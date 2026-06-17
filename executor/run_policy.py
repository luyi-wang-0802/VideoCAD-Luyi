"""Roll out a trained primitive-action policy."""

from __future__ import annotations

import argparse
import ctypes
import json
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_loader.data_loader import PrimitiveActionDataset, read_json
from data_loader.image_loader import ScreenshotImageLoader
from data_process.transform_dataset import summarize_resplan_for_model
from executor.vectorworks_executor import VectorworksExecutor
from model.model_factory import ModelFactory, _strip_wrappers


PROGRESS_ENTITY_KEYS = [
    "exterior_walls",
    "interior_walls",
    "windows",
    "doors",
    "slabs",
    "roofs",
]
HIGH_LEVEL_TO_PROGRESS_KEY = {
    "CREATE_EXTERIOR_WALL": "exterior_walls",
    "CREATE_INTERIOR_WALL": "interior_walls",
    "CREATE_WINDOW": "windows",
    "CREATE_DOOR": "doors",
    "CREATE_SLAB": "slabs",
    "CREATE_ROOF": "roofs",
}


class RolloutAbortMonitor:
    VK_C = 0x43
    VK_CONTROL = 0x11
    VK_LCONTROL = 0xA2
    VK_RCONTROL = 0xA3

    def __init__(self, enabled: bool = True, poll_interval_s: float = 0.05) -> None:
        self.enabled = enabled and sys.platform.startswith("win")
        self.poll_interval_s = poll_interval_s
        self.abort_event = threading.Event()
        self.stop_event = threading.Event()
        self.reason: str | None = None
        self.thread: threading.Thread | None = None

    def __enter__(self) -> "RolloutAbortMonitor":
        if self.enabled:
            self.thread = threading.Thread(target=self._poll_hotkey, name="rollout-abort-monitor", daemon=True)
            self.thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=0.2)

    def _key_down(self, virtual_key: int) -> bool:
        return bool(ctypes.windll.user32.GetAsyncKeyState(virtual_key) & 0x8000)

    def _poll_hotkey(self) -> None:
        while not self.stop_event.is_set() and not self.abort_event.is_set():
            control_down = (
                self._key_down(self.VK_CONTROL)
                or self._key_down(self.VK_LCONTROL)
                or self._key_down(self.VK_RCONTROL)
            )
            if control_down and self._key_down(self.VK_C):
                self.reason = "ctrl+c"
                self.abort_event.set()
                break
            self.stop_event.wait(self.poll_interval_s)

    def requested(self) -> bool:
        return self.abort_event.is_set()


def move_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    return value


def inverse_mapping(mapping: dict[str, int]) -> dict[int, str]:
    return {int(value): key for key, value in mapping.items()}


def progress_vector(task_counts: dict[str, int], done_counts: dict[str, int]) -> list[float]:
    vector = []
    for key in PROGRESS_ENTITY_KEYS:
        total = max(int(task_counts.get(key, 0) or 0), 0)
        done = max(int(done_counts.get(key, 0) or 0), 0)
        done_clamped = min(done, total) if total else done
        ratio = float(done_clamped) / float(total) if total else 1.0
        vector.extend([float(total) / 100.0, float(done_clamped) / 100.0, ratio])
    return vector


def initial_progress_state(sample: dict[str, Any]) -> dict[str, dict[str, int]]:
    counts = sample["model_inputs"].get("task_entity_counts") or {}
    task_counts = {key: int(counts.get(key, 0) or 0) for key in PROGRESS_ENTITY_KEYS}
    return {
        "task_counts": task_counts,
        "done_counts": {key: 0 for key in PROGRESS_ENTITY_KEYS},
    }


def should_count_completed_entity(decoded_action: dict[str, Any]) -> bool:
    high_level = decoded_action.get("high_level_action")
    action = decoded_action.get("executor_action", {})
    action_type = action.get("action_type")
    if high_level in {"CREATE_EXTERIOR_WALL", "CREATE_INTERIOR_WALL"}:
        return action_type == "PRESS_KEY" and action.get("key") == "enter"
    if high_level == "CREATE_SLAB":
        return action_type == "CLICK"
    if high_level in {"CREATE_WINDOW", "CREATE_DOOR"}:
        return action_type == "DOUBLE_CLICK"
    if high_level == "CREATE_ROOF":
        return action_type == "PRESS_KEY" and action.get("key") == "enter"
    return False


def update_progress_state(progress_state: dict[str, dict[str, int]], decoded_action: dict[str, Any]) -> None:
    progress_key = HIGH_LEVEL_TO_PROGRESS_KEY.get(decoded_action.get("high_level_action"))
    if not progress_key or not should_count_completed_entity(decoded_action):
        return
    done_counts = progress_state["done_counts"]
    task_counts = progress_state["task_counts"]
    done_counts[progress_key] = min(done_counts.get(progress_key, 0) + 1, max(task_counts.get(progress_key, 0), 1))


def load_primitive_model(
    checkpoint_path: Path,
    model_config_path: Path,
    model_name: str,
    dataset_path: Path,
    action_vocab_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    model_params = read_json(model_config_path)
    model_config = dict(model_params[model_name])
    model_config["dataset_path"] = str(dataset_path)
    model_config["action_vocab_path"] = str(action_vocab_path)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model, _ = ModelFactory().create_model("primitive_action_policy", model_config, device)
    model.load_state_dict(_strip_wrappers(state_dict), strict=False)
    model.eval()
    return model


def infer_action_vocab_path(checkpoint_path: Path, explicit_path: Path | None = None) -> Path:
    if explicit_path is not None:
        return explicit_path
    checkpoint_vocab = checkpoint_path.parent / "action_vocab.json"
    if checkpoint_vocab.exists():
        return checkpoint_vocab
    raise FileNotFoundError(
        f"Expected action vocab next to checkpoint: {checkpoint_vocab}. "
        "Live rollout requires checkpoint_dir/action_vocab.json so decoded ids match the checkpoint."
    )


def infer_checkpoint_load_global_floorplan(checkpoint_path: Path) -> bool:
    run_dir = checkpoint_path.parent.parent
    command_args_path = run_dir / "configs" / "command_args.json"
    if not command_args_path.exists():
        return True
    command_args = read_json(command_args_path)
    return bool(command_args.get("load_global_floorplan", True))


def make_batch(
    dataset: PrimitiveActionDataset,
    sample: dict[str, Any],
    screenshot_path: str | None,
    encoded_history: list[dict[str, torch.Tensor]],
    progress_state: dict[str, dict[str, int]],
    image_loader: ScreenshotImageLoader,
    device: torch.device,
    step_index: int,
    load_global_floorplan: bool,
) -> dict[str, Any]:
    plan = dataset._encode_plan(sample["model_inputs"]["encoded_resplan"])
    history = dataset._build_history(encoded_history, len(encoded_history))
    observation, available = image_loader.load(screenshot_path)
    batch = {
        "sample_id": [sample["sample_id"]],
        "step_index": torch.tensor([step_index], dtype=torch.long),
        "observation": observation.unsqueeze(0),
        "observation_available": torch.tensor([available], dtype=torch.bool),
        "global_floorplan": None,
        "global_floorplan_available": torch.tensor([False], dtype=torch.bool),
        "progress": torch.tensor(
            [progress_vector(progress_state["task_counts"], progress_state["done_counts"])],
            dtype=torch.float32,
        ),
        "plan": {
            "walls": plan["walls"].unsqueeze(0),
            "wall_mask": torch.ones((1, plan["walls"].shape[0]), dtype=torch.bool),
            "insertions": plan["insertions"].unsqueeze(0),
            "insertion_mask": torch.ones((1, plan["insertions"].shape[0]), dtype=torch.bool),
        },
        "history": {key: value.unsqueeze(0) for key, value in history.items()},
    }
    if load_global_floorplan:
        global_floorplan, global_floorplan_available = image_loader.load(
            sample["model_inputs"].get("global_floorplan_path")
        )
        batch["global_floorplan"] = global_floorplan.unsqueeze(0)
        batch["global_floorplan_available"] = torch.tensor([global_floorplan_available], dtype=torch.bool)
    return move_to_device(batch, device)


def infer_run_id_from_resplan_path(resplan_json_path: Path) -> str | None:
    match = re.search(r"resplan_to_JSON_(\d+)", resplan_json_path.stem)
    if not match:
        return None
    return f"plan_{int(match.group(1)):03d}"


def infer_global_floorplan_path(
    resplan_json_path: Path,
    dataset_path: Path,
    explicit_path: Path | None = None,
) -> str | None:
    if explicit_path is not None:
        return str(explicit_path).replace("\\", "/")

    run_id = infer_run_id_from_resplan_path(resplan_json_path)
    if run_id is None:
        return None

    candidates = [
        dataset_path / "images" / run_id / "global_floorplan" / "floorplan_before_roof.png",
        Path("raw_data") / "trajectory_data" / run_id / "global_floorplan" / "floorplan_before_roof.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate).replace("\\", "/")
    return None


def infer_runtime_plan_path(resplan_json_path: Path, dataset_path: Path) -> Path | None:
    run_id = infer_run_id_from_resplan_path(resplan_json_path)
    if run_id is None:
        return None
    candidate = dataset_path / "runtime_plans" / f"{run_id}_runtime_plan.json"
    return candidate if candidate.exists() else None


def runtime_sample_from_resplan(
    resplan_json_path: Path,
    runtime_plan_path: Path | None = None,
    dataset_path: Path = Path("processed_data"),
    global_floorplan_path: Path | None = None,
    grounding_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if runtime_plan_path is not None:
        runtime_plan = read_json(runtime_plan_path)
        encoded_resplan = runtime_plan.get("encoded_resplan")
        if not encoded_resplan:
            raise ValueError(f"{runtime_plan_path} does not contain encoded_resplan")
        return {
            "sample_id": runtime_plan.get("sample_id") or resplan_json_path.stem,
            "model_inputs": {
                "resplan_json_path": str(resplan_json_path).replace("\\", "/"),
                "global_floorplan_path": str(global_floorplan_path).replace("\\", "/")
                if global_floorplan_path is not None
                else runtime_plan.get("global_floorplan_path"),
                "task_entity_counts": runtime_plan.get("task_entity_counts")
                or (encoded_resplan.get("task_entity_counts") if isinstance(encoded_resplan, dict) else {}),
                "encoded_resplan": encoded_resplan,
            },
            "runtime_plan_path": str(runtime_plan_path).replace("\\", "/"),
            "steps": [],
        }

    resplan = read_json(resplan_json_path)
    encoded_resplan = summarize_resplan_for_model(resplan, primitive_plan={}, grounding_config=None)
    return {
        "sample_id": resplan_json_path.stem,
        "model_inputs": {
            "resplan_json_path": str(resplan_json_path).replace("\\", "/"),
            "global_floorplan_path": infer_global_floorplan_path(
                resplan_json_path,
                dataset_path=dataset_path,
                explicit_path=global_floorplan_path,
            ),
            "task_entity_counts": resplan.get("task_entity_counts")
            or encoded_resplan.get("task_entity_counts", {}),
            "encoded_resplan": encoded_resplan,
        },
        "runtime_plan_path": None,
        "steps": [],
    }


def primitive_action_to_executor_action(
    primitive_action: list[float],
    key_name: str | None,
) -> dict[str, Any]:
    action_type_by_id = {
        1: "MOVE_TO",
        2: "CLICK",
        3: "PRESS_KEY",
        4: "HOTKEY",
        5: "DOUBLE_CLICK",
    }
    action_type_id = int(round(float(primitive_action[0])))
    action_type = action_type_by_id.get(action_type_id, "<unknown>")
    x = float(primitive_action[1])
    y = float(primitive_action[2])
    repeat_count = int(round(float(primitive_action[4]))) if primitive_action[4] >= 0 else 1
    key_interval = float(primitive_action[5]) if primitive_action[5] >= 0 else -1.0

    action: dict[str, Any] = {
        "action_type": action_type,
        "coordinate_frame": "model" if action_type == "MOVE_TO" and x != -1 and y != -1 else "none",
    }
    if action_type in {"MOVE_TO", "CLICK", "DOUBLE_CLICK"}:
        if action_type == "MOVE_TO" and x != -1 and y != -1:
            action["target"] = {"model_point": [round(x, 6), round(y, 6)]}
        if action_type in {"CLICK", "DOUBLE_CLICK"}:
            action["button"] = "left"
    elif action_type == "PRESS_KEY":
        action["key"] = key_name or "enter"
        action["repeat_count"] = max(repeat_count, 1)
        if key_interval >= 0:
            action["key_interval_ms"] = round(key_interval, 3)
    elif action_type == "HOTKEY":
        action["keys"] = (key_name or "").split("+")
        action["repeat_count"] = max(repeat_count, 2) if key_name == "ctrl+2" else max(repeat_count, 1)
        if key_name == "ctrl+2":
            key_interval = max(key_interval, 450.0)
        if key_interval >= 0:
            action["key_interval_ms"] = round(key_interval, 3)
    return action


def decode_action(outputs: dict[str, torch.Tensor], model: torch.nn.Module, dataset: PrimitiveActionDataset) -> dict[str, Any]:
    decoded = model.decode_action(outputs)
    primitive_action = decoded["primitive_action"][0].detach().cpu().tolist()

    action_type_by_id = inverse_mapping(dataset.action_type_to_id)
    high_level_by_id = inverse_mapping(dataset.high_level_to_id)
    gui_action_by_id = inverse_mapping(dataset.gui_action_to_id)
    key_by_id = inverse_mapping(dataset.key_to_id)
    frame_by_id = inverse_mapping(dataset.coordinate_frame_to_id)
    action_type_id = int(decoded["action_type_id"][0].item())
    high_level_id = int(decoded["high_level_id"][0].item())
    gui_action_id = int(decoded["gui_action_id"][0].item())
    frame_id = int(decoded["coordinate_frame_id"][0].item())
    key_id = int(round(primitive_action[3])) if primitive_action[3] >= 0 else -1

    primitive_action_type_id = int(round(float(primitive_action[0])))
    action_type = action_type_by_id.get(primitive_action_type_id, "<unknown>")
    key_name = key_by_id.get(key_id)
    coordinate_frame = "model" if action_type == "MOVE_TO" and primitive_action[1] != -1 and primitive_action[2] != -1 else "none"

    return {
        "policy_level": "primitive_action",
        "high_level_action": high_level_by_id.get(high_level_id, "<unknown>"),
        "gui_action": gui_action_by_id.get(gui_action_id, "<unknown>"),
        "coordinate_frame": coordinate_frame,
        "primitive_action": [round(float(value), 6) for value in primitive_action],
        "executor_action": primitive_action_to_executor_action(
            primitive_action,
            key_name,
        ),
        "raw_prediction": {
            "action_type_id": action_type_id,
            "high_level_id": high_level_id,
            "gui_action_id": gui_action_id,
            "coordinate_frame_id": frame_id,
            "key_id": key_id,
        },
    }


def encode_decoded_action(dataset: PrimitiveActionDataset, decoded_action: dict[str, Any]) -> dict[str, torch.Tensor]:
    action = torch.tensor(decoded_action["primitive_action"], dtype=torch.float32)
    action_type_id = int(action[0].item())
    key_id = int(action[3].item())
    repeat_count = int(action[4].item()) if action[4].item() >= 0 else -1
    raw = decoded_action["raw_prediction"]
    is_wall_move = action_type_id == 1 and (
        decoded_action.get("high_level_action") in {"CREATE_EXTERIOR_WALL", "CREATE_INTERIOR_WALL"}
        or decoded_action.get("gui_action") == "DRAW_WALL_FROM_ENTITY_GEOMETRY"
    )
    return {
        "primitive_action": action,
        "action_type_id": torch.tensor(action_type_id, dtype=torch.long),
        "xy": action[1:3].clone(),
        "key_id": torch.tensor(key_id, dtype=torch.long),
        "key_repeat_count": torch.tensor(repeat_count, dtype=torch.long),
        "key_interval": action[5].clone().to(torch.float32),
        "high_level_id": torch.tensor(int(raw["high_level_id"]), dtype=torch.long),
        "gui_action_id": torch.tensor(int(raw["gui_action_id"]), dtype=torch.long),
        "coordinate_frame_id": torch.tensor(int(raw["coordinate_frame_id"]), dtype=torch.long),
        "is_move": torch.tensor(action_type_id == 1, dtype=torch.bool),
        "is_wall_move": torch.tensor(is_wall_move, dtype=torch.bool),
        "is_key_action": torch.tensor(action_type_id in {3, 4}, dtype=torch.bool),
    }


def target_from_sample_step(step: dict[str, Any]) -> dict[str, Any]:
    target = step["supervision_target"]
    return {
        "high_level_action": target["high_level_action"],
        "gui_action": target["gui_action"],
        "coordinate_frame": target["coordinate_frame"],
        "primitive_action": target["primitive_action"],
    }


def compare_to_sample(decoded_action: dict[str, Any], sample_step: dict[str, Any]) -> dict[str, Any]:
    expected = target_from_sample_step(sample_step)
    pred = decoded_action["primitive_action"]
    exp = expected["primitive_action"]
    xy_error = None
    if exp[1] != -1 and exp[2] != -1 and pred[1] != -1 and pred[2] != -1:
        xy_error = ((float(pred[1]) - float(exp[1])) ** 2 + (float(pred[2]) - float(exp[2])) ** 2) ** 0.5
    return {
        "expected": expected,
        "action_type_match": int(round(pred[0])) == int(round(exp[0])),
        "high_level_match": decoded_action["high_level_action"] == expected["high_level_action"],
        "gui_action_match": decoded_action["gui_action"] == expected["gui_action"],
        "coordinate_frame_match": decoded_action["coordinate_frame"] == expected["coordinate_frame"],
        "xy_error": xy_error,
    }


def inspect_step_trajectory(
    sample_step: dict[str, Any],
    decoded_action: dict[str, Any],
    predicted_screen_point: dict[str, int] | None,
    executor: VectorworksExecutor,
) -> dict[str, Any] | None:
    expected = sample_step["supervision_target"]
    expected_xy = expected["primitive_action"][1:3]
    predicted_xy = decoded_action["primitive_action"][1:3]
    if len(expected_xy) < 2 or len(predicted_xy) < 2:
        return None

    expected_xy = [float(expected_xy[0]), float(expected_xy[1])]
    predicted_xy = [float(predicted_xy[0]), float(predicted_xy[1])]
    expected_have_xy = expected_xy[0] != -1 and expected_xy[1] != -1
    predicted_have_xy = predicted_xy[0] != -1 and predicted_xy[1] != -1
    expected_screen_point = None
    if expected_have_xy:
        expected_screen_point = executor.resolve_action_point(
            {
                "action_type": "CLICK",
                "target": {"model_point": expected_xy},
            }
        )

    xy_error = None
    if expected_have_xy and predicted_have_xy:
        dx = predicted_xy[0] - expected_xy[0]
        dy = predicted_xy[1] - expected_xy[1]
        xy_error = (dx * dx + dy * dy) ** 0.5

    return {
        "gt_xy": expected_xy if expected_have_xy else None,
        "pred_xy": predicted_xy if predicted_have_xy else None,
        "gt_screen_xy": {"x": expected_screen_point[0], "y": expected_screen_point[1]} if expected_screen_point else None,
        "pred_screen_xy": predicted_screen_point,
        "xy_error": xy_error,
        "gt_frame": expected.get("coordinate_frame"),
        "pred_frame": decoded_action.get("coordinate_frame"),
        "gt_action_type": expected.get("high_level_action") or "<none>",
        "pred_action_type": decoded_action.get("high_level_action"),
    }


def execute_decoded_action(executor: VectorworksExecutor, decoded_action: dict[str, Any]) -> dict[str, Any]:
    action = decoded_action["executor_action"]
    if action["action_type"] == "MOVE_TO":
        event = {
            "timestamp_utc_before": None,
            "action": action,
            "dry_run": executor.dry_run,
        }
        point = executor.resolve_action_point(action)
        if point is not None:
            event["resolved_screen_point"] = {"x": point[0], "y": point[1]}
        if not executor.dry_run:
            if point is None:
                raise ValueError("MOVE_TO requires target.model_point for live execution")
            assert executor.pyautogui is not None
            executor.pyautogui.moveTo(point[0], point[1], duration=0.08)
        event["timestamp_utc_after"] = None
        return event
    return executor.execute(action)


def compact_event(
    event: dict[str, Any],
    step_index: int,
    decoded_action: dict[str, Any],
    screenshot_before: str | None,
    screenshot_after: str | None,
) -> dict[str, Any]:
    prediction = {
        "primitive_action": decoded_action["primitive_action"],
    }
    compact = {
        "step_index": step_index,
        "prediction": prediction,
        "executed_action": event.get("action"),
        "debug_prediction": {
            "high_level_action": decoded_action.get("high_level_action"),
            "gui_action": decoded_action.get("gui_action"),
            "coordinate_frame": decoded_action.get("coordinate_frame"),
        },
        "screenshot_before": screenshot_before,
        "screenshot_after": screenshot_after,
    }
    if event.get("resolved_screen_point"):
        compact["resolved_screen_point"] = event["resolved_screen_point"]
    if event.get("timestamp_utc_before") or event.get("timestamp_utc_after"):
        compact["timestamps"] = {
            "before": event.get("timestamp_utc_before"),
            "after": event.get("timestamp_utc_after"),
        }
    if "sample_comparison" in event:
        compact["sample_comparison"] = event["sample_comparison"]
    if "trajectory_inspect" in event:
        compact["trajectory_inspect"] = event["trajectory_inspect"]
    return compact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resplan-json", required=True, type=Path, help="Raw ResPlan JSON used as the global plan input.")
    parser.add_argument(
        "--runtime-plan",
        default=None,
        type=Path,
        help="Optional legacy encoded plan override. If omitted, rollout uses only --resplan-json as the plan input.",
    )
    parser.add_argument("--global-floorplan", default=None, type=Path, help="Optional floorplan image override. If omitted, inferred from the ResPlan filename.")
    parser.add_argument("--sample", default=None, type=Path, help="Optional processed sample JSON for debug comparison only.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset-path", default="processed_data", type=Path)
    parser.add_argument(
        "--action-vocab",
        default=None,
        type=Path,
        help="Optional vocab override. Defaults to checkpoint_dir/action_vocab.json.",
    )
    parser.add_argument("--model-config", default="model_configs/primitive_action_policy.json", type=Path)
    parser.add_argument("--model-name", default="default_params")
    parser.add_argument("--calibration", default="configs/vectorworks_grounding_template.json", type=Path)
    parser.add_argument("--run-dir", default="outputs/policy_rollouts/manual_run", type=Path)
    parser.add_argument("--initial-screenshot", default=None, type=str)
    parser.add_argument("--max-steps", default=20, type=int)
    parser.add_argument("--history-length", default=32, type=int)
    parser.add_argument("--image-size", default=224, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--no-global-floorplan",
        action="store_true",
        help="Disable loading the global floorplan image during rollout, regardless of checkpoint command args.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--use-recorded-observations", action="store_true")
    parser.add_argument("--teacher-force-history", action="store_true")
    parser.add_argument("--compare-sample", action="store_true")
    parser.add_argument("--live-primitive-actions", action="store_true")
    parser.add_argument("--countdown", default=5, type=int)
    parser.add_argument(
        "--inspect-step",
        type=int,
        default=None,
        help="Print a one-step coordinate path trace: gt xy, pred xy, gt/pred screen points (requires --sample).",
    )
    args = parser.parse_args()

    if (args.use_recorded_observations or args.teacher_force_history or args.compare_sample) and args.sample is None:
        raise ValueError("--sample is required when using recorded observations, teacher-forced history, or comparison.")
    if args.inspect_step is not None and args.sample is None:
        raise ValueError("--inspect-step requires --sample.")

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    action_vocab_path = infer_action_vocab_path(args.checkpoint, args.action_vocab)
    load_global_floorplan = (not args.no_global_floorplan) and infer_checkpoint_load_global_floorplan(args.checkpoint)
    sample = runtime_sample_from_resplan(
        args.resplan_json,
        runtime_plan_path=args.runtime_plan,
        dataset_path=args.dataset_path,
        global_floorplan_path=args.global_floorplan,
        grounding_config=read_json(args.calibration),
    )
    debug_sample = read_json(args.sample) if args.sample is not None else None
    dataset = PrimitiveActionDataset(
        dataset_path=args.dataset_path,
        action_vocab_path=action_vocab_path,
        split=None,
        image_size=(args.image_size, args.image_size),
        history_length=args.history_length,
        load_images=False,
    )
    image_loader = ScreenshotImageLoader(image_size=(args.image_size, args.image_size))
    model = load_primitive_model(
        args.checkpoint,
        args.model_config,
        args.model_name,
        args.dataset_path,
        action_vocab_path,
        device,
    )
    executor = VectorworksExecutor(args.calibration, dry_run=args.dry_run)
    args.run_dir.mkdir(parents=True, exist_ok=True)

    if not args.dry_run and not args.live_primitive_actions:
        raise RuntimeError("Refusing live execution without --live-primitive-actions. Run --dry-run first.")
    if not args.dry_run and args.countdown > 0:
        print("Switch focus to Vectorworks.")
        for remaining in range(args.countdown, 0, -1):
            print(f"Starting in {remaining}...")
            time.sleep(1)

    screenshot = args.initial_screenshot
    encoded_history: list[dict[str, torch.Tensor]] = []
    progress_state = initial_progress_state(sample)
    log_path = args.run_dir / "policy_events.jsonl"
    log_handle = log_path.open("w", encoding="utf-8")
    sample_steps = debug_sample.get("steps", []) if debug_sample is not None else []
    abort_monitor = RolloutAbortMonitor(enabled=not args.dry_run)
    try:
        with abort_monitor:
            for step_index in range(args.max_steps):
                if abort_monitor.requested():
                    print(f"Abort requested by {abort_monitor.reason}; stopping before step {step_index}.")
                    break
                if args.use_recorded_observations and step_index < len(sample_steps):
                    screenshot = sample_steps[step_index]["model_input"].get("observation_screenshot_path")
                if screenshot is None and not args.dry_run:
                    screenshot = executor.capture_screenshot(args.run_dir, step_index * 2)
                if abort_monitor.requested():
                    print(f"Abort requested by {abort_monitor.reason}; stopping before inference at step {step_index}.")
                    break

                batch = make_batch(
                    dataset,
                    sample,
                    screenshot,
                    encoded_history,
                    progress_state,
                    image_loader,
                    device,
                    step_index,
                    load_global_floorplan=load_global_floorplan,
                )
                with torch.no_grad():
                    outputs = model(batch)
                decoded_action = decode_action(outputs, model, dataset)
                if abort_monitor.requested():
                    print(f"Abort requested by {abort_monitor.reason}; stopping before execution at step {step_index}.")
                    break

                event = execute_decoded_action(executor, decoded_action)
                event["screenshot_before"] = screenshot
                if step_index < len(sample_steps):
                    sample_step = sample_steps[step_index]
                    if args.compare_sample:
                        event["sample_comparison"] = compare_to_sample(decoded_action, sample_step)
                    if args.inspect_step is not None and step_index == args.inspect_step:
                        event["trajectory_inspect"] = inspect_step_trajectory(
                            sample_step,
                            decoded_action,
                            event.get("resolved_screen_point"),
                            executor,
                        )

                if args.inspect_step is not None and step_index == args.inspect_step:
                    if args.inspect_step >= len(sample_steps):
                        print(f"[inspect-step {step_index}] sample has no corresponding step for comparison.")
                    else:
                        inspect_info = event.get("trajectory_inspect")
                        if inspect_info is None:
                            print(f"[inspect-step {step_index}] no valid xy values to compare.")
                        else:
                            print(
                                json.dumps(
                                    {
                                        "step": step_index,
                                        "gt_xy": inspect_info["gt_xy"],
                                        "pred_xy": inspect_info["pred_xy"],
                                        "gt_screen_xy": inspect_info["gt_screen_xy"],
                                        "pred_screen_xy": inspect_info["pred_screen_xy"],
                                        "xy_error": inspect_info["xy_error"],
                                        "gt_action_type": inspect_info["gt_action_type"],
                                        "pred_action_type": inspect_info["pred_action_type"],
                                        "gt_frame": inspect_info["gt_frame"],
                                        "pred_frame": inspect_info["pred_frame"],
                                    },
                                    ensure_ascii=False,
                                )
                            )

                screenshot_after = executor.capture_screenshot(args.run_dir, step_index * 2 + 1)
                event["screenshot_after"] = screenshot_after
                compact = compact_event(event, step_index, decoded_action, screenshot, screenshot_after)
                compact["task_progress"] = {
                    "entity_order": PROGRESS_ENTITY_KEYS,
                    "task_entity_counts": dict(progress_state["task_counts"]),
                    "done_entity_counts_before": dict(progress_state["done_counts"]),
                    "vector": progress_vector(progress_state["task_counts"], progress_state["done_counts"]),
                }
                log_handle.write(json.dumps(compact, ensure_ascii=False) + "\n")
                log_handle.flush()
                print(json.dumps(compact, ensure_ascii=False, indent=2))

                if abort_monitor.requested():
                    print(f"Abort requested by {abort_monitor.reason}; stopping after step {step_index}.")
                    break
                if args.teacher_force_history and step_index < len(sample_steps):
                    encoded_history.append(dataset._encode_step(sample_steps[step_index]))
                else:
                    encoded_history.append(encode_decoded_action(dataset, decoded_action))
                update_progress_state(progress_state, decoded_action)
                screenshot = screenshot_after or screenshot
    except KeyboardInterrupt:
        print("Abort requested by console KeyboardInterrupt; stopping rollout.")
    finally:
        log_handle.close()
    print(f"Wrote {log_path}")


if __name__ == "__main__":
    main()
