"""Prepare training data from the reorganized raw_data directory.

Expected raw_data layout:

raw_data/
  resplan_to_JSON/resplan_to_JSON_001.json
  bim_sequence/resplan_to_JSON_001_bim_sequence/
    high_level_sequence.json
    gui_action_sequence.json
    primitive_plan.json
  trajectory_data/plan_001/
    imitation_trajectory.jsonl
    primitive_actions.jsonl
    screenshots/*.png

Training contract:

- Model inputs at inference time:
  - ResPlan JSON / encoded plan features
  - current observation screenshot
  - historical primitive actions and historical predicted/planned labels
- Supervision targets during training:
  - next high-level action
  - next GUI action
  - next primitive action

The rule-based BIM/GUI/primitive sequences are used only as supervision
references and provenance, not as model input answers.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps


MISSING = -1
DEFAULT_KEY_INTERVAL_MS = 100.0
DEFAULT_PROCESSED_IMAGE_SIZE = (384, 216)
SPLIT_RATIOS = {"train": 0.7, "val": 0.2, "test": 0.1}
SPLIT_ORDER = ("train", "val", "test")
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

ACTION_TYPE_TO_ID = {
    "MOVE_TO": 1,
    "CLICK": 2,
    "PRESS_KEY": 3,
    "HOTKEY": 4,
    "DOUBLE_CLICK": 5,
}
ID_TO_ACTION_TYPE = {value: key for key, value in ACTION_TYPE_TO_ID.items()}

COORDINATE_FRAME_TO_ID = {
    "none": 0,
    "model": 1,
    "window_norm": 2,
    "screen": 3,
}

DEFAULT_WINDOW_EDGE_OFFSET_SOURCE_UNITS = 1.5
DEFAULT_DOOR_EDGE_OFFSET_SOURCE_UNITS = 1.0
DEFAULT_GROUNDING_CONFIG_PATH = Path("configs/vectorworks_grounding_template.json")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_optional_json(path: Path | None, default: Any) -> Any:
    if path is None or not path.exists():
        return default
    return read_json(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_number}: {exc}") from exc
    return records


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def relpath(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_root).as_posix()
    except ValueError:
        return resolved.as_posix()


def relpath_if_exists(path: Path | None, repo_root: Path) -> str | None:
    if path is None or not path.exists():
        return None
    return relpath(path, repo_root)


def sequence_root(raw_data_dir: Path) -> Path:
    return raw_data_dir / "bim_sequence"


def stable_mapping(names: list[str], include_none: bool = True) -> dict[str, int]:
    unique_names = sorted({str(name) for name in names if name and str(name) != "<none>"})
    if include_none:
        return {"<none>": 0, **{name: index + 1 for index, name in enumerate(unique_names)}}
    return {name: index for index, name in enumerate(unique_names)}


def plan_suffix_from_run_dir(run_dir: Path) -> str:
    if not run_dir.name.startswith("plan_"):
        raise ValueError(f"Trajectory run directory must be named plan_XXX: {run_dir}")
    return run_dir.name.removeprefix("plan_")


def discover_runs(raw_data_dir: Path) -> list[dict[str, Path | str]]:
    trajectory_root = raw_data_dir / "trajectory_data"
    resplan_root = raw_data_dir / "resplan_to_JSON"
    sequence_root_dir = sequence_root(raw_data_dir)
    run_dirs = sorted(
        path
        for path in trajectory_root.glob("plan_*")
        if path.is_dir() and (path / "imitation_trajectory.jsonl").exists()
    )
    runs = []
    for run_dir in run_dirs:
        suffix = plan_suffix_from_run_dir(run_dir)
        resplan_path = resplan_root / f"resplan_to_JSON_{suffix}.json"
        sequence_dir = sequence_root_dir / f"resplan_to_JSON_{suffix}_bim_sequence"
        paths = {
            "run_id": run_dir.name,
            "suffix": suffix,
            "trajectory_dir": run_dir,
            "imitation_path": run_dir / "imitation_trajectory.jsonl",
            "primitive_actions_path": run_dir / "primitive_actions.jsonl",
            "run_summary_path": run_dir / "run_summary.json",
            "resplan_path": resplan_path,
            "sequence_dir": sequence_dir,
            "high_level_sequence_path": sequence_dir / "high_level_sequence.json",
            "gui_action_sequence_path": sequence_dir / "gui_action_sequence.json",
            "primitive_plan_path": sequence_dir / "primitive_plan.json",
        }
        optional = {
            "run_id",
            "suffix",
            "primitive_actions_path",
            "sequence_dir",
            "high_level_sequence_path",
            "gui_action_sequence_path",
            "primitive_plan_path",
        }
        missing = [str(path) for key, path in paths.items() if key not in optional and not Path(path).exists()]
        if missing:
            raise FileNotFoundError(f"Missing files for {run_dir.name}: {missing}")
        runs.append(paths)
    if not runs:
        raise FileNotFoundError(f"No trajectory_data/plan_* runs found under {raw_data_dir}")
    return runs


def key_name(action: dict[str, Any]) -> str | None:
    if action.get("type") == "PRESS_KEY":
        key = action.get("key")
        return str(key).lower() if key is not None else None
    if action.get("type") == "HOTKEY":
        keys = action.get("keys") or []
        return "+".join(str(key).lower() for key in keys)
    return None


def collect_key_names(actions: list[dict[str, Any]]) -> list[str]:
    return [name for action in actions if (name := key_name(action))]


def high_level_name(action: dict[str, Any]) -> str:
    return action.get("parent_high_level_type") or action.get("high_level_action_type") or "<none>"


def gui_action_name(action: dict[str, Any]) -> str:
    return action.get("parent_gui_action") or action.get("gui_action_type") or "<none>"


def get_xy_and_frame(action: dict[str, Any]) -> tuple[float, float, str]:
    model_point = action.get("model_point_mm")
    if isinstance(model_point, list) and len(model_point) >= 2:
        return float(model_point[0]), float(model_point[1]), "model"

    resolved_norm = action.get("resolved_window_point_norm")
    if isinstance(resolved_norm, list) and len(resolved_norm) >= 2:
        return float(resolved_norm[0]), float(resolved_norm[1]), "window_norm"

    coordinates = action.get("coordinates") or {}
    window = coordinates.get("window") or {}
    if "x_norm" in window and "y_norm" in window:
        return float(window["x_norm"]), float(window["y_norm"]), "window_norm"

    resolved = action.get("resolved_window_point")
    if isinstance(resolved, list) and len(resolved) >= 2:
        return float(resolved[0]), float(resolved[1]), "screen"
    if "x" in window and "y" in window:
        return float(window["x"]), float(window["y"]), "screen"

    screen = coordinates.get("screen") or {}
    if "x" in screen and "y" in screen:
        return float(screen["x"]), float(screen["y"]), "screen"

    return float(MISSING), float(MISSING), "none"


def is_repeatable_key_action(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if left.get("type") not in {"PRESS_KEY", "HOTKEY"}:
        return False
    if right.get("type") != left.get("type"):
        return False
    if left.get("parent_high_level_type") != right.get("parent_high_level_type"):
        return False
    if left.get("parent_gui_action") != right.get("parent_gui_action"):
        return False
    return key_name(left) == key_name(right)


def condense_key_repeats(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    condensed = []
    index = 0
    while index < len(actions):
        current = dict(actions[index])
        group = [actions[index]]
        next_index = index + 1
        while next_index < len(actions) and is_repeatable_key_action(group[-1], actions[next_index]):
            group.append(actions[next_index])
            next_index += 1

        if len(group) > 1:
            intervals = []
            for prev, nxt in zip(group, group[1:]):
                prev_time = prev.get("time_s")
                next_time = nxt.get("time_s")
                if prev_time is not None and next_time is not None:
                    intervals.append(max(0.0, float(next_time) - float(prev_time)) * 1000.0)
            current["key_repeat_count"] = len(group)
            current["key_interval"] = round(sum(intervals) / len(intervals), 3) if intervals else DEFAULT_KEY_INTERVAL_MS
            current["condensed_primitive_ids"] = [item.get("primitive_id") for item in group]
            current["timestamp_utc_after"] = group[-1].get("timestamp_utc_after")
            current["timestamp_utc"] = group[-1].get("timestamp_utc")
            current["time_s_after"] = group[-1].get("time_s_after")
            current["time_s"] = group[-1].get("time_s")
            current["screenshot_path"] = group[0].get("screenshot_path") or current.get("screenshot_path")
        else:
            current["key_repeat_count"] = 1 if current.get("type") in {"PRESS_KEY", "HOTKEY"} else MISSING
            current["key_interval"] = DEFAULT_KEY_INTERVAL_MS if current.get("type") in {"PRESS_KEY", "HOTKEY"} else MISSING

        condensed.append(current)
        index = next_index
    return condensed


def resolve_screenshot_path(path_value: str | None, run_dir: Path, repo_root: Path) -> str | None:
    if not path_value:
        return None

    raw_path = Path(path_value)
    candidates = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.append(repo_root / raw_path)
        if "screenshots" in raw_path.parts:
            candidates.append(run_dir / "screenshots" / raw_path.name)

    for candidate in candidates:
        if candidate.exists():
            return relpath(candidate, repo_root)
    return path_value.replace("\\", "/")


def resolve_global_floorplan_path(run: dict[str, Path | str], repo_root: Path) -> str | None:
    run_dir = Path(run["trajectory_dir"])
    run_summary_path = Path(run["run_summary_path"])
    path_value = None
    if run_summary_path.exists():
        summary = read_json(run_summary_path)
        capture = summary.get("global_floorplan_capture") or {}
        path_value = capture.get("screenshot_path")

    metadata_path = run_dir / "global_floorplan" / "metadata.json"
    if path_value is None and metadata_path.exists():
        metadata = read_json(metadata_path)
        path_value = metadata.get("screenshot_path")

    candidates: list[Path] = []
    if path_value:
        raw_path = Path(path_value)
        if raw_path.is_absolute():
            candidates.append(raw_path)
        else:
            candidates.extend(
                [
                    repo_root / raw_path,
                    run_dir.parent / raw_path,
                    run_dir / raw_path.name,
                    run_dir / "global_floorplan" / raw_path.name,
                ]
            )
    candidates.append(run_dir / "global_floorplan" / "floorplan_before_roof.png")

    for candidate in candidates:
        if candidate.exists():
            return relpath(candidate, repo_root)
    return path_value.replace("\\", "/") if path_value else None


def resize_image_to_training_size(source: Path, target: Path, image_size: tuple[int, int]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(source).convert("L")
    resized = ImageOps.contain(image, image_size, Image.Resampling.BILINEAR)
    canvas = Image.new("L", image_size, color=0)
    offset = ((image_size[0] - resized.width) // 2, (image_size[1] - resized.height) // 2)
    canvas.paste(resized, offset)
    canvas.save(target)


def materialize_training_image(
    path_value: str | None,
    output_dir: Path,
    repo_root: Path,
    run_id: str,
    image_group: str,
    image_size: tuple[int, int],
    cache: dict[str, str | None],
) -> str | None:
    if not path_value:
        return None
    if path_value in cache:
        return cache[path_value]

    source = Path(path_value)
    if not source.is_absolute():
        source = repo_root / source
    if not source.exists():
        cache[path_value] = None
        return None

    target = output_dir / "images" / run_id / image_group / source.name
    resize_image_to_training_size(source, target, image_size)
    resized_path = relpath(target, repo_root)
    cache[path_value] = resized_path
    return resized_path


def encode_primitive_action(action: dict[str, Any], key_to_id: dict[str, int]) -> tuple[list[float], str]:
    action_type = action.get("type")
    action_type_id = ACTION_TYPE_TO_ID.get(str(action_type), 0)
    if action_type == "MOVE_TO":
        x, y, coordinate_frame = get_xy_and_frame(action)
    else:
        x, y, coordinate_frame = float(MISSING), float(MISSING), "none"

    key = key_name(action)
    key_id = key_to_id.get(key, MISSING) if key else MISSING
    repeat_count = action.get("key_repeat_count", MISSING)
    key_interval = action.get("key_interval", MISSING)
    if action_type in {"PRESS_KEY", "HOTKEY"}:
        if repeat_count in {None, MISSING}:
            repeat_count = 1
        if key_interval in {None, MISSING}:
            key_interval = DEFAULT_KEY_INTERVAL_MS
    return (
        [
            float(action_type_id),
            float(x),
            float(y),
            float(key_id),
            float(repeat_count),
            float(key_interval),
        ],
        coordinate_frame,
    )


def summarize_resplan_for_model(
    resplan: dict[str, Any],
    primitive_plan: dict[str, Any],
    grounding_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compact structured input features. The full JSON path is also kept."""
    coordinate_system = resplan.get("coordinate_system", {})
    walls = resplan.get("walls") or resplan.get("wall_segments") or resplan.get("wall_center_lines") or []
    compression = infer_model_point_axis_compression(resplan, primitive_plan, grounding_config)
    compression_scale = compression["scale"]
    insertions = build_insertion_features(walls, coordinate_system, compression_scale)
    execution_walls = build_execution_wall_features(walls, coordinate_system, compression_scale)
    execution_coordinate_policy = build_execution_coordinate_policy(coordinate_system, compression)
    rooms = resplan.get("rooms") or []
    return {
        "metadata": resplan.get("metadata", {}),
        "task_entity_counts": normalize_task_entity_counts(resplan),
        "coordinate_system": coordinate_system,
        "execution_coordinate_policy": execution_coordinate_policy,
        "counts": {
            "rooms": len(rooms) if isinstance(rooms, list) else 0,
            "walls": len(walls) if isinstance(walls, list) else 0,
            "insertions": len(insertions),
        },
        "rooms": rooms,
        "walls": walls,
        "execution_walls": execution_walls,
        "insertions": insertions,
    }


def normalize_task_entity_counts(resplan: dict[str, Any]) -> dict[str, int]:
    counts = resplan.get("task_entity_counts") or {}
    walls = resplan.get("walls") or resplan.get("wall_segments") or resplan.get("wall_center_lines") or []
    insertions = resplan.get("insertions") or resplan.get("openings") or []

    result = {key: int(counts.get(key, 0) or 0) for key in PROGRESS_ENTITY_KEYS}
    if not result["exterior_walls"] or not result["interior_walls"]:
        for wall in walls if isinstance(walls, list) else []:
            location = str(wall.get("physical", {}).get("wall_location", "unknown")).lower()
            if location == "exterior":
                result["exterior_walls"] += 1
            elif location == "interior":
                result["interior_walls"] += 1
    if not result["windows"] or not result["doors"]:
        for insertion in insertions if isinstance(insertions, list) else []:
            opening_type = str(insertion.get("opening_type") or insertion.get("type") or "").lower()
            if opening_type == "window":
                result["windows"] += 1
            elif opening_type in {"door", "front_door"}:
                result["doors"] += 1
    if "slabs" not in counts:
        result["slabs"] = 1
    if "roofs" not in counts:
        result["roofs"] = 1
    return result


def progress_vector(task_counts: dict[str, int], done_counts: dict[str, int]) -> list[float]:
    vector = []
    for key in PROGRESS_ENTITY_KEYS:
        total = max(int(task_counts.get(key, 0) or 0), 0)
        done = max(int(done_counts.get(key, 0) or 0), 0)
        done_clamped = min(done, total) if total else done
        ratio = float(done_clamped) / float(total) if total else 1.0
        vector.extend([float(total) / 100.0, float(done_clamped) / 100.0, ratio])
    return vector


def build_progress_by_step(actions: list[dict[str, Any]], task_counts: dict[str, int]) -> list[dict[str, Any]]:
    entity_last_step: dict[tuple[str, str], int] = {}
    for index, action in enumerate(actions):
        progress_key = HIGH_LEVEL_TO_PROGRESS_KEY.get(high_level_name(action))
        if not progress_key:
            continue
        entity = action.get("parent_entity") or f"{progress_key}:{action.get('parent_high_level_index', index)}"
        entity_last_step[(progress_key, str(entity))] = index

    progress_rows = []
    for step_index in range(len(actions)):
        done_counts = {key: 0 for key in PROGRESS_ENTITY_KEYS}
        for (progress_key, _entity), last_step in entity_last_step.items():
            if last_step < step_index:
                done_counts[progress_key] += 1
        progress_rows.append(
            {
                "entity_order": PROGRESS_ENTITY_KEYS,
                "task_entity_counts": task_counts,
                "done_entity_counts": done_counts,
                "vector": progress_vector(task_counts, done_counts),
            }
        )
    return progress_rows


def model_point_axis_compression_scale(primitive_plan: dict[str, Any]) -> list[float]:
    compression = primitive_plan.get("execution_policy", {}).get("model_point_axis_compression") or {}
    scale = compression.get("scale")
    if isinstance(scale, list) and len(scale) >= 2:
        return [float(scale[0]), float(scale[1])]
    return [1.0, 1.0]


def apply_axis_compression(point: list[float], scale: list[float]) -> list[float]:
    return [round(float(point[0]) * float(scale[0]), 6), round(float(point[1]) * float(scale[1]), 6)]


def model_axis_safe_limits(grounding_config: dict[str, Any] | None) -> tuple[float, float] | None:
    if not grounding_config:
        return None
    canvas = grounding_config.get("canvas", {})
    model_range = canvas.get("model_range_mm", {})
    coordinate_mapping = canvas.get("coordinate_mapping", {})
    if not bool(coordinate_mapping.get("auto_compress_primitive_model_points", True)):
        return None
    scale_multiplier = float(coordinate_mapping.get("scale_multiplier", 1.0) or 1.0)
    if scale_multiplier <= 0:
        return None
    try:
        x_limit = min(abs(float(model_range["x_min"])), abs(float(model_range["x_max"]))) / scale_multiplier
        y_limit = min(abs(float(model_range["y_min"])), abs(float(model_range["y_max"]))) / scale_multiplier
    except KeyError:
        return None
    if x_limit <= 0 or y_limit <= 0:
        return None
    return x_limit, y_limit


def resplan_model_points_for_compression(resplan: dict[str, Any], coordinate_system: dict[str, Any]) -> list[list[float]]:
    walls = resplan.get("walls") or resplan.get("wall_segments") or resplan.get("wall_center_lines") or []
    points: list[list[float]] = []
    for wall in walls if isinstance(walls, list) else []:
        geometry = wall.get("geometry", {})
        for key in ("start", "end"):
            point = geometry.get(key)
            if isinstance(point, list) and len(point) >= 2:
                points.append(normalize_source_point_for_json(point, coordinate_system))
        for opening in wall.get("openings", []) or []:
            click_point = opening.get("insertion_point")
            if click_point:
                points.append(normalize_source_point_for_json(click_point, coordinate_system))
    slab_point = resplan.get("slab_generate_point")
    if isinstance(slab_point, list) and len(slab_point) >= 2:
        points.append(normalize_source_point_for_json(slab_point, coordinate_system))
    return points


def infer_model_point_axis_compression(
    resplan: dict[str, Any],
    primitive_plan: dict[str, Any],
    grounding_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    primitive_compression = primitive_plan.get("execution_policy", {}).get("model_point_axis_compression") or {}
    primitive_scale = primitive_compression.get("scale")
    if isinstance(primitive_scale, list) and len(primitive_scale) >= 2:
        return {
            "enabled": bool(primitive_compression.get("enabled", True)),
            "scale": [float(primitive_scale[0]), float(primitive_scale[1])],
            "source": "primitive_plan.execution_policy.model_point_axis_compression",
            "raw": primitive_compression,
        }

    limits = model_axis_safe_limits(grounding_config)
    if not limits:
        return {"enabled": False, "scale": [1.0, 1.0], "source": "none", "raw": {}}
    coordinate_system = resplan.get("coordinate_system", {})
    points = resplan_model_points_for_compression(resplan, coordinate_system)
    if not points:
        return {"enabled": False, "scale": [1.0, 1.0], "source": "none", "raw": {}}

    x_limit, y_limit = limits
    max_abs_x = max(abs(float(point[0])) for point in points)
    max_abs_y = max(abs(float(point[1])) for point in points)
    scale_x = x_limit / max_abs_x if max_abs_x > x_limit else 1.0
    scale_y = y_limit / max_abs_y if max_abs_y > y_limit else 1.0
    compression = {
        "enabled": scale_x < 0.999999 or scale_y < 0.999999,
        "scale": [round(scale_x, 6), round(scale_y, 6)],
        "source": "grounding_config.canvas.coordinate_mapping.auto_compress_primitive_model_points",
        "raw": {
            "safe_abs_limit": [round(x_limit, 6), round(y_limit, 6)],
            "original_max_abs": [round(max_abs_x, 6), round(max_abs_y, 6)],
        },
    }
    return compression


def build_execution_coordinate_policy(
    coordinate_system: dict[str, Any],
    compression: dict[str, Any],
) -> dict[str, Any]:
    return {
        "coordinate_space": "centered_normalized_execution_model",
        "normalization": {
            "method": "source_bbox_center_span",
            "source_bbox": coordinate_system.get("source_bbox"),
            "source_span": coordinate_system.get("source_span"),
            "x_range": coordinate_system.get("x_range"),
            "y_range": coordinate_system.get("y_range"),
        },
        "opening_offsets_source_units": {
            "window": DEFAULT_WINDOW_EDGE_OFFSET_SOURCE_UNITS,
            "door": DEFAULT_DOOR_EDGE_OFFSET_SOURCE_UNITS,
            "front_door": DEFAULT_DOOR_EDGE_OFFSET_SOURCE_UNITS,
        },
        "model_point_axis_compression": {
            "enabled": bool(compression.get("enabled", bool(compression))),
            "scale": compression.get("scale", [1.0, 1.0]),
            "source": compression.get("source", "none"),
            "raw": compression.get("raw", {}),
        },
    }


def build_execution_wall_features(
    walls: list[dict[str, Any]],
    coordinate_system: dict[str, Any],
    compression_scale: list[float],
) -> list[dict[str, Any]]:
    execution_walls = []
    for wall in walls:
        geometry = wall.get("geometry", {})
        start = geometry.get("start")
        end = geometry.get("end")
        if not start or not end:
            continue
        start_model = normalize_source_point_for_json(start, coordinate_system)
        end_model = normalize_source_point_for_json(end, coordinate_system)
        execution_walls.append(
            {
                "wall_id": wall.get("wall_id"),
                "wall_location": wall.get("physical", {}).get("wall_location", "unknown"),
                "source_start": [round(float(start[0]), 6), round(float(start[1]), 6)],
                "source_end": [round(float(end[0]), 6), round(float(end[1]), 6)],
                "start": start_model,
                "end": end_model,
                "execution_start": apply_axis_compression(start_model, compression_scale),
                "execution_end": apply_axis_compression(end_model, compression_scale),
            }
        )
    return execution_walls


def wall_unit_normal(wall: dict[str, Any]) -> tuple[float, float]:
    geometry = wall.get("geometry", {})
    start = geometry.get("start") or [0.0, 0.0]
    end = geometry.get("end") or [0.0, 0.0]
    dx = float(end[0]) - float(start[0])
    dy = float(end[1]) - float(start[1])
    length = (dx * dx + dy * dy) ** 0.5
    if length == 0:
        return 0.0, 1.0
    return dy / length, -dx / length


def offset_opening_click_point(opening: dict[str, Any], wall: dict[str, Any]) -> list[float] | None:
    centerline = opening.get("insertion_point")
    if not centerline:
        return None
    normal_x, normal_y = wall_unit_normal(wall)
    opening_type = str(opening.get("opening_type", "")).lower()
    edge_offset = DEFAULT_DOOR_EDGE_OFFSET_SOURCE_UNITS if "door" in opening_type else DEFAULT_WINDOW_EDGE_OFFSET_SOURCE_UNITS
    return [
        round(float(centerline[0]) + normal_x * edge_offset, 6),
        round(float(centerline[1]) + normal_y * edge_offset, 6),
    ]


def build_insertion_features(
    walls: list[dict[str, Any]],
    coordinate_system: dict[str, Any],
    compression_scale: list[float],
) -> list[dict[str, Any]]:
    insertions = []
    for wall in walls:
        wall_id = wall.get("wall_id")
        for opening in wall.get("openings", []) or []:
            centerline = opening.get("insertion_point")
            click_point = centerline
            if not centerline or not click_point:
                continue
            insertions.append(
                {
                    "opening_id": opening.get("opening_id"),
                    "opening_type": opening.get("opening_type"),
                    "host_wall_id": opening.get("host_wall_id") or wall_id,
                    "source_insertion_point": [round(float(centerline[0]), 6), round(float(centerline[1]), 6)],
                    "source_click_point": click_point,
                    "insertion_point": normalize_source_point_for_json(centerline, coordinate_system),
                    "click_point": normalize_source_point_for_json(click_point, coordinate_system),
                    "execution_click_point": apply_axis_compression(
                        normalize_source_point_for_json(click_point, coordinate_system), compression_scale
                    ),
                    "click_point_rule": (
                        "source opening insertion point, matching rule-base primitive target_click_point_source"
                    ),
                }
            )
    return insertions


def normalize_source_point_for_json(point: list[float], coordinate_system: dict[str, Any]) -> list[float]:
    if coordinate_system.get("input_coordinates") in {"model_units", "normalized", "centered_normalized"}:
        return [round(float(point[0]), 6), round(float(point[1]), 6)]

    bbox = coordinate_system.get("source_bbox", {})
    x_range = coordinate_system.get("x_range", [0, 1])
    y_range = coordinate_system.get("y_range", [0, 1])
    source_span = float(coordinate_system.get("source_span") or 0)
    if not source_span:
        source_span = max(float(x_range[1]) - float(x_range[0]), float(y_range[1]) - float(y_range[0]))
    if not source_span:
        return [0.0, 0.0]

    min_x = float(bbox.get("min_x", x_range[0]))
    max_x = float(bbox.get("max_x", x_range[1]))
    min_y = float(bbox.get("min_y", y_range[0]))
    max_y = float(bbox.get("max_y", y_range[1]))
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    return [
        round((float(point[0]) - center_x) / source_span, 6),
        round((float(point[1]) - center_y) / source_span, 6),
    ]


def sequence_stats(high_sequence: dict[str, Any], gui_sequence: dict[str, Any], primitive_plan: dict[str, Any]) -> dict[str, Any]:
    high_steps = high_sequence.get("sequence", [])
    gui_steps = gui_sequence.get("sequence", [])
    primitives = primitive_plan.get("primitives", [])
    return {
        "num_high_level_steps": len(high_steps),
        "num_gui_sequence_steps": sum(len(step.get("gui_action_sequence", [])) for step in gui_steps),
        "num_primitive_plan_steps": len(primitives),
        "primitive_plan_action_space": primitive_plan.get("primitive_action_space", []),
        "model_point_axis_compression": primitive_plan.get("execution_policy", {}).get("model_point_axis_compression"),
    }


def target_split_counts(total: int, split_ratios: dict[str, float] | None = None) -> dict[str, int]:
    ratios = split_ratios or SPLIT_RATIOS
    if total <= 0:
        return {split: 0 for split in SPLIT_ORDER}

    normalized_total = sum(ratios.get(split, 0.0) for split in SPLIT_ORDER)
    if normalized_total <= 0:
        raise ValueError("Split ratios must sum to a positive value")

    raw_counts = {split: total * ratios.get(split, 0.0) / normalized_total for split in SPLIT_ORDER}
    counts = {split: int(raw_counts[split]) for split in SPLIT_ORDER}
    remainder = total - sum(counts.values())
    fractions = sorted(
        SPLIT_ORDER,
        key=lambda split: (raw_counts[split] - counts[split], ratios.get(split, 0.0)),
        reverse=True,
    )
    for split in fractions[:remainder]:
        counts[split] += 1
    return counts


def build_run_split_stats(run: dict[str, Path | str]) -> dict[str, Any]:
    actions = condense_key_repeats(read_jsonl(Path(run["imitation_path"])))
    feature_counts: Counter[str] = Counter()
    for action in actions:
        feature_counts[f"action_type:{action.get('type') or '<none>'}"] += 1
        feature_counts[f"high_level_action:{high_level_name(action)}"] += 1
        feature_counts[f"gui_action:{gui_action_name(action)}"] += 1
    return {
        "run_id": str(run["run_id"]),
        "num_steps": len(actions),
        "feature_counts": feature_counts,
    }


def split_assignment_score(
    run_stats: dict[str, Any],
    split: str,
    target_run_counts: dict[str, int],
    target_step_counts: dict[str, float],
    target_feature_shares: dict[str, float],
    assigned_run_counts: dict[str, int],
    assigned_step_counts: dict[str, int],
    assigned_feature_counts: dict[str, Counter[str]],
    global_feature_counts: Counter[str],
) -> float:
    projected_run_count = assigned_run_counts[split] + 1
    projected_step_count = assigned_step_counts[split] + int(run_stats["num_steps"])
    run_fill = projected_run_count / max(target_run_counts[split], 1)
    step_fill = projected_step_count / max(target_step_counts[split], 1.0)

    score = 0.45 * run_fill + 0.45 * step_fill
    run_feature_counts: Counter[str] = run_stats["feature_counts"]
    for feature, count in run_feature_counts.items():
        global_count = global_feature_counts[feature]
        if global_count <= 0:
            continue
        projected_feature_count = assigned_feature_counts[split][feature] + count
        projected_share = projected_feature_count / global_count
        over_target_share = max(0.0, projected_share - target_feature_shares[split])
        score += 0.10 * over_target_share * over_target_share
    return score


def assign_balanced_splits(run_stats: list[dict[str, Any]]) -> dict[str, str]:
    target_run_counts = target_split_counts(len(run_stats))
    total_steps = sum(int(stats["num_steps"]) for stats in run_stats)
    global_feature_counts: Counter[str] = Counter()
    for stats in run_stats:
        global_feature_counts.update(stats["feature_counts"])

    target_feature_shares = {
        split: target_run_counts[split] / max(len(run_stats), 1) for split in SPLIT_ORDER
    }
    target_step_counts = {
        split: total_steps * target_feature_shares[split] for split in SPLIT_ORDER
    }
    assigned_run_counts = {split: 0 for split in SPLIT_ORDER}
    assigned_step_counts = {split: 0 for split in SPLIT_ORDER}
    assigned_feature_counts = {split: Counter() for split in SPLIT_ORDER}
    split_by_run: dict[str, str] = {}

    def rarity_score(stats: dict[str, Any]) -> float:
        return sum(
            count / max(global_feature_counts[feature], 1)
            for feature, count in stats["feature_counts"].items()
        )

    ordered_stats = sorted(
        run_stats,
        key=lambda stats: (-rarity_score(stats), -int(stats["num_steps"]), str(stats["run_id"])),
    )
    for stats in ordered_stats:
        candidates = [
            split for split in SPLIT_ORDER if assigned_run_counts[split] < target_run_counts[split]
        ]
        if not candidates:
            raise RuntimeError("No split has remaining capacity")
        split = min(
            candidates,
            key=lambda candidate: split_assignment_score(
                stats,
                candidate,
                target_run_counts,
                target_step_counts,
                target_feature_shares,
                assigned_run_counts,
                assigned_step_counts,
                assigned_feature_counts,
                global_feature_counts,
            ),
        )
        split_by_run[str(stats["run_id"])] = split
        assigned_run_counts[split] += 1
        assigned_step_counts[split] += int(stats["num_steps"])
        assigned_feature_counts[split].update(stats["feature_counts"])

    return split_by_run


def build_balanced_split_map(runs: list[dict[str, Path | str]]) -> dict[str, str]:
    return assign_balanced_splits([build_run_split_stats(run) for run in runs])


def build_split_distribution(index: list[dict[str, Any]], repo_root: Path) -> dict[str, Any]:
    distribution: dict[str, Any] = {}
    for item in index:
        split = item["split"]
        split_record = distribution.setdefault(
            split,
            {
                "num_runs": 0,
                "num_training_steps": 0,
                "samples": [],
                "high_level_action": {},
                "gui_action": {},
                "action_type": {},
            },
        )
        split_record["num_runs"] += 1
        split_record["num_training_steps"] += int(item["num_training_steps"])
        split_record["samples"].append(item["sample_id"])
        sample = read_json(repo_root / item["path"])
        for step in sample.get("steps", []):
            target = step.get("supervision_target", {})
            for key, value in [
                ("high_level_action", target.get("high_level_action")),
                ("gui_action", target.get("gui_action")),
                ("action_type", target.get("primitive_action", [None])[0]),
            ]:
                if value is None:
                    continue
                value = str(value)
                split_record[key][value] = split_record[key].get(value, 0) + 1
    return distribution


def build_vocab(
    runs: list[dict[str, Path | str]],
) -> tuple[dict[str, int], dict[str, int], dict[str, int], dict[str, Any]]:
    all_actions = []
    for run in runs:
        all_actions.extend(condense_key_repeats(read_jsonl(Path(run["imitation_path"]))))

    high_levels = [high_level_name(action) for action in all_actions]
    gui_actions = [gui_action_name(action) for action in all_actions]
    keys = collect_key_names(all_actions)

    high_level_to_id = stable_mapping(high_levels)
    gui_action_to_id = stable_mapping(gui_actions)
    key_to_id = stable_mapping(keys, include_none=False)

    vocab = {
        "action_type_to_id": ACTION_TYPE_TO_ID,
        "id_to_action_type": {str(key): value for key, value in ID_TO_ACTION_TYPE.items()},
        "coordinate_frame_to_id": COORDINATE_FRAME_TO_ID,
        "high_level_to_id": high_level_to_id,
        "gui_action_to_id": gui_action_to_id,
        "key_to_id": key_to_id,
        "counts": {
            "action_type": dict(Counter(action.get("type") for action in all_actions)),
            "high_level_action": dict(Counter(high_levels)),
            "gui_action": dict(Counter(gui_actions)),
            "key": dict(Counter(keys)),
        },
    }
    return high_level_to_id, gui_action_to_id, key_to_id, vocab


def transform_run(
    run: dict[str, Path | str],
    output_dir: Path,
    repo_root: Path,
    high_level_to_id: dict[str, int],
    gui_action_to_id: dict[str, int],
    key_to_id: dict[str, int],
    split: str,
    image_size: tuple[int, int] = DEFAULT_PROCESSED_IMAGE_SIZE,
    grounding_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run_id = str(run["run_id"])
    run_dir = Path(run["trajectory_dir"])
    raw_actions = read_jsonl(Path(run["imitation_path"]))
    actions = condense_key_repeats(raw_actions)
    resplan = read_json(Path(run["resplan_path"]))
    high_sequence = read_optional_json(Path(run["high_level_sequence_path"]), {"sequence": []})
    gui_sequence = read_optional_json(Path(run["gui_action_sequence_path"]), {"sequence": []})
    primitive_plan = read_optional_json(Path(run["primitive_plan_path"]), {})
    task_entity_counts = normalize_task_entity_counts(resplan)
    progress_by_step = build_progress_by_step(actions, task_entity_counts)
    image_cache: dict[str, str | None] = {}
    global_floorplan_raw_path = resolve_global_floorplan_path(run, repo_root)
    global_floorplan_path = materialize_training_image(
        global_floorplan_raw_path,
        output_dir=output_dir,
        repo_root=repo_root,
        run_id=run_id,
        image_group="global_floorplan",
        image_size=image_size,
        cache=image_cache,
    )

    steps = []
    trajectory_rows = []
    primitive_actions = []
    high_level_ids = []
    gui_action_ids = []
    coordinate_frame_ids = []
    flat_actions = []
    progress_vectors = []
    for step_index, action in enumerate(actions):
        high_name = high_level_name(action)
        gui_name = gui_action_name(action)
        high_level_id = high_level_to_id.get(high_name, 0)
        gui_action_id = gui_action_to_id.get(gui_name, 0)
        primitive_action, coordinate_frame = encode_primitive_action(action, key_to_id)
        coordinate_frame_id = COORDINATE_FRAME_TO_ID[coordinate_frame]
        flat_action = [float(high_level_id), float(gui_action_id), *primitive_action]
        task_progress = progress_by_step[step_index]
        raw_action_screenshot_path = resolve_screenshot_path(action.get("screenshot_path"), run_dir, repo_root)
        action_screenshot_path = materialize_training_image(
            raw_action_screenshot_path,
            output_dir=output_dir,
            repo_root=repo_root,
            run_id=run_id,
            image_group="screenshots",
            image_size=image_size,
            cache=image_cache,
        )
        observation_screenshot_path = action_screenshot_path

        primitive_actions.append(primitive_action)
        high_level_ids.append(high_level_id)
        gui_action_ids.append(gui_action_id)
        coordinate_frame_ids.append(coordinate_frame_id)
        flat_actions.append(flat_action)
        progress_vectors.append(task_progress["vector"])

        step = {
            "step_index": step_index,
            "run_id": run_id,
            "primitive_id": action.get("primitive_id"),
            "condensed_primitive_ids": action.get("condensed_primitive_ids"),
            "model_input": {
                "resplan_json_path": relpath(Path(run["resplan_path"]), repo_root),
                "global_floorplan_path": global_floorplan_path,
                "observation_screenshot_path": observation_screenshot_path,
                "task_progress": task_progress,
                "history_policy": "use previous steps in this sample only; do not include future sequence labels",
            },
            "supervision_target": {
                "high_level_action": high_name,
                "high_level_id": high_level_id,
                "gui_action": gui_name,
                "gui_action_id": gui_action_id,
                "coordinate_frame": coordinate_frame,
                "coordinate_frame_id": coordinate_frame_id,
                "primitive_action": primitive_action,
                "flat_action": flat_action,
                "progress_vector": task_progress["vector"],
            },
            "debug_provenance": {
                "raw_action_type": action.get("type"),
                "key": key_name(action),
                "parent_high_level_index": action.get("parent_high_level_index"),
                "parent_entity": action.get("parent_entity"),
                "model_point_mm": action.get("model_point_mm"),
                "source_insertion_point": action.get("source_insertion_point"),
                "source_click_point": action.get("source_click_point"),
                "window_point": action.get("resolved_window_point")
                or (action.get("coordinates", {}).get("window") if action.get("coordinates") else None),
                "window_point_norm": action.get("resolved_window_point_norm"),
                "raw_action_screenshot_path": raw_action_screenshot_path,
                "action_screenshot_path": action_screenshot_path,
                "time_s_before": action.get("time_s_before"),
                "time_s_after": action.get("time_s_after"),
                "timestamp_utc_before": action.get("timestamp_utc_before"),
                "timestamp_utc_after": action.get("timestamp_utc_after"),
            },
        }
        steps.append(step)
        trajectory_rows.append(
            {
                "step_index": step_index,
                "primitive_id": action.get("primitive_id"),
                "model_input": step["model_input"],
                "supervision_target": step["supervision_target"],
                "debug_provenance": step["debug_provenance"],
            }
        )

    teacher_forcing = {
        "mode": "autoregressive_next_step",
        "global_input": "model_input.resplan_json_path or encoded_resplan + global_floorplan_path",
        "per_step_input": "observation_screenshot_path + previous primitive/high/gui history",
        "target": "supervision_target at the current step",
        "no_leakage_rule": "Do not feed future trajectory rows or optional sequence/provenance files as model input.",
    }
    sample = {
        "sample_id": run_id,
        "run_id": run_id,
        "split": split,
        "schema": {
            "primitive_action": ["action_type", "x", "y", "key_pressed", "key_repeat_count", "key_interval"],
            "flat_action": [
                "high_level_id",
                "gui_action_id",
                "action_type",
                "x",
                "y",
                "key_pressed",
                "key_repeat_count",
                "key_interval",
            ],
        },
        "model_inputs": {
            "resplan_json_path": relpath(Path(run["resplan_path"]), repo_root),
            "global_floorplan_path": global_floorplan_path,
            "global_floorplan_raw_path": global_floorplan_raw_path,
            "processed_image_size": list(image_size),
            "task_entity_counts": task_entity_counts,
            "encoded_resplan": summarize_resplan_for_model(resplan, primitive_plan, grounding_config),
        },
        "supervision_sources": {
            "high_level_sequence_path": relpath_if_exists(Path(run["high_level_sequence_path"]), repo_root),
            "gui_action_sequence_path": relpath_if_exists(Path(run["gui_action_sequence_path"]), repo_root),
            "primitive_plan_path": relpath_if_exists(Path(run["primitive_plan_path"]), repo_root),
            "imitation_trajectory_path": relpath(Path(run["imitation_path"]), repo_root),
            "note": "Trajectory rows provide training labels. Sequence files are optional provenance only and are not inference-time model inputs.",
        },
        "sequence_stats": sequence_stats(high_sequence, gui_sequence, primitive_plan),
        "teacher_forcing": teacher_forcing,
        "steps": steps,
        "tensors": {
            "primitive_actions": primitive_actions,
            "high_level_ids": high_level_ids,
            "gui_action_ids": gui_action_ids,
            "coordinate_frame_ids": coordinate_frame_ids,
            "flat_actions": flat_actions,
            "progress": progress_vectors,
        },
    }

    sample_path = output_dir / "samples" / f"{run_id}.json"
    trajectory_path = output_dir / "trajectories" / f"{run_id}_training_steps.jsonl"
    tensor_path = output_dir / "tensors" / f"{run_id}.npz"
    runtime_plan_path = output_dir / "runtime_plans" / f"{run_id}_runtime_plan.json"
    write_json(sample_path, sample)
    write_jsonl(trajectory_path, trajectory_rows)
    write_json(
        runtime_plan_path,
        {
            "sample_id": run_id,
            "resplan_json_path": relpath(Path(run["resplan_path"]), repo_root),
            "global_floorplan_path": global_floorplan_path,
            "global_floorplan_raw_path": global_floorplan_raw_path,
            "processed_image_size": list(image_size),
            "task_entity_counts": task_entity_counts,
            "encoded_resplan": sample["model_inputs"]["encoded_resplan"],
            "execution_coordinate_policy": sample["model_inputs"]["encoded_resplan"]["execution_coordinate_policy"],
            "note": "Runtime coordinate metadata only. It contains coordinate transforms, not the action sequence.",
        },
    )
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        tensor_path,
        primitive_actions=np.asarray(primitive_actions, dtype=np.float32),
        high_level_ids=np.asarray(high_level_ids, dtype=np.int64),
        gui_action_ids=np.asarray(gui_action_ids, dtype=np.int64),
        coordinate_frame_ids=np.asarray(coordinate_frame_ids, dtype=np.int64),
        flat_actions=np.asarray(flat_actions, dtype=np.float32),
        progress=np.asarray(progress_vectors, dtype=np.float32),
    )

    observation_missing = sum(
        1 for row in trajectory_rows if not row["model_input"].get("observation_screenshot_path")
    )
    global_floorplan_missing = 0 if global_floorplan_path else 1
    action_screenshot_missing = sum(
        1 for row in trajectory_rows if not row["debug_provenance"].get("action_screenshot_path")
    )
    return {
        "sample_id": run_id,
        "split": split,
        "path": relpath(sample_path, repo_root),
        "trajectory_path": relpath(trajectory_path, repo_root),
        "tensor_path": relpath(tensor_path, repo_root),
        "runtime_plan_path": relpath(runtime_plan_path, repo_root),
        "resplan_json_path": relpath(Path(run["resplan_path"]), repo_root),
        "global_floorplan_path": global_floorplan_path,
        "global_floorplan_raw_path": global_floorplan_raw_path,
        "processed_image_size": list(image_size),
        "task_entity_counts": task_entity_counts,
        "supervision_sources": {
            "high_level_sequence_path": relpath_if_exists(Path(run["high_level_sequence_path"]), repo_root),
            "gui_action_sequence_path": relpath_if_exists(Path(run["gui_action_sequence_path"]), repo_root),
            "primitive_plan_path": relpath_if_exists(Path(run["primitive_plan_path"]), repo_root),
        },
        "num_raw_steps": len(raw_actions),
        "num_training_steps": len(actions),
        "missing_observation_screenshots": observation_missing,
        "missing_global_floorplan": global_floorplan_missing,
        "missing_action_screenshots": action_screenshot_missing,
        "missing_action_after_screenshots": action_screenshot_missing,
    }


def transform_dataset(
    raw_data_dir: Path,
    output_dir: Path,
    repo_root: Path,
    overwrite: bool = False,
    image_size: tuple[int, int] = DEFAULT_PROCESSED_IMAGE_SIZE,
    grounding_config_path: Path | None = DEFAULT_GROUNDING_CONFIG_PATH,
) -> None:
    runs = discover_runs(raw_data_dir)
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    grounding_config = read_optional_json(repo_root / grounding_config_path, {}) if grounding_config_path else {}

    high_level_to_id, gui_action_to_id, key_to_id, vocab = build_vocab(runs)
    write_json(output_dir / "action_vocab.json", vocab)

    split_by_run = build_balanced_split_map(runs)
    index = []
    for run in runs:
        split = split_by_run[str(run["run_id"])]
        index.append(
            transform_run(
                run=run,
                output_dir=output_dir,
                repo_root=repo_root,
                high_level_to_id=high_level_to_id,
                gui_action_to_id=gui_action_to_id,
                key_to_id=key_to_id,
                split=split,
                image_size=image_size,
                grounding_config=grounding_config,
            )
        )

    summary = {
        "raw_data_dir": relpath(raw_data_dir, repo_root),
        "output_dir": relpath(output_dir, repo_root),
        "num_runs": len(index),
        "num_raw_steps": sum(item["num_raw_steps"] for item in index),
        "num_training_steps": sum(item["num_training_steps"] for item in index),
        "split_distribution": build_split_distribution(index, repo_root),
        "split_policy": {
            "unit": "trajectory_run",
            "target_ratios": SPLIT_RATIOS,
            "target_run_counts": target_split_counts(len(runs)),
            "strategy": (
                "greedy stratified assignment by condensed primitive action type, "
                "high-level action, GUI action, and step count"
            ),
        },
        "primitive_action_dim": 6,
        "flat_action_dim": 8,
        "progress_feature_dim": len(PROGRESS_ENTITY_KEYS) * 3,
        "processed_image_size": list(image_size),
        "grounding_config_path": relpath_if_exists(repo_root / grounding_config_path, repo_root)
        if grounding_config_path
        else None,
        "missing_global_floorplans": sum(item["missing_global_floorplan"] for item in index),
        "training_contract": {
            "inference_inputs": [
                "resplan_json_or_encoded_plan",
                "global_floorplan_image",
                "current_screenshot",
                "historical_actions",
            ],
            "training_targets": [
                "next_high_level_id",
                "next_gui_action_id",
                "next_primitive_action",
            ],
            "sequence_files_are": "supervision/provenance, not model inputs",
        },
        "schema": {
            "primitive_action": ["action_type", "x", "y", "key_pressed", "key_repeat_count", "key_interval"],
            "flat_action": [
                "high_level_id",
                "gui_action_id",
                "action_type",
                "x",
                "y",
                "key_pressed",
                "key_repeat_count",
                "key_interval",
            ],
        },
    }
    write_json(output_dir / "dataset_index.json", index)
    write_json(output_dir / "summary.json", summary)
    print(
        f"Processed {summary['num_runs']} runs, {summary['num_raw_steps']} raw steps "
        f"-> {summary['num_training_steps']} training steps into {output_dir}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-data-dir", type=Path, default=Path("raw_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("processed_data"))
    parser.add_argument("--image-width", type=int, default=DEFAULT_PROCESSED_IMAGE_SIZE[0])
    parser.add_argument("--image-height", type=int, default=DEFAULT_PROCESSED_IMAGE_SIZE[1])
    parser.add_argument("--grounding-config", type=Path, default=DEFAULT_GROUNDING_CONFIG_PATH)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd().resolve()
    transform_dataset(
        raw_data_dir=(repo_root / args.raw_data_dir).resolve(),
        output_dir=(repo_root / args.output_dir).resolve(),
        repo_root=repo_root,
        overwrite=args.overwrite,
        image_size=(args.image_width, args.image_height),
        grounding_config_path=args.grounding_config,
    )


if __name__ == "__main__":
    main()
