"""Prepare low-level GUI sequence data from collected Vectorworks runs.

The resulting dataset treats pointer movement as executor grounding, not as a
policy action. Labels therefore contain only state-changing GUI primitives:
CLICK, DOUBLE_CLICK, HOTKEY, and PRESS_KEY. For click actions, the preceding
MOVE_TO context is attached as the click target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


ACTION_TYPES = ["CLICK", "DOUBLE_CLICK", "HOTKEY", "PRESS_KEY"]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def plan_digits(value: str) -> str:
    digits = "".join(ch for ch in value if ch.isdigit())
    if not digits:
        raise ValueError(f"Cannot derive numeric plan id from {value!r}")
    return str(int(digits))


def normalized_plan_id(value: str, width: int = 5) -> str:
    return f"plan_{int(plan_digits(value)):0{width}d}"


def path_posix(path: Path) -> str:
    return str(path).replace("\\", "/")


def stable_sort_key(plan_id: str) -> str:
    return hashlib.sha1(plan_id.encode("utf-8")).hexdigest()


def find_run_dir(runs_root: Path, plan_id: str, width: int) -> Path | None:
    numeric = int(plan_digits(plan_id))
    candidates = [
        runs_root / f"plan_{numeric}",
        runs_root / f"plan_{numeric:0{width}d}",
        runs_root / plan_id,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def compact_source_plan(source: dict[str, Any]) -> dict[str, Any]:
    """Keep the structured geometry needed for GUI grounding.

    Rooms are intentionally omitted. Openings are reduced to insertion targets,
    and front_door is merged into door.
    """

    walls = []
    insertions = []
    for wall in source.get("walls", []):
        walls.append(
            {
                "wall_id": wall.get("wall_id"),
                "wall_location": wall.get("physical", {}).get("wall_location", "unknown"),
                "start": wall.get("geometry", {}).get("start"),
                "end": wall.get("geometry", {}).get("end"),
            }
        )
        for opening in wall.get("openings", []) or []:
            opening_type = opening.get("opening_type")
            if opening_type == "front_door":
                opening_type = "door"
            if opening_type not in {"door", "window"}:
                continue
            insertions.append(
                {
                    "opening_id": opening.get("opening_id"),
                    "opening_type": opening_type,
                    "host_wall_id": opening.get("host_wall_id", wall.get("wall_id")),
                    "insertion_point": opening.get("insertion_point"),
                }
            )

    metadata = source.get("metadata", {})
    quality_check = source.get("quality_check", {})
    return {
        "metadata": {
            "plan_id": metadata.get("plan_id"),
            "plan_index": metadata.get("plan_index"),
            "source_dataset": metadata.get("source_dataset"),
            "unit": metadata.get("unit"),
        },
        "coordinate_system": source.get("coordinate_system", {}),
        "quality_ok": quality_check.get("ok"),
        "walls": walls,
        "insertions": insertions,
        "counts": {
            "walls": len(walls),
            "insertions": len(insertions),
            "doors": sum(1 for item in insertions if item["opening_type"] == "door"),
            "windows": sum(1 for item in insertions if item["opening_type"] == "window"),
        },
    }


def click_target_from_record(record: dict[str, Any]) -> dict[str, Any]:
    move = record.get("preceding_move") or {}
    coordinates = move.get("coordinates") or record.get("coordinates") or {}
    window = coordinates.get("window") or {}
    screen = coordinates.get("screen") or {}
    target: dict[str, Any] = {
        "target_entity": move.get("target_entity") or record.get("target_entity"),
        "point_role": move.get("point_role") or record.get("point_role"),
        "model_point": move.get("model_point_mm") or record.get("model_point_mm"),
        "screen_point": [screen.get("x"), screen.get("y")] if screen else None,
        "window_point": [window.get("x"), window.get("y")] if window else None,
        "window_norm": [window.get("x_norm"), window.get("y_norm")] if window else None,
        "clamped_to_canvas": move.get("clamped_to_canvas"),
    }
    return {key: value for key, value in target.items() if value is not None}


def action_label(record: dict[str, Any]) -> dict[str, Any] | None:
    action_type = record.get("type")
    if action_type not in ACTION_TYPES:
        return None

    label: dict[str, Any] = {
        "primitive_id": record.get("primitive_id"),
        "action_type": action_type,
        "parent_high_level_index": record.get("parent_high_level_index"),
        "parent_high_level_type": record.get("parent_high_level_type"),
        "parent_entity": record.get("parent_entity"),
        "parent_gui_action": record.get("parent_gui_action"),
        "point_role": record.get("point_role"),
        "screenshot_path": record.get("screenshot_path"),
        "time_s_after": record.get("time_s_after"),
    }

    if action_type in {"CLICK", "DOUBLE_CLICK"}:
        label["button"] = record.get("button", "left")
        label["target"] = click_target_from_record(record)
    elif action_type == "HOTKEY":
        label["keys"] = record.get("keys", [])
    elif action_type == "PRESS_KEY":
        label["key"] = record.get("key")

    return label


def point_target(
    target_entity: str | None,
    point_role: str,
    model_point: list[float] | None,
) -> dict[str, Any]:
    target: dict[str, Any] = {
        "target_entity": target_entity,
        "point_role": point_role,
        "model_point": model_point,
    }
    return {key: value for key, value in target.items() if value is not None}


def attach_interior_wall_enter_targets(
    labels: list[dict[str, Any]],
    entities: dict[str, Any],
) -> None:
    """Attach the hidden end-point MOVE_TO to the first interior-wall Enter.

    Interior wall execution is MOVE_TO(start) -> CLICK -> MOVE_TO(end) ->
    PRESS_KEY(enter) -> PRESS_KEY(enter). Since MOVE_TO is not a policy action,
    the first Enter must carry the end-point target for the executor.
    """

    for label in labels:
        if (
            label.get("action_type") == "PRESS_KEY"
            and label.get("key") == "enter"
            and label.get("parent_high_level_type") == "CREATE_INTERIOR_WALL"
            and label.get("parent_gui_action") == "DRAW_WALL_FROM_ENTITY_GEOMETRY"
            and label.get("point_role") == "interior_wall_end_enter_confirm_1"
        ):
            entity_id = label.get("parent_entity")
            geometry = entities.get(entity_id, {}).get("geometry", {}) if entity_id else {}
            end_point = geometry.get("end_mm")
            label["target"] = point_target(entity_id, "end_mm", end_point)
            label["executor_note"] = "move_to_target_before_press_key"


def attach_observations(labels: list[dict[str, Any]]) -> None:
    """Attach screenshot observations around each policy action.

    The collector captures screenshots after state-changing primitives. For
    behavior cloning, the best available observation before action t is the
    screenshot after action t-1. The first action has no recorded before image.
    """

    previous_screenshot = None
    for index, label in enumerate(labels):
        current_screenshot = label.get("screenshot_path")
        label["step_index"] = index
        label["observation_before"] = {
            "screenshot_path": previous_screenshot,
            "source": "previous_action_after_screenshot" if previous_screenshot else "not_recorded",
        }
        label["observation_after"] = {
            "screenshot_path": current_screenshot,
            "source": "current_action_after_screenshot",
        }
        previous_screenshot = current_screenshot or previous_screenshot


def label_text(label: dict[str, Any]) -> str:
    action_type = label["action_type"]
    if action_type in {"CLICK", "DOUBLE_CLICK"}:
        target = label.get("target", {})
        point = target.get("model_point") or target.get("window_norm") or target.get("screen_point")
        return f"{action_type} at {point}"
    if action_type == "PRESS_KEY" and "target" in label:
        target = label.get("target", {})
        point = target.get("model_point") or target.get("window_norm") or target.get("screen_point")
        return f"PRESS_KEY {label.get('key')} at {point}"
    if action_type == "HOTKEY":
        return f"HOTKEY {'+'.join(label.get('keys', []))}"
    return f"PRESS_KEY {label.get('key')}"


def build_sample(
    source_path: Path,
    sequence_root: Path,
    runs_root: Path,
    output_id_width: int,
    exclude_quality_fail: bool,
) -> tuple[dict[str, Any] | None, str | None]:
    source = read_json(source_path)
    raw_plan_id = source.get("metadata", {}).get("plan_id") or source_path.stem
    sample_id = normalized_plan_id(raw_plan_id, output_id_width)

    if exclude_quality_fail and source.get("quality_check", {}).get("ok") is False:
        return None, f"excluded_quality_fail:{sample_id}"

    sequence_dir = sequence_root / f"{source_path.stem}_bim_sequence"
    gui_path = sequence_dir / "gui_action_sequence.json"
    high_path = sequence_dir / "high_level_sequence.json"
    run_dir = find_run_dir(runs_root, raw_plan_id, output_id_width)
    if not gui_path.exists():
        return None, f"missing_gui_sequence:{sample_id}"
    if not high_path.exists():
        return None, f"missing_high_level_sequence:{sample_id}"
    if run_dir is None:
        return None, f"missing_run:{sample_id}"

    trajectory_path = run_dir / "imitation_trajectory.jsonl"
    if not trajectory_path.exists():
        return None, f"missing_trajectory:{sample_id}"

    labels = []
    for record in read_jsonl(trajectory_path):
        label = action_label(record)
        if label is not None:
            labels.append(label)

    if not labels:
        return None, f"empty_labels:{sample_id}"

    gui_sequence = read_json(gui_path)
    high_sequence = read_json(high_path)
    attach_interior_wall_enter_targets(labels, high_sequence.get("entities", {}))
    attach_observations(labels)
    for label in labels:
        label["label_text"] = label_text(label)
    return (
        {
            "sample_id": sample_id,
            "raw_plan_id": raw_plan_id,
            "source_json": path_posix(source_path),
            "high_level_sequence_path": path_posix(high_path),
            "gui_action_sequence_path": path_posix(gui_path),
            "trajectory_path": path_posix(trajectory_path),
            "compact_plan": compact_source_plan(source),
            "high_level_sequence": high_sequence.get("sequence", []),
            "gui_sequence": gui_sequence.get("sequence", []),
            "low_level_actions": labels,
        },
        None,
    )


def assign_splits(samples: list[dict[str, Any]], train_ratio: float, val_ratio: float, seed: int) -> dict[str, str]:
    shuffled = sorted(samples, key=lambda sample: stable_sort_key(sample["sample_id"]))
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    if total >= 3:
        train_count = max(train_count, 1)
        val_count = max(val_count, 1)
        test_count = total - train_count - val_count
        if test_count < 1:
            train_count = max(total - 2, 1)
            val_count = 1
    split = {}
    for index, sample in enumerate(shuffled):
        if index < train_count:
            split[sample["sample_id"]] = "train"
        elif index < train_count + val_count:
            split[sample["sample_id"]] = "val"
        else:
            split[sample["sample_id"]] = "test"
    return split


def make_vocab(samples: list[dict[str, Any]]) -> dict[str, Any]:
    action_counter = Counter()
    key_counter = Counter()
    gui_counter = Counter()
    high_counter = Counter()
    point_role_counter = Counter()
    for sample in samples:
        for label in sample["low_level_actions"]:
            action_counter[label["action_type"]] += 1
            gui_counter[label.get("parent_gui_action")] += 1
            high_counter[label.get("parent_high_level_type")] += 1
            target = label.get("target", {})
            if target.get("point_role"):
                point_role_counter[target["point_role"]] += 1
            if label["action_type"] == "HOTKEY":
                key_counter["+".join(label.get("keys", []))] += 1
            if label["action_type"] == "PRESS_KEY":
                key_counter[label.get("key")] += 1
    return {
        "action_types": ACTION_TYPES,
        "action_counts": dict(action_counter),
        "key_counts": dict(key_counter),
        "parent_gui_action_counts": dict(gui_counter),
        "parent_high_level_type_counts": dict(high_counter),
        "point_role_counts": dict(point_role_counter),
    }


def write_sample_files(output_dir: Path, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for sample in sorted(samples, key=lambda item: item["sample_id"]):
        sample_path = samples_dir / f"{sample['sample_id']}.json"
        write_json(sample_path, sample)
        index.append(
            {
                "sample_id": sample["sample_id"],
                "raw_plan_id": sample["raw_plan_id"],
                "split": sample["split"],
                "path": path_posix(sample_path),
                "source_json": sample["source_json"],
                "low_level_action_count": len(sample["low_level_actions"]),
                "screenshot_count": sum(
                    1
                    for action in sample["low_level_actions"]
                    if action.get("observation_after", {}).get("screenshot_path")
                ),
                "high_level_step_count": len(sample["high_level_sequence"]),
                "quality_ok": sample["compact_plan"].get("quality_ok"),
            }
        )
    return index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-json-dir", default="resplan_to_JSON", type=Path)
    parser.add_argument("--sequence-root", default="outputs/bim_sequences", type=Path)
    parser.add_argument("--runs-root", default="outputs/runs", type=Path)
    parser.add_argument("--output-dir", default="data_process/low_level_gui_sequence/results", type=Path)
    parser.add_argument("--id-width", default=5, type=int)
    parser.add_argument("--train-ratio", default=0.8, type=float)
    parser.add_argument("--val-ratio", default=0.1, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--exclude-quality-fail", action="store_true")
    args = parser.parse_args()

    samples = []
    skipped = Counter()
    for source_path in sorted(args.source_json_dir.glob("*.json")):
        sample, reason = build_sample(
            source_path=source_path,
            sequence_root=args.sequence_root,
            runs_root=args.runs_root,
            output_id_width=args.id_width,
            exclude_quality_fail=args.exclude_quality_fail,
        )
        if sample is None:
            skipped[reason or "unknown"] += 1
        else:
            samples.append(sample)

    if not samples:
        raise SystemExit("No samples were prepared. Check input paths and generated runs.")

    split = assign_splits(samples, args.train_ratio, args.val_ratio, args.seed)
    for sample in samples:
        sample["split"] = split[sample["sample_id"]]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    index = write_sample_files(args.output_dir, samples)
    write_json(args.output_dir / "dataset_index.json", index)
    write_json(args.output_dir / "dataset_split.json", split)
    write_json(args.output_dir / "action_vocab.json", make_vocab(samples))
    legacy_jsonl = args.output_dir / "dataset.jsonl"
    if legacy_jsonl.exists():
        legacy_jsonl.unlink()

    action_count = sum(len(sample["low_level_actions"]) for sample in samples)
    split_counts = Counter(split.values())
    summary = {
        "sample_count": len(samples),
        "low_level_action_count": action_count,
        "split_counts": dict(split_counts),
        "id_width": args.id_width,
        "excluded_action_types": ["MOVE_TO", "SLEEP", "KEY_DOWN", "KEY_UP", "TYPE_TEXT", "SCROLL"],
        "policy_action_types": ACTION_TYPES,
        "skipped": dict(skipped),
        "files": {
            "samples_dir": path_posix(args.output_dir / "samples"),
            "dataset_index": path_posix(args.output_dir / "dataset_index.json"),
            "dataset_split": path_posix(args.output_dir / "dataset_split.json"),
            "action_vocab": path_posix(args.output_dir / "action_vocab.json"),
        },
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
