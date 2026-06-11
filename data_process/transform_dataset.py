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


MISSING = -1

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


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


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
    return path.resolve().relative_to(repo_root).as_posix()


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
    sequence_root = raw_data_dir / "bim_sequence"
    run_dirs = sorted(
        path
        for path in trajectory_root.glob("plan_*")
        if path.is_dir() and (path / "imitation_trajectory.jsonl").exists()
    )
    runs = []
    for run_dir in run_dirs:
        suffix = plan_suffix_from_run_dir(run_dir)
        resplan_path = resplan_root / f"resplan_to_JSON_{suffix}.json"
        sequence_dir = sequence_root / f"resplan_to_JSON_{suffix}_bim_sequence"
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
        missing = [str(path) for key, path in paths.items() if key not in {"run_id", "suffix"} and not Path(path).exists()]
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
            current["key_interval"] = round(sum(intervals) / len(intervals), 3) if intervals else MISSING
            current["condensed_primitive_ids"] = [item.get("primitive_id") for item in group]
            current["timestamp_utc_after"] = group[-1].get("timestamp_utc_after")
            current["timestamp_utc"] = group[-1].get("timestamp_utc")
            current["time_s_after"] = group[-1].get("time_s_after")
            current["time_s"] = group[-1].get("time_s")
            current["screenshot_path"] = group[-1].get("screenshot_path") or current.get("screenshot_path")
        else:
            current["key_repeat_count"] = 1 if current.get("type") in {"PRESS_KEY", "HOTKEY"} else MISSING
            current["key_interval"] = MISSING

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


def summarize_resplan_for_model(resplan: dict[str, Any], primitive_plan: dict[str, Any]) -> dict[str, Any]:
    """Compact structured input features. The full JSON path is also kept."""
    coordinate_system = resplan.get("coordinate_system", {})
    walls = resplan.get("walls") or resplan.get("wall_segments") or resplan.get("wall_center_lines") or []
    compression_scale = model_point_axis_compression_scale(primitive_plan)
    insertions = build_insertion_features(walls, coordinate_system, compression_scale)
    execution_walls = build_execution_wall_features(walls, coordinate_system, compression_scale)
    execution_coordinate_policy = build_execution_coordinate_policy(coordinate_system, primitive_plan)
    entity_points = build_entity_points(execution_walls, insertions, primitive_plan)
    rooms = resplan.get("rooms") or []
    return {
        "metadata": resplan.get("metadata", {}),
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
        "entity_points": entity_points,
    }


def model_point_axis_compression_scale(primitive_plan: dict[str, Any]) -> list[float]:
    compression = primitive_plan.get("execution_policy", {}).get("model_point_axis_compression") or {}
    scale = compression.get("scale")
    if isinstance(scale, list) and len(scale) >= 2:
        return [float(scale[0]), float(scale[1])]
    return [1.0, 1.0]


def apply_axis_compression(point: list[float], scale: list[float]) -> list[float]:
    return [round(float(point[0]) * float(scale[0]), 6), round(float(point[1]) * float(scale[1]), 6)]


def build_execution_coordinate_policy(
    coordinate_system: dict[str, Any],
    primitive_plan: dict[str, Any],
) -> dict[str, Any]:
    compression = primitive_plan.get("execution_policy", {}).get("model_point_axis_compression") or {}
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
            "scale": model_point_axis_compression_scale(primitive_plan),
            "source": "primitive_plan.execution_policy.model_point_axis_compression",
            "raw": compression,
        },
    }


def build_entity_points(
    execution_walls: list[dict[str, Any]],
    insertions: list[dict[str, Any]],
    primitive_plan: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    entity_points: dict[str, dict[str, Any]] = {}
    for wall in execution_walls:
        wall_id = wall.get("wall_id")
        if not wall_id:
            continue
        entity_points[str(wall_id)] = {
            "entity_type": "wall",
            "start_mm": wall.get("execution_start"),
            "end_mm": wall.get("execution_end"),
            "source_start": wall.get("source_start"),
            "source_end": wall.get("source_end"),
        }
    for insertion in insertions:
        opening_id = insertion.get("opening_id")
        if not opening_id:
            continue
        entity_points[str(opening_id)] = {
            "entity_type": insertion.get("opening_type") or "opening",
            "intersection_point": insertion.get("execution_click_point"),
            "insertion_point": insertion.get("execution_click_point"),
            "centerline_insertion_point": insertion.get("insertion_point"),
            "source_insertion_point": insertion.get("source_insertion_point"),
            "source_click_point": insertion.get("source_click_point"),
            "host_wall_id": insertion.get("host_wall_id"),
        }
    if primitive_plan:
        for primitive in primitive_plan.get("primitives", []) or []:
            entity_id = primitive.get("target_entity")
            point_role = primitive.get("point_role")
            model_point = primitive.get("model_point_mm")
            if not entity_id or not point_role or not isinstance(model_point, list) or len(model_point) < 2:
                continue
            record = entity_points.setdefault(str(entity_id), {"entity_type": str(entity_id).split("_", 1)[0]})
            record[str(point_role)] = [round(float(model_point[0]), 6), round(float(model_point[1]), 6)]
    return entity_points


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
            click_point = offset_opening_click_point(opening, wall)
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
                        "source insertion point + wall normal * edge offset "
                        "(window=1.5 source units, door/front_door=1.0 source unit)"
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


def split_for_index(index: int, total: int) -> str:
    if total <= 1:
        return "train"
    ratio = index / total
    if ratio < 0.8:
        return "train"
    if ratio < 0.9:
        return "val"
    return "test"


def action_target_entity(action: dict[str, Any]) -> str:
    entity = action.get("target_entity")
    return str(entity) if entity else "<none>"


def action_point_role(action: dict[str, Any]) -> str:
    role = action.get("point_role")
    return str(role) if role else "<none>"


def build_vocab(
    runs: list[dict[str, Path | str]],
) -> tuple[dict[str, int], dict[str, int], dict[str, int], dict[str, int], dict[str, int], dict[str, Any]]:
    all_actions = []
    for run in runs:
        all_actions.extend(condense_key_repeats(read_jsonl(Path(run["imitation_path"]))))

    high_levels = [action.get("parent_high_level_type") or "<none>" for action in all_actions]
    gui_actions = [action.get("parent_gui_action") or "<none>" for action in all_actions]
    keys = collect_key_names(all_actions)
    target_entities = [action_target_entity(action) for action in all_actions]
    point_roles = [action_point_role(action) for action in all_actions]

    high_level_to_id = stable_mapping(high_levels)
    gui_action_to_id = stable_mapping(gui_actions)
    key_to_id = stable_mapping(keys, include_none=False)
    target_entity_to_id = stable_mapping(target_entities)
    point_role_to_id = stable_mapping(point_roles)

    vocab = {
        "action_type_to_id": ACTION_TYPE_TO_ID,
        "id_to_action_type": {str(key): value for key, value in ID_TO_ACTION_TYPE.items()},
        "coordinate_frame_to_id": COORDINATE_FRAME_TO_ID,
        "high_level_to_id": high_level_to_id,
        "gui_action_to_id": gui_action_to_id,
        "key_to_id": key_to_id,
        "target_entity_to_id": target_entity_to_id,
        "point_role_to_id": point_role_to_id,
        "counts": {
            "action_type": dict(Counter(action.get("type") for action in all_actions)),
            "high_level_action": dict(Counter(high_levels)),
            "gui_action": dict(Counter(gui_actions)),
            "key": dict(Counter(keys)),
            "target_entity": dict(Counter(target_entities)),
            "point_role": dict(Counter(point_roles)),
        },
    }
    return high_level_to_id, gui_action_to_id, key_to_id, target_entity_to_id, point_role_to_id, vocab


def transform_run(
    run: dict[str, Path | str],
    output_dir: Path,
    repo_root: Path,
    high_level_to_id: dict[str, int],
    gui_action_to_id: dict[str, int],
    key_to_id: dict[str, int],
    target_entity_to_id: dict[str, int],
    point_role_to_id: dict[str, int],
    split: str,
) -> dict[str, Any]:
    run_id = str(run["run_id"])
    run_dir = Path(run["trajectory_dir"])
    raw_actions = read_jsonl(Path(run["imitation_path"]))
    actions = condense_key_repeats(raw_actions)
    resplan = read_json(Path(run["resplan_path"]))
    high_sequence = read_json(Path(run["high_level_sequence_path"]))
    gui_sequence = read_json(Path(run["gui_action_sequence_path"]))
    primitive_plan = read_json(Path(run["primitive_plan_path"]))

    steps = []
    trajectory_rows = []
    primitive_actions = []
    high_level_ids = []
    gui_action_ids = []
    coordinate_frame_ids = []
    target_entity_ids = []
    point_role_ids = []
    flat_actions = []
    last_screenshot_path = None

    for step_index, action in enumerate(actions):
        high_level_name = action.get("parent_high_level_type") or "<none>"
        gui_action_name = action.get("parent_gui_action") or "<none>"
        high_level_id = high_level_to_id.get(high_level_name, 0)
        gui_action_id = gui_action_to_id.get(gui_action_name, 0)
        primitive_action, coordinate_frame = encode_primitive_action(action, key_to_id)
        coordinate_frame_id = COORDINATE_FRAME_TO_ID[coordinate_frame]
        target_entity = action_target_entity(action)
        point_role = action_point_role(action)
        target_entity_id = target_entity_to_id.get(target_entity, 0)
        point_role_id = point_role_to_id.get(point_role, 0)
        flat_action = [float(high_level_id), float(gui_action_id), *primitive_action]
        action_screenshot_path = resolve_screenshot_path(action.get("screenshot_path"), run_dir, repo_root)
        observation_screenshot_path = last_screenshot_path or action_screenshot_path

        primitive_actions.append(primitive_action)
        high_level_ids.append(high_level_id)
        gui_action_ids.append(gui_action_id)
        coordinate_frame_ids.append(coordinate_frame_id)
        target_entity_ids.append(target_entity_id)
        point_role_ids.append(point_role_id)
        flat_actions.append(flat_action)

        step = {
            "step_index": step_index,
            "run_id": run_id,
            "primitive_id": action.get("primitive_id"),
            "condensed_primitive_ids": action.get("condensed_primitive_ids"),
            "model_input": {
                "resplan_json_path": relpath(Path(run["resplan_path"]), repo_root),
                "observation_screenshot_path": observation_screenshot_path,
                "history_policy": "use previous steps in this sample only; do not include future sequence labels",
            },
            "supervision_target": {
                "high_level_action": high_level_name,
                "high_level_id": high_level_id,
                "gui_action": gui_action_name,
                "gui_action_id": gui_action_id,
                "coordinate_frame": coordinate_frame,
                "coordinate_frame_id": coordinate_frame_id,
                "target_entity": target_entity,
                "target_entity_id": target_entity_id,
                "point_role": point_role,
                "point_role_id": point_role_id,
                "primitive_action": primitive_action,
                "flat_action": flat_action,
            },
            "debug_provenance": {
                "raw_action_type": action.get("type"),
                "key": key_name(action),
                "parent_high_level_index": action.get("parent_high_level_index"),
                "parent_entity": action.get("parent_entity"),
                "target_entity": action.get("target_entity"),
                "point_role": action.get("point_role"),
                "model_point_mm": action.get("model_point_mm"),
                "source_insertion_point": action.get("source_insertion_point"),
                "source_click_point": action.get("source_click_point"),
                "window_point": action.get("resolved_window_point")
                or (action.get("coordinates", {}).get("window") if action.get("coordinates") else None),
                "window_point_norm": action.get("resolved_window_point_norm"),
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

        if action_screenshot_path:
            last_screenshot_path = action_screenshot_path

    teacher_forcing = {
        "mode": "autoregressive_next_step",
        "global_input": "model_input.resplan_json_path or encoded_resplan",
        "per_step_input": "observation_screenshot_path + previous primitive/high/gui history",
        "target": "supervision_target at the current step",
        "no_leakage_rule": "Do not feed full high_level_sequence/gui_action_sequence/primitive_plan as model input.",
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
            "encoded_resplan": summarize_resplan_for_model(resplan, primitive_plan),
        },
        "supervision_sources": {
            "high_level_sequence_path": relpath(Path(run["high_level_sequence_path"]), repo_root),
            "gui_action_sequence_path": relpath(Path(run["gui_action_sequence_path"]), repo_root),
            "primitive_plan_path": relpath(Path(run["primitive_plan_path"]), repo_root),
            "imitation_trajectory_path": relpath(Path(run["imitation_path"]), repo_root),
            "note": "These sequence files provide labels/provenance only. They are not inference-time model inputs.",
        },
        "sequence_stats": sequence_stats(high_sequence, gui_sequence, primitive_plan),
        "teacher_forcing": teacher_forcing,
        "steps": steps,
        "tensors": {
            "primitive_actions": primitive_actions,
            "high_level_ids": high_level_ids,
            "gui_action_ids": gui_action_ids,
            "coordinate_frame_ids": coordinate_frame_ids,
            "target_entity_ids": target_entity_ids,
            "point_role_ids": point_role_ids,
            "flat_actions": flat_actions,
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
            "encoded_resplan": sample["model_inputs"]["encoded_resplan"],
            "execution_coordinate_policy": sample["model_inputs"]["encoded_resplan"]["execution_coordinate_policy"],
            "entity_points": sample["model_inputs"]["encoded_resplan"]["entity_points"],
            "note": "Runtime coordinate metadata only. It contains coordinate transforms and entity point lookup, not the action sequence.",
        },
    )
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        tensor_path,
        primitive_actions=np.asarray(primitive_actions, dtype=np.float32),
        high_level_ids=np.asarray(high_level_ids, dtype=np.int64),
        gui_action_ids=np.asarray(gui_action_ids, dtype=np.int64),
        coordinate_frame_ids=np.asarray(coordinate_frame_ids, dtype=np.int64),
        target_entity_ids=np.asarray(target_entity_ids, dtype=np.int64),
        point_role_ids=np.asarray(point_role_ids, dtype=np.int64),
        flat_actions=np.asarray(flat_actions, dtype=np.float32),
    )

    observation_missing = sum(
        1 for row in trajectory_rows if not row["model_input"].get("observation_screenshot_path")
    )
    action_after_missing = sum(
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
        "supervision_sources": {
            "high_level_sequence_path": relpath(Path(run["high_level_sequence_path"]), repo_root),
            "gui_action_sequence_path": relpath(Path(run["gui_action_sequence_path"]), repo_root),
            "primitive_plan_path": relpath(Path(run["primitive_plan_path"]), repo_root),
        },
        "num_raw_steps": len(raw_actions),
        "num_training_steps": len(actions),
        "missing_observation_screenshots": observation_missing,
        "missing_action_after_screenshots": action_after_missing,
    }


def transform_dataset(raw_data_dir: Path, output_dir: Path, repo_root: Path, overwrite: bool = False) -> None:
    runs = discover_runs(raw_data_dir)
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    high_level_to_id, gui_action_to_id, key_to_id, target_entity_to_id, point_role_to_id, vocab = build_vocab(runs)
    write_json(output_dir / "action_vocab.json", vocab)

    index = []
    for run_index, run in enumerate(runs):
        split = split_for_index(run_index, len(runs))
        index.append(
            transform_run(
                run=run,
                output_dir=output_dir,
                repo_root=repo_root,
                high_level_to_id=high_level_to_id,
                gui_action_to_id=gui_action_to_id,
                key_to_id=key_to_id,
                target_entity_to_id=target_entity_to_id,
                point_role_to_id=point_role_to_id,
                split=split,
            )
        )

    summary = {
        "raw_data_dir": relpath(raw_data_dir, repo_root),
        "output_dir": relpath(output_dir, repo_root),
        "num_runs": len(index),
        "num_raw_steps": sum(item["num_raw_steps"] for item in index),
        "num_training_steps": sum(item["num_training_steps"] for item in index),
        "primitive_action_dim": 6,
        "flat_action_dim": 8,
        "training_contract": {
            "inference_inputs": ["resplan_json", "current_screenshot", "historical_actions"],
            "training_targets": [
                "next_high_level_id",
                "next_gui_action_id",
                "next_primitive_action",
                "next_target_entity_id",
                "next_point_role_id",
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
    )


if __name__ == "__main__":
    main()
