import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_process.transform_dataset import (
    DATASET_PROFILE_STRUCTURED,
    DATASET_PROFILE_VISUAL,
    condense_key_repeats,
    default_output_dir_for_profile,
    target_split_counts,
    transform_run,
)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 9), color=(255, 255, 255)).save(path)


def test_target_split_counts_use_7_2_1_ratio() -> None:
    assert target_split_counts(301) == {"train": 211, "val": 60, "test": 30}


def test_dataset_profile_default_output_dirs_are_separate() -> None:
    assert default_output_dir_for_profile(DATASET_PROFILE_STRUCTURED) == Path(
        "processed_data/structured_primitive_action_policy"
    )
    assert default_output_dir_for_profile(DATASET_PROFILE_VISUAL) == Path(
        "processed_data/visual_primitive_action_policy"
    )


def test_key_repeat_uses_first_pre_action_screenshot_and_default_interval() -> None:
    actions = [
        {
            "type": "PRESS_KEY",
            "key": "a",
            "parent_high_level_type": "draw",
            "parent_gui_action": "type_name",
            "screenshot_path": "before_first.png",
        },
        {
            "type": "PRESS_KEY",
            "key": "a",
            "parent_high_level_type": "draw",
            "parent_gui_action": "type_name",
            "screenshot_path": "before_second.png",
        },
    ]

    [condensed] = condense_key_repeats(actions)

    assert condensed["key_repeat_count"] == 2
    assert condensed["key_interval"] == 100.0
    assert condensed["screenshot_path"] == "before_first.png"


def test_transform_run_uses_current_action_screenshot_as_observation(tmp_path: Path) -> None:
    repo_root = tmp_path
    raw_dir = tmp_path / "raw_data"
    run_dir = raw_dir / "trajectory_data" / "plan_0001"
    output_dir = tmp_path / "processed_data"
    before_move = run_dir / "screenshots" / "before_move.png"
    before_click = run_dir / "screenshots" / "before_click.png"
    floorplan = run_dir / "global_floorplan" / "floorplan_before_roof.png"
    for image_path in (before_move, before_click, floorplan):
        write_image(image_path)

    resplan_path = raw_dir / "resplan_to_JSON" / "resplan_to_JSON_001.json"
    write_json(
        resplan_path,
        {
            "coordinate_system": {"input_coordinates": "normalized"},
            "walls": [],
            "rooms": [],
        },
    )
    write_jsonl(
        run_dir / "imitation_trajectory.jsonl",
        [
            {
                "type": "MOVE_TO",
                "model_point_mm": [0.1, 0.2],
                "parent_high_level_type": "draw",
                "parent_gui_action": "move",
                "screenshot_path": str(before_move.relative_to(repo_root)),
            },
            {
                "type": "CLICK",
                "parent_high_level_type": "draw",
                "parent_gui_action": "click",
                "screenshot_path": str(before_click.relative_to(repo_root)),
            },
        ],
    )

    transform_run(
        {
            "run_id": "plan_0001",
            "trajectory_dir": run_dir,
            "imitation_path": run_dir / "imitation_trajectory.jsonl",
            "run_summary_path": run_dir / "run_summary.json",
            "resplan_path": resplan_path,
            "high_level_sequence_path": run_dir / "missing_high_level.json",
            "gui_action_sequence_path": run_dir / "missing_gui.json",
            "primitive_plan_path": run_dir / "missing_primitive.json",
        },
        output_dir=output_dir,
        repo_root=repo_root,
        high_level_to_id={"draw": 1},
        gui_action_to_id={"move": 1, "click": 2},
        key_to_id={},
        split="train",
        image_size=(16, 9),
    )

    sample = json.loads((output_dir / "samples" / "plan_0001.json").read_text(encoding="utf-8"))
    first_observation = sample["steps"][0]["model_input"]["observation_screenshot_path"]
    second_observation = sample["steps"][1]["model_input"]["observation_screenshot_path"]

    assert first_observation.endswith("processed_data/images/plan_0001/screenshots/before_move.png")
    assert second_observation.endswith("processed_data/images/plan_0001/screenshots/before_click.png")


def test_structured_transform_run_does_not_materialize_or_expose_image_inputs(tmp_path: Path) -> None:
    repo_root = tmp_path
    raw_dir = tmp_path / "raw_data"
    run_dir = raw_dir / "trajectory_data" / "plan_0001"
    output_dir = tmp_path / "processed_data" / "structured_primitive_action_policy"
    before_move = run_dir / "screenshots" / "before_move.png"
    floorplan = run_dir / "global_floorplan" / "floorplan_before_roof.png"
    for image_path in (before_move, floorplan):
        write_image(image_path)

    resplan_path = raw_dir / "resplan_to_JSON" / "resplan_to_JSON_001.json"
    write_json(
        resplan_path,
        {
            "coordinate_system": {"input_coordinates": "normalized"},
            "walls": [],
            "rooms": [],
        },
    )
    write_jsonl(
        run_dir / "imitation_trajectory.jsonl",
        [
            {
                "type": "MOVE_TO",
                "model_point_mm": [0.1, 0.2],
                "parent_high_level_type": "draw",
                "parent_gui_action": "move",
                "screenshot_path": str(before_move.relative_to(repo_root)),
            },
        ],
    )

    transform_run(
        {
            "run_id": "plan_0001",
            "trajectory_dir": run_dir,
            "imitation_path": run_dir / "imitation_trajectory.jsonl",
            "run_summary_path": run_dir / "run_summary.json",
            "resplan_path": resplan_path,
            "high_level_sequence_path": run_dir / "missing_high_level.json",
            "gui_action_sequence_path": run_dir / "missing_gui.json",
            "primitive_plan_path": run_dir / "missing_primitive.json",
        },
        output_dir=output_dir,
        repo_root=repo_root,
        high_level_to_id={"draw": 1},
        gui_action_to_id={"move": 1},
        key_to_id={},
        split="train",
        image_size=(16, 9),
        dataset_profile=DATASET_PROFILE_STRUCTURED,
        materialize_images=False,
    )

    sample = json.loads((output_dir / "samples" / "plan_0001.json").read_text(encoding="utf-8"))
    model_inputs = sample["model_inputs"]
    step_model_input = sample["steps"][0]["model_input"]
    debug = sample["steps"][0]["debug_provenance"]

    assert not (output_dir / "images").exists()
    assert "global_floorplan_path" not in model_inputs
    assert "global_floorplan_raw_path" not in model_inputs
    assert "processed_image_size" not in model_inputs
    assert "global_floorplan_path" not in step_model_input
    assert "observation_screenshot_path" not in step_model_input
    assert debug["raw_action_screenshot_path"].endswith("raw_data/trajectory_data/plan_0001/screenshots/before_move.png")
    assert debug["action_screenshot_path"] is None
