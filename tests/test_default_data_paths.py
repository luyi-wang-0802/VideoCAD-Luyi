from pathlib import Path
import json
import os
import subprocess
import sys


def test_vectorworks_data_paths_point_to_external_data_root() -> None:
    from data_paths import (
        DEFAULT_RAW_DATA_DIR,
        DEFAULT_STRUCTURED_DATASET_PATH,
        DEFAULT_VISUAL_DATASET_PATH,
        VECTORWORKS_DATA_ROOT,
    )

    assert VECTORWORKS_DATA_ROOT == Path("/home/ray/data/vectorworks")
    assert DEFAULT_RAW_DATA_DIR == Path("/home/ray/data/vectorworks/raw_data")
    assert DEFAULT_STRUCTURED_DATASET_PATH == Path(
        "/home/ray/data/vectorworks/processed_data/structured_primitive_action_policy"
    )
    assert DEFAULT_VISUAL_DATASET_PATH == Path(
        "/home/ray/data/vectorworks/processed_data/visual_primitive_action_policy"
    )


def test_structured_model_config_uses_external_processed_dataset() -> None:
    config = json.loads(Path("model_configs/structured_primitive_action_policy.json").read_text())
    params = config["default_params"]

    assert params["dataset_path"] == "/home/ray/data/vectorworks/processed_data/structured_primitive_action_policy"
    assert (
        params["action_vocab_path"]
        == "/home/ray/data/vectorworks/processed_data/structured_primitive_action_policy/action_vocab.json"
    )


def test_transform_dataset_script_runs_directly_from_repo_root() -> None:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "data_process/transform_dataset.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert "--raw-data-dir" in result.stdout
