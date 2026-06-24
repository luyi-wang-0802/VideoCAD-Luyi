#!/usr/bin/env python3
"""Launch the regularized structured primitive-action policy experiment."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from select_structured_policy_checkpoint import select_checkpoint


REPO_ROOT = Path(__file__).resolve().parent
RUN_NAME_PREFIX = "structured_primitive_action_policy_regularized_"


def _training_command() -> list[str]:
    return [
        sys.executable,
        "main.py",
        "--gpu_ids",
        "0",
        "--dataset_path",
        "/home/ray/data/vectorworks/processed_data/structured_primitive_action_policy",
        "--model_config",
        "model_configs/structured_primitive_action_policy_regularized.json",
        "--model_name",
        "structured_primitive_action_policy_regularized",
        "--output_dir",
        "outputs",
        "--training_runs_dir",
        "training_runs",
        "--epochs",
        "800",
        "--batch_size",
        "128",
        "--lr",
        "1e-4",
        "--num_workers",
        "15",
        "--cache_dataset_on_device",
        "true",
        "--early_stopping_enabled",
        "true",
        "--early_stopping_patience",
        "80",
        "--early_stopping_start_epoch",
        "150",
        "--early_stopping_min_delta",
        "0.001",
        "--early_stopping_metric",
        "loss",
        "--early_stopping_mode",
        "min",
        "--val_frequency",
        "5",
        "--use_wandb",
        "true",
        "--wandb_project",
        "videocad-primitive-action",
        "--wandb_entity",
        "luyi-wang-technical-university-of-munich",
        "--wandb_run_name",
        "structured_primitive_action_policy_regularizedforgeneralization_bs128_lr1e-4_drop0.3_aux0.05_ep800_pat80_loss",
    ]


def _latest_regularized_run() -> Path | None:
    runs_dir = REPO_ROOT / "outputs" / "training_runs"
    candidates = [
        path
        for path in runs_dir.glob(f"{RUN_NAME_PREFIX}*")
        if path.is_dir()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def main() -> None:
    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(REPO_ROOT) if not existing_pythonpath else f"{REPO_ROOT}:{existing_pythonpath}"

    subprocess.run(_training_command(), cwd=REPO_ROOT, env=env, check=True)

    latest_run = _latest_regularized_run()
    if latest_run is not None:
        select_checkpoint(latest_run)


if __name__ == "__main__":
    main()
