#!/usr/bin/env python3
"""Select a rollout-oriented checkpoint from saved validation logs."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any


LOWER_IS_BETTER = {
    "loss": 0.45,
    "xy_mae": 0.20,
}
HIGHER_IS_BETTER = {
    "aux_wall_acc": 0.15,
    "aux_point_role_acc": 0.10,
    "action_type_acc": 0.04,
    "high_level_acc": 0.03,
    "gui_action_acc": 0.03,
}


def _epoch_from_val_log(path: Path) -> int:
    match = re.fullmatch(r"val_epoch_(\d+)\.json", path.name)
    if match is None:
        raise ValueError(f"Unexpected validation log name: {path.name}")
    return int(match.group(1))


def _load_candidates(run_dir: Path) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    logs_dir = run_dir / "logs"
    checkpoint_dir = run_dir / "checkpoints"
    for log_path in sorted(logs_dir.glob("val_epoch_*.json"), key=_epoch_from_val_log):
        epoch = _epoch_from_val_log(log_path)
        checkpoint_path = checkpoint_dir / f"epoch_{epoch:04d}.pt"
        if not checkpoint_path.exists():
            continue
        metrics = json.loads(log_path.read_text())
        candidates.append(
            {
                "epoch": epoch,
                "log_path": str(log_path),
                "checkpoint_path": str(checkpoint_path),
                "metrics": metrics,
            }
        )
    if not candidates:
        raise FileNotFoundError(
            f"No periodic checkpoint candidates found in {run_dir}. "
            "Expected matching logs/val_epoch_*.json and checkpoints/epoch_XXXX.pt files."
        )
    return candidates


def _normalized_scores(values: list[float], higher_is_better: bool) -> list[float]:
    low = min(values)
    high = max(values)
    if high == low:
        return [1.0 for _ in values]
    if higher_is_better:
        return [(value - low) / (high - low) for value in values]
    return [(high - value) / (high - low) for value in values]


def score_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scored = [dict(candidate, proxy_score=0.0, score_parts={}) for candidate in candidates]
    metric_specs = [(name, weight, False) for name, weight in LOWER_IS_BETTER.items()]
    metric_specs += [(name, weight, True) for name, weight in HIGHER_IS_BETTER.items()]

    for metric_name, weight, higher_is_better in metric_specs:
        available = [
            (idx, float(candidate["metrics"][metric_name]))
            for idx, candidate in enumerate(scored)
            if isinstance(candidate["metrics"].get(metric_name), (int, float))
        ]
        if not available:
            continue
        indices = [idx for idx, _ in available]
        values = [value for _, value in available]
        normalized = _normalized_scores(values, higher_is_better=higher_is_better)
        for idx, normalized_value in zip(indices, normalized):
            contribution = weight * normalized_value
            scored[idx]["proxy_score"] += contribution
            scored[idx]["score_parts"][metric_name] = contribution

    return sorted(scored, key=lambda item: (-item["proxy_score"], item["epoch"]))


def write_selection(run_dir: Path, selected: dict[str, Any], top: list[dict[str, Any]], output_name: str) -> Path:
    checkpoint_dir = run_dir / "checkpoints"
    output_path = checkpoint_dir / output_name
    shutil.copy2(selected["checkpoint_path"], output_path)
    metadata = {
        "selected_epoch": selected["epoch"],
        "selected_checkpoint": selected["checkpoint_path"],
        "output_checkpoint": str(output_path),
        "proxy_score": selected["proxy_score"],
        "score_parts": selected["score_parts"],
        "metrics": selected["metrics"],
        "top_candidates": [
            {
                "epoch": item["epoch"],
                "checkpoint_path": item["checkpoint_path"],
                "proxy_score": item["proxy_score"],
                "score_parts": item["score_parts"],
                "metrics": item["metrics"],
            }
            for item in top
        ],
    }
    metadata_path = checkpoint_dir / f"{output_path.stem}_selection.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    return output_path


def select_checkpoint(run_dir: Path, output_name: str = "best_proxy.pt", top_k: int = 10) -> Path:
    run_dir = run_dir.resolve()
    candidates = _load_candidates(run_dir)
    ranked = score_candidates(candidates)
    selected = ranked[0]
    output_path = write_selection(run_dir, selected, ranked[:top_k], output_name)

    print(f"Selected epoch {selected['epoch']} as {output_path}")
    print(f"proxy_score={selected['proxy_score']:.6f}")
    for key, value in selected["score_parts"].items():
        print(f"  {key}: {value:.6f}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Training run directory containing logs/ and checkpoints/.")
    parser.add_argument("--output-name", default="best_proxy.pt", help="Checkpoint filename to create.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of candidates to include in metadata.")
    args = parser.parse_args()

    select_checkpoint(args.run_dir, output_name=args.output_name, top_k=args.top_k)


if __name__ == "__main__":
    main()
