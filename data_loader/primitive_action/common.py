"""Shared primitive-action dataloader core."""

from typing import Any

from data_loader.data_loader import (
    PrimitiveActionDataset,
    create_dataloader,
    create_dataset_from_config,
)

ALLOWED_DATASET_KWARGS = {
    "repo_root",
    "action_vocab_path",
    "image_size",
    "history_length",
    "load_observation",
    "load_images",
    "load_global_floorplan",
    "image_dtype",
    "normalize_images",
    "include_raw",
}


def filter_dataset_kwargs(dataset_kwargs: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in dataset_kwargs.items() if key in ALLOWED_DATASET_KWARGS}


__all__ = [
    "ALLOWED_DATASET_KWARGS",
    "PrimitiveActionDataset",
    "create_dataloader",
    "create_dataset_from_config",
    "filter_dataset_kwargs",
]
