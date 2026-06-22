"""Structured no-screenshot primitive-action dataloader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from data_paths import DEFAULT_STRUCTURED_DATASET_PATH
from data_loader.data_loader import DEFAULT_IMAGE_SIZE, PrimitiveActionDataset
from data_loader.primitive_action.common import filter_dataset_kwargs


class StructuredPrimitiveActionDataset(PrimitiveActionDataset):
    """Primitive-action dataset profile for structured plan/history inputs only."""

    def __init__(
        self,
        dataset_path: str | Path = DEFAULT_STRUCTURED_DATASET_PATH,
        split: str | None = None,
        repo_root: str | Path = ".",
        action_vocab_path: str | Path | None = None,
        image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
        history_length: int = 32,
        load_observation: bool = False,
        load_images: bool = False,
        load_global_floorplan: bool = False,
        normalize_images: bool = True,
        image_dtype: str | torch.dtype = torch.float32,
        include_raw: bool = False,
    ) -> None:
        super().__init__(
            dataset_path=dataset_path,
            split=split,
            repo_root=repo_root,
            action_vocab_path=action_vocab_path,
            image_size=image_size,
            history_length=history_length,
            load_observation=load_observation,
            load_images=load_images,
            load_global_floorplan=load_global_floorplan,
            normalize_images=normalize_images,
            image_dtype=image_dtype,
            include_raw=include_raw,
        )


def create_structured_dataloader(
    dataset_path: str | Path = DEFAULT_STRUCTURED_DATASET_PATH,
    split: str | None = "train",
    batch_size: int = 8,
    shuffle: bool | None = None,
    num_workers: int = 0,
    **dataset_kwargs: Any,
) -> DataLoader:
    dataset_kwargs = filter_dataset_kwargs(dataset_kwargs)
    dataset = StructuredPrimitiveActionDataset(dataset_path=dataset_path, split=split, **dataset_kwargs)
    if shuffle is None:
        shuffle = split == "train"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=PrimitiveActionDataset.collate_fn,
        pin_memory=torch.cuda.is_available(),
    )


def create_structured_dataset_from_config(
    dataset_path: str | Path = DEFAULT_STRUCTURED_DATASET_PATH,
    config: str | Path | None = None,
    batch_size: int = 8,
    num_workers: int = 0,
    splits: tuple[str, ...] = ("train", "val", "test"),
    overfit: bool = False,
    **dataset_kwargs: Any,
) -> tuple[dict[str, Any], ...]:
    del config
    dataset_kwargs = filter_dataset_kwargs(dataset_kwargs)
    packets = []
    for split in splits:
        effective_split = None if overfit else split
        loader = create_structured_dataloader(
            dataset_path=dataset_path,
            split=effective_split,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=split == "train",
            **dataset_kwargs,
        )
        packets.append({"loader": loader, "sampler": None})
    return tuple(packets)


__all__ = [
    "DEFAULT_STRUCTURED_DATASET_PATH",
    "StructuredPrimitiveActionDataset",
    "create_structured_dataloader",
    "create_structured_dataset_from_config",
]
