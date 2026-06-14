"""Utilities for keeping small datasets resident on the training device."""

from __future__ import annotations

import random
from typing import Any, Iterable

import torch
from torch.utils.data import RandomSampler


def move_to_device(value: Any, device: torch.device | str) -> Any:
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value


class DeviceCachedDataLoader:
    """A DataLoader-like wrapper that builds batches from preloaded device samples."""

    def __init__(
        self,
        samples: list[Any],
        batch_size: int,
        collate_fn,
        drop_last: bool = False,
        shuffle_samples: bool = False,
    ) -> None:
        self.samples = samples
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.drop_last = drop_last
        self.shuffle_samples = shuffle_samples

    @classmethod
    def from_loader(
        cls,
        loader: Iterable[Any],
        device: torch.device | str,
    ) -> "DeviceCachedDataLoader":
        if not hasattr(loader, "dataset"):
            raise TypeError("DeviceCachedDataLoader.from_loader expects a torch DataLoader with a dataset")
        dataset = loader.dataset
        samples = [move_to_device(dataset[index], device) for index in range(len(dataset))]
        shuffle_samples = isinstance(getattr(loader, "sampler", None), RandomSampler)
        return cls(
            samples=samples,
            batch_size=loader.batch_size,
            collate_fn=loader.collate_fn,
            drop_last=loader.drop_last,
            shuffle_samples=shuffle_samples,
        )

    def __iter__(self):
        order = list(range(len(self.samples)))
        if self.shuffle_samples:
            random.shuffle(order)
        for start in range(0, len(order), self.batch_size):
            indices = order[start : start + self.batch_size]
            if self.drop_last and len(indices) < self.batch_size:
                continue
            yield self.collate_fn([self.samples[index] for index in indices])

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.samples) // self.batch_size
        return (len(self.samples) + self.batch_size - 1) // self.batch_size
