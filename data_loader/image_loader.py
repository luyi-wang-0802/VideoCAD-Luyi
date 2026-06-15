"""Image loading helpers for collected Vectorworks screenshots."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


class ScreenshotImageLoader:
    """Load action-aligned screenshots as normalized tensors."""

    def __init__(
        self,
        repo_root: str | Path = ".",
        image_size: tuple[int, int] = (384, 216),
        load_images: bool = True,
        normalize: bool = True,
        channels: int = 1,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.image_size = image_size
        self.load_images = load_images
        self.normalize = normalize
        self.channels = channels
        self.dtype = dtype
        if channels not in {1, 3}:
            raise ValueError("channels must be 1 or 3")

    def resolve(self, path: str | Path | None) -> Path | None:
        if not path:
            return None
        resolved = Path(path)
        if resolved.is_absolute():
            return resolved
        return self.repo_root / resolved

    def empty(self) -> torch.Tensor:
        width, height = self.image_size
        return torch.zeros((self.channels, height, width), dtype=self.dtype)

    def load(self, path: str | Path | None) -> tuple[torch.Tensor, bool]:
        if not self.load_images:
            return self.empty(), False

        resolved = self.resolve(path)
        if resolved is None or not resolved.exists():
            return self.empty(), False

        mode = "L" if self.channels == 1 else "RGB"
        image = Image.open(resolved).convert(mode).resize(self.image_size, Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        if self.channels == 1:
            tensor = torch.from_numpy(array).unsqueeze(0)
        else:
            tensor = torch.from_numpy(array).permute(2, 0, 1)
        if self.normalize:
            tensor = (tensor - 0.5) / 0.5
        tensor = tensor.to(dtype=self.dtype)
        return tensor, True
