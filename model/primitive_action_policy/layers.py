"""Small reusable layers/helpers for primitive-action policy models."""

from __future__ import annotations

import torch
import torch.nn as nn


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if values.shape[1] == 0:
        return torch.zeros(values.shape[0], values.shape[-1], device=values.device, dtype=values.dtype)
    mask_f = mask.to(values.dtype).unsqueeze(-1)
    denom = mask_f.sum(dim=1).clamp_min(1.0)
    return (values * mask_f).sum(dim=1) / denom


def make_image_encoder(image_channels: int, hidden_size: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(image_channels, 32, kernel_size=7, stride=2, padding=3),
        nn.GroupNorm(8, 32),
        nn.GELU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
        nn.GroupNorm(8, 64),
        nn.GELU(),
        nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
        nn.GroupNorm(8, 128),
        nn.GELU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(128, hidden_size),
        nn.LayerNorm(hidden_size),
    )
