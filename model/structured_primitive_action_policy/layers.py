"""Small reusable layers/helpers for structured primitive-action policy models."""

from __future__ import annotations

import torch


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if values.shape[1] == 0:
        return torch.zeros(values.shape[0], values.shape[-1], device=values.device, dtype=values.dtype)
    mask_f = mask.to(values.dtype).unsqueeze(-1)
    denom = mask_f.sum(dim=1).clamp_min(1.0)
    return (values * mask_f).sum(dim=1) / denom
