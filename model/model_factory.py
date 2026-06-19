from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

import torch

from model.structured_primitive_action_policy import (
    StructuredPrimitiveActionPolicyConfig,
    StructuredPrimitiveActionPolicyModel,
)


class ModelType(Enum):
    STRUCTURED_PRIMITIVE_ACTION_POLICY = "structured_primitive_action_policy"


def _load_vocab_counts(model_config: dict[str, Any]) -> dict[str, int]:
    vocab_path = model_config.get("action_vocab_path")
    if not vocab_path:
        dataset_path = model_config.get("dataset_path") or "processed_data"
        vocab_path = Path(dataset_path) / "action_vocab.json"
    vocab_path = Path(vocab_path)
    if not vocab_path.exists():
        return {}

    vocab = json.loads(vocab_path.read_text(encoding="utf-8-sig"))
    action_type_to_id = vocab.get("action_type_to_id", {})
    high_level_to_id = vocab.get("high_level_to_id", {})
    gui_action_to_id = vocab.get("gui_action_to_id", {})
    key_to_id = vocab.get("key_to_id", {})
    return {
        "num_action_types": max(action_type_to_id.values(), default=0) + 1,
        "num_high_level_actions": max(high_level_to_id.values(), default=0) + 1,
        "num_gui_actions": max(gui_action_to_id.values(), default=0) + 1,
        "num_keys": len(key_to_id),
    }


def _strip_wrappers(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module._orig_mod."):
            new_state_dict[key.replace("module._orig_mod.", "")] = value
        elif key.startswith("module."):
            new_state_dict[key.replace("module.", "")] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


class ModelFactory:
    def load_model(self, model_name: str, model_path: str | Path, device: str | torch.device):
        ckpt = torch.load(model_path, map_location=device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model, model_type = self.create_model(model_name, ckpt.get("model_config", {}), device)
        model.load_state_dict(_strip_wrappers(state_dict), strict=False)
        return model, model_type

    def create_model(
        self,
        model_name: str,
        model_config: dict[str, Any],
        device: str | torch.device,
        state_dict: dict[str, torch.Tensor] | None = None,
    ):
        if model_name == "structured_primitive_action_policy":
            config_kwargs = dict(model_config)
            config_kwargs.update({k: v for k, v in _load_vocab_counts(model_config).items() if v})
            allowed_keys = set(StructuredPrimitiveActionPolicyConfig.__dataclass_fields__.keys())
            config = StructuredPrimitiveActionPolicyConfig(
                **{k: v for k, v in config_kwargs.items() if k in allowed_keys}
            )
            model = StructuredPrimitiveActionPolicyModel(config).to(device)
            model_type = ModelType.STRUCTURED_PRIMITIVE_ACTION_POLICY
        else:
            raise ValueError(
                f"Unsupported model_name={model_name!r}. "
                "This project currently keeps only model_name='structured_primitive_action_policy'."
            )

        if state_dict:
            print("Loading state dict")
            model.load_state_dict(_strip_wrappers(state_dict), strict=False)
        return model, model_type
