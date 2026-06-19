"""Configuration for the structured primitive-action policy model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StructuredPrimitiveActionPolicyConfig:
    hidden_size: int = 256
    history_length: int = 32
    max_wall_tokens: int = 256
    max_insertion_tokens: int = 64
    num_transformer_layers: int = 4
    num_attention_heads: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1
    num_action_types: int = 6
    num_high_level_actions: int = 8
    num_gui_actions: int = 16
    num_keys: int = 16
    max_step_index: int = 512
    max_repeat_count: int = 8
    wall_feature_dim: int = 5
    insertion_feature_dim: int = 5
    progress_feature_dim: int = 18
    primitive_action_dim: int = 6
    xy_output_activation: str = "sigmoid"
    ignore_interval_loss: bool = True
    default_key_interval_ms: float = 100.0
    xy_smooth_l1_beta: float = 0.02
    loss_action_type_weight: float = 1.0
    loss_high_level_weight: float = 1.0
    loss_gui_action_weight: float = 1.0
    loss_xy_weight: float = 200.0
    loss_aux_wall_weight: float = 1.0
    loss_aux_point_role_weight: float = 1.0
    loss_key_weight: float = 1.0
    loss_repeat_weight: float = 0.5
    loss_interval_weight: float = 0.0
