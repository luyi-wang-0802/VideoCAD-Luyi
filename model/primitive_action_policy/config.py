"""Configuration for the primitive-action policy model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PrimitiveActionPolicyConfig:
    hidden_size: int = 256
    image_channels: int = 1
    history_length: int = 32
    num_transformer_layers: int = 4
    num_attention_heads: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1
    num_action_types: int = 6
    num_high_level_actions: int = 8
    num_gui_actions: int = 16
    num_keys: int = 16
    num_coordinate_frames: int = 4
    num_target_entities: int = 1
    num_point_roles: int = 1
    max_step_index: int = 512
    max_repeat_count: int = 8
    wall_feature_dim: int = 5
    insertion_feature_dim: int = 5
    entity_feature_dim: int = 11
    progress_feature_dim: int = 18
    primitive_action_dim: int = 6
    primitive_param_dim: int = 5
    num_param_bins: int = 1000
    key_bin_size: int = 50
    repeat_bin_size: int = 100
    model_coord_min: float = -0.5
    model_coord_max: float = 0.5
    use_binned_primitive_params: bool = False
    soft_xy_bin_loss: bool = True
    xy_bin_tolerance: int = 5
    xy_bin_soft_sigma: float = 2.0
    ignore_interval_bin_loss: bool = True
    ignore_interval_loss: bool = True
    default_key_interval_ms: float = 100.0
    xy_smooth_l1_beta: float = 0.02
    loss_param_bins_weight: float = 1.0
    loss_action_type_weight: float = 1.0
    loss_high_level_weight: float = 1.0
    loss_gui_action_weight: float = 1.0
    loss_xy_weight: float = 50.0
    loss_key_weight: float = 1.0
    loss_repeat_weight: float = 0.5
    loss_interval_weight: float = 0.0
    loss_target_entity_weight: float = 1.0
    loss_point_role_weight: float = 1.0
