import pytest
import torch

from model.model_factory import ModelFactory, ModelType
from model.primitive_action_policy import PrimitiveActionPolicyConfig, PrimitiveActionPolicyModel


def minimal_batch() -> dict:
    return {
        "observation": torch.randn(2, 1, 216, 384),
        "global_floorplan": torch.randn(2, 1, 216, 384),
        "global_floorplan_available": torch.tensor([True, True]),
        "step_index": torch.tensor([0, 1]),
        "plan": {
            "walls": torch.randn(2, 1, 5),
            "wall_mask": torch.tensor([[True], [True]]),
            "insertions": torch.randn(2, 1, 5),
            "insertion_mask": torch.tensor([[True], [False]]),
        },
        "history": {
            "primitive_action": torch.zeros(2, 2, 6),
            "action_type_id": torch.zeros(2, 2, dtype=torch.long),
            "xy": torch.zeros(2, 2, 2),
            "key_id": torch.full((2, 2), -1, dtype=torch.long),
            "key_repeat_count": torch.full((2, 2), -1, dtype=torch.long),
            "key_interval": torch.full((2, 2), -1.0),
            "high_level_id": torch.zeros(2, 2, dtype=torch.long),
            "gui_action_id": torch.zeros(2, 2, dtype=torch.long),
            "coordinate_frame_id": torch.zeros(2, 2, dtype=torch.long),
            "is_move": torch.zeros(2, 2, dtype=torch.bool),
            "is_key_action": torch.zeros(2, 2, dtype=torch.bool),
            "mask": torch.tensor([[False, False], [True, False]]),
        },
        "target": {
            "primitive_action": torch.tensor(
                [[1.0, 0.1, 0.2, -1.0, -1.0, -1.0], [2.0, -1.0, -1.0, -1.0, -1.0, -1.0]]
            ),
            "action_type_id": torch.tensor([1, 2]),
            "xy": torch.tensor([[0.1, 0.2], [-1.0, -1.0]]),
            "key_id": torch.tensor([-1, -1]),
            "key_repeat_count": torch.tensor([-1, -1]),
            "key_interval": torch.tensor([-1.0, -1.0]),
            "high_level_id": torch.tensor([1, 2]),
            "gui_action_id": torch.tensor([1, 3]),
            "coordinate_frame_id": torch.tensor([1, 0]),
            "is_move": torch.tensor([True, False]),
            "is_key_action": torch.tensor([False, False]),
        },
    }


def test_primitive_policy_package_import_and_forward() -> None:
    model = PrimitiveActionPolicyModel(
        PrimitiveActionPolicyConfig(
            hidden_size=32,
            num_transformer_layers=1,
            num_attention_heads=4,
            dim_feedforward=64,
            history_length=2,
            num_high_level_actions=3,
            num_gui_actions=4,
            num_keys=2,
        )
    )
    batch = minimal_batch()

    outputs = model(batch)
    losses = model.compute_loss(batch, outputs)

    assert outputs["action_type_logits"].shape == torch.Size([2, 6])
    assert "target_entity_logits" not in outputs
    assert "point_role_logits" not in outputs
    assert torch.isfinite(losses["loss"])


def test_primitive_policy_forward_without_global_floorplan() -> None:
    model = PrimitiveActionPolicyModel(
        PrimitiveActionPolicyConfig(
            hidden_size=32,
            num_transformer_layers=1,
            num_attention_heads=4,
            dim_feedforward=64,
            history_length=2,
            num_high_level_actions=3,
            num_gui_actions=4,
            num_keys=2,
        )
    )
    batch = minimal_batch()
    batch["global_floorplan"] = None
    batch["global_floorplan_available"] = torch.tensor([False, False])

    outputs = model(batch)
    losses = model.compute_loss(batch, outputs)

    assert outputs["action_type_logits"].shape == torch.Size([2, 6])
    assert torch.isfinite(losses["loss"])


def test_primitive_policy_xy_loss_backward() -> None:
    model = PrimitiveActionPolicyModel(
        PrimitiveActionPolicyConfig(
            hidden_size=32,
            num_transformer_layers=1,
            num_attention_heads=4,
            dim_feedforward=64,
            history_length=2,
            num_high_level_actions=3,
            num_gui_actions=4,
            num_keys=2,
        )
    )
    batch = minimal_batch()

    outputs = model(batch)
    losses = model.compute_loss(batch, outputs)
    losses["loss"].backward()

    assert torch.isfinite(losses["loss"])


def test_model_factory_only_creates_current_primitive_policy() -> None:
    model, model_type = ModelFactory().create_model(
        "primitive_action_policy",
        {
            "hidden_size": 32,
            "num_transformer_layers": 1,
            "num_attention_heads": 4,
            "dim_feedforward": 64,
            "history_length": 2,
            "num_high_level_actions": 3,
            "num_gui_actions": 4,
            "num_keys": 2,
        },
        "cpu",
    )

    assert isinstance(model, PrimitiveActionPolicyModel)
    assert model_type == ModelType.PRIMITIVE_ACTION_POLICY


def test_model_factory_rejects_legacy_model_names() -> None:
    with pytest.raises(ValueError, match="primitive_action_policy"):
        ModelFactory().create_model("cad_past_10_actions_and_states_timestep_embedding", {}, "cpu")
