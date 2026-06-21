import inspect
import json
from pathlib import Path

import torch
import pytest

import data_loader.data_loader as data_loader_module
from data_loader.data_loader import PrimitiveActionDataset
from data_loader.primitive_action import (
    StructuredPrimitiveActionDataset,
    VisualPrimitiveActionDataset,
    create_structured_dataset_from_config,
    create_structured_dataloader,
    create_visual_dataloader,
)


def minimal_loader_item() -> dict:
    return {
        "sample_id": "plan_0001",
        "split": "train",
        "step_index": torch.tensor(0, dtype=torch.long),
        "observation": torch.zeros((1, 216, 384), dtype=torch.float32),
        "observation_available": torch.tensor(True, dtype=torch.bool),
        "global_floorplan": None,
        "global_floorplan_available": torch.tensor(False, dtype=torch.bool),
        "plan": {
            "walls": torch.zeros((1, 5), dtype=torch.float32),
            "insertions": torch.zeros((0, 5), dtype=torch.float32),
        },
        "progress": torch.zeros((18,), dtype=torch.float32),
        "history": {
            "primitive_action": torch.zeros((2, 6), dtype=torch.float32),
            "mask": torch.tensor([False, False], dtype=torch.bool),
        },
        "target": {
            "primitive_action": torch.tensor([1.0, 0.0, 0.0, -1.0, -1.0, -1.0], dtype=torch.float32),
        },
    }


def write_minimal_processed_dataset(dataset_path: Path) -> None:
    sample_path = dataset_path / "samples" / "plan_0001.json"
    sample = {
        "sample_id": "plan_0001",
        "split": "train",
        "model_inputs": {
            "encoded_resplan": {
                "coordinate_system": {"input_coordinates": "normalized"},
                "execution_walls": [],
                "insertions": [],
            },
            "task_entity_counts": {},
        },
        "steps": [
            {
                "step_index": 0,
                "model_input": {"task_progress": {"vector": [0.0] * 18}},
                "supervision_target": {
                    "primitive_action": [2.0, -1.0, -1.0, -1.0, -1.0, -1.0],
                    "high_level_id": 0,
                    "gui_action_id": 0,
                    "coordinate_frame_id": 0,
                    "high_level_action": "<none>",
                    "gui_action": "<none>",
                },
            }
        ],
    }
    vocab = {
        "action_type_to_id": {"CLICK": 2},
        "high_level_to_id": {"<none>": 0},
        "gui_action_to_id": {"<none>": 0},
        "key_to_id": {},
        "coordinate_frame_to_id": {"none": 0},
    }
    sample_path.parent.mkdir(parents=True)
    sample_path.write_text(json.dumps(sample), encoding="utf-8")
    (dataset_path / "dataset_index.json").write_text(
        json.dumps([{"sample_id": "plan_0001", "split": "train", "path": str(sample_path)}]),
        encoding="utf-8",
    )
    (dataset_path / "action_vocab.json").write_text(json.dumps(vocab), encoding="utf-8")


def test_collate_keeps_global_floorplan_disabled() -> None:
    batch = PrimitiveActionDataset.collate_fn([minimal_loader_item(), minimal_loader_item()])

    assert batch["global_floorplan"] is None
    assert batch["global_floorplan_available"].tolist() == [False, False]


def test_collate_omits_compatibility_aliases() -> None:
    batch = PrimitiveActionDataset.collate_fn([minimal_loader_item(), minimal_loader_item()])

    assert "observation_before" not in batch
    assert "action" not in batch


def test_dataset_module_does_not_export_legacy_dataset_aliases() -> None:
    assert not hasattr(data_loader_module, "OnlineGuiPolicyDataset")
    assert not hasattr(data_loader_module, "LowLevelGuiDataset")


def test_primitive_action_dataset_defaults_to_structured_inputs() -> None:
    signature = inspect.signature(PrimitiveActionDataset)

    assert signature.parameters["load_observation"].default is False
    assert signature.parameters["load_images"].default is False
    assert signature.parameters["load_global_floorplan"].default is False


def test_primitive_action_dataset_docstring_describes_structured_inputs() -> None:
    docstring = inspect.getdoc(PrimitiveActionDataset) or ""

    assert "structured" in docstring.lower()
    assert "current observation screenshot" not in docstring


def test_profile_specific_dataset_wrappers_set_input_defaults() -> None:
    structured_signature = inspect.signature(StructuredPrimitiveActionDataset)
    visual_signature = inspect.signature(VisualPrimitiveActionDataset)

    assert structured_signature.parameters["load_observation"].default is False
    assert structured_signature.parameters["load_images"].default is False
    assert structured_signature.parameters["load_global_floorplan"].default is False
    assert visual_signature.parameters["load_observation"].default is True
    assert visual_signature.parameters["load_images"].default is True
    assert visual_signature.parameters["load_global_floorplan"].default is True


def test_profile_specific_dataloader_factories_use_profile_dataset_classes(tmp_path: Path) -> None:
    write_minimal_processed_dataset(tmp_path)

    structured_loader = create_structured_dataloader(dataset_path=tmp_path, split="train", batch_size=1)
    visual_loader = create_visual_dataloader(dataset_path=tmp_path, split="train", batch_size=1)

    assert isinstance(structured_loader.dataset, StructuredPrimitiveActionDataset)
    assert isinstance(visual_loader.dataset, VisualPrimitiveActionDataset)


def test_structured_dataset_from_config_ignores_unrelated_legacy_loader_kwargs(tmp_path: Path) -> None:
    write_minimal_processed_dataset(tmp_path)

    [packet] = create_structured_dataset_from_config(
        dataset_path=tmp_path,
        splits=("train",),
        batch_size=1,
        multiview_dir="unused",
        view_ids=["05"],
        frame_transform=object(),
    )

    assert isinstance(packet["loader"].dataset, StructuredPrimitiveActionDataset)


def test_image_dtype_rejects_legacy_short_names() -> None:
    with pytest.raises(ValueError, match="image_dtype"):
        PrimitiveActionDataset(dataset_path="processed_data", split="train", image_dtype="fp16")


def move_tensors_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_tensors_to_device(item, device) for key, item in value.items()}
    return value


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for device mismatch regression")
def test_collate_keeps_vector_masks_on_sample_device() -> None:
    device = torch.device("cuda:0")
    items = [move_tensors_to_device(minimal_loader_item(), device) for _ in range(2)]

    batch = PrimitiveActionDataset.collate_fn(items)

    assert batch["plan"]["wall_mask"].device == device
    assert batch["plan"]["insertion_mask"].device == device
