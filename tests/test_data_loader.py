import torch
import pytest

import data_loader.data_loader as data_loader_module
from data_loader.data_loader import PrimitiveActionDataset


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
