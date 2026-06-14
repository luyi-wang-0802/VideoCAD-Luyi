import torch
from torch.utils.data import DataLoader, Dataset

from data_loader.device_cached_loader import DeviceCachedDataLoader


class DictDataset(Dataset):
    def __len__(self) -> int:
        return 3

    def __getitem__(self, index: int):
        return {
            "x": torch.tensor([float(index)]),
            "nested": {"y": torch.tensor([index])},
            "ids": [f"sample_{index}"],
        }


def collate_dicts(batch):
    return {
        "x": torch.stack([item["x"] for item in batch]),
        "nested": {"y": torch.stack([item["nested"]["y"] for item in batch])},
        "ids": [item["ids"][0] for item in batch],
    }


def test_device_cached_loader_moves_nested_tensors_and_preserves_len() -> None:
    source_loader = DataLoader(DictDataset(), batch_size=2, collate_fn=collate_dicts)

    loader = DeviceCachedDataLoader.from_loader(source_loader, torch.device("cpu"))

    assert len(loader) == 2
    batch = next(iter(loader))
    assert batch["x"].device.type == "cpu"
    assert batch["nested"]["y"].device.type == "cpu"
    assert batch["ids"] == ["sample_0", "sample_1"]
