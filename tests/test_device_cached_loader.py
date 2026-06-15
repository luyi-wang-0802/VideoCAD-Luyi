import torch
from torch.utils.data import DataLoader, Dataset

import data_loader.device_cached_loader as cached_loader_module
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


class AliasDataset(Dataset):
    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int):
        tensor = torch.tensor([float(index)])
        return {"primary": tensor, "duplicate_reference": tensor}


def collate_aliases(batch):
    return batch[0]


def test_device_cached_loader_preserves_tensor_aliases(monkeypatch) -> None:
    source_loader = DataLoader(AliasDataset(), batch_size=1, collate_fn=collate_aliases)

    def clone_to(self, *args, **kwargs):
        return self.clone()

    monkeypatch.setattr(torch.Tensor, "to", clone_to)
    loader = DeviceCachedDataLoader.from_loader(source_loader, torch.device("cpu"))
    cached = loader.samples[0]

    assert cached["primary"] is cached["duplicate_reference"]


class SharedTensorDataset(Dataset):
    def __init__(self) -> None:
        self.shared = torch.tensor([1.0])

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int):
        return {"shared": self.shared, "index": torch.tensor([index])}


def test_device_cached_loader_preserves_shared_tensors_across_samples(monkeypatch) -> None:
    source_loader = DataLoader(SharedTensorDataset(), batch_size=1, collate_fn=collate_aliases)

    def clone_to(self, *args, **kwargs):
        return self.clone()

    monkeypatch.setattr(torch.Tensor, "to", clone_to)
    loader = DeviceCachedDataLoader.from_loader(source_loader, torch.device("cpu"))

    assert loader.samples[0]["shared"] is loader.samples[1]["shared"]


class ShapeChangingDataset(Dataset):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int):
        if index == 0:
            return {"step_index": torch.zeros((1, 216, 384))}
        return {"step_index": torch.tensor(index)}


def test_device_cached_loader_handles_tensor_id_collisions(monkeypatch) -> None:
    source_loader = DataLoader(ShapeChangingDataset(), batch_size=2, collate_fn=collate_aliases)

    def clone_to(self, *args, **kwargs):
        return self.clone()

    monkeypatch.setattr(torch.Tensor, "to", clone_to)
    monkeypatch.setattr(cached_loader_module, "id", lambda value: 1, raising=False)

    loader = DeviceCachedDataLoader.from_loader(source_loader, torch.device("cpu"))

    assert loader.samples[0]["step_index"].shape == torch.Size([1, 216, 384])
    assert loader.samples[1]["step_index"].shape == torch.Size([])
