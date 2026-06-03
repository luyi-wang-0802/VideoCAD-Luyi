"""Dataset and dataloader for low-level Vectorworks GUI imitation learning.

This repository fork uses processed samples from:

    data_process/low_level_gui_sequence/results/

Each sample is one floor plan, and each dataset item is one low-level policy
action with JSON-derived plan context and screenshot observations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from data_loader.image_loader import ScreenshotImageLoader


LOCATION_TO_ID = {"exterior": 0, "interior": 1, "unknown": 2}
OPENING_TO_ID = {"door": 0, "window": 1}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def resolve_path(path: str | Path, repo_root: Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return repo_root / path


def normalize_source_point(point: list[float] | None, coordinate_system: dict[str, Any]) -> list[float]:
    """Convert ResPlan source coordinates to centered normalized model coordinates."""

    if not point:
        return [0.0, 0.0]
    if coordinate_system.get("input_coordinates") in {"model_units", "normalized", "centered_normalized"}:
        return [float(point[0]), float(point[1])]

    bbox = coordinate_system.get("source_bbox", {})
    x_range = coordinate_system.get("x_range", [0, 1])
    y_range = coordinate_system.get("y_range", [0, 1])
    source_span = float(coordinate_system.get("source_span") or 0)
    if not source_span:
        source_span = max(float(x_range[1]) - float(x_range[0]), float(y_range[1]) - float(y_range[0]))
    if not source_span:
        return [0.0, 0.0]

    min_x = float(bbox.get("min_x", x_range[0]))
    max_x = float(bbox.get("max_x", x_range[1]))
    min_y = float(bbox.get("min_y", y_range[0]))
    max_y = float(bbox.get("max_y", y_range[1]))
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    return [
        (float(point[0]) - center_x) / source_span,
        (float(point[1]) - center_y) / source_span,
    ]


class LowLevelGuiDataset(Dataset):
    """Step-level dataset for low-level GUI behavior cloning.

    Output action types are:

    - CLICK
    - DOUBLE_CLICK
    - HOTKEY
    - PRESS_KEY

    MOVE_TO is represented as target information attached to the state-changing
    action that needs it.
    """

    def __init__(
        self,
        dataset_dir: str | Path = "data_process/low_level_gui_sequence/results",
        split: str | None = None,
        repo_root: str | Path = ".",
        image_size: tuple[int, int] = (224, 224),
        load_images: bool = True,
        include_after_image: bool = False,
        normalize_images: bool = True,
        include_raw: bool = False,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.dataset_dir = resolve_path(dataset_dir, self.repo_root)
        self.split = split
        self.include_after_image = include_after_image
        self.include_raw = include_raw
        self.image_loader = ScreenshotImageLoader(
            repo_root=self.repo_root,
            image_size=image_size,
            load_images=load_images,
            normalize=normalize_images,
        )

        self.index = read_json(self.dataset_dir / "dataset_index.json")
        self.vocab = read_json(self.dataset_dir / "action_vocab.json")
        self.action_type_to_id = {name: idx for idx, name in enumerate(self.vocab["action_types"])}
        self.key_to_id = self._make_vocab_mapping(self.vocab.get("key_counts", {}))
        self.gui_action_to_id = self._make_vocab_mapping(self.vocab.get("parent_gui_action_counts", {}))
        self.high_level_to_id = self._make_vocab_mapping(self.vocab.get("parent_high_level_type_counts", {}))
        self.point_role_to_id = self._make_vocab_mapping(self.vocab.get("point_role_counts", {}))

        self.samples: dict[str, dict[str, Any]] = {}
        self.steps: list[dict[str, Any]] = []
        for entry in self.index:
            if split is not None and entry.get("split") != split:
                continue
            sample_path = resolve_path(entry["path"], self.repo_root)
            sample = read_json(sample_path)
            self.samples[sample["sample_id"]] = sample
            plan_tensors = self._encode_plan(sample["compact_plan"])
            for action in sample["low_level_actions"]:
                self.steps.append(
                    {
                        "sample_id": sample["sample_id"],
                        "split": sample.get("split"),
                        "sample_path": str(sample_path),
                        "action": action,
                        "plan": plan_tensors,
                    }
                )

    @staticmethod
    def _make_vocab_mapping(counts: dict[str, int]) -> dict[str, int]:
        names = sorted(name for name in counts if name is not None)
        return {"<none>": 0, **{name: idx + 1 for idx, name in enumerate(names)}}

    def _encode_plan(self, compact_plan: dict[str, Any]) -> dict[str, torch.Tensor]:
        coordinate_system = compact_plan.get("coordinate_system", {})

        wall_rows = []
        for wall in compact_plan.get("walls", []):
            start = normalize_source_point(wall.get("start"), coordinate_system)
            end = normalize_source_point(wall.get("end"), coordinate_system)
            location_id = LOCATION_TO_ID.get(wall.get("wall_location", "unknown"), LOCATION_TO_ID["unknown"])
            wall_rows.append([start[0], start[1], end[0], end[1], float(location_id)])

        insertion_rows = []
        for insertion in compact_plan.get("insertions", []):
            point = normalize_source_point(insertion.get("insertion_point"), coordinate_system)
            opening_id = OPENING_TO_ID.get(insertion.get("opening_type"), -1)
            insertion_rows.append([point[0], point[1], float(opening_id)])

        return {
            "walls": torch.tensor(wall_rows, dtype=torch.float32) if wall_rows else torch.zeros((0, 5)),
            "insertions": (
                torch.tensor(insertion_rows, dtype=torch.float32) if insertion_rows else torch.zeros((0, 3))
            ),
        }

    def _encode_target(self, action: dict[str, Any]) -> dict[str, torch.Tensor]:
        target = action.get("target", {})
        model_point = target.get("model_point")
        window_norm = target.get("window_norm")
        screen_point = target.get("screen_point")
        point_role = target.get("point_role") or action.get("point_role") or "<none>"

        return {
            "model_point": torch.tensor(model_point or [0.0, 0.0], dtype=torch.float32),
            "model_point_mask": torch.tensor(model_point is not None, dtype=torch.bool),
            "window_norm": torch.tensor(window_norm or [0.0, 0.0], dtype=torch.float32),
            "window_norm_mask": torch.tensor(window_norm is not None, dtype=torch.bool),
            "screen_point": torch.tensor(screen_point or [0.0, 0.0], dtype=torch.float32),
            "screen_point_mask": torch.tensor(screen_point is not None, dtype=torch.bool),
            "point_role_id": torch.tensor(self.point_role_to_id.get(point_role, 0), dtype=torch.long),
        }

    def _encode_action(self, action: dict[str, Any]) -> dict[str, torch.Tensor]:
        key_name = "+".join(action.get("keys", [])) if action.get("action_type") == "HOTKEY" else action.get("key")
        encoded = {
            "action_type_id": torch.tensor(self.action_type_to_id[action["action_type"]], dtype=torch.long),
            "key_id": torch.tensor(self.key_to_id.get(key_name or "<none>", 0), dtype=torch.long),
            "gui_action_id": torch.tensor(
                self.gui_action_to_id.get(action.get("parent_gui_action") or "<none>", 0), dtype=torch.long
            ),
            "high_level_id": torch.tensor(
                self.high_level_to_id.get(action.get("parent_high_level_type") or "<none>", 0), dtype=torch.long
            ),
            "step_index": torch.tensor(int(action.get("step_index", 0)), dtype=torch.long),
        }
        encoded.update(self._encode_target(action))
        return encoded

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.steps[index]
        action = item["action"]
        before_image, before_available = self.image_loader.load(
            action.get("observation_before", {}).get("screenshot_path")
        )
        result: dict[str, Any] = {
            "sample_id": item["sample_id"],
            "split": item["split"],
            "primitive_id": action.get("primitive_id"),
            "observation_before": before_image,
            "observation_before_available": torch.tensor(before_available, dtype=torch.bool),
            "action": self._encode_action(action),
            "plan": item["plan"],
        }
        if self.include_after_image:
            after_image, after_available = self.image_loader.load(
                action.get("observation_after", {}).get("screenshot_path")
            )
            result["observation_after"] = after_image
            result["observation_after_available"] = torch.tensor(after_available, dtype=torch.bool)
        if self.include_raw:
            result["raw_action"] = action
            result["sample_path"] = item["sample_path"]
        return result

    def __len__(self) -> int:
        return len(self.steps)

    @staticmethod
    def _pad_tensor_list(values: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        max_len = max((value.shape[0] for value in values), default=0)
        feature_dim = values[0].shape[1] if values else 0
        padded = torch.zeros((len(values), max_len, feature_dim), dtype=values[0].dtype)
        mask = torch.zeros((len(values), max_len), dtype=torch.bool)
        for index, value in enumerate(values):
            length = value.shape[0]
            if length:
                padded[index, :length] = value
                mask[index, :length] = True
        return padded, mask

    @classmethod
    def collate_fn(cls, batch: list[dict[str, Any]]) -> dict[str, Any]:
        action_keys = batch[0]["action"].keys()
        walls, wall_mask = cls._pad_tensor_list([item["plan"]["walls"] for item in batch])
        insertions, insertion_mask = cls._pad_tensor_list([item["plan"]["insertions"] for item in batch])

        result: dict[str, Any] = {
            "sample_id": [item["sample_id"] for item in batch],
            "split": [item["split"] for item in batch],
            "primitive_id": [item["primitive_id"] for item in batch],
            "observation_before": torch.stack([item["observation_before"] for item in batch]),
            "observation_before_available": torch.stack([item["observation_before_available"] for item in batch]),
            "action": {key: torch.stack([item["action"][key] for item in batch]) for key in action_keys},
            "plan": {
                "walls": walls,
                "wall_mask": wall_mask,
                "insertions": insertions,
                "insertion_mask": insertion_mask,
            },
        }
        if "observation_after" in batch[0]:
            result["observation_after"] = torch.stack([item["observation_after"] for item in batch])
            result["observation_after_available"] = torch.stack(
                [item["observation_after_available"] for item in batch]
            )
        if "raw_action" in batch[0]:
            result["raw_action"] = [item["raw_action"] for item in batch]
            result["sample_path"] = [item["sample_path"] for item in batch]
        return result


def create_dataloader(
    dataset_dir: str | Path = "data_process/low_level_gui_sequence/results",
    split: str | None = "train",
    batch_size: int = 8,
    shuffle: bool | None = None,
    num_workers: int = 0,
    **dataset_kwargs: Any,
) -> DataLoader:
    dataset = LowLevelGuiDataset(dataset_dir=dataset_dir, split=split, **dataset_kwargs)
    if shuffle is None:
        shuffle = split == "train"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=LowLevelGuiDataset.collate_fn,
        pin_memory=torch.cuda.is_available(),
    )


def create_dataset_from_config(
    dataset_path: str | Path = "data_process/low_level_gui_sequence/results",
    config: str | Path | None = None,
    batch_size: int = 8,
    num_workers: int = 0,
    splits: tuple[str, ...] = ("train", "val", "test"),
    **dataset_kwargs: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Compatibility wrapper returning train/val/test loader packets.

    The old VideoCAD code passed a split config path. The processed dataset now
    stores split information in dataset_index.json, so ``config`` is accepted
    only for API compatibility.
    """

    del config
    packets = []
    for split in splits:
        loader = create_dataloader(
            dataset_dir=dataset_path,
            split=split,
            batch_size=batch_size,
            num_workers=num_workers,
            **dataset_kwargs,
        )
        packets.append({"loader": loader, "sampler": None})
    return tuple(packets)  # type: ignore[return-value]
