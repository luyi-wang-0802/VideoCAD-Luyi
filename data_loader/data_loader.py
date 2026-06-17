"""Dataset and dataloader for ResPlan-conditioned primitive-action training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from data_loader.image_loader import ScreenshotImageLoader


DEFAULT_IMAGE_SIZE = (384, 216)
LOCATION_TO_ID = {"exterior": 0, "interior": 1, "unknown": 2}
OPENING_TO_ID = {"door": 0, "window": 1, "front_door": 2, "unknown": 3}
PRIMITIVE_ACTION_DIM = 6
PROGRESS_FEATURE_DIM = 18


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def resolve_path(path: str | Path, repo_root: Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return repo_root / path


def normalize_source_point(point: list[float] | None, coordinate_system: dict[str, Any]) -> list[float]:
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


class PrimitiveActionDataset(Dataset):
    """Step-level dataset for autoregressive next-action training.

    Each item is one training step:

    Inputs:
    - ResPlan-derived plan tensors
    - current observation screenshot
    - fixed-length history of previous primitive/high-level/GUI labels

    Targets:
    - next high-level id
    - next GUI action id
    - next primitive action vector
    - next coordinate frame id
    """

    def __init__(
        self,
        dataset_path: str | Path = "processed_data",
        split: str | None = None,
        repo_root: str | Path = ".",
        action_vocab_path: str | Path | None = None,
        image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
        history_length: int = 32,
        load_images: bool = True,
        load_global_floorplan: bool = False,
        normalize_images: bool = True,
        image_dtype: str | torch.dtype = torch.float32,
        include_raw: bool = False,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.dataset_path = resolve_path(dataset_path, self.repo_root)
        self.split = split
        self.history_length = history_length
        self.include_raw = include_raw
        self.load_global_floorplan = load_global_floorplan
        if isinstance(image_dtype, str):
            dtype_map = {"float32": torch.float32, "float16": torch.float16}
            if image_dtype not in dtype_map:
                raise ValueError("image_dtype must be one of: float32, float16")
            image_dtype = dtype_map[image_dtype]
        self.image_loader = ScreenshotImageLoader(
            repo_root=self.repo_root,
            image_size=image_size,
            load_images=load_images,
            normalize=normalize_images,
            dtype=image_dtype,
        )

        self.index = read_json(self.dataset_path / "dataset_index.json")
        vocab_path = resolve_path(action_vocab_path, self.repo_root) if action_vocab_path else self.dataset_path / "action_vocab.json"
        self.vocab = read_json(vocab_path)
        self.action_type_to_id = self.vocab.get("action_type_to_id", {})
        self.high_level_to_id = self.vocab.get("high_level_to_id", {})
        self.gui_action_to_id = self.vocab.get("gui_action_to_id", {})
        self.key_to_id = self.vocab.get("key_to_id", {})
        self.coordinate_frame_to_id = self.vocab.get("coordinate_frame_to_id", {})

        self.samples: dict[str, dict[str, Any]] = {}
        self.steps: list[dict[str, Any]] = []
        for entry in self.index:
            if split is not None and entry.get("split") != split:
                continue
            sample_path = resolve_path(entry["path"], self.repo_root)
            sample = read_json(sample_path)
            self.samples[sample["sample_id"]] = sample
            global_floorplan_path = sample.get("model_inputs", {}).get("global_floorplan_path")
            plan_tensors = self._encode_plan(sample["model_inputs"]["encoded_resplan"])
            encoded_steps = [self._encode_step(step) for step in sample["steps"]]
            for step_index, step in enumerate(sample["steps"]):
                step_global_floorplan_path = step.get("model_input", {}).get("global_floorplan_path")
                self.steps.append(
                    {
                        "sample_id": sample["sample_id"],
                        "split": sample.get("split"),
                        "sample_path": str(sample_path),
                        "step": step,
                        "global_floorplan_path": step_global_floorplan_path or global_floorplan_path,
                        "target": encoded_steps[step_index],
                        "history": self._build_history(encoded_steps, step_index),
                        "plan": plan_tensors,
                        "progress": self._encode_progress(step),
                    }
                )

    def _encode_plan(self, encoded_resplan: dict[str, Any]) -> dict[str, torch.Tensor]:
        coordinate_system = encoded_resplan.get("coordinate_system", {})

        wall_rows = []
        execution_walls = encoded_resplan.get("execution_walls")
        wall_source = execution_walls if execution_walls is not None else encoded_resplan.get("walls", [])
        for wall in wall_source:
            if execution_walls is not None:
                start = wall.get("execution_start") or wall.get("start") or [0.0, 0.0]
                end = wall.get("execution_end") or wall.get("end") or [0.0, 0.0]
                location_name = wall.get("wall_location", "unknown")
            else:
                geometry = wall.get("geometry", {})
                start = normalize_source_point(geometry.get("start"), coordinate_system)
                end = normalize_source_point(geometry.get("end"), coordinate_system)
                location_name = wall.get("physical", {}).get("wall_location", "unknown")
            location_id = LOCATION_TO_ID.get(location_name, LOCATION_TO_ID["unknown"])
            wall_rows.append([start[0], start[1], end[0], end[1], float(location_id)])

        insertion_rows = []
        for insertion in encoded_resplan.get("insertions", []):
            center = insertion.get("insertion_point") or normalize_source_point(
                insertion.get("source_insertion_point"), coordinate_system
            )
            click = insertion.get("execution_click_point") or insertion.get("click_point") or normalize_source_point(
                insertion.get("source_click_point"), coordinate_system
            )
            opening_type = str(insertion.get("opening_type", "unknown")).lower()
            opening_id = OPENING_TO_ID.get(opening_type, OPENING_TO_ID["unknown"])
            insertion_rows.append([center[0], center[1], click[0], click[1], float(opening_id)])

        return {
            "walls": torch.tensor(wall_rows, dtype=torch.float32) if wall_rows else torch.zeros((0, 5)),
            "insertions": (
                torch.tensor(insertion_rows, dtype=torch.float32) if insertion_rows else torch.zeros((0, 5))
            ),
        }

    @staticmethod
    def _encode_progress(step: dict[str, Any]) -> torch.Tensor:
        progress = step.get("model_input", {}).get("task_progress", {})
        vector = progress.get("vector") if isinstance(progress, dict) else None
        if not isinstance(vector, list):
            vector = [0.0] * PROGRESS_FEATURE_DIM
        values = [float(value) for value in vector[:PROGRESS_FEATURE_DIM]]
        values += [0.0] * (PROGRESS_FEATURE_DIM - len(values))
        return torch.tensor(values, dtype=torch.float32)

    @staticmethod
    def _encode_step(step: dict[str, Any]) -> dict[str, torch.Tensor]:
        target = step["supervision_target"]
        primitive_action = torch.tensor(target["primitive_action"], dtype=torch.float32)
        action_type_id = int(primitive_action[0].item())
        key_id = int(primitive_action[3].item())
        repeat_count = int(primitive_action[4].item()) if primitive_action[4].item() >= 0 else -1
        high_level_action = str(target.get("high_level_action", ""))
        gui_action = str(target.get("gui_action", ""))
        is_wall_move = action_type_id == 1 and (
            high_level_action in {"CREATE_EXTERIOR_WALL", "CREATE_INTERIOR_WALL"}
            or gui_action == "DRAW_WALL_FROM_ENTITY_GEOMETRY"
        )
        return {
            "primitive_action": primitive_action,
            "action_type_id": torch.tensor(action_type_id, dtype=torch.long),
            "xy": primitive_action[1:3].clone(),
            "key_id": torch.tensor(key_id, dtype=torch.long),
            "key_repeat_count": torch.tensor(repeat_count, dtype=torch.long),
            "key_interval": primitive_action[5].clone().to(torch.float32),
            "high_level_id": torch.tensor(int(target["high_level_id"]), dtype=torch.long),
            "gui_action_id": torch.tensor(int(target["gui_action_id"]), dtype=torch.long),
            "coordinate_frame_id": torch.tensor(int(target["coordinate_frame_id"]), dtype=torch.long),
            "is_move": torch.tensor(action_type_id == 1, dtype=torch.bool),
            "is_wall_move": torch.tensor(is_wall_move, dtype=torch.bool),
            "is_key_action": torch.tensor(action_type_id in {3, 4}, dtype=torch.bool),
        }

    @staticmethod
    def _empty_encoded_step() -> dict[str, torch.Tensor]:
        return {
            "primitive_action": torch.full((PRIMITIVE_ACTION_DIM,), -1.0, dtype=torch.float32),
            "action_type_id": torch.tensor(0, dtype=torch.long),
            "xy": torch.zeros((2,), dtype=torch.float32),
            "key_id": torch.tensor(-1, dtype=torch.long),
            "key_repeat_count": torch.tensor(-1, dtype=torch.long),
            "key_interval": torch.tensor(-1.0, dtype=torch.float32),
            "high_level_id": torch.tensor(0, dtype=torch.long),
            "gui_action_id": torch.tensor(0, dtype=torch.long),
            "coordinate_frame_id": torch.tensor(0, dtype=torch.long),
            "is_move": torch.tensor(False, dtype=torch.bool),
            "is_wall_move": torch.tensor(False, dtype=torch.bool),
            "is_key_action": torch.tensor(False, dtype=torch.bool),
        }

    def _build_history(self, encoded_steps: list[dict[str, torch.Tensor]], step_index: int) -> dict[str, torch.Tensor]:
        start = max(0, step_index - self.history_length)
        history = encoded_steps[start:step_index]
        pad_count = self.history_length - len(history)
        padded = history + [self._empty_encoded_step() for _ in range(pad_count)]
        mask = [True] * len(history) + [False] * pad_count
        keys = padded[0].keys() if padded else self._empty_encoded_step().keys()
        stacked = {key: torch.stack([item[key] for item in padded]) for key in keys}
        stacked["mask"] = torch.tensor(mask, dtype=torch.bool)
        return stacked

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.steps[index]
        step = item["step"]
        screenshot_path = step["model_input"].get("observation_screenshot_path")
        observation, observation_available = self.image_loader.load(screenshot_path)
        global_floorplan = None
        global_floorplan_available = False
        if self.load_global_floorplan:
            global_floorplan, global_floorplan_available = self.image_loader.load(item.get("global_floorplan_path"))
        result: dict[str, Any] = {
            "sample_id": item["sample_id"],
            "split": item["split"],
            "step_index": torch.tensor(int(step["step_index"]), dtype=torch.long),
            "observation": observation,
            "observation_available": torch.tensor(observation_available, dtype=torch.bool),
            "global_floorplan": global_floorplan,
            "global_floorplan_available": torch.tensor(global_floorplan_available, dtype=torch.bool),
            "plan": item["plan"],
            "progress": item["progress"],
            "history": item["history"],
            "target": item["target"],
        }
        if self.include_raw:
            result["raw_step"] = step
            result["sample_path"] = item["sample_path"]
            result["global_floorplan_path"] = item.get("global_floorplan_path")
        return result

    def __len__(self) -> int:
        return len(self.steps)

    @staticmethod
    def _pad_tensor_list(values: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        max_len = max((value.shape[0] for value in values), default=0)
        feature_dim = values[0].shape[1] if values else 0
        device = values[0].device if values else torch.device("cpu")
        padded = torch.zeros((len(values), max_len, feature_dim), dtype=values[0].dtype, device=device)
        mask = torch.zeros((len(values), max_len), dtype=torch.bool, device=device)
        for index, value in enumerate(values):
            length = value.shape[0]
            if length:
                padded[index, :length] = value
                mask[index, :length] = True
        return padded, mask

    @staticmethod
    def _pad_vector_list(values: list[torch.Tensor], pad_value: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        max_len = max((value.shape[0] for value in values), default=0)
        device = values[0].device if values else torch.device("cpu")
        padded = torch.full(
            (len(values), max_len),
            pad_value,
            dtype=values[0].dtype if values else torch.long,
            device=device,
        )
        mask = torch.zeros((len(values), max_len), dtype=torch.bool, device=device)
        for index, value in enumerate(values):
            length = value.shape[0]
            if length:
                padded[index, :length] = value
                mask[index, :length] = True
        return padded, mask

    @classmethod
    def collate_fn(cls, batch: list[dict[str, Any]]) -> dict[str, Any]:
        target_keys = batch[0]["target"].keys()
        history_keys = batch[0]["history"].keys()
        walls, wall_mask = cls._pad_tensor_list([item["plan"]["walls"] for item in batch])
        insertions, insertion_mask = cls._pad_tensor_list([item["plan"]["insertions"] for item in batch])
        result: dict[str, Any] = {
            "sample_id": [item["sample_id"] for item in batch],
            "split": [item["split"] for item in batch],
            "step_index": torch.stack([item["step_index"] for item in batch]),
            "observation": torch.stack([item["observation"] for item in batch]),
            "observation_available": torch.stack([item["observation_available"] for item in batch]),
            "global_floorplan_available": torch.stack([item["global_floorplan_available"] for item in batch]),
            "plan": {
                "walls": walls,
                "wall_mask": wall_mask,
                "insertions": insertions,
                "insertion_mask": insertion_mask,
            },
            "progress": torch.stack([item["progress"] for item in batch]),
            "history": {key: torch.stack([item["history"][key] for item in batch]) for key in history_keys},
            "target": {key: torch.stack([item["target"][key] for item in batch]) for key in target_keys},
        }
        if "raw_step" in batch[0]:
            result["raw_step"] = [item["raw_step"] for item in batch]
            result["sample_path"] = [item["sample_path"] for item in batch]
            result["global_floorplan_path"] = [item["global_floorplan_path"] for item in batch]
        global_floorplans = [item.get("global_floorplan") for item in batch]
        if all(torch.is_tensor(item) for item in global_floorplans):
            result["global_floorplan"] = torch.stack(global_floorplans)
        else:
            result["global_floorplan"] = None
        return result


def create_dataloader(
    dataset_path: str | Path = "processed_data",
    split: str | None = "train",
    batch_size: int = 8,
    shuffle: bool | None = None,
    num_workers: int = 0,
    **dataset_kwargs: Any,
) -> DataLoader:
    dataset = PrimitiveActionDataset(dataset_path=dataset_path, split=split, **dataset_kwargs)
    if shuffle is None:
        shuffle = split == "train"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=PrimitiveActionDataset.collate_fn,
        pin_memory=torch.cuda.is_available(),
    )


def create_dataset_from_config(
    dataset_path: str | Path = "processed_data",
    config: str | Path | None = None,
    batch_size: int = 8,
    num_workers: int = 0,
    splits: tuple[str, ...] = ("train", "val", "test"),
    overfit: bool = False,
    **dataset_kwargs: Any,
) -> tuple[dict[str, Any], ...]:
    del config
    allowed_dataset_kwargs = {
        "repo_root",
        "action_vocab_path",
        "image_size",
        "history_length",
        "load_images",
        "load_global_floorplan",
        "image_dtype",
        "normalize_images",
        "include_raw",
    }
    filtered_kwargs = {key: value for key, value in dataset_kwargs.items() if key in allowed_dataset_kwargs}
    packets = []
    for split in splits:
        effective_split = None if overfit else split
        loader = create_dataloader(
            dataset_path=dataset_path,
            split=effective_split,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=split == "train",
            **filtered_kwargs,
        )
        packets.append({"loader": loader, "sampler": None})
    return tuple(packets)
