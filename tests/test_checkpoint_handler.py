import ast
from pathlib import Path

import pytest
import torch

from trainer import BaseTrainer
from trainer import PrimitiveActionTrainer


TRAINER_SOURCE = Path(__file__).resolve().parents[1] / "trainer.py"


def _trainer_tree() -> ast.Module:
    return ast.parse(TRAINER_SOURCE.read_text())


def _class_node(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"Class {name} not found")


def _method_node(class_node: ast.ClassDef, name: str) -> ast.FunctionDef:
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"Method {class_node.name}.{name} not found")


def test_checkpoint_handler_supports_last_best_and_numbered_epoch_files() -> None:
    tree = _trainer_tree()
    handler = _class_node(tree, "CheckpointHandler")
    save_checkpoint = _method_node(handler, "save_checkpoint")
    source = ast.get_source_segment(TRAINER_SOURCE.read_text(), save_checkpoint)

    assert '{"best", "last", "epoch"}' in source
    assert 'f"epoch_{epoch + 1:04d}.pt"' in source
    assert 'f"{kind}.pt"' in source


def test_training_loop_saves_periodic_checkpoint_every_hundred_epochs() -> None:
    tree = _trainer_tree()
    base_trainer = _class_node(tree, "BaseTrainer")
    train = _method_node(base_trainer, "train")
    source = ast.get_source_segment(TRAINER_SOURCE.read_text(), train)

    assert "checkpoint_frequency = int(self.training_config.get('checkpoint_frequency', 100))" in source
    assert "self._should_save_periodic_checkpoint(epoch, checkpoint_frequency)" in source
    assert 'self.save_checkpoint(epoch, avg_loss, kind="epoch")' in source


def test_early_stopping_skips_best_update_when_validation_did_not_run() -> None:
    trainer = object.__new__(BaseTrainer)
    trainer.early_stopping_enabled = True
    trainer.early_stopping_patience = 3
    trainer.early_stopping_metric = "xy_mae"
    trainer.early_stopping_mode = "min"
    trainer.early_stopping_min_delta = 0.0
    trainer.device = torch.device("cpu")
    trainer.log = lambda _message: None

    saved = []

    def save_checkpoint(*args, **kwargs):
        saved.append((args, kwargs))
        return {"model_state_dict": {}, "epoch": args[0] + 1}

    trainer.save_checkpoint = save_checkpoint

    best_metric, patience, best_state, should_stop = trainer._handle_early_stopping(
        epoch=0,
        avg_loss=12.3,
        val_metrics=None,
        best_metric_value=float("inf"),
        patience_counter=2,
        best_model_state=None,
    )

    assert best_metric == float("inf")
    assert patience == 2
    assert best_state is None
    assert should_stop is False
    assert saved == []


def test_validation_metric_must_exist_when_early_stopping_metric_is_named() -> None:
    trainer = object.__new__(BaseTrainer)
    trainer.early_stopping_metric = "xy_mae"

    with pytest.raises(KeyError, match="xy_mae"):
        trainer._get_current_metric(avg_loss=1.23, val_metrics={"loss": 9.87})


def test_early_stopping_start_epoch_delays_patience_counting() -> None:
    trainer = object.__new__(BaseTrainer)
    trainer.early_stopping_enabled = True
    trainer.early_stopping_patience = 1
    trainer.early_stopping_start_epoch = 10
    trainer.early_stopping_metric = "xy_mae"
    trainer.early_stopping_mode = "min"
    trainer.early_stopping_min_delta = 0.0
    trainer.device = torch.device("cpu")
    trainer.log = lambda _message: None
    trainer.save_checkpoint = lambda *args, **kwargs: pytest.fail("checkpoint should not be saved")

    best_metric, patience, best_state, should_stop = trainer._handle_early_stopping(
        epoch=4,
        avg_loss=12.3,
        val_metrics={"xy_mae": 0.2},
        best_metric_value=0.1,
        patience_counter=0,
        best_model_state=None,
    )

    assert best_metric == 0.1
    assert patience == 0
    assert best_state is None
    assert should_stop is False


def test_early_stopping_start_epoch_allows_stop_after_boundary() -> None:
    trainer = object.__new__(BaseTrainer)
    trainer.early_stopping_enabled = True
    trainer.early_stopping_patience = 1
    trainer.early_stopping_start_epoch = 10
    trainer.early_stopping_metric = "xy_mae"
    trainer.early_stopping_mode = "min"
    trainer.early_stopping_min_delta = 0.0
    trainer.device = torch.device("cpu")
    trainer.log = lambda _message: None
    trainer.save_checkpoint = lambda *args, **kwargs: pytest.fail("checkpoint should not be saved")

    best_metric, patience, best_state, should_stop = trainer._handle_early_stopping(
        epoch=9,
        avg_loss=12.3,
        val_metrics={"xy_mae": 0.2},
        best_metric_value=0.1,
        patience_counter=0,
        best_model_state=None,
    )

    assert best_metric == 0.1
    assert patience == 1
    assert best_state is None
    assert should_stop is True


def test_primitive_action_final_wandb_metrics_are_filtered_to_plottable_keys() -> None:
    trainer = object.__new__(PrimitiveActionTrainer)
    metrics = {
        "loss": 1.0,
        "xy_mae": 0.12,
        "action_type_acc": 0.5,
        "action_type_correct": 10,
        "count": 20,
        "xy_count": 8,
    }

    logged = trainer._final_wandb_metrics(metrics, "test")

    assert logged == {
        "test/loss": 1.0,
        "test/action_type_acc": 0.5,
        "test/xy_mae": 0.12,
    }
