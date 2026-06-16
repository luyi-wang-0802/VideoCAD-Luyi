import ast
from pathlib import Path


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
