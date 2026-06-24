import json
import pathlib


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_regularized_structured_policy_config_uses_generalization_settings() -> None:
    config_path = REPO_ROOT / "model_configs" / "structured_primitive_action_policy_regularized.json"

    config = json.loads(config_path.read_text())
    params = config["structured_primitive_action_policy_regularized"]
    train_config = params["train_config"]

    assert params["model_name"] == "structured_primitive_action_policy"
    assert params["dropout"] == 0.3
    assert params["loss_aux_wall_weight"] == 0.05
    assert params["loss_aux_point_role_weight"] == 0.05
    assert params["hidden_size"] == 384
    assert params["num_transformer_layers"] == 6
    assert train_config["lr"] == 1e-4
    assert train_config["early_stopping_metric"] == "loss"
    assert train_config["early_stopping_mode"] == "min"
    assert train_config["early_stopping_start_epoch"] == 150
    assert train_config["early_stopping_patience"] == 80
    assert train_config["checkpoint_frequency"] == 20


def test_regularized_training_script_uses_regularized_config_and_loss_best() -> None:
    script_path = REPO_ROOT / "train_structured_policy_regularized.py"

    script = script_path.read_text()

    assert '"--model_config"' in script
    assert '"model_configs/structured_primitive_action_policy_regularized.json"' in script
    assert '"--model_name"' in script
    assert '"structured_primitive_action_policy_regularized"' in script
    assert '"--epochs"' in script
    assert '"800"' in script
    assert '"--lr"' in script
    assert '"1e-4"' in script
    assert '"--early_stopping_metric"' in script
    assert '"loss"' in script
    assert '"--early_stopping_mode"' in script
    assert '"min"' in script
    assert "structured_primitive_action_policy_regularized" in script


def test_checkpoint_selector_defaults_to_periodic_checkpoint_candidates() -> None:
    script_path = REPO_ROOT / "select_structured_policy_checkpoint.py"

    script = script_path.read_text()

    assert "best_proxy.pt" in script
    assert "val_epoch_*.json" in script
    assert "epoch_{epoch:04d}.pt" in script
    assert "xy_mae" in script
    assert "aux_wall_acc" in script
    assert "aux_point_role_acc" in script
