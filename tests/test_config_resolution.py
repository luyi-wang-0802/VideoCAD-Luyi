from experiment import apply_command_line_training_overrides, resolve_config_entry


def test_resolve_config_entry_uses_matching_inner_model_name() -> None:
    configs = {
        "default_params": {
            "model_name": "structured_primitive_action_policy",
            "hidden_size": 384,
        }
    }

    key, params = resolve_config_entry(configs, "structured_primitive_action_policy")

    assert key == "default_params"
    assert params["model_name"] == "structured_primitive_action_policy"


def test_resolve_config_entry_prefers_explicit_top_level_key() -> None:
    configs = {
        "structured_primitive_action_policy": {
            "model_name": "structured_primitive_action_policy",
            "hidden_size": 128,
        },
        "default_params": {"model_name": "structured_primitive_action_policy", "hidden_size": 384},
    }

    key, params = resolve_config_entry(configs, "structured_primitive_action_policy")

    assert key == "structured_primitive_action_policy"
    assert params["hidden_size"] == 128


def test_command_line_training_overrides_ignore_none_and_apply_values() -> None:
    training_config = {
        "early_stopping_patience": 20,
        "val_frequency": 5,
        "command_line_overrides": {
            "early_stopping_patience": 7,
            "val_frequency": None,
        },
    }

    apply_command_line_training_overrides(training_config)

    assert training_config["early_stopping_patience"] == 7
    assert training_config["val_frequency"] == 5
