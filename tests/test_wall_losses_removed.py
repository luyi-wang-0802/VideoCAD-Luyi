from pathlib import Path


REMOVED_WALL_LOSS_TERMS = (
    "loss_wall_orthogonal",
    "loss_wall_length",
    "loss_wall_endpoint",
    "loss_wall_orthogonal_weight",
    "loss_wall_length_weight",
    "loss_wall_endpoint_weight",
)


def test_wall_geometry_losses_are_not_part_of_training_strategy() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    checked_files = [
        repo_root / "model/primitive_action_policy/config.py",
        repo_root / "model/primitive_action_policy/policy.py",
        repo_root / "model_configs/primitive_action_policy.json",
        repo_root / "trainer.py",
    ]

    offenders = []
    for path in checked_files:
        text = path.read_text()
        for term in REMOVED_WALL_LOSS_TERMS:
            if term in text:
                offenders.append(f"{path.relative_to(repo_root)} contains {term}")

    assert offenders == []
