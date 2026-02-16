from ae_tirt.config import merge_with_training_defaults, validate_training_config


def test_training_config_validation_and_merge():
    cfg = merge_with_training_defaults({"optimizer": "adamw", "learning_rate": 5e-4})
    validate_training_config(cfg)
    assert cfg["optimizer"] == "adamw"
    assert cfg["learning_rate"] == 5e-4
