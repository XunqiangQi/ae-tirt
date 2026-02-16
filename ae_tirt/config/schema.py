"""Configuration schema validators."""

from __future__ import annotations

from .defaults import TrainingConfig


def validate_training_config(config: dict):
    required = ["optimizer", "learning_rate", "batch_size", "num_epochs"]
    missing = [k for k in required if k not in config]
    if missing:
        raise ValueError(f"Missing training config keys: {missing}")

    if str(config["optimizer"]).lower() not in {"adam", "sgd", "adamw"}:
        raise ValueError("optimizer must be one of: adam, sgd, adamw")
    if float(config["learning_rate"]) <= 0:
        raise ValueError("learning_rate must be positive")
    if int(config["batch_size"]) <= 0 or int(config["num_epochs"]) <= 0:
        raise ValueError("batch_size and num_epochs must be positive integers")


def validate_config(config: dict):
    """Backward-compatible alias for training config validation."""
    # Accept legacy key lr and normalize to learning_rate.
    cfg = dict(config)
    if "learning_rate" not in cfg and "lr" in cfg:
        cfg["learning_rate"] = cfg["lr"]
    validate_training_config(cfg)


def merge_with_training_defaults(config: dict | None = None) -> dict:
    merged = TrainingConfig().__dict__.copy()
    if config:
        merged.update(config)
    validate_config(merged)
    return merged
