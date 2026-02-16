"""AE-TIRT public API."""

from .__version__ import __version__

__all__ = [
    "__version__",
    "AE_TIRT",
    "Sim_data_TIRT",
    "train_model",
    "evaluate_traits",
    "evaluate_model",
    "evaluate_item_parameters",
    "generate_tirt_conditions",
    "generate_tirt_conditions_study2",
    "batch_simulate_and_train",
    "DEFAULT_TRAINING_CONFIG",
    "DEFAULT_SIMULATION_CONFIG",
    "DEFAULT_EXPERIMENT_CONFIG",
]


def __getattr__(name):
    if name == "AE_TIRT":
        from .models.ae_tirt import AE_TIRT

        return AE_TIRT
    if name == "Sim_data_TIRT":
        from .data.simulator import Sim_data_TIRT

        return Sim_data_TIRT
    if name in {"train_model"}:
        from .training import train_model

        return train_model
    if name in {"evaluate_traits", "evaluate_model", "evaluate_item_parameters"}:
        from .evaluation import evaluate_item_parameters, evaluate_model, evaluate_traits

        return {"evaluate_traits": evaluate_traits, "evaluate_model": evaluate_model, "evaluate_item_parameters": evaluate_item_parameters}[name]
    if name in {"generate_tirt_conditions", "generate_tirt_conditions_study2", "batch_simulate_and_train"}:
        from .experiments import batch_simulate_and_train, generate_tirt_conditions, generate_tirt_conditions_study2

        return {
            "generate_tirt_conditions": generate_tirt_conditions,
            "generate_tirt_conditions_study2": generate_tirt_conditions_study2,
            "batch_simulate_and_train": batch_simulate_and_train,
        }[name]
    if name in {"DEFAULT_TRAINING_CONFIG", "DEFAULT_SIMULATION_CONFIG", "DEFAULT_EXPERIMENT_CONFIG"}:
        from .config import DEFAULT_EXPERIMENT_CONFIG, DEFAULT_SIMULATION_CONFIG, DEFAULT_TRAINING_CONFIG

        return {
            "DEFAULT_TRAINING_CONFIG": DEFAULT_TRAINING_CONFIG,
            "DEFAULT_SIMULATION_CONFIG": DEFAULT_SIMULATION_CONFIG,
            "DEFAULT_EXPERIMENT_CONFIG": DEFAULT_EXPERIMENT_CONFIG,
        }[name]
    raise AttributeError(f"module 'ae_tirt' has no attribute {name!r}")
