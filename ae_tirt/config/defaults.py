"""Default configuration objects for reproducible studies."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TrainingConfig:
    optimizer: str = "adam"
    learning_rate: float = 1e-3
    batch_size: int = 16
    num_epochs: int = 500
    early_stopping_patience: int = 20
    penalty_weight_factor: float = 1.0
    weight_constraint: str = "standardized"
    link_function: str = "probit"
    seed: int = 42


@dataclass(frozen=True)
class SimulationConfig:
    npersons: int = 500
    ntraits: int = 5
    nblocks_per_trait: int = 12
    nitems_per_block: int = 2
    weight_sign: float = 0.5
    w_range: tuple[float, float] = (0.65, 0.95)
    b_range: tuple[float, float] = (-1.0, 1.0)
    comb_blocks: str = "random"


@dataclass(frozen=True)
class ExperimentConfig:
    repeat: int = 1
    sim_data_root: str = "Sim_data_result"
    use_gpu_if_available: bool = True


DEFAULT_TRAINING_CONFIG = TrainingConfig()
DEFAULT_SIMULATION_CONFIG = SimulationConfig()
DEFAULT_EXPERIMENT_CONFIG = ExperimentConfig()

# Backward-compatible dict style configuration.
DEFAULT_CONFIG = asdict(DEFAULT_TRAINING_CONFIG)
