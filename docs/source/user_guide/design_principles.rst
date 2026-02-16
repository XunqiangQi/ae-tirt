Design Principles
=================

The package is organized around research workflow stages:

1. data generation / loading,
2. model fitting,
3. recovery evaluation,
4. condition-wise experiment orchestration.

Architectural principles
------------------------

- **Model transparency**: decoder follows explicit Thurstonian equations.
- **Reproducibility first**: deterministic seeds and persistent output files.
- **Modularity**: decoupled modules for training, evaluation, and experiments.
- **Benchmark readiness**: scripts aligned with manuscript settings.
