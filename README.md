# AE-TIRT: Autoencoder-Based Thurstonian IRT Toolkit

`ae-tirt` is a research-oriented Python package for estimating **Thurstonian IRT (TIRT)** models for forced-choice (FC) assessments via an autoencoder-based estimation strategy.

The package implements the methodological framework described in the manuscript:
**An Autoencoder-Based Thurstonian IRT Model for Forced-Choice Assessment: Estimation Framework and Evaluation (2026)**.

## Overview

Forced-choice measurement requires estimation procedures that are both psychometrically interpretable and computationally scalable. AE-TIRT integrates:

- an amortized **encoder** for fast latent trait inference,
- a theory-constrained **decoder** that implements the Thurstonian comparison equation,
- end-to-end optimization for the joint estimation of item parameters and person traits.

Across simulation conditions (3 x 2 x 2 x 2 design, 50 replications per condition), AE-TIRT targets numerically stable and scalable estimation while preserving parameter interpretability under the Thurstonian measurement model.

## Standard Errors (Observed-Information, Decoder Likelihood)

AE-TIRT includes an optional, post hoc standard error routine based on the **observed information** (negative Hessian) of the decoder log-likelihood. This yields asymptotic, approximate SEs **conditional on encoder-based trait estimates**. Two options are supported:

- **Full Hessian**: more accurate, computationally more expensive.
- **Diagonal approximation**: faster, approximate.

For standardized loadings (`w = sigmoid(u) * sign`), SEs are transformed to the loading scale via the chain rule.

## Core Components

- `ae_tirt/models`: AE-TIRT architecture and decoder-constrained model logic.
- `ae_tirt/data`: simulation, data loaders, preprocessing, and transforms.
- `ae_tirt/training`: training loop, optimizer utilities, and callbacks.
- `ae_tirt/evaluation`: trait and item recovery metrics and validators.
- `ae_tirt/experiments`: factorial condition generators and batch runners.
- `ae_tirt/config`: typed defaults and validation schema.

## Installation

```bash
pip install -e .
```

For development:

```bash
pip install -r requirements-dev.txt
```

For documentation:

```bash
pip install -r requirements-docs.txt
```

## Minimal Reproducible Example

```python
import torch
from ae_tirt import AE_TIRT, Sim_data_TIRT, train_model, evaluate_model

sim = Sim_data_TIRT(
    npersons=300,
    ntraits=5,
    nblocks_per_trait=12,
    nitems_per_block=2,
    weight_sign=0.5,
    w_range=(0.65, 0.95),
    b_range=(-1.0, 1.0),
).simulate()

model = AE_TIRT(
    input_dim=sim.responses.shape[1],
    latent_dim=sim.theta.shape[1],
    item_trait_map=torch.tensor(sim.item_trait_map, dtype=torch.long),
    pair_definitions=torch.tensor(sim.pair_definitions, dtype=torch.long),
    weight_sign=torch.tensor(sim.weight_sign_array, dtype=torch.float32),
    weight_constraint="standardized",
    link_function="probit",
)

history = train_model(
    model=model,
    train_data=sim.responses,
    batch_size=16,
    num_epochs=500,
    learning_rate=0.001,
    early_stopping_patience=20,
    penalty_weight=sim.responses.shape[1] * 1.0,
)

metrics = evaluate_model(model, sim.responses, sim.theta)
print(metrics["traits"]["overall"])

# Compute standard errors 
se = model.compute_standard_errors(
    torch.tensor(sim.responses, dtype=torch.float32),
    method="observed",
    use_hessian_diag=True,  # set False for full Hessian 
    regularization=1e-4,
)
model.print_standard_errors_summary(se, num_examples=3)

# report table with estimate, SE, z, p, CI
se_tables = model.build_standard_error_report(se, alpha=0.05)
print(se_tables["theta"].head(3))
print(se_tables["w"].head(3))
print(se_tables["b"].head(3))

# export CSV reports for theta / w / b 
paths = model.save_standard_errors(
    se_results=se,
    output_dir="se_reports",
    repeat=1,
    alpha=0.05,
)
print(paths)
```

## Typical Usage Scenarios

### Scenario A: Simulate Data and Analyze

Use this workflow when conducting methodological studies (parameter recovery, stability, runtime).

```bash
python examples/01_basic_usage.py
python examples/02_simulation_study.py
```

For manuscript-scale Study-1 settings (24 conditions x 50 replications):

```bash
python scripts/run_all_experiments.py
```

### Scenario B: Analyze Externally Provided Data

Use this workflow when FC response files and design metadata are provided by an external team.

```bash
python scripts/run_real_data_analysis.py --data-dir data/real
```

Dataset example aligned with the paper in this workspace:

```bash
python scripts/run_real_data_analysis.py \
  --data-dir data/real \
  --responses-file X_responses.csv \
  --batch-size 16 \
  --num-epochs 500 \
  --learning-rate 0.001 \
  --early-stopping-patience 20 \
  --penalty-weight-factor batch_size * 1.0\
```

Alternatively, run the packaged example:

```bash
python examples/05_paper_real_data_example.py
```

Required inputs: binary response matrix (or long-format), `pair_definitions.csv`, `item_trait_map.csv`, `weight_sign.csv`. Response columns = number of pairs; pair indices 1..D; D = number of statements. See `data/real/README.md` for file provenance (OSF source), AE-TIRT-derived files, and R-format.

**Evaluate with pre-trained weights** (paper-aligned results):

```bash
python scripts/evaluate_pretrained_model.py --weights data/real/pretrained_model.pth
```

This loads pre-trained model weights and reports key metrics (-2LL, AIC, BIC, ACC, AUC, F1, F2, parameter estimates) in the terminal, ensuring consistency with published results.

Paper real data: 5 traits, 50 statements, 25 pairs. Run `python scripts/verify_real_data.py` to check; use `--strict-real-data` to enforce. Defaults in `run_real_data_analysis.py`:

- `optimizer = adam`
- `batch_size = 16`
- `num_epochs = 500`
- `learning_rate = 0.001`
- `early_stopping_patience = 20`
- `penalty_weight_factor = 1.0`
- `seed = 42`
- `weight_constraint = standardized`
- `link_function = probit`

## Paper-Aligned Batch Experiments

Run the manuscript-aligned Study-1 settings (24 conditions x 50 replications):

```bash
python scripts/run_all_experiments.py
```

This script fixes:

- `repeat = 50`
- `num_epochs = 500`
- `batch_size = 16`
- `learning_rate = 0.001`
- `early_stopping_patience = 20`
- `penalty_weight_factor = 1`
- `seed = 42`

## Project Documentation

See `docs/source/` for:

- theory and architecture notes,
- simulation protocol,
- empirical analysis protocol,
- API reference and reproducibility guidance.

## Citation

Please cite this software and the accompanying manuscript. Metadata is provided in `CITATION.cff`.
