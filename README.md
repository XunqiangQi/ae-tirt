# AE-TIRT: Autoencoder-Based Thurstonian IRT Toolkit

`ae-tirt` is a research-oriented Python package for estimating **Thurstonian IRT (TIRT)** models for forced-choice (FC) assessments via an autoencoder-based estimation strategy.

This package implements the methodological framework described in the manuscript:  
**An Autoencoder-Based Thurstonian IRT Model for Forced-Choice Assessment: Estimation Framework and Evaluation (2026)**.

## Overview

Forced-choice measurement requires estimation procedures that are both psychometrically interpretable and computationally scalable. AE-TIRT integrates:

- an amortized **encoder** for fast latent trait inference,
- a theory-constrained **decoder** that implements the Thurstonian pairwise comparison equation,
- end-to-end optimization for the joint estimation of item parameters and person traits.

The model is evaluated across a 3 × 2 × 2 × 2 simulation design (50 replications per condition), targeting numerically stable and scalable estimation while preserving parameter interpretability under the Thurstonian measurement model.

## Standard Errors

AE-TIRT provides an optional, post hoc standard error routine based on the **observed information** (negative Hessian) of the item response log-likelihood. The resulting SEs are asymptotic approximations **conditional on the encoder-based trait estimates**, analogous to post hoc SEs reported by conventional IRT software (e.g., BILOG-MG, flexMIRT). Two computation modes are available:

- **Full Hessian** (`use_hessian_diag=False`): more accurate; O(P²) computational cost.
- **Diagonal approximation** (`use_hessian_diag=True`): faster; neglects off-diagonal parameter covariances.

For standardized loadings (`w = sigmoid(u) × sign`), SEs are mapped to the loading scale via the delta method (chain rule on sigmoid). SE computation is **disabled by default**; see `examples/01_basic_usage.py` for a documented usage example.

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

## Quick Start

```python
import torch
from ae_tirt import AE_TIRT, Sim_data_TIRT, train_model, evaluate_model

sim = Sim_data_TIRT(
    npersons=300, ntraits=5, nblocks_per_trait=12, nitems_per_block=3,
    weight_sign=1, w_range=(0.65, 0.95), b_range=(-1.0, 1.0),
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
    model=model, train_data=sim.responses,
    batch_size=16, num_epochs=500, learning_rate=0.001,
    early_stopping_patience=20, penalty_weight=sim.responses.shape[1] * 1.0,
)

metrics = evaluate_model(model, sim.responses, sim.theta)
print(metrics["traits"]["overall"])

# Optional: post hoc standard errors (disabled by default)
# se = model.compute_standard_errors(
#     torch.tensor(sim.responses, dtype=torch.float32),
#     method="observed", use_hessian_diag=True, regularization=1e-4,
# )
# model.print_standard_errors_summary(se, num_examples=3)
# se_tables = model.build_standard_error_report(se, alpha=0.05)
```

## Usage

### Simulation Studies

```bash
python examples/01_basic_usage.py
python examples/02_simulation_study.py
```

For manuscript-scale Study-1 settings (24 conditions × 50 replications):

```bash
python scripts/run_all_experiments.py
```

### Real Data Analysis

Reproduce paper-aligned results from pre-trained model weights:

```bash
python scripts/run_real_data_analysis.py --weights data/real/pretrained_model.pth
```

The script loads pre-trained weights and reports: −2LL, AIC, BIC, ACC, AUC, F1, F2, Precision, Recall, Specificity, MCC, and parameter estimates (θ, w, b), ensuring consistency with published results.

Required inputs (`--data-dir data/real`): `X_responses.csv` (binary response matrix, persons × pairs), `pair_definitions.csv`, `item_trait_map.csv`, `weight_sign.csv`. See `data/real/README.md` for file provenance and format specification.

Alternatively, run the packaged example:

```bash
python examples/05_paper_real_data_example.py
```

## Project Documentation

See `docs/source/` for:

- theory and architecture notes,
- simulation protocol,
- empirical analysis protocol,
- API reference and reproducibility guidance.

## Citation

Please cite this software and the accompanying manuscript. Metadata is provided in `CITATION.cff`.