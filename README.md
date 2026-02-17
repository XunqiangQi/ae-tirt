# AE-TIRT: Autoencoder-Based Thurstonian IRT Toolkit

`ae-tirt` is a research-oriented Python package for estimating **Thurstonian IRT (TIRT)** models for forced-choice (FC) assessments using an autoencoder-based estimation strategy.

The package operationalizes the methodological framework described in the manuscript:
**An Autoencoder-Based Thurstonian IRT Model for Forced-Choice Assessment: Estimation Framework and Evaluation (2026)**.

## Abstract-Style Overview

Forced-choice measurement requires estimation methods that are both psychometrically interpretable and computationally scalable. AE-TIRT combines:

- an amortized **encoder** for fast latent trait inference,
- a theory-constrained **decoder** that implements the Thurstonian comparison equation,
- end-to-end optimization for joint recovery of item parameters and person traits.

Across simulation conditions (3 x 2 x 2 x 2 design, 50 replications per condition), AE-TIRT is designed to provide strong numerical stability and substantially reduced runtime relative to full MCMC workflows, while preserving parameter interpretability under the Thurstonian measurement model.

## Methodological Positioning

AE-TIRT is intended as a **computationally efficient complement** to SEM- and MCMC-based TIRT estimation.

- **Compared with SEM**: avoids dependence on ill-conditioned weight-matrix inversion workflows in fragile settings.
- **Compared with MCMC**: trades posterior uncertainty quantification for substantial speed gains.
- **Scope condition**: best suited to exploratory, operational, and large-scale scoring settings where speed and stability are prioritized.

### Optional Standard Errors (Post Hoc)

Following the paper's formulation, AE-TIRT is trained as a point-estimation framework and **does not compute standard errors by default**.  
When inferential support is needed, standard errors can be computed post hoc from the observed information matrix (negative Hessian of decoder log-likelihood):

- **Full Hessian**: more accurate, computationally expensive.
- **Diagonal approximation**: faster, approximate.
- **Scale handling**: for standardized loadings (`w = sigmoid(u) * sign`), SEs are transformed to the loading scale via the chain rule.

This provides asymptotic, approximate SEs conditional on encoder-based trait estimates.

## Core Components

- `ae_tirt/models`: AE-TIRT architecture and decoder-constrained model logic.
- `ae_tirt/data`: simulation, loaders, preprocessing, and transforms.
- `ae_tirt/training`: training loop, optimizer utilities, and callbacks.
- `ae_tirt/evaluation`: trait/item recovery metrics and validators.
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

## Minimal Reproducible Usage

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

# Optional: post-hoc standard errors (not computed during training)
se = model.compute_standard_errors(
    torch.tensor(sim.responses, dtype=torch.float32),
    method="observed",
    use_hessian_diag=True,  # set False for full Hessian
    regularization=1e-4,
)
model.print_standard_errors_summary(se, num_examples=3)

# Optional: academic-style report table (estimate, SE, z, p, CI)
se_tables = model.build_standard_error_report(se, alpha=0.05)
print(se_tables["theta"].head(3))
print(se_tables["w"].head(3))
print(se_tables["b"].head(3))

# Optional: export CSV reports for theta / w / b (no true-value columns)
paths = model.save_standard_errors(
    se_results=se,
    output_dir="se_reports",
    repeat=1,
    alpha=0.05,
)
print(paths)
```

## Standard Usage Scenarios

### Scenario A: Simulate Data Then Analyze

Use this path when conducting methodological studies (parameter recovery, stability, runtime).

```bash
python examples/01_basic_usage.py
python examples/02_simulation_study.py
```

For manuscript-scale Study-1 settings (24 conditions x 50 replications):

```bash
python scripts/run_all_experiments.py
```

### Scenario B: Analyze Externally Provided Data

Use this path when another team provides FC response files and design metadata.

```bash
python scripts/run_real_data_analysis.py --data-dir data/real
```

Paper dataset example in this workspace:

```bash
python scripts/run_real_data_analysis.py \
  --data-dir data/real \
  --responses-file X_responses.csv \
  --batch-size 16 \
  --num-epochs 500 \
  --learning-rate 0.001 \
  --early-stopping-patience 20 \
  --penalty-weight-factor 1.0
```

or run the packaged example:

```bash
python examples/05_paper_real_data_example.py
```

Required files in `data/real`:

- `responses.csv`: binary response matrix (`N x L`, values must be `0/1`).
- `pair_definitions.csv`: exactly `L` rows with columns `item1,item2` (1-based item indices).
- `item_trait_map.csv`: exactly `D` rows with column `item_trait` (trait ID per statement).
- `weight_sign.csv`: exactly `D` rows with column `weight_sign` (typically `+1/-1`).

Consistency constraints checked by the script:

- number of response columns must equal rows in `pair_definitions.csv`,
- max item index in pair definitions must be within `1..D`,
- length of `weight_sign.csv` must equal length of `item_trait_map.csv`,
- responses must be strictly binary (`0/1`).

Supported alternative (R-compatible) schemas:

- `pair_definitions.csv` can also be `statement_j,statement_k`.
- `item_trait_map.csv` can also be `statement_id,trait_id`.
- responses can be loaded from long-format `Real_data.csv` by adding `--use-long-format`.

Bundled real-data files and provenance notes are provided in `data/real/README.md`.

Paper-aligned real-data defaults in `scripts/run_real_data_analysis.py`:

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
