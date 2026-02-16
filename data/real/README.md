# Paper Real Data (Included)

This directory directly includes the empirical data files used in the AE-TIRT paper workflow.

## Included Files

- `X_responses.csv`: wide binary response matrix (`userid` + pair columns).
- `Real_data.csv`: long-format response table compatible with R thurstonian-style pipelines.
- `item_trait_map.csv`: statement-to-trait mapping (`statement_id,trait_id`).
- `pair_definitions.csv`: pair definitions (`statement_j,statement_k`).
- `weight_sign.csv`: fixed sign constraints (`statement_id,weight_sign`).
- `items_2MFC.csv`: item/block metadata used during preprocessing.

## Data Source

The empirical FC dataset is adapted from the Japanese Big-Five Scale short form study:

- Namikawa et al. (2012)
- OSF repository referenced in manuscript: `https://osf.io/xzcrs/`

The files in this folder are project-ready derivatives used by the AE-TIRT analysis scripts.

## Usage

Run the real-data analysis directly:

```bash
python scripts/run_real_data_analysis.py --data-dir data/real --responses-file X_responses.csv
```

Paper-aligned hyperparameters are set as defaults in the script:

- optimizer: `adam`
- batch size: `16`
- epochs: `500`
- learning rate: `0.001`
- early stopping patience: `20`
- penalty weight factor: `1.0`
- seed: `42`
- weight constraint: `standardized`
- link function: `probit`

Or use long-format input:

```bash
python scripts/run_real_data_analysis.py --data-dir data/real --use-long-format
```

## Notes

- Keep participant-level identifiers anonymized.
- Check redistribution permissions and citation requirements before external release.
