# Real Data

Empirical forced-choice data used in the AE-TIRT paper. Source: Japanese Big-Five Scale short form (Namikawa et al., 2012); repository: https://osf.io/xzcrs/

## File provenance

| File | Description |
|------|-------------|
| `dat_2MFC.csv` | Raw response data from OSF (participant × block responses). |
| `items_2MFC.csv` | Block design from OSF: M1/M2 = traits of the two statements per block, R1/R2 = keying. |
| `item_trait_map.csv` | Statement–trait mapping in AE-TIRT format (50 statements, trait_id 1..5). |
| `X_responses.csv` | Binary response matrix (persons × 25 pairs) for AE-TIRT. |
| `pair_definitions.csv` | Pair definitions (25 pairs; statement IDs 1..50) for AE-TIRT. |
| `weight_sign.csv` | Statement keying (+1/−1) for AE-TIRT. |
| `Real_data.csv` | Long-format (person, itemC, response) for use with R Thurstonian IRT packages. |

The AE-TIRT inputs (`item_trait_map.csv`, `X_responses.csv`, `pair_definitions.csv`, `weight_sign.csv`) were derived from `dat_2MFC.csv` and `items_2MFC.csv`. `Real_data.csv` is the long-format export for R.

## Design

5 traits, 50 statements, 25 blocks (one pair per block). Trait IDs are consecutive 1..5.

## Usage

Install the project first (`pip install -e .` from the repo root). Then:

```bash
# Verify 5 traits, 50 statements, 25 pairs and consistency with items_2MFC
python scripts/verify_real_data.py

# Fit AE-TIRT (matrix input)
python scripts/run_real_data_analysis.py --data-dir data/real --responses-file X_responses.csv

# Fit AE-TIRT (long-format input)
python scripts/run_real_data_analysis.py --data-dir data/real --use-long-format
```

Default hyperparameters in the script match the paper (e.g., Adam, batch size 16, 500 epochs). Use `--strict-real-data` to enforce 50 statements, 5 traits, 25 pairs.

## Citation

Cite the original data source (Namikawa et al., 2012) and the OSF repository when using or redistributing these files.
