# External Data Template for AE-TIRT

Place collaborator-provided files under `data/real/` and follow these schemas:

1. `responses.csv`
   - Shape: `N x L`
   - Values: `0` or `1`
   - One row per respondent, one column per pair comparison.

2. `pair_definitions.csv`
   - Required columns: `item1,item2`
   - Row count must equal `L`
   - Item IDs are 1-based indices.

3. `item_trait_map.csv`
   - Required column: `item_trait`
   - Row count defines number of items `D`.

4. `weight_sign.csv`
   - Required column: `weight_sign`
   - Row count must equal `D`
   - Typical values: `1` and `-1`.

After preparing files, run:

```bash
python scripts/run_real_data_analysis.py --data-dir data/real
```

R-compatible long format is also supported:

- `Real_data.csv` with columns `person,itemC,response` (+ optional metadata columns),
- `item_trait_map.csv` may use `statement_id,trait_id`,
- `pair_definitions.csv` may use `statement_j,statement_k`.

Run with:

```bash
python scripts/run_real_data_analysis.py --data-dir data/real --use-long-format
```
