# Canonical FI-JEPA macrodata results

This file is auto-generated from `experiments/paper_results.json`. Historical result families are not inputs.

| Condition | Regression MSE | Regression MAE | Directional accuracy | Classification accuracy |
|---|---:|---:|---:|---:|
| Full FI-JEPA | 0.920145 ± 0.074493 | 0.683738 ± 0.048334 | 0.259259 ± 0.064150 | 0.234568 ± 0.021383 |
| No EMA | 0.872222 ± 0.033997 | 0.650183 ± 0.032533 | 0.259259 ± 0.064150 | 0.234568 ± 0.021383 |
| No financial regularizers | 0.915821 ± 0.071222 | 0.681149 ± 0.045780 | 0.259259 ± 0.064150 | 0.234568 ± 0.021383 |
| Monolithic predictor | 0.920145 ± 0.074493 | 0.683738 ± 0.048334 | 0.259259 ± 0.064150 | 0.234568 ± 0.021383 |
| No uncertainty heads | 0.912092 ± 0.051924 | 0.682212 ± 0.043498 | 0.271605 ± 0.085533 | 0.234568 ± 0.021383 |
| No memory | 0.895743 ± 0.044221 | 0.668672 ± 0.027983 | 0.271605 ± 0.085533 | 0.234568 ± 0.021383 |
| Raw-context Ridge | 0.441924 ± 0.000000 | 0.537767 ± 0.000000 | 0.629630 ± 0.000000 | 0.777778 ± 0.000000 |

## Paired seed-level comparisons

Deltas are `full - comparator`; for MSE, a negative value favors full FI-JEPA. The bootstrap intervals are descriptive because the frozen study contains only three seeds.

- `full_latent_minus_raw_context_ridge_mse`: mean delta 0.478221, descriptive 95% bootstrap interval [0.419935, 0.562148], n=3 paired seeds.
- `full_minus_no_ema_probe_mse`: mean delta 0.047923, descriptive 95% bootstrap interval [0.015231, 0.101932], n=3 paired seeds.
- `full_minus_no_financial_regularizers_probe_mse`: mean delta 0.004324, descriptive 95% bootstrap interval [0.001249, 0.007858], n=3 paired seeds.
- `full_minus_no_memory_probe_mse`: mean delta 0.024402, descriptive 95% bootstrap interval [0.000022, 0.058310], n=3 paired seeds.
- `full_minus_no_operator_split_probe_mse`: mean delta 0.000000, descriptive 95% bootstrap interval [0.000000, 0.000000], n=3 paired seeds.
- `full_minus_no_uncertainty_heads_probe_mse`: mean delta 0.008053, descriptive 95% bootstrap interval [-0.026530, 0.045108], n=3 paired seeds.

## Claim boundary

- This bounded macrodata suite is not evidence of trading alpha, market profitability, or state of the art forecasting.
- Three paired seeds provide descriptive uncertainty only; confidence intervals are retained but not treated as definitive significance evidence.
- Ablation differences are interpreted as component-removal evidence, not universal causal claims.
- The raw-context Ridge comparison is a matched downstream-evaluation control, not a parameter-matched neural architecture baseline.
