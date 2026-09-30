# FI-JEPA macrodata v1 post-run validity audit

Status: `EXECUTED_NEGATIVE / ONE_NONIDENTIFYING_CONTROL`

Canonical retained result: `experiments/paper_results.json` from protocol `FIJEPA_MACRODATA_PAPER_V1_20260909`.

## What survived the audit

The bounded three-seed macrodata suite executed successfully from a validation-selected checkpoint for every model condition. Per-seed run directories are isolated, the test split is not used for checkpoint selection, train-only normalization is retained, the deterministic raw-context Ridge control uses the same target/split/probe hyperparameters as the latent probe, and the following component removals alter executed numerical computation:

- `no_ema`;
- `no_financial_regularizers`;
- `no_uncertainty_heads`;
- `no_memory`.

The historical `deterministic` alias is correctly excluded because it is identical to `no_uncertainty_heads`.

## P0 finding: operator-split control is non-identifying

The frozen macro config sets `stage_names: [macro]`. In `FIJEPA.__init__`, split mode therefore instantiates one `PredictorStage`. The planned `no_operator_split` control changes the mode/name to `monolithic`, but also instantiates one `PredictorStage` with the same layer topology. `PredictorStage.name` is metadata and is not read by the forward computation.

The retained outcomes confirm the structural audit: `full` and `no_operator_split` are exactly equal across all reported downstream metrics for all three seeds.

Consequences:

1. Preserve this row in the raw evidence ledger.
2. Do not count it as a real ablation.
3. Do not infer that operator factorization helps, hurts, or is neutral from v1.
4. Do not retrofit a multi-stage full model into v1 after seeing the outcome.
5. Any future operator-factorization experiment must be separately versioned and frozen before its own outcome access and must have at least two executed stages in the full condition.

Tracked in issue #4.

## Scientific result

The real-data result is adverse to the maintained representation claim on this compact task. The fixed raw-context Ridge control substantially outperforms the full learned-latent probe on regression MSE, MAE, directional accuracy, and binary classification accuracy. In addition, the full model does not outperform `no_ema`, `no_financial_regularizers`, or `no_memory` on mean regression MSE in this three-seed matrix. `no_uncertainty_heads` is mixed but does not provide evidence favoring the full model.

This means the evidence does **not** support a claim that FI-JEPA improves GDP-growth prediction, that its finance-specific components are beneficial on this dataset, or that the learned latent is superior to the raw-context linear control.

## Inference boundary

There are only three training seeds and a small quarterly dataset. Paired bootstrap intervals are descriptive finite-seed summaries, not broad significance guarantees. The raw-context baseline is a matched downstream-evaluation control, not a parameter-matched neural architecture. No trading, profitability, market-generalization, or SOTA claim is authorized.

## Paper policy

The preprint must be regenerated from the canonical retained artifact and must:

- show the adverse raw-context baseline;
- label the monolithic/operator row as non-identifying;
- remove historical hand-entered seed-17 result tables and stale figures not generated from the canonical artifact;
- preserve the negative/mixed result without post-outcome rescue tuning.
