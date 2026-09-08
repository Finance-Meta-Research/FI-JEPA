# FI-JEPA macrodata paper protocol v1 — frozen before canonical rerun

Status: `FROZEN_PRE_OUTCOME_PROTOCOL`

Protocol ID: `FIJEPA_MACRODATA_PAPER_V1_20260909`

## Scientific question

Does the FI-JEPA latent representation improve a fixed downstream GDP-growth prediction probe on the repository's real `statsmodels.datasets.macrodata` benchmark, and which implemented components materially affect that bounded result?

This is a small real-data representation-learning study. It is **not** a trading-alpha, profitability, market-SOTA, or broad finance-generalization experiment.

## Data and split

- dataset: `statsmodels.datasets.macrodata` packaged public macroeconomic time series;
- target: `gdp_growth = realgdp.pct_change()`;
- chronological split: 70% train / 15% validation / 15% test;
- normalization: fit on train rows only;
- context length: 4 quarters;
- horizon: 1 quarter;
- target window: 1;
- the test split is never used for checkpoint selection or hyperparameter selection.

The benchmark implementation constructs windows independently inside each split. This is conservative because it drops cross-boundary context rather than allowing information from future splits into earlier training data.

## Frozen seeds and budget

Seeds: `7, 17, 27`.

Training budget: 15 epochs for every FI-JEPA mechanism condition, identical batch size, optimizer, learning rate, weight decay, split, and downstream probe implementation.

Checkpoint selection: minimum validation total loss only. Paper-facing test metrics are computed from that validation-selected checkpoint.

## Paper-facing model conditions

1. `full`
2. `no_ema`
3. `no_financial_regularizers`
4. `no_operator_split`
5. `no_uncertainty_heads`
6. `no_memory`

The legacy `deterministic` name is excluded from the paper matrix because it is implementation-identical to `no_uncertainty_heads`; counting both would create a fake ablation. The macro full configuration is stochastic so `no_uncertainty_heads` is a genuine intervention.

Every mechanism condition must have an implementation signature distinct from `full` or the evidence gate fails.

## Matched downstream baseline

`raw_context_ridge` uses the same training targets, held-out test rows, normalization, and scikit-learn Ridge/RidgeClassifier hyperparameters as the latent probe. The only difference is representation input: flattened raw context versus FI-JEPA latent embeddings.

This is a matched **downstream evaluation control**, not a parameter-matched neural architecture baseline. The paper must say so explicitly.

## Primary and secondary outcomes

Primary descriptive outcome:

- test GDP-growth regression MSE from `Ridge(alpha=1.0)` on the learned latent representation.

Secondary diagnostics:

- regression MAE;
- directional accuracy;
- RidgeClassifier accuracy;
- best validation total loss;
- latent dispersion/memory diagnostics retained in raw run artifacts;
- trainable parameter counts.

## Statistics

All comparisons are paired by training seed. The independent unit is the seed, not a timestep or window.

For each ablation and the raw-context baseline, retain full-minus-comparator seed-level deltas and a fixed-seed 10,000-replicate paired bootstrap interval. With only three seeds these intervals are descriptive and must not be presented as definitive statistical significance.

## Fail-closed rules

The canonical artifact is `experiments/paper_results.json` and must be produced by `scripts/run_paper_suite.py`.

The paper evidence gate fails if:

- fewer than three unique predeclared seeds exist;
- any required variant × seed cell is missing or duplicated;
- any mechanism condition is a no-op relative to `full`;
- a per-seed raw-context downstream baseline is missing;
- any numeric paper metric is non-finite;
- paired seed-level statistics are absent;
- the protocol identity changes.

Historical `current`, `new`, `publishable`, or single-seed result families remain historical evidence and cannot be selected for paper claims after outcomes are seen.

## Claim boundary

Positive, null, mixed, or negative outcomes are all acceptable. Do not rescue-tune after inspecting the canonical result. Any later architecture, seed, metric, threshold, or dataset change is a separately versioned successor protocol.
