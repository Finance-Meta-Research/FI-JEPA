# FI-JEPA canonical paper evidence gate

## Current status: FAIL until canonical v1 artifact executes

The repository contains overlapping historical result families such as `current`, `new`, `publishable`, seed-7, and seed-17 outputs. Those files remain useful audit evidence, but they are not a predeclared paper evidence package and may not be selected after outcomes are seen.

The frozen paper protocol is `research/PAPER_PROTOCOL_V1.md` with protocol ID:

`FIJEPA_MACRODATA_PAPER_V1_20260909`

## Canonical execution

```bash
python scripts/run_paper_suite.py \
  --config configs/benchmark_macro.yaml \
  --seeds 7 17 27 \
  --epochs 15 \
  --out experiments/paper_results.json
python scripts/check_paper_evidence_gate.py
python scripts/make_canonical_paper_assets.py
```

Paper-facing tables and numerical prose must be generated from exactly one retained result artifact:

`experiments/paper_results.json`

## Required mechanism conditions

- `full`
- `no_ema`
- `no_financial_regularizers`
- `no_operator_split`
- `no_uncertainty_heads`
- `no_memory`

The historical `deterministic` alias is **not** a separate paper condition because the current implementation makes it identical to `no_uncertainty_heads`. Counting both would be pseudoreplication by configuration name rather than a real ablation.

The macro full configuration is stochastic, so removing uncertainty heads changes the executed model.

## Matched downstream control

Every seed must include exactly one explicit `raw_context_ridge` record. It uses the same train/test targets and the same Ridge/RidgeClassifier probe implementation as the FI-JEPA latent probe, but consumes flattened raw context instead of learned embeddings.

This is a matched downstream-evaluation control, not a claim of parameter-matched neural architecture comparison.

## Gate behavior

`scripts/check_paper_evidence_gate.py` fails closed when the canonical artifact is missing, under-seeded, contains duplicate/missing variant × seed cells, contains a no-op mechanism ablation, lacks the per-seed raw-context baseline, contains non-finite paper metrics, omits paired seed-level statistics, or has a mismatched protocol identity.

## Evidence policy

Older result files are not deleted merely because they are adverse or inconsistent. Once the canonical v1 run executes, the manuscript must not hand-pick historical numbers. Generated result assets live under `paper/generated/` and are derived only from `paper_results.json`.

Positive, null, mixed, and negative outcomes are all admissible. Do not change seeds, metrics, model conditions, or training budget after observing the canonical result and call it the same protocol.
