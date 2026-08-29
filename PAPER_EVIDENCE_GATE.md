# FI-JEPA canonical paper evidence gate

## Current status: FAIL

The repository currently contains overlapping result families such as `current`, `new`, `publishable`, seed-7, and seed-17 outputs. Those files are useful historical artifacts, but they are not a single predeclared paper evidence package.

The manuscript should not choose between those families based on which one gives the stronger narrative.

## Canonical artifact

Paper-facing tables and figures should be generated from exactly one file:

`experiments/paper_results.json`

The artifact must contain:

- `seed_list`: at least 3 unique seeds frozen before the paper run;
- `runs`: per-seed records containing `seed`, `variant`, and `metrics`;
- exactly one record for every required variant × seed cell;
- an explicit downstream baseline for every seed, marked with `is_downstream_baseline: true`.

Required mechanism/model variants are currently:

- `full`
- `no_ema`
- `no_financial_regularizers`
- `no_operator_split`
- `no_uncertainty_heads`
- `no_memory`
- `deterministic`

Additional reconstruction/control baselines may be included, but required cells may not be omitted.

## Gate

Run:

```bash
python scripts/check_paper_evidence_gate.py
```

It fails closed when the canonical artifact is missing, under-seeded, has duplicate/missing cells, uses undeclared seeds, omits metrics, or lacks an explicit downstream baseline for any seed.

## Evidence policy

Older result files remain part of the audit trail and should not be deleted merely because they are adverse or inconsistent. Once the canonical run is complete, label older `current/new/publishable` files as historical/non-paper-facing and regenerate the manuscript tables and figures only from `paper_results.json`.

Do not reduce the seed requirement or remove a required variant to make the gate pass.
