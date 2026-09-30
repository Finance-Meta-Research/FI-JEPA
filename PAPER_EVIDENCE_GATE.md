# FI-JEPA canonical paper evidence gate

## Current status: EXECUTED_NEGATIVE / ONE_NONIDENTIFYING_CONTROL

The canonical v1 macrodata evidence run has executed and the retained artifact is frozen. Canonical workflow `34286562818` completed the predeclared 3-seed × 6-condition suite, evidence gate, asset generation, and retained-output commit. Reporting-only workflow `34287184603` regenerated the evidence-bound negative-result preprint from the unchanged retained result. Historical source head `969783021472a93b32e0df71cc71b94fec706edb` also passed the dedicated ablation-identifiability workflow `34287367098`.

The v1 result is adverse/negative on this compact macrodata task. The matched raw-context Ridge control outperforms the full FI-JEPA latent probe on the retained downstream metrics. `no_operator_split` is non-identifying under the frozen one-stage macro configuration, so that row remains retained for provenance but is excluded from mechanism inference. This status does not authorize post-outcome retrofit of v1, publication-strength superiority claims, or successor outcome access.

The repository also contains overlapping historical result families such as `current`, `new`, `publishable`, seed-7, and seed-17 outputs. Those files remain useful audit evidence, but they are not the canonical predeclared paper evidence package and may not be selected after outcomes are seen.

The frozen paper protocol is `research/PAPER_PROTOCOL_V1.md` with protocol ID:

`FIJEPA_MACRODATA_PAPER_V1_20260909`

## Canonical execution

The retained v1 execution used:

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

The v1 outcome is frozen. Do not rerun or alter v1 to repair a negative result or the non-identifying operator-split control. Any causal-preprocessing, identifying multi-stage operator control, or stronger matched-neural-baseline work belongs to a separately preregistered successor.

## Required mechanism conditions

- `full`
- `no_ema`
- `no_financial_regularizers`
- `no_operator_split`
- `no_uncertainty_heads`
- `no_memory`

The historical `deterministic` alias is **not** a separate paper condition because the current implementation makes it identical to `no_uncertainty_heads`. Counting both would be pseudoreplication by configuration name rather than a real ablation.

The macro full configuration is stochastic, so removing uncertainty heads changes the executed model.

For v1 interpretation, `no_operator_split` is retained as an executed provenance row but is not an identifying mechanism ablation under the frozen one-stage macro configuration.

## Matched downstream control

Every seed includes exactly one explicit `raw_context_ridge` record. It uses the same train/test targets and the same Ridge/RidgeClassifier probe implementation as the FI-JEPA latent probe, but consumes flattened raw context instead of learned embeddings.

This is a matched downstream-evaluation control, not a claim of parameter-matched neural architecture comparison.

## Gate behavior

`scripts/check_paper_evidence_gate.py` validates the frozen v1 saved evidence without importing model/training code. It requires the exact predeclared seeds, six-condition declaration, 15-epoch budget, explicit downstream-control flags, complete finite numeric paper metrics, and retained v1 configuration signatures. Missing/duplicate/extra run cells, booleans or numeric strings in metrics, duplicate JSON keys, and changes to those frozen declarations fail closed.

It checks the 84 downstream report values against the corresponding retained raw rows, checks model validation totals, and recomputes all six full-minus-comparator paired summaries using the frozen 10,000-replicate bootstrap specification. A missing comparison, altered mean/standard deviation/interval, or changed bootstrap metadata fails. This is arithmetic verification of saved evidence, not independent model reproduction or authentication of the original execution. Full raw configuration/history, dataset provenance, and execution receipts are outside this reader validator; inspect their retained records separately. The four frozen claim-boundary statements are also required unchanged before rendering.

`scripts/make_canonical_paper_assets.py` calls the same validator before writing either output, orders each condition by seed, and refuses to alias an output with its input or the other output. Focused saved-data regressions run with `python -m unittest discover -s tests -p 'test_paper_evidence_gate.py' -v` and require NumPy only.

A distinct configuration signature is not proof of an identifying intervention. The gate retains all six model conditions and explicitly reports `no_operator_split` as non-identifying; it does not call them six real mechanism ablations or declare publication readiness.

The post-run identifiability audit is additionally authoritative for interpretation: a retained row can exist in the frozen artifact without supporting a mechanism claim when the intervention does not change executed structure.

## Evidence policy

Older result files are not deleted merely because they are adverse or inconsistent. The manuscript must not hand-pick historical numbers. Generated result assets live under `paper/generated/` and are derived only from the retained `paper_results.json`.

Positive, null, mixed, and negative outcomes are all admissible. Do not change seeds, metrics, model conditions, or training budget after observing the canonical result and call it the same protocol.
