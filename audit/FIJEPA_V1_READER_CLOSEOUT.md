# FI-JEPA v1 saved-evidence reader closeout

Source reviewed: `c94e1616eba1b2a7415ead4695def1d3b93094db` on public `Finance-Meta-Research/FI-JEPA`.

## Scope and preserved decision

This is a reporting-code and documentation closeout, using only already-public retained evidence. V1 remains **NEGATIVE / BOUNDARY, CLOSED**. V2 remains a separate **unauthorized, unexecuted** prospective study. No training, checkpoint loading/inference, new model outcome, protected data, workflow dispatch, merge, or deployment was used for this closeout.

The review checked the canonical JSON, original run and preprint receipts, the evidence gate, canonical asset generator, paper protocol tests, frozen protocol, post-run validity audit, final research status, README, current `paper/main.tex`, generated table/summary, and relevant workflow triggers. Branch/PR inventory was checked before edits: #2/#3/#7 were merged, #6 was closed unmerged, no open PR duplicated this repair, and the existing reporting branch was identical to main. The prospective v2 branches were not executed or incorporated.

## Actual saved-row arithmetic

All **84** downstream metric entries (21 report rows × four metrics) match their corresponding raw-run values exactly. The model validation totals also agree. All six paired MSE comparisons reproduce the retained mean, sample standard deviation, and descriptive bootstrap endpoints from saved seed-level values using the retained seed `20260909` and 10,000 resamples.

The full-model probe MSE mean is `0.920145054658254`; raw-context Ridge is `0.44192370772361755`. Their full-minus-control paired mean is **`+0.4782213469346364`**, with retained and recomputed descriptive interval **`[0.4199346601963043, 0.5621477663516998]`**. These are calculations from saved outcomes, not independently rerun estimates. The three-seed intervals do not establish population-level significance.

`full` and `no_operator_split` have exactly identical saved downstream metrics at each seed (also identical retained validation totals). Their MSE deltas and interval are zero. The post-run structural audit remains authoritative: both frozen arms contain one executed predictor stage, so this comparison is **non-identifying**, not evidence of neutrality or operator-factorization benefit. Ridge remains a matched downstream-evaluation control, not a parameter-matched neural baseline.

## Reproduced reader defects and targeted repairs

Before repair, the CLI returned success for a missing MSE, boolean MSE, string `"NaN"` MSE, forged paired mean, missing ablation comparison, changed declared epoch budget, mismatched raw MSE, and missing full signature. The ungated generator rendered a forged paired mean of `-99.000000`.

The shared saved-evidence validator now rejects those cases, enforces the exact frozen seed/condition/budget declarations and baseline flags, checks typed finite metrics against saved raw rows, and recomputes every paired summary. Duplicate JSON keys, malformed intervals, missing/extra cells, and changed bootstrap metadata fail closed. Frozen claim-boundary text must remain intact. These checks do not authenticate execution or comprehensively validate raw configurations/history or dataset provenance.

The canonical generator invokes this validator before writing either output, pairs rows by seed rather than incidental order, and rejects same-path, symlink, and hardlink output aliases of its input or the other output. The gate's success text no longer calls six conditions six real mechanism ablations. The configuration-signature test name/comment now distinguishes metadata from executed identifiability.

README replaces the publication-readiness claim and old quickstart with the closed negative result and saved-evidence-only inspection commands. Historical two-seed/one-epoch paths are explicitly outside the frozen three-seed/six-condition/15-epoch evidence package. The manuscript already carries the correct negative-result and non-identifiability boundaries, so it was not rewritten.

## Verification performed

- `python -m unittest discover -s tests -p 'test_paper_evidence_gate.py' -v`: **11 tests passed**, including malformed-input subcases and generator regressions
- `python scripts/check_paper_evidence_gate.py`: **passed** on the retained artifact
- Canonical generator to temporary outputs: **passed**, and both table and summary are **byte-identical** to the retained generated files
- `python -m compileall -q src scripts tests`: **passed** (syntax only; no model imports)
- `git diff --check`: **passed**

Verification ran with one-CPU affinity, one-thread numerical-library settings, and a 2 GiB address-space cap; peak child RSS was approximately 30 MiB. Runtime was Python 3.12 with NumPy 2.3.5. The original experiment's environment remains recorded unchanged in its artifact/receipt. PyTorch and pytest are unavailable in this review environment; the full project suite, original model/protocol/temporal tests, and executable structural audit were **not run**. No dependency installation or model execution was attempted to fill that gap.

## Retained bytes and historical receipt discrepancy

Unchanged artifact SHA-256:

`525fe5f141bba765b3346c1aca3b327303703e5f98142b0f25d664211e10f7c8`

This matches `experiments/PAPER_RUN_RECEIPT.txt`. The retained config's SHA-256 also matches the config hash in the artifact:

`a1b82ab4df0c3a5c97ebd2660d8c2a20dbc62355f8ad8f4aafa16bb23867dccc`

However, `paper/PREPRINT_RECEIPT.txt` records artifact hash:

`fa42b42d1ba593dc9bcbb2fa4e3d98fc28715ea210760a0f800ebccd6a7aa493`

That differs from the current retained artifact. This historical discrepancy is preserved and flagged, not repaired by rewriting provenance. Both receipts, JSON, config, manuscript, final closure, and retained generated assets are unchanged by this patch. Regenerated table and summary hashes remain:

- Table: `c9fefacea09b532667af800643c9c5a065d5abe9b7e01540ee2f5694c120f3ca`
- Summary: `6d4d3dc41af182b747d59458e8a47f83b90dd6e3573a01483f1ffc70938e5d8b`

## Remaining gates and limits

1. Human review of this source-only reader patch; exact-head CI is not claimed (`[skip ci]` prevents automatic execution in this closeout).
2. If a newly verified PDF is wanted later, a separately scoped reporting-only build/visual inspection and an additive provenance reconciliation are still needed. No PDF build, visual inspection, or verification against the historical PDF hash was performed here.
3. No new outcome generation is required to close v1. Any v2 work needs separate protocol/code review and explicit execution authorization; this patch grants none.

No publication-readiness or independently reproduced scientific-result claim follows from these checks.
