# Development correction: shared panel calendar

This is an engineering correction for future panel runs. The closed v1
macrodata result and every retained artifact remain unchanged. No training,
protected evaluation, new research outcome, or trading claim is part of this
change. The canonical disposition remains in `FINAL_STATUS_2026-09-30.md`.

## Follow-up: reject context leakage from invalid window counts

The previous constructor applied `int()` to every length and horizon without
admission checks. A declared horizon of zero starts its target on the final
context observation, and a negative horizon starts even earlier. Fractional
counts silently selected a different experiment (for example, `0.9` became
zero). Empty horizon lists failed later with an unrelated `max()` exception.

Both the direct dataset API and `build_datasets` now require nonempty, strictly
positive integer horizons and positive integer context/target lengths, rejecting
booleans and lossy casts. The top-level builder validates before generating or
loading observations. Valid integer configurations retain the same row indices,
target convention, sorted/deduplicated horizons, and tensor values. NumPy integer
counts remain accepted. No model, loss, trained parameter, seed, frozen protocol,
retained result, or transaction-cost assumption is changed.

Twenty additional generated-fixture regression cases exercise rejection,
early failure before data generation, and exact multihorizon row membership.
The local admission probe uses the exact source validation function; complete
tensor integration is checked by the existing bounded Temporal Integrity CI job.

## Defects

1. `build_datasets` split each asset's lifetime separately. For staggered
   listings, a late asset's training rows could occur after an early asset's
   validation rows. The shared normalizer and model could therefore learn
   information from a future calendar interval.
2. `split_panel_by_time` counted rows, so a timestamp with multiple assets could
   be divided across training and validation.
3. Each dataset built its own asset-to-embedding mapping. If an asset disappeared
   before validation, a surviving asset could be assigned a different embedding.

## Corrected contract

Fractions count sorted unique observed timestamps. Every row from a timestamp
belongs to the same split. The train/validation/test calendar boundaries are
shared by every asset, with windows still constructed entirely within a split.
Normalization is fit only on the globally admitted training rows. One stable
identifier mapping is copied to all three datasets; it does not fit return
statistics or establish that a newly observed asset has a trained embedding.

Balanced panels and the single-asset v1 macrodata panel retain their previous
split membership. Staggered panels deliberately change membership and require a
new versioned run/protocol; historical numbers must never be relabelled as if
they had used the corrected pipeline.

## Regression checks

`tests/test_panel_calendar_integrity.py` uses generated development fixtures to
check staggered dates, complete-timestamp membership, invariance of training
statistics/windows to future shocks, asset identity after a delisting, balanced
panel equivalence, and invalid split fractions. The existing temporal CI job now
includes this file.

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest tests/test_temporal_integrity.py tests/test_panel_calendar_integrity.py -q
```

Local source compilation and direct execution of the exact pandas split
function passed. The old function divided a five-row timestamp between splits;
the corrected function retained all five rows together. PyTorch was unavailable
locally, so whole-package/model integration is delegated to the PR's temporal
CI job and must not be described as locally passed.
