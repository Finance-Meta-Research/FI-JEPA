# Development record: paired metrics and sample-mean evaluation

Date: 2026-10-10. Parent: `20a1fc3f533b484e6ed88f7180c38b340378755f`
(gradient-accumulation PR #11, based on main `c94e1616eba1b2a7415ead4695def1d3b93094db`).
Development began at `853dbb4649281920f8ca412e663f9e3eb797da62`;
the concurrently published derivative-validation follow-up was integrated before
publication, preserving its complete canonical-state history and code.
Scientific disposition remains **NEGATIVE / BOUNDARY; v1 CLOSED**.

## Reproduced defects

The shared directional-accuracy function compared arrays without aligning their
observation/output axes. A four-observation target column `[-2, -1, 1, 2]` and an
identical flat prediction vector produced an unintended 4-by-4 comparison. The
unchanged source returned MSE 0, MAE 0, and directional accuracy **0.5** for those
perfect predictions. The same expression was duplicated in the benchmark score
helper. Ridge can naturally produce the flat prediction side of this mismatch.

The original standalone evaluator averaged batch means. For three observations
with squared errors `[0, 0, 9]` and context norms `[1, 1, 4]`, partitioned as two
observations plus one, that reduction gives MSE **4.5** and mean norm **2.5**. The
observation means are **3.0** and **2.0**. These constructed numbers demonstrate
implementation errors; they are not empirical findings about financial data.

## Implemented numerical contract

The regression metric path treats `[N]` and `[N, 1]` as the same scalar-target
representation. Other multi-output inputs must have exactly matching `[N, K]`
shapes. Empty, nonfinite, complex, nonnumeric and mismatched inputs fail before
a score is emitted. Zero retains the existing nonnegative-direction convention.
MSE and MAE keep the existing uniform-output scikit-learn definitions. The
benchmark helper delegates to this same contract instead of maintaining a second
copy. Valid one-dimensional inputs, including the canonical paper probe's input
shape, retain their previous numerical definitions.

The new `fijepa.evaluation.evaluate_latents` computes final-stage mean prediction
error against the final target horizon. For N observations and latent width L,
MSE is the sum of all squared coordinate errors divided by N times L. Mean norm
is the sum of the N context-vector norms divided by N. It counts observations
actually consumed rather than relying on loader length, and handles generator
loaders and unequal batches. Subtraction, squaring and reductions use float64;
this avoids overflowing float16 before the aggregate is computed. Empty loaders,
invalid batch/output shapes and nonfinite results fail explicitly.

Evaluation runs without an autograd graph or memory writes. It restores each
module's original train/eval flag, including a target encoder held in a different
mode from the parent. This helper does not choose a checkpoint or perform any
optimizer, EMA or memory update.

## Explicit version and executable path

`scripts/eval_v2.py` opts into `fijepa_latent_sample_mean_v2`. It loads a real
checkpoint, validates its saved model/data/seed configuration, builds the chosen
validation or test loader through the installable package's nested configuration
loader, and invokes the new evaluator. Explicit execution devices are honored.
Batch size, workers and execution device can differ from training; the resulting
resolved configuration hash records those choices.

The JSON report includes the SHA-256 of the exact checkpoint bytes loaded,
resolved configuration SHA-256, split, seed, batch size, sample count, batch count,
feature ordering, horizons, Torch version, execution device and stochastic-policy
flag. Nonfinite JSON numbers are disallowed. These identities help trace an
evaluation; a configuration hash does not replace a source-data content manifest
or a predeclared scientific protocol.

The existing stochastic predictor still samples intermediate stages even in eval
mode. The CLI seeds those draws separately from model initialization. For fixed
predictions the reduction is invariant to how observations are partitioned, up
to floating-point rounding. The implementation does not claim that a stochastic
multi-stage model produces identical predictions across different batch sizes;
seed and batch size must both be retained in prospective run records.

## Historical and scientific boundaries

`scripts/eval.py`, `Trainer.evaluate`, training/checkpoint selection, the frozen
protocol, retained evidence, paper assets, negative outcomes and source receipts
are unchanged. In particular, this change does not retroactively replace v1's
batch-mean validation metrics or reopen the non-identifying operator-split row.
The accumulation repair in PR #11 remains intact; the calendar-split repair in
PR #10 remains separate. No outcome-generating campaign was executed, no
protected evaluation data were opened, and no paper evidence was regenerated.

A successor scientific run must separately freeze its hypothesis, data identities,
split, baselines, metrics, source revision and finite budget before execution.
The new evaluator is an implementation tool, not a scientific success verdict.

## Verification

Environment: Python 3.12, CPU PyTorch 2.14.1. The initial focused run passed all
**60 new tests**. They include an independent closed-form partition oracle, a
float16 overflow case, actual Ridge fitting with column test targets, an actual
FI-JEPA model with occupied memory and multiple predictor stages, unchanged
weights/buffers/modes, and subprocess execution of both validation and test CLI
paths using a newly generated, untrained checkpoint. The CLI tests also reject
shape-compatible model, data and seed configuration mismatches before output.

The existing optimizer, model, temporal-integrity and paper-protocol suite adds
**23 passing tests**, for **83 total**. Existing AMP deprecation warnings remain.
The unchanged ablation-identifiability audit passes and continues to classify
`no_operator_split` as non-identifying. New evaluation code passes isolated Ruff
checks and formatting; original-source whitespace checks pass. The first shell
Ruff invocation was unavailable on PATH; invoking the installed module resolved
that environment issue. An initial local style check also inherited unrelated
parent-directory settings; the committed workflow uses explicit isolated rules.

Commands (with the package and test dependencies installed):

```bash
python -m pytest -q tests
python scripts/check_ablation_identifiability.py
python -m ruff check --isolated --select E4,E7,E9,F,I src/fijepa/evaluation.py scripts/eval_v2.py tests/test_regression_metrics.py tests/test_latent_evaluation.py
```

Independent source review found no blocking issue after inspecting the paired
metric contract, reductions, actual-model memory path, mode restoration, CLI,
checkpoint binding and scientific boundaries. The reviewer did not rerun tests.
The added workflow executes the bounded generated-fixture suite against the
exact PR commit on CPU. Hosted results and newest-parent verification are
recorded in the canonical state and PR as they complete. Existing scientific
workflows and their activation branches are unchanged.

After integrating the concurrent parent at `20a1fc3f533b484e6ed88f7180c38b340378755f`, all **25** of its optimizer/model/temporal/protocol tests passed locally, including both new derivative-failure cases. Together with the unchanged 60 new tests this verifies **85 cases**. Six existing AMP deprecation warnings occur in this expanded parent suite.

## Hosted failure and pandas compatibility correction

The first exact-commit evaluation workflow at
https://github.com/Finance-Meta-Research/FI-JEPA/actions/runs/38025913766
ran 80 tests successfully and failed five generated-checkpoint setup cases.
It installed pandas 3.0.6 and NumPy 2.4.6. The unchanged feature loader attempted
to assign float32 data in place into integer columns; pandas 3 rejects that dtype
conversion. The same failure was reproduced locally using an isolated pandas
3.0.6 installation. The initial ablation audit succeeded.

The follow-up replaces whole feature columns with the existing grouped,
forward-filled, float32 values. This allows the explicit dtype conversion to
take effect instead of assigning into an incompatible retained dtype. It changes
neither per-asset carry-forward, index alignment, temporal splitting nor the
training population. FitNormalizer already converts inputs to float32, so its
arithmetic and downstream tensor values remain the same. The new regression
checks mixed integer/nullable/float columns, interleaved assets, nonsequential
indexes, input and metadata immutability, exact float32 values and normalization.

This follows pandas' documented ban on incompatible in-place dtype changes:
https://pandas.pydata.org/docs/whatsnew/v3.0.0.html#removal-of-prior-version-deprecations-changes

After the repair, the complete local suite passes **86 tests** on pandas 3.0.6.
The new dtype regression and existing temporal suite also pass **7 tests** on
the prior pandas environment. These runs do not lower dependency requirements or
remove any test. Independent review approved the two-line production change
and regression contract.

Six inherited AMP deprecation warnings remain. Pandas 3 also exposes an existing
non-writable NumPy-to-Torch view warning in MarketWindowDataset; the separate
calendar/asset-vocabulary PR #10 already includes copied sample tensors. This
evaluation revision neither performs in-place input mutation nor duplicates
that separate change.

The failed hosted run and its environment are retained here. No scientific
dataset, protected outcome, retained paper evidence or training campaign was
used to diagnose this compatibility failure.
