# Prospective development contract: retained, bounded training attempts

Declared before implementation on 2026-10-10. Initial source: FI-JEPA evaluation
PR #12, `578cabafb02368985f73765c741bce46aedf11d7`. Planned integration: calendar,
normalization and sample ownership PR #10, `fb3f950414cc1c965702a4b83001dbc91624b44e`.
The v1 scientific result remains **NEGATIVE / BOUNDARY, CLOSED**.

## Observation and cause

`Trainer.__init__` reuses an existing output directory and replaces `config.json`.
`Trainer.fit` appends epochs to the previous history and replaces `best.pt` and
`last.pt`. A failed or interrupted command has no durable terminal receipt. A
retry can therefore overwrite earlier evidence or combine unrelated histories.
The data repair and optimizer/evaluation repair also occupy separate branches;
their combined behavior needs a real training and evaluation integration check.

## Proposed change and predicted consequence

Add an explicitly selected development command, `scripts/train_recorded.py`, and
an installable helper. The helper will run the existing `scripts/train.py` as a
child process, rather than implementing another optimizer or loss. Each invocation
will reserve a new private attempt directory, retain the supplied configuration,
retain a streamed copy of a selected local CSV/parquet input, record its effective
configuration and source identities, and capture both output streams to files.
The caller must supply a finite positive wall-time limit. No prior attempt path
will be reused, and completed, failed, cancelled and timed-out attempts will
remain distinguishable. A started attempt without a terminal receipt means
interrupted/incomplete work, not success.

Completion requires a zero child exit status and the expected configuration,
history and checkpoints. The terminal receipt will bind retained files by their
actual byte counts and SHA-256 values. This is a record of execution, not a
scientific verdict, authentication of dataset provenance, or a claim that a
checkpoint contains optimizer/RNG state for exact training resumption. A retry
starts a separate attempt. No automatic protected evaluation, benchmark matrix,
successor study, publication or release is part of this command.

## Discriminating verification and fixed development budget

Use only generated local fixtures. Reproduce replacement/mixed-history behavior
on the initial trainer, then require two recorded attempts to retain separate,
unchanged artifacts. Compare an actual one-epoch recorded training run with the
same ordinary training command, including every checkpoint tensor and reported
metric. Exercise malformed configuration, data-loader failure, child failure,
timeout and cancellation. Check that failure logs and partial outputs survive,
that finite JSON receipts match retained bytes, and that input snapshots remain
usable after an original input changes. Run the full combined inherited tests,
static checks, and independent source review. Each subprocess fixture has a
bounded wall-time; no paid compute or outcome-generating scientific campaign is
authorized or needed for this development verification.

## Integration and historical boundaries

Preserve both dependency histories and all their tests. Keep the pandas dtype
repair from #12 and the finite normalizer, calendar split, asset vocabulary and
copied samples from #10. Preserve all paper artifacts, frozen protocols, prior
failures and scientific conclusions. Append execution evidence and the independent
review result below after they exist; the contract above is prospective.
