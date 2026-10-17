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

## Implementation scope and retained review amendments

The helper is included in the Python package; its child command requires a
repository checkout containing `scripts/train.py`. It does not package a second
trainer or change the existing training, evaluation, optimizer or checkpoint
selection rule. A supplied output path is replaced only in this explicit
command's effective configuration, with a newly reserved private attempt path.

The finite positive budget begins when that attempt is allocated and covers
preparation and child execution. Interpreter imports precede that budget. Cleanup
allows up to five seconds for the leader to exit; artifact validation and hashing
then run outside the child budget. POSIX cleanup signals the owned process group
with SIGTERM and then SIGKILL, including workers whose leader has already exited,
and reaps the direct child. Other platforms terminate the direct child only.
This is not a process sandbox or a guarantee about descendants that deliberately
leave their inherited process group.

JSON receipts use exclusive atomic publication and finite JSON. Configurations
and selected local tabular files are copied through regular-file descriptors.
Output validation parses exactly the same byte snapshots whose identities it
records. The final inventory must still match those validated identities before
an attempt can remain COMPLETED. Source identities are observations before
launch, not authentication of the executed code amid concurrent edits. The
command does not provide filesystem tamper isolation, exact resumed training,
recovery from disk failure, or a terminal record after uncatchable termination.

Independent review identified three concrete candidate defects before
publication. All original witnesses are retained with the final verification:

1. Opening a named FIFO could block before the next deadline check. The amended
   reader rejects nonregular inputs and checks the opened descriptor, using
   nonblocking open where available. Its bounded fork regression excludes
   interpreter import/shutdown from the attempt-budget assertion.
2. Validation and a later hash could observe different output bytes. Validation
   now retains each exact byte snapshot's hash and size; a changed final
   inventory changes the attempt to FAILED and retains both identities.
3. Waiting only for a process-group leader could leave a SIGTERM-ignoring worker
   alive after timeout. POSIX cleanup now signals the remaining group even when
   the leader has exited, then reaps the direct child. A controlled independent
   Linux subreaper fixture checks this case and reaps its test worker.

An initial 31-case author rerun recorded 30 passes and one subprocess-watchdog
failure even though the rejected FIFO's terminal FAILED receipt had been emitted.
The FIFO assertion was replaced with the bounded fork check above to isolate
file admission from Python/Torch startup and shutdown. The unsuccessful run is
retained; it is not silently relabeled as a passing run. Subsequent test counts
and the final independent decision are recorded in the verification receipt.

The inherited temporal workflow uses the same existing pinned checkout/setup
actions and explicit CPU dependency installation as the evaluation workflow,
verifies the tested commit, and limits numerical-library threads. The evaluation
workflow includes the new command/process tests and their lint checks. Neither
workflow dispatches the frozen real-data paper suite or regenerates its outputs.

## Verified development outcome — 2026-10-10

The combined final suite passed **169 tests** in 171.42 seconds: all 137 inherited
cases and 32 new cases. The run used Python 3.12.14, CPU Torch 2.5.1, NumPy 2.3.5
and pandas 2.2.3. Its ten retained warnings comprise seven inherited AMP
deprecations and three deprecations in bounded fork-based fixture supervision.
Isolated lint and formatting passed for the new implementation and both test
files; both amended workflows and their eight shell steps parsed successfully.

Independent review ran the 31-case suite after the first two corrections, then
seven affected cases on the final process-group correction, and inspected the
actual-validator replacement probe. Those seven overlap the existing cases;
they are not seven additional unique tests. The reviewer also verified the
intended integration parents and preservation of inherited tests, scripts,
protocols and scientific artifacts. The author ran the complete 169-case final
suite. The final review found no remaining concrete blocker within the declared
development scope.

See `verification/recorded_training_20261010/receipt.json` for source hashes,
environment, exact commands and evidence inventory. The raw initial failures,
subsequent corrections, independent review and full test output remain alongside
that receipt. GitHub integration and hosted verification are tracked separately;
local development verification does not reopen the closed scientific study.
