# Prospective development contract: active prototype batch writes

Recorded 2026-10-10 09:57:23 UTC, before implementation or reproduction. Parent: FI-JEPA
PR #13 at `950bbed595e750498e7f8360720eff0ec703aa78`. New session:
`20261010-0945-active-prototype-writes`. The user's new continuation authorizes
this finite engineering pass. All earlier budgets remain closed, and the v1
scientific disposition remains **NEGATIVE / BOUNDARY, CLOSED**.

## Observed code defect and falsifier

`PrototypeMemory.update` currently compares a proposed key against every allocated
slot, including inactive zero slots. With an allowed cosine merge threshold of
zero, an inactive zero vector can outrank a negatively aligned active prototype.
The method then merges into that inactive slot without increasing `filled`.
A subsequent insertion can overwrite that newly activated prototype. The existing
method also mutates early rows before discovering a malformed or nonfinite later
row, and caller tensor views can change while the same batch is being written.

The discriminating generated witness will require a new opposite key to occupy a
new slot when no active prototype reaches the threshold. Invalid later rows must
leave all five memory buffers byte-identical to their pre-call values. The original
source and all observed failures will be retained, even if the initial diagnosis
does not reproduce.

## Exact proposed runtime contract

Repair only `PrototypeMemory.update` and narrowly required helpers; preserve
checkpoint buffer names, shapes, ordinary forward/retrieval and the model/trainer
interfaces. This is a prospective implementation correction, not a new empirical
model claim.

- Admit a nonempty batch `key[B,key_dim]`, `value[B,value_dim]` and one salience
  per row; retain the existing single-key/vector shorthand. Empty batches may
  retain the historical one age increment when their shapes are valid.
- Keys and values must be finite real floating tensors matching the corresponding
  buffer dtype/device. Salience must be finite, real and nonnegative and convertible
  to the salience-buffer dtype without nonfinite values. Shapes are checked for the
  entire batch before mutation. Buffers must have the declared shapes and finite
  values, and `filled` must be an exact count within capacity.
- Require finite `decay` in [0,1], cosine `merge_threshold` in [-1,1], and
  nonnegative finite `write_threshold`. These are admission bounds, not tuning.
- Own detached copies of every input before processing. Compute changes on private
  buffer copies. A numerical or input failure leaves the live buffers unchanged.
  This is call-level failure behavior, not concurrency or interruption isolation.
- Merge candidates are only slots whose salience is positive. Choose the first
  maximum in ascending active-slot order, retaining Torch's existing tie rule.
  Preserve the existing threshold comparison, decay update, unused-slot insertion,
  lowest-salience replacement and once-per-call age increment.
- Form scalar salience values in the salience-buffer dtype. Ordinary float32
  default arithmetic remains identical; nondefault double precision no longer
  rounds an intermediate scalar through the ambient default dtype.
- Commit only complete finite candidate buffers. Caller-owned tensors and their
  autograd graphs receive no writes or gradients.

These controls do not authenticate a dataset or establish physical/temporal memory
independence. They do not repair the scientific meaning of unused or detached
auxiliary losses, change memory retrieval gradients, or reinterpret v1 results.

## Bounded implementation evidence

Use generated finite tensors to exercise active-only matching, merge/insertion/
replacement/ties, no-active and zero-salience behavior, empty/single/batched inputs,
caller aliasing and all-or-nothing shape/numerical failure. Compare ordinary valid
updates against the actual retained parent implementation. Exercise the existing
FIJEPA model and loss through one bounded optimizer/target/memory update.

Budget: at most five local test or reproduction commands, 360 seconds cumulative
command wall time, at most two review repair cycles, one independent reviewer
test/probe invocation of at most 90 seconds, and one retained generated demo
attempt of at most 60 seconds. Each new actual-model case uses at most four
examples, context length eight, latent width eight and two optimizer steps.
One CPU numerical thread; no GPU, provider, paid, protected or scientific campaign.
The demo will retain its exact command, environment, source and input identities,
stdout/stderr/exit or failure, checkpoint/memory bytes and interpretation.
Stop optional verification after sufficient success.

## Publication and history

Append each attempt, failure, review and actual budget use to the canonical state
and this record. Publish only after independent review, as a distinct stacked draft
on the refreshed #13 head, with complete tree/blob verification. Preserve all
prior source bindings, protocols, adverse outcomes and parallel PRs. New scientific
use requires its own source-bound successor protocol and evaluation boundary.

