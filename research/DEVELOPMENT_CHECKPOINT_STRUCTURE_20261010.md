# Recorded training: checkpoint structure admission

Date: 2026-10-10. Parent: PR #13 at
`950bbed595e750498e7f8360720eff0ec703aa78`.
Scientific disposition: **NEGATIVE / BOUNDARY; v1 CLOSED**.

## Specification before implementation

The recorded-training wrapper verifies checkpoint configuration, selected epoch,
history and finite tensor values, but accepts any nonempty tensor dictionary as
the model. A finite dictionary containing unrelated keys can therefore help
produce a `COMPLETED` execution receipt despite being unusable as the declared
FI-JEPA architecture.

Engineering hypothesis: comparing complete checkpoint keys, tensor shapes and
dtypes with the configured model prevents structural false completion, without
changing trainer equations, checkpoint bytes, validation selection, metrics,
global RNG state or any retained scientific result. Both best and last
checkpoints must satisfy one common structure. The falsifier is admission of an
incompatible state, rejection of a genuine generated fixture, or changed RNG or
artifact bytes during validation.

The expected structure will be built on the Torch meta device, which allocates
no parameter storage. Explicit feature columns determine input width. When the
configuration requests automatic feature inference, width is inferred from the
first checkpoint's designated input projection and held fixed for the second
checkpoint. That fallback checks internal structural consistency; it does not
independently attest actual feature names, ordering or source-data semantics.
The normal trainer's float32 parameter contract remains in effect.

Budget: at most 24 new generated-fixture cases, one pre-fix run, one targeted
post-fix run, and at most two complete existing CPU correctness suites of 120
seconds each. No scientific campaign, protected evaluation, paid compute,
historical checkpoint repair or frozen-protocol change is authorized by this
engineering record. Existing small artificial trainer equivalence tests may run
as part of the correctness suite.

## Verification

Implementation and verification are pending at specification time. Append the
observed results and source identities below; preserve this earlier status as
the record of the specification stage.

## Implemented result and retained verification

Complete model keys, tensor shapes and float32 dtypes are now compared with the
configured FI-JEPA structure. The expected model is constructed on the meta
device; both best and last checkpoints use the same expected structure. Explicit
feature columns remain authoritative. Automatic feature width uses the first
checkpoint input projection, or the downstream model width for a supported
empty MLP/conv backbone. Original trainer, checkpoint bytes, selection and
scientific evidence are unchanged.

The original 23 generated cases produced **17 failures / 6 passes** before the
repair, then **23 passes in 11.23 seconds**. Independent review found that a
valid zero-layer MLP with inferred features has no input projection parameter.
The 24th fixture reproduced that compatibility failure (**1 failure in 11.83
seconds**); the follow-up handles empty MLP/conv backbones without weakening
complete-state admission. Review then found no scoped blocker and did not rerun
any tests.

The final complete CPU suite passed **193 tests in 112.59 seconds**: all 169
inherited tests and 24 new tests. It ran under an explicit 120-second timeout,
with plugin autoload disabled and full test selection preserved. **13 warnings**
remain, including inherited Torch AMP and Python fork deprecations. The earlier
full run was manually interrupted during prolonged inherited subprocess checks
(exit130); its partial output is retained and is not counted as a successful
complete suite. Its precise duration was not recorded. The extra isolated
review regression was a concrete compatibility check within the declared
24-case scope, not a new scientific hypothesis.

Changed implementation and tests pass isolated Ruff E4/E7/E9/F/I. Environment:
Python 3.12.14, CPU Torch 2.14.1, pandas 3.0.6. Raw baseline, intermediate,
interrupted, review-failure and final logs and source hashes are retained in
`research/verification/checkpoint_structure_20261010/`.

A complete execution receipt still does not prove scientific efficacy or that
an adversarially fabricated history represents actual training. This repair
checks checkpoint structure. The closed negative v1 result is preserved; no
protected outcome or scientific campaign was opened. Exact-source hosted
verification and review remain required for the published change.
