# Active prototype writes: prospective implementation correction

This implementation follows the pre-code contract in
[`research/DEVELOPMENT_ACTIVE_PROTOTYPE_WRITES_20261010.md`](../../research/DEVELOPMENT_ACTIVE_PROTOTYPE_WRITES_20261010.md).
FI-JEPA v1 remains **CLOSED, NEGATIVE / BOUNDARY**. Generated correctness tests and
the small demonstration below do not supply new market evidence.

## Runtime behavior

`PrototypeMemory.update(key, value, salience)` validates the complete batch and
the existing five buffers before committing any changes. Keys and values must
match the bank's floating dtype/device; salience is one finite, nonnegative real
number per row and must remain finite in the salience-buffer dtype. The supplied
salience precision is used for the existing write-threshold comparison. Double
bank writes retain double scalar precision.

Only positive-salience prototypes can be cosine merge candidates. Ties choose the
first maximum in ascending active-slot order. The existing merge decay,
unused-slot insertion, lowest-salience replacement, and once-per-call age update
are retained. A valid empty batch still increments ages once. Zero-salience writes
with a zero write threshold retain their previous inactive-slot semantics.

The method owns copies of all input tensors and stages all five buffers before
writing them back into their existing registered storage. Malformed later rows,
nonfinite values and unrepresentable cosine norms leave the live state unchanged.
Views into bank storage are read from their pre-call snapshot; external input
tensors are not directly changed. Once committed, a caller's view of bank storage
naturally observes the new bank state.

The additional storage is proportional to the bank plus the submitted batch:
`O(M * (key_dim + value_dim) + B * (key_dim + value_dim))`. This is a bounded
correctness cost. It is not concurrency isolation, crash recovery or a general
transaction system. Retrieval, loss, model and trainer code remain unchanged.

## Reproduction

From a fresh checkout/interpreter with the package and development dependencies
installed, run the relevant generated gate:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m pytest -q tests/test_active_prototype_writes.py \
  tests/test_gradient_accumulation.py tests/test_gradient_failure_atomicity.py \
  tests/test_model_smoke.py tests/test_recorded_checkpoint_structure.py
```

The new suite has 42 cases. It includes the four retained parent failures, exact
ordinary float32 comparison with the retained parent source, numerical rollback,
aliased inputs, all registered buffer identities, and the actual model/loss/
optimizer/EMA/memory path. The initial combined gate passed 58 cases; six existing
trainer `GradScaler` deprecation warnings remain visible in its raw receipt.
The publication parent later advanced to `18f96ad4849b7e7dbc1593513ab64c0e733b284f`
with an independent checkpoint-structure correction. Both histories and all 24
new parent cases were preserved. The final integrated gate includes those cases
alongside the original 58; its separate receipt records the actual result.

The fixed demonstration uses two generated examples, context length six, latent
width eight, and two SGD updates. It writes fresh artifacts and refuses an
existing output directory:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python scripts/demo_active_prototype_writes.py --output /tmp/fijepa-active-write-demo
```

It retains input tensors, the initial model, both raw gradient tensors, the final
model/bank checkpoint, exact replay, and the active-slot falsifier. It does not
implement exact optimizer/RNG resumption or authorize a scientific run.

## Evidence and remaining work

`research/verification/active_prototype_writes_20261010/` retains the original
source, four failing baseline witnesses, successful candidate command, independent
source review, and one separately capped demonstration attempt. Command receipts
include stdout, stderr, exit/timeout, wall time, environment, peak child RSS, and
source hashes before/after. Artifact hashes identify retained bytes; a fresh
interpreter and unchanged checkout are execution preconditions, not secure
attestation of arbitrary running code.

Any prospective scientific use must name this implementation correction in its
source-bound protocol. Earlier negative financial results, nonidentifying
ablations, frozen configurations and all prior development receipts remain
preserved. This change does not resolve those scientific boundaries.
