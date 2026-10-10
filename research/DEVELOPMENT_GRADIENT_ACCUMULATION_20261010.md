# Development record: correct gradient-accumulation weights

Date: 2026-10-10. Base: `c94e1616eba1b2a7415ead4695def1d3b93094db`.
Scientific disposition: v1 remains **NEGATIVE / BOUNDARY, CLOSED**.

## Defect and correction

`Trainer.train_epoch` divided every microbatch mean loss by the configured
accumulation count, including a final group containing fewer batches. A tail
containing one batch with an accumulation count of four therefore contributed
only one quarter of its intended gradient. Unequal microbatch sizes also gave
individual observations unequal weights.

The implementation now reads at most one accumulation group ahead, counts its
actual observations, and weights each microbatch mean by `batch_size / group_size`.
It executes exactly one optimizer step, EMA update and existing memory update per
group. Empty batches and non-integer or nonpositive accumulation counts fail
before that group's model execution. The progress bar closes on failures.

For decomposable example losses this equals the gradient of the concatenated
group's mean. Batch-dependent penalties, such as covariance regularization, still
execute on each original microbatch. This change does **not** claim equivalence
between microbatch covariance penalties and a covariance penalty computed over
one larger batch. The existing last-microbatch memory write policy is unchanged.

When accumulation is one, the loss multiplier remains exactly one. The historical
batch-mean metric reporting and validation selection code is unchanged. The
canonical macrodata run did not configure accumulation above the default one;
its existing evidence, source revisions, configurations and paper are preserved.
Runs that use unequal or incomplete accumulation groups will change their update
trajectory and must record this revised source in a prospective run manifest.
Buffering more than one microbatch also changes the order in which a stochastic
dataset and model may consume shared random-number streams. This applies only
when accumulation exceeds one; it is another reason to treat such future runs as
source-versioned development rather than replays of historical outcomes.

## Verification

`PYTHONPATH=src python -m pytest -q tests/test_gradient_accumulation.py tests/test_model_smoke.py tests/test_temporal_integrity.py tests/test_paper_protocol.py`

The first local run could not collect because `tqdm` was unavailable; it generated
no model result. After installing test dependencies in an isolated target,
**23 tests passed** using Python 3.12 and CPU PyTorch 2.14.1. Four existing AMP API
deprecation warnings remain. The new optimizer tests compare actual SGD parameter
updates against independently concatenated groups, including unequal sizes,
partial tails and accumulation one, to `1e-12` tolerance in float64. Other tests
cover bounded loader consumption, invalid counts and empty batches.
A direct probe of the unchanged main-source trainer reproduced a final weight of
`0.965` where the actual grouped sample-mean reference was `1.175` on a four-row
linear-regression fixture. The corrected optimizer matches that reference.

The new correctness workflow runs this same bounded suite. These are constructed
engineering fixtures, not benchmark outcomes. No protected data, scientific
training campaign, new paper result or paid compute was used. Existing calendar
and asset-vocabulary work in PR #10 remains separate and unmodified.

## Next gate

An independent source review found no blocking correctness issue. GitHub integration
remains pending. A scientific successor remains
separate from this correction and from the closed predecessor result.
