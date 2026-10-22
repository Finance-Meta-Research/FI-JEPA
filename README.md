# FI-JEPA

Financial-Informed Joint Embedding Predictive Architecture.

## Research status: NEGATIVE / BOUNDARY (v1 closed)

This repository retains code and an evidence-bounded negative-result manuscript for a compact public macroeconomic forecasting study. It does **not** establish publication readiness or a positive financial-prediction result. See [the final research status](FINAL_STATUS_2026-09-30.md).

The canonical v1 artifact is [`experiments/paper_results.json`](experiments/paper_results.json): six model conditions, seeds **7, 17, 27**, and **15 epochs** per condition. Full-latent minus raw-context Ridge probe MSE is **+0.4782213469**, with a descriptive three-seed bootstrap interval **[0.4199346602, 0.5621477664]**. Positive MSE deltas favor the control. Ridge is a matched downstream-evaluation control, not a parameter-matched neural baseline.

`no_operator_split` is numerically identical to `full` and structurally **non-identifying** under the frozen one-stage configuration. Retain it for provenance; it cannot support operator-factorization inference. No trading-alpha, profitability, broad market-generalization, or SOTA claim follows from v1.

The prospective v2 successor is **unauthorized and unexecuted** for outcome generation. It is a separate study and is not needed to close v1. Do not rerun or retrofit v1 to repair its negative result or failed control.

## Inspect the saved evidence (no training)

From the repository root, with NumPy installed:

```bash
python scripts/check_paper_evidence_gate.py
python -m unittest discover -s tests -p 'test_paper_evidence_gate.py' -v
python scripts/make_canonical_paper_assets.py \
  --tex /tmp/fijepa-review/canonical_results_table.tex \
  --md /tmp/fijepa-review/canonical_results_summary.md
```

These commands validate saved rows, recompute descriptive summaries from those rows, and render report assets. They do not load checkpoints or run models. The asset generator invokes the same gate before writing either output. Passing the reader gate establishes consistency of the saved report inputs, not publication readiness, causal identification, or independently reproduced model outcomes.

## What is included

- `src/fijepa/` — PyTorch implementation with EMA targets, operator heads, and optional prototype memory
- `configs/`, `scripts/` — model configs and historical training/evaluation entry points
- `experiments/` — retained results and original run receipt
- `research/PAPER_PROTOCOL_V1.md` — frozen pre-outcome protocol, preserved as historical protocol text
- `audit/FIJEPA_V1_POSTRUN_VALIDITY_AUDIT.md` — authoritative post-run interpretation, including the failed operator control
- `paper/` — negative-result LaTeX source and retained generated assets
- `PAPER_EVIDENCE_GATE.md` — reader checks and evidence policy

## Historical commands and provenance

The old two-seed, one-epoch benchmark quickstart, `make bench`, `make suite`, and `scripts/make_paper_assets.py` are exploratory/historical paths. They are **not** the canonical three-seed, six-condition, 15-epoch paper evidence path. Historical `current`, `new`, `publishable`, and single-seed result families cannot substitute for the frozen artifact. Training and the canonical runner are not needed for this saved-evidence review.

The manuscript can be built from `paper/main.tex` with a suitable LaTeX toolchain. Existing receipts remain historical records: the artifact matches `experiments/PAPER_RUN_RECEIPT.txt`, but the artifact hash recorded in `paper/PREPRINT_RECEIPT.txt` differs. The source-only reader closeout preserves this discrepancy rather than overwriting a receipt or claiming a fresh PDF rebuild. See [the reader closeout](audit/FIJEPA_V1_READER_CLOSEOUT.md) for exact checks and remaining review limits.

## Prospective evaluation

`scripts/eval_v2.py` evaluates a supplied checkpoint with observation-weighted
latent metrics, including a short final batch:

```bash
python scripts/eval_v2.py --config path/to/run.yaml --checkpoint path/to/best.pt --split validation
```

The checkpoint's saved model, data and seed configuration must match the supplied
configuration. The report records its metric version, checkpoint and resolved
configuration hashes, sample/batch counts, device and seed. For stochastic
multi-stage predictors, retain the batch size as well: random draws can still
depend on batching. The reducer is invariant to partitioning fixed predictions;
it does not redefine the model's stochastic policy.

The original evaluator, validation-based checkpoint selection, v1 artifacts and
closed negative/boundary conclusion retain their historical meaning. This command
does not authorize a successor experiment. See
`research/DEVELOPMENT_PAIRED_EVALUATION_20261010.md` for the numerical contract and
generated-fixture verification.

## Recorded development training

Use the explicit recorded command when a development run needs retained inputs,
logs and a terminal execution receipt:

```bash
python scripts/train_recorded.py --config path/to/dev.yaml --attempt-root path/to/attempts --max-seconds 120
```

Run this from a repository checkout with the package's dependencies installed.
Each invocation creates a new `attempt-*` directory and runs the existing trainer
against copied configuration and local CSV/parquet inputs. The configured output
directory is replaced in the effective configuration with that attempt's own
`training/` directory. Retries allocate another directory and retain the earlier
attempt. The optional `--source synthetic` or `--source macrodata` override is
recorded in the receipt.

`started.json` records the request. `finished.json` records `COMPLETED`, `FAILED`,
`TIMED_OUT` or `CANCELLED`, along with input/output identities and retained logs.
Completion also requires the requested epoch history and finite, matching
checkpoints; the bytes validated must match the final artifact inventory. A
started receipt without a terminal receipt marks an interrupted/incomplete
attempt. Ordinary errors return exit status 2; cancellation retains its receipt
and propagates the interruption.

The time budget covers attempt preparation and the training child. Child cleanup
allows a five-second termination grace; output validation and hashing follow
termination. POSIX cleanup signals the inherited process group and reaps the
direct child; other platforms terminate the direct child only. Regular local
files are required for configuration and tabular input.
This command does not provide exact checkpoint resumption, authenticate input
provenance, or protect against hostile filesystem changes or machine failure.
Its execution status always keeps `scientific_result: NOT_EVALUATED`. The closed
v1 negative/boundary finding and the requirements for a prospective successor
remain in force. See `research/DEVELOPMENT_RECORDED_TRAINING_20261010.md`.
