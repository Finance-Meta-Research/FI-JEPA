# FI-JEPA

Financial-Informed Joint Embedding Predictive Architecture.

This repository contains the code, configs, benchmark harness, figures, and LaTeX source for a publishable FI-JEPA preprint.

## What is included

- `src/fijepa/` - PyTorch implementation of FI-JEPA with EMA targets, operator heads, and optional prototype memory
- `configs/` - base training config, benchmark configs, and ablations
- `scripts/` - training, evaluation, sweep, benchmark, and figure-generation entry points
- `experiments/` - saved benchmark outputs and summaries
- `runs/` - checkpoints from compact benchmark jobs
- `paper/` - LaTeX source and generated figures

## Quick start

```bash
pip install -e .
python scripts/train.py --config configs/base.yaml --source synthetic
python scripts/benchmark.py --config configs/benchmark_macro.yaml --source macrodata --suite --ablations full no_ema no_financial_regularizers no_operator_split no_uncertainty_heads no_memory deterministic mlp_backbone no_multi_horizon reconstruction --seeds 7 17 --epochs 1
python scripts/make_paper_assets.py
```

## Notes

The paper is written in LaTeX and is meant to compile directly from `paper/main.tex` without BibTeX.

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
