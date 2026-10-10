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
