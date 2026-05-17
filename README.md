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
