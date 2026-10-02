# FI-JEPA

## Current research status

The current v1 study is **closed as negative / boundary**, as recorded in [the final status](FINAL_STATUS_2026-09-30.md). The full learned representation performed worse than the matched raw-context Ridge control on the retained compact macrodata study. The one-stage `no_operator_split` condition is non-identifying and cannot support an operator-factorization claim.

Use [the canonical evidence gate](PAPER_EVIDENCE_GATE.md) and [post-run validity audit](audit/FIJEPA_V1_POSTRUN_VALIDITY_AUDIT.md) to interpret the retained evidence. This repository does not establish trading alpha or broad forecasting superiority. Any v2 outcome generation is a separate prospective study; the commands below are historical usage examples, not authorization to rerun the frozen v1 or access successor outcomes.

Financial-Informed Joint Embedding Predictive Architecture.

This repository contains the code, configs, benchmark harness, figures, and LaTeX source for an evidence-bounded FI-JEPA negative-result preprint.

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
