.PHONY: bench figures paper test suite

bench:
	python scripts/benchmark.py --config configs/benchmark_macro.yaml --source macrodata --epochs 1 --ablations full no_ema no_financial_regularizers no_operator_split no_uncertainty_heads no_memory deterministic mlp_backbone no_multi_horizon reconstruction --seeds 7 17

suite:
	python scripts/benchmark.py --config configs/benchmark_macro.yaml --source macrodata --suite --ablations full no_ema no_financial_regularizers no_operator_split no_uncertainty_heads no_memory deterministic mlp_backbone no_multi_horizon reconstruction --seeds 7 17 --epochs 1 --out experiments/results_suite_macro.json

figures:
	python scripts/make_paper_assets.py

paper:
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

test:
	pytest -q
