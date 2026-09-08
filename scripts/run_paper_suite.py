#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from scripts.train import load_config
from fijepa.benchmark import ablation_signature, benchmark_suite

PAPER_VARIANTS = [
    "full",
    "no_ema",
    "no_financial_regularizers",
    "no_operator_split",
    "no_uncertainty_heads",
    "no_memory",
]
DEFAULT_SEEDS = [7, 17, 27]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "UNKNOWN"


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "UNKNOWN"


def bootstrap_paired(deltas: list[float], seed: int = 20260909, reps: int = 10000) -> dict:
    arr = np.asarray(deltas, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(reps, dtype=float)
    for i in range(reps):
        means[i] = rng.choice(arr, size=len(arr), replace=True).mean()
    return {
        "mean_delta": float(arr.mean()),
        "std_delta": float(arr.std(ddof=1) if len(arr) > 1 else 0.0),
        "bootstrap_95ci": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
        "n_seed_pairs": int(len(arr)),
        "bootstrap_seed": seed,
        "bootstrap_reps": reps,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Run the canonical, evidence-bound FI-JEPA macrodata paper suite.")
    p.add_argument("--config", default="configs/benchmark_macro.yaml")
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--out", default="experiments/paper_results.json")
    args = p.parse_args()

    if len(args.seeds) != len(set(args.seeds)) or len(args.seeds) < 3:
        raise SystemExit("paper suite requires at least three unique predeclared seeds")

    cfg_path = Path(args.config)
    cfg = load_config(str(cfg_path))
    cfg.data.source = "macrodata"
    if not cfg.model.stochastic:
        raise SystemExit("canonical full model must be stochastic so no_uncertainty_heads is a real intervention")

    full_sig = ablation_signature(cfg, "full")
    signatures = {name: ablation_signature(cfg, name) for name in PAPER_VARIANTS}
    for name in PAPER_VARIANTS:
        if name != "full" and signatures[name] == full_sig:
            raise SystemExit(f"paper ablation {name} is a no-op relative to full")

    raw = benchmark_suite(cfg, seeds=args.seeds, ablations=PAPER_VARIANTS, max_epochs=args.epochs)

    by_cell = {(r["ablation"], int(r["seed"])): r for r in raw}
    expected = {(variant, seed) for variant in PAPER_VARIANTS for seed in args.seeds}
    missing = sorted(expected - set(by_cell))
    if missing:
        raise SystemExit(f"missing benchmark cells: {missing}")

    runs = []
    for variant in PAPER_VARIANTS:
        for seed in args.seeds:
            r = by_cell[(variant, seed)]
            runs.append(
                {
                    "seed": seed,
                    "variant": variant,
                    "is_downstream_baseline": False,
                    "metrics": {
                        "probe_regression_mse": r["probe_regression"]["mse"],
                        "probe_regression_mae": r["probe_regression"]["mae"],
                        "probe_directional_accuracy": r["probe_regression"]["directional_acc"],
                        "probe_classification_accuracy": r["probe_classification"]["accuracy"],
                        "best_validation_total": r["best_val_total"],
                        "final_validation_total": r["final_val_total"],
                    },
                    "trainable_params": r["trainable_params"],
                    "checkpoint": r["checkpoint"],
                    "ablation_signature": signatures[variant],
                }
            )

    # Matched downstream control: the exact same Ridge(alpha=1) estimator,
    # target, train/test rows and normalization as the latent probe, but on raw
    # flattened context windows. It is derived only from the full-model run so
    # there is one explicit baseline record per seed rather than duplicated
    # pseudo-replicates across ablations.
    for seed in args.seeds:
        r = by_cell[("full", seed)]
        runs.append(
            {
                "seed": seed,
                "variant": "raw_context_ridge",
                "is_downstream_baseline": True,
                "metrics": {
                    "probe_regression_mse": r["baseline_regression"]["mse"],
                    "probe_regression_mae": r["baseline_regression"]["mae"],
                    "probe_directional_accuracy": r["baseline_regression"]["directional_acc"],
                    "probe_classification_accuracy": r["baseline_classification"]["accuracy"],
                },
                "estimator": "sklearn.linear_model.Ridge(alpha=1.0) / RidgeClassifier(alpha=1.0)",
            }
        )

    paired = {}
    full_mse = {seed: by_cell[("full", seed)]["probe_regression"]["mse"] for seed in args.seeds}
    for variant in PAPER_VARIANTS[1:]:
        deltas = [
            full_mse[seed] - by_cell[(variant, seed)]["probe_regression"]["mse"]
            for seed in args.seeds
        ]
        paired[f"full_minus_{variant}_probe_mse"] = bootstrap_paired(deltas)

    baseline_deltas = [
        full_mse[seed] - by_cell[("full", seed)]["baseline_regression"]["mse"]
        for seed in args.seeds
    ]
    paired["full_latent_minus_raw_context_ridge_mse"] = bootstrap_paired(baseline_deltas)

    payload = {
        "protocol_id": "FIJEPA_MACRODATA_PAPER_V1_20260909",
        "status": "EXECUTED_CANONICAL_PAPER_EVIDENCE",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "config_path": str(cfg_path),
        "config_sha256": sha256_file(cfg_path),
        "dataset": {
            "name": "statsmodels.datasets.macrodata",
            "source": "statsmodels packaged public macrodata dataset",
            "task": "chronological GDP-growth proxy prediction",
            "target": "gdp_growth",
            "split": {"train_frac": cfg.data.train_frac, "val_frac": cfg.data.val_frac},
            "normalization_fit": "train split only",
            "test_selection_policy": "never used for checkpoint selection",
        },
        "seed_list": list(args.seeds),
        "epochs": int(args.epochs),
        "required_variants": PAPER_VARIANTS,
        "downstream_baseline": "raw_context_ridge",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "statsmodels": package_version("statsmodels"),
            "scikit_learn": package_version("scikit-learn"),
        },
        "runs": runs,
        "paired_seed_level_statistics": paired,
        "raw_runs": raw,
        "claim_boundary": [
            "This bounded macrodata suite is not evidence of trading alpha, market profitability, or state of the art forecasting.",
            "Three paired seeds provide descriptive uncertainty only; confidence intervals are retained but not treated as definitive significance evidence.",
            "Ablation differences are interpreted as component-removal evidence, not universal causal claims.",
            "The raw-context Ridge comparison is a matched downstream-evaluation control, not a parameter-matched neural architecture baseline.",
        ],
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {out} sha256={sha256_file(out)}")


if __name__ == "__main__":
    main()
