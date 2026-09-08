#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

LABELS = {
    "full": "Full FI-JEPA",
    "no_ema": "No EMA",
    "no_financial_regularizers": "No financial regularizers",
    "no_operator_split": "Monolithic predictor",
    "no_uncertainty_heads": "No uncertainty heads",
    "no_memory": "No memory",
    "raw_context_ridge": "Raw-context Ridge",
}


def fmt(mean: float, std: float) -> str:
    return f"{mean:.6f} $\\pm$ {std:.6f}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--artifact", default="experiments/paper_results.json")
    p.add_argument("--tex", default="paper/generated/canonical_results_table.tex")
    p.add_argument("--md", default="paper/generated/canonical_results_summary.md")
    args = p.parse_args()

    data = json.loads(Path(args.artifact).read_text(encoding="utf-8"))
    grouped = defaultdict(list)
    for run in data["runs"]:
        grouped[run["variant"]].append(run)

    order = list(data["required_variants"]) + [data["downstream_baseline"]]
    stats = {}
    for variant in order:
        rows = grouped[variant]
        mse = np.asarray([r["metrics"]["probe_regression_mse"] for r in rows], dtype=float)
        mae = np.asarray([r["metrics"]["probe_regression_mae"] for r in rows], dtype=float)
        direction = np.asarray([r["metrics"]["probe_directional_accuracy"] for r in rows], dtype=float)
        accuracy = np.asarray([r["metrics"]["probe_classification_accuracy"] for r in rows], dtype=float)
        stats[variant] = {
            "mse": (float(mse.mean()), float(mse.std(ddof=1))),
            "mae": (float(mae.mean()), float(mae.std(ddof=1))),
            "direction": (float(direction.mean()), float(direction.std(ddof=1))),
            "accuracy": (float(accuracy.mean()), float(accuracy.std(ddof=1))),
        }

    tex_lines = [
        "% AUTO-GENERATED from experiments/paper_results.json; do not hand edit.",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Canonical three-seed macrodata study. Mean $\\pm$ sample standard deviation across predeclared seeds. Regression MSE/MAE are lower-is-better; directional/classification accuracy are higher-is-better.}",
        "\\label{tab:canonical_macro}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Condition & Regression MSE & Regression MAE & Directional acc. & Classification acc. \\\\",
        "\\midrule",
    ]
    for variant in order:
        s = stats[variant]
        tex_lines.append(
            f"{LABELS.get(variant, variant)} & {fmt(*s['mse'])} & {fmt(*s['mae'])} & {fmt(*s['direction'])} & {fmt(*s['accuracy'])} \\\\" 
        )
    tex_lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]

    tex_path = Path(args.tex)
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")

    md = [
        "# Canonical FI-JEPA macrodata results",
        "",
        "This file is auto-generated from `experiments/paper_results.json`. Historical result families are not inputs.",
        "",
        "| Condition | Regression MSE | Regression MAE | Directional accuracy | Classification accuracy |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant in order:
        s = stats[variant]
        md.append(
            f"| {LABELS.get(variant, variant)} | {s['mse'][0]:.6f} ± {s['mse'][1]:.6f} | "
            f"{s['mae'][0]:.6f} ± {s['mae'][1]:.6f} | {s['direction'][0]:.6f} ± {s['direction'][1]:.6f} | "
            f"{s['accuracy'][0]:.6f} ± {s['accuracy'][1]:.6f} |"
        )

    md += [
        "",
        "## Paired seed-level comparisons",
        "",
        "Deltas are `full - comparator`; for MSE, a negative value favors full FI-JEPA. The bootstrap intervals are descriptive because the frozen study contains only three seeds.",
        "",
    ]
    for name, result in sorted(data["paired_seed_level_statistics"].items()):
        lo, hi = result["bootstrap_95ci"]
        md.append(f"- `{name}`: mean delta {result['mean_delta']:.6f}, descriptive 95% bootstrap interval [{lo:.6f}, {hi:.6f}], n={result['n_seed_pairs']} paired seeds.")

    md += [
        "",
        "## Claim boundary",
        "",
    ] + [f"- {line}" for line in data.get("claim_boundary", [])]

    md_path = Path(args.md)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(tex_path)
    print(md_path)


if __name__ == "__main__":
    main()
