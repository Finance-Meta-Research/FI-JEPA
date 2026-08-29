#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

REQUIRED_VARIANTS = {
    "full",
    "no_ema",
    "no_financial_regularizers",
    "no_operator_split",
    "no_uncertainty_heads",
    "no_memory",
    "deterministic",
}


def fail(message: str) -> None:
    raise SystemExit(f"PAPER_EVIDENCE_GATE_FAIL: {message}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the one canonical paper-facing FI-JEPA evidence artifact.")
    parser.add_argument("--artifact", default="experiments/paper_results.json")
    parser.add_argument("--min-seeds", type=int, default=3)
    args = parser.parse_args()

    path = Path(args.artifact)
    if not path.is_file():
        fail(
            f"missing canonical artifact {path}; do not substitute current/new/publishable result families"
        )

    data = json.loads(path.read_text(encoding="utf-8"))
    seeds = data.get("seed_list")
    runs = data.get("runs")

    if not isinstance(seeds, list) or len(set(seeds)) < args.min_seeds:
        fail(f"paper artifact needs at least {args.min_seeds} unique predeclared seeds")
    if len(seeds) != len(set(seeds)):
        fail("seed_list contains duplicates")
    if not isinstance(runs, list) or not runs:
        fail("runs must contain per-seed paper-facing records")

    expected_seeds = set(seeds)
    cells = Counter()
    baseline_by_seed = set()

    for index, run in enumerate(runs):
        if not isinstance(run, dict):
            fail(f"run {index} is not an object")
        seed = run.get("seed")
        variant = run.get("variant")
        metrics = run.get("metrics")
        if seed not in expected_seeds:
            fail(f"run {index} uses undeclared seed {seed!r}")
        if not isinstance(variant, str) or not variant:
            fail(f"run {index} is missing variant")
        if not isinstance(metrics, dict) or not metrics:
            fail(f"run {index} has no metrics")
        cells[(variant, seed)] += 1
        if bool(run.get("is_downstream_baseline")):
            baseline_by_seed.add(seed)

    variants = {variant for variant, _ in cells}
    missing_variants = sorted(REQUIRED_VARIANTS - variants)
    if missing_variants:
        fail(f"missing required paper variants: {', '.join(missing_variants)}")

    for variant in sorted(REQUIRED_VARIANTS):
        for seed in seeds:
            count = cells[(variant, seed)]
            if count != 1:
                fail(f"expected exactly one {variant}/seed={seed} record; found {count}")

    missing_baseline_seeds = sorted(expected_seeds - baseline_by_seed)
    if missing_baseline_seeds:
        fail(
            "explicit downstream baseline is missing for seed(s): "
            + ", ".join(str(seed) for seed in missing_baseline_seeds)
        )

    print(
        "PAPER_EVIDENCE_GATE_PASS: "
        f"{len(seeds)} seeds; {len(runs)} per-seed records; "
        f"required variants complete; downstream baseline present for every seed."
    )


if __name__ == "__main__":
    main()
