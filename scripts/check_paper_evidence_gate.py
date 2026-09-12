#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

REQUIRED_VARIANTS = {
    "full",
    "no_ema",
    "no_financial_regularizers",
    "no_operator_split",
    "no_uncertainty_heads",
    "no_memory",
}
REQUIRED_BASELINE = "raw_context_ridge"
EXPECTED_PROTOCOL = "FIJEPA_MACRODATA_PAPER_V1_20260909"


def fail(message: str) -> None:
    raise SystemExit(f"PAPER_EVIDENCE_GATE_FAIL: {message}")


def finite_metrics(metrics: dict, label: str) -> None:
    for key, value in metrics.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and not math.isfinite(float(value)):
            fail(f"{label} metric {key} is non-finite")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the one canonical paper-facing FI-JEPA evidence artifact.")
    parser.add_argument("--artifact", default="experiments/paper_results.json")
    parser.add_argument("--min-seeds", type=int, default=3)
    args = parser.parse_args()

    path = Path(args.artifact)
    if not path.is_file():
        fail(f"missing canonical artifact {path}; do not substitute current/new/publishable result families")

    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("protocol_id") != EXPECTED_PROTOCOL:
        fail(f"unexpected protocol_id {data.get('protocol_id')!r}")
    if data.get("status") != "EXECUTED_CANONICAL_PAPER_EVIDENCE":
        fail("canonical artifact is not marked as executed evidence")

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
    baselines = Counter()

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
        finite_metrics(metrics, f"{variant}/seed={seed}")
        cells[(variant, seed)] += 1
        if bool(run.get("is_downstream_baseline")):
            baselines[(variant, seed)] += 1

    variants = {variant for variant, _ in cells}
    missing_variants = sorted(REQUIRED_VARIANTS - variants)
    if missing_variants:
        fail(f"missing required paper variants: {', '.join(missing_variants)}")

    for variant in sorted(REQUIRED_VARIANTS):
        for seed in seeds:
            count = cells[(variant, seed)]
            if count != 1:
                fail(f"expected exactly one {variant}/seed={seed} record; found {count}")

    for seed in seeds:
        count = baselines[(REQUIRED_BASELINE, seed)]
        if count != 1:
            fail(f"expected exactly one explicit {REQUIRED_BASELINE}/seed={seed} baseline; found {count}")
        if cells[(REQUIRED_BASELINE, seed)] != 1:
            fail(f"duplicate/noncanonical {REQUIRED_BASELINE}/seed={seed} record")

    # Guard the previously discovered no-op ablation bug.
    full_sigs = {json.dumps(run.get("ablation_signature"), sort_keys=True) for run in runs if run.get("variant") == "full"}
    if len(full_sigs) != 1:
        fail("full ablation signature is missing or inconsistent across seeds")
    full_sig = next(iter(full_sigs))
    for variant in sorted(REQUIRED_VARIANTS - {"full"}):
        sigs = {json.dumps(run.get("ablation_signature"), sort_keys=True) for run in runs if run.get("variant") == variant}
        if len(sigs) != 1:
            fail(f"{variant} ablation signature is missing or inconsistent across seeds")
        if next(iter(sigs)) == full_sig:
            fail(f"{variant} is a no-op relative to full")

    paired = data.get("paired_seed_level_statistics")
    if not isinstance(paired, dict) or not paired:
        fail("missing paired seed-level statistics")
    expected_stat = "full_latent_minus_raw_context_ridge_mse"
    if expected_stat not in paired:
        fail(f"missing paired downstream baseline statistic {expected_stat}")

    print(
        "PAPER_EVIDENCE_GATE_PASS: "
        f"protocol={EXPECTED_PROTOCOL}; {len(seeds)} seeds; "
        f"{len(REQUIRED_VARIANTS)} real mechanism variants; matched downstream baseline present; "
        "paired seed-level statistics retained."
    )


if __name__ == "__main__":
    main()
