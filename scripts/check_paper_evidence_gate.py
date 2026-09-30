#!/usr/bin/env python3
"""Validate the frozen v1 report inputs using saved rows only (no model imports)."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

REQUIRED_VARIANTS = (
    "full", "no_ema", "no_financial_regularizers", "no_operator_split",
    "no_uncertainty_heads", "no_memory",
)
REQUIRED_BASELINE = "raw_context_ridge"
EXPECTED_PROTOCOL = "FIJEPA_MACRODATA_PAPER_V1_20260909"
EXPECTED_SEEDS = [7, 17, 27]
BOOTSTRAP_SEED = 20260909
BOOTSTRAP_REPS = 10000
EXPECTED_CLAIM_BOUNDARY = [
    "This bounded macrodata suite is not evidence of trading alpha, market profitability, or state of the art forecasting.",
    "Three paired seeds provide descriptive uncertainty only; confidence intervals are retained but not treated as definitive significance evidence.",
    "Ablation differences are interpreted as component-removal evidence, not universal causal claims.",
    "The raw-context Ridge comparison is a matched downstream-evaluation control, not a parameter-matched neural architecture baseline."
]
PROBE_METRICS = (
    "probe_regression_mse", "probe_regression_mae",
    "probe_directional_accuracy", "probe_classification_accuracy",
)
RAW_METRICS = {
    "probe_regression_mse": ("probe_regression", "mse"),
    "probe_regression_mae": ("probe_regression", "mae"),
    "probe_directional_accuracy": ("probe_regression", "directional_acc"),
    "probe_classification_accuracy": ("probe_classification", "accuracy"),
}


def fail(message: str) -> None:
    raise SystemExit(f"PAPER_EVIDENCE_GATE_FAIL: {message}")


def finite_number(value, label: str) -> float:
    if type(value) not in (int, float):
        fail(f"{label} must be a finite number, not a boolean/string/null")
    try:
        number = float(value)
    except OverflowError:
        fail(f"{label} is outside the finite numeric range")
    if not math.isfinite(number):
        fail(f"{label} is non-finite")
    return number


def same_number(actual, expected, label: str) -> None:
    if not math.isclose(finite_number(actual, label), finite_number(expected, label),
                        rel_tol=1e-12, abs_tol=1e-12):
        fail(f"{label} disagrees with saved-row arithmetic")


def expected_signature(variant: str) -> dict:
    signature = dict(predictor_mode="split", stage_names=["macro"], stochastic=True,
                     use_financial_regularizers=True, use_memory=True, use_target_ema=True)
    switches = {"no_ema": "use_target_ema", "no_financial_regularizers": "use_financial_regularizers",
                "no_uncertainty_heads": "stochastic", "no_memory": "use_memory"}
    if variant in switches:
        signature[switches[variant]] = False
    elif variant == "no_operator_split":
        # This metadata differs, but both frozen arms execute one predictor stage.
        signature.update(predictor_mode="monolithic", stage_names=["monolithic"])
    return signature


def index_rows(rows, variant_key: str, variants: tuple, label: str) -> dict:
    if not isinstance(rows, list):
        fail(f"{label} must be a list")
    indexed = {}
    for row in rows:
        if not isinstance(row, dict):
            fail(f"{label} contains a non-object row")
        seed, variant = row.get("seed"), row.get(variant_key)
        if type(seed) is not int or seed not in EXPECTED_SEEDS:
            fail(f"{label} contains an undeclared/non-integer seed")
        if not isinstance(variant, str) or variant not in variants:
            fail(f"{label} contains an unexpected variant")
        if (variant, seed) in indexed:
            fail(f"{label} duplicates {variant}/seed={seed}")
        indexed[(variant, seed)] = row
    expected = {(variant, seed) for variant in variants for seed in EXPECTED_SEEDS}
    if set(indexed) != expected:
        fail(f"{label} is missing required variant/seed cells")
    return indexed


def paired_from_saved(deltas: list[float]) -> dict:
    """Recompute the retained descriptive bootstrap, never a new model outcome."""
    arr = np.asarray(deltas, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = rng.choice(arr, size=(BOOTSTRAP_REPS, len(arr)), replace=True).mean(axis=1)
    return {
        "mean_delta": float(arr.mean()), "std_delta": float(arr.std(ddof=1)),
        "bootstrap_95ci": np.quantile(means, [0.025, 0.975]).tolist(),
        "n_seed_pairs": len(arr), "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_reps": BOOTSTRAP_REPS,
    }


def validate_artifact(data: dict, min_seeds: int = 3) -> None:
    if not isinstance(data, dict):
        fail("artifact must be an object")
    if data.get("protocol_id") != EXPECTED_PROTOCOL:
        fail(f"unexpected protocol_id {data.get('protocol_id')!r}")
    if data.get("status") != "EXECUTED_CANONICAL_PAPER_EVIDENCE":
        fail("canonical artifact is not marked as executed evidence")
    seeds = data.get("seed_list")
    if not isinstance(seeds, list) or any(type(seed) is not int for seed in seeds) or seeds != EXPECTED_SEEDS:
        fail("frozen v1 requires the ordered predeclared seeds [7, 17, 27]")
    if min_seeds < 3 or len(seeds) < min_seeds:
        fail("--min-seeds cannot relax the frozen three-seed protocol")
    if type(data.get("epochs")) is not int or data["epochs"] != 15:
        fail("frozen v1 requires 15 epochs")
    if data.get("required_variants") != list(REQUIRED_VARIANTS):
        fail("required_variants differs from the frozen six-condition matrix")
    if data.get("downstream_baseline") != REQUIRED_BASELINE:
        fail("unexpected downstream baseline")
    if data.get("claim_boundary") != EXPECTED_CLAIM_BOUNDARY:
        fail("claim_boundary differs from the frozen v1 limitations")

    rows = index_rows(data.get("runs"), "variant", REQUIRED_VARIANTS + (REQUIRED_BASELINE,), "runs")
    raw = index_rows(data.get("raw_runs"), "ablation", REQUIRED_VARIANTS, "raw_runs")
    for (variant, seed), row in rows.items():
        label = f"{variant}/seed={seed}"
        baseline = variant == REQUIRED_BASELINE
        if row.get("is_downstream_baseline") is not baseline:
            fail(f"{label} has an incorrect baseline flag")
        metrics = row.get("metrics")
        required = set(PROBE_METRICS) | (set() if baseline else {"best_validation_total", "final_validation_total"})
        if not isinstance(metrics, dict) or not required.issubset(metrics):
            fail(f"{label} is missing required paper metrics")
        for name, value in metrics.items():
            number = finite_number(value, f"{label}/{name}")
            if name in PROBE_METRICS and number < 0:
                fail(f"{label}/{name} must be nonnegative")
            if name in PROBE_METRICS[2:] and number > 1:
                fail(f"{label}/{name} must lie in [0, 1]")
        raw_row = raw[("full" if baseline else variant, seed)]
        for name, (group, field) in RAW_METRICS.items():
            if baseline:
                group = group.replace("probe_", "baseline_")
            nested = raw_row.get(group)
            if not isinstance(nested, dict) or field not in nested:
                fail(f"raw {label} is missing {group}/{field}")
            same_number(metrics[name], nested[field], f"{label}/{name} raw mapping")
        if not baseline:
            signature = row.get("ablation_signature")
            # JSON comparison distinguishes booleans from 0/1 and requires all fields.
            if json.dumps(signature, sort_keys=True) != json.dumps(expected_signature(variant), sort_keys=True):
                fail(f"{label} signature differs from the frozen v1 signature")
            for report_key, raw_key in [("best_validation_total", "best_val_total"),
                                        ("final_validation_total", "final_val_total")]:
                same_number(metrics[report_key], raw_row.get(raw_key), f"{label}/{report_key} raw mapping")

    comparisons = {f"full_minus_{v}_probe_mse": v for v in REQUIRED_VARIANTS if v != "full"}
    comparisons["full_latent_minus_raw_context_ridge_mse"] = REQUIRED_BASELINE
    paired = data.get("paired_seed_level_statistics")
    if not isinstance(paired, dict) or set(paired) != set(comparisons):
        fail("paired statistics must contain all six frozen comparisons, without extras")
    for name, variant in comparisons.items():
        retained = paired[name]
        if not isinstance(retained, dict):
            fail(f"{name} must be an object")
        expected = paired_from_saved([
            rows[("full", seed)]["metrics"]["probe_regression_mse"]
            - rows[(variant, seed)]["metrics"]["probe_regression_mse"] for seed in seeds
        ])
        for field in ("n_seed_pairs", "bootstrap_seed", "bootstrap_reps"):
            if type(retained.get(field)) is not int or retained[field] != expected[field]:
                fail(f"{name}/{field} differs from the frozen bootstrap specification")
        for field in ("mean_delta", "std_delta"):
            same_number(retained.get(field), expected[field], f"{name}/{field}")
        ci = retained.get("bootstrap_95ci")
        if not isinstance(ci, list) or len(ci) != 2:
            fail(f"{name} requires a two-endpoint bootstrap interval")
        for index in (0, 1):
            same_number(ci[index], expected["bootstrap_95ci"][index], f"{name}/bootstrap_95ci[{index}]")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            fail(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def load_artifact(path: str | Path, min_seeds: int = 3) -> dict:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_object,
                          parse_constant=lambda value: fail(f"non-finite JSON constant {value}"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        fail(f"cannot read canonical artifact {path}: {exc}")
    validate_artifact(data, min_seeds)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate frozen v1 saved evidence without model execution.")
    parser.add_argument("--artifact", default="experiments/paper_results.json")
    parser.add_argument("--min-seeds", type=int, default=3)
    args = parser.parse_args()
    load_artifact(args.artifact, args.min_seeds)
    print("PAPER_EVIDENCE_GATE_PASS: "
          f"protocol={EXPECTED_PROTOCOL}; 3 seeds; 6 retained model conditions; "
          "raw-context Ridge downstream control; saved-row metrics and six paired summaries verified; "
          "no_operator_split remains non-identifying; this is not a publication-readiness gate.")


if __name__ == "__main__":
    main()
