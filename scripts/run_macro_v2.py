#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import statsmodels.api as sm
import torch

from fijepa.benchmark import benchmark_suite
from fijepa.config import load_config
from fijepa.controls import DirectMLPTrainingSpec, evaluate_raw_context_controls
from fijepa.data import build_datasets


PROTOCOL_ID = "FIJEPA_MACRODATA_V2_CANDIDATE_20260926"
AUTHORIZED_STATUS = "FROZEN_PRE_OUTCOME_AUTHORIZED"
VARIANTS = (
    "full",
    "no_ema",
    "no_financial_regularizers",
    "no_operator_split",
    "no_uncertainty_heads",
    "no_memory",
)


class MacroV2ProtocolError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_macrodata_sha256() -> str:
    raw = sm.datasets.macrodata.load_pandas().data.copy()
    payload = raw.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.17g",
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise MacroV2ProtocolError("protocol must be a JSON object")
    return obj


def validate_protocol(protocol: Mapping[str, Any], cfg) -> None:
    if protocol.get("schema_version") != 1:
        raise MacroV2ProtocolError("schema_version must equal 1")
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise MacroV2ProtocolError("unexpected protocol_id")
    if tuple(protocol.get("seeds_exact", ())) != (101, 211, 307, 401, 503):
        raise MacroV2ProtocolError("seed list drift")
    if cfg.data.source != "macrodata":
        raise MacroV2ProtocolError("v2 source must be macrodata")
    if cfg.data.causal_preprocessing is not True:
        raise MacroV2ProtocolError("v2 requires causal_preprocessing=true")
    if tuple(cfg.data.horizons) != (1, 4):
        raise MacroV2ProtocolError("v2 horizon drift")
    if tuple(cfg.model.stage_names) != ("transition_1", "transition_2"):
        raise MacroV2ProtocolError("v2 stage-name drift")
    if cfg.model.predictor_mode != "split":
        raise MacroV2ProtocolError("v2 full model must use split predictor mode")
    if len(cfg.model.stage_names) < 2:
        raise MacroV2ProtocolError("v2 full model must execute at least two stages")
    metric = protocol.get("primary_metric", {})
    if float(metric.get("minimum_relative_mse_improvement", -1.0)) != 0.05:
        raise MacroV2ProtocolError("primary practical-effect threshold drift")
    if int(metric.get("bootstrap_reps", 0)) != 20000:
        raise MacroV2ProtocolError("bootstrap-rep drift")


def assert_execution_authorized(
    protocol: Mapping[str, Any],
    cfg,
    *,
    config_path: Path,
) -> None:
    validate_protocol(protocol, cfg)
    if protocol.get("status") != AUTHORIZED_STATUS:
        raise MacroV2ProtocolError(
            f"protocol status must be {AUTHORIZED_STATUS!r} before any v2 outcome access"
        )
    if protocol.get("execution_authorized") is not True:
        raise MacroV2ProtocolError("execution_authorized must be true")

    freeze = protocol.get("freeze")
    if not isinstance(freeze, Mapping):
        raise MacroV2ProtocolError("missing freeze block")
    expected_version = freeze.get("statsmodels_version")
    actual_version = importlib.metadata.version("statsmodels")
    if expected_version != actual_version:
        raise MacroV2ProtocolError(
            f"statsmodels version drift: expected {expected_version}, got {actual_version}"
        )
    if freeze.get("raw_macrodata_sha256") != canonical_macrodata_sha256():
        raise MacroV2ProtocolError("statsmodels macrodata hash drift")
    if freeze.get("config_sha256") != sha256_file(config_path):
        raise MacroV2ProtocolError("v2 config hash drift")

    source_hashes = freeze.get("source_sha256")
    if not isinstance(source_hashes, Mapping) or not source_hashes:
        raise MacroV2ProtocolError("source hashes are not frozen")
    for raw_path, expected in source_hashes.items():
        path = Path(raw_path)
        if not path.is_file() or sha256_file(path) != expected:
            raise MacroV2ProtocolError(f"source hash drift: {raw_path}")


def _dataset_arrays(dataset, target_idx: int):
    xs = []
    ys = []
    for i in range(len(dataset)):
        row = dataset[i]
        xs.append(row["context"].detach().cpu().numpy().reshape(-1))
        ys.append(
            float(
                row["futures"][0, -1, target_idx]
                .detach()
                .cpu()
                .item()
            )
        )
    if not xs:
        raise RuntimeError("control dataset has zero windows")
    x = np.asarray(xs, dtype=np.float32)
    y = np.asarray(ys, dtype=np.float32)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise RuntimeError("control arrays contain non-finite values")
    return x, y


def _controls_for_seed(cfg, *, seed: int, target_params: int):
    local = copy.deepcopy(cfg)
    local.seed = int(seed)
    train_ds, val_ds, test_ds, feature_cols, _ = build_datasets(local)
    target_name = local.data.target_cols[0] if local.data.target_cols else feature_cols[0]
    target_idx = feature_cols.index(target_name) if target_name in feature_cols else 0
    x_train, y_train = _dataset_arrays(train_ds, target_idx)
    x_val, y_val = _dataset_arrays(val_ds, target_idx)
    x_test, y_test = _dataset_arrays(test_ds, target_idx)
    return evaluate_raw_context_controls(
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        seed=seed,
        target_params=target_params,
        training=DirectMLPTrainingSpec(
            epochs=int(local.train.max_epochs),
            batch_size=int(local.train.batch_size),
            lr=float(local.train.lr),
            weight_decay=float(local.train.weight_decay),
        ),
    )


def _select_strongest_control(controls: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    validation_mse = {
        "raw_context_ridge": float(controls["raw_context_ridge"]["validation_mse"]),
        "raw_context_hist_gradient_boosting": float(
            controls["raw_context_hist_gradient_boosting"]["validation_mse"]
        ),
        "capacity_matched_direct_mlp": float(
            controls["capacity_matched_direct_mlp"]["best_validation_mse"]
        ),
    }
    name = min(validation_mse, key=validation_mse.get)
    return name, {
        "selected_by": "lowest_validation_mse_only",
        "validation_mse_by_control": validation_mse,
        "test": controls[name]["test"],
    }


def _paired_bootstrap(deltas: list[float], *, reps: int, seed: int) -> dict[str, Any]:
    arr = np.asarray(deltas, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(reps, dtype=np.float64)
    for i in range(reps):
        means[i] = rng.choice(arr, size=len(arr), replace=True).mean()
    return {
        "mean_delta": float(arr.mean()),
        "bootstrap_95ci": [
            float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)),
        ],
        "n_seed_pairs": int(len(arr)),
        "bootstrap_seed": int(seed),
        "bootstrap_reps": int(reps),
    }


def run_v2(protocol: Mapping[str, Any], cfg) -> dict[str, Any]:
    seeds = [int(x) for x in protocol["seeds_exact"]]
    raw = benchmark_suite(
        cfg,
        seeds=seeds,
        ablations=VARIANTS,
        max_epochs=int(cfg.train.max_epochs),
        probe_validation_selection=True,
    )
    by_cell = {(row["ablation"], int(row["seed"])): row for row in raw}
    expected = {(variant, seed) for variant in VARIANTS for seed in seeds}
    missing = expected - set(by_cell)
    if missing:
        raise RuntimeError(f"missing v2 cells: {sorted(missing)}")

    controls_by_seed: dict[str, Any] = {}
    selected_controls: dict[str, Any] = {}
    full_mse = {}
    selected_control_mse = {}
    relative_improvements = []

    for seed in seeds:
        full = by_cell[("full", seed)]
        full_mse[seed] = float(full["probe_regression"]["mse"])
        controls = _controls_for_seed(
            cfg,
            seed=seed,
            target_params=int(full["trainable_params"]),
        )
        gap = float(
            controls["capacity_matched_direct_mlp"]["relative_parameter_gap"]
        )
        if gap > 0.10:
            raise RuntimeError(
                f"seed {seed}: capacity-matched MLP parameter gap {gap:.4f} exceeds 10%"
            )
        name, selected = _select_strongest_control(controls)
        controls_by_seed[str(seed)] = controls
        selected_controls[str(seed)] = {"name": name, **selected}
        control_mse = float(selected["test"]["mse"])
        selected_control_mse[seed] = control_mse
        relative_improvements.append(
            (control_mse - full_mse[seed]) / max(control_mse, 1e-12)
        )

    deltas = [
        full_mse[seed] - selected_control_mse[seed]
        for seed in seeds
    ]
    metric = protocol["primary_metric"]
    paired = _paired_bootstrap(
        deltas,
        reps=int(metric["bootstrap_reps"]),
        seed=int(metric["bootstrap_seed"]),
    )
    mean_rel = float(np.mean(relative_improvements))
    passes_effect = mean_rel >= float(metric["minimum_relative_mse_improvement"])
    passes_interval = paired["bootstrap_95ci"][1] < 0.0

    operator_capacity = {}
    for seed in seeds:
        full_p = int(by_cell[("full", seed)]["trainable_params"])
        mono_p = int(by_cell[("no_operator_split", seed)]["trainable_params"])
        gap = abs(full_p - mono_p) / max(full_p, 1)
        operator_capacity[str(seed)] = {
            "full_trainable_params": full_p,
            "no_operator_split_trainable_params": mono_p,
            "relative_gap": float(gap),
            "inferentially_capacity_matched": bool(gap <= 0.10),
        }

    return {
        "protocol_id": PROTOCOL_ID,
        "status": "EXECUTED_V2_EVIDENCE",
        "seed_list": seeds,
        "variants": list(VARIANTS),
        "raw_runs": raw,
        "raw_context_controls_by_seed": controls_by_seed,
        "validation_selected_strongest_control_by_seed": selected_controls,
        "primary_comparison": {
            "delta_definition": "full_latent_probe_test_mse - validation_selected_strongest_raw_control_test_mse",
            "paired_seed_statistics": paired,
            "mean_relative_mse_improvement": mean_rel,
            "minimum_relative_mse_improvement": float(
                metric["minimum_relative_mse_improvement"]
            ),
            "effect_threshold_pass": passes_effect,
            "bootstrap_upper_below_zero": passes_interval,
            "confirmatory_success": bool(passes_effect and passes_interval),
        },
        "operator_ablation_capacity_audit": operator_capacity,
        "claim_boundary": protocol["claim_boundary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail-closed prospective FI-JEPA macrodata v2 runner."
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("protocols/macrodata_v2_candidate_20260926.json"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/benchmark_macro_v2_candidate.yaml"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/macrodata_v2_results.json"),
    )
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()

    protocol = _load_json(args.protocol)
    cfg = load_config(str(args.config))
    validate_protocol(protocol, cfg)

    plan = {
        "protocol_id": PROTOCOL_ID,
        "status": "PREOUTCOME_PLAN_ONLY",
        "seeds": list(protocol["seeds_exact"]),
        "variants": list(VARIANTS),
        "neural_cells": len(protocol["seeds_exact"]) * len(VARIANTS),
        "controls_per_seed": [
            "raw_context_ridge",
            "raw_context_hist_gradient_boosting",
            "capacity_matched_direct_mlp",
        ],
        "primary_selection": "strongest raw-context control selected by validation MSE only",
        "test_policy": "single access only after explicit freeze and authorization",
    }
    if args.plan_only:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return

    assert_execution_authorized(protocol, cfg, config_path=args.config)
    payload = run_v2(protocol, cfg)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite retained v2 evidence: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.out} sha256={sha256_file(args.out)}")


if __name__ == "__main__":
    main()
