#!/usr/bin/env python3
"""Fail-closed structural check for FI-JEPA paper-facing mechanism ablations.

The v1 macro protocol was already executed before a post-run source audit noticed
that split-vs-monolithic prediction was non-identifying when the full model had
only one predictor stage. This script prevents a configuration-name difference
from being mistaken for a numerical mechanism difference again.
"""
from __future__ import annotations

from dataclasses import asdict

from fijepa.benchmark import apply_ablation
from fijepa.config import load_config

CONFIG = "configs/benchmark_macro.yaml"
IDENTIFYING_V1 = {
    "no_ema",
    "no_financial_regularizers",
    "no_uncertainty_heads",
    "no_memory",
}
NONIDENTIFYING_V1 = {"no_operator_split"}


def effective_signature(cfg) -> dict:
    """Describe executed mechanisms while excluding labels that do no math."""
    stages = len(cfg.model.stage_names) if cfg.model.predictor_mode == "split" else 1
    return {
        "target_ema": bool(cfg.model.use_target_ema),
        "financial_regularizers": bool(cfg.loss.use_financial_regularizers),
        "stochastic_predictor": bool(cfg.model.stochastic),
        "prototype_memory": bool(cfg.model.use_memory),
        "effective_predictor_stage_count": int(stages),
        # Numerical stage architecture shared by split and monolithic paths.
        "predictor_hidden": int(cfg.model.predictor_hidden),
        "latent_dim": int(cfg.model.latent_dim),
    }


def main() -> None:
    cfg = load_config(CONFIG)
    full = effective_signature(cfg)
    observed_identifying = set()
    observed_nonidentifying = set()

    for variant in sorted(IDENTIFYING_V1 | NONIDENTIFYING_V1):
        ablated = apply_ablation(cfg, variant)
        signature = effective_signature(ablated)
        if signature == full:
            observed_nonidentifying.add(variant)
        else:
            observed_identifying.add(variant)
        print(f"{variant}: {'IDENTIFYING' if signature != full else 'NONIDENTIFYING'} {signature}")

    if observed_identifying != IDENTIFYING_V1:
        raise SystemExit(
            "ABLATION_IDENTIFIABILITY_FAIL: identifying set drifted; "
            f"expected={sorted(IDENTIFYING_V1)} observed={sorted(observed_identifying)}"
        )
    if observed_nonidentifying != NONIDENTIFYING_V1:
        raise SystemExit(
            "ABLATION_IDENTIFIABILITY_FAIL: non-identifying set drifted; "
            f"expected={sorted(NONIDENTIFYING_V1)} observed={sorted(observed_nonidentifying)}"
        )

    # The historical deterministic alias must remain equivalent to the real
    # no-uncertainty intervention rather than being counted as another result.
    det = effective_signature(apply_ablation(cfg, "deterministic"))
    nounc = effective_signature(apply_ablation(cfg, "no_uncertainty_heads"))
    if det != nounc:
        raise SystemExit("ABLATION_IDENTIFIABILITY_FAIL: deterministic alias semantics drifted")

    print(
        "ABLATION_IDENTIFIABILITY_PASS: four v1 mechanism removals alter executed "
        "structure; no_operator_split is correctly classified non-identifying; "
        "deterministic remains a duplicate alias."
    )


if __name__ == "__main__":
    main()
