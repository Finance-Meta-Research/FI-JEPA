"""Paper-protocol structural tests; no historical results are regenerated.

The frozen macrodata v1 config has *one* split predictor stage. Renaming the
stage to monolithic does not change executed mathematics and cannot be counted
as evidence for operator factorization. Any v2 multi-stage study needs its own
pre-outcome protocol and must not retrofit the closed v1 results.
"""
import copy

from fijepa.benchmark import ablation_signature, apply_ablation
from fijepa.config import load_config
from scripts.check_ablation_identifiability import effective_signature


IDENTIFYING_V1 = (
    "no_ema",
    "no_financial_regularizers",
    "no_uncertainty_heads",
    "no_memory",
)


def test_macro_full_enables_uncertainty_so_ablation_is_real():
    cfg = load_config("configs/benchmark_macro.yaml")
    assert cfg.model.stochastic is True
    assert effective_signature(apply_ablation(cfg, "no_uncertainty_heads")) != effective_signature(cfg)


def test_only_identifying_v1_conditions_change_executed_mechanisms():
    cfg = load_config("configs/benchmark_macro.yaml")
    full = effective_signature(cfg)
    for variant in IDENTIFYING_V1:
        assert effective_signature(apply_ablation(cfg, variant)) != full, variant


def test_operator_split_v1_changes_only_label_not_execution():
    cfg = load_config("configs/benchmark_macro.yaml")
    assert cfg.model.predictor_mode == "split"
    assert len(cfg.model.stage_names) == 1
    other = apply_ablation(cfg, "no_operator_split")
    # The old *configuration* signature correctly records different labels;
    # it is not a test that the numerical mechanism differs.
    assert ablation_signature(cfg, "no_operator_split") != ablation_signature(cfg, "full")
    assert effective_signature(other) == effective_signature(cfg)


def test_multi_stage_prospective_configuration_would_be_distinct():
    # Structural test only; does not authorize executing a new study.
    cfg = copy.deepcopy(load_config("configs/benchmark_macro.yaml"))
    cfg.model.stage_names = ["macro", "liquidity"]
    assert effective_signature(apply_ablation(cfg, "no_operator_split")) != effective_signature(cfg)


def test_renaming_single_stage_cannot_create_a_mechanism_ablation():
    cfg = load_config("configs/benchmark_macro.yaml")
    renamed = copy.deepcopy(cfg)
    renamed.model.stage_names = ["another_label"]
    assert effective_signature(renamed) == effective_signature(cfg)


def test_legacy_deterministic_alias_is_not_a_distinct_ablation():
    cfg = load_config("configs/benchmark_macro.yaml")
    assert ablation_signature(cfg, "deterministic") == ablation_signature(cfg, "no_uncertainty_heads")
    assert effective_signature(apply_ablation(cfg, "deterministic")) == effective_signature(
        apply_ablation(cfg, "no_uncertainty_heads")
    )
