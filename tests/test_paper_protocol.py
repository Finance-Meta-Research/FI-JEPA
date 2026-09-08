from scripts.train import load_config
from fijepa.benchmark import ablation_signature


PAPER_VARIANTS = [
    "full",
    "no_ema",
    "no_financial_regularizers",
    "no_operator_split",
    "no_uncertainty_heads",
    "no_memory",
]


def test_macro_full_enables_uncertainty_so_ablation_is_real():
    cfg = load_config("configs/benchmark_macro.yaml")
    assert cfg.model.stochastic is True
    assert ablation_signature(cfg, "no_uncertainty_heads") != ablation_signature(cfg, "full")


def test_all_paper_mechanism_ablations_change_execution_signature():
    cfg = load_config("configs/benchmark_macro.yaml")
    full = ablation_signature(cfg, "full")
    for variant in PAPER_VARIANTS[1:]:
        assert ablation_signature(cfg, variant) != full, variant


def test_legacy_deterministic_alias_is_not_a_distinct_ablation():
    cfg = load_config("configs/benchmark_macro.yaml")
    assert ablation_signature(cfg, "deterministic") == ablation_signature(cfg, "no_uncertainty_heads")
