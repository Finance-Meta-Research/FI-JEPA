import copy
import json
from pathlib import Path

import pytest

from fijepa.config import load_config
from scripts.run_macro_v2 import (
    AUTHORIZED_STATUS,
    MacroV2ProtocolError,
    _select_strongest_control,
    assert_execution_authorized,
    canonical_macrodata_sha256,
    validate_protocol,
)


def _protocol():
    return json.loads(
        Path("protocols/macrodata_v2_candidate_20260926.json").read_text(
            encoding="utf-8"
        )
    )


def test_candidate_protocol_and_config_are_shape_valid():
    protocol = _protocol()
    cfg = load_config("configs/benchmark_macro_v2_candidate.yaml")
    validate_protocol(protocol, cfg)
    assert len(protocol["seeds_exact"]) == 5
    assert cfg.data.causal_preprocessing is True
    assert len(cfg.model.stage_names) == 2


def test_candidate_protocol_cannot_execute_outcomes():
    protocol = _protocol()
    cfg = load_config("configs/benchmark_macro_v2_candidate.yaml")
    assert protocol["status"] != AUTHORIZED_STATUS
    with pytest.raises(MacroV2ProtocolError, match="protocol status must be"):
        assert_execution_authorized(
            protocol,
            cfg,
            config_path=Path("configs/benchmark_macro_v2_candidate.yaml"),
        )


def test_validation_only_control_selector():
    controls = {
        "raw_context_ridge": {
            "validation_mse": 0.5,
            "test": {"mse": 0.1},
        },
        "raw_context_hist_gradient_boosting": {
            "validation_mse": 0.3,
            "test": {"mse": 99.0},
        },
        "capacity_matched_direct_mlp": {
            "best_validation_mse": 0.4,
            "test": {"mse": 0.01},
        },
    }
    name, row = _select_strongest_control(controls)
    assert name == "raw_context_hist_gradient_boosting"
    assert row["test"]["mse"] == 99.0


def test_macrodata_hash_is_stable_shape():
    digest = canonical_macrodata_sha256()
    assert len(digest) == 64
    int(digest, 16)
