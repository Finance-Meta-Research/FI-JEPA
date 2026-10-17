"""Constructed checkpoint structure must match the declared FI-JEPA model."""

from __future__ import annotations

import json
from dataclasses import asdict

import pytest
import torch

from fijepa import FIJEPA, FIJEPAConfig
from fijepa.recorded_training import _validate_outputs


def fixture_run(
    root, backbone, *, infer_features=False, last_width=None, input_width=2, num_layers=1,
):
    cfg = FIJEPAConfig(device="cpu")
    cfg.data.feature_cols = None if infer_features else ["feature_a", "feature_b"]
    cfg.model.backbone = backbone
    cfg.model.d_model = 8
    cfg.model.latent_dim = 4
    cfg.model.num_heads = 2
    cfg.model.num_layers = num_layers
    cfg.model.predictor_hidden = 8
    cfg.model.stage_names = ["generated_fixture"]
    cfg.model.memory_size = 3
    cfg.model.memory_topk = 1
    cfg.model.stochastic = backbone == "transformer"
    cfg.model.use_memory = backbone != "conv"
    cfg.model.use_reconstruction_head = backbone != "mlp"
    cfg.train.max_epochs = 1
    config = asdict(cfg)
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "history.jsonl").write_text(
        json.dumps({"epoch": 1, "train_total": 0.5, "val_total": 0.25}) + "\n",
        encoding="utf-8",
    )
    if last_width is None:
        last_width = input_width
    for name, width in (("best.pt", input_width), ("last.pt", last_width)):
        # Untrained, artificial checkpoints. These numbers are no study result.
        state = FIJEPA(input_dim=width, config=cfg.model).state_dict()
        torch.save(
            {"config": config, "epoch": 1, "metric": 0.25, "model": state},
            root / name,
        )
    return config


@pytest.mark.parametrize("backbone", ["mlp", "conv", "transformer"])
@pytest.mark.parametrize("infer_features", [False, True])
def test_genuine_structure_keeps_rng_and_checkpoint_bytes(tmp_path, backbone, infer_features):
    config = fixture_run(tmp_path, backbone, infer_features=infer_features)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    rng = torch.get_rng_state().clone()
    report = _validate_outputs(tmp_path, config)
    assert report["epochs_completed"] == 1
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}


@pytest.mark.parametrize("backbone", ["mlp", "conv", "transformer"])
@pytest.mark.parametrize("tamper", ["unrelated", "missing", "unexpected", "shape", "dtype"])
def test_finite_but_incompatible_checkpoint_cannot_complete(tmp_path, backbone, tamper):
    config = fixture_run(tmp_path, backbone)
    path = tmp_path / "last.pt"
    checkpoint = torch.load(path, weights_only=True)
    state = checkpoint["model"]
    key = next(iter(state))
    if tamper == "unrelated":
        checkpoint["model"] = {"unrelated": torch.ones(1)}
    elif tamper == "missing":
        del state[key]
    elif tamper == "unexpected":
        state["unexpected"] = torch.ones(1)
    elif tamper == "shape":
        state[key] = state[key][:-1]
    else:
        state[key] = state[key].to(torch.float16)
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="model (keys|structure)"):
        _validate_outputs(tmp_path, config)


def test_automatic_width_is_shared_by_both_checkpoints(tmp_path):
    config = fixture_run(tmp_path, "conv", infer_features=True, last_width=3)
    with pytest.raises(ValueError, match="model structure"):
        _validate_outputs(tmp_path, config)


def test_declared_feature_width_is_not_replaced_by_checkpoint_width(tmp_path):
    config = fixture_run(tmp_path, "mlp")
    config["data"]["feature_cols"] = ["one_declared_feature"]
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    for name in ("best.pt", "last.pt"):
        checkpoint = torch.load(tmp_path / name, weights_only=True)
        checkpoint["config"] = config
        torch.save(checkpoint, tmp_path / name)
    with pytest.raises(ValueError, match="model structure"):
        _validate_outputs(tmp_path, config)


def test_zero_layer_inferred_mlp_state_is_admissible(tmp_path):
    config = fixture_run(
        tmp_path, "mlp", infer_features=True, input_width=8, num_layers=0,
    )
    rng = torch.get_rng_state().clone()
    assert _validate_outputs(tmp_path, config)["epochs_completed"] == 1
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
