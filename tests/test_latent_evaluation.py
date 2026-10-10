import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
import yaml

from fijepa import FIJEPA, FIJEPAConfig, build_datasets
from fijepa.data import make_loader
from fijepa.evaluation import METRIC_SCHEMA, evaluate_latents

ROOT = Path(__file__).resolve().parents[1]


class DirectLatents(torch.nn.Module):
    """Analytic oracle: supplied data are the latents, without a learned map."""

    def forward(self, context, future, asset_id):
        assert not torch.is_grad_enabled()
        return {
            "z_context": context[:, 0],
            "z_targets": future[:, :, 0],
            "stages": [{"mu": context[:, 0]}],
        }


def _batches(context, future, sizes):
    assert sum(sizes) == len(context)
    offset = 0
    for size in sizes:
        stop = offset + size
        yield {
            "context": context[offset:stop],
            "futures": future[offset:stop],
            "asset_id": torch.zeros(size, dtype=torch.long),
        }
        offset = stop


@pytest.mark.parametrize("sizes", [[3], [2, 1], [1, 2], [1, 1, 1]])
def test_sample_mean_matches_independent_closed_form_for_every_partition(sizes):
    context = torch.tensor([1.0, 1.0, 4.0]).reshape(3, 1, 1)
    future = torch.ones(3, 1, 1, 1)
    report = evaluate_latents(DirectLatents(), _batches(context, future, sizes))
    # Coordinate errors are [0, 0, 9]; context norms are [1, 1, 4].
    # The historical mean of two batch means for [2, 1] was 4.5 and 2.5.
    assert report["latent_mse"] == 3.0
    assert report["latent_mean_norm"] == 2.0
    assert report["sample_count"] == 3
    assert report["batch_count"] == len(sizes)
    assert report["metric_schema"] == METRIC_SCHEMA


def test_float16_inference_values_are_promoted_before_squaring():
    context = torch.tensor([1000.0, -1000.0], dtype=torch.float16).reshape(2, 1, 1)
    future = torch.zeros(2, 1, 1, 1, dtype=torch.float16)
    report = evaluate_latents(DirectLatents(), _batches(context, future, [2]))
    assert report["latent_mse"] == 1_000_000.0
    assert report["latent_mean_norm"] == 1000.0


def _config():
    cfg = FIJEPAConfig(seed=143, device="cpu")
    cfg.data.synthetic.num_assets = 1
    cfg.data.synthetic.num_steps = 100
    cfg.data.synthetic.feature_dim = 2
    cfg.data.context_length = 3
    cfg.data.horizons = [1, 2]
    cfg.data.target_window = 2
    cfg.model.d_model = 8
    cfg.model.latent_dim = 4
    cfg.model.num_layers = 1
    cfg.model.num_heads = 2
    cfg.model.predictor_hidden = 8
    cfg.model.backbone = "mlp"
    cfg.model.dropout = 0.0
    cfg.model.stochastic = False
    cfg.model.stage_names = ["macro", "residual"]
    cfg.model.memory_size = 4
    cfg.model.memory_topk = 2
    cfg.train.batch_size = 5
    return cfg


def _model(cfg, input_dim):
    with torch.random.fork_rng():
        torch.manual_seed(491)
        return FIJEPA(input_dim=input_dim, config=cfg.model)


def test_actual_model_matches_full_cohort_oracle_and_preserves_weights_memory_and_modes():
    cfg = _config()
    model = _model(cfg, 2)
    context = torch.linspace(-1.0, 1.0, 7 * 3 * 2).reshape(7, 3, 2)
    future = torch.linspace(-0.6, 1.3, 7 * 2 * 2 * 2).reshape(7, 2, 2, 2)
    # Exercise an occupied memory, not just its all-empty branch.
    with torch.no_grad():
        model.update_memory(
            torch.ones(1, cfg.model.latent_dim),
            torch.full((1, cfg.model.latent_dim), 2.0),
            salience=torch.ones(1),
        )
        model.eval()
        expected = model(context, future, torch.zeros(7, 1, dtype=torch.long))
    delta = (
        expected["stages"][-1]["mu"].double() - expected["z_targets"][:, -1].double()
    )
    expected_mse = delta.square().mean().item()
    expected_norm = expected["z_context"].double().norm(dim=-1).mean().item()
    model.train()
    model.target_encoder.eval()
    modes = [module.training for module in model.modules()]
    state = {key: value.clone() for key, value in model.state_dict().items()}

    for partition in ([7], [3, 3, 1], [2, 2, 2, 1]):
        report = evaluate_latents(model, _batches(context, future, partition))
        assert report["latent_mse"] == pytest.approx(expected_mse, abs=2e-7)
        assert report["latent_mean_norm"] == pytest.approx(expected_norm, abs=2e-7)
        assert [module.training for module in model.modules()] == modes
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, state[key], rtol=0, atol=0)
        assert all(parameter.grad is None for parameter in model.parameters())


def test_empty_loader_fails_without_a_success_shaped_empty_report():
    model = DirectLatents().train()
    with pytest.raises(RuntimeError, match="zero examples"):
        evaluate_latents(model, iter(()))
    assert model.training


@pytest.mark.parametrize("invalid", ["empty", "pairing", "nan", "asset_shape"])
def test_invalid_batch_is_rejected_and_training_mode_is_restored(invalid):
    batch = next(_batches(torch.ones(2, 1, 1), torch.ones(2, 1, 1, 1), [2]))
    if invalid == "empty":
        batch = {key: value[:0] for key, value in batch.items()}
    elif invalid == "pairing":
        batch["futures"] = batch["futures"][:1]
    elif invalid == "nan":
        batch["context"][0, 0, 0] = float("nan")
    else:
        batch["asset_id"] = torch.zeros(2, 2, dtype=torch.long)
    model = DirectLatents().train()
    with pytest.raises(ValueError):
        evaluate_latents(model, [batch])
    assert model.training


def test_nonfinite_model_outputs_are_rejected():
    class Nonfinite(DirectLatents):
        def forward(self, *args, **kwargs):
            result = super().forward(*args, **kwargs)
            result["stages"][0]["mu"] = result["z_context"] * float("inf")
            return result

    batch = _batches(torch.ones(2, 1, 1), torch.ones(2, 1, 1, 1), [2])
    with pytest.raises(FloatingPointError, match="non-finite model output"):
        evaluate_latents(Nonfinite(), batch)


def _checkpoint_fixture(tmp_path):
    cfg = _config()
    _, val_ds, test_ds, feature_cols, _ = build_datasets(cfg)
    model = _model(cfg, len(feature_cols))
    checkpoint_path = tmp_path / "generated_untrained_checkpoint.pt"
    torch.save({"model": model.state_dict(), "config": asdict(cfg)}, checkpoint_path)
    config_path = tmp_path / "generated_config.yaml"
    config_path.write_text(yaml.safe_dump(asdict(cfg)), encoding="utf-8")
    return cfg, model, val_ds, test_ds, config_path, checkpoint_path


def _run_cli(config_path, checkpoint_path, *extra):
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/eval_v2.py"),
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint_path),
            *extra,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


@pytest.mark.parametrize("split", ["validation", "test"])
def test_real_checkpoint_cli_matches_full_dataset_oracle_and_records_identity(
    tmp_path, split
):
    cfg, model, val_ds, test_ds, config_path, checkpoint_path = _checkpoint_fixture(
        tmp_path
    )
    dataset = val_ds if split == "validation" else test_ds
    batch = next(iter(make_loader(dataset, len(dataset), shuffle=False)))
    with torch.no_grad():
        model.eval()
        out = model(
            batch["context"], batch["futures"], batch["asset_id"].reshape(-1, 1)
        )
    expected = (
        (out["stages"][-1]["mu"].double() - out["z_targets"][:, -1].double())
        .square()
        .mean()
    )
    result = _run_cli(config_path, checkpoint_path, "--split", split)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["latent_mse"] == pytest.approx(expected.item(), abs=2e-7)
    assert report["sample_count"] == len(dataset)
    assert (
        report["batch_count"]
        == (len(dataset) + cfg.train.batch_size - 1) // cfg.train.batch_size
    )
    assert (
        report["checkpoint_sha256"]
        == hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    )
    assert (
        report["resolved_config_sha256"]
        == hashlib.sha256(
            json.dumps(asdict(cfg), sort_keys=True, allow_nan=False).encode("utf-8")
        ).hexdigest()
    )
    assert report["metric_schema"] == METRIC_SCHEMA
    assert report["device"] == "cpu"
    assert report["split"] == split


@pytest.mark.parametrize("section", ["seed", "model", "data"])
def test_cli_rejects_shape_compatible_but_semantically_different_checkpoint_config(
    tmp_path, section
):
    cfg, _, _, _, config_path, checkpoint_path = _checkpoint_fixture(tmp_path)
    if section == "seed":
        cfg.seed += 1
    elif section == "model":
        cfg.model.dropout = 0.2
    else:
        cfg.data.normalize = False
    config_path.write_text(yaml.safe_dump(asdict(cfg)), encoding="utf-8")
    result = _run_cli(config_path, checkpoint_path)
    assert result.returncode != 0
    assert f"checkpoint {section} configuration does not match" in result.stderr
    assert result.stdout == ""
