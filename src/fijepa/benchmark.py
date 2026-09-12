from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .data import build_datasets, make_loader
from .metrics import linear_probe_classification, linear_probe_regression, flatten_context_windows
from .model import FIJEPA
from .trainer import Trainer


ABLATION_MAP = {
    "full": {},
    "no_ema": {"model.use_target_ema": False},
    "no_operator_split": {"model.predictor_mode": "monolithic", "model.stage_names": ["monolithic"]},
    "no_financial_regularizers": {"loss.use_financial_regularizers": False},
    "no_uncertainty_heads": {"model.stochastic": False},
    # Kept as a legacy alias for older commands. It is intentionally not a
    # distinct paper-facing ablation because it is identical to
    # no_uncertainty_heads in the current model implementation.
    "deterministic": {"model.stochastic": False},
    "no_multi_horizon": {"data.horizons": [1], "loss.multi_horizon_weights": [1.0]},
    "reconstruction": {"model.use_reconstruction_head": True, "loss.lambda_recon": 0.25},
    "mlp_backbone": {"model.backbone": "mlp", "model.num_layers": 2},
    "no_memory": {"model.use_memory": False},
}

# Backward-compatible misspelling used by older imports.
ABLATON_MAP = ABLATION_MAP


def _set_by_path(obj, path: str, value):
    parts = path.split(".")
    cur = obj
    for p in parts[:-1]:
        cur = getattr(cur, p)
    setattr(cur, parts[-1], value)


def apply_ablation(cfg, name: str):
    cfg = copy.deepcopy(cfg)
    if name not in ABLATION_MAP:
        raise ValueError(f"Unknown ablation: {name}")
    for path, value in ABLATION_MAP[name].items():
        _set_by_path(cfg, path, value)
    return cfg


def ablation_signature(cfg, name: str) -> Dict[str, object]:
    """Return the fields that define the intended mechanism intervention."""
    ablated = apply_ablation(cfg, name)
    return {
        "use_target_ema": ablated.model.use_target_ema,
        "predictor_mode": ablated.model.predictor_mode,
        "stage_names": tuple(ablated.model.stage_names),
        "use_financial_regularizers": ablated.loss.use_financial_regularizers,
        "stochastic": ablated.model.stochastic,
        "use_memory": ablated.model.use_memory,
    }


def collect_embeddings(model: FIJEPA, loader, device: torch.device):
    model.eval()
    zs, contexts, futures, gates, mems = [], [], [], [], []
    for batch in loader:
        context = batch["context"].to(device)
        future = batch["futures"].to(device)
        asset_id = batch["asset_id"].to(device)
        if asset_id.ndim == 1:
            asset_id = asset_id.unsqueeze(-1)
        out = model(context=context, future=future, asset_id=asset_id)
        zs.append(out["z_context"].detach().cpu())
        contexts.append(batch["context"].cpu())
        futures.append(batch["futures"].cpu())
        if out.get("memory_gate") is not None:
            gates.append(out["memory_gate"].detach().cpu())
        if out.get("memory_occupancy") is not None:
            mems.append(out["memory_occupancy"].detach().cpu().reshape(1))
    if not zs:
        raise RuntimeError("benchmark loader produced zero examples")
    return (
        torch.cat(zs, dim=0),
        torch.cat(contexts, dim=0),
        torch.cat(futures, dim=0),
        (torch.cat(gates, dim=0) if gates else None),
        (torch.cat(mems, dim=0) if mems else None),
    )


def _metrics_from_scores(y_true, y_pred):
    return {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "directional_acc": float(np.mean((np.asarray(y_true) >= 0) == (np.asarray(y_pred) >= 0))),
    }


def benchmark_fijepa(
    config,
    seed: int = 7,
    max_epochs: int = 3,
    ablation: str = "full",
    compute_probes: bool = True,
) -> Dict:
    cfg = apply_ablation(config, ablation)
    cfg.seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Every variant/seed gets an isolated provenance directory so checkpoints
    # and histories cannot overwrite one another.
    base_run_dir = Path(cfg.experiment.output_dir)
    cfg.experiment.output_dir = str(base_run_dir / f"{ablation}_seed{seed}")

    train_ds, val_ds, test_ds, feature_cols, _ = build_datasets(cfg)
    if min(len(train_ds), len(val_ds), len(test_ds)) <= 0:
        raise RuntimeError(
            f"empty benchmark split: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}"
        )

    train_loader = make_loader(
        train_ds,
        cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
    )
    train_eval_loader = make_loader(
        train_ds,
        cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
    )
    val_loader = make_loader(
        val_ds,
        cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
    )
    test_loader = make_loader(
        test_ds,
        cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
    )

    model = FIJEPA(input_dim=len(feature_cols), config=cfg.model)
    trainable_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    total_params = int(sum(p.numel() for p in model.parameters()))
    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        cfg,
        feature_dim=len(feature_cols),
        run_dir=cfg.experiment.output_dir,
    )
    history = trainer.fit(max_epochs=max_epochs)

    # All downstream paper metrics are computed from the checkpoint selected
    # by validation loss, never the arbitrary last epoch or test performance.
    best_path = Path(cfg.experiment.output_dir) / "best.pt"
    if not best_path.is_file():
        raise RuntimeError(f"missing validation-selected checkpoint: {best_path}")
    checkpoint = torch.load(best_path, map_location=trainer.device)
    model.load_state_dict(checkpoint["model"])

    device = trainer.device
    if compute_probes:
        z_train, ctx_train, fut_train, _, _ = collect_embeddings(model, train_eval_loader, device)
        z_test, ctx_test, fut_test, gates, mems = collect_embeddings(model, test_loader, device)

        target_name = cfg.data.target_cols[0] if cfg.data.target_cols else feature_cols[0]
        target_idx = feature_cols.index(target_name) if target_name in feature_cols else 0
        y_train = fut_train[:, 0, -1, target_idx].numpy()
        y_test = fut_test[:, 0, -1, target_idx].numpy()
        z_train_np = z_train.numpy()
        z_test_np = z_test.numpy()
        context_flat_train = flatten_context_windows(ctx_train)
        context_flat_test = flatten_context_windows(ctx_test)

        probe_reg = linear_probe_regression(z_train_np, y_train, z_test_np, y_test)
        base_reg = linear_probe_regression(context_flat_train, y_train, context_flat_test, y_test)

        median = np.median(y_train)
        y_bin_train = (y_train > median).astype(int)
        y_bin_test = (y_test > median).astype(int)
        probe_cls = linear_probe_classification(z_train_np, y_bin_train, z_test_np, y_bin_test)
        base_cls = linear_probe_classification(context_flat_train, y_bin_train, context_flat_test, y_bin_test)

        latent_stats = {
            "train_mean_norm": float(np.linalg.norm(z_train_np.mean(axis=0))),
            "test_mean_norm": float(np.linalg.norm(z_test_np.mean(axis=0))),
            "train_std_mean": float(z_train_np.std(axis=0).mean()),
            "test_std_mean": float(z_test_np.std(axis=0).mean()),
            "test_cov_trace": float(np.trace(np.cov(z_test_np.T))) if z_test_np.shape[0] > 1 else 0.0,
            "memory_gate_mean": float(gates.mean().item()) if gates is not None else 0.0,
            "memory_occupancy_mean": float(mems.mean().item()) if mems is not None else 0.0,
        }
    else:
        probe_reg = base_reg = probe_cls = base_cls = {}
        latent_stats = {}

    val_totals = [float(row["val_total"]) for row in history if "val_total" in row]
    if not val_totals:
        raise RuntimeError("training history has no validation total metric")

    return {
        "ablation": ablation,
        "seed": seed,
        "history": history,
        "best_val_total": float(min(val_totals)),
        "final_val_total": float(val_totals[-1]),
        "probe_regression": probe_reg,
        "baseline_regression": base_reg,
        "probe_classification": probe_cls,
        "baseline_classification": base_cls,
        "latent_stats": latent_stats,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "checkpoint": str(best_path),
        "config": {
            "seed": cfg.seed,
            "model": asdict(cfg.model),
            "data": asdict(cfg.data),
            "train": asdict(cfg.train),
            "loss": asdict(cfg.loss),
            "experiment": asdict(cfg.experiment),
        },
    }


def benchmark_suite(
    config,
    seeds: Sequence[int] = (7, 17, 27),
    ablations: Sequence[str] = ("full",),
    max_epochs: int = 1,
):
    results = []
    for ablation in ablations:
        for seed in seeds:
            results.append(
                benchmark_fijepa(
                    config,
                    seed=seed,
                    max_epochs=max_epochs,
                    ablation=ablation,
                )
            )
    return results


def summarize_suite(results: List[Dict]):
    summary = {}
    by_ablation = {}
    for r in results:
        by_ablation.setdefault(r["ablation"], []).append(r)

    def agg_values(xs):
        xs = np.asarray(xs, dtype=float)
        return {
            "mean": float(xs.mean()),
            "std": float(xs.std(ddof=1) if len(xs) > 1 else 0.0),
            "n": int(len(xs)),
        }

    for ablation, vals in by_ablation.items():
        summary[ablation] = {
            "probe_regression_mse": agg_values([v["probe_regression"]["mse"] for v in vals]),
            "probe_regression_directional_acc": agg_values(
                [v["probe_regression"]["directional_acc"] for v in vals]
            ),
            "baseline_regression_mse": agg_values([v["baseline_regression"]["mse"] for v in vals]),
            "probe_classification_accuracy": agg_values(
                [v["probe_classification"]["accuracy"] for v in vals]
            ),
            "baseline_classification_accuracy": agg_values(
                [v["baseline_classification"]["accuracy"] for v in vals]
            ),
            "latent_train_std_mean": agg_values([v["latent_stats"]["train_std_mean"] for v in vals]),
            "latent_test_std_mean": agg_values([v["latent_stats"]["test_std_mean"] for v in vals]),
            "latent_test_cov_trace": agg_values([v["latent_stats"]["test_cov_trace"] for v in vals]),
            "memory_gate_mean": agg_values([v["latent_stats"]["memory_gate_mean"] for v in vals]),
            "memory_occupancy_mean": agg_values([v["latent_stats"]["memory_occupancy_mean"] for v in vals]),
            "best_val_total": agg_values([v["best_val_total"] for v in vals]),
            "final_val_total": agg_values([v["final_val_total"] for v in vals]),
            "trainable_params": sorted({int(v["trainable_params"]) for v in vals}),
        }
    return summary
