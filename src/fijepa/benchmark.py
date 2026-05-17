from __future__ import annotations

import copy
from dataclasses import asdict
from typing import Dict, List, Sequence

import numpy as np
import torch
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, roc_auc_score, f1_score

from .data import build_datasets, make_loader
from .metrics import linear_probe_regression, linear_probe_classification, flatten_context_windows
from .model import FIJEPA
from .trainer import Trainer


ABLATON_MAP = {
    "full": {},
    "no_ema": {"model.use_target_ema": False},
    "no_operator_split": {"model.predictor_mode": "monolithic", "model.stage_names": ["monolithic"]},
    "no_financial_regularizers": {"loss.use_financial_regularizers": False},
    "no_uncertainty_heads": {"model.stochastic": False},
    "deterministic": {"model.stochastic": False},
    "no_multi_horizon": {"data.horizons": [1], "loss.multi_horizon_weights": [1.0]},
    "reconstruction": {"model.use_reconstruction_head": True, "loss.lambda_recon": 0.25},
    "mlp_backbone": {"model.backbone": "mlp", "model.num_layers": 2},
    "no_memory": {"model.use_memory": False},
}


def _set_by_path(obj, path: str, value):
    parts = path.split(".")
    cur = obj
    for p in parts[:-1]:
        cur = getattr(cur, p)
    setattr(cur, parts[-1], value)


def apply_ablation(cfg, name: str):
    cfg = copy.deepcopy(cfg)
    if name not in ABLATON_MAP:
        raise ValueError(f"Unknown ablation: {name}")
    for path, value in ABLATON_MAP[name].items():
        _set_by_path(cfg, path, value)
    return cfg


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
    return torch.cat(zs, dim=0), torch.cat(contexts, dim=0), torch.cat(futures, dim=0), (torch.cat(gates, dim=0) if gates else None), (torch.cat(mems, dim=0) if mems else None)


def _metrics_from_scores(y_true, y_pred):
    return {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "directional_acc": float(np.mean((np.asarray(y_true) >= 0) == (np.asarray(y_pred) >= 0))),
    }


def benchmark_fijepa(config, seed: int = 7, max_epochs: int = 3, ablation: str = "full", compute_probes: bool = True) -> Dict:
    cfg = apply_ablation(config, ablation)
    cfg.seed = seed
    torch.manual_seed(seed)
    train_ds, val_ds, test_ds, feature_cols, _ = build_datasets(cfg)
    train_loader = make_loader(train_ds, cfg.train.batch_size, shuffle=True, num_workers=cfg.train.num_workers)
    val_loader = make_loader(val_ds, cfg.train.batch_size, shuffle=False, num_workers=cfg.train.num_workers)
    test_loader = make_loader(test_ds, cfg.train.batch_size, shuffle=False, num_workers=cfg.train.num_workers)

    model = FIJEPA(input_dim=len(feature_cols), config=cfg.model)
    trainer = Trainer(model, train_loader, val_loader, cfg, feature_dim=len(feature_cols), run_dir=cfg.experiment.output_dir)
    history = trainer.fit(max_epochs=max_epochs)

    device = trainer.device
    if compute_probes:
        z_train, ctx_train, fut_train, _, _ = collect_embeddings(model, train_loader, device)
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

    return {
        "ablation": ablation,
        "seed": seed,
        "history": history,
        "probe_regression": probe_reg,
        "baseline_regression": base_reg,
        "probe_classification": probe_cls,
        "baseline_classification": base_cls,
        "latent_stats": latent_stats,
        "config": {
            "seed": cfg.seed,
            "model": asdict(cfg.model),
            "data": asdict(cfg.data),
            "train": asdict(cfg.train),
            "loss": asdict(cfg.loss),
            "experiment": asdict(cfg.experiment),
        },
    }


def benchmark_suite(config, seeds: Sequence[int] = (7, 17, 27), ablations: Sequence[str] = ("full",), max_epochs: int = 1):
    results = []
    for ablation in ablations:
        for seed in seeds:
            results.append(benchmark_fijepa(config, seed=seed, max_epochs=max_epochs, ablation=ablation))
    return results


def summarize_suite(results: List[Dict]):
    summary = {}
    by_ablation = {}
    for r in results:
        by_ablation.setdefault(r["ablation"], []).append(r)

    def agg(vals, key_path):
        def get(d, path):
            cur = d
            for p in path.split('.'):
                cur = cur[p]
            return float(cur)
        xs = np.array([get(v, key_path) for v in vals], dtype=float)
        return {"mean": float(xs.mean()), "std": float(xs.std(ddof=1) if len(xs) > 1 else 0.0), "n": int(len(xs))}

    for ablation, vals in by_ablation.items():
        summary[ablation] = {
            "probe_regression_mse": agg(vals, "probe_regression.mse"),
            "probe_regression_directional_acc": agg(vals, "probe_regression.directional_acc"),
            "baseline_regression_mse": agg(vals, "baseline_regression.mse"),
            "probe_classification_accuracy": agg(vals, "probe_classification.accuracy"),
            "baseline_classification_accuracy": agg(vals, "baseline_classification.accuracy"),
            "latent_train_std_mean": agg(vals, "latent_stats.train_std_mean"),
            "latent_test_std_mean": agg(vals, "latent_stats.test_std_mean"),
            "latent_test_cov_trace": agg(vals, "latent_stats.test_cov_trace"),
            "memory_gate_mean": agg(vals, "latent_stats.memory_gate_mean"),
            "memory_occupancy_mean": agg(vals, "latent_stats.memory_occupancy_mean"),
            "final_val_total": agg(vals, "history.0.val_total"),
        }
    return summary
