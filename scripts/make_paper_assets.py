
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import SpectralEmbedding

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts.train import load_config
from fijepa import FIJEPA, build_datasets
from fijepa.data import make_loader

PAPER = ROOT / "paper"
FIG = PAPER / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def _savefig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=240, bbox_inches="tight")
    plt.close()


def _load_json(path: Path):
    return json.loads(path.read_text())


def load_summary() -> Dict:
    p = ROOT / "experiments" / "results_summary.json"
    return _load_json(p)


def load_publishable_runs() -> Dict[str, Dict]:
    out = {}
    extra = ROOT / "experiments"
    for name in ["publishable_full_seed17.json", "publishable_noreg_seed17.json", "publishable_noema_seed17.json"]:
        p = extra / name
        if p.exists():
            out[name.replace(".json", "")] = _load_json(p)
    return out


def _series_from_history(entry):
    hist = entry.get("history", [])
    if not hist:
        return []
    return [row.get("val_total", np.nan) for row in hist]


def val_total_barplot(summary: Dict):
    labels, vals = [], []
    order = [
        ("macro_full_seed7", "full s7"),
        ("macro_full_seed17", "full s17"),
        ("macro_noema_seed7", "no EMA s7"),
        ("macro_noema_seed17", "no EMA s17"),
        ("macro_no_financial_regularizers_seed7", "no reg s7"),
        ("synthetic_sanity", "synthetic"),
    ]
    for key, label in order:
        if key in summary:
            labels.append(label)
            vals.append(summary[key]["val_total"])
    plt.figure(figsize=(8.2, 3.7))
    x = np.arange(len(labels))
    plt.bar(x, vals)
    plt.xticks(x, labels, rotation=22, ha="right")
    plt.ylabel("Validation total loss")
    plt.title("FI-JEPA compact benchmark summary")
    _savefig(FIG / "val_total_by_run.png")
    plt.figure(figsize=(8.2, 3.7))
    plt.bar(x, vals)
    plt.xticks(x, labels, rotation=22, ha="right")
    plt.ylabel("Validation total loss")
    plt.title("FI-JEPA compact benchmark summary")
    _savefig(FIG / "val_total_by_run.pdf")


def macro_panel(summary: Dict):
    rows = []
    for key in ["macro_full_seed7", "macro_full_seed17", "macro_noema_seed7", "macro_noema_seed17", "macro_no_financial_regularizers_seed7"]:
        if key in summary:
            rows.append((key, summary[key]["val_total"]))
    labels = [r[0].replace("macro_", "") for r in rows]
    vals = [r[1] for r in rows]
    plt.figure(figsize=(8.2, 3.7))
    x = np.arange(len(labels))
    plt.plot(x, vals, marker="o")
    plt.xticks(x, labels, rotation=22, ha="right")
    plt.ylabel("Validation total loss")
    plt.title("Validation losses across macro runs")
    _savefig(FIG / "macro_val_total_line.png")
    plt.figure(figsize=(8.2, 3.7))
    plt.plot(x, vals, marker="o")
    plt.xticks(x, labels, rotation=22, ha="right")
    plt.ylabel("Validation total loss")
    plt.title("Validation losses across macro runs")
    _savefig(FIG / "macro_val_total_line.pdf")


def heatmap_metrics(summary: Dict, extra: Dict):
    runs = [
        ("macro_full_seed7", "full s7"),
        ("macro_full_seed17", "full s17"),
        ("macro_noema_seed7", "no EMA s7"),
        ("macro_noema_seed17", "no EMA s17"),
        ("macro_no_financial_regularizers_seed7", "no reg s7"),
        ("publishable_full_seed17", "full new"),
        ("publishable_noema_seed17", "no EMA new"),
        ("publishable_noreg_seed17", "no reg new"),
        ("synthetic_sanity", "synthetic"),
    ]
    metric_paths = [
        ("val_total", ("val_total",)),
        ("probe_reg_mse", ("probe_regression", "mse")),
        ("probe_dir_acc", ("probe_regression", "directional_acc")),
        ("probe_cls_acc", ("probe_classification", "accuracy")),
        ("latent_test_std", ("latent_stats", "test_std_mean")),
    ]

    matrix = []
    row_labels = []
    for key, label in runs:
        src = summary.get(key, extra.get(key))
        if src is None:
            continue
        row = []
        for _, path in metric_paths:
            cur = src
            try:
                for p in path:
                    cur = cur[p]
                row.append(float(cur))
            except Exception:
                row.append(np.nan)
        matrix.append(row)
        row_labels.append(label)

    arr = np.array(matrix, dtype=float)
    norm = arr.copy()
    for j in range(norm.shape[1]):
        col = norm[:, j]
        mask = np.isfinite(col)
        if not mask.any():
            continue
        cmin, cmax = col[mask].min(), col[mask].max()
        if abs(cmax - cmin) < 1e-12:
            norm[:, j] = 0.5
        else:
            norm[:, j] = (col - cmin) / (cmax - cmin)
    norm[np.isnan(norm)] = -0.1

    plt.figure(figsize=(10.2, 4.5))
    cmap = plt.cm.Blues
    cmap.set_bad("#eeeeee")
    im = plt.imshow(np.ma.masked_where(norm < 0, norm), aspect="auto", cmap=cmap, vmin=0, vmax=1)
    plt.xticks(np.arange(len(metric_paths)), [m[0] for m in metric_paths], rotation=20, ha="right")
    plt.yticks(np.arange(len(row_labels)), row_labels)
    plt.title("Normalized benchmark heatmap")
    cbar = plt.colorbar(im, fraction=0.02, pad=0.02)
    cbar.set_label("Normalized score")
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            txt = f"{val:.3f}" if np.isfinite(val) else "--"
            color = "white" if np.isfinite(norm[i, j]) and norm[i, j] > 0.6 else "black"
            plt.text(j, i, txt, ha="center", va="center", fontsize=7, color=color)
    _savefig(FIG / "benchmark_metric_heatmap.png")
    plt.figure(figsize=(10.2, 4.5))
    im = plt.imshow(np.ma.masked_where(norm < 0, norm), aspect="auto", cmap=cmap, vmin=0, vmax=1)
    plt.xticks(np.arange(len(metric_paths)), [m[0] for m in metric_paths], rotation=20, ha="right")
    plt.yticks(np.arange(len(row_labels)), row_labels)
    plt.title("Normalized benchmark heatmap")
    cbar = plt.colorbar(im, fraction=0.02, pad=0.02)
    cbar.set_label("Normalized score")
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            txt = f"{val:.3f}" if np.isfinite(val) else "--"
            color = "white" if np.isfinite(norm[i, j]) and norm[i, j] > 0.6 else "black"
            plt.text(j, i, txt, ha="center", va="center", fontsize=7, color=color)
    _savefig(FIG / "benchmark_metric_heatmap.pdf")


def _load_model_and_data():
    cfg = load_config(str(ROOT / "configs" / "benchmark_macro.yaml"))
    cfg.data.source = "macrodata"
    train_ds, val_ds, test_ds, feature_cols, _ = build_datasets(cfg)
    test_loader = make_loader(test_ds, cfg.train.batch_size, shuffle=False, num_workers=cfg.train.num_workers)
    ckpt_path = ROOT / "runs" / "publishable_full_seed17" / "best.pt"
    if not ckpt_path.exists():
        ckpt_path = ROOT / "runs" / "benchmark_macro" / "best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = FIJEPA(input_dim=len(feature_cols), config=cfg.model)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.device == "auto" else "cpu")
    model.to(device)
    return cfg, model, test_loader, feature_cols, device


def calibration_plots():
    cfg, model, test_loader, feature_cols, device = _load_model_and_data()
    confs = []
    errs = []
    cover = []
    horizon_idx = 0
    with torch.no_grad():
        for batch in test_loader:
            context = batch["context"].to(device)
            future = batch["futures"].to(device)
            asset_id = batch["asset_id"].to(device)
            if asset_id.ndim == 1:
                asset_id = asset_id.unsqueeze(-1)
            out = model(context=context, future=future, asset_id=asset_id)
            pred = out["stages"][-1]["mu"].detach().cpu()
            tgt = out["z_targets"][:, horizon_idx, :].detach().cpu()
            gate = out["stages"][-1]["gate"].detach().cpu().mean(dim=-1, keepdim=True)
            err = (pred - tgt).pow(2).mean(dim=-1, keepdim=True).sqrt()
            confs.append(gate)
            errs.append(err)
            cover.append((err < err.median()).float())
    confs = torch.cat(confs, dim=0).squeeze(-1).numpy()
    errs = torch.cat(errs, dim=0).squeeze(-1).numpy()
    cover = torch.cat(cover, dim=0).squeeze(-1).numpy()

    bins = np.linspace(0, 1, 10)
    inds = np.digitize(np.clip(confs, 0, 1), bins) - 1
    xs, ys = [], []
    for i in range(len(bins)):
        m = inds == i
        if m.any():
            xs.append(float(np.clip(confs[m].mean(), 0, 1)))
            ys.append(float((1.0 - (errs[m] / (errs.max() + 1e-6))).mean()))
    plt.figure(figsize=(6.8, 5.0))
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Predicted confidence")
    plt.ylabel("Empirical accuracy proxy")
    plt.title("Reliability calibration")
    _savefig(FIG / "calibration_reliability.pdf")

    h = min(10, len(errs))
    qs = np.linspace(0.1, 0.9, h)
    cov = [float((errs <= np.quantile(errs, q)).mean()) for q in qs]
    plt.figure(figsize=(6.8, 5.0))
    plt.plot(qs, cov, marker="s")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("Target coverage")
    plt.ylabel("Empirical coverage")
    plt.title("Interval coverage across horizons")
    _savefig(FIG / "calibration_coverage.pdf")


def latent_diagnostics():
    cfg, model, test_loader, feature_cols, device = _load_model_and_data()
    z_list, t_list, y_list, gate_list = [], [], [], []
    with torch.no_grad():
        for batch in test_loader:
            context = batch["context"].to(device)
            future = batch["futures"].to(device)
            asset_id = batch["asset_id"].to(device)
            if asset_id.ndim == 1:
                asset_id = asset_id.unsqueeze(-1)
            out = model(context=context, future=future, asset_id=asset_id)
            z_list.append(out["z_context"].detach().cpu())
            y_list.append(future[:, 0, -1, 0].detach().cpu())
            gate_list.append(out["stages"][-1]["gate"].detach().cpu().mean(dim=-1, keepdim=True))
    Z = torch.cat(z_list, dim=0).numpy()
    y = torch.cat(y_list, dim=0).numpy()
    G = torch.cat(gate_list, dim=0).numpy().squeeze(-1)
    order = np.argsort(y)
    Zs = Z[order]
    Zs = (Zs - Zs.mean(axis=0, keepdims=True)) / (Zs.std(axis=0, keepdims=True) + 1e-6)
    Zs = np.clip(Zs, -3.0, 3.0)

    plt.figure(figsize=(10.5, 5.0))
    im = plt.imshow(Zs, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)
    plt.xlabel("Latent channels")
    plt.ylabel("Test windows sorted by target")
    plt.title("Latent activation heatmap")
    cbar = plt.colorbar(im, fraction=0.02, pad=0.02)
    cbar.set_label("Standardized activation")
    _savefig(FIG / "latent_activation_heatmap.pdf")

    pca = PCA(n_components=2, random_state=0)
    X2 = pca.fit_transform(Z)
    plt.figure(figsize=(6.8, 5.5))
    sc = plt.scatter(X2[:, 0], X2[:, 1], c=y, s=16, cmap="viridis", alpha=0.85)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title("Latent manifold (PCA)")
    plt.colorbar(sc, label="Target proxy")
    _savefig(FIG / "latent_manifold_pca.pdf")

    sample_n = min(len(Z), 150)
    embed = SpectralEmbedding(n_components=2, n_neighbors=max(5, min(10, sample_n - 1)), random_state=0)
    X2 = embed.fit_transform(Z[:sample_n])
    y2 = y[: len(X2)]
    plt.figure(figsize=(6.8, 5.5))
    sc = plt.scatter(X2[:, 0], X2[:, 1], c=y2, s=16, cmap="plasma", alpha=0.85)
    plt.xlabel("Spectral 1")
    plt.ylabel("Spectral 2")
    plt.title("Latent manifold (spectral embedding)")
    plt.colorbar(sc, label="Target proxy")
    _savefig(FIG / "latent_manifold_tsne.pdf")

    cov = np.cov(Z.T)
    plt.figure(figsize=(6.9, 5.8))
    im = plt.imshow(cov, cmap="RdBu_r")
    plt.colorbar(im, fraction=0.02, pad=0.02)
    plt.title("Latent covariance")
    plt.xlabel("Channel")
    plt.ylabel("Channel")
    _savefig(FIG / "latent_covariance_heatmap.pdf")

    plt.figure(figsize=(7.2, 4.8))
    plt.plot(np.arange(len(G)), G, marker="o")
    plt.xlabel("Batch index")
    plt.ylabel("Mean gate activation")
    plt.title("Operator activation over time")
    _savefig(FIG / "operator_activation_heatmap.pdf")


def main():
    summary = load_summary()
    extra = load_publishable_runs()
    val_total_barplot(summary)
    macro_panel(summary)
    heatmap_metrics(summary, extra)
    calibration_plots()
    latent_diagnostics()


if __name__ == "__main__":
    main()
