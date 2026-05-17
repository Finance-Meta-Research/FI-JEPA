from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
import yaml

from fijepa import FIJEPAConfig, FIJEPA, build_datasets
from fijepa.data import make_loader


def load_config(path: str) -> FIJEPAConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}
    cfg = FIJEPAConfig()
    for section in ["seed", "device"]:
        if section in raw:
            setattr(cfg, section, raw[section])
    for key in ["data", "model", "train", "loss", "experiment"]:
        if key in raw:
            obj = getattr(cfg, key)
            for k, v in raw[key].items():
                if hasattr(obj, k):
                    setattr(obj, k, v)
    return cfg


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    args = p.parse_args()

    cfg = load_config(args.config)
    _, _, test_ds, feature_cols, _ = build_datasets(cfg)
    test_loader = make_loader(test_ds, cfg.train.batch_size, shuffle=False, num_workers=cfg.train.num_workers)

    model = FIJEPA(input_dim=len(feature_cols), config=cfg.model)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.device == "auto" else "cpu")
    model.to(device)

    metrics = {}
    for batch in test_loader:
        context = batch["context"].to(device)
        future = batch["futures"].to(device)
        asset_id = batch["asset_id"].to(device)
        if asset_id.ndim == 1:
            asset_id = asset_id.unsqueeze(-1)
        out = model(context, future=future, asset_id=asset_id)
        z_context = out["z_context"].detach().cpu()
        z_target = out["z_targets"].detach().cpu()
        mse = torch.mean((out["stages"][-1]["mu"].detach().cpu() - z_target[:, -1, :]) ** 2).item()
        metrics.setdefault("latent_mse", 0.0)
        metrics["latent_mse"] += mse
        metrics.setdefault("latent_mean_norm", 0.0)
        metrics["latent_mean_norm"] += float(z_context.norm(dim=-1).mean().item())

    for k in metrics:
        metrics[k] /= max(1, len(test_loader))

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
