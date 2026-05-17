from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
import yaml

from fijepa import FIJEPAConfig, FIJEPA, build_datasets
from fijepa.data import make_loader
from fijepa.trainer import Trainer


def _apply_updates(obj, raw: dict):
    for k, v in raw.items():
        if not hasattr(obj, k):
            continue
        current = getattr(obj, k)
        if isinstance(v, dict) and not isinstance(current, (str, int, float, bool, list, tuple, type(None))):
            _apply_updates(current, v)
        else:
            setattr(obj, k, v)


def load_config(path: str) -> FIJEPAConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}
    cfg = FIJEPAConfig()
    for section in ["seed", "device"]:
        if section in raw:
            setattr(cfg, section, raw[section])
    for key in ["data", "model", "train", "loss", "experiment"]:
        if key in raw:
            _apply_updates(getattr(cfg, key), raw[key])
    return cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--source", choices=["synthetic", "macrodata"], default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.source is not None:
        cfg.data.source = args.source

    torch.manual_seed(cfg.seed)

    train_ds, val_ds, test_ds, feature_cols, _ = build_datasets(cfg)
    train_loader = make_loader(train_ds, cfg.train.batch_size, shuffle=True, num_workers=cfg.train.num_workers)
    val_loader = make_loader(val_ds, cfg.train.batch_size, shuffle=False, num_workers=cfg.train.num_workers)

    model = FIJEPA(input_dim=len(feature_cols), config=cfg.model)
    trainer = Trainer(model, train_loader, val_loader, cfg, feature_dim=len(feature_cols), run_dir=cfg.experiment.output_dir)
    history = trainer.fit()

    print(json.dumps(history[-1], indent=2))


if __name__ == "__main__":
    main()
