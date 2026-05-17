from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
import itertools
import json
from pathlib import Path
import copy

from scripts.train import load_config
from fijepa import FIJEPA, build_datasets
from fijepa.data import make_loader
from fijepa.trainer import Trainer
import torch


def run_one(cfg, seed, output_dir):
    cfg = copy.deepcopy(cfg)
    cfg.seed = seed
    cfg.experiment.output_dir = str(output_dir)
    torch.manual_seed(seed)
    train_ds, val_ds, _, feature_cols, _ = build_datasets(cfg)
    train_loader = make_loader(train_ds, cfg.train.batch_size, shuffle=True, num_workers=cfg.train.num_workers)
    val_loader = make_loader(val_ds, cfg.train.batch_size, shuffle=False, num_workers=cfg.train.num_workers)
    model = FIJEPA(input_dim=len(feature_cols), config=cfg.model)
    trainer = Trainer(model, train_loader, val_loader, cfg, feature_dim=len(feature_cols), run_dir=cfg.experiment.output_dir)
    hist = trainer.fit(max_epochs=cfg.train.max_epochs)
    return hist[-1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 27])
    args = p.parse_args()

    cfg = load_config(args.config)
    out = []
    for s in args.seeds:
        res = run_one(cfg, s, Path(cfg.experiment.output_dir) / f"seed_{s}")
        out.append({"seed": s, **res})
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
