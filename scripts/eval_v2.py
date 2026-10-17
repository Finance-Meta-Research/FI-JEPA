"""Explicit sample-mean evaluation of a supplied, configuration-bound checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from fijepa import FIJEPA, build_datasets  # noqa: E402
from fijepa.config import load_config  # noqa: E402
from fijepa.data import make_loader  # noqa: E402
from fijepa.evaluation import evaluate_latents  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    args = parser.parse_args()
    cfg = load_config(args.config)
    resolved_config = asdict(cfg)
    checkpoint_bytes = Path(args.checkpoint).read_bytes()
    checkpoint = torch.load(
        io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True
    )
    # Shape-compatible weights alone do not bind preprocessing or model behavior.
    # Batch size, loader workers and execution device may be changed explicitly.
    saved_config = checkpoint.get("config", {})
    for section in ("seed", "model", "data"):
        if saved_config.get(section) != resolved_config[section]:
            raise ValueError(
                f"checkpoint {section} configuration does not match evaluation config"
            )

    torch.manual_seed(cfg.seed)
    _, val_ds, test_ds, feature_cols, _ = build_datasets(cfg)
    dataset = val_ds if args.split == "validation" else test_ds
    loader = make_loader(
        dataset, cfg.train.batch_size, shuffle=False, num_workers=cfg.train.num_workers
    )
    model = FIJEPA(input_dim=len(feature_cols), config=cfg.model)
    model.load_state_dict(checkpoint["model"], strict=True)
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if cfg.device == "auto"
        else torch.device(cfg.device)
    )
    model.to(device)
    # Separate evaluation draws from the discarded random parameter initialization.
    torch.manual_seed(cfg.seed)
    metrics = evaluate_latents(model, loader, device=device)
    result = {
        **metrics,
        "split": args.split,
        "checkpoint_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
        "resolved_config_sha256": hashlib.sha256(
            json.dumps(resolved_config, sort_keys=True, allow_nan=False).encode("utf-8")
        ).hexdigest(),
        "seed": cfg.seed,
        "batch_size": cfg.train.batch_size,
        "device": str(device),
        "torch_version": str(torch.__version__),
        "feature_cols": feature_cols,
        "horizons": cfg.data.horizons,
        "stochastic_predictor": cfg.model.stochastic,
    }
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
