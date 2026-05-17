from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
from fijepa import FIJEPAConfig, FIJEPA


def test_forward_smoke():
    cfg = FIJEPAConfig()
    model = FIJEPA(input_dim=12, config=cfg.model)
    x = torch.randn(4, 16, 12)
    y = torch.randn(4, 3, 8, 12)
    asset = torch.zeros(4, 1, dtype=torch.long)
    out = model(x, future=y, asset_id=asset)
    assert "z_context" in out
    assert out["z_context"].shape[0] == 4
    assert out["z_targets"].shape[0] == 4
    assert len(out["stages"]) == len(cfg.model.stage_names)
