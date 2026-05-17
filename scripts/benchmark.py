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
import json
import os
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("TQDM_DISABLE", "1")

from scripts.train import load_config
from fijepa import benchmark_fijepa
from fijepa.benchmark import benchmark_suite, summarize_suite


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--source", choices=["synthetic", "macrodata"], default=None)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 27])
    p.add_argument("--ablations", nargs="+", default=["full"])
    p.add_argument("--suite", action="store_true")
    p.add_argument("--out", default=None)
    p.add_argument("--no-probes", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.source is not None:
        cfg.data.source = args.source
    if cfg.data.source == "macrodata" and not cfg.data.target_cols:
        cfg.data.target_cols = ["gdp_growth"]

    if args.suite:
        results = benchmark_suite(cfg, seeds=args.seeds, ablations=args.ablations, max_epochs=args.epochs)
        payload = {"results": results, "summary": summarize_suite(results)}
    else:
        payload = benchmark_fijepa(cfg, seed=args.seeds[0], max_epochs=args.epochs, ablation=args.ablations[0], compute_probes=not args.no_probes)

    text = json.dumps(payload, indent=2, default=float)
    if args.out:
        Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
