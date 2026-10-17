"""Run development training with retained inputs, logs and terminal receipts."""

from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fijepa.recorded_training import run_training_attempt  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--attempt-root", required=True)
    parser.add_argument("--max-seconds", required=True, type=float)
    parser.add_argument("--source", choices=("synthetic", "macrodata"))
    args = parser.parse_args()

    def terminate(signum, frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    previous = signal.signal(signal.SIGTERM, terminate)
    try:
        report = run_training_attempt(
            args.config,
            args.attempt_root,
            max_seconds=args.max_seconds,
            source=args.source,
        )
    finally:
        signal.signal(signal.SIGTERM, previous)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["status"] == "COMPLETED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
