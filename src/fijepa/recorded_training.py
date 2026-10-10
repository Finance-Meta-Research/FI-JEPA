"""Run the existing trainer in a fresh, retained, time-bounded attempt."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import signal
import stat
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import torch
import yaml

from .config import ModelConfig, load_config
from .model import FIJEPA

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "fijepa_recorded_training_v1"


def _publish_json(path: Path, value: dict) -> None:
    """Publish a complete receipt without replacing a pre-existing receipt."""
    payload = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    descriptor, temporary = tempfile.mkstemp(prefix=".receipt-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("attempt wall-time budget exhausted")
    return remaining


def _regular_reader(path: Path, *, allow_symlinks: bool = True):
    """Reject FIFOs/devices before reading; check the opened descriptor as well."""
    if not allow_symlinks and path.is_symlink():
        raise ValueError(f"symlink is not an owned output file: {path}")
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError(f"a regular file is required: {path}")
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
    if not allow_symlinks:
        flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"a regular file is required: {path}")
        return os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise


def _copy_snapshot(source: Path, destination: Path, deadline: float) -> dict:
    digest = hashlib.sha256()
    size = 0
    with _regular_reader(source) as reader, destination.open("xb") as writer:
        while True:
            _remaining(deadline)
            chunk = reader.read(1024 * 1024)
            if not chunk:
                break
            writer.write(chunk)
            digest.update(chunk)
            size += len(chunk)
        writer.flush()
        os.fsync(writer.fileno())
    return {"sha256": digest.hexdigest(), "bytes": size}


def _identity(path: Path) -> dict:
    digest = hashlib.sha256()
    size = 0
    with _regular_reader(path) as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return {"sha256": digest.hexdigest(), "bytes": size}


def _strict_json(raw: bytes):
    def invalid_constant(value):
        raise ValueError(f"non-finite JSON constant: {value}")

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def finite_float(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("non-finite JSON number")
        return result

    return json.loads(
        raw,
        parse_constant=invalid_constant,
        object_pairs_hook=unique_object,
        parse_float=finite_float,
    )


def _validate_outputs(run_dir: Path, config: dict) -> dict:
    def canonical(value):
        return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))

    snapshots = {}

    def read_snapshot(name):
        with _regular_reader(run_dir / name, allow_symlinks=False) as handle:
            raw = handle.read()
        snapshots[f"training/{name}"] = {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
        }
        return raw

    expected_config = canonical(config)
    if canonical(_strict_json(read_snapshot("config.json"))) != expected_config:
        raise ValueError(
            "trainer output configuration differs from the retained configuration"
        )
    history = [
        _strict_json(line)
        for line in read_snapshot("history.jsonl").splitlines()
        if line.strip()
    ]
    epochs = config["train"]["max_epochs"]
    if len(history) != epochs or any(
        not isinstance(row, dict)
        or type(row.get("epoch")) is not int
        or row["epoch"] != index
        for index, row in enumerate(history, start=1)
    ):
        raise ValueError(
            "trainer history does not contain exactly the requested epochs"
        )
    for row in history:
        for name in ("train_total", "val_total"):
            value = row.get(name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"history contains an invalid {name}")
    expected_best = min(history, key=lambda row: row["val_total"])
    expected_state = None
    for name in ("best.pt", "last.pt"):
        checkpoint = torch.load(
            io.BytesIO(read_snapshot(name)),
            map_location="cpu",
            weights_only=True,
        )
        if (
            not isinstance(checkpoint, dict)
            or canonical(checkpoint.get("config")) != expected_config
        ):
            raise ValueError(f"{name} is not bound to the retained configuration")
        expected_epoch = expected_best["epoch"] if name == "best.pt" else epochs
        if (
            type(checkpoint.get("epoch")) is not int
            or checkpoint["epoch"] != expected_epoch
        ):
            raise ValueError(f"{name} has an inconsistent epoch")
        if name == "best.pt":
            metric = checkpoint.get("metric")
            if (
                isinstance(metric, bool)
                or not isinstance(metric, (int, float))
                or metric != expected_best["val_total"]
            ):
                raise ValueError(
                    "best.pt does not match the validation-selected history row"
                )
        state = checkpoint.get("model")
        if (
            not isinstance(state, dict)
            or not state
            or any(
                not isinstance(value, torch.Tensor) or not torch.isfinite(value).all()
                for value in state.values()
            )
        ):
            raise ValueError(f"{name} has missing or non-finite model tensors")
        if expected_state is None:
            feature_cols = config["data"].get("feature_cols")
            if feature_cols is not None:
                input_dim = len(feature_cols)
            elif (
                config["model"]["backbone"] in ("mlp", "conv")
                and config["model"]["num_layers"] == 0
            ):
                # An empty backbone is supported when the original input width
                # already equals its downstream projection/normalization width.
                input_dim = config["model"]["d_model"]
            else:
                backbone = config["model"]["backbone"]
                input_key = (
                    "online_encoder.backbone.input_proj.weight"
                    if backbone == "transformer"
                    else "online_encoder.backbone.net.0.weight"
                )
                projection = state.get(input_key)
                expected_rank = 3 if backbone == "conv" else 2
                if (
                    projection is None
                    or projection.ndim != expected_rank
                    or projection.shape[1] < 1
                ):
                    raise ValueError(f"{name} has invalid model structure at input")
                input_dim = projection.shape[1]
            if input_dim < 1:
                raise ValueError(f"{name} has invalid model structure at input")
            # Meta construction validates structure without allocating weights
            # or consuming the caller's RNG. The trainer owns float32 state.
            with torch.device("meta"):
                expected_state = FIJEPA(
                    input_dim=input_dim, config=ModelConfig(**config["model"])
                ).float().state_dict()
        if state.keys() != expected_state.keys():
            raise ValueError(f"{name} model keys differ from configured FI-JEPA")
        for key, value in state.items():
            expected = expected_state[key]
            if value.shape != expected.shape or value.dtype != expected.dtype:
                raise ValueError(
                    f"{name} model structure differs from configured FI-JEPA: {key}"
                )
    return {
        "epochs_completed": epochs,
        "last_metrics": history[-1],
        "validated_outputs": snapshots,
    }


def _stop_child(process: subprocess.Popen) -> None:
    if os.name == "posix":
        # The leader can exit before its workers. Its process group still owns
        # those descendants, so cleanup must not stop at process.poll()/wait().
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        finally:
            # A worker may ignore SIGTERM while the leader exits immediately.
            # Finish the owned group even when waiting on the leader succeeded.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        return
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def run_training_attempt(
    config_path: str | Path,
    attempt_root: str | Path,
    *,
    max_seconds: float,
    source: str | None = None,
) -> dict:
    """Retain a development attempt; retries always allocate a new directory.

    The budget covers preparation and the training child. Cleanup permits a
    five-second termination grace; output verification and receipt hashing follow
    child termination. Neither this receipt nor a zero exit code is a scientific
    result. Abrupt machine/process termination can leave only ``started.json``.
    """
    if isinstance(max_seconds, bool) or not isinstance(max_seconds, (int, float)):
        raise ValueError("max_seconds must be a finite positive number")
    if not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("max_seconds must be a finite positive number")
    if source not in (None, "synthetic", "macrodata"):
        raise ValueError("source override must be synthetic or macrodata")
    parent = Path(attempt_root).resolve()
    parent.mkdir(parents=True, exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix="attempt-", dir=parent))
    started = time.monotonic()
    deadline = started + max_seconds
    record = {
        "schema": SCHEMA,
        "status": "STARTED",
        "attempt_directory": str(attempt),
        "started_unix_seconds": time.time(),
        "max_seconds": max_seconds,
        "phase": "DEVELOPMENT",
        "scientific_result": "NOT_EVALUATED",
        "request": {
            "config_path": str(Path(config_path).resolve()),
            "source_override": source,
        },
        "environment": {
            "python": sys.version,
            "torch": str(torch.__version__),
            "platform": sys.platform,
        },
    }
    _publish_json(attempt / "started.json", record)
    process = None
    cancellation = None
    try:
        inputs = attempt / "inputs"
        inputs.mkdir()
        identities = {
            "config.yaml": _copy_snapshot(
                Path(config_path), inputs / "config.yaml", deadline
            )
        }
        cfg = load_config(str(inputs / "config.yaml"))
        if source is not None:
            cfg.data.source = source
        if type(cfg.train.max_epochs) is not int or cfg.train.max_epochs <= 0:
            raise ValueError("recorded training requires a positive integer max_epochs")
        if cfg.data.source not in ("synthetic", "macrodata"):
            field = "csv_path" if cfg.data.csv_path is not None else "parquet_path"
            original = getattr(cfg.data, field)
            if original is not None:
                name = (
                    "observations.csv"
                    if field == "csv_path"
                    else "observations.parquet"
                )
                identities[name] = _copy_snapshot(
                    Path(original), inputs / name, deadline
                )
                setattr(cfg.data, field, str(inputs / name))
        cfg.experiment.output_dir = str(attempt / "training")
        resolved = asdict(cfg)
        _publish_json(attempt / "resolved_config.json", resolved)
        with (inputs / "effective_config.yaml").open("x", encoding="utf-8") as handle:
            yaml.safe_dump(resolved, handle)
            handle.flush()
            os.fsync(handle.fileno())
        source_files = sorted((ROOT / "src" / "fijepa").glob("*.py"))
        source_files.append(ROOT / "scripts" / "train.py")
        _publish_json(
            attempt / "inputs.json",
            {
                "retained_inputs": identities,
                "source_files_at_launch": {
                    str(path.relative_to(ROOT)): _identity(path)
                    for path in source_files
                },
                "source_identity_scope": "Files observed before launch; no authentication or concurrent-source-edit guarantee",
            },
        )
        command = [
            sys.executable,
            str(ROOT / "scripts" / "train.py"),
            "--config",
            str(inputs / "effective_config.yaml"),
        ]
        record["command"] = command
        with (
            (attempt / "stdout.log").open("xb", buffering=0) as stdout,
            (attempt / "stderr.log").open("xb", buffering=0) as stderr,
        ):
            _remaining(deadline)
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=stdout,
                stderr=stderr,
                start_new_session=os.name == "posix",
            )
            record["child_pid"] = process.pid
            record["exit_code"] = process.wait(timeout=_remaining(deadline))
        if record["exit_code"] != 0:
            record.update(
                status="FAILED", error="training child returned a nonzero exit status"
            )
        else:
            record.update(_validate_outputs(attempt / "training", resolved))
            record["status"] = "COMPLETED"
    except (subprocess.TimeoutExpired, TimeoutError) as exc:
        record.update(status="TIMED_OUT", error=str(exc), error_type=type(exc).__name__)
    except KeyboardInterrupt as exc:
        cancellation = exc
        record.update(
            status="CANCELLED",
            error="caller interrupted training",
            error_type="KeyboardInterrupt",
        )
    except Exception as exc:
        record.update(
            status="FAILED",
            error=str(exc),
            error_type=type(exc).__name__,
            traceback=traceback.format_exc(),
        )
    finally:
        if process is not None:
            _stop_child(process)
            record["exit_code"] = process.returncode
        record["elapsed_seconds"] = time.monotonic() - started
        try:
            files = {}
            for path in sorted(attempt.rglob("*")):
                if path.is_symlink():
                    raise ValueError(
                        f"unexpected symlink in attempt: {path.relative_to(attempt)}"
                    )
                if path.is_file():
                    files[str(path.relative_to(attempt))] = _identity(path)
            record["retained_files"] = files
            if any(
                files.get(name) != identity
                for name, identity in record.get("validated_outputs", {}).items()
            ):
                raise ValueError(
                    "trainer outputs changed after their completion validation"
                )
        except Exception as exc:
            record["artifact_inventory_error"] = str(exc)
            if record["status"] == "COMPLETED":
                record["status"] = "FAILED"
        _publish_json(attempt / "finished.json", record)
    if cancellation is not None:
        raise KeyboardInterrupt(
            f"Cancelled attempt retained at {attempt}"
        ) from cancellation
    return record
