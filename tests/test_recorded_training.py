"""Actual trainer equivalence and retained-failure process regressions."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path

import pytest
import torch
import yaml

from fijepa import FIJEPA, FIJEPAConfig
from fijepa import recorded_training as recorded
from fijepa.trainer import Trainer

ROOT = Path(__file__).resolve().parents[1]


def small_config(directory):
    cfg = FIJEPAConfig(seed=819, device="cpu")
    cfg.data.synthetic.num_assets = 2
    cfg.data.synthetic.num_steps = 40
    cfg.data.feature_cols = ["return", "spread"]
    cfg.data.context_length = 3
    cfg.data.horizons = [1]
    cfg.data.target_window = 1
    cfg.data.train_frac = 0.6
    cfg.data.val_frac = 0.2
    cfg.model.backbone = "mlp"
    cfg.model.d_model = 8
    cfg.model.latent_dim = 4
    cfg.model.num_layers = 1
    cfg.model.num_heads = 2
    cfg.model.predictor_hidden = 8
    cfg.model.stage_names = ["macro"]
    cfg.model.dropout = 0.0
    cfg.model.stochastic = False
    cfg.model.memory_size = 4
    cfg.model.memory_topk = 2
    cfg.train.batch_size = 8
    cfg.train.accumulate_grad_batches = 2
    cfg.train.max_epochs = 1
    cfg.experiment.output_dir = str(directory)
    return cfg


def save_config(path, cfg):
    path.write_text(yaml.safe_dump(asdict(cfg)), encoding="utf-8")


@pytest.fixture(scope="module")
def completed_runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("recorded-real-training")
    existing = root / "earlier-run"
    existing.mkdir()
    originals = {
        "config.json": b'{"earlier_run":true}\n',
        "history.jsonl": b'{"epoch":1,"earlier_run":true}\n',
        "best.pt": b"earlier best checkpoint fixture",
        "last.pt": b"earlier last checkpoint fixture",
    }
    for name, data in originals.items():
        (existing / name).write_bytes(data)
    cfg = small_config(existing)
    path = root / "requested.yaml"
    save_config(path, cfg)
    original_yaml = path.read_bytes()
    rng = torch.get_rng_state().clone()
    first = recorded.run_training_attempt(path, root / "attempts", max_seconds=90)
    assert first["status"] == "COMPLETED", first
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    second = recorded.run_training_attempt(path, root / "attempts", max_seconds=90)
    assert second["status"] == "COMPLETED", second
    assert path.read_bytes() == original_yaml
    for name, data in originals.items():
        assert (existing / name).read_bytes() == data
    # Independent reference is the existing, ordinary command, not a second
    # implementation of training or the wrapper's artifact validation helper.
    cfg.experiment.output_dir = str(root / "plain")
    plain_path = root / "plain.yaml"
    save_config(plain_path, cfg)
    plain = subprocess.run(
        [sys.executable, str(ROOT / "scripts/train.py"), "--config", str(plain_path)],
        cwd=ROOT,
        capture_output=True,
        timeout=90,
        check=False,
    )
    assert plain.returncode == 0, plain.stderr.decode()
    return root, first, second


def test_two_attempts_preserve_prior_artifacts_and_have_distinct_histories(
    completed_runs,
):
    _, first, second = completed_runs
    assert first["attempt_directory"] != second["attempt_directory"]
    for report in (first, second):
        attempt = Path(report["attempt_directory"])
        assert json.loads((attempt / "started.json").read_text())["status"] == "STARTED"
        assert json.loads((attempt / "finished.json").read_text()) == report
        assert report["scientific_result"] == "NOT_EVALUATED"
        assert report["epochs_completed"] == 1
        rows = (attempt / "training/history.jsonl").read_text().splitlines()
        assert len(rows) == 1
        assert json.loads(rows[0])["epoch"] == 1
        assert report["exit_code"] == 0


def test_recorded_command_matches_every_plain_checkpoint_tensor_and_metric(
    completed_runs,
):
    root, first, second = completed_runs
    plain = root / "plain"
    for report in (first, second):
        actual = Path(report["attempt_directory"]) / "training"
        assert (actual / "history.jsonl").read_bytes() == (
            plain / "history.jsonl"
        ).read_bytes()
        for name in ("best.pt", "last.pt"):
            expected = torch.load(plain / name, weights_only=True, map_location="cpu")
            obtained = torch.load(actual / name, weights_only=True, map_location="cpu")
            assert obtained["epoch"] == expected["epoch"]
            assert obtained["model"].keys() == expected["model"].keys()
            for key in expected["model"]:
                torch.testing.assert_close(
                    obtained["model"][key], expected["model"][key], rtol=0, atol=0
                )
            if "metric" in expected:
                assert obtained["metric"] == expected["metric"]


def test_receipt_hashes_and_sizes_match_actual_retained_bytes(completed_runs):
    _, report, _ = completed_runs
    attempt = Path(report["attempt_directory"])
    assert {
        "stdout.log",
        "stderr.log",
        "inputs/config.yaml",
        "training/best.pt",
        "training/last.pt",
    } <= report["retained_files"].keys()
    for name, identity in report["retained_files"].items():
        raw = (attempt / name).read_bytes()
        assert identity == {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
        }


def test_legacy_constructor_reproduces_original_config_replacement(tmp_path):
    cfg = small_config(tmp_path)
    before = b'{"earlier_run":true}\n'
    (tmp_path / "config.json").write_bytes(before)
    model = FIJEPA(input_dim=2, config=cfg.model)
    Trainer(model, [], [], cfg, feature_dim=2, run_dir=str(tmp_path))
    assert (tmp_path / "config.json").read_bytes() != before
    assert json.loads((tmp_path / "config.json").read_text()) == asdict(cfg)


@pytest.mark.parametrize("budget", [0, -1, float("nan"), float("inf"), True, "30"])
def test_invalid_budgets_do_not_start_attempts(tmp_path, budget):
    with pytest.raises(ValueError, match="finite positive"):
        recorded.run_training_attempt(
            tmp_path / "missing.yaml", tmp_path / "attempts", max_seconds=budget
        )
    assert not (tmp_path / "attempts").exists()


@pytest.mark.parametrize(
    "contents", [b"data: [", b"train:\n  max_epochs: false\n", None]
)
def test_configuration_failures_leave_terminal_evidence(tmp_path, contents):
    config = tmp_path / "config.yaml"
    if contents is not None:
        config.write_bytes(contents)
    report = recorded.run_training_attempt(
        config, tmp_path / "attempts", max_seconds=30
    )
    assert report["status"] == "FAILED"
    attempt = Path(report["attempt_directory"])
    assert (attempt / "finished.json").is_file()
    assert "error_type" in report
    assert "child_pid" not in report
    if contents is not None:
        assert (attempt / "inputs/config.yaml").read_bytes() == contents


def test_failed_loader_retains_data_snapshot_and_child_traceback(tmp_path):
    original = tmp_path / "source.csv"
    data = b"timestamp,asset_id,return,spread\ninvalid-time,0,1.0,2.0\n"
    original.write_bytes(data)
    cfg = small_config(tmp_path / "unused")
    cfg.data.source = "csv"
    cfg.data.csv_path = str(original)
    path = tmp_path / "config.yaml"
    save_config(path, cfg)
    report = recorded.run_training_attempt(path, tmp_path / "attempts", max_seconds=90)
    assert report["status"] == "FAILED"
    assert report["exit_code"] != 0
    attempt = Path(report["attempt_directory"])
    assert "missing/unparsable" in (attempt / "stderr.log").read_text()
    original.write_text("later replacement", encoding="utf-8")
    assert (attempt / "inputs/observations.csv").read_bytes() == data
    resolved = json.loads((attempt / "resolved_config.json").read_text())
    assert resolved["data"]["csv_path"] == str(attempt / "inputs/observations.csv")


def substituted_child(monkeypatch, code, interrupt=False):
    real_popen = subprocess.Popen

    def launch(command, **kwargs):
        process = real_popen([sys.executable, "-c", code], **kwargs)
        if interrupt:
            real_wait = process.wait
            first = True

            def wait(*args, **kw):
                nonlocal first
                if first:
                    first = False
                    raise KeyboardInterrupt
                return real_wait(*args, **kw)

            process.wait = wait
        return process

    monkeypatch.setattr(recorded.subprocess, "Popen", launch)


def test_timeout_stops_real_child_and_retains_partial_output(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    save_config(path, small_config(tmp_path / "unused"))
    substituted_child(
        monkeypatch, "import time; print('partial output', flush=True); time.sleep(30)"
    )
    report = recorded.run_training_attempt(path, tmp_path / "attempts", max_seconds=1)
    assert report["status"] == "TIMED_OUT"
    assert report["exit_code"] != 0
    assert (
        "partial output"
        in (Path(report["attempt_directory"]) / "stdout.log").read_text()
    )
    if os.name == "posix":
        with pytest.raises(ProcessLookupError):
            os.kill(report["child_pid"], 0)


def test_cancellation_stops_real_child_before_reraising(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    save_config(path, small_config(tmp_path / "unused"))
    substituted_child(monkeypatch, "import time; time.sleep(30)", interrupt=True)
    with pytest.raises(KeyboardInterrupt, match="Cancelled attempt retained"):
        recorded.run_training_attempt(path, tmp_path / "attempts", max_seconds=30)
    [terminal] = list((tmp_path / "attempts").glob("*/finished.json"))
    report = json.loads(terminal.read_text())
    assert report["status"] == "CANCELLED"
    assert report["exit_code"] != 0
    if os.name == "posix":
        with pytest.raises(ProcessLookupError):
            os.kill(report["child_pid"], 0)


def test_zero_exit_without_trainer_artifacts_cannot_be_completed(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    save_config(path, small_config(tmp_path / "unused"))
    substituted_child(monkeypatch, "pass")
    report = recorded.run_training_attempt(path, tmp_path / "attempts", max_seconds=30)
    assert report["exit_code"] == 0
    assert report["status"] == "FAILED"
    assert report["error_type"] == "FileNotFoundError"


@pytest.mark.parametrize(
    "tamper", ["config_type", "epoch", "selection", "nonfinite_weight"]
)
def test_checkpoint_tampering_is_rejected(completed_runs, tmp_path, tamper):
    _, report, _ = completed_runs
    source = Path(report["attempt_directory"])
    copied = tmp_path / "training"
    shutil.copytree(source / "training", copied)
    config = json.loads((source / "resolved_config.json").read_text())
    path = copied / "best.pt"
    checkpoint = torch.load(path, weights_only=True, map_location="cpu")
    if tamper == "config_type":
        checkpoint["config"]["model"]["dropout"] = False
    elif tamper == "epoch":
        checkpoint["epoch"] = True
    elif tamper == "selection":
        checkpoint["metric"] += 1
    else:
        next(iter(checkpoint["model"].values())).flatten()[0] = float("nan")
    torch.save(checkpoint, path)
    with pytest.raises(ValueError):
        recorded._validate_outputs(copied, config)


@pytest.mark.parametrize(
    "raw",
    [
        b'{"epoch":1,"epoch":1}',
        b'{"epoch":1,"val_total":NaN}',
        b'{"epoch":1,"val_total":1e999}',
    ],
)
def test_ambiguous_or_nonfinite_history_fails(completed_runs, tmp_path, raw):
    _, report, _ = completed_runs
    source = Path(report["attempt_directory"])
    copied = tmp_path / "training"
    shutil.copytree(source / "training", copied)
    config = json.loads((source / "resolved_config.json").read_text())
    (copied / "history.jsonl").write_bytes(raw + b"\n")
    with pytest.raises(ValueError):
        recorded._validate_outputs(copied, config)


def test_receipt_publication_never_overwrites_existing_evidence(tmp_path):
    path = tmp_path / "finished.json"
    before = b'{"earlier":"evidence"}\n'
    path.write_bytes(before)
    with pytest.raises(FileExistsError):
        recorded._publish_json(path, {"later": "evidence"})
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_cli_preserves_failure_and_returns_nonzero(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("data: [", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/train_recorded.py"),
            "--config",
            str(path),
            "--attempt-root",
            str(tmp_path / "attempts"),
            "--max-seconds",
            "30",
        ],
        cwd=ROOT,
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 2, result.stderr.decode()
    report = json.loads(result.stdout)
    assert report["status"] == "FAILED"
    assert (Path(report["attempt_directory"]) / "finished.json").is_file()


def test_valid_csv_snapshot_is_consumed_after_original_is_removed(
    tmp_path, monkeypatch
):
    source = tmp_path / "source.csv"
    lines = ["timestamp,asset_id,return,spread"]
    # Forty distinct dates retain complete train/validation/test windows.
    for day in range(40):
        timestamp = date(2020, 1, 1) + timedelta(days=day)
        lines.append(f"{timestamp.isoformat()},0,{day / 100},{0.01 + day / 10000}")
    raw = ("\n".join(lines) + "\n").encode()
    source.write_bytes(raw)
    cfg = small_config(tmp_path / "unused")
    cfg.data.source = "csv"
    cfg.data.csv_path = str(source)
    path = tmp_path / "config.yaml"
    save_config(path, cfg)
    real_popen = subprocess.Popen

    def launch(command, **kwargs):
        source.unlink()
        path.unlink()
        return real_popen(command, **kwargs)

    monkeypatch.setattr(recorded.subprocess, "Popen", launch)
    report = recorded.run_training_attempt(path, tmp_path / "attempts", max_seconds=90)
    assert report["status"] == "COMPLETED", report
    attempt = Path(report["attempt_directory"])
    assert (attempt / "inputs/observations.csv").read_bytes() == raw
    assert (
        report["retained_files"]["inputs/observations.csv"]["sha256"]
        == hashlib.sha256(raw).hexdigest()
    )
    assert report["epochs_completed"] == 1


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups and signals")
def test_cli_sigterm_records_cancellation_and_stops_its_child(tmp_path):
    path = tmp_path / "config.yaml"
    save_config(path, small_config(tmp_path / "unused"))
    marker = tmp_path / "child_pid"
    sleeper = f"import os,time; from pathlib import Path; Path({str(marker)!r}).write_text(str(os.getpid())); time.sleep(30)"
    driver = (
        "import runpy,subprocess,sys\n"
        "from fijepa import recorded_training as recorded\n"
        "real_popen=subprocess.Popen\n"
        "def launch(command, **kwargs):\n"
        f"    return real_popen([sys.executable, '-c', {sleeper!r}], **kwargs)\n"
        "recorded.subprocess.Popen=launch\n"
        f"sys.argv={[str(ROOT / 'scripts/train_recorded.py'), '--config', str(path), '--attempt-root', str(tmp_path / 'attempts'), '--max-seconds', '30']!r}\n"
        f"runpy.run_path({str(ROOT / 'scripts/train_recorded.py')!r}, run_name='__main__')\n"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", driver],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 30
        while (
            not marker.exists()
            and process.poll() is None
            and time.monotonic() < deadline
        ):
            time.sleep(0.02)
        assert marker.exists(), "controlled child did not start"
        process.send_signal(signal.SIGTERM)
        process.communicate(timeout=10)
        assert process.returncode != 0
        [terminal] = list((tmp_path / "attempts").glob("*/finished.json"))
        report = json.loads(terminal.read_text())
        assert report["status"] == "CANCELLED"
        assert report["child_pid"] == int(marker.read_text())
        with pytest.raises(ProcessLookupError):
            os.kill(report["child_pid"], 0)
    finally:
        if process.poll() is None:
            process.terminate()
            process.communicate(timeout=10)


@pytest.mark.skipif(os.name != "posix", reason="POSIX FIFO admission")
@pytest.mark.parametrize("which", ["config", "csv"])
def test_fifo_input_is_rejected_before_any_blocking_read(tmp_path, which):
    path = tmp_path / "config.yaml"
    if which == "config":
        os.mkfifo(path)
    else:
        fifo = tmp_path / "source.csv"
        os.mkfifo(fifo)
        cfg = small_config(tmp_path / "unused")
        cfg.data.source = "csv"
        cfg.data.csv_path = str(fifo)
        save_config(path, cfg)
    # Fork after imports so the watchdog measures file admission rather than
    # interpreter/Torch startup or shutdown. The predecessor blocks in open().
    pid = os.fork()
    if pid == 0:
        try:
            report = recorded.run_training_attempt(
                path, tmp_path / "attempts", max_seconds=0.5
            )
            os._exit(0 if report["status"] == "FAILED" else 2)
        except BaseException:
            os._exit(3)
    reaped = False
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            found, status = os.waitpid(pid, os.WNOHANG)
            if found:
                reaped = True
                assert os.waitstatus_to_exitcode(status) == 0
                break
            time.sleep(0.01)
        assert reaped, "regular-file admission blocked on a FIFO"
    finally:
        if not reaped:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
    [terminal] = list((tmp_path / "attempts").glob("*/finished.json"))
    report = json.loads(terminal.read_text())
    assert report["status"] == "FAILED"
    assert "regular file" in report["error"]
    assert "child_pid" not in report


def test_replaced_output_cannot_be_sealed_as_completed(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    save_config(path, small_config(tmp_path / "unused"))
    substituted_child(monkeypatch, "pass")

    def validation(run_dir, config):
        run_dir.mkdir()
        output = run_dir / "history.jsonl"
        before = b"validated bytes\n"
        output.write_bytes(before)
        identity = {"sha256": hashlib.sha256(before).hexdigest(), "bytes": len(before)}
        output.write_bytes(b"replacement after validation\n")
        return {"validated_outputs": {"training/history.jsonl": identity}}

    monkeypatch.setattr(recorded, "_validate_outputs", validation)
    report = recorded.run_training_attempt(path, tmp_path / "attempts", max_seconds=30)
    assert report["status"] == "FAILED"
    assert "changed after" in report["artifact_inventory_error"]
    assert (
        report["validated_outputs"]["training/history.jsonl"]
        != report["retained_files"]["training/history.jsonl"]
    )
