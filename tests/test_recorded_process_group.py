"""Independent, bounded regression for a live worker after its leader exits."""

from __future__ import annotations

import ctypes
import json
import multiprocessing
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from unittest.mock import patch

import pytest

from fijepa import recorded_training as recorded


def _kill_group(pid):
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _reap_with_deadline(pid, seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        observed, status = os.waitpid(pid, os.WNOHANG)
        if observed == pid:
            return status
        time.sleep(0.01)
    raise TimeoutError("controlled descendant remained alive after cleanup")


def _supervise_attempt(directory, sender):
    """Confine Linux subreaper state and all fixture children to this process."""
    processes = []
    result = {"ok": False}
    marker = directory / "worker.pid"
    worker_reaped = False

    def stop_supervisor(signum, frame):
        raise TimeoutError("independent supervisor watchdog expired")

    signal.signal(signal.SIGTERM, stop_supervisor)
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        # PR_SET_CHILD_SUBREAPER: adopt the orphan when its original parent exits.
        # A dead adopted child can then be reaped without relying on PID 1.
        if libc.prctl(36, 1, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "cannot enable fixture subreaper")
        config = directory / "config.yaml"
        config.write_text("train:\n  max_epochs: 1\n", encoding="utf-8")
        worker = (
            "import os,signal,time; from pathlib import Path; "
            "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            f"Path({str(marker)!r}).write_text(str(os.getpid())); time.sleep(30)"
        )
        leader = (
            "import subprocess,sys,time; "
            f"subprocess.Popen([sys.executable, '-c', {worker!r}]); time.sleep(30)"
        )
        real_popen = subprocess.Popen

        def launch(command, **kwargs):
            # Keep the wrapper's actual file streams, session, wait and cleanup.
            process = real_popen([sys.executable, "-c", leader], **kwargs)
            processes.append(process)
            (directory / "leader.pid").write_text(str(process.pid))
            return process

        with patch.object(recorded.subprocess, "Popen", launch):
            report = recorded.run_training_attempt(
                config, directory / "attempts", max_seconds=2
            )
        assert marker.exists(), "controlled worker never installed SIGTERM ignore"
        assert report["status"] == "TIMED_OUT", report
        assert report["exit_code"] == -signal.SIGTERM, report
        terminal = Path(report["attempt_directory"]) / "finished.json"
        assert json.loads(terminal.read_text()) == report
        worker_pid = int(marker.read_text())
        # kill(pid, 0) alone would mistake an unreaped zombie for a live leak.
        # Waiting for this adopted child verifies actual signal termination.
        status = _reap_with_deadline(worker_pid, 3)
        worker_reaped = True
        assert os.WIFSIGNALED(status)
        assert os.WTERMSIG(status) == signal.SIGKILL
        result.update(ok=True, worker_exit_code=os.waitstatus_to_exitcode(status))
    except BaseException:  # noqa: BLE001 - transport every child failure to pytest
        result["failure"] = traceback.format_exc()
    finally:
        # Preserve the failure above before reviewer cleanup kills any survivor.
        # On the broken predecessor this path kills and reaps the live worker.
        try:
            for process in processes:
                _kill_group(process.pid)
                process.wait(timeout=5)
            if marker.exists() and not worker_reaped:
                worker_pid = int(marker.read_text())
                try:
                    _reap_with_deadline(worker_pid, 3)
                except ChildProcessError:
                    pass
        except BaseException:  # noqa: BLE001 - a cleanup failure must fail the test
            result["ok"] = False
            result["cleanup_failure"] = traceback.format_exc()
        sender.send(result)
        sender.close()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux subreaper fixture")
def test_timeout_stops_sigterm_ignoring_worker_after_leader_exits(tmp_path):
    # Fork after Python/Torch imports so startup cost is outside the fixture's
    # attempt deadline. Subreaper state never affects the main pytest process.
    context = multiprocessing.get_context("fork")
    receiver, sender = context.Pipe(duplex=False)
    supervisor = context.Process(target=_supervise_attempt, args=(tmp_path, sender))
    supervisor.start()
    sender.close()
    try:
        supervisor.join(timeout=20)
        assert not supervisor.is_alive(), "independent process watchdog expired"
        assert supervisor.exitcode == 0
        assert receiver.poll(1), "independent supervisor returned no evidence"
        result = receiver.recv()
        assert result["ok"], result
        assert result["worker_exit_code"] == -signal.SIGKILL
    finally:
        if supervisor.is_alive():
            marker = tmp_path / "leader.pid"
            if marker.exists():
                _kill_group(int(marker.read_text()))
            # The supervisor's handler enters its bounded kill/reap finally.
            supervisor.terminate()
            supervisor.join(timeout=10)
            if supervisor.is_alive():
                supervisor.kill()
                supervisor.join(timeout=5)
        receiver.close()
