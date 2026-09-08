"""The benchmark must expose progress before its rendering child exits."""

from __future__ import annotations

import importlib.util
import sys
import threading
import time
from pathlib import Path


def _run_logged():
    path = Path(__file__).resolve().parents[2] / "benchmarks/_mac_depth_buffer_ab.py"
    spec = importlib.util.spec_from_file_location("depth_buffer_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run_logged


def test_child_output_is_visible_before_exit(tmp_path, capsys):
    release = tmp_path / "release"
    log = tmp_path / "child.log"
    command = [
        sys.executable,
        "-u",
        "-c",
        "import pathlib,sys,time\n"
        "print('child ready', flush=True)\n"
        "while not pathlib.Path(sys.argv[1]).exists(): time.sleep(0.01)\n"
        "print('child finished', flush=True)\n",
        str(release),
    ]
    outcomes = []
    runner = _run_logged()
    worker = threading.Thread(target=lambda: outcomes.append(runner(command, log, 5)))
    worker.start()
    visible = ""
    try:
        deadline = time.monotonic() + 3
        while "child ready" not in visible and time.monotonic() < deadline:
            visible += capsys.readouterr().out
            time.sleep(0.01)
        assert "child ready" in visible
        assert worker.is_alive(), "Progress appeared only after the child exited"
    finally:
        release.touch()
        worker.join(timeout=6)
    assert outcomes == [0]
    assert "child finished" in capsys.readouterr().out
    assert log.read_text() == "child ready\nchild finished\n"


def test_child_error_preserves_exit_code_and_stderr(tmp_path, capsys):
    log = tmp_path / "error.log"
    code = _run_logged()(
        [
            sys.executable,
            "-u",
            "-c",
            "import sys; print('failed', file=sys.stderr); sys.exit(7)",
        ],
        log,
        5,
    )
    assert code == 7
    assert log.read_text() == "failed\n"
    assert "failed" in capsys.readouterr().out


def test_child_timeout_is_bounded_and_preserves_output(tmp_path, capsys, monkeypatch):
    # Native Mac sampling is deliberately excluded from this process test.
    monkeypatch.setattr(sys, "platform", "linux")
    log = tmp_path / "timeout.log"
    started = time.monotonic()
    code = _run_logged()(
        [
            sys.executable,
            "-u",
            "-c",
            "import time; print('started', flush=True); time.sleep(20)",
        ],
        log,
        0.5,
    )
    assert code == 124
    assert time.monotonic() - started < 5
    assert log.read_text() == "started\n"
    visible = capsys.readouterr().out
    assert "started" in visible
    assert "timeout:" in visible
