"""Deterministic regression tests for asynchronous viewer-client state."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


def test_viewer_client_async_responses():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed to test the viewer's JavaScript client")
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [node, "--test", str(root / "tests/viewer/test_viewer_async.cjs")],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
