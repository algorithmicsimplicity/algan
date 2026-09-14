"""Deterministic regression tests for asynchronous viewer-client state."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "script", ["test_viewer_async.cjs", "test_transcript.cjs", "test_project.cjs"]
)
def test_viewer_client_async_responses(script):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed to test the viewer's JavaScript client")
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [node, "--test", str(root / "tests/viewer" / script)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
