"""A cache validation benchmark must fail when only pixel parity passed."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

_PATH = Path(__file__).resolve().parents[2] / "benchmarks" / "_taichi_source_key_check.py"
_SPEC = importlib.util.spec_from_file_location("source_key_check_harness", _PATH)
check = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(check)


def _stats(**updates):
    return {"hits": 0, "misses": 0, "poisoned": 0, "verified": 0, "keyed": 0} | updates


@pytest.mark.parametrize(
    ("name", "stats"),
    [
        ("on", _stats(misses=3, keyed=3)),
        ("on", _stats(hits=2, misses=1, keyed=3)),
        ("on", _stats(hits=3, poisoned=1, keyed=3)),
        ("on", _stats()),
        ("on", {}),
        ("verify", _stats(misses=3, keyed=3)),
        ("verify", _stats(hits=3, keyed=3)),
        ("verify", _stats(verified=2, misses=1, keyed=3)),
        ("warm", _stats()),
        ("off", _stats(hits=3, keyed=3)),
    ],
)
def test_cache_failures_are_errors_even_when_frames_match(monkeypatch, capsys, name, stats):
    monkeypatch.setattr(
        check,
        "run_arm",
        lambda *args: {
            "process": 1.0,
            "render": 0.5,
            "frontend": 0.2,
            "digest": "same-frame",
            "stats": stats,
        },
    )
    assert check.main(["--arms", name]) == 1
    assert "SOURCE-KEY-CHECK: FAILED" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("name", "stats"),
    [
        ("off", _stats()),
        ("warm", _stats(misses=3, keyed=3)),
        ("warm", _stats(hits=3, keyed=3)),
        ("on", _stats(hits=3, keyed=3)),
        ("verify", _stats(verified=3, keyed=3)),
    ],
)
def test_valid_cache_arms_pass(name, stats):
    assert check._cache_failure(name, stats) is None


@pytest.mark.parametrize(
    ("name", "verify"), [("warm", "0"), ("on", "0"), ("verify", "1")]
)
def test_children_force_fresh_process_mode_and_isolate_verify(
    monkeypatch, tmp_path, name, verify
):
    monkeypatch.setenv("ALGAN_AUTO_DAEMON", "1")
    monkeypatch.setenv("ALGAN_USE_DAEMON", "1")
    monkeypatch.setenv("ALGAN_TAICHI_SOURCE_KEY_VERIFY", "1")

    def run(*args, **kwargs):
        env = kwargs["env"]
        assert env["ALGAN_AUTO_DAEMON"] == "0"
        assert env["ALGAN_USE_DAEMON"] == "0"
        assert env["ALGAN_TAICHI_SOURCE_KEY_VERIFY"] == verify
        return SimpleNamespace(returncode=1, stdout="", stderr="")

    monkeypatch.setattr(check.subprocess, "run", run)
    check.run_arm(name, tmp_path, quiet=True)
