"""Native wheel contents, build failures, and non-Mac packaging stay explicit."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def build_hook():
    pytest.importorskip("hatchling")
    spec = importlib.util.spec_from_file_location(
        "algan_native_build_test", ROOT / "hatch_build.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _hook(module, root, target="wheel"):
    return module.MetalArenaBuildHook(
        str(root), {}, None, None, str(root / "dist"), target
    )


@pytest.mark.parametrize(
    ("platform", "target"),
    [("linux", "wheel"), ("win32", "wheel"), ("darwin", "sdist")],
)
def test_non_native_builds_never_invoke_a_compiler(
    build_hook, monkeypatch, tmp_path, platform, target
):
    monkeypatch.setattr(build_hook.sys, "platform", platform)
    monkeypatch.setattr(
        build_hook.subprocess, "run", lambda *a, **k: pytest.fail("compiled")
    )
    data = {"force_include": {}}
    _hook(build_hook, tmp_path, target).initialize("standard", data)
    assert data == {"force_include": {}}


@pytest.mark.parametrize("version", ["standard", "editable"])
def test_mac_build_is_abi3_and_includes_the_binary_not_a_torch_link(
    build_hook, monkeypatch, tmp_path, version
):
    monkeypatch.setattr(build_hook.sys, "platform", "darwin")
    source = tmp_path / "algan/rendering/_mps_arena_native.mm"
    source.parent.mkdir(parents=True)
    source.write_text("// synthetic source for a fake compiler\n")
    calls = []

    def compile_native(command, **kwargs):
        calls.append(command)
        assert kwargs["check"]
        assert kwargs["timeout"] > 0
        assert "-fno-objc-arc" in command
        assert command.count("-arch") == 2
        assert "arm64" in command
        assert "x86_64" in command
        assert not any("torch" in arg.lower() for arg in command)
        Path(command[-1]).write_bytes(b"native library")

    monkeypatch.setattr(build_hook.subprocess, "run", compile_native)
    data = {"force_include": {}}
    hook = _hook(build_hook, tmp_path)
    hook.initialize(version, data)
    assert len(calls) == 1
    assert data["pure_python"] is False
    assert data["tag"] == "cp310-abi3-macosx_11_0_universal2"
    relative = "algan/rendering/_mps_arena_native.abi3.so"
    if version == "editable":
        assert (tmp_path / relative).read_bytes() == b"native library"
    else:
        [(binary, destination)] = data["force_include"].items()
        assert destination == relative
        assert Path(binary).read_bytes() == b"native library"
    hook.finalize(version, data, "unused")
    assert not Path(calls[0][-1]).exists()


def test_missing_mac_toolchain_fails_the_build_actionably(
    build_hook, monkeypatch, tmp_path
):
    monkeypatch.setattr(build_hook.sys, "platform", "darwin")

    def missing(*args, **kwargs):
        raise FileNotFoundError("xcrun")

    monkeypatch.setattr(build_hook.subprocess, "run", missing)
    with pytest.raises(RuntimeError, match="xcode-select --install"):
        _hook(build_hook, tmp_path).initialize("standard", {"force_include": {}})


def test_native_compiler_failure_carries_its_diagnostic(
    build_hook, monkeypatch, tmp_path
):
    monkeypatch.setattr(build_hook.sys, "platform", "darwin")

    def rejected(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "clang++", stderr="missing SDK header")

    monkeypatch.setattr(build_hook.subprocess, "run", rejected)
    with pytest.raises(RuntimeError, match="missing SDK header"):
        _hook(build_hook, tmp_path).initialize("standard", {"force_include": {}})


def test_ci_has_one_unpartitioned_portable_suite_for_all_devices():
    workflow = yaml.safe_load((ROOT / ".github/workflows/test.yaml").read_text())
    steps = workflow["jobs"]["build"]["steps"]
    runs = [
        step
        for step in steps
        if "pytest tests/unit_tests tests/fast" in step.get("run", "")
    ]
    assert len(runs) == 1
    assert "if" not in runs[0]
    assert "--ci-batch" not in runs[0]["run"]
    assert "-n 1" in runs[0]["run"]
    assert "--cov=algan" in runs[0]["run"]
