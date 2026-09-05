"""One rule about ``ti.init``, enforced over the whole repository.

``ti.init`` is process-global and takes Taichi's *default* for every kwarg it is
not given, so a bare call anywhere reconfigures Taichi for everything that runs
after it in the same process -- including kernels compiled later, which is what
makes it invisible at the call site.

That is not hypothetical. ``tests/unit_tests/test_color_space.py`` called
``ti.init(arch=ti.cpu)`` to make sure Taichi was up before defining a kernel,
which turned ``advanced_optimization`` back on (Algan runs with it off), and
under it Taichi miscompiles ``pbr_neutral_tonemap``: the in-place peak rescale
inside the compression branch is dropped, so an authored white tonemapped to
244 instead of 222. Three guards in ``test_tonemapping.py`` failed in CI for
weeks and every one of them passed when run on its own, because the file that
broke them sorts earlier in the run and nothing in either file mentions the
other.

The rule: call :func:`algan.rendering.taichi_runtime.init_taichi` (idempotent),
or pass ``**taichi_init_kwargs()`` to start from Algan's own configuration and
override deliberately from there.

The one place Taichi is ever re-initialized is
:func:`~algan.rendering.taichi_runtime.ensure_taichi_for_render`, when the
render device has moved across the CPU/GPU line since the last render. The rest
of this file is about the arrangement that makes that possible: nothing may
need a live program while algan imports, because at that moment the device is
still the script's to choose.
"""

from __future__ import annotations

import ast
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

from algan.taichi_compat import (
    BACKEND,
    PROGRAM_ATTR,
    kernel_specializations,
    program,
    submodule,
    ti,
)

_REPOSITORY_ROOT = Path(__file__).parents[2]

#: Where ``ti.init`` is *defined* to be called from, so it is exempt.
_RUNTIME_MODULE = _REPOSITORY_ROOT / "algan" / "rendering" / "taichi_runtime.py"

#: Scanned in full. Vendored code is excluded: it is read-only here, and a
#: vendored library initializing its own Taichi is not Algan's call to make.
_SCANNED_ROOTS = ("algan", "tests", "docs")


#: Names the kernel compiler is bound to anywhere in the tree. ``ti`` is the one
#: this project writes (``from algan.taichi_compat import ti``); the other three
#: are the spellings a module that imported an implementation directly would
#: use, and the rule below is about the *call*, so it has to recognise all of
#: them rather than trusting the import to be the sanctioned one.
_COMPILER_BINDINGS = frozenset({"ti", "taichi", "qd", "quadrants"})


def _is_taichi_init(node):
    """``ti.init(...)``, however the compiler module happens to be bound."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "init"
        and (isinstance(func.value, ast.Name) and func.value.id in _COMPILER_BINDINGS)
    )


def _starts_from_algan_config(node):
    return any(
        keyword.arg is None
        and isinstance(keyword.value, ast.Call)
        and isinstance(keyword.value.func, ast.Name)
        and keyword.value.func.id == "taichi_init_kwargs"
        for keyword in node.keywords
    )


def _source_files():
    for root in _SCANNED_ROOTS:
        directory = _REPOSITORY_ROOT / root
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*.py")):
            if "external_libraries" in path.parts or path == _RUNTIME_MODULE:
                continue
            yield path


def test_every_taichi_init_starts_from_algans_configuration():
    violations = []
    for path in _source_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - a file that is not importable
            continue
        for node in ast.walk(tree):
            if _is_taichi_init(node) and not _starts_from_algan_config(node):
                violations.append(f"{path.relative_to(_REPOSITORY_ROOT)}:{node.lineno}")

    assert not violations, (
        "call init_taichi(), or pass **taichi_init_kwargs() and override from "
        "there -- a bare ti.init() takes Taichi's defaults for everything it "
        "omits and applies them to the whole process:\n" + "\n".join(violations)
    )


def test_init_taichi_does_not_reinitialize_a_running_program():
    """The property the rule above leans on: calling it again is free.

    A second ``ti.init`` would discard every kernel compiled so far, so a test
    that just wants Taichi up must be able to say so without paying for it.
    """
    from algan.rendering.taichi_runtime import init_taichi

    init_taichi()
    live = program()
    assert live is not None
    init_taichi()
    assert program() is live


def test_ensure_taichi_for_render_leaves_a_matching_arch_alone():
    """The common case: every render that did not change the device.

    A re-init discards every compiled kernel in the process, so this has to be
    free when there is nothing to do -- otherwise each render would pay a
    kernel-preparation pass.
    """
    from algan.rendering.taichi_runtime import ensure_taichi_for_render

    ensure_taichi_for_render()
    live = program()
    assert live is not None
    assert ensure_taichi_for_render() is False
    assert program() is live


def test_ensure_taichi_for_render_reinitializes_when_the_arch_changes(monkeypatch):
    """And the uncommon one: a render device that moved across the CPU/GPU line.

    Driven by faking the *wanted* arch rather than by setting a device this
    machine may not have, so it runs everywhere. What it pins down is the
    decision, not the backend: a mismatch must produce a new ``Program``, and
    the kernels compiled against the old one must be dropped rather than
    silently reused on the wrong device.
    """
    from algan.rendering import taichi_runtime

    taichi_runtime.ensure_taichi_for_render()
    live = program()
    assert live is not None

    monkeypatch.setattr(taichi_runtime, "_arch_matches_render_device", lambda: False)
    assert taichi_runtime.ensure_taichi_for_render() is True
    assert program() is not live
    runtime = submodule("lang.impl").get_runtime()
    assert all(not kernel_specializations(kernel) for kernel in runtime.kernels)

    # Leave the process on the arch the rest of the suite expects.
    monkeypatch.undo()
    taichi_runtime.ensure_taichi_for_render()
    assert taichi_runtime._arch_matches_render_device()


@pytest.mark.parametrize(
    ("device", "expected"),
    [("cpu", "cpu"), ("cuda", "cuda"), ("mps", "metal"), ("xpu", "cpu")],
)
def test_the_arch_is_concrete_for_every_render_device(monkeypatch, device, expected):
    """Never ``ti.gpu``: that is a preference list ``init`` resolves by probing
    Vulkan and OpenGL and, when every probe fails, by falling back to the CPU
    with a warning -- leaving the live arch ``cpu`` against a ``cuda`` render
    device, which made ``ensure_taichi_for_render`` re-initialise on every
    render. The device is faked rather than selected, so this runs anywhere.
    """
    from algan.rendering import taichi_runtime

    monkeypatch.setattr(taichi_runtime, "render_device", lambda: torch.device(device))
    arch = taichi_runtime._taichi_arch()
    assert arch == getattr(ti, expected)
    assert arch != ti.gpu
    assert taichi_runtime.taichi_init_kwargs()["arch"] == arch


def test_init_kwargs_refuse_the_cpu_fallback_and_name_the_cache_directory():
    from algan.rendering import taichi_runtime

    kwargs = taichi_runtime.taichi_init_kwargs()
    assert kwargs["enable_fallback"] is False
    assert "gpu_max_reg" not in kwargs, "the knob never reached ptxas and is gone"
    # The kwarg is how the directory reaches Quadrants at all: it reads QD_
    # variables, never the TI_OFFLINE_CACHE_FILE_PATH name Algan honours.
    if not (BACKEND == "taichi" and os.environ.get("TI_OFFLINE_CACHE_FILE_PATH")):
        assert kwargs["offline_cache_file_path"] == str(
            taichi_runtime._TAICHI_CACHE_DIRECTORY
        )


def test_full_traceback_is_passed_only_when_asked_for(monkeypatch):
    from algan.rendering import taichi_runtime

    monkeypatch.delenv("ALGAN_TI_FULL_TRACEBACK", raising=False)
    assert "print_full_traceback" not in taichi_runtime.taichi_init_kwargs()
    monkeypatch.setenv("ALGAN_TI_FULL_TRACEBACK", "1")
    assert taichi_runtime.taichi_init_kwargs()["print_full_traceback"] is True


def _touch(path, age_seconds, now):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")
    os.utime(path, (now - age_seconds, now - age_seconds))
    return path


def test_stale_offline_cache_locks_are_removed_before_init(tmp_path, caplog):
    """A process killed while holding the cache's metadata lock leaves a bare
    O_EXCL file behind, and the compilers have no staleness rule: every later
    run fails to take it (five 50 ms retries), loads nothing and saves nothing,
    with a warning. Both compilers' names and both depths are covered; a fresh
    lock, an unrelated ``.lock`` and a lock two levels down are left alone.
    """
    from algan.rendering import taichi_runtime

    now = time.time()
    stale = [
        _touch(tmp_path / "ticache.lock", 3600, now),
        _touch(tmp_path / "kernel_compilation_manager" / "qdcache.lock", 3600, now),
        _touch(tmp_path / "ptx_cache_sm_86" / "ptxcache.lock", 3600, now),
    ]
    kept = [
        _touch(tmp_path / "kernel_compilation_manager" / "qdcache.lock.fresh", 1, now),
        _touch(tmp_path / "other.lock", 3600, now),
        _touch(tmp_path / "a" / "b" / "qdcache.lock", 3600, now),
    ]
    fresh = _touch(tmp_path / "fresh" / "qdcache.lock", 30, now)

    # Algan's logger does not propagate to the root, so caplog's handler is
    # attached to it directly rather than relying on ``at_level``.
    algan_logger = logging.getLogger("algan")
    algan_logger.addHandler(caplog.handler)
    try:
        removed = taichi_runtime._remove_stale_offline_cache_locks(tmp_path, now=now)
    finally:
        algan_logger.removeHandler(caplog.handler)

    assert sorted(removed) == sorted(stale)
    assert not any(path.exists() for path in stale)
    assert all(path.exists() for path in [*kept, fresh])
    assert sum("stale kernel-cache lock" in r.message for r in caplog.records) == 3


def test_stale_lock_sweep_tolerates_a_missing_cache_directory(tmp_path):
    from algan.rendering import taichi_runtime

    assert taichi_runtime._remove_stale_offline_cache_locks(tmp_path / "none") == []


def test_starting_a_program_sweeps_the_cache_directory(monkeypatch, tmp_path):
    """The sweep is wired into the one place ``ti.init`` is called from."""
    from algan.rendering import taichi_runtime

    now = time.time()
    lock = _touch(tmp_path / "kernel_compilation_manager" / "qdcache.lock", 3600, now)
    monkeypatch.setattr(taichi_runtime, "_TAICHI_CACHE_DIRECTORY", tmp_path)
    monkeypatch.setattr(taichi_runtime.ti, "init", lambda **kwargs: None)
    monkeypatch.setattr(taichi_runtime, "_install_taichi_compile_logger", lambda: None)

    taichi_runtime._start_program()

    assert not lock.exists()


def test_importing_algan_does_not_start_taichi():
    """Nothing may need a live Taichi program while algan imports.

    The arch is chosen from ``SETTINGS.computing.render_device``, which a script
    can still change after ``import algan``, so the program is created at the
    first kernel launch (``install_render_arch_guard``) or at the start of a
    render (``ensure_taichi_for_render``) -- never before. Defining a kernel
    needs no program; materializing one does, and that is the moment this
    defers to.

    Run in a subprocess with ``ti.init`` stubbed out, so a module that quietly
    starts needing a program at import fails here instead of re-pinning the
    device to whatever the environment said.

    The compiler is named by ``importlib`` rather than reached through
    ``algan.taichi_compat``, because importing anything under ``algan`` runs
    ``algan/__init__.py`` -- which is the very import the stub has to be in
    place before. The backend name is resolved here and pinned into the child's
    environment so the two processes cannot disagree about it.
    """
    probe = f"""
import importlib

ti = importlib.import_module({BACKEND!r})
_misc = importlib.import_module({BACKEND!r} + ".lang.misc")
_impl = importlib.import_module({BACKEND!r} + ".lang.impl")

_calls = []
ti.init = _misc.init = lambda *a, **k: _calls.append(k.get("arch"))

import algan  # noqa: F401

print("CALLS", len(_calls))
print("PROG", getattr(_impl.get_runtime(), {PROGRAM_ATTR!r}, None) is not None)
print("KERNELS", len(_impl.get_runtime().kernels))
"""
    environment = dict(os.environ, ALGAN_USE_DAEMON="0", ALGAN_TAICHI_BACKEND=BACKEND)
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        env=environment,
        cwd=_REPOSITORY_ROOT,
    )
    assert result.returncode == 0, result.stderr
    reported = dict(
        line.split(" ", 1)
        for line in result.stdout.splitlines()
        if line.startswith(("CALLS ", "PROG ", "KERNELS "))
    )
    assert reported["CALLS"] == "0", (
        "something called ti.init while algan imported; the render device can "
        "still change at that point:\n" + result.stdout
    )
    assert reported["PROG"] == "False"
    # The point of the exercise: kernels register fine without a program.
    assert int(reported["KERNELS"]) > 0
