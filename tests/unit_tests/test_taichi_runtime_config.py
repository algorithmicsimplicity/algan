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
import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPOSITORY_ROOT = Path(__file__).parents[2]

#: Where ``ti.init`` is *defined* to be called from, so it is exempt.
_RUNTIME_MODULE = _REPOSITORY_ROOT / "algan" / "rendering" / "taichi_runtime.py"

#: Scanned in full. Vendored code is excluded: it is read-only here, and a
#: vendored library initializing its own Taichi is not Algan's call to make.
_SCANNED_ROOTS = ("algan", "tests", "benchmarks", "docs")


def _is_taichi_init(node):
    """``ti.init(...)`` / ``taichi.init(...)``, however the module is bound."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "init"
        and (isinstance(func.value, ast.Name) and func.value.id in {"ti", "taichi"})
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
    ti = pytest.importorskip("taichi")
    from algan.rendering.taichi_runtime import init_taichi

    init_taichi()
    program = ti.lang.impl.get_runtime().prog
    assert program is not None
    init_taichi()
    assert ti.lang.impl.get_runtime().prog is program


def test_ensure_taichi_for_render_leaves_a_matching_arch_alone():
    """The common case: every render that did not change the device.

    A re-init discards every compiled kernel in the process, so this has to be
    free when there is nothing to do -- otherwise each render would pay a
    kernel-preparation pass.
    """
    ti = pytest.importorskip("taichi")
    from algan.rendering.taichi_runtime import ensure_taichi_for_render

    ensure_taichi_for_render()
    program = ti.lang.impl.get_runtime().prog
    assert program is not None
    assert ensure_taichi_for_render() is False
    assert ti.lang.impl.get_runtime().prog is program


def test_ensure_taichi_for_render_reinitializes_when_the_arch_changes(monkeypatch):
    """And the uncommon one: a render device that moved across the CPU/GPU line.

    Driven by faking the *wanted* arch rather than by setting a device this
    machine may not have, so it runs everywhere. What it pins down is the
    decision, not the backend: a mismatch must produce a new ``Program``, and
    the kernels compiled against the old one must be dropped rather than
    silently reused on the wrong device.
    """
    ti = pytest.importorskip("taichi")
    from algan.rendering import taichi_runtime

    taichi_runtime.ensure_taichi_for_render()
    program = ti.lang.impl.get_runtime().prog
    assert program is not None

    monkeypatch.setattr(taichi_runtime, "_arch_matches_render_device", lambda: False)
    assert taichi_runtime.ensure_taichi_for_render() is True
    assert ti.lang.impl.get_runtime().prog is not program
    assert all(
        not kernel.compiled_kernels for kernel in ti.lang.impl.get_runtime().kernels
    )

    # Leave the process on the arch the rest of the suite expects.
    monkeypatch.undo()
    taichi_runtime.ensure_taichi_for_render()
    assert taichi_runtime._arch_matches_render_device()


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
    """
    pytest.importorskip("taichi")
    probe = """
import taichi as ti
import taichi.lang.misc as _misc

_calls = []
ti.init = _misc.init = lambda *a, **k: _calls.append(k.get("arch"))

import algan  # noqa: F401

print("CALLS", len(_calls))
print("PROG", ti.lang.impl.get_runtime().prog is not None)
print("KERNELS", len(ti.lang.impl.get_runtime().kernels))
"""
    environment = dict(os.environ, ALGAN_USE_DAEMON="0")
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
