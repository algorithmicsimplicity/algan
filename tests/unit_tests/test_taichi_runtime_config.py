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
"""

from __future__ import annotations

import ast
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
