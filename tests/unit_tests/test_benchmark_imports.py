"""Every ``algan`` name a benchmark script imports still exists.

``benchmarks/`` holds the A/B, parity and profiling scripts that source
comments and design documents cite as the evidence for a default. Nothing ran
them, so a rename in ``algan/`` broke them silently: 32 of them imported names
that no longer existed (``algan.RenderSettings``, ``render_to_file``,
``enable_ray_tracing``, ...), 13 of those cited as evidence. This resolves
each script's ``from algan... import X``, ``import algan.x`` and ``algan.x.y``
references against the live package without running the script -- many
render at module level -- so a rename that breaks one fails here instead.

The scripts already broken when this check was added are frozen in
``KNOWN_BROKEN``: each is cited from outside ``benchmarks/`` (or by a script
that is), so it was kept rather than deleted, and none could be repaired by
a mechanical rename. The list only shrinks.
"""

from __future__ import annotations

import ast
import importlib
from functools import cache
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ROOT / "benchmarks"

KNOWN_BROKEN = frozenset(
    {
        "_aa_run_gate_check.py",
        "_aa_sample_count_cost.py",
        "_aa_secondary_check.py",
        "_analytic_aa_bez_ab.py",
        "_glossy_determinism.py",
        "_glossy_ggx_check.py",
        "_glossy_pool_check.py",
        "_node_pack_parity.py",
        "_notch_scene_check.py",
        "_prep_profile.py",
        "_raster_empty_skip_parity.py",
        "_raster_sparse_coverage_parity.py",
        "_wf_gen_fused_ab.py",
        "_wf_mem_trim_ab.py",
        "_wf_parity_check.py",
    }
)


@cache
def _module(name):
    """The imported module, or None when ``name`` is no algan module."""
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as e:
        if (e.name or "").startswith("algan"):
            return None
        raise


def _missing(parts):
    """The first dotted prefix of ``parts`` that does not resolve, or None."""
    for i in range(len(parts), 0, -1):
        obj = _module(".".join(parts[:i]))
        if obj is None:
            continue
        for j, part in enumerate(parts[i:], start=i):
            if hasattr(obj, part):
                obj = getattr(obj, part)
                continue
            # A submodule its package does not import by itself.
            obj = _module(".".join(parts[: j + 1]))
            if obj is None:
                return ".".join(parts[: j + 1])
        return None
    return ".".join(parts)


def _references(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            module = node.module.split(".")
            if module[0] == "algan":
                for alias in node.names:
                    yield module if alias.name == "*" else [*module, alias.name]
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "algan":
                    yield alias.name.split(".")
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
            parts, current = [], node
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name) and current.id == "algan":
                yield ["algan", *reversed(parts)]


def _unresolved(path):
    tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    return sorted({m for parts in _references(tree) if (m := _missing(parts))})


SCRIPTS = sorted(BENCHMARKS.rglob("*.py"))


def test_every_benchmark_script_imports_only_names_algan_still_has():
    broken = {}
    for path in SCRIPTS:
        rel = path.relative_to(BENCHMARKS).as_posix()
        if rel in KNOWN_BROKEN:
            continue
        missing = _unresolved(path)
        if missing:
            broken[rel] = missing
    assert not broken, (
        f"benchmark scripts reference names algan/ no longer has: {broken}. "
        f"Follow the rename in the script, or delete a script nothing cites."
    )


@pytest.mark.parametrize("name", sorted(KNOWN_BROKEN))
def test_a_known_broken_benchmark_is_still_broken(name):
    """An entry whose script was repaired or deleted leaves ``KNOWN_BROKEN``."""
    path = BENCHMARKS / name
    assert path.is_file(), f"{name} is gone; drop it from KNOWN_BROKEN"
    assert _unresolved(path), f"{name} resolves now; drop it from KNOWN_BROKEN"
