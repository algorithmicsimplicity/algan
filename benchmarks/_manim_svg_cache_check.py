"""Parity + timing check for the persistent Manim SVG/Tex geometry cache.

Validates that rebuilding Tex glyphs from the on-disk cache is byte-identical to
a fresh svgelements parse, and reports cold / warm-memory / disk-reload timings.

Run: .venv/Scripts/python.exe benchmarks/_manim_svg_cache_check.py
"""

from __future__ import annotations

import shutil
import time

import manim as mn
import numpy as np

import algan  # noqa: F401  installs the cache patch
from algan.external_libraries.manim.mobject.text.tex_mobject import (
    SingleStringMathTex as _V,  # noqa: F401
)
from algan.utils import manim_svg_cache as C

TEX = ("a" * 50 + "\n") * 50


def leaf_state(mob):
    """Flat list of (points, fill_rgbas, stroke_rgbas) over the whole family."""
    out = []

    # deterministic pre-order over submobjects
    def walk(m):
        if len(m.submobjects) == 0:
            out.append(
                (
                    np.array(m.points, dtype=np.float64),
                    np.array(m.fill_rgbas, dtype=np.float64),
                    np.array(m.stroke_rgbas, dtype=np.float64),
                )
            )
        for sm in m.submobjects:
            walk(sm)

    walk(mob)
    return out


def compare(a, b, label):
    assert len(a) == len(b), f"{label}: leaf count {len(a)} != {len(b)}"
    max_pts = 0.0
    for i, ((pa, fa, sa), (pb, fb, sb)) in enumerate(zip(a, b)):
        assert pa.shape == pb.shape, (
            f"{label}: leaf {i} points shape {pa.shape} != {pb.shape}"
        )
        max_pts = max(max_pts, float(np.abs(pa - pb).max()) if pa.size else 0.0)
        assert np.array_equal(fa, fb), f"{label}: leaf {i} fill_rgbas differ"
        assert np.array_equal(sa, sb), f"{label}: leaf {i} stroke_rgbas differ"
    assert max_pts == 0.0, f"{label}: max point delta {max_pts} != 0"
    print(f"[OK] {label}: {len(a)} leaves byte-identical (max point delta 0)")


def build(cached):
    t = time.time()
    m = SingleStringMathTex(TEX, use_svg_cache=cached)
    return m, time.time() - t


# Use the installed manim's SingleStringMathTex (the one that is patched).
SingleStringMathTex = mn.SingleStringMathTex

# Fresh, un-cached reference (patched fn early-returns to a real parse).
ref, t_ref = build(cached=False)
ref_state = leaf_state(ref)
print(f"reference (uncached parse): {t_ref:.2f}s, {len(ref_state)} leaves")

# Cold: wipe disk + memo, build cached -> parses once, writes disk.
shutil.rmtree(C._cache_dir(), ignore_errors=True)
C._MEM_CACHE.clear()
cold, t_cold = build(cached=True)
compare(ref_state, leaf_state(cold), "cold(parse+save)")
print(f"cold build: {t_cold:.2f}s")

# Warm memory: same process, memo hit -> rebuild from recipe.
warm, t_warm = build(cached=True)
compare(ref_state, leaf_state(warm), "warm(mem rebuild)")
print(f"warm(mem) build: {t_warm:.2f}s")

# Cross-run simulation: drop the in-memory memo, force a disk load + rebuild.
C._MEM_CACHE.clear()
disk, t_disk = build(cached=True)
compare(ref_state, leaf_state(disk), "disk(load+rebuild)")
print(f"disk(reload) build: {t_disk:.2f}s")

print(
    f"\nSUMMARY  uncached={t_ref:.2f}s  cold={t_cold:.2f}s  "
    f"warm={t_warm:.2f}s  disk={t_disk:.2f}s  "
    f"(disk speedup {t_ref / max(t_disk, 1e-6):.1f}x)"
)
