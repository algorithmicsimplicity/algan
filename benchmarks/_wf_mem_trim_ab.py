"""In-process A/B for the full 'Family A+B' material-field memory trim
(settings.wf_mem_trim). Baseline (off) vs trim (on): triangles reordered into
material-class bands so tri_norm/tri_mat are compacted prefixes, tri_colors/
tri_extra addressed via a per-prim col_row remap, tex_meta/uvs widened to full
band-order arrays. The trim saves per-primitive memory but pays a per-hit
indirection gather -- this measures how much slower (if at all) that makes the
occupancy-bound shade kernel.

Verifies the trim is byte-identical to the baseline (same PNG) and that it
actually engaged (tracer._MEM_TRIM_ENGAGED bumped), then times the wavefront
render stage in isolation, alternating off/on to cancel thermal drift.

    .venv/Scripts/python.exe benchmarks/_wf_mem_trim_ab.py [reps]
"""

from __future__ import annotations

import os
import statistics
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import algan.rendering.raytracing.settings as rt_settings  # noqa: E402
import algan.rendering.raytracing.tracer as tracer_mod  # noqa: E402
from algan import (  # noqa: E402
    BLUE,
    GREEN,
    IN,
    ORANGE,
    PURPLE,
    RED,
    RIGHT,
    TEAL,
    UP,
    WHITE,
    YELLOW,
    MeshBasicMaterial,
    MeshLambertMaterial,
    SceneManager,
    Sphere,
    Sync,
)
from algan.rendering.raytracing import (  # noqa: E402
    set_fragment_shading,
    set_reflectivity,
    set_shadows,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), "_tc_out")
os.makedirs(OUT_DIR, exist_ok=True)

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
_COLORS = [BLUE, RED, GREEN, YELLOW, WHITE, ORANGE, PURPLE, TEAL]


def build():
    """A jittered cloud of overlapping, semi-transparent spheres with a MIX of
    material classes -- lit (Lambert), unlit (Basic) and reflective -- so the
    band reorder produces all three bands and col_row is exercised.
    """
    rng = np.random.default_rng(20260705)
    with Sync():
        n = 45
        for i in range(n):
            x = float(rng.uniform(-3.2, 3.2))
            y = float(rng.uniform(-1.9, 1.9))
            z = float(rng.uniform(-2.0, 2.0))
            col = _COLORS[i % len(_COLORS)]
            klass = i % 3
            if klass == 0:
                mat = MeshLambertMaterial(color=col, opacity=0.32)  # lit
            elif klass == 1:
                mat = MeshBasicMaterial(color=col, opacity=0.32)  # unlit
            else:
                mat = MeshLambertMaterial(color=col, opacity=0.32)  # reflective
            m = (
                Sphere(grid_height=10, grid_width=10)
                .scale(0.85)
                .move(RIGHT * x + UP * y + IN * z)
                .set_material(mat)
            )
            if klass == 2:
                set_reflectivity(m, 0.4)
            m.spawn()


_wf_times = []
_orig_wf = tracer_mod.raytrace_render_wavefront


def _timed_wf(*a, **k):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    r = _orig_wf(*a, **k)
    torch.cuda.synchronize()
    _wf_times.append(time.perf_counter() - t0)
    return r


tracer_mod.raytrace_render_wavefront = _timed_wf


def render_once(trim_on, tag):
    SceneManager.reset()
    set_fragment_shading(True)
    set_shadows(False)
    rt_settings.wf_mem_trim = bool(trim_on)
    build()
    scene = SceneManager.instance()
    _wf_times.clear()
    path = os.path.join(OUT_DIR, f"memtrim_{tag}.png")
    t0 = time.perf_counter()
    scene.save_frame(path)
    return sum(_wf_times), time.perf_counter() - t0, path


def main():
    # correctness + engagement
    e0 = tracer_mod._MEM_TRIM_ENGAGED[0]
    _, _, p_off = render_once(False, "id_off")
    e_off = tracer_mod._MEM_TRIM_ENGAGED[0] - e0
    _, _, p_on = render_once(True, "id_on")
    e_on = tracer_mod._MEM_TRIM_ENGAGED[0] - e0 - e_off
    a = cv2.imread(p_off, cv2.IMREAD_UNCHANGED).astype(np.int32)
    b = cv2.imread(p_on, cv2.IMREAD_UNCHANGED).astype(np.int32)
    ident = a.shape == b.shape and np.array_equal(a, b)
    mx = -1 if a.shape != b.shape else int(np.abs(a - b).max())
    print(
        f"engaged: off={e_off} launches, on={e_on} launches "
        f"({'TRIM FIRED' if e_on > 0 else 'TRIM DID NOT FIRE'})",
        flush=True,
    )
    print(
        f"byte-identity baseline vs trim: "
        f"{'IDENTICAL' if ident else f'DIFFER (max |d|={mx})'}",
        flush=True,
    )

    # timing: warm up (compile both template variants), then alternate
    render_once(False, "warm_off")
    render_once(True, "warm_on")
    off_wf, on_wf, off_tot, on_tot = [], [], [], []
    for _ in range(REPS):
        wf, tot, _ = render_once(False, "off")
        off_wf.append(wf)
        off_tot.append(tot)
        wf, tot, _ = render_once(True, "on")
        on_wf.append(wf)
        on_tot.append(tot)
    ow, nw = statistics.median(off_wf), statistics.median(on_wf)
    ot, nt = statistics.median(off_tot), statistics.median(on_tot)
    print(
        f"wavefront: baseline {ow * 1e3:8.1f} ms   trim {nw * 1e3:8.1f} ms   "
        f"(trim is {nw / ow:5.3f}x baseline)   "
        f"end-to-end: off {ot:6.2f}s on {nt:6.2f}s",
        flush=True,
    )


if __name__ == "__main__":
    with torch.inference_mode():
        main()
