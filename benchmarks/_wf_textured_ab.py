"""In-process A/B: current per-vertex wavefront vs the experimental
texture-lookup wavefront (settings.wf_textured) on an all-Surface scene.

Builds a scene of solid-colour lit spheres + cylinders (one reflective, one
glass), renders it once with the textured shader OFF (the current per-vertex
wavefront) and once ON (three per-triangle texture lookups), and reports:

* a pixel-wise correctness diff of the two rendered frames, and
* the wavefront render-stage time (CUDA-synced around
  ``raytrace_render_wavefront``) and end-to-end time for each.

Alternating in one process cancels thermal-throttle drift; the first pair is a
compile / clock warm-up and discarded.

    .venv/Scripts/python.exe benchmarks/_wf_textured_ab.py [reps]
"""

from __future__ import annotations

import os
import statistics
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import algan.rendering.raytracing.tracer as tracer_mod  # noqa: E402
from algan import (  # noqa: E402
    BLUE,
    DOWN,
    GREEN,
    IN,
    LEFT,
    ORANGE,
    RED,
    RIGHT,
    UP,
    WHITE,
    YELLOW,
    Cylinder,
    MeshLambertMaterial,
    MeshPhongMaterial,
    MeshPhysicalMaterial,
    MeshStandardMaterial,
    SceneManager,
    Sphere,
    Sync,
)
from algan.rendering.raytracing import set_fragment_shading  # noqa: E402
from algan.rendering.raytracing.settings import set_wf_textured  # noqa: E402
from algan.rendering.taichi_runtime import _sync_devices  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "_tc_out")
os.makedirs(OUT_DIR, exist_ok=True)

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 4

_MATERIALS = [
    lambda: MeshLambertMaterial(color=BLUE),
    lambda: MeshPhongMaterial(color=RED, specular=0xFFFFFF, shininess=50),
    lambda: MeshStandardMaterial(color=GREEN, metalness=0.8, roughness=0.3),
    lambda: MeshLambertMaterial(color=ORANGE),
]


def build():
    with Sync():
        for i in range(12):
            row, col = divmod(i, 4)
            (
                Sphere()
                .scale(0.55)
                .move(RIGHT * (col - 1.5) * 1.5 + UP * (row - 1) * 1.6)
                .set_material(_MATERIALS[i % 4]())
                .spawn()
            )
        # Reflection / glass come from the materials (the renderer-side
        # set_reflectivity / set_refractive_index controls were removed by
        # the material-transport rework).
        (
            Cylinder(radius=0.35, height=2.2)
            .move(DOWN * 2.6 + LEFT * 3)
            .set_material(
                MeshStandardMaterial(color=WHITE, metalness=0.6, roughness=0.4)
            )
            .spawn()
        )
        (
            Sphere()
            .scale(0.7)
            .move(DOWN * 2.4 + RIGHT * 2.5 + IN * 1.0)
            .set_material(MeshPhysicalMaterial(color=YELLOW, ior=1.5, transmission=1.0))
            .spawn()
        )


_wf_times = []
_orig_wf = tracer_mod.raytrace_render_wavefront


def _timed_wf(*a, **k):
    _sync_devices()
    t0 = time.perf_counter()
    r = _orig_wf(*a, **k)
    _sync_devices()
    _wf_times.append(time.perf_counter() - t0)
    return r


tracer_mod.raytrace_render_wavefront = _timed_wf


def render_once(textured, tag):
    SceneManager.reset()
    set_fragment_shading(True)
    set_wf_textured(textured)
    build()
    scene = SceneManager.instance()
    _wf_times.clear()
    path = os.path.join(OUT_DIR, f"textured_{tag}.png")
    t0 = time.perf_counter()
    scene.save_frame(path)
    total = time.perf_counter() - t0
    return sum(_wf_times), total, path


def _load(path):
    import cv2

    return cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.int32)


def main():
    # Warm-up pair (kernel compiles + cold GPU clocks), then alternate.
    _, _, base_p = render_once(False, "warm_base")
    _, _, tex_p = render_once(True, "warm_tex")

    base = _load(base_p)
    tex = _load(tex_p)
    diff = np.abs(base - tex)
    print(
        f"correctness: shape {base.shape}  max|diff| {diff.max()}  "
        f"mean|diff| {diff.mean():.4f}  "
        f"px >2 {(diff.max(-1) > 2).sum()} / {base.shape[0] * base.shape[1]}",
        flush=True,
    )

    base_wf, base_tot, tex_wf, tex_tot = [], [], [], []
    for _rep in range(REPS):
        wf, tot, _ = render_once(False, "base")
        base_wf.append(wf)
        base_tot.append(tot)
        wf, tot, _ = render_once(True, "tex")
        tex_wf.append(wf)
        tex_tot.append(tot)
    bw, tw = statistics.median(base_wf), statistics.median(tex_wf)
    bt, tt = statistics.median(base_tot), statistics.median(tex_tot)
    print(
        f"wavefront:  per-vertex {bw * 1e3:8.1f} ms   textured {tw * 1e3:8.1f} ms"
        f"   ({bw / tw:5.2f}x)",
        flush=True,
    )
    print(
        f"end-to-end: per-vertex {bt:6.2f} s    textured {tt:6.2f} s"
        f"   ({bt / tt:5.2f}x)",
        flush=True,
    )


if __name__ == "__main__":
    with torch.inference_mode():
        main()
