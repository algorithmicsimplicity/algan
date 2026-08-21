"""In-process A/B benchmark: monolithic vs sorted-material wavefront shading.

Builds a material-heavy scene (15 spheres cycling Lambert / Phong / Standard /
Basic materials + a reflective cylinder), then alternates rendering it with
material sorting OFF (monolithic ``wavefront_shade``) and ON (peel / sort /
per-material ``wf_shade_event``), timing the wavefront render stage in
isolation (CUDA-synced around ``raytrace_render_wavefront``) as well as end to
end. Alternating in one process cancels thermal-throttle drift (see
``lean-triangle-only-kernel``); the first pair is discarded as compile/clock
warm-up. Runs the frag-only config and the frag+shadows config.

    .venv/Scripts/python.exe benchmarks/_wf_sorted_ab.py [reps]
"""

from __future__ import annotations

import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import algan.rendering.raytracing.tracer as tracer_mod  # noqa: E402
from algan import (  # noqa: E402
    BLUE,
    DOWN,
    GREEN,
    LEFT,
    RED,
    RIGHT,
    UP,
    WHITE,
    YELLOW,
    Cylinder,
    MeshBasicMaterial,
    MeshLambertMaterial,
    MeshPhongMaterial,
    MeshStandardMaterial,
    SceneManager,
    Sphere,
    Sync,
)
from algan.rendering.raytracing import (  # noqa: E402
    set_fragment_shading,
    set_ray_traced_shadows,
    set_reflectivity,
)
from algan.rendering.raytracing.settings import set_material_sorting  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "_tc_out")
os.makedirs(OUT_DIR, exist_ok=True)

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 4

_MATERIALS = [
    lambda: MeshLambertMaterial(color=BLUE),
    lambda: MeshPhongMaterial(color=RED, specular=0xFFFFFF, shininess=50),
    lambda: MeshStandardMaterial(color=GREEN, metalness=0.8, roughness=0.3),
    lambda: MeshBasicMaterial(color=YELLOW),
]


def build():
    with Sync():
        for i in range(15):
            row, col = divmod(i, 5)
            (
                Sphere()
                .scale(0.55)
                .move(RIGHT * (col - 2) * 1.5 + UP * (row - 1) * 1.6)
                .set_material(_MATERIALS[i % 4]())
                .spawn()
            )
        mirror = (
            Cylinder(radius=0.35, height=2.2)
            .move(DOWN * 2.6 + LEFT * 3)
            .set_material(MeshLambertMaterial(color=WHITE))
        )
        set_reflectivity(mirror, 0.6)
        mirror.spawn()


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


def render_once(sort_on, shadows, tag):
    SceneManager.reset()
    set_fragment_shading(True)
    set_ray_traced_shadows(shadows)
    set_material_sorting(sort_on)
    build()
    scene = SceneManager.instance()
    _wf_times.clear()
    t0 = time.perf_counter()
    scene.save_frame(os.path.join(OUT_DIR, f"wfab_{tag}.png"))
    total = time.perf_counter() - t0
    return sum(_wf_times), total


def bench(shadows, label):
    # Warm-up pair (kernel compiles + cold GPU clocks), then alternate.
    render_once(False, shadows, f"{label}_warm_mono")
    render_once(True, shadows, f"{label}_warm_sort")
    mono_wf, mono_tot, sort_wf, sort_tot = [], [], [], []
    for _rep in range(REPS):
        wf, tot = render_once(False, shadows, f"{label}_mono")
        mono_wf.append(wf)
        mono_tot.append(tot)
        wf, tot = render_once(True, shadows, f"{label}_sort")
        sort_wf.append(wf)
        sort_tot.append(tot)
    mw, sw = statistics.median(mono_wf), statistics.median(sort_wf)
    mt, st = statistics.median(mono_tot), statistics.median(sort_tot)
    print(
        f"[{label:11s}] wavefront: mono {mw * 1e3:8.1f} ms  "
        f"sorted {sw * 1e3:8.1f} ms  ({mw / sw:5.2f}x)   "
        f"end-to-end: mono {mt:6.2f} s  sorted {st:6.2f} s  "
        f"({mt / st:5.2f}x)",
        flush=True,
    )


def main():
    bench(False, "frag")
    bench(True, "frag+shadow")


if __name__ == "__main__":
    with torch.inference_mode():
        main()
