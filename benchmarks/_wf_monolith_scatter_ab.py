"""In-process A/B: monolith-with-custom-scatter vs the sorted pipeline.

The monolithic ``wavefront_shade`` now supports custom ray bouncing (scatter)
and normal-mapped lighting, so it -- not the sorted Cycles-style pipeline -- is
the default even for scenes that customise bouncing. This benchmark renders a
material-heavy scene that *includes a custom-scatter mob* (so both paths run
their scatter machinery) and alternates the monolith (``set_wavefront_sort_materials
(False)``) against the forced sorted pipeline (``True``), timing the wavefront
render stage in isolation (CUDA-synced) plus end to end. Alternating in one
process cancels thermal-throttle drift; the first pair is discarded as
compile/clock warm-up.

    .venv/Scripts/python.exe benchmarks/_wf_monolith_scatter_ab.py [reps]
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
    set_reflectivity,
    set_shadows,
)
from algan.rendering.raytracing.settings import (
    set_wavefront_sort_materials,  # noqa: E402
)
from algan.rendering.taichi_runtime import _sync_devices  # noqa: E402

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
    from algan.rendering.shaders.fragment_shaders import forced_mirror_scatter

    with Sync():
        for i in range(15):
            row, col = divmod(i, 5)
            m = (
                Sphere()
                .scale(0.55)
                .move(RIGHT * (col - 2) * 1.5 + UP * (row - 1) * 1.6)
            )
            if i % 5 == 0:
                # A custom-scatter mob: forces frag_scatters non-empty, so the
                # monolith exercises its per-material scatter dispatch.
                m.set_fragment_shader(forced_mirror_scatter)
            else:
                m.set_material(_MATERIALS[i % 4]())
            m.spawn()
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
    _sync_devices()
    t0 = time.perf_counter()
    r = _orig_wf(*a, **k)
    _sync_devices()
    _wf_times.append(time.perf_counter() - t0)
    return r


tracer_mod.raytrace_render_wavefront = _timed_wf


def render_once(sort_on, shadows, tag):
    SceneManager.reset()
    set_fragment_shading(True)
    set_shadows(shadows)
    set_wavefront_sort_materials(sort_on)
    build()
    scene = SceneManager.instance()
    _wf_times.clear()
    t0 = time.perf_counter()
    scene.save_frame(os.path.join(OUT_DIR, f"wfmsab_{tag}.png"))
    return sum(_wf_times), time.perf_counter() - t0


def bench(shadows, label):
    render_once(False, shadows, f"{label}_warm_mono")
    render_once(True, shadows, f"{label}_warm_sort")
    mono_wf, mono_tot, sort_wf, sort_tot = [], [], [], []
    for _ in range(REPS):
        wf, tot = render_once(False, shadows, f"{label}_mono")
        mono_wf.append(wf)
        mono_tot.append(tot)
        wf, tot = render_once(True, shadows, f"{label}_sort")
        sort_wf.append(wf)
        sort_tot.append(tot)
    mw, sw = statistics.median(mono_wf), statistics.median(sort_wf)
    mt, st = statistics.median(mono_tot), statistics.median(sort_tot)
    print(
        f"[{label:11s}] wavefront: monolith {mw * 1e3:8.1f} ms  "
        f"sorted {sw * 1e3:8.1f} ms  (monolith is {sw / mw:5.2f}x sorted)   "
        f"end-to-end: mono {mt:6.2f}s sorted {st:6.2f}s",
        flush=True,
    )


def main():
    bench(False, "scatter")
    bench(True, "scatter+shad")


if __name__ == "__main__":
    with torch.inference_mode():
        main()
