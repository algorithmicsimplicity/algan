"""Video-scale A/B: monolithic vs sorted-material wavefront on a real render.

Same material-heavy scene as ``_wf_sorted_ab.py`` but *animated* and rendered
to video (PREVIEW settings), so the wavefront processes multi-frame batches
with full ~1.4M-ray tiles -- the regime real renders run in, where per-tile
launch/sync overhead amortizes (a single LD save_frame is the worst case for
the sorted path's fixed per-iteration cost). Times the wavefront stage in
isolation via the monkeypatched ``raytrace_render_wavefront``.

    .venv/Scripts/python.exe benchmarks/_wf_sorted_video_ab.py [reps]
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
    PREVIEW,
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
from algan.utils.algan_utils import render_to_file  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "_tc_out")
os.makedirs(OUT_DIR, exist_ok=True)

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 2

_MATERIALS = [
    lambda: MeshLambertMaterial(color=BLUE),
    lambda: MeshPhongMaterial(color=RED, specular=0xFFFFFF, shininess=50),
    lambda: MeshStandardMaterial(color=GREEN, metalness=0.8, roughness=0.3),
    lambda: MeshBasicMaterial(color=YELLOW),
]


def build_and_animate():
    mobs = []
    with Sync():
        for i in range(15):
            row, col = divmod(i, 5)
            m = (
                Sphere()
                .scale(0.55)
                .move(RIGHT * (col - 2) * 1.5 + UP * (row - 1) * 1.6)
                .set_material(_MATERIALS[i % 4]())
            )
            m.spawn()
            mobs.append(m)
        mirror = (
            Cylinder(radius=0.35, height=2.2)
            .move(DOWN * 2.6 + LEFT * 3)
            .set_material(MeshLambertMaterial(color=WHITE))
        )
        set_reflectivity(mirror, 0.6)
        mirror.spawn()
    with Sync():  # ~1s of simultaneous motion -> a real multi-frame batch
        for i, m in enumerate(mobs):
            m.move((RIGHT if i % 2 else LEFT) * 0.5 + UP * 0.2)


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


def render_once(sort_on, shadows):
    SceneManager.reset()
    set_fragment_shading(True)
    set_shadows(shadows)
    set_wavefront_sort_materials(sort_on)
    build_and_animate()
    _wf_times.clear()
    t0 = time.perf_counter()
    render_to_file(file_name="wfab_video", output_dir=OUT_DIR, render_settings=PREVIEW)
    return sum(_wf_times), time.perf_counter() - t0


def bench(shadows, label):
    render_once(False, shadows)  # warm-up pair (compiles + GPU clocks)
    render_once(True, shadows)
    mono_wf, mono_tot, sort_wf, sort_tot = [], [], [], []
    for _ in range(REPS):
        wf, tot = render_once(False, shadows)
        mono_wf.append(wf)
        mono_tot.append(tot)
        wf, tot = render_once(True, shadows)
        sort_wf.append(wf)
        sort_tot.append(tot)
    mw, sw = statistics.median(mono_wf), statistics.median(sort_wf)
    mt, st = statistics.median(mono_tot), statistics.median(sort_tot)
    print(
        f"[{label:11s}] wavefront: mono {mw:7.2f} s  sorted {sw:7.2f} s  "
        f"({mw / sw:5.2f}x)   end-to-end: mono {mt:6.1f} s  "
        f"sorted {st:6.1f} s  ({mt / st:5.2f}x)",
        flush=True,
    )


def main():
    bench(False, "frag")
    bench(True, "frag+shadow")


if __name__ == "__main__":
    main()
