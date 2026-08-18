"""Cost of the analytic-coverage sample loop, per kernel, on a triangle-dense
scene.

The sample count (``_AA_SAMPLES`` in ``raster_taichi.py``) is a compile-time
constant, so the two arms are two RUNS of this script with the constant edited
and the kernel cache cleared in between -- there is no in-process A/B available.
Report DEVICE times per kernel, not wall clock: the sample loop lives inside
``raster_tri_count`` / ``raster_tri_write`` (and ``raster_tri_z`` on the dense
path), and cross-process wall time on this machine swings ~2x with thermals.

Run: .venv/Scripts/python.exe benchmarks/_aa_sample_count_cost.py
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("ALGAN_PREFETCH_BATCHES", "0")
# Two runs so the reported numbers exclude JIT and cold clocks.
os.environ.setdefault("ALGAN_PROFILE_RUNS", "2")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Benchmarks must never be measured inside a warm daemon: it keeps adaptive
# renderer state (the memory model's batch-size fit) across runs, so one
# benchmark would be timed against whatever ran before it.
os.environ.setdefault("ALGAN_USE_DAEMON", "0")

from algan import (  # noqa: E402
    BLUE,
    GREEN,
    LEFT,
    RIGHT,
    UP,
    Off,
    RenderSettings,
    Sphere,
)
from algan.rendering.raytracing import settings as rt_settings  # noqa: E402
from algan.rendering.raytracing.raster_taichi import (  # noqa: E402
    _AA_NUM_SAMPLES,
)
from algan.utils.profiling_utils import profile_scene  # noqa: E402


def spheres():
    # Dense flat-triangle meshes with long silhouettes and a moving one, so the
    # candidate/fragment counts are representative rather than static-friendly.
    with Off():
        a = Sphere().scale(1.3).move(LEFT * 1.2).set_color(BLUE)
        a.spawn()
        b = Sphere().scale(0.9).move(RIGHT * 1.2 + UP * 0.4).set_color(GREEN)
        b.spawn()
    a.move(RIGHT * 0.6)


if __name__ == "__main__":
    rt_settings.set_analytic_aa(True, bezier=True, triangles=True)
    print(f"analytic coverage samples per pixel: {_AA_NUM_SAMPLES}")
    profile_scene(
        spheres,
        RenderSettings((1280, 720), 30, anti_alias_level=1),
        f"aa_samples_{_AA_NUM_SAMPLES}",
    )
