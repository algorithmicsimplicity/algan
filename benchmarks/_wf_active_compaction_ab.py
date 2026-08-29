"""A/B and parity check for previous-active-set wavefront compaction.

The baseline scans the entire tile-sized ray pool after every shade pass.  The
optimized non-splitting path filters only the prior active indexes; refraction
and custom-scatter paths deliberately retain the full scan because they can
activate spare ray slots.

The first pair warms the kernel/GPU and is discarded.  Remaining pairs
alternate baseline/optimized in one process.

    .venv/Scripts/python.exe benchmarks/_wf_active_compaction_ab.py [reps]
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
    RED,
    RIGHT,
    UP,
    MeshLambertMaterial,
    SceneManager,
    Sphere,
    Sync,
)
from algan.rendering.raytracing import (  # noqa: E402
    RayTracedTrianglePrimitive,
    set_fragment_shading,
    set_samples_per_pixel,
    set_shadows,
)
from algan.rendering.taichi_runtime import _sync_devices  # noqa: E402
from algan.settings.render_settings import RenderSettings  # noqa: E402
from algan.settings.renderer_settings import RENDERER_SETTINGS  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "_tc_out")
os.makedirs(OUT_DIR, exist_ok=True)
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
BENCH_SETTINGS = RenderSettings(
    (640, 360), 1, super_sampling_anti_aliasing=1, fxaa=False
)


def build():
    """Overlapping translucent spheres require several wavefront iterations;
    the active population falls on every pass.
    """
    with Sync():
        for i in range(30):
            x = (i % 6 - 2.5) * 1.0
            y = (i // 6 - 2.0) * 0.78
            z = (i % 5 - 2.0) * 0.38
            color = (BLUE, RED, GREEN)[i % 3]
            (
                Sphere(grid_height=10, grid_width=10)
                .scale(0.72)
                .move(RIGHT * x + UP * y + IN * z)
                .set_material(MeshLambertMaterial(color=color, opacity=0.28))
                .spawn()
            )


_wf_times = []
_orig_wf = tracer_mod.raytrace_render_wavefront


def _timed_wf(*args, **kwargs):
    _sync_devices()
    t0 = time.perf_counter()
    result = _orig_wf(*args, **kwargs)
    _sync_devices()
    _wf_times.append(time.perf_counter() - t0)
    return result


tracer_mod.raytrace_render_wavefront = _timed_wf


def render_once(optimized, tag):
    SceneManager.reset()
    SceneManager.instance().set_render_settings(BENCH_SETTINGS)
    RENDERER_SETTINGS.triangle_primitive = RayTracedTrianglePrimitive
    set_samples_per_pixel(1)
    set_fragment_shading(True)
    set_shadows(False)
    rt_settings.wf_compact_active_only = bool(optimized)
    build()
    scene = SceneManager.instance()
    _wf_times.clear()
    path = os.path.join(OUT_DIR, f"wfcompact_{tag}.png")
    t0 = time.perf_counter()
    scene.save_frame(path)
    return sum(_wf_times), time.perf_counter() - t0, path


def parity():
    _, _, base_path = render_once(False, "identity_base")
    _, _, opt_path = render_once(True, "identity_opt")
    base = cv2.imread(base_path, cv2.IMREAD_UNCHANGED).astype(np.int32)
    opt = cv2.imread(opt_path, cv2.IMREAD_UNCHANGED).astype(np.int32)
    diff = np.abs(base - opt)
    return bool(np.array_equal(base, opt)), int(diff.max())


def bench():
    render_once(False, "warm_base")
    render_once(True, "warm_opt")
    base_wf, opt_wf, base_total, opt_total = [], [], [], []
    for rep in range(REPS):
        wf, total, _ = render_once(False, f"base_{rep}")
        base_wf.append(wf)
        base_total.append(total)
        wf, total, _ = render_once(True, f"opt_{rep}")
        opt_wf.append(wf)
        opt_total.append(total)
    bw, ow = statistics.median(base_wf), statistics.median(opt_wf)
    bt, ot = statistics.median(base_total), statistics.median(opt_total)
    print(
        f"wavefront base {bw * 1e3:8.1f} ms  optimized {ow * 1e3:8.1f} ms  "
        f"({bw / ow:5.3f}x); end-to-end base {bt:6.2f}s optimized {ot:6.2f}s",
        flush=True,
    )


def main():
    identical, max_diff = parity()
    print(
        "byte parity: "
        + ("IDENTICAL" if identical else f"DIFFER (max |d|={max_diff})"),
        flush=True,
    )
    if not identical:
        raise SystemExit(1)
    bench()


if __name__ == "__main__":
    with torch.inference_mode():
        main()
