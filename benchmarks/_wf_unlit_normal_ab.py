"""In-process A/B for Family A (Stage 1): skip the up-front per-fragment
shading-normal computation for UNLIT hits on the fragment-shading wavefront.

The shade kernel's ``skip_unlit_normal`` template (``ALGAN_WF_SKIP_UNLIT_NORMAL``
/ ``settings.wf_skip_unlit_normal``) is toggled between renders. UNLIT hits pass
their colour through unchanged and never consume the shading normal, so skipping
``_flat_triangle_normal`` for them is byte-identical -- this harness first proves
that (identical PNG), then measures the speed effect.

Scene: a dense cloud of overlapping *transparent* spheres so each primary ray
passes through many surfaces (many shade invocations per ray, where the skip
lives). Two material regimes:
  * ``unlit``  -- every sphere MeshBasicMaterial (pid == UNLIT): the skip fires
    on every hit, so this is the *upper bound* on the benefit.
  * ``lit``    -- every sphere MeshLambertMaterial (never UNLIT): the skip never
    fires, so any A/B delta here is the noise floor / measurement bias.

Alternating the two settings in one process cancels thermal-throttle drift; the
first pair is discarded as compile/clock warm-up (each template value compiles a
distinct kernel the first time).

    .venv/Scripts/python.exe benchmarks/_wf_unlit_normal_ab.py [reps]
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
    set_shadows,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), "_tc_out")
os.makedirs(OUT_DIR, exist_ok=True)

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
_COLORS = [BLUE, RED, GREEN, YELLOW, WHITE, ORANGE, PURPLE, TEAL]


def build(regime):
    """A jittered 3D cloud of overlapping, semi-transparent spheres filling the
    view, so primary rays accumulate through many surfaces.
    """
    rng = np.random.default_rng(1234)
    with Sync():
        n = 45
        for i in range(n):
            x = float(rng.uniform(-3.2, 3.2))
            y = float(rng.uniform(-1.9, 1.9))
            z = float(rng.uniform(-2.0, 2.0))
            col = _COLORS[i % len(_COLORS)]
            # opacity < 1 -> rays pass through, hitting several surfaces
            if regime == "unlit":
                mat = MeshBasicMaterial(color=col, opacity=0.30)
            else:
                mat = MeshLambertMaterial(color=col, opacity=0.30)
            (
                Sphere(grid_height=10, grid_width=10)
                .scale(0.85)
                .move(RIGHT * x + UP * y + IN * z)
                .set_material(mat)
                .spawn()
            )


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


def render_once(skip_on, regime, tag):
    SceneManager.reset()
    set_fragment_shading(True)
    set_shadows(False)
    rt_settings.wf_skip_unlit_normal = bool(skip_on)
    build(regime)
    scene = SceneManager.instance()
    _wf_times.clear()
    path = os.path.join(OUT_DIR, f"unlitnrm_{tag}.png")
    t0 = time.perf_counter()
    scene.save_frame(path)
    return sum(_wf_times), time.perf_counter() - t0, path


def byte_identity(regime):
    _, _, p_off = render_once(False, regime, f"{regime}_id_off")
    _, _, p_on = render_once(True, regime, f"{regime}_id_on")
    a = cv2.imread(p_off, cv2.IMREAD_UNCHANGED).astype(np.int32)
    b = cv2.imread(p_on, cv2.IMREAD_UNCHANGED).astype(np.int32)
    if a.shape != b.shape:
        return False, -1
    return bool(np.array_equal(a, b)), int(np.abs(a - b).max())


def bench(regime):
    # warm-up: compile both template variants + settle clocks
    render_once(False, regime, f"{regime}_warm_off")
    render_once(True, regime, f"{regime}_warm_on")
    off_wf, off_tot, on_wf, on_tot = [], [], [], []
    for _ in range(REPS):
        wf, tot, _ = render_once(False, regime, f"{regime}_off")
        off_wf.append(wf)
        off_tot.append(tot)
        wf, tot, _ = render_once(True, regime, f"{regime}_on")
        on_wf.append(wf)
        on_tot.append(tot)
    ow, nw = statistics.median(off_wf), statistics.median(on_wf)
    ot, nt = statistics.median(off_tot), statistics.median(on_tot)
    print(
        f"[{regime:5s}] wavefront: baseline(off) {ow * 1e3:8.1f} ms   "
        f"skip(on) {nw * 1e3:8.1f} ms   (skip is {ow / nw:5.3f}x baseline)   "
        f"end-to-end: off {ot:6.2f}s on {nt:6.2f}s",
        flush=True,
    )


def main():
    for regime in ("unlit", "lit"):
        ok, mx = byte_identity(regime)
        print(
            f"[{regime:5s}] byte-identity off vs on: "
            f"{'IDENTICAL' if ok else f'DIFFER (max |d|={mx})'}",
            flush=True,
        )
    for regime in ("unlit", "lit"):
        bench(regime)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
