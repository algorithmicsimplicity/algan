"""In-process parity + A/B for fused primary-ray generation
(settings.WF_GEN_FUSED).

The classic wavefront opens every tile with a standalone
``wavefront_generate_rays`` pass (~104 B/ray of initial state written, read
straight back by the first traverse/shade). Fused generation computes the
rays inside the tile's first traverse (persisting only ro/rd) and treats the
initial state as constants in the first shade -- same math, same order, so it
must be byte-identical.

Parity scenes cover the state paths the fusion touches:
  * translucent bezier + triangle mix  -> multi-surface peel, K-buffer refill
    (survivor write-back feeds classic iterations >= 1), background escape
    (num_hits == 0 first-iteration branch);
  * mirrors (reflectivity)             -> bounced rays surviving into later
    iterations with mutated ro/rd/weight/bounces_left;
  * ray-traced shadows                 -> per-fragment lighting + shadow rays
    inside the first-iteration shade.

Timing: HD frames on a merge-cached scene, ABBA ordering (thermal drift on
this GTX 1050 is monotonic; ABBA cancels it).

    .venv/Scripts/python.exe benchmarks/_wf_gen_fused_ab.py [reps]
"""

from __future__ import annotations

import gc
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
    HD,
    IN,
    ORANGE,
    PURPLE,
    RED,
    RIGHT,
    TEAL,
    UP,
    WHITE,
    YELLOW,
    Off,
    SceneManager,
    Sphere,
    Square,
)
from algan.rendering.raytracing import (  # noqa: E402
    set_ray_traced_shadows,
    set_reflectivity,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), "_tc_out")
os.makedirs(OUT_DIR, exist_ok=True)

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 4
MODE = sys.argv[2] if len(sys.argv) > 2 else "all"  # parity | timing | all


def _release_gpu():
    """Successive SceneManager.reset() cycles in one process accumulate GPU
    memory (each new render pool is sized from *free* VRAM, so leftovers from
    earlier scenes shrink later pools until an HD frame no longer fits).
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


_COLORS = [BLUE, RED, GREEN, YELLOW, WHITE, ORANGE, PURPLE, TEAL]

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


def build(mirrors=False):
    rng = np.random.default_rng(20260713)
    with Off():
        for i in range(10):
            x = float(rng.uniform(-3.0, 3.0))
            y = float(rng.uniform(-1.8, 1.8))
            z = float(rng.uniform(-1.5, 1.5))
            col = _COLORS[i % len(_COLORS)].set_opacity(0.55)
            if i % 3 == 0:
                m = Sphere(grid_height=10, grid_width=10).scale(0.8)
                if mirrors:
                    set_reflectivity(m, 0.5)
            else:
                m = Square(color=col).scale(1.1)
            m.move(RIGHT * x + UP * y + IN * z)
            m.spawn()


def render_frame(tag, fused, mirrors=False, shadows=False):
    SceneManager.reset()
    _release_gpu()
    set_ray_traced_shadows(bool(shadows))
    rt_settings.WF_GEN_FUSED = bool(fused)
    build(mirrors=mirrors)
    scene = SceneManager.instance()
    path = os.path.join(OUT_DIR, f"gen_fused_{tag}.png")
    scene.save_frame(path)
    return path


def main():
    # ---- parity: fused off vs on across the touched state paths ----
    if MODE in ("parity", "all"):
        all_ok = True
        for name, kw in (
            ("peel", {}),
            ("mirrors", {"mirrors": True}),
            ("shadows", {"shadows": True}),
        ):
            p_off = render_frame(f"{name}_off", False, **kw)
            p_on = render_frame(f"{name}_on", True, **kw)
            a = cv2.imread(p_off, cv2.IMREAD_UNCHANGED).astype(np.int32)
            b = cv2.imread(p_on, cv2.IMREAD_UNCHANGED).astype(np.int32)
            same = a.shape == b.shape and np.array_equal(a, b)
            all_ok &= same
            mx = -1 if a.shape != b.shape else int(np.abs(a - b).max())
            print(
                f"  {name:<10} classic vs fused: "
                f"{'IDENTICAL' if same else f'DIFFER (max |d|={mx})'}",
                flush=True,
            )
        print(f"gen-fused parity: {'PASS' if all_ok else 'FAIL'}", flush=True)
        set_ray_traced_shadows(False)
        if not all_ok:
            return
    if MODE == "parity":
        return

    # ---- timing: ABBA on HD frames, merge-cached scene ----
    SceneManager.reset()
    _release_gpu()
    rt_settings.WF_GEN_FUSED = False
    build()
    scene = SceneManager.instance()
    scene.set_render_settings(HD)

    def frame(fused):
        rt_settings.WF_GEN_FUSED = bool(fused)
        _wf_times.clear()
        scene.save_frame(os.path.join(OUT_DIR, "gen_fused_timing.png"))
        _release_gpu()
        return sum(_wf_times)

    frame(False)  # warm both kernel instantiations
    frame(True)
    off_t, on_t = [], []
    for rep in range(max(4, REPS * 2)):
        order = (False, True) if rep % 2 == 0 else (True, False)
        for fused in order:
            (on_t if fused else off_t).append(frame(fused))
    rt_settings.WF_GEN_FUSED = True
    ow, nw = statistics.median(off_t), statistics.median(on_t)
    print(f"raw classic: {[f'{t:.3f}' for t in off_t]}", flush=True)
    print(f"raw fused:   {[f'{t:.3f}' for t in on_t]}", flush=True)
    print(
        f"wavefront/frame: classic {ow:7.3f}s   fused {nw:7.3f}s   "
        f"(fused is {nw / ow:5.3f}x classic)",
        flush=True,
    )


if __name__ == "__main__":
    with torch.inference_mode():
        main()
