"""In-process A/B + parity for adaptive wavefront tile sizing
(settings.wavefront_tile_auto).

Tiles partition pixels and every per-pixel computation is independent of its
tile, so ANY tile size must produce byte-identical frames. Part 1 verifies
that directly: the same frame is rendered with the static default (one tile),
a forced-small static tile (many tiles), natural auto sizing and forced-small
auto sizing, and all four PNGs must match bytewise.

Part 2 measures what auto sizing is for: fewer, bigger tiles mean fewer
traverse/shade/generate launches, and each launch pays a fixed host-side cost
(Taichi marshals the ~60 ndarray args per launch; ~7-9 ms/launch measured on
the GTX 1050). Renders a short HD video alternating auto off/on to cancel
thermal drift, timing the wavefront render stage and end-to-end.

    .venv/Scripts/python.exe benchmarks/_wf_tile_auto_ab.py [reps]
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
    HD,
    IN,
    LEFT,
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
    Sync,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), "_tc_out")
os.makedirs(OUT_DIR, exist_ok=True)

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
_COLORS = [BLUE, RED, GREEN, YELLOW, WHITE, ORANGE, PURPLE, TEAL]

_DEF_TILE = rt_settings.wavefront_tile_rays
_DEF_MIN = rt_settings.wavefront_tile_min
_DEF_MAX = rt_settings.wavefront_tile_max


def _set_tiling(auto, tile=_DEF_TILE, tmin=_DEF_MIN, tmax=_DEF_MAX):
    rt_settings.wavefront_tile_auto = bool(auto)
    rt_settings.wavefront_tile_rays = int(tile)
    rt_settings.wavefront_tile_min = int(tmin)
    rt_settings.wavefront_tile_max = int(tmax)


# Record every tile size the orchestrators actually use.
_tile_log = []
_orig_auto = tracer_mod._auto_primary_per_tile


def _auto_spy(memory, split_k, static_primary, *a, **k):
    v = _orig_auto(memory, split_k, static_primary, *a, **k)
    _tile_log.append(v)
    return v


tracer_mod._auto_primary_per_tile = _auto_spy

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


def build(animate=False):
    """Overlapping semi-transparent bezier shapes over triangle meshes, so
    rays peel multiple surfaces (several wavefront iterations) across both
    geometry types -- the tile loop's worst case.
    """
    rng = np.random.default_rng(20260713)
    mobs = []
    with Off():
        for i in range(10):
            x = float(rng.uniform(-3.0, 3.0))
            y = float(rng.uniform(-1.8, 1.8))
            z = float(rng.uniform(-1.5, 1.5))
            col = _COLORS[i % len(_COLORS)].set_opacity(0.55)
            if i % 3 == 0:
                m = Sphere(grid_height=10, grid_width=10).scale(0.8)
            else:
                m = Square(color=col).scale(1.1)
            m.move(RIGHT * x + UP * y + IN * z)
            m.spawn()
            mobs.append(m)
    if animate:
        with Sync(run_time=0.5):
            for m in mobs:
                m.move(LEFT * 0.8)


def render_frame(tag):
    SceneManager.reset()
    build(animate=False)
    scene = SceneManager.instance()
    path = os.path.join(OUT_DIR, f"tile_auto_{tag}.png")
    _tile_log.clear()
    scene.save_frame(path)
    return path, list(_tile_log)


def timed_frame(scene, auto):
    """One HD frame on the already-built (merge-cached) scene: isolates the
    wavefront render stage from prep/merge/encode.
    """
    _set_tiling(auto)
    _wf_times.clear()
    scene.save_frame(os.path.join(OUT_DIR, "tile_auto_timing.png"))
    return sum(_wf_times)


def main():
    # ---- Part 1: tile-size invariance (byte parity) ----
    cases = []
    _set_tiling(False)
    cases.append(("static_default", *render_frame("static_default")))
    _set_tiling(False, tile=1 << 16)
    cases.append(("static_64k", *render_frame("static_64k")))
    _set_tiling(True)
    cases.append(("auto", *render_frame("auto")))
    _set_tiling(True, tmin=1 << 16, tmax=1 << 16)
    cases.append(("auto_64k", *render_frame("auto_64k")))
    _set_tiling(False)

    ref = cv2.imread(cases[0][1], cv2.IMREAD_UNCHANGED)
    all_ok = True
    for name, path, tiles in cases:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        same = ref.shape == img.shape and np.array_equal(ref, img)
        all_ok &= same
        mx = (
            0
            if same
            else int(np.abs(ref.astype(np.int32) - img.astype(np.int32)).max())
        )
        print(
            f"  {name:<16} tile sizes {sorted(set(tiles))} -> "
            f"{'IDENTICAL' if same else f'DIFFER (max |d|={mx})'}",
            flush=True,
        )
    print(f"tile-size parity: {'PASS' if all_ok else 'FAIL'}", flush=True)
    if not all_ok:
        return

    # ---- Part 2: timing. Single HD frames on one merge-cached scene, ABBA
    # ordering so the GTX 1050's monotonic thermal drift cancels instead of
    # biasing whichever mode runs second. ----
    SceneManager.reset()
    build(animate=False)
    scene = SceneManager.instance()
    scene.set_render_settings(HD)
    timed_frame(scene, False)  # warm both paths (+ merge cache)
    timed_frame(scene, True)
    off_wf, on_wf = [], []
    for rep in range(max(4, REPS * 2)):
        order = (False, True) if rep % 2 == 0 else (True, False)
        for auto in order:
            (on_wf if auto else off_wf).append(timed_frame(scene, auto))
    _set_tiling(False)
    ow, nw = statistics.median(off_wf), statistics.median(on_wf)
    print(f"raw static: {[f'{t:.3f}' for t in off_wf]}", flush=True)
    print(f"raw auto:   {[f'{t:.3f}' for t in on_wf]}", flush=True)
    print(
        f"wavefront/frame: static {ow:7.3f}s   auto {nw:7.3f}s   "
        f"(auto is {nw / ow:5.3f}x static)",
        flush=True,
    )


if __name__ == "__main__":
    with torch.inference_mode():
        main()
