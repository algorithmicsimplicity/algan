"""A/B the hybrid-raster camera-plane clip's cost.

The clip is pure host-side overhead on a scene with no straddlers, and a large
saving on one full of them. This alternates clip-off/clip-on renders in one
process (wall-clock across processes swings with thermal throttling) and reports
both regimes: a turntable that never straddles, and the orbit that motivated the
clip.

    .venv/Scripts/python.exe benchmarks/_raster_straddle_clip_ab.py [repeats]
"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("ALGAN_PREFETCH_BATCHES", "0")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from algan import *  # noqa: F401,F403,E402
from algan.rendering.raytracing import settings as rt_settings  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "_orbit_out")
os.makedirs(OUT_DIR, exist_ok=True)
REPEATS = int(sys.argv[1]) if len(sys.argv) > 1 else 3


def run(mode, clip):
    SceneManager.reset()
    rt_settings.set_raster_straddle_clip(clip)
    SETTINGS.video.set(MD)
    with Off():
        Group(
            [
                Cube(side_length=0.4, color=BLUE).move(
                    RIGHT * 0.8 * i + UP * 0.8 * j + OUT * 0.8 * k
                )
                for i in range(-2, 3)
                for j in range(-2, 3)
                for k in range(-2, 3)
            ]
        ).spawn()
    with Seq(run_time=2, rate_func=rate_funcs.identity):
        camera = Scene.get_camera()
        if mode == "rotate":
            camera.rotate(360, UP, about_point=ORIGIN)
        else:
            camera.orbit(360, UP, about_point=ORIGIN)
    start = time.perf_counter()
    Scene.save_video(os.path.join(OUT_DIR, f"ab_{mode}_{int(clip)}"), reset=True)
    return time.perf_counter() - start


if __name__ == "__main__":
    try:
        for mode in ("rotate", "orbit"):
            run(mode, True)  # warm kernels/caches for this scene
            times = {False: [], True: []}
            for _ in range(REPEATS):
                for clip in (False, True):
                    times[clip].append(run(mode, clip))
            off = min(times[False])
            on = min(times[True])
            print(
                f"{mode:7s} clip off {off:6.2f}s  on {on:6.2f}s  ({on / off:.2f}x)",
                flush=True,
            )
    finally:
        rt_settings.set_raster_straddle_clip(True)
