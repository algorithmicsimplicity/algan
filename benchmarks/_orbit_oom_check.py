"""Reproduce the camera.orbit() OOM and confirm the raster fallback fixes it.

``camera.rotate(deg, UP, about=ORIGIN)`` keeps the scene in frame; the
same circle travelled with ``camera.orbit`` leaves the camera pointing in its
original direction, so the geometry sweeps out of frame -- and, crucially, past
the camera plane. Primitives that straddle the camera plane get a full-window
conservative candidate bbox in the hybrid raster front-end, which is where the
memory goes.

    .venv/Scripts/python.exe benchmarks/_orbit_oom_check.py [rotate|orbit]
"""

from __future__ import annotations

import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from algan import *  # noqa: F401,F403,E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "_orbit_out")
os.makedirs(OUT_DIR, exist_ok=True)

mode = sys.argv[1] if len(sys.argv) > 1 else "orbit"

SETTINGS.video.set(LD)

with Off():
    Group(
        [Cube(size=0.8, color=BLUE).move(RIGHT * 1.6 * i) for i in (-1, 0, 1)]
    ).spawn()

with Seq(run_time=4, rate_func=rate_funcs.identity):
    if mode == "rotate":
        Scene.get_camera().rotate(360, UP, about=ORIGIN)
    else:
        Scene.get_camera().orbit(360, UP, about=ORIGIN)

try:
    res = Scene.save_video(os.path.join(OUT_DIR, f"camera_{mode}"))
    print(f"OK {mode}: {res.output_path} ({res.duration_seconds:.1f}s)")
except Exception:
    traceback.print_exc()
    print(f"FAILED {mode}")
    sys.exit(1)
