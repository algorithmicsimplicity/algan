"""Check the logical-PN screen guard costs no visible tessellation detail.

``_required_subdivision_levels`` now ignores flatness error that happens
outside a guard box around the frame. That is only safe if the box is wide
enough that in-frame geometry is unaffected. This renders a turntable (camera
``rotate``, so the subject stays framed the whole way round) twice -- once with
the shipped guard, once with an effectively infinite one that restores the old
"measure error everywhere" behaviour -- and requires the two videos to match
pixel-exactly.

It also reports the subdivision levels each run chose, so a difference shows up
as a level change rather than only as pixels.

    .venv/Scripts/python.exe benchmarks/_orbit_lod_guard_check.py
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("ALGAN_PREFETCH_BATCHES", "0")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from algan import *  # noqa: F401,F403,E402
from algan.rendering.raytracing.primitives import (  # noqa: E402
    LogicalPNTrianglePrimitive,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), "_orbit_out")
os.makedirs(OUT_DIR, exist_ok=True)

_orig_levels = LogicalPNTrianglePrimitive._required_subdivision_levels
_levels = []


def _record(self, *a, **k):
    out = _orig_levels(self, *a, **k)
    _levels.extend(out.tolist())
    return out


LogicalPNTrianglePrimitive._required_subdivision_levels = _record


def render(guard, tag):
    SceneManager.reset()
    LogicalPNTrianglePrimitive.screen_guard_factor = guard
    _levels.clear()
    SETTINGS.video.set(LD)
    # A tight render_tolerance and a subject large enough to reach the frame
    # edges: the level must land above 0, and geometry must sit right up
    # against the guard box, or the comparison proves nothing.
    with Off():
        Group(
            [
                Sphere(color=BLUE, render_tolerance=0.0008, max_grid_resolution=24)
                .scale(1.7)
                .move(RIGHT * 2.2 * i)
                for i in (-1, 0, 1)
            ]
        ).spawn()
    with Seq(duration=2, rate_func=rate_funcs.identity):
        Scene.get_camera().rotate(360, UP, about=ORIGIN)
    path = os.path.join(OUT_DIR, f"guard_{tag}")
    Scene.save_video(path, reset=True)
    print(f"  {tag}: guard={guard} levels={sorted(set(_levels))}", flush=True)
    return path + ".mp4"


def frames(path):
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f.astype(np.int32))
    cap.release()
    return out


if __name__ == "__main__":
    try:
        shipped = render(LogicalPNTrianglePrimitive.screen_guard_factor, "shipped")
        wide = render(1e9, "wide")
    finally:
        LogicalPNTrianglePrimitive.screen_guard_factor = 1.5
    a, b = frames(shipped), frames(wide)
    ok = (
        len(a) == len(b)
        and a
        and all(int(np.abs(x - y).max()) == 0 for x, y in zip(a, b))
    )
    worst = max((int(np.abs(x - y).max()) for x, y in zip(a, b)), default=-1)
    print(f"frames={len(a)}/{len(b)} max|d|={worst} {'OK' if ok else 'MISMATCH'}")
    print("LOD_GUARD_OK:", bool(ok))
    sys.exit(0 if ok else 1)
