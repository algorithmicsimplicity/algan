"""Trace how high the logical-PN level search climbs during a camera orbit.

Wraps ``subdivision_triangle_uvs`` as seen by the level search so every trial
level is visible, and reports the camera-plane straddle / behind classification
the search uses to decide a frame is unresolvable.

    .venv/Scripts/python.exe benchmarks/_orbit_pn_level_probe.py [rotate|orbit] [LD|MD|HD]
"""

from __future__ import annotations

import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from algan import *  # noqa: F401,F403,E402
from algan.rendering.raytracing import primitives as rprim  # noqa: E402

mode = sys.argv[1] if len(sys.argv) > 1 else "orbit"
quality = sys.argv[2] if len(sys.argv) > 2 else "HD"

_orig_uvs = rprim.subdivision_triangle_uvs
_orig_levels = rprim.LogicalPNTrianglePrimitive._required_subdivision_levels
_state = {"in_search": False}


def _probe_uvs(level, **k):
    if _state["in_search"]:
        print(f"      trial level {level}", flush=True)
    return _orig_uvs(level, **k)


rprim.subdivision_triangle_uvs = _probe_uvs


def _probe_levels(self, control_points, cam_o, sp, sb, screen_height):
    camera_shape = (-1, 1, 1, 3)
    screen_normal = sb[:, 2].view(camera_shape)
    depth = ((control_points - cam_o.view(camera_shape)) * screen_normal).sum(-1)
    wholly_behind = depth.amax(dim=(-1, -2)) < -1e-7
    crosses = ~((depth.amin(-1) > 1e-7) | (depth.amax(-1) < -1e-7)).all(-1)
    print(
        f"    search: frames={control_points.shape[0]} "
        f"patches={control_points.shape[1]} "
        f"behind={wholly_behind.tolist()} crosses={crosses.tolist()}",
        flush=True,
    )
    _state["in_search"] = True
    try:
        out = _orig_levels(self, control_points, cam_o, sp, sb, screen_height)
    finally:
        _state["in_search"] = False
    print(f"    -> levels={out.tolist()}", flush=True)
    return out


rprim.LogicalPNTrianglePrimitive._required_subdivision_levels = _probe_levels

SETTINGS.video.set({"LD": LD, "MD": MD, "HD": HD}[quality])

with Off():
    Group([Sphere(color=BLUE).move(RIGHT * 1.6 * i) for i in (-1, 0, 1)]).spawn()

with Seq(duration=4, easing=easings.identity):
    if mode == "rotate":
        Scene.get_camera().rotate(360, UP, about=ORIGIN)
    else:
        Scene.get_camera().orbit(360, UP, about=ORIGIN)

out_dir = os.path.join(os.path.dirname(__file__), "_orbit_out")
os.makedirs(out_dir, exist_ok=True)
try:
    Scene.save_video(os.path.join(out_dir, f"lv_{mode}_{quality}"))
    print("OK")
except Exception:
    traceback.print_exc()
    print("FAILED")
