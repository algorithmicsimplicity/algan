"""Byte parity: the hybrid-raster camera-plane clip (ALGAN_RASTER_STRADDLE_CLIP).

The clip shrinks a straddling primitive's candidate bbox from "the whole
window" to the real screen extent of the part a primary ray can reach. Candidate
pixels are exact-tested either way, so a tighter bbox may only skip pixels that
would have missed -- the rendered frames must be identical. Each scene is
rendered twice, clip off (the old full-window straddler bbox) and on, and the
decoded videos must match pixel-exactly.

The scenes are built to straddle: geometry sweeping past the camera, the camera
flying through a mesh, an orbiting camera, and text passing the camera plane.

Note this cannot be checked against the classic wavefront instead: that route
renders at ``anti_alias_level`` while the analytic raster route renders at 1, so
the two differ everywhere for reasons unrelated to the clip.

    .venv/Scripts/python.exe benchmarks/_raster_straddle_clip_parity.py [configs...]
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("ALGAN_PREFETCH_BATCHES", "0")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from algan import *  # noqa: F401,F403,E402
from algan.rendering.raytracing import settings as rt_settings  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "_orbit_out")
os.makedirs(OUT_DIR, exist_ok=True)


def build(cfg):
    if cfg == "sweep_past":
        # Cubes swing sideways past the camera: most frames have geometry
        # straddling the camera plane far off to one side.
        with Off():
            g = Group(
                [
                    Cube(side_length=0.8, color=BLUE).move(RIGHT * 1.6 * i)
                    for i in (-1, 0, 1)
                ]
            ).spawn()
        with Seq(run_time=1, rate_func=rate_funcs.identity):
            g.move(RIGHT * 14)
    elif cfg == "fly_through":
        # The camera passes through the mesh: the "unbounded" fallback case,
        # where the clip must give up and keep the whole window.
        with Off():
            Group(
                [
                    Cube(side_length=1.2, color=RED).move(
                        RIGHT * 1.5 * i + UP * 1.5 * j
                    )
                    for i in (-1, 0, 1)
                    for j in (-1, 0, 1)
                ]
            ).spawn()
        with Seq(run_time=1, rate_func=rate_funcs.identity):
            Scene.get_camera().move(IN * 12)
    elif cfg == "orbit":
        with Off():
            Group(
                [
                    Cube(side_length=0.8, color=BLUE).move(RIGHT * 1.6 * i)
                    for i in (-1, 0, 1)
                ]
            ).spawn()
        with Seq(run_time=1, rate_func=rate_funcs.identity):
            Scene.get_camera().orbit(360, UP, about_point=ORIGIN)
    elif cfg == "text_past":
        with Off():
            t = Group([Text("straddle").move(UP * 1.2 * i) for i in (-1, 0, 1)]).spawn()
        with Seq(run_time=1, rate_func=rate_funcs.identity):
            t.move(RIGHT * 12)
    else:
        raise SystemExit(f"unknown config {cfg}")


def render(cfg, clip, tag):
    SceneManager.reset()
    rt_settings.set_raster_straddle_clip(clip)
    SETTINGS.video.set(LD)
    build(cfg)
    path = os.path.join(OUT_DIR, f"straddleclip_{cfg}_{tag}")
    Scene.save_video(path, reset=True)
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


def main():
    configs = sys.argv[1:] or ["sweep_past", "fly_through", "orbit", "text_past"]
    all_ok = True
    try:
        for cfg in configs:
            ref = render(cfg, False, "off")
            got = render(cfg, True, "on")
            fa, fb = frames(ref), frames(got)
            if len(fa) != len(fb) or not fa:
                print(f"[{cfg:12s}] FAIL: frames {len(fa)} vs {len(fb)}")
                all_ok = False
                continue
            worst = max(int(np.abs(a - b).max()) for a, b in zip(fa, fb))
            ok = worst == 0
            all_ok = all_ok and ok
            print(
                f"[{cfg:12s}] frames={len(fa):3d} max|d|={worst:3d}  "
                f"{'OK' if ok else 'MISMATCH'}",
                flush=True,
            )
    finally:
        rt_settings.set_raster_straddle_clip(True)
    print("\nSTRADDLE_CLIP_PARITY_OK:", all_ok)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
