"""Measure hybrid-raster candidate-pair / fragment counts for rotate vs orbit.

Monkeypatches the raster front-end's candidate emission to tally how many
(primitive, frame, bbox, chunk) rows and how many emitted fragments each render
produces. The OOM hypothesis is that an un-turned orbiting camera pushes
geometry across the camera plane, where the conservative bbox becomes the whole
window.

    .venv/Scripts/python.exe benchmarks/_orbit_pairs_probe.py [rotate|orbit] [LD|MD|HD] [scene]
"""

from __future__ import annotations

import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from algan import *  # noqa: F401,F403,E402
from algan.rendering.raytracing import raster_pipeline as rp  # noqa: E402

mode = sys.argv[1] if len(sys.argv) > 1 else "orbit"
quality = sys.argv[2] if len(sys.argv) > 2 else "LD"
scene_name = sys.argv[3] if len(sys.argv) > 3 else "cubes"

stats = {"rows": 0, "calls": 0, "max_rows": 0, "frags": 0, "max_frags": 0}

_orig_flat = rp._class_pairs_flat


def _probe_flat(mask, x0, x1, y0, y1, f_abs, device):
    out = _orig_flat(mask, x0, x1, y0, y1, f_abs, device)
    if out is not None:
        n = int(out.shape[0])
        stats["rows"] += n
        stats["calls"] += 1
        stats["max_rows"] = max(stats["max_rows"], n)
    return out


rp._class_pairs_flat = _probe_flat

_orig_prepare = rp.prepare_sparse_raster_coverage


def _probe_prepare(*a, **k):
    out = _orig_prepare(*a, **k)
    if out is not None:
        stats["frags"] += out["num_fragments"]
        stats["max_frags"] = max(stats["max_frags"], out["num_fragments"])
    return out


rp.prepare_sparse_raster_coverage = _probe_prepare

SETTINGS.video.set({"LD": LD, "MD": MD, "HD": HD}[quality])

with Off():
    if scene_name == "cubes":
        Group(
            [
                Cube(side_length=0.8, color=BLUE).move(RIGHT * 1.6 * i)
                for i in (-1, 0, 1)
            ]
        ).spawn()
    elif scene_name == "spheres":
        Group([Sphere(color=BLUE).move(RIGHT * 1.6 * i) for i in (-1, 0, 1)]).spawn()
    elif scene_name == "grid":
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
    elif scene_name == "text":
        Group([Text("Orbiting camera").move(UP * 1.2 * i) for i in (-1, 0, 1)]).spawn()

with Seq(run_time=4, rate_func=rate_funcs.identity):
    if mode == "rotate":
        Scene.get_camera().rotate(360, UP, about_point=ORIGIN)
    else:
        Scene.get_camera().orbit(360, UP, about_point=ORIGIN)

out_dir = os.path.join(os.path.dirname(__file__), "_orbit_out")
os.makedirs(out_dir, exist_ok=True)
status = "OK"
try:
    Scene.save_video(os.path.join(out_dir, f"probe_{mode}_{quality}_{scene_name}"))
except Exception:
    traceback.print_exc()
    status = "FAILED"
print(
    f"[{mode} {quality} {scene_name}] {status} pair rows total={stats['rows']:,} "
    f"calls={stats['calls']} max_call={stats['max_rows']:,} "
    f"(={stats['max_rows'] * 32 / 1e6:.1f} MB) "
    f"frags total={stats['frags']:,} max_batch={stats['max_frags']:,} "
    f"(={stats['max_frags'] * 29 / 1e6:.1f} MB)"
)
