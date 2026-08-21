"""Smoke test: the Monte-Carlo path tracer against the packed node layout.

``path_trace_scene_stbvh`` is only reached with ``set_samples_per_pixel > 1``,
so the parity/perf gates for node-layout changes never touch it. This renders
one small MC frame through the real pipeline purely to prove the kernel still
compiles and binds the packed ``[num_nodes, 2, 4]`` node arrays (expect a
multi-minute cold compile on first run).

    .venv/Scripts/python.exe benchmarks/_node_pack_mc_smoke.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from algan import (  # noqa: E402
    BLUE,
    GREEN,
    LEFT,
    RIGHT,
    MeshLambertMaterial,
    SceneManager,
    Sphere,
    Square,
    Sync,
)
from algan.rendering.raytracing.settings import set_samples_per_pixel  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "_tc_out")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    SceneManager.reset()
    set_samples_per_pixel(2)
    try:
        with Sync():
            Sphere().scale(0.9).move(LEFT * 1.2).set_material(
                MeshLambertMaterial(color=BLUE)
            ).spawn()
            Square(color=GREEN).scale(0.8).move(RIGHT * 1.2).spawn()
        scene = SceneManager.instance()
        frames = scene.save_frame(os.path.join(OUT_DIR, "node_pack_mc.png"))
        frame = frames[-1]
        nonzero = float((frame > 0).float().mean())
        print(f"MC frame rendered: shape={tuple(frame.shape)} nonzero={nonzero:.3f}")
        print("MC_SMOKE_OK", nonzero > 0.01)
    finally:
        set_samples_per_pixel(1)


if __name__ == "__main__":
    main()
