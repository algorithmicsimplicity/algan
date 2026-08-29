"""Smoke test: the path tracer against the packed node layout.

The path-traced route is only reached with ``set_samples_per_pixel > 1``, so
the parity/perf gates for node-layout changes never touch it. This renders
one small path-traced frame through the real pipeline purely to prove the
route still compiles and binds the packed ``[num_nodes, 2, 4]`` node arrays.

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
    Scene,
    SceneManager,
    Sphere,
    Square,
    Sync,
)
from algan.rendering.raytracing.settings import set_samples_per_pixel  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "_tc_out")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    import cv2

    SceneManager.reset()
    set_samples_per_pixel(2)
    try:
        with Sync():
            Sphere().scale(0.9).move(LEFT * 1.2).set_material(
                MeshLambertMaterial(color=BLUE)
            ).spawn()
            Square(color=GREEN).scale(0.8).move(RIGHT * 1.2).spawn()
        result = Scene.save_frame(
            os.path.join(OUT_DIR, "node_pack_mc.png"), overwrite=True
        )
        frame = cv2.imread(str(result.output_path))
        nonzero = float((frame > 0).mean())
        print(f"PT frame rendered: shape={frame.shape} nonzero={nonzero:.3f}")
        print("MC_SMOKE_OK", nonzero > 0.01)
    finally:
        set_samples_per_pixel(1)


if __name__ == "__main__":
    main()
