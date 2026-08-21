"""Byte-identity gate for STBVH node-layout changes.

Renders two animated multi-frame scenes to video (PREVIEW settings) through
the default deterministic wavefront pipeline -- one flat-triangle scene with
materials, bezier squares and ray-traced shadows, and one PN-patch variant --
then decodes every frame with cv2 and hashes the raw pixel bytes. Run with
``--save`` before a kernel/layout change to record the reference manifest;
run without arguments after the change to compare against it. Any hash
mismatch means the change is not output-preserving.

    .venv/Scripts/python.exe benchmarks/_node_pack_parity.py --save   # before
    .venv/Scripts/python.exe benchmarks/_node_pack_parity.py          # after
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from algan import (  # noqa: E402
    BLUE,
    DOWN,
    GREEN,
    LEFT,
    OUT,
    PREVIEW,
    RED,
    RIGHT,
    UP,
    WHITE,
    YELLOW,
    Cylinder,
    MeshBasicMaterial,
    MeshLambertMaterial,
    MeshPhongMaterial,
    MeshStandardMaterial,
    SceneManager,
    Sphere,
    Square,
    Sync,
)
from algan.rendering.raytracing import (  # noqa: E402
    set_fragment_shading,
    set_ray_traced_shadows,
    set_reflectivity,
)
from algan.utils.algan_utils import render_to_file  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "_tc_out")
os.makedirs(OUT_DIR, exist_ok=True)
MANIFEST = os.path.join(OUT_DIR, "node_pack_parity.json")

_MATERIALS = [
    lambda: MeshLambertMaterial(color=BLUE),
    lambda: MeshPhongMaterial(color=RED, specular=0xFFFFFF, shininess=50),
    lambda: MeshStandardMaterial(color=GREEN, metalness=0.8, roughness=0.3),
    lambda: MeshBasicMaterial(color=YELLOW),
]


def build_and_animate():
    """Sphere grid (all built-in materials) + reflective cylinder + bezier
    squares, all moving, so the STBVH gets real temporal segmentation.
    """
    mobs = []
    with Sync():
        for i in range(9):
            row, col = divmod(i, 3)
            m = (
                Sphere()
                .scale(0.55)
                .move(RIGHT * (col - 1) * 1.8 + UP * (row - 1) * 1.7)
                .set_material(_MATERIALS[i % 4]())
            )
            m.spawn()
            mobs.append(m)
        mirror = (
            Cylinder(radius=0.35, height=2.2)
            .move(DOWN * 2.6 + LEFT * 3)
            .set_material(MeshLambertMaterial(color=WHITE))
        )
        set_reflectivity(mirror, 0.6)
        mirror.spawn()
        sq1 = Square(color=GREEN).scale(0.6).move(UP * 2.2 + OUT * 0.5)
        sq1.spawn()
        sq2 = Square(color=RED).scale(0.6).move(DOWN * 2.2 + OUT * 0.5)
        sq2.spawn()
        mobs += [sq1, sq2]
    with Sync():  # ~1s of simultaneous motion -> a real multi-frame batch
        for i, m in enumerate(mobs):
            m.move((RIGHT if i % 2 else LEFT) * 0.5 + UP * 0.2)


def video_frame_hash(path):
    """SHA-256 over the decoded raw frames of a video (order-sensitive)."""
    cap = cv2.VideoCapture(path)
    h = hashlib.sha256()
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h.update(frame.tobytes())
        n += 1
    cap.release()
    return h.hexdigest(), n


def render_config(tag, pn):
    SceneManager.reset()
    if pn:
        from algan.rendering.raytracing.primitives import RayTracedPNTrianglePrimitive
        from algan.settings.renderer_settings import RENDERER_SETTINGS

        prev = RENDERER_SETTINGS.triangle_primitive
        RENDERER_SETTINGS.triangle_primitive = RayTracedPNTrianglePrimitive
    set_fragment_shading(True)
    set_ray_traced_shadows(True)
    build_and_animate()
    name = f"node_pack_{tag}"
    render_to_file(file_name=name, output_dir=OUT_DIR, render_settings=PREVIEW)
    if pn:
        RENDERER_SETTINGS.triangle_primitive = prev
    path = os.path.join(OUT_DIR, name + ".mp4")
    digest, n = video_frame_hash(path)
    if n == 0:
        print(f"[{tag:4s}] no decodable frames at {path}; aborting")
        sys.exit(2)
    print(f"[{tag:4s}] {n} frames  sha256={digest}", flush=True)
    return digest, n


def main():
    save = "--save" in sys.argv
    results = {}
    for tag, pn in (("flat", False), ("pn", True)):
        digest, n = render_config(tag, pn)
        results[tag] = {"sha256": digest, "frames": n}
    if save:
        with open(MANIFEST, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"saved reference manifest -> {MANIFEST}")
        return
    if not os.path.exists(MANIFEST):
        print("no reference manifest; run with --save first")
        sys.exit(2)
    with open(MANIFEST) as fh:
        ref = json.load(fh)
    ok = True
    for tag, got in results.items():
        want = ref.get(tag)
        match = want == got
        ok = ok and match
        print(
            f"[{tag:4s}] {'BYTE-IDENTICAL' if match else 'MISMATCH'}"
            + ("" if match else f"  want={want} got={got}")
        )
    print("PARITY_OK:", ok)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
