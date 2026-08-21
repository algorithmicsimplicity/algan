"""Parity check: general wavefront tracer vs the megakernel (render_scene_stbvh).

Builds a mixed scene (Sphere, Cylinder, a 2-D shape) so the general path
exercises every geometry BVH present, then renders the same frame with the
megakernel (set_wavefront(False)) and the wavefront (set_wavefront(True)) for
three configs:

    vertex   -- default Gouraud (frag_shading off, shadows off)
    frag     -- deterministic per-fragment shading
    shadow   -- per-fragment shading + binary hard shadows

For each config it reports the max / mean per-pixel abs difference (0-255). The
wavefront math is the megakernel's, so at AA=1 (save_frame) the two should match
to within a couple of LSBs.

    .venv/Scripts/python.exe benchmarks/_wf_parity_check.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import algan.rendering.raytracing.primitives as rtp  # noqa: E402
from algan import (  # noqa: E402
    BLUE,
    GREEN,
    LEFT,
    OUT,
    RED,
    RIGHT,
    UP,
    Cylinder,
    MeshLambertMaterial,
    SceneManager,
    Sphere,
    Square,
    Sync,
)
from algan.rendering.raytracing import enable_ray_tracing  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "_tc_out")
os.makedirs(OUT_DIR, exist_ok=True)

# Wrap _merge_scene to print which geometry BVHs the scene actually populates.
_orig_merge = rtp._merge_scene
_last_counts = {}


def _merge_probe(primitives):
    merged = _orig_merge(primitives)
    _last_counts.clear()
    _last_counts.update(
        tri=int(merged["num_triangles"]),
        pn=int(merged["num_pn"]),
        bez=int(merged["num_circuits"]),
    )
    return merged


rtp._merge_scene = _merge_probe


# Include a 2-D bezier circuit only when ALGAN_PARITY_BEZIER=1 (its static
# save_frame color setup is finicky and unrelated to the trace kernels).
WITH_BEZIER = os.environ.get("ALGAN_PARITY_BEZIER", "0") == "1"
# Render the mesh mobs as curved PN patches (ALGAN_PARITY_PN=1) instead of flat
# triangles, so the parity check exercises the PN BVH + Matrix Pencil solver and
# PN fragment shading.
PN_TRIANGLES = os.environ.get("ALGAN_PARITY_PN", "0") == "1"


def build():
    with Sync():
        Sphere().scale(1.3).move(LEFT * 2.2).set_material(
            MeshLambertMaterial(color=BLUE)
        ).spawn()
        Cylinder(radius=0.5, height=2.0).move(RIGHT * 2.2).set_material(
            MeshLambertMaterial(color=RED)
        ).spawn()
        if WITH_BEZIER:
            # 2-D filled shapes -> bezier circuits. Two of them so the packed
            # circuit-opacity tensor has numel > 1 (a single circuit collapses
            # Color.opacity to a Python float in the static save_frame setup --
            # a pre-existing quirk unrelated to the trace kernels).
            Square(color=GREEN).scale(0.7).move(UP * 1.7 + OUT * 0.5).spawn()
            Square(color=GREEN).scale(0.7).move(
                UP * 1.7 + LEFT * 1.4 + OUT * 0.5
            ).spawn()


def render_once(wf, frag, shadows, tag):
    SceneManager.reset()
    enable_ray_tracing(
        samples_per_pixel=1,
        fragment_shading=frag,
        shadows=shadows,
        pn_triangles=PN_TRIANGLES,
    )
    rtp.set_wavefront(wf)
    build()
    scene = SceneManager.instance()
    out = os.path.join(OUT_DIR, f"wfparity_{tag}.png")
    frames = scene.save_frame(out)
    # save_frame returns CHW float tensors in [0, 1]; take the last.
    arr = frames[-1].permute(1, 2, 0).float().cpu().numpy() * 255.0
    return arr


def compare(a, b):
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return d.max(), d.mean(), float((d > 1.0).mean() * 100.0)


def main():
    configs = [
        ("vertex", {"frag": False, "shadows": False}),
        ("frag", {"frag": True, "shadows": False}),
        ("shadow", {"frag": True, "shadows": True}),
    ]
    print("Building reference (megakernel) + wavefront renders...\n")
    all_ok = True
    for name, cfg in configs:
        mega = render_once(False, cfg["frag"], cfg["shadows"], f"{name}_mega")
        counts = dict(_last_counts)
        wf = render_once(True, cfg["frag"], cfg["shadows"], f"{name}_wf")
        mx, mn, pct = compare(mega, wf)
        ok = mx <= 2.0
        all_ok = all_ok and ok
        print(
            f"[{name:6s}] geom(tri={counts['tri']} pn={counts['pn']} "
            f"bez={counts['bez']})  max|d|={mx:5.1f}  mean|d|={mn:6.3f}  "
            f">1LSB={pct:5.2f}%  {'OK' if ok else 'MISMATCH'}"
        )
    print("\nPARITY_OK:", all_ok)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
