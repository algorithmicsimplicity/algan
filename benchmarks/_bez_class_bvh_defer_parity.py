"""Parity check: bezier cell classification + deferred BVH builds.

Renders one mixed scene (text bezier circuits incl. a translucent fade, a
filled 2D shape, and a moving triangle-mesh sphere) four ways in one process:

  A. baseline        - cell classification OFF, BVH deferral OFF
  B. class only      - cell classification ON
  C. defer only      - BVH deferral ON
  D. both (defaults) - classification + deferral ON

B/C/D must decode byte-identically to A (max pixel diff 0).

Then two safety-net checks force deferral where the merge-time predicate
would refuse it, so the tracer's lazy ``build_deferred_bvhs`` hooks are
actually exercised:

  E. reflective sphere with the deferral predicate monkeypatched to True:
     a continuation ray spawns in iteration zero and the run_tile hook must
     build the trees mid-render. Compared byte-wise against the same scene
     rendered eagerly.
  F. same scene with hard shadows enabled and the predicate patched: the
     shadow_flag hook at wavefront entry must build the trees before the
     sparse shadow queue traces. Compared byte-wise against eager.

Run: .venv/Scripts/python.exe benchmarks/_bez_class_bvh_defer_parity.py
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("ALGAN_PREFETCH_BATCHES", "0")

import numpy as np

from algan import (
    BLUE,
    DOWN,
    LEFT,
    OUT,
    RED,
    RIGHT,
    UP,
    MeshStandardMaterial,
    Off,
    RenderSettings,
    Sphere,
    Square,
    Sync,
    Text,
    render_to_file,
)
from algan.rendering.raytracing import bezier_acceleration as bez_accel
from algan.rendering.raytracing import scene_builder
from algan.rendering.raytracing import settings as rt_settings

OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "_tc_out"))
SETTINGS = RenderSettings((352, 198), 10)


def build_scene(reflective=False):
    with Off():
        t = Text("Parity 123").scale(0.7).move(UP * 1.5)
        t.spawn()
        sq = Square().scale(0.8).move(LEFT * 2 + DOWN)
        sq.set_color(BLUE)
        sq.spawn()
        s = Sphere().scale(0.6).move(RIGHT * 2)
        if reflective:
            s.set_color(RED)
            s.set_material(MeshStandardMaterial(metalness=1.0, roughness=0.1))
        s.spawn()
    with Sync():
        t.opacity = 0.4
        sq.rotate(90, OUT)
        s.move(LEFT)


def render(
    name, *, bez_class, bvh_defer, reflective=False, force_defer=False, shadows=False
):
    bez_accel.BEZIER_CLASS_ENABLED = bez_class
    rt_settings.set_bvh_defer(bvh_defer)
    rt_settings.set_ray_traced_shadows(shadows)
    orig_eligible = scene_builder._bvh_deferral_eligible
    if force_defer:
        # Force deferral for a batch the predicate would (correctly) refuse,
        # to exercise the tracer's lazy build_deferred_bvhs hooks.
        scene_builder._bvh_deferral_eligible = lambda scene: True
    try:
        build_scene(reflective=reflective)
        render_to_file(file_name=name, output_dir=OUT_DIR, render_settings=SETTINGS)
    finally:
        scene_builder._bvh_deferral_eligible = orig_eligible
        rt_settings.set_ray_traced_shadows(False)
    return os.path.join(OUT_DIR, name + ".mp4")


def read_frames(path):
    import cv2

    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame.astype(np.int16))
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    return frames


def compare(path_a, path_b, label):
    fa = read_frames(path_a)
    fb = read_frames(path_b)
    assert len(fa) == len(fb), (label, len(fa), len(fb))
    worst = 0
    ndiff = 0
    mean_lum = 0.0
    for a, b in zip(fa, fb):
        d = np.abs(a - b)
        worst = max(worst, int(d.max()))
        ndiff += int((d > 0).any(-1).sum())
        mean_lum += float(a.mean())
    status = "OK (byte-identical)" if worst == 0 else "DIFF"
    print(
        f"{label:34s} max diff {worst:3d}  diff px {ndiff:8d}  "
        f"mean lum {mean_lum / len(fa):6.1f}  {status}"
    )
    return worst


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    a = render("parity_base", bez_class=False, bvh_defer=False)
    b = render("parity_class", bez_class=True, bvh_defer=False)
    c = render("parity_defer", bez_class=False, bvh_defer=True)
    d = render("parity_both", bez_class=True, bvh_defer=True)

    failures = 0
    failures += compare(a, b, "cell classification") != 0
    failures += compare(a, c, "bvh deferral") != 0
    failures += compare(a, d, "both (defaults)") != 0

    # Safety net: forced deferral on a reflective (continuation-spawning)
    # scene -> run_tile lazy build; with shadows -> wavefront-entry lazy build.
    e_ref = render("lazy_refl_eager", bez_class=True, bvh_defer=False, reflective=True)
    e = render(
        "lazy_refl_forced",
        bez_class=True,
        bvh_defer=True,
        reflective=True,
        force_defer=True,
    )
    failures += compare(e_ref, e, "forced defer + reflection") != 0

    f_ref = render(
        "lazy_shadow_eager",
        bez_class=True,
        bvh_defer=False,
        reflective=True,
        shadows=True,
    )
    f = render(
        "lazy_shadow_forced",
        bez_class=True,
        bvh_defer=True,
        reflective=True,
        shadows=True,
        force_defer=True,
    )
    failures += compare(f_ref, f, "forced defer + shadows") != 0

    print("PARITY FAILED" if failures else "PARITY PASSED")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
