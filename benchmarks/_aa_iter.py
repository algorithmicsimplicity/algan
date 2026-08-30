"""Fast analytic-AA iteration harness: one frame per scene against cached refs.

Rebuilt for DESIGN_analytic_aa_v2.md (the ss21 original was lost to a truncated
write). One ``save_frame`` per scene under the LIVE settings (arms are selected
by env vars before launch, so switching arms never edits code), compared
against a cached ``supersampling=4`` supersampled reference and a cached
aliased (analytic off, aa=1) arm. ~12s warm for all scenes, vs ~15min for the
video gates -- use those (``_analytic_aa_bez_check.py``, ``_aa_match_aa2.py``)
before shipping, and this in the loop.

Scenes: slant / stem / corner / glyph (circuits: silhouette, sub-pixel stems,
convex+concave corners, real text with holes), mesh / thin (triangles:
silhouette gradation, sub-pixel rods). Metrics per scene:

    L1       mean |arm - ref| over the frame (the silhouette quality)
    ink      sum(arm luminance) / sum(ref luminance) -- dilation vs erosion;
             the metric that says whether sub-pixel geometry VANISHED
    notches  interior pixels measurably darker than the reference (the seam:
             background bleeding through fully covered pixels)
    maxdev   max |arm - ref| over the frame

MEASUREMENT TRAP (ss21.2): the aa=4 reference dilates filled circuits by 0.15
output px while the analytic default is analytic_aa_bez_min_half_width = 0.3.
A sharper filter amplifies that fixed offset, so exact-coverage arms score
WORSE for being MORE faithful unless measured at a matched 0.15:

    ALGAN_ANALYTIC_AA_BEZ_MIN_HALF_WIDTH=0.15 python benchmarks/_aa_iter.py

Run:
    .venv/Scripts/python.exe benchmarks/_aa_iter.py [scene ...] [--refresh-refs]

Arms via env, e.g.:
    ALGAN_ANALYTIC_AA_BEZ_WEDGE=1 ... (the oriented wedge)
    ALGAN_ANALYTIC_AA_RUN=1       ... (run-corrected triangle coverage)
    ALGAN_ANALYTIC_AA_EXACT=0     ... (the box filter)
"""

import os
import sys

os.environ.setdefault("ALGAN_PREFETCH_BATCHES", "0")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

# Benchmarks must never be measured inside a warm daemon: it keeps adaptive
# renderer state (the memory model's batch-size fit) across runs, so one
# benchmark would be timed against whatever ran before it.
os.environ.setdefault("ALGAN_USE_DAEMON", "0")

from algan import (  # noqa: E402
    BLUE,
    DOWN,
    GREEN,
    LEFT,
    OUT,
    RED,
    RIGHT,
    UP,
    WHITE,
    YELLOW,
    Line3D,
    Off,
    Polygon,
    Rectangle,
    Scene,
    SceneManager,
    Sphere,
    Square,
    Text,
    Triangle,
    VideoSettings,
)
from algan.rendering.raytracing import settings as rt_settings  # noqa: E402

BASE = os.path.dirname(os.path.abspath(__file__))
REF_DIR = os.path.join(BASE, "_aa_iter_ref")
OUT_DIR = os.path.join(BASE, "_aa_iter_out")
os.makedirs(REF_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

W, H = 320, 180
SCENES = ("slant", "stem", "corner", "glyph", "mesh", "thin")


def build_scene(name):
    if name == "slant":
        # One filled square rotated off-axis: the cleanest staircase.
        with Off():
            Square(color=RED).scale(1.4).rotate(24, OUT).spawn()
        return
    if name == "stem":
        # Sub-pixel and near-pixel STEMS: pairs of walls a fraction of a pixel
        # apart, which a single half-plane model reads as solid (ss21.2). Both
        # axis-aligned (where the box filter is exact) and slightly rotated
        # (where it is not).
        with Off():
            for i, w in enumerate((0.055, 0.028, 0.014)):
                Rectangle(width=w, height=1.4, color=WHITE).move(
                    LEFT * 1.5 + RIGHT * 0.5 * i + UP * 0.8
                ).spawn()
            for i, w in enumerate((0.055, 0.028, 0.014)):
                Rectangle(width=w, height=1.4, color=WHITE).rotate(10, OUT).move(
                    RIGHT * (0.2 + 0.5 * i) + UP * 0.8
                ).spawn()
            Text("lllll iiii 1111").scale(0.4).move(DOWN * 0.8).spawn()
        return
    if name == "corner":
        # Convex corners (triangle, rotated square) and CONCAVE corners (an
        # L-polygon) -- the configuration ss21.6's wedge broke on and the one
        # the flatten-time inward signs exist to fix.
        with Off():
            Triangle(color=YELLOW).scale(0.7).move(LEFT * 1.6 + UP * 0.5).spawn()
            Square(color=RED).scale(0.55).rotate(24, OUT).move(
                LEFT * 0.2 + UP * 0.55
            ).spawn()
            ell = Polygon(
                np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [1.6, 0.0, 0.0],
                        [1.6, 0.55, 0.0],
                        [0.55, 0.55, 0.0],
                        [0.55, 1.6, 0.0],
                        [0.0, 1.6, 0.0],
                    ],
                    dtype=np.float32,
                ),
                color=GREEN,
            )
            ell.rotate(18, OUT).move(RIGHT * 1.2 + DOWN * 1.3).spawn()
        return
    if name == "glyph":
        # Real text: holes, both winding conventions, thin stems at glyph
        # scale (the ss21.10 "o8B" line).
        with Off():
            Text("o8B gjq 0123").scale(0.5).move(UP * 0.4).spawn()
            Text("The quick brown fox").scale(0.28).move(DOWN * 0.6).spawn()
        return
    if name == "mesh":
        # Opaque triangle meshes (the ss16.2 "tri" config): silhouette
        # gradation and interior shared edges.
        with Off():
            s1 = Sphere().scale(1.0).move(LEFT * 1.2 + RIGHT * 0.6)
            s1.set_color(BLUE)
            s1.spawn()
            s2 = Sphere().scale(0.7).move(RIGHT * 1.2 + UP * 0.4)
            s2.set_color(GREEN)
            s2.spawn()
        return
    if name == "thin":
        # Sub-pixel triangle rods (~0.9 / 0.45 / 0.22 px): the sliver-policy
        # config. Read INK -- "it disappeared" is a brightness question.
        with Off():
            for i, th in enumerate((0.02, 0.01, 0.005)):
                Line3D(
                    start=LEFT * 1.4 + UP * (0.7 - 0.7 * i),
                    end=RIGHT * 1.4 + UP * (0.7 - 0.7 * i),
                    thickness=th,
                    color=YELLOW,
                ).spawn()
            sm = Sphere().scale(0.02).move(DOWN * 1.3 + RIGHT * 0.4)
            sm.set_color(BLUE)
            sm.spawn()
        return
    raise SystemExit(f"unknown scene {name}")


def render_frame(name, path, aa_level, analytic):
    """One frame of ``name`` at ``aa_level``; analytic AA forced on or off."""
    SceneManager.reset()
    prev = (
        rt_settings.analytic_aa,
        rt_settings.analytic_aa_tri,
    )
    rt_settings.set_analytic_aa(analytic, triangles=analytic and prev[1])
    settings = VideoSettings((W, H), frames_per_second=4, supersampling=aa_level)
    try:
        with Scene(video_settings=settings) as scene:
            build_scene(name)
            scene.save_frame(path, video_settings=settings, overwrite=True)
    finally:
        rt_settings.set_analytic_aa(prev[0], triangles=prev[1])
    return cv2.imread(path).astype(np.float64)


def interior_notches(arm, ref, dark=5.0):
    """Interior pixels measurably darker than the reference (the seam)."""
    lit = (arm.max(axis=2) > 8) & (ref.max(axis=2) > 8)
    inner = lit
    for _ in range(3):
        inner = (
            inner[1:-1, 1:-1]
            & inner[:-2, 1:-1]
            & inner[2:, 1:-1]
            & inner[1:-1, :-2]
            & inner[1:-1, 2:]
        )
        inner = np.pad(inner, 1, constant_values=False)
    if not inner.any():
        return 0
    delta = (arm.mean(axis=2) - ref.mean(axis=2))[inner]
    return int((delta < -dark).sum())


def metrics(arm, ref):
    l1 = float(np.abs(arm - ref).mean())
    ink = float(arm.sum() / max(ref.sum(), 1.0))
    n = interior_notches(arm, ref)
    maxdev = float(np.abs(arm - ref).max())
    return l1, ink, n, maxdev


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    refresh = "--refresh-refs" in sys.argv
    scenes = args or SCENES
    rows = []
    for name in scenes:
        ref_png = os.path.join(REF_DIR, f"{name}_ref.png")
        ali_png = os.path.join(REF_DIR, f"{name}_aliased.png")
        if refresh or not (os.path.exists(ref_png) and os.path.getsize(ref_png)):
            render_frame(name, ref_png, 4, False)
            render_frame(name, ali_png, 1, False)
        ref = cv2.imread(ref_png).astype(np.float64)
        ali = cv2.imread(ali_png).astype(np.float64)
        arm = render_frame(name, os.path.join(OUT_DIR, f"{name}_aa.png"), 1, True)
        l1, ink, n, maxdev = metrics(arm, ref)
        al1, aink, an, amax = metrics(ali, ref)
        rows.append((name, l1, ink, n, maxdev, al1, aink, an))
        print(
            f"[{name:6s}] L1 {l1:7.4f}  ink {ink:5.3f}  notches {n:4d}  "
            f"maxdev {maxdev:5.1f}   (aliased: L1 {al1:7.4f}  ink {aink:5.3f}"
            f"  notches {an:4d})"
        )
    print()
    print("scene    L1        ink     notches  (vs aliased L1)")
    for name, l1, ink, n, _m, al1, _ai, _an in rows:
        rel = l1 / max(al1, 1e-9)
        print(f"{name:8s} {l1:8.4f} {ink:7.3f} {n:7d}   {rel:5.2f}x")


if __name__ == "__main__":
    main()
