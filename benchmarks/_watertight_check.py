"""Qualification runs for the watertight ray/triangle test (``ALGAN_WATERTIGHT_TRI``).

``DESIGN_mesh_identity.md`` ss3.2 built the test and ss4.7 lists what has to be
measured before its default can flip. ``tests/unit_tests/test_watertight_triangle.py``
already pins the *unit* property (exactly one hit per shared edge). This script
covers the three ss4.7 items that need a rendered frame:

**1. No cracks in f32.** The dilation the shipped arm applies
(``BARYCENTRIC_EPSILON``) exists so a ray on a shared edge cannot miss *both*
neighbours. A watertight test removes the dilation, so the question is whether
exact edge-function negation really does close every seam at f32. Two scenes
attack it: adjacent quads at **grazing incidence** (where the projected edge is
nearly degenerate and the intersection is worst-conditioned), and a finely diced
``Sphere`` filling the frame (thousands of interior edges, and a silhouette where
patches meet at extreme angles). A crack is a background-coloured pixel STRICTLY
INSIDE the shape, found by filling the silhouette's holes and asking which filled
pixels came back as background -- so it is counted, not eyeballed.

**2. No double blend.** A duplicate hit on a shared edge blends the same surface
twice, which on a TRANSLUCENT solid paints a one-pixel ridge along every interior
edge. Measured as the count of interior pixels deviating from the median of their
3x3 neighbourhood by more than a threshold, at several alphas: a ridge is exactly
a thin-line deviation, and a smooth shading gradient is not.

**3. Register pressure -- NOT MEASURED HERE, despite what this used to claim.**
ss4.7 asks for occupancy against the 21-25% resolve ceiling; Nsight does not
support this machine's Pascal GPU (memory note "RT kernel occupancy diagnosis"),
and the per-kernel device times this docstring promised were never implemented --
the JSON report has no such key and never did. Read the two sections below as
what they are: a QUALITY comparison, not a cost one. The cost question is still
open, and ssK's decision is written not to depend on it (both arms are
compile-time dead at shipped defaults, so keeping them costs nothing).

All of it runs with the **hybrid raster front-end off**, so primary visibility
goes through the ray path this flag changes. With the front-end on (the default)
the flag is byte-identical, because primary visibility never reaches this code --
which is also why a default flip is lower-risk than it sounds, and why the
front-end-off numbers are the ones that matter.

``watertight_tri`` is read at **import** (it changes the compiled kernel body),
so one process exercises one arm. Run it twice. A separate cache directory per
arm is optional -- the offline cache is keyed on the compiled IR, so the two
arms' kernels never collide in one cache -- but it keeps each arm's cold-compile
time attributable to that arm::

**The default is now True**, so it is the Moller-Trumbore arm that needs the env
var, not the watertight one::

    ALGAN_CACHE_DIR=<dir>/wt_on .venv/Scripts/python.exe benchmarks/_watertight_check.py on
    ALGAN_WATERTIGHT_TRI=0 ALGAN_CACHE_DIR=<dir>/wt_moller \
        .venv/Scripts/python.exe benchmarks/_watertight_check.py moller

Then diff the two reports. The crack counts must be zero in BOTH arms (a nonzero
count in the watertight arm is a blocker) and the ridge counts must not grow.

MEASURED on CUDA with exactly that pair: the arms are IDENTICAL on both items --
0 cracks on grazing quads and on a diced Sphere in each, and the same ridge
counts (114 / 0 / 0 at alpha 0.35 / 0.6 / 0.85, max deviation 9.0 in both). The
dilation the Moller arm applies buys nothing here, so it has no remaining future
as a candidate default; its value is as the A/B CONTROL for ss3.2, which is the
argument for keeping it as a ``ti.static`` arm rather than deleting it (ssK).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from algan import *  # noqa: F403,E402
from algan.rendering.raytracing import settings as rt_settings  # noqa: E402
from algan.rendering.shaders.materials import MeshBasicMaterial  # noqa: E402
from algan.scene_manager import SceneManager  # noqa: E402

OUT_DIR = os.path.join("algan_outputs", "watertight_check")
# Forcing the ray path (hybrid_raster off) allocates per-ray state for every
# pixel, so it needs far more per-frame memory than the raster front-end: at MD
# with a 1.4 GB budget this OOMs on "a single frame". Measured on a 4 GB GTX 1050,
# 2.2 GB is affordable and PREVIEW/LD fit inside it. The figure REPLACES the
# free-VRAM measurement rather than capping it, so it must stay affordable on the
# device -- raise the resolution before raising this.
PINNED_BYTES = 2_200_000_000

# A background no unlit surface in these scenes can produce -- every shape is
# red or green. ``set_background`` also accepts image paths and procedural
# callables, so rather than hard-code what BLUE decodes to after tonemapping and
# encoding, the reference is READ BACK from a corner pixel of the rendered frame
# (every scene here is centred, so a corner is background by construction).
BACKGROUND = BLUE
# Slack on the read-back reference: the encoder is lossy, so background pixels
# are not all bit-equal. Still far below the distance to any shape colour.
BG_TOL = 8


def _background_mask(img):
    """Which pixels are background, keyed on the frame's own corner pixel."""
    corner = img[0, 0].astype(np.int16)
    return np.all(np.abs(img - corner) <= BG_TOL, axis=-1)


def _unlit(mob, color):
    """Flat, unlit and opaque: the crack test must not depend on shading."""
    mob.set_material(MeshBasicMaterial())
    mob.color = color
    return mob


def scene_grazing():
    """Adjacent quads seen almost edge-on.

    Grazing incidence is where a ray/triangle test is worst-conditioned: the
    projected triangle is nearly degenerate, so an edge function that is not
    *exactly* negated between neighbours produces a visible seam. Several
    strips at several angles, each a run of quads sharing interior edges.
    """
    Scene.set_background(BACKGROUND)
    with Off():
        for row, tilt in enumerate((84.0, 87.0, 89.0)):
            for i in range(6):
                quad = Square(color=WHITE).scale(0.55)
                quad.move(RIGHT * (i - 2.5) * 1.1 + UP * (row - 1) * 1.9)
                quad.rotate(tilt, RIGHT)
                _unlit(quad, RED).spawn(animate=False)


def scene_diced_sphere():
    """One finely diced Sphere filling the frame.

    Thousands of interior shared edges, and a silhouette where adjacent patches
    meet at extreme angles -- the ss4.7 "extreme silhouette" case.
    """
    Scene.set_background(BACKGROUND)
    with Off():
        sphere = Sphere(radius=1.9, resolution=(192, 96))
        _unlit(sphere, GREEN).spawn(animate=False)


def scene_translucent(alpha):
    """A translucent Sphere and Cylinder: the double-blend case.

    A duplicate hit on a shared edge composites the surface twice, which shows
    as a one-pixel ridge along interior edges. Opacity has to be < 1 for the
    second blend to be visible at all.
    """
    Scene.set_background(BACKGROUND)
    with Off():
        sphere = Sphere(radius=1.1, resolution=(96, 48)).move(LEFT * 1.3)
        _unlit(sphere, GREEN)
        sphere.opacity = alpha
        sphere.spawn(animate=False)
        cyl = Cylinder(radius=0.8, height=2.2, resolution=(96, 8)).move(RIGHT * 1.3)
        _unlit(cyl, RED)
        cyl.opacity = alpha
        cyl.spawn(animate=False)


def render_frame(build, tag, quality):
    path = os.path.join(OUT_DIR, f"wt_{tag}.png")
    SceneManager.reset()
    SETTINGS.computing.set(available_memory_override=PINNED_BYTES)
    # The whole point: primary visibility must go through the RAY path, which is
    # the only path _tri_hit is on.
    SETTINGS.raytracing.experimental.set(hybrid_raster=False)
    scene = SceneManager.instance().current_scene
    scene.set_video_settings(quality)
    build()
    Scene.save_frame(path, quality)
    import cv2

    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"could not read back {path}")
    return img.astype(np.int16)


def count_cracks(img):
    """Background-coloured pixels strictly inside a shape.

    ``binary_fill_holes`` closes the shape's interior, so the filled-minus-drawn
    set is exactly the enclosed background -- a crack. Reported with the shape
    size, because "0 cracks" only means something if the shape is actually there.
    """
    from scipy import ndimage

    is_bg = _background_mask(img)
    drawn = ~is_bg
    if not drawn.any():
        return {"drawn_px": 0, "cracks": -1, "note": "nothing drawn"}
    filled = ndimage.binary_fill_holes(drawn)
    cracks = int((filled & is_bg).sum())
    labels, n = ndimage.label(filled & is_bg)
    sizes = sorted((int((labels == i).sum()) for i in range(1, n + 1)), reverse=True)[
        :5
    ]
    return {
        "drawn_px": int(drawn.sum()),
        "cracks": cracks,
        "crack_blobs": n,
        "largest_blobs": sizes,
    }


def count_ridges(img, threshold=6):
    """Interior pixels deviating from their 3x3 median by more than threshold.

    A double blend is a one-pixel line, which a median filter removes and a
    smooth shading gradient survives -- so the difference isolates thin-line
    artifacts. Restricted to the shape's interior (eroded, so the silhouette's
    own anti-aliased edge is not counted as a ridge).
    """
    from scipy import ndimage

    is_bg = _background_mask(img)
    interior = ndimage.binary_erosion(~is_bg, iterations=3)
    if not interior.any():
        return {"interior_px": 0, "ridges": -1, "note": "no interior"}
    gray = img.max(axis=-1).astype(np.float32)
    med = ndimage.median_filter(gray, size=3)
    dev = np.abs(gray - med)
    return {
        "interior_px": int(interior.sum()),
        "ridges": int(((dev > threshold) & interior).sum()),
        "max_dev": float(dev[interior].max()),
        "mean_dev": float(dev[interior].mean()),
    }


def main():
    arm = sys.argv[1] if len(sys.argv) > 1 else "arm"
    quality = globals()[sys.argv[2]] if len(sys.argv) > 2 else LD
    os.makedirs(OUT_DIR, exist_ok=True)

    from algan.rendering.raytracing import raytrace_kernels_taichi as k

    watertight = k.watertight_tri()
    print(f"arm={arm}  WATERTIGHT_TRI={watertight}  quality={quality.resolution}")
    print(f"hybrid raster: OFF (ray path)  analytic AA: {rt_settings.analytic_aa}")
    print()

    report = {"arm": arm, "watertight": watertight}

    print("== 1. CRACKS (must be 0; a crack is enclosed background) ==")
    for name, build in (
        ("grazing", scene_grazing),
        ("diced_sphere", scene_diced_sphere),
    ):
        img = render_frame(build, f"{arm}_{name}", quality)
        stats = count_cracks(img)
        report[f"cracks_{name}"] = stats
        print(f"  {name:14s} {stats}")

    print()
    print("== 2. DOUBLE BLEND (ridge count on interior edges) ==")
    for alpha in (0.35, 0.6, 0.85):
        img = render_frame(
            lambda a=alpha: scene_translucent(a), f"{arm}_alpha{alpha}", quality
        )
        stats = count_ridges(img)
        report[f"ridges_alpha{alpha}"] = stats
        print(f"  alpha={alpha:<5} {stats}")

    out = os.path.join(OUT_DIR, f"report_{arm}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
