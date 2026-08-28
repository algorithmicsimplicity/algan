"""Does the shadow-terminator offset (``ALGAN_SHADOW_TERMINATOR``, work-queue
item 20) remove terminator acne, and does it move nothing it should not?

A PN patch is diced to FLAT triangles under a SMOOTH normal field, so a shadow
ray lifted along the *face* normal starts below the surface it approximates and
can strike a neighbouring facet at grazing incidence. Two things follow, and
the script is built around the second:

* the face-normal horizon cull (``fnrm . wi > 1e-3``) is what keeps that from
  showing today -- it refuses to trace exactly the rays that would go wrong;
* so the offset's value cannot be read off the default image. It is what makes
  the cull safe to relax, and the only way to see that is to relax the cull
  *without* the offset and watch the acne appear.

Hence three arms, and the setting is a tri-state:

* ``0``     -- off. Today's origin, today's cull.
* ``1``     -- on (the default). Hanika offset (Ray Tracing Gems II ch. 4) plus
  the relaxed cull.
* ``relax`` -- DIAGNOSTIC, not a supported configuration: the cull is relaxed
  but the origin is NOT moved. This is the arm that exposes the acne, and
  therefore the only one that can show the offset is what removes it.

Four still frames, each rendered in all three arms in one process, written as
PNG so the comparison is lossless (an mp4 A/B folds the codec into the delta --
DESIGN_mesh_identity_open.md §Y.7):

* ``torus``   -- a concave single mesh, where a shadow ray leaving one part of
  the tube strikes another: the acne population and the legitimate self-shadow
  population overlap, which is what makes it the item's own recommended shape.
  THE headline: the default arm must carry materially LESS speckle than both
  the other two. Measured at LD -- off 41, relax 38, **on 4**. That the relax
  arm sits with ``off`` rather than with ``on`` is the attribution: the cull
  relaxation is not what cleans the image, the offset is.
* ``sphere``  -- a diced convex solid on a lit plane, lit side-on so the
  terminator crosses the frame. Convex, so it cannot legitimately shadow
  itself across the terminator, and what ``relax`` newly admits there is
  therefore false: a majority of the pixels it moves must move DARKER, and
  ``on`` must darken none of them. Measured at LD: relax moves 24, 20 of them
  darker; ``on`` darkens 0 of the same pixels. (The other 4 go brighter --
  a newly admitted sample that comes back LIT enters the fan's average too.)
* ``cube``    -- a flat-faced solid on the same plane. Every vertex normal
  equals its face normal, so the Hanika delta is EXACTLY zero, ``lifted`` stays
  0 and both the origin and the cull are bit-for-bit today's. MUST be
  byte-identical in all three arms. This is the control that says flat-shaded
  geometry cannot move.
* ``circuit`` -- 2-D shapes only. No triangle in the scene, so no shadow event
  reaches the offset at all. MUST be byte-identical.

**Every mob here carries an explicit lit material.** Algan's default material
renders unlit, an unlit fragment builds no shadow event, and a scene of them
measures nothing at all -- which is how an earlier version of this file
reported "no change" from a feature that was working.

Usage:
    <venv-python> benchmarks/_shadow_terminator_ab.py [quality] [--map]

``quality`` is a preset name (PREVIEW, LD, MD, HD); it defaults to LD, which is
enough to resolve the band and cheap enough to run on a CPU-only box. ``--map``
adds a coarse occupancy grid of where each frame moved.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# A warm daemon carries the previous run's adaptive renderer state, and this
# script flips a renderer gate between renders (benchmarks/ convention).
os.environ.setdefault("ALGAN_USE_DAEMON", "0")

import numpy as np  # noqa: E402

from algan import *  # noqa: F403,E402
from algan.rendering.raytracing import settings as rt_settings  # noqa: E402
from algan.rendering.shaders.materials import MeshStandardMaterial  # noqa: E402
from algan.scene_manager import SceneManager  # noqa: E402

OUT_DIR = os.path.join("algan_outputs", "profiling")

ARMS = ("0", "1", "relax")

# (label, what this frame asserts).
#   "speckle" -- the default arm must carry materially less speckle than either
#                other arm, and buy it inside BAND_FRACTION of the frame.
#   "acne"    -- relax must move pixels, a majority of them darker, and the
#                default arm must darken none of the same pixels.
#   "same"    -- must be byte-identical in every arm.
SCENES = (
    ("torus", "speckle"),
    ("sphere", "acne"),
    ("cube", "same"),
    ("circuit", "same"),
)

# How much of the speckle the offset has to remove on ``torus`` for the run to
# count. Measured 41 -> 4, a factor of ten; half is a wide margin against the
# arm-to-arm noise of a CPU render and still impossible to pass by accident.
SPECKLE_RATIO = 0.5

# What either non-default arm may move on ``torus``, as a fraction of the
# frame. The relaxed cull only ever admits rays within 0.06 degrees of a
# facet's own plane, so the population is a thin band around the terminator
# and the contact: 177 pixels (0.04%) measured. One percent is far above that
# and far below a whole-object change.
BAND_FRACTION = 0.01

# A pixel counts as speckle when it is darker than the median of its 3x3
# neighbourhood by more than this many 8-bit levels. Acne is high-frequency by
# construction -- individual wrongly-occluded fragments inside a smoothly
# shaded region -- so a median filter separates it from the terminator's own
# gradient, which the median tracks.
SPECKLE_LEVELS = 6


def _lit(mob, color):
    return mob.set_material(MeshStandardMaterial(color=color, roughness=0.75))


def _ground():
    """A lit plane under the solid, so contact and cast shadow are both in
    frame. A Prism, not a Square: a 2-D circuit renders unlit and would receive
    no shadow at all.
    """
    ground = _lit(Prism(dimensions=(14, 0.25, 8), color=GRAY), GRAY)
    ground.move(DOWN * 1.7)
    ground.spawn(animate=False)


def build_scene(label):
    Scene.set_background(BLACK)
    with Off():
        AmbientLight(color=WHITE, intensity=0.06).spawn(animate=False)

        if label == "circuit":
            # 2-D only: no triangle anywhere, so no shadow event can reach the
            # offset by construction.
            PointLight(location=RIGHT * 5 + UP * 2 + OUT * 4).spawn(animate=False)
            for i, col in enumerate((RED, GREEN, BLUE)):
                disc = Circle(color=col).scale(0.5)
                disc.move(LEFT * 1.2 + RIGHT * 1.2 * i + UP * 0.3 * i)
                disc.spawn(animate=False)
            return

        # Almost exactly side-on: the terminator sits square in frame and the
        # shadow rays there leave the surface almost tangentially, which is the
        # population the offset is for.
        PointLight(location=RIGHT * 9 + UP * 0.6 + OUT * 0.8, intensity=2.2).spawn(
            animate=False
        )

        if label == "torus":
            # Concave and single-mesh: a shadow ray leaving the tube can strike
            # the tube again, so the acne population and the legitimate
            # self-shadow population overlap here.
            torus = _lit(
                Torus(major_radius=1.5, minor_radius=0.38, color=BLUE_B), BLUE_B
            )
            torus.rotate(58, RIGHT)
            torus.spawn(animate=False)
            return

        _ground()
        if label == "sphere":
            solid = _lit(Sphere(radius=1.35, color=BLUE_B), BLUE_B)
        else:
            solid = _lit(Cube(color=BLUE_B).scale(1.2), BLUE_B)
            solid.rotate(28, UP).rotate(17, RIGHT)
        solid.spawn(animate=False)


def render_once(label, arm, quality):
    path = os.path.join(OUT_DIR, f"shadow_term_{label}_{arm}.png")
    rt_settings.set_shadow_terminator(arm if arm == "relax" else int(arm))
    SceneManager.reset()
    scene = SceneManager.instance().current_scene
    scene.set_video_settings(quality)
    SETTINGS.raytracing.set(shadows=True)
    build_scene(label)
    Scene.save_frame(path, quality, overwrite=True)
    return path


def read_frame(path):
    import cv2

    frame = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if frame is None:
        raise SystemExit(f"could not read {path}")
    return frame


def luma(frame):
    import cv2

    return cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2GRAY).astype(np.int16)


def speckle_count(frame):
    """Pixels darker than their 3x3 neighbourhood median by SPECKLE_LEVELS."""
    import cv2

    gray = luma(frame)
    med = cv2.medianBlur(gray.astype(np.uint8), 3).astype(np.int16)
    return int(((med - gray) > SPECKLE_LEVELS).sum())


def moved(a, b):
    delta = np.abs(a.astype(np.int16) - b.astype(np.int16))
    per_pixel = delta.max(axis=2)
    return int(delta.max()), int((per_pixel > 0).sum()), per_pixel


def occupancy_map(pixel_delta, rows=9, cols=16):
    """Coarse grid of where a frame moved, as printable lines."""
    h, w = pixel_delta.shape
    grid = np.zeros((rows, cols), int)
    ys, xs = np.nonzero(pixel_delta)
    for y, x in zip(ys.tolist(), xs.tolist()):
        grid[min(y * rows // h, rows - 1), min(x * cols // w, cols - 1)] += 1
    return ["    " + "".join(f"{v:5d}" if v else "    ." for v in row) for row in grid]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    show_map = "--map" in sys.argv[1:]
    quality = globals()[args[0]] if args else LD
    os.makedirs(OUT_DIR, exist_ok=True)
    failures = []
    for label, expectation in SCENES:
        frames = {arm: read_frame(render_once(label, arm, quality)) for arm in ARMS}
        if len({f.shape for f in frames.values()}) != 1:
            raise SystemExit(f"{label}: frame shapes differ")
        off, on, relax = frames["0"], frames["1"], frames["relax"]

        max_d, px, per_pixel = moved(off, on)
        fraction = px / per_pixel.size
        speck = {arm: speckle_count(frames[arm]) for arm in ARMS}

        if expectation == "same":
            ok = px == 0 and moved(off, relax)[1] == 0
            wanted = "identical, every arm"
        elif expectation == "speckle":
            # The headline. The default arm has to be materially cleaner than
            # both the untouched arm and the cull-only arm, and it has to buy
            # that inside the thin band the cull relaxation can reach.
            floor = SPECKLE_RATIO * min(speck["0"], speck["relax"])
            ok = speck["1"] <= floor and fraction <= BAND_FRACTION
            wanted = f"on speckle <= {SPECKLE_RATIO:.0%} of both"
        else:
            # The acne assertion. ``relax`` must move pixels (the cull it drops
            # was doing work), every pixel it moves must go DARKER (a shadow
            # ray that should have missed reported a hit), and ``on`` must
            # carry none of that darkening.
            _, relax_px, relax_map = moved(off, relax)
            ys, xs = np.nonzero(relax_map)
            darker = int((luma(relax)[ys, xs] < luma(off)[ys, xs]).sum())
            on_dark = (
                int((luma(on)[ys, xs] < luma(off)[ys, xs]).sum()) if relax_px else 0
            )
            ok = relax_px > 0 and darker * 2 > relax_px and on_dark == 0
            wanted = "relax darkens, on does not"
            print(
                f"{label:8s} acne: relax moved {relax_px} px, {darker} of them "
                f"darker; on darkens {on_dark} of the same pixels",
                flush=True,
            )
        if not ok:
            failures.append(label)
        print(
            f"{label:8s} off->on {'moved' if px else 'identical':9s} "
            f"({wanted:26s}) max|d|={max_d:3d} pixels={px} ({fraction:.2%}) "
            f"speckle off/on/relax={speck['0']}/{speck['1']}/{speck['relax']} "
            f"-> {'OK' if ok else 'FAIL'}",
            flush=True,
        )
        if show_map and px:
            print("\n".join(occupancy_map(per_pixel)), flush=True)

    print(f"\nframes in {OUT_DIR}")
    if failures:
        raise SystemExit(f"FAILED: {', '.join(failures)}")
    print("all arms as expected")


if __name__ == "__main__":
    main()
