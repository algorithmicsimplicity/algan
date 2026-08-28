"""Does nested-IOR refraction (``ALGAN_NESTED_IOR``, §H) change the image, and
only where it should?

Three still frames, each rendered gate-off then gate-on in one process, written
as PNG so the comparison is lossless (an mp4 A/B would fold the codec into the
delta -- see DESIGN_mesh_identity_open.md §Y.7):

* ``nested``    -- a glass sphere of ior 1.2 sitting inside a glass sphere of
  ior 1.5, over three colour bars. The inner interface's relative index is
  1.2/1.5 = 0.8 with the gate on and 1.2 with it off: the transmitted ray bends
  the OTHER WAY, so this frame MUST move. It is the only positive evidence the
  mechanism engages, which is what the §I session shipped without.
* ``cube`` and ``sphere`` -- ONE glass solid, alone, flat-faced and diced
  respectively. Nothing is nested, so the naive expectation is byte-identity.
  Both move anyway, and only at edges and grazing silhouettes: 0.05% and 0.10%
  of pixels here. That is not noise and not the plumbing -- the renders are
  bit-reproducible within each arm, and the movement survives disabling the
  entry half of the arithmetic, which means the stack reaches depth >= 2 on a
  CONVEX solid. Depth 2 needs two entries without an exit between them, which
  a straight ray cannot do to a convex shell: it is the epsilon artifact
  ``_offset_transmitted_origin``'s own docstring describes, where a
  transmitted origin at a shared edge lands outside the neighbouring face and
  the ray "enters" a solid it never left. With the gate on, the stack reads
  that as already-inside-1.5-entering-1.5 and refuses to bend, which is the
  physically right answer at a hit where there is no interface; the
  air-outside assumption bent it again. So the gate quietly fixes a second
  thing, and these two frames are bounded rather than pinned. Mesh identity
  (DESIGN_mesh_identity_open.md §I, unbuilt) is what would remove the artifact
  at its source, by refusing to enter a mesh the ray is already inside.
* ``pane``      -- a transmissive circuit (a thin pane, which never refracts)
  in front of an opaque solid. Circuits take no slot on the stack at all; this
  frame MUST be byte-identical, and is the control that says the plumbing
  itself -- the wider ray state, the second kernel variant -- moves nothing.

So the run is an assertion, not a measurement: one frame that has to move, one
that has to not, and two whose movement has to stay inside a stated band.
Byte-identity of the gate-off arm against a pre-feature build is a separate
question -- for that, render ``nested`` with the gate off on both builds and
compare.

Usage:
    <venv-python> benchmarks/_nested_ior_ab.py [quality] [--map]

``quality`` is a preset name (PREVIEW, LD, MD, HD); it defaults to LD, which is
enough to see the refraction and cheap enough to run on a CPU-only box.
``--map`` adds a coarse occupancy grid of where each frame moved, which is how
the silhouette band was identified in the first place.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from algan import *  # noqa: F403,E402
from algan.rendering.raytracing import settings as rt_settings  # noqa: E402
from algan.rendering.shaders.materials import (  # noqa: E402
    MeshPhysicalMaterial,
)
from algan.scene_manager import SceneManager  # noqa: E402

OUT_DIR = os.path.join("algan_outputs", "profiling")

# (label, expectation). "move" = must differ, "same" = must be byte-identical,
# "band" = may differ, but only in the edge/silhouette band the module docstring
# accounts for -- at most BAND_FRACTION of the frame.
SCENES = (
    ("nested", "move"),
    ("cube", "band"),
    ("sphere", "band"),
    ("pane", "same"),
)

# Measured at PREVIEW: cube 148 and sphere 270 of 278784 pixels, 0.05% and
# 0.10%. One percent is an order of magnitude above both and two below the
# ``nested`` arm's 2.17%, which is what a leak of the nesting arithmetic into
# an un-nested scene would look like.
BAND_FRACTION = 0.01


def _backdrop():
    """Three colour bars behind the glass, so a change of bend is legible."""
    for i in range(3):
        bar = Square(color=(YELLOW, GREEN, BLUE)[i]).scale(0.6)
        bar.rotate(25 * i - 25, OUT)
        bar.move(UP * (0.9 - 0.9 * i) + LEFT * (0.9 - 0.9 * i) - OUT * 2.5)
        bar.spawn(animate=False)


def build_scene(label):
    Scene.set_background(DARKER_GRAY)
    with Off():
        AmbientLight(color=WHITE, intensity=0.35).spawn(animate=False)
        PointLight(location=LEFT * 4 + UP * 3 + OUT * 4).spawn(animate=False)
        _backdrop()

        if label == "pane":
            solid = Sphere(radius=0.7, color=RED).move(-OUT * 0.8)
            solid.spawn(animate=False)
            pane = Square(color=WHITE).scale(1.6)
            pane.set_material(
                MeshPhysicalMaterial(transmission=0.95, roughness=0.02, ior=1.5)
            )
            pane.spawn(animate=False)
            return

        if label == "cube":
            # Convex and flat-faced: a straight ray crosses the shell exactly
            # twice, so any stack depth beyond one is an epsilon artifact
            # rather than geometry. Rotated off-axis so the transmitted rays
            # are not all at normal incidence.
            box = Cube().scale(1.2)
            box.rotate(28, UP).rotate(17, RIGHT)
            box.set_material(
                MeshPhysicalMaterial(transmission=0.95, roughness=0.02, ior=1.5)
            )
            box.spawn(animate=False)
            return

        outer = Sphere(radius=1.4)
        outer.set_material(
            MeshPhysicalMaterial(transmission=0.95, roughness=0.02, ior=1.5)
        )
        outer.spawn(animate=False)

        if label == "nested":
            # Strictly inside the outer shell, and off-centre so the two
            # interfaces are not concentric (a concentric pair bends
            # symmetrically and hides half the effect).
            inner = Sphere(radius=0.55).move(RIGHT * 0.25 + UP * 0.15)
            inner.set_material(
                MeshPhysicalMaterial(transmission=0.95, roughness=0.02, ior=1.2)
            )
            inner.spawn(animate=False)


def render_once(label, gate, quality):
    path = os.path.join(OUT_DIR, f"nested_ior_{label}_{'on' if gate else 'off'}.png")
    rt_settings.set_nested_ior(gate)
    SceneManager.reset()
    scene = SceneManager.instance().current_scene
    scene.set_video_settings(quality)
    build_scene(label)
    Scene.save_frame(path, quality, overwrite=True)
    return path


def read_frame(path):
    import cv2

    frame = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if frame is None:
        raise SystemExit(f"could not read {path}")
    return frame


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
        off = read_frame(render_once(label, False, quality))
        on = read_frame(render_once(label, True, quality))
        if off.shape != on.shape:
            raise SystemExit(f"{label}: frame shapes differ, {off.shape} vs {on.shape}")
        delta = np.abs(off.astype(np.int16) - on.astype(np.int16))
        pixel_delta = delta.max(axis=2)
        moved = int((delta > 0).sum())
        moved_pixels = int((pixel_delta > 0).sum())
        fraction = moved_pixels / pixel_delta.size
        if expectation == "move":
            ok, wanted = moved > 0, "must move"
        elif expectation == "same":
            ok, wanted = moved == 0, "must be identical"
        else:
            ok, wanted = fraction <= BAND_FRACTION, f"band <= {BAND_FRACTION:.0%}"
        if not ok:
            failures.append(label)
        print(
            f"{label:8s} {'moved' if moved else 'identical':9s} ({wanted:18s}) "
            f"max|d|={int(delta.max()):3d} channels>0={moved} "
            f"pixels={moved_pixels} ({fraction:.2%}) -> {'OK' if ok else 'FAIL'}",
            flush=True,
        )
        if show_map and moved_pixels:
            print("\n".join(occupancy_map(pixel_delta)), flush=True)
    if failures:
        raise SystemExit(f"FAILED: {', '.join(failures)}")
    print("every arm behaved as required", flush=True)


if __name__ == "__main__":
    main()
