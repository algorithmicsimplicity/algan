"""Render a capped solid's end disc face-on, so its rim can be looked at.

A flat cap's rim is the one piece of a logical PN surface the renderer cannot
refine: the disc's normals are all one constant, so its PN patch and its PN
edge curves *are* the flat triangle and its straight chords, and every
render-time criterion returns zero at level 0.  Whatever polygon the rim is
built as is the polygon that ships.  ``_CapDisc`` therefore sizes the rim
against ``geometry_tolerance`` at construction; this probe is how you look at
the result.

    <venv-python> benchmarks/_cap_rim_probe.py [outname] [rim_multiplier]

``rim_multiplier`` scales the rim's chord count away from what the disc chose,
keeping the body's own ring vertices as a subset.  Values below 1 are how you
reproduce the pre-refinement look: ``0`` pins the rim to the body's ring count
exactly, which is what a cap inherited before it sized its own rim, and which
renders the green ``solids_and_camera`` cylinder as a visible 14-gon with dark
notches where its chords fall inside the tube's PN-bowed silhouette.
"""

import sys

import algan.mobs.shapes_3d as shapes_3d
from algan import *  # noqa: F403

name = sys.argv[1] if len(sys.argv) > 1 else "cap_rim_probe"
multiplier = int(sys.argv[2]) if len(sys.argv) > 2 else 1

if multiplier != 1:
    _CapDiscBase = shapes_3d._CapDisc

    class _ScaledCapDisc(_CapDiscBase):
        """A cap whose rim refinement is forced, not measured."""

        def _rimmed_grid_width(self, segments, *_args):
            if multiplier <= 0:  # the pre-refinement behaviour: inherit and stop
                return max(3, int(segments))
            return max(3, (int(segments) - 1) * multiplier + 1)

    shapes_3d._CapDisc = _ScaledCapDisc

Scene.set_background(DARKER_GRAY)

with Off():
    AmbientLight(color=WHITE, intensity=0.45).spawn(animate=False)
    DirectionalLight(
        location=RIGHT * 5 + UP * 6 + OUT * 4,
        target=ORIGIN,
        color=WHITE,
        intensity=0.8,
    ).spawn(animate=False)
    HemisphereLight(color=BLUE_A, ground_color=MAROON_E, intensity=0.3).spawn(
        animate=False
    )

    # The solids_and_camera cylinder, scaled up and tilted the way Act 2 tilts
    # it, so the cap faces the camera.
    cylinder = Cylinder(radius=0.45, height=1.0, show_ends=True).set_material(
        MeshLambertMaterial(color=GREEN)
    )
    cylinder.scale(2.4).rotate(75, RIGHT).spawn(animate=False)

print(
    f"tube grid {cylinder.grid_width}x{cylinder.grid_height}, "
    f"cap grid {cylinder.top_cap.grid_width}x{cylinder.top_cap.grid_height}"
)

Scene.save_frame(name, HD)
