"""A bezier circuit's stroke gets wider the further it is drawn off-axis.

FIXED -- this is now a regression check, and passes. It is kept because the
property it asserts is not obvious and is easy to break again.

A horizontal line has identical geometry in every column of the frame, so
every column must carry identical ink. Before the fix it did not: at MD a
default ``Line`` across the frame was drawn 9.09 px wide at the centre --
exactly its authored width -- and 12.18 px at the left and right edges, 34.7%
fatter, the growth tracking ``1 / cos(theta)`` with theta the pixel's angle off
the camera's optical axis (pure ``1 / cos`` predicts 33.8% on its own).

WHY. A circuit is drawn from a signed distance field evaluated in the plane, in
WORLD units, and ``pixel_size`` converts the authored stroke width (which is in
output pixels) into those world units. Both the raster path
(``raster_taichi._bez_pixel_hit``) and the wavefront/path-tracer paths
computed it as

    pixel_size = pixel_world_scale[f] * t          # t = ray parameter

and ``_generate_ray`` returns a NORMALISED direction, so ``t`` is the slant
range from the camera to the hit, not the perpendicular depth. The camera is a
pinhole with a FLAT image plane, whose world-to-screen Jacobian on a
fronto-parallel plane depends on the perpendicular depth ``Z`` alone and is
constant across the frame. Scaling by ``t = Z / cos(theta)`` therefore inflates
every drawn width by ``1 / cos(theta)``. The comment at the wavefront site says
what was intended -- "world size of one screen pixel at this hit, for
screen-constant border/outline widths" -- so this is an implementation slip,
not a design choice.

SCOPE. Only bezier circuits consult ``pixel_size``; triangles are rasterised
from exact screen-space edge functions and hold their width EXACTLY constant
across the frame, which is what the Cylinder control below is for. Within
circuits it splits by what sets the drawn boundary:

* an UNFILLED stroke (``Line``, ``filled=False``) is a band of half-width
  ``border_w / 2``, all of which scales -- the full 34%;
* a filled circuit's BORDER band scales the same way;
* a FILL boundary (``Text``, ``Tex``, filled 2-D shapes) only scales its
  sub-pixel anti-crack dilation, so it grows by ~0.2 px edge to edge.

THE FIX (``_axis_cos`` in ``raytrace_kernels_taichi``): multiply by the cosine
between the pixel's PRIMARY ray and the optical axis, which converts the slant
range into perpendicular depth. It is applied on all four paths that set a
circuit's width from a primary ray -- the hybrid raster path (which also covers
the supersampled aa=2 fallback), the wavefront traversal and both path
tracers -- and this script reports 0.000 px spread on every one of them.

``wavefront_shade`` keeps the accumulated-path heuristic on purpose: its
``pixel_size_per_t`` reaches only
``_shadow_occluded``, so it never sets a drawn width, and growth with path
length is defensible for a secondary ray. In the traversals the cosine is
rebuilt from the PIXEL rather than from the current ray, because ``gen_first``
is a compile-time template and the non-fused path makes its primaries in a
separate pass.

The inflation was ANGULAR, so it was essentially resolution-independent --
34.7% at MD, 35.4% at LD -- because a pixel at the frame edge sits at the same
angle off the optical axis whatever the pixel count. Both sat ~1 point above
the 33.8% that pure ``1 / cos(theta)`` predicts, so ``1 / cos`` accounted for
about 96% of it; the fix removes the whole effect (0.000 px spread), so the
residual belonged to the prediction's own second order, not to a second
mechanism.

Run:  <venv-python> benchmarks/_stroke_width_check.py [--res ld|md|hd]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#: Tolerated spread, in pixels, across the frame. A correct conversion holds a
#: constant-geometry line to a constant width; this leaves room only for 8-bit
#: readback noise.
TOLERANCE_PX = 0.05


def _profile(scene, video_settings, path, lut):
    """Ink per column of a frame containing one horizontal bar.

    Coverage is recovered through the measured transfer LUT rather than by
    dividing by the full-coverage value: the curve has a shoulder near white,
    and assuming linearity biases the width by ~1% -- enough to move the
    centre width off the authored one and muddy the comparison.
    """
    from benchmarks._aa_line_check import render, to_coverage

    return to_coverage(render(scene, path, video_settings), *lut).sum(0)


def _report(label, width, focal_px, scaling_px):
    """Report one bar's width profile against what the diagnosis predicts.

    ``scaling_px`` is how much of the drawn width is set by a quantity that
    ``pixel_size`` scales: the whole width for an unfilled stroke, only the
    sub-pixel anti-crack dilation for a fill, and nothing for triangles. The
    predicted spread is that amount times ``1 / cos(theta) - 1`` at the frame's
    corner-most column, with the focal length taken from the camera's own fov
    rather than fitted -- so agreement identifies the cause, it does not merely
    describe the curve.
    """
    half_frame = width.size / 2
    spread = float(np.ptp(width))
    centre = float(width[width.size // 2])
    growth = math.sqrt(1.0 + (half_frame / focal_px) ** 2) - 1.0
    print(
        f"{label:<34} centre {centre:6.3f}px  "
        f"edge {float(width[5]):6.3f}px  "
        f"spread {spread:5.3f}px ({100 * spread / centre:5.2f}% of centre)"
    )
    print(
        f"{'':<34} predicted spread {scaling_px * growth:5.3f}px "
        f"({scaling_px:.2f}px of the width scales by 1/cos, f={focal_px:.0f}px)"
    )
    return spread


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--res", default="md", choices=("ld", "md", "hd"))
    args = parser.parse_args(argv)

    from algan.constants.color import WHITE
    from algan.constants.spatial import LEFT, RIGHT
    from algan.mobs.shapes_2d import Line, Rectangle
    from algan.mobs.shapes_3d import Cylinder
    from algan.rendering.shaders.materials import MeshBasicMaterial
    from algan.scene import Scene
    from algan.settings.video_settings import HD, LD, MD

    out_dir = REPO_ROOT / "algan_outputs" / f"aa_check_{args.res}"
    out_dir.mkdir(parents=True, exist_ok=True)
    video_settings = {"ld": LD, "md": MD, "hd": HD}[args.res]
    height = video_settings.resolution[1]
    print(f"resolution {video_settings.resolution}")

    from benchmarks._aa_line_check import build_lut

    scene = Scene(video_settings=video_settings)
    focal_px = height / 2 / math.tan(math.radians(scene.camera.fov) / 2)
    print(f"camera fov {scene.camera.fov:.2f} deg -> f = {focal_px:.0f} px")
    print(
        "a horizontal line has identical geometry in every column, so every "
        "column must carry identical ink\n"
    )
    levels, curve, _ = build_lut(out_dir, video_settings)
    lut = (levels, curve)
    authored = 5 * height / 396  # Line's default stroke_width, in render pixels
    print(f"authored Line width: 5 * {height}/396 = {authored:.3f} px\n")

    from algan.rendering.raytracing import settings as rt_settings

    Line(LEFT * 40, RIGHT * 40, color=WHITE, scene=scene).spawn()
    stroke = _profile(scene, video_settings, out_dir / "stroke_unfilled.png", lut)
    # The whole band is border_w / 2 either side of the path, so all of it
    # scales.
    spread = _report(
        "Line (unfilled bezier stroke)",
        stroke,
        focal_px,
        float(stroke[stroke.size // 2]),
    )

    scene = Scene(video_settings=video_settings)
    Rectangle(width=80, height=0.09, color=WHITE, scene=scene, stroke_width=0).spawn()
    filled = _profile(scene, video_settings, out_dir / "stroke_filled.png", lut)
    # A fill's boundary is d > -min_half_width, so only that dilation scales,
    # once on each side.
    _report(
        "Rectangle (filled bezier)",
        filled,
        focal_px,
        2 * rt_settings.analytic_aa_bez_min_half_width,
    )

    scene = Scene(video_settings=video_settings)
    bar = Cylinder(radius=0.045, height=80, direction=RIGHT, color=WHITE, scene=scene)
    bar.set_material(MeshBasicMaterial(color=WHITE))
    bar.spawn()
    tri = _profile(scene, video_settings, out_dir / "stroke_triangles.png", lut)
    # Triangle coverage never consults pixel_size, so nothing scales.
    _report("Cylinder (triangles, the control)", tri, focal_px, 0.0)

    print()
    if spread > TOLERANCE_PX:
        print(
            f"FAIL: the unfilled stroke's width varies by {spread:.3f}px across "
            f"the frame (tolerance {TOLERANCE_PX}px). See this module's "
            f"docstring for the cause and the verified fix."
        )
        return 1
    print("OK: stroke width is constant across the frame.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
