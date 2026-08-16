"""A bezier circuit's stroke gets wider the further it is drawn off-axis.

A horizontal line has identical geometry in every column of the frame, so
every column must carry identical ink. It does not: a default ``Line`` across
the frame is drawn 9.16 px wide at the centre and 12.28 px wide at the left and
right edges, 34% fatter, and the growth follows ``1 / cos(theta)`` exactly,
where theta is the pixel's angle off the camera's optical axis.

WHY. A circuit is drawn from a signed distance field evaluated in the plane, in
WORLD units, and ``pixel_size`` converts the authored stroke width (which is in
output pixels) into those world units. Both the raster path
(``raster_taichi._bez_pixel_hit``) and the wavefront/path-tracer paths
(``raytrace_kernels_taichi``, three sites) compute it as

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

FIX, verified: multiply by the cosine, which is one dot product against a
camera basis vector already passed to the kernel::

    fwd = (vec3(screen_point[f, 0], screen_point[f, 1],
                screen_point[f, 2]) - ro).normalized()
    pixel_size = pixel_world_scale[f] * th * rd.dot(fwd)

With that in place this script reports a spread of 0.000 px. It is NOT applied:
it changes every off-centre stroke, so it moves rendered output (the fast
suite's baseline by up to 116 channel values, against a tolerance of 2) and
would need both the CPU and the CUDA baselines regenerated -- and the CUDA set
can only be made on a CUDA machine. The three wavefront sites need the same
treatment, and their ``base_dist + t`` accumulation raises a separate question
about what the intended footprint is on a SECONDARY ray, where growth with path
length is a reasonable ray-differential heuristic rather than a bug.

Run:  <venv-python> benchmarks/_stroke_width_check.py
"""

from __future__ import annotations

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


def _profile(scene, video_settings, path):
    """Ink per column of a frame containing one horizontal bar."""
    import cv2

    from algan.constants.color import BLACK

    scene.save_frame(str(path), video_settings, background_color=BLACK)
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)[..., :3].astype(np.float64)
    # Full coverage reads 222 and the transfer curve is very nearly linear
    # below it; for a width PROFILE that suffices, since the same two edge
    # pixels per column are the only ones in the curved region and they are
    # the same at every column. _aa_line_check.py does the exact inversion.
    return image.mean(-1).sum(0) / 222.0


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
    growth = math.sqrt(1.0 + (half_frame / focal_px) ** 2) - 1.0
    print(
        f"{label:<34} centre {float(width[width.size // 2]):6.3f}px  "
        f"edge {float(width[5]):6.3f}px  "
        f"spread {spread:5.3f}px ({100 * spread / width.mean():5.2f}%)"
    )
    print(
        f"{'':<34} predicted spread {scaling_px * growth:5.3f}px "
        f"({scaling_px:.2f}px of the width scales by 1/cos, f={focal_px:.0f}px)"
    )
    return spread


def main():
    from algan.constants.color import WHITE
    from algan.constants.spatial import LEFT, RIGHT
    from algan.mobs.shapes_2d import Line, Rectangle
    from algan.mobs.shapes_3d import Cylinder
    from algan.rendering.shaders.materials import MeshBasicMaterial
    from algan.scene import Scene
    from algan.settings.video_settings import MD

    out_dir = REPO_ROOT / "algan_outputs" / "aa_check"
    out_dir.mkdir(parents=True, exist_ok=True)
    video_settings = MD
    height = video_settings.resolution[1]

    scene = Scene(video_settings=video_settings)
    focal_px = height / 2 / math.tan(math.radians(scene.camera.fov) / 2)
    print(f"camera fov {scene.camera.fov:.2f} deg -> f = {focal_px:.0f} px")
    print(
        "a horizontal line has identical geometry in every column, so every "
        "column must carry identical ink\n"
    )

    from algan.rendering.raytracing import settings as rt_settings

    Line(LEFT * 40, RIGHT * 40, color=WHITE, scene=scene).spawn()
    stroke = _profile(scene, video_settings, out_dir / "stroke_unfilled.png")
    # The whole band is border_w / 2 either side of the path, so all of it
    # scales.
    spread = _report(
        "Line (unfilled bezier stroke)",
        stroke,
        focal_px,
        float(stroke[stroke.size // 2]),
    )

    scene = Scene(video_settings=video_settings)
    Rectangle(
        width=80, height=0.09, color=WHITE, scene=scene, border_width=0
    ).spawn()
    filled = _profile(scene, video_settings, out_dir / "stroke_filled.png")
    # A fill's boundary is d > -min_half_width, so only that dilation scales,
    # once on each side.
    _report(
        "Rectangle (filled bezier)",
        filled,
        focal_px,
        2 * rt_settings.ANALYTIC_AA_BEZ_MIN_HALF_WIDTH,
    )

    scene = Scene(video_settings=video_settings)
    bar = Cylinder(
        radius=0.045, height=80, direction=RIGHT, color=WHITE, scene=scene
    )
    bar.set_material(MeshBasicMaterial(color=WHITE))
    bar.spawn()
    tri = _profile(scene, video_settings, out_dir / "stroke_triangles.png")
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
