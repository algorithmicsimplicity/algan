r"""Match a Scene's viewpoint, lighting and framing to Manim's defaults.

:func:`apply_manim_defaults` -- reached as
:meth:`Scene.use_manim_defaults() <algan.scene.Scene.use_manim_defaults>` -- repoints
a Scene's camera and lights so that geometry authored for Manim lands on the same
pixels Manim would put it on. It is the setting half of the Manim compatibility
layer, next to the mob half in :mod:`algan.mobs.manim_compat`.

What Manim's defaults actually are
----------------------------------
Manim's frame is **8 world units tall**, always; its width follows from the output
aspect ratio (14.222 units at 16:9). That is the same convention Algan's camera
uses -- ``fov`` is vertical and the horizontal angle is derived -- so the two
line up without a special case.

Manim has *two* cameras. A plain ``Scene`` draws through the Cairo camera, which
is a flat orthographic projection. A ``ThreeDScene`` draws through
``ThreeDCamera``, a pinhole perspective whose eye sits ``focal_distance = 20``
units from the frame plane: it scales a point by ``f / (f - z)``. At ``z = 0``
that factor is exactly 1, so the two agree on everything a 2-D scene contains,
and one Algan perspective camera at distance 20 with

.. math:: \\mathrm{fov} = 2 \\arctan\\frac{8 / 2}{20} = 22.62^\\circ

reproduces both: 2-D content exactly, 3-D content with Manim's own perspective.

The z axis agrees
-----------------
Manim's ``OUT`` and Algan's ``OUTWARD`` are both ``+z``, so a point keeps the
coordinates it was written with and there is nothing to convert. Algan used to
put ``OUTWARD`` at ``-z``, which made the two screen bases mirror images and
needed a ``Scene.manim_coordinates`` flag to mirror imported geometry and the
camera together; that flag and its ``from_manim_coordinates`` /
``to_manim_coordinates`` helpers are gone with the reason for them.
"""

from __future__ import annotations

import math

import torch

from algan.constants.color import BLACK, WHITE
from algan.constants.spatial import ORIGIN, OUTWARD
from algan.settings import SETTINGS
from algan.animation_timeline.animation_contexts import Off

#: Height of Manim's frame in world units. Manim pins the vertical extent and
#: derives the width from the aspect ratio, exactly as Algan's vertical ``fov``
#: does (``manim.constants``/``manim._config``: ``frame_height``).
MANIM_FRAME_HEIGHT = 8.0

#: Distance from Manim's ``ThreeDCamera`` eye to the frame plane, in world units
#: (``ThreeDCamera.focal_distance``). Its projection scales a point by
#: ``f / (f - z)``, which is a pinhole camera this far from ``z = 0``.
MANIM_FOCAL_DISTANCE = 20.0

#: Manim's ``ThreeDCamera.light_source_start_point``
#: (``9 * DOWN + 7 * LEFT + 10 * OUT``). Manim's coordinates are Algan's, so it
#: is usable as an Algan light position directly.
MANIM_LIGHT_SOURCE = (-7.0, -9.0, 10.0)

#: Manim's default output resolution and frame rate (``manim/_config/default.cfg``).
MANIM_RESOLUTION = (1920, 1080)
MANIM_FRAMES_PER_SECOND = 60

#: Cairo user units a Manim stroke of width 1 is drawn across
#: (``manim.camera.camera.Camera.cairo_line_width_multiple``). Manim's cairo
#: context is scaled to the frame, so a stroke of width ``w`` comes out
#: ``w * this * frame_height_px / MANIM_FRAME_HEIGHT`` pixels across --
#: measured exactly: width 20 at 854x480 draws 12.00 px.
MANIM_CAIRO_LINE_WIDTH_MULTIPLE = 0.01

#: What Algan's stroke-width unit is worth in Manim's, under Algan's own
#: convention: Manim's number is simply twice Algan's.
ALGAN_STROKE_WIDTH_RATIO = 2.0


def manim_stroke_width_ratio() -> float:
    """Get the Manim stroke-width units per Algan unit that draw the same weight.

    Both engines size a stroke against the frame height, so the ratio is one
    number at every resolution. Manim draws ``w`` as
    ``w * MANIM_CAIRO_LINE_WIDTH_MULTIPLE * H / MANIM_FRAME_HEIGHT`` pixels;
    Algan draws it as ``w * H / PREVIEW.resolution[1]``
    (``_stroke_width_in_render_pixels``). Equating them leaves

    ``MANIM_FRAME_HEIGHT / (PREVIEW_height * MANIM_CAIRO_LINE_WIDTH_MULTIPLE)``,

    which is ``2.0202``, not the flat ``2.0`` Algan's own convention uses --
    they would agree exactly if ``PREVIEW`` were 400 px tall instead of 396, so
    the standing convention is 1.01% off Manim rather than wrong in kind.

    Returns
    -------
    float
        Manim stroke-width units per Algan unit.
    """
    from algan.settings.video_settings import PREVIEW

    return MANIM_FRAME_HEIGHT / (
        PREVIEW.resolution[1] * MANIM_CAIRO_LINE_WIDTH_MULTIPLE
    )


def manim_fov(
    frame_height: float = MANIM_FRAME_HEIGHT,
    focal_distance: float = MANIM_FOCAL_DISTANCE,
) -> float:
    """Get the vertical field of view that reproduces Manim's 3-D projection.

    Manim's ``ThreeDCamera`` is a pinhole camera whose eye sits ``focal_distance``
    from the frame plane, and whose frame is ``frame_height`` units tall there. The
    angle it subtends is what Algan's :meth:`~algan.rendering.camera.Camera.set_fov`
    wants.

    Parameters
    ----------
    frame_height
        Height of the frame in world units, at the frame plane. Defaults to
        ``8.0``, Manim's value.
    focal_distance
        Distance from the eye to the frame plane in world units. Defaults to
        ``20.0``, Manim's value.

    Returns
    -------
    float
        The vertical field of view in **degrees** -- ``22.62`` for Manim's
        defaults.

    Examples
    --------
    ::

        import algan.manim as mn

        Scene.get_camera().set_fov(mn.manim_fov())
    """
    return math.degrees(2.0 * math.atan((frame_height * 0.5) / focal_distance))


def apply_manim_defaults(
    scene,
    *,
    camera: bool = True,
    shading: bool = True,
    background: bool = True,
    video_settings: bool = False,
    shape_defaults: bool = False,
    stroke_geometry: bool = True,
):
    """Point ``scene`` at Manim's defaults. See :meth:`~algan.scene.Scene.use_manim_defaults`."""
    from algan.rendering.lights import PointLight
    from algan.rendering.shaders.materials import ManimMaterial
    from algan.settings.video_settings import VideoSettings

    if video_settings:
        scene.set_video_settings(
            VideoSettings(
                resolution=MANIM_RESOLUTION,
                frames_per_second=MANIM_FRAMES_PER_SECOND,
            )
        )

    with Off():
        if background:
            scene.set_background(BLACK.clone())

        if camera:
            scene_camera = scene.get_camera()
            if scene_camera is not None:
                # Manim's eye sits focal_distance from the frame plane on its +z
                # side, which is OUTWARD * focal_distance -- already where an Algan
                # camera looks from.
                scene_camera.move_to(ORIGIN + OUTWARD * MANIM_FOCAL_DISTANCE)
                scene_camera.look_at(ORIGIN)
                scene_camera.set_fov(manim_fov())

        if shading:
            scene.clear_lights()
            PointLight(
                scene=scene,
                location=torch.tensor(MANIM_LIGHT_SOURCE).view(1, 1, 3),
                color=WHITE,
            ).spawn(animate=False)
            # What the default material reaches, engine by engine. Manim gates its
            # shading on ONE flag -- ``VMobject.shade_in_3d``, which its
            # ThreeDCamera checks in ``modified_rgbas`` before calling
            # ``get_shaded_rgb`` -- so a plain 2-D VMobject is Cairo-filled in its
            # own colour and a ThreeDVMobject or Surface is lit. Algan reads the
            # same flag: ``ManimMob`` carries it across, and a circuit that has it
            # renders as PN patches (3-D geometry a material and the lights reach)
            # rather than as an unlit analytic circuit. That is what makes a flat
            # ``Cube`` face shade -- geometry alone would call it planar and leave
            # it unlit where Manim shades it -- while a plain 2-D shape keeps
            # matching on both sides by getting no lighting at all.
            # ``ManimMaterial`` is what reproduces the law itself. The light above
            # is what it responds to, placed where Manim's own light sits; a Mob
            # given an explicitly lit material keeps it.
            SETTINGS.style.set(default_material=ManimMaterial())
            # Manim writes its colors straight out. Algan does too now -- this is
            # its own default since 2026-08-22 -- so this is belt-and-braces against
            # a Scene that turned tonemapping on. With the curve on, every fill
            # darkens by about 10/255 and white lands on 222, uniformly enough to
            # read as a color error rather than as a highlight roll-off. Off, a
            # flat fill comes out byte-identical to Manim's.
            SETTINGS.raytracing.set(tonemapping=False)
            # And Manim does its ARITHMETIC in that same display-referred space:
            # it composites alpha, antialiases and gradients sRGB values directly.
            # Algan's default is a linear working space, which is the physically
            # correct choice and the one three.js makes, but it puts a fill of
            # opacity a on a^(1/2.2) of the colour -- MAROON at 0.55 lands on
            # (150,71,87) where Manim puts (108,52,63). Matching Manim means
            # matching its space. Measured on the parity scene: whole-frame mean
            # 2.630 -> 1.636, the 0.55 fill 13.35 -> 0.49.
            #
            # This is a WHOLE-PIPELINE switch, not an alpha one: lighting goes
            # display-referred with it, which is the rest of what Manim does. It is
            # read through ti.static, so it costs nothing at runtime but compiles a
            # separate kernel variant -- the first render after the switch pays a
            # cold compile.
            #
            # And ti.static means it is really a PROCESS-START decision: flipping
            # it after a render has already compiled kernels leaves those kernels
            # in the old space while the host half of the pipeline moves, and the
            # two disagree by ~24/255. ``use_manim_defaults`` is documented as a
            # call you make once before building the Scene, which is before any
            # render and therefore safe; a process that renders more than once
            # wants ALGAN_LINEAR_COLOR=0 instead.
            SETTINGS.raytracing.set(linear_color_space=False)

        if stroke_geometry:
            # Where the stroke goes. Manim strokes an SVG path: the stroke is
            # centred on the outline and spends half its width outside. Algan lays
            # a filled shape's stroke entirely inside instead, which keeps glyphs
            # from fusing but leaves an imported Manim shape's silhouette half a
            # stroke width too small -- the double outline that shows up in a
            # parity diff of any stroked shape.
            #
            # And how wide it is. Algan's convention is the round "Manim's number
            # is twice Algan's"; the exact figure is 2.0202, so an imported stroke
            # is otherwise 1.01% too wide. Placement and width move together
            # because the export conversions invert this one, and a ratio applied
            # on the way in but not out would drift a round trip.
            SETTINGS.style.set(
                border_placement="centered",
                manim_stroke_width_ratio=manim_stroke_width_ratio(),
            )

        if shape_defaults:
            SETTINGS.style.set(shape_style_profile="manim")

    return scene
