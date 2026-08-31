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

        from algan.manim_defaults import manim_fov

        Scene.get_camera().set_fov(manim_fov())
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
        # What the default material reaches, engine by engine. Manim applies
        # no lighting at all to a flat 2-D VMobject -- Cairo fills it in its
        # own color -- and Algan's 2-D content never consults this setting
        # (circuits and images are drawn unlit by construction), so flat
        # 2-D matches on both sides without it. The setting's audience is
        # 3-D geometry, which is Manim's ``ThreeDVMobject`` territory: there
        # Manim DOES shade, per light via ``get_shaded_rgb``, and that is
        # exactly what ManimMaterial reproduces -- so an imported 3-D mob with
        # no material of its own shades the way Manim would have shaded it.
        # The light above is what it responds to, placed where Manim's own
        # light sits; a Mob given an explicitly lit material keeps it.
        SETTINGS.style.set(default_material=ManimMaterial())
        # Manim writes its colors straight out. Algan does too now -- this is
        # its own default since 2026-08-22 -- so this is belt-and-braces against
        # a Scene that turned tonemapping on. With the curve on, every fill
        # darkens by about 10/255 and white lands on 222, uniformly enough to
        # read as a color error rather than as a highlight roll-off. Off, a
        # flat fill comes out byte-identical to Manim's.
        SETTINGS.raytracing.set(tonemapping=False)

    if shape_defaults:
        SETTINGS.style.set(shape_style_profile="manim")

    return scene
