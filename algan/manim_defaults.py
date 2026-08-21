"""Match a Scene's viewpoint, lighting and framing to Manim's defaults.

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

The z axis points the other way
-------------------------------
Manim's ``OUT`` is ``+z`` and Algan's is ``-z``, so the two screen bases are
mirror images: with the same numbers a Manim scene's near objects become Algan's
far ones. Reproducing Manim's picture therefore means mirroring *both* the
geometry and the camera in z, which is what
:func:`from_manim_coordinates` does and why
:attr:`Scene.manim_coordinates <algan.scene.Scene.manim_coordinates>` exists.
Mirroring only the camera would render the scene back-to-front; mirroring only
the geometry would render it from behind.
"""

from __future__ import annotations

import math

import torch

from algan.constants.color import BLACK, WHITE
from algan.constants.spatial import ORIGIN, OUT
from algan.settings import SETTINGS

#: Height of Manim's frame in world units. Manim pins the vertical extent and
#: derives the width from the aspect ratio, exactly as Algan's vertical ``fov``
#: does (``manim.constants``/``manim._config``: ``frame_height``).
MANIM_FRAME_HEIGHT = 8.0

#: Distance from Manim's ``ThreeDCamera`` eye to the frame plane, in world units
#: (``ThreeDCamera.focal_distance``). Its projection scales a point by
#: ``f / (f - z)``, which is a pinhole camera this far from ``z = 0``.
MANIM_FOCAL_DISTANCE = 20.0

#: Manim's ``ThreeDCamera.light_source_start_point``, in **Manim** coordinates
#: (``9 * DOWN + 7 * LEFT + 10 * OUT``). Pass it through
#: :func:`from_manim_coordinates` before handing it to an Algan light.
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


def _mirror_z(points):
    """Negate the z component of a point, tensor of points, or 3-sequence."""
    tensor = torch.as_tensor(points, dtype=torch.get_default_dtype())
    flip = torch.ones_like(tensor)
    flip[..., 2] = -1.0
    return tensor * flip


def from_manim_coordinates(points):
    """Convert points from Manim's coordinate system into Algan's.

    Manim's ``OUT`` is ``+z`` and Algan's is ``-z``, so the conversion is a mirror
    in z. x and y are untouched: both engines put ``+x`` to the right and ``+y``
    up.

    The mirror is an involution, so this and :func:`to_manim_coordinates` do the
    same arithmetic; they are separate names so a call site says which way it is
    going.

    Parameters
    ----------
    points
        A point or an array of points whose last dimension is 3. Anything
        ``torch.as_tensor`` accepts -- a tensor, a NumPy array, or a sequence.

    Returns
    -------
    :class:`torch.Tensor`
        The converted points, same shape as the input.

    See Also
    --------
    :func:`to_manim_coordinates` : The other direction.
    """
    return _mirror_z(points)


def to_manim_coordinates(points):
    """Convert points from Algan's coordinate system into Manim's.

    The inverse of :func:`from_manim_coordinates`, and the same arithmetic -- a
    mirror in z.

    Parameters
    ----------
    points
        A point or an array of points whose last dimension is 3.

    Returns
    -------
    :class:`torch.Tensor`
        The converted points, same shape as the input.

    See Also
    --------
    :func:`from_manim_coordinates` : The other direction.
    """
    return _mirror_z(points)


def apply_manim_defaults(
    scene,
    *,
    camera: bool = True,
    shading: bool = True,
    background: bool = True,
    coordinates: bool = True,
    video_settings: bool = False,
    shape_defaults: bool = False,
):
    """Point ``scene`` at Manim's defaults. See :meth:`~algan.scene.Scene.use_manim_defaults`."""
    from algan.rendering.lights import PointLight
    from algan.rendering.shaders.material_shaders import basic_material_shader
    from algan.settings.video_settings import VideoSettings

    if coordinates:
        scene.manim_coordinates = True

    if video_settings:
        scene.set_video_settings(
            VideoSettings(
                resolution=MANIM_RESOLUTION,
                frames_per_second=MANIM_FRAMES_PER_SECOND,
            )
        )

    if background:
        scene.set_background_color(BLACK.clone())

    if camera:
        scene_camera = scene.get_camera()
        if scene_camera is not None:
            # Manim's eye sits focal_distance from the frame plane on its +z
            # side; mirrored into Algan that is OUT * focal_distance, which is
            # already where an Algan camera looks from.
            scene_camera.move_to(ORIGIN + OUT * MANIM_FOCAL_DISTANCE)
            scene_camera.look_at(ORIGIN)
            scene_camera.set_fov(manim_fov())

    if shading:
        scene.clear_light_sources()
        PointLight(
            scene=scene,
            location=from_manim_coordinates(MANIM_LIGHT_SOURCE).view(1, 1, 3),
            color=WHITE,
        ).spawn(animate=False)
        # Manim's renderer draws a VMobject as flat colour -- it applies no
        # lighting at all -- so an unlit default is what actually reproduces it.
        # The light above still shades any Mob given an explicitly lit material,
        # and it shades it from where Manim's own light sits.
        SETTINGS.style.set(default_shader=basic_material_shader)
        # Manim writes its colours straight out; Algan tonemaps by default, which
        # darkens every fill by about 10/255 -- uniformly, so it reads as a colour
        # error rather than as a highlight roll-off. Off, a flat fill comes out
        # byte-identical to Manim's.
        SETTINGS.raytracing.set(tonemapping=False)

    if shape_defaults:
        SETTINGS.style.set(shape_style_profile="manim")

    return scene
