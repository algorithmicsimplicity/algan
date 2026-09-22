"""Live secondary cameras displayed on ordinary textured surfaces."""

from __future__ import annotations

import math
import operator
from collections.abc import Iterable
from typing import Any

import torch

from algan.animatable_base.mob import Mob
from algan.animation_timeline.animation_contexts import Off, active_scene_for_new_mob
from algan.errors import AlganConfigurationError
from algan.mobs.image_mob import ImageMob
from algan.rendering.camera import Camera

__all__ = ["CameraView"]


def _capture_resolution(resolution):
    try:
        values = tuple(resolution)
        if len(values) != 2 or any(isinstance(v, bool) for v in values):
            raise ValueError
        result = tuple(operator.index(v) for v in values)
        if min(result) <= 0:
            raise ValueError
        return result
    except (TypeError, ValueError) as exc:
        raise AlganConfigurationError(
            "CameraView resolution must be two positive integers (width, height)"
        ) from exc


class CameraView(ImageMob):
    """A live camera image on a movable, unlit rectangular surface.

    Animate ``view.camera`` to change the source region and animate ``view`` to
    move, resize, rotate or fade its display. Both use the owning Scene's timeline.
    Multiple views can show different perspectives of the same animated objects.
    All camera-view displays and their children are omitted from secondary
    captures, preventing recursive pictures. Other scene geometry, lights and
    the background are rendered normally.

    Animation
    ---------
    Construction takes effect immediately. Call ``spawn()`` to show the display;
    subsequent transforms animate over the current context's runtime (1 second
    by default). The camera is already spawned and can be animated independently.
    Use ``Off()`` for initial positioning. Configure the resolution and exclusions
    at construction; they are not animated.

    Parameters
    ----------
    camera
        A camera belonging to this Scene. Defaults to ``None``, creating an
        independent camera with the main camera's current position and lens.
    resolution
        Capture size in pixels, ``(width, height)``. Its aspect ratio also sets
        the display's initial proportions. Defaults to ``(640, 360)``.
    height
        Initial display height in world units. Defaults to ``3``.
    exclude
        Mobs, including their descendants, to omit from this capture while
        keeping them visible in the main view. Defaults to an empty iterable.
    **kwargs
        Passed to :class:`~.ImageMob`, notably ``scene`` and ``location``.

    Attributes
    ----------
    camera
        The secondary :class:`~.Camera`; its transforms and field of view are
        independent of the display's transforms.

    Raises
    ------
    :class:`.AlganConfigurationError`
        If the resolution or height is invalid, or a camera or excluded Mob
        belongs to another Scene.

    Examples
    --------
    Keep a close-up beside the moving subject:

    .. algan:: Example1CameraView

        from algan import *

        dot = Circle(radius=0.3, color=YELLOW).spawn()
        view = CameraView(resolution=(400, 300), height=2.5)
        with Off():
            view.focus_on(dot, buffer_portion=1)
            view.move_to(RIGHT * 3 + UP * 1.5)
        view.spawn()
        with Sync():
            dot.move(LEFT)
            view.camera.move(LEFT)
        Scene.save_video()
    """

    _is_camera_view = True

    def __init__(
        self,
        camera: Camera | None = None,
        *,
        resolution: tuple[int, int] = (640, 360),
        height: float = 3.0,
        exclude: Iterable[Mob] = (),
        **kwargs: Any,
    ) -> None:
        scene = kwargs.get("scene")
        if scene is None:
            scene = (
                camera.scene
                if isinstance(camera, Camera)
                else active_scene_for_new_mob()
            )
        kwargs["scene"] = scene
        resolution = _capture_resolution(resolution)
        try:
            height = float(height)
            if not math.isfinite(height) or height <= 0:
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise AlganConfigurationError(
                "CameraView height must be positive and finite"
            ) from exc
        if camera is not None and (
            not isinstance(camera, Camera) or camera.scene is not scene
        ):
            raise AlganConfigurationError(
                "CameraView camera must be an Algan Camera in the same Scene"
            )
        exclude = tuple(exclude)
        if any(not isinstance(mob, Mob) or mob.scene is not scene for mob in exclude):
            raise AlganConfigurationError(
                "CameraView exclusions must be Mobs in the same Scene"
            )
        if camera is None:
            main = scene.camera
            with Off(animation_manager=scene.animation_manager):
                camera = Camera(
                    scene=scene,
                    location=main.location,
                    basis=main.basis,
                    screen_distance=float(
                        (main.screen.location - main.location).norm()
                    ),
                    screen_half_height=main.screen_half_height,
                    near=main.near,
                    far=main.far,
                )
                camera.screen.location = main.screen.location
                camera.screen.basis = main.screen.basis
                camera.orthographic = main.orthographic
        self.camera = camera
        self.capture_resolution = resolution
        self.capture_exclude = exclude
        # A tiny placeholder supplies UVs without authoring a full image in the
        # timeline. Captured pixels belong to a render pass, never to this Mob.
        super().__init__(torch.ones((2, 2, 4)), **kwargs)
        with Off(animation_manager=self.animation_manager):
            self.scale(
                torch.tensor((height * resolution[0] / resolution[1], height, 1.0))
            )

    def focus_on(self, mob: Mob, buffer_portion: float = 0.7) -> CameraView:
        """Frame a Mob in the secondary camera without moving the display.

        Animation
        ---------
        Record a camera move over the current context's runtime (1 second by
        default). Use ``with Seq(runtime=3): view.focus_on(mob)`` to change its
        duration, or ``Off()`` for an immediate move. The target's extent is
        measured when called; later target motion does not automatically follow.
        The display need not be spawned.

        Parameters
        ----------
        mob
            The Mob to frame, belonging to this Scene.
        buffer_portion
            Extra margin as a fraction of the Mob's extent. Defaults to ``0.7``,
            making the framed area 1.7 times its size. Must be non-negative.

        Returns
        -------
        :class:`~.CameraView`
            This Mob, so calls can be chained.

        Raises
        ------
        :class:`.AlganConfigurationError`
            If the target belongs to another Scene or the margin is invalid.

        Examples
        --------
        Zoom from a whole group to one member:

        .. algan:: Example1CameraViewFocusOn

            from algan import *

            square = Square().spawn()
            view = CameraView().spawn()
            with Off():
                view.move_to(RIGHT * 3)
            view.focus_on(square, buffer_portion=0.2)
            Scene.save_video()
        """
        if not isinstance(mob, Mob) or mob.scene is not self.scene:
            raise AlganConfigurationError(
                "CameraView focus target must belong to the same Scene"
            )
        try:
            margin = float(buffer_portion)
            if not math.isfinite(margin) or margin < 0:
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise AlganConfigurationError(
                "buffer_portion must be finite and non-negative"
            ) from exc
        self.camera._center_on(
            mob, margin, self.capture_resolution[0] / self.capture_resolution[1]
        )
        return self

    def _get_camera_view_primitives(self, texture):
        primitive = ImageMob.get_render_primitives(self)
        if primitive is not None:
            primitive.texture_map = texture
            primitive.texture_lerp = None
            primitive.texture_u8_ok = False
            primitive.texture_opacity = self.opacity.reshape(-1)
        return primitive

    def _get_memory_used_per_timestep(self):
        # Captured windows live on the host; allow for the texture packing and
        # color decode copies alongside the ordinary surface geometry.
        w, h = self.capture_resolution
        return super()._get_memory_used_per_timestep() + w * h * 5 * 4 * 4
