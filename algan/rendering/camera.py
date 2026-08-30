"""The Scene's camera.

:class:`Camera` is a :class:`~algan.animatable_base.mob.Mob`, so it is moved,
rotated and animated with exactly the methods every other Mob uses, inside the
same animation contexts -- there is no separate camera-animation API. Camera
moves usually want ``easing=easings.identity``, since easing in and out of a
pan reads as a wobble.

It carries the projection: perspective or orthographic, field of view in degrees,
and near/far planes. :meth:`Camera.center_on` frames a
given Mob, and :meth:`~algan.animatable_base.mob_orientation.MobOrientationMixin.look_at`
aims at a point.

Each Scene owns its own camera. The renderer consumes an immutable camera and
light snapshot per frame batch, which is what lets batch preparation for the next
batch run on a worker thread while the current one renders.

See :doc:`/advanced_user_tutorials/cameras`.
"""

from __future__ import annotations

import math

import torch.nn.functional as F

from algan.animatable_base.mob import Mob
from algan.animation_timeline.animation_contexts import Off, Sync
from algan.constants.spatial import *  # CAMERA_ORIGIN
from algan.errors import AlganConfigurationError
from algan.geometry.geometry import intersect_line_with_plane_colinear
from algan.utils.tensor_utils import (
    broadcast_gather,
    dot_product,
    unsquish,
)


class Camera(Mob):
    """The Scene's viewpoint. A Mob, so it animates like one.

    Positioning
    -----------
    Use :meth:`~algan.animatable_base.mob_movement.MobMovementMixin.move_to` (or
    ``move`` for a relative step) -- they set the camera's own location and carry
    its screen along. Do **not** use
    :meth:`~algan.animatable_base.mob_layout.MobLayoutMixin.move_center_to`: that
    centres a *bounding box*, and a Camera's box spans it and its internal screen
    plane, so the camera lands half the screen distance behind where you asked.

    Field of view and aspect ratio
    ------------------------------
    ``fov`` / :meth:`set_fov` are **vertical**, matching Three.js's
    ``PerspectiveCamera``. The horizontal field of view is derived from it and
    the output aspect ratio, so changing the resolution's shape changes how wide
    the camera sees while the vertical stays put. The default 53 degree vertical
    fov gives about 82 degrees horizontally at 16:9, but about 119 degrees on a
    3.4:1 banner -- wide enough that an off-axis sphere is visibly projected as an
    ellipse (stretched by ``1 / cos(angle off axis)``, so ~1.9x at that frame's
    edge). That is correct perspective, not a bug, but it is rarely what a wide
    still is after: for the near-orthographic look of a long lens, narrow the fov
    and pull back by the same factor, keeping ``distance * tan(fov / 2)``
    constant::

        camera = Scene.get_camera()
        camera.set_fov(math.degrees(2 * math.atan(3.5 / 70)))
        camera.move_to(OUT * 70)  # was OUT * 7, so 10x the distance

    Alternatively :meth:`set_near_orthographic` flattens it almost completely --
    it is an approximation, not true parallel projection; see
    :doc:`/advanced_user_tutorials/renderer_limitations`.
    """

    def __init__(
        self,
        orthographic=False,
        screen_distance=5,
        screen_half_height=2.5,
        fov=None,
        near=0.0,
        far=0.0,
        *args,
        **kwargs,
    ):
        # fov (vertical, degrees) is an alternative way to specify the
        # perspective: it fixes the camera-to-screen distance for the given
        # screen size. near/far are the clip distances (0 disables each);
        # near is a plane, far is a distance along the ray.
        screen_distance = self._validated_positive("screen_distance", screen_distance)
        screen_half_height = self._validated_positive(
            "screen_half_height", screen_half_height
        )
        if fov is not None:
            fov = self._validated_fov(fov)
            screen_distance = screen_half_height / math.tan(math.radians(fov) * 0.5)
        self._near = self._validated_clip("near", near)
        self._far = self._validated_clip("far", far)
        self._validate_clip_order(self._near, self._far)
        # Camera ownership is managed by Scene; tolerate the common generic
        # Mob kwargs without passing duplicates into the base constructor.
        kwargs.pop("add_to_scene", None)
        kwargs.pop("init", None)
        super().__init__(*args, add_to_scene=False, init=False, **kwargs)
        self.animatable_attrs.remove("color")
        with Off(animation_manager=self.animation_manager):
            self.orthographic = orthographic
            self.screen = Mob(
                scene=self.scene,
                location=self.location + screen_distance * self.get_forward_direction(),
                add_to_scene=False,
                init=False,
            )
            self.screen.scale(
                torch.tensor((1 / screen_half_height, 1 / screen_half_height, 1))
            )
            self.screen_half_height = screen_half_height
            self.screen.is_primitive = True
            self.is_primitive = True
            self.add_children(self.screen)
            self.screen_distance = screen_distance
            self.corner_x_coords = torch.tensor([-1, -1, 1, 1]).view(-1, 1, 1, 1)
            self.corner_y_coords = torch.tensor([-1, 1, 1, -1]).view(-1, 1, 1, 1)
            self.spawn(animate=False)
        if orthographic:
            with Off(animation_manager=self.animation_manager):
                self.set_near_orthographic()

    @property
    def pixel_height(self):
        """Current normalized screen height represented by one output pixel."""
        return 2.0 / self.scene.num_pixels_screen_height

    @property
    def pixel_width(self):
        """Current normalized screen width represented by one output pixel."""
        return 2.0 / self.scene.num_pixels_screen_width

    @staticmethod
    def _as_finite_number(name, value):
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise AlganConfigurationError(f"{name} must be a finite number") from exc
        if not math.isfinite(result):
            raise AlganConfigurationError(f"{name} must be a finite number")
        return result

    @classmethod
    def _validated_positive(cls, name, value):
        result = cls._as_finite_number(name, value)
        if result <= 0:
            raise AlganConfigurationError(f"{name} must be positive")
        return result

    @classmethod
    def _validated_fov(cls, fov):
        value = cls._as_finite_number("fov", fov)
        if not math.isfinite(value) or not 0.0 < value < 180.0:
            raise AlganConfigurationError("fov must be finite and in (0, 180)")
        return value

    @classmethod
    def _validated_clip(cls, name, value):
        value = cls._as_finite_number(name, value)
        if value < 0:
            raise AlganConfigurationError(
                f"{name} clip distance must be finite and non-negative"
            )
        return value

    @staticmethod
    def _validate_clip_order(near, far):
        if near > 0 and far > 0 and near >= far:
            raise AlganConfigurationError(
                "near clip distance must be less than far clip distance"
            )

    def set_near_orthographic(self, distance=1e5):
        """Flatten perspective by moving the camera far from its screen."""
        distance = self._validated_positive("distance", distance)
        self.orthographic = True
        return self._set_distance_to_screen(distance, preserve_mode=True)

    def get_fov(self):
        """The camera's vertical field of view in degrees (like Three.js's
        ``PerspectiveCamera.fov``), derived from the screen size and the
        camera-to-screen distance.
        """
        d = (self.screen.location - self.location).norm(p=2, dim=-1).flatten()[0].item()
        return math.degrees(2.0 * math.atan(self.screen_half_height / max(d, 1e-9)))

    def set_fov(self, fov):
        """Set the vertical field of view (degrees). The camera stays where it
        is; its screen moves along the forward axis so that the given angle is
        spanned (small fov = telephoto, large fov = wide angle). Animatable.

        The horizontal field of view follows from this and the output aspect
        ratio, so a wide frame sees much wider than this angle -- see the
        :class:`Camera` class docstring.

        Parameters
        ----------
        fov
            Vertical field of view in degrees, in (0, 180).
        """
        fov = self._validated_fov(fov)
        d = self.screen_half_height / math.tan(math.radians(fov) * 0.5)
        self.orthographic = False
        self.screen.move_to(self.location + self.get_forward_direction() * d)
        return self

    fov = property(get_fov, set_fov)

    def get_near(self):
        """Near clip distance (world units from the camera along its forward
        axis); geometry closer than this is not rendered. 0 = disabled.
        """
        return getattr(self, "_near", 0.0)

    def set_near(self, near):
        """Set the near clip plane distance (0 disables near clipping)."""
        near = self._validated_clip("near", near)
        self._validate_clip_order(near, self.far)
        self._near = near
        return self

    near = property(get_near, set_near)

    def get_far(self):
        """Far clip distance (world units of ray travel from the camera);
        geometry farther than this shows the background/environment instead.
        0 = disabled.
        """
        return getattr(self, "_far", 0.0)

    def set_far(self, far):
        """Set the far clip distance (0 disables far clipping)."""
        far = self._validated_clip("far", far)
        self._validate_clip_order(self.near, far)
        self._far = far
        return self

    far = property(get_far, set_far)

    def set_distance_to_screen(self, distance):
        """Moves the camera focus to be the given distance away from its screen, thereby changing the perspective.

        Parameters
        ----------
        distance
            The camera focus will be set to be this distance away from the screen.
        """
        return self._set_distance_to_screen(distance, preserve_mode=False)

    def _set_distance_to_screen(self, distance, *, preserve_mode):
        distance = self._validated_positive("distance", distance)
        if not preserve_mode:
            self.orthographic = False
        self.set_non_recursive(
            location=self.screen.location - self.get_forward_direction() * distance
        )
        return self

    def set_euler_angles(
        self,
        yaw: float | torch.Tensor,
        pitch: float | torch.Tensor,
        roll: float | torch.Tensor,
        *,
        degrees: bool = True,
    ):
        """Point the camera using three Euler rotations about the origin.

        The rotations are applied about the world x, y and z axes and are performed
        together, so the camera swings around the origin rather than turning in place.
        Note these are *added* to the camera's current orientation rather than
        replacing it, despite the name.

        Animation
        ---------
        Recorded as an animation: all three rotations run inside a :class:`~.Sync`,
        over the current context's duration (1 second by default).

        Parameters
        ----------
        yaw
            Rotation about the world x axis (``RIGHT``).
        pitch
            Rotation about the world y axis (``UP``).
        roll
            Rotation about the world z axis (``OUTWARD``).
        degrees
            Whether the three angles are in degrees. Defaults to True; pass False
            to give them in radians.

        Returns
        -------
        :class:`~.Camera`
            This camera, so calls can be chained.
        """
        with Sync(animation_manager=self.animation_manager):
            self.rotate(yaw, RIGHT, about=ORIGIN, degrees=degrees)
            self.rotate(pitch, UP, about=ORIGIN, degrees=degrees)
            self.rotate(roll, OUTWARD, about=ORIGIN, degrees=degrees)
        return self

    def _get_render_screen_basis(self):
        """Per-frame screen basis used by the renderers to project the scene.

        Derived from the camera's own basis -- which is purely rotational,
        the camera mob is never scaled -- with the screen's in-plane scale
        applied along the screen's *local* axes. The screen mob's stored
        basis cannot be used directly: its non-uniform scale is applied
        along world axes (basis = rotation @ scale), so once the camera
        rotates the rows skew and shrink/stretch, which makes the projection
        anisotropic -- the image visibly squashes with the orbit angle
        (e.g. a sphere renders as an ellipse).
        """
        basis = unsquish(self.basis, -1, 3).clone()
        basis[..., :2, :] = basis[..., :2, :] / self.screen_half_height
        return basis

    def _retroactive_center(self, mob, **kwargs):
        """Frame a Mob, with the camera move recorded earlier in the video.

        The same framing as :meth:`~.Camera.center_on`, but
        recorded at the camera's retroactive timestamp -- so the camera has already
        arrived by the time the Mob does its thing, instead of chasing it.

        Animation
        ---------
        Recorded as an animation, inserted at the retroactive timestamp rather than the
        current one.

        Parameters
        ----------
        mob
            The Mob to frame.
        **kwargs
            Passed to :meth:`~.Camera.center_on` -- notably
            ``buffer_portion``.

        Returns
        -------
        :class:`~.Camera`
            This camera, so calls can be chained.
        """
        with self.retroactive():
            self.center_on(mob, **kwargs)
        return self

    def center_on(self, mob, buffer_portion: float = 0.7):
        """Move the camera so a Mob fills the frame, centred.

        The camera slides sideways to centre the Mob and in or out until the Mob just
        fits, leaving ``buffer_portion`` of margin. It does not rotate, so the viewing
        angle you set is preserved.

        Animation
        ---------
        Recorded as an animation: the moves run together inside a :class:`~.Sync`, over
        the current context's duration (1 second by default). The Mob's extent is
        measured when the call is recorded, so a Mob that changes size afterwards will
        not stay framed.

        Parameters
        ----------
        mob
            The Mob to frame.
        buffer_portion
            Extra margin around the Mob, as a fraction of its size. Defaults to
            ``0.7``, i.e. the framed area is 1.7 times the Mob's extent.

        Returns
        -------
        :class:`~.Camera`
            This camera, so calls can be chained.
        """
        f = self.get_forward_direction()
        r = self.get_right_direction()
        u = self.get_up_direction()
        # mob_boundary_points = [mob.get_boundary_point(_) + _ * buffer for _ in [u, r, -u, -r]]
        mob_boundary_points = [mob.get_boundary_point(_) for _ in [-r, u, r, -u]]
        mob_boundary_points = torch.stack(mob_boundary_points)
        # mobl = sum(mob_boundary_points) / len(mob_boundary_points)
        mobl = 0.5 * ((mob_boundary_points).amax(0) + (mob_boundary_points).amin(0))
        mob_boundary_points = (mob_boundary_points - mobl) * (1 + buffer_portion) + mobl
        selfl = self.location

        with Sync(animation_manager=self.animation_manager):
            self.move_to(mobl - f * dot_product(mobl - selfl, f))
            selfl = self.location

            corner_rays = F.normalize(self.get_corner_pixels() - selfl, dim=-1, p=2)
            edge_plane_rays = torch.stack(
                (corner_rays, torch.cat((corner_rays[1:], corner_rays[:1]))), 1
            )
            up_plane = edge_plane_rays[1]
            right_plane = edge_plane_rays[0]

            vertical_move, vertical_dist = intersect_line_with_plane_colinear(
                -f, selfl, up_plane[0], up_plane[1], mob_boundary_points[1]
            )
            horizontal_move, horizontal_dist = intersect_line_with_plane_colinear(
                -f, selfl, right_plane[0], right_plane[1], mob_boundary_points[0]
            )

            return self.move(
                (mob_boundary_points[1] - vertical_move)
                if vertical_dist <= horizontal_dist
                else (mob_boundary_points[0] - horizontal_move)
            )

    def project_point_onto_screen_border(
        self, point: torch.Tensor, direction: torch.Tensor
    ) -> torch.Tensor:
        """Find where a point would leave the frame travelling in a direction.

        Casts from ``point`` along ``direction`` and returns the intersection with the
        edge of the visible frustum. This is what the screen-edge placement methods use
        to know where "against the edge" is.

        Parameters
        ----------
        point
            Starting point, shape ``(*, 3)``.
        direction
            Direction to travel, shape ``(*, 3)``, e.g. ``RIGHT`` or ``UP``.

        Returns
        -------
        torch.Tensor
            The point on the frame border, shape ``(*, 3)``.
        """
        corner_rays = F.normalize(
            (self.get_corner_pixels()) - self.location, dim=-1, p=2
        )
        edge_plane_rays = torch.stack(
            (corner_rays, torch.cat((corner_rays[1:], corner_rays[:1])))
        )
        intersection_points, intersection_distances = (
            torch.stack(_)
            for _ in zip(
                *[
                    intersect_line_with_plane_colinear(
                        direction,
                        self.location,
                        edge_plane_rays[0, i],
                        edge_plane_rays[1, i],
                        point,
                    )
                    for i in range(edge_plane_rays.shape[1])
                ]
            )
        )
        intersection_distances = intersection_distances.nan_to_num(
            nan=1e12, posinf=1e12, neginf=1e12
        )
        m = (
            intersection_distances.sign()
            == torch.cat(
                (intersection_distances[2:], intersection_distances[:2])
            ).sign()
        ).float()
        diff_sign_intersection_distances = intersection_distances.clone()
        diff_sign_intersection_distances[diff_sign_intersection_distances < 0] = 1e12
        intersection_distances = (
            intersection_distances.abs() * m
            + (1 - m) * diff_sign_intersection_distances
        )
        closest_ind = intersection_distances.argmin(0, keepdim=True)
        closest_point = broadcast_gather(
            intersection_points, 0, closest_ind, keepdim=False
        )
        return closest_point

    def get_corner_pixels(self) -> torch.Tensor:
        """Get the four corners of the visible frame, in world space.

        The corners of the camera's screen plane, accounting for the current aspect
        ratio, ordered around the frame. Screen-relative layout is built on these.

        Returns
        -------
        torch.Tensor
            The four corner points, shape ``(4, *, 3)``.
        """
        b = unsquish(self.screen.basis, -1, 3)
        b = b / b.norm(p=2, dim=-1, keepdim=True).square().clamp_min(1e-6)
        aspect_ratio = (
            self.scene.video_settings.resolution[0]
            / self.scene.video_settings.resolution[1]
        )
        return (
            self.screen.location
            + b[..., 0, :] * self.corner_x_coords * aspect_ratio
            + b[..., 1, :] * self.corner_y_coords
        )
