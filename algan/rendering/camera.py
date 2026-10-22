"""The Scene's camera.

:class:`Camera` is a :class:`~algan.animatable_base.mob.Mob`, so it is moved,
rotated and animated with exactly the methods every other Mob uses, inside the
same animation contexts -- there is no separate camera-animation API. Camera
moves usually want ``easing=easings.identity``, since easing in and out of a
pan reads as a wobble.

It carries a perspective projection with a field of view in degrees, or a true
orthographic one -- parallel rays from a frame of fixed size, selected with
:meth:`Camera.set_orthographic` and undone with :meth:`Camera.set_perspective`,
either of which can be animated -- and near/far clipping distances.
:meth:`Camera.center_on` frames a given Mob, and
:meth:`~algan.animatable_base.mob_orientation.MobOrientationMixin.look_at`
aims at a point. Its animatable ``aperture`` and ``focus_distance`` describe a
thin lens, which the path tracer (``samples_per_pixel > 1``) renders as depth of
field; :meth:`Camera.focus_at` racks the focus onto a Mob.

The projection is one family (``algan.rendering.projection``): a pinhole, a
true orthographic camera, and every blend between the two that an animated
switch passes through, carried to the renderer as a per-frame
``(persp, spread)`` pair beside the camera's position and screen.

Each Scene owns its own camera. The renderer consumes an immutable camera and
light snapshot per frame batch, which is what lets batch preparation for the next
batch run on a worker thread while the current one renders.

See :doc:`/advanced_user_tutorials/cameras`.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from algan.animatable_base.animatable import animated_function
from algan.animatable_base.mob import Mob
from algan.animation_timeline.animation_contexts import Off, Sync
from algan.constants.spatial import *  # CAMERA_ORIGIN
from algan.errors import AlganConfigurationError
from algan.geometry.geometry import intersect_line_with_plane_colinear
from algan.rendering.projection import effective_depth
from algan.utils.tensor_utils import (
    broadcast_gather,
    cast_to_direction,
    cast_to_tensor,
    dot_product,
    squish,
    unsquish,
)

#: A camera's own default orientation, and the one Mob orientation that is not
#: :data:`~algan.constants.spatial.DEFAULT_BASIS`. A Mob's forward axis is the
#: way it faces, so it starts ``OUTWARD``, towards the viewer; a camera's
#: forward axis is the way it *looks*, so it starts ``INWARD``, at the scene --
#: which is what puts its screen between it and the ``ORIGIN`` rather than
#: behind it.
_CAMERA_BASIS = torch.stack((RIGHT, UP, INWARD))

#: The thin-lens attributes, in the order the render snapshot packs them.
_LENS_ATTRS = ("aperture", "focus_distance")

#: The projection's animatable state, in the order the render snapshot packs
#: it (``algan.rendering.projection``): how much of a pinhole's perspective is
#: left -- 1 for a pinhole, 0 for orthographic -- and half the height, in world
#: units, of the patch of the camera plane the rays start from -- 0 for a
#: pinhole, whose rays all start at the eye, and the frame's half-height for an
#: orthographic camera. A linear tween between any two such states is itself a
#: camera, which is what lets the switch animate. Private because the public
#: spelling is :meth:`Camera.set_orthographic` / :meth:`Camera.set_perspective`,
#: which keep the pair consistent.
_PROJECTION_ATTRS = ("_perspective_weight", "_parallel_half_height")

#: Default distance of the plane in focus: the default Scene camera's distance
#: from ``ORIGIN`` (``algan.manim_defaults.MANIM_FOCAL_DISTANCE``), so opening
#: the aperture of a camera that has not been moved keeps ``ORIGIN`` sharp.
DEFAULT_FOCUS_DISTANCE = 20.0

#: Smallest focus distance the renderer is handed. ``focus_at`` a target at or
#: behind the camera plane clamps here rather than inverting the lens.
_MIN_FOCUS_DISTANCE = 1e-4


def _validated_lens_value(name, value):
    """Validate an ``aperture`` / ``focus_distance`` write.

    Tensors are expected, not a defensive nicety: a materialized row is a
    ``[T, 1, 1]`` tensor and ``Animatable.__deepcopy__`` copies every animatable
    attribute through its setter.
    """
    if torch.is_tensor(value):
        if not bool(torch.isfinite(value).all()):
            raise AlganConfigurationError(f"{name} must be a finite number")
        if name == "aperture" and bool((value < 0).any()):
            raise AlganConfigurationError("aperture must be at least 0.0")
        if name == "focus_distance" and bool((value <= 0).any()):
            raise AlganConfigurationError("focus_distance must be greater than 0.0")
        return value
    if isinstance(value, bool):
        raise AlganConfigurationError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AlganConfigurationError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise AlganConfigurationError(f"{name} must be a finite number")
    if name == "aperture" and result < 0:
        raise AlganConfigurationError("aperture must be at least 0.0")
    if name == "focus_distance" and result <= 0:
        raise AlganConfigurationError("focus_distance must be greater than 0.0")
    return result


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
    still is after: for the flatter look of a long lens, narrow the fov and pull
    back by the same factor, keeping ``distance * tan(fov / 2)`` constant::

        camera = Scene.get_camera()
        camera.set_fov(math.degrees(2 * math.atan(3.5 / 70)))
        camera.move_to(OUT * 70)  # was OUT * 7, so 10x the distance

    Or remove perspective altogether with :meth:`set_orthographic`.

    Orthographic projection
    -----------------------
    :meth:`set_orthographic` makes the camera truly orthographic: every ray runs
    along the forward axis, starting from the plane through the camera
    perpendicular to it, so a Mob's size on screen does not depend on its
    distance and parallel lines stay parallel. The frame is a rectangle of fixed
    world size -- by default the one the perspective view had on the plane
    through ``ORIGIN``, which from the default camera is Manim's 8 units tall.
    Only what lies in front of the camera's plane is seen, so moving the camera
    along its forward axis changes nothing else; :meth:`center_on` frames a Mob
    by resizing the frame rather than by moving. :meth:`set_perspective`
    switches back, and either switch animates as a dolly zoom: the frame on one
    plane holds still while the perspective drains away or returns.

    Depth of field
    --------------
    The camera is a pinhole until ``aperture`` is opened. With an aperture the
    path tracer (``SETTINGS.raytracing.samples_per_pixel > 1``) renders a thin
    lens: points ``focus_distance`` in front of the camera are sharp, and
    everything nearer or farther blurs by a disk that grows with its distance
    from that plane. The deterministic renderer (``samples_per_pixel == 1``)
    has no lens model, so a render there with an open aperture reports depth of
    field as unsupported -- see
    :ref:`renderer-capabilities <renderer-capabilities>`.

    Parameters
    ----------
    orthographic
        Whether to start with the orthographic projection of
        :meth:`set_orthographic`, at its default frame. Defaults to False, a
        perspective camera.
    screen_distance
        Distance from the camera to its screen plane, in world units. Defaults
        to ``5``. Ignored when ``fov`` is given.
    screen_half_height
        Half the height of the screen plane, in world units. Defaults to
        ``2.5``.
    fov
        Vertical field of view in degrees, in (0, 180); fixes
        ``screen_distance`` for the given screen size. Defaults to ``None``,
        meaning ``screen_distance`` decides.
    near
        Near clip plane distance in world units along the forward axis. Defaults
        to ``0.0``, meaning no near clipping.
    far
        Far clip distance in world units along each ray. Defaults to ``0.0``,
        meaning no far clipping.
    aperture
        Initial lens aperture diameter, in world units. Defaults to ``0.0``,
        a pinhole: everything in focus.
    focus_distance
        Initial distance of the plane in focus, in world units along the
        camera's forward axis. Defaults to ``20.0``, the distance from the
        default Scene camera to ``ORIGIN``.

    Attributes
    ----------
    aperture
        Diameter of the lens, in world units; a finite number of at least
        ``0.0``. A point at distance ``d`` along the forward axis is blurred
        into a disk whose diameter, measured on the plane in focus, is
        ``aperture * |d - focus_distance| / d`` -- so an object at infinity blurs
        by exactly the aperture's own size seen at the focus plane. An
        orthographic camera's lens magnifies nothing, so there the disk is
        ``aperture * |d - focus_distance| / focus_distance`` across, at every
        depth. ``0`` is a pinhole. Only the path tracer renders it.
    focus_distance
        Distance from the camera to the plane in sharp focus, measured along
        the forward axis (a plane parallel to the screen, not a sphere), in
        world units; a finite number greater than ``0.0``. Has no visible
        effect while ``aperture`` is ``0``.

    Animation
    ---------
    ``aperture`` and ``focus_distance`` are animatable attributes: writing
    ``camera.focus_distance = 12`` after spawn (the Scene's camera is always
    spawned) is *recorded* and interpolates over the current context's runtime
    (1 second by default), which is how a rack focus is animated. Wrap setup in
    ``with Off():`` to apply it instantly. :meth:`focus_at` animates the focus
    onto a Mob or point, following it while the camera or target moves.
    :meth:`set_orthographic` and :meth:`set_perspective` are recorded the same
    way. ``near`` and ``far`` are configuration, not animation.
    """

    #: Width over height of the :class:`~.CameraView` capture this camera
    #: renders, set by the view. None for a camera whose image is the Scene's
    #: own output.
    _view_aspect_ratio: float | None = None

    def __init__(
        self,
        orthographic=False,
        screen_distance=5,
        screen_half_height=2.5,
        fov=None,
        near=0.0,
        far=0.0,
        *args,
        aperture=0.0,
        focus_distance=DEFAULT_FOCUS_DISTANCE,
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
        # Validated into locals: once the properties are attached, assigning
        # self.aperture would route into set_animated_attribute before
        # Animatable.__init__ has built the state it reads (the Light.intensity
        # pattern).
        aperture = _validated_lens_value("aperture", aperture)
        focus_distance = _validated_lens_value("focus_distance", focus_distance)
        # Registered before super().__init__() so Animatable's accessor
        # generation installs set_aperture / get_focus_distance and friends.
        self.register_attrs_as_animatable(list(_LENS_ATTRS + _PROJECTION_ATTRS), Camera)
        # Camera ownership is managed by Scene; tolerate the common generic
        # Mob kwargs without passing duplicates into the base constructor.
        kwargs.pop("add_to_scene", None)
        kwargs.setdefault("basis", squish(_CAMERA_BASIS))
        super().__init__(*args, add_to_scene=False, **kwargs)
        self.animatable_attrs.remove("color")
        # Before the screen exists: with no children, _init_default_attr
        # allocates only the camera's own row instead of falling back to the
        # recursive setter, which would give the screen a lens as well.
        self._init_default_attr("aperture", cast_to_tensor(aperture))
        self._init_default_attr("focus_distance", cast_to_tensor(focus_distance))
        # A pinhole: all of the perspective, every ray from the eye.
        self._init_default_attr("_perspective_weight", cast_to_tensor(1.0))
        self._init_default_attr("_parallel_half_height", cast_to_tensor(0.0))
        with Off(animation_manager=self.animation_manager):
            self.screen = Mob(
                scene=self.scene,
                location=self.location + screen_distance * self.get_forward_direction(),
                basis=squish(
                    unsquish(self.basis, -1, 3)
                    * self.basis.new_tensor([1, 1, -1]).view(3, 1)
                ),
                add_to_scene=False,
            )
            # The screen is a projection proxy, not an optical element: a lens
            # or projection write on the camera must not allocate (or tween)
            # rows on it.
            self.screen._excluded_from_parent_attrs = frozenset(
                _LENS_ATTRS + _PROJECTION_ATTRS
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
                self.set_orthographic()

    def set_animated_attribute(self, attr, value, recursive=True):
        """Animate one animatable attribute to a new value, by name.

        As :meth:`~.Mob.set_animated_attribute`, and additionally the one place
        a lens write is checked. Every route to ``aperture`` and
        ``focus_distance`` -- assignment, ``set_aperture``, :meth:`~.Mob.set`
        and :meth:`~.Mob.set_non_recursive` -- passes through here.

        Raises
        ------
        :class:`.AlganConfigurationError`
            If ``attr`` is ``"aperture"`` and ``value`` is not a finite number
            of at least ``0.0``, or ``"focus_distance"`` and ``value`` is not a
            finite number greater than ``0.0``.
        """
        if attr in _LENS_ATTRS:
            value = _validated_lens_value(attr, value)
        return super().set_animated_attribute(attr, value, recursive=recursive)

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

    @property
    def orthographic(self) -> bool:
        """Whether the camera projects orthographically: parallel rays, no perspective.

        True once :meth:`set_orthographic` has been called and False for a
        perspective camera. Like every recorded change, a switch is read as
        its end state as soon as it is recorded, and during an animated switch
        an updater sees False until the switch completes. Assigning ``True``
        calls :meth:`set_orthographic`, at its default frame, and ``False``
        calls :meth:`set_perspective`, so an assignment animates like them.
        """
        return bool((self._projection_state()[0] <= 0).all())

    @orthographic.setter
    def orthographic(self, value: bool):
        if value:
            self.set_orthographic()
        else:
            self.set_perspective()

    def set_orthographic(self, frame_height: float | None = None) -> Camera:
        """Switch to true orthographic projection: parallel rays, no perspective.

        Every camera ray runs along the forward axis, starting from the plane
        through the camera perpendicular to it, so a Mob's size on screen does
        not depend on its distance from the camera and parallel lines stay
        parallel. The visible frame is a rectangle of fixed world size,
        ``frame_height`` tall and as wide as the output's aspect ratio makes
        it, at every depth. Only what lies in front of the camera's plane is
        seen: moving the camera along its forward axis changes nothing else,
        and moving it sideways pans.

        Animation
        ---------
        Recorded as an animation over the current context's runtime (1 second
        by default), as a dolly zoom: the perspective drains away while the
        frame on one plane holds still -- for the default frame, the plane
        through ``ORIGIN``. On a camera that is already orthographic the frame
        zooms to its new size instead. Use ``with Seq(runtime=3): ...`` to
        change the duration, or ``with Off(): ...`` to switch immediately. Only
        this camera's projection changes; its position, screen and focus stay
        where they are. May be called before or after spawning scene objects.

        Parameters
        ----------
        frame_height
            Height of the visible frame, in world units; positive and finite.
            The width follows from the output's aspect ratio (a
            :class:`~.CameraView`'s capture resolution, for its camera).
            Defaults to ``None``, meaning the height the camera sees on the
            plane through ``ORIGIN`` parallel to its screen -- 8 units from the
            default camera, Manim's frame -- or, when ``ORIGIN`` is not in front
            of the camera, on its screen plane. So an orthographic camera keeps
            its current frame.

        Returns
        -------
        :class:`~.Camera`
            This camera, so calls can be chained.

        Raises
        ------
        :class:`~.AlganConfigurationError`
            If ``frame_height`` is non-finite or not positive.

        See Also
        --------
        :meth:`~.Camera.set_perspective` : Switch back to a perspective projection.

        Examples
        --------
        Flatten the view of a cube while keeping its framing:

        .. algan:: Example1CameraSetOrthographic

            from algan import *

            Cube().rotate(30, UP).rotate(20, RIGHT).spawn()
            with Seq(runtime=2):
                Scene.get_camera().set_orthographic()
            Scene.save_video()
        """
        if frame_height is None:
            half_height = self._default_orthographic_half_height()
        else:
            half_height = 0.5 * self._validated_positive("frame_height", frame_height)
        self.set_non_recursive(
            _perspective_weight=torch.zeros_like(cast_to_tensor(half_height)),
            _parallel_half_height=cast_to_tensor(half_height),
        )
        return self

    def set_perspective(self) -> Camera:
        """Switch back to a perspective projection, undoing :meth:`set_orthographic`.

        The camera becomes a pinhole at its own location again, with the field
        of view its screen gives it (:meth:`get_fov`, unchanged by
        :meth:`set_orthographic`). A perspective camera stays as it is.

        Animation
        ---------
        Recorded as an animation over the current context's runtime (1 second
        by default): the reverse dolly zoom, the perspective returning while
        the frame holds still on the plane where the orthographic and the
        perspective frames agree -- the plane through ``ORIGIN``, for the
        default frame of a camera that has not moved along its axis since.
        Use ``with Seq(runtime=3): ...`` to change the duration, or
        ``with Off(): ...`` to switch immediately. Only this camera's
        projection changes.

        Returns
        -------
        :class:`~.Camera`
            This camera, so calls can be chained.

        See Also
        --------
        :meth:`~.Camera.set_orthographic` : Switch to an orthographic projection.

        Examples
        --------
        Bring perspective back into an orthographic view:

        .. algan:: Example1CameraSetPerspective

            from algan import *

            Cube().rotate(30, UP).rotate(20, RIGHT).spawn()
            camera = Scene.get_camera()
            with Off():
                camera.set_orthographic()
            with Seq(runtime=2):
                camera.set_perspective()
            Scene.save_video()
        """
        self.set_non_recursive(
            _perspective_weight=cast_to_tensor(1.0),
            _parallel_half_height=cast_to_tensor(0.0),
        )
        return self

    def set_near_orthographic(self, distance: float | None = None) -> Camera:
        """Switch to true orthographic projection, by the name it had before.

        Before Algan had an orthographic camera, this approximated one with a
        distant perspective camera. It now switches to the real thing with the
        same framing, exactly as ``set_orthographic()`` does.

        Animation
        ---------
        As :meth:`set_orthographic`: recorded over the current context's
        runtime (1 second by default), as a dolly zoom that keeps the frame on
        the plane through ``ORIGIN``. Use ``with Off(): ...`` to switch
        immediately.

        Parameters
        ----------
        distance
            No longer accepted: it set how far out the approximation's eye
            went, and an orthographic camera has no eye to move. Defaults to
            ``None``; any other value raises.

        Returns
        -------
        :class:`~.Camera`
            This camera, so calls can be chained.

        Raises
        ------
        :class:`~.AlganConfigurationError`
            If ``distance`` is given. :meth:`set_orthographic`'s
            ``frame_height`` chooses the frame instead.

        Examples
        --------
        .. code-block:: python

            Scene.get_camera().set_near_orthographic()  # set_orthographic()
        """
        if distance is not None:
            raise AlganConfigurationError(
                "set_near_orthographic no longer takes a distance: the camera "
                "is truly orthographic now, with no eye to move back. Call "
                "set_orthographic(), or set_orthographic(frame_height=...) to "
                "choose the frame."
            )
        return self.set_orthographic()

    def _default_orthographic_half_height(self):
        """Half the frame :meth:`set_orthographic` keeps when given no height.

        The half-height of what the camera sees now on the plane through
        ``ORIGIN`` parallel to its screen, or on the screen plane when
        ``ORIGIN`` is not in front of it -- a camera constructed at ``ORIGIN``
        has not been positioned yet. ``[*, 1, 1]``, in world units.
        """
        forward = self.get_forward_direction()
        screen_distance = (self.screen.location - self.location).norm(
            dim=-1, keepdim=True
        )
        depth = (-self.location * forward).sum(-1, keepdim=True)
        depth = torch.where(depth > 0, depth, screen_distance)
        persp, parallel_half_height = self._projection_state()
        return (
            parallel_half_height
            + persp * self.screen_half_height * depth / screen_distance
        )

    def _projection_state(self):
        """The stored ``(perspective weight, parallel half-height)``, ``[*, 1, 1]`` each."""
        return (
            cast_to_tensor(self._perspective_weight),
            cast_to_tensor(self._parallel_half_height),
        )

    def _projection_terms(self):
        """The ``(persp, spread)`` the renderer sees, ``[*, 1, 1]`` each.

        See ``algan.rendering.projection``. Writes are only ever endpoints,
        but a recorded tween is interpolated by the timeline: an overshooting
        easing can carry either value past its range between two valid
        endpoints, so both are clamped here, where every reader gets them.
        """
        persp, parallel_half_height = self._projection_state()
        persp = persp.clamp(0.0, 1.0)
        spread = (parallel_half_height / self.screen_half_height).clamp_min(0.0)
        # With no parallel part the camera is the pinhole, whatever the
        # weight says: no tween between valid endpoints arrives anywhere
        # else, and the renderer tells the two cases apart by spread alone.
        persp = torch.where(spread <= 0, torch.ones_like(persp), persp)
        return persp, spread

    def _is_pinhole(self) -> bool:
        """Whether the camera is a plain pinhole at every frame it describes.

        The pinhole keeps the historical arithmetic everywhere, so a
        perspective camera's framing and renders do not change by a bit.
        """
        persp, spread = self._projection_terms()
        return bool(((persp == 1.0) & (spread == 0.0)).all())

    def _get_render_projection(self):
        """Per-frame ``(persp, spread)`` for the render snapshot.

        Shape ``[T, 1, 2]`` after materialization (``[1, 1, 2]`` before), read
        by the render loop beside :meth:`_get_render_lens`;
        ``algan.rendering.projection`` describes how every renderer path uses
        it.
        """
        persp, spread = torch.broadcast_tensors(*self._projection_terms())
        return torch.cat((persp, spread), dim=-1)

    def _frame_rays(self, frame_points):
        """The camera rays through points of the visible frame on the screen plane.

        ``frame_points`` are world points on the screen plane, like the
        corners :meth:`get_corner_pixels` returns. Returns ``(origins,
        directions)`` broadcast against them, the directions unit length.
        A pinhole's rays all start at the camera.
        """
        persp, spread = self._projection_terms()
        screen = self.screen.location
        offset = (frame_points - screen) / (persp + spread)
        origins = self.location + spread * offset
        directions = F.normalize((screen - self.location) + persp * offset, dim=-1, p=2)
        return torch.broadcast_tensors(origins, directions)

    def _frame_edge_planes(self, corners):
        """The four planes the visible frame's edges sweep out, one per edge.

        ``corners`` as :meth:`get_corner_pixels` orders them; edge ``i`` runs
        from corner ``i`` to corner ``i + 1``. Returns ``(points, spans_a,
        spans_b)``, each ``(4, *, 3)``: edge ``i``'s plane passes through
        ``points[i]`` and contains both span directions, which is what
        :func:`~algan.geometry.geometry.intersect_line_with_plane_colinear`
        takes. For a pinhole these are the frustum's sides through the camera,
        spanned by the two corner rays; otherwise each passes through the
        corner ray's origin, spanned by that ray and the edge.
        """
        if self._is_pinhole():
            corner_rays = F.normalize(corners - self.location, dim=-1, p=2)
            next_rays = torch.cat((corner_rays[1:], corner_rays[:1]))
            points = self.location.expand_as(corner_rays)
            return points, corner_rays, next_rays
        origins, directions = self._frame_rays(corners)
        edges = F.normalize(
            torch.cat((corners[1:], corners[:1])) - corners, dim=-1, p=2
        )
        return origins, directions, torch.broadcast_to(edges, directions.shape)

    def get_fov(self):
        """The camera's vertical field of view in degrees (like Three.js's
        ``PerspectiveCamera.fov``), derived from the screen size and the
        camera-to-screen distance. For an orthographic camera, the field of
        view it has again after :meth:`set_perspective`.
        """
        d = (self.screen.location - self.location).norm(p=2, dim=-1).flatten()[0].item()
        return math.degrees(2.0 * math.atan(self.screen_half_height / max(d, 1e-9)))

    def set_fov(self, fov):
        """Set the vertical field of view (degrees). The camera stays where it
        is; its screen moves along the forward axis so that the given angle is
        spanned (small fov = telephoto, large fov = wide angle). Animatable.
        An orthographic camera switches back to perspective
        (:meth:`set_perspective`) at the same time.

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
        location = self.location + self.get_forward_direction() * d
        if self._is_pinhole():
            self.screen.move_to(location)
            return self
        with Sync(animation_manager=self.animation_manager):
            self.screen.move_to(location)
            self.set_perspective()
        return self

    fov = property(get_fov, set_fov)

    def get_near(self):
        """Near clip distance (world units from the camera along its forward
        axis); geometry closer than this is not rendered. 0 = disabled.
        """
        return getattr(self, "_near", 0.0)

    def set_near(self, near):
        """Set the near clip plane distance; 0 disables near clipping.

        Configuration, not animation: takes effect immediately for the whole
        render rather than being recorded.

        Parameters
        ----------
        near
            Distance in world units along the forward axis; finite and at least
            ``0.0``.

        Returns
        -------
        :class:`~.Camera`
            This camera, so calls can be chained.

        Raises
        ------
        :class:`~.AlganConfigurationError`
            If ``near`` is non-finite or negative, or if both clips are enabled
            and ``near`` is not less than ``far``.
        """
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
        """Set the far clip distance; 0 disables far clipping.

        Configuration, not animation: takes effect immediately for the whole
        render rather than being recorded.

        Parameters
        ----------
        far
            Distance in world units of ray travel from the camera; finite and at
            least ``0.0``.

        Returns
        -------
        :class:`~.Camera`
            This camera, so calls can be chained.

        Raises
        ------
        :class:`~.AlganConfigurationError`
            If ``far`` is non-finite or negative, or if both clips are enabled
            and ``far`` is not greater than ``near``.
        """
        far = self._validated_clip("far", far)
        self._validate_clip_order(self.near, far)
        self._far = far
        return self

    far = property(get_far, set_far)

    def set_distance_to_screen(self, distance: float) -> Camera:
        """Move the camera along its axis to a given distance from its screen.

        Moving the camera towards or away from a screen that stays put changes
        how strongly the perspective converges: the frame on the screen plane
        holds still while nearer and farther depths grow or shrink. An
        orthographic camera switches back to perspective
        (:meth:`set_perspective`) as it moves.

        Animation
        ---------
        Recorded as an animation over the current context's runtime (1 second
        by default); ``with Off(): ...`` applies it immediately. Only the
        camera moves; its screen stays where it is.

        Parameters
        ----------
        distance
            The new distance from the camera to its screen plane, in world
            units; positive and finite.

        Returns
        -------
        :class:`~.Camera`
            This camera, so calls can be chained.

        Raises
        ------
        :class:`~.AlganConfigurationError`
            If ``distance`` is non-finite or not positive.
        """
        distance = self._validated_positive("distance", distance)
        location = self.screen.location - self.get_forward_direction() * distance
        if self._is_pinhole():
            self.set_non_recursive(location=location)
            return self
        with Sync(animation_manager=self.animation_manager):
            self.set_non_recursive(location=location)
            self.set_perspective()
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
        over the current context's runtime (1 second by default).

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

    def fly_to(
        self,
        position: torch.Tensor | tuple | list,
        *,
        look_at: torch.Tensor | tuple | list | None = None,
        via: torch.Tensor | tuple | list | None = None,
        look_at_via: torch.Tensor | tuple | list | None = None,
    ) -> Camera:
        """Move the camera and its aim together while keeping the horizon level.

        The camera faces an interpolated target from its actual position at
        every frame. World ``UP`` defines the horizon, so turns do not accumulate
        roll. A camera that starts tilted about its viewing direction levels out
        gradually over the move rather than snapping level on the first frame.
        At a vertical view, the starting right direction defines the frame.

        Animation
        ---------
        Recorded over the current context's runtime (1 second by default).
        Use ``with Seq(runtime=3): camera.fly_to(...)`` to retime the shot or
        ``with Off(): ...`` to place it immediately. Moves this camera and its
        screen together; the camera is already spawned by its Scene.

        Parameters
        ----------
        position
            Destination in world units, a tensor or sequence of shape ``(3,)``.
        look_at
            Destination target in world units, shape ``(3,)``. The initial target
            lies along the current viewing direction, at the distance from the
            starting camera to this target. Both targets interpolate together
            with the position. Defaults to None, preserving the viewing direction.
        via
            World-space waypoint, shape ``(3,)``, on a quadratic curve through
            the start and destination. The camera passes through it halfway
            through the eased motion. Defaults to None, taking a straight path.
        look_at_via
            World-space waypoint for the look-at target, shape ``(3,)``. The
            target follows a quadratic curve through it, reaching it halfway
            through the eased motion, so the aim can sweep along an arc. Requires
            ``look_at``. Defaults to None, moving the target in a straight line.

        Returns
        -------
        :class:`~.Camera`
            This camera, so calls can be chained.

        Raises
        ------
        :class:`~.AlganConfigurationError`
            If a point is not a finite 3-vector, the destination equals its
            look-at target, or ``look_at_via`` is given without ``look_at``. A
            coincident intermediate target retains the starting viewing
            direction.

        Examples
        --------
        .. algan:: Example1CameraFlyTo

            from algan import *

            Sphere().spawn()
            with Seq(runtime=3):
                Scene.get_camera().fly_to(
                    (4, 2, 7), look_at=ORIGIN, via=(2, 3, 8)
                )
            Scene.save_video()
        """
        if look_at_via is not None and look_at is None:
            raise AlganConfigurationError("look_at_via requires look_at")
        position = self._shot_point("position", position)
        target = None if look_at is None else self._shot_point("look_at", look_at)
        waypoint = None if via is None else self._shot_point("via", via)
        target_waypoint = (
            None
            if look_at_via is None
            else self._shot_point("look_at_via", look_at_via)
        )
        if target is not None and torch.linalg.vector_norm(target - position) < 1e-7:
            raise AlganConfigurationError("position and look_at must be different")
        return self._fly_to(position, target, waypoint, target_waypoint)

    @staticmethod
    def _shot_point(name, value):
        point = cast_to_direction(name, value)
        if point.numel() != 3 or not torch.isfinite(point).all():
            raise AlganConfigurationError(f"{name} must be a finite 3-vector")
        return point.reshape(1, 1, 3).clone()

    @staticmethod
    def _level_right(forward, right):
        """The right direction of a frame facing ``forward`` with world UP up.

        At a vertical view the horizon is undefined, so ``right`` projected onto
        the view plane stands in for it.
        """
        level_right = torch.linalg.cross(forward, UP.to(forward).expand_as(forward))
        fallback = right - (right * forward).sum(-1, keepdim=True) * forward
        fallback = torch.where(
            fallback.norm(dim=-1, keepdim=True) > 1e-7,
            fallback,
            RIGHT.to(forward).expand_as(forward),
        )
        return F.normalize(
            torch.where(
                level_right.norm(dim=-1, keepdim=True) > 1e-7, level_right, fallback
            ),
            dim=-1,
        )

    @animated_function(animated_args={"interpolation": 0})
    def _fly_to(self, position, target, via, target_via=None, interpolation=1):
        start = self.location
        forward = self.get_forward_direction()
        right = self.get_right_direction()
        # The starting roll: the signed angle about the view direction from the
        # level right to the actual one. It eases out over the move, so a tilted
        # camera levels gradually instead of jumping on the first frame.
        start_level = self._level_right(forward, right)
        roll = torch.atan2(
            (torch.linalg.cross(start_level, right) * forward).sum(-1, keepdim=True),
            (start_level * right).sum(-1, keepdim=True),
        )
        t = interpolation
        destination = start * (1 - t) + position * t
        if via is not None:
            destination = destination + 4 * t * (1 - t) * (
                via - (start + position) * 0.5
            )
        if target is not None:
            initial_target = start + forward * (target - start).norm(
                dim=-1, keepdim=True
            )
            aim_target = initial_target * (1 - t) + target * t
            if target_via is not None:
                aim_target = aim_target + 4 * t * (1 - t) * (
                    target_via - (initial_target + target) * 0.5
                )
            aim = aim_target - destination
            forward = torch.where(
                aim.norm(dim=-1, keepdim=True) > 1e-7,
                F.normalize(aim, dim=-1),
                forward,
            )
        level_right = self._level_right(forward, right)
        roll = roll * (1 - t)
        # Rodrigues' rotation of the level right about ``forward``.
        new_right = level_right * torch.cos(roll) + torch.linalg.cross(
            forward, level_right
        ) * torch.sin(roll)
        new_up = torch.linalg.cross(new_right, forward)
        self.move_to(destination)
        self.basis = squish(torch.stack((new_right, new_up, forward), dim=-2), -2, -1)
        return self

    def visible_size_at(self, point: torch.Tensor | tuple | list) -> torch.Tensor:
        """Measure the visible width and height at a point's camera depth.

        Measures a plane parallel to the screen through ``point``. Off-axis
        points use forward depth, not distance along their viewing ray. An
        orthographic camera sees the same frame at every depth.

        Animation
        ---------
        Queries the current camera pose immediately without recording anything.
        During an updater or frame query, reads that frame's pose.

        Parameters
        ----------
        point
            World-space point, a tensor or sequence of shape ``(*, 3)``.

        Returns
        -------
        torch.Tensor
            Visible ``(width, height)`` in world units, shape ``(*, 2)`` with
            the camera's leading frame and batch dimensions. Uses the Scene's
            current output aspect ratio, including portrait resolutions. For
            the camera of a :class:`~.CameraView`, uses that view's capture
            resolution instead.

        Raises
        ------
        :class:`~.AlganConfigurationError`
            If the point is non-finite or lies at or behind the camera plane.

        Examples
        --------
        .. algan:: Example1CameraVisibleSizeAt

            from algan import *

            width, height = Scene.get_camera().visible_size_at(ORIGIN).flatten()
            Rectangle(width=float(width) * 0.8, height=float(height) * 0.8).spawn()
            Scene.save_video()
        """
        point = cast_to_direction("point", point)
        depth = self._depth_of(point)
        if not torch.isfinite(point).all() or (depth <= 0).any():
            raise AlganConfigurationError(
                "point must be finite and in front of the camera"
            )
        distance = (self.screen.location - self.location).norm(dim=-1, keepdim=True)
        persp, spread = self._projection_terms()
        # A pinhole's z_eff is the depth itself, bit for bit.
        z_eff = effective_depth(depth, distance, persp, spread)
        height = 2 * self.screen_half_height * z_eff / distance
        aspect = self._view_aspect_ratio or (
            self.scene.num_pixels_screen_width / self.scene.num_pixels_screen_height
        )
        return torch.cat((height * aspect, height), dim=-1)

    def _depth_of(self, point):
        """Distance of ``point`` in front of the camera along its forward axis.

        The planar depth the near clip, :meth:`visible_size_at` and the thin
        lens's focus plane all use: a point anywhere on the plane parallel to
        the screen at depth ``d`` answers ``d``, not its slant range.
        """
        return ((point - self.location) * self.get_forward_direction()).sum(
            -1, keepdim=True
        )

    def focus_at(self, target: Mob | torch.Tensor | tuple | list) -> Camera:
        """Pull the camera's focus onto a Mob or a point.

        Sets ``focus_distance`` to the target's depth: its distance in front of
        the camera along the forward axis, so the plane in focus passes through
        the target's centre. Only visible where depth of field is rendered --
        with an open ``aperture`` under the path tracer
        (``samples_per_pixel > 1``).

        Animation
        ---------
        Recorded as an animation over the current context's runtime (1 second
        by default): the plane in focus travels from its current distance to
        the target, the classic rack focus. The target's depth is measured
        again on every frame of the pull, so a subject or a camera move that
        runs alongside it is followed -- provided that move is recorded
        *before* ``focus_at`` in the same block (``with Sync(): subject.move(...);
        camera.focus_at(subject)``): animations replay in the order they were
        recorded, so a move recorded after the pull is not seen by it.
        Afterwards the distance stays at the value measured when the call was
        recorded, and a target that keeps moving drifts out of focus (call
        ``focus_at`` again). Use ``with Seq(runtime=2): ...`` to retime the
        pull or ``with Off(): ...`` to focus instantly. Changes only this
        camera's ``focus_distance``.

        Parameters
        ----------
        target
            The Mob to focus on (its centre), or a world-space point, a tensor
            or sequence of shape ``(3,)``.

        Returns
        -------
        :class:`~.Camera`
            This camera, so calls can be chained.

        Raises
        ------
        :class:`~.AlganConfigurationError`
            If a point is not a finite 3-vector, or the target is not in front
            of the camera when the call is recorded.

        Examples
        --------
        .. code-block:: python

            SETTINGS.raytracing.set(samples_per_pixel=64)
            near = Sphere(radius=0.5).move_to(LEFT * 2 + OUT * 4).spawn()
            far = Cube().move_to(RIGHT * 2 + IN * 4).spawn()
            camera = Scene.get_camera()
            with Off():
                camera.aperture = 0.8
                camera.focus_at(near)
            with Seq(runtime=2):
                camera.focus_at(far)
            Scene.save_video()
        """
        if not isinstance(target, Mob):
            target = self._shot_point("target", target)
            point = target
        else:
            point = target.get_center()
        depth = self._depth_of(point)
        if not bool(torch.isfinite(depth).all()) or bool((depth <= 0).any()):
            raise AlganConfigurationError(
                "focus_at target must be in front of the camera"
            )
        return self._focus_at(target)

    @animated_function(animated_args={"interpolation": 0})
    def _focus_at(self, target, interpolation=1):
        point = target.get_center() if isinstance(target, Mob) else target
        # During replay this is the frame's own camera and target, and a
        # [T, 1, 1] interpolation. Both the target depth and the blend are
        # clamped: a target passing behind the camera, or an overshooting
        # easing (ease_out_back takes the interpolation past 1), must not
        # write a non-positive distance, which the setter would refuse in the
        # middle of a render.
        depth = self._depth_of(point).clamp_min(_MIN_FOCUS_DISTANCE)
        start = self.focus_distance
        blended = start * (1 - interpolation) + depth * interpolation
        self.focus_distance = blended.clamp_min(_MIN_FOCUS_DISTANCE)
        return self

    def _get_render_lens(self):
        """Per-frame ``(aperture, focus_distance)`` for the render snapshot.

        Shape ``[T, 1, 2]`` after materialization (``[1, 1, 2]`` before), in
        world units, the lens diameter first. Read by the render loop beside
        :meth:`_get_render_screen_basis`; only the path tracer consumes it.
        """
        # Writes are validated, but a recorded tween is interpolated by the
        # timeline without the setter: an overshooting easing can carry
        # either value past its bound between two valid endpoints. A negative
        # focus would turn every lens ray around, so both are clamped here,
        # where every render reads them.
        aperture = cast_to_tensor(self.aperture).clamp_min(0.0)
        focus = cast_to_tensor(self.focus_distance).clamp_min(_MIN_FOCUS_DISTANCE)
        aperture, focus = torch.broadcast_tensors(aperture, focus)
        return torch.cat((aperture, focus), dim=-1)

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
        angle you set is preserved. An orthographic camera, whose view does not change
        with distance, instead resizes its frame to fit (backing off only if the Mob
        is not wholly in front of it).

        Animation
        ---------
        Recorded as an animation: the moves run together inside a :class:`~.Sync`, over
        the current context's runtime (1 second by default). The Mob's extent is
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
        return self._center_on(mob, buffer_portion)

    def _center_on(self, mob, buffer_portion, aspect_ratio=None):
        f = self.get_forward_direction()
        r = self.get_right_direction()
        u = self.get_up_direction()
        mob_boundary_points = [mob.get_boundary_point(_) for _ in [-r, u, r, -u]]
        mob_boundary_points = torch.stack(mob_boundary_points)
        mobl = 0.5 * ((mob_boundary_points).amax(0) + (mob_boundary_points).amin(0))
        mob_boundary_points = (mob_boundary_points - mobl) * (1 + buffer_portion) + mobl
        selfl = self.location
        orthographic = self.orthographic

        with Sync(animation_manager=self.animation_manager):
            self.move_to(mobl - f * dot_product(mobl - selfl, f))
            if orthographic:
                # Moving along the axis changes nothing an orthographic camera
                # sees, so it frames by resizing.
                return self._fit_orthographic_frame(
                    mob, mobl, mob_boundary_points, aspect_ratio
                )

            corners = (
                self.get_corner_pixels()
                if aspect_ratio is None
                else self._corner_pixels(aspect_ratio)
            )
            # The top edge's plane (corners 1 -> 2) and the left edge's
            # (corners 0 -> 1); they move rigidly with the camera, so sliding
            # it back along the axis until a boundary point lies on one makes
            # that side of the Mob touch that side of the frame.
            points, spans_a, spans_b = self._frame_edge_planes(corners)
            vertical_move, vertical_dist = intersect_line_with_plane_colinear(
                -f, points[1], spans_a[1], spans_b[1], mob_boundary_points[1]
            )
            horizontal_move, horizontal_dist = intersect_line_with_plane_colinear(
                -f, points[0], spans_a[0], spans_b[0], mob_boundary_points[0]
            )

            return self.move(
                (mob_boundary_points[1] - vertical_move)
                if vertical_dist <= horizontal_dist
                else (mob_boundary_points[0] - horizontal_move)
            )

    def _fit_orthographic_frame(self, mob, center, boundary, aspect_ratio):
        """Size an orthographic frame to the buffered ``boundary`` of a Mob.

        ``boundary`` holds the Mob's extreme points towards ``-right``,
        ``up``, ``right`` and ``-up`` (in that order) and ``center`` the point
        the camera is now centred on. The frame's half-height becomes the
        larger of the vertical reach and the horizontal reach over the aspect
        ratio. A Mob that is not wholly in front of the camera's plane, which
        an orthographic camera cannot see past, also backs the camera off
        until its nearest point is that half-height in front.
        """
        forward = self.get_forward_direction()
        if aspect_ratio is None:
            aspect_ratio = self._view_aspect_ratio or (
                self.scene.video_settings.resolution[0]
                / self.scene.video_settings.resolution[1]
            )
        half_height = torch.maximum(
            dot_product(boundary[1] - center, self.get_up_direction()).abs(),
            dot_product(center - boundary[0], self.get_right_direction()).abs()
            / aspect_ratio,
        ).clamp_min(1e-6)
        nearest = dot_product(mob.get_boundary_point(-forward) - self.location, forward)
        if bool((nearest <= 0).any()):
            self.move(forward * (nearest - half_height))
        self.set_non_recursive(_parallel_half_height=half_height)
        return self

    def project_point_onto_screen_border(
        self, point: torch.Tensor, direction: torch.Tensor
    ) -> torch.Tensor:
        """Find where a point would leave the frame travelling in a direction.

        Casts from ``point`` along ``direction`` and returns the intersection with the
        edge of the visible volume -- a frustum for a perspective camera, a box for an
        orthographic one. This is what the screen-edge placement methods use to know
        where "against the edge" is.

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
        points, spans_a, spans_b = self._frame_edge_planes(self.get_corner_pixels())
        intersection_points, intersection_distances = (
            torch.stack(_)
            for _ in zip(
                *[
                    intersect_line_with_plane_colinear(
                        direction, points[i], spans_a[i], spans_b[i], point
                    )
                    for i in range(points.shape[0])
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

        Where the visible frame crosses the camera's screen plane, accounting for the
        current aspect ratio, ordered around the frame: bottom-left, top-left,
        top-right, bottom-right. Screen-relative layout is built on these.

        Returns
        -------
        torch.Tensor
            The four corner points, shape ``(4, *, 3)``.
        """
        return self._corner_pixels()

    def _corner_pixels(self, aspect_ratio=None):
        b = unsquish(self.screen.basis, -1, 3)
        b = b / b.norm(p=2, dim=-1, keepdim=True).square().clamp_min(1e-6)
        if aspect_ratio is None:
            aspect_ratio = self._view_aspect_ratio or (
                self.scene.video_settings.resolution[0]
                / self.scene.video_settings.resolution[1]
            )
        if self._is_pinhole():
            return (
                self.screen.location
                + b[..., 0, :] * self.corner_x_coords * aspect_ratio
                + b[..., 1, :] * self.corner_y_coords
            )
        # The ray through screen offset L crosses the screen plane at
        # (persp + spread) * L (``algan.rendering.projection``), so that is
        # where the visible frame meets it.
        persp, spread = self._projection_terms()
        return self.screen.location + (
            b[..., 0, :] * self.corner_x_coords * aspect_ratio
            + b[..., 1, :] * self.corner_y_coords
        ) * (persp + spread)
