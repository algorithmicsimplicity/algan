"""Translation and placement methods for :class:`~algan.animatable_base.mob.Mob`.

Split out of ``mob.py`` for readability; :class:`MobMovementMixin` is mixed into
``Mob`` and is not useful standalone (``self`` is always a Mob).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from algan import animated_function
from algan.animation_timeline.animation_contexts import Off, Seq, Sync
from algan.constants.spatial import *
from algan.errors import AlganConfigurationError
from algan.settings import SETTINGS
from algan.utils.tensor_utils import (
    broadcast_cross_product,
    broadcast_gather,
    cast_to_direction,
    cast_to_tensor,
    dot_product,
)


def _resolve_buffer(buffer):
    return SETTINGS.style.buffer if buffer is None else buffer


class MobMovementMixin:
    """Methods for moving Mobs around, mixed into
    :class:`~algan.animatable_base.mob.Mob`.
    """

    def _screen_relative_direction(self, direction: torch.Tensor) -> torch.Tensor:
        """Read a direction in the camera's frame, and return it in world space.

        ``RIGHT`` becomes the camera's right, ``UP`` its up and ``OUT`` the
        direction back towards the viewer, so a direction with no ``z``
        component stays in the plane parallel to the screen however the camera
        is posed. Without this the screen-relative helpers cast along world
        axes: under a 60-degree yaw ``move_to_screen_edge(LEFT)`` used to
        displace a Mob along world *+x*, landing it off the right of the frame.

        The frame is
        :meth:`~algan.animatable_base.mob_layout.MobLayoutMixin._screen_axes`,
        shared with the layout mixin. Its ``None`` -- the camera's frame is the
        world frame -- is what keeps an unrotated camera on exactly the
        arithmetic it always did, rather than a matmul by an identity that is
        only identity up to rounding.
        """
        basis = self._screen_axes()
        if basis is None:
            return direction
        return direction.to(device=basis.device, dtype=basis.dtype) @ basis

    def move_between(self, start: Mob | torch.Tensor, end: Mob | torch.Tensor) -> Mob:
        """Move the Mob to the midpoint between two locations.

        Animation
        ---------
        Recorded as an animation: the Mob travels to the midpoint over the
        current context's duration (1 second by default). Applies to this Mob
        and its descendants.

        Parameters
        ----------
        start
            First endpoint: a 3-D point of shape ``(*, 3)``, or a Mob, in which
            case its center is used.
        end
            Second endpoint, in the same forms as ``start``.

        Returns
        -------
        :class:`~algan.animatable_base.mob.Mob`
            This Mob, so calls can be chained.
        """
        start, end = [
            _.get_center() if hasattr(_, "get_center") else _ for _ in [start, end]
        ]
        return self.move_to((start + end) / 2)

    def _move_along_arc(
        self,
        point: torch.Tensor,
        arc_angle: float | torch.Tensor,
        arc_normal: torch.Tensor = OUTWARD,
        recursive: bool = True,
    ) -> Mob:
        """Move the Mob to ``point`` along a signed circular arc.

        The start and target points form the chord of the arc. ``arc_normal``
        fixes the plane of the circle, and ``arc_angle`` is the signed
        sweep from the start to the target. Its sign follows the same rotation
        convention as
        :meth:`~algan.animatable_base.mob_orientation.MobOrientationMixin.rotate`;
        angles whose magnitude exceeds
        180 degrees therefore trace the corresponding major arc.

        A zero sweep is treated as the limiting straight-line path. Coincident
        endpoints are a no-op because the radius of a non-trivial closed arc
        cannot be inferred from the endpoint alone.

        Animation
        ---------
        Recorded as an animation: the Mob sweeps along the arc over the current
        context's duration (1 second by default). Wrap the call to retime it --
        ``with Seq(duration=3): mob.move_to_point_along_arc(RIGHT, 90)``.

        Parameters
        ----------
        point
            The target location, shape ``(*, 3)``.
        arc_angle
            Signed arc sweep **in degrees**. Sweeps outside ``[-360, 360]`` are
            supported, except exact non-zero multiples of 360 degrees when the
            endpoints differ; such a path would require an infinite radius.
        arc_normal
            Normal vector of the arc plane; the chord from the current location
            to ``point`` must be perpendicular to it. Defaults to ``OUTWARD`` (the
            +z axis, out of the screen), which arcs in the screen plane.
        recursive
            Whether to propagate the location change to descendants, preserving
            their offsets from this Mob. Defaults to True; False moves only this
            Mob and leaves its children where they are.

        Returns
        -------
        :class:`~algan.animatable_base.mob.Mob`
            This Mob, so calls can be chained.

        Raises
        ------
        ValueError
            If ``arc_normal`` is zero, if the endpoints do not lie in a plane
            perpendicular to ``arc_normal``, or if distinct endpoints are
            paired with a non-zero whole-turn sweep.
        """
        start = self.location
        dtype = start.dtype
        device = start.device

        target = cast_to_tensor(point).to(device=device, dtype=dtype)
        normal = cast_to_tensor(arc_normal).to(device=device, dtype=dtype)
        angle_degrees = cast_to_tensor(arc_angle).to(device=device, dtype=dtype)

        if not torch.all(torch.isfinite(target)):
            raise ValueError("point must contain only finite values")
        if not torch.all(torch.isfinite(angle_degrees)):
            raise ValueError("arc_angle must contain only finite values")
        if not torch.all(torch.isfinite(normal)):
            raise ValueError("arc_normal must contain only finite values")

        # A circular rotation around ``normal`` preserves the coordinate along
        # that axis, so the chord must lie in the perpendicular plane.
        normal_length = normal.norm(p=2, dim=-1, keepdim=True)
        if torch.any(normal_length == 0):
            raise ValueError("arc_normal must be a non-zero vector")
        normal = normal / normal_length

        chord = target - start
        chord_length = chord.norm(p=2, dim=-1, keepdim=True)
        coplanar_error = dot_product(chord, normal).abs()
        coplanar_tolerance = (
            torch.maximum(torch.ones_like(chord_length), chord_length) * 1e-5
        )
        if torch.any(coplanar_error > coplanar_tolerance):
            raise ValueError(
                "The start and target points must lie in a plane "
                "perpendicular to arc_normal"
            )

        coincident = chord_length == 0
        whole_turn_count = torch.round(angle_degrees / 360)
        exact_whole_turn = (whole_turn_count != 0) & (
            angle_degrees == whole_turn_count * 360
        )
        invalid_whole_turn = exact_whole_turn & ~coincident
        if torch.any(invalid_whole_turn):
            raise ValueError(
                "Distinct endpoints cannot be connected by an exact non-zero "
                "multiple-of-360-degree circular arc"
            )

        return self._move_along_arc_displacement(
            chord,
            angle_degrees,
            normal,
            recursive=recursive,
        )

    @animated_function(animated_args={"interpolation": 0.0}, unique_args=["recursive"])
    def _move_along_arc_displacement(
        self,
        chord: torch.Tensor,
        arc_angle: torch.Tensor,
        arc_normal: torch.Tensor,
        recursive: bool = True,
        interpolation: float | torch.Tensor = 1.0,
    ) -> Mob:
        """Apply a pre-validated arc displacement at ``interpolation``."""
        dtype = self.location.dtype
        device = self.location.device
        chord = cast_to_tensor(chord).to(device=device, dtype=dtype)
        normal = cast_to_tensor(arc_normal).to(device=device, dtype=dtype)
        angle_degrees = cast_to_tensor(arc_angle).to(device=device, dtype=dtype)
        interpolation = cast_to_tensor(interpolation).to(device=device, dtype=dtype)

        # Let h be half the total sweep. Direct circular interpolation can be
        # written using only the chord d and n x d:
        #
        #   offset(t) = A(t) d + B(t) (n x d)
        #   A(t) = sin(t h) cos((1-t) h) / sin(h)
        #   B(t) = sin(t h) sin((1-t) h) / sin(h)
        #
        # This is algebraically equivalent to rotating around the circle
        # centre, but avoids constructing that centre. For shallow arcs the
        # centre can be arbitrarily far away and subtraction around it loses
        # substantial precision.
        half_angle = torch.deg2rad(angle_degrees) * 0.5
        sin_half_angle = torch.sin(half_angle)
        zero_angle = angle_degrees == 0
        coincident = chord.norm(p=2, dim=-1, keepdim=True) == 0

        # Substitute a harmless denominator for entries whose result is taken
        # from the linear/no-op branch below. This keeps mixed batched inputs
        # finite without weakening the validation performed by the public
        # method before the animation is recorded.
        linear_path = zero_angle | coincident
        safe_sin_half_angle = torch.where(
            linear_path,
            torch.ones_like(sin_half_angle),
            sin_half_angle,
        )
        traversed_half_angle = interpolation * half_angle
        remaining_half_angle = (1 - interpolation) * half_angle
        chord_coefficient = (
            torch.sin(traversed_half_angle)
            * torch.cos(remaining_half_angle)
            / safe_sin_half_angle
        )
        normal_coefficient = (
            torch.sin(traversed_half_angle)
            * torch.sin(remaining_half_angle)
            / safe_sin_half_angle
        )

        circular_offset = (
            chord_coefficient * chord
            + normal_coefficient * broadcast_cross_product(normal, chord)
        )
        linear_offset = interpolation * chord
        offset = torch.where(
            linear_path.expand_as(circular_offset),
            linear_offset,
            circular_offset,
        )
        new_location = self.location + offset

        if recursive:
            self.location = new_location
        else:
            self.set_non_recursive(location=new_location)
        return self

    def move_to(
        self, location: torch.Tensor, arc_angle: float | None = None, **kwargs
    ) -> Mob:
        """Move the Mob to an absolute location.

        The path is a straight line unless ``arc_angle`` is given, in which
        case the Mob swings to the target along a circular arc.

        Animation
        ---------
        Recorded as an animation: the Mob travels from where it is to
        ``location`` over the current context's duration (1 second by default).
        Use ``with Off(): mob.move_to(...)`` to teleport it instead. Applies to
        this Mob and its descendants.

        Parameters
        ----------
        location
            The target location, shape ``(*, 3)``.
        arc_angle
            Signed sweep of the curved path, **in degrees**. Defaults to
            ``None``, meaning travel in a straight line.
        **kwargs
            Passed to :meth:`~algan.animatable_base.mob.Mob.set_location` (notably
            ``recursive``), or to
            :meth:`~algan.animatable_base.mob_movement.MobMovementMixin.move_to_point_along_arc`
            when ``arc_angle`` is
            given (notably ``arc_normal``).

        Returns
        -------
        :class:`~algan.animatable_base.mob.Mob`
            This Mob, so calls can be chained.

        See Also
        --------
        :meth:`~algan.animatable_base.mob_movement.MobMovementMixin.move`
            Move by a relative displacement instead.
        :meth:`~algan.animatable_base.mob_movement.MobMovementMixin.move_to_screen_position`
            Place the Mob in screen space.
        """
        if arc_angle is None:
            return self.set_location(location, **kwargs)
        return self._move_along_arc(location, arc_angle, **kwargs)

    def move(self, displacement: torch.Tensor, **kwargs) -> Mob:
        """Move the Mob by a displacement from wherever it currently is.

        Animation
        ---------
        Recorded as an animation: the Mob travels the displacement over the
        current context's duration (1 second by default). Retime it with
        ``with Seq(duration=2): mob.move(RIGHT)``, or apply it instantly with
        ``with Off(): mob.move(RIGHT)``. Applies to this Mob and its descendants.

        Parameters
        ----------
        displacement
            How far and in which direction to move, shape ``(*, 3)``, in world
            units. The spatial constants (``RIGHT``, ``UP``, ``OUT``, ...) are
            unit vectors, so ``mob.move(RIGHT * 3)`` moves three units right.
        **kwargs
            Passed to
            :meth:`~algan.animatable_base.mob_movement.MobMovementMixin.move_to`
            -- notably ``arc_angle`` to
            travel along a curve rather than a straight line.

        Returns
        -------
        :class:`~algan.animatable_base.mob.Mob`
            This Mob, so calls can be chained.

        Examples
        --------
        .. algan:: Example1MobMove

            from algan import *

            square = Square().spawn()
            square.move(RIGHT)
            square.move(UP * 2 + LEFT)
            square.move(DOWN, arc_angle=120)

            Scene.save_video()
        """
        self.move_to(
            self.location + cast_to_direction("displacement", displacement), **kwargs
        )
        return self

    def move_next_to(
        self,
        target_mob: Mob | torch.Tensor,
        direction: torch.Tensor,
        buffer: float | None = None,
        align_edge=None,
        **kwargs,
    ) -> Mob:
        """Move this Mob so it sits just beside another Mob or point.

        Placement is edge-to-edge, not center-to-center: this Mob's near
        boundary is set ``buffer`` away from the target's boundary, so shapes of
        different sizes end up with an even gap between them.

        Animation
        ---------
        Recorded as an animation over the current context's duration (1 second
        by default). Applies to this Mob and its descendants.

        Parameters
        ----------
        target_mob
            The Mob to sit beside, or a point of shape ``(*, 3)`` to treat as
            the target. A Mob contributes its boundary, a point only itself.
        direction
            Which side of ``target_mob`` to move to (e.g. ``RIGHT``, ``UP``);
            need not be normalized.
        buffer
            Gap to leave between the two boundaries, in world units. Defaults to
            ``SETTINGS.style.buffer`` (``0.6``).
        align_edge
            Direction along which to additionally align the two Mobs' boundaries
            (see
            :meth:`~algan.animatable_base.mob_movement.MobMovementMixin.align_with`),
            so e.g. two Mobs
            placed side by side can also share a bottom edge. Defaults to
            ``None``, meaning no secondary alignment.
        **kwargs
            Passed to
            :meth:`~algan.animatable_base.mob_movement.MobMovementMixin.move_to`.

        Returns
        -------
        :class:`~algan.animatable_base.mob.Mob`
            This Mob, so calls can be chained.

        See Also
        --------
        :meth:`~algan.animatable_base.mob_movement.MobMovementMixin.align_with`
            Align centers, edges or boundaries along one axis without changing
            the others.
        """
        buffer = _resolve_buffer(buffer)
        direction = cast_to_direction("direction", direction)
        normalized_direction = F.normalize(direction, p=2, dim=-1)
        # Get the boundary point of the target_mob along the given direction
        target_edge_point = (
            target_mob.get_boundary_point(normalized_direction)
            if not isinstance(target_mob, torch.Tensor)
            else target_mob
        )
        # Get the boundary point of this mob in the opposite direction
        my_edge_point = self.get_boundary_point(-normalized_direction)

        # Calculate the required displacement to move 'my_edge_point' to 'target_edge_point'
        # plus the buffer distance, and then apply it to the Mob's current location.
        displacement_to_align_edges = (
            target_edge_point + normalized_direction * buffer - my_edge_point
        )
        self.move(displacement_to_align_edges, **kwargs)
        if align_edge is not None:
            self.align_with(target_mob, align_edge, anchor="boundary")
        return self

    def align_with(
        self,
        mob: Mob,
        direction: torch.Tensor,
        anchor: str = "center",
        buffer: float | None = None,
        from_mob: Mob | None = None,
    ) -> Mob:
        """Line this Mob up with another along one axis.

        Only the component of the movement along ``direction`` is applied, so
        the two end up aligned on that axis with their positions on the other
        axes untouched: ``a.align_with(b, UP)`` puts them at the same height
        without changing how far apart they are horizontally.

        Animation
        ---------
        Recorded as an animation over the current context's duration (1 second
        by default). Applies to this Mob and its descendants.

        Parameters
        ----------
        mob
            The Mob to align with.
        direction
            Axis to align along, and which of ``mob``'s sides to measure from
            (e.g. ``RIGHT``, ``UP``); need not be normalized.
        anchor
            Which point on each Mob is brought into line, matched
            case-insensitively. ``'center'`` (the default) aligns the two
            anchors. ``'boundary'`` aligns the two ``direction``-side
            boundaries, so the Mobs end up flush -- sharing a bottom edge, say.
            ``'edge'`` brings this Mob's opposite side up against ``mob``'s
            ``direction`` side, so the two abut rather than overlap.
        buffer
            Extra gap along ``direction``, in world units. Defaults to ``None``,
            which means ``SETTINGS.style.buffer`` (``0.6``) for
            ``anchor='edge'`` and no gap for the other two.
        from_mob
            Mob supplying the reference point that is moved into alignment,
            useful when aligning a group by one of its members. Defaults to
            ``None``, meaning use this Mob.

        Returns
        -------
        :class:`~algan.animatable_base.mob.Mob`
            This Mob, so calls can be chained.

        Raises
        ------
        :class:`.AlganConfigurationError`
            If ``anchor`` is not one of the three names above.
        """
        anchor = str(anchor).lower()
        if anchor not in ("center", "edge", "boundary"):
            raise AlganConfigurationError(
                f"anchor must be 'center', 'edge' or 'boundary', got {anchor!r}"
            )
        direction = F.normalize(cast_to_direction("direction", direction), p=2, dim=-1)
        source = self if from_mob is None else from_mob
        if buffer is None:
            buffer = SETTINGS.style.buffer if anchor == "edge" else 0.0

        if anchor == "center":
            target_point = mob.location
            source_point = source.location
        elif anchor == "boundary":
            target_point = mob.get_boundary_point(direction)
            source_point = source.get_boundary_point(direction)
        else:
            target_point = mob.get_boundary_point(direction)
            source_point = source.get_boundary_point(-direction)

        displacement = target_point + direction * buffer - source_point
        return self.move(dot_product(displacement, direction) * direction)

    def move_to_screen_position(
        self, x: float | torch.Tensor, y: float | torch.Tensor
    ) -> Mob:
        """Move the Mob so it appears at a given position on the screen.

        The world location is worked out from the current camera, so this places
        the Mob where the viewer sees it rather than where it sits in 3-D space.
        The Mob keeps its distance from the camera; only its apparent position
        changes.

        Animation
        ---------
        Recorded as an animation over the current context's duration (1 second
        by default). The screen position is resolved once, when the call is
        recorded -- a later camera move will not keep the Mob pinned there (use
        an updater for that). Applies to this Mob and its descendants.

        Parameters
        ----------
        x
            Horizontal position in screen units: ``0`` is the left edge, ``1``
            the right edge, ``0.5`` the middle. Values outside ``[0, 1]`` are
            off-screen.
        y
            Vertical position in screen units: ``0`` is the bottom edge, ``1``
            the top edge.

        Returns
        -------
        :class:`~algan.animatable_base.mob.Mob`
            This Mob, so calls can be chained.
        """
        with Off(animation_manager=self.animation_manager):
            clone = self.clone(add_to_scene=False)
            clone.move_to_screen_corner((DOWN, LEFT))
            bottom_left = clone.location.clone()
            clone.move_to_screen_corner((UP, LEFT))
            top_left = clone.location.clone()
            clone.move_to_screen_corner((UP, RIGHT))
            top_right = clone.location.clone()
            clone.move_to_screen_corner((DOWN, RIGHT))
            bottom_right = clone.location.clone()
            bottom = bottom_left * (1 - x) + x * bottom_right
            top = top_left * (1 - x) + x * top_right
            new_loc = bottom * (1 - y) + y * top
        return self.move_to(new_loc)

    def move_to_screen_edge(
        self, direction: torch.Tensor, buffer: float | None = None
    ) -> Mob:
        """Move the Mob against one edge of the screen.

        The Mob's own boundary is what comes to rest ``buffer`` *inside* the
        border, so a large and a small shape both end up looking equally inset.
        Where the Mob starts makes no difference: one that is already off-screen
        past that edge is brought back in, and calling this twice leaves it where
        the first call put it.

        ``direction`` is read in the camera's frame, not the world's, so
        ``RIGHT`` means the right of the *screen* whatever angle the camera is
        posed at, and the Mob travels in the plane parallel to the screen rather
        than towards or away from the viewer.

        Animation
        ---------
        Recorded as an animation over the current context's duration (1 second
        by default). The edge position is resolved from the camera when the call
        is recorded. Applies to this Mob and its descendants.

        Parameters
        ----------
        direction
            Which screen edge to move to, in the camera's frame: ``RIGHT``,
            ``LEFT``, ``UP`` or ``DOWN``. ``x`` runs across the screen, ``y`` up
            it and ``z`` out of it towards the viewer, so ``OUT`` is the camera's
            ``-forward``; ``RIGHT + OUT`` casts along the diagonal of the two and
            stops where that ray leaves the frustum.
        buffer
            Gap to leave between the Mob's boundary and the screen border, in
            world units. Defaults to ``SETTINGS.style.buffer`` (``0.6``).

        Returns
        -------
        :class:`~algan.animatable_base.mob.Mob`
            This Mob, so calls can be chained.

        See Also
        --------
        :meth:`~algan.animatable_base.mob_movement.MobMovementMixin.move_to_screen_corner`
            Move against two edges at once.
        :meth:`~algan.animatable_base.mob_movement.MobMovementMixin.move_off_screen`
            Move all the way off-screen.
        """
        buffer = _resolve_buffer(buffer)
        direction = self._screen_relative_direction(
            cast_to_direction("direction", direction)
        )
        normalized_edge = F.normalize(direction, p=2, dim=-1)
        # Get the boundary point of this Mob that is furthest towards the 'edge' direction
        mob_boundary_point = self.get_boundary_point(normalized_edge)
        # Project this point onto the screen border to find the target point on the border
        edge_point_on_screen = self.scene.camera.project_point_onto_screen_border(
            mob_boundary_point, normalized_edge
        )
        # Step back from the border along the edge direction. The inset used to
        # be taken as ``normalize(boundary - border) * buffer``, which reads the
        # direction off the Mob's current position instead of off the edge that
        # was asked for. The border is cast *from* the boundary along the edge,
        # so that difference is antiparallel to the edge whenever the Mob is
        # inside the frame -- but it flips for a Mob already outside (leaving it
        # ``buffer`` outside rather than bringing it in), and it degenerates to
        # zero for a boundary resting exactly on the border, where float noise
        # then chose the sign. A Manim ``Title`` lands exactly there.
        target_location = edge_point_on_screen - normalized_edge * buffer
        # Calculate the displacement needed and move the Mob
        displacement = target_location - mob_boundary_point
        self.move(displacement)
        return self

    def move_to_screen_corner(self, directions, buffer: float | None = None) -> Mob:
        """Move the Mob into a corner of the screen.

        The corner is named by the edges that meet there, e.g.
        ``mob.move_to_screen_corner((UP, RIGHT))`` for the top-right.

        Animation
        ---------
        Recorded as an animation. The two edge moves run inside a
        :class:`~algan.animation_timeline.animation_contexts.Sync`, so they happen
        simultaneously and the whole call still
        takes the current context's duration (1 second by default) rather than
        two seconds. Applies to this Mob and its descendants.

        Parameters
        ----------
        directions
            The screen edges meeting at the corner, as an iterable of direction
            vectors in the camera's frame -- ``(UP, RIGHT)`` for the top-right,
            whatever angle the camera is posed at.
        buffer
            Gap to leave from every screen border named, in world units.
            Defaults to ``SETTINGS.style.buffer`` (``0.6``).

        Returns
        -------
        :class:`~algan.animatable_base.mob.Mob`
            This Mob, so calls can be chained.
        """
        # Chained move_to_screen_edge calls, run together so the whole corner
        # move still takes one context duration.
        with Sync(animation_manager=self.animation_manager):
            for direction in directions:
                self.move_to_screen_edge(direction, buffer=buffer)
        return self

    def move_off_screen(
        self, direction: torch.Tensor, buffer: float | None = None, despawn: bool = True
    ) -> Mob:
        """Slide the Mob off the screen, and by default despawn it there.

        The Mob travels far enough that its whole bounding box clears the border,
        so nothing is left poking into frame.

        Animation
        ---------
        Recorded as an animation: the slide takes the current context's duration
        (1 second by default), and the despawn follows it in a
        :class:`~algan.animation_timeline.animation_contexts.Seq`
        without an extra fade, so the Mob is simply gone once it is out of sight.
        Applies to this Mob and its descendants.

        Parameters
        ----------
        direction
            Which way to leave, in the camera's frame: ``RIGHT``, ``LEFT``,
            ``UP`` or ``DOWN``, meaning the sides of the *screen* whatever angle
            the camera is posed at.
        buffer
            Extra distance to travel beyond the screen border, in world units.
            Defaults to ``SETTINGS.style.buffer`` (``0.6``).
        despawn
            Whether to despawn the Mob once it is off-screen. Defaults to True;
            pass False to keep it alive out of frame so it can slide back in
            later.

        Returns
        -------
        :class:`~algan.animatable_base.mob.Mob`
            This Mob, so calls can be chained.
        """
        buffer = _resolve_buffer(buffer)
        direction = self._screen_relative_direction(
            cast_to_direction("direction", direction)
        )
        bbox = self.get_bounding_box()

        points_on_screen_edge = self.scene.camera.project_point_onto_screen_border(
            bbox, direction
        )

        disps = points_on_screen_edge - bbox
        largest_disp = broadcast_gather(
            disps,
            -2,
            disps.norm(p=2, dim=-1, keepdim=True).argmax(-2, keepdim=True),
            keepdim=True,
        )

        with Seq(
            animation_manager=self.animation_manager
        ):  # Ensure movement and despawn happen sequentially
            self.move(largest_disp + buffer * F.normalize(direction, p=2, dim=-1))
            if despawn:
                self.despawn(animate=False)
        return self

    def move_to_point_with_displacement(
        self, destination: torch.Tensor, displacement: torch.Tensor
    ) -> Mob:
        """Move the Mob to a destination along a right-angled, three-leg path.

        The Mob first travels along ``displacement``, then along the component of
        the remaining distance orthogonal to it, then closes any remainder --
        tracing a bracket-shaped route instead of a diagonal. Useful for routing
        a Mob around something in the way.

        Animation
        ---------
        Recorded as an animation. All three legs run inside a
        ``Seq(duration=1)``, so the whole path takes 1 second regardless of the
        current context's duration. Applies to this Mob and its descendants.

        Parameters
        ----------
        destination
            The final location, shape ``(*, 3)``.
        displacement
            Direction and length of the first leg, shape ``(*, 3)``; this is what
            decides which way the path bends.

        Returns
        -------
        :class:`~algan.animatable_base.mob.Mob`
            This Mob, so calls can be chained.
        """
        # Vector from current location to destination
        destination_displacement = destination - self.location
        # Normalize the initial displacement direction
        normalized_displacement_direction = F.normalize(displacement, p=2, dim=-1)
        # Calculate the orthogonal component of the destination displacement relative to the initial displacement
        orthogonal_displacement = (
            destination_displacement
            - dot_product(destination_displacement, normalized_displacement_direction)
            * normalized_displacement_direction
        )

        with Seq(duration=1, animation_manager=self.animation_manager):
            self.move(displacement)
            self.move(orthogonal_displacement)
            self.location = destination
        return self
