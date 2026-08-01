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
from algan.geometry.geometry import project_point_onto_line
from algan.settings import SETTINGS
from algan.utils.tensor_utils import (
    broadcast_cross_product,
    broadcast_gather,
    cast_to_tensor,
    dot_product,
)


def _resolve_buffer(buffer):
    return SETTINGS.style.buffer if buffer is None else buffer


class MobMovementMixin:
    """Methods for moving Mobs around, mixed into
    :class:`~algan.animatable_base.mob.Mob`.
    """

    def move_between(self, loc1: Mob | torch.Tensor, loc2: Mob | torch.Tensor) -> Mob:
        """Move the Mob to the midpoint between two locations.

        Animation
        ---------
        Recorded as an animation: the Mob travels to the midpoint over the
        current context's duration (1 second by default). Applies to this Mob
        and its descendants.

        Parameters
        ----------
        loc1
            First endpoint: a 3-D point of shape ``(*, 3)``, or a Mob, in which
            case its center is used.
        loc2
            Second endpoint, in the same forms as ``loc1``.

        Returns
        -------
        :class:`~algan.animatable_base.mob.Mob`
            This Mob, so calls can be chained.
        """
        loc1, loc2 = [
            _.get_center() if hasattr(_, "get_center") else _ for _ in [loc1, loc2]
        ]
        return self.move_to((loc1 + loc2) / 2)

    def move_to_point_along_arc(
        self,
        point: torch.Tensor,
        arc_angle_degrees: float | torch.Tensor,
        arc_normal: torch.Tensor = OUT,
        recursive: bool = True,
    ) -> Mob:
        """Move the Mob to ``point`` along a signed circular arc.

        The start and target points form the chord of the arc. ``arc_normal``
        fixes the plane of the circle, and ``arc_angle_degrees`` is the signed
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
        ``with Seq(run_time=3): mob.move_to_point_along_arc(RIGHT, 90)``.

        Parameters
        ----------
        point
            The target location, shape ``(*, 3)``.
        arc_angle_degrees
            Signed arc sweep **in degrees**. Sweeps outside ``[-360, 360]`` are
            supported, except exact non-zero multiples of 360 degrees when the
            endpoints differ; such a path would require an infinite radius.
        arc_normal
            Normal vector of the arc plane; the chord from the current location
            to ``point`` must be perpendicular to it. Defaults to ``OUT`` (the
            -z axis, out of the screen), which arcs in the screen plane.
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
        angle_degrees = cast_to_tensor(arc_angle_degrees).to(device=device, dtype=dtype)

        if not torch.all(torch.isfinite(target)):
            raise ValueError("point must contain only finite values")
        if not torch.all(torch.isfinite(angle_degrees)):
            raise ValueError("arc_angle_degrees must contain only finite values")
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
        arc_angle_degrees: torch.Tensor,
        arc_normal: torch.Tensor,
        recursive: bool = True,
        interpolation: float | torch.Tensor = 1.0,
    ) -> Mob:
        """Apply a pre-validated arc displacement at ``interpolation``."""
        dtype = self.location.dtype
        device = self.location.device
        chord = cast_to_tensor(chord).to(device=device, dtype=dtype)
        normal = cast_to_tensor(arc_normal).to(device=device, dtype=dtype)
        angle_degrees = cast_to_tensor(arc_angle_degrees).to(device=device, dtype=dtype)
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
        self, location: torch.Tensor, path_arc_angle: float | None = None, **kwargs
    ) -> Mob:
        """Move the Mob to an absolute location.

        The path is a straight line unless ``path_arc_angle`` is given, in which
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
        path_arc_angle
            Signed sweep of the curved path, **in degrees**. Defaults to
            ``None``, meaning travel in a straight line.
        **kwargs
            Passed to :meth:`~algan.animatable_base.mob.Mob.set_location` (notably
            ``recursive``), or to
            :meth:`~algan.animatable_base.mob_movement.MobMovementMixin.move_to_point_along_arc`
            when ``path_arc_angle`` is
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
        if path_arc_angle is None:
            return self.set_location(location, **kwargs)
        return self.move_to_point_along_arc(location, path_arc_angle, **kwargs)

    def move(self, displacement: torch.Tensor, **kwargs) -> Mob:
        """Move the Mob by a displacement from wherever it currently is.

        Animation
        ---------
        Recorded as an animation: the Mob travels the displacement over the
        current context's duration (1 second by default). Retime it with
        ``with Seq(run_time=2): mob.move(RIGHT)``, or apply it instantly with
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
            -- notably ``path_arc_angle`` to
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
            square.move(DOWN, path_arc_angle=120)

            Scene.save_video()
        """
        self.move_to(self.location + cast_to_tensor(displacement), **kwargs)
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
            :meth:`~algan.animatable_base.mob_movement.MobMovementMixin.move_inline_with_boundary`),
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
        :meth:`~algan.animatable_base.mob_movement.MobMovementMixin.move_inline_with_center`
            Align centers along one axis without changing the others.
        :meth:`~algan.animatable_base.mob_movement.MobMovementMixin.move_inline_with_edge`
            Align edges along one axis without changing the others.
        """
        buffer = _resolve_buffer(buffer)
        normalized_direction = F.normalize(direction, p=2, dim=-1)
        # Get the boundary point of the target_mob along the given direction
        target_edge_point = (
            target_mob.get_boundary_in_direction(normalized_direction)
            if not isinstance(target_mob, torch.Tensor)
            else target_mob
        )
        # Get the boundary point of this mob in the opposite direction
        my_edge_point = self.get_boundary_in_direction(-normalized_direction)

        # Calculate the required displacement to move 'my_edge_point' to 'target_edge_point'
        # plus the buffer distance, and then apply it to the Mob's current location.
        displacement_to_align_edges = (
            target_edge_point + normalized_direction * buffer - my_edge_point
        )
        self.move(displacement_to_align_edges, **kwargs)
        if align_edge is not None:
            self.move_inline_with_boundary(target_mob, align_edge)
        return self

    def move_inline_with_edge(
        self,
        mob: Mob,
        direction: torch.Tensor,
        edge: torch.Tensor | None = None,
        buffer: float | None = None,
        **kwargs,
    ) -> Mob:
        """Line this Mob's edge up with another Mob's edge along one axis.

        Only the component of the movement along ``direction`` is applied, so
        this aligns the two Mobs on that axis and leaves their positions on the
        other axes untouched.

        Animation
        ---------
        Recorded as an animation over the current context's duration (1 second
        by default). Applies to this Mob and its descendants.

        Parameters
        ----------
        mob
            The Mob to align with.
        direction
            Axis to align along, and which of ``mob``'s edges to use (e.g.
            ``RIGHT``, ``UP``).
        edge
            Which of *this* Mob's edges to align. Defaults to ``None``, meaning
            use ``direction`` for both.
        buffer
            Gap to leave between the aligned edges, in world units. Defaults to
            ``SETTINGS.style.buffer`` (``0.6``).
        **kwargs
            Passed to
            :meth:`~algan.animatable_base.mob_movement.MobMovementMixin.move`.

        Returns
        -------
        :class:`~algan.animatable_base.mob.Mob`
            This Mob, so calls can be chained.
        """
        from algan.animatable_base.mob import Mob

        # Calculate the target location for this Mob if it were moved next to itself
        # using the specified `edge` direction and `buffer`. This acts as a reference point.
        old_location_reference = (
            Mob(scene=self.scene, add_to_scene=False)
            .move_next_to(self, direction if edge is None else edge, buffer)
            .location
        )
        # Calculate the target location for this Mob if it were moved next to the `mob`
        # using the primary `direction` and `buffer`.
        new_location_target = (
            Mob(scene=self.scene, add_to_scene=False)
            .move_next_to(mob, direction, buffer)
            .location
        )
        # Calculate the displacement needed to move from the reference point to the target point,
        # projected onto the `direction` to ensure alignment only along that axis.
        displacement = project_point_onto_line(
            new_location_target - old_location_reference, direction
        )
        self.move(displacement, **kwargs)
        return self

    def move_inline_with_center(self, mob: Mob, direction: torch.Tensor) -> Mob:
        """Line this Mob's center up with another Mob's center along one axis.

        Only the component of the movement along ``direction`` is applied:
        ``mob_a.move_inline_with_center(mob_b, UP)`` puts the two at the same
        height without changing how far apart they are horizontally.

        Animation
        ---------
        Recorded as an animation over the current context's duration (1 second
        by default). Applies to this Mob and its descendants.

        Parameters
        ----------
        mob
            The Mob whose center to align with.
        direction
            Axis to align along; need not be normalized.

        Returns
        -------
        :class:`~algan.animatable_base.mob.Mob`
            This Mob, so calls can be chained.
        """
        # Calculate the displacement vector from this Mob's center to the target Mob's center.
        displacement_to_target_center = mob.location - self.location
        # Project this displacement onto the `direction` to get the movement needed for alignment.
        alignment_displacement = project_point_onto_line(
            displacement_to_target_center, direction
        )
        self.move(alignment_displacement)
        return self

    def move_inline_with_mob(
        self,
        mob: Mob,
        align_direction: torch.Tensor,
        center: bool = False,
        from_mob: Mob | None = None,
        buffer: float | None = None,
    ) -> Mob:
        """Align this Mob with another along one axis, by edge or by center.

        The general form of
        :meth:`~algan.animatable_base.mob_movement.MobMovementMixin.move_inline_with_edge`
        and
        :meth:`~algan.animatable_base.mob_movement.MobMovementMixin.move_inline_with_center`:
        ``center`` picks which of the two
        behaviours you get, and ``from_mob`` lets a third Mob supply the
        reference point being moved into place.

        Animation
        ---------
        Recorded as an animation over the current context's duration (1 second
        by default). Applies to this Mob and its descendants.

        Parameters
        ----------
        mob
            The Mob to align with.
        align_direction
            Axis to align along; only movement along this axis is applied.
        center
            Whether to align centers rather than edges. Defaults to False,
            meaning this Mob's boundary is brought to ``mob``'s boundary.
        from_mob
            Mob supplying the reference point that is moved into alignment,
            useful when aligning a group by one of its members. Defaults to
            ``None``, meaning use this Mob.
        buffer
            Accepted for symmetry with the other alignment methods and **has no
            effect** here. Defaults to ``None``.

        Returns
        -------
        :class:`~algan.animatable_base.mob.Mob`
            This Mob, so calls can be chained.
        """
        if center:
            # Align centers
            mob_reference_point = mob.location
            from_mob_reference_point = (
                self.location if from_mob is None else from_mob.location
            )
        else:
            # Align edges
            mob_reference_point = mob.get_boundary_in_direction(align_direction)
            from_mob_reference_point = (
                self.get_boundary_in_direction(-align_direction)
                if from_mob is None
                else from_mob.get_boundary_in_direction(-align_direction)
            )

        # Calculate the overall displacement needed for alignment
        displacement = mob_reference_point - from_mob_reference_point
        # Normalize the alignment direction
        normalized_align_direction = F.normalize(align_direction, p=2, dim=-1)
        # Project the displacement onto the normalized direction to ensure movement only along that axis
        return self.move(
            dot_product(displacement, normalized_align_direction)
            * normalized_align_direction
        )

    def move_inline_with_boundary(self, mob: Mob, direction: torch.Tensor) -> Mob:
        """Align this Mob's boundary flush with another Mob's boundary.

        Unlike
        :meth:`~algan.animatable_base.mob_movement.MobMovementMixin.move_next_to`,
        no gap is left: the two boundaries end
        up coincident along ``direction``, which is what makes two Mobs share a
        bottom edge or a left edge.

        Animation
        ---------
        Recorded as an animation over the current context's duration (1 second
        by default). Applies to this Mob and its descendants.

        Parameters
        ----------
        mob
            The Mob whose boundary to align with.
        direction
            Which boundary to align (e.g. ``DOWN`` for bottom edges).

        Returns
        -------
        :class:`~algan.animatable_base.mob.Mob`
            This Mob, so calls can be chained.
        """
        return self.move(self.get_displacement_to_boundary(mob, direction))

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
            clone.move_to_corner(DOWN, LEFT)
            bottom_left = clone.location.clone()
            clone.move_to_corner(UP, LEFT)
            top_left = clone.location.clone()
            clone.move_to_corner(UP, RIGHT)
            top_right = clone.location.clone()
            clone.move_to_corner(DOWN, RIGHT)
            bottom_right = clone.location.clone()
            bottom = bottom_left * (1 - x) + x * bottom_right
            top = top_left * (1 - x) + x * top_right
            new_loc = bottom * (1 - y) + y * top
        return self.move_to(new_loc)

    def move_to_edge(self, edge: torch.Tensor, buffer: float | None = None) -> Mob:
        """Move the Mob against one edge of the screen.

        The Mob's own boundary is what comes to rest ``buffer`` from the border,
        so a large and a small shape both end up looking equally inset.

        Animation
        ---------
        Recorded as an animation over the current context's duration (1 second
        by default). The edge position is resolved from the camera when the call
        is recorded. Applies to this Mob and its descendants.

        Parameters
        ----------
        edge
            Which screen edge to move to: ``RIGHT``, ``LEFT``, ``UP`` or
            ``DOWN``.
        buffer
            Gap to leave between the Mob's boundary and the screen border, in
            world units. Defaults to ``SETTINGS.style.buffer`` (``0.6``).

        Returns
        -------
        :class:`~algan.animatable_base.mob.Mob`
            This Mob, so calls can be chained.

        See Also
        --------
        :meth:`~algan.animatable_base.mob_movement.MobMovementMixin.move_to_corner`
            Move against two edges at once.
        :meth:`~algan.animatable_base.mob_movement.MobMovementMixin.move_out_of_screen`
            Move all the way off-screen.
        """
        buffer = _resolve_buffer(buffer)
        normalized_edge = F.normalize(edge, p=2, dim=-1)
        # Get the boundary point of this Mob that is furthest towards the 'edge' direction
        mob_boundary_point = self.get_boundary_in_direction(normalized_edge)
        # Project this point onto the screen border to find the target point on the border
        edge_point_on_screen = self.scene.camera.project_point_onto_screen_border(
            mob_boundary_point, normalized_edge
        )
        # Calculate the final target location for the Mob, accounting for the buffer
        target_location = (
            edge_point_on_screen
            + F.normalize(mob_boundary_point - edge_point_on_screen, p=2, dim=-1)
            * buffer
        )
        # Calculate the displacement needed and move the Mob
        displacement = target_location - mob_boundary_point
        self.move(displacement)
        return self

    def move_to_corner(
        self, edge1: torch.Tensor, edge2: torch.Tensor, buffer: float | None = None
    ) -> Mob:
        """Move the Mob into a corner of the screen.

        The corner is named by the two edges that meet there, e.g.
        ``mob.move_to_corner(UP, RIGHT)`` for the top-right.

        Animation
        ---------
        Recorded as an animation. The two edge moves run inside a
        :class:`~algan.animation_timeline.animation_contexts.Sync`, so they happen
        simultaneously and the whole call still
        takes the current context's duration (1 second by default) rather than
        two seconds. Applies to this Mob and its descendants.

        Parameters
        ----------
        edge1
            First screen edge of the corner (e.g. ``UP``).
        edge2
            Second screen edge of the corner (e.g. ``RIGHT``).
        buffer
            Gap to leave from both screen borders, in world units. Defaults to
            ``SETTINGS.style.buffer`` (``0.6``).

        Returns
        -------
        :class:`~algan.animatable_base.mob.Mob`
            This Mob, so calls can be chained.
        """
        # Chain two calls to move_to_edge to reach the corner
        with Sync(animation_manager=self.animation_manager):
            return self.move_to_edge(edge1, buffer=buffer).move_to_edge(
                edge2, buffer=buffer
            )

    def move_out_of_screen(
        self, edge: torch.Tensor, buffer: float | None = None, despawn: bool = True
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
        edge
            Which way to leave: ``RIGHT``, ``LEFT``, ``UP`` or ``DOWN``.
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
        bbox = self.get_bounding_box()

        points_on_screen_edge = self.scene.camera.project_point_onto_screen_border(
            bbox, edge
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
            self.move(largest_disp + buffer * F.normalize(edge, p=2, dim=-1))
            if despawn:
                self.despawn(animate=False)
        return self

    def move_to_point_along_square(
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
        ``Seq(run_time=1)``, so the whole path takes 1 second regardless of the
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

        with Seq(run_time=1, animation_manager=self.animation_manager):
            self.move(displacement)
            self.move(orthogonal_displacement)
            self.location = destination
        return self
