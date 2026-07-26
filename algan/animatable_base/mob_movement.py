"""Screen-relative layout and bounding-box queries for :class:`~algan.mobs.mob.Mob`.

Split out of ``mob.py`` for readability; :class:`MobLayoutMixin` is mixed into
``Mob`` and is not useful standalone (``self`` is always a Mob).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

import warnings
from algan import RADIANS_TO_DEGREES
from algan.animation_timeline.animation_contexts import Off, Seq, Sync
from algan.constants.spatial import *
from algan.geometry.geometry import project_point_onto_line, rotate_vector_around_axis
from algan.settings.style_defaults import STYLE_DEFAULTS
from algan.utils.tensor_utils import broadcast_gather, cast_to_tensor, dot_product, broadcast_cross_product

DEFAULT_BUFFER = STYLE_DEFAULTS.buffer


class MobMovementMixin:
    """Helpful methods for moving mobs around. """

    def move_between(self, loc1, loc2):
        loc1, loc2 = [_.get_center() if hasattr(_, 'get_center') else _ for _ in [loc1, loc2]]
        return self.move_to((loc1 + loc2) / 2)

    def move_to_point_along_arc(
        self,
        point: torch.Tensor,
        arc_angle_degrees: float | torch.Tensor,
        arc_normal: torch.Tensor = OUT,
        recursive: bool = True,
    ) -> Mob:
        # TODO: This is bugged and needs to be fixed. The mathematical implementation for arc center calculation might be unstable or incorrect for all cases.
        """Moves the Mob to a target point along a circular arc. ***Currently bugged***

        Parameters
        ----------
        point : torch.Tensor
            The target 3-D location.
        arc_angle_degrees : float or torch.Tensor
            The angle subtended by the arc, in degrees. The sign determines
            the direction of rotation along the arc
            (clockwise/counter-clockwise).
        arc_normal : torch.Tensor, optional
            The normal vector to the plane of the arc. Defaults to `OUT`
            (positive Z-axis).
        recursive : bool, optional
            If True, applies the rotation recursively to children,
            maintaining their relative positions. Defaults to True.

        Returns
        -------
        Mob
            The Mob instance itself, allowing for method chaining.
        """
        warnings.warn(
            "move_to_point_along_arc (also reached via move_to(path_arc_angle=...)) "
            "is known to be bugged: the arc-center calculation can be unstable or "
            "wrong for some configurations.",
            stacklevel=2,
        )
        my_location = self.location
        displacement_unnormalized = point - my_location
        # Normalize the displacement for consistent direction calculations
        displacement_normalized = F.normalize(displacement_unnormalized, p=2, dim=-1)

        # Calculate a vector orthogonal to both displacement and arc_normal, which will define one axis for arc plane
        displacement_normal_orthogonal = F.normalize(
            broadcast_cross_product(displacement_normalized, arc_normal), p=2, dim=-1
        )

        angle_sign = cast_to_tensor(arc_angle_degrees).sign()
        abs_arc_angle_degrees = (
            abs(arc_angle_degrees)
            if not isinstance(arc_angle_degrees, torch.Tensor)
            else arc_angle_degrees.abs()
        )

        # Calculate two vectors `in1` and `in2` that define the tangents or radii for arc center calculation.
        # These are rotated versions of the normalized displacement, used to form a geometric intersection.
        in1 = F.normalize(
            rotate_vector_around_axis(
                displacement_normalized, abs_arc_angle_degrees - 90, arc_normal, -1
            ),
            p=2,
            dim=-1,
        )
        in2 = F.normalize(
            rotate_vector_around_axis(
                displacement_normalized, -(abs_arc_angle_degrees + 90), arc_normal, -1
            ),
            p=2,
            dim=-1,
        )

        # Calculate the angle of the full circumference based on the dot product of in1 and in2
        arc_circumference_angle = (
            dot_product(-in1, -in2).clamp_(min=-1, max=1).arccos_()
        )

        # Handle edge cases where angle is exactly 180 degrees or displacement is zero,
        # which can lead to division by zero or ambiguous arc centers.
        # In such cases, a simple midpoint is used as the arc center.
        zero_displacement_mask = (
            ((math.pi - arc_circumference_angle).abs() <= 1e-5)
            | (displacement_unnormalized.norm(p=2, dim=-1, keepdim=True) <= 1e-5)
        ).float()

        # Calculate arc center candidates using geometric intersection formulas.
        # These involve solving linear equations based on the dot products of vectors.
        arc_center1 = (
            my_location + point
        ) * 0.5  # Midpoint for 180-degree or zero-displacement cases

        x1, y1 = 0.0, 0.0
        x2, y2 = (
            dot_product(in1, displacement_normal_orthogonal),
            dot_product(in1, displacement_normalized),
        )
        x3, y3 = (
            dot_product(displacement_normalized, displacement_normal_orthogonal),
            dot_product(displacement_normalized, displacement_normalized),
        )
        x4, y4 = (
            dot_product(in2, displacement_normal_orthogonal),
            dot_product(in2, displacement_normalized),
        )

        # Solving for intersection point in a 2D plane defined by displacement_normal_orthogonal and displacement_normalized
        # These are standard formulas for line-line intersection, adapted for vector components.
        intersect_x = (
            (x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)
        ) / ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
        intersect_y = (
            (x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)
        ) / ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))

        # Reconstruct the arc center from the intersection point and the initial location
        arc_center2 = (
            my_location
            + intersect_x * displacement_normal_orthogonal
            + intersect_y * displacement_normalized
        )
        arc_center2 = arc_center2.nan_to_num_(
            0, 0, 0
        )  # Handle potential NaNs from division by zero

        # Select the appropriate arc center based on the edge case mask
        final_arc_center = (
            arc_center1 * (zero_displacement_mask)
            + (1 - zero_displacement_mask) * arc_center2
        )

        # Perform the rotation around the calculated arc center
        if recursive:
            return self.rotate_around_point(
                final_arc_center,
                arc_circumference_angle * RADIANS_TO_DEGREES * angle_sign,
                arc_normal,
            )
        else:
            return self.rotate_around_point_non_recursive(
                final_arc_center,
                arc_circumference_angle * RADIANS_TO_DEGREES * angle_sign,
                arc_normal,
            )

    def move_to(
        self, location: torch.Tensor, path_arc_angle: float | None = None, **kwargs
    ) -> Mob:
        """Moves the Mob to a specified location.

        If `path_arc_angle` is provided, the Mob moves along a circular arc.
        Otherwise, it moves in a straight line.

        Parameters
        ----------
        location : torch.Tensor
            The target 3-D location.
        path_arc_angle : float, optional
            The angle of the arc in degrees for curved movement. If None,
            movement is linear. Defaults to None.
        **kwargs
            Additional arguments passed to `set_location` or
            `move_to_point_along_arc`.

        Returns
        -------
        Mob
            The Mob instance itself.
        """
        if path_arc_angle is None:
            return self.set_location(location, **kwargs)
        return self.move_to_point_along_arc(location, path_arc_angle, **kwargs)

    def move(self, displacement: torch.Tensor, **kwargs) -> Mob:
        """Moves the Mob by a given displacement vector from its current location.

        Parameters
        ----------
        displacement : torch.Tensor
            The 3-D vector by which to move the Mob.
        **kwargs
            Additional arguments passed to `move_to` (e.g., `path_arc_angle`).

        Returns
        -------
        Mob
            The Mob instance itself, allowing for method chaining.
        """
        self.move_to(self.location + cast_to_tensor(displacement), **kwargs)
        return self

    def move_next_to(
        self,
        target_mob: Mob | torch.Tensor,
        direction: torch.Tensor,
        buffer: float = DEFAULT_BUFFER,
        align_edge=None,
        **kwargs,
    ) -> Mob:
        """Moves this Mob to be adjacent to another Mob (or a point) in a given direction.

        Parameters
        ----------
        target_mob
            The target Mob or a 3-D point (torch.Tensor) to move next to.
        direction
            The 3-D vector indicating the direction
            from `target_mob` towards where this Mob should be placed.
            This vector does not need to be normalized.
        buffer
            The minimum distance to maintain between
            the closest edges of the two Mobs. Defaults to `DEFAULT_BUFFER`.
        **kwargs
            Passed to :meth:`~.Mob.move_to` .

        Returns
        -------
        :class:`~.Mob`
            The Mob instance itself, allowing for method chaining.

        """
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
        self.move_to(self.location + displacement_to_align_edges, **kwargs)
        if align_edge is not None:
            self.move_inline_with_boundary(target_mob, align_edge)
        return self

    def move_inline_with_edge(
        self,
        mob: Mob,
        direction: torch.Tensor,
        edge: torch.Tensor | None = None,
        buffer: float = DEFAULT_BUFFER,
        **kwargs,
    ) -> Mob:
        """Moves this Mob so its specified edge is aligned with another Mob's edge
        along a given direction, while maintaining a buffer.

        Parameters
        ----------
        mob : Mob
            The target Mob to align with.
        direction : torch.Tensor
            The primary direction along which the alignment should occur
            (e.g., `RIGHT`, `UP`).
        edge : torch.Tensor, optional
            If specified, this direction is used to determine "which side" of
            *this* Mob to use for alignment. If None, `direction` is used for
            both. Defaults to None.
        buffer : float, optional
            The buffer distance to maintain between the edges. Defaults to
            `DEFAULT_BUFFER`.
        **kwargs
            Additional arguments for :meth:`~.Mob.move`.

        Returns
        -------
        Mob
            The Mob instance itself, allowing for method chaining.
        """
        from algan.animatable_base.mob import Mob

        # Calculate the target location for this Mob if it were moved next to itself
        # using the specified `edge` direction and `buffer`. This acts as a reference point.
        old_location_reference = (
            Mob(add_to_scene=False)
            .move_next_to(self, direction if edge is None else edge, buffer)
            .location
        )
        # Calculate the target location for this Mob if it were moved next to the `mob`
        # using the primary `direction` and `buffer`.
        new_location_target = (
            Mob(add_to_scene=False).move_next_to(mob, direction, buffer).location
        )
        # Calculate the displacement needed to move from the reference point to the target point,
        # projected onto the `direction` to ensure alignment only along that axis.
        displacement = project_point_onto_line(
            new_location_target - old_location_reference, direction
        )
        self.move(displacement, **kwargs)
        return self

    def move_inline_with_center(
        self, mob: Mob, direction: torch.Tensor, buffer: float = DEFAULT_BUFFER
    ) -> Mob:
        """Moves this Mob so its center is aligned with another Mob's center
        along a given direction.

        Parameters
        ----------
        mob : Mob
            The target Mob whose center will be aligned with.
        direction : torch.Tensor
            The 3-D vector specifying the alignment direction.
        buffer : float, optional
            Buffer distance (currently unused in this specific
            implementation, as it aligns centers, not edges). Defaults to
            `DEFAULT_BUFFER`.

        Returns
        -------
        Mob
            The Mob instance itself, allowing for method chaining.
        """
        # Calculate the displacement vector from this Mob's center to the target Mob's center.
        displacement_to_target_center = mob.location - self.location
        # Project this displacement onto the `direction` to get the movement needed for alignment.
        alignment_displacement = project_point_onto_line(
            displacement_to_target_center, direction
        )
        self.location = self.location + alignment_displacement
        return self

    def move_inline_with_mob(
        self,
        mob: Mob,
        align_direction: torch.Tensor,
        center: bool = False,
        from_mob: Mob | None = None,
        buffer: float = DEFAULT_BUFFER,
    ) -> Mob:
        """Moves this Mob to align with another Mob along a specific direction,
        either by their edges or by their centers.

        Parameters
        ----------
        mob
            The target Mob to align with.
        align_direction
            The 3-D vector defining the direction along which alignment should occur.
        center
            If True, aligns the centers of the Mobs. If False, aligns their edges.
        from_mob
            The Mob whose edge/center is considered the starting point for calculating displacement. If None,
            this Mob itself is used.
        buffer
            Buffer distance between aligned edges (only relevant
            if `center` is False). Defaults to `DEFAULT_BUFFER`.

        Returns
        -------
        :class:`~.Mob`
            The Mob instance itself, allowing for method chaining.
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
        """
        Moves this Mob so its boundary aligns with another Mob's boundary
        along a specific direction.

        Parameters
        ----------
        mob
            The target Mob whose boundary will be aligned with.
        direction
            The direction along which to align the boundaries.

        Returns
        -------
        :class:`~.Mob`
            The Mob instance itself, allowing for method chaining.

        """
        return self.move(self.get_displacement_to_boundary(mob, direction))

    def move_to_screen_position(self, x, y):
        """Moves the mob so that it appears at coordinate (x, y) on the screen.

        Parameters
        ----------
        x
            Horizontal position given between 0 (left edge) and 1 (right edge).
        y
            Vertical position given between 0 (bottom edge) and 1 (top edge).

        Returns
        -------
        :class:`~.Mob`
            The Mob instance itself, allowing for method chaining.

        """
        with Off():
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

    def move_to_edge(self, edge: torch.Tensor, buffer: float = DEFAULT_BUFFER) -> Mob:
        """Moves the Mob to an edge of the screen.

        Parameters
        ----------
        edge
            A 3-D vector indicating the screen edge direction (e.g., `RIGHT`, `LEFT`, `UP`, `DOWN`).
        buffer
            Distance to maintain from the screen border after moving. Defaults to `DEFAULT_BUFFER`.

        Returns
        -------
        :class:`~.Mob`
            The Mob instance itself, allowing for method chaining.
        """
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
        self, edge1: torch.Tensor, edge2: torch.Tensor, buffer: float = DEFAULT_BUFFER
    ) -> Mob:
        """Moves the Mob to a corner of the screen, defined by two intersecting edge directions.

        Parameters
        ----------
        edge1
            Vector for the first screen edge.
        edge2
            Vector for the second screen edge.
        buffer
            Distance to maintain from both screen borders. Defaults to `DEFAULT_BUFFER`.

        Returns
        -------
        :class:`~.Mob`
            The Mob instance itself, allowing for method chaining.
        """
        # Chain two calls to move_to_edge to reach the corner
        with Sync():
            return self.move_to_edge(edge1, buffer=buffer).move_to_edge(
                edge2, buffer=buffer
            )

    def move_out_of_screen(
        self, edge: torch.Tensor, buffer: float = DEFAULT_BUFFER, despawn: bool = True
    ) -> Mob:
        """Animates the Mob moving off-screen in a given edge direction and then optionally despawns it.

        Parameters
        ----------
        edge
            Vector indicating the direction to move off-screen.
        buffer
            Additional distance beyond the screen edge to move the Mob. Defaults to `DEFAULT_BUFFER`.
        despawn
            If True, the Mob is despawned immediately after moving off-screen.

        Returns
        -------
        :class:`~.Mob`
            The Mob instance itself, allowing for method chaining.

        """
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

        with Seq():  # Ensure movement and despawn happen sequentially
            self.move(largest_disp + buffer * F.normalize(edge, p=2, dim=-1))
            if despawn:
                self.despawn(animate=False)
        return self

    def move_to_point_along_square(
        self, destination: torch.Tensor, displacement: torch.Tensor
    ) -> Mob:
        """Moves the Mob to a destination in a two-step "square" path.
        First, it moves by the `displacement` vector. Then, it moves orthogonally
        to align with the `destination` point, and finally reaches the `destination`.
        This creates an [-shaped path.

        Parameters
        ----------
        destination
            The final target 3-D location.
        displacement
            The initial 3-D displacement vector for the first segment of the path.

        Returns
        -------
        :class:`~.Mob`
            The Mob instance itself, allowing for method chaining.

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

        with Seq(run_time=1):
            self.move(displacement)
            self.move(orthogonal_displacement)
            self.location = destination
        return self
