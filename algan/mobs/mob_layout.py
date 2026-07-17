"""Screen-relative layout and bounding-box queries for :class:`~algan.mobs.mob.Mob`.

Split out of ``mob.py`` for readability; :class:`MobLayoutMixin` is mixed into
``Mob`` and is not useful standalone (``self`` is always a Mob).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from algan.animation.animation_contexts import Off, Seq, Sync
from algan.constants.spatial import *
from algan.geometry.geometry import project_point_onto_line
from algan.settings.style_defaults import STYLE_DEFAULTS
from algan.utils.tensor_utils import broadcast_gather, cast_to_tensor, dot_product

DEFAULT_BUFFER = STYLE_DEFAULTS.buffer


class MobLayoutMixin:
    """Bounding boxes, boundary queries, and screen-relative placement
    (``move_to_edge``, ``move_next_to``, ``move_inline_with_*``, ...)."""

    def get_axis_aligned_lower_corner(self):
        return self.location.amin(-2, keepdim=True)

    def get_axis_aligned_upper_corner(self):
        return self.location.amax(-2, keepdim=True)

    def _get_bounding_box_recursive(self, lower_corner, upper_corner):
        if not self.exclude_from_boundary:
            lower_corner = torch.minimum(
                lower_corner, self.get_axis_aligned_lower_corner()
            )
            upper_corner = torch.maximum(
                upper_corner, self.get_axis_aligned_upper_corner()
            )
        for c in self.children:
            lower_corner, upper_corner = c._get_bounding_box_recursive(
                lower_corner, upper_corner
            )
        return lower_corner, upper_corner

    def get_bounding_box(self):
        lower_corner, upper_corner = self._get_bounding_box_recursive(
            self.location.amin(-2, keepdim=True), self.location.amax(-2, keepdim=True)
        )
        out = torch.empty(*lower_corner.shape[:-2], 8, 3)
        lower_corner = lower_corner.squeeze(-2)
        upper_corner = upper_corner.squeeze(-2)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    a = torch.tensor((i, j, k), device=lower_corner.device)
                    out[..., i * 4 + j * 2 + k, :] = (
                        lower_corner * (1 - a) + (a) * upper_corner
                    )
        return out

    def get_boundary_points(self) -> torch.Tensor:
        """Returns the current location of the Mob, serving as its boundary point.
        For more complex Mobs, this should be overridden to provide actual boundary points.
        """
        return self.location

    def get_boundary_points_recursive(self) -> torch.Tensor:
        """Recursively collects boundary points from this Mob and all its descendants.

        Returns
        -------
        torch.Tensor
            A concatenated tensor of boundary points from all relevant Mobs
            in the hierarchy.

        """
        num_children = len(self.children)
        if num_children == 0:
            return self.get_boundary_points()
        elif num_children == 1:
            return self.children[0].get_boundary_points_recursive()
        return torch.cat(
            [
                child.get_boundary_points_recursive()
                for child in self.children
                if not child.exclude_from_boundary
            ],
            -2,
        )

    def _select_in_direction(self, points, direction):
        ind = dot_product(
            points, direction, dim=-1, keepdim=True
        ).argmax(-2, keepdim=True)
        return broadcast_gather(points, -2, ind, keepdim=True)

    def get_boundary_edge_point_recursive(self, direction):
        num_children = len(self.children)
        if num_children == 0:
            return self._select_in_direction(self.get_boundary_points(), direction)
        elif num_children == 1:
            return self.children[0].get_boundary_edge_point_recursive(direction)
        return self._select_in_direction(torch.cat([
                child.get_boundary_edge_point_recursive(direction)
                for child in self.children
                if not child.exclude_from_boundary
            ],
            -2,
            ), direction)

    def get_boundary_edge_point(self, direction: torch.Tensor) -> torch.Tensor:
        """Finds the point on the Mob's recursive boundary that is furthest in a given direction.

        Parameters
        ----------
        direction
            The 3-D vector indicating the direction
            along which to find the extreme boundary point.

        Returns
        -------
        torch.Tensor
            The 3-D coordinate of the boundary point furthest in `direction`.

        """
        return self.get_boundary_edge_point_recursive(direction)

    def get_center(self) -> torch.Tensor:
        """Gets the center (median mid-point) of the Mob and its descendants.

        """

        def get_median_location(tensor_values: torch.Tensor) -> torch.Tensor:
            """Calculates the median (midpoint of min/max) of a tensor's values."""
            max_val = tensor_values.amax(-2, keepdim=True)
            min_val = tensor_values.amin(-2, keepdim=True)
            return (max_val + min_val) * 0.5

        bbox = self.get_bounding_box()
        return get_median_location(bbox)

    def get_boundary_in_direction(self, direction: torch.Tensor) -> torch.Tensor:
        """Gets the point on the Mob's boundary (including children) that lies along
        the given direction from its center, and is furthest in that direction.

        Parameters
        ----------
        direction : torch.Tensor
            The 3-D vector defining the direction.

        Returns
        -------
        torch.Tensor
            The 3-D coordinate of the boundary point.

        """
        direction = F.normalize(direction, p=2, dim=-1)
        edge_point = self.get_boundary_edge_point(direction)

        # Get the logical center of the Mob (or its current location if no complex center is defined)
        mob_center = self.get_center()
        # Project the offset from the center to the edge point onto the direction
        # and add it back to the center to get the boundary point in that direction.
        return (
            project_point_onto_line(edge_point - mob_center, direction, dim=-1)
            + mob_center
        )

    def set_x_coord(self, target):
        return self.set_individual_coords(target, 0)

    def set_y_coord(self, target):
        return self.set_individual_coords(target, 1)

    def set_z_coord(self, target):
        return self.set_individual_coords(target, 2)

    def set_individual_coords(self, target, coord_indexes):
        from algan.mobs.mob import Mob

        if isinstance(target, Mob):
            target = target.location
        target = cast_to_tensor(target)
        if not hasattr(coord_indexes, "__len__"):
            coord_indexes = [coord_indexes]
        if target.shape[-1] != 1:
            target = target[..., coord_indexes]
        new_location = self.location.clone()
        new_location[..., coord_indexes] = target
        self.location = new_location
        return self

    def get_x_coord(self, *args, **kwargs):
        return self.get_individual_coords(0, *args, **kwargs)

    def get_y_coord(self, *args, **kwargs):
        return self.get_individual_coords(1, *args, **kwargs)

    def get_z_coord(self, *args, **kwargs):
        return self.get_individual_coords(2, *args, **kwargs)

    def get_individual_coords(self, coord_indexes, centered=False):
        l = self.get_center() if centered else self.location
        return l[..., coord_indexes].clone()

    def set_x_y_coord(self, xy_coords: torch.Tensor):
        """Sets the x and y coordinates of the Mob's location, preserving z."""
        new_location = self.location.clone()
        new_location[..., :2] = xy_coords[..., :2]
        self.location = new_location

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

    def get_length_in_direction(self, direction: torch.Tensor) -> torch.Tensor:
        """Calculates the spatial extent of the Mob along a given direction.
        This is the distance between the furthest points on its boundary
        in that direction and its opposite.

        Parameters
        ----------
        direction : torch.Tensor
            The 3-D vector defining the direction.

        Returns
        -------
        torch.Tensor
            The length of the Mob along the specified direction.

        """
        # Get the boundary points in the positive and negative directions and calculate their distance
        return (
            self.get_boundary_in_direction(direction)
            - self.get_boundary_in_direction(-direction)
        ).norm(p=2, dim=-1, keepdim=True)

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
        from algan.mobs.mob import Mob

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

    def get_displacement_to_boundary(
        self, mob: Mob, direction: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the vector displacement required to move this Mob's boundary
        to match another Mob's boundary along a given direction.

        Parameters
        ----------
        mob : Mob
            The target Mob.
        direction : torch.Tensor
            The direction along which to calculate the displacement.

        Returns
        -------
        torch.Tensor
            The displacement vector.
        """
        my_boundary = self.get_boundary_in_direction(direction)
        other_boundary = mob.get_boundary_in_direction(direction)
        return other_boundary - my_boundary

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
