"""Screen-relative layout and bounding-box queries for :class:`~algan.mobs.mob.Mob`.

Split out of ``mob.py`` for readability; :class:`MobLayoutMixin` is mixed into
``Mob`` and is not useful standalone (``self`` is always a Mob).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

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
        from algan.animatable_base.mob import Mob

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

    def get_points_evenly_along_direction(self, direction, num_points=3):
        e, s = (
            self.get_boundary_edge_point(direction),
            self.get_boundary_edge_point(-direction),
        )
        return [s * t + (1 - t) * e for t in torch.linspace(0, 1, num_points + 2)[1:-1]]

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
