"""Screen-relative layout and bounding-box queries for :class:`~algan.mobs.mob.Mob`.

Split out of ``mob.py`` for readability; :class:`MobLayoutMixin` is mixed into
``Mob`` and is not useful standalone (``self`` is always a Mob).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from algan.animatable_base.animatable import animated_function
from algan.animation_timeline.animation_contexts import Sync
from algan.constants.spatial import *
from algan.errors import AlganConfigurationError
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

    def get_axis_aligned_size(self) -> torch.Tensor:
        """Return the width, height, and depth of the recursive bounding box."""
        bbox = self.get_bounding_box()
        return bbox.amax(-2, keepdim=True) - bbox.amin(-2, keepdim=True)

    def get_width(self) -> torch.Tensor:
        """Return the width of the recursive axis-aligned bounding box."""
        return self.get_axis_aligned_size()[..., 0:1]

    def get_height(self) -> torch.Tensor:
        """Return the height of the recursive axis-aligned bounding box."""
        return self.get_axis_aligned_size()[..., 1:2]

    def get_depth(self) -> torch.Tensor:
        """Return the depth of the recursive axis-aligned bounding box."""
        return self.get_axis_aligned_size()[..., 2:3]

    def move_center_to(self, point) -> MobLayoutMixin:
        """Move the center of the recursive bounding box to ``point``.

        Unlike :meth:`~.Mob.move_to`, this aligns the visible bounding-box
        center rather than the Mob's anchor/location.
        """
        point = cast_to_tensor(point).to(
            device=self.location.device, dtype=self.location.dtype
        )
        return self.move(point - self.get_center())

    def _screen_point_at_depth(
        self, screen_position: torch.Tensor, depth_point: torch.Tensor
    ) -> torch.Tensor:
        """Map a normalized screen position onto ``depth_point``'s screen-parallel plane."""
        camera = self.scene.camera
        if camera is None:
            raise AlganConfigurationError(
                "Screen-relative layout requires the Scene to have a Camera"
            )

        corners = camera.get_corner_pixels()
        x, y = screen_position.unbind()
        bottom = torch.lerp(corners[0], corners[3], x)
        top = torch.lerp(corners[1], corners[2], x)
        point_on_screen = torch.lerp(bottom, top, y)

        camera_location = camera.location
        ray = point_on_screen - camera_location
        screen_normal = camera.get_forward_direction()
        denominator = dot_product(ray, screen_normal)
        if torch.any(denominator.abs() <= torch.finfo(ray.dtype).eps):
            raise AlganConfigurationError(
                "Cannot map screen coordinates at a depth parallel to the view ray"
            )
        distance = (
            dot_product(depth_point - camera_location, screen_normal) / denominator
        )
        return camera_location + ray * distance

    def _validate_screen_rectangle(self, bottom_left, top_right):
        defaults = ((0.0, 0.0), (1.0, 1.0))
        corners = []
        for name, value, default in zip(
            ("bottom_left", "top_right"),
            (bottom_left, top_right),
            defaults,
        ):
            if value is None:
                value = default
            try:
                value = torch.as_tensor(
                    value, device=self.location.device, dtype=self.location.dtype
                ).reshape(-1)
            except (TypeError, ValueError, RuntimeError) as exc:
                raise AlganConfigurationError(
                    f"screen rectangle {name} must be a pair of finite coordinates"
                ) from exc
            if value.numel() != 2 or not torch.isfinite(value).all():
                raise AlganConfigurationError(
                    f"screen rectangle {name} must be a pair of finite coordinates"
                )
            corners.append(value)

        bottom_left, top_right = corners
        if (
            torch.any(bottom_left < 0)
            or torch.any(top_right > 1)
            or torch.any(bottom_left >= top_right)
        ):
            raise AlganConfigurationError(
                "screen rectangle coordinates must satisfy "
                "0 <= bottom_left < top_right <= 1"
            )
        return bottom_left, top_right

    def move_center_to_screen_position(
        self, screen_position=(0.5, 0.5)
    ) -> MobLayoutMixin:
        """Move the bounding-box center to a normalized 2-D screen position.

        Screen coordinates range from ``(0, 0)`` at the bottom-left to
        ``(1, 1)`` at the top-right. The Mob keeps its current view depth.
        """
        try:
            position = torch.as_tensor(
                screen_position,
                device=self.location.device,
                dtype=self.location.dtype,
            ).reshape(-1)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise AlganConfigurationError(
                "screen_position must be a pair of finite screen coordinates"
            ) from exc
        if (
            position.numel() != 2
            or not torch.isfinite(position).all()
            or torch.any(position < 0)
            or torch.any(position > 1)
        ):
            raise AlganConfigurationError(
                "screen_position must contain two coordinates in [0, 1]"
            )
        center = self.get_center()
        return self.move_center_to(self._screen_point_at_depth(position, center))

    def scale_to_width(self, width) -> MobLayoutMixin:
        """Uniformly scale the Mob so its recursive bounding box has ``width``."""
        width = cast_to_tensor(width).to(
            device=self.location.device, dtype=self.location.dtype
        )
        current_width = self.get_width()
        if torch.any(width <= 0):
            raise AlganConfigurationError("width must be positive")
        if torch.any(current_width <= torch.finfo(current_width.dtype).eps):
            raise AlganConfigurationError("cannot scale a Mob with zero width")
        return self.scale(width / current_width)

    def scale_to_height(self, height) -> MobLayoutMixin:
        """Uniformly scale the Mob so its recursive bounding box has ``height``."""
        height = cast_to_tensor(height).to(
            device=self.location.device, dtype=self.location.dtype
        )
        current_height = self.get_height()
        if torch.any(height <= 0):
            raise AlganConfigurationError("height must be positive")
        if torch.any(current_height <= torch.finfo(current_height.dtype).eps):
            raise AlganConfigurationError("cannot scale a Mob with zero height")
        return self.scale(height / current_height)

    @animated_function(animated_args={"interpolation": 0.0})
    def _scale_about_point_along_world_axes(
        self, scale, about_point, interpolation=1.0
    ):
        """Apply an axis-aligned scale without depending on the Mob's local basis."""
        scale = torch.lerp(torch.ones_like(scale), scale, interpolation)
        locations = self.get_animated_attribute(
            "location", include_descendants=True, copy=False
        )
        bases = self.get_animated_attribute(
            "basis", include_descendants=True, copy=False
        )
        new_locations = about_point + (locations - about_point) * scale
        basis_matrices = bases.reshape(*bases.shape[:-1], 3, 3)
        new_bases = (basis_matrices * scale.unsqueeze(-2)).reshape_as(bases)
        self._apply_set("location", new_locations, recursive=True)
        self._apply_set("basis", new_bases, recursive=True)
        return self

    def fit_to_screen_rectangle(
        self,
        bottom_left=None,
        top_right=None,
        *,
        preserve_aspect_ratio: bool = False,
    ) -> MobLayoutMixin:
        """Scale and move this Mob's bounding box into a screen rectangle.

        This operates recursively, so calling it on a :class:`~.Group` lays
        out an entire collection while preserving the members' relative
        positions.

        Parameters
        ----------
        bottom_left
            Normalized ``(x, y)`` screen coordinates for the rectangle's
            bottom-left corner. Defaults to ``(0, 0)``.
        top_right
            Normalized ``(x, y)`` screen coordinates for the rectangle's
            top-right corner. Defaults to ``(1, 1)``.
        preserve_aspect_ratio
            If False (the default), scale x and y independently so the
            axis-aligned bounding box exactly occupies the rectangle. If True,
            use the largest uniform scale that keeps the Mob inside it.

        Returns
        -------
        :class:`~.Mob`
            The Mob instance itself, allowing for method chaining.
        """
        bottom_left, top_right = self._validate_screen_rectangle(
            bottom_left, top_right
        )
        bbox = self.get_bounding_box()
        source_lower = bbox.amin(-2, keepdim=True)
        source_upper = bbox.amax(-2, keepdim=True)
        source_size = source_upper - source_lower
        source_xy = source_size[..., :2]
        if torch.any(source_xy <= torch.finfo(source_xy.dtype).eps):
            raise AlganConfigurationError(
                "cannot fit a Mob whose bounding box has zero width or height"
            )

        source_center = (source_lower + source_upper) * 0.5
        target_lower = self._screen_point_at_depth(bottom_left, source_center)
        target_upper = self._screen_point_at_depth(top_right, source_center)
        target_center = (target_lower + target_upper) * 0.5
        target_xy = (target_upper - target_lower).abs()[..., :2]
        xy_scale = target_xy / source_xy

        if preserve_aspect_ratio:
            scale = xy_scale.amin(-1, keepdim=True)
        else:
            scale = torch.cat((xy_scale, torch.ones_like(xy_scale[..., :1])), -1)

        with Sync(animation_manager=self.animation_manager):
            if preserve_aspect_ratio:
                self.scale(scale)
            else:
                self._scale_about_point_along_world_axes(scale, source_center)
            self.move_center_to(target_center)
        return self

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
