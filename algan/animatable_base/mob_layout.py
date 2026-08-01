"""Screen-relative layout and bounding-box queries for
:class:`~algan.animatable_base.mob.Mob`.

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
from algan.utils.tensor_utils import broadcast_gather, cast_to_tensor, dot_product


class MobLayoutMixin:
    """Bounding boxes, boundary queries, and screen-relative placement
    (``move_to_edge``, ``move_next_to``, ``move_inline_with_*``, ...).
    """

    def get_axis_aligned_lower_corner(self) -> torch.Tensor:
        """Get the minimum corner of this Mob's own points, ignoring children.

        Returns
        -------
        torch.Tensor
            Per-axis minimum of this Mob's points, shape ``(*, 1, 3)``. For the
            corner of the whole hierarchy, use
            :meth:`~algan.animatable_base.mob_layout.MobLayoutMixin.get_bounding_box`.
        """
        return self.location.amin(-2, keepdim=True)

    def get_axis_aligned_upper_corner(self) -> torch.Tensor:
        """Get the maximum corner of this Mob's own points, ignoring children.

        Returns
        -------
        torch.Tensor
            Per-axis maximum of this Mob's points, shape ``(*, 1, 3)``. For the
            corner of the whole hierarchy, use
            :meth:`~algan.animatable_base.mob_layout.MobLayoutMixin.get_bounding_box`.
        """
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

    def get_bounding_box(self) -> torch.Tensor:
        """Get the eight corners of the box enclosing this Mob and its children.

        The box is axis-aligned in world space, and children marked
        ``exclude_from_boundary`` (labels, helper geometry) are left out of it.

        Returns
        -------
        torch.Tensor
            The eight corner points, shape ``(*, 8, 3)``.
        """
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
        """Get the points that define this Mob's own outline.

        The base implementation returns the Mob's location, which is the whole
        outline of a point-like Mob. Shape classes override this to return their
        real silhouette, and that is what the placement methods measure against.

        Returns
        -------
        torch.Tensor
            Boundary points, shape ``(*, N, 3)``.
        """
        return self.location

    def get_boundary_points_recursive(self) -> torch.Tensor:
        """Get the outline points of this Mob and all its descendants.

        Children marked ``exclude_from_boundary`` contribute nothing, so helper
        geometry does not enlarge a Mob's apparent extent.

        Returns
        -------
        torch.Tensor
            All boundary points concatenated, shape ``(*, N, 3)``.
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
        ind = dot_product(points, direction, dim=-1, keepdim=True).argmax(
            -2, keepdim=True
        )
        return broadcast_gather(points, -2, ind, keepdim=True)

    def get_boundary_edge_point_recursive(
        self, direction: torch.Tensor
    ) -> torch.Tensor:
        """Get the outermost point of this Mob's hierarchy along a direction.

        Walks the hierarchy so a Group reports the extreme point of whichever
        member reaches furthest.
        :meth:`~algan.animatable_base.mob_layout.MobLayoutMixin.get_boundary_edge_point`
        is the
        public spelling of this.

        Parameters
        ----------
        direction
            Direction to search along, shape ``(*, 3)``.

        Returns
        -------
        torch.Tensor
            The extreme boundary point, shape ``(*, 1, 3)``.
        """
        num_children = len(self.children)
        if num_children == 0:
            return self._select_in_direction(self.get_boundary_points(), direction)
        elif num_children == 1:
            return self.children[0].get_boundary_edge_point_recursive(direction)
        return self._select_in_direction(
            torch.cat(
                [
                    child.get_boundary_edge_point_recursive(direction)
                    for child in self.children
                    if not child.exclude_from_boundary
                ],
                -2,
            ),
            direction,
        )

    def get_boundary_edge_point(self, direction: torch.Tensor) -> torch.Tensor:
        """Get the point on the Mob furthest along a direction.

        The actual outermost point of the geometry, which for an irregular shape
        is off to one side rather than straight out from the center. For the
        point straight out from the center, use
        :meth:`~algan.animatable_base.mob_layout.MobLayoutMixin.get_boundary_in_direction`.

        Parameters
        ----------
        direction
            Direction to search along, shape ``(*, 3)``; need not be normalized.

        Returns
        -------
        torch.Tensor
            The extreme boundary point, shape ``(*, 1, 3)``.
        """
        return self.get_boundary_edge_point_recursive(direction)

    def get_center(self) -> torch.Tensor:
        """Get the center of the box enclosing this Mob and its descendants.

        This is the midpoint of the bounding box, not the average of the Mob's
        points and not necessarily its
        :attr:`~algan.animatable_base.mob.Mob.location` -- the location is
        an anchor that can sit anywhere, while this is the middle of what the
        viewer sees.

        Returns
        -------
        torch.Tensor
            The bounding-box center, shape ``(*, 1, 3)``.
        """

        def get_median_location(tensor_values: torch.Tensor) -> torch.Tensor:
            """Calculates the median (midpoint of min/max) of a tensor's values."""
            max_val = tensor_values.amax(-2, keepdim=True)
            min_val = tensor_values.amin(-2, keepdim=True)
            return (max_val + min_val) * 0.5

        bbox = self.get_bounding_box()
        return get_median_location(bbox)

    def get_axis_aligned_size(self) -> torch.Tensor:
        """Get the Mob's size along the world x, y and z axes.

        Measured from the bounding box of this Mob and its descendants, in world
        units. Because the box is world-axis-aligned, a rotated Mob reports the
        size of its footprint, not of the shape itself.

        Returns
        -------
        torch.Tensor
            Width, height and depth, shape ``(*, 1, 3)``.
        """
        bbox = self.get_bounding_box()
        return bbox.amax(-2, keepdim=True) - bbox.amin(-2, keepdim=True)

    def get_width(self) -> torch.Tensor:
        """Get the Mob's extent along the world x axis, in world units.

        Returns
        -------
        torch.Tensor
            Width of this Mob and its descendants, shape ``(*, 1, 1)``.
        """
        return self.get_axis_aligned_size()[..., 0:1]

    def get_height(self) -> torch.Tensor:
        """Get the Mob's extent along the world y axis, in world units.

        Returns
        -------
        torch.Tensor
            Height of this Mob and its descendants, shape ``(*, 1, 1)``.
        """
        return self.get_axis_aligned_size()[..., 1:2]

    def get_depth(self) -> torch.Tensor:
        """Get the Mob's extent along the world z axis, in world units.

        Returns
        -------
        torch.Tensor
            Depth of this Mob and its descendants, shape ``(*, 1, 1)``.
        """
        return self.get_axis_aligned_size()[..., 2:3]

    def move_center_to(self, point: torch.Tensor) -> MobLayoutMixin:
        """Move the Mob so the middle of its bounding box lands on a point.

        Unlike
        :meth:`~algan.animatable_base.mob_movement.MobMovementMixin.move_to`, which
        places the Mob's anchor, this places
        what the viewer perceives as the middle -- the right choice for centering
        text or a Group whose anchor is off to one side.

        Animation
        ---------
        Recorded as an animation over the current context's duration (1 second by
        default). Applies to this Mob and its descendants.

        Parameters
        ----------
        point
            Where the bounding-box center should end up, shape ``(*, 3)``.

        Returns
        -------
        :class:`~algan.animatable_base.mob.Mob`
            This Mob, so calls can be chained.
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
        x, y = screen_position.unbind(-1)
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
        """Move the Mob so it appears centered at a position on the screen.

        The Mob keeps its distance from the camera, so this slides it across the
        view rather than towards or away from the viewer.

        Animation
        ---------
        Recorded as an animation over the current context's duration (1 second by
        default). The screen position is resolved from the camera when the call is
        recorded, so a later camera move will not keep the Mob pinned there.

        Parameters
        ----------
        screen_position
            Target ``(x, y)`` in screen units, from ``(0, 0)`` at the bottom-left
            to ``(1, 1)`` at the top-right. Defaults to ``(0.5, 0.5)``, the middle
            of the screen.

        Returns
        -------
        :class:`~algan.animatable_base.mob.Mob`
            This Mob, so calls can be chained.

        Raises
        ------
        :class:`~algan.errors.AlganConfigurationError`
            If ``screen_position`` is not two finite coordinates within
            ``[0, 1]``, or if the Scene has no camera.
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

    def scale_to_width(self, width: float | torch.Tensor) -> MobLayoutMixin:
        """Resize the Mob uniformly until it is a given width.

        The scale is uniform, so height and depth change by the same factor and
        the Mob keeps its proportions.

        Animation
        ---------
        Recorded as an animation over the current context's duration (1 second by
        default). Applies to this Mob and its descendants.

        Parameters
        ----------
        width
            Target width along the world x axis, in world units. Must be
            positive.

        Returns
        -------
        :class:`~algan.animatable_base.mob.Mob`
            This Mob, so calls can be chained.

        Raises
        ------
        :class:`~algan.errors.AlganConfigurationError`
            If ``width`` is not positive, or if the Mob's current width is zero
            and no scale factor could produce the target.
        """
        width = cast_to_tensor(width).to(
            device=self.location.device, dtype=self.location.dtype
        )
        current_width = self.get_width()
        if torch.any(width <= 0):
            raise AlganConfigurationError("width must be positive")
        if torch.any(current_width <= torch.finfo(current_width.dtype).eps):
            raise AlganConfigurationError("cannot scale a Mob with zero width")
        return self.scale(width / current_width)

    def scale_to_height(self, height: float | torch.Tensor) -> MobLayoutMixin:
        """Resize the Mob uniformly until it is a given height.

        The scale is uniform, so width and depth change by the same factor and the
        Mob keeps its proportions.

        Animation
        ---------
        Recorded as an animation over the current context's duration (1 second by
        default). Applies to this Mob and its descendants.

        Parameters
        ----------
        height
            Target height along the world y axis, in world units. Must be
            positive.

        Returns
        -------
        :class:`~algan.animatable_base.mob.Mob`
            This Mob, so calls can be chained.

        Raises
        ------
        :class:`~algan.errors.AlganConfigurationError`
            If ``height`` is not positive, or if the Mob's current height is zero
            and no scale factor could produce the target.
        """
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
        preserve_aspect_ratio: bool = True,
    ) -> MobLayoutMixin:
        """Scale and move the Mob to fill a rectangle of the screen.

        Works on the bounding box of the whole hierarchy, so calling it on a
        :class:`~algan.mobs.group.Group` lays out the entire collection at once and keeps its
        members' relative positions. Handy for "put this diagram in the left half
        of the frame" without hand-tuning coordinates.

        Animation
        ---------
        Recorded as an animation: the scale and the move run together inside a
        :class:`~algan.animation_timeline.animation_contexts.Sync`, over the current context's duration (1 second by
        default). The rectangle is resolved from the camera when the call is
        recorded.

        Parameters
        ----------
        bottom_left
            ``(x, y)`` screen coordinates of the rectangle's bottom-left corner,
            where ``(0, 0)`` is the bottom-left of the screen. Defaults to
            ``None``, meaning ``(0, 0)``.
        top_right
            ``(x, y)`` screen coordinates of the rectangle's top-right corner,
            where ``(1, 1)`` is the top-right of the screen. Defaults to
            ``None``, meaning ``(1, 1)`` -- the whole screen.
        preserve_aspect_ratio
            Whether to keep the Mob's proportions. Defaults to True: the largest
            uniform scale that still fits inside the rectangle, so the Mob is
            undistorted but may leave slack on one axis. False stretches x and y
            independently so the bounding box fills the rectangle exactly.

        Returns
        -------
        :class:`~algan.animatable_base.mob.Mob`
            This Mob, so calls can be chained.

        Raises
        ------
        :class:`~algan.errors.AlganConfigurationError`
            If the corners are not finite pairs satisfying
            ``0 <= bottom_left < top_right <= 1``, if the Mob's bounding box has
            zero width or height, or if the Scene has no camera.
        """
        bottom_left, top_right = self._validate_screen_rectangle(bottom_left, top_right)
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
        """Get the point where the Mob's boundary sits along a direction.

        The outermost boundary point projected back onto the line through the
        Mob's center, i.e. how far the Mob reaches that way, measured from its
        middle. This is what the placement methods use, so
        ``mob.move_next_to(other, RIGHT)`` leaves an even gap for irregular
        shapes too.

        Parameters
        ----------
        direction
            Direction to measure along, shape ``(*, 3)``; need not be normalized.

        Returns
        -------
        torch.Tensor
            The boundary point in that direction, shape ``(*, 1, 3)``.

        See Also
        --------
        :meth:`~algan.animatable_base.mob_layout.MobLayoutMixin.get_boundary_edge_point`
            The true extreme point, which may be off-axis.
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

    def set_x_coord(self, target: Mob | torch.Tensor | float) -> MobLayoutMixin:
        """Move the Mob along the world x axis only, leaving y and z alone.

        Animation
        ---------
        Recorded as an animation over the current context's duration (1 second by
        default). Applies to this Mob and its descendants.

        Parameters
        ----------
        target
            New x coordinate in world units, or a Mob whose x coordinate to take.

        Returns
        -------
        :class:`~algan.animatable_base.mob.Mob`
            This Mob, so calls can be chained.
        """
        return self.set_individual_coords(target, 0)

    def set_y_coord(self, target: Mob | torch.Tensor | float) -> MobLayoutMixin:
        """Move the Mob along the world y axis only, leaving x and z alone.

        Animation
        ---------
        Recorded as an animation over the current context's duration (1 second by
        default). Applies to this Mob and its descendants.

        Parameters
        ----------
        target
            New y coordinate in world units, or a Mob whose y coordinate to take.

        Returns
        -------
        :class:`~algan.animatable_base.mob.Mob`
            This Mob, so calls can be chained.
        """
        return self.set_individual_coords(target, 1)

    def set_z_coord(self, target: Mob | torch.Tensor | float) -> MobLayoutMixin:
        """Move the Mob along the world z axis only, leaving x and y alone.

        The z axis runs out of the screen, so this changes how far the Mob is
        from the camera.

        Animation
        ---------
        Recorded as an animation over the current context's duration (1 second by
        default). Applies to this Mob and its descendants.

        Parameters
        ----------
        target
            New z coordinate in world units, or a Mob whose z coordinate to take.

        Returns
        -------
        :class:`~algan.animatable_base.mob.Mob`
            This Mob, so calls can be chained.
        """
        return self.set_individual_coords(target, 2)

    def set_individual_coords(
        self, target: Mob | torch.Tensor | float, coord_indexes: int | list[int]
    ) -> MobLayoutMixin:
        """Move the Mob along selected world axes, leaving the others alone.

        The general form of
        :meth:`~algan.animatable_base.mob_layout.MobLayoutMixin.set_x_coord` and
        friends; pass a list to
        set several axes at once.

        Animation
        ---------
        Recorded as an animation over the current context's duration (1 second by
        default). Applies to this Mob and its descendants.

        Parameters
        ----------
        target
            New coordinate values, or a Mob whose location supplies them. A
            multi-component value is indexed with ``coord_indexes``, so a full 3-D
            point can be passed and only the selected axes are taken from it.
        coord_indexes
            Which axes to write: ``0`` for x, ``1`` for y, ``2`` for z, or a list
            such as ``[0, 2]``.

        Returns
        -------
        :class:`~algan.animatable_base.mob.Mob`
            This Mob, so calls can be chained.
        """
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

    def get_x_coord(self, *args, **kwargs) -> torch.Tensor:
        """Get the Mob's x coordinate in world units.

        Parameters
        ----------
        *args, **kwargs
            Passed to
            :meth:`~algan.animatable_base.mob_layout.MobLayoutMixin.get_individual_coords`
            -- notably
            ``centered=True`` to measure the bounding-box center instead of the
            Mob's anchor.

        Returns
        -------
        torch.Tensor
            The x coordinate, shape ``(*, 1)``.
        """
        return self.get_individual_coords(0, *args, **kwargs)

    def get_y_coord(self, *args, **kwargs) -> torch.Tensor:
        """Get the Mob's y coordinate in world units.

        Parameters
        ----------
        *args, **kwargs
            Passed to
            :meth:`~algan.animatable_base.mob_layout.MobLayoutMixin.get_individual_coords`
            -- notably
            ``centered=True``.

        Returns
        -------
        torch.Tensor
            The y coordinate, shape ``(*, 1)``.
        """
        return self.get_individual_coords(1, *args, **kwargs)

    def get_z_coord(self, *args, **kwargs) -> torch.Tensor:
        """Get the Mob's z coordinate in world units.

        Parameters
        ----------
        *args, **kwargs
            Passed to
            :meth:`~algan.animatable_base.mob_layout.MobLayoutMixin.get_individual_coords`
            -- notably
            ``centered=True``.

        Returns
        -------
        torch.Tensor
            The z coordinate, shape ``(*, 1)``.
        """
        return self.get_individual_coords(2, *args, **kwargs)

    def get_individual_coords(
        self, coord_indexes: int | list[int], centered: bool = False
    ) -> torch.Tensor:
        """Get selected world coordinates of the Mob.

        Parameters
        ----------
        coord_indexes
            Which axes to read: ``0`` for x, ``1`` for y, ``2`` for z, or a list
            of them.
        centered
            Whether to read the bounding-box center rather than the Mob's anchor
            location. Defaults to False, meaning the anchor.

        Returns
        -------
        torch.Tensor
            A copy of the requested coordinates, safe to keep and modify.
        """
        location = self.get_center() if centered else self.location
        return location[..., coord_indexes].clone()

    def set_x_y_coord(self, xy_coords: torch.Tensor):
        """Move the Mob in the screen plane, keeping its distance from the camera.

        Writes the x and y coordinates and leaves z as it is. Unlike its
        neighbours, this returns nothing, so it cannot be chained.

        Animation
        ---------
        Recorded as an animation over the current context's duration (1 second by
        default). Applies to this Mob and its descendants.

        Parameters
        ----------
        xy_coords
            New x and y in world units; the first two components of a tensor of
            shape ``(*, 2)`` or larger are used, so a 3-D point may be passed and
            its z ignored.

        Returns
        -------
        :class:`~algan.animatable_base.mob.Mob`
            This Mob, so calls can be chained.
        """
        new_location = self.location.clone()
        new_location[..., :2] = xy_coords[..., :2]
        self.location = new_location
        return self

    def get_length_in_direction(self, direction: torch.Tensor) -> torch.Tensor:
        """Get how far the Mob extends along an arbitrary direction.

        The distance between its two boundary points along that axis, so
        ``mob.get_length_in_direction(RIGHT)`` is its width and any other vector
        measures a diagonal extent.

        Parameters
        ----------
        direction
            Direction to measure along, shape ``(*, 3)``; need not be normalized.

        Returns
        -------
        torch.Tensor
            Length along that direction in world units, shape ``(*, 1, 1)``.
        """
        # Get the boundary points in the positive and negative directions and calculate their distance
        return (
            self.get_boundary_in_direction(direction)
            - self.get_boundary_in_direction(-direction)
        ).norm(p=2, dim=-1, keepdim=True)

    def get_points_evenly_along_direction(
        self, direction: torch.Tensor, num_points: int = 3
    ) -> list[torch.Tensor]:
        """Get evenly spaced points spanning the Mob along a direction.

        The points are strictly inside the Mob's extent -- the two boundary
        points themselves are excluded -- which makes this convenient for hanging
        labels or ticks off a shape without landing on its edges.

        Parameters
        ----------
        direction
            Direction to space the points along, shape ``(*, 3)``.
        num_points
            How many points to return. Defaults to ``3``.

        Returns
        -------
        list[torch.Tensor]
            ``num_points`` points, each of shape ``(*, 1, 3)``, ordered from the
            ``direction`` end towards the opposite end.
        """
        e, s = (
            self.get_boundary_edge_point(direction),
            self.get_boundary_edge_point(-direction),
        )
        return [s * t + (1 - t) * e for t in torch.linspace(0, 1, num_points + 2)[1:-1]]

    def get_displacement_to_boundary(
        self, mob: Mob, direction: torch.Tensor
    ) -> torch.Tensor:
        """Get the displacement that would align this Mob's boundary with another's.

        The vector
        :meth:`~algan.animatable_base.mob_movement.MobMovementMixin.move_inline_with_boundary`
        applies; useful when you
        want the number rather than the movement.

        Parameters
        ----------
        mob
            The Mob whose boundary is the target.
        direction
            Which boundary to align, e.g. ``DOWN`` for bottom edges.

        Returns
        -------
        torch.Tensor
            Displacement from this Mob's boundary to ``mob``'s, shape
            ``(*, 1, 3)``.
        """
        my_boundary = self.get_boundary_in_direction(direction)
        other_boundary = mob.get_boundary_in_direction(direction)
        return other_boundary - my_boundary
