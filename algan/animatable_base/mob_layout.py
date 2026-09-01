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
from algan.utils.tensor_utils import (
    broadcast_gather,
    cast_to_direction,
    cast_to_tensor,
    dot_product,
)

# Refinement budget for fit_to_screen. The scale converges
# superlinearly, but re-centering under perspective only converges linearly, so
# the cap is set by the latter; the common cases (a flat Mob, or any Mob under an
# unrotated camera) are exact after one pass and stop on the tolerance. The
# tolerance is in screen units, so it is orders of magnitude below one pixel.
_SCREEN_FIT_REFINEMENT_PASSES = 24
_SCREEN_FIT_TOLERANCE = 1e-6


class MobLayoutMixin:
    """Bounding boxes, boundary queries, and screen-relative placement
    (``move_to_screen_edge``, ``move_next_to``, ``align_with``, ...).
    """

    def get_bounding_box_min(self) -> torch.Tensor:
        """Get the minimum corner of the box enclosing this Mob and its children.

        The same box :meth:`get_bounding_box` returns the corners of, and the
        box :meth:`get_center` is the middle of.

        Returns
        -------
        torch.Tensor
            Per-axis minimum, shape ``(*, 1, 3)``.
        """
        return self.get_bounding_box().amin(-2, keepdim=True)

    def get_bounding_box_max(self) -> torch.Tensor:
        """Get the maximum corner of the box enclosing this Mob and its children.

        Returns
        -------
        torch.Tensor
            Per-axis maximum, shape ``(*, 1, 3)``.
        """
        return self.get_bounding_box().amax(-2, keepdim=True)

    @staticmethod
    def _box_corners(lower_corner, upper_corner):
        """Expand a min/max pair, ``(*, 1, 3)``, into the box's eight corners."""
        out = torch.empty(*lower_corner.shape[:-2], 8, 3, device=lower_corner.device)
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

    def _get_bounding_box_recursive(self, lower_corner, upper_corner, axes=None):
        location = self.location
        if axes is not None:
            location = location @ axes.transpose(-1, -2)
        if not self.exclude_from_boundary:
            lower_corner = torch.minimum(lower_corner, location.amin(-2, keepdim=True))
            upper_corner = torch.maximum(upper_corner, location.amax(-2, keepdim=True))
        for c in self.children:
            lower_corner, upper_corner = c._get_bounding_box_recursive(
                lower_corner, upper_corner, axes
            )
        return lower_corner, upper_corner

    def _get_bounding_box_aligned_to(self, axes=None) -> torch.Tensor:
        """Get this hierarchy's bounding box aligned to ``axes``' rows."""
        location = self.location
        if axes is not None:
            location = location @ axes.transpose(-1, -2)
        corners = self._box_corners(
            *self._get_bounding_box_recursive(
                location.amin(-2, keepdim=True),
                location.amax(-2, keepdim=True),
                axes,
            )
        )
        return corners if axes is None else corners @ axes

    def get_bounding_box(self) -> torch.Tensor:
        """Get the eight corners of the box enclosing this Mob and its children.

        The box is axis-aligned in world space, and children marked
        ``exclude_from_boundary`` (labels, helper geometry) are left out of it.

        Returns
        -------
        torch.Tensor
            The eight corner points, shape ``(*, 8, 3)``.
        """
        return self._get_bounding_box_aligned_to()

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

    def _edge_point_recursive(self, direction: torch.Tensor) -> torch.Tensor:
        """The outermost point of this Mob's whole hierarchy along a direction."""
        num_children = len(self.children)
        if num_children == 0:
            return self._select_in_direction(self.get_boundary_points(), direction)
        elif num_children == 1:
            return self.children[0]._edge_point_recursive(direction)
        return self._select_in_direction(
            torch.cat(
                [
                    child._edge_point_recursive(direction)
                    for child in self.children
                    if not child.exclude_from_boundary
                ],
                -2,
            ),
            direction,
        )

    def get_edge_point(
        self, direction: torch.Tensor, recursive: bool = True
    ) -> torch.Tensor:
        """Get the point on the Mob furthest along a direction.

        The actual outermost point of the geometry, which for an irregular shape
        is off to one side rather than straight out from the center. For the
        point straight out from the center, use
        :meth:`~algan.animatable_base.mob_layout.MobLayoutMixin.get_boundary_point`.

        Parameters
        ----------
        direction
            Direction to search along, shape ``(*, 3)``; need not be normalized.
        recursive
            Whether to search this Mob's descendants as well, so a Group reports
            the extreme point of whichever member reaches furthest. Defaults to
            True; pass False to search only this Mob's own points.

        Returns
        -------
        torch.Tensor
            The extreme boundary point, shape ``(*, 1, 3)``.
        """
        if not recursive:
            return self._select_in_direction(self.get_boundary_points(), direction)
        return self._edge_point_recursive(direction)

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

    def get_bounding_box_size(self) -> torch.Tensor:
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
        return self.get_bounding_box_size()[..., 0:1]

    def get_height(self) -> torch.Tensor:
        """Get the Mob's extent along the world y axis, in world units.

        Returns
        -------
        torch.Tensor
            Height of this Mob and its descendants, shape ``(*, 1, 1)``.
        """
        return self.get_bounding_box_size()[..., 1:2]

    def get_depth(self) -> torch.Tensor:
        """Get the Mob's extent along the world z axis, in world units.

        Returns
        -------
        torch.Tensor
            Depth of this Mob and its descendants, shape ``(*, 1, 1)``.
        """
        return self.get_bounding_box_size()[..., 2:3]

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

    def _screen_frame(self):
        """The camera plus the parameterization of its screen plane.

        Returns ``(camera, origin, edge_x, edge_y)`` where
        ``origin + sx * edge_x + sy * edge_y`` is the point of the screen plane
        at normalized screen coordinate ``(sx, sy)``, ``(0, 0)`` being the
        bottom-left of the frame.
        """
        camera = self.scene.camera
        if camera is None:
            raise AlganConfigurationError(
                "Screen-relative layout requires the Scene to have a Camera"
            )
        corners = camera.get_corner_pixels()
        origin = corners[0]
        return camera, origin, corners[3] - origin, corners[1] - origin

    def _resolved_screen_frame(self):
        """Everything :meth:`_project_to_screen_coords` needs, read once.

        ``(camera_location, normal, origin, edge_x, edge_y)``. Reading the camera
        costs animated-attribute lookups, so the fit solver resolves this before
        its loop rather than inside it.
        """
        camera, origin, edge_x, edge_y = self._screen_frame()
        return camera.location, camera.get_forward_direction(), origin, edge_x, edge_y

    def _project_to_screen_coords(
        self, points: torch.Tensor, frame=None
    ) -> torch.Tensor:
        """Normalized screen coordinates of world points, ``(*, N, 2)``.

        The inverse of :meth:`_screen_point_at_depth`: ``(0, 0)`` is the
        bottom-left of the frame and ``(1, 1)`` the top-right. This is the
        projection the renderer's ray generator inverts, so it accounts for
        perspective -- two points of equal world size project to different
        screen sizes when they sit at different depths.

        ``frame`` is an already-resolved ``(camera_location, normal, origin,
        edge_x, edge_y)``; callers that project repeatedly pass one so the loop
        does not re-read the camera's animated attributes every time.
        """
        if frame is None:
            frame = self._resolved_screen_frame()
        camera_location, normal, origin, edge_x, edge_y = frame
        ray = points - camera_location
        depth = dot_product(ray, normal)
        if torch.any(depth <= torch.finfo(ray.dtype).eps):
            raise AlganConfigurationError(
                "cannot project a Mob that is not entirely in front of the camera"
            )
        screen_depth = dot_product(origin - camera_location, normal)
        offset = camera_location + ray * (screen_depth / depth) - origin
        return torch.cat(
            (
                dot_product(offset, edge_x) / dot_product(edge_x, edge_x),
                dot_product(offset, edge_y) / dot_product(edge_y, edge_y),
            ),
            -1,
        )

    def _screen_axes(self):
        """The camera's frame as an orthonormal basis, or ``None``.

        Rows ``(right, up, -forward)``: ``x`` runs across the screen, ``y`` up
        it and ``z`` out of it towards the viewer. The third row is *minus* the
        camera's forward so that the default camera -- which looks down world
        ``-z`` from in front of the scene -- comes out as exactly the identity,
        and so the basis is right-handed like the world's.

        ``None`` means the camera's frame is the world frame, which lets callers
        keep the cheaper world-axis code paths for an unrotated camera. Built
        from ``forward`` this never fired: the default camera gave
        ``diag(1, 1, -1)``, never equal to the identity, so every caller took
        the rotated path to compute a result it already had. Flipping that row
        is invisible to them -- the depth axis is only ever used as the span of
        a bounding box (whose corner *set* a sign flip leaves alone) and in
        ``transpose(axes) @ diag(scale) @ axes``, a sum of per-row outer
        products in which each row's sign cancels with itself.
        """
        camera = self._screen_frame()[0]
        axes = torch.cat(
            (
                camera.get_right_direction(),
                camera.get_up_direction(),
                -camera.get_forward_direction(),
            ),
            -2,
        )
        identity = torch.eye(3, device=axes.device, dtype=axes.dtype).expand_as(axes)
        return None if torch.equal(axes, identity) else axes

    def _solve_screen_rectangle_fit(
        self, bbox, source_center, bottom_left, top_right, preserve_aspect_ratio, axes
    ):
        """Land a camera-aligned ``bbox`` on a screen rectangle.

        Iterative rather than closed-form because the projection is perspective:
        the screen size of a Mob with any depth depends on where its near face
        ends up, so a scale derived from a single depth plane (that of the Mob's
        center) overshoots -- a Cube fitted to the whole frame used to hang half
        its height off-screen.

        The projected extent is monotone in the scale, so each pass takes a
        secant step in log-log space towards the size that is still wanted, and
        slides the center back onto the rectangle's middle. A plain
        ``scale *= wanted / measured`` would do for a stretch (which leaves every
        depth alone, making the projection linear in the scale) but oscillates
        for a uniform scale, where growing the Mob also brings its near face
        closer to the camera and the projection grows faster than the scale.

        Returns ``(scale, center)``: the scale to apply about ``source_center``
        (uniform when ``preserve_aspect_ratio``, otherwise per ``axes`` axis with
        depth left alone) and the world point ``source_center`` must end up at.
        """
        target_size = top_right - bottom_left
        target_center = (bottom_left + top_right) * 0.5
        offsets = bbox - source_center
        tiny = torch.finfo(offsets.dtype).eps
        frame = self._resolved_screen_frame()
        camera_location, normal, origin, edge_x, edge_y = frame
        screen_depth = dot_product(origin - camera_location, normal)

        def as_scale(log_scale):
            scale = log_scale.exp()
            if preserve_aspect_ratio:
                return scale
            return torch.cat((scale, torch.ones_like(scale[..., :1])), -1)

        def scaled(scale):
            if axes is None or preserve_aspect_ratio:
                return offsets * scale
            return ((offsets @ axes.transpose(-1, -2)) * scale) @ axes

        def measure(log_scale, center):
            """Log excess over the target size, and the center's screen drift."""
            points = center + scaled(as_scale(log_scale))
            screen = self._project_to_screen_coords(points, frame)
            lower = screen.amin(-2, keepdim=True)
            upper = screen.amax(-2, keepdim=True)
            size = upper - lower
            if torch.any(size <= tiny):
                raise AlganConfigurationError(
                    "cannot fit a Mob whose bounding box projects to zero width "
                    "or height"
                )
            excess = size.log() - target_size.log()
            if preserve_aspect_ratio:
                # Only the axis that runs out of room first constrains a uniform
                # scale; the other is left with slack.
                excess = excess.amax(-1, keepdim=True)
            return excess, target_center - (lower + upper) * 0.5

        width = 1 if preserve_aspect_ratio else 2
        log_scale = torch.zeros_like(source_center[..., :width])
        # Seed from the Mob where it stands, so the refinement starts at roughly
        # the right size even for a Mob that has to travel far to its rectangle.
        log_scale = log_scale - measure(log_scale, source_center)[0]
        center = self._screen_point_at_depth(target_center, source_center)
        previous = None
        for _ in range(_SCREEN_FIT_REFINEMENT_PASSES):
            excess, drift = measure(log_scale, center)
            if torch.all(excess.abs() < _SCREEN_FIT_TOLERANCE) and torch.all(
                drift.abs() < _SCREEN_FIT_TOLERANCE
            ):
                break
            if previous is None:
                slope = torch.ones_like(excess)
            else:
                previous_log_scale, previous_excess = previous
                step = log_scale - previous_log_scale
                slope = (excess - previous_excess) / torch.where(
                    step.abs() > tiny, step, torch.ones_like(step)
                )
                # The projection can only grow with the scale, and never slower
                # than the scale itself; clamping keeps a degenerate secant (a
                # pass that barely moved, or one where the binding axis changed)
                # from throwing the step away.
                slope = slope.clamp(1.0, 32.0)
            previous = (log_scale, excess)
            log_scale = log_scale - excess / slope
            # Perspective makes the projected box drift off the rectangle's
            # center as the scale changes; slide the center back along the
            # screen-parallel plane it already sits on, which leaves its depth
            # (and therefore the scale just solved for) untouched.
            depth_ratio = dot_product(center - camera_location, normal) / screen_depth
            center = center + depth_ratio * (
                drift[..., :1] * edge_x + drift[..., 1:] * edge_y
            )
        return as_scale(log_scale), center

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
        self, scale, about_point, axes=None, interpolation=1.0
    ):
        """Apply an axis-aligned scale without depending on the Mob's local basis.

        ``axes`` is an optional orthonormal frame (rows are the axes, shape
        ``(*, 3, 3)``) to scale along instead of the world axes; ``None`` keeps
        the plain per-world-axis path.
        """
        scale = torch.lerp(torch.ones_like(scale), scale, interpolation)
        locations = self.get_animated_attribute(
            "location", include_descendants=True, copy=False
        )
        bases = self.get_animated_attribute(
            "basis", include_descendants=True, copy=False
        )
        basis_matrices = bases.reshape(*bases.shape[:-1], 3, 3)
        if axes is None:
            new_locations = about_point + (locations - about_point) * scale
            new_bases = (basis_matrices * scale.unsqueeze(-2)).reshape_as(bases)
        else:
            # Scaling along a rotated frame is the same map expressed in world
            # coordinates: transpose(axes) @ diag(scale) @ axes, applied on the
            # right because basis rows are local axes written in world space.
            transform = (axes.transpose(-1, -2) * scale) @ axes
            new_locations = about_point + (locations - about_point) @ transform
            new_bases = (basis_matrices @ transform).reshape_as(bases)
        self._apply_set("location", new_locations, recursive=True)
        self._apply_set("basis", new_bases, recursive=True)
        return self

    def fit_to_screen(
        self,
        bottom_left=None,
        top_right=None,
        *,
        preserve_aspect_ratio: bool = True,
    ) -> MobLayoutMixin:
        """Scale and move the Mob to fill a rectangle of the screen.

        Works on the camera-screen-aligned bounding box of the whole hierarchy,
        so its near face is parallel to the screen. Calling it on a
        :class:`~algan.mobs.group.Group` lays out the entire collection at once
        and keeps its members' relative positions. Handy for "put this diagram
        in the left half of the frame" without hand-tuning coordinates.

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
            undistorted but may leave slack on one axis. False stretches the Mob
            across the camera's right and up axes independently so its projection
            fills the rectangle exactly.

        Returns
        -------
        :class:`~algan.animatable_base.mob.Mob`
            This Mob, so calls can be chained.

        Raises
        ------
        :class:`~algan.errors.AlganConfigurationError`
            If the corners are not finite pairs satisfying
            ``0 <= bottom_left < top_right <= 1``, if the Mob's bounding box
            projects to zero width or height, if any of it sits behind the
            camera, or if the Scene has no camera.

        Notes
        -----
        The fit is measured on the projection of a camera-screen-aligned AABB,
        so it holds for a Mob with depth under perspective (the near face of a
        Cube is bigger on screen than its center slice) and for a rotated camera
        (the box follows the camera's right, up and forward axes, not the world
        axes).
        """
        bottom_left, top_right = self._validate_screen_rectangle(bottom_left, top_right)
        axes = self._screen_axes()
        bbox = self._get_bounding_box_aligned_to(axes)
        source_center = (
            bbox.amin(-2, keepdim=True) + bbox.amax(-2, keepdim=True)
        ) * 0.5
        scale, target_center = self._solve_screen_rectangle_fit(
            bbox, source_center, bottom_left, top_right, preserve_aspect_ratio, axes
        )

        with Sync(animation_manager=self.animation_manager):
            self._scale_about_point_along_world_axes(
                scale,
                source_center,
                axes=None if preserve_aspect_ratio else axes,
            )
            # Scaling about the camera-aligned box center leaves that point
            # fixed, so translating by the solved residual is exact.
            self.move(target_center - source_center)
        return self

    def get_boundary_point(self, direction: torch.Tensor) -> torch.Tensor:
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
        :meth:`~algan.animatable_base.mob_layout.MobLayoutMixin.get_edge_point`
            The true extreme point, which may be off-axis.
        """
        # Funnels get_length_in_direction, align_with, move_next_to and
        # Group.arrange_in_line, all of which otherwise fail with a bare
        # AttributeError on 'int' when handed a scalar.
        direction = F.normalize(cast_to_direction("direction", direction), p=2, dim=-1)
        edge_point = self.get_edge_point(direction)

        # Get the logical center of the Mob (or its current location if no complex center is defined)
        mob_center = self.get_center()
        # Project the offset from the center to the edge point onto the direction
        # and add it back to the center to get the boundary point in that direction.
        return (
            project_point_onto_line(edge_point - mob_center, direction, dim=-1)
            + mob_center
        )

    def set_coord(
        self,
        indices: int | list[int],
        value: Mob | torch.Tensor | float,
    ) -> MobLayoutMixin:
        """Move the Mob along selected world axes, leaving the others alone.

        The general form of the :attr:`x`, :attr:`y`, :attr:`z` and :attr:`xy`
        properties; pass a list to set several axes at once.

        Animation
        ---------
        Recorded as an animation over the current context's duration (1 second by
        default). Applies to this Mob and its descendants.

        Parameters
        ----------
        indices
            Which axes to write: ``0`` for x, ``1`` for y, ``2`` for z, or a list
            such as ``[0, 2]``.
        value
            New coordinate values in world units, or a Mob whose location
            supplies them. A multi-component value is indexed with ``indices``,
            so a full 3-D point can be passed and only the selected axes taken
            from it.

        Returns
        -------
        :class:`~algan.animatable_base.mob.Mob`
            This Mob, so calls can be chained.
        """
        from algan.animatable_base.mob import Mob

        if isinstance(value, Mob):
            value = value.location
        value = cast_to_tensor(value)
        if not hasattr(indices, "__len__"):
            indices = [indices]
        if value.shape[-1] != 1:
            value = value[..., indices]
        new_location = self.location.clone()
        new_location[..., indices] = value
        self.location = new_location
        return self

    def get_coord(
        self, indices: int | list[int], centered: bool = False
    ) -> torch.Tensor:
        """Get selected world coordinates of the Mob.

        Parameters
        ----------
        indices
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
        return location[..., indices].clone()

    @property
    def x(self) -> torch.Tensor:
        """The Mob's x coordinate in world units, shape ``(*, 1)``.

        Animation
        ---------
        Assignment is recorded like any other Mob attribute: ``mob.x = 3``
        slides the Mob over the current context's duration, and
        ``with Off(): mob.x = 3`` teleports it. Reading is not animated.
        """
        return self.get_coord(0)

    @x.setter
    def x(self, value):
        self.set_coord(0, value)

    @property
    def y(self) -> torch.Tensor:
        """The Mob's y coordinate in world units, shape ``(*, 1)``.

        Animation
        ---------
        Assignment is recorded, exactly as for :attr:`x`.
        """
        return self.get_coord(1)

    @y.setter
    def y(self, value):
        self.set_coord(1, value)

    @property
    def z(self) -> torch.Tensor:
        """The Mob's z coordinate in world units, shape ``(*, 1)``.

        The z axis runs out of the screen, so this is how far the Mob is from
        the camera.

        Animation
        ---------
        Assignment is recorded, exactly as for :attr:`x`.
        """
        return self.get_coord(2)

    @z.setter
    def z(self, value):
        self.set_coord(2, value)

    @property
    def xy(self) -> torch.Tensor:
        """The Mob's x and y coordinates in world units, shape ``(*, 2)``.

        Writing it moves the Mob in the screen plane and leaves z -- its
        distance from the camera -- alone. A value with more than two components
        may be assigned; the extra components are ignored.

        Animation
        ---------
        Assignment is recorded, exactly as for :attr:`x`.
        """
        return self.get_coord([0, 1])

    @xy.setter
    def xy(self, value):
        from algan.animatable_base.mob import Mob

        if isinstance(value, Mob):
            value = value.location
        self.set_coord([0, 1], cast_to_tensor(value)[..., :2])

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
            self.get_boundary_point(direction) - self.get_boundary_point(-direction)
        ).norm(p=2, dim=-1, keepdim=True)

    def sample_points_in_direction(
        self, direction: torch.Tensor, count: int = 3
    ) -> list[torch.Tensor]:
        """Get evenly spaced points spanning the Mob along a direction.

        The points are strictly inside the Mob's extent -- the two boundary
        points themselves are excluded -- which makes this convenient for hanging
        labels or ticks off a shape without landing on its edges.

        Parameters
        ----------
        direction
            Direction to space the points along, shape ``(*, 3)``.
        count
            How many points to return. Defaults to ``3``.

        Returns
        -------
        list[torch.Tensor]
            ``count`` points, each of shape ``(*, 1, 3)``, ordered from the
            ``direction`` end towards the opposite end.
        """
        e, s = (
            self.get_edge_point(direction),
            self.get_edge_point(-direction),
        )
        return [s * t + (1 - t) * e for t in torch.linspace(0, 1, count + 2)[1:-1]]

    def get_displacement_to_boundary(
        self, mob: Mob, direction: torch.Tensor
    ) -> torch.Tensor:
        """Get the displacement that would align this Mob's boundary with another's.

        The vector
        :meth:`~algan.animatable_base.mob_movement.MobMovementMixin.align_with`
        applies with ``anchor='boundary'``; useful when you want the number
        rather than the movement.

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
        my_boundary = self.get_boundary_point(direction)
        other_boundary = mob.get_boundary_point(direction)
        return other_boundary - my_boundary
