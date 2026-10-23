"""The 2-D shapes.

:class:`Circle`, :class:`Square`, :class:`Rectangle`, :class:`RegularPolygon`,
:class:`Polygon`, :class:`Quad`, :class:`Line`, :class:`Point`,
:class:`Dot` and :class:`SurroundingRectangle` are all cubic bezier circuits
(:class:`~algan.mobs.bezier_circuit.BezierCircuitCubic`), so they stay exactly
smooth at any zoom and morph into one another. They take ``color``,
``stroke_color`` and ``stroke_width``, plus ``grid_width`` /
``grid_height`` for shapes that carry a gradient or an image rather than
one flat color.

A second family -- :class:`~.Arc`, :class:`~.Annulus`, :class:`~.Ellipse`,
:class:`~.Star`, :class:`~.Arrow` -- comes from the Manim compatibility layer and
is constructed with Manim's arguments, including ``stroke_width`` and
``stroke_color`` in place of the border ones.

The ``Triangulated`` classes here build filled triangle meshes instead, for
interiors that need per-fragment shading.

See :doc:`/galleries/mob_gallery`.
"""

from __future__ import annotations

import math
import numbers
import operator

import torch.nn.functional as F

from algan.animatable_base.mob import Mob
from algan.animation_timeline.animation_contexts import (
    Off,
    active_scene_for_new_mob,
)
from algan.constants.color import *
from algan.constants.spatial import INWARD, LEFT, ORIGIN, RIGHT
from algan.errors import AlganConfigurationError
from algan.geometry.geometry import get_orthonormal_vector, map_local_to_global_coords
from algan.mobs.bezier_circuit import BezierCircuitCubic
from algan.mobs.stroke_style import _warn_background_stroke
from algan.settings import SETTINGS
from algan.settings.renderer_settings import effective_triangle_primitive
from algan.settings.shape_style_profiles import _manim_shape_style_for
from algan.utils.api_renames import (
    _reject_renamed_keywords,
    _warn_if_angle_looks_like_radians,
)
from algan.utils.tensor_utils import (
    broadcast_all,
    cast_to_direction,
    cast_to_tensor,
    mean,
)


def _coerce_algan_color(value, opacity=None):
    """Return an Algan :class:`Color` from a Manim-style color value."""
    # Parsed as ``mob.color = ...`` parses it. A name used to be looked up among
    # the palette constants first, so "red" was Manim's #FC6255 here and CSS red
    # everywhere else: ``Square(color="red")`` changed hue when given a
    # ``fill_opacity``, and ``Line(color="red")`` was not the red of a Square.
    # A hex int or an RGB sequence, which a bare cast took for a tensor of
    # channels -- 0xFF0000 for every one of them -- goes the same way.
    value = cast_to_tensor(to_color(value))
    value = Color.add_defaults(value).as_subclass(Color)
    if opacity is not None:
        value = value.set_opacity(opacity)
    return value


def _translate_vector_style_kwargs(
    kwargs, *, default_color=None, line=False, shape=None
):
    """Translate common VMobject style keywords to BezierCircuitCubic.

    Algan stores fill color on ``color`` and outline style on
    ``stroke_color``/``stroke_width``, which is what Manim calls them too --
    so ``stroke_*`` now passes straight through, and the work here is
    translating Manim's ``fill_*`` onto ``color`` and consuming the
    renderer-only VMobject settings that would otherwise leak into
    ``Animatable``.

    **``stroke_width`` is in Algan's unit here, not Manim's.** Manim means
    twice this by the same number, and that conversion lives in
    ``algan.manim`` -- ``mn.Square(stroke_width=4)`` is the exact-parity
    spelling.

    ``shape``, when given, is the Mob class being constructed; under the
    opt-in Manim shape profile (``SETTINGS.style.shape_style_profile``) its
    Manim constructor defaults are fed in for anything the caller did not pass
    -- an explicit keyword always wins over the profile.
    """
    kwargs = dict(kwargs)
    has_color = "color" in kwargs
    color = kwargs.get("color", default_color)

    fill_color = kwargs.pop("fill_color", None)
    fill_opacity = kwargs.pop("fill_opacity", None)
    stroke_color = kwargs.pop("stroke_color", None)
    stroke_opacity = kwargs.pop("stroke_opacity", None)
    stroke_width = kwargs.pop("stroke_width", None)

    style = _manim_shape_style_for(shape) if shape is not None else None
    if style is not None:
        if not has_color and style["color"] is not None:
            color = style["color"]
            has_color = True
        if (
            stroke_color is None
            and stroke_opacity is None
            and style["stroke_color"] is not None
        ):
            kwargs["stroke_color"] = style["stroke_color"]
        if stroke_width is None:
            kwargs["stroke_width"] = style["stroke_width"]
        if "filled" not in kwargs:
            kwargs["filled"] = style["filled"]

    if line:
        # A Line is an unfilled path; Manim's generic ``color`` controls its
        # stroke, while fill settings are accepted but have no visible effect.
        # The profile's own border color stands in for that stroke when it has
        # already been injected, so the (invisible) profile fill never leaks
        # into the stroke.
        if stroke_color is None and has_color:
            stroke_color = color
        kwargs["filled"] = False
    else:
        if fill_color is not None:
            color = fill_color
            has_color = True
        if fill_opacity is not None:
            if color is None:
                color = WHITE
            color = _coerce_algan_color(color, fill_opacity)
            has_color = True
        if has_color:
            kwargs["color"] = color

    if stroke_color is not None:
        kwargs["stroke_color"] = _coerce_algan_color(
            stroke_color,
            stroke_opacity,
        )
    elif stroke_opacity is not None:
        kwargs["stroke_color"] = _coerce_algan_color(
            WHITE if default_color is None else default_color,
            stroke_opacity,
        )
    if stroke_width is not None:
        kwargs["stroke_width"] = float(stroke_width)
    # Forwarded as well as folded into the colors above, so the circuit records
    # them as explicit component opacities that later color writes keep. A
    # line's fill is invisible, so only its stroke opacity matters.
    if fill_opacity is not None and not line:
        kwargs["fill_opacity"] = fill_opacity
    if stroke_opacity is not None:
        kwargs["stroke_opacity"] = stroke_opacity

    _warn_background_stroke(
        kwargs.get("background_stroke_width", 0),
        kwargs.get("background_stroke_opacity", 1),
    )
    # Accepted by Manim's VMobject/Mobject constructors but not represented by
    # Algan's ray-traced Bezier primitive.
    for key in (
        "background_stroke_color",
        "background_stroke_width",
        "background_stroke_opacity",
        "sheen_factor",
        "sheen_direction",
        "shade_in_3d",
        "tolerance_for_point_equality",
        "z_index",
    ):
        kwargs.pop(key, None)
    return kwargs


class Line(BezierCircuitCubic):
    """A straight or circular-arc line segment, drawn as a cubic bezier circuit.

    A :class:`Line` has no interior, so its stroke stays centred on the path
    rather than being drawn inside an outline the way a filled shape's border is.

    A straight line's control points are collinear, which has two consequences
    worth relying on. Its local frame's **first basis row points from the
    line's centre toward** :meth:`~.Line.get_start` -- that is a guarantee, not
    an accident of how the frame is derived. And its second row is synthesized
    perpendicular to the path rather than measured from it, so the texture grid
    defaults to ``grid_width`` samples along the line and a single row
    across it. Color along the line with
    :meth:`~.Line.set_color_by_function`, which parametrizes it by ``t`` running
    from the start to the end.

    To move a line's ends, animate them with :meth:`~.Line.put_start_and_end_on`.
    To lengthen a straight line about its centre, scale it along its own first
    axis -- ``line.scale(torch.tensor([2.0, 1.0, 1.0]))`` doubles it, whichever
    way it points -- because :attr:`~algan.animatable_base.mob.Mob.scale_coefficient`
    is in the line's own axes, not the world's: on a vertical line, a factor on
    component 1 scales it across its width, which a line does not have, and
    changes nothing visible.

    Parameters
    ----------
    start, end
        The endpoints, shape ``(*, 3)`` in world units, or a :class:`~.Mob` to
        attach to. Given a Mob, the line stops at its boundary in the direction
        of travel rather than at its center, so it never disappears underneath it.
        Default to ``LEFT`` and ``RIGHT``.
    buff
        Gap left at each end, in world units, so a line between two labelled
        points does not touch them. Defaults to ``0`` (no gap).
    path_arc
        Angle of the circular arc bulging the line away from straight, in
        degrees. Positive and negative values bulge opposite ways. A value that
        looks like radians (a non-integer smaller than a full turn) warns.
        Defaults to ``0`` (a straight segment). ``algan.manim.Line`` takes
        Manim's radians.
    *args, **kwargs
        Passed to :class:`~.BezierCircuitCubic` -- notably ``color``,
        ``stroke_width`` and ``grid_width``.

    Examples
    --------
    A straight line and an arc between the same two points:

    .. algan:: Example1Line
        :save_last_frame:

        from algan import *

        Line(LEFT, RIGHT, color=BLUE).spawn()
        Line(LEFT, RIGHT, path_arc=60, color=YELLOW).spawn()

        Scene.save_video()
    """

    def __init__(self, start=LEFT, end=RIGHT, buff=0, path_arc=0, *args, **kwargs):
        _warn_if_angle_looks_like_radians("path_arc", path_arc)
        start_center = (
            start.get_center()
            if isinstance(start, Mob)
            else cast_to_direction("start", start)
        )
        end_center = (
            end.get_center() if isinstance(end, Mob) else cast_to_direction("end", end)
        )
        direction = end_center - start_center
        if isinstance(start, Mob):
            start_center = start.get_boundary_point(direction)
        if isinstance(end, Mob):
            end_center = end.get_boundary_point(-direction)
        direction = end_center - start_center
        length = direction.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-10)
        if buff:
            unit = direction / length
            effective_buff = min(
                float(buff), max(float(length.reshape(-1)[0]) * 0.5, 0.0)
            )
            start_center = start_center + unit * effective_buff
            end_center = end_center - unit * effective_buff

        kwargs = _translate_vector_style_kwargs(kwargs, line=True, shape=type(self))

        if abs(float(path_arc)) > 1e-10:
            import manim as mn

            arc = mn.ArcBetweenPoints(
                start_center.detach().reshape(-1, 3)[0].cpu().numpy(),
                end_center.detach().reshape(-1, 3)[0].cpu().numpy(),
                angle=math.radians(float(path_arc)),  # Algan degrees -> Manim radians
            )
            control_points = torch.from_numpy(arc.points).to(
                device=start_center.device, dtype=start_center.dtype
            )
        else:
            control_points = torch.cat(
                [
                    start_center * (1 - a) + a * end_center
                    for a in torch.linspace(0, 1, 4, device=start_center.device)
                ],
                -2,
            )
        super().__init__(control_points, *args, **kwargs)

    def _path_ends(self):
        """The first and last control point of each line, one row per line.

        Not the first point of every cubic: an arc is drawn with several, and a
        pack of lines (``batch_mobs``) holds each line's cubics end to end.
        """
        points = self.control_points.location
        sizes = self.control_points.parent_batch_sizes
        if sizes is None or int(sizes.sum()) != points.shape[-2]:
            return points[..., :1, :], points[..., -1:, :]
        last = sizes.to(points.device).cumsum(0) - 1
        first = last - sizes.to(points.device) + 1
        return points[..., first, :], points[..., last, :]

    def get_start(self) -> torch.Tensor:
        """Get the point the line starts at.

        For an arc this is where the whole arc starts, not where each of the
        cubic segments it is drawn with does.

        Returns
        -------
        torch.Tensor
            The start point in world units, shape ``(*, 1, 3)``; a pack of lines
            made with :func:`~algan.utils.mob_utils.batch_mobs` has one row per
            line.
        """
        return self._path_ends()[0]

    def get_end(self) -> torch.Tensor:
        """Get the point the line ends at.

        For an arc this is where the whole arc ends, not where each of the
        cubic segments it is drawn with does.

        Returns
        -------
        torch.Tensor
            The end point in world units, shape ``(*, 1, 3)``; a pack of lines
            made with :func:`~algan.utils.mob_utils.batch_mobs` has one row per
            line.
        """
        return self._path_ends()[1]

    def get_vector(self) -> torch.Tensor:
        """Get the vector from the line's start to its end.

        For an arc this is its chord.

        Returns
        -------
        torch.Tensor
            :meth:`get_end` minus :meth:`get_start`, shape ``(*, 1, 3)``.
        """
        return self.get_end() - self.get_start()

    def get_unit_vector(self) -> torch.Tensor:
        """Get the direction from the line's start to its end.

        Returns
        -------
        torch.Tensor
            :meth:`get_vector` normalized to length 1, shape ``(*, 1, 3)``.
        """
        return F.normalize(self.get_vector(), p=2, dim=-1)

    def get_length(self) -> torch.Tensor:
        """Get the distance from the line's start to its end.

        For an arc this is the length of its chord, as in Manim, not of the
        curve.

        Returns
        -------
        torch.Tensor
            The length in world units, shape ``(*, 1)``.
        """
        return self.get_vector().norm(p=2, dim=-1)

    def set_color_by_function(self, function):
        """Color the line by a function of ``t``, its progress from start to end.

        The same as
        :meth:`BezierCircuitCubic.set_color_by_function <algan.mobs.bezier_circuit.BezierCircuitCubic.set_color_by_function>`,
        except that a line is one-dimensional: instead of ``(u, v)`` coordinates
        over a 2-D frame, the function is handed a single ``t`` running from 0 at
        :meth:`~.Line.get_start` to 1 at :meth:`~.Line.get_end`. A gradient along
        a line, or a value plotted onto one, is written directly in those terms.

        A line is unfilled, so this colors its stroke. The resolution is the
        line's texture grid, which is one flat color unless you asked for more:
        build it with ``grid_width`` (``Line(LEFT, RIGHT,
        grid_width=64)``). The height defaults to a single row across
        the line, so ``grid_width`` alone is the number of color
        samples along it.

        Animation
        ---------
        Recorded as an animation over the current context's runtime (1 second
        by default), so the colors cross-fade smoothly. Wrap the call in
        ``Off()`` to apply it instantly.

        Parameters
        ----------
        function
            Callable taking a ``t`` tensor of shape ``[..., 1]``, in ``[0, 1]``,
            and returning colors of shape ``[..., 3]`` (RGB), ``[..., 4]``
            (RGBA) or ``[..., 5]`` (RGB, glow, alpha -- Algan's internal channel
            order). Channels are in ``[0, 1]``; a missing alpha defaults to 1 and
            a missing glow to 0. Must be vectorized -- it is called once on the
            whole grid, not per texel.

        Returns
        -------
        :class:`~.Line`
            This line, so calls can be chained.

        Raises
        ------
        ValueError
            If the line has a single-texel texture grid, or if ``function``
            returns the wrong number of colors.

        Examples
        --------
        .. algan:: Example1LineSetColorByFunction
            :save_last_frame:

            from algan import *
            import torch

            line = Line(LEFT * 3, RIGHT * 3, stroke_width=20, grid_width=64)
            line.set_color_by_function(
                lambda t: torch.cat((t, torch.zeros_like(t), 1 - t), -1)
            )
            line.spawn()

            Scene.save_video()
        """
        self._require_texture_grid("set_color_by_function")
        return self._apply_texture_grid_colors(
            function(self._path_parameters()), "set_color_by_function's function"
        )

    def _path_parameters(self):
        """Internal: ``t`` at every texel of the texture grid, shape ``[W, H, 1]``.

        Each texel is projected onto the chord from :meth:`~.Line.get_start` to
        :meth:`~.Line.get_end`. For a straight line, whose first basis row points
        at the start and whose second is perpendicular to the path, that is
        exactly ``1 - u``; for an arc it measures progress along the chord,
        which is the only well-defined reading of "along the line" there.
        """
        uv = self.get_base_grid()
        center = self.location.reshape(-1, 3)[:1]
        basis = self.basis.reshape(-1, 9)[:1]
        points = (
            center
            + (2 * uv[..., :1] - 1) * basis[..., :3]
            + (2 * uv[..., 1:] - 1) * basis[..., 3:6]
        )
        # One row per bezier segment: the path starts at the first segment's
        # start and ends at the last one's end.
        start = self.get_start().reshape(-1, 3)[:1]
        chord = self.get_end().reshape(-1, 3)[-1:] - start
        length_squared = chord.square().sum(-1, keepdim=True).clamp_min(1e-12)
        return (
            ((points - start) * chord).sum(-1, keepdim=True) / length_squared
        ).clamp(0.0, 1.0)

    def put_start_and_end_on(
        self,
        start: torch.Tensor | tuple[float, float, float],
        end: torch.Tensor | tuple[float, float, float],
    ) -> Line:
        """Move the line's endpoints while preserving its appearance.

        Animation
        ---------
        For a spawned line, record the endpoint change over the current
        context's runtime (1 second by default). Use ``Sync(runtime=2)`` to
        change the duration or ``Off()`` to apply it immediately. Each end
        travels straight to its new point, so a line turned this way is shorter
        part-way through;
        :meth:`~algan.animatable_base.mob_orientation.MobOrientationMixin.rotate`
        turns it at its full length.
        Stroke color, opacity, width, caps, joins and draw order are preserved.

        Inside an updater it takes effect on every frame without animating, so
        ``line.add_updater(lambda mob, t: mob.put_start_and_end_on(
        a.get_center(), b.get_center()))`` keeps a line between two moving Mobs.

        Parameters
        ----------
        start
            New start point in world units, shape ``(*, 3)``.
        end
            New end point in world units, shape ``(*, 3)``.

        Returns
        -------
        :class:`~.Line`
            This Mob, so calls can be chained.

        Examples
        --------
        Turn a thick red line upright:

        .. algan:: Example1LinePutStartAndEndOn

            from algan import *

            line = Line(LEFT, RIGHT, color=RED, stroke_width=10).spawn()
            line.put_start_and_end_on(DOWN, UP)
            Scene.save_video()
        """
        if self.scene.timeline_manager.is_replaying():
            # An updater, rendering a batch of frames at once: its points hold
            # one row per frame, and no Line can be built for a batch to become.
            return self._place_ends_now(start, end)
        target = Line(
            start,
            end,
            scene=self.scene,
            add_to_scene=False,
            color=self.color,
            stroke_color=self.stroke_color,
            stroke_width=self.stroke_width,
            opacity=self.opacity,
            grid_width=self.grid_width,
            grid_height=self.grid_height,
            cap_style=self.cap_style,
            joint_type=self.joint_type,
            miter_limit=self.miter_limit,
            z_index=self.z_index,
            shade_in_3d=self.shade_in_3d,
        )
        target.z_index = self.z_index
        return self.become(target, detach_history=False)

    def _place_ends_now(self, start, end):
        """Turn, scale and move the line so its ends land on ``start`` and ``end``.

        Vectorized over the frames of a batch. A line with no length has no
        direction to turn from, so its points are laid along the new ends
        directly.
        """
        start, end = cast_to_tensor(start), cast_to_tensor(end)
        old_start, old_end = self.get_start(), self.get_end()
        old_vector, new_vector = torch.broadcast_tensors(
            old_end - old_start, end - start
        )
        old_length = old_vector.norm(p=2, dim=-1, keepdim=True)
        new_length = new_vector.norm(p=2, dim=-1, keepdim=True)
        pointless = old_length <= 1e-9
        cross = torch.cross(old_vector, new_vector, dim=-1)
        sine = cross.norm(p=2, dim=-1, keepdim=True)
        cosine = (old_vector * new_vector).sum(-1, keepdim=True)
        # Parallel or opposite ends have no cross product to turn about; any
        # axis perpendicular to the line turns it the 0 or 180 degrees needed.
        axis = torch.where(
            sine > 1e-9 * old_length * new_length,
            cross,
            get_orthonormal_vector(old_vector),
        )
        angle = torch.where(pointless, 0.0, torch.rad2deg(torch.atan2(sine, cosine)))
        factor = torch.where(pointless, 1.0, new_length / old_length.clamp_min(1e-9))
        self.rotate(angle, axis, about=old_start)
        self.scale(factor)
        self.move(start - self.get_start())
        if bool(pointless.any()):
            points = self.control_points.location
            cubics = points.shape[-2] // 4
            along = (
                torch.arange(cubics, device=points.device, dtype=points.dtype)[:, None]
                + torch.tensor([0, 1 / 3, 2 / 3, 1], device=points.device)
            ).reshape(-1, 1) / cubics
            laid = start + along * (end - start)
            self.control_points.location = torch.where(pointless, laid, points)
        return self


class Point(BezierCircuitCubic):
    """A single point, mostly useful as an invisible anchor.

    It is a degenerate bezier circuit -- four control points stacked on the same
    location -- so it occupies no area and renders as nothing at the default
    black. Reach for it to give a :class:`~algan.mobs.group.Group` a reference position, or as a
    :meth:`~algan.animatable_base.mob.Mob.become` target that collapses a shape
    to a dot. Use :class:`Dot` when you want something visible.

    Parameters
    ----------
    location
        Where the point sits, shape ``(*, 3)`` in world units. Defaults to
        ``ORIGIN``.
    *args, **kwargs
        Passed to :class:`~.BezierCircuitCubic`. ``color`` defaults to ``BLACK``
        here rather than the usual shape default.
    """

    def __init__(self, location=ORIGIN, *args, **kwargs):
        location = cast_to_tensor(location)
        kwargs = _translate_vector_style_kwargs(
            kwargs, default_color=BLACK, shape=type(self)
        )
        super().__init__(torch.cat([location for _ in range(4)], -2), *args, **kwargs)

    def get_num_points(self):
        """Get how many points this is, for Manim compatibility: always ``1``.

        Returns
        -------
        int
            ``1``.
        """
        return 1

    def get_points(self):
        """Get the point's location as a list of points, as Manim's would be.

        Returns
        -------
        torch.Tensor
            The location in world units, shape ``(1, 3)``.
        """
        return self.get_center().reshape(1, 3)

    def point_from_proportion(self, alpha):
        """Get the point a proportion of the way along: always its location.

        Parameters
        ----------
        alpha
            The proportion, from ``0`` to ``1``. Ignored: a point has no length.

        Returns
        -------
        torch.Tensor
            The location in world units, shape ``(*, 3)``.
        """
        return self.get_center()


class TriangleTriangulated(Mob):
    """A filled triangle mesh, built from one or more triangles given by their corners.

    Each triangle becomes a child holding its three corner vertices. The Mob is placed at
    the triangle's circumcentre, and its color is the average of its corners' colors.

    Parameters
    ----------
    corner_locations
        The corners in world units, shape ``(*, 3, 3)``: three points per triangle. A
        batch of triangles gives one Mob per triangle.
    vertices
        The Mob class used for the corners. Defaults to :class:`TriangleVertices`.
    normals
        Per-corner surface normals, shape ``(*, 3, 3)``, used for shading. Defaults to
        ``None``.
    **kwargs
        Passed to :class:`~.Mob`, and to the corner Mobs -- notably ``color``, which
        defaults to ``YELLOW``.
    """

    _morph_family = "mesh"

    def __init__(self, corner_locations, vertices=None, normals=None, **kwargs):
        corner_locations = cast_to_tensor(corner_locations)
        if vertices is None:
            vertices = TriangleVertices
        if "color" in kwargs:
            color = kwargs["color"]
            del kwargs["color"]
        else:
            color = self.get_default_color()
        super().__init__(**kwargs)
        kwargs["color"] = color
        if vertices == TriangleVertices:
            corner_locations = corner_locations.view(-1, 3, 3)
            kwargs["parent_batch_sizes"] = torch.full((len(corner_locations),), 3)
        else:
            corner_locations = corner_locations.view(-1, corner_locations.shape[-2], 3)
            kwargs["parent_batch_sizes"] = torch.full(
                (len(corner_locations),), corner_locations.shape[-2]
            )
        self.corners = vertices(corner_locations, normals, **kwargs)
        if vertices != TriangleVertices:
            with Off(
                record_attr_modifications=False,
                animation_manager=self.animation_manager,
            ):
                self.location = self.corners.location.mean(-2, keepdim=True)
            self.add_children(self.corners)
            return
        a = corner_locations[..., 0, :]
        b = corner_locations[..., 1, :]
        c = corner_locations[..., 2, :]
        m = (c - a).norm(p=2, dim=-1, keepdim=True).square() * torch.cross(
            torch.cross(b - a, c - a, -1), b - a, -1
        ) + (b - a).norm(p=2, dim=-1, keepdim=True).square() * torch.cross(
            torch.cross(c - a, b - a, -1), c - a, -1
        )
        m = a + m / (
            2
            * torch.cross(b - a, c - a, -1)
            .norm(p=2, dim=-1, keepdim=True)
            .square()
            .clamp_min_(1e-10)
        )
        with Off(
            record_attr_modifications=False, animation_manager=self.animation_manager
        ):
            self.location = m  # .unsqueeze(-2)
            if self.corners.color.shape[-2] > 1:
                corner_colors = self.corners.color.view(
                    -1, 3, self.corners.color.shape[-1]
                ).mean(-2)
            else:
                corner_colors = self.corners.color
            self.color = corner_colors
        self.add_children(self.corners)

    def get_default_color(self):
        """Get the color this Mob uses when none was given: ``YELLOW``."""
        return YELLOW


class TriangleVertices(Mob):
    """The corner vertices of a triangle mesh, rendered as filled triangles.

    Usually created for you by :class:`TriangleTriangulated`; each group of three
    consecutive locations is drawn as one triangle.

    Parameters
    ----------
    corner_locations
        The corners in world units, shape ``(*, 3, 3)``: three points per triangle.
    normals
        Per-corner surface normals, shape ``(*, 3, 3)``, used for shading. Defaults to
        ``None``.
    **kwargs
        Passed to :class:`~.Mob` -- notably ``color``, which defaults to ``PURE_RED``
        and may be given per corner.
    """

    _morph_family = "mesh"

    def __init__(self, corner_locations, normals=None, **kwargs):
        corner_locations = cast_to_tensor(corner_locations)
        kwargs2 = dict(kwargs.items())
        if "location" in kwargs2:
            del kwargs2["location"]
        kwargs2["location"] = corner_locations.reshape(-1, 3)
        if "color" in kwargs2:
            # Parsed here, as a circuit's color is: the reshape needs a tensor,
            # so a hex string or an RGB tuple failed every polyhedron with
            # "'str' object has no attribute 'reshape'".
            color = to_color(kwargs2["color"])
            kwargs2["color"] = color.reshape(-1, color.shape[-1])
        if normals is not None:
            normals = normals.reshape(-1, 3)
        super().__init__(**kwargs2)
        self.normals = normals
        self.is_primitive = True
        self.num_points_per_object = 3

    def _rebatch_structural_attrs(self, repeat_indices, *, child=None):
        if self.normals is None:
            return self
        repeat_indices = repeat_indices.to(self.normals.device)
        corner_offsets = torch.arange(3, device=self.normals.device)
        corner_indices = (repeat_indices[:, None] * 3 + corner_offsets).reshape(-1)
        self.normals = self.normals.index_select(-2, corner_indices)
        return self

    def _reorder_structural_attrs(self, permutation, *, child=None):
        return self._rebatch_structural_attrs(permutation, child=child)

    def _adopt_structural_attrs(self, target):
        super()._adopt_structural_attrs(target)
        self.normals = None if target.normals is None else target.normals.clone()
        return self

    def _get_memory_used_per_timestep(self):
        # Read uncopied: only the row count is wanted (``location`` is this
        # very read, plus a defensive clone nothing here keeps).
        n = self.get_animated_attribute("location", copy=False).shape[-2]
        # Source/animation-device state only: location(3) + color(5) +
        # normal(3), plus the primitive's cloned color(5). Ray-tracing bounds,
        # packed geometry and BVHs are built on the source device and their
        # finished storages are charged exactly when uploaded to ManualMemory.
        num_vars = 16
        for _ in self.get_shader_params().values():
            num_vars += _.shape[-1]
        return n * num_vars * 4

    def get_default_color(self):
        """Get the color this Mob uses when none was given: ``PURE_RED``."""
        return PURE_RED

    def get_render_primitives(self):
        """Internal: build the triangle primitive the renderer draws for this Mob."""
        # Read once: the corners and the normal frame below are the same
        # values, and each property read is a timeline query and a copy.
        location = self.location
        locations, c, o, n, g = broadcast_all(
            [
                location,
                # Plain tensor, not the Color subclass the public property
                # returns: the primitive only does arithmetic on these numbers,
                # and a subclass sends every operation it and the collection
                # merge apply through __torch_function__ (see the matching note
                # in Surface._build_render_primitive).
                self.color.as_subclass(torch.Tensor),
                self.opacity,
                self.normals,
                self.glow,
            ],
            ignored_dims=[-1],
        )
        if n is None:
            n = torch.zeros_like(locations)
        primitive = effective_triangle_primitive()(
            locations,
            c,
            o,
            F.normalize(
                map_local_to_global_coords(location, self.basis, n) - location,
                p=2,
                dim=-1,
            ),
            glow=g,
            shader=self.shader,
            **self.get_shader_params(),
        )
        # One SURFACE per mob, the Polyhedron pattern: without a declared
        # identity the batcher can merge several triangle mobs into one
        # collection member, and everything downstream that groups by surface
        # id then treats two unrelated mobs as one mesh. The fragment walk's
        # run rule carried that exposure bounded (it still shaded per
        # fragment); the sheet resolve shades once per surface per pixel, so
        # a quad and a backdrop sharing an id fused into one sheet and took
        # the backdrop's color (measured, DESIGN_sheet_resolve.md Phase 4a).
        # A mob's own triangles being one surface is also simply true: a
        # quad's diagonal is an interior edge, not a boundary.
        primitive.mesh_key = ("trimob", self.id)
        # A bare triangle mob has no outside (a Polyhedron's faces do -- it
        # sets ``two_sided`` False on them, and this is what carries that to
        # the renderer) and no closed shell for the same reason: a face of a
        # solid is skin, and it is the solid that declares the shell closed
        # (``Mob.closed_shell``) when its geometry proves one.
        primitive.declare_one_sided(not self.two_sided)
        primitive.declare_closed_shell(bool(getattr(self, "closed_shell", False)))
        primitive.declare_shadow_flags(*self._resolved_shadow_flags())
        return primitive


class QuadTriangulated(Mob):
    """A filled quadrilateral, drawn as two :class:`TriangleTriangulated` halves.

    The corners ``0, 1, 2`` form one triangle and ``2, 3, 0`` the other, so the corners
    should be given in order around the quad.

    Parameters
    ----------
    corner_locations
        The four corners in world units, shape ``(*, 4, 3)``.
    **kwargs
        Passed to both triangles and to :class:`~.Mob` -- notably ``color``, which
        defaults to ``YELLOW``.
    """

    def __init__(self, corner_locations, **kwargs):
        if kwargs.get("scene") is None:
            kwargs["scene"] = active_scene_for_new_mob()

        def q(_):
            return torch.cat((_[..., 2:4, :], _[..., :1, :]), -2)

        triangles = [
            TriangleTriangulated(corner_locations[..., :3, :], **kwargs),
            TriangleTriangulated(q(corner_locations), **kwargs),
        ]
        kwargs["location"] = mean([_.location for _ in triangles])
        super().__init__(**kwargs)
        self.triangles = triangles
        self.add_children(triangles)


def _is_point_sequence(value):
    """Whether ``value`` is a list or tuple of points rather than one point."""
    return isinstance(value, (list, tuple)) and not any(
        isinstance(item, numbers.Number) for item in value
    )


class Polygon(BezierCircuitCubic):
    """A 2-D planar polygon with N vertices, joined by straight edges.

    Parameters
    ----------
    *vertex_locations
        The vertices in order, in world units: one point per argument,
        ``Polygon(LEFT, RIGHT, UP)``, one list of them, or a single tensor of
        shape ``(N, 3)``. Each vertex is joined to the next and the last back
        to the first. At least three are required.
    **kwargs
        Passed to :class:`~.BezierCircuitCubic` -- notably ``color``, which
        defaults to ``RED``, and the ``stroke_*`` styling.

    Raises
    ------
    :class:`.AlganConfigurationError`
        If fewer than three vertices are given.
    """

    def __init__(self, *vertex_locations: torch.Tensor, **kwargs):
        kwargs = _translate_vector_style_kwargs(
            kwargs, default_color=RED, shape=type(self)
        )
        if not vertex_locations or (
            len(vertex_locations) == 1
            and isinstance(vertex_locations[0], (list, tuple))
            and not vertex_locations[0]
        ):
            # ``torch.stack([])`` below would answer "stack expects a
            # non-empty TensorList", which says nothing about polygons.
            raise AlganConfigurationError(
                f"{type(self).__name__} needs its vertices: pass them as "
                f"points ({type(self).__name__}(LEFT, RIGHT, UP)), as a list "
                f"of points or as one [N, 3] tensor."
            )
        if len(vertex_locations) == 1 and _is_point_sequence(vertex_locations[0]):
            # One list of points, which may mix tensors and tuples; a bare cast
            # of the list takes a tensor in it for a number.
            vertex_locations = tuple(vertex_locations[0])
        if len(vertex_locations) == 1:
            corner_locations = cast_to_tensor(vertex_locations[0])
            while corner_locations.dim() > 2 and corner_locations.shape[0] == 1:
                corner_locations = corner_locations[0]
        else:
            corner_locations = torch.stack(
                [
                    cast_to_direction(f"vertex {index}", vertex).reshape(-1, 3)[0]
                    for index, vertex in enumerate(vertex_locations)
                ],
                dim=0,
            )
        if corner_locations.shape[-2] < 3:
            raise AlganConfigurationError("Polygon requires at least three vertices")
        control_points = []
        for line_start, line_end in zip(
            corner_locations, corner_locations.roll(-1, -2)
        ):
            control_points.append(
                torch.stack(
                    [
                        line_start * (1 - a) + a * line_end
                        for a in torch.linspace(0, 1, 4)
                    ]
                )
            )

        control_points = torch.cat(control_points, -2)
        super().__init__(control_points, **kwargs)

    def get_default_color(self):
        """Get the color this Mob uses when none was given.

        Returns
        -------
        :class:`~.Color`
            ``RED`` for a Polygon and the shapes built on it.
        """
        return RED


class RegularPolygon(Polygon):
    """A regular polygon: ``n`` equal sides on a circle.

    Parameters
    ----------
    n
        Number of sides. Defaults to ``6``; must be at least 3.
    radius
        Distance from the center to each vertex, in world units. Defaults to ``1``.
    start_angle
        Angle of the first vertex, in degrees. A value that looks like radians
        (a non-integer smaller than a full turn, such as ``PI / 2``) warns.
        Defaults to ``None``, which puts the first vertex at the top and repeats
        the closing vertex -- a topology that matters when morphing with
        :meth:`~algan.animatable_base.mob_morph.MobMorphMixin.become`.
    **kwargs
        Passed to :class:`~.Polygon` and on to :class:`~.BezierCircuitCubic`.

    Animation
    ---------
    Constructs an unspawned polygon immediately. Call ``spawn()`` before
    recording animations. ``algan.manim.RegularPolygon`` retains radians.

    Raises
    ------
    :class:`~.AlganConfigurationError`
        If fewer than 3 sides are requested.

    Examples
    --------
    .. algan:: Example1RegularPolygonAngle

        from algan import *

        RegularPolygon(n=5, start_angle=90, color=BLUE).spawn()
        Scene.save_video()
    """

    def __init__(
        self,
        n: int = 6,
        *,
        radius: float = 1,
        start_angle: float | None = None,
        **kwargs,
    ):
        _reject_renamed_keywords(
            "RegularPolygon",
            kwargs,
            {"num_vertices": "n"},
            manim_alternative="RegularPolygon",
        )
        if isinstance(n, float) and n.is_integer():
            n = int(n)
        try:
            n = operator.index(n)
        except TypeError:
            raise AlganConfigurationError(
                f"RegularPolygon's n must be a whole number of sides; got {n!r}"
            ) from None
        if n < 3:
            raise AlganConfigurationError("RegularPolygon requires n >= 3")
        if start_angle is None:
            # Preserve Algan's original topology and orientation: the first
            # vertex is at the top and the closing vertex is repeated.  The
            # latter is observable during ``become`` because it determines
            # how cubic segments are paired.
            angles = torch.linspace(math.pi / 2, -math.pi * 1.5, n + 1)
            self.start_angle = 90.0
        else:
            _warn_if_angle_looks_like_radians("start_angle", start_angle)
            angles = math.radians(start_angle) + torch.arange(n) * (2 * math.pi / n)
            self.start_angle = start_angle
        vertices = torch.stack(
            (
                radius * torch.cos(angles),
                radius * torch.sin(angles),
                torch.zeros_like(angles),
            ),
            dim=-1,
        )
        self.n = n
        super().__init__(*vertices, **kwargs)


class Quad(Polygon):
    """A four-sided polygon, given its four corners.

    The base class of :class:`~.Rectangle`; use it directly for a quadrilateral that is
    not axis-aligned.

    Parameters
    ----------
    *args, **kwargs
        Passed to :class:`~.Polygon` -- the four corner points, plus
        :class:`~.BezierCircuitCubic` styling.
    """

    pass


class Triangle(RegularPolygon):
    """An equilateral triangle pointing up.

    Parameters
    ----------
    **kwargs
        Passed to :class:`~.RegularPolygon` with ``n=3`` -- notably ``radius``,
        ``start_angle`` and ``color``.
    """

    def __init__(self, **kwargs):
        super().__init__(n=3, **kwargs)


class Rectangle(Quad):
    """A rectangle, drawn as a cubic bezier circuit.

    Parameters
    ----------
    width
        Width in world units. Defaults to ``2``.
    height
        Height in world units. Defaults to ``2`` -- so the default ``Rectangle()``
        is a 2x2 square.
    color
        Fill color. Defaults to ``None``, meaning the class default (see
        :meth:`~algan.animatable_base.animatable.Animatable.get_default_color`).
    **kwargs
        Passed to :class:`~.BezierCircuitCubic` -- notably ``location``,
        ``stroke_color``, ``stroke_width`` and ``filled``.
    """

    def __init__(self, width=2, height=2, color=None, **kwargs):
        for key in (
            "grid_xstep",
            "grid_ystep",
            "mark_paths_closed",
            "close_new_points",
        ):
            kwargs.pop(key, None)
        if color is not None:
            kwargs.setdefault("color", color)
        corners = (
            torch.tensor(
                (
                    (-width, height, 0),
                    (width, height, 0),
                    (width, -height, 0),
                    (-width, -height, 0),
                )
            )
            * 0.5
        )
        if "location" in kwargs:
            corners = corners + cast_to_direction("location", kwargs["location"])
            del kwargs["location"]
        super().__init__(corners, **kwargs)


class SurroundingRectangle(Quad):
    """A rectangle sized to enclose one or more other Mobs.

    The rectangle is built around the Mobs' combined bounding box at construction
    time, so it does not track them afterwards -- rebuild it, or add an updater, if
    the contents move.

    Parameters
    ----------
    *mobjects
        The Mobs to enclose. At least one is required.
    color
        Color of the rectangle. Defaults to ``None``, meaning the class default.
    buff
        Gap between the contents and the rectangle, in world units. A ``(horizontal,
        vertical)`` pair sets the two axes separately. Defaults to ``None``, meaning
        half of ``SETTINGS.style.buffer`` (``0.3``).
    corner_radius
        Corner rounding in world units. Defaults to ``0.0``, i.e. square corners.
    buffer
        Alias of ``buff``, for consistency with Algan's other spacing parameters. When
        both are given, this one wins. Defaults to ``None``.
    bottom_buffer
        Additional gap below the contents, in world units, for leaving room for a
        caption. Defaults to ``None``, meaning none.
    **kwargs
        Passed to :class:`~.BezierCircuitCubic` -- notably ``stroke_color``,
        ``stroke_width`` and ``filled=False`` for an outline-only frame.

    Raises
    ------
    ValueError
        If no Mobs are given.
    """

    def __init__(
        self,
        *mobjects,
        color=None,
        buff=None,
        corner_radius=0.0,
        buffer=None,
        bottom_buffer=None,
        **kwargs,
    ):
        if not mobjects:
            raise AlganConfigurationError(
                "SurroundingRectangle requires at least one Mobject"
            )
        if buffer is not None:
            buff = buffer
        elif buff is None:
            buff = SETTINGS.style.buffer * 0.5
        if isinstance(buff, (tuple, list)):
            horizontal_buff, vertical_buff = buff
        else:
            horizontal_buff = vertical_buff = buff
        bboxes = torch.cat([mob.get_bounding_box() for mob in mobjects], -2)
        bbox = bboxes
        mn = bbox.amin(-2)
        mn[..., 0] -= horizontal_buff
        mn[..., 1] -= vertical_buff
        mx = bbox.amax(-2)
        mx[..., 0] += horizontal_buff
        mx[..., 1] += vertical_buff
        if bottom_buffer is not None:
            mn[..., 1:2] -= bottom_buffer
        md = (mn + mx) * 0.5

        corners = torch.stack(
            (
                torch.stack((mn[..., 0], mx[..., 1], md[..., 2]), -1),
                torch.stack((mx[..., 0], mx[..., 1], md[..., 2]), -1),
                torch.stack((mx[..., 0], mn[..., 1], md[..., 2]), -1),
                torch.stack((mn[..., 0], mn[..., 1], md[..., 2]), -1),
            ),
            -2,
        )
        self.corner_radius = corner_radius
        if color is not None:
            kwargs.setdefault("color", color)
        if corner_radius > 0:
            import manim as manim_ce

            width = float((mx[..., 0] - mn[..., 0]).reshape(-1)[0])
            height = float((mx[..., 1] - mn[..., 1]).reshape(-1)[0])
            rounded = manim_ce.RoundedRectangle(
                width=width,
                height=height,
                corner_radius=float(corner_radius),
            )
            control_points = torch.from_numpy(rounded.points).to(
                device=md.device,
                dtype=md.dtype,
            )
            control_points = control_points + md.reshape(-1, 3)[0] + INWARD * 0.01
            kwargs = _translate_vector_style_kwargs(
                kwargs, default_color=RED, shape=type(self)
            )
            BezierCircuitCubic.__init__(self, control_points, **kwargs)
        else:
            super().__init__(corners + INWARD * 0.01, **kwargs)


class Square(Rectangle):
    """A square, drawn as a cubic bezier circuit.

    Parameters
    ----------
    size
        Length of each side, in world units. Defaults to ``2``.
    **kwargs
        Passed to :class:`~.Rectangle` and on to :class:`~.BezierCircuitCubic` --
        notably ``color``, ``location``, ``stroke_color``, ``stroke_width`` and
        ``filled``.

    Examples
    --------
    .. algan:: Example1Square

        from algan import *

        square = Square(color=BLUE).spawn()
        square.rotate(45)

        Scene.save_video()
    """

    def __init__(self, size=2, **kwargs):
        super().__init__(width=size, height=size, **kwargs)


class Circle(BezierCircuitCubic):
    """A circle, drawn as a cubic bezier circuit.

    Parameters
    ----------
    radius
        Radius in world units. Defaults to ``1``; ``None`` is also treated as ``1``.
    color
        Fill color. Defaults to ``None``, meaning ``BLUE``.
    *args, **kwargs
        Passed to :class:`~.BezierCircuitCubic` -- notably ``location``,
        ``stroke_color``, ``stroke_width`` and ``filled``.
    """

    def __init__(self, radius=1, color=None, *args, **kwargs):
        if radius is None:
            radius = 1
        if color is not None:
            kwargs.setdefault("color", color)
        kwargs = _translate_vector_style_kwargs(
            kwargs, default_color=BLUE, shape=type(self)
        )
        a = 1.00005519
        b = 0.55342686
        c = 0.99873585
        control_points_quarter = torch.tensor([[0, a], [b, c], [c, b], [a, 0]])

        def rot90_in_2d(x):
            return torch.stack([x[..., 1], -x[..., 0]], -1)

        def rot_n_quarters(x, n):
            for _i in range(n):
                x = rot90_in_2d(x)
            return x

        control_points = torch.cat(
            [rot_n_quarters(control_points_quarter, i) for i in range(4)], -2
        )
        control_points = torch.cat(
            [control_points, torch.zeros_like(control_points[..., :1])], -1
        )
        mob_location = ORIGIN
        if "location" in kwargs:
            mob_location = kwargs["location"]
            del kwargs["location"]

        super().__init__(control_points, *args, **kwargs)
        self.scale(radius)
        self.move_to(mob_location)

    #: How far the unit circle's furthest control point lies from its centre.
    #: A circuit's ``scale_coefficient`` is that distance (the bezier handles
    #: sit outside the curve), not the radius the curve is drawn at.
    _EXTENT_PER_RADIUS = float(
        torch.tensor([[1.00005519, 0.0], [0.55342686, 0.99873585]]).norm(dim=-1).max()
    )

    @property
    def radius(self):
        """The radius the circle is drawn at, in world units, shape ``(*)``.

        Animation
        ---------
        Assignment resizes the circle about its centre, recorded as an
        animation over the current context's runtime (1 second by default);
        wrap it in ``Off()`` to apply it instantly. Reading is not animated.
        """
        return self.scale_coefficient[..., 0] / self._EXTENT_PER_RADIUS

    @radius.setter
    def radius(self, radius):
        radius = cast_to_tensor(radius)
        if radius.ndim > 0 and radius.shape[-1] == 1:
            radius = radius[..., 0]
        factor = radius / self.radius.clamp_min(1e-12)
        self.scale_coefficient = self.scale_coefficient * factor.unsqueeze(-1)

    def get_default_color(self):
        """Get the color this Mob uses when none was given.

        Returns
        -------
        :class:`~.Color`
            ``BLUE`` for a Circle.
        """
        return BLUE


class Dot(Circle):
    """A small filled :class:`Circle`, for marking a point.

    Parameters
    ----------
    location
        Where to put it, shape ``(*, 3)`` in world units. Defaults to ``ORIGIN``
        (the centre of the scene). The same name as :attr:`~.Mob.location`, which
        is what it sets.
    radius
        Radius in world units. Defaults to ``0.08`` -- small enough to read as a
        marker beside shapes of unit size.
    stroke_width
        Width of the outline, in world units. Defaults to ``0``: a dot is solid
        fill with no border.
    fill_opacity
        Opacity of the fill, from ``0`` (invisible) to ``1`` (opaque). Defaults to
        ``1.0``.
    color
        An Algan :class:`~algan.constants.color.Color`, a named constant such as
        ``BLUE``, or anything ``Color()`` accepts. Defaults to ``WHITE``.
    **kwargs
        Passed to :class:`Circle` and on to :class:`~.BezierCircuitCubic`.

    Examples
    --------
    Three dots marking the corners of a triangle:

    .. algan:: Example1Dot
        :save_last_frame:

        from algan import *

        for position in (UP, LEFT + DOWN, RIGHT + DOWN):
            Dot(location=position, radius=0.15, color=BLUE).spawn()

        Scene.save_video()
    """

    def __init__(
        self,
        location=ORIGIN,
        radius=0.08,
        stroke_width=0,
        fill_opacity=1.0,
        color=WHITE,
        **kwargs,
    ):
        _reject_renamed_keywords(
            "Dot", kwargs, {"point": "location"}, manim_alternative="Dot"
        )
        kwargs.setdefault("color", color)
        kwargs.setdefault("stroke_width", stroke_width)
        kwargs.setdefault("fill_opacity", fill_opacity)
        super().__init__(radius=radius, location=location, **kwargs)
