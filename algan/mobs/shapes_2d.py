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

import torch.nn.functional as F

from algan.animatable_base.mob import Mob
from algan.animation_timeline.animation_contexts import (
    Off,
    active_scene_for_new_mob,
)
from algan.constants.color import *
from algan.constants.spatial import INWARD, LEFT, ORIGIN, RIGHT
from algan.geometry.geometry import map_local_to_global_coords
from algan.mobs.bezier_circuit import BezierCircuitCubic
from algan.settings import SETTINGS
from algan.settings.renderer_settings import effective_triangle_primitive
from algan.settings.shape_style_profiles import _manim_shape_style_for
from algan.utils.tensor_utils import (
    broadcast_all,
    cast_to_tensor,
    mean,
    unsquish,
)


def _coerce_algan_color(value, opacity=None):
    """Return an Algan :class:`Color` from a Manim-style color value."""
    if isinstance(value, str):
        named = globals().get(value.upper())
        value = named if isinstance(named, Color) else Color(value)
    value = cast_to_tensor(value)
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
        "joint_type",
        "cap_style",
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
        Angle of the circular arc bulging the line away from straight, **in
        radians** -- a Manim-parity argument, which is why it contradicts Algan's
        usual degrees. Positive and negative values bulge opposite ways. Defaults
        to ``0`` (a straight segment).
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
        Line(LEFT, RIGHT, path_arc=1.0, color=YELLOW).spawn()

        Scene.save_video()
    """

    def __init__(self, start=LEFT, end=RIGHT, buff=0, path_arc=0, *args, **kwargs):
        start_center = (
            start.get_center() if isinstance(start, Mob) else cast_to_tensor(start)
        )
        end_center = end.get_center() if isinstance(end, Mob) else cast_to_tensor(end)
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
                angle=float(path_arc),
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

    def get_start(self):
        return unsquish(self.control_points.location, -2, 4)[..., 0, :]

    def get_end(self):
        return unsquish(self.control_points.location, -2, 4)[..., -1, :]

    def get_vector(self):
        return self.get_end() - self.get_start()

    def get_unit_vector(self):
        return F.normalize(self.get_vector(), p=2, dim=-1)

    def get_length(self):
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

    def put_start_and_end_on(self, start, end):
        target = Line(start, end, scene=self.scene, add_to_scene=False)
        return self.become(target, detach_history=False)


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
        return 1

    def get_points(self):
        return self.get_center().reshape(1, 3)

    def point_from_proportion(self, alpha):
        return self.get_center()


class TriangleTriangulated(Mob):
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
        # scl = squish(corner_locations, 0, -2)
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
        return YELLOW


class TriangleVertices(Mob):
    _morph_family = "mesh"

    def __init__(self, corner_locations, normals=None, **kwargs):
        corner_locations = cast_to_tensor(corner_locations)
        kwargs2 = dict(kwargs.items())
        if "location" in kwargs2:
            del kwargs2["location"]
        kwargs2["location"] = corner_locations.reshape(-1, 3)
        if "color" in kwargs2:
            kwargs2["color"] = kwargs2["color"].reshape(-1, kwargs2["color"].shape[-1])
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
        n = self.location.shape[-2]
        # Source/animation-device state only: location(3) + color(5) +
        # normal(3), plus the primitive's cloned color(5). Ray-tracing bounds,
        # packed geometry and BVHs are built on the source device and their
        # finished storages are charged exactly when uploaded to ManualMemory.
        num_vars = 16
        for _ in self.get_shader_params().values():
            num_vars += _.shape[-1]
        return n * num_vars * 4

    def get_default_color(self):
        return PURE_RED

    def get_render_primitives(self):
        locations, c, o, n, g = broadcast_all(
            [
                self.location,
                self.color,
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
                map_local_to_global_coords(self.location, self.basis, n)
                - self.location,
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


class Polygon(BezierCircuitCubic):
    """A 2-D planar polygon with N vertices.

    Parameters
    ----------
    vertex_locations : torch.Tensor[N, 3]
        3-D coordinates for each of the N vertex points.
    *args, **kwargs
        Passed to :class:`~.BezierCircuitCubic`

    """

    def __init__(self, *vertex_locations: torch.Tensor, **kwargs):
        kwargs = _translate_vector_style_kwargs(
            kwargs, default_color=RED, shape=type(self)
        )
        if not vertex_locations:
            # ``torch.stack([])`` below would answer "stack expects a
            # non-empty TensorList", which says nothing about polygons.
            raise ValueError(
                f"{type(self).__name__} needs its vertices: pass them as "
                f"points ({type(self).__name__}(LEFT, RIGHT, UP)) or as one "
                f"[N, 3] tensor."
            )
        if len(vertex_locations) == 1:
            corner_locations = cast_to_tensor(vertex_locations[0])
            while corner_locations.dim() > 2 and corner_locations.shape[0] == 1:
                corner_locations = corner_locations[0]
        else:
            corner_locations = torch.stack(
                [
                    cast_to_tensor(vertex).reshape(-1, 3)[0]
                    for vertex in vertex_locations
                ],
                dim=0,
            )
        if corner_locations.shape[-2] < 3:
            raise ValueError("Polygon requires at least three vertices")
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
        return RED


class RegularPolygon(Polygon):
    """A regular polygon: ``n`` equal sides on a circle.

    Parameters
    ----------
    n
        Number of sides. Defaults to ``6``; must be at least 3.
    num_vertices
        Alias of ``n``, for Manim compatibility. When given it overrides ``n``.
        Defaults to ``None``.
    radius
        Distance from the center to each vertex, in world units. Defaults to ``1``.
    start_angle
        Angle of the first vertex, **in radians** (Manim's convention). Defaults to
        ``None``, which puts the first vertex at the top and repeats the closing
        vertex -- a topology that matters when morphing with
        :meth:`~algan.animatable_base.mob_morph.MobMorphMixin.become`.
    **kwargs
        Passed to :class:`~.Polygon` and on to :class:`~.BezierCircuitCubic`.

    Raises
    ------
    ValueError
        If fewer than 3 sides are requested.
    """

    def __init__(
        self,
        n: int = 6,
        *,
        num_vertices: int | None = None,
        radius: float = 1,
        start_angle: float | None = None,
        **kwargs,
    ):
        if num_vertices is not None:
            n = num_vertices
        if n < 3:
            raise ValueError("RegularPolygon requires n >= 3")
        if start_angle is None:
            # Preserve Algan's original topology and orientation: the first
            # vertex is at the top and the closing vertex is repeated.  The
            # latter is observable during ``become`` because it determines
            # how cubic segments are paired.
            angles = torch.linspace(math.pi / 2, -math.pi * 1.5, n + 1)
            self.start_angle = math.pi / 2
        else:
            angles = start_angle + torch.arange(n) * (2 * math.pi / n)
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
            corners = corners + cast_to_tensor(kwargs["location"])
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
            raise ValueError("SurroundingRectangle requires at least one Mobject")
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

    @property
    def radius(self):
        return self.scale_coefficient[..., 0]

    @radius.setter
    def radius(self, radius):
        self.scale_coefficient = radius

    def get_default_color(self):
        return BLUE


class Dot(Circle):
    """A small filled :class:`Circle`, for marking a point.

    Parameters
    ----------
    point
        Where to put it, shape ``(*, 3)`` in world units. Defaults to ``ORIGIN``.
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
            Dot(point=position, radius=0.15, color=BLUE).spawn()

        Scene.save_video()
    """

    def __init__(
        self,
        point=ORIGIN,
        radius=0.08,
        stroke_width=0,
        fill_opacity=1.0,
        color=WHITE,
        **kwargs,
    ):
        kwargs.setdefault("color", color)
        kwargs.setdefault("stroke_width", stroke_width)
        kwargs.setdefault("fill_opacity", fill_opacity)
        super().__init__(radius=radius, location=point, **kwargs)
