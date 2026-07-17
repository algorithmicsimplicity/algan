import math

import torch
import torch.nn.functional as F

from algan.animation.animation_contexts import Off, Sync
from algan.constants.spatial import ORIGIN, UP, RIGHT, LEFT, IN
from algan.constants.color import *
from algan.geometry.geometry import map_local_to_global_coords
from algan.mobs.bezier_circuit import BezierCircuitCubic
from algan.mobs.mob import Mob
from algan.mobs.renderable import Renderable
from algan.settings.style_defaults import STYLE_DEFAULTS
from algan.settings.renderer_settings import RENDERER_SETTINGS
from algan.utils.tensor_utils import (
    unsqueeze_left,
    broadcast_all,
    cast_to_tensor,
    unsquish,
)
from algan.utils.tensor_utils import mean


def _coerce_algan_color(value, opacity=None):
    """Return an Algan :class:`Color` from a Manim-style color value."""
    if isinstance(value, str):
        named = globals().get(value.upper())
        if isinstance(named, Color):
            value = named
        else:
            value = Color(value)
    value = cast_to_tensor(value)
    value = Color.add_defaults(value).as_subclass(Color)
    if opacity is not None:
        value = value.set_opacity(opacity)
    return value


def _translate_vector_style_kwargs(kwargs, *, default_color=None, line=False):
    """Translate common VMobject style keywords to BezierCircuitCubic.

    Algan stores fill color on ``color`` and outline style on
    ``border_color``/``border_width``.  Manim exposes the same concepts as
    ``fill_*`` and ``stroke_*``.  Consuming the keywords here also prevents
    renderer-only VMobject settings from leaking into ``Animatable``.
    """
    kwargs = dict(kwargs)
    has_color = "color" in kwargs
    color = kwargs.get("color", default_color)

    fill_color = kwargs.pop("fill_color", None)
    fill_opacity = kwargs.pop("fill_opacity", None)
    stroke_color = kwargs.pop("stroke_color", None)
    stroke_opacity = kwargs.pop("stroke_opacity", None)
    stroke_width = kwargs.pop("stroke_width", None)

    if line:
        # A Line is an unfilled path; Manim's generic ``color`` controls its
        # stroke, while fill settings are accepted but have no visible effect.
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
        kwargs["border_color"] = _coerce_algan_color(
            stroke_color,
            stroke_opacity,
        )
    elif stroke_opacity is not None:
        kwargs["border_color"] = _coerce_algan_color(
            WHITE if default_color is None else default_color,
            stroke_opacity,
        )
    if stroke_width is not None:
        # ManimMob performs the inverse conversion when importing VMobjects.
        kwargs["border_width"] = float(stroke_width) / 2

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
    """A straight or circular-arc line segment with Manim-compatible arguments."""

    def __init__(self, start=LEFT, end=RIGHT, buff=0, path_arc=0, *args, **kwargs):
        start_center = start.get_center() if isinstance(start, Mob) else cast_to_tensor(start)
        end_center = end.get_center() if isinstance(end, Mob) else cast_to_tensor(end)
        direction = end_center - start_center
        if isinstance(start, Mob):
            start_center = start.get_boundary_in_direction(direction)
        if isinstance(end, Mob):
            end_center = end.get_boundary_in_direction(-direction)
        direction = end_center - start_center
        length = direction.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-10)
        if buff:
            unit = direction / length
            effective_buff = min(float(buff), max(float(length.reshape(-1)[0]) * 0.5, 0.0))
            start_center = start_center + unit * effective_buff
            end_center = end_center - unit * effective_buff

        kwargs = _translate_vector_style_kwargs(kwargs, line=True)

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

    def put_start_and_end_on(self, start, end):
        target = Line(start, end, add_to_scene=False)
        return self.become(target, detach_history=False)


class Point(BezierCircuitCubic):
    def __init__(self, location=ORIGIN, *args, **kwargs):
        location = cast_to_tensor(location)
        kwargs = _translate_vector_style_kwargs(kwargs, default_color=BLACK)
        super().__init__(torch.cat([location for _ in range(4)], -2), *args, **kwargs)

    def get_num_points(self):
        return 1

    def get_points(self):
        return self.get_center().reshape(1, 3)

    def point_from_proportion(self, alpha):
        return self.get_center()


class TriangleTriangulated(Mob):
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
        k = self.location
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
            with Off(record_attr_modifications=False):
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
        with Off(record_attr_modifications=False):
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


class TriangleVertices(Renderable):
    def __init__(self, corner_locations, normals=None, **kwargs):
        corner_locations = cast_to_tensor(corner_locations)
        kwargs2 = {k: v for k, v in kwargs.items()}
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

    def get_memory_used_per_timestep(self):
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
        l, c, o, n, g, gr = broadcast_all(
            [
                self.location,
                self.color,
                self.opacity,# * self.max_opacity,
                self.normals,
                self.glow,
                self.glow_radius,
            ],
            ignored_dims=[-1],
        )
        if n is None:
            n = torch.zeros_like(l)
        return RENDERER_SETTINGS.triangle_primitive(
            l,
            c,
            o,
            F.normalize(
                map_local_to_global_coords(self.location, self.basis, n)
                - self.location,
                p=2,
                dim=-1,
            ),
            glow=g,
            glow_radius=gr,
            shader=self.shader,
            **self.get_shader_params(),
        )


class QuadTriangulated(Mob):
    def __init__(self, corner_locations, **kwargs):
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
        kwargs = _translate_vector_style_kwargs(kwargs, default_color=RED)
        if len(vertex_locations) == 1:
            corner_locations = cast_to_tensor(vertex_locations[0])
            while corner_locations.dim() > 2 and corner_locations.shape[0] == 1:
                corner_locations = corner_locations[0]
        else:
            corner_locations = torch.stack(
                [cast_to_tensor(vertex).reshape(-1, 3)[0] for vertex in vertex_locations],
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
    """A regular polygon with Manim-compatible ``n``/``start_angle`` arguments."""

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
            (radius * torch.cos(angles), radius * torch.sin(angles), torch.zeros_like(angles)),
            dim=-1,
        )
        self.n = n
        super().__init__(*vertices, **kwargs)


class Quad(Polygon):
    pass


class Triangle(RegularPolygon):
    def __init__(self, **kwargs):
        super().__init__(n=3, **kwargs)


class Rectangle(Quad):
    """A rectangle.

    Parameters
    ----------
    height
        Rectangle height.
    width
        Rectangle width.
    *args, **kwargs
        Passed to :class:`~.BezierCircuitCubic`

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
    """A rectangle.

    Parameters
    ----------
    height
        Rectangle height.
    width
        Rectangle width.
    *args, **kwargs
        Passed to :class:`~.BezierCircuitCubic`

    """

    def __init__(
        self,
        *mobjects,
        color=None,
        buff=STYLE_DEFAULTS.buffer * 0.5,
        corner_radius=0.0,
        buffer=None,
        bottom_buffer=None,
        **kwargs,
    ):
        if not mobjects:
            raise ValueError("SurroundingRectangle requires at least one Mobject")
        if buffer is not None:
            buff = buffer
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
            mn[...,1:2] -= bottom_buffer
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
            control_points = control_points + md.reshape(-1, 3)[0] + IN * 0.01
            kwargs = _translate_vector_style_kwargs(kwargs, default_color=RED)
            BezierCircuitCubic.__init__(self, control_points, **kwargs)
        else:
            super().__init__(corners + IN * 0.01, **kwargs)


class Square(Rectangle):
    """A square.

    Parameters
    ----------
    side_length
        Length of each side of the square.
    *args, **kwargs
        Passed to :class:`~.BezierCircuitCubic`

    """

    def __init__(self, side_length=2, **kwargs):
        super().__init__(width=side_length, height=side_length, **kwargs)


class Circle(BezierCircuitCubic):
    """A circle.

    Parameters
    ----------
    radius
        Circle radius.
    *args, **kwargs
        Passed to :class:`~.BezierCircuitCubic`

    """

    def __init__(self, radius=1, color=None, *args, **kwargs):
        if radius is None:
            radius = 1
        if color is not None:
            kwargs.setdefault("color", color)
        kwargs = _translate_vector_style_kwargs(kwargs, default_color=BLUE)
        a = 1.00005519
        b = 0.55342686
        c = 0.99873585
        control_points_quarter = torch.tensor([[0, a], [b, c], [c, b], [a, 0]])

        def rot90_in_2d(x):
            return torch.stack([x[..., 1], -x[..., 0]], -1)

        def rot_n_quarters(x, n):
            for i in range(n):
                x = rot90_in_2d(x)
            return x

        control_points = torch.cat(
            [rot_n_quarters(control_points_quarter, i) for i in range(4)], -2
        )
        control_points = torch.cat(
            [control_points, torch.zeros_like(control_points[..., :1])], -1
        )
        l = ORIGIN
        if "location" in kwargs:
            l = kwargs["location"]
            del kwargs["location"]

        super().__init__(control_points, *args, **kwargs)
        self.scale(radius)
        self.move_to(l)

    @property
    def radius(self):
        return self.scale_coefficient[..., 0]

    @radius.setter
    def radius(self, radius):
        self.scale_coefficient = radius

    def get_default_color(self):
        return BLUE


class Dot(Circle):
    """A small filled circle with Manim-compatible constructor arguments."""

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
