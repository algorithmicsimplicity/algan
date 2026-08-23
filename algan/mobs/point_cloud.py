"""Manim-compatible point-cloud Mobs.

Algan's ray tracer has triangle and cubic-Bezier primitives rather than a
point-sprite primitive. Visible point clouds are therefore represented by one
batched collection of small native spheres while retaining Manim's point-array
mutation and query API.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import torch

from algan.animation_timeline.animation_contexts import active_scene_for_new_mob
from algan.constants.color import BLACK, WHITE, YELLOW, Color
from algan.constants.spatial import ORIGIN
from algan.mobs.group import Group
from algan.mobs.shapes_2d import Point
from algan.mobs.shapes_3d import Dot3D
from algan.utils.tensor_utils import cast_to_tensor


def _as_points(points) -> torch.Tensor:
    if points is None:
        return torch.empty(
            (0, 3), dtype=torch.get_default_dtype(), device=torch.get_default_device()
        )
    if isinstance(points, torch.Tensor):
        value = points
    elif isinstance(points, (list, tuple)) and any(
        isinstance(item, torch.Tensor) for item in points
    ):
        value = torch.stack([cast_to_tensor(item).reshape(3) for item in points])
    else:
        value = torch.as_tensor(np.asarray(points), device=torch.get_default_device())
    return value.to(dtype=torch.get_default_dtype()).reshape(-1, 3)


def _as_rgba(color, alpha=1.0) -> torch.Tensor:
    if color is None:
        color = YELLOW
    if isinstance(color, Color):
        rgb = color.rgb.detach().to(dtype=torch.get_default_dtype())
        opacity = color.opacity.detach().reshape(-1)[0] * alpha
    else:
        value = cast_to_tensor(color).to(dtype=torch.get_default_dtype()).reshape(-1)
        rgb = value[:3]
        opacity = (
            value[3] if value.numel() == 4 else value[-1] if value.numel() >= 5 else 1.0
        ) * alpha
    return torch.cat(
        (rgb[:3], torch.as_tensor([opacity], device=rgb.device, dtype=rgb.dtype))
    )


def _rgba_to_color(rgba: torch.Tensor) -> Color:
    rgba = rgba.detach().reshape(-1)
    return Color(tuple(float(x) for x in rgba[:3]), opacity=float(rgba[3]))


def _gradient(colors, length: int) -> torch.Tensor:
    if length <= 0:
        return torch.empty(
            (0, 4), dtype=torch.get_default_dtype(), device=torch.get_default_device()
        )
    stops = torch.stack([_as_rgba(color) for color in colors])
    if len(stops) == 1:
        return stops.expand(length, -1).clone()
    position = torch.linspace(0, len(stops) - 1, length, device=stops.device)
    lower = position.floor().long().clamp(max=len(stops) - 2)
    t = (position - lower).unsqueeze(-1)
    return stops[lower] * (1 - t) + stops[lower + 1] * t


class PMobject(Group):
    """Point-array Mobject rendered as a batched set of tiny spheres."""

    #: ``get_render_primitives`` below builds every point's sphere itself and
    #: none of them is a Scene actor, so the cloud is one unit to ``become``
    #: too: converted through the "aggregate" adapter and with its spheres kept
    #: out of the actor list rather than published and drawn a second time.
    _morph_family = "aggregate"
    draws_descendants = True

    def morph_soup_parts(self):
        return self._primitive_children()

    def __init__(
        self,
        stroke_width: int = 4,
        *,
        points=None,
        rgbas=None,
        color=YELLOW,
        point_radius=None,
        **kwargs,
    ):
        if point_radius is None and "radius" in kwargs:
            # Algan historically used ``radius`` for the rendered point size;
            # retain it as an extension while exposing Manim's stroke-width API.
            point_radius = kwargs.pop("radius")
        self.stroke_width = stroke_width
        self.point_radius = float(
            stroke_width * 0.01 if point_radius is None else point_radius
        )
        self.points = _as_points(points)
        if rgbas is None:
            rgba = _as_rgba(color)
            self.rgbas = rgba.expand(len(self.points), -1).clone()
        else:
            self.rgbas = (
                cast_to_tensor(rgbas).to(dtype=torch.get_default_dtype()).reshape(-1, 4)
            )
            if len(self.rgbas) != len(self.points):
                raise ValueError("points and rgbas must have same length")
        self.point_color = color
        if kwargs.get("scene") is None:
            kwargs["scene"] = active_scene_for_new_mob()
        geometry = self._build_geometry(scene=kwargs["scene"])
        super().__init__(*([] if geometry is None else [geometry]), **kwargs)
        # Animatable installs instance-level generic color accessors while
        # registering Mob attributes. Point clouds need color accessors that
        # update their per-point RGBA array and rebuilt geometry.
        self.set_color = PMobject.set_color.__get__(self, type(self))
        self.get_color = PMobject.get_color.__get__(self, type(self))

    def _build_geometry(self, scene=None):
        if scene is None and hasattr(self, "scene"):
            scene = self.scene
        if len(self.points) == 0:
            return None
        if len(self.points) == 1:
            return Dot3D(
                point=self.points[0],
                radius=self.point_radius,
                resolution=None,
                color=_rgba_to_color(self.rgbas[0]),
                add_to_scene=False,
                scene=scene,
            )
        # One packed Mob rather than one Dot3D per point: a cloud of N points
        # costs one construction and one primitive build instead of N of each.
        return Dot3D.from_batches(
            self.points,
            radius=self.point_radius,
            resolution=None,
            colors=self.rgbas,
            add_to_scene=False,
            scene=scene,
        )

    def _rebuild_geometry(self):
        geometry = self._build_geometry()
        self.replace_children([] if geometry is None else [geometry])
        if geometry is not None and self.is_spawned() and not self.is_despawned():
            geometry._create_recursive(animate=False)
        return self

    def _primitive_children(self):
        """Return hidden geometry not already rendered as its own Scene actor."""
        registered = {id(actor) for actor in self.scene.actors}
        return [
            child
            for child in self.children
            if id(child) not in registered and hasattr(child, "get_render_primitives")
        ]

    def _get_memory_used_per_timestep(self):
        return sum(
            child._get_memory_used_per_timestep()
            for child in self._primitive_children()
        )

    def get_render_primitives(self):
        """Build the native sphere primitives representing this point cloud."""
        primitives = []
        for child in self._primitive_children():
            primitive = child.get_render_primitives()
            if primitive is None:
                continue
            primitives.extend(primitive if isinstance(primitive, list) else [primitive])
        return primitives or None

    def reset_points(self):
        self.points = self.points.new_empty((0, 3))
        self.rgbas = self.rgbas.new_empty((0, 4))
        return self._rebuild_geometry()

    def get_array_attrs(self):
        return ["points", "rgbas"]

    def get_num_points(self):
        return len(self.points)

    def get_points(self):
        return self.points.clone()

    def set_points(self, points):
        points = _as_points(points)
        if len(self.rgbas) == 0:
            self.rgbas = _as_rgba(self.point_color).expand(len(points), -1).clone()
        elif len(points) != len(self.rgbas):
            self.rgbas = self.rgbas[:1].expand(len(points), -1).clone()
        self.points = points
        return self._rebuild_geometry()

    def add_points(self, points, rgbas=None, color=None, alpha=1, opacity=None):
        points = _as_points(points)
        if opacity is not None:
            alpha = opacity
        if rgbas is None:
            rgba = _as_rgba(self.point_color if color is None else color, alpha)
            rgbas = rgba.expand(len(points), -1).clone()
        else:
            rgbas = (
                cast_to_tensor(rgbas).to(dtype=torch.get_default_dtype()).reshape(-1, 4)
            )
            if len(rgbas) != len(points):
                raise ValueError("points and rgbas must have same length")
        self.points = torch.cat((self.points, points), 0)
        self.rgbas = torch.cat((self.rgbas, rgbas.to(self.rgbas.device)), 0)
        return self._rebuild_geometry()

    def get_all_rgbas(self):
        return self.rgbas.clone()

    def set_color(self, color=YELLOW, family=True):
        self.point_color = color
        if len(self.rgbas):
            self.rgbas[:] = _as_rgba(color).to(self.rgbas.device)
        if family:
            for mob in self.mobs:
                if isinstance(mob, PMobject):
                    mob.set_color(color, family=True)
        return self._rebuild_geometry()

    def get_color(self):
        return (
            self.point_color if len(self.rgbas) == 0 else _rgba_to_color(self.rgbas[0])
        )

    def get_stroke_width(self):
        return self.stroke_width

    def set_stroke_width(self, width, family=True):
        old_radius = self.point_radius
        self.stroke_width = width
        self.point_radius = float(width) * 0.01
        if old_radius != self.point_radius:
            self._rebuild_geometry()
        if family:
            for mob in self.mobs:
                if isinstance(mob, PMobject):
                    mob.set_stroke_width(width, family=True)
        return self

    def set_color_by_gradient(self, *colors):
        self.rgbas = _gradient(colors, len(self.points)).to(self.points.device)
        return self._rebuild_geometry()

    def set_colors_by_radial_gradient(
        self,
        center=None,
        radius=1,
        inner_color=WHITE,
        outer_color=BLACK,
    ):
        if center is None:
            center = self.points.mean(0) if len(self.points) else _as_points(ORIGIN)[0]
        center = _as_points(center)[0]
        alpha = ((self.points - center).norm(dim=-1) / radius).clamp(0, 1).unsqueeze(-1)
        start, end = _as_rgba(inner_color), _as_rgba(outer_color)
        self.rgbas = start * (1 - alpha) + end * alpha
        return self._rebuild_geometry()

    def match_colors(self, mobject):
        if len(self.points) == 0:
            self.rgbas = self.rgbas.new_empty((0, 4))
        elif len(mobject.rgbas) == 0:
            self.rgbas = (
                _as_rgba(mobject.point_color).expand(len(self.points), -1).clone()
            )
        else:
            indices = (
                torch.linspace(
                    0,
                    len(mobject.rgbas) - 1,
                    len(self.points),
                    device=mobject.rgbas.device,
                )
                .round()
                .long()
            )
            self.rgbas = mobject.rgbas[indices].clone()
        return self._rebuild_geometry()

    def filter_out(self, condition: Callable):
        keep = torch.tensor(
            [
                not bool(condition(point.detach().cpu().numpy()))
                for point in self.points
            ],
            dtype=torch.bool,
            device=self.points.device,
        )
        self.points = self.points[keep]
        self.rgbas = self.rgbas[keep]
        return self._rebuild_geometry()

    def thin_out(self, factor=5):
        indices = torch.arange(0, len(self.points), factor, device=self.points.device)
        self.points = self.points[indices]
        self.rgbas = self.rgbas[indices]
        return self._rebuild_geometry()

    def sort_points(self, function=lambda point: point[0]):
        order = sorted(
            range(len(self.points)),
            key=lambda i: function(self.points[i].detach().cpu().numpy()),
        )
        order = torch.as_tensor(order, dtype=torch.long, device=self.points.device)
        self.points = self.points[order]
        self.rgbas = self.rgbas[order]
        return self._rebuild_geometry()

    def fade_to(self, color, alpha, family=True):
        target = _as_rgba(color).to(self.rgbas.device)
        self.rgbas = self.rgbas * (1 - alpha) + target * alpha
        return self._rebuild_geometry()

    def ingest_submobjects(self):
        point_arrays = [self.points]
        rgba_arrays = [self.rgbas]
        for mob in self.mobs:
            if isinstance(mob, PMobject):
                point_arrays.append(mob.points)
                rgba_arrays.append(mob.rgbas)
        self.points = torch.cat(point_arrays, 0)
        self.rgbas = torch.cat(rgba_arrays, 0)
        return self._rebuild_geometry()

    def point_from_proportion(self, alpha):
        if len(self.points) == 0:
            raise IndexError("point cloud is empty")
        index = int(alpha * (len(self.points) - 1))
        return self.points[index]

    def align_points_with_larger(self, larger_mobject):
        if len(self.points) == 0:
            self.points = torch.zeros_like(larger_mobject.points)
            self.rgbas = _as_rgba(self.point_color).expand(len(self.points), -1).clone()
        else:
            indices = (
                torch.linspace(
                    0,
                    len(self.points) - 1,
                    len(larger_mobject.points),
                    device=self.points.device,
                )
                .round()
                .long()
            )
            self.points = self.points[indices]
            self.rgbas = self.rgbas[indices]
        self._rebuild_geometry()

    def get_point_mobject(self, center=None):
        if center is None:
            center = self.points.mean(0) if len(self.points) else ORIGIN
        return Point(center, scene=self.scene, add_to_scene=False)

    def interpolate_color(self, mobject1, mobject2, alpha):
        self.rgbas = mobject1.rgbas * (1 - alpha) + mobject2.rgbas * alpha
        self.stroke_width = (
            mobject1.get_stroke_width() * (1 - alpha)
            + mobject2.get_stroke_width() * alpha
        )
        self.point_radius = float(self.stroke_width) * 0.01
        return self._rebuild_geometry()

    def pointwise_become_partial(self, mobject, a, b):
        lower = int(a * len(mobject.points))
        upper = int(b * len(mobject.points))
        self.points = mobject.points[lower:upper].clone()
        self.rgbas = mobject.rgbas[lower:upper].clone()
        return self._rebuild_geometry()

    @staticmethod
    def get_mobject_type_class():
        return PMobject


class Mobject1D(PMobject):
    def __init__(self, density: int = 10, **kwargs):
        self.density = density
        self.epsilon = 1.0 / density
        super().__init__(**kwargs)

    def add_line(self, start, end, color=None):
        start, end = _as_points([start, end])
        length = float((end - start).norm())
        if length == 0:
            points = start.unsqueeze(0)
        else:
            count = max(1, int(math.ceil(length / self.epsilon)))
            t = torch.arange(count, device=start.device, dtype=start.dtype) / count
            points = start + (end - start) * t.unsqueeze(-1)
        self.add_points(points, color=color)


class Mobject2D(PMobject):
    def __init__(self, density: int = 25, **kwargs):
        self.density = density
        self.epsilon = 1.0 / density
        super().__init__(**kwargs)


class PGroup(PMobject):
    def __init__(self, *pmobs, **kwargs):
        if not all(isinstance(mob, PMobject) for mob in pmobs):
            raise ValueError("All submobjects must be of type PMobject")
        super().__init__(points=None, **kwargs)
        self.replace_children(pmobs)

    def fade_to(self, color, alpha, family=True):
        if family:
            for mob in self.mobs:
                mob.fade_to(color, alpha, family=True)
        return self


def _disc_points(radius, density, center=ORIGIN):
    epsilon = 1.0 / density
    result = []
    for r in np.arange(epsilon, radius, epsilon):
        count = int(2 * np.pi * (r + epsilon) / epsilon)
        for theta in np.linspace(0, 2 * np.pi, num=count):
            result.append((r * np.cos(theta), r * np.sin(theta), 0.0))
    points = _as_points(result)
    if len(points):
        points = points + _as_points(center)[0]
    return points


class PointCloudDot(Mobject1D):
    def __init__(
        self,
        center=ORIGIN,
        radius: float = 2.0,
        stroke_width: int = 2,
        density: int = 10,
        color=YELLOW,
        **kwargs,
    ):
        self.radius = radius
        points = _disc_points(radius, density, center)
        super().__init__(
            density=density,
            points=points,
            stroke_width=stroke_width,
            color=color,
            **kwargs,
        )

    def generate_points(self):
        return self.set_points(_disc_points(self.radius, self.density))

    init_points = generate_points


class OpenGLPMobject(PMobject):
    OPENGL_POINT_RADIUS_SCALE_FACTOR = 0.01

    def __init__(self, stroke_width=2.0, color=YELLOW, render_primitive=1, **kwargs):
        self.render_primitive = render_primitive
        super().__init__(stroke_width=stroke_width, color=color, **kwargs)
        self.point_radius = self.stroke_width * self.OPENGL_POINT_RADIUS_SCALE_FACTOR


class OpenGLPGroup(PGroup, OpenGLPMobject):
    pass


class OpenGLPMPoint(OpenGLPMobject):
    def __init__(self, location=ORIGIN, stroke_width=4.0, **kwargs):
        self.location_value = location
        super().__init__(points=[location], stroke_width=stroke_width, **kwargs)


class DotCloud(OpenGLPMobject):
    def __init__(
        self,
        color=YELLOW,
        stroke_width=2.0,
        radius=2.0,
        density=10,
        **kwargs,
    ):
        self.radius = radius
        self.density = density
        self.epsilon = 1.0 / density
        points = kwargs.pop("points", None)
        if points is None:
            points = _disc_points(radius, density)
        super().__init__(
            points=points,
            stroke_width=stroke_width,
            color=color,
            **kwargs,
        )

    def make_3d(self, gloss=0.5, shadow=0.2):
        self.gloss = gloss
        self.shadow = shadow
        return self

    def to_grid(self, n_rows, n_cols, n_layers=1, buff_ratio=0.5, height=6):
        spacing = self.point_radius * 2 * (1 + buff_ratio)
        coords = [
            (
                (j - (n_cols - 1) / 2) * spacing,
                ((n_rows - 1) / 2 - i) * spacing,
                (k - (n_layers - 1) / 2) * spacing,
            )
            for k in range(n_layers)
            for i in range(n_rows)
            for j in range(n_cols)
        ]
        self.set_points(coords)
        if height is not None and len(self.points):
            extent = float(self.points[:, 1].max() - self.points[:, 1].min())
            if extent > 0:
                self.points *= height / extent
                self._rebuild_geometry()
        return self


class TrueDot(DotCloud):
    def __init__(self, center=ORIGIN, stroke_width=2.0, **kwargs):
        super().__init__(points=[center], stroke_width=stroke_width, **kwargs)


__all__ = [
    "PMobject",
    "Mobject1D",
    "Mobject2D",
    "PGroup",
    "PointCloudDot",
    "DotCloud",
    "TrueDot",
    "OpenGLPMobject",
    "OpenGLPMPoint",
    "OpenGLPGroup",
]
