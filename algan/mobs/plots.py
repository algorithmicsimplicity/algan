"""Plotting and mathematical graph mobs for Algan.

Provides 2D coordinate axes, function curve plotting, directional arrows with ticks,
and bar graphs.
"""

from __future__ import annotations

import torch.nn.functional as F
from svgelements import Close, Line, Move, Path

from algan.animatable_base.animatable import animated_function
from algan.animatable_base.mob import Mob
from algan.animation_timeline.animation_contexts import (
    Off,
    Sync,
    active_scene_for_new_mob,
)
from algan.constants.color import *
from algan.constants.spatial import DOWN, LEFT, ORIGIN, OUTWARD, RIGHT, UP
from algan.mobs.group import Group
from algan.mobs.shapes_2d import Quad, Rectangle, TriangleTriangulated
from algan.mobs.triangulated_bezier_circuit import TriangulatedBezierCircuit
from algan.utils.tensor_utils import (
    broadcast_cross_product,
    interpolate,
    mean,
    squish,
    unsquish,
)


def convert_points_to_path(points):
    path = [Move(points[0])]
    for i in range(len(points) - 1):
        path.append(Line(points[i], points[i + 1]))
    path.append(Close(points[-1], points[0]))
    return Path(*path)


class Arrow(TriangulatedBezierCircuit):
    """Directional arrow with optional tick marks along its shaft.

    Constructs a 2D triangulated arrow pointing from `start` to `end`.

    Parameters
    ----------
    start : torch.Tensor | list[float]
        Starting coordinate vector `[x, y, z]`.
    end : torch.Tensor | list[float]
        Ending coordinate vector `[x, y, z]` where the arrow head points.
    facing_direction : torch.Tensor, default=OUTWARD
        Normal orientation vector perpendicular to the arrow plane.
    width : float, default=0.009
        Width of the arrow shaft.
    bidirectional : bool, default=False
        Whether to add arrow heads at both ends.
    num_ticks : int, default=4
        Number of tick subdivisions along the shaft.
    **kwargs
        Additional keyword arguments forwarded to :class:`TriangulatedBezierCircuit`.
    """

    def __init__(
        self,
        start,
        end,
        facing_direction=OUTWARD,
        width=0.009,
        bidirectional=False,
        num_ticks=4,
        **kwargs,
    ):
        direction = F.normalize(end - start, p=2, dim=-1)
        perp = F.normalize(
            broadcast_cross_product(direction, facing_direction), p=2, dim=-1
        )
        k = 2
        tick_height = width * 2
        tick_a = torch.linspace(0, 1, 2 * num_ticks + 3)[1:-1]

        def get_tick_at(x):
            return [
                x - width * 0.25 * direction,
                x - width * 0.25 * direction + tick_height * perp,
                x + width * 0.25 * direction + tick_height * perp,
                x + width * 0.25 * direction,
            ]

        tick_points = [
            interpolate(start + perp * width * 0.5, end + perp * width * 0.5, _)
            for _ in tick_a
        ]
        tick_points = [
            x
            for tick_segment in [get_tick_at(_) for _ in tick_points]
            for x in tick_segment
        ]
        points = torch.stack(
            [
                start + perp * width * 0.5,
                *tick_points,
                end + perp * width * 0.5,
                end + perp * width * k,
                end + direction * width * k * 1.5,
                end - perp * width * k,
                end - perp * width * 0.5,
                start - perp * width * 0.5,
                start - perp * width * k,
                start - direction * width * k * 1.5,
                start + perp * width * k,
            ]
        )[..., :2]
        path = convert_points_to_path(points)

        super().__init__([path], **kwargs)


class AxesMob(Mob):
    """2D coordinate axis system with horizontal and vertical axes.

    Parameters
    ----------
    width : float, default=1.0
        Span width of the coordinate axes.
    quadrant : bool, default=False
        If True, renders axes starting from the origin (first quadrant only).
        If False, centers axes symmetrically around the origin.
    **kwargs
        Additional keyword arguments forwarded to child :class:`Arrow` mobs.
    """

    def __init__(self, width=1.0, quadrant=False, **kwargs):
        super().__init__(**kwargs)
        kwargs["scene"] = self.scene
        self.width = width
        self.horizontal_axis = Arrow(
            LEFT * width * 0.5 if not quadrant else ORIGIN,
            RIGHT * width * 0.5,
            **kwargs,
        ).scale(2)
        self.vertical_axis = Arrow(
            DOWN * width * 0.5 if not quadrant else ORIGIN, UP * width * 0.5, **kwargs
        ).scale(2)
        self.width = width * 2
        self.add_children(self.horizontal_axis, self.vertical_axis)


def get_corners(start, direction, width, height, facing_direction):
    perp = broadcast_cross_product(direction, facing_direction)
    end = start + direction * height
    return torch.stack(
        [
            start + perp * width,
            end + perp * width,
            end - perp * width,
            start - perp * width,
        ],
        -2,
    )


class Bar(Quad):
    """Rectangular 2D bar mob for discrete plots and histograms.

    Parameters
    ----------
    start : torch.Tensor | list[float]
        Base center point of the bar.
    end : torch.Tensor | list[float]
        Top center point of the bar.
    width : float, default=0.05
        Half-width of the bar quad.
    facing_direction : torch.Tensor, default=OUTWARD
        Normal vector for orientation.
    **kwargs
        Additional keyword arguments forwarded to :class:`Quad`.
    """

    def __init__(self, start, end, width=0.05, facing_direction=OUTWARD, **kwargs):
        self.direction = F.normalize(end - start, p=2, dim=-1)
        self.height = (end - start).norm(p=2, dim=-1)
        self.width = width
        self.facing_direction = facing_direction
        super().__init__(
            get_corners(start, self.direction, width, self.height, facing_direction),
            **kwargs,
        )
        with Off(animation_manager=self.animation_manager):
            self.set_non_recursive(location=start)

    @animated_function(animated_args={"interpolation": 0}, unique_args=["height_func"])
    def move_to_with_height_matching(
        self, location, height_func, original_loc, interpolation=1
    ):
        """Smoothly moves the bar to `location` while adjusting height dynamically via `height_func`."""
        self.location = original_loc * (1 - interpolation) + interpolation * location
        loc1 = unsquish(self.triangles[0].corners.location, -2, 3).clone()
        loc2 = unsquish(self.triangles[1].corners.location, -2, 3).clone()
        ray1 = F.normalize(loc1[..., 1, :] - loc1[..., 0, :], p=2, dim=-1)
        ray2 = F.normalize(loc2[..., 0, :] - loc2[..., 1, :], p=2, dim=-1)
        h = height_func(self.location[..., 0])[..., 1].unsqueeze(-1)
        loc1[..., 1, :] = loc1[..., 0, :] + ray1 * h
        loc1[..., 2, :] = loc2[..., 1, :] + ray2 * h
        loc2[..., 0, :] = loc2[..., 1, :] + ray2 * h
        self.triangles[0].corners.location = squish(loc1, -3, -2)
        self.triangles[1].corners.location = squish(loc2, -3, -2)


class FunctionPlotMob(Mob):
    """Plot of a mathematical function `y = f(x)` on a coordinate axis system.

    Parameters
    ----------
    func : callable
        Mathematical function taking an `x` tensor and returning `y`.
    axes : AxesMob, optional
        Pre-existing :class:`AxesMob` coordinate system to plot upon. If None, a new axes system is created.
    width : float, default=0.02
        Stroke width of the plotted curve.
    func_color : Color, default=RED_A
        Colour of the plotted function curve or bars.
    num_points : int, default=200
        Number of sample points evaluated along the domain.
    offset : float, default=1.0
        Depth offset along OUTWARD axis to prevent z-fighting with axes.
    scale : float, default=1.0
        Domain scaling factor.
    max_value : float, optional
        Maximum range value for vertical scaling.
    bar_plot : bool, default=False
        If True, renders discrete bar rectangles instead of a continuous curve.
    **kwargs
        Additional keyword arguments forwarded to :class:`Mob`.
    """

    def __init__(
        self,
        func,
        axes=None,
        width=0.02,
        func_color=RED_A,
        num_points=200,
        offset=1,
        scale=1,
        max_value=None,
        bar_plot=False,
        **kwargs,
    ):
        create = kwargs.get("create", True)
        init = kwargs.get("init", True)
        kwargs["create"] = False
        kwargs["init"] = False
        super().__init__(**kwargs)
        kwargs["scene"] = self.scene
        new_axes = axes is None
        if axes is None:
            axes = AxesMob(**kwargs)
            axes.max_value = max_value

        self.axes = axes
        self.s = scale * 2

        self.func_callable = func

        xs = (
            torch.linspace(
                -axes.width * 0.5, axes.width * 0.5, num_points + (1 - (num_points % 2))
            )
            + 1e-3
        )
        func_points = self.map_input_domain_to_curve_location(xs)
        func_points = func_points[~func_points[..., 1].isnan()]
        func_points = func_points[func_points[..., 1].abs() <= self.get_scaler()]

        points = func_points
        perps = [
            F.normalize(
                broadcast_cross_product(
                    (points[i + 1] if i < len(points) - 1 else points[i])
                    - (points[i - 1] if i > 0 else points[i]),
                    OUTWARD,
                ),
                p=2,
                dim=-1,
            )
            for i in range(len(points))
        ]
        func_points = torch.stack(
            [
                *[points[i] + perps[i] * width * 0.5 for i in range(len(points))],
                *reversed(
                    [points[i] - perps[i] * width * 0.5 for i in range(len(points))]
                ),
            ]
        )[..., :2]

        kwargs["constants"] = func_color
        with Off(animation_manager=self.animation_manager):
            if not bar_plot:
                self.func = TriangulatedBezierCircuit(
                    [convert_points_to_path(func_points)], **kwargs
                ).move(OUTWARD * 0.001 * offset)
            else:
                x = self.map_input_domain_to_scaled_domain(xs)
                locs = (func_points + x) * 0.5
                heights = func_points - x
                widths = (x[..., 1] - x[..., 0]) * 0.5
                self.func = Group(
                    [
                        Rectangle(h, w, scene=self.scene).move_to(location)
                        for h, w, location in zip(heights, widths, locs)
                    ],
                    scene=self.scene,
                )
        self.add_children(self.func)
        if new_axes:
            self.add_children(axes)
        if init:
            self.init()
        if create:
            self.spawn()

    def map_input_domain_to_scaled_domain(self, xs):
        def get_func_point_at_x(x):
            return torch.stack((x, torch.zeros_like(x), torch.zeros_like(x)), -1)

        func_points = get_func_point_at_x(xs)
        return func_points

    def get_scaler(self):
        return self.axes.width * 0.5 * 1.1

    def map_input_domain_to_curve_location(self, xs):
        def get_func_point_at_x(x):
            return torch.stack(
                (x, self.func_callable(x * self.s) / self.s, torch.zeros_like(x)), -1
            )

        func_points = get_func_point_at_x(xs)
        if self.axes.max_value is None:
            max_value = (
                func_points[..., 1]
                .nan_to_num(0)
                .abs()
                .amax(keepdim=True)
                .clamp_min_(1e-6)
                .item()
            )
            self.axes.max_value = max_value
        max_value = self.axes.max_value
        func_points[..., 1] = (func_points[..., 1] / max_value) * self.get_scaler()
        return func_points

    def on_create(self):
        self.spawn_tilewise_recursive()

    def on_destroy(self):
        self.despawn_tilewise_recursive()
