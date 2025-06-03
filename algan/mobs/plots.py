from svgelements import Path, Move, Close, Line

import torch
from algan.animation.animatable import animated_function
from algan.animation.animation_contexts import Off, Sync
from algan.constants.spatial import OUT, LEFT, RIGHT, DOWN, UP, ORIGIN
from algan.constants.color import *
from algan.mobs.triangulated_bezier_circuit import TriangulatedBezierCircuit
from algan.mobs.group import Group
from algan.mobs.mob import Mob
from algan.mobs.renderable import Renderable
from algan.mobs.shapes_2d import Quad, TriangleTriangulated, Rectangle
from algan.rendering.primitives.triangle import TrianglePrimitive
from algan.utils.tensor_utils import squish, broadcast_all
from algan.utils.tensor_utils import mean, broadcast_cross_product, interpolate, unsquish
import torch.nn.functional as F


class Line2(Quad):
    def __init__(self, start, end, facing_direction=OUT, width=0.05, **kwargs):
        direction = F.normalize(end-start, p=2, dim=-1)
        perp = F.normalize(broadcast_cross_product(direction, facing_direction), p=2, dim=-1)
        super().__init__(torch.stack([start+perp*width*0.5, end+perp*width*0.5, end-perp*width*0.5, start-perp*width*0.5], -2), **kwargs)


def convert_points_to_path(points):
    path = [Move(points[0])]
    for i in range(len(points) - 1):
        path.append(Line(points[i], points[i + 1]))
    path.append(Close(points[-1], points[0]))
    return Path(*path)


class Arrow(TriangulatedBezierCircuit):
    def __init__(self, start, end, facing_direction=OUT, width=0.009, bidirectional=False, num_ticks=4, **kwargs):
        #line = Line(start, end, facing_direction=OUT, width=0.05, **kwargs)
        #head = TriangleTriangulated(end )
        direction = F.normalize(end - start, p=2, dim=-1)
        perp = F.normalize(broadcast_cross_product(direction, facing_direction), p=2, dim=-1)
        k = 2
        tick_height = width*2
        tick_a = torch.linspace(0, 1, 2*num_ticks+3)[1:-1]

        def get_tick_at(x):
            return [x-width*0.25*direction, x-width*0.25*direction + tick_height * perp,
                    x+width*0.25*direction + tick_height * perp, x+width*0.25*direction]
        tick_points = [interpolate(start+perp*width*0.5, end+perp*width*0.5, _) for _ in tick_a]
        tick_points = [x for l in [get_tick_at(_) for _ in tick_points] for x in l]
        points = torch.stack([start+perp*width*0.5,
                              *tick_points,
                              end+perp*width*0.5, end+perp*width*k, end+direction*width*k*1.5, end-perp*width*k, end-perp*width*0.5,

                              start-perp*width*0.5, start-perp*width*k, start-direction*width*k*1.5, start+perp*width*k])[..., :2]
        path = convert_points_to_path(points)

        super().__init__([path], **kwargs)


class AxesMob(Mob):
    def __init__(self, width=1.0, quadrant=False, **kwargs):
        super().__init__(**kwargs)
        self.width = width
        self.horizontal_axis = Arrow(LEFT*width*0.5 if not quadrant else ORIGIN, RIGHT*width*0.5, **kwargs).scale(2)
        self.vertical_axis = Arrow(DOWN*width*0.5 if not quadrant else ORIGIN, UP*width*0.5, **kwargs).scale(2)
        self.width = width * 2
        self.add_children(self.horizontal_axis, self.vertical_axis)


def get_corners(start, direction, width, height, facing_direction):
    perp = broadcast_cross_product(direction, facing_direction)
    end = start + direction * height
    return torch.stack([start + perp * width, end + perp * width, end - perp * width, start - perp * width], -2)


class Bar(Quad):
    def __init__(self, start, end, width=0.05, facing_direction=OUT, **kwargs):
        self.direction = F.normalize(end - start, p=2, dim=-1)
        self.height = (end - start).norm(p=2, dim=-1)
        self.width = width
        self.facing_direction = facing_direction
        super().__init__(get_corners(start, self.direction, width, self.height, facing_direction), **kwargs)
        with Off():
            self.setattr_non_recursive('location', start)

    @animated_function(animated_args={'interpolation': 0}, unique_args=['height_func'])
    def move_to_with_height_matching(self, location, height_func, original_loc, interpolation=1):
        self.location = original_loc * (1-interpolation) + interpolation * location
        loc1 = unsquish(self.triangles[0].corners.location, -2, 3).clone()
        loc2 = unsquish(self.triangles[1].corners.location, -2, 3).clone()
        ray1 = F.normalize(loc1[...,1,:] - loc1[...,0,:], p=2, dim=-1)
        ray2 = F.normalize(loc2[..., 0, :] - loc2[..., 1, :], p=2, dim=-1)
        h = height_func(self.location[...,0])[...,1].unsqueeze(-1)
        loc1[...,1,:] = loc1[...,0,:] + ray1 * h
        loc1[..., 2,:] = loc2[..., 1,:] + ray2 * h
        loc2[...,0,:] = loc2[..., 1,:] + ray2 * h
        self.triangles[0].corners.location = squish(loc1, -3, -2)
        self.triangles[1].corners.location = squish(loc2, -3, -2)


class FunctionPlotMob(Mob):
    def __init__(self, func, axes=None, width=0.02, func_color=RED_A, num_points=200, offset=1, scale=1, max_value=None, bar_plot=False, **kwargs):
        create = kwargs['create'] if 'create' in kwargs else True
        init = kwargs['init'] if 'init' in kwargs else True
        kwargs['create'] = False
        kwargs['init'] = False
        super().__init__(**kwargs)
        new_axes = (axes is None)
        if axes is None:
            axes = AxesMob(**kwargs)
            axes.max_value = max_value

        self.axes = axes
        self.s = scale*2

        self.func_callable = func

        xs = torch.linspace(-axes.width*0.5, axes.width*0.5, num_points + (1-(num_points%2))) + 1e-3
        func_points = self.map_input_domain_to_curve_location(xs)
        func_points = func_points[~func_points[...,1].isnan()]
        func_points = func_points[func_points[...,1].abs() <= self.get_scaler()]
        #s = 1.1
        #func_points = func_points.clamp_(min=-(axes.width * 0.5 * s), max=(axes.width * 0.5 * s))

        points = func_points
        perps = [F.normalize(broadcast_cross_product((points[i + 1] if i < len(points) - 1 else points[i]) - (points[i - 1] if i > 0 else points[i]), OUT), p=2, dim=-1) for i in range(len(points))]
        #perps = [UP for i in range(len(points))]
        func_points = torch.stack([*[points[i] + perps[i] * width * 0.5 for i in range(len(points))], *reversed([points[i] - perps[i] * width * 0.5 for i in range(len(points))])])[..., :2]

        kwargs['constants'] = func_color
        with Off():
            if not bar_plot:
                self.func = TriangulatedBezierCircuit([convert_points_to_path(func_points)], **kwargs).move(OUT*0.001 * offset)
            else:
                x = self.map_input_domain_to_scaled_domain(xs)
                locs = (func_points + x) * 0.5
                heights = (func_points - x)
                widths = (x[...,1] - x[...,0]) * 0.5
                self.func = Group([Rectangle(h, w).move_to(l) for h, w, l in zip(heights, widths, locs)])
        #self.axes = axes
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
        func_points = get_func_point_at_x(xs)#torch.stack([get_func_point_at_x(x) for x in xs])
        return func_points

    def get_scaler(self):
        return self.axes.width * 0.5 * 1.1

    def map_input_domain_to_curve_location(self, xs):
        def get_func_point_at_x(x):
            return torch.stack((x, self.func_callable(x*self.s)/self.s, torch.zeros_like(x)), -1)
        func_points = get_func_point_at_x(xs)#torch.stack([get_func_point_at_x(x) for x in xs])
        if self.axes.max_value is None:
            max_value = func_points[..., 1].nan_to_num(0).abs().amax(keepdim=True).clamp_min_(1e-6).item()
            self.axes.max_value = max_value
        max_value = self.axes.max_value
        #func_points = func_points[(func_points[..., 1].abs() <= max_value+1e-6)]  # (axes.width*0.5*1.25))]
        func_points[..., 1] = (func_points[..., 1] / max_value) * self.get_scaler()
        #func_points = func_points.nan_to_num_(max_value + 1)
        return func_points

    def on_create(self):
        self.spawn_tilewise_recursive()

    def on_destroy(self):
        self.despawn_tilewise_recursive()


class TriangleVertices2(Renderable):
    def __init__(self, corner_locations, **kwargs):
        kwargs2 = {k: v for k, v in kwargs.items()}
        if 'location' in kwargs2:
            del kwargs2['location']
        kwargs2['location'] = corner_locations.view(-1, 3)
        super().__init__(**kwargs2)
        self.is_primitive = True

    def get_default_color(self):
        return RED

    def get_boundary_points_recursive(self):
        return self.location.view(-1, 3)

    def get_render_primitives(self):
        l, c, o = broadcast_all([self.location, self.color, self.opacity], ignore_dims=[-1])
        return TrianglePrimitive(l, c, o)


class Quad(Mob):
    def __init__(self, corner_locations, color=None, *args, **kwargs):
        if color is None:
            color = self.get_default_color()
        if color.dim() == 1:
            color = color.unsqueeze(0)
        if color.shape[0] == 1:
            color = color.expand(corner_locations.shape[-2], -1)
        with Sync():
            def q(_):
                return torch.cat((_[..., 2:4, :], _[..., :1, :]), -2)
            triangles = [TriangleTriangulated(corner_locations[..., :3, :], color=color[..., :3, :], *args, **kwargs),
                         TriangleTriangulated(q(corner_locations), color=q(color), *args, **kwargs)]
            kwargs['location'] = mean([_.location for _ in triangles])
            super().__init__(*args, **kwargs)
            self.triangles = triangles
            self.add_children(triangles)
