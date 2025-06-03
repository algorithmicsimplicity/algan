import math

import torch
import torch.nn.functional as F

from algan.animation.animation_contexts import Off, Sync
from algan.constants.spatial import UP, RIGHT
from algan.geometry.geometry import map_local_to_global_coords
from algan.mobs.bezier_circuit import BezierCircuitCubic
from algan.mobs.mob import Mob
from algan.mobs.renderable import Renderable
from algan.rendering.primitives.triangle import TrianglePrimitive
from algan.defaults.style_defaults import *
from algan.utils.tensor_utils import unsqueeze_left, broadcast_all, cast_to_tensor
from algan.utils.tensor_utils import mean


class TriangleTriangulated(Mob):
    def __init__(self, corner_locations, vertices=None, normals=None, **kwargs):
        if vertices is None:
            vertices = TriangleVertices
        if 'color' in kwargs:
            color = kwargs['color']
            del kwargs['color']
        else:
            color = self.get_default_color()
        super().__init__(**kwargs)
        kwargs['color'] = color
        k = self.location
        #scl = squish(corner_locations, 0, -2)
        if vertices == TriangleVertices:
            corner_locations = corner_locations.view(-1,3,3)
            kwargs['parent_batch_sizes'] = torch.full((len(corner_locations),), 3)
        else:
            corner_locations = corner_locations.view(-1, corner_locations.shape[-2], 3)
            kwargs['parent_batch_sizes'] = torch.full((len(corner_locations),), corner_locations.shape[-2])
        self.corners = vertices(corner_locations, normals, **kwargs)
        if vertices != TriangleVertices:
            with Off(record_attr_modifications=False):
                self.location = self.corners.location.mean(-2, keepdim=True)
            self.add_children(self.corners)
            return
        a = corner_locations[..., 0, :]
        b = corner_locations[..., 1, :]
        c = corner_locations[..., 2, :]
        m = (c - a).norm(p=2, dim=-1, keepdim=True).square() * torch.cross(torch.cross(b - a, c - a, -1), b - a, -1) + \
            (b - a).norm(p=2, dim=-1, keepdim=True).square() * torch.cross(torch.cross(c - a, b - a, -1), c - a, -1)
        m = a + m / (2 * torch.cross(b - a, c - a, -1).norm(p=2, dim=-1, keepdim=True).square().clamp_min_(1e-10))
        with Off(record_attr_modifications=False):
            self.location = m#.unsqueeze(-2)
            if self.corners.color.shape[-2] > 1:
                corner_colors = self.corners.color.view(-1, 3, self.corners.color.shape[-1]).mean(-2)
            else:
                corner_colors = self.corners.color
            self.color = corner_colors
        self.add_children(self.corners)

    def get_default_color(self):
        return YELLOW


class TriangleVertices(Renderable):
    def __init__(self, corner_locations, normals=None, **kwargs):
        kwargs2 = {k: v for k, v in kwargs.items()}
        if 'location' in kwargs2:
            del kwargs2['location']
        kwargs2['location'] = corner_locations.reshape(-1, 3)
        if 'color' in kwargs2:
            kwargs2['color'] = kwargs2['color'].reshape(-1,kwargs2['color'].shape[-1])
        if normals is not None:
            normals = normals.reshape(-1, 3)
        super().__init__(**kwargs2)
        self.normals = normals
        self.is_primitive = True
        self.num_points_per_object = 3

    def get_default_color(self):
        return PURE_RED

    def get_render_primitives(self):
        l, c, o, n, g = broadcast_all([self.location, self.color, self.opacity * self.max_opacity, self.normals, self.glow], ignore_dims=[-1])
        if n is None:
            n = torch.zeros_like(l)
        return TrianglePrimitive(l, c, o, F.normalize(map_local_to_global_coords(self.location, self.basis, n) - self.location, p=2, dim=-1), glow=g)


class QuadTriangulated(Mob):
    def __init__(self, corner_locations, **kwargs):
        def q(_):
            return torch.cat((_[..., 2:4, :], _[..., :1, :]), -2)
        triangles = [TriangleTriangulated(corner_locations[..., :3, :], **kwargs),
                     TriangleTriangulated(q(corner_locations), **kwargs)]
        kwargs['location'] = mean([_.location for _ in triangles])
        super().__init__(**kwargs)
        self.triangles = triangles
        self.add_children(triangles)


class Polygon(BezierCircuitCubic):
    """A 2-D planar polygon with N vertices.

    Parameters
    ----------
    vertex_locations : torch.Tensor[N, 3]
        3-D coordinates for each of the N vertex points.
    """
    def __init__(self, vertex_locations, *args, **kwargs):
        corner_locations = cast_to_tensor(vertex_locations)[0]
        control_points = []
        for line_start, line_end in zip(corner_locations, corner_locations.roll(-1,-2)):
            control_points.append(torch.stack([line_start * (1-a) + a * line_end for a in torch.linspace(0, 1, 4)]))

        control_points = torch.cat(control_points, -2)
        super().__init__(control_points, *args, **kwargs)

    def get_default_color(self):
        return RED


class RegularPolygon(Polygon):
    def __init__(self, num_vertices, *args, **kwargs):
        vertices = torch.stack([UP * torch.sin(a) + RIGHT * torch.cos(a) for a in torch.linspace(math.pi/2, -math.pi * 1.5, num_vertices+1)[:-1]], -2)
        super().__init__(vertices, *args, **kwargs)


class Quad(Polygon):
    pass


class Triangle(Polygon):
    pass


class Rectangle(Quad):
    def __init__(self, height=2, width=2, **kwargs):
        corners = torch.tensor(((-width, height,0),
                                      (width, height,0),
                                      (width, -height,0),
                                      (-width, -height,0),
                                      ))*0.5
        if 'location' in kwargs:
            corners += kwargs['location']
            del kwargs['location']
        super().__init__(corners, **kwargs)


class Square(Rectangle):
    def __init__(self, height=2, **kwargs):
        super().__init__(height, height, **kwargs)


class Circle(BezierCircuitCubic):
    def __init__(self, radius=1, *args, **kwargs):
        a = 1.00005519
        b = 0.55342686
        c = 0.99873585
        control_points_quarter = torch.tensor([[0,a], [b,c], [c,b], [a,0]])

        def rot90_in_2d(x):
            return torch.stack([x[...,1], -x[...,0]],-1)

        def rot_n_quarters(x, n):
            for i in range(n):
                x = rot90_in_2d(x)
            return x
        control_points = torch.cat([rot_n_quarters(control_points_quarter, i) for i in range(4)], -2)
        control_points = torch.cat([control_points, torch.zeros_like(control_points[...,:1])], -1)
        l = 0
        if 'location' in kwargs:
            l = kwargs['location']
            del kwargs['location']

        super().__init__(control_points, *args, **kwargs)
        self.scale(radius)
        self.move_to(l)

    @property
    def radius(self):
        return self.scale_coefficient[...,0]

    @radius.setter
    def radius(self, radius):
        self.scale_coefficient = radius

    def get_default_color(self):
        return BLUE
