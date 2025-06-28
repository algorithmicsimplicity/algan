import math

import torch
import torch.nn.functional as F

from algan.mobs.renderable import Renderable
from algan.utils.tensor_utils import broadcast_cross_product
from algan.rendering.primitives.triangle_primitive import TrianglePrimitive
from algan.animation.animation_contexts import Sync, Off
from algan.constants.color import *
from algan.constants.spatial import ORIGIN, OUT
from algan.mobs.mob import Mob
from algan.mobs.shapes_2d import TriangleTriangulated
from algan.utils.file_utils import get_image
from algan.utils.tensor_utils import unsqueeze_left, squish, unsquish, cast_to_tensor


def grid_to_triangle_vertices(grid):
    if grid.dim() == 1:
        return grid
    transformed_grid = grid

    triangle_corners = torch.stack((
        torch.stack((transformed_grid[...,:-1, :-1,:],
                     transformed_grid[...,:-1, 1:,:],
                     transformed_grid[...,1:, :-1,:]), -2),
        torch.stack((transformed_grid[...,1:, :-1,:],
                     transformed_grid[...,:-1, 1:,:],
                     transformed_grid[...,1:, 1:,:]), -2)), -3)
    return triangle_corners.reshape(*grid.shape[:-3], -1, transformed_grid.shape[-1])#unsquish(triangle_corners, -2, 3)

class Surface(Renderable):
    """A smooth 2-D surface, embedded in 3-D space, A.K.A a manifold.
    The surface is implemented by sampling a uniform grid of 2-D points
    from the unit square (known as intrinsic coordinates, or "UV coordinates"),
    tiling this grid with triangles, and then mapping the triangle corners
    to 3-D world coordinates as defined by the manifold function.

    Parameters
    ----------
    function
        The function mapping 2-D intrinsic coordinates (ranging from [0,1]), to 3-D world coordinates,
        which defines the manifold's shape.
    function_normals
        Function mapping 3-D world coordinates to their normal vectors (i.e. vectors facing directly out
        of the surface), used for lighting.
    grid_height
        Height of the grid from which internal coordinates are sampled.
    grid_width
        Width of the grid from which internal coordinates are sampled.
    grid_aspect_radio
        If not None, set the grid_height to be equal to grid_width * grid_aspect_ratio
    *args, **kwargs
        Passed to :class:`~.Mob`

    """
    def __init__(self, coord_function=None, normal_function=None, grid_height=50, grid_width=None,
                 grid_aspect_ratio = None, checkered_color=None, color_texture=None, ignore_normals=False, *args, **kwargs):
        if coord_function is None:
            coord_function = self.coord_function
        if normal_function is None:
            normal_function = self.normal_function
        if grid_width is None:
            grid_width = grid_height
        if grid_height is None:
            grid_height = grid_width
        if grid_aspect_ratio is not None:
            grid_height = int(grid_width * grid_aspect_ratio)

        self.coord_function_active = coord_function
        self.normal_function_active = normal_function
        self.ignore_normals = ignore_normals
        #triangle_normals = grid_to_triangle_vertices(F.normalize(normal_function(base_grid), p=2, dim=-1)) if not ignore_normals else None
        super().__init__(*args, **kwargs)
        self.grid_height, self.grid_width = grid_height, grid_width
        base_grid = self.get_base_grid()
        grid_points = squish(coord_function(base_grid), -3, -2) + self.location

        color = kwargs['color'] if 'color' in kwargs else self.get_default_color()
        if checkered_color is None:
            checkered_color = color
        else:
            checkered_color = unsqueeze_left(checkered_color, color)

        if color_texture is not None:
            color = squish(color_texture, -3, -2)
        else:
            color_grid = (BLACK * 0).view(1,-1).expand((self.grid_width * self.grid_height, -1)).contiguous()
            color_grid[::2] = color
            color_grid[1::2] = checkered_color
            color_grid = color_grid.view(self.grid_height, self.grid_width, 5)
            color = squish(color_grid, -3, -2)
        #color = grid_to_triangle_vertices(color)
        kwargs['color'] = color
        kwargs['location'] = grid_points
        self.grid = Renderable(**kwargs)
        self.add_children(self.grid)
        self.grid.is_primitive = True
        self.is_primitive = True
        self.ignore_wave_animations = True

    def get_render_primitives(self):
        self.grid.set_time_inds_to(self)
        grid = unsquish(self.grid.location, -2, self.grid_height)
        if not self.ignore_normals:
            grid_x_plus_1 = grid.roll(-1,-3)
            grid_x_minus_1 = grid.roll(1,-3)
            grid_y_plus_1 = grid.roll(-1,-2)
            grid_y_minus_1 = grid.roll(1, -2)
            triangle_sides = unsquish(torch.stack((grid_x_minus_1, grid_y_minus_1,
                                                   grid_y_minus_1, grid_x_plus_1,
                                                   grid_x_plus_1, grid_y_plus_1,
                                                   grid_y_plus_1, grid_x_minus_1
                                                   ), -2) - grid.unsqueeze(-2), -2, 2)
            triangle_normals = broadcast_cross_product(triangle_sides[...,0,:], triangle_sides[...,1,:])
            triangle_normals[...,  0, :, [0,3], :] = 0
            triangle_normals[..., -1, :, [1,2], :] = 0
            triangle_normals[..., :, 0, [0,1], :] = 0
            triangle_normals[..., :, -1, [2, 3], :] = 0
            vertex_normals = -F.normalize(triangle_normals.sum(-2), p=2, dim=-1)
            vertex_normals = grid_to_triangle_vertices(vertex_normals)
        else:
            vertex_normals = None

        def expand_grid_to_verts(x):
            if x.shape[-2] == 1:
                x = x.expand(*[-1 for _ in x.shape[:-2]], grid.shape[-2]*grid.shape[-3], -1)
            x = unsquish(x, -2, self.grid_height)
            return grid_to_triangle_vertices(x)

        grid_color = self.grid.color.clone()
        grid_color[...,-1:] *= self.grid.opacity
        grid_color[..., -2:-1] += self.grid.glow
        return TrianglePrimitive(corners=grid_to_triangle_vertices(grid),
                                 colors=expand_grid_to_verts(grid_color),
                                 normals=vertex_normals,
                                 shader=self.shader,
                                 **{k: expand_grid_to_verts(v) for k, v in self.grid.get_shader_params().items()},
                                 )

    def coord_function(self, uv:torch.Tensor):
        """Default function used to map intrinsic coordinates to world space to define
        manifold shape. This method is overwritten by subclasses to define new shapes.

        Parameters
        ----------
        uv : torch.Tensor[*, 2]
            Collection of 2-D coordinates to be mapped.

        """
        return torch.cat(((uv - 0.5)*2, torch.zeros_like(uv[..., :1])), -1)

    def normal_function(self, uv):
        """Default function used to map intrinsic coordinates to world space normals to define
        manifold normal directions. This method is overwritten by subclasses to define new shapes.

        Parameters
        ----------
        uv : torch.Tensor[*, 2]
            Collection of 2-D coordinates to be mapped.

        """
        return OUT

    def get_base_grid(self):
        grid = torch.stack((torch.linspace(0, 1, self.grid_width).view(-1, 1).expand(-1, self.grid_height),
                                 torch.linspace(0, 1, self.grid_height).view(1, -1).expand(self.grid_width, -1)), -1)
        return grid

    def set_shape_to(self, other_surface:'Surface'):
        """Changes this surface's shape to the shape defined by another surface's :meth:`~.Surface.coord_function` .

        Parameters
        ----------
        other_surface
            The surface from which to get coord_function.

        """
        with Sync():
            self.set_location_by_function(other_surface.coord_function)
            #TODO setting normals currently doesn't work, implement it.
            #self.set_normal_by_function(other_surface.normal_function)

    def set_location_by_function(self, function):
        new_loc = squish(function(self.get_base_grid()), -3, -2) + self.location
        self.grid.location = new_loc
        return self

    def set_normal_by_function(self, function):
        new_normals = grid_to_triangle_vertices(function(self.get_base_grid()))
        new_triangles = TriangleTriangulated(unsquish(self.triangles.corners.location, -2, 3), normals=new_normals, add_to_scene=False)
        with Sync():
            self.triangles.basis = new_triangles.basis
            self.triangles.corners.basis = new_triangles.corners.basis
        return self

    def get_default_color(self):
        return GREEN

    def set_color_by_function(self, function):
        new_color = grid_to_triangle_vertices(function(self.get_base_grid()))
        new_triangles = TriangleTriangulated(unsquish(self.triangles.corners.location, -2, 3), color=new_color, normals=None, add_to_scene=False)
        with Sync():
            self.triangles.color = new_triangles.color
            self.triangles.corners.color = new_triangles.corners.color
        return self

    def set_color_by_texture(self, rgba_array_or_file_path):
        texture_image = get_image(rgba_array_or_file_path)
        texture_image = F.interpolate(texture_image.permute(2,0,1).unsqueeze(0), (self.grid_height, self.grid_width), mode='bilinear', antialias=True).squeeze(0).permute(2,1,0).flip(-2)
        self.triangles.corners.color = squish(grid_to_triangle_vertices(texture_image), -3, -2)
        return self
