import math

import torch
import torch.nn.functional as F

from algan.animation.animatable import animated_function
from algan.animation.animation_contexts import Sync, Off
from algan.constants.color import *
from algan.constants.spatial import ORIGIN, OUT
from algan.geometry.geometry import map_global_to_local_coords
from algan.mobs.mob import Mob
from algan.mobs.shapes_2d import TriangleTriangulated
from algan.utils.file_utils import get_image
from algan.utils.tensor_utils import unsqueeze_left, squish, unsquish


def grid_to_triangle_vertices(grid):
    if grid.dim() == 1:
        return grid
    transformed_grid = grid

    triangle_corners = torch.stack((
        torch.stack((transformed_grid[:-1, :-1],
                     transformed_grid[:-1, 1:],
                     transformed_grid[1:, :-1]), -2),
        torch.stack((transformed_grid[1:, :-1],
                     transformed_grid[:-1, 1:],
                     transformed_grid[1:, 1:]), -2)), -3)
    return triangle_corners.reshape(-1, 3, transformed_grid.shape[-1])#unsquish(triangle_corners, -2, 3)

class Surface(Mob):
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

    """
    def __init__(self, coord_function=None, normal_function=None, grid_height=50, grid_width=None, grid_aspect_ratio = None, checkered_color=None, color_texture=None, ignore_normals=False, **kwargs):
        if coord_function is None:
            coord_function = self.coord_function
        if normal_function is None:
            normal_function = self.normal_function
        if grid_width is None:
            grid_width = grid_height
        if grid_aspect_ratio is not None:
            grid_height = int(grid_width * grid_aspect_ratio)

        self.grid_height, self.grid_width = grid_height, grid_width
        base_grid = self.get_base_grid()
        triangle_corners = coord_function(base_grid)
        triangle_normals = grid_to_triangle_vertices(normal_function(triangle_corners)) if not ignore_normals else None
        if 'location' in kwargs:
            triangle_corners = triangle_corners + kwargs['location']
        triangle_corners = grid_to_triangle_vertices(triangle_corners)

        color = kwargs['color'] if 'color' in kwargs else self.get_default_color()
        if checkered_color is None:
            checkered_color = color
        else:
            checkered_color = unsqueeze_left(checkered_color, color)

        if color_texture is not None:
            color = color_texture
        else:
            color_grid = torch.zeros((self.grid_width * self.grid_height, 5))
            color_grid[::2] = color
            color_grid[1::2] = checkered_color
            color_grid = color_grid.view(self.grid_height, self.grid_width, 5)
            color = color_grid
        color = grid_to_triangle_vertices(color)
        super().__init__(**kwargs)
        kwargs['color'] = color
        self.triangles = TriangleTriangulated(triangle_corners, normals=triangle_normals, **kwargs)
        self.add_children(self.triangles)

    def coord_function(self, uv):
        return torch.cat(((uv - 0.5)*2, torch.zeros_like(uv[..., :1])), -1)

    def normal_function(self, uv):
        return OUT

    def get_base_grid(self):
        grid = torch.stack((torch.linspace(0, 1, self.grid_width).view(-1, 1).expand(-1, self.grid_height),
                                 torch.linspace(0, 1, self.grid_height).view(1, -1).expand(self.grid_width, -1)), -1)
        return grid

    def set_shape_to(self, other_surface):
        with Sync():
            self.set_location_by_function(other_surface.coord_function)
            #TODO setting normals currently doesn't work, implement it.
            #self.set_normal_by_function(other_surface.normal_function)

    def set_location_by_function(self, function):
        new_loc = grid_to_triangle_vertices(function(self.get_base_grid()) + self.location)
        new_triangles = TriangleTriangulated(new_loc, normals=None, add_to_scene=False)
        with Sync():
            self.triangles.location = new_triangles.location
            self.triangles.corners.location = new_triangles.corners.location
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
