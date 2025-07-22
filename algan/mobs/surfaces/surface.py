from __future__ import annotations

import torch

from algan.constants.color import *
from algan.mobs.mob import Mob
from algan.mobs.shapes_2d import TriangleTriangulated
from algan.utils.tensor_utils import unsqueeze_left


class Surface(Mob):
    def __init__(self, function, function_normals=None, num_grid_pieces=50, checkered_color=None, **kwargs):
        base_grid = torch.stack((torch.linspace(0, 1, num_grid_pieces).view(-1,1).expand(-1, num_grid_pieces),
                                 torch.linspace(0, 1, num_grid_pieces).view(1,-1).expand(num_grid_pieces, -1)), -1)
        transformed_grid = function(base_grid)

        triangle_corners = torch.stack((
            torch.stack((transformed_grid[:-1, :-1],
                         transformed_grid[:-1, 1:],
                         transformed_grid[1:, :-1]), -2),
            torch.stack((transformed_grid[1:, :-1],
                         transformed_grid[:-1, 1:],
                         transformed_grid[1:, 1:]), -2)), -3)

        triangle_normals = function_normals(triangle_corners).reshape(-1,3,3) if function_normals is not None else None
        color = kwargs['color'] if 'color' in kwargs else self.get_default_color()
        if checkered_color is None:
            checkered_color = color
        else:
            checkered_color = unsqueeze_left(checkered_color, color)
        triangle_corners = triangle_corners.reshape(-1, 3, 3)
        color = torch.stack((color, color, checkered_color, checkered_color), -2).repeat((len(triangle_corners)//4)+1, 1)[:len(triangle_corners)].unsqueeze(-2).expand(-1,3,-1)
        super().__init__(**kwargs)
        kwargs['color'] = color
        self.triangles = TriangleTriangulated(triangle_corners, normals=triangle_normals, **kwargs)
        self.add_children(self.triangles)

    def get_default_color(self):
        return GREEN
