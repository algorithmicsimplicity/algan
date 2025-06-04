import torch
import torchvision.io

from algan.mobs.mob import Mob
from algan.mobs.shapes_2d import TriangleVertices


class ImageMob(Mob):#TriangleVertices):
    def __init__(self, file_path=None, rgba_array=None, ):
        if file_path is None and rgba_array is None:
            raise RuntimeError('At least one of file_path and rgba_array must be provided.')
        if file_path is not None:
            rgba_array = torchvision.io.read_image(file_path).permute(1,2,0)
            rgba_array = rgba_array.float() / 255

        if rgba_array.shape[-1] < 4:
            rgba_array = torch.cat((rgba_array, torch.ones_like(rgba_array[...,:1])), -1)
        if rgba_array.shape[-1] < 5:
            rgba_array = torch.cat((rgba_array[...,:-1], torch.zeros_like(rgba_array[...,:1]), rgba_array[...,-1:]), -1)

        h = rgba_array.shape[-3]
        w = rgba_array.shape[-2]
        aspec_ratio = w / max(h, 1)
        pixel_locs = torch.stack(((torch.linspace(-1, 1, h).view(-1,1) * 0.5).expand(-1,w), (torch.linspace(-1, 1, w) * 0.5 * aspec_ratio).view(1,-1).expand(h, w)), -1).transpose(-3, -2).flip(-3)
        pixel_locs = torch.cat((pixel_locs, torch.zeros_like(pixel_locs[...,:1])), -1)

        def expand_grid_to_triangles(grid):
            upper_triangles = torch.stack((grid[:-1, :-1], grid[1:, :-1], grid[:-1, 1:]), -2)
            lower_triangles = torch.stack((grid[1:, :-1], grid[:-1, 1:], grid[1:, 1:]), -2)
            triangles = torch.stack((lower_triangles, upper_triangles), -3)
            triangles = triangles.reshape(-1, triangles.shape[-1])
            return triangles

        super().__init__()
        self.pixels = TriangleVertices(expand_grid_to_triangles(pixel_locs), color=expand_grid_to_triangles(rgba_array))
        self.add_children(self.pixels)
