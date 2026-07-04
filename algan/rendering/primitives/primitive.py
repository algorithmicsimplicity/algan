import torch
import torchvision
from torch.export.dynamic_shapes import Dim
import torch.nn.functional as F
import sys
import traceback
import gc

from algan import compiled, exported, not_compiled
from algan.settings.defaults import COMPUTING_DEFAULTS
from algan.constants.color import BLUE, BLACK, WHITE, RED, GREEN
from algan.geometry.geometry import intersect_line_with_plane, distance, normalize
from algan.utils.memory_utils import InsufficientMemoryException, empty_cache
from algan.utils.tensor_utils import (
    dot_product,
    squish,
    broadcast_gather,
    unsquish,
    unsqueeze_right,
)


class OutOfRenderMemory(Exception):
    pass

class RenderPrimitive:
    def __init__(
        self,
        corners=None,
        colors=BLUE,
        opacity=0,
        normals=None,
        perimeter_points=None,
        reverse_perimeter=False,
        triangle_collection=None,
        glow=0,
        glow_radius=0.2,
    ):
        self.corners = corners
        self.colors = colors
        self.normals = normals
        self.glow_radius = glow_radius
        self.padding = 1

    def get_batch_identifier(self):
        return f"{self.__class__}"

    def get_memory_used_per_timestep(self):
        return self.num_fragments_per_frame * (128)

    def get_memory_used_for_blending(self, start_ind, end_ind):
        mem_used_for_blending = self.num_fragments_per_frame * (9 * 4 + 8) * 2  # * 3 for buffers
        return mem_used_for_blending * (end_ind - start_ind)

    def get_memory_used(self, start_ind, end_ind):
        # The blending process uses, for each fragment, 1 4-channel color and 1 5-channel color (9 floats), and one index (long), so 9*4+1*8 bytes.
        return self.get_memory_used_per_timestep() * (end_ind - start_ind)

    def project_to_screen(self, camera, light_sources):
        ray_origin = camera.ray_origin
        screen_point = camera.screen_point
        screen_basis = camera.screen_basis
        screen_width = camera.screen_width
        screen_height = camera.screen_height

        light_intensity = 1
        ambient_light_intensity = 1
        d = -1
        if hasattr(self, "shader") and self.shader is not None:
            for light_source in light_sources:
                with self.memory.temp():
                    self.colors[..., :d] = self.shader(self.memory,
                        self.corners,
                        self.normals,
                        self.colors[..., :d],
                        ray_origin,
                        light_source.origin,
                        light_source.light_color,
                        light_intensity,
                        ambient_light_intensity,
                        *self.shader_param_values,
                    )

        self.first_projection = True
        (
            self.corners,
            self.corners_int,
            self.projected_distances,
            self.bounding_corners,
            self.bounding_box_sizes,
            self.bbss,
            self.num_fragments_per_object,
            self.num_fragments_per_frame,
            self.num_fragments,
            _,
        ) = self.project_and_get_bounding_boxes(
            self.corners,
            ray_origin,
            screen_point,
            screen_basis,
            screen_width,
            screen_height,
            memory=self.memory
        )
        self.first_projection = False
        return self

class RenderPrimitive2D(RenderPrimitive):
    def raycast_onto_plane(self, ray_origins, ray_directions, plane_point, plane_basis):
        dists = -dot_product(ray_origins - plane_point, plane_basis) / dot_product(
            ray_directions, plane_basis
        )
        dists.nan_to_num_()
        return dists
