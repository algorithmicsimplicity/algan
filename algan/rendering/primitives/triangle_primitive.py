import torch

from algan.constants.color import BLUE
from algan.settings.defaults import COMPUTING_DEFAULTS, RENDERING_DEFAULTS
from algan.rendering.primitives.primitive import RenderPrimitive
from algan.utils.tensor_utils import (
    broadcast_all,
    cast_to_tensor,
    squish,
    unsquish,
)


class TrianglePrimitive(RenderPrimitive):
    def __init__(
        self,
        corners=None,
        colors=BLUE,
        opacity=1,
        normals=None,
        perimeter_points=None,
        reverse_perimeter=False,
        triangle_collection=None,
        glow=0,
        shader=None,
        uvs=None,
        texture_map=None,
        material_texture_map=None,
        material_texture_flags=0,
        normal_texture_map=None,
        **shader_kwargs,
    ):
        device = COMPUTING_DEFAULTS.animation_device
        glow = cast_to_tensor(glow).to(device)
        opacity = cast_to_tensor(opacity).to(device)
        """
        corners: Tensor[batch[*], num_corners[3], corner_locations[3]]
            Location of triangle vertices/corners in 3d world space.
        constants: Tensor[batch[*], num_corners[3], RGBA[3|4]]
            RGBA constants values for each vertex. A value is optional, if not provided then transparency parameter will be used.
        transparency: Tensor[batch[*], num_corners[3], 1]
            Transparency value for each vertex. Only used if A is not provided in constants value.
        triangle_collection: Iterable[TrianglePrimitive]
            Collection of n Triangles, if provided then this instance will be a batch containing the corners and constants
            of all n triangles in one. If provided, all other parameters are ignored.
        """
        self.reverse_perimeter = reverse_perimeter
        self.min_interpolation_coord = 0
        self.uvs = None
        self.texture_map = None
        self.material_texture_map = None
        self.material_texture_flags = 0
        self.normal_texture_map = None

        if triangle_collection is not None:
            self.shader = triangle_collection[0].shader
            # Names of the positional shader_param_values, in the same order
            # (kept so the ray tracer can map them to its material slots).
            self.shader_param_names = getattr(
                triangle_collection[0], "shader_param_names", [])
            self.corners, self.colors, self.normals, *self.shader_param_values = (
                unsquish(torch.cat(_, 1), -2, 3)
                for _ in zip(
                    *(
                        broadcast_all(
                            [
                                triangle.corners,
                                triangle.colors,
                                triangle.normals,
                                *triangle.shader_param_values,
                            ],
                            ignored_dims=[-1],
                        )
                        for triangle in triangle_collection
                    )
                )
            )

            # Check if any triangle in the collection has uvs or texture_map
            has_uvs = any(getattr(t, "uvs", None) is not None for t in triangle_collection)
            if has_uvs:
                uv_list = []
                for triangle in triangle_collection:
                    uv = getattr(triangle, "uvs", None)
                    if uv is None:
                        uv = torch.zeros((*triangle.corners.shape[:-1], 2), device=triangle.corners.device)
                    else:
                        if uv.dim() == 4:
                            uv = squish(uv, -3, -2)
                        uv = uv.to(triangle.corners.device)
                    uv_list.append(uv)
                merged_uvs = []
                for i, triangle in enumerate(triangle_collection):
                    cor, uv = broadcast_all([triangle.corners, uv_list[i]], ignored_dims=[-1])
                    merged_uvs.append(uv)
                self.uvs = unsquish(torch.cat(merged_uvs, 1), -2, 3)

            for triangle in triangle_collection:
                tex = getattr(triangle, "texture_map", None)
                if tex is not None:
                    self.texture_map = tex.to(self.corners.device)
                    break

            # Texture maps cannot be concatenated across primitives (each map
            # keeps its own resolution), so like texture_map above they are
            # taken from the first primitive that has one. The scene batcher
            # puts every textured primitive in its own singleton collection,
            # which makes this exact.
            for triangle in triangle_collection:
                tex = getattr(triangle, "material_texture_map", None)
                if tex is not None:
                    self.material_texture_map = tex.to(self.corners.device)
                    self.material_texture_flags = getattr(
                        triangle, "material_texture_flags", 0)
                    break
            for triangle in triangle_collection:
                tex = getattr(triangle, "normal_texture_map", None)
                if tex is not None:
                    self.normal_texture_map = tex.to(self.corners.device)
                    break
            return

        self.corners = corners
        if normals is None:
            normals = torch.zeros_like(corners)
        colors, opacity, glow = broadcast_all(
            [colors, opacity, glow], ignored_dims=[-1]
        )
        self.colors = colors.clone()
        self.colors[..., -2:-1] += glow
        self.colors[..., -1:] *= opacity
        self.glow = glow
        self.normals = normals
        self.shader_param_names = list(shader_kwargs.keys())
        self.shader_param_values = broadcast_all(
            [colors, *shader_kwargs.values()], ignored_dims=[-1]
        )[1:]

        if uvs is not None:
            if uvs.dim() == 3:
                uvs = unsquish(uvs, -2, 3)
            self.uvs = uvs.to(self.corners.device)
        self.texture_map = texture_map.to(self.corners.device) if texture_map is not None else None
        self.material_texture_map = (material_texture_map.to(self.corners.device)
                                     if material_texture_map is not None else None)
        self.material_texture_flags = material_texture_flags
        self.normal_texture_map = (normal_texture_map.to(self.corners.device)
                                   if normal_texture_map is not None else None)

        if shader is None:
            shader = RENDERING_DEFAULTS.shader
        self.shader = shader

    def get_batch_identifier(self):
        return f"{self.__class__}_{id(self.shader)}"
