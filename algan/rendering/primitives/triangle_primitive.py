from __future__ import annotations

import torch

from algan.constants.color import BLUE
from algan.rendering.primitives.primitive import RenderPrimitive
from algan.settings import SETTINGS
from algan.settings._startup import _ANIMATION_DEVICE
from algan.utils.tensor_utils import (
    broadcast_all,
    cast_to_tensor,
    squish,
    unsquish,
)


def _triangle_count(corners):
    """Number of triangles in flat ``[...,3F,3]`` or grouped ``[...,F,3,3]``."""
    if corners.ndim >= 4 and corners.shape[-2:] == (3, 3):
        return int(corners.shape[-3])
    if corners.shape[-2] % 3:
        raise ValueError("triangle corners must contain a multiple of three vertices")
    return int(corners.shape[-2] // 3)


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
        component_ids=None,
        **shader_kwargs,
    ):
        device = _ANIMATION_DEVICE
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
        self.component_ids = None

        if triangle_collection is not None:
            self.shader = triangle_collection[0].shader
            # Names of the positional shader_param_values, in the same order
            # (kept so the ray tracer can map them to its material slots).
            self.shader_param_names = getattr(
                triangle_collection[0], "shader_param_names", []
            )
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
            has_uvs = any(
                getattr(t, "uvs", None) is not None for t in triangle_collection
            )
            if has_uvs:
                uv_list = []
                for triangle in triangle_collection:
                    uv = getattr(triangle, "uvs", None)
                    if uv is None:
                        uv = torch.zeros(
                            (*triangle.corners.shape[:-1], 2),
                            device=triangle.corners.device,
                        )
                    else:
                        if uv.dim() == 4:
                            uv = squish(uv, -3, -2)
                        uv = uv.to(triangle.corners.device)
                    uv_list.append(uv)
                merged_uvs = []
                for i, triangle in enumerate(triangle_collection):
                    cor, uv = broadcast_all(
                        [triangle.corners, uv_list[i]], ignored_dims=[-1]
                    )
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
                        triangle, "material_texture_flags", 0
                    )
                    break
            for triangle in triangle_collection:
                tex = getattr(triangle, "normal_texture_map", None)
                if tex is not None:
                    self.normal_texture_map = tex.to(self.corners.device)
                    break

            # Component ids are renderer-internal topology, local to each
            # source primitive.  Offset each source's id range while batching
            # so disconnected mobs can never alias merely because both called
            # their first component zero.  A primitive with no topology marks
            # every face independent, which is conservative for scalar AA.
            component_parts = []
            component_base = 0
            for triangle in triangle_collection:
                num_faces = _triangle_count(triangle.corners)
                ids = getattr(triangle, "component_ids", None)
                if ids is None:
                    ids = torch.arange(
                        num_faces, dtype=torch.int32, device=self.corners.device
                    )
                else:
                    ids = ids.to(device=self.corners.device, dtype=torch.int32).view(-1)
                    if ids.numel() != num_faces:
                        raise ValueError(
                            "component_ids must contain one id per triangle "
                            f"({ids.numel()} vs {num_faces})"
                        )
                    ids = ids - ids.min()
                component_parts.append(ids + component_base)
                component_base += int(ids.max().item()) + 1 if ids.numel() else 0
            self.component_ids = torch.cat(component_parts).contiguous()
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
        self.texture_map = (
            texture_map.to(self.corners.device) if texture_map is not None else None
        )
        self.material_texture_map = (
            material_texture_map.to(self.corners.device)
            if material_texture_map is not None
            else None
        )
        self.material_texture_flags = material_texture_flags
        self.normal_texture_map = (
            normal_texture_map.to(self.corners.device)
            if normal_texture_map is not None
            else None
        )

        num_faces = _triangle_count(corners)
        if component_ids is None:
            component_ids = torch.arange(num_faces, device=corners.device)
        component_ids = (
            cast_to_tensor(component_ids)
            .to(device=corners.device, dtype=torch.int32)
            .view(-1)
        )
        if component_ids.numel() != num_faces:
            raise ValueError(
                "component_ids must contain one id per triangle "
                f"({component_ids.numel()} vs {num_faces})"
            )
        self.component_ids = component_ids.contiguous()

        if shader is None:
            shader = SETTINGS.style.default_shader
        self.shader = shader

    def get_batch_identifier(self):
        return f"{self.__class__}_{id(self.shader)}"
