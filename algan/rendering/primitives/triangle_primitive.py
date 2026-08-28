"""Batched triangle geometry for the renderer.

:class:`TrianglePrimitive` is what a mesh Mob hands the renderer: a batch of
triangles as corner positions, normals and per-vertex colors, along with the
material parameters the shading kernel reads.

Primitives from many Mobs are grouped by
:meth:`~algan.rendering.primitives.primitive.RenderPrimitive.get_batch_identifier`
and rebuilt into one merged primitive per group, so a Scene full of like objects
costs one kernel launch rather than many. ``project_to_screen`` then shades and
packs a batch once per frame window.

This is the flat-triangle case; curved geometry arrives as PN triangles carrying
corner normals, and bezier outlines as
:class:`~algan.rendering.primitives.bezier_circuit_primitive.BezierCircuitPrimitive`.
"""

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


def _broadcast_channel(values, rows):
    """A contiguous 1-D buffer plus the row stride that reads ``values`` from it.

    ``glow`` and ``opacity`` arrive as ``broadcast_all`` results -- ``expand``ed
    stride-0 views over the color rows -- and Taichi takes only contiguous
    ndarrays. Rather than materialize the expansion (which is most of the traffic
    the kernel exists to avoid), hand the kernel the underlying element(s) and
    let a stride of 0 do the broadcasting. Returns None for anything else, which
    makes the caller fall back to torch.
    """
    if values.shape[-1] != 1 or values.dtype != torch.float32 or values.numel() == 0:
        return None
    if all(stride == 0 for stride in values.stride()[:-1]):
        # Broadcast from a single element: take that one element, not N copies.
        return values[(0,) * (values.dim() - 1)].contiguous().view(-1), 0
    if values.numel() != rows:
        return None
    return values.contiguous().view(-1), 1


def _bake_glow_and_opacity(colors, opacity, glow):
    """``colors`` with glow added to channel ``-2`` and opacity scaled into ``-1``.

    The torch form is a full-size clone plus two in-place passes over strided
    one-channel views of it -- three passes for one add and one multiply per row,
    and P10b measures it at 13.5% of ``get_render_primitives_batched``. On a CPU
    Taichi arch a single kernel pass replaces all three, byte-identically; every
    other case keeps the torch path.
    """
    from algan.rendering.taichi_runtime import cpu_prep_kernel_enabled

    channels = colors.shape[-1]
    if (
        channels >= 2
        and colors.numel() > 0
        and colors.dtype == torch.float32
        and colors.device.type == "cpu"
        and colors.is_contiguous()
        and cpu_prep_kernel_enabled("cpucolors")
    ):
        rows = colors.numel() // channels
        packed_glow = _broadcast_channel(glow, rows)
        packed_opacity = _broadcast_channel(opacity, rows)
        if packed_glow is not None and packed_opacity is not None:
            from algan.rendering.primitives.triangle_primitive_kernels_taichi import (
                apply_glow_and_opacity,
            )

            out = torch.empty_like(colors)
            apply_glow_and_opacity(
                colors.view(-1),
                packed_glow[0],
                packed_opacity[0],
                out.view(-1),
                packed_glow[1],
                packed_opacity[1],
                channels,
            )
            return out

    out = colors.clone()
    out[..., -2:-1] += glow
    out[..., -1:] *= opacity
    return out


class TrianglePrimitive(RenderPrimitive):
    #: Parameter values of ``SETTINGS.style.default_material``, carried by a
    #: primitive built through the no-material fallback so the process-wide
    #: default material's look reaches the packed material block and the
    #: vertex bake, not just its shader. Empty when the mob set its own
    #: material or shader (its own values are already registered) -- every
    #: consumer treats an empty mapping as an exact no-op.
    default_material_params: dict = {}

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
        # Per-frame mob opacity for the color map, applied in the sampler
        # instead of premultiplied into the map (texture_opacity_in_kernel);
        # None = the map arrived premultiplied. Set post-construction by the
        # mob that builds the primitive, like ``mesh_ids``.
        self.texture_opacity = None
        # Authoring-side proof that every color-map texel is exactly k/255
        # with zero glow (texture_u8_storage); the merge trusts this rather
        # than probing texels (a probe is a device sync on the prefetch
        # worker).
        self.texture_u8_ok = False
        # Per-frame endpoint interpolation for the color map
        # (texture_time_lerp): a ``[T, 3]`` float tensor of (endpoint index,
        # endpoint index, weight) rows. When set, ``texture_map`` is a
        # ``[1, K, H, W, 5]`` stack of AUTHORED endpoint images (the leading
        # singleton keeps frame slicing away from the endpoint axis) and the
        # sampler lerps the two endpoint texels before decoding. None = the
        # map's leading axis is batch time, as always.
        self.texture_lerp = None
        self.material_texture_map = None
        self.material_texture_flags = 0
        self.normal_texture_map = None

        if triangle_collection is not None:
            self.shader = triangle_collection[0].shader
            # The seed rides along: members grouped under one identifier all
            # went through the same fallback (see get_batch_identifier), so
            # the first member's mapping is every member's.
            self.default_material_params = dict(
                getattr(triangle_collection[0], "default_material_params", {})
            )
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

            # Texture maps stay on whatever device built them. A color map
            # is a wide animated attribute whose frame window materializes on
            # the render device (AttributeTimeline.materialize_device), and
            # relocating it beside the corners meant copying every frame of it
            # back to the CPU here only for the projection upload to copy it
            # to the GPU again; the scene merge moves what it concatenates to
            # its own device (scene_builder._append_texture).
            for triangle in triangle_collection:
                tex = getattr(triangle, "texture_map", None)
                if tex is not None:
                    self.texture_map = tex
                    # The opacity scalars and the u8-provenance proof describe
                    # THIS map, so they ride the same first-member-with-a-map
                    # contract (exact because textured primitives are batched
                    # one per collection).
                    self.texture_opacity = getattr(triangle, "texture_opacity", None)
                    self.texture_u8_ok = bool(getattr(triangle, "texture_u8_ok", False))
                    self.texture_lerp = getattr(triangle, "texture_lerp", None)
                    break

            # Texture maps cannot be concatenated across primitives (each map
            # keeps its own resolution), so like texture_map above they are
            # taken from the first primitive that has one. The scene batcher
            # puts every textured primitive in its own singleton collection,
            # which makes this exact.
            for triangle in triangle_collection:
                tex = getattr(triangle, "material_texture_map", None)
                if tex is not None:
                    self.material_texture_map = tex
                    self.material_texture_flags = getattr(
                        triangle, "material_texture_flags", 0
                    )
                    break
            for triangle in triangle_collection:
                tex = getattr(triangle, "normal_texture_map", None)
                if tex is not None:
                    self.normal_texture_map = tex
                    break
            return

        self.corners = corners
        if normals is None:
            normals = torch.zeros_like(corners)
        colors, opacity, glow = broadcast_all(
            [colors, opacity, glow], ignored_dims=[-1]
        )
        self.colors = _bake_glow_and_opacity(colors, opacity, glow)
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
        # Left on the device that built them; see the collection branch above.
        self.texture_map = texture_map
        self.material_texture_map = material_texture_map
        self.material_texture_flags = material_texture_flags
        self.normal_texture_map = normal_texture_map

        if shader is None:
            # A mob that set no material of its own renders as the process
            # default material (SETTINGS.style.default_material, installed at
            # import as a DiffuseMaterial). Its parameters ride along so an
            # explicitly configured default -- e.g.
            # MeshStandardMaterial(roughness=0.3) -- is honoured rather than
            # silently rendering at the packed block's built-in defaults; see
            # _pack_material and _ordered_shader_param_values.
            default_material = SETTINGS.style.default_material
            if default_material is not None:
                shader = default_material.shader
                self.default_material_params = dict(
                    default_material.get_shader_param_values()
                )
        self.shader = shader

    def get_batch_identifier(self):
        # The trailing flag separates a mob that authored its own shader
        # parameter values from one shaded through the default-material seed
        # (empty ``shader_param_names``). Both can carry the same shader
        # function now that the fallback resolves to DiffuseMaterial's, but
        # their parameter rows have different widths, and the collection
        # merge transposes members column-wise -- mixing them would silently
        # truncate the authored rows to the bare mob's zero.
        return f"{self.__class__}_{id(self.shader)}_{bool(getattr(self, 'shader_param_names', None))}"
