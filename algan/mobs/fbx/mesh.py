"""A general indexed triangle-mesh :class:`~.Mob`.

Unlike :class:`~algan.mobs.surfaces.surface.Surface` (which is parametric -- a
``coord_function`` sampled on a regular UV grid), :class:`TriangleMesh` renders
an arbitrary triangle soup: explicit per-corner positions, normals and UVs plus
an optional texture map. This is the geometry container that
:class:`~algan.mobs.fbx.model_mob.ThreeDModelMob` builds each imported mesh node
from, but it is usable on its own for any hand-built or procedurally generated
mesh.

The class mirrors ``Surface``'s render structure exactly -- a parent
``Renderable`` holding a child ``self.grid`` ``Renderable`` that carries the
per-corner geometry as its animatable ``location``/``color`` -- so it plugs into
the same scene materialization, batching and (ray traced) texture pipeline that
already drives ``Surface``/``ImageMob``. Storing the corners as an animatable
attribute is also what lets a later animation phase drive per-frame vertex
positions (skinning / morph baking) with no renderer changes: the
spatio-temporal BVH already consumes ``[T, N, 3, 3]`` corners.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from algan.constants.color import Color, WHITE
from algan.geometry.geometry import map_local_to_global_coords
from algan.mobs.renderable import Renderable
from algan.rendering.primitives.triangle_primitive import TrianglePrimitive
from algan.utils.tensor_utils import cast_to_tensor, unsquish


def image_to_texture_map(image):
    """Convert an image ``[H, W, C]`` (rows top-to-bottom, ``C`` in 1/3/4
    channels) to the ``[W, H, 5]`` texture-map layout the ray tracer samples,
    matching the convention used by :class:`~algan.mobs.image_mob.ImageMob`.

    The kernel indexes a texture by ``u`` along the first stored axis and ``v``
    along the second (``local_idx = u_idx * height + v_idx``, see
    ``_sample_texture`` in ``ray_trace_taichi``), with ``v`` measured
    bottom-up. An image is stored ``[row, col]`` with ``row`` top-down, so we
    transpose ``[H, W] -> [W, H]`` (columns become the ``u`` axis) and flip the
    ``v`` axis. Channels are padded to the engine's 5-slot colour
    ``(r, g, b, glow, opacity)`` by :meth:`Color.add_defaults`.
    """
    image = cast_to_tensor(image).float()
    if image.dim() != 3:
        raise ValueError(
            f"texture image must be [H, W, C], got shape {tuple(image.shape)}")
    if image.shape[-1] > 1 and image.shape[-1] <= 4:
        image = Color.add_defaults(image)
    elif image.shape[-1] == 1:
        image = Color.add_defaults(image.expand(*image.shape[:-1], 3))
    return image.transpose(-3, -2).flip(-2).as_subclass(Color)


class TriangleMesh(Renderable):
    """An arbitrary indexed triangle mesh with optional per-corner normals,
    UVs and a texture map.

    Parameters
    ----------
    vertices : torch.Tensor[V, 3]
        Vertex positions.
    faces : torch.Tensor[F, 3]
        Triangle vertex indices into ``vertices``.
    normals : torch.Tensor[V, 3], optional
        Per-vertex normals (in the mesh's local frame). When omitted, flat
        per-face normals are computed from the geometry each frame (so they
        stay correct under rotation and deformation).
    uvs : torch.Tensor[V, 2], optional
        Per-vertex texture coordinates. Required to show ``texture`` /
        ``material_texture_map`` / ``normal_texture_map``.
    texture : torch.Tensor, optional
        Either a ``[W, H, 5]`` texture map already in engine layout (as built
        by :func:`image_to_texture_map`) or a raw image ``[H, W, C]`` which is
        converted automatically. Provides the surface albedo; replaces the
        per-vertex colour where present.
    vertex_colors : torch.Tensor[V, 5], optional
        Per-vertex colours (RGB + glow + opacity). Ignored when a ``texture``
        is given. Falls back to a single ``color`` otherwise.
    material_texture_map, material_texture_flags, normal_texture_map
        Optional per-texel material-property / tangent-space normal maps,
        forwarded to the ray tracer exactly as for
        :class:`~algan.mobs.surfaces.surface.Surface`.
    ignore_normals : bool
        If True the mesh carries zero normals (no lighting interaction).
    *args, **kwargs
        Passed to :class:`~.Mob`.
    """

    def __init__(
        self,
        vertices,
        faces,
        normals=None,
        uvs=None,
        texture=None,
        vertex_colors=None,
        material_texture_map=None,
        material_texture_flags=0,
        normal_texture_map=None,
        ignore_normals=False,
        **kwargs,
    ):
        vertices = cast_to_tensor(vertices).view(-1, 3)
        faces = torch.as_tensor(faces, device=vertices.device).long().view(-1, 3)
        # Flatten to per-corner ("triangle soup") arrays: three corners per
        # face, laid out so consecutive triples are one triangle -- exactly the
        # ordering the TrianglePrimitive / trace kernel expect.
        corner_index = faces.reshape(-1)  # [3F]
        self.num_triangles = faces.shape[0]

        super().__init__(**kwargs)
        device = self.location.device
        vertices = vertices.to(device)
        corner_index = corner_index.to(device)

        corner_positions = vertices[corner_index]  # [3F, 3]

        self.ignore_normals = ignore_normals
        if ignore_normals or normals is None:
            self.corner_normals = None
        else:
            normals = cast_to_tensor(normals).view(-1, 3).to(device)
            self.corner_normals = F.normalize(
                normals[corner_index], p=2, dim=-1)

        # Texture / material maps.
        self.texture_map = None
        if texture is not None:
            texture = cast_to_tensor(texture).to(device)
            # Already-in-layout maps are [W, H, 5] (or [T, W, H, 5]); a raw
            # image is [H, W, C] with C < 5 -> convert it.
            if texture.dim() == 3 and texture.shape[-1] != 5:
                texture = image_to_texture_map(texture)
            self.texture_map = texture.to(device).as_subclass(Color)
        self.material_texture_map = (
            cast_to_tensor(material_texture_map).to(device)
            if material_texture_map is not None else None)
        self.material_texture_flags = material_texture_flags
        self.normal_texture_map = (
            cast_to_tensor(normal_texture_map).to(device)
            if normal_texture_map is not None else None)

        has_any_texture = (self.texture_map is not None
                           or self.material_texture_map is not None
                           or self.normal_texture_map is not None)
        if has_any_texture:
            if uvs is None:
                raise ValueError(
                    "TriangleMesh with a texture/material/normal map requires "
                    "per-vertex `uvs`")
            uvs = cast_to_tensor(uvs).view(-1, 2).to(device)
            # [1, 3F, 2] -- static per-corner UVs (broadcast over time in the
            # primitive, matching Surface.get_render_primitives).
            self.corner_uvs = uvs[corner_index].unsqueeze(0)
        else:
            self.corner_uvs = None

        # Per-corner base colours. A colour map supplies the real albedo per
        # fragment, so the per-vertex colours are only a fallback / frame
        # visibility signal there -- keep them opaque white. Otherwise use the
        # supplied per-vertex colours (or the single mob colour).
        if self.texture_map is not None:
            corner_colors = WHITE.view(1, -1).expand(corner_positions.shape[0], -1)
        elif vertex_colors is not None:
            vertex_colors = Color.add_defaults(
                cast_to_tensor(vertex_colors).to(device))
            corner_colors = vertex_colors[corner_index]
        else:
            corner_colors = self.color.view(1, -1).expand(
                corner_positions.shape[0], -1)
        corner_colors = corner_colors.contiguous().as_subclass(Color)

        # The child Renderable carries the per-corner geometry as its animatable
        # location/colour (mirrors Surface.self.grid). Style attributes
        # (opacity, glow, glow_radius) are inherited from kwargs.
        grid_kwargs = {k: v for k, v in kwargs.items()
                       if k not in ("location", "color", "basis")}
        grid_kwargs["location"] = corner_positions
        grid_kwargs["color"] = corner_colors
        self.grid = Renderable(**grid_kwargs)
        self.add_children(self.grid)
        self.components = [self.grid]
        self.grid.is_primitive = True
        self.is_primitive = True
        self.ignore_wave_animations = True

    def _compute_corner_normals(self, corners_flat):
        """World-space per-corner normals for the current (already
        transformed) corner positions ``[T, 3F, 3]``.

        Authored normals are rotated out of the mesh's local frame by the
        child's materialized basis (exactly as :class:`TriangleVertices`
        does). Without authored normals, flat per-face normals are derived from
        the corner geometry so shading stays correct under any deformation.
        """
        if self.ignore_normals:
            return None
        if self.corner_normals is not None:
            n_local = self.corner_normals.unsqueeze(0).expand_as(corners_flat)
            world = map_local_to_global_coords(
                corners_flat, self.grid.basis, n_local) - corners_flat
            return F.normalize(world, p=2, dim=-1)
        # Flat face normals: cross product of two edges, shared by all 3 corners.
        tris = unsquish(corners_flat, -2, 3)  # [T, F, 3, 3]
        e1 = tris[..., 1, :] - tris[..., 0, :]
        e2 = tris[..., 2, :] - tris[..., 0, :]
        face_n = F.normalize(torch.cross(e1, e2, dim=-1), p=2, dim=-1)  # [T, F, 3]
        return face_n.unsqueeze(-2).expand(*face_n.shape[:-1], 3, 3).reshape(
            corners_flat.shape)

    def get_render_primitives(self):
        self.grid.set_time_inds_to(self)
        corners = self.grid.location  # [T, 3F, 3]

        normals = self._compute_corner_normals(corners)
        if normals is None:
            normals = torch.zeros_like(corners)

        grid_color = self.grid.color.clone()
        grid_color[..., -1:] = grid_color[..., -1:] * self.grid.opacity
        grid_color[..., -2:-1] = grid_color[..., -2:-1] + self.grid.glow
        colors = grid_color

        texture_map = None
        if self.texture_map is not None:
            # Pre-multiply the texture's opacity by the mob's (per-frame)
            # opacity so the standard spawn/despawn fade drives textured meshes
            # too -- the same trick Surface/ImageMob use. The map is [W, H, 5]
            # (add a leading frame axis) and the mob opacity is one value per
            # frame; broadcast both to [T, W, H, *].
            tmap = self.texture_map
            if tmap.dim() == 3:
                tmap = tmap.unsqueeze(0)  # [1, W, H, 5]
            op = self.opacity.reshape(-1, 1, 1, 1)
            texture_map = tmap.as_subclass(Color).mult_opacity(op)

        return TrianglePrimitive(
            corners=corners,
            colors=colors,
            normals=normals,
            glow=colors[..., -2:-1].as_subclass(torch.Tensor),
            glow_radius=self.grid.glow_radius,
            shader=self.shader,
            uvs=self.corner_uvs,
            texture_map=texture_map,
            material_texture_map=self.material_texture_map,
            material_texture_flags=self.material_texture_flags,
            normal_texture_map=self.normal_texture_map,
            **self.grid.get_shader_params(),
        )

    def get_memory_used_per_timestep(self):
        n_v = self.num_triangles * 3
        # location(3) + color(5) animation state, plus primitive corners/colors/
        # normals and ~24 bytes/vertex of RT frame-bound + BVH overhead.
        num_vars = 16
        for _ in self.grid.get_shader_params().values():
            num_vars += _.shape[-1]
        return n_v * (num_vars * 4 + 24)

    def get_default_color(self):
        return WHITE
