"""Import a 3-D model file (glB/glTF, FBX, OBJ, ...) as an Algan :class:`~.Mob`.

:class:`ThreeDModelMob` parses the file into the backend-independent
:class:`~algan.mobs.fbx.scene_data.SceneData` IR and builds one
:class:`~algan.mobs.fbx.mesh.TriangleMesh` child per mesh instance, with
geometry baked into the model's world space.

The parser backend is chosen from the file extension:

* ``.glb`` / ``.gltf`` (and other trimesh formats) -> trimesh, which is
  pure-Python and needs no native library.
* ``.fbx`` -> pyassimp, which needs the native ``assimp`` library installed
  separately (see :mod:`algan.mobs.fbx.assimp_loader`).

Baking the node hierarchy to world-space vertices (rather than reconstructing a
live Algan transform hierarchy) keeps static import unambiguous and is also the
substrate every animation phase uses: rigid node animation, skeletal skinning
and blend-shape morphs all reduce to *per-frame vertex positions*, which the
ray tracer's spatio-temporal BVH already consumes through ``TriangleMesh``'s
animatable per-corner ``location``.
"""
from __future__ import annotations

import os

import torch

from algan.animation.animation_contexts import Off
from algan.constants.color import Color
from algan.mobs.fbx.mesh import TriangleMesh, image_to_texture_map
from algan.mobs.fbx.scene_data import SceneData
from algan.mobs.mob import Mob

# File extensions routed to the trimesh backend; everything else falls to the
# assimp (FBX) backend.
_TRIMESH_EXTS = {".glb", ".gltf", ".obj", ".ply", ".stl", ".dae", ".off"}


def _eye4(device):
    return torch.eye(4, dtype=torch.float32, device=device)


def _load_scene_for(file_path):
    """Dispatch to a parser backend by file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext in _TRIMESH_EXTS:
        from algan.mobs.fbx.gltf_loader import load_scene
    else:  # .fbx and anything else -> assimp
        from algan.mobs.fbx.assimp_loader import load_scene
    return load_scene(file_path)


def _compose_world_transforms(nodes, device):
    """World (model-space) 4x4 transform per node. Nodes are depth-first with
    ``parent < child``, so a single forward pass suffices: ``world[i] =
    world[parent] @ local[i]`` (assimp/glTF matrices transform column vectors,
    translation in the last column)."""
    world = [None] * len(nodes)
    for i, node in enumerate(nodes):
        local = node.transform
        local = _eye4(device) if local is None else local.to(device).float()
        if node.parent < 0:
            world[i] = local
        else:
            world[i] = world[node.parent] @ local
    return world


def _mesh_to_node_map(nodes):
    """Map each mesh index to the list of nodes that instance it."""
    out = {}
    for i, node in enumerate(nodes):
        for m in node.mesh_indices:
            out.setdefault(m, []).append(i)
    return out


def _transform_points(points, matrix):
    """Apply a 4x4 (column-vector) transform to ``[N, 3]`` points."""
    linear = matrix[:3, :3]
    translation = matrix[:3, 3]
    return points @ linear.T + translation


def _transform_normals(normals, matrix):
    """Transform normals by the inverse-transpose of the linear part (correct
    under non-uniform scale); falls back to the linear part if singular."""
    linear = matrix[:3, :3]
    try:
        inv_t = torch.linalg.inv(linear).T
    except Exception:
        inv_t = linear
    return normals @ inv_t.T


def _load_image_hwc(path):
    """Load an image file to ``[H, W, C]`` float in ``[0, 1]`` (C in {3, 4}),
    or ``None`` on failure."""
    try:
        import torchvision

        img = torchvision.io.read_image(path)  # [C, H, W] uint8
        return img.permute(1, 2, 0).float() / 255.0
    except Exception:
        return None


class ThreeDModelMob(Mob):
    """Load a 3-D model file and build its geometry/textures as a Mob.

    Parameters
    ----------
    file_path : str
        Path to the model file. ``.glb``/``.gltf`` (and ``.obj``/``.ply``/...)
        load through trimesh (no native dependency); ``.fbx`` loads through
        pyassimp (needs the native ``assimp`` library installed).
    scene_data : SceneData, optional
        Pre-parsed IR to build from instead of reading ``file_path`` (used by
        tests and alternative importer backends). When given, ``file_path`` may
        be a label only.
    load_textures : bool
        Load and apply diffuse texture maps referenced by materials
        (default True). When False (or a texture fails to load) meshes fall
        back to their material's flat base colour.
    normalize : bool
        Recenter and uniformly scale the whole model to fit a box of
        ``normalize_size`` (handy since model files use wildly different unit
        scales). Off by default.
    normalize_size : float
        Target bounding-box diagonal when ``normalize`` is True.
    smooth_normals : bool
        Use the mesh's authored / generated per-vertex normals for smooth
        shading (default True). When False, flat per-face normals are derived
        at render time.
    *args, **kwargs
        Passed to :class:`~.Mob` (e.g. ``location`` to place the model).
    """

    def __init__(
        self,
        file_path=None,
        scene_data: SceneData | None = None,
        load_textures: bool = True,
        normalize: bool = False,
        normalize_size: float = 2.0,
        smooth_normals: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        device = self.location.device

        if scene_data is None:
            if file_path is None:
                raise ValueError(
                    "ThreeDModelMob requires a file_path or scene_data")
            scene_data = _load_scene_for(file_path)
        self.scene_data = scene_data
        self.source_path = scene_data.source_path or (file_path or "")

        world = _compose_world_transforms(scene_data.nodes, device)
        mesh_nodes = _mesh_to_node_map(scene_data.nodes)

        # Cache loaded texture maps by file path (materials commonly share one).
        self._texture_cache: dict[str, object] = {}

        self.mesh_mobs: list[TriangleMesh] = []
        with Off():
            for mesh_idx, mesh in enumerate(scene_data.meshes):
                node_indices = mesh_nodes.get(mesh_idx, [-1])
                for node_idx in node_indices:
                    matrix = (world[node_idx] if 0 <= node_idx < len(world)
                              else _eye4(device))
                    self.mesh_mobs.append(
                        self._build_mesh_mob(
                            scene_data, mesh, matrix, device,
                            load_textures, smooth_normals))

        if not self.mesh_mobs:
            raise ValueError(
                f"No renderable meshes found in {self.source_path!r}")

        self.add_children(self.mesh_mobs)

        if normalize:
            self._normalize(normalize_size)

    def _build_mesh_mob(self, scene_data, mesh, matrix, device,
                        load_textures, smooth_normals):
        vertices = _transform_points(mesh.vertices.to(device), matrix)
        normals = None
        if smooth_normals and mesh.normals is not None:
            normals = _transform_normals(mesh.normals.to(device), matrix)

        material = scene_data.material_for(mesh)
        texture = None
        color = None
        if material is not None:
            r, g, b, a = material.base_color
            color = Color((r, g, b), opacity=a)
            if load_textures:
                texture = self._resolve_texture(material, device)

        uvs = mesh.uvs.to(device) if mesh.uvs is not None else None
        vertex_colors = (mesh.vertex_colors.to(device)
                         if (texture is None and mesh.vertex_colors is not None)
                         else None)
        # A texture needs UVs; without them, fall back to the flat colour.
        if texture is not None and uvs is None:
            texture = None

        mesh_kwargs = {}
        if color is not None and texture is None and vertex_colors is None:
            mesh_kwargs["color"] = color

        # Meshes stay lit: with authored normals (smooth_normals) they shade
        # smooth; otherwise TriangleMesh derives flat per-face normals.
        mob = TriangleMesh(
            vertices=vertices,
            faces=mesh.faces.to(device),
            normals=normals,
            uvs=uvs,
            texture=texture,
            vertex_colors=vertex_colors,
            **mesh_kwargs,
        )

        # Optional non-default material response, applied before spawn.
        if material is not None:
            if material.reflectivity and material.reflectivity > 0:
                from algan.rendering.raytracing.primitives import set_reflectivity
                set_reflectivity(mob, float(material.reflectivity))
            if material.refractive_index and material.refractive_index > 1.0:
                from algan.rendering.raytracing.primitives import (
                    set_refractive_index,
                )
                set_refractive_index(mob, float(material.refractive_index))
        return mob

    def _resolve_texture(self, material, device):
        """Diffuse texture map for a material: an embedded in-memory image
        (``diffuse_image``, e.g. from glB) takes precedence over an external
        ``diffuse_texture`` file path. Returns a ``[W, H, 5]`` map or ``None``."""
        if material.diffuse_image is not None:
            return image_to_texture_map(
                material.diffuse_image.to(device)).to(device)
        if material.diffuse_texture:
            return self._load_texture(material.diffuse_texture, device)
        return None

    def _load_texture(self, path, device):
        if path in self._texture_cache:
            return self._texture_cache[path]
        tex_map = None
        image = _load_image_hwc(path) if os.path.exists(path) else None
        if image is not None:
            tex_map = image_to_texture_map(image).to(device)
        self._texture_cache[path] = tex_map
        return tex_map

    def _normalize(self, target_size):
        """Recenter to the origin and uniformly scale so the model's bounding
        box diagonal equals ``target_size``."""
        mins, maxs = [], []
        for mob in self.mesh_mobs:
            loc = mob.grid.location.reshape(-1, 3)
            mins.append(loc.amin(0))
            maxs.append(loc.amax(0))
        lo = torch.stack(mins).amin(0)
        hi = torch.stack(maxs).amax(0)
        center = (lo + hi) * 0.5
        diagonal = (hi - lo).norm().clamp_min(1e-8)
        scale = float(target_size) / float(diagonal)
        with Off():
            for mob in self.mesh_mobs:
                mob.grid.location = (mob.grid.location - center) * scale
        return self

    def get_default_color(self):
        return Color("#CCCCCC")
