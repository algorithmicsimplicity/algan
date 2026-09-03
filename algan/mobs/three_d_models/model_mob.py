"""Import a 3-D model file (glB/glTF, FBX, OBJ, ...) as an Algan
:class:`~algan.animatable_base.mob.Mob`.

:class:`Model3D` parses the file into the backend-independent
:class:`~algan.mobs.three_d_models.scene_data.SceneData` IR and builds one
:class:`~algan.mobs.three_d_models.mesh.TriangleMesh` child per mesh instance, with
geometry baked into the model's world space.

The parser backend is chosen from the file extension:

* ``.glb`` / ``.gltf`` (and other trimesh formats) -> trimesh, which is
  pure-Python and needs no native library.
* ``.fbx`` -> pyassimp, which needs the native ``assimp`` library installed
  separately (see :mod:`~algan.mobs.three_d_models.assimp_loader`).

Baking the node hierarchy to world-space vertices (rather than reconstructing a
live Algan transform hierarchy) keeps static import unambiguous and is also the
substrate every animation phase uses: rigid node animation, skeletal skinning
and blend-shape morphs all reduce to *per-frame vertex positions*, which the
ray tracer's spatio-temporal BVH already consumes through ``TriangleMesh``'s
animatable per-corner ``location``.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from algan.animatable_base.mob import Mob
from algan.animation_timeline.animation_contexts import Off, Seq, Sync
from algan.constants.color import Color
from algan.constants.easings import identity
from algan.errors import AlganConfigurationError
from algan.mobs.three_d_models import animation as _anim
from algan.mobs.three_d_models.mesh import (
    TriangleMesh,
    image_to_normal_map,
    image_to_texture_map,
)
from algan.mobs.three_d_models.scene_data import SceneData

# File extensions routed to the trimesh backend; everything else falls to the
# assimp (FBX) backend.
_TRIMESH_EXTS = {".glb", ".gltf", ".obj", ".ply", ".stl", ".dae", ".off"}


def _eye4(device):
    return torch.eye(4, dtype=torch.float32, device=device)


def _load_scene_for(file_path):
    """Dispatch to a parser backend by file extension.

    The path goes through :func:`~algan.utils.file_utils.resolve_asset_path`
    first, so a model sitting beside the running script loads regardless of the
    working directory -- the same rule images follow.
    """
    from algan.utils.file_utils import resolve_asset_path

    file_path = resolve_asset_path(file_path)
    ext = os.path.splitext(file_path)[1].lower()
    if ext in _TRIMESH_EXTS:
        from algan.mobs.three_d_models.gltf_loader import load_scene
    else:  # .fbx and anything else -> assimp
        from algan.mobs.three_d_models.assimp_loader import load_scene
    return load_scene(file_path)


def _compose_world_transforms(nodes, device):
    """World (model-space) 4x4 transform per node. Nodes are depth-first with
    ``parent < child``, so a single forward pass suffices: ``world[i] =
    world[parent] @ local[i]`` (assimp/glTF matrices transform column vectors,
    translation in the last column).
    """
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
    under non-uniform scale); falls back to the linear part if singular.
    """
    linear = matrix[:3, :3]
    try:
        inv_t = torch.linalg.inv(linear).T
    except Exception:
        inv_t = linear
    return normals @ inv_t.T


def _load_image_hwc(path):
    """Load an image file to ``[H, W, C]`` float in ``[0, 1]`` (C in {3, 4}),
    or ``None`` on failure.
    """
    try:
        from PIL import Image

        with Image.open(path) as pil_image:
            array = np.array(pil_image)  # [H, W, C] (or [H, W] for grayscale)
        if array.ndim == 2:
            array = array[:, :, None]
        img = torch.from_numpy(array)  # [H, W, C] uint8
        return img.float() / 255.0
    except Exception:
        return None


class Model3D(Mob):
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
        back to their material's flat base color.
    fit_to_size : float, optional
        Recenter the model and uniformly scale it so its bounding-box diagonal
        is this many world units -- handy, since model files use wildly
        different unit scales. Defaults to ``None``, which leaves the model at
        the size the file gives it.
    smooth_normals : bool
        Use the mesh's authored / generated per-vertex normals for smooth
        shading (default True). When False, flat per-face normals are derived
        at render time.
    normal_maps : bool
        Apply tangent-space normal maps from materials (default True), adding
        per-fragment surface detail. Requires per-vertex UVs; batches carrying a
        normal map render through the general wavefront tracer.
    pbr_materials : bool
        Apply each material's PBR parameters (metalness / roughness / emissive)
        as a :class:`~algan.rendering.shaders.materials.MeshStandardMaterial`
        (default True), so imported meshes shade with Cook-Torrance GGX. When
        False the default lit shader is kept.
    *args, **kwargs
        Passed to :class:`~algan.animatable_base.mob.Mob` (e.g. ``location`` to
        place the model).
    """

    def __init__(
        self,
        file_path=None,
        scene_data: SceneData | None = None,
        load_textures: bool = True,
        fit_to_size: float | None = None,
        smooth_normals: bool = True,
        normal_maps: bool = True,
        pbr_materials: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        device = self.location.device
        self.normal_maps = normal_maps
        self.pbr_materials = pbr_materials

        if scene_data is None:
            if file_path is None:
                raise AlganConfigurationError(
                    "Model3D requires a file_path or scene_data"
                )
            scene_data = _load_scene_for(file_path)
        self.scene_data = scene_data
        self.source_path = scene_data.source_path or (file_path or "")

        world = _compose_world_transforms(scene_data.nodes, device)
        mesh_nodes = _mesh_to_node_map(scene_data.nodes)

        # Cache loaded texture maps by file path (materials commonly share one).
        self._texture_cache: dict[str, object] = {}

        # Recenter/scale applied by normalize(), also folded into animation
        # baking so baked poses land in the same space as the built geometry.
        self._norm_center = torch.zeros(3, device=device)
        self._norm_scale = 1.0

        self.mesh_mobs: list[TriangleMesh] = []
        # Node name -> the mesh mobs built for that node (for part access).
        self.parts: dict[str, list[TriangleMesh]] = {}
        with Off(animation_manager=self.animation_manager):
            for mesh_idx, mesh in enumerate(scene_data.meshes):
                node_indices = mesh_nodes.get(mesh_idx, [-1])
                for node_idx in node_indices:
                    matrix = (
                        world[node_idx] if 0 <= node_idx < len(world) else _eye4(device)
                    )
                    mob = self._build_mesh_mob(
                        scene_data, mesh, matrix, device, load_textures, smooth_normals
                    )
                    node_name = (
                        (
                            scene_data.nodes[node_idx].name
                            if 0 <= node_idx < len(scene_data.nodes)
                            else mesh.name
                        )
                        or mesh.name
                        or f"mesh_{mesh_idx}"
                    )
                    mob.node_name = node_name
                    # Animation hooks: the node this instance came from, and the
                    # mesh's *local* (pre-world-bake) vertices, so per-frame
                    # world corners can be re-baked from animated transforms.
                    mob._node_idx = node_idx
                    mob._local_vertices = mesh.vertices.to(device)
                    self.mesh_mobs.append(mob)
                    self.parts.setdefault(node_name, []).append(mob)

        if not self.mesh_mobs:
            raise AlganConfigurationError(
                f"No renderable meshes found in {self.source_path!r}"
            )

        self.add_children(self.mesh_mobs)

        if fit_to_size is not None:
            self._normalize(fit_to_size)

    def _build_mesh_mob(
        self, scene_data, mesh, matrix, device, load_textures, smooth_normals
    ):
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
        vertex_colors = (
            mesh.vertex_colors.to(device)
            if (texture is None and mesh.vertex_colors is not None)
            else None
        )
        # A texture needs UVs; without them, fall back to the flat color.
        if texture is not None and uvs is None:
            texture = None

        normal_texture_map = None
        if self.normal_maps and uvs is not None and material is not None:
            normal_texture_map = self._resolve_normal_map(material, device)

        mesh_kwargs = {}
        if color is not None and texture is None and vertex_colors is None:
            mesh_kwargs["color"] = color

        # Meshes stay lit: with authored normals (smooth_normals) they shade
        # smooth; otherwise TriangleMesh derives flat per-face normals.
        mob = TriangleMesh(
            scene=self.scene,
            vertices=vertices,
            faces=mesh.faces.to(device),
            normals=normals,
            uvs=uvs,
            texture=texture,
            vertex_colors=vertex_colors,
            normal_texture_map=normal_texture_map,
            **mesh_kwargs,
        )

        # PBR shading: apply the material's metalness/roughness/emissive as a
        # MeshStandardMaterial (Cook-Torrance GGX per fragment). The texture (or
        # per-vertex color) still supplies albedo; the material color is only
        # the flat fallback.
        if self.pbr_materials and material is not None:
            self._apply_pbr_material(
                mob, material, color, has_texture=texture is not None
            )

        # Legacy model formats may expose reflectivity / IOR without a full
        # metallic-roughness material. Convert those values to the same public
        # Three.js-style material workflow rather than mutating ray parameters.
        if material is not None and not self.pbr_materials:
            legacy_metalness = float(material.reflectivity or 0.0)
            legacy_ior = float(material.refractive_index or 0.0)
            if legacy_metalness > 0.0 or legacy_ior > 1.0:
                from algan.rendering.shaders.materials import MeshPhysicalMaterial

                mob.set_material(
                    MeshPhysicalMaterial(
                        color=color,
                        metalness=legacy_metalness,
                        roughness=float(material.roughness_factor),
                        ior=legacy_ior if legacy_ior > 1.0 else 1.5,
                        transmission=1.0 if legacy_ior > 1.0 else 0.0,
                    )
                )
        return mob

    def _apply_pbr_material(self, mob, material, color, has_texture):
        """Apply ``material``'s PBR params as a MeshStandardMaterial. Metalness
        and roughness are per-primitive constants for the in-kernel GGX shader;
        when a packed metallic-roughness map is present they are taken as its
        mean (modulated by the factors) since the deterministic fragment shader
        reads them per triangle, not per texel.
        """
        from algan.rendering.shaders.materials import (
            MeshPhysicalMaterial,
            MeshStandardMaterial,
        )

        metalness = float(material.metallic_factor)
        roughness = float(material.roughness_factor)
        mr = material.metallic_roughness_image
        if mr is not None and mr.shape[-1] >= 3:
            # glTF packing: G = roughness, B = metallic.
            roughness *= float(mr[..., 1].mean())
            metalness *= float(mr[..., 2].mean())
        emissive = [float(x) for x in material.emissive]
        ei = material.emissive_image
        if ei is not None and ei.shape[-1] >= 3:
            # Scale emissive by the mean of the emissive texture.
            emissive[0] *= float(ei[..., 0].mean())
            emissive[1] *= float(ei[..., 1].mean())
            emissive[2] *= float(ei[..., 2].mean())
        emissive = tuple(emissive)
        material_cls = (
            MeshPhysicalMaterial
            if float(material.refractive_index or 0.0) > 1.0
            else MeshStandardMaterial
        )
        material_kwargs = {
            "color": color,
            "metalness": metalness,
            "roughness": roughness,
            "emissive": (emissive if any(e > 0 for e in emissive) else 0x000000),
        }
        if material_cls is MeshPhysicalMaterial:
            material_kwargs.update(
                ior=float(material.refractive_index),
                transmission=max(0.0, 1.0 - float(material.base_color[3])),
            )
        mat = material_cls(**material_kwargs)
        # No map slots are passed: the model's own maps are already wired
        # through TriangleMesh above, and re-forwarding them through the
        # material would only resample what the loader has decoded.
        mob.set_material(mat)
        return mob

    def _resolve_normal_map(self, material, device):
        """Tangent-space normal map for a material as a ``[W, H, 3]`` tensor in
        ``[-1, 1]``: an embedded image takes precedence over a file path.
        """
        if material.normal_image is not None:
            return image_to_normal_map(material.normal_image.to(device)).to(device)
        if material.normal_texture and os.path.exists(material.normal_texture):
            image = _load_image_hwc(material.normal_texture)
            if image is not None:
                return image_to_normal_map(image).to(device)
        return None

    def _resolve_texture(self, material, device):
        """Diffuse texture map for a material: an embedded in-memory image
        (``diffuse_image``, e.g. from glB) takes precedence over an external
        ``diffuse_texture`` file path. Returns a ``[W, H, 5]`` map or ``None``.
        """
        if material.diffuse_image is not None:
            return image_to_texture_map(material.diffuse_image.to(device)).to(device)
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
        box diagonal equals ``target_size``.
        """
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
        with Off(animation_manager=self.animation_manager):
            for mob in self.mesh_mobs:
                mob.grid.location = (mob.grid.location - center) * scale
        # Record so precompute_animation lands baked poses in the same space.
        self._norm_center = center
        self._norm_scale = scale
        return self

    @property
    def node_names(self):
        """The names of the model's nodes that carry geometry."""
        return list(self.parts.keys())

    def get_part(self, name):
        """The imported mesh mob(s) for a named node, so a sub-part of the model
        can be manipulated (moved, colored, animated) on its own. Returns a
        single :class:`~algan.mobs.three_d_models.mesh.TriangleMesh` when the node has one
        mesh, else a list. Raises ``KeyError`` for an unknown node.
        """
        if name not in self.parts:
            raise KeyError(f"no node named {name!r}; available: {self.node_names}")
        mobs = self.parts[name]
        return mobs[0] if len(mobs) == 1 else list(mobs)

    # --- Phase 3: rigid node-keyframe animation -----------------------------
    @property
    def animations(self):
        """The animation clips
        (:class:`~algan.mobs.three_d_models.scene_data.AnimationData`) parsed from the model
        file, if any.
        """
        return self.scene_data.animations

    @property
    def animation_names(self):
        """Names of the model's animation clips."""
        return [a.name for a in self.scene_data.animations]

    def _resolve_clip(self, name):
        clips = self.scene_data.animations
        if not clips:
            raise AlganConfigurationError(
                f"model {self.source_path!r} carries no animation clips"
            )
        if name is None:
            return clips[0]
        for clip in clips:
            if clip.name == name:
                return clip
        raise KeyError(
            f"no animation named {name!r}; available: {self.animation_names}"
        )

    def precompute_animation(self, name=None, times=None, fps=30):
        """Bake an animation clip to per-frame world-space corner positions.

        Evaluates every node's animated local transform at each sample time,
        composes them down the hierarchy and transforms each mesh instance's
        local vertices, yielding the exact geometry the ray tracer renders per
        frame. Pure computation (no scene mutation), so it is unit-testable and
        also drives :meth:`play_animation`.

        Parameters
        ----------
        name : str, optional
            Clip name; defaults to the first clip.
        times : sequence of float, optional
            Explicit sample times (seconds). Defaults to an ``fps`` grid over
            the clip runtime unioned with the authored keyframe times.
        fps : int
            Sampling rate used when ``times`` is not given.

        Returns
        -------
        (times, corners) : (list[float], dict[TriangleMesh, torch.Tensor])
            The sample times and, per mesh mob, a ``[T, 3F, 3]`` stack of
            world-space corner positions (one slice per sample time).
        """
        clip = self._resolve_clip(name)
        device = self.location.device
        if times is None:
            times = _anim.sample_times(clip.runtime, fps, _anim.clip_key_times(clip))
        times = [float(t) for t in times]

        nodes = self.scene_data.nodes
        # World transform per node at each sample time.
        worlds = []
        for t in times:
            locals_ = _anim.evaluate_animated_locals(nodes, clip, t, device=device)
            worlds.append(_anim.compose_world_from_locals(nodes, locals_))

        center = self._norm_center.to(device)
        scale = self._norm_scale
        corners = {}
        for mob in self.mesh_mobs:
            local_corners = mob._local_vertices[mob.corner_index]  # [3F, 3] local
            frames = []
            for w in worlds:
                matrix = (
                    w[mob._node_idx] if 0 <= mob._node_idx < len(w) else _eye4(device)
                )
                world_corners = _transform_points(local_corners, matrix)
                # Fold in the same recenter/scale normalize() applied.
                frames.append((world_corners - center) * scale)
            corners[mob] = torch.stack(frames, dim=0)  # [T, 3F, 3]
        return times, corners

    def play_animation(self, name=None, runtime=None, fps=30, loop=1, easing=identity):
        """Play a baked node-keyframe animation on the timeline.

        The clip is baked to per-frame world corners (see
        :meth:`precompute_animation`) and each mesh instance's corners are driven
        through those poses, so rigid node motion (a bone/part translating,
        rotating or scaling, composed down the hierarchy) plays back. Meshes
        with authored normals are switched to per-frame smooth-normal
        recomputation so shading stays correct as the geometry moves.

        Parameters
        ----------
        name : str, optional
            Clip name; defaults to the first clip.
        runtime : float, optional
            Playback runtime in seconds (per loop). Defaults to the clip's
            authored runtime.
        fps : int
            Sampling rate for baking (higher = smoother rotation, since corners
            are linearly interpolated between baked poses).
        loop : int
            Number of times to repeat the clip.
        easing : callable
            Timeline rate function; defaults to linear playback.
        """
        clip = self._resolve_clip(name)
        times, corners = self.precompute_animation(name, fps=fps)
        if len(times) < 2:
            return self
        if runtime is None:
            runtime = clip.runtime or float(times[-1]) or 1.0

        # Recompute smooth normals per frame so authored-normal meshes shade
        # correctly under the deformation.
        for mob in self.mesh_mobs:
            if mob.corner_normals is not None:
                mob.recompute_normals = True

        # Frame 0 is set instantly, then the geometry is swept through the
        # remaining baked poses; each Sync step moves every mesh together and
        # Seq sequences the steps (rescaled to runtime).
        with Off(animation_manager=self.animation_manager):
            for mob in self.mesh_mobs:
                mob.grid.set_location(corners[mob][0])
        for _lap in range(max(1, int(loop))):
            with Seq(
                runtime=runtime,
                easing=easing,
                animation_manager=self.animation_manager,
            ):
                for k in range(1, len(times)):
                    with Sync(animation_manager=self.animation_manager):
                        for mob in self.mesh_mobs:
                            mob.grid.set_location(corners[mob][k])
        return self

    def get_default_color(self):
        return Color("#CCCCCC")
