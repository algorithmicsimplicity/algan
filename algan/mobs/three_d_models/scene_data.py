"""Backend-independent intermediate representation (IR) for imported 3-D models.

An importer backend (:mod:`algan.mobs.three_d_models.gltf_loader` for glB/glTF via trimesh,
or :mod:`algan.mobs.three_d_models.assimp_loader` for FBX via pyassimp) parses a model file
into these plain dataclasses; :class:`~algan.mobs.three_d_models.model_mob.Model3D`
consumes them and builds the Algan mob tree. Keeping the IR separate from both
the parser and the mob builder means:

* the parsing backend is swappable (trimesh, FBX SDK, pyassimp, ...) without
  touching :class:`~algan.mobs.three_d_models.model_mob.Model3D`;
* the IR can be hand-built in tests, so the mob builder is testable with no
  parser backend present.

The IR is intentionally a superset of what the current (static-geometry) import
phase consumes: ``SkinData``, morph targets and ``AnimationData`` are populated
by the loader where available and are the hooks the skeletal / morph animation
phases build on. Phase 1 reads only
:attr:`~algan.mobs.three_d_models.scene_data.SceneData.meshes`,
:attr:`~algan.mobs.three_d_models.scene_data.SceneData.materials` and the node
transforms.

All arrays are plain :class:`torch.Tensor` (or ``None``); coordinates are in the
model's own space (the loader applies assimp's post-process transforms, e.g.
triangulation and, when requested, axis conversion).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class MaterialData:
    """A surface material. Texture fields are absolute file paths (resolved
    relative to the model file) or ``None``; ``*_factor`` fields are the flat
    fallbacks used when no map is present.
    """

    name: str = ""
    base_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    diffuse_texture: str | None = None
    # In-memory images [H, W, C] in [0, 1] (for embedded textures, e.g. glB),
    # used in preference to the corresponding ``*_texture`` file path.
    diffuse_image: torch.Tensor | None = None
    normal_texture: str | None = None
    # Tangent-space normal map [H, W, 3] in [0, 1] (rgb-encoded; = 2*n-1).
    normal_image: torch.Tensor | None = None
    metallic_factor: float = 0.0
    roughness_factor: float = 1.0
    metallic_roughness_texture: str | None = None
    # glTF-packed metallic-roughness map [H, W, 3]: G = roughness, B = metallic.
    metallic_roughness_image: torch.Tensor | None = None
    emissive: tuple[float, float, float] = (0.0, 0.0, 0.0)
    emissive_image: torch.Tensor | None = None
    reflectivity: float = 0.0
    opacity: float = 1.0
    refractive_index: float = 0.0


@dataclass
class SkinData:
    """Linear-blend-skinning data for a mesh. Consumed by the skeletal
    animation phase (baked to per-frame vertex positions); ignored by static
    import.
    """

    # Names of the bones (indices into these are what `weights` refers to).
    bone_names: list[str] = field(default_factory=list)
    # Bind-pose inverse transform per bone, [B, 4, 4].
    inverse_bind_matrices: torch.Tensor | None = None
    # Per-vertex bone influences, both [V, K] (K = max influences per vertex):
    # `bone_indices` indexes `bone_names`, `weights` sums to ~1 per vertex.
    bone_indices: torch.Tensor | None = None
    weights: torch.Tensor | None = None


@dataclass
class MeshData:
    """One triangulated mesh: per-vertex arrays plus triangle indices."""

    vertices: torch.Tensor  # [V, 3]
    faces: torch.Tensor  # [F, 3] int
    normals: torch.Tensor | None = None  # [V, 3]
    uvs: torch.Tensor | None = None  # [V, 2]
    vertex_colors: torch.Tensor | None = None  # [V, 4] or [V, 5]
    material_index: int = -1
    name: str = ""
    skin: SkinData | None = None
    # Morph / blend-shape targets: each a full [V, 3] vertex-position set.
    morph_targets: list[torch.Tensor] = field(default_factory=list)
    morph_names: list[str] = field(default_factory=list)


@dataclass
class NodeData:
    """A node in the model's scene graph. ``transform`` is the node-local
    ``4x4`` transform (relative to its parent).
    """

    name: str = ""
    transform: torch.Tensor | None = None  # [4, 4], local to parent
    parent: int = -1  # index into SceneData.nodes
    mesh_indices: list[int] = field(default_factory=list)


@dataclass
class NodeAnimation:
    """Keyframed local transform track for a single node (rigid / node
    animation, and the per-bone tracks skeletal animation reads). Times are in
    seconds; each channel may key independently.
    """

    node_name: str = ""
    position_times: torch.Tensor | None = None  # [Kp]
    positions: torch.Tensor | None = None  # [Kp, 3]
    rotation_times: torch.Tensor | None = None  # [Kr]
    rotations: torch.Tensor | None = None  # [Kr, 4] quaternion (x,y,z,w)
    scaling_times: torch.Tensor | None = None  # [Ks]
    scalings: torch.Tensor | None = None  # [Ks, 3]


@dataclass
class AnimationData:
    """A named animation clip: a set of per-node keyframe tracks."""

    name: str = ""
    duration: float = 0.0  # seconds
    channels: list[NodeAnimation] = field(default_factory=list)


@dataclass
class SceneData:
    """A whole imported model."""

    meshes: list[MeshData] = field(default_factory=list)
    materials: list[MaterialData] = field(default_factory=list)
    nodes: list[NodeData] = field(default_factory=list)
    animations: list[AnimationData] = field(default_factory=list)
    # Units-per-metre / global scale hint from the file, if any.
    unit_scale: float = 1.0
    source_path: str = ""

    def material_for(self, mesh: MeshData) -> MaterialData | None:
        if 0 <= mesh.material_index < len(self.materials):
            return self.materials[mesh.material_index]
        return None
