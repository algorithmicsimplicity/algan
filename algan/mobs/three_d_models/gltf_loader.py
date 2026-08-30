"""trimesh-backed importer for glTF / glB (and other trimesh-supported
formats): parse a model file into a
:class:`~algan.mobs.three_d_models.scene_data.SceneData`.

trimesh is pure-Python (plus numpy) and needs no native library, so unlike the
pyassimp/FBX path this works out of the box. It loads geometry, per-vertex
normals / UVs / colors, the node-instance transforms, and PBR materials
(including *embedded* base-color textures, which are handed to the IR as
in-memory images).

The output is the same backend-independent :class:`SceneData` the FBX loader
produces, so :class:`~algan.mobs.three_d_models.model_mob.Model3D` builds both
through one shared path.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from algan.mobs.three_d_models import animation as _anim
from algan.mobs.three_d_models.scene_data import (
    AnimationData,
    MaterialData,
    MeshData,
    NodeAnimation,
    NodeData,
    SceneData,
)


def _to_tensor(array, dtype=torch.float32):
    if array is None:
        return None
    a = np.asarray(array)
    if a.size == 0:
        return None
    # copy=True: trimesh often hands back read-only views of its cached arrays.
    return torch.as_tensor(np.array(a, copy=True), dtype=dtype)


def _image_to_float_hwc(image):
    """PIL image (or array) -> torch ``[H, W, C]`` float in ``[0, 1]``."""
    arr = np.asarray(image)
    if arr.ndim == 2:  # grayscale
        arr = arr[..., None]
    # copy=True: PIL/trimesh arrays are commonly read-only views.
    t = torch.as_tensor(np.array(arr, copy=True)).float()
    if t.dtype == torch.uint8 or t.max() > 1.0 + 1e-4:
        t = t / 255.0
    return t


def _normalize_color(color):
    """glTF color factors may be uint8 [0, 255] or float [0, 1]; return a
    4-tuple in [0, 1].
    """
    if color is None:
        return (1.0, 1.0, 1.0, 1.0)
    c = [float(x) for x in np.asarray(color).reshape(-1)[:4]]
    while len(c) < 4:
        c.append(1.0)
    if max(c) > 1.0 + 1e-4:
        c = [x / 255.0 for x in c]
    return tuple(c)


def _convert_material(visual):
    """trimesh visual -> :class:`MaterialData` (PBR base color + embedded
    diffuse texture where present).
    """
    material = getattr(visual, "material", None)
    if material is None:
        return MaterialData()
    base = _normalize_color(
        getattr(material, "baseColorFactor", None)
        if getattr(material, "baseColorFactor", None) is not None
        else getattr(material, "main_color", None)
    )

    def _img(attr):
        tex = getattr(material, attr, None)
        if tex is None:
            return None
        try:
            return _image_to_float_hwc(tex)
        except Exception:
            return None

    diffuse_image = _img("baseColorTexture")
    if diffuse_image is None:
        diffuse_image = _img("image")  # some trimesh materials
    metallic = getattr(material, "metallicFactor", None)
    roughness = getattr(material, "roughnessFactor", None)
    emissive = getattr(material, "emissiveFactor", None)
    return MaterialData(
        name=getattr(material, "name", "") or "",
        base_color=base,
        diffuse_image=diffuse_image,
        normal_image=_img("normalTexture"),
        metallic_roughness_image=_img("metallicRoughnessTexture"),
        emissive_image=_img("emissiveTexture"),
        metallic_factor=float(metallic) if metallic is not None else 0.0,
        roughness_factor=float(roughness) if roughness is not None else 1.0,
        emissive=(
            tuple(float(x) for x in np.asarray(emissive).reshape(-1)[:3])
            if emissive is not None
            else (0.0, 0.0, 0.0)
        ),
        opacity=base[3],
    )


def _convert_mesh(geom, material_index, name):
    vertices = _to_tensor(geom.vertices)
    faces = _to_tensor(getattr(geom, "faces", None), dtype=torch.int64)
    if vertices is None or faces is None:
        return None
    normals = None
    try:
        vn = geom.vertex_normals
        if vn is not None and len(vn) == len(geom.vertices):
            normals = _to_tensor(vn)
    except Exception:
        normals = None

    uvs = None
    visual = getattr(geom, "visual", None)
    uv = getattr(visual, "uv", None) if visual is not None else None
    if uv is not None and len(uv) == vertices.shape[0]:
        # trimesh already flips the v axis to be v-up when loading glTF, so we
        # don't need to flip it again.
        uvs = _to_tensor(uv)[..., :2].contiguous()

    vertex_colors = None
    vc = getattr(visual, "vertex_colors", None) if visual is not None else None
    if uvs is None and vc is not None and len(vc) == vertices.shape[0]:
        vertex_colors = _image_to_float_hwc(np.asarray(vc)[None])[0]

    return MeshData(
        vertices=vertices,
        faces=faces.view(-1, faces.shape[-1]),
        normals=normals,
        uvs=uvs,
        vertex_colors=vertex_colors,
        material_index=material_index,
        name=name or "",
    )


# --- animation / node-hierarchy parsing (best-effort, via pygltflib) --------
# trimesh discards glTF keyframe animation and flattens the node graph, so when
# a file carries animations we re-read the raw glTF with pygltflib to recover
# the true node hierarchy (needed to compose animated transforms) and the
# keyframe tracks. This path is additive: a file without animations is left
# exactly as the trimesh-only static import produced it. It is UNVALIDATED
# against real animated assets (no such asset on hand) -- like the FBX path.

_COMPONENT_DTYPE = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}
_TYPE_COMPONENTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}


def _accessor_array(gltf, accessor_idx, blob):
    """Read glTF accessor ``accessor_idx`` into a torch tensor ``[count, C]``
    (or ``[count]`` for SCALAR).
    """
    acc = gltf.accessors[accessor_idx]
    view = gltf.bufferViews[acc.bufferView]
    dtype = _COMPONENT_DTYPE[acc.componentType]
    ncomp = _TYPE_COMPONENTS[acc.type]
    start = (view.byteOffset or 0) + (acc.byteOffset or 0)
    count = acc.count * ncomp
    arr = np.frombuffer(blob, dtype=dtype, count=count, offset=start)
    arr = np.array(arr, copy=True)
    if ncomp > 1:
        arr = arr.reshape(acc.count, ncomp)
    t = torch.as_tensor(arr)
    if acc.componentType == 5126:
        t = t.float()
    return t


def _node_local_transform(node):
    """Local ``4x4`` transform of a glTF node (its ``matrix`` if present, else
    its TRS components).
    """
    if node.matrix:
        # glTF matrices are column-major; reshape row-major then transpose.
        m = torch.tensor(node.matrix, dtype=torch.float32).reshape(4, 4).T
        return m
    translation = node.translation or [0.0, 0.0, 0.0]
    rotation = node.rotation or [0.0, 0.0, 0.0, 1.0]
    scale = node.scale or [1.0, 1.0, 1.0]
    return _anim.compose_trs(translation, rotation, scale)


def _gltf_mesh_counts(gltf, blob):
    """Per glTF-mesh (vertex_count, face_count) from its first primitive, to map
    glTF meshes onto the trimesh-built :class:`MeshData` list by shape.
    """
    counts = []
    for mesh in gltf.meshes or []:
        vc = fc = -1
        if mesh.primitives:
            prim = mesh.primitives[0]
            pos = getattr(prim.attributes, "POSITION", None)
            if pos is not None:
                vc = gltf.accessors[pos].count
            if prim.indices is not None:
                fc = gltf.accessors[prim.indices].count // 3
        counts.append((vc, fc))
    return counts


def _map_gltf_meshes(gltf, meshes, blob):
    """Map each glTF mesh index to a unique built ``MeshData`` index by matching
    (vertex_count, face_count). Returns ``None`` if any glTF mesh is ambiguous
    or unmatched (caller then falls back to the flat static nodes).
    """
    our = [(m.vertices.shape[0], m.faces.shape[0]) for m in meshes]
    counts = _gltf_mesh_counts(gltf, blob)
    mapping = {}
    used = set()
    for gi, key in enumerate(counts):
        matches = [i for i, k in enumerate(our) if k == key and i not in used]
        if len(matches) != 1:
            return None
        mapping[gi] = matches[0]
        used.add(matches[0])
    return mapping


def _build_hierarchy(gltf, meshes, blob):
    """Recover the real node hierarchy from the glTF, mapping mesh-carrying
    nodes onto built ``MeshData`` indices. Returns an ordered ``list[NodeData]``
    with ``parent < child`` (so a single forward compose pass works), or
    ``None`` if the mesh mapping is ambiguous.
    """
    mesh_map = _map_gltf_meshes(gltf, meshes, blob)
    if mesh_map is None:
        return None
    gnodes = gltf.nodes or []
    parent = [-1] * len(gnodes)
    for i, n in enumerate(gnodes):
        for c in n.children or []:
            parent[c] = i
    # Depth-first order from the scene roots so parents precede children.
    roots = []
    if gltf.scenes and gltf.scene is not None:
        roots = list(gltf.scenes[gltf.scene].nodes or [])
    if not roots:
        roots = [i for i in range(len(gnodes)) if parent[i] < 0]
    order = []
    seen = set()
    stack = list(reversed(roots))
    while stack:
        i = stack.pop()
        if i in seen:
            continue
        seen.add(i)
        order.append(i)
        for c in reversed(gnodes[i].children or []):
            stack.append(c)
    remap = {g: o for o, g in enumerate(order)}
    out = []
    for g in order:
        n = gnodes[g]
        mesh_indices = (
            [mesh_map[n.mesh]] if n.mesh is not None and n.mesh in mesh_map else []
        )
        out.append(
            NodeData(
                name=n.name or f"node_{g}",
                transform=_node_local_transform(n),
                parent=remap[parent[g]] if parent[g] >= 0 else -1,
                mesh_indices=mesh_indices,
            )
        )
    return out


def _parse_animations(gltf, blob):
    """Parse glTF animation clips into ``list[AnimationData]`` keyed by node
    name (matching :class:`NodeData.name`).
    """
    gnodes = gltf.nodes or []

    def node_name(idx):
        if 0 <= idx < len(gnodes):
            return gnodes[idx].name or f"node_{idx}"
        return f"node_{idx}"

    clips = []
    for ai, anim in enumerate(gltf.animations or []):
        per_node = {}
        duration = 0.0
        for chan in anim.channels:
            target = chan.target
            if target is None or target.node is None:
                continue
            sampler = anim.samplers[chan.sampler]
            times = _accessor_array(gltf, sampler.input, blob).float().reshape(-1)
            values = _accessor_array(gltf, sampler.output, blob).float()
            duration = max(duration, float(times[-1]) if times.numel() else 0.0)
            na = per_node.setdefault(
                target.node, NodeAnimation(node_name=node_name(target.node))
            )
            path = target.path
            if path == "translation":
                na.position_times, na.positions = times, values.reshape(-1, 3)
            elif path == "rotation":
                na.rotation_times, na.rotations = times, values.reshape(-1, 4)
            elif path == "scale":
                na.scaling_times, na.scalings = times, values.reshape(-1, 3)
            # 'weights' (morph) is left for the morph phase.
        if per_node:
            clips.append(
                AnimationData(
                    name=anim.name or f"animation_{ai}",
                    duration=duration,
                    channels=list(per_node.values()),
                )
            )
    return clips


def _augment_with_animations(file_path, meshes, static_nodes):
    """When ``file_path`` carries glTF animations, return ``(nodes, animations)``
    with the real hierarchy and parsed clips; otherwise return the static nodes
    and no animations. Fully defensive: any failure degrades to static.
    """
    try:
        from pygltflib import GLTF2

        gltf = GLTF2().load(file_path)
        if not gltf.animations:
            return static_nodes, []
        blob = gltf.binary_blob()
        if blob is None and gltf.buffers:
            # .gltf with an external/base64 buffer.
            blob = gltf.get_data_from_buffer_uri(gltf.buffers[0].uri)
        nodes = _build_hierarchy(gltf, meshes, blob)
        animations = _parse_animations(gltf, blob)
        if nodes is None:
            # Couldn't map meshes to the real hierarchy; keep flat static nodes
            # (animations still bind to directly-animated leaf nodes by name).
            return static_nodes, animations
        return nodes, animations
    except Exception:
        return static_nodes, []


def load_scene(file_path):
    """Parse ``file_path`` (glTF/glB/OBJ/PLY/... anything trimesh reads) into a
    :class:`SceneData`. Node-instance world transforms from the scene graph are
    baked into flat root nodes (parent -1), which the shared model builder then
    applies to the geometry.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)
    import trimesh

    # process=False keeps the authored vertex/UV/normal arrays aligned (no
    # vertex merging), which matters for textured meshes.
    loaded = trimesh.load(file_path, process=False, force="scene")

    meshes: list[MeshData] = []
    materials: list[MaterialData] = []
    nodes: list[NodeData] = []
    geom_to_mesh: dict[str, int] = {}

    for name, geom in loaded.geometry.items():
        if not hasattr(geom, "vertices") or not hasattr(geom, "faces"):
            continue
        material_index = len(materials)
        materials.append(_convert_material(getattr(geom, "visual", None)))
        md = _convert_mesh(geom, material_index, name)
        if md is None:
            materials.pop()
            continue
        geom_to_mesh[name] = len(meshes)
        meshes.append(md)

    # One node per geometry instance, carrying its (already world-space)
    # transform from the scene graph.
    graph = loaded.graph
    for node_name in getattr(graph, "nodes_geometry", []):
        try:
            transform, geom_name = graph.get(node_name)
        except Exception:
            continue
        if geom_name not in geom_to_mesh:
            continue
        nodes.append(
            NodeData(
                name=str(node_name),
                transform=_to_tensor(transform),
                parent=-1,
                mesh_indices=[geom_to_mesh[geom_name]],
            )
        )

    # Fallback: no graph instances (single mesh) -> identity-placed nodes.
    if not nodes:
        for i in range(len(meshes)):
            nodes.append(NodeData(name=meshes[i].name, parent=-1, mesh_indices=[i]))

    # If the file carries keyframe animation, recover the real hierarchy + clips
    # (trimesh drops both). No-op for static files, so the static path is
    # untouched.
    nodes, animations = _augment_with_animations(file_path, meshes, nodes)

    return SceneData(
        meshes=meshes,
        materials=materials,
        nodes=nodes,
        animations=animations,
        source_path=os.path.abspath(file_path),
    )
