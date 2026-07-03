"""trimesh-backed importer for glTF / glB (and other trimesh-supported
formats): parse a model file into a :class:`SceneData`.

trimesh is pure-Python (plus numpy) and needs no native library, so unlike the
pyassimp/FBX path this works out of the box. It loads geometry, per-vertex
normals / UVs / colours, the node-instance transforms, and PBR materials
(including *embedded* base-colour textures, which are handed to the IR as
in-memory images).

The output is the same backend-independent :class:`SceneData` the FBX loader
produces, so :class:`~algan.mobs.fbx.model_mob.ThreeDModelMob` builds both
through one shared path.
"""
from __future__ import annotations

import os

import numpy as np
import torch

from algan.mobs.fbx.scene_data import MaterialData, MeshData, NodeData, SceneData


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
    """glTF colour factors may be uint8 [0, 255] or float [0, 1]; return a
    4-tuple in [0, 1]."""
    if color is None:
        return (1.0, 1.0, 1.0, 1.0)
    c = [float(x) for x in np.asarray(color).reshape(-1)[:4]]
    while len(c) < 4:
        c.append(1.0)
    if max(c) > 1.0 + 1e-4:
        c = [x / 255.0 for x in c]
    return tuple(c)


def _convert_material(visual):
    """trimesh visual -> :class:`MaterialData` (PBR base colour + embedded
    diffuse texture where present)."""
    material = getattr(visual, "material", None)
    if material is None:
        return MaterialData()
    base = _normalize_color(
        getattr(material, "baseColorFactor", None)
        if getattr(material, "baseColorFactor", None) is not None
        else getattr(material, "main_color", None))
    diffuse_image = None
    tex = getattr(material, "baseColorTexture", None)
    if tex is None:
        tex = getattr(material, "image", None)  # some trimesh materials
    if tex is not None:
        try:
            diffuse_image = _image_to_float_hwc(tex)
        except Exception:
            diffuse_image = None
    metallic = getattr(material, "metallicFactor", None)
    roughness = getattr(material, "roughnessFactor", None)
    emissive = getattr(material, "emissiveFactor", None)
    return MaterialData(
        name=getattr(material, "name", "") or "",
        base_color=base,
        diffuse_image=diffuse_image,
        metallic_factor=float(metallic) if metallic is not None else 0.0,
        roughness_factor=float(roughness) if roughness is not None else 1.0,
        emissive=(tuple(float(x) for x in np.asarray(emissive).reshape(-1)[:3])
                  if emissive is not None else (0.0, 0.0, 0.0)),
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
        uvs = _to_tensor(uv)[..., :2].contiguous()
        # glTF UV origin is the *top-left* (v increases downward); the engine's
        # texture sampler is v-up (see image_to_texture_map), so flip v.
        uvs = uvs.clone()
        uvs[:, 1] = 1.0 - uvs[:, 1]

    vertex_colors = None
    vc = getattr(visual, "vertex_colors", None) if visual is not None else None
    if (uvs is None and vc is not None and len(vc) == vertices.shape[0]):
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


def load_scene(file_path):
    """Parse ``file_path`` (glTF/glB/OBJ/PLY/... anything trimesh reads) into a
    :class:`SceneData`. Node-instance world transforms from the scene graph are
    baked into flat root nodes (parent -1), which the shared model builder then
    applies to the geometry."""
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
            nodes.append(NodeData(name=meshes[i].name, parent=-1,
                                  mesh_indices=[i]))

    return SceneData(
        meshes=meshes,
        materials=materials,
        nodes=nodes,
        source_path=os.path.abspath(file_path),
    )
