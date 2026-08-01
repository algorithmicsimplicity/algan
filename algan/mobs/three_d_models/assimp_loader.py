"""pyassimp-backed importer: parse a model file into a
:class:`~algan.mobs.three_d_models.scene_data.SceneData`.

pyassimp is a thin ``ctypes`` wrapper around the native Open Asset Import
Library (``assimp``). The Python package (``pip install pyassimp``) does **not**
ship that native library, so a clear, actionable error is raised if it is
missing rather than a bare ``ctypes`` failure. On Windows install the native
DLL (e.g. ``conda install -c conda-forge assimp`` and put ``assimp*.dll`` on the
PATH, or drop it beside the app); on Linux/mac install ``libassimp`` from the
system package manager.

Only the static-geometry portion of the IR is required by the current import
phase; bones and animation channels are extracted here too so the later
skeletal / morph phases have their inputs without a loader change.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from algan.mobs.three_d_models.scene_data import (
    AnimationData,
    MaterialData,
    MeshData,
    NodeAnimation,
    NodeData,
    SceneData,
    SkinData,
)

_INSTALL_HINT = (
    "FBX/mesh import needs the native 'assimp' library, which the pyassimp "
    "Python package does not bundle.\n"
    "  * pip install pyassimp   (Python bindings)\n"
    "  * then install the native library:\n"
    "      - conda:   conda install -c conda-forge assimp\n"
    "      - Windows: put assimp*.dll on PATH (or beside the executable)\n"
    "      - Linux:   apt install libassimp5   (or your distro's package)\n"
    "      - macOS:   brew install assimp\n"
)


def _require_pyassimp():
    try:
        import pyassimp
        import pyassimp.postprocess as pp
    except ImportError as e:
        raise ImportError("pyassimp is not installed.\n" + _INSTALL_HINT) from e
    except Exception as e:  # pragma: no cover - native lib lookup failure
        # pyassimp raises AssimpError (not ImportError) when the DLL is absent.
        raise ImportError(
            f"pyassimp could not load the native assimp library ({e}).\n"
            + _INSTALL_HINT
        ) from e
    return pyassimp, pp


def _default_flags(pp):
    """Post-process flags: triangulate everything, generate smooth normals
    where absent, and drop degenerate/duplicate data. Left UV-space as authored
    (:func:`image_to_texture_map` handles the engine's v-orientation), so no
    ``FlipUVs`` here.
    """
    return (
        pp.aiProcess_Triangulate
        | pp.aiProcess_GenSmoothNormals
        | pp.aiProcess_JoinIdenticalVertices
        | pp.aiProcess_ImproveCacheLocality
        | pp.aiProcess_RemoveRedundantMaterials
        | pp.aiProcess_FindInvalidData
        | pp.aiProcess_GenUVCoords
        | pp.aiProcess_SortByPType
    )


def _t(array, dtype=torch.float32):
    if array is None:
        return None
    a = np.asarray(array)
    if a.size == 0:
        return None
    return torch.as_tensor(np.ascontiguousarray(a), dtype=dtype)


def _material_string(props, *keys):
    for k in keys:
        v = props.get(k)
        if isinstance(v, (bytes, bytearray)):
            return v.decode("utf-8", "ignore")
        if isinstance(v, str):
            return v
    return None


def _material_tuple(props, key, n):
    v = props.get(key)
    if v is None:
        return None
    try:
        seq = list(v)
    except TypeError:
        seq = [v]
    if len(seq) < n:
        return None
    return tuple(float(x) for x in seq[:n])


def _parse_material(mat, base_dir):
    """assimp material -> :class:`MaterialData`. pyassimp exposes a material's
    properties as a plain ``dict`` keyed by short semantic names (e.g.
    ``'diffuse'``, ``'file'``); different assimp versions vary, so every lookup
    is defensive.
    """
    props = getattr(mat, "properties", {}) or {}
    diffuse = _material_tuple(props, "diffuse", 4) or _material_tuple(
        props, "diffuse", 3
    )
    if diffuse is not None and len(diffuse) == 3:
        diffuse = (*diffuse, 1.0)
    emissive = _material_tuple(props, "emissive", 3) or (0.0, 0.0, 0.0)
    opacity = props.get("opacity", 1.0)
    reflectivity = props.get("reflectivity", 0.0)
    refracti = props.get("refracti", 0.0)
    metallic = props.get("metallicFactor", props.get("reflectivity", 0.0))
    roughness = props.get("roughnessFactor", 1.0)

    def _resolve(name):
        if not name:
            return None
        name = name.replace("\\", os.sep).replace("/", os.sep)
        cand = name if os.path.isabs(name) else os.path.join(base_dir, name)
        return cand if os.path.exists(cand) else name

    # Texture file properties: pyassimp collapses the per-semantic '$tex.file'
    # into 'file'; keep it simple (diffuse) and fall through gracefully.
    diffuse_tex = _resolve(_material_string(props, "file"))
    return MaterialData(
        name=_material_string(props, "name") or "",
        base_color=diffuse or (1.0, 1.0, 1.0, 1.0),
        diffuse_texture=diffuse_tex,
        metallic_factor=float(metallic) if metallic is not None else 0.0,
        roughness_factor=float(roughness) if roughness is not None else 1.0,
        emissive=emissive,
        reflectivity=float(reflectivity) if reflectivity is not None else 0.0,
        opacity=float(opacity) if opacity is not None else 1.0,
        refractive_index=float(refracti) if refracti and refracti > 1.0 else 0.0,
    )


def _parse_mesh(mesh):
    vertices = _t(mesh.vertices)
    faces = _t(getattr(mesh, "faces", None), dtype=torch.int64)
    if vertices is None or faces is None:
        return None
    faces = faces.view(-1, faces.shape[-1])
    if faces.shape[-1] != 3:  # non-triangle primitives slipped through
        faces = faces[:, :3]

    normals = _t(getattr(mesh, "normals", None))
    uvs = None
    texcoords = getattr(mesh, "texturecoords", None)
    if texcoords is not None and len(texcoords) > 0:
        uv = _t(texcoords[0])
        if uv is not None:
            uvs = uv[..., :2].contiguous()
    colors = None
    vcolors = getattr(mesh, "colors", None)
    if vcolors is not None and len(vcolors) > 0:
        colors = _t(vcolors[0])

    skin = _parse_skin(mesh, vertices.shape[0])

    return MeshData(
        vertices=vertices,
        faces=faces,
        normals=normals,
        uvs=uvs,
        vertex_colors=colors,
        material_index=int(getattr(mesh, "materialindex", -1)),
        name=getattr(mesh, "name", "") or "",
        skin=skin,
    )


def _parse_skin(mesh, num_vertices):
    bones = getattr(mesh, "bones", None)
    if not bones:
        return None
    K = 4  # keep the 4 strongest influences per vertex (standard for LBS)
    names = []
    inv_bind = []
    # Accumulate (weight, bone) per vertex, then keep the top K.
    acc = [[] for _ in range(num_vertices)]
    for b, bone in enumerate(bones):
        names.append(getattr(bone, "name", f"bone_{b}") or f"bone_{b}")
        inv_bind.append(_t(getattr(bone, "offsetmatrix", None)))
        for w in getattr(bone, "weights", []) or []:
            vid = int(getattr(w, "vertexid", getattr(w, "vertex_id", 0)))
            if 0 <= vid < num_vertices:
                acc[vid].append((float(getattr(w, "weight", 0.0)), b))
    bone_indices = torch.zeros((num_vertices, K), dtype=torch.int64)
    weights = torch.zeros((num_vertices, K), dtype=torch.float32)
    for vid, infl in enumerate(acc):
        infl.sort(key=lambda x: -x[0])
        for j, (wgt, bidx) in enumerate(infl[:K]):
            bone_indices[vid, j] = bidx
            weights[vid, j] = wgt
        s = weights[vid].sum()
        if s > 1e-8:
            weights[vid] /= s
    inv_bind = [m for m in inv_bind if m is not None]
    inverse_bind = torch.stack(inv_bind) if len(inv_bind) == len(names) else None
    return SkinData(
        bone_names=names,
        inverse_bind_matrices=inverse_bind,
        bone_indices=bone_indices,
        weights=weights,
    )


def _parse_nodes(root):
    """Flatten the node tree to a list with parent indices (depth first)."""
    nodes: list[NodeData] = []

    def visit(node, parent_idx):
        idx = len(nodes)
        nodes.append(
            NodeData(
                name=getattr(node, "name", "") or "",
                transform=_t(getattr(node, "transformation", None)),
                parent=parent_idx,
                mesh_indices=[int(m) for m in _node_mesh_indices(node)],
            )
        )
        for child in getattr(node, "children", []) or []:
            visit(child, idx)

    if root is not None:
        visit(root, -1)
    return nodes


def _node_mesh_indices(node):
    meshes = getattr(node, "meshes", None)
    if meshes is None:
        return []
    out = []
    for m in meshes:
        # pyassimp gives either an int index or the mesh object itself.
        if isinstance(m, (int, np.integer)):
            out.append(int(m))
        else:
            idx = getattr(m, "index", None)
            if idx is not None:
                out.append(int(idx))
    return out


def _parse_animations(scene):
    out = []
    for anim in getattr(scene, "animations", []) or []:
        tps = getattr(anim, "tickspersecond", 0.0) or 25.0
        duration_ticks = getattr(anim, "duration", 0.0) or 0.0
        channels = []
        for ch in getattr(anim, "channels", []) or []:
            channels.append(_parse_channel(ch, tps))
        out.append(
            AnimationData(
                name=getattr(anim, "name", "") or "",
                duration=float(duration_ticks) / float(tps),
                channels=channels,
            )
        )
    return out


def _parse_channel(ch, tps):
    def keys(attr):
        arr = getattr(ch, attr, None)
        if arr is None or len(arr) == 0:
            return None, None
        times = torch.as_tensor([float(k.time) / tps for k in arr], dtype=torch.float32)
        vals = torch.as_tensor(
            np.stack([np.asarray(k.value) for k in arr]), dtype=torch.float32
        )
        return times, vals

    pt, pv = keys("positionkeys")
    rt, rv = keys("rotationkeys")
    st, sv = keys("scalingkeys")
    # assimp quaternions are (w, x, y, z); store as (x, y, z, w).
    if rv is not None and rv.shape[-1] == 4:
        rv = rv[:, [1, 2, 3, 0]]
    return NodeAnimation(
        node_name=getattr(ch, "nodename", getattr(ch, "node_name", "")) or "",
        position_times=pt,
        positions=pv,
        rotation_times=rt,
        rotations=rv,
        scaling_times=st,
        scalings=sv,
    )


def load_scene(file_path, extra_flags=0):
    """Parse ``file_path`` into a :class:`SceneData` using pyassimp.

    Raises :class:`ImportError` with install guidance if the native assimp
    library is unavailable, and :class:`FileNotFoundError` if the model file
    does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)
    pyassimp, pp = _require_pyassimp()
    base_dir = os.path.dirname(os.path.abspath(file_path))
    flags = _default_flags(pp) | extra_flags
    # pyassimp.load frees native memory on context exit, so pull everything we
    # need into torch tensors inside the `with` block.
    with pyassimp.load(file_path, processing=flags) as scene:
        materials = [
            _parse_material(m, base_dir) for m in getattr(scene, "materials", []) or []
        ]
        meshes = []
        for m in getattr(scene, "meshes", []) or []:
            md = _parse_mesh(m)
            if md is not None:
                meshes.append(md)
        nodes = _parse_nodes(getattr(scene, "rootnode", None))
        animations = _parse_animations(scene)
    return SceneData(
        meshes=meshes,
        materials=materials,
        nodes=nodes,
        animations=animations,
        source_path=os.path.abspath(file_path),
    )
