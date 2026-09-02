"""Tests for the 3-D model import path (``Model3D`` + ``TriangleMesh``).

Run directly: python tests/unit_tests/test_fbx_import.py

The importer is split so the mob-building logic is testable without any parser
backend: the backend-independent :class:`SceneData` IR is hand-built here and
fed to :class:`Model3D` directly (the same entry alternative backends
use). Layers checked:

1. Pure logic (always runs): world-transform composition/baking and mob-tree
   construction from a hand-built ``SceneData``.
2. A real glB import (``tests/dragon_mesh.glb`` via trimesh, no native lib) --
   parse + build assertions; skipped if the asset is absent.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from algan import SceneManager
from algan.mobs.three_d_models import Model3D, TriangleMesh
from algan.mobs.three_d_models.mesh import image_to_normal_map, image_to_texture_map
from algan.mobs.three_d_models.scene_data import (
    AnimationData,
    MaterialData,
    MeshData,
    NodeAnimation,
    NodeData,
    SceneData,
)
from algan.rendering.shaders.material_shaders import standard_shader

DRAGON_GLB = Path(__file__).resolve().parents[1] / "dragon_mesh.glb"
FULL_RENDER_GLB = (
    Path(__file__).resolve().parents[1]
    / "full_renders"
    / "assets"
    / "textured_icosphere.glb"
)


# --- geometry / IR builders -------------------------------------------------
def _checker_texture(n=32):
    """Asymmetric texture: RED in the top-left image quadrant, a green/blue
    checker elsewhere -- asymmetry lets the render test verify UV orientation.
    """
    img = torch.zeros(n, n, 4)
    img[..., 3] = 1.0
    for r in range(n):
        for c in range(n):
            if r < n // 2 and c < n // 2:
                img[r, c, 0] = 1.0
            elif (r // 4 + c // 4) % 2 == 0:
                img[r, c, 1] = 1.0
            else:
                img[r, c, 2] = 1.0
    return img


def _quad():
    v = torch.tensor(
        [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], dtype=torch.float32
    )
    uv = torch.tensor([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=torch.float32)
    n = torch.tensor([[0, 0, 1]] * 4, dtype=torch.float32)
    f = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long)
    return v, f, n, uv


def _octahedron():
    v = torch.tensor(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
        dtype=torch.float32,
    )
    n = torch.nn.functional.normalize(v, dim=-1)
    f = torch.tensor(
        [
            [0, 2, 4],
            [2, 1, 4],
            [1, 3, 4],
            [3, 0, 4],
            [2, 0, 5],
            [1, 2, 5],
            [3, 1, 5],
            [0, 3, 5],
        ],
        dtype=torch.long,
    )
    return v, f, n


def _translate(tx):
    m = torch.eye(4)
    m[0, 3] = tx
    return m


def _two_mesh_scene():
    qv, qf, qn, quv = _quad()
    ov, of, on = _octahedron()
    meshes = [
        MeshData(
            vertices=qv, faces=qf, normals=qn, uvs=quv, material_index=0, name="quad"
        ),
        MeshData(vertices=ov, faces=of, normals=on, material_index=1, name="octa"),
    ]
    materials = [
        MaterialData(name="tex", diffuse_texture="__CHECKER__"),
        MaterialData(name="red", base_color=(0.9, 0.2, 0.2, 1.0)),
    ]
    nodes = [
        NodeData(name="root", transform=torch.eye(4), parent=-1),
        NodeData(
            name="quad_node", transform=_translate(-1.6), parent=0, mesh_indices=[0]
        ),
        NodeData(
            name="octa_node", transform=_translate(1.6), parent=0, mesh_indices=[1]
        ),
    ]
    return SceneData(
        meshes=meshes, materials=materials, nodes=nodes, source_path="synthetic"
    )


class _CheckerModel(Model3D):
    """Model3D resolving the synthetic ``__CHECKER__`` texture in-memory."""

    def _load_texture(self, path, device):
        if path == "__CHECKER__":
            return image_to_texture_map(_checker_texture()).to(device)
        return super()._load_texture(path, device)


# --- pure-logic tests (no rendering) ----------------------------------------
def test_world_transform_baking():
    """Node-local transforms compose down the hierarchy and bake into
    world-space vertices.
    """
    from algan.mobs.three_d_models.model_mob import (
        _compose_world_transforms,
        _transform_points,
    )

    parent = _translate(1.0)
    child = _translate(2.0)
    child[1, 3] = 0.5  # +y
    nodes = [
        NodeData(name="root", transform=parent, parent=-1),
        NodeData(name="child", transform=child, parent=0),
    ]
    world = _compose_world_transforms(nodes, torch.device("cpu"))
    # child world translation = parent (x=1) composed with child (x=2, y=0.5).
    assert torch.allclose(world[1][:3, 3], torch.tensor([3.0, 0.5, 0.0]))

    pts = torch.zeros(1, 3)
    baked = _transform_points(pts, world[1])
    assert torch.allclose(baked[0], torch.tensor([3.0, 0.5, 0.0]))


def test_model_builds_mesh_tree():
    SceneManager.reset()
    model = _CheckerModel(scene_data=_two_mesh_scene())
    # One TriangleMesh per mesh instance, both registered as children.
    assert len(model.mesh_mobs) == 2
    assert all(isinstance(m, TriangleMesh) for m in model.mesh_mobs)
    assert all(m in model.children for m in model.mesh_mobs)

    # Node transforms baked: quad centered near x=-1.6, octahedron near x=+1.6.
    quad, octa = model.mesh_mobs
    quad_x = quad.grid.location.reshape(-1, 3)[:, 0].mean().item()
    octa_x = octa.grid.location.reshape(-1, 3)[:, 0].mean().item()
    assert quad_x < -1.0 < 1.0 < octa_x


def test_triangle_mesh_requires_uvs_for_texture():
    SceneManager.reset()
    v, f, n, uv = _quad()
    tex = image_to_texture_map(_checker_texture())
    with pytest.raises(ValueError):
        TriangleMesh(vertices=v, faces=f, normals=n, texture=tex)  # no uvs


def test_assimp_loader_missing_file():
    from algan.mobs.three_d_models.assimp_loader import load_scene

    with pytest.raises(FileNotFoundError):
        load_scene("does_not_exist.fbx")


# --- Phase 2: PBR materials + normal maps + node access ---------------------
def test_normal_map_conversion():
    """glTF rgb normal map -> engine [-1, 1] tangent map, transposed/flipped
    like the colour texture with the green axis flipped.
    """
    img = torch.zeros(4, 6, 3)  # [H, W, 3] in [0, 1]
    img[..., 2] = 1.0  # flat normal rgb (0,0,1) -> vector (-1,-1,+1)
    img[..., 1] = 1.0  # green = 1 -> y = +1, flipped to -1
    nm = image_to_normal_map(img)
    assert nm.shape == (6, 4, 3)  # [W, H, 3] (transposed)
    assert nm[..., 2].min() > 0.99  # z stays +1
    assert nm[..., 1].max() < -0.99  # green flipped: y = -(2*1-1) = -1


def _material_scene_with_maps():
    """Two-triangle quad whose material carries an embedded metallic-roughness
    map and a normal map, for wiring assertions (no render).
    """
    v, f, n, uv = _quad()
    mr = torch.zeros(8, 8, 3)
    mr[..., 1] = 0.5  # roughness channel (G)
    mr[..., 2] = 0.25  # metallic channel (B)
    normal = torch.zeros(8, 8, 3)
    normal[..., 2] = 1.0  # flat normal map
    mat = MaterialData(
        name="pbr",
        base_color=(0.8, 0.7, 0.6, 1.0),
        metallic_factor=1.0,
        roughness_factor=1.0,
        metallic_roughness_image=mr,
        normal_image=normal,
    )
    mesh = MeshData(
        vertices=v, faces=f, normals=n, uvs=uv, material_index=0, name="quad"
    )
    node = NodeData(name="quad_node", parent=-1, mesh_indices=[0])
    return SceneData(
        meshes=[mesh], materials=[mat], nodes=[node], source_path="synthetic"
    )


def test_pbr_and_normal_map_wiring():
    SceneManager.reset()
    model = Model3D(scene_data=_material_scene_with_maps())
    mesh = model.mesh_mobs[0]
    # Normal map wired through to the mesh.
    assert mesh.normal_texture_map is not None
    assert mesh.normal_texture_map.shape[-1] == 3
    # PBR material -> standard (GGX) shader with metalness/roughness params.
    assert mesh.shader is standard_shader
    params = mesh.grid.get_shader_params()
    assert "metalness" in params
    assert "roughness" in params
    # Flat metalness/roughness = factor * mean(map channel): 1*0.25, 1*0.5.
    assert abs(float(mesh.grid.metalness.reshape(-1)[0]) - 0.25) < 1e-3
    assert abs(float(mesh.grid.roughness.reshape(-1)[0]) - 0.5) < 1e-3


def test_normal_maps_and_pbr_can_be_disabled():
    SceneManager.reset()
    model = Model3D(
        scene_data=_material_scene_with_maps(), normal_maps=False, pbr_materials=False
    )
    mesh = model.mesh_mobs[0]
    assert mesh.normal_texture_map is None
    assert mesh.shader is not standard_shader


def test_node_part_access():
    SceneManager.reset()
    model = _CheckerModel(scene_data=_two_mesh_scene())
    assert set(model.node_names) == {"quad_node", "octa_node"}
    quad_part = model.get_part("quad_node")
    assert isinstance(quad_part, TriangleMesh)
    assert quad_part in model.mesh_mobs
    with pytest.raises(KeyError):
        model.get_part("nonexistent")


# --- Phase 3: rigid node-keyframe animation ---------------------------------
def test_animation_math_roundtrips():
    """Quaternion<->matrix, TRS decompose/recompose and slerp midpoint."""
    from algan.mobs.three_d_models import animation as anim

    q = torch.tensor([0.0, 0.7071068, 0.0, 0.7071068])  # 90 deg about +Y
    R = anim.quaternion_to_matrix(q)
    # +90 about Y maps (x,y,z) -> (z, y, -x).
    v = torch.tensor([1.0, 2.0, 3.0])
    assert torch.allclose(R @ v, torch.tensor([3.0, 2.0, -1.0]), atol=1e-5)
    q2 = anim.matrix_to_quaternion(R)
    assert torch.allclose(anim.quaternion_to_matrix(q2), R, atol=1e-5)

    # Recompose a T,R,S matrix and get the same components back.
    T = torch.tensor([1.0, -2.0, 3.0])
    S = torch.tensor([2.0, 0.5, 1.5])
    M = anim.compose_trs(T, q, S)
    t2, r2, s2 = anim.decompose_trs(M)
    assert torch.allclose(t2, T, atol=1e-5)
    assert torch.allclose(s2, S, atol=1e-5)
    assert torch.allclose(
        anim.quaternion_to_matrix(r2), anim.quaternion_to_matrix(q), atol=1e-5
    )

    # slerp halfway between identity and 90-about-Y is 45-about-Y.
    times = torch.tensor([0.0, 1.0])
    quats = torch.stack([torch.tensor([0.0, 0.0, 0.0, 1.0]), q])
    half = anim.sample_quaternion_track(times, quats, 0.5)
    Rh = anim.quaternion_to_matrix(half)
    c = 0.70710678
    assert torch.allclose(
        Rh @ torch.tensor([1.0, 0.0, 0.0]), torch.tensor([c, 0.0, -c]), atol=1e-4
    )


def _triangle_mesh(v=None):
    if v is None:
        v = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32)
    n = torch.nn.functional.normalize(v, dim=-1)
    f = torch.tensor([[0, 1, 2]], dtype=torch.long)
    return MeshData(vertices=v, faces=f, normals=n, name="tri")


def _spin_translate_scene():
    """Single triangle on a node that rotates 90 deg about +Y and translates to
    x=+2 over one second.
    """
    times = torch.tensor([0.0, 1.0])
    quats = torch.stack(
        [
            torch.tensor([0.0, 0.0, 0.0, 1.0]),
            torch.tensor([0.0, 0.7071068, 0.0, 0.7071068]),
        ]
    )
    positions = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    channel = NodeAnimation(
        node_name="spin",
        position_times=times,
        positions=positions,
        rotation_times=times,
        rotations=quats,
    )
    clip = AnimationData(name="clip", runtime=1.0, channels=[channel])
    node = NodeData(name="spin", transform=torch.eye(4), parent=-1, mesh_indices=[0])
    return SceneData(
        meshes=[_triangle_mesh()],
        materials=[MaterialData()],
        nodes=[node],
        animations=[clip],
    )


def test_bake_rigid_node_animation():
    """Baked per-frame corners match the analytic rotate-then-translate pose."""
    SceneManager.reset()
    model = Model3D(scene_data=_spin_translate_scene())
    times, corners = model.precompute_animation(times=[0.0, 0.5, 1.0])
    mob = model.mesh_mobs[0]
    baked = corners[mob]  # [3, 3F, 3]
    v0 = torch.tensor([1.0, 0.0, 0.0])  # first triangle corner
    # t=0: rest pose (identity, no translation).
    assert torch.allclose(baked[0, 0], v0, atol=1e-5)
    # t=1: R_y(90) v + (2,0,0) = (0,0,-1)+(2,0,0).
    assert torch.allclose(baked[2, 0], torch.tensor([2.0, 0.0, -1.0]), atol=1e-4)
    # t=0.5: R_y(45) v + (1,0,0).
    c = 0.70710678
    assert torch.allclose(baked[1, 0], torch.tensor([1.0 + c, 0.0, -c]), atol=1e-4)
    assert model.animation_names == ["clip"]


def test_bake_hierarchy_animation():
    """A mesh under an animated parent inherits the parent's motion."""
    SceneManager.reset()
    times = torch.tensor([0.0, 1.0])
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    clip = AnimationData(
        name="c",
        runtime=1.0,
        channels=[
            NodeAnimation(node_name="root", position_times=times, positions=positions)
        ],
    )
    nodes = [
        NodeData(name="root", transform=torch.eye(4), parent=-1),
        NodeData(name="child", transform=torch.eye(4), parent=0, mesh_indices=[0]),
    ]
    scene = SceneData(
        meshes=[_triangle_mesh()],
        materials=[MaterialData()],
        nodes=nodes,
        animations=[clip],
    )
    model = Model3D(scene_data=scene)
    _, corners = model.precompute_animation(times=[0.0, 1.0])
    baked = corners[model.mesh_mobs[0]]
    v0 = torch.tensor([1.0, 0.0, 0.0])
    assert torch.allclose(baked[0, 0], v0, atol=1e-5)
    assert torch.allclose(baked[1, 0], v0 + torch.tensor([0.0, 3.0, 0.0]), atol=1e-5)


def test_play_animation_sets_recompute_normals():
    """Playing an animation drives the corner geometry and switches meshes to
    per-frame smooth-normal recomputation.
    """
    SceneManager.reset()
    with torch.inference_mode():
        model = Model3D(scene_data=_spin_translate_scene()).spawn()
        model.play_animation(runtime=1.0)
    assert model.mesh_mobs[0].recompute_normals is True


def _write_animated_glb(path):
    """Author a minimal animated glB (one triangle on a node spinning 90 deg
    about +Y over 1s) with pygltflib, to exercise the real loader path.
    """
    import pygltflib as g

    verts = np.array([[-0.5, -0.5, 0], [0.5, -0.5, 0], [0.0, 0.5, 0]], dtype=np.float32)
    idx = np.array([0, 1, 2], dtype=np.uint16)
    times = np.array([0.0, 1.0], dtype=np.float32)
    rots = np.array([[0, 0, 0, 1], [0, 0.7071068, 0, 0.7071068]], dtype=np.float32)

    pos_b = verts.tobytes()
    idx_b = idx.tobytes()
    idx_pad = (4 - len(idx_b) % 4) % 4
    times_b = times.tobytes()
    rot_b = rots.tobytes()
    blob = pos_b + idx_b + b"\x00" * idx_pad + times_b + rot_b
    o_pos, o_idx = 0, len(pos_b)
    o_times = o_idx + len(idx_b) + idx_pad
    o_rot = o_times + len(times_b)

    gltf = g.GLTF2(
        scene=0,
        scenes=[g.Scene(nodes=[0])],
        nodes=[g.Node(mesh=0, name="spinner")],
        meshes=[
            g.Mesh(
                primitives=[g.Primitive(attributes=g.Attributes(POSITION=0), indices=1)]
            )
        ],
        accessors=[
            g.Accessor(
                bufferView=0,
                componentType=5126,
                count=3,
                type="VEC3",
                min=verts.min(0).tolist(),
                max=verts.max(0).tolist(),
            ),
            g.Accessor(bufferView=1, componentType=5123, count=3, type="SCALAR"),
            g.Accessor(
                bufferView=2,
                componentType=5126,
                count=2,
                type="SCALAR",
                min=[0.0],
                max=[1.0],
            ),
            g.Accessor(bufferView=3, componentType=5126, count=2, type="VEC4"),
        ],
        bufferViews=[
            g.BufferView(buffer=0, byteOffset=o_pos, byteLength=len(pos_b)),
            g.BufferView(buffer=0, byteOffset=o_idx, byteLength=len(idx_b)),
            g.BufferView(buffer=0, byteOffset=o_times, byteLength=len(times_b)),
            g.BufferView(buffer=0, byteOffset=o_rot, byteLength=len(rot_b)),
        ],
        buffers=[g.Buffer(byteLength=len(blob))],
        animations=[
            g.Animation(
                name="spin",
                samplers=[
                    g.AnimationSampler(input=2, output=3, interpolation="LINEAR")
                ],
                channels=[
                    g.AnimationChannel(
                        sampler=0,
                        target=g.AnimationChannelTarget(node=0, path="rotation"),
                    )
                ],
            )
        ],
    )
    gltf.set_binary_blob(blob)
    gltf.save_binary(path)


def test_glb_animation_roundtrip(tmp_path):
    """Author, load and bake an animated glB through the real trimesh+pygltflib
    loader path: the clip is recovered and the pose is correct.
    """
    pytest.importorskip("pygltflib")
    SceneManager.reset()
    path = tmp_path / "spinner.glb"
    _write_animated_glb(str(path))

    model = Model3D(str(path))
    assert len(model.animations) == 1
    assert model.animations[0].name == "spin"
    assert "spinner" in model.node_names
    _, corners = model.precompute_animation(times=[0.0, 1.0])
    baked = corners[model.mesh_mobs[0]]
    # A corner at +x rotates 90 about Y to -z; check the pose moved as expected.
    rest = baked[0].reshape(-1, 3)
    end = baked[1].reshape(-1, 3)
    assert not torch.allclose(rest, end, atol=1e-3)  # motion happened
    # Rotation preserves distance from the Y axis for every corner.
    r_rest = rest[:, [0, 2]].norm(dim=-1)
    r_end = end[:, [0, 2]].norm(dim=-1)
    assert torch.allclose(r_rest.sort().values, r_end.sort().values, atol=1e-4)


# --- real glB import (trimesh, no native library) ---------------------------
def _skip_if_no_dragon():
    if not DRAGON_GLB.exists():
        pytest.skip(f"{DRAGON_GLB} not present")


def test_glb_load_and_build():
    """Parse and build the textured dragon glB via trimesh (embedded texture,
    per-vertex normals/UVs, node-instance transform).
    """
    _skip_if_no_dragon()
    SceneManager.reset()
    model = Model3D(str(DRAGON_GLB), fit_to_size=2.0)
    assert len(model.mesh_mobs) >= 1
    mesh = model.mesh_mobs[0]
    assert mesh.num_triangles > 1000  # a real, detailed mesh
    assert mesh.texture_map is not None  # embedded diffuse texture
    assert mesh.texture_map.shape[-1] == 5  # engine colour layout
    assert mesh.corner_normals is not None  # smooth normals
    # Phase 2: the dragon carries a normal map + PBR material.
    assert mesh.normal_texture_map is not None
    assert mesh.normal_texture_map.shape[-1] == 3
    assert mesh.shader is standard_shader
    # normalize() fit the model to the target box.
    loc = torch.cat([m.grid.location.reshape(-1, 3) for m in model.mesh_mobs])
    diag = (loc.amax(0) - loc.amin(0)).norm().item()
    assert abs(diag - 2.0) < 1e-2


def test_full_render_glb_fixture_retains_pbr_maps():
    SceneManager.reset()
    model = Model3D(
        str(FULL_RENDER_GLB),
        fit_to_size=2.0,
    )
    mesh = model.mesh_mobs[0]

    assert mesh.num_triangles == 320
    assert mesh.texture_map is not None
    assert mesh.normal_texture_map is not None
    assert mesh.shader is standard_shader
