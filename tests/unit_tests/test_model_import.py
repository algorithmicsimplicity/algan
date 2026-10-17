"""Importing a 3-D model: ``Model3D``, the ``TriangleMesh`` it builds, and the
glTF reader behind them.

What is pinned here, none of it rendering a frame:

* clips -- the first by default, others by name, a clear error for an unknown
  name or a model with none -- and playback: rotation keys recorded on the
  timeline with the shading following the moving surface, and poses that stay
  where the model was placed, moved, turned or scaled to;
* placement -- ``location`` puts the model there, with or without
  ``fit_to_size``;
* a ``TriangleMesh``'s normals (flat, authored, and recomputed area-weighted
  from the current geometry) and the render primitive it hands the renderer:
  opacity folded into the colours, the texture map with its UVs and per-frame
  opacity, and shell ids split on shared edges;
* textures and normal maps -- the per-path cache, an embedded image's
  precedence over a file -- and the material each imported material becomes;
* the glTF reader: node matrices, scale keys, untargeted channels, files with
  no default scene, animation in a second buffer, malformed animation,
  non-triangle primitives, and the other formats trimesh reads.

Models are hand-built ``SceneData`` or tiny files written into ``tmp_path``
(glTF through pygltflib, OBJ as text, PNG through PIL); no binary fixture is
involved. Feature tests, outside the fast suite.
"""

from __future__ import annotations

import base64
import math
from types import SimpleNamespace

import numpy as np
import pygltflib as g
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from algan import OUT, RIGHT, UP, Off, Scene, Sync
from algan.errors import AlganConfigurationError, UnsupportedFeatureWarning
from algan.mobs.three_d_models import model_mob
from algan.mobs.three_d_models.gltf_loader import _convert_material
from algan.mobs.three_d_models.mesh import (
    TriangleMesh,
    image_to_normal_map,
    image_to_texture_map,
)
from algan.mobs.three_d_models.model_mob import Model3D
from algan.mobs.three_d_models.scene_data import (
    AnimationData,
    MaterialData,
    MeshData,
    NodeAnimation,
    NodeData,
    SceneData,
)
from algan.rendering.shaders.materials import (
    MeshPhysicalMaterial,
    MeshStandardMaterial,
)

# --- hand-built models ---------------------------------------------------------

# One triangle whose first corner sits on +x, so a turn is easy to read off it.
_TRIANGLE = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])


def _model_data(*clips, normals=False):
    """A one-triangle model on a node named ``root``, carrying ``clips``."""
    mesh = MeshData(
        vertices=_TRIANGLE,
        faces=torch.tensor([[0, 1, 2]]),
        normals=torch.tensor([[0.0, 0.0, 1.0]] * 3) if normals else None,
    )
    return SceneData(
        meshes=[mesh],
        nodes=[NodeData(name="root", mesh_indices=[0])],
        animations=list(clips),
    )


def _slide(name, end, runtime=1.0):
    """A clip translating ``root`` from where it is to ``end``."""
    return AnimationData(
        name=name,
        runtime=runtime,
        channels=[
            NodeAnimation(
                node_name="root",
                position_times=torch.tensor([0.0, runtime]),
                positions=torch.tensor([[0.0, 0.0, 0.0], list(end)]),
            )
        ],
    )


def _spin(name, degrees, axis, runtime=1.0):
    """A clip turning ``root`` by ``degrees`` about ``axis``."""
    half = math.radians(degrees) / 2
    end = [a * math.sin(half) for a in axis] + [math.cos(half)]
    return AnimationData(
        name=name,
        runtime=runtime,
        channels=[
            NodeAnimation(
                node_name="root",
                rotation_times=torch.tensor([0.0, runtime]),
                rotations=torch.tensor([[0.0, 0.0, 0.0, 1.0], end]),
            )
        ],
    )


def _corners_at(scene, mesh, times):
    """``mesh``'s corner positions ``[T, 3F, 3]`` at ``times`` seconds."""
    scene.timeline_manager.set_state_to_times(torch.tensor(times))
    try:
        return mesh.grid.location.clone()
    finally:
        scene.timeline_manager.clear_buffers()


# A unit quad in z = 0, two triangles sharing the diagonal 0-2.
_QUAD = torch.tensor([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
_QUAD_UVS = torch.tensor([[0.0, 0], [1, 0], [1, 1], [0, 1]])
_QUAD_FACES = torch.tensor([[0, 1, 2], [0, 2, 3]])


def _quads(*materials, uvs=True):
    """One textured-ready quad per material, each on its own node."""
    meshes = [
        MeshData(
            vertices=_QUAD,
            faces=_QUAD_FACES,
            uvs=_QUAD_UVS if uvs else None,
            material_index=i,
        )
        for i in range(len(materials))
    ]
    nodes = [NodeData(name=f"quad{i}", mesh_indices=[i]) for i in range(len(meshes))]
    return SceneData(meshes=meshes, materials=list(materials), nodes=nodes)


# --- clips and playback --------------------------------------------------------


def test_play_animation_plays_the_first_clip_unless_another_is_named():
    data = _model_data(_slide("up", (0, 2, 0)), _slide("right", (3, 0, 0)))
    for name, end in [(None, (0, 2, 0)), ("up", (0, 2, 0)), ("right", (3, 0, 0))]:
        with Scene() as scene:
            model = Model3D(scene_data=data).spawn(False)
            assert model.animation_names == ["up", "right"]
            assert model.play_animation(name) is model
            corners = _corners_at(scene, model.mesh_mobs[0], [1.0])
        torch.testing.assert_close(corners[0], _TRIANGLE + torch.tensor(end))


def test_an_unknown_clip_is_a_key_error_naming_the_clips_there_are():
    with Scene():
        model = Model3D(
            scene_data=_model_data(_slide("up", (0, 1, 0)), _slide("down", (0, -1, 0)))
        )
        for call in (model.play_animation, model.precompute_animation):
            with pytest.raises(KeyError, match=r"'sideways'.*\['up', 'down'\]"):
                call("sideways")


def test_a_model_without_clips_refuses_to_play():
    with Scene():
        model = Model3D(scene_data=_model_data())
        assert model.animation_names == []
        for call in (model.play_animation, model.precompute_animation):
            with pytest.raises(AlganConfigurationError, match="no animation clips"):
                call()


def test_playback_records_a_turn_and_the_shading_turns_with_the_surface():
    """A clip with no ``runtime`` given plays over its own, and a mesh with
    authored normals has them recomputed from the moving geometry -- the
    authored ones describe the rest pose only.
    """
    clip = _spin("spin", 90, axis=(0, 1, 0), runtime=2.0)
    with Scene() as scene:
        model = Model3D(scene_data=_model_data(clip, normals=True)).spawn(False)
        model.play_animation()
        assert scene.animation_manager.context.timespan.original_end == pytest.approx(2)
        mesh = model.mesh_mobs[0]
        scene.timeline_manager.set_state_to_times(torch.tensor([0.0, 1.0, 2.0]))
        corners = mesh.grid.location.clone()
        normals = mesh.get_render_primitives().normals.clone()
        scene.timeline_manager.clear_buffers()

    c = math.sqrt(0.5)
    # A quarter turn about +y takes +x to -z, and the +z normal to +x.
    torch.testing.assert_close(
        corners[:, 0],
        torch.tensor([[1.0, 0, 0], [c, 0, -c], [0, 0, -1]]),
        atol=1e-5,
        rtol=0,
    )
    torch.testing.assert_close(
        normals[:, 0],
        torch.tensor([[0.0, 0, 1], [c, 0, c], [1, 0, 0]]),
        atol=1e-5,
        rtol=0,
    )


@pytest.mark.parametrize(
    ("place", "kwargs", "expected"),
    [
        (lambda model: model.move(RIGHT * 3), {}, lambda t: [4.0, t, 0.0]),
        (
            lambda model: model.rotate(90, OUT).scale(2),
            {},
            lambda t: [-2.0 * t, 2.0, 0.0],
        ),
        (None, {"location": (0.0, 0.0, 2.0)}, lambda t: [1.0, t, 2.0]),
    ],
    ids=["moved", "turned_and_scaled", "built_at_a_location"],
)
def test_playback_stays_where_the_model_was_placed(place, kwargs, expected):
    """REGRESSION. The baked poses were absolute positions in the file's own
    coordinates, so a model moved, turned or scaled before ``play_animation``
    (or built at a ``location``) snapped back to where the file put it.
    """
    with Scene() as scene:
        model = Model3D(scene_data=_model_data(_slide("up", (0, 1, 0))), **kwargs)
        model.spawn(False)
        if place is not None:
            with Off():
                place(model)
        model.play_animation()
        corners = _corners_at(scene, model.mesh_mobs[0], [0.5, 1.0])
    torch.testing.assert_close(
        corners[:, 0], torch.tensor([expected(0.5), expected(1.0)]), atol=1e-5, rtol=0
    )


def test_playback_composes_with_motion_recorded_after_it_in_a_sync():
    """The composition the tutorial shows: a clip playing while the model
    moves across the frame.
    """
    with Scene() as scene:
        model = Model3D(scene_data=_model_data(_slide("up", (0, 1, 0)))).spawn(False)
        with Sync(runtime=1):
            model.play_animation()
            model.move(RIGHT * 6)
        corners = _corners_at(scene, model.mesh_mobs[0], [0.5, 1.0])
    torch.testing.assert_close(
        corners[:, 0],
        torch.tensor([[4.0, 0.5, 0.0], [7.0, 1.0, 0.0]]),
        atol=1e-5,
        rtol=0,
    )


# --- placement -----------------------------------------------------------------


def test_location_places_the_model():
    """REGRESSION. ``location`` moved only the model's own anchor and left its
    geometry at the origin, so the model drew in the wrong place and turned
    about a point beside it.
    """
    offset = torch.tensor([2.0, -1.0, 0.5])
    with Scene():
        placed = Model3D(scene_data=_model_data(), location=offset)
        torch.testing.assert_close(
            placed.mesh_mobs[0].grid.location[0], _TRIANGLE + offset
        )

        with Off():
            placed.rotate(90, OUT)
        torch.testing.assert_close(
            placed.mesh_mobs[0].grid.location[0, 0], offset + torch.tensor([0.0, 1, 0])
        )

        fitted = Model3D(scene_data=_model_data(), location=offset, fit_to_size=3.0)
        corners = fitted.mesh_mobs[0].grid.location.reshape(-1, 3)
        lo, hi = corners.amin(0), corners.amax(0)
        torch.testing.assert_close((lo + hi) / 2, offset)
        torch.testing.assert_close((hi - lo).norm(), torch.tensor(3.0))


def test_a_model_needs_a_source_with_geometry():
    with Scene():
        with pytest.raises(AlganConfigurationError, match="file_path or scene_data"):
            Model3D()
        with pytest.raises(AlganConfigurationError, match="No renderable meshes"):
            Model3D(scene_data=SceneData(source_path="empty.glb"))


# --- TriangleMesh normals ------------------------------------------------------

# A regular tetrahedron about the origin, every face wound so that +cross is
# outward (the convention every triangle Algan builds follows).
_TETRA = torch.tensor([[1.0, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])
_TETRA_FACES = torch.tensor([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]])


def _render_normals(mesh):
    return mesh.get_render_primitives().normals[0]


def test_a_mesh_without_normals_shades_each_face_by_its_outward_normal():
    with Scene():
        normals = _render_normals(TriangleMesh(_TETRA, _TETRA_FACES)).reshape(4, 3, 3)
    for face, corner_normals in zip(_TETRA_FACES, normals):
        # About the centre of a regular tetrahedron, a face's outward normal
        # points along its centroid; all three of its corners share it.
        outward = F.normalize(_TETRA[face].mean(0), dim=0)
        torch.testing.assert_close(corner_normals, outward.expand(3, 3))


def test_authored_normals_are_normalized_and_turn_with_the_mesh():
    with Scene():
        mesh = TriangleMesh(_TETRA, _TETRA_FACES, normals=_TETRA * 5)
        # Authored here as each vertex's own direction, which any turn about
        # the centre keeps true.
        torch.testing.assert_close(
            _render_normals(mesh), F.normalize(mesh.grid.location[0], dim=-1)
        )
        with Off():
            mesh.rotate(90, UP).rotate(30, RIGHT)
        turned = mesh.grid.location[0]
        assert not torch.allclose(turned, _TETRA[_TETRA_FACES.reshape(-1)])
        torch.testing.assert_close(
            _render_normals(mesh), F.normalize(turned, dim=-1), atol=1e-6, rtol=0
        )


def test_recomputed_normals_weight_shared_vertices_by_face_area():
    """Two faces hinged on the edge v0-v1: one of area 1 facing +z, one of
    area 1/2 facing +y. The shared corners lean twice as far towards +z, and
    the authored normals (all +x here) are ignored in favour of the geometry.
    """
    vertices = torch.tensor([[0.0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 1]])
    faces = torch.tensor([[0, 1, 2], [0, 3, 1]])
    with Scene():
        mesh = TriangleMesh(
            vertices,
            faces,
            normals=torch.tensor([[1.0, 0, 0]] * 4),
            recompute_normals=True,
        )
        rest = mesh.grid.location
        # A second frame with v2 lifted out of the plane: the normals follow it.
        lifted = rest.clone()
        lifted[0, 2] = torch.tensor([0.0, 2.0, 2.0])
        normals = mesh._compute_corner_normals(torch.cat([rest, lifted]))

    shared = F.normalize(torch.tensor([0.0, 1.0, 2.0]), dim=0)
    z, y = torch.tensor([0.0, 0, 1]), torch.tensor([0.0, 1, 0])
    torch.testing.assert_close(
        normals[0], torch.stack([shared, shared, z, shared, y, shared])
    )
    # Face 0's doubled-area normal is now (0, -2, 2); face 1's is unchanged.
    shared = F.normalize(torch.tensor([0.0, -1.0, 2.0]), dim=0)
    tilted = F.normalize(torch.tensor([0.0, -1.0, 1.0]), dim=0)
    torch.testing.assert_close(
        normals[1], torch.stack([shared, shared, tilted, shared, y, shared])
    )


def test_ignore_normals_hands_the_renderer_zero_normals():
    with Scene():
        mesh = TriangleMesh(_TETRA, _TETRA_FACES, normals=_TETRA, ignore_normals=True)
        assert mesh.corner_normals is None
        assert not _render_normals(mesh).any()


# --- TriangleMesh render primitive ---------------------------------------------


def test_a_raw_image_texture_is_laid_out_like_a_prebuilt_map():
    image = torch.rand(4, 6, 3)
    with Scene():
        raw = TriangleMesh(_QUAD, _QUAD_FACES, uvs=_QUAD_UVS, texture=image)
        prebuilt = TriangleMesh(
            _QUAD, _QUAD_FACES, uvs=_QUAD_UVS, texture=image_to_texture_map(image)
        )
    assert raw.texture_map.shape == (6, 4, 5)
    torch.testing.assert_close(raw.texture_map, prebuilt.texture_map)


@pytest.mark.parametrize("opacity_in_sampler", [True, False])
def test_a_textured_primitive_carries_its_map_uvs_and_opacity(
    monkeypatch, opacity_in_sampler
):
    from algan.rendering.raytracing import settings as raytracing_settings

    monkeypatch.setattr(
        raytracing_settings,
        "texture_opacity_in_kernel_active",
        lambda: opacity_in_sampler,
    )
    image = torch.rand(4, 6, 4)
    with Scene():
        with Off():
            mesh = TriangleMesh(
                _QUAD, _QUAD_FACES, uvs=_QUAD_UVS, texture=image, opacity=0.5
            ).spawn()
        primitive = mesh.get_render_primitives()

    # The map is the albedo; the corners are white placeholders at the opacity.
    torch.testing.assert_close(primitive.colors[0, :, :3], torch.ones(6, 3))
    torch.testing.assert_close(primitive.colors[0, :, 4], torch.full((6,), 0.5))
    torch.testing.assert_close(primitive.uvs[0], _QUAD_UVS[_QUAD_FACES])
    authored = image_to_texture_map(image).unsqueeze(0)
    if opacity_in_sampler:
        # The map stays as authored and the sampler applies the opacity.
        torch.testing.assert_close(primitive.texture_map, authored)
        torch.testing.assert_close(primitive.texture_opacity, torch.tensor([0.5]))
    else:
        premultiplied = authored.clone()
        premultiplied[..., 4] *= 0.5
        torch.testing.assert_close(primitive.texture_map, premultiplied)
        assert primitive.texture_opacity is None


def test_vertex_colours_reach_the_corners_with_the_mob_opacity_applied():
    colours = torch.tensor([[1.0, 0, 0, 0.5], [0, 1, 0, 1], [0, 0, 1, 1], [1, 1, 1, 1]])
    with Scene():
        with Off():
            mesh = TriangleMesh(
                _QUAD, _QUAD_FACES, vertex_colors=colours, opacity=0.5
            ).spawn()
        primitive = mesh.get_render_primitives()
    per_corner = colours[_QUAD_FACES.reshape(-1)]
    torch.testing.assert_close(primitive.colors[0, :, :3], per_corner[:, :3])
    torch.testing.assert_close(primitive.colors[0, :, 4], per_corner[:, 3] * 0.5)


@pytest.mark.parametrize(
    ("faces", "shells"),
    [
        ([[0, 1, 2]], None),
        ([[0, 1, 2], [0, 2, 3]], None),
        ([[0, 1, 2], [3, 4, 5]], [0, 1]),
        ([[0, 1, 2], [0, 3, 4]], [0, 1]),
        ([[0, 1, 2], [4, 5, 6], [0, 2, 3], [4, 6, 7]], [0, 1, 0, 1]),
    ],
    ids=["triangle", "quad", "two_triangles", "bow_tie", "two_interleaved_quads"],
)
def test_shells_split_a_mesh_on_shared_edges_not_shared_vertices(faces, shells):
    """A bow tie -- two triangles meeting at one vertex -- is two shells, the
    way two cones meeting at an apex are two surfaces. One shell is no ids.
    """
    vertices = torch.rand(8, 3, generator=torch.Generator().manual_seed(0))
    with Scene():
        mesh = TriangleMesh(vertices, torch.tensor(faces))
        ids = mesh.triangle_shell_ids()
        primitive = mesh.get_render_primitives()
    if shells is None:
        assert ids is None
        assert getattr(primitive, "mesh_ids", None) is None
    else:
        assert ids.tolist() == shells
        assert mesh.triangle_shell_ids() is ids, "computed once from the topology"
        assert primitive.mesh_ids is ids


def test_rebatching_carries_each_triangles_normals_and_uvs_with_it():
    normals = F.normalize(torch.rand(4, 3, generator=torch.Generator().manual_seed(1)))
    with Scene():
        mesh = TriangleMesh(
            _QUAD,
            _QUAD_FACES,
            normals=normals,
            uvs=_QUAD_UVS,
            texture=torch.rand(2, 2, 3),
        )
        index = mesh.corner_index.view(2, 3).clone()
        corner_normals = mesh.corner_normals.view(2, 3, 3).clone()
        corner_uvs = mesh.corner_uvs.view(2, 3, 2).clone()
        mesh._rebatch_structural_attrs(torch.tensor([1, 1, 0]))
    assert mesh.num_triangles == 3
    torch.testing.assert_close(mesh.corner_index.view(3, 3), index[[1, 1, 0]])
    torch.testing.assert_close(
        mesh.corner_normals.view(3, 3, 3), corner_normals[[1, 1, 0]]
    )
    torch.testing.assert_close(mesh.corner_uvs.view(3, 3, 2), corner_uvs[[1, 1, 0]])


# --- textures and normal maps --------------------------------------------------

_RGB = np.array([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]], np.uint8)


def test_texture_files_load_once_per_path_and_a_missing_one_falls_back(
    tmp_path, monkeypatch
):
    path = str(tmp_path / "shared.png")
    Image.fromarray(_RGB).save(path)
    loads = []
    load = model_mob._load_image_hwc
    monkeypatch.setattr(
        model_mob,
        "_load_image_hwc",
        lambda p, *rest: loads.append(p) or load(p, *rest),
    )
    data = _quads(
        MaterialData(diffuse_texture=path),
        MaterialData(diffuse_texture=path),
        MaterialData(base_color=(0.2, 0.4, 0.6, 1.0), diffuse_texture=path + ".gone"),
    )
    with Scene():
        first, second, missing = Model3D(scene_data=data).mesh_mobs
        untextured = Model3D(scene_data=data, load_textures=False).mesh_mobs

    assert loads.count(path) == 1, "two materials sharing a file must load it once"
    expected = image_to_texture_map(torch.tensor(_RGB / 255.0, dtype=torch.float32))
    torch.testing.assert_close(first.texture_map, expected)
    torch.testing.assert_close(second.texture_map, expected)
    assert missing.texture_map is None
    torch.testing.assert_close(
        missing.grid.color[0, :, :3], torch.tensor([0.2, 0.4, 0.6]).expand(6, 3)
    )
    assert all(mesh.texture_map is None for mesh in untextured)


def test_an_embedded_image_takes_precedence_over_a_texture_file(tmp_path):
    path = str(tmp_path / "blue.png")
    Image.fromarray(np.full((2, 2, 3), (0, 0, 255), np.uint8)).save(path)
    red = torch.zeros(2, 2, 3)
    red[..., 0] = 1.0
    data = _quads(MaterialData(diffuse_image=red, diffuse_texture=path))
    with Scene():
        mesh = Model3D(scene_data=data).mesh_mobs[0]
    torch.testing.assert_close(mesh.texture_map, image_to_texture_map(red))


def test_a_texture_without_uvs_falls_back_to_the_base_colour():
    material = MaterialData(
        base_color=(0.1, 0.2, 0.3, 1.0), diffuse_image=torch.rand(2, 2, 3)
    )
    with Scene():
        mesh = Model3D(scene_data=_quads(material, uvs=False)).mesh_mobs[0]
    assert mesh.texture_map is None
    torch.testing.assert_close(
        mesh.grid.color[0, :, :3], torch.tensor([0.1, 0.2, 0.3]).expand(6, 3)
    )


def test_normal_maps_load_from_files_but_an_embedded_image_wins(tmp_path):
    path = str(tmp_path / "tilted.png")
    tilted = np.full((2, 2, 3), (255, 128, 128), np.uint8)
    Image.fromarray(tilted).save(path)
    embedded = torch.zeros(2, 2, 3)
    embedded[..., 2] = 1.0
    materials = [
        MaterialData(normal_texture=path),
        MaterialData(normal_texture=path, normal_image=embedded),
        MaterialData(normal_texture=path + ".gone"),
    ]
    with Scene():
        from_file, from_image, missing = Model3D(
            scene_data=_quads(*materials)
        ).mesh_mobs
        no_uvs = Model3D(scene_data=_quads(*materials, uvs=False)).mesh_mobs
        disabled = Model3D(scene_data=_quads(*materials), normal_maps=False).mesh_mobs

    torch.testing.assert_close(
        from_file.normal_texture_map,
        image_to_normal_map(torch.tensor(tilted / 255.0, dtype=torch.float32)),
    )
    torch.testing.assert_close(
        from_image.normal_texture_map, image_to_normal_map(embedded)
    )
    assert missing.normal_texture_map is None
    assert all(mesh.normal_texture_map is None for mesh in no_uvs + disabled)


def test_imported_materials_become_standard_or_physical_materials():
    glowing = MaterialData(
        base_color=(0.2, 0.4, 0.6, 1.0),
        metallic_factor=0.3,
        roughness_factor=0.7,
        emissive=(1.0, 0.5, 0.0),
        emissive_image=torch.full((2, 2, 3), 0.5),
    )
    glass = MaterialData(base_color=(1.0, 1.0, 1.0, 0.25), refractive_index=1.5)
    with Scene():
        standard, physical = Model3D(scene_data=_quads(glowing, glass)).mesh_mobs

    assert type(standard.material) is MeshStandardMaterial
    assert standard.material.metalness == pytest.approx(0.3)
    assert standard.material.roughness == pytest.approx(0.7)
    # The emissive factor is scaled by the emissive map's mean.
    torch.testing.assert_close(
        standard.material.emissive[:3].as_subclass(torch.Tensor),
        torch.tensor([0.5, 0.25, 0.0]),
    )
    assert type(physical.material) is MeshPhysicalMaterial
    assert physical.material.ior == pytest.approx(1.5)
    assert physical.material.transmission == pytest.approx(0.75)


def test_without_pbr_legacy_reflectivity_and_ior_become_a_physical_material():
    materials = [
        MaterialData(reflectivity=0.6, roughness_factor=0.4),
        MaterialData(refractive_index=1.33),
        MaterialData(),
    ]
    with Scene():
        mirror, glass, plain = Model3D(
            scene_data=_quads(*materials), pbr_materials=False
        ).mesh_mobs

    assert type(mirror.material) is MeshPhysicalMaterial
    assert (mirror.material.metalness, mirror.material.roughness) == pytest.approx(
        (0.6, 0.4)
    )
    assert mirror.material.transmission == 0
    assert type(glass.material) is MeshPhysicalMaterial
    assert (glass.material.ior, glass.material.transmission) == pytest.approx((1.33, 1))
    assert getattr(plain, "material", None) is None


# --- the glTF reader -------------------------------------------------------------


class _Gltf:
    """A glTF scene assembled in memory, its data packed into buffers."""

    def __init__(self):
        self.blobs = [bytearray()]
        self.views = []
        self.accessors = []

    def data(self, payload, buffer=0):
        """A buffer view over ``payload``, 4-byte aligned as glTF requires."""
        while len(self.blobs) <= buffer:
            self.blobs.append(bytearray())
        blob = self.blobs[buffer]
        while len(blob) % 4:
            blob.append(0)
        self.views.append(
            g.BufferView(buffer=buffer, byteOffset=len(blob), byteLength=len(payload))
        )
        blob.extend(payload)
        return len(self.views) - 1

    def accessor(self, values, kind, component=5126, buffer=0):
        array = np.asarray(values, np.float32 if component == 5126 else np.uint16)
        self.accessors.append(
            g.Accessor(
                bufferView=self.data(array.tobytes(), buffer),
                componentType=component,
                count=len(array),
                type=kind,
            )
        )
        return len(self.accessors) - 1

    def triangle(self, offset=0.0, mode=None, uvs=False, material=None):
        positions = [[offset, 0, 0], [offset + 1, 0, 0], [offset, 1, 0]]
        attributes = g.Attributes(POSITION=self.accessor(positions, "VEC3"))
        if uvs:
            attributes.TEXCOORD_0 = self.accessor([[0, 0], [1, 0], [0, 1]], "VEC2")
        primitive = g.Primitive(attributes=attributes, material=material)
        if mode is None:
            primitive.indices = self.accessor([0, 1, 2], "SCALAR", 5123)
        else:
            primitive.mode = mode
        return primitive

    def keys(self, times, values, kind="VEC3", buffer=0):
        """An animation sampler over ``times`` and ``values``."""
        return g.AnimationSampler(
            input=self.accessor(times, "SCALAR", buffer=buffer),
            output=self.accessor(values, kind, buffer=buffer),
        )

    def save(self, path, **fields):
        """Write a ``.glb``, or a ``.gltf`` whose buffers are base64 data URIs."""
        glb = path.suffix == ".glb"
        buffers = [
            g.Buffer(
                byteLength=len(blob),
                uri=None
                if glb
                else "data:application/octet-stream;base64,"
                + base64.b64encode(bytes(blob)).decode(),
            )
            for blob in self.blobs
        ]
        model = g.GLTF2(
            accessors=self.accessors, bufferViews=self.views, buffers=buffers, **fields
        )
        if glb:
            assert len(self.blobs) == 1, "a glB carries one binary chunk"
            model.set_binary_blob(bytes(self.blobs[0]))
            model.save_binary(str(path))
        else:
            model.save_json(str(path))
        return str(path)


def _channel(sampler, node, path):
    return g.AnimationChannel(
        sampler=sampler, target=g.AnimationChannelTarget(node=node, path=path)
    )


def test_gltf_node_matrices_and_scale_keys_compose_down_the_hierarchy(tmp_path):
    """A parent placed by a column-major ``matrix``, a child keyed by
    ``scale``, and a channel targeting no node, which is skipped.
    """
    gltf = _Gltf()
    mesh = g.Mesh(primitives=[gltf.triangle()])
    sampler = gltf.keys([0.0, 1.0], [[1, 1, 1], [2, 3, 1]])
    matrix = np.eye(4)
    matrix[:3, 3] = (5, 0, 0)
    path = gltf.save(
        tmp_path / "matrix.glb",
        scene=0,
        scenes=[g.Scene(nodes=[0])],
        nodes=[
            g.Node(name="parent", matrix=matrix.T.reshape(-1).tolist(), children=[1]),
            g.Node(name="child", mesh=0),
        ],
        meshes=[mesh],
        animations=[
            g.Animation(
                name="grow",
                samplers=[sampler],
                channels=[_channel(0, 1, "scale"), _channel(0, None, "scale")],
            )
        ],
    )
    with Scene():
        model = Model3D(path)
        (clip,) = model.animations
        assert [channel.node_name for channel in clip.channels] == ["child"]
        assert clip.runtime == 1.0
        mesh = model.mesh_mobs[0]
        torch.testing.assert_close(
            mesh.grid.location[0], torch.tensor([[5.0, 0, 0], [6, 0, 0], [5, 1, 0]])
        )
        _, corners = model.precompute_animation(times=[1.0])
    torch.testing.assert_close(
        corners[mesh][0], torch.tensor([[5.0, 0, 0], [7, 0, 0], [5, 3, 0]])
    )


def test_a_gltf_without_a_default_scene_roots_at_its_parentless_nodes(tmp_path):
    gltf = _Gltf()
    mesh = g.Mesh(primitives=[gltf.triangle()])
    sampler = gltf.keys([0.0, 1.0], [[0, 0, 0], [0, 2, 0]])
    path = gltf.save(
        tmp_path / "sceneless.glb",
        nodes=[g.Node(name="root", children=[1]), g.Node(name="leaf", mesh=0)],
        meshes=[mesh],
        animations=[
            g.Animation(samplers=[sampler], channels=[_channel(0, 0, "translation")])
        ],
    )
    with Scene():
        model = Model3D(path)
        assert [(n.name, n.parent) for n in model.scene_data.nodes] == [
            ("root", -1),
            ("leaf", 0),
        ]
        assert model.animation_names == ["animation_0"]
        _, corners = model.precompute_animation(times=[1.0])
    torch.testing.assert_close(
        corners[model.mesh_mobs[0]][0],
        torch.tensor([[0.0, 2, 0], [1, 2, 0], [0, 3, 0]]),
    )


def test_gltf_animation_in_a_second_buffer_is_read_from_that_buffer(tmp_path):
    """REGRESSION. Every accessor was read from the first buffer whatever its
    view named, so keys stored in another buffer silently came back as the
    first buffer's vertex data -- here two zero times and a zero runtime.
    """
    gltf = _Gltf()
    mesh = g.Mesh(primitives=[gltf.triangle()])
    sampler = gltf.keys([0.0, 1.0], [[0, 0, 0], [0, 4, 0]], buffer=1)
    path = gltf.save(
        tmp_path / "two_buffers.gltf",
        scene=0,
        scenes=[g.Scene(nodes=[0])],
        nodes=[g.Node(name="root", mesh=0)],
        meshes=[mesh],
        animations=[
            g.Animation(
                name="rise",
                samplers=[sampler],
                channels=[_channel(0, 0, "translation")],
            )
        ],
    )
    with Scene():
        model = Model3D(path)
        (clip,) = model.animations
        (channel,) = clip.channels
        torch.testing.assert_close(channel.position_times, torch.tensor([0.0, 1.0]))
        torch.testing.assert_close(
            channel.positions, torch.tensor([[0.0, 0, 0], [0, 4, 0]])
        )
        assert clip.runtime == 1.0
        _, corners = model.precompute_animation()
    torch.testing.assert_close(
        corners[model.mesh_mobs[0]][-1],
        torch.tensor([[0.0, 4, 0], [1, 4, 0], [0, 5, 0]]),
    )


def test_a_malformed_gltf_animation_warns_and_loads_the_model_static(tmp_path):
    gltf = _Gltf()
    mesh = g.Mesh(primitives=[gltf.triangle()])
    sampler = gltf.keys([0.0, 1.0], [[0, 0, 0], [0, 2, 0]])
    path = gltf.save(
        tmp_path / "broken.glb",
        scene=0,
        scenes=[g.Scene(nodes=[0])],
        nodes=[g.Node(name="root", mesh=0)],
        meshes=[mesh],
        animations=[
            # Sampler 7 does not exist.
            g.Animation(samplers=[sampler], channels=[_channel(7, 0, "translation")])
        ],
    )
    with Scene():
        with pytest.warns(
            UnsupportedFeatureWarning, match="loading the model as static"
        ):
            model = Model3D(path)
        assert model.animations == []
        assert len(model.mesh_mobs) == 1


@pytest.mark.parametrize("animated", [False, True])
def test_non_triangle_gltf_primitives_are_left_out(tmp_path, animated):
    gltf = _Gltf()
    meshes = [
        g.Mesh(name="tri", primitives=[gltf.triangle()]),
        g.Mesh(name="dots", primitives=[gltf.triangle(offset=3.0, mode=0)]),
    ]
    fields = {}
    if animated:
        sampler = gltf.keys([0.0, 1.0], [[0, 0, 0], [0, 2, 0]])
        fields["animations"] = [
            g.Animation(samplers=[sampler], channels=[_channel(0, 0, "translation")])
        ]
    path = gltf.save(
        tmp_path / "mixed.glb",
        scene=0,
        scenes=[g.Scene(nodes=[0, 1])],
        nodes=[g.Node(name="tri", mesh=0), g.Node(name="dots", mesh=1)],
        meshes=meshes,
        **fields,
    )
    with Scene():
        model = Model3D(path)
    assert model.node_names == ["tri"]
    assert model.mesh_mobs[0].num_triangles == 1
    assert len(model.animations) == (1 if animated else 0)


def test_other_trimesh_formats_load_static_with_their_authored_normals(tmp_path):
    path = tmp_path / "quad.obj"
    path.write_text(
        "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\n"
        "vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\n"
        "vn 0 0 1\n"
        "f 1/1/1 2/2/1 3/3/1 4/4/1\n"
    )
    with Scene():
        model = Model3D(str(path))
    assert model.animations == []
    (mesh,) = model.mesh_mobs
    assert mesh.num_triangles == 2
    corners = {tuple(c) for c in mesh.grid.location[0].tolist()}
    assert corners == {tuple(v) for v in _QUAD.tolist()}
    torch.testing.assert_close(
        mesh.corner_normals, torch.tensor([[0.0, 0, 1]]).expand(6, 3)
    )


def test_a_missing_model_file_is_a_file_not_found_error(tmp_path):
    with Scene(), pytest.raises(FileNotFoundError):
        Model3D(str(tmp_path / "absent.glb"))


def test_an_unreadable_embedded_texture_is_dropped_rather_than_fatal():
    material = _convert_material(
        SimpleNamespace(
            material=SimpleNamespace(baseColorTexture=np.full((1, 1, 3), 2.0))
        )
    )
    assert material.diffuse_image is None
    assert material.base_color == (1.0, 1.0, 1.0, 1.0)
