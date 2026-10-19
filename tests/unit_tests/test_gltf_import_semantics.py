"""glTF import follows the glTF 2.0 specification: URIs, material factors and
vertex colors.

Each model is a tiny ``.gltf`` written into ``tmp_path`` by :func:`_write_gltf`.
"""

from __future__ import annotations

import base64
import io
import json
import warnings

import numpy as np
import pytest
import torch
from PIL import Image

from algan import Model3D, Scene
from algan.mobs.three_d_models.gltf_loader import load_scene
from algan.utils.color_space import linear_to_srgb, srgb_to_linear

FLOAT, UBYTE, USHORT = 5126, 5121, 5123


def _png(pixels):
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(pixels, dtype=np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def _quad():
    """A unit quad facing +z; glTF UVs (v = 0 is the image's top row)."""
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], np.float32)
    uvs = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], np.float32)
    faces = np.array([0, 1, 2, 0, 2, 3], np.uint16)
    return vertices, uvs, faces


def _write_gltf(path, material=None, *, uvs=True, colors=None, images=()):
    """Write a one-quad ``.gltf``. ``images`` holds ``(texture_slot, image)``
    pairs, each either PNG bytes (embedded) or a URI string, wired into the
    material's ``pbrMetallicRoughness`` slot of that name.
    """
    blob = bytearray()
    doc = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "quad"}],
        "accessors": [],
        "bufferViews": [],
    }

    def accessor(data, kind, component=FLOAT, normalized=False):
        data = np.asarray(data)
        while len(blob) % 4:
            blob.append(0)
        doc["bufferViews"].append(
            {"buffer": 0, "byteOffset": len(blob), "byteLength": data.nbytes}
        )
        blob.extend(data.tobytes())
        entry = {
            "bufferView": len(doc["bufferViews"]) - 1,
            "componentType": component,
            "count": len(data),
            "type": kind,
        }
        if normalized:
            entry["normalized"] = True
        if kind == "VEC3" and component == FLOAT:
            entry["min"] = data.min(0).tolist()
            entry["max"] = data.max(0).tolist()
        doc["accessors"].append(entry)
        return len(doc["accessors"]) - 1

    vertices, quad_uvs, faces = _quad()
    attributes = {"POSITION": accessor(vertices, "VEC3")}
    if uvs:
        attributes["TEXCOORD_0"] = accessor(quad_uvs, "VEC2")
    if colors is not None:
        colors = np.asarray(colors)
        component = {np.dtype(np.float32): FLOAT, np.dtype(np.uint8): UBYTE}[
            colors.dtype
        ]
        attributes["COLOR_0"] = accessor(
            colors,
            "VEC4" if colors.shape[-1] == 4 else "VEC3",
            component,
            normalized=component != FLOAT,
        )
    primitive = {
        "attributes": attributes,
        "indices": accessor(faces, "SCALAR", USHORT),
    }
    if material is not None:
        material = json.loads(json.dumps(material))
        for slot, image in images:
            if isinstance(image, bytes):
                image = {
                    "uri": "data:image/png;base64," + base64.b64encode(image).decode()
                }
            else:
                image = {"uri": image}
            doc.setdefault("images", []).append(image)
            doc.setdefault("textures", []).append({"source": len(doc["images"]) - 1})
            material.setdefault("pbrMetallicRoughness", {})[slot] = {
                "index": len(doc["textures"]) - 1
            }
        doc["materials"] = [material]
        primitive["material"] = 0
    doc["meshes"] = [{"name": "quad", "primitives": [primitive]}]
    doc["buffers"] = [
        {
            "byteLength": len(blob),
            "uri": "data:application/octet-stream;base64,"
            + base64.b64encode(bytes(blob)).decode(),
        }
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


_RED = np.zeros((2, 2, 4), np.uint8)
_RED[..., 0] = _RED[..., 3] = 255


@pytest.mark.parametrize(
    ("uri", "file_name"),
    [
        ("my%20tex.png", "models/my tex.png"),
        ("text%C3%BCr.png", "models/textür.png"),
        ("../textures/tex.png", "textures/tex.png"),
    ],
)
def test_encoded_and_parent_relative_texture_uris_load(tmp_path, uri, file_name):
    (tmp_path / file_name).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / file_name).write_bytes(_png(_RED))
    path = _write_gltf(
        tmp_path / "models" / "model.gltf",
        {"pbrMetallicRoughness": {"metallicFactor": 0}},
        images=[("baseColorTexture", uri)],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        image = load_scene(str(path)).materials[0].diffuse_image
    assert image is not None
    torch.testing.assert_close(image, torch.tensor(_RED / 255, dtype=torch.float32))


def test_unreadable_texture_uri_warns_with_its_name(tmp_path):
    path = _write_gltf(
        tmp_path / "model.gltf",
        {"pbrMetallicRoughness": {"metallicFactor": 0}},
        images=[("baseColorTexture", "missing%20wood.png")],
    )
    with pytest.warns(UserWarning, match="missing%20wood.png"):
        material = load_scene(str(path)).materials[0]
    assert material.diffuse_image is None


def _material(tmp_path, material, **kwargs):
    path = _write_gltf(tmp_path / "material.gltf", material, **kwargs)
    return load_scene(str(path)).materials[0]


def test_omitted_factors_take_the_gltf_defaults(tmp_path):
    material = _material(tmp_path, {"pbrMetallicRoughness": {}})
    assert material.base_color == (1.0, 1.0, 1.0, 1.0)
    assert material.metallic_factor == 1.0
    assert material.roughness_factor == 1.0

    # A metallic-roughness map with B = 1 is fully metallic under the default.
    mr = np.zeros((2, 2, 3), np.uint8)
    mr[..., 1:] = 255
    path = _write_gltf(
        tmp_path / "mr.gltf",
        {"pbrMetallicRoughness": {}},
        images=[("metallicRoughnessTexture", _png(mr))],
    )
    with Scene():
        mesh = Model3D(str(path)).mesh_mobs[0]
        assert float(mesh.grid.metalness.reshape(-1)[0]) == pytest.approx(1.0)
        torch.testing.assert_close(
            mesh.grid.color.reshape(-1, 5)[:, :3], torch.ones(6, 3)
        )


def test_color_factors_are_linear(tmp_path):
    # 0.2 and 0.6 survive trimesh's 8-bit storage of the factors exactly.
    material = _material(
        tmp_path,
        {
            "pbrMetallicRoughness": {"baseColorFactor": [0.2, 0.0, 1.0, 1.0]},
            "emissiveFactor": [0.6, 0.0, 1.0],
        },
    )
    srgb = linear_to_srgb(torch.tensor([0.2, 0.6], dtype=torch.float64)).tolist()
    assert material.base_color == pytest.approx((srgb[0], 0.0, 1.0, 1.0), abs=1e-6)
    assert material.base_color[2] == 1.0
    assert material.emissive == pytest.approx((srgb[1], 0.0, 1.0), abs=1e-6)


def test_base_color_factor_multiplies_the_texture_in_linear_light(tmp_path):
    texture = np.full((2, 2, 4), 255, np.uint8)
    texture[0, 0] = (128, 64, 255, 200)
    material = _material(
        tmp_path,
        {
            "pbrMetallicRoughness": {"baseColorFactor": [1.0, 0.2, 0.0, 0.6]},
            "alphaMode": "BLEND",
        },
        images=[("baseColorTexture", _png(texture))],
    )
    image = material.diffuse_image
    source = torch.tensor(texture / 255, dtype=torch.float32)
    # A factor of 1 leaves its channel bit-exact; 0 clears it.
    torch.testing.assert_close(image[..., 0], source[..., 0], rtol=0, atol=0)
    assert float(image[..., 2].abs().max()) == 0.0
    expected_green = linear_to_srgb(srgb_to_linear(source[..., 1]) * 0.2)
    torch.testing.assert_close(image[..., 1], expected_green)
    torch.testing.assert_close(image[..., 3], source[..., 3] * 0.6)


@pytest.mark.parametrize(
    ("mode", "cutoff", "expected_base", "expected_texels"),
    [
        (None, None, 1.0, [1.0, 1.0]),  # OPAQUE by default: alpha is ignored
        ("OPAQUE", None, 1.0, [1.0, 1.0]),
        ("MASK", None, 1.0, [0.0, 1.0]),  # default cutoff 0.5
        ("MASK", 0.9, 0.0, [0.0, 0.0]),
        ("BLEND", None, 0.8, [0.8 * 100 / 255, 0.8]),
    ],
)
def test_alpha_mode_decides_what_alpha_means(
    tmp_path, mode, cutoff, expected_base, expected_texels
):
    material = {"pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 0.8]}}
    if mode is not None:
        material["alphaMode"] = mode
    if cutoff is not None:
        material["alphaCutoff"] = cutoff
    assert _material(tmp_path, material).base_color[3] == pytest.approx(
        expected_base, abs=2e-3
    )
    texture = np.full((1, 2, 4), 255, np.uint8)
    texture[0, 0, 3] = 100
    textured = _material(
        tmp_path, material, images=[("baseColorTexture", _png(texture))]
    )
    assert textured.diffuse_image[0, :, 3].tolist() == pytest.approx(
        expected_texels, abs=2e-3
    )


_VERTEX_RGBA = np.array(
    [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [1, 1, 1, 0.25]], np.float32
)


def _corner_colors(mesh):
    """Per-vertex RGBA the renderer receives, in the quad's vertex order."""
    colors = mesh.grid.color.reshape(-1, 5)
    order = mesh.corner_index.tolist()
    return torch.stack([colors[order.index(v)] for v in range(4)])[:, [0, 1, 2, 4]]


@pytest.mark.parametrize("uvs", [True, False])
@pytest.mark.parametrize("pbr_materials", [True, False])
def test_gltf_vertex_colors_are_the_albedo_beside_a_material(
    tmp_path, uvs, pbr_materials
):
    path = _write_gltf(
        tmp_path / "painted.gltf",
        {"pbrMetallicRoughness": {"metallicFactor": 0}},
        uvs=uvs,
        colors=_VERTEX_RGBA,
    )
    with Scene():
        mesh = Model3D(str(path), pbr_materials=pbr_materials).mesh_mobs[0]
        expected = torch.tensor(_VERTEX_RGBA)
        expected[:, 3] = 1  # OPAQUE (the default): COLOR_0's alpha is ignored
        torch.testing.assert_close(_corner_colors(mesh), expected)


def test_gltf_vertex_colors_are_linear_and_multiply_the_factor(tmp_path):
    colors = np.array(
        [[0.2, 0.2, 0.2, 1], [1, 1, 1, 0.5], [0, 0, 0, 1], [1, 1, 1, 1]], np.float32
    )
    path = _write_gltf(
        tmp_path / "tinted.gltf",
        {
            "pbrMetallicRoughness": {
                "baseColorFactor": [1.0, 0.2, 1.0, 0.6],
                "metallicFactor": 0,
            },
            "alphaMode": "BLEND",
        },
        colors=colors,
    )
    srgb = linear_to_srgb(torch.tensor([0.2, 0.04, 0.2], dtype=torch.float64))
    with Scene():
        actual = _corner_colors(Model3D(str(path)).mesh_mobs[0])
    # Linear COLOR_0 times the linear factor, shown as display sRGB.
    torch.testing.assert_close(
        actual[0], torch.tensor([srgb[0], srgb[1], srgb[0], 0.6]).float()
    )
    torch.testing.assert_close(
        actual[1], torch.tensor([1.0, float(srgb[2]), 1.0, 0.3]), atol=2e-3, rtol=0
    )


def test_ply_vertex_colors_survive_the_default_material(tmp_path):
    import trimesh

    vertices, _, faces = _quad()
    mesh = trimesh.Trimesh(vertices, faces.reshape(-1, 3), process=False)
    mesh.visual.vertex_colors = (_VERTEX_RGBA * 255).round().astype(np.uint8)
    path = tmp_path / "painted.ply"
    mesh.export(str(path))
    with Scene():
        actual = _corner_colors(Model3D(str(path)).mesh_mobs[0])
    expected = torch.tensor((_VERTEX_RGBA * 255).round() / 255, dtype=torch.float32)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
