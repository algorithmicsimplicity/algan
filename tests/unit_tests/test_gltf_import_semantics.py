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

from algan.mobs.three_d_models.gltf_loader import load_scene

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
