"""A model file that cannot be parsed is reported by its name.

pygltflib and trimesh report a damaged file by whatever it tripped -- a
``struct.error`` for an empty glB, ``chunk was not expected length!`` for a
truncated one, an ``IndexError`` from a broken OBJ -- and never by the file.
"""

from __future__ import annotations

import re

import pytest
import trimesh

from algan import Model3D, Scene


def _box(file_type):
    return trimesh.creation.box().export(file_type=file_type)


@pytest.mark.parametrize(
    ("name", "content"),
    [
        ("empty.glb", lambda: b""),
        ("truncated.glb", lambda: _box("glb")[: len(_box("glb")) // 2]),
        ("text.glb", lambda: b"this is not a model\n"),
        ("broken.gltf", lambda: b"{ not json"),
        ("broken.obj", lambda: b"v 1 2\nf 1 2 3 4 5\n"),
    ],
)
def test_a_model_file_that_cannot_be_parsed_is_named(tmp_path, name, content):
    path = tmp_path / name
    path.write_bytes(content())
    with Scene(), pytest.raises(OSError) as raised:
        Model3D(str(path))
    assert re.match(
        rf"Could not read {re.escape(repr(str(path)))} as a {re.escape(path.suffix)} "
        r"model \(.+\)\. Check that it opens in another 3D viewer",
        str(raised.value),
    ), str(raised.value)
    # What the parser raised stays in the traceback.
    assert raised.value.__cause__ is not None


def test_a_gltf_whose_buffer_is_missing_names_the_model_and_the_buffer(tmp_path):
    files = _box("gltf")
    (tmp_path / "model.gltf").write_bytes(files["model.gltf"])
    with (
        Scene(),
        pytest.raises(
            OSError,
            match=r"model\.gltf' as a \.gltf model \(FileNotFoundError: .+\.bin\)",
        ),
    ):
        Model3D(str(tmp_path / "model.gltf"))


def test_a_file_that_cannot_be_opened_keeps_its_own_error(tmp_path):
    with Scene():
        with pytest.raises(FileNotFoundError):
            Model3D(str(tmp_path / "missing.glb"))
        (tmp_path / "folder.glb").mkdir()
        # Windows refuses to open a directory with a PermissionError.
        with pytest.raises((IsADirectoryError, PermissionError)):
            Model3D(str(tmp_path / "folder.glb"))


def test_a_parser_that_is_not_installed_is_not_taken_for_a_damaged_file(
    tmp_path, monkeypatch
):
    path = tmp_path / "box.glb"
    path.write_bytes(_box("glb"))
    missing = ImportError("No module named 'pygltflib'", name="pygltflib")

    def not_installed(*args, **kwargs):
        raise missing

    monkeypatch.setattr(trimesh, "load", not_installed)
    with Scene(), pytest.raises(ImportError) as raised:
        Model3D(str(path))
    assert raised.value is missing
