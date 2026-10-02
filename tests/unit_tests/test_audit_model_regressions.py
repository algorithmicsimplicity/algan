"""Real glTF import and recorded playback regressions from the bug audit."""

import numpy as np
import pytest
import torch

from algan import Model3D, Scene
from algan.errors import UnsupportedFeatureError
from algan.mobs.three_d_models.scene_data import (
    AnimationData,
    MeshData,
    NodeAnimation,
    NodeData,
    SceneData,
)


@pytest.mark.parametrize("keys", [[0.0, 1.0], [0.0, 0.1, 1.0]])
@pytest.mark.parametrize("runtime", [1.0, 2.0])
def test_playback_keeps_key_timing_and_resets_each_lap(keys, runtime):
    times = torch.tensor(keys)
    positions = torch.zeros(len(times), 3)
    positions[:, 0] = times
    vertices = torch.tensor([[0.0, 0, 0], [1.0, 0, 0], [0.0, 1, 0]])
    data = SceneData(
        meshes=[MeshData(vertices=vertices, faces=torch.tensor([[0, 1, 2]]))],
        nodes=[NodeData(name="root", mesh_indices=[0])],
        animations=[
            AnimationData(
                name="move",
                runtime=1,
                channels=[
                    NodeAnimation(
                        node_name="root", position_times=times, positions=positions
                    ),
                ],
            )
        ],
    )
    with Scene() as scene:
        model = Model3D(scene_data=data).spawn(False)
        model.play_animation(runtime=runtime, fps=2, loop=2)
        assert scene.animation_manager.context.timespan.original_end == pytest.approx(
            2 * runtime
        )
        samples = torch.tensor([0.1, 0.5, 0.999, 1.0, 1.1, 1.5, 2.0]) * runtime
        scene.timeline_manager.set_state_to_times(samples)
        actual = model.mesh_mobs[0].grid.location[:, 0, 0]
        torch.testing.assert_close(
            actual,
            torch.tensor([0.1, 0.5, 0.999, 0.0, 0.1, 0.5, 1.0]),
            atol=2e-5,
            rtol=1e-5,
        )


def _write_glb(
    path, *, multi_primitive=False, interpolation="LINEAR", duplicate_names=False
):
    import pygltflib as g

    blob = bytearray()
    views, accessors = [], []

    def accessor(data, kind, component=5126):
        data = np.asarray(data, dtype=np.float32 if component == 5126 else np.uint16)
        while len(blob) % 4:
            blob.append(0)
        views.append(
            g.BufferView(buffer=0, byteOffset=len(blob), byteLength=data.nbytes)
        )
        blob.extend(data.tobytes())
        accessors.append(
            g.Accessor(
                bufferView=len(views) - 1,
                componentType=component,
                count=len(data),
                type=kind,
            )
        )
        return len(accessors) - 1

    primitives = []
    for index in range(2):
        vertices = [[index * 2.0, 0, 0], [index * 2.0 + 1, 0, 0], [index * 2.0, 1, 0]]
        faces = [0, 1, 2]
        if multi_primitive and index == 1:
            vertices.append([index * 2.0 + 1, 1, 0])
            faces += [1, 3, 2]
        primitives.append(
            g.Primitive(
                attributes=g.Attributes(POSITION=accessor(vertices, "VEC3")),
                indices=accessor(faces, "SCALAR", 5123),
                material=index,
            )
        )
    times = accessor([0, 1], "SCALAR")
    values = [[0, 0, 0], [0, 3, 0]]
    if interpolation == "CUBICSPLINE":
        values = [[0, 0, 0], values[0], [0, 0, 0], [0, 0, 0], values[1], [0, 0, 0]]
    poses = accessor(values, "VEC3")
    meshes = (
        [g.Mesh(name="parts", primitives=primitives)]
        if multi_primitive
        else [
            g.Mesh(name="same" if duplicate_names else f"part-{i}", primitives=[p])
            for i, p in enumerate(primitives)
        ]
    )
    model = g.GLTF2(
        scene=0,
        scenes=[g.Scene(nodes=[0])],
        nodes=[g.Node(name="parent", children=list(range(1, len(meshes) + 1)))]
        + [g.Node(name=f"child-{i}", mesh=i) for i in range(len(meshes))],
        meshes=meshes,
        accessors=accessors,
        bufferViews=views,
        buffers=[g.Buffer(byteLength=len(blob))],
        materials=[
            g.Material(pbrMetallicRoughness=g.PbrMetallicRoughness(baseColorFactor=c))
            for c in ([1, 0, 0, 1], [0, 1, 0, 1])
        ],
        animations=[
            g.Animation(
                name="move",
                samplers=[
                    g.AnimationSampler(
                        input=times, output=poses, interpolation=interpolation
                    )
                ],
                channels=[
                    g.AnimationChannel(
                        sampler=0,
                        target=g.AnimationChannelTarget(node=0, path="translation"),
                    )
                ],
            )
        ],
    )
    model.set_binary_blob(bytes(blob))
    if path.suffix == ".gltf":
        model.buffers[0].uri = "mesh.bin"
        path.with_name("mesh.bin").write_bytes(blob)
        model.save_json(str(path))
    else:
        model.save_binary(str(path))


@pytest.mark.parametrize("multi_primitive", [False, True])
@pytest.mark.parametrize("duplicate_names", [False, True])
@pytest.mark.parametrize("extension", ["glb", "gltf"])
def test_every_imported_primitive_inherits_animated_parent(
    tmp_path, multi_primitive, duplicate_names, extension
):
    path = tmp_path / f"parent.{extension}"
    _write_glb(path, multi_primitive=multi_primitive, duplicate_names=duplicate_names)
    original_bytes = path.read_bytes()
    with Scene():
        model = Model3D(str(path))
        assert len(model.mesh_mobs) == 2
        assert sum(len(mesh.faces) for mesh in model.scene_data.meshes) == (
            3 if multi_primitive else 2
        )
        assert model.animation_names == ["move"]
        _, frames = model.precompute_animation(times=[0.0, 1.0])
        for mesh, corners in frames.items():
            assert mesh._node_idx >= 0
            torch.testing.assert_close(
                corners[1] - corners[0],
                torch.tensor([0.0, 3.0, 0.0]).expand_as(corners[0]),
            )
    assert path.read_bytes() == original_bytes


@pytest.mark.parametrize("interpolation", ["STEP", "CUBICSPLINE"])
def test_unsupported_gltf_interpolation_is_never_silently_reinterpreted(
    tmp_path, interpolation
):
    path = tmp_path / "unsupported.glb"
    _write_glb(path, interpolation=interpolation)
    with pytest.raises(UnsupportedFeatureError, match=interpolation):
        Model3D(str(path))
