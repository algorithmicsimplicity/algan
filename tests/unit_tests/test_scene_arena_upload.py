import pytest
import torch

from algan.rendering.raytracing import settings as rt_settings
from algan.rendering.raytracing.scene_builder import (
    _prefill_background,
    _projected_scene_device,
    copy_merged_scene_to_arena,
    get_merged_scene_arena_nbytes,
    get_merged_scene_tensor_nbytes,
)
from algan.rendering.raytracing.stbvh import BVH_ARITY, STBVH
from algan.utils.memory_utils import ManualMemory


def _prebuilt_bvh():
    num_leaves = BVH_ARITY
    num_nodes = (BVH_ARITY * num_leaves - 1) // (BVH_ARITY - 1)
    first_leaf = num_nodes - num_leaves
    nodes = torch.arange(num_nodes * 8, dtype=torch.float32).reshape(num_nodes, 8)
    blocks = torch.arange(first_leaf * 8 * BVH_ARITY, dtype=torch.float16).reshape(
        first_leaf, 8, BVH_ARITY
    )
    node_miss = torch.arange(num_nodes, dtype=torch.int32)
    leaf_prim = torch.arange(num_leaves, dtype=torch.int32)
    leaf_tspan = torch.arange(num_leaves, dtype=torch.int32) << 16
    return STBVH.from_prebuilt(nodes, node_miss, leaf_prim, leaf_tspan, blocks)


def _cpu_arena():
    # CPU availability is a fixed 2 GiB budget in ManualMemory; this reserves
    # roughly 2 MiB without relying on host free-memory telemetry.
    return ManualMemory(0.001, device=torch.device("cpu"), managed=True)


def test_scene_arena_upload_preserves_aliases_values_and_exact_bytes(monkeypatch):
    bvh = _prebuilt_bvh()
    base = torch.arange(24, dtype=torch.float32)
    strided = base[2:18:2]
    matrix_view = base.view(4, 6)[:, 1:5]
    scene = {
        "tri_bvh": bvh,
        "tri_opaque_bvh": bvh,
        "nodes_alias": bvh.nodes,
        "strided": strided,
        "strided_alias": strided,
        "matrix_view": matrix_view,
        "metadata": {"num_frames": 3, "enabled": True},
    }

    # The raw count charges each source storage once: five independent BVH
    # fields plus ``base`` (despite its two different views and aliases).
    expected_raw = sum(
        tensor.untyped_storage().nbytes()
        for tensor in (
            bvh.nodes,
            bvh.blocks,
            bvh.node_miss,
            bvh.leaf_prim,
            bvh.leaf_tspan,
            base,
        )
    )
    assert get_merged_scene_tensor_nbytes(scene) == expected_raw

    memory = _cpu_arena()
    # Make the reverse pointer deliberately unaligned so the exact accounting
    # assertion exercises per-storage padding as well as payload bytes.
    memory.get_tensor((1,), dtype=torch.uint8, persist=True)
    before = memory.get_pointers()
    expected_arena = get_merged_scene_arena_nbytes(scene, memory)

    # Uploading an STBVH must attach its existing blocks, never rebuild them on
    # the destination device.
    def fail_build(*_args, **_kwargs):
        raise AssertionError("destination unexpectedly rebuilt BVH blocks")

    monkeypatch.setattr("algan.rendering.raytracing.stbvh._build_blocks", fail_build)
    uploaded = copy_merged_scene_to_arena(scene, memory)
    after = memory.get_pointers()

    assert before[0] == after[0]
    assert before[1] - after[1] == expected_arena
    assert uploaded["tri_bvh"] is uploaded["tri_opaque_bvh"]
    assert uploaded["tri_bvh"].nodes is uploaded["nodes_alias"]
    assert uploaded["strided"] is uploaded["strided_alias"]
    assert uploaded["metadata"] == scene["metadata"]

    for field in ("nodes", "blocks", "node_miss", "leaf_prim", "leaf_tspan"):
        assert torch.equal(getattr(uploaded["tri_bvh"], field), getattr(bvh, field))
    assert uploaded["tri_bvh"].first_leaf == bvh.first_leaf
    assert uploaded["tri_bvh"].num_leaves == bvh.num_leaves
    assert torch.equal(uploaded["strided"], strided)
    assert torch.equal(uploaded["matrix_view"], matrix_view)

    # The two views keep their offset relationship within their copied storage.
    source_delta = matrix_view.storage_offset() - strided.storage_offset()
    uploaded_delta = (
        uploaded["matrix_view"].storage_offset() - uploaded["strided"].storage_offset()
    )
    assert uploaded_delta == source_delta

    arena_storage = memory.data.untyped_storage()._cdata
    for tensor in (
        uploaded["tri_bvh"].nodes,
        uploaded["tri_bvh"].blocks,
        uploaded["tri_bvh"].node_miss,
        uploaded["tri_bvh"].leaf_prim,
        uploaded["tri_bvh"].leaf_tspan,
        uploaded["strided"],
        uploaded["matrix_view"],
    ):
        assert tensor.device.type == "cpu"
        assert tensor.untyped_storage()._cdata == arena_storage

    # Idempotence is by identity and consumes no additional arena bytes.
    stable_pointers = memory.get_pointers()
    assert copy_merged_scene_to_arena(uploaded, memory) is uploaded
    assert memory.get_pointers() == stable_pointers
    assert get_merged_scene_arena_nbytes(uploaded, memory) == 0


def test_projected_scene_device_comes_from_primitive_tensor():
    class Primitive:
        pass

    primitive = Primitive()
    primitive._rt_num_frames = 2
    primitive._rt_tri_pos = torch.zeros((2, 1, 9), device="cpu")
    assert _projected_scene_device([primitive]) == torch.device("cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_unmanaged_scene_upload_keeps_regular_copy_fallback_and_aliases():
    base = torch.arange(16, dtype=torch.float32)
    view = base[2:12:2]
    scene = {"base": base, "view": view, "view_alias": view}
    memory = ManualMemory(0, device="cuda", managed=False)

    uploaded = copy_merged_scene_to_arena(scene, memory)

    assert uploaded["view"] is uploaded["view_alias"]
    assert uploaded["base"].device.type == "cuda"
    assert uploaded["base"].untyped_storage()._cdata == (
        uploaded["view"].untyped_storage()._cdata
    )
    assert torch.equal(uploaded["base"].cpu(), base)
    assert torch.equal(uploaded["view"].cpu(), view)


def _reference_to_linear(c):
    """The sRGB EOTF, written out from the specification.

    Not imported from ``algan.utils.color_space``, so this states what the
    ingest owes rather than agreeing with the renderer's own transcription of
    the standard.
    """
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


@pytest.mark.parametrize("linear", [False, True], ids=["display", "linear"])
def test_prefill_background_copies_and_casts_directly_into_arena(linear):
    """The arena mechanics, in whichever colour space the renderer composites in.

    A background is authored display-referred either way. Under the linear
    working space it is decoded on the way in -- it composites against geometry
    that has already been decoded, so it has to arrive in the same space --
    while the display-referred space copies the authored value through. What
    both arms check is the same: the copy casts straight into the reserved
    destination, a nonzero frame offset selects the right rows, and a missing
    channel is filled from the source's last one.
    """
    previous = rt_settings.LINEAR_COLOR_SPACE
    rt_settings.set_linear_color_space(linear)
    try:
        memory = _cpu_arena()

        # Under the linear space the destination is the float HDR buffer, never
        # a byte one: ``_prefill_background`` deliberately does not round a
        # linear value onto the byte grid (that is what would crush the darks),
        # and the tracer makes the float buffer a precondition of the space.
        solid_out = memory.get_tensor(
            (2, 3, 5), dtype=torch.float32 if linear else torch.uint8
        )
        _prefill_background(
            solid_out, torch.tensor([0.0, 0.5, 1.0]), 0, torch.device("cpu")
        )
        authored = [0.0, 0.5, 1.0]
        if linear:
            # 0.5 decodes to 0.21404 -- the anchor anyone can look up.
            channels = [255 * _reference_to_linear(c) for c in authored]
            expected_solid = torch.tensor(channels + channels[-1:] * 2)
            assert torch.allclose(
                solid_out, expected_solid.expand_as(solid_out), atol=1e-3
            )
        else:
            channels = [round(255 * c) for c in authored]
            expected_solid = torch.tensor(
                channels + channels[-1:] * 2, dtype=torch.uint8
            )
            assert torch.equal(solid_out, expected_solid.expand_as(solid_out))

        # Animated/image backgrounds carry one padding row followed by flattened
        # frame/pixel rows. Exercise a nonzero frame offset, uint8 -> float32 copy,
        # and missing-channel fill from the source's final channel.
        rows = torch.tensor(
            [
                [99, 99, 99],  # leading padding row
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
                [10, 11, 12],
                [13, 14, 15],
                [16, 17, 18],
            ],
            dtype=torch.uint8,
        )
        animated_out = memory.get_tensor((2, 2, 4), dtype=torch.float32)
        _prefill_background(animated_out, rows, 1, torch.device("cpu"))

        def colour(byte):
            """An image background is 8-bit sRGB like any other texture."""
            return 255 * _reference_to_linear(byte / 255.0) if linear else byte

        # The fourth channel is the missing-channel fill, and it is a raw copy
        # of the source's last channel in both spaces: it stands in for alpha,
        # which is a coverage weight rather than a colour, so it is not decoded.
        expected_animated = torch.tensor(
            [
                [[*map(colour, (7, 8, 9)), 9], [*map(colour, (10, 11, 12)), 12]],
                [[*map(colour, (13, 14, 15)), 15], [*map(colour, (16, 17, 18)), 18]],
            ],
            dtype=torch.float32,
        )
        assert torch.allclose(animated_out, expected_animated, atol=1e-3)
        assert (
            animated_out.untyped_storage()._cdata
            == memory.data.untyped_storage()._cdata
        )
    finally:
        rt_settings.set_linear_color_space(previous)
