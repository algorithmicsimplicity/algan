"""Arena shadow copies preserve exact indices, masks and visibility layout."""

from __future__ import annotations

import pytest
import torch

from algan import SETTINGS
from algan.rendering.raytracing.shadow_queue import (
    _gather_shadow_payload,
    _scatter_shadow_visibility,
)
from algan.rendering.taichi_runtime import init_taichi
from algan.utils.memory_utils import ManualMemory


@pytest.mark.parametrize("with_footprint", [False, True])
@pytest.mark.parametrize("with_terminator", [False, True])
@pytest.mark.parametrize("order", [[], [3], [3, 0, 2]])
def test_shadow_gather_is_exact_and_scoped(with_footprint, with_terminator, order):
    init_taichi()
    device = SETTINGS.computing.render_device
    memory = ManualMemory(0, device=device, num_bytes=8192)
    memory._poison = 255
    position = torch.arange(12, dtype=torch.float32).view(4, 3)
    smooth = position * 0.1
    face = position * -0.5
    # These values catch integer->float->integer copying on MPS.
    frame = torch.tensor([2**25 + 1, 2**25 + 3, 1, 9], dtype=torch.int32)
    mask = torch.tensor([-2147221505, 0x7FFFFFFF, -1, 0x01000001], dtype=torch.int32)
    footprint = torch.arange(24, dtype=torch.float32).view(4, 6)
    terminator = position * 0.02
    if not with_footprint:
        footprint = torch.full((1, 6), float("nan"))
    if not with_terminator:
        terminator = torch.full((1, 3), float("nan"))
    inputs_cpu = (position, smooth, face, frame, mask, footprint, terminator)
    inputs = [value.to(device) for value in inputs_cpu]
    indices = torch.tensor(order, dtype=torch.int64, device=device)
    before = memory.get_pointers()
    with memory.temp():
        result = _gather_shadow_payload(
            memory,
            indices,
            *inputs,
            with_footprint=with_footprint,
            with_terminator=with_terminator,
        )
        for i, (actual, source) in enumerate(zip(result, inputs_cpu)):
            enabled = (
                i < 5 or (i == 5 and with_footprint) or (i == 6 and with_terminator)
            )
            if enabled:
                expected = source.index_select(0, indices.cpu())
                assert torch.equal(actual.cpu(), expected)
                assert (
                    actual.untyped_storage()._cdata
                    == memory.data.untyped_storage()._cdata
                )
            else:
                assert actual is inputs[i]
    assert memory.get_pointers() == before


@pytest.mark.parametrize("lights", [1, 3, 8])
@pytest.mark.parametrize("order", [[], [3], [3, 0, 2]])
def test_visibility_scatter_keeps_padding_and_unaccepted_rows(lights, order):
    init_taichi()
    device = SETTINGS.computing.render_device
    memory = ManualMemory(0, device=device, num_bytes=8192)
    destination = memory.get_tensor((5, 24), torch.float32)
    destination.fill_(1.0)
    indices = torch.tensor(order, dtype=torch.int64, device=device)
    source = (
        torch.arange(len(order) * lights * 3, dtype=torch.float32).view(
            len(order), lights, 3
        )
        / 100
    )
    expected = torch.ones(5, 24)
    for i, row in enumerate(order):
        expected[row, : lights * 3] = source[i].reshape(-1)
    assert (
        _scatter_shadow_visibility(destination, indices, source.to(device))
        is destination
    )
    assert torch.equal(destination.cpu(), expected)


@pytest.mark.parametrize("identity", [False, True])
@pytest.mark.parametrize("refit", [False, True])
def test_shared_trace_context_binds_current_trees_and_explicit_policies(
    monkeypatch, identity, refit
):
    from types import SimpleNamespace

    from algan.rendering.raytracing import raster_taichi
    from algan.rendering.raytracing.refit_bvh import RefitBVH
    from algan.rendering.raytracing.shadow_queue import (
        ShadowTraceContext,
        _ShadowPayload,
    )

    names = (
        "tri_pos",
        "tri_colors",
        "tri_uvs",
        "tri_tex_meta",
        "textures",
        "tri_extra",
        "circuit_meta",
        "circuit_colors",
        "circuit_border_colors",
        "edges_2d",
        "edge_accel",
        "tri_obj",
    )
    scene = dict.fromkeys(names)
    for name in names:
        scene[name] = object()
    scene["num_colored_triangles"] = 7
    context = ShadowTraceContext(scene, object(), object(), 3, object(), 1.75)

    def tree(is_refit):
        bvh = object.__new__(RefitBVH) if is_refit else SimpleNamespace()
        for field in ("blocks", "node_miss", "leaf_prim", "leaf_tspan"):
            setattr(bvh, field, object())
        bvh.first_leaf = 5 if is_refit else 11
        return bvh

    # The context precedes the build; each call must use the replacement tree,
    # including its concrete refit type, rather than capturing a placeholder.
    placeholder = tree(False)
    final = tree(refit)
    bezier = tree(False)
    payload = _ShadowPayload(
        object(), object(), object(), torch.empty(2), object(), object(), object()
    )
    source, visibility = object(), object()
    launches = []

    def capture(*args):
        assert len(args) == 48
        launches.append(
            dict(zip(raster_taichi._RASTER_SHADOW_TRACE_PARAMS, args, strict=True))
        )

    monkeypatch.setattr(raster_taichi, "raster_shadow_trace", capture)
    for triangle, samples in ((placeholder, 1), (final, 8)):
        context.trace(
            payload,
            triangle,
            bezier,
            visibility,
            samples=samples,
            shadow_mode=4,
            has_triangles=1,
            has_beziers=0,
            source_primitives=source,
            identity_enabled=identity,
            self_epsilon=1e-4,
            near_epsilon=2e-5,
            terminator_mode=2,
            adaptive_taps=1,
        )
    result = launches[-1]
    for field in names:
        if field != "tri_obj":
            assert result[field] is scene[field]
    for prefix, bvh in (("t", final), ("b", bezier)):
        assert result[f"{prefix}_nodes"] is bvh.blocks
        assert result[f"{prefix}_node_miss"] is bvh.node_miss
        assert result[f"{prefix}_leaf_prim"] is bvh.leaf_prim
        assert result[f"{prefix}_leaf_tspan"] is bvh.leaf_tspan
        assert result[f"{prefix}_first_leaf"] == bvh.first_leaf
    assert result["t_nodes"] is not launches[0]["t_nodes"]
    assert result["refit"] == int(refit)
    for name, value in {
        "event_pos": payload.position,
        "event_snrm": payload.smooth_normal,
        "event_fnrm": payload.face_normal,
        "event_frame": payload.frame,
        "event_msk": payload.mask,
        "event_dp": payload.footprint,
        "event_toff": payload.terminator,
        "shadow_vis": visibility,
        "event_src_prim": source,
        "tri_obj": scene["tri_obj"] if identity else source,
        "light_pos": context.light_position,
        "light_col": context.light_color,
        "pixel_world_scale": context.pixel_world_scale,
    }.items():
        assert result[name] is value
    for name, value in {
        "num_events": 2,
        "num_lights": 3,
        "num_colored_triangles": 7,
        "layer_offset_triangles": 1.75,
        "has_tri": 1,
        "has_bez": 0,
        "sec_aa": 8,
        "shadow_anyhit": 4,
        "shadow_identity": int(identity),
        "shadow_term": 2,
        "adaptive_taps": 1,
        "eps_self": 1e-4,
        "eps_near": 2e-5,
    }.items():
        assert result[name] == value
