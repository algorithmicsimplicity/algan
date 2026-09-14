"""Shared bounds and scoped ray state preserve the renderer's existing contracts."""

from __future__ import annotations

import math

import pytest
import torch

from algan import SETTINGS
from algan.rendering.raytracing import raster_pipeline, tracer
from algan.rendering.raytracing.scene_bounds import triangle_scene_bounds
from algan.rendering.raytracing.wavefront_state import RayState
from algan.utils.memory_utils import ManualMemory


@pytest.mark.parametrize("kind", ["finite", "empty", "nan", "infinite", "degenerate"])
def test_shared_bounds_preserve_original_extrema_and_shadow_norm(kind):
    device = SETTINGS.computing.render_device
    positions = (
        torch.arange(54, dtype=torch.float32).reshape(2, 3, 9).to(device) * 0.137
    )
    if kind == "empty":
        positions = positions[:, :0]
    elif kind == "nan":
        positions[0, 0, 0] = float("nan")
    elif kind == "infinite":
        positions[0, 0, 0] = float("inf")
    elif kind == "degenerate":
        positions.fill_(0.375)
    merged = {"tri_pos": positions}
    bounds = triangle_scene_bounds(merged)
    if positions.numel():
        vertices = positions.reshape(-1, 3, 3)
        lo, hi = vertices.amin((0, 1)), vertices.amax((0, 1))
        diagonal = (hi - lo).norm().item()
        assert torch.allclose(torch.tensor(bounds.lower), lo.cpu(), equal_nan=True)
        assert torch.allclose(torch.tensor(bounds.upper), hi.cpu(), equal_nan=True)
        assert bounds.diagonal == diagonal or (
            math.isnan(bounds.diagonal) and math.isnan(diagonal)
        )
    else:
        assert bounds.lower == (0.0, 0.0, 0.0)
        assert bounds.upper == (1.0, 1.0, 1.0)
        assert bounds.diagonal == 0.0
    assert triangle_scene_bounds(merged) is bounds
    self_eps, near_eps = raster_pipeline._shadow_identity_epsilons(merged)
    assert math.isfinite(self_eps)
    assert self_eps > 0.0
    assert near_eps >= 0.0
    assert merged["_triangle_scene_bounds"] is bounds


@pytest.mark.parametrize("persist", [False, True])
@pytest.mark.parametrize("flags", [False, True])
def test_shared_screen_bound_packing_matches_original_rules(persist, flags):
    SETTINGS.raytracing.experimental.set(raster_pair_flags=flags)
    device = SETTINGS.computing.render_device
    memory = ManualMemory(0, device=device, num_bytes=8192)
    memory._poison = 255
    xmin = torch.tensor([[-9.0, -0.5, 1.1, 7.0], [12.0, -3.0, 2.0, 8.0]], device=device)
    xmax, ymin, ymax = xmin + 2.25, xmin * 0.5, xmin * 0.5 + 1.125
    bounded = torch.tensor(
        [[True, True, False, False], [True, False, True, False]], device=device
    )
    front = torch.tensor(
        [[True, True, True, False], [True, False, True, True]], device=device
    )
    valid = torch.tensor(
        [[True, True, True, False], [True, True, False, True]], device=device
    )
    opaque = torch.tensor(
        [[True, False, True, False], [False, True, False, False]], device=device
    )
    width = 8
    expected_x = torch.stack(
        (
            torch.where(bounded, (xmin - 1).floor().clamp(0, width - 1).long(), 0),
            torch.where(
                bounded, (xmax + 1).ceil().clamp(0, width - 1).long(), width - 1
            ),
        ),
        -1,
    )
    expected_f = torch.stack(((ymin - 1).floor(), (ymax + 1).ceil(), ymin, ymax), -1)
    x_on = (xmax >= -1.0) & (xmin <= width + 1.0)
    expected_m = torch.stack(
        (bounded, bounded & x_on, ~bounded & front, opaque, valid & ~opaque), -1
    )
    before = memory.get_pointers()
    actual = raster_pipeline._pack_screen_bounds(
        xmin,
        xmax,
        ymin,
        ymax,
        bounded,
        front,
        valid,
        opaque,
        width,
        memory,
        persist=persist,
    )
    for output, reference in zip(actual[:3], (expected_f, expected_x, expected_m)):
        assert torch.equal(output.cpu(), reference.cpu())
        assert output.untyped_storage()._cdata == memory.data.untyped_storage()._cdata
    reach = expected_m[..., 1] | expected_m[..., 2]
    expected_flags = torch.stack(
        ((opaque & reach).any(1), (valid & ~opaque & reach).any(1)), -1
    )
    assert actual[3] == (expected_flags.cpu().tolist() if flags else None)
    if persist:
        assert memory.current_pointer == before[0]
    else:
        assert memory.current_reverse_pointer == before[1]


@pytest.mark.parametrize("global_hits", [False, True])
def test_named_ray_state_retains_tuple_abi_and_sparse_accumulator_slot(global_hits):
    memory = ManualMemory(0, device="cpu", num_bytes=8192)
    state = tracer._alloc_wavefront_state(memory, 3, 7, global_hits=global_hits)
    assert isinstance(state, RayState)
    assert isinstance(state, tuple)
    assert len(state) == 11
    assert state.origin is state[0]
    assert state.scalars is state[3]
    assert state.integers is state[4]
    state.integers[:, 4].fill_(123)
    assert state[4][:, 4].tolist() == [123, 123, 123]
    if not global_hits:
        assert state.hit_distance is state.hit_u is state.hit_v
        assert state.hit_primitive is state.hit_flags


@pytest.mark.parametrize("failure", [None, "drain", "readback", "composite"])
def test_classic_tile_attempt_restores_both_arena_ends_on_every_exit(
    monkeypatch, failure
):
    memory = ManualMemory(0, device="cpu", num_bytes=16384)
    out = memory.get_tensor((1, 4, 4), torch.float32)
    marker = memory.get_tensor((3,), torch.uint8, persist=True)
    marker.fill_(171)
    before = memory.get_pointers()
    entries = []

    def fail_or_pass(stage):
        if stage == failure:
            raise LookupError(f"injected {stage} failure")

    def run_tile(*args):
        entries.append(memory.get_pointers())
        memory.get_tensor((7,), torch.float32, persist=True).fill_(0)
        fail_or_pass("drain")

    def readback(_):
        fail_or_pass("readback")
        return [0] * tracer.ALLOC_WIDTH

    def composite(*args):
        fail_or_pass("composite")

    monkeypatch.setattr(tracer, "_read_tile_alloc", readback)
    monkeypatch.setattr(tracer, "_record_tile_truncations", lambda *args: None)
    monkeypatch.setattr(tracer, "wf_composite_accum", composite)
    monkeypatch.setattr(tracer, "_auto_primary_per_tile", lambda *args: 2)
    kwargs = {
        "n": 4,
        "width": 4,
        "height": 1,
        "time_start": 0,
        "transparent": False,
        "aa_level": 1,
        "pool_ratio": 1,
        "primary_per_tile": 2,
        "cam_origin": None,
        "screen_point": None,
        "pixel_basis_x": None,
        "pixel_basis_y": None,
        "half_screen_w": 2.0,
        "half_screen_h": 0.5,
        "max_bounces": 0,
        "near_clip": 0.0,
        "run_tile": run_tile,
        "gen_fused": True,
        "global_hits": False,
    }
    if failure is None:
        tracer._run_wavefront_tiles(memory, out, **kwargs)
        assert len(entries) == 2
        assert entries[0] == entries[1]
    else:
        with pytest.raises(LookupError, match=f"injected {failure}"):
            tracer._run_wavefront_tiles(memory, out, **kwargs)
    assert memory.get_pointers() == before
    assert marker.tolist() == [171, 171, 171]
