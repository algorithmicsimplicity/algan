"""Exact preprocessing destinations and their pre-reduction scratch lifetime."""

from __future__ import annotations

import pytest
import torch

from algan.rendering import mps_compat
from algan.rendering.raytracing import array_copy_taichi, sheets
from algan.rendering.raytracing.array_ops import gather_frame_table
from algan.rendering.raytracing.sheet_preprocessing import (
    FragmentMetadata,
    fragment_metadata,
)
from tests.unit_tests.test_sheet_compaction import _coverage
from tests.unit_tests.test_sheet_fragment_workspace import (
    _bits_equal,
    _poison,
    _workspace,
)


def _strided(value):
    storage = torch.empty((*value.shape, 2), dtype=value.dtype, device=value.device)
    storage[..., 0].copy_(value)
    return storage[..., 0]


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.float32])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("owned", [False, True])
@pytest.mark.parametrize("size", [0, 1, 5])
def test_frame_table_lookup_is_exact_and_releases_indices(dtype, strided, owned, size):
    memory, ws = _workspace()
    table = torch.arange(28, device=ws.device).reshape(4, 7)
    if dtype == torch.float32:
        table = (table + 0x7FC00001).to(torch.int32).view(torch.float32)
    else:
        table = (table + (2**40 if dtype == torch.int64 else 2**24)).to(dtype)
    frames = torch.tensor([-3, 0, 2, 4, 8], dtype=torch.int32, device=ws.device)[:size]
    columns = torch.tensor([6, 2, 1, 5, 0], device=ws.device)[:size]
    if strided:
        table, frames, columns = map(_strided, (table, frames, columns))
    expected = table[(frames.to(torch.int64) + 3) % 4, columns].clone()
    with ws.stage():
        out = ws.tensor((size,), dtype) if owned else None
        floor = memory.current_pointer
        result = gather_frame_table(
            table, frames, columns, time_start=3, out=out, workspace=ws
        )
        assert not owned or result is out
        assert memory.current_pointer == floor
        _poison(memory)
        assert _bits_equal(result, expected)
    assert memory.current_pointer == ws._depth == ws._live_bytes == 0


@pytest.mark.parametrize("local", [False, True])
def test_frame_lookup_uses_exact_mps_integer_copy_policy(monkeypatch, local):
    from algan.rendering import taichi_runtime

    memory, ws = _workspace()
    monkeypatch.setattr(mps_compat, "mps_friendly", lambda: True)
    calls = []
    original = array_copy_taichi.gather_rows_into

    def spy(*args):
        calls.append(True)
        return original(*args)

    monkeypatch.setattr(array_copy_taichi, "gather_rows_into", spy)
    if not local:
        monkeypatch.setattr(taichi_runtime, "_live_arch", lambda: None)
    table = torch.tensor([[2**45 + 3, -(2**45) + 7], [2**30, -17]], device=ws.device)
    frames = torch.tensor([0, 1], device=ws.device)
    columns = torch.tensor([1, 0], device=ws.device)
    with ws.stage():
        out = ws.tensor((2,), torch.int64)
        gather_frame_table(table, frames, columns, out=out, workspace=ws)
        _poison(memory)
        assert torch.equal(out, table[frames, columns])
    assert bool(calls) is local


@pytest.mark.parametrize(
    "problem",
    [
        "shape",
        "dtype",
        "stride",
        "table_alias",
        "frame_alias",
        "column_alias",
        "device",
        "frames",
        "columns",
        "table",
        "workspace",
    ],
)
def test_frame_lookup_checks_before_writing(problem):
    memory, ws = _workspace()
    table = torch.arange(16, device=ws.device).reshape(4, 4)
    frames = torch.tensor([0, 1, 2, 3], device=ws.device)
    cols = torch.tensor([3, 2, 1, 0], device=ws.device)
    out = torch.full((4,), 97, dtype=torch.int64, device=ws.device)
    if problem == "shape":
        out = out[:3]
    elif problem == "dtype":
        out = out.to(torch.int32)
    elif problem == "stride":
        out = _strided(out)
    elif problem == "table_alias":
        out = table[0]
    elif problem == "frame_alias":
        out = frames
    elif problem == "column_alias":
        out = cols
    elif problem == "device":
        out = torch.empty(4, dtype=torch.int64, device="meta")
    elif problem == "frames":
        frames = frames.float()
    elif problem == "columns":
        cols = cols[:3]
    elif problem == "table":
        table = table.view(16)
    elif problem == "workspace":
        from algan.rendering.raytracing.sheet_workspace import CompactionWorkspace

        ws = CompactionWorkspace(device="meta")
    tensors = [x for x in (table, frames, cols, out) if x.device.type != "meta"]
    snapshots = [x.clone() for x in tensors]
    with pytest.raises(ValueError):
        gather_frame_table(table, frames, cols, out=out, workspace=ws)
    assert all(_bits_equal(x, y) for x, y in zip(tensors, snapshots))
    assert memory.current_pointer == 0


def _metadata_inputs(device, dtype=torch.int64, size=4, strided=False):
    pixel = torch.tensor([1, 17, 33, 49], device=device)[:size]
    bits = torch.tensor(
        [0x80000000, 0x7FC00023, 0x7F800000, 0x3F800001], device=device
    )[:size]
    keys = (pixel << 32) | bits
    refs = torch.tensor([0, 1, -2, 2], dtype=torch.int32, device=device)[:size]
    masks = torch.tensor(
        [0, sheets.AA_BACKFACE_BIT, 1, sheets.AA_BACKFACE_BIT],
        dtype=torch.int32,
        device=device,
    )[:size]
    positions = torch.arange(size, device=device)
    table = torch.tensor([[2**30, 3, 9], [8, 7, 2**30 + 5]], dtype=dtype, device=device)
    if dtype == torch.int64:
        table += 2**40
    values = (keys, refs, masks, positions, table)
    return tuple(map(_strided, values)) if strided else values


def _metadata_oracle(keys, refs, masks, positions, table):
    pixel = keys >> 32
    depth = (keys & 0xFFFFFFFF).to(torch.int32).view(torch.float32)
    frame = pixel // 16
    triangle = refs >= 0
    reference = refs.clamp_min(0).to(torch.int64)
    surface = table[(frame + 1) % table.shape[0], reference].to(torch.int64)
    facing = ((masks & sheets.AA_BACKFACE_BIT) != 0).to(torch.int64)
    group = torch.where(triangle, surface * 2 + facing, -(positions + 2))
    return FragmentMetadata(pixel, depth, frame, triangle, reference, group)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("owned", [False, True])
@pytest.mark.parametrize("size", [0, 1, 4])
def test_metadata_preserves_packed_bits_and_widens_surfaces(
    dtype, strided, owned, size
):
    memory, ws = _workspace()
    values = _metadata_inputs(ws.device, dtype, size, strided)
    expected = _metadata_oracle(*values)
    with ws.stage():
        out = FragmentMetadata.allocate(ws, size) if owned else None
        floor = memory.current_pointer
        actual = fragment_metadata(*values, 16, 1, out=out, workspace=ws)
        assert not owned or actual is out
        assert memory.current_pointer == floor
        _poison(memory)
        assert all(_bits_equal(a, b) for a, b in zip(actual, expected))
    assert ws._depth == ws._live_bytes == memory.current_pointer == 0


@pytest.mark.parametrize(
    "problem",
    [
        "shape",
        "dtype",
        "stride",
        "input_alias",
        "output_alias",
        "device",
        "source",
        "table",
        "pixels",
    ],
)
def test_metadata_validates_all_outputs_before_mutation(problem):
    memory, ws = _workspace()
    values = list(_metadata_inputs(ws.device))
    with ws.stage():
        out = list(FragmentMetadata.allocate(ws, 4))
        for field in out:
            field.fill_(1)
        if problem == "shape":
            out[-1] = out[-1][:3]
        elif problem == "dtype":
            out[-1] = out[-1].float()
        elif problem == "stride":
            out[-1] = _strided(out[-1])
        elif problem == "input_alias":
            out[-1] = values[0]
        elif problem == "output_alias":
            out[-1] = out[0]
        elif problem == "device":
            out[-1] = torch.empty(4, dtype=torch.int64, device="meta")
        elif problem == "source":
            values[1] = values[1].float()
        elif problem == "table":
            values[-1] = values[-1].float()
        tensors = [x for x in (*values, *out) if x.device.type != "meta"]
        snapshots = [x.clone() for x in tensors]
        floor = memory.current_pointer
        with pytest.raises(ValueError):
            fragment_metadata(
                *values,
                0 if problem == "pixels" else 16,
                1,
                out=FragmentMetadata(*out),
                workspace=ws,
            )
        assert all(_bits_equal(a, b) for a, b in zip(tensors, snapshots))
        assert memory.current_pointer == floor


@pytest.mark.parametrize("method", ["class", "prim"])
@pytest.mark.parametrize("owned", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_preprocessing_tables_release_workspace(method, owned, empty):
    memory, ws = _workspace()
    size = 0 if empty else 4
    frame = torch.arange(size, device=ws.device) % 2
    ref = torch.arange(size, device=ws.device) % 3
    tri = torch.ones(size, dtype=torch.bool, device=ws.device)
    normals = torch.tensor([[[0.0, 0.0, 1.0] * 3] * 3], device=ws.device)
    pos = torch.arange(27, dtype=torch.float32, device=ws.device).view(1, 3, 9) / 10
    merged = {"tri_norm": normals, "tri_pos": pos}
    if method == "class":
        fn = sheets._shade_class
        args = (merged, frame, 1, ref, tri, bool(size), 2)
        shape, dtype = (size,), torch.int64
    else:
        fn = sheets._prim_split_after
        depth = torch.arange(size, dtype=torch.float32, device=ws.device)
        order = torch.arange(size, device=ws.device)
        args = (
            merged,
            torch.zeros(1, 3, device=ws.device),
            torch.ones(1, device=ws.device),
            None,
            frame,
            1,
            ref,
            tri,
            depth,
            depth.clone(),
            order,
            2.0,
            2,
        )
        shape, dtype = (max(0, size - 1),), torch.bool
    expected = fn(*args)
    with ws.stage():
        out = ws.tensor(shape, dtype) if owned else None
        floor = memory.current_pointer
        result = fn(*args, out=out, workspace=ws)
        assert not owned or result is out
        assert memory.current_pointer == floor
        _poison(memory)
        assert torch.equal(result, expected)


@pytest.mark.parametrize("method", ["class", "prim"])
@pytest.mark.parametrize(
    "problem", ["dtype", "shape", "stride", "input_alias", "device"]
)
def test_preprocessing_rule_outputs_are_checked(method, problem):
    memory, ws = _workspace()
    frame = torch.tensor([0, 0, 0, 0], device=ws.device)
    ref = frame.clone()
    tri = torch.ones(4, dtype=torch.bool, device=ws.device)
    merged = {
        "tri_norm": torch.zeros(1, 1, 9, device=ws.device),
        "tri_pos": torch.ones(1, 1, 9, device=ws.device),
    }
    if method == "class":
        fn, args = sheets._shade_class, (merged, frame, 0, ref, tri)
        out = torch.full((4,), 11, device=ws.device)
        alias = frame
    else:
        depth = torch.ones(4, device=ws.device)
        fn, args = (
            sheets._prim_split_after,
            (
                merged,
                torch.zeros(1, 3, device=ws.device),
                depth[:1],
                None,
                frame,
                0,
                ref,
                tri,
                depth,
                depth,
                ref,
                2.0,
            ),
        )
        out, alias = torch.ones(3, dtype=torch.bool, device=ws.device), tri[:3]
    if problem == "dtype":
        out = out.float()
    elif problem == "shape":
        out = out[:-1]
    elif problem == "stride":
        out = _strided(out)
    elif problem == "input_alias":
        out = alias
    else:
        out = torch.empty_like(out, device="meta")
    snapshot = None if out.device.type == "meta" else out.clone()
    with pytest.raises(ValueError):
        fn(*args, out=out, workspace=ws)
    if snapshot is not None:
        assert _bits_equal(out, snapshot)
    assert memory.current_pointer == 0


def test_decoding_storage_is_reused_by_rank_destination(monkeypatch):
    memory, ws = _workspace()
    coverage, merged, cam, pws = _coverage(
        [(0, 1.0, 0, 0.4, 15), (0, 1.001, 1, 0.6, 240)]
    )
    coverage = {
        k: v.to(ws.device) if torch.is_tensor(v) else v for k, v in coverage.items()
    }
    merged = {k: v.to(ws.device) for k, v in merged.items()}
    captured = []
    decode, rank = sheets.fragment_metadata, sheets._sheet_rank_groups

    def capture(*a, **kw):
        result = decode(*a, **kw)
        captured.append(result.pixel.data_ptr())
        return result

    def assert_reuse(*a, **kw):
        assert kw["out"].data_ptr() == captured[-1]
        return rank(*a, **kw)

    monkeypatch.setattr(sheets, "fragment_metadata", capture)
    monkeypatch.setattr(sheets, "_sheet_rank_groups", assert_reuse)
    result = sheets.compact_sheets(
        coverage,
        merged,
        cam.to(ws.device),
        pws.to(ws.device),
        0,
        4,
        4,
        diagnostics=False,
        resolver_memory=memory,
        workspace=ws,
    )
    assert captured
    _poison(memory)
    assert result.num_sheets == 1
    assert torch.equal(result.sheet_cov, torch.tensor([1.0], device=ws.device))


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("with_workspace", [False, True])
def test_pixel_sort_accepts_preprocessing_destination(
    monkeypatch, native, with_workspace
):
    memory, ws = _workspace()
    monkeypatch.setattr(sheets, "sheet_pixel_sort", native)
    monkeypatch.setattr(sheets, "sheet_packed_sort", False)
    pixel = torch.tensor([0, 0, 0, 1, 1], device=ws.device)
    group = torch.tensor([1, 0, 1, 3, 2], device=ws.device)
    depth = torch.tensor([1.0, 2.0, 1.0, 0.0, 2.0], device=ws.device)
    offsets = torch.tensor([0, 3, 5], dtype=torch.int32, device=ws.device)
    with ws.stage():
        out = ws.tensor((5,), torch.int64)
        floor = memory.current_pointer
        result = sheets._pixel_group_order(
            pixel,
            group,
            depth,
            offsets,
            out=out,
            workspace=ws if with_workspace else None,
        )
        assert result is out
        assert memory.current_pointer == floor
        _poison(memory)
        assert result.tolist() == [1, 0, 2, 4, 3]


@pytest.mark.parametrize(
    "problem", ["shape", "dtype", "stride", "input_alias", "device"]
)
def test_pixel_sort_checks_early_destination(problem):
    memory, ws = _workspace()
    pixel = torch.tensor([0, 0, 0], device=ws.device)
    group = torch.tensor([1, 0, 2], device=ws.device)
    depth = torch.ones(3, device=ws.device)
    out = torch.full((3,), 113, device=ws.device)
    if problem == "shape":
        out = out[:2]
    elif problem == "dtype":
        out = out.float()
    elif problem == "stride":
        out = _strided(out)
    elif problem == "input_alias":
        out = group
    else:
        out = torch.empty(3, dtype=torch.int64, device="meta")
    snapshot = None if out.device.type == "meta" else out.clone()
    with pytest.raises(ValueError):
        sheets._pixel_group_order(pixel, group, depth, None, out=out, workspace=ws)
    if snapshot is not None:
        assert torch.equal(out, snapshot)
    assert memory.current_pointer == 0
