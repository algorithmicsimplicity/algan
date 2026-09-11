"""Closed-shell metadata, exact arithmetic and scratch/result ownership."""

from __future__ import annotations

import pytest
import torch

from algan.rendering import mps_compat
from algan.rendering.raytracing import sheet_compact_taichi, sheet_shells, sheets
from algan.rendering.raytracing.sheet_shells import (
    ShellSegments,
    apply_shell_ceiling,
    shell_segments,
)
from algan.rendering.raytracing.sheet_workspace import CompactionWorkspace
from tests.unit_tests.test_sheet_fragment_workspace import (
    _bits_equal,
    _poison,
    _workspace,
)
from tests.unit_tests.test_sheet_preprocessing_workspace import _strided


def _metadata(device, count=6, *, wide=False, animated=True, strided=False):
    pixel = torch.tensor([0, 0, 2, 2, 3, 3], device=device)[:count]
    mask = torch.tensor(
        [1, sheets.AA_BACKFACE_BIT, 0, sheets.AA_BACKFACE_BIT, 2, 0],
        dtype=torch.int32,
        device=device,
    )[:count]
    frame = torch.tensor([-3, 7, 2, 5, 0, 1], device=device)[:count]
    ref = torch.tensor([1, 2, 0, 2, 1, 0], device=device)[:count]
    tri = torch.tensor([True, False, True, True, False, True], device=device)[:count]
    order = torch.arange(count - 1, -1, -1, device=device)
    positions = torch.arange(count, device=device)
    obj = torch.tensor(
        [[17, 23, 7], [19, 11, 29]],
        dtype=torch.int64 if wide else torch.int32,
        device=device,
    )
    if wide:
        obj += 2**40
    closed = torch.tensor(
        [[0.5, 1.0, 0.75], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0]], device=device
    )
    if not animated:
        obj, closed = obj[:1], closed[:1]
    values = pixel, mask, frame, ref, tri, order, positions, obj, closed
    return tuple(map(_strided, values)) if strided else values


def _metadata_oracle(pixel, mask, frame, ref, tri, order, pos, obj, closed):
    flags = (closed[(frame + 5) % len(closed), ref] > 0.5) & tri
    if not bool(flags.any()):
        return None
    sid = obj[(frame + 5) % len(obj), ref].to(torch.int64)[order]
    key = torch.where(flags[order], pixel * (int(sid.max()) + 2) + sid, -(pos + 1))
    return ShellSegments(key, (mask & sheets.AA_BACKFACE_BIT) != 0)


@pytest.mark.parametrize("count", [0, 1, 6])
@pytest.mark.parametrize("owned", [False, True])
@pytest.mark.parametrize("wide", [False, True])
@pytest.mark.parametrize("animated", [False, True])
@pytest.mark.parametrize("strided", [False, True])
def test_shell_metadata_is_exact_and_releases_lookup_scratch(
    count, owned, wide, animated, strided
):
    memory, ws = _workspace()
    values = _metadata(ws.device, count, wide=wide, animated=animated, strided=strided)
    expected = _metadata_oracle(*values)
    with ws.stage():
        out = ShellSegments.allocate(ws, count) if owned else None
        floor = memory.get_pointers()
        result = shell_segments(*values, 5, out=out, workspace=ws)
        assert memory.get_pointers() == floor
        _poison(memory)
        if expected is None:
            assert result is None
        else:
            assert not owned or result is out
            assert all(_bits_equal(a, b) for a, b in zip(result, expected))
    assert memory.current_pointer == ws._depth == ws._live_bytes == 0
    if not owned and result is not None:
        _poison(memory)
        assert all(_bits_equal(a, b) for a, b in zip(result, expected))


@pytest.mark.parametrize("mode", ["empty", "open", "circuits"])
def test_inactive_shells_skip_surface_lookup_and_leave_outputs_untouched(
    monkeypatch, mode
):
    memory, ws = _workspace()
    values = list(_metadata(ws.device, 0 if mode == "empty" else 6))
    if mode == "open":
        values[-1].fill_(0.5)
    elif mode == "circuits":
        values[4].zero_()

    def forbidden(*args, **kwargs):
        raise AssertionError("inactive shells must not gather surface IDs")

    monkeypatch.setattr(sheet_shells, "_surface_ids", forbidden)
    with ws.stage():
        out = ShellSegments.allocate(ws, values[0].numel())
        out.key.fill_(73)
        out.back.fill_(True)
        floor = memory.get_pointers()
        assert shell_segments(*values, 5, out=out, workspace=ws) is None
        assert memory.get_pointers() == floor
        assert torch.all(out.key == 73)
        assert torch.all(out.back)


@pytest.mark.parametrize(
    "problem",
    [
        "shape",
        "dtype",
        "stride",
        "input_alias",
        "output_alias",
        "table_alias",
        "table",
        "closed",
        "source",
        "device",
        "workspace",
    ],
)
def test_shell_metadata_validation_precedes_output_mutation(problem):
    memory, ws = _workspace()
    values = list(_metadata(ws.device))
    out = [
        torch.full((6,), 41, device=ws.device),
        torch.ones(6, dtype=torch.bool, device=ws.device),
    ]
    if problem == "shape":
        out[1] = out[1][:5]
    elif problem == "dtype":
        out[1] = out[1].to(torch.uint8)
    elif problem == "stride":
        out[1] = _strided(out[1])
    elif problem == "input_alias":
        out[0] = values[0]
    elif problem == "output_alias":
        out[1] = out[0].view(torch.bool)[:6]
    elif problem == "table_alias":
        values[-2] = out[0].reshape(2, 3)
    elif problem == "table":
        values[-2] = values[-2].float()
    elif problem == "closed":
        values[-1] = values[-1][:, :2]
    elif problem == "source":
        values[2] = values[2].int()
    elif problem == "device":
        out[1] = torch.empty(6, dtype=torch.bool, device="meta")
    else:
        ws = CompactionWorkspace(device="meta")
    tensors = [x for x in (*values, *out) if x.device.type != "meta"]
    saved = [x.clone() for x in tensors]
    with pytest.raises(ValueError):
        shell_segments(*values, 5, out=ShellSegments(*out), workspace=ws)
    assert all(_bits_equal(a, b) for a, b in zip(tensors, saved))
    assert memory.current_pointer == 0


def _ceiling_oracle(segments, depth, coverage, *, native):
    """Previous implementation's allocating formula, independent of new stages."""
    key, back = segments
    # Python sorting independently pins stable key/depth order, including ties.
    order = torch.tensor(
        sorted(range(len(depth)), key=lambda i: (int(key[i]), float(depth[i]))),
        dtype=torch.int64,
        device=depth.device,
    )
    result = coverage.clone()
    acc = mps_compat.accumulate_dtype()
    scratch = result.to(acc, copy=True)
    c2 = scratch[order]
    exclusive = c2.cumsum(0) - c2
    if native:
        sheet_compact_taichi.solid_shell_ceiling(
            key.contiguous(),
            mps_compat.kernel_index(order),
            back.contiguous().view(torch.uint8),
            exclusive,
            scratch,
            len(depth),
            result,
            mps_compat.taichi_accumulate_dtype(),
        )
        return result
    starts = torch.ones(len(depth), dtype=torch.bool, device=depth.device)
    starts[1:] = key[order][1:] != key[order][:-1]
    groups = starts.long().cumsum(0) - 1
    count = int(groups[-1]) + 1
    first = torch.zeros(count, dtype=torch.int64, device=depth.device)
    first.scatter_(0, groups[starts], torch.nonzero(starts).flatten())
    spent = exclusive - exclusive[first][groups]
    front = torch.zeros(count, dtype=acc, device=depth.device)
    back_sum = torch.zeros_like(front)
    facing = back[order]
    front.scatter_add_(0, groups, torch.where(facing, 0.0, c2))
    back_sum.scatter_add_(0, groups, torch.where(facing, c2, 0.0))
    cap = torch.maximum(front, back_sum).float().to(acc)
    scale = (cap[groups] - spent).clamp_min_(0).div_(c2.clamp_min_(1e-12)).clamp_max_(1)
    result[order] = (c2 * scale).float()
    return result


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("friendly", [False, True])
@pytest.mark.parametrize("count", [0, 1, 12, 89])
@pytest.mark.parametrize("strided", [False, True])
def test_shell_ceiling_preserves_arithmetic_and_inout_bits(
    monkeypatch, native, friendly, count, strided
):
    memory, ws = _workspace()
    monkeypatch.setattr(mps_compat, "mps_friendly", lambda: friendly)
    rng = torch.Generator().manual_seed(814 + count)
    key = torch.randint(-3, 5, (count,), generator=rng, device=ws.device)
    depth = torch.randint(0, 3, (count,), generator=rng, device=ws.device).float() / 7
    back = torch.rand(count, generator=rng, device=ws.device) > 0.45
    area = torch.rand(count, generator=rng, device=ws.device) * 2
    if count >= 12:
        area[:6] = torch.tensor(
            [0.0, -0.0, 3e-14, 0.33333334, 1.0000001, 1001.125], device=ws.device
        )
        depth[0] = -0.0
    if strided:
        key, back, depth = map(_strided, (key, back, depth))
    segments = ShellSegments(key, back)
    snapshots = [v.clone() for v in (*segments, depth)]
    expected = _ceiling_oracle(segments, depth, area, native=native) if count else area
    with ws.stage():
        out = ws.copy(area)
        floor = memory.get_pointers()
        result = apply_shell_ceiling(
            segments,
            depth,
            out,
            order_builder=sheets._key_depth_order,
            use_kernel=native,
            workspace=ws,
        )
        assert result is out
        assert memory.get_pointers() == floor
        _poison(memory)
        assert _bits_equal(result, expected)
        assert all(_bits_equal(v, s) for v, s in zip((*segments, depth), snapshots))
    assert memory.current_pointer == ws._depth == ws._live_bytes == 0


@pytest.mark.parametrize(
    "problem",
    ["shape", "dtype", "stride", "input_alias", "key_alias", "device", "workspace"],
)
def test_ceiling_validates_inout_before_sorting_or_writing(monkeypatch, problem):
    memory, ws = _workspace()
    key = torch.tensor([0, 0, 1, 1], device=ws.device)
    depth = torch.arange(4, dtype=torch.float32, device=ws.device)
    back = torch.tensor([False, True, False, True], device=ws.device)
    coverage = torch.ones(4, device=ws.device)
    if problem == "shape":
        coverage = coverage[:3]
    elif problem == "dtype":
        coverage = coverage.double()
    elif problem == "stride":
        coverage = _strided(coverage)
    elif problem == "input_alias":
        coverage = depth
    elif problem == "key_alias":
        coverage = key.view(torch.float32)[:4]
    elif problem == "device":
        coverage = torch.empty(4, device="meta")
    else:
        ws = CompactionWorkspace(device="meta")
    saved = [v.clone() for v in (key, depth, back)]

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid destinations must fail before sorting")

    with pytest.raises(ValueError):
        apply_shell_ceiling(
            ShellSegments(key, back),
            depth,
            coverage,
            order_builder=forbidden,
            use_kernel=True,
            workspace=ws,
        )
    assert all(_bits_equal(v, s) for v, s in zip((key, depth, back), saved))
    assert memory.current_pointer == 0


@pytest.mark.parametrize(
    "site", ["closed", "surface", "sort", "scan", "reduction", "copy", "native"]
)
def test_shell_failures_release_all_scratch(monkeypatch, site):
    memory, ws = _workspace()
    sentinel = memory.get_tensor((3,), torch.int64, persist=True).fill_(79)
    memory.get_tensor((7,), torch.uint8).fill_(113)
    before = memory.get_pointers()

    def wrapped(original):
        def fail_after(*args, **kwargs):
            original(*args, **kwargs)
            raise LookupError("shell failure")

        return fail_after

    if site in ("closed", "surface"):
        name = "gather_frame_table" if site == "closed" else "_surface_ids"
        monkeypatch.setattr(sheet_shells, name, wrapped(getattr(sheet_shells, name)))

        def call():
            return shell_segments(*_metadata(ws.device), 5, workspace=ws)
    else:
        if site == "sort":
            sorter = wrapped(sheets._key_depth_order)
        else:
            sorter = sheets._key_depth_order
        if site == "scan":
            monkeypatch.setattr(torch, "cumsum", wrapped(torch.cumsum))
        elif site == "reduction":
            monkeypatch.setattr(torch, "maximum", wrapped(torch.maximum))
        elif site == "copy":
            monkeypatch.setattr(
                sheet_shells, "index_copy_rows", wrapped(sheet_shells.index_copy_rows)
            )
        elif site == "native":
            monkeypatch.setattr(
                sheet_compact_taichi,
                "solid_shell_ceiling",
                wrapped(sheet_compact_taichi.solid_shell_ceiling),
            )
        segments = ShellSegments(
            torch.tensor([0, 0], device=ws.device),
            torch.tensor([False, True], device=ws.device),
        )

        def call():
            return apply_shell_ceiling(
                segments,
                torch.tensor([1.0, 2.0], device=ws.device),
                torch.ones(2, device=ws.device),
                order_builder=sorter,
                use_kernel=site == "native",
                workspace=ws,
            )

    with pytest.raises(LookupError, match="shell failure"):
        call()
    assert memory.get_pointers() == before
    assert ws._depth == ws._live_bytes == 0
    _poison(memory)
    assert torch.all(sentinel == 79)


@pytest.mark.parametrize("native", [False, True])
def test_shell_stages_are_reclaimed_before_rank_grouping(monkeypatch, native):
    from tests.unit_tests.test_closed_shell_ceiling import _coverage

    memory, ws = _workspace()
    monkeypatch.setattr(sheets.rt_settings, "sheet_shell_ceiling_kernel", native)
    coverage, merged, cam, pws = _coverage(
        [
            (0, 1.0, 0, 0.75, 255),
            (0, 1.01, 1, 0.5, 255 | sheets.AA_BACKFACE_BIT),
            (0, 1.02, 2, 0.25, 15),
            (1, 1.0, 4, 1.0, 255),
        ],
        tri_closed=[1.0] * 8,
    )
    coverage = {
        k: v.to(ws.device) if torch.is_tensor(v) else v for k, v in coverage.items()
    }
    merged = {
        k: v.to(ws.device) if torch.is_tensor(v) else v for k, v in merged.items()
    }
    args = coverage, merged, cam.to(ws.device), pws.to(ws.device), 0, 4, 4
    expected = sheets.compact_sheets(*args)
    original_ceiling = sheets.apply_shell_ceiling
    original_rank = sheets._sheet_rank_groups
    observed = []

    def ceiling(*a, **kw):
        observed.append((memory.current_pointer, ws._depth))
        return original_ceiling(*a, **kw)

    def rank(*a, **kw):
        assert len(observed) == 1
        pointer, depth = observed[0]
        assert memory.current_pointer < pointer
        assert ws._depth < depth
        _poison(memory)
        return original_rank(*a, **kw)

    monkeypatch.setattr(sheets, "apply_shell_ceiling", ceiling)
    monkeypatch.setattr(sheets, "_sheet_rank_groups", rank)
    result = sheets.compact_sheets(
        *args, diagnostics=False, resolver_memory=memory, workspace=ws
    )
    _poison(memory)
    names = {"sheet_cov": "sheet_wgt", "sheet_msk": "sheet_wmsk"}
    for name, value in result._asdict().items():
        if torch.is_tensor(value):
            wanted = expected[names.get(name, name)]
            if name == "sheet_offsets":
                wanted = wanted.to(torch.int32)  # persistent CSR contract
            assert _bits_equal(value, wanted), name
    assert memory.current_pointer == ws._depth == ws._live_bytes == 0
