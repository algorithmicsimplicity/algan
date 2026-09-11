"""Expanded sample-depth reference: exact results and reclaimed scratch."""

from __future__ import annotations

import pytest
import torch

from algan.rendering.raytracing import sheets
from algan.rendering.raytracing.sheet_workspace import CompactionWorkspace
from tests.unit_tests.test_sheet_fragment_workspace import (
    _bits_equal,
    _poison,
    _workspace,
)
from tests.unit_tests.test_sheet_preprocessing_workspace import _strided


def _inputs(device, count, active=True):
    gen = torch.Generator().manual_seed(1309 + count)
    pixels = torch.randint(0, 4, (count,), generator=gen).long() + 2**24 + 3
    surfaces = torch.randint(0, 3, (count,), generator=gen).long() + 2**40 + 1
    depth = torch.randint(0, 5, (count, sheets.AA_NUM_SAMPLES), generator=gen).float()
    # Infinite/missing lanes, signed zero, exact ties and the epsilon boundary.
    depth[::3, 0] = float("inf")
    depth[::5, 1] = -0.0
    depth[::4, 2] = sheets.depth_tie_epsilon
    depth[1::4, 2] = torch.nextafter(
        torch.tensor(sheets.depth_tie_epsilon), torch.tensor(float("inf"))
    )
    mask = torch.randint(0, 256, (count,), generator=gen, dtype=torch.int32)
    enforcer = torch.randint(0, 2, (count,), generator=gen).bool()
    enforcer[pixels == 2**24 + 6] = False
    if not active:
        enforcer.zero_()
    subject = torch.randint(0, 2, (count,), generator=gen).bool()
    return tuple(
        v.to(device) for v in (pixels, depth, surfaces, enforcer, subject, mask)
    )


def _oracle(pixels, depths, sid, enforcer, subject, masks, cede):
    # An independent direct competitor search, without grouped sorted minima.
    result = torch.zeros(pixels.numel(), dtype=torch.int32, device=pixels.device)
    lanes = torch.arange(sheets.AA_NUM_SAMPLES, device=pixels.device)
    for i in range(pixels.numel()):
        competitors = enforcer & (pixels == pixels[i]) & (sid != sid[i])
        other = (
            depths[competitors].amin(0)
            if bool(competitors.any())
            else torch.full_like(depths[i], float("inf"))
        )
        owns = ((masks[i] >> lanes) & 1).bool()
        lost = owns & (other < depths[i] - sheets.depth_tie_epsilon) & subject[i]
        if bool(lost.sum().float() > float(cede) * owns.sum().float()):
            result[i] = (
                sum(
                    1 << lane
                    for lane in range(sheets.AA_NUM_SAMPLES)
                    if bool(lost[lane])
                )
                << sheets.AA_LOSE_SHIFT
            )
    return result


@pytest.mark.parametrize("count", [0, 1, 9, 65])
@pytest.mark.parametrize("friendly", [False, True])
@pytest.mark.parametrize("owned", [False, True])
@pytest.mark.parametrize("cede", [0.0, 0.25, 1.0])
def test_depth_reference_matches_independent_oracle_and_owns_result(
    monkeypatch, count, friendly, owned, cede
):
    memory, ws = _workspace()
    monkeypatch.setenv("ALGAN_MPS_FRIENDLY", str(int(friendly)))
    monkeypatch.setattr(sheets, "sheet_sample_depth_cede", cede)
    values = _inputs(ws.device, count)
    expected = _oracle(*values, cede)
    snapshots = [v.clone() for v in values]
    with ws.stage():
        out = ws.tensor((count,), torch.int32) if owned else None
        pointer = memory.current_pointer
        actual = sheets._sample_depth_lose_reference(*values, out=out, workspace=ws)
        assert out is None or actual is out
        assert memory.current_pointer == pointer
        _poison(memory)
        assert _bits_equal(actual, expected)
        assert all(_bits_equal(v, saved) for v, saved in zip(values, snapshots))
    assert ws._depth == ws._live_bytes == memory.current_pointer == 0


@pytest.mark.parametrize("active", [False, True])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("integer", [torch.int32, torch.int64])
def test_depth_reference_accepts_strides_and_integer_widths(active, strided, integer):
    memory, ws = _workspace()
    pix, depths, sid, enf, sub, masks = _inputs(ws.device, 19, active)
    values = (
        (pix - 2**24).to(integer),
        depths,
        (sid - 2**40).to(integer),
        enf,
        sub,
        masks.to(integer),
    )
    if strided:
        values = tuple(_strided(v) for v in values)
    expected = _oracle(*values, sheets.sheet_sample_depth_cede)
    result = sheets._sample_depth_lose_reference(*values, workspace=ws)
    _poison(memory)
    assert _bits_equal(result, expected)


@pytest.mark.parametrize(
    "problem",
    [
        "shape",
        "dtype",
        "stride",
        "device",
        "low_alias",
        "depth_alias",
        "pixel_alias",
        "workspace",
        "input_shape",
        "input_dtype",
        "input_device",
    ],
)
def test_depth_reference_validates_before_mutation(problem):
    memory, ws = _workspace()
    values = list(_inputs(ws.device, 9))
    out = torch.full((9,), 37, dtype=torch.int32, device=ws.device)
    if problem == "shape":
        out = out[:8]
    elif problem == "dtype":
        out = out.long()
    elif problem == "stride":
        out = _strided(out)
    elif problem == "device":
        out = torch.empty(9, dtype=torch.int32, device="meta")
    elif problem == "low_alias":
        out = values[-1]
    elif problem == "depth_alias":
        out = values[1].view(torch.int32).view(-1)[:9]
    elif problem == "pixel_alias":
        out = values[0].view(torch.int32)[:9]
    elif problem == "workspace":
        ws = CompactionWorkspace(device="meta")
    elif problem == "input_shape":
        values[1] = values[1][:, :7]
    elif problem == "input_dtype":
        values[3] = values[3].int()
    elif problem == "input_device":
        values[2] = torch.empty_like(values[2], device="meta")
    tensors = [v for v in (*values, out) if v.device.type != "meta"]
    snapshots = [v.clone() for v in tensors]
    with pytest.raises(ValueError):
        sheets._sample_depth_lose_reference(*values, out=out, workspace=ws)
    assert all(_bits_equal(a, b) for a, b in zip(tensors, snapshots))
    assert memory.current_pointer == ws._depth == ws._live_bytes == 0


@pytest.mark.parametrize("failure", ["sort", "query", "gate"])
def test_depth_reference_failure_unwinds_every_stage(monkeypatch, failure):
    memory, ws = _workspace()
    values = _inputs(ws.device, 19)
    sentinel = memory.get_tensor((9,), torch.int64, persist=True).fill_(471)
    memory.get_tensor((5,), torch.uint8).fill_(199)
    before = memory.get_pointers()
    target, name = (
        (sheets, "_lexsort")
        if failure == "sort"
        else (torch, "searchsorted" if failure == "query" else "sum")
    )
    original = getattr(target, name)
    visited = []

    def fail(*args, **kwargs):
        result = original(*args, **kwargs)
        # The sort helper may sum its own boundaries. Inject only in the final
        # expanded-lane gate, where sum receives a 2D boolean tensor.
        if failure != "gate" or (args[0].ndim == 2 and args[0].dtype == torch.bool):
            visited.append(True)
            raise LookupError("injected depth-reference failure")
        return result

    monkeypatch.setattr(target, name, fail)
    with pytest.raises(LookupError, match="depth-reference"):
        sheets._sample_depth_lose_reference(*values, workspace=ws)
    assert visited == [True]
    assert memory.get_pointers() == before
    assert ws._depth == ws._live_bytes == 0
    _poison(memory)
    assert sentinel.tolist() == [471] * 9
