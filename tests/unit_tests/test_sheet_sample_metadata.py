"""Sample-depth predicates retain the exact mask, dust and signed-weight rules."""

from __future__ import annotations

import pytest
import torch

from algan.rendering.raytracing import sheets
from algan.rendering.raytracing.raster_taichi import _AA_MAT_OPAQUE_BIT as MAT_OPAQUE
from algan.rendering.raytracing.sheet_preprocessing import (
    SampleDepthMetadata,
    sample_depth_metadata,
)
from tests.unit_tests.test_sheet_fragment_workspace import (
    _bits_equal,
    _poison,
    _workspace,
)
from tests.unit_tests.test_sheet_preprocessing_workspace import _strided


def _inputs(device, size, strided=False):
    index = torch.arange(size, device=device)
    pixel = index * 17
    reference = (index % 3).to(torch.int32)
    reference[::7] = -1
    pattern = [
        255,
        255 | MAT_OPAQUE,
        0,
        15,
        255 | sheets.AA_SLIVER_BIT | MAT_OPAQUE,
    ]
    mask = torch.tensor(pattern * ((size + 4) // 5), dtype=torch.int32, device=device)[
        :size
    ]
    coverage = torch.ones(size, device=device)
    coverage[::4] -= sheets.FULL_DUST * 2
    if size > 5:
        coverage[5] = float("nan")
    weight = torch.ones(size, device=device)
    weight[::6] = -1
    weight[1::6] = -0.0
    if size > 3:
        weight[3] = float("nan")
    only = index % 4 != 3
    table = torch.tensor([[2**40 + 3, 2**24 + 1, 7], [2**32, -9, 13]], device=device)
    values = (pixel, reference, mask, coverage, weight, only, table)
    return tuple(map(_strided, values)) if strided else values


def _oracle(pixel, reference, mask, coverage, weight, only, table):
    triangle = reference >= 0
    low = mask & sheets.AA_MASK_ALL
    nonareal = (low != 0) & ((mask & sheets.AA_SLIVER_BIT) == 0)
    enforcer = (
        triangle
        & ((mask & MAT_OPAQUE) != 0)
        & (low == sheets.AA_MASK_ALL)
        & ((coverage - 1).abs() <= sheets.FULL_DUST)
        & only
        & (weight >= 0)
    )
    subject = triangle & nonareal & only & (weight >= 0)
    surface = table[
        (pixel // 16 + 3) % table.shape[0], reference.clamp_min(0).long()
    ].long()
    return SampleDepthMetadata(low, surface, enforcer, subject)


@pytest.mark.parametrize("size", [0, 1, 41])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("owned", [False, True])
@pytest.mark.parametrize("narrow", [False, True])
def test_sample_metadata_matches_original_predicates(size, strided, owned, narrow):
    memory, ws = _workspace()
    values = list(_inputs(ws.device, size, strided))
    if narrow:
        values[-1] = values[-1].to(torch.int32)
    expected = _oracle(*values)
    before = [x.clone() for x in values]
    with ws.stage():
        out = SampleDepthMetadata.allocate(ws, size) if owned else None
        floor = memory.current_pointer
        actual = sample_depth_metadata(*values, 16, 3, out=out, workspace=ws)
        assert not owned or actual is out
        assert memory.current_pointer == floor
        _poison(memory)
        assert all(_bits_equal(a, b) for a, b in zip(actual, expected))
        assert all(_bits_equal(a, b) for a, b in zip(values, before))
    assert memory.current_pointer == ws._depth == ws._live_bytes == 0


@pytest.mark.parametrize(
    "problem",
    [
        "shape",
        "dtype",
        "stride",
        "output_alias",
        "mask_alias",
        "flag_alias",
        "device",
        "input",
    ],
)
def test_sample_metadata_checks_every_destination_before_writes(problem):
    memory, ws = _workspace()
    values = list(_inputs(ws.device, 8))
    with ws.stage():
        out = list(SampleDepthMetadata.allocate(ws, 8))
        for x in out:
            x.fill_(1)
        if problem == "shape":
            out[-1] = out[-1][:-1]
        elif problem == "dtype":
            out[-1] = out[-1].to(torch.uint8)
        elif problem == "stride":
            out[-1] = _strided(out[-1])
        elif problem == "output_alias":
            out[-1] = out[-2]
        elif problem == "mask_alias":
            out[0] = values[2]
        elif problem == "flag_alias":
            out[-1] = values[5]
        elif problem == "device":
            out[-1] = torch.empty(8, dtype=torch.bool, device="meta")
        else:
            values[3] = values[3].double()
        tensors = [x for x in (*values, *out) if x.device.type != "meta"]
        before = [x.clone() for x in tensors]
        floor = memory.current_pointer
        with pytest.raises(ValueError):
            sample_depth_metadata(
                *values, 16, 3, out=SampleDepthMetadata(*out), workspace=ws
            )
        assert all(_bits_equal(a, b) for a, b in zip(tensors, before))
        assert memory.current_pointer == floor
