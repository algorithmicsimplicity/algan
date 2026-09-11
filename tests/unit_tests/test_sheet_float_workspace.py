"""Floating sheet work keeps its arithmetic and releases stage-owned temporaries."""

from __future__ import annotations

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

from algan.rendering import mps_compat
from algan.rendering.raytracing import sheets
from algan.rendering.raytracing.sheet_reduction_buffers import (
    BandComposite,
    SheetWeights,
)
from algan.rendering.raytracing.sheet_statistics import (
    SheetStatistics,
    sheet_statistics,
)
from tests.unit_tests.test_sheet_fragment_workspace import (
    _bits_equal,
    _poison,
    _workspace,
)
from tests.unit_tests.test_sheet_preprocessing_workspace import _strided


class _NewStorage(TorchDispatchMode):
    """Record visible tensor results outside the arena and known input storages.

    This does not observe compiler, operator-internal or driver workspace.
    """

    def __init__(self, memory, inputs):
        super().__init__()
        self.known = {v.untyped_storage().data_ptr() for v in [memory.data, *inputs]}
        self.external = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        for value in tree_leaves(result):
            if (
                isinstance(value, torch.Tensor)
                and value.numel() > 1
                and value.untyped_storage().data_ptr() not in self.known
            ):
                self.external.append((str(func), tuple(value.shape), value.dtype))
        return result


def _supported(device, *dtypes):
    if device.type == "mps" and torch.float64 in dtypes:
        pytest.skip("MPS has no float64 tensors")


@pytest.mark.parametrize("friendly", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("floor", [1e-12, 1e-7])
@pytest.mark.parametrize("destination", ["default", "owned", "inplace"])
def test_clamp_floor_destinations_preserve_bits_and_nan(
    monkeypatch, friendly, dtype, floor, destination
):
    memory, ws = _workspace()
    _supported(ws.device, dtype)
    monkeypatch.setenv("ALGAN_MPS_FRIENDLY", str(int(friendly)))
    values = torch.tensor(
        [
            -float("inf"),
            -0.0,
            0.0,
            1e-40,
            floor / 2,
            floor,
            floor * 2,
            float("inf"),
            float("nan"),
        ],
        dtype=dtype,
        device=ws.device,
    )
    expected = torch.where(values < floor, floor, values)
    with ws.stage():
        source = ws.copy(values) if destination == "inplace" else values
        out = (
            source
            if destination == "inplace"
            else ws.tensor(values.shape, dtype)
            if destination == "owned"
            else None
        )
        pointer = memory.current_pointer
        actual = mps_compat.clamp_floor(source, floor, out=out, workspace=ws)
        assert out is None or actual is out
        assert memory.current_pointer == pointer
        _poison(memory)
        assert _bits_equal(actual, expected)
    assert ws._depth == ws._live_bytes == memory.current_pointer == 0


@pytest.mark.parametrize("friendly", [False, True])
def test_default_floor_still_supports_gradients(monkeypatch, friendly):
    monkeypatch.setenv("ALGAN_MPS_FRIENDLY", str(int(friendly)))
    source = torch.tensor([0.0, 1.0], requires_grad=True)
    mps_compat.clamp_floor(source, 1e-12).sum().backward()
    assert source.grad.tolist() == [0.0, 1.0]


def _geometry_inputs(device, dtype, strided=False):
    gen = torch.Generator().manual_seed(691)
    norm = torch.randn(3, 5, 9, generator=gen, dtype=dtype)
    norm[:, 0] = torch.tensor([0.0, 0.0, 1.0] * 3)
    norm[:, 1] = 0
    norm[:, 2] = torch.tensor([0.707, 0.707, 0.0] * 3)
    norm[:, 3] *= 1e-8
    pos = torch.randn(2, 5, 9, generator=gen, dtype=dtype)
    pos[..., 2::3] += 4
    screen = torch.randn(3, 5, 10, generator=gen, dtype=dtype) * 5
    screen[..., 9] = torch.tensor([1.0, 0.0, 1.0, 0.5, 1.0])
    cam = torch.randn(4, 3, generator=gen, dtype=dtype)
    pws = torch.tensor([1e-3, 2e-4], dtype=dtype)
    frame = torch.arange(35) % 7
    ref = torch.arange(35) % 5
    tri = torch.arange(35) % 4 != 1
    depth = torch.rand(35, generator=gen, dtype=dtype) + 2
    order = torch.argsort(depth, stable=True)
    values = [norm, pos, screen, cam, pws, frame, ref, tri, depth, order]
    return [(_strided(v.to(device)) if strided else v.to(device)) for v in values]


def _shade_oracle(norm, pos, frame, ref, tri, offset):
    absolute = torch.arange(7, device=norm.device) + offset
    nrm = norm[absolute % norm.shape[0]].reshape(7, -1, 3, 3)
    mag = nrm.norm(dim=3)
    unit = nrm / mps_compat.clamp_floor(mag.unsqueeze(3), 1e-12)
    spread = torch.maximum(
        (unit[:, :, 1] - unit[:, :, 0]).abs().amax(dim=2),
        (unit[:, :, 2] - unit[:, :, 0]).abs().amax(dim=2),
    )
    flat = (mag.amin(dim=2) > 1e-6) & (spread < 1e-6)
    geometric = mag.amax(dim=2) < 1e-6
    positions = pos[absolute % pos.shape[0]]
    gn = torch.cross(
        positions[..., 3:6] - positions[..., :3],
        positions[..., 6:9] - positions[..., :3],
        dim=-1,
    )
    gn = gn / mps_compat.clamp_floor(gn.norm(dim=-1, keepdim=True), 1e-12)
    face = torch.where(geometric.unsqueeze(-1), gn, unit[:, :, 0])
    quant = sheets.SHADE_CLASS_QUANT
    q = torch.round(face * float(quant)).long().clamp_(-quant, quant) + quant
    packed = (q[..., 0] << 16) | (q[..., 1] << 8) | q[..., 2]
    table = torch.where(
        flat | geometric,
        packed + 1,
        torch.zeros((), dtype=torch.int64, device=norm.device),
    )
    return torch.where(tri, table[frame, ref], 0)


def _prim_oracle(pos, screen, cam, pws, frame, ref, tri, depth, order, offset):
    absolute = torch.arange(7, device=pos.device) + offset
    positions = pos[absolute % pos.shape[0]]
    origin = cam[absolute % cam.shape[0]].view(7, 1, 3)
    dmin = dmax = None
    for k in range(3):
        distance = torch.linalg.norm(positions[..., 3 * k : 3 * k + 3] - origin, dim=-1)
        dmin = distance if dmin is None else torch.minimum(dmin, distance)
        dmax = distance if dmax is None else torch.maximum(dmax, distance)
    extent = dmax - dmin
    if screen is not None:
        scr = screen[absolute % screen.shape[0]]
        x = scr[..., :3]
        y = scr[..., 3:6]
        proj = torch.maximum(
            x.amax(-1) - x.amin(-1), y.amax(-1) - y.amin(-1)
        ).clamp_min_(1.0)
        extent = torch.where(scr[..., 9] > 0.5, extent / proj, extent)
    table = extent.to(pos.dtype)  # original table assignment's rounding
    scale = torch.where(
        tri,
        table[frame, ref] + pws[(frame + offset) % pws.shape[0]] * depth,
        torch.zeros_like(depth),
    )
    ordered = scale[order]
    threshold = 2.0 * (ordered[1:] + ordered[:-1])
    ordered_depth = depth[order]
    return ordered_depth[1:] - ordered_depth[:-1] > threshold


@pytest.mark.parametrize("method", ["class", "prim", "prim_no_projection"])
@pytest.mark.parametrize("friendly", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("owned", [False, True])
def test_geometry_blocks_match_original_arithmetic_and_reuse_storage(
    monkeypatch, method, friendly, dtype, strided, owned
):
    memory, ws = _workspace()
    _supported(ws.device, dtype)
    monkeypatch.setenv("ALGAN_MPS_FRIENDLY", str(int(friendly)))
    norm, pos, screen, cam, pws, frame, ref, tri, depth, order = _geometry_inputs(
        ws.device, dtype, strided
    )
    offset = -3
    if method == "prim_no_projection":
        screen = None
    monkeypatch.setattr(
        sheets, "_FRAME_TABLE_BUDGET", 10
    )  # three equal blocks and one short tail
    calls = []
    name = "shade_class_block" if method == "class" else "depth_slope_block"
    original = getattr(sheets, name)

    def block(*args):
        before = memory.current_pointer
        original(*args)
        assert memory.current_pointer == before
        calls.append(before)

    monkeypatch.setattr(sheets, name, block)
    merged = {"tri_norm": norm, "tri_pos": pos}
    if method == "class":
        expected = _shade_oracle(norm, pos, frame, ref, tri, offset)
        fn, args = sheets._shade_class, (merged, frame, offset, ref, tri, True, 7)
    else:
        expected = _prim_oracle(
            pos, screen, cam, pws, frame, ref, tri, depth, order, offset
        )
        fn, args = (
            sheets._prim_split_after,
            (
                merged,
                cam,
                pws,
                screen,
                frame,
                offset,
                ref,
                tri,
                depth,
                depth[order],
                order,
                2.0,
                7,
            ),
        )
    snapshots = [
        v.clone() for v in (norm, pos, cam, pws, frame, ref, tri, depth, order)
    ]
    with ws.stage():
        out = ws.tensor(expected.shape, expected.dtype) if owned else None
        pointer = memory.current_pointer
        actual = fn(*args, out=out, workspace=ws)
        assert memory.current_pointer == pointer
        _poison(memory)
        assert _bits_equal(actual, expected)
    assert len(calls) == 4
    assert calls[0] == calls[1] == calls[2]
    assert all(
        _bits_equal(a, b)
        for a, b in zip((norm, pos, cam, pws, frame, ref, tri, depth, order), snapshots)
    )
    assert ws._depth == ws._live_bytes == memory.current_pointer == 0


@pytest.mark.parametrize("method", ["class", "prim"])
def test_geometry_promotes_mixed_input_precision_before_table_storage(
    monkeypatch, method
):
    memory, ws = _workspace()
    _supported(ws.device, torch.float64)
    norm, pos, screen, cam, pws, frame, ref, tri, depth, order = _geometry_inputs(
        ws.device, torch.float32
    )
    norm, cam, screen, pws = [v.double() for v in (norm, cam, screen, pws)]
    monkeypatch.setattr(sheets, "_FRAME_TABLE_BUDGET", 5)
    merged = {"tri_pos": pos, "tri_norm": norm}
    if method == "class":
        expected = _shade_oracle(norm, pos, frame, ref, tri, 2)
        actual = sheets._shade_class(merged, frame, 2, ref, tri, True, 7, workspace=ws)
    else:
        expected = _prim_oracle(pos, screen, cam, pws, frame, ref, tri, depth, order, 2)
        actual = sheets._prim_split_after(
            merged,
            cam,
            pws,
            screen,
            frame,
            2,
            ref,
            tri,
            depth,
            depth[order],
            order,
            2.0,
            7,
            workspace=ws,
        )
    _poison(memory)
    assert _bits_equal(actual, expected)


def _weight_oracle(band, cov, masks, area, union, corr):
    count = torch.zeros_like(area, dtype=torch.int64)
    count.scatter_add_(0, band, torch.ones_like(band))
    runs = torch.zeros_like(count)
    starts = torch.ones_like(band)
    starts[1:] = (band[1:] != band[:-1]).long()
    runs.scatter_add_(0, band, starts)
    multi = (count[band] > 1) & (runs[band] == 1)
    if not multi.any():
        return cov, masks
    acc = mps_compat.accumulate_dtype()
    share = cov.to(acc) / mps_compat.clamp_floor(area[band].to(acc), 1e-12)
    p = corr[band].to(acc) * share
    unions = union[band]
    pop = torch.zeros_like(unions, dtype=torch.int32)
    for lane in range(sheets.AA_NUM_SAMPLES):
        pop += ((unions >> lane) & 1).int()
    weight = torch.where(
        unions == sheets.AA_MASK_ALL,
        p,
        p * pop.clamp_min(1).to(acc) / float(sheets.AA_NUM_SAMPLES),
    ).float()
    cont = torch.zeros_like(multi)
    cont[:-1] = band[1:] == band[:-1]
    weight = torch.where(multi & cont, -weight, weight)
    flags = masks & ~sheets.AA_MASK_ALL & ~sheets.AA_SLIVER_BIT
    return torch.where(multi, weight, cov), torch.where(
        multi, unions.to(masks.dtype) | flags, masks
    )


@pytest.mark.parametrize("friendly", [False, True])
@pytest.mark.parametrize("owned", [False, True])
@pytest.mark.parametrize(
    "case", ["partial", "interleaved", "zero", "wide_coverage", "empty", "single"]
)
def test_reference_weights_preserve_wide_rounding_and_signed_zero(
    monkeypatch, friendly, owned, case
):
    memory, ws = _workspace()
    if case == "wide_coverage":
        _supported(ws.device, torch.float64)
    monkeypatch.setenv("ALGAN_MPS_FRIENDLY", str(int(friendly)))
    monkeypatch.setattr(sheets.rt_settings, "sheet_sibling_weights_kernel", False)
    band = torch.tensor([0, 0, 1, 1, 2, 2], device=ws.device)
    if case == "interleaved":
        band = torch.tensor([0, 1, 0, 2, 1, 2], device=ws.device)
    cov = torch.tensor([0.0, 0.75, 0.1, 0.2, -0.0, 0.0], device=ws.device)
    if case == "wide_coverage":
        cov = cov.double() + 1e-10
    masks = torch.tensor(
        [sheets.AA_SLIVER_BIT, 255, 1, 6, 15, 240], dtype=torch.int32, device=ws.device
    )
    area = torch.tensor([0.75, 0.3, 0.0], device=ws.device)
    if case == "zero":
        area.zero_()
    union = torch.tensor([255, 7, 255], dtype=torch.int32, device=ws.device)
    corr = torch.tensor([0.75, 0.8, 1.0], device=ws.device)
    if case in ("empty", "single"):
        size = 0 if case == "empty" else 1
        band, cov, masks = [v[:size] for v in (band, cov, masks)]
    expected = _weight_oracle(band, cov, masks, area, union, corr)
    with ws.stage():
        out = (
            SheetWeights(
                ws.tensor(cov.shape, cov.dtype), ws.tensor(masks.shape, masks.dtype)
            )
            if owned
            else None
        )
        pointer = memory.current_pointer
        result = sheets._sibling_weights(
            band, cov, masks, area, union, corr, out=out, workspace=ws
        )
        assert memory.current_pointer == pointer
        _poison(memory)
        assert all(_bits_equal(a, b) for a, b in zip(result, expected))
        if case == "partial":
            assert torch.signbit(result[0][0])  # continuation includes negative zero
    assert ws._depth == ws._live_bytes == 0


@pytest.mark.parametrize(
    "method", ["class", "prim", "composite", "weights", "statistics"]
)
@pytest.mark.parametrize("friendly", [False, True])
def test_fixed_math_has_no_visible_allocator_owned_result(
    monkeypatch, method, friendly
):
    memory, ws = _workspace()
    monkeypatch.setenv("ALGAN_MPS_FRIENDLY", str(int(friendly)))
    monkeypatch.setattr(sheets.rt_settings, "sheet_mask_kernel", False)
    monkeypatch.setattr(sheets.rt_settings, "sheet_sibling_weights_kernel", False)
    monkeypatch.setattr(sheets.rt_settings, "sheet_band_stats_kernel", False)
    with ws.stage():
        if method in ("class", "prim"):
            norm, pos, screen, cam, pws, frame, ref, tri, depth, order = (
                _geometry_inputs(ws.device, torch.float32)
            )
            merged = {"tri_norm": norm, "tri_pos": pos}
            if method == "class":
                fn, args = sheets._shade_class, (merged, frame, 0, ref, tri, True, 7)
                out = ws.tensor(frame.shape, torch.int64)
            else:
                fn, args = (
                    sheets._prim_split_after,
                    (
                        merged,
                        cam,
                        pws,
                        screen,
                        frame,
                        0,
                        ref,
                        tri,
                        depth,
                        depth[order],
                        order,
                        2.0,
                        7,
                    ),
                )
                out = ws.tensor((34,), torch.bool)
            inputs = [
                norm,
                pos,
                screen,
                cam,
                pws,
                frame,
                ref,
                tri,
                depth,
                order,
                *(a for a in args if isinstance(a, torch.Tensor)),
            ]
            kwargs = {}
        else:
            band = torch.tensor([0, 0, 1, 1], device=ws.device)
            masks = torch.tensor([15, 240, 1, 6], dtype=torch.int32, device=ws.device)
            cov = torch.tensor([0.25, 0.75, 0.1, 0.2], device=ws.device)
            if method == "composite":
                fn, args, out, kwargs = (
                    sheets._band_composite,
                    (band, 2, cov, masks),
                    BandComposite.allocate(ws, 2),
                    {},
                )
            elif method == "weights":
                area = torch.tensor([1.0, 0.3], device=ws.device)
                union = torch.tensor([255, 7], dtype=torch.int32, device=ws.device)
                corr = torch.tensor([1.0, 0.8], device=ws.device)
                fn, args, out, kwargs = (
                    sheets._sibling_weights,
                    (band, cov, masks, area, union, corr),
                    SheetWeights.allocate(ws, 4),
                    {},
                )
            else:
                positions = torch.arange(4, device=ws.device)
                original = torch.tensor([1, 0, 3, 2], device=ws.device)
                pixels = torch.tensor([7, 7, 9, 9], device=ws.device)
                fn, args = (
                    sheet_statistics,
                    (band, masks, positions, original, pixels, cov, 2),
                )
                out, kwargs = (
                    SheetStatistics.allocate(ws, 2, diagnostics=True),
                    {"mask_all": 255, "positioned": True, "diagnostics": True},
                )
            inputs = [a for a in args if isinstance(a, torch.Tensor)]
        tracker = _NewStorage(memory, inputs)
        with tracker:
            fn(*args, out=out, workspace=ws, **kwargs)
        assert tracker.external == []


@pytest.mark.parametrize("method", ["class", "prim", "floor"])
def test_nested_float_failure_restores_arena_and_preserves_inputs(monkeypatch, method):
    memory, ws = _workspace()
    monkeypatch.setenv("ALGAN_MPS_FRIENDLY", "1")
    sentinel = memory.get_tensor((7,), torch.int32, persist=True).fill_(313)
    memory.get_tensor((3,), torch.uint8).fill_(199)
    before = memory.get_pointers()
    norm, pos, screen, cam, pws, frame, ref, tri, depth, order = _geometry_inputs(
        ws.device, torch.float32
    )
    original_norm, original_pos = norm.clone(), pos.clone()
    if method == "floor":
        name, target = "where", torch
    elif method == "class":
        name, target = "cross", torch
    else:
        name, target = "vector_norm", torch.linalg
    original = getattr(target, name)

    def fail(*args, **kwargs):
        original(*args, **kwargs)
        raise LookupError("injected floating-stage failure")

    monkeypatch.setattr(target, name, fail)

    def invoke():
        if method == "floor":
            mps_compat.clamp_floor(
                depth, 1e-12, out=torch.empty_like(depth), workspace=ws
            )
        elif method == "class":
            sheets._shade_class(
                {"tri_norm": norm, "tri_pos": pos},
                frame,
                0,
                ref,
                tri,
                True,
                7,
                workspace=ws,
            )
        else:
            sheets._prim_split_after(
                {"tri_pos": pos},
                cam,
                pws,
                screen,
                frame,
                0,
                ref,
                tri,
                depth,
                depth[order],
                order,
                2.0,
                7,
                workspace=ws,
            )

    with pytest.raises(LookupError, match="floating-stage"):
        invoke()
    assert memory.get_pointers() == before
    _poison(memory)
    assert sentinel.tolist() == [313] * 7
    assert _bits_equal(norm, original_norm)
    assert _bits_equal(pos, original_pos)
    assert ws._depth == ws._live_bytes == 0


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("shape", [(), (0,), (5,), (2, 5)])
@pytest.mark.parametrize("owned", [False, True])
def test_popcount_owned_scratch_and_default_outputs(
    monkeypatch, native, dtype, shape, owned
):
    memory, ws = _workspace()
    monkeypatch.setattr(sheets.rt_settings, "sheet_mask_kernel", native)
    values = torch.tensor([-1, 0, 0x101, 255, 0x102], dtype=dtype, device=ws.device)
    if shape == ():
        values = values[0]
    elif shape == (0,):
        values = values[:0]
    elif len(shape) == 2:
        values = values.repeat(2, 1)
    values = _strided(values) if values.ndim else values
    expected = torch.tensor(
        [(int(v) & sheets.AA_MASK_ALL).bit_count() for v in values.cpu().reshape(-1)],
        dtype=torch.int32,
        device=ws.device,
    ).reshape(shape)
    before = values.clone()
    with ws.stage():
        out = ws.tensor(shape, torch.int32) if owned else None
        pointer = memory.current_pointer
        actual = sheets._popcount_lanes(values, out=out, workspace=ws)
        assert out is None or actual is out
        assert memory.current_pointer == pointer
        _poison(memory)
        assert _bits_equal(actual.reshape(-1), expected.reshape(-1))
        assert _bits_equal(values.reshape(-1), before.reshape(-1))
    assert ws._depth == ws._live_bytes == memory.current_pointer == 0


@pytest.mark.parametrize(
    "problem", ["shape", "dtype", "stride", "device", "alias", "workspace"]
)
def test_popcount_rejects_invalid_destination_before_mutation(problem):
    memory, ws = _workspace()
    values = torch.tensor([0, 1, 255], dtype=torch.int32, device=ws.device)
    out = torch.full((3,), 13, dtype=torch.int32, device=ws.device)
    if problem == "shape":
        out = out[:2]
    elif problem == "dtype":
        out = out.float()
    elif problem == "stride":
        out = torch.full((6,), 13, dtype=torch.int32, device=ws.device)[::2]
    elif problem == "device":
        out = torch.empty(3, dtype=torch.int32, device="meta")
    elif problem == "alias":
        out = values
    elif problem == "workspace":
        ws = sheets.CompactionWorkspace(device="meta")
    before = values.clone()
    with pytest.raises(ValueError):
        sheets._popcount_lanes(values, out=out, workspace=ws)
    assert _bits_equal(values, before)
    assert memory.current_pointer == 0
    if problem not in ("device", "alias"):
        assert torch.all(out == 13)


@pytest.mark.parametrize("friendly", [False, True])
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("owned", [False, True])
def test_band_composite_retains_float32_dust_and_reduction_boundaries(
    monkeypatch, friendly, native, owned
):
    memory, ws = _workspace()
    monkeypatch.setenv("ALGAN_MPS_FRIENDLY", str(int(friendly)))
    monkeypatch.setattr(sheets.rt_settings, "sheet_mask_kernel", native)
    cov = torch.tensor(
        [
            0.25,
            0.5,
            0.1,
            0.2,
            0.0,
            0.0,
            0.5,
            0.50000006,
            0.5,
            0.5,
            1e-12,
            1e-13,
            1.0,
            1.0001,
        ],
        device=ws.device,
    )
    msk = torch.tensor(
        [15, 240, 1, 6, sheets.AA_SLIVER_BIT, 0, 15, 240, 255, 1, 1, 2, 255, 255],
        dtype=torch.int32,
        device=ws.device,
    )
    band = torch.arange(7, device=ws.device).repeat_interleave(2)
    # Independent integer union and summation, then the former literal formula.
    sums = torch.zeros(7, dtype=mps_compat.accumulate_dtype(), device=ws.device)
    sums.scatter_add_(0, band, cov.to(sums.dtype))
    area = sums.float()
    union = torch.tensor(
        [
            (int(msk[2 * i]) | int(msk[2 * i + 1])) & sheets.AA_MASK_ALL
            for i in range(7)
        ],
        dtype=torch.int32,
        device=ws.device,
    )
    pop = torch.tensor(
        [int(v).bit_count() for v in union.cpu()], dtype=torch.float32, device=ws.device
    )
    full = union == sheets.AA_MASK_ALL
    expected = torch.where(
        full,
        torch.where((1.0 - area).abs() <= sheets.FULL_DUST, 1.0, area.clamp(max=1.0)),
        area.clamp(max=1.0) * float(sheets.AA_NUM_SAMPLES) / pop.clamp_min(1),
    )
    with ws.stage():
        out = BandComposite.allocate(ws, 7) if owned else None
        pointer = memory.current_pointer
        actual = sheets._band_composite(band, 7, cov, msk, out=out, workspace=ws)
        assert memory.current_pointer == pointer
        _poison(memory)
        assert _bits_equal(actual.area, area)
        assert _bits_equal(actual.union, union)
        assert _bits_equal(actual.correction, expected)


@pytest.mark.parametrize("friendly", [False, True])
def test_default_floor_does_not_save_reclaimed_predicate_for_backward(
    monkeypatch, friendly
):
    memory, ws = _workspace()
    monkeypatch.setenv("ALGAN_MPS_FRIENDLY", str(int(friendly)))
    source = torch.tensor([0.0, 1.0, -2.0], device=ws.device, requires_grad=True)
    result = mps_compat.clamp_floor(source, 1e-12, workspace=ws)
    _poison(memory)
    result.sum().backward()
    assert source.grad.tolist() == [0.0, 1.0, 0.0]


@pytest.mark.parametrize("precision", ["float32", "float64", "mixed"])
@pytest.mark.parametrize("friendly", [False, True])
@pytest.mark.parametrize("projected", [False, True])
def test_depth_slope_block_preserves_exact_intermediate_table(
    monkeypatch, precision, friendly, projected
):
    from algan.rendering.raytracing.sheet_geometry import depth_slope_block

    memory, ws = _workspace()
    if precision != "float32":
        _supported(ws.device, torch.float64)
    monkeypatch.setenv("ALGAN_MPS_FRIENDLY", str(int(friendly)))
    dtype = torch.float64 if precision == "float64" else torch.float32
    _norm, pos, screen, cam, *_ = _geometry_inputs(ws.device, dtype, strided=True)
    if precision == "mixed":
        cam, screen = cam.double(), screen.double()
    if not projected:
        screen = None
    frames = torch.tensor([-3, 0, 4], device=ws.device)
    positions = pos[frames % pos.shape[0]]
    origins = cam[frames % cam.shape[0]].view(3, 1, 3)
    distances = [
        torch.linalg.norm(positions[..., k : k + 3] - origins, dim=-1)
        for k in (0, 3, 6)
    ]
    nearest = torch.minimum(torch.minimum(distances[0], distances[1]), distances[2])
    farthest = torch.maximum(torch.maximum(distances[0], distances[1]), distances[2])
    extent = farthest - nearest
    if projected:
        scr = screen[frames % screen.shape[0]]
        span = torch.maximum(
            scr[..., :3].amax(-1) - scr[..., :3].amin(-1),
            scr[..., 3:6].amax(-1) - scr[..., 3:6].amin(-1),
        ).clamp_min(1.0)
        extent = torch.where(scr[..., 9] > 0.5, extent / span, extent)
    expected = extent.to(dtype)
    with ws.stage():
        out = ws.tensor(expected.shape, dtype)
        pointer = memory.current_pointer
        depth_slope_block(pos, cam, screen, frames, out, ws)
        assert memory.current_pointer == pointer
        _poison(memory)
        assert _bits_equal(out, expected)
