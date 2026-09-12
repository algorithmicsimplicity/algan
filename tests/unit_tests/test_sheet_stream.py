"""Fused sheet data movement and run construction against independent torch oracles."""

from __future__ import annotations

import pytest
import torch

from algan import SETTINGS
from algan.rendering.raytracing import sheet_stream as stream
from algan.rendering.taichi_runtime import init_taichi
from algan.settings._startup import render_device
from algan.utils.memory_utils import ManualMemory


@pytest.fixture
def device():
    init_taichi()
    return render_device()


def assert_exact(actual, expected):
    assert actual.dtype == expected.dtype
    assert torch.equal(actual.cpu(), expected.cpu())


@pytest.mark.parametrize("lengths", [[], [1], [1, 1, 1], [4, 257, 3, 1025], [8193]])
def test_pixel_runs_and_truncation(device, lengths):
    # Gapped pixel IDs cross frame/low-word boundaries; depth bits must not
    # participate in run detection. All computation of the oracle is on CPU.
    counts = torch.tensor(lengths, dtype=torch.int64)
    covered = torch.arange(len(lengths), dtype=torch.int64) * 8192 + 12345
    pix = torch.repeat_interleave(covered, counts)
    n = pix.numel()
    key = (pix << 32) | (torch.arange(n) * 7919 % (2**32))
    memory = ManualMemory(1.0, device=device, num_bytes=16 * n + 4096)
    before = memory.get_pointers()
    actual = stream.pixel_runs(key.to(device), memory=memory)
    assert memory.get_pointers() == before
    offsets = torch.cat((torch.zeros(1, dtype=torch.int64), counts.cumsum(0))).to(
        torch.int32
    )
    for a, e in zip(actual, (covered, counts, offsets)):
        assert_exact(a, e)
    # Reuse the freed arena aggressively: returned CSR must own its storage.
    with memory.temp():
        memory.get_tensor((memory.get_num_bytes_remaining(),), torch.uint8).fill_(255)
    assert_exact(actual[2], offsets)
    for mode in ("none", "first", "last", "middle"):
        opaque = torch.zeros(n, dtype=torch.bool)
        expected_indices = []
        retained = []
        for p, length in enumerate(lengths):
            start = int(offsets[p])
            used = length
            if mode != "none":
                at = {"first": 0, "last": length - 1, "middle": length // 2}[mode]
                opaque[start + at] = True
                used = at + 1
            retained.append(used)
            expected_indices.extend(range(start, start + used))
        idx, cnt, off = stream.truncate_pixel_runs(opaque.to(device), actual[2])
        assert_exact(cnt, torch.tensor(retained, dtype=torch.int64))
        assert_exact(
            off,
            torch.tensor(
                [0, *torch.tensor(retained).cumsum(0).tolist()], dtype=torch.int32
            ),
        )
        if len(expected_indices) == n:
            assert idx is None
        else:
            assert_exact(idx, torch.tensor(expected_indices, dtype=torch.int32))


def test_pixel_runs_restores_scratch_on_failure(device, monkeypatch):
    memory = ManualMemory(1.0, device=device, num_bytes=4096)
    before = memory.get_pointers()
    from algan.rendering.raytracing import sheet_stream_taichi as kernels

    def fail(*args):
        raise RuntimeError("injected scatter failure")

    monkeypatch.setattr(kernels, "write_pixel_runs", fail)
    with pytest.raises(RuntimeError, match="injected"):
        stream.pixel_runs(
            torch.zeros(100, dtype=torch.int64, device=device), memory=memory
        )
    assert memory.get_pointers() == before


def test_gathered_groups_and_composed_final_records(device):
    n = 1003
    gen = torch.Generator().manual_seed(9)
    order = torch.randperm(n, generator=gen)
    pix = torch.arange(n, dtype=torch.int64) // 4
    group = torch.randint(-100, 100, (n,), generator=gen, dtype=torch.int64)
    depth = torch.randn(n, generator=gen)
    cov = torch.rand(n, generator=gen)
    mask = torch.randint(0, 2**28, (n,), generator=gen, dtype=torch.int32)
    got = stream.gather_group_stream(
        *(a.to(device) for a in (order, pix, group, depth, cov, mask))
    )
    expected = [a[order] for a in (pix, depth, cov, mask)]
    starts = torch.ones(n, dtype=torch.bool)
    starts[1:] = (pix[order][1:] != pix[order][:-1]) | (
        group[order][1:] != group[order][:-1]
    )
    for a, e in zip(got, [*expected, starts]):
        assert_exact(a, e)
    final = torch.randperm(n, generator=gen)
    nearest = torch.randperm(n, generator=gen)
    rep = torch.randperm(n, generator=gen)
    # Include high keys and nontrivial low bits: a float-mediated int64 gather
    # would corrupt these on MPS. Nearest and dominant refs deliberately differ.
    key = (pix << 32) | (torch.arange(n) * 7919 + 123456789)
    ref = torch.arange(n, dtype=torch.int32) - n // 2
    ab = torch.rand(n, 2, generator=gen)
    cap = torch.rand(n, generator=gen)
    nfrag = torch.randint(1, 100, (n,), generator=gen, dtype=torch.int64)
    fused = ref < 0
    for band in (None, group):
        args = (final, nearest, rep, key, ref, ab, cap, cov, mask, nfrag, fused)
        out, out_band = stream.gather_sheet_records(
            *(a.to(device) for a in args), None if band is None else band.to(device)
        )
        expected = {
            "sheet_key": key[nearest[final]],
            "sheet_pix": pix[nearest[final]],
            "sheet_ref": ref[rep[final]],
            "sheet_ab": ab[rep[final]],
            "sheet_cap": cap[rep[final]],
            "sheet_cov": cov[final],
            "sheet_msk": mask[final],
            "sheet_nfrag": nfrag[final],
            "sheet_fused": fused[final],
        }
        for name in expected:
            assert_exact(out[name], expected[name])
        if band is None:
            assert out_band is None
        else:
            assert_exact(out_band, band[final])


@pytest.mark.parametrize("n", [0, 1, 31, 1033, 8193])
def test_fragment_run_sort_is_stable(device, n):
    gen = torch.Generator().manual_seed(7)
    # A long equal-key run exercises heapsort without a per-pixel ceiling.
    keys = (
        torch.randint(0, max(1, n // 100), (n,), generator=gen, dtype=torch.int64) << 35
    )
    layers = torch.randint(-(2**31), 2**31 - 1, (n,), generator=gen, dtype=torch.int32)
    if n > 1:
        layers[1::2] = 3
    layer_order = torch.argsort(layers, descending=True, stable=True)
    expected = layer_order[torch.argsort(keys[layer_order], stable=True)]
    got = stream.fragment_run_order(keys.to(device), layers.to(device))
    assert_exact(got, expected)


@pytest.mark.parametrize("shade_split", [False, True])
@pytest.mark.parametrize("sample_depth", [False, True])
def test_complete_sheet_compaction_parity(device, shade_split, sample_depth):
    from algan.rendering.raytracing.sheets import compact_sheets
    from tests.unit_tests.test_sheet_compaction import _coverage

    frags = [
        (0, 1.0, 0, 0.25, 1),
        (0, 1.001, 1, 0.25, 2),
        (0, 1.002, 2, 0.25, 1),
        (0, 1.01, -1, 0.5, 15),
        (3, 1.0, 4, 0.5, 3),
        (3, 1.001, 5, 0.5, 12),
    ]
    coverage, merged, cam, pws = _coverage(frags)
    coverage = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in coverage.items()
    }
    merged = {k: v.to(device) for k, v in merged.items()}
    memory = ManualMemory(1.0, device=device, num_bytes=1 << 20)
    results = []
    before = memory.get_pointers()
    for enabled in (False, True):
        SETTINGS.raytracing.experimental.sheet_fused_stream = enabled
        with memory.temp():
            results.append(
                compact_sheets(
                    coverage,
                    merged,
                    cam.to(device),
                    pws.to(device),
                    0,
                    8,
                    4,
                    shade_split=shade_split,
                    sample_depth=sample_depth,
                    memory=memory,
                )
            )
        assert memory.get_pointers() == before
    # Overwrite every freed scratch byte before inspecting returned records.
    # This catches a final output accidentally aliasing the sorted arena stream.
    with memory.temp():
        memory.get_tensor((memory.get_num_bytes_remaining(),), torch.uint8).fill_(255)
    for name, expected in results[0].items():
        actual = results[1][name]
        if isinstance(expected, torch.Tensor):
            assert_exact(actual, expected)
        else:
            assert actual == expected


def test_live_switches():
    from algan.rendering.raytracing import settings

    for name in ("sheet_device_runs", "sheet_fused_stream", "sheet_fragment_run_sort"):
        setattr(SETTINGS.raytracing.experimental, name, True)
        assert getattr(settings, name) is True
        setattr(SETTINGS.raytracing.experimental, name, False)
        assert getattr(settings, name) is False


def test_fragment_order_preserves_depth_bins_and_geometry_layers(device, monkeypatch):
    from algan.rendering.raytracing import raster_pipeline as pipeline

    monkeypatch.setattr(pipeline.device_sort, "radix_sort_available", lambda _: False)
    generator = torch.Generator().manual_seed(147)
    count = 513
    pixels = torch.randint(0, 17, (count,), generator=generator, dtype=torch.int64)
    depths = torch.randint(0, 20, (count,), generator=generator).float() + 0.25
    depths *= pipeline.depth_tie_epsilon
    # Deliberately shuffle depth within an equal bin: true depth must not
    # displace the reference's geometry-layer tie breaker.
    depths[::5] += 0.5 * pipeline.depth_tie_epsilon
    key = (pixels << 32) | (depths.view(torch.int32).to(torch.int64) & 0xFFFFFFFF)
    refs = torch.randint(-2048, 127, (count,), generator=generator, dtype=torch.int32)
    key, refs = key.to(device), refs.to(device)
    SETTINGS.raytracing.experimental.sheet_fragment_run_sort = False
    expected = pipeline._exact_fragment_order(key, refs, 211)
    SETTINGS.raytracing.experimental.sheet_fragment_run_sort = True
    actual = pipeline._exact_fragment_order(key, refs, 211)
    assert_exact(actual, expected)


def test_kernel_gate_rejects_uninitialized_or_oversized_inputs(monkeypatch):
    from types import SimpleNamespace

    from algan.rendering import taichi_runtime

    monkeypatch.setattr(taichi_runtime, "_live_arch", lambda: None)
    assert not stream.stream_kernel_available(torch.empty(4))
    monkeypatch.setattr(taichi_runtime, "_live_arch", lambda: object())
    monkeypatch.setattr(taichi_runtime, "taichi_launch_is_local", lambda _: True)
    # Exercise the integer capacity bound without allocating enormous tensors.
    assert not stream.stream_kernel_available(SimpleNamespace(shape=(2**31,)))
