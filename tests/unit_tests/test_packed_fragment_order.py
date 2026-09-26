"""The packed one-sort fragment order equals the two-sort reference exactly.

``raster_pipeline._packed_fragment_order`` replaces the emission's two stable
sorts (descending layer, then ``(pixel << 32) | depth bin``) with one stable
sort of a mixed-radix key. These compare the permutations on random streams
with heavy ties in every column -- repeated pixels, repeated depth bins,
repeated layers -- on the CPU (forced past the CUDA-only gate) and on CUDA
when it is present.
"""

from __future__ import annotations

import pytest
import torch

from algan.rendering.raytracing import raster_pipeline as rpl

_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _stream(seed, n, pixels, depths, layers, device):
    g = torch.Generator().manual_seed(seed)
    pix = torch.randint(0, pixels, (n,), generator=g, dtype=torch.int64)
    # A few distinct depths so depth bins tie; some exactly on bin edges.
    choices = torch.rand(depths, generator=g) * 20.0
    t = choices[torch.randint(0, depths, (n,), generator=g)]
    key = (pix << 32) | (t.view(torch.int32).to(torch.int64) & 0xFFFFFFFF)
    ref = torch.randint(-(layers << 8), layers, (n,), generator=g, dtype=torch.int32)
    return key.to(device), ref.to(device)


def _reference(key, ref, offset):
    is_bez = ref < 0
    bez_layer = (-ref - 1).clamp_min(0) >> rpl._BEZ_BORDER_BITS
    layer = torch.where(is_bez, bez_layer, ref + offset).to(torch.int32)
    layer_order = torch.argsort(layer, descending=True, stable=True)
    primary = rpl._primary_depth_key(key.index_select(0, layer_order))
    return layer_order.index_select(0, torch.argsort(primary, stable=True)), layer


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize(
    ("n", "pixels", "depths", "layers"),
    [(5000, 40, 3, 7), (20000, 5000, 50, 300), (3000, 3000, 1, 1), (4000, 1, 2, 4000)],
)
def test_packed_order_matches_two_sorts(device, n, pixels, depths, layers):
    key, ref = _stream(n + pixels, n, pixels, depths, layers, device)
    want, layer = _reference(key, ref, offset=layers)
    got = rpl._packed_fragment_order(key, layer, min_fragments=0, devices=(device,))
    assert got is not None
    assert got.dtype == want.dtype
    assert torch.equal(got, want)


def test_packed_order_declines_off_gate():
    key, ref = _stream(0, 100, 10, 2, 3, "cpu")
    _want, layer = _reference(key, ref, offset=3)
    assert rpl._packed_fragment_order(key, layer) is None


def _kernel_device():
    """The device whose tensors the live Taichi arch launches on in place."""
    from algan.rendering.taichi_runtime import _live_arch, init_taichi
    from algan.taichi_compat import ti

    init_taichi()
    return "cuda" if _live_arch() == ti.cuda else "cpu"


@pytest.mark.parametrize(
    ("n", "pixels", "depths", "layers"),
    [
        (5000, 40, 3, 7),
        (20000, 5000, 50, 300),
        (3000, 3000, 1, 1),
        (4000, 1, 2, 4000),
        (30000, 700, 4, 20),
    ],
)
def test_binned_order_matches_two_sorts(n, pixels, depths, layers):
    device = _kernel_device()
    key, ref = _stream(n + pixels + 7, n, pixels, depths, layers, device)
    want, layer = _reference(key, ref, offset=layers)
    got = rpl._binned_fragment_order(
        key, layer, pixels, min_fragments=0, devices=(device,)
    )
    assert got is not None
    assert got.dtype == want.dtype
    assert torch.equal(got, want)


def test_binned_order_declines_out_of_range_pixels():
    device = _kernel_device()
    key, ref = _stream(1, 1000, 50, 2, 3, device)
    _want, layer = _reference(key, ref, offset=3)
    # 49 is the largest pixel; a window of 40 bins leaves keys outside it.
    assert (
        rpl._binned_fragment_order(key, layer, 40, min_fragments=0, devices=(device,))
        is None
    )
