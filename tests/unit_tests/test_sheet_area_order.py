"""The compaction's float32 area sums come back bit for bit on every replay.

In MPS-friendly mode the compaction sums each sheet's exact area in float32
(``mps_compat.accumulate_dtype``: Metal has no float64). Summed with float32
atomics, a sheet of three or more fragments rounds by the order the device
happens to land them in, which a GPU does not fix from one launch to the next;
``sheets._band_reduce`` sums each band in stream order instead
(``sheet_compact_taichi.sheet_band_reduce_runs``).

These build streams whose every pixel holds 33 to 96 fragments of one surface,
with areas spread over several orders of magnitude -- more fragments than a
SIMD group has lanes, so an atomic sum of one sheet crosses SIMD groups -- and
compact each one several times on every device, in MPS-friendly mode.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from algan.rendering.raytracing.raster_taichi import (
    _AA_MASK_ALL as MASK_ALL,
)
from algan.rendering.raytracing.raster_taichi import (
    _AA_NUM_SAMPLES as NUM_SAMPLES,
)
from algan.rendering.raytracing.sheets import compact_sheets
from algan.settings import SETTINGS

_DEVICES = (
    ["cpu"]
    + (["cuda"] if torch.cuda.is_available() else [])
    + (["mps"] if SETTINGS.computing.render_device.type == "mps" else [])
)
#: Compactions of each stream; every one must match the first.
_REPLAYS = 5


@pytest.fixture
def float32_areas(monkeypatch):
    """MPS-friendly mode, whatever the device, and restored afterwards.
    ``ALGAN_MPS_FRIENDLY`` would override the setting, so it is cleared.
    """
    monkeypatch.delenv("ALGAN_MPS_FRIENDLY", raising=False)
    snapshot = SETTINGS.snapshot()
    SETTINGS.computing.set(mps_friendly=True)
    try:
        yield
    finally:
        SETTINGS.restore(snapshot)


def _deep_stream(device, *, seed, sampled, pixels=1024, width=64, height=32):
    """A one-frame stream of ``pixels`` covered pixels, each holding 33 to 96
    fragments of one surface at strictly increasing depths 1e-6 apart -- far
    inside one band under the ``prim`` rule -- so the compaction keeps them
    in stream order.

    With ``sampled`` false each pixel's first fragment owns every sample and
    the rest none, so no two fragments conflict and every pixel is one sheet
    summing all of its fragments. With it true every fragment owns one
    random sample and each of eight triangles shades flat with its own normal,
    so the other two sums matter as well: the shading classes split bands
    into siblings weighted by their band's area (``sheets._band_composite``),
    and the samples conflict into ranks that pool or not by their whole
    band's area (``sheets._rank_pool_groups``). Every fourth pixel is built
    to sit on that threshold: a first fragment owning every sample with an
    area of exactly ``sheet_rank_pool_layers`` in float32, then fragments of
    4e-8 each -- under half an ulp of it, so added after it they round away
    and it pools, while added before it they count and it does not.

    Returns the ``compact_sheets`` arguments and the fragment count per pixel.
    """
    from algan.rendering.raytracing.sheets import sheet_rank_pool_layers

    g = torch.Generator().manual_seed(seed)
    per = torch.randint(33, 97, (pixels,), generator=g)
    pix = torch.randperm(width * height, generator=g)[:pixels].sort().values
    pix = pix.repeat_interleave(per)
    n = int(pix.numel())
    first = torch.zeros(n, dtype=torch.bool)
    first[torch.cumsum(per, 0) - per] = True
    within = torch.arange(n) - (torch.cumsum(per, 0) - per).repeat_interleave(per)
    t = 1.0 + within.to(torch.float32) * 1e-6
    key = (pix << 32) | (t.view(torch.int32).to(torch.int64) & 0xFFFFFFFF)
    num_tris = 8
    ref = torch.randint(0, num_tris, (n,), generator=g, dtype=torch.int32)
    cov = torch.rand(n, generator=g) * torch.exp(torch.randn(n, generator=g) * 2.0)
    if sampled:
        lane = torch.randint(0, NUM_SAMPLES, (n,), generator=g, dtype=torch.int32)
        msk = (torch.ones(n, dtype=torch.int32) << lane).to(torch.int32)
        # Band sums about 1, where a sibling's weight is its share of an
        # unclamped band area.
        cov /= 240.0
        edge = (torch.arange(pixels) % 4 == 0).repeat_interleave(per)
        cov[edge] = 4e-8
        cov[edge & first] = float(sheet_rank_pool_layers)
        msk[edge & first] = MASK_ALL
    else:
        msk = torch.where(
            first, torch.tensor(MASK_ALL, dtype=torch.int32), torch.tensor(0)
        ).to(torch.int32)
    run_offsets = torch.zeros(pixels + 1, dtype=torch.int32)
    run_offsets[1:] = torch.cumsum(per, 0).to(torch.int32)
    coverage = {
        "frag_key": key,
        "frag_ref": ref,
        "frag_ab": torch.rand(n, 2, generator=g),
        "frag_cov": cov,
        "frag_msk": msk,
        "frag_cap": torch.full((n,), 2.0),
        "covered_idx": torch.unique_consecutive(pix).to(torch.int32),
        "run_offsets": run_offsets,
        "num_fragments": n,
        "num_covered": pixels,
        "num_tri_fragments": n,
    }
    # Flat-on triangles at depth 1 (the band rule reads vertex distances).
    tp = torch.tensor([0.0, 0, 1, 0.05, 0, 1, 0, 0.05, 1]).repeat(1, num_tris, 1)
    tn = torch.randn(1, num_tris, 3, generator=g).repeat(1, 1, 3)
    merged = {
        "tri_obj": torch.zeros(1, num_tris, dtype=torch.int32),
        "tri_pos": tp,
        "tri_norm": tn,
    }
    merged = {k: v.to(device) for k, v in merged.items()}
    coverage = {
        k: v.to(device) if torch.is_tensor(v) else v for k, v in coverage.items()
    }
    cam = torch.zeros(1, 3, device=device)
    pws = torch.full((1,), 1e-3, device=device)
    return (coverage, merged, cam, pws), per


def _compact(stream, *, shade_split):
    coverage, merged, cam, pws = stream
    return compact_sheets(
        dict(coverage),
        merged,
        cam,
        pws,
        time_start=0,
        width=64,
        height=32,
        band_rule="prim",
        band_c=4.0,
        shade_split=shade_split,
    )


def _bits(t):
    return t.view(torch.int32) if t.dtype == torch.float32 else t


def _assert_replays_match(outs):
    """Every compaction in ``outs`` returned what the first did, bit for bit."""
    want = outs[0]
    for r, got in enumerate(outs[1:], 1):
        assert set(got) == set(want)
        for name, w in want.items():
            if not torch.is_tensor(w):
                assert got[name] == w, (r, name)
                continue
            g = got[name]
            assert g.shape == w.shape, (r, name)
            assert g.dtype == w.dtype, (r, name)
            if w.numel() == 0:
                continue
            differs = (_bits(g) != _bits(w)).reshape(w.shape[0] if w.dim() else 1, -1)
            moved = int(differs.any(1).sum())
            assert moved == 0, f"replay {r}: {moved} of {len(differs)} rows of {name}"


def _stream_order_sums(cov, per):
    """Each pixel's areas summed one by one in stream order, in float32."""
    values = cov.cpu().numpy()
    sums = np.empty(per.numel(), dtype=np.float32)
    i = 0
    for p, m in enumerate(per.tolist()):
        total = np.float32(0.0)
        for v in values[i : i + m]:
            total = np.float32(total + v)
        sums[p] = total
        i += m
    return torch.from_numpy(sums)


@pytest.mark.parametrize("device", _DEVICES)
def test_float32_sheet_areas_replay_as_stream_order_sums(device, float32_areas):
    """Every replay returns the same sheets, each with its fragments' areas
    added in stream order. Replays are compared first, so that a failure
    says whether the sums moved between launches or only disagreed with the
    stream order.
    """
    from algan.rendering.mps_compat import accumulate_dtype

    assert accumulate_dtype() == torch.float32
    stream, per = _deep_stream(device, seed=0, sampled=False)
    outs = [_compact(stream, shade_split=False) for _ in range(_REPLAYS)]
    _assert_replays_match(outs)
    out = outs[0]
    # The stream compacts to one sheet per pixel, of all its fragments.
    assert int(out["num_sheets"]) == per.numel()
    assert torch.equal(out["sheet_nfrag"].cpu(), per)
    want = _stream_order_sums(stream[0]["frag_cov"], per)
    got = out["sheet_cov"].cpu()
    moved = int((got.view(torch.int32) != want.view(torch.int32)).sum())
    assert moved == 0, f"{moved} of {per.numel()} sheets are not the stream-order sum"
    assert torch.equal(out["sheet_wgt"].cpu().view(torch.int32), want.view(torch.int32))


@pytest.mark.parametrize("device", _DEVICES)
def test_float32_pooling_and_sibling_weights_replay_bit_identically(
    device, float32_areas, monkeypatch
):
    """The two other sums the compaction takes -- a conflict-ranked band's,
    which decides whether its ranks pool, and a shading-split band's, which
    weights its siblings -- replay bit for bit too.
    """
    import algan.rendering.raytracing.sheets as sheets_mod

    reached = []
    real_pool = sheets_mod._rank_pool_groups
    real_composite = sheets_mod._band_composite

    def pool(*args, **kwargs):
        n_group, group_of_cid = real_pool(*args, **kwargs)
        reached.append(("pooled", n_group < args[5]))
        return n_group, group_of_cid

    def composite(*args, **kwargs):
        reached.append(("composited", True))
        return real_composite(*args, **kwargs)

    monkeypatch.setattr(sheets_mod, "_rank_pool_groups", pool)
    monkeypatch.setattr(sheets_mod, "_band_composite", composite)
    stream, per = _deep_stream(device, seed=1, sampled=True)
    outs = [_compact(stream, shade_split=True) for _ in range(_REPLAYS)]
    # The stream reaches both sums on every replay, and some ranks pool.
    assert reached == [("pooled", True), ("composited", True)] * _REPLAYS
    _assert_replays_match(outs)
