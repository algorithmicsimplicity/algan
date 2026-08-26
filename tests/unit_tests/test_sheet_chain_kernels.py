"""The five sheet-chain host-pass kernels against the torch they replaced.

``solid_shell_ceiling``, ``band_stats_reduce``, ``band_stats_rep_orig``,
``pair_expand_count`` and ``pair_expand_write`` each replace a multi-pass
torch block in the sheet chain, and every replacement claims bit-identity.
These tests launch each kernel on small real inputs and compare bitwise
against the exact torch statements they replaced -- the only way to catch a
Taichi scoping/compile error, which no host-side inspection of the kernel
source can see. ``benchmarks/_sheet_kernel_check.py`` covers 4K-scale shapes;
this file pins small deterministic cases, including the routing gate in
``_class_pairs_flat``.

Not marked fast: each test pays a Taichi compile its own module owns (the
convention in ``test_sheet_compaction.py``).
"""

from __future__ import annotations

import pytest
import torch

from algan.rendering.raytracing.raster_pipeline import (
    RASTER_CHUNK,
    _class_pairs_flat,
)
from algan.rendering.raytracing.raster_taichi import _AA_MASK_ALL as MASK_ALL
from algan.rendering.raytracing.sheet_compact_taichi import (
    band_stats_reduce,
    band_stats_rep_orig,
    pair_expand_count,
    pair_expand_write,
    solid_shell_ceiling,
)
from algan.rendering.raytracing.sheets import _lexsort
from algan.settings import SETTINGS

EXPERIMENTAL = SETTINGS.raytracing.experimental


def _bits_equal(a, b):
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if a.dtype.is_floating_point:
        width = torch.int32 if a.dtype == torch.float32 else torch.int64
        return bool((a.view(width) == b.view(width)).all())
    return bool((a == b).all())


# ----------------------------------------------------- solid_shell_ceiling


def _shell_stream(n, seed, zero_areas=False):
    g = torch.Generator().manual_seed(seed)
    # Mostly closed-surface (pixel, surface) segments plus unique-negative
    # pass-through keys, exactly the two kinds compact_sheets emits.
    pix = torch.randint(0, 50, (n,), generator=g)
    sid = torch.randint(0, 3, (n,), generator=g)
    closed = torch.rand(n, generator=g) < 0.75
    key = torch.where(closed, pix * 5 + sid, -(torch.arange(n) + 1))
    depth = torch.rand(n, generator=g)
    o2 = _lexsort(key, depth)
    back = torch.rand(n, generator=g) < 0.5
    cov = torch.rand(n, generator=g)
    if zero_areas:
        cov = torch.full((n,), 1e-13)
        cov[::7] = 0.0
    return key.contiguous(), o2.contiguous(), back, cov.contiguous()


def _torch_shell_ceiling(key, o2, back, excl, c2, cov_src):
    """The verbatim torch block from compact_sheets (toggle-off arm)."""
    n = int(key.numel())
    device = key.device
    k2 = key.index_select(0, o2)
    seg_start = torch.ones(n, dtype=torch.bool, device=device)
    if n > 1:
        seg_start[1:] = k2[1:] != k2[:-1]
    seg = torch.cumsum(seg_start.to(torch.int64), 0) - 1
    nseg = int(seg[-1].item()) + 1
    first = torch.zeros(nseg, dtype=torch.int64, device=device)
    first.scatter_(0, seg[seg_start], torch.nonzero(seg_start).reshape(-1))
    spent = excl - excl.index_select(0, first).index_select(0, seg)
    backf2 = back.index_select(0, o2)
    z64 = torch.zeros((), dtype=torch.float64, device=device)
    front = torch.zeros(nseg, dtype=torch.float64, device=device)
    back_s = torch.zeros(nseg, dtype=torch.float64, device=device)
    front.scatter_add_(0, seg, torch.where(backf2, z64, c2))
    back_s.scatter_add_(0, seg, torch.where(backf2, c2, z64))
    cap = torch.maximum(front, back_s).to(torch.float32).to(torch.float64)
    scale = (
        cap.index_select(0, seg)
        .sub_(spent)
        .clamp_min_(0.0)
        .div_(c2.clamp_min_(1e-12))
        .clamp_max_(1.0)
    )
    out = cov_src.clone()
    out.index_copy_(0, o2, (c2 * scale).to(torch.float32))
    return out


@pytest.mark.parametrize("zero_areas", [False, True])
def test_solid_shell_ceiling_matches_torch_segment_clamp(zero_areas):
    n = 4096
    key, o2, back, cov = _shell_stream(n, seed=11, zero_areas=zero_areas)
    cov64 = cov.to(torch.float64)
    c2 = cov64.index_select(0, o2)
    excl = torch.cumsum(c2, 0).sub_(c2)

    want = _torch_shell_ceiling(key, o2, back, excl, c2.clone(), cov)

    got = cov.clone()
    scratch = cov.to(torch.float64)
    solid_shell_ceiling(
        key, o2, back.contiguous().view(torch.uint8), excl, scratch, n, got
    )
    assert _bits_equal(want, got), (
        f"max abs diff {(want - got).abs().max().item()}, "
        f"{int((want.view(torch.int32) != got.view(torch.int32)).sum())} bits-differing rows"
    )

    again = cov.clone()
    solid_shell_ceiling(
        key, o2, back.contiguous().view(torch.uint8), excl, scratch, n, again
    )
    assert _bits_equal(got, again), "kernel is not deterministic across runs"


def test_solid_shell_ceiling_single_fragment_and_whole_stream_segments():
    for n, key_val in ((1, 0), (257, 0)):
        key = torch.full((n,), key_val, dtype=torch.int64)
        o2 = torch.arange(n, dtype=torch.int64)
        back = torch.zeros(n, dtype=torch.bool)
        if n > 1:
            back[::3] = True
        cov = torch.linspace(0.01, 0.9, n)
        cov64 = cov.to(torch.float64)
        c2 = cov64.index_select(0, o2)
        excl = torch.cumsum(c2, 0).sub_(c2)
        want = _torch_shell_ceiling(key, o2, back, excl, c2.clone(), cov)
        got = cov.clone()
        solid_shell_ceiling(
            key,
            o2,
            back.contiguous().view(torch.uint8),
            excl,
            cov.to(torch.float64),
            n,
            got,
        )
        assert _bits_equal(want, got), f"n={n}"


# ------------------------------------------------------- band stats fusion


def _band_torch_arms(band, msk, pos_o, cov, nb, positioned):
    """The verbatim scatters from compact_sheets (toggle-off arms)."""
    n = int(band.numel())
    dev = band.device
    fs = torch.full((nb,), n, dtype=torch.int64, device=dev)
    fs.scatter_reduce_(
        0,
        band,
        torch.arange(n, dtype=torch.int64, device=dev),
        reduce="amin",
        include_self=True,
    )
    mp = torch.full((nb,), n, dtype=torch.int64, device=dev)
    mp.scatter_reduce_(0, band, pos_o, reduce="amin", include_self=True)
    fsp = torch.full((nb,), n, dtype=torch.int64, device=dev)
    mpp = torch.full((nb,), n, dtype=torch.int64, device=dev)
    if positioned:
        big = torch.full((), n, dtype=torch.int64, device=dev)
        posn = (msk & MASK_ALL) != 0
        masked = torch.where(posn, torch.arange(n, dtype=torch.int64, device=dev), big)
        fsp.scatter_reduce_(0, band, masked, reduce="amin", include_self=True)
        masked = torch.where(posn, pos_o, big)
        mpp.scatter_reduce_(0, band, masked, reduce="amin", include_self=True)
    cm = torch.zeros(nb, dtype=torch.float32, device=dev)
    cm.scatter_reduce_(0, band, cov, reduce="amax", include_self=True)
    nf = torch.zeros(nb, dtype=torch.int64, device=dev)
    nf.scatter_add_(0, band, torch.ones_like(band))
    is_max = cov >= cm.index_select(0, band)
    cand = torch.where(
        is_max, pos_o, torch.full((n,), n, dtype=torch.int64, device=dev)
    )
    rp = torch.full((nb,), n, dtype=torch.int64, device=dev)
    rp.scatter_reduce_(0, band, cand, reduce="amin", include_self=True)
    return fs, mp, fsp, mpp, cm, nf, rp


@pytest.mark.parametrize("positioned", [False, True])
def test_band_stats_kernels_match_the_five_scatters(positioned):
    g = torch.Generator().manual_seed(23)
    n, nb = 30_000, 4_001
    band = torch.randint(0, nb, (n,), generator=g)
    pos_o = torch.randperm(n, generator=g)
    msk = torch.randint(0, 1 << 22, (n,), generator=g, dtype=torch.int32)
    cov = torch.rand(n, generator=g)
    cov[::11] = 0.0

    w_fs, w_mp, w_fsp, w_mpp, w_cm, w_nf, w_rp = _band_torch_arms(
        band, msk, pos_o, cov, nb, positioned
    )

    fs = torch.full((nb,), n, dtype=torch.int64)
    mp = torch.full((nb,), n, dtype=torch.int64)
    fsp = torch.full((nb,), n, dtype=torch.int64)
    mpp = torch.full((nb,), n, dtype=torch.int64)
    cm = torch.zeros(nb, dtype=torch.float32)
    nf = torch.zeros(nb, dtype=torch.int64)
    band_stats_reduce(
        band.contiguous(),
        msk.contiguous(),
        pos_o.contiguous(),
        cov.contiguous(),
        n,
        int(MASK_ALL),
        fs,
        mp,
        fsp,
        mpp,
        cm,
        nf,
        bool(positioned),
    )
    rp = torch.full((nb,), n, dtype=torch.int64)
    band_stats_rep_orig(
        band.contiguous(), pos_o.contiguous(), cov.contiguous(), cm, n, rp
    )

    assert _bits_equal(w_fs, fs), "first_sorted"
    assert _bits_equal(w_mp, mp), "min_pos"
    assert _bits_equal(w_fsp, fsp), "first_sorted_p"
    assert _bits_equal(w_mpp, mpp), "min_pos_p"
    assert _bits_equal(w_cm, cm), "cmax"
    assert _bits_equal(w_nf, nf), "nfrag"
    assert _bits_equal(w_rp, rp), "rep_orig"


def test_band_stats_leaves_unused_band_rows_at_sentinel():
    g = torch.Generator().manual_seed(29)
    n, nb = 512, 128
    band = torch.randint(0, nb // 2, (n,), generator=g)  # upper half unused
    pos_o = torch.randperm(n, generator=g)
    msk = torch.full((n,), MASK_ALL, dtype=torch.int32)
    cov = torch.rand(n, generator=g)

    w_fs, w_mp, w_fsp, w_mpp, w_cm, w_nf, w_rp = _band_torch_arms(
        band, msk, pos_o, cov, nb, True
    )
    fs = torch.full((nb,), n, dtype=torch.int64)
    mp = torch.full((nb,), n, dtype=torch.int64)
    fsp = torch.full((nb,), n, dtype=torch.int64)
    mpp = torch.full((nb,), n, dtype=torch.int64)
    cm = torch.zeros(nb, dtype=torch.float32)
    nf = torch.zeros(nb, dtype=torch.int64)
    band_stats_reduce(
        band.contiguous(),
        msk.contiguous(),
        pos_o.contiguous(),
        cov.contiguous(),
        n,
        int(MASK_ALL),
        fs,
        mp,
        fsp,
        mpp,
        cm,
        nf,
        True,
    )
    rp = torch.full((nb,), n, dtype=torch.int64)
    band_stats_rep_orig(
        band.contiguous(), pos_o.contiguous(), cov.contiguous(), cm, n, rp
    )
    assert _bits_equal(w_fs, fs), "first_sorted"
    assert _bits_equal(w_mp, mp), "min_pos"
    assert _bits_equal(w_fsp, fsp), "first_sorted_p"
    assert _bits_equal(w_mpp, mpp), "min_pos_p"
    assert _bits_equal(w_rp, rp)


# ---------------------------------------------------- pair-row expansion


def _torch_pair_rows(mask, x0, x1, y0, y1, f_abs):
    """The verbatim torch body of _class_pairs_flat."""
    ncirc = mask.shape[1]
    idx = mask.reshape(-1).nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return None
    bx0 = x0.reshape(-1)[idx]
    by0 = y0.reshape(-1)[idx]
    bw = x1.reshape(-1)[idx] - bx0 + 1
    bh = y1.reshape(-1)[idx] - by0 + 1
    area = bw * bh
    nch = (area + (RASTER_CHUNK - 1)) // RASTER_CHUNK
    rep = torch.repeat_interleave(torch.arange(idx.numel()), nch)
    if rep.numel() == 0:
        return None
    base = torch.cumsum(nch, 0) - nch
    off = (torch.arange(rep.shape[0]) - base[rep]) * RASTER_CHUNK
    rows = torch.stack(
        [
            (idx % ncirc)[rep],
            f_abs.index_select(0, idx // ncirc)[rep],
            bx0[rep],
            by0[rep],
            bw[rep],
            bh[rep],
            off,
            torch.zeros_like(rep),
        ],
        -1,
    )
    return rows.to(torch.int32).contiguous()


def _kernel_pair_rows(mask, x0, x1, y0, y1, f_abs):
    """The host shape _pair_expand_rows gives the two kernels."""
    numel = int(mask.numel())
    ncirc = mask.shape[1]
    mflat = mask.reshape(-1).contiguous()
    x0f = x0.reshape(-1).contiguous()
    x1f = x1.reshape(-1).contiguous()
    y0f = y0.reshape(-1).contiguous()
    y1f = y1.reshape(-1).contiguous()
    counts = torch.empty(numel, dtype=torch.int64)
    pair_expand_count(
        mflat.view(torch.uint8), x0f, x1f, y0f, y1f, numel, RASTER_CHUNK, counts
    )
    offs = torch.cumsum(counts, 0) - counts
    total = int(counts.sum().item())
    if total == 0:
        return None
    rows = torch.empty((total, 8), dtype=torch.int32)
    pair_expand_write(
        x0f,
        x1f,
        y0f,
        y1f,
        f_abs.contiguous(),
        offs,
        numel,
        ncirc,
        RASTER_CHUNK,
        total,
        rows,
    )
    return rows


def test_pair_expand_kernels_match_class_pairs_flat_body():
    g = torch.Generator().manual_seed(31)
    ft, ncirc = 3, 64
    # Interleaved candidates and non-candidates, multi-frame so both e % ncirc
    # and e // ncirc carry real values; one candidate's area is an exact
    # multiple of RASTER_CHUNK, another's degenerate (x0 == x1, y0 == y1).
    mask = torch.rand(ft, ncirc, generator=g) < 0.4
    mask[0, ::16] = True
    x0 = torch.randint(0, 800, (ft, ncirc), generator=g)
    x1 = x0 + torch.randint(1, 100, (ft, ncirc), generator=g)
    y0 = torch.randint(0, 450, (ft, ncirc), generator=g)
    y1 = y0 + torch.randint(1, 80, (ft, ncirc), generator=g)
    # exact-multiple area: width 32 x height 4 -> 128 = 4 chunks
    mask[1, 5] = True
    x0[1, 5], x1[1, 5], y0[1, 5], y1[1, 5] = 10, 41, 20, 23
    # degenerate one-pixel bbox
    mask[2, 7] = True
    x0[2, 7], x1[2, 7], y0[2, 7], y1[2, 7] = 300, 300, 200, 200
    f_abs = torch.arange(ft) + 17

    want = _torch_pair_rows(mask, x0, x1, y0, y1, f_abs)
    got = _kernel_pair_rows(mask, x0, x1, y0, y1, f_abs)
    assert want is not None
    assert got is not None
    assert got.dtype == torch.int32
    assert got.is_contiguous()
    assert _bits_equal(want, got), (
        f"row mismatch at "
        f"{int((want.view(torch.int32) != got.view(torch.int32)).any(-1).sum())}"
        f"/{want.shape[0]} rows"
    )


def test_pair_expand_empty_window_returns_none():
    mask = torch.zeros(2, 33, dtype=torch.bool)
    zeros = torch.zeros(2, 33, dtype=torch.int64)
    assert _kernel_pair_rows(mask, zeros, zeros, zeros, zeros, torch.arange(2)) is None


def test_class_pairs_flat_routes_by_toggle_and_agrees():
    g = torch.Generator().manual_seed(37)
    ft, ncirc = 2, 128
    mask = torch.rand(ft, ncirc, generator=g) < 0.35
    mask[0, ::9] = True
    x0 = torch.randint(0, 800, (ft, ncirc), generator=g)
    x1 = x0 + torch.randint(1, 90, (ft, ncirc), generator=g)
    y0 = torch.randint(0, 450, (ft, ncirc), generator=g)
    y1 = y0 + torch.randint(1, 60, (ft, ncirc), generator=g)
    f_abs = torch.arange(ft) + 4

    import algan.rendering.raytracing.sheet_compact_taichi as sct

    launches = {"count": 0, "write": 0}
    real_count, real_write = sct.pair_expand_count, sct.pair_expand_write

    def counted_count(*a, **k):
        launches["count"] += 1
        return real_count(*a, **k)

    def counted_write(*a, **k):
        launches["write"] += 1
        return real_write(*a, **k)

    sct.pair_expand_count, sct.pair_expand_write = counted_count, counted_write
    try:
        EXPERIMENTAL.set(raster_pair_expand_kernel=False)
        want = _class_pairs_flat(mask, x0, x1, y0, y1, f_abs, torch.device("cpu"))
        assert launches == {"count": 0, "write": 0}, "torch arm launched a kernel"

        EXPERIMENTAL.set(raster_pair_expand_kernel=True)
        got = _class_pairs_flat(mask, x0, x1, y0, y1, f_abs, torch.device("cpu"))
        assert launches["count"] >= 1, (
            "toggle ON did not route to the pair-expand kernels"
        )
        assert launches["write"] >= 1, (
            "toggle ON did not route to the pair-expand kernels"
        )
    finally:
        sct.pair_expand_count, sct.pair_expand_write = real_count, real_write
        EXPERIMENTAL.set(raster_pair_expand_kernel=True)

    assert want is not None
    assert got is not None
    assert got.dtype == torch.int32
    assert got.is_contiguous()
    assert _bits_equal(want, got)
