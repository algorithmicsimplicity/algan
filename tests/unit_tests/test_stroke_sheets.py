"""The stroke-only compaction shortcut reproduces ``compact_sheets`` bit for bit.

A stream holding no triangle fragment -- every chunk of a 2-D scene -- makes
one sheet per fragment, so ``compact_sheets`` writes it directly
(``sheets._stroke_sheets``) instead of sorting, grouping and reducing. These
build random bezier-only streams and compare the shortcut (taken when the
emission reports ``num_tri_fragments == 0``) against the general path (taken
when the count is absent), output by output and bit by bit.
"""

from __future__ import annotations

import pytest
import torch

from algan.rendering.raytracing.raster_taichi import (
    _AA_MASK_ALL as MASK_ALL,
)
from algan.rendering.raytracing.raster_taichi import (
    _AA_ONE_MESH_BIT as ONE_MESH,
)
from algan.rendering.raytracing.raster_taichi import (
    _AA_SLIVER_BIT as SLIVER,
)
from algan.rendering.raytracing.sheets import compact_sheets

_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _stroke_stream(seed, device, pixels=300, width=64, height=32, frames=2):
    g = torch.Generator().manual_seed(seed)
    ppf = width * height
    pix = torch.randperm(frames * ppf, generator=g)[:pixels].sort().values
    per = torch.randint(1, 7, (pixels,), generator=g)
    pix = pix.repeat_interleave(per)
    n = int(pix.numel())
    t = torch.rand(n, generator=g) * 5.0 + 0.5
    key = (pix << 32) | (t.view(torch.int32).to(torch.int64) & 0xFFFFFFFF)
    # Circuit references are negative: -(code << border bits | weight) - 1.
    ref = -torch.randint(1, 1 << 20, (n,), generator=g, dtype=torch.int32)
    cov = torch.rand(n, generator=g) * 1.2
    pick = torch.randint(0, 6, (n,), generator=g)
    cov[pick == 0] = 0.0
    cov[pick == 1] = -0.0
    cov[pick == 2] = 1.0
    # No negative areas: the general path's dominant-fragment search starts
    # from 0 and cannot represent a band whose areas are all below it.
    cov[pick == 3] = 1e-9
    msk = torch.randint(0, MASK_ALL + 1, (n,), generator=g, dtype=torch.int32)
    msk[torch.randint(0, 4, (n,), generator=g) == 0] = 0
    msk[torch.randint(0, 5, (n,), generator=g) == 0] |= SLIVER
    msk[torch.randint(0, 5, (n,), generator=g) == 0] |= ONE_MESH
    covered, counts = torch.unique_consecutive(pix, return_counts=True)
    run_offsets = torch.zeros(covered.numel() + 1, dtype=torch.int32)
    run_offsets[1:] = torch.cumsum(counts.to(torch.int32), 0)
    coverage = {
        "frag_key": key,
        "frag_ref": ref,
        "frag_ab": torch.rand(n, 2, generator=g),
        "frag_cov": cov,
        "frag_msk": msk,
        "frag_cap": torch.where(
            torch.rand(n, generator=g) < 0.5,
            torch.full((n,), 2.0),
            torch.rand(n, generator=g),
        ),
        "covered_idx": covered.to(torch.int32),
        "run_offsets": run_offsets,
        "num_fragments": n,
        "num_covered": int(covered.numel()),
    }
    coverage = {
        k: v.to(device) if torch.is_tensor(v) else v for k, v in coverage.items()
    }
    num_tris = 2
    merged = {
        "tri_obj": torch.zeros(frames, num_tris, dtype=torch.int32, device=device),
        "tri_pos": torch.rand(frames, num_tris, 9, generator=g).to(device),
        "tri_norm": torch.rand(frames, num_tris, 9, generator=g).to(device),
    }
    cam = torch.zeros(frames, 3, device=device)
    pws = torch.full((frames,), 1e-3, device=device)
    return coverage, merged, cam, pws, width, height


def _run(stream, *, shortcut, resolve_only, **flags):
    coverage, merged, cam, pws, width, height = stream
    coverage = dict(coverage)
    if shortcut:
        coverage["num_tri_fragments"] = 0
    return compact_sheets(
        coverage,
        merged,
        cam,
        pws,
        time_start=0,
        width=width,
        height=height,
        band_rule="prim",
        band_c=2.0,
        resolve_only=resolve_only,
        **flags,
    )


def _bits(t):
    return t.view(torch.int32) if t.dtype == torch.float32 else t


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("resolve_only", [True, False])
@pytest.mark.parametrize(
    "flags",
    [
        {"shade_split": True, "positioned_depth": True, "sample_depth": True},
        {"shade_split": False, "positioned_depth": False, "sample_depth": False},
    ],
)
@pytest.mark.parametrize("seed", [0, 1])
def test_stroke_shortcut_matches_general_path(device, resolve_only, flags, seed):
    stream = _stroke_stream(seed, device)
    fast = _run(stream, shortcut=True, resolve_only=resolve_only, **flags)
    ref = _run(stream, shortcut=False, resolve_only=resolve_only, **flags)
    assert set(fast) == set(ref)
    for name, want in ref.items():
        got = fast[name]
        if torch.is_tensor(want):
            assert got.dtype == want.dtype, name
            assert got.shape == want.shape, name
            assert torch.equal(_bits(got), _bits(want)), name
        else:
            assert got == want, name


def test_stroke_shortcut_off_switch(monkeypatch):
    import algan.animation_timeline.timeline as tl

    stream = _stroke_stream(3, "cpu")
    monkeypatch.setattr(tl, "_OPT_DISABLED", frozenset({"strokesheets"}))
    off = _run(stream, shortcut=True, resolve_only=True, shade_split=True)
    monkeypatch.setattr(tl, "_OPT_DISABLED", frozenset())
    on = _run(stream, shortcut=True, resolve_only=True, shade_split=True)
    for name, want in off.items():
        if torch.is_tensor(want):
            assert torch.equal(_bits(on[name]), _bits(want)), name


def _mixed_stream(
    seed,
    device,
    flat,
    disjoint=False,
    pixels=200,
    width=32,
    height=16,
    num_tris=8,
    tri_share=0.6,
):
    """Triangle and circuit fragments sharing pixels; ``flat`` gives the
    triangles declared-flat normals (distinct shading classes per face), and
    ``disjoint`` gives each fragment of a pixel its own sample bit, so no band
    is rank-split (and none pools).
    """
    g = torch.Generator().manual_seed(seed)
    ppf = width * height
    pix = torch.randperm(ppf, generator=g)[:pixels].sort().values
    per = torch.randint(1, 6, (pixels,), generator=g)
    pix = pix.repeat_interleave(per)
    n = int(pix.numel())
    tri = torch.rand(n, generator=g) < tri_share
    tref = torch.randint(0, num_tris, (n,), generator=g, dtype=torch.int32)
    bref = -torch.randint(1, 1 << 16, (n,), generator=g, dtype=torch.int32)
    ref = torch.where(tri, tref, bref)
    t = 1.0 + 0.001 * tref.float() + torch.rand(n, generator=g) * 1e-4
    t = torch.where(tri, t, torch.rand(n, generator=g) * 3.0 + 0.5)
    # Emission order within a pixel: ascending depth.
    order = torch.argsort(pix.double() * 16 + t.double(), stable=True)
    pix, t, ref = pix[order], t[order], ref[order]
    key = (pix << 32) | (t.view(torch.int32).to(torch.int64) & 0xFFFFFFFF)
    cov = torch.rand(n, generator=g) * 0.9 + 0.05
    msk = torch.randint(0, MASK_ALL + 1, (n,), generator=g, dtype=torch.int32)
    covered, counts = torch.unique_consecutive(pix, return_counts=True)
    run_offsets = torch.zeros(covered.numel() + 1, dtype=torch.int32)
    run_offsets[1:] = torch.cumsum(counts.to(torch.int32), 0)
    if disjoint:
        starts = torch.repeat_interleave(run_offsets[:-1].to(torch.int64), counts)
        msk = (1 << (torch.arange(n) - starts)).to(torch.int32)
    coverage = {
        "frag_key": key,
        "frag_ref": ref,
        "frag_ab": torch.rand(n, 2, generator=g),
        "frag_cov": cov,
        "frag_msk": msk,
        "frag_cap": torch.full((n,), 2.0),
        "covered_idx": covered.to(torch.int32),
        "run_offsets": run_offsets,
        "num_fragments": n,
        "num_covered": int(covered.numel()),
        "num_tri_fragments": int(tri.sum()),
    }
    coverage = {
        k: v.to(device) if torch.is_tensor(v) else v for k, v in coverage.items()
    }
    tp = torch.zeros(1, num_tris, 9)
    for r in range(num_tris):
        z = 1.0 + 0.001 * r
        tp[0, r] = torch.tensor([0.0, 0, z, 0.05, 0, z, 0, 0.05, z])
    if flat:
        nrm = torch.randn(1, num_tris, 3, generator=g)
        tn = nrm.repeat(1, 1, 3)
    else:
        tn = torch.randn(1, num_tris, 9, generator=g)
    merged = {
        "tri_obj": torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1][:num_tris]], device=device),
        "tri_pos": tp.to(device),
        "tri_norm": tn.to(device),
    }
    cam = torch.zeros(1, 3, device=device)
    pws = torch.full((1,), 1e-3, device=device)
    return coverage, merged, cam, pws, width, height


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("resolve_only", [True, False])
@pytest.mark.parametrize("sample_depth", [True, False])
@pytest.mark.parametrize("disjoint", [False, True])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_solo_band_skip_matches_sibling_weights(
    device, flat, resolve_only, sample_depth, disjoint, seed, monkeypatch
):
    import algan.animation_timeline.timeline as tl

    stream = _mixed_stream(seed, device, flat, disjoint)
    flags = {
        "shade_split": True,
        "positioned_depth": True,
        "sample_depth": sample_depth,
    }
    monkeypatch.setattr(tl, "_OPT_DISABLED", frozenset({"solobands"}))
    ref = _run(stream, shortcut=False, resolve_only=resolve_only, **flags)
    monkeypatch.setattr(tl, "_OPT_DISABLED", frozenset())
    got = _run(stream, shortcut=False, resolve_only=resolve_only, **flags)
    assert set(got) == set(ref)
    for name, want in ref.items():
        if torch.is_tensor(want):
            assert got[name].dtype == want.dtype, name
            assert torch.equal(_bits(got[name]), _bits(want)), name
        else:
            assert got[name] == want, name


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("resolve_only", [True, False])
@pytest.mark.parametrize("sample_depth", [True, False])
@pytest.mark.parametrize("disjoint", [False, True])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_stroke_pixel_split_matches_general_path(
    device, flat, resolve_only, sample_depth, disjoint, seed, monkeypatch
):
    import algan.animation_timeline.timeline as tl
    import algan.rendering.raytracing.sheets as sheets_mod

    stream = _mixed_stream(seed, device, flat, disjoint, tri_share=0.08)
    flags = {
        "shade_split": True,
        "positioned_depth": True,
        "sample_depth": sample_depth,
    }
    monkeypatch.setattr(tl, "_OPT_DISABLED", frozenset({"strokepixels"}))
    ref = _run(stream, shortcut=False, resolve_only=resolve_only, **flags)
    monkeypatch.setattr(tl, "_OPT_DISABLED", frozenset())
    calls = []
    real = sheets_mod._split_stroke_pixels

    def spy(*a, **k):
        out = real(*a, **k)
        calls.append(out is not None)
        return out

    monkeypatch.setattr(sheets_mod, "_split_stroke_pixels", spy)
    got = _run(stream, shortcut=False, resolve_only=resolve_only, **flags)
    assert calls == [True]  # the split was taken, not declined
    assert set(got) == set(ref)
    for name, want in ref.items():
        if torch.is_tensor(want):
            assert got[name].dtype == want.dtype, name
            assert got[name].shape == want.shape, name
            assert torch.equal(_bits(got[name]), _bits(want)), name
        else:
            assert got[name] == want, name
