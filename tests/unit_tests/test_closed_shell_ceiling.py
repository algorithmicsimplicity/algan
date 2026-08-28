"""The solid-shell coverage ceiling (``solid_shell_alpha``) on synthetic streams.

What ``Mob.opacity`` delivers on a declared closed shell: one attenuation of
what is behind it, per pixel, per surface -- not one per shell crossing. These
tests pin the ceiling's semantics at the compaction, on hand-built fragment
streams, the same way ``test_sheet_compaction`` pins the machinery it rides:

* a closed solid's far sheet is left no coverage where both shells tile the
  pixel, whatever runtime sample visibility would have said;
* the cap is ``max(front, back)`` of the surface's own footprints -- so the
  silhouette keeps its ink (leftover allowance flows to whichever sheet owns
  more of the pixel) and a genuine self-overlap still attenuates twice;
* every exemption holds: undeclared surfaces, transmissive ones (folded open
  at pack time, tested in ``test_closed_shell_declaration``), N independent
  translucent solids composing to ``1 - (1 - a)^N``, and the view from inside
  a solid where every crossing is back-facing.

Pure tensor assertions; no render. Feature tests of the renderer's compositing
semantics, not of anything the timeline can break: not ``fast``.
"""

from __future__ import annotations

import torch

from algan import SETTINGS
from algan.rendering.raytracing.raster_taichi import (
    _AA_BACKFACE_BIT as BACKFACE,
)
from algan.rendering.raytracing.raster_taichi import (
    _AA_MASK_ALL as MASK_ALL,
)
from algan.rendering.raytracing.sheets import compact_sheets, resolve_pixel_reference


def _key(pix, t):
    tb = int(torch.tensor([t], dtype=torch.float32).view(torch.int32)[0].item())
    return (int(pix) << 32) | (tb & 0xFFFFFFFF)


def _coverage(frags, num_tris=8, tri_closed=None):
    """A coverage dict + merged scene from ``(pix, t, ref, cov, msk)`` rows,
    with optional per-triangle closed-shell flags. Triangles 0-3 are surface 0,
    4-7 surface 1 (mirroring ``test_sheet_compaction``'s fixture).
    """
    n = len(frags)
    pix = torch.tensor([f[0] for f in frags], dtype=torch.int64)
    t = torch.tensor([f[1] for f in frags], dtype=torch.float32)
    ref = torch.tensor([f[2] for f in frags], dtype=torch.int32)
    cov = torch.tensor([f[3] for f in frags], dtype=torch.float32)
    msk = torch.tensor([f[4] for f in frags], dtype=torch.int32)
    key = torch.tensor([_key(f[0], f[1]) for f in frags], dtype=torch.int64)
    covered, counts = torch.unique_consecutive(pix, return_counts=True)
    run_offsets = torch.zeros(covered.numel() + 1, dtype=torch.int32)
    run_offsets[1:] = torch.cumsum(counts.to(torch.int32), 0)
    coverage = {
        "frag_key": key,
        "frag_ref": ref,
        "frag_ab": torch.zeros(n, 2),
        "frag_cov": cov,
        "frag_msk": msk,
        "frag_cap": torch.full((n,), 2.0),
        "covered_idx": covered.to(torch.int32),
        "run_offsets": run_offsets,
        "num_fragments": n,
        "num_covered": int(covered.numel()),
    }
    tri_obj = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1][:num_tris]])
    tp = torch.zeros(1, num_tris, 9)
    for r in range(num_tris):
        z = 1.0 + 0.001 * r
        tp[0, r] = torch.tensor([0.0, 0, z, 0.05, 0, z, 0, 0.05, z])
    merged = {"tri_obj": tri_obj, "tri_pos": tp}
    if tri_closed is not None:
        merged["tri_closed"] = torch.tensor(
            [tri_closed][:num_tris], dtype=torch.float32
        )
    cam = torch.zeros(1, 3)
    pws = torch.full((1,), 1e-3)
    return coverage, merged, cam, pws


def _compact(frags, tri_closed=None, **kw):
    return compact_sheets(
        *_coverage(frags, tri_closed=tri_closed),
        time_start=0,
        width=4,
        height=4,
        **kw,
    )


def test_the_far_shell_of_an_interior_pixel_gets_nothing():
    """Both shells tile the pixel (front = back = 1): the near crossing spends
    the whole allowance in depth order and the far one drops out -- regardless
    of what sample visibility would have granted it. The near sheet carries the
    backface bit here, matching measured emission.
    """
    out = _compact(
        [
            (3, 5.91, 0, 1.0, BACKFACE | MASK_ALL),
            (3, 8.09, 1, 1.0, MASK_ALL),
        ],
        tri_closed=[1, 1, 1, 1, 0, 0, 0, 0],
    )
    assert out["num_sheets"] == 2
    assert torch.allclose(out["sheet_cov"], torch.tensor([1.0, 0.0]))


def test_disabling_the_toggle_restores_today_behaviour():
    """With ``solid_shell_alpha`` off nothing is clamped: byte-identical to the
    pre-change compaction on the same stream.
    """
    frags = [
        (3, 5.91, 0, 1.0, BACKFACE | MASK_ALL),
        (3, 8.09, 1, 1.0, MASK_ALL),
    ]
    old = SETTINGS.raytracing.experimental.solid_shell_alpha
    try:
        assert old is True, "the fix is supposed to ship enabled"
        SETTINGS.raytracing.experimental.set(solid_shell_alpha=False)
        off = _compact(frags, tri_closed=[1, 1, 1, 1, 0, 0, 0, 0])
        SETTINGS.raytracing.experimental.set(solid_shell_alpha=True)
        # And with the toggle on but NO declaration in merged at all.
        none = _compact(frags, tri_closed=None)
    finally:
        SETTINGS.raytracing.experimental.set(solid_shell_alpha=old)
    assert torch.allclose(off["sheet_cov"], torch.tensor([1.0, 1.0]))
    assert torch.allclose(none["sheet_cov"], torch.tensor([1.0, 1.0]))


def test_undeclared_surfaces_keep_per_crossing_attenuation():
    """A surface that does not declare (an open cone, an unprovable Polyhedron)
    is exempt: its two layers survive untouched, so a ray crossing it twice
    still attenuates twice.
    """
    out = _compact(
        [
            (3, 1.0, 4, 0.6, MASK_ALL),
            (3, 2.0, 5, 0.6, MASK_ALL),
        ],
        tri_closed=[1, 1, 1, 1, 0, 0, 0, 0],
    )
    assert torch.allclose(out["sheet_cov"], torch.tensor([0.6, 0.6]))
    claims, _T = resolve_pixel_reference(
        [0.6, 0.6],
        [MASK_ALL, MASK_ALL],
        [False, False],
        alphas=[0.55, 0.55],
    )
    # Second layer sees attenuated samples: (1 - .55*.6) factor.
    assert claims[1] < claims[0]


def test_independent_translucent_solids_still_compose():
    """Two DIFFERENT declared solids stacked over one pixel each composite once:
    their sheets do not share an allowance, so the stack telescopes toward
    1 - (1 - a)^2 exactly as N independent surfaces must.
    """
    out = _compact(
        [
            (3, 1.0, 0, 1.0, BACKFACE | MASK_ALL),  # A near
            (3, 3.0, 1, 1.0, MASK_ALL),  # A far
            (3, 2.0, 4, 1.0, BACKFACE | MASK_ALL),  # B near (between A's shells)
            (3, 4.0, 5, 1.0, MASK_ALL),  # B far
        ],
        tri_closed=[1, 1, 1, 1, 1, 1, 1, 1],
    )
    covs = out["sheet_cov"].tolist()
    assert sorted(covs) == [0.0, 0.0, 1.0, 1.0]
    a = 0.55
    claims, T = resolve_pixel_reference(
        [1.0, 1.0, 0.0, 0.0],
        [BACKFACE | MASK_ALL, MASK_ALL, BACKFACE | MASK_ALL, MASK_ALL],
        [False] * 4,
        alphas=[a] * 4,
    )
    total_paint = sum(claims)
    assert abs(total_paint - (a + a * (1 - a))) < 1e-9


def test_the_view_from_inside_keeps_its_sole_crossing():
    """Camera inside the solid: the single crossing is back-facing, front sums
    to zero, and max(front, back) leaves that sheet everything.
    """
    out = _compact(
        [(3, 1.0, 0, 1.0, BACKFACE | MASK_ALL)],
        tri_closed=[1, 1, 1, 1, 0, 0, 0, 0],
    )
    assert torch.allclose(out["sheet_cov"], torch.tensor([1.0]))


def test_a_genuine_self_overlap_still_attenuates_twice():
    """The conflict-rank contract survives: a declared surface whose ray
    crosses it twice on the SAME side sums front past 1, the unclamped cap
    keeps both crossings, nothing fuses.
    """
    out = _compact(
        [
            (3, 1.0, 0, 1.0, MASK_ALL),
            (3, 1.0001, 1, 1.0, MASK_ALL),
        ],
        tri_closed=[1, 1, 1, 1, 0, 0, 0, 0],
    )
    assert torch.allclose(out["sheet_cov"], torch.tensor([1.0, 1.0]))


def test_silhouette_leftover_flows_to_the_sheet_that_owns_more():
    """At the rim the two shells' footprints differ. The near sheet spends its
    own area first; the far sheet keeps the REMAINDER of max(front, back) on
    the samples only it covers -- the ink guard. Plain suppression would have
    zeroed it.
    """
    low = 0b00000011  # near sheet covers a thin sliver
    high = 0b11111100  # far sheet owns much more of this boundary pixel
    out = _compact(
        [
            (3, 1.0, 0, 0.25, low),
            (3, 2.0, 1, 0.75, BACKFACE | high),
        ],
        tri_closed=[1, 1, 1, 1, 0, 0, 0, 0],
    )
    assert torch.allclose(out["sheet_cov"], torch.tensor([0.25, 0.50]))
