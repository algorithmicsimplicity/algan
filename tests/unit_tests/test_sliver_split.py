"""The sliver leaf split (``raytracing.sliver_split``).

A long, thin triangle bounds a BVH box that is almost all empty space; the
refit tree gives it one leaf per strip across its long axis, each leaf a tight
box whose payload is the PARENT triangle. These tests pin what the renderer
relies on: the strips cover the parent (a hit anywhere on it lies in some
strip's box), the strips are genuinely short, unsplit triangles keep their one
column untouched, every flag follows its parent, the leaf payload the tree
carries is the parent's index, and the whole thing is inert for a scene with
no slivers.
"""

from __future__ import annotations

import torch

from algan.rendering.raytracing import sliver_split as S
from algan.rendering.raytracing.refit_bvh import (
    LINK_INVALID,
    LINK_PRIM_MASK,
    build_refit_bvh,
)
from algan.rendering.raytracing.stbvh import EMPTY_HI, EMPTY_LO
from algan.settings import SETTINGS


def _geometry():
    """Two triangles tiling a 3 x 0.01 quad (sharing its diagonal), one fat
    triangle and one short sliver, as ``[T=2, N=4, 9]`` moving corners.
    """
    q0 = torch.tensor([0.0, 0.0, 0.0])
    q1 = torch.tensor([3.0, 0.0, 0.0])
    q2 = torch.tensor([3.0, 0.01, 0.0])
    q3 = torch.tensor([0.0, 0.01, 0.0])
    tri = torch.stack(
        [
            torch.stack([q0, q1, q2]),
            torch.stack([q2, q3, q0]),
            torch.stack(
                [
                    torch.tensor([5.0, 0.0, 0.0]),
                    torch.tensor([6.0, 0.0, 0.0]),
                    torch.tensor([5.5, 1.0, 0.0]),
                ]
            ),
            torch.stack(
                [
                    torch.tensor([0.0, 2.0, 0.0]),
                    torch.tensor([0.03, 2.0, 0.0]),
                    torch.tensor([0.015, 2.0001, 0.0]),
                ]
            ),
        ]
    )
    corners = torch.stack([tri, tri + torch.tensor([0.0, 0.5, 0.0])])  # [2, 4, 3, 3]
    tri_pos = corners.reshape(2, 4, 9)
    lo = corners.amin(2)
    hi = corners.amax(2)
    opaque = torch.tensor([[True, True, False, True]])
    casts = torch.tensor([[True, False, True, True]])
    return tri_pos, lo, hi, opaque, casts


def _columns(**overrides):
    values = {
        "sliver_split_max_pieces": 64,
        "sliver_split_aspect": 4.0,
        "sliver_split_min_piece": 0.02,
    }
    values.update(overrides)
    tri_pos, lo, hi, opaque, casts = _geometry()
    with SETTINGS.raytracing.experimental.override(**values):
        return S.sliver_leaf_columns(tri_pos, lo, hi, opaque, casts), (
            tri_pos,
            lo,
            hi,
            opaque,
            casts,
        )


def test_slivers_get_one_column_per_strip_and_the_rest_pass_through():
    cols, (tri_pos, lo, hi, opaque, casts) = _columns()
    assert cols is not None
    lo_m, hi_m, opq_m, casts_m, leaf_prim = cols
    # Two slivers x 64 strips (the cap wins over the aspect rule).
    assert lo_m.shape == (2, 4 + 128, 3)
    assert leaf_prim.tolist() == [0, 1, 2, 3] + [0] * 64 + [1] * 64
    # The parents' own columns: cut ones emptied, the others bit-identical.
    assert bool((lo_m[:, :2] == EMPTY_LO).all() and (hi_m[:, :2] == EMPTY_HI).all())
    assert torch.equal(lo_m[:, 2:4], lo[:, 2:4])
    assert torch.equal(hi_m[:, 2:4], hi[:, 2:4])
    # Flags follow the parent.
    assert opq_m.tolist()[0] == [True, True, False, True] + [True] * 128
    assert casts_m.tolist()[0] == [True, False, True, True] + [True] * 64 + [False] * 64


def test_strips_cover_the_parent_and_are_short():
    cols, (tri_pos, lo, hi, _o, _c) = _columns()
    lo_m, hi_m, _opq, _casts, leaf_prim = cols
    for frame in range(2):
        for parent in (0, 1):
            k = (leaf_prim[4:] == parent).nonzero(as_tuple=True)[0] + 4
            s_lo, s_hi = lo_m[frame, k], hi_m[frame, k]
            # Union of the strips' boxes contains the parent's box (with the
            # ulp pad) and no strip box leaves it by more than that pad.
            assert bool((s_lo.amin(0) <= lo[frame, parent] + 1e-6).all())
            assert bool((s_hi.amax(0) >= hi[frame, parent] - 1e-6).all())
            assert bool((s_lo >= lo[frame, parent] - 1e-5).all())
            assert bool((s_hi <= hi[frame, parent] + 1e-5).all())
            # Every strip spans about L / 64 along x -- the wire's long axis.
            span = (s_hi - s_lo)[:, 0]
            assert float(span.max()) <= 3.0 / 64 * 1.05 + 1e-5
            # ...and consecutive strips overlap only at their shared cut.
            order = torch.argsort(s_lo[:, 0])
            gaps = s_lo[order][1:, 0] - s_hi[order][:-1, 0]
            assert float(gaps.max()) <= 0.0
            assert float(gaps.min()) >= -1e-5
            # Every corner of the parent lies in some strip's box.
            corners = tri_pos[frame, parent].view(3, 3)
            for v in corners:
                inside = ((s_lo <= v) & (v <= s_hi)).all(-1)
                assert bool(inside.any())


def test_the_cap_and_the_floor_bound_the_strip_count():
    cols, _g = _columns(sliver_split_max_pieces=16)
    assert cols[0].shape[1] == 4 + 32
    # A 0.5-unit floor allows floor(3 / 0.5) = 6 strips per sliver.
    cols, _g = _columns(sliver_split_min_piece=0.5)
    assert cols[0].shape[1] == 4 + 12


def test_disabled_or_sliver_free_geometry_is_inert():
    cols, _g = _columns(sliver_split_max_pieces=1)
    assert cols is None
    tri_pos, lo, hi, opaque, casts = _geometry()
    fat_only = slice(2, 3)
    with SETTINGS.raytracing.experimental.override(sliver_split_max_pieces=64):
        assert (
            S.sliver_leaf_columns(
                tri_pos[:, fat_only].contiguous(),
                lo[:, fat_only],
                hi[:, fat_only],
                opaque[:, fat_only],
                casts[:, fat_only],
            )
            is None
        )


def test_an_invisible_frame_leaves_every_strip_empty():
    tri_pos, lo, hi, opaque, casts = _geometry()
    lo = lo.clone()
    hi = hi.clone()
    lo[1, 0] = EMPTY_LO  # parent 0 invisible in frame 1
    hi[1, 0] = EMPTY_HI
    with SETTINGS.raytracing.experimental.override(sliver_split_max_pieces=8):
        lo_m, hi_m, _o, _c, leaf_prim = S.sliver_leaf_columns(
            tri_pos, lo, hi, opaque, casts
        )
    k = (leaf_prim == 0).nonzero(as_tuple=True)[0][
        1:
    ]  # the strips, not the parent column
    assert bool((lo_m[1, k] == EMPTY_LO).all() and (hi_m[1, k] == EMPTY_HI).all())
    assert bool((hi_m[0, k] >= lo_m[0, k]).all())


def _link_words(bvh):
    blocks = bvh.blocks
    if blocks.dtype == torch.float32:
        return blocks[:, 6].contiguous().view(torch.int32).flatten()
    halves = blocks.view(torch.int16).to(torch.int64)
    lo = halves[:, 6] & 0xFFFF
    hi = halves[:, 7] & 0xFFFF
    words = (lo | (hi << 16)).flatten()
    return torch.where(words >= 2**31, words - 2**32, words).to(torch.int32)


def test_the_tree_carries_the_parent_as_the_leaf_payload():
    cols, _g = _columns(sliver_split_max_pieces=8)
    lo_m, hi_m, opq_m, casts_m, leaf_prim = cols
    bvh = build_refit_bvh(
        lo_m, hi_m, num_frames=2, opaque=opq_m, casts=casts_m, leaf_prim=leaf_prim
    )
    words = _link_words(bvh)
    leaves = words[(words < 0) & (words != LINK_INVALID)]
    prims = (leaves & LINK_PRIM_MASK).unique().tolist()
    # Every leaf names one of the four real triangles, never a strip column.
    assert prims == [0, 1, 2, 3]
    # The two slivers are reachable through many leaves, the others through one
    # per frame.
    per_frame = leaves.numel() // 2
    assert per_frame == 8 + 8 + 1 + 1
