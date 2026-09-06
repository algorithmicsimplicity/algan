"""Sliver triangles as several BVH leaves (``sliver_split_*`` settings).

A triangle's axis-aligned box is what the BVH tests, and a sliver -- a wire,
an axis line, the side of a thin cylinder -- bounds a box that is almost all
empty space: one face of a 1.5 x 0.003 world-unit synapse tube claims a volume
hundreds of times its own, and every ray crossing that box tests the triangle.
A scene of such wires funnels most of its traversal work into them. Measured on
the nn benchmark (``benchmarks/performance/nn_scene_UHD.py``): 2,722 of 44k
triangles held 96% of the total box surface area, and tessellating them along
their length took the warm UHD render from 24.1 s to 13.7 s -- shadow trace
6.2 -> 3.0 s, wavefront shade 5.9 -> 2.5 s, traversal 2.8 -> 1.2 s.

Splitting the GEOMETRY is the wrong tool for that: the extra triangles carry
every per-triangle table (the material block six frames deep on this scene),
which on a 4 GB card cost the render its frame window and re-merged the scene
per batch, and the raster's per-sheet shading point moves when a wire's
fragments come from two pieces, which speckled the wire edges. So the split
lives in the acceleration structure alone. :func:`sliver_leaf_columns` cuts
each sliver's LONGEST-edge axis into strips and hands the BVH builder one leaf
column per strip -- the strip's own box, the parent's opacity and caster flags,
and the PARENT's primitive index as the leaf payload -- while the parent's own
column is emptied. A ray entering a strip's box tests the parent triangle
exactly as before, so primary visibility, shading, coverage and shadows are
bit-for-bit what they were; only the boxes a ray has to open changed.

A triangle reached through two of its strips' leaves reports the same hit
twice. The nearest-hit and shadow marches only accept a hit strictly nearer
than (or strictly after) what they hold, so the duplicate is ignored there for
free; the K-buffer gather (``_collect_hits``) checks a candidate triangle
against the slots it already holds, the one place a duplicate could otherwise
have composited a surface twice.

``n = ceil(L / (aspect * h))`` strips bring a strip's own aspect ratio down to
``sliver_split_aspect`` (``L`` the longest edge, ``h`` the triangle's height
above it, both averaged over the batch's frames), capped by
``sliver_split_max_pieces`` and by the ``sliver_split_min_piece`` floor on strip
length -- a short sliver is not a traversal problem, and cutting it only adds
leaves. A fat or short triangle keeps its one column, so a scene with no slivers
pays one reduction and no extra leaf.
"""

from __future__ import annotations

import torch

from algan.rendering.mps_compat import clamp_floor
from algan.rendering.raytracing.stbvh import EMPTY_HI, EMPTY_LO
from algan.settings import SETTINGS

rt_settings = SETTINGS.raytracing


def sliver_strip_plan(tri_pos):
    """Per-triangle strip plan from ``tri_pos`` ``[Tp, N, 9]``.

    Returns ``(strips, longest, n_ac)`` -- the strip count per triangle (1
    for anything not worth cutting), the corner index that starts its
    longest edge, and how many of the cut points on the other two edges fall
    on the edge from that corner to the apex -- or ``None`` when no triangle
    is cut.
    """
    max_pieces = int(rt_settings.sliver_split_max_pieces)
    if (
        max_pieces <= 1
        or tri_pos is None
        or tri_pos.dim() != 3
        or tri_pos.shape[1] == 0
    ):
        return None
    aspect = max(float(rt_settings.sliver_split_aspect), 1e-6)
    min_piece = max(float(rt_settings.sliver_split_min_piece), 0.0)
    c = tri_pos.float().view(tri_pos.shape[0], tri_pos.shape[1], 3, 3)
    v0, v1, v2 = c[:, :, 0], c[:, :, 1], c[:, :, 2]
    # Edge k runs from corner k to corner (k + 1) % 3; averaged over frames.
    edge_len = torch.stack(
        ((v1 - v0).norm(dim=-1), (v2 - v1).norm(dim=-1), (v0 - v2).norm(dim=-1)), -1
    ).mean(0)
    length, longest = edge_len.max(-1)
    area2 = torch.linalg.cross(v1 - v0, v2 - v0, dim=-1).norm(dim=-1).mean(0)
    height = area2 / clamp_floor(length, 1e-30)
    strips = torch.ceil(length / clamp_floor(aspect * height, 1e-30))
    if min_piece > 0.0:
        strips = torch.minimum(strips, torch.floor(length / min_piece))
    strips = strips.clamp(1, max_pieces)
    strips = torch.where(
        (height > 1e-12) & (length > 1e-12), strips, torch.ones_like(strips)
    ).long()
    if not bool((strips > 1).any()):
        return None
    # Where the apex projects onto the longest edge decides how the cut points
    # on the other two edges are shared out between them, so every cut runs
    # roughly across the triangle rather than fanning from a corner.
    cm = c.mean(0)
    rows = torch.arange(cm.shape[0], device=cm.device)
    pa = cm[rows, longest]
    pb = cm[rows, (longest + 1) % 3]
    pc = cm[rows, (longest + 2) % 3]
    ab = pb - pa
    proj = ((pc - pa) * ab).sum(-1) / clamp_floor((ab * ab).sum(-1), 1e-30)
    n_ac = torch.minimum(
        torch.round(proj.clamp(0.0, 1.0) * strips.to(torch.float32))
        .long()
        .clamp_min(0),
        strips,
    )
    return strips, longest, n_ac


def sliver_leaf_columns(tri_pos, lo, hi, opaque, casts):
    """Leaf columns for a triangle BVH build with slivers cut into strips.

    ``tri_pos`` is the merged ``[Tp, N, 9]`` corner table; ``lo`` / ``hi``
    ``[Tc, N, 3]`` the per-frame boxes the build would otherwise take (with
    invisible frames marked empty), ``opaque`` ``[To, N]`` and ``casts``
    ``[1, N]`` its flags. Returns ``(lo, hi, opaque, casts, leaf_prim)`` over
    ``N + K`` columns -- the ``N`` parents (a cut parent's box emptied) plus one
    column per strip, ``leaf_prim`` naming each column's primitive -- or
    ``None`` when nothing is cut, so the caller builds exactly as before.

    A strip's box is the box of its quad (the two cut points on the longest
    edge and the two matching points on the other edges), per frame, padded by
    a few ulps: neighbouring strips share their cut points exactly, so the
    strips cover the parent without a gap, and the pad keeps a hit that lands
    on a cut line inside at least one box under float rounding. A frame in
    which the parent is invisible leaves every strip empty in that frame too.
    """
    plan = sliver_strip_plan(tri_pos)
    if plan is None:
        return None
    strips, longest, n_ac = plan
    device = tri_pos.device
    N = tri_pos.shape[1]
    split = strips > 1
    sidx = split.nonzero(as_tuple=True)[0]
    n = strips.index_select(0, sidx)
    K = int(n.sum())
    owner = torch.repeat_interleave(torch.arange(sidx.numel(), device=device), n)
    first = torch.cumsum(n, 0) - n
    k = torch.arange(K, device=device) - first.index_select(0, owner)
    n_o = n.index_select(0, owner).to(torch.float32)
    nac_o = n_ac.index_select(0, sidx).index_select(0, owner)
    ncb_o = (n.index_select(0, owner) - nac_o).clamp_min(1).to(torch.float32)
    nac_f = nac_o.clamp_min(1).to(torch.float32)
    parent = sidx.index_select(0, owner)  # [K]

    c = tri_pos.float().view(tri_pos.shape[0], N, 3, 3)
    Tp = c.shape[0]
    ia = longest.index_select(0, parent)
    rows = torch.arange(K, device=device)
    A = c[:, parent][:, rows, ia]  # [Tp, K, 3]
    B = c[:, parent][:, rows, (ia + 1) % 3]
    C = c[:, parent][:, rows, (ia + 2) % 3]

    def p_point(j):
        t = (j.to(torch.float32) / n_o).view(1, -1, 1)
        return A + (B - A) * t

    def q_point(j):
        on_ac = (j <= nac_o) & (nac_o > 0)
        t_ac = (j.to(torch.float32) / nac_f).clamp(0.0, 1.0).view(1, -1, 1)
        t_cb = ((j - nac_o).to(torch.float32) / ncb_o).clamp(0.0, 1.0).view(1, -1, 1)
        return torch.where(on_ac.view(1, -1, 1), A + (C - A) * t_ac, C + (B - C) * t_cb)

    pts = torch.stack(
        (p_point(k), p_point(k + 1), q_point(k), q_point(k + 1)), 2
    )  # [Tp, K, 4, 3]
    s_lo = pts.amin(2)
    s_hi = pts.amax(2)
    pad = 4e-7 * (pts.abs().amax(2) + 1.0)
    s_lo = s_lo - pad
    s_hi = s_hi + pad

    # Frame rows: the strips exist in every frame their parent is visible in.
    Tc = lo.shape[0]
    T = max(Tc, Tp)
    tp_rows = torch.arange(T, device=device) % Tp
    tc_rows = torch.arange(T, device=device) % Tc
    s_lo = s_lo.index_select(0, tp_rows)
    s_hi = s_hi.index_select(0, tp_rows)
    valid = (hi >= lo).all(-1).index_select(0, tc_rows)  # [T, N]
    valid_k = valid.index_select(1, parent).unsqueeze(-1)  # [T, K, 1]
    s_lo = torch.where(valid_k, s_lo, torch.full_like(s_lo, EMPTY_LO))
    s_hi = torch.where(valid_k, s_hi, torch.full_like(s_hi, EMPTY_HI))

    lo_p = lo.index_select(0, tc_rows).clone()
    hi_p = hi.index_select(0, tc_rows).clone()
    lo_p[:, sidx] = EMPTY_LO
    hi_p[:, sidx] = EMPTY_HI
    lo_m = torch.cat((lo_p, s_lo), 1).contiguous()
    hi_m = torch.cat((hi_p, s_hi), 1).contiguous()
    opaque_m = torch.cat((opaque, opaque.index_select(1, parent)), 1).contiguous()
    casts_m = None
    if casts is not None:
        casts_m = torch.cat((casts, casts.index_select(1, parent)), 1).contiguous()
    leaf_prim = torch.cat((torch.arange(N, device=device), parent))
    return lo_m, hi_m, opaque_m, casts_m, leaf_prim
