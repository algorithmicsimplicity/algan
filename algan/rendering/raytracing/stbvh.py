"""Spatio-Temporal Bounding Volume Hierarchy (STBVH).

Algan renders animations in batches of hundreds of frames at once. A naive ray
tracer would build one BVH per frame, costing O(num_frames * num_primitives)
nodes even though most geometry barely moves between consecutive frames. The
STBVH instead treats time as a fourth dimension and builds a single tree over
*primitive instances*, where an instance is a primitive together with a frame
interval ``[t0, t1]`` and the union of its spatial bounds over that interval.

Memory footprint is optimized in two ways:

1. **Adaptive temporal segmentation** -- each primitive's timeline is split
   into dyadic intervals only where its union bound becomes loose. A static
   primitive contributes a single instance covering the whole batch; a fast
   moving one degrades gracefully towards one instance per frame (never worse
   than a per-frame BVH). Frames in which a primitive is invisible contribute
   nothing at all.
2. **Pointer-free layout** -- instances are sorted along a 4D (x, y, z, t)
   Morton curve so that nodes are coherent in space *and* time, then packed
   into an implicit complete binary tree (heap order). Children are found by
   index arithmetic and traversal is stackless via precomputed "miss" links,
   so a node costs just 8 floats + 3 ints.

Intersection tests remain exact: a leaf stores only the primitive index and
its frame interval; the traversal kernel fetches the primitive's geometry at
the ray's exact frame.

Everything in this module is implemented with vectorized PyTorch ops (the
per-ray traversal lives in ``ray_trace_taichi.py``).
"""
from __future__ import annotations

import torch

# Bounds used to mark an empty/invisible AABB. Unions and costs are computed
# with clamping so that empty boxes behave as the identity element. The
# magnitude is kept small enough that the float32 slab test in the traversal
# kernel ((bound - origin) * inv_dir, with inv_dir clamped to 1e12) stays
# finite.
EMPTY_LO = 1e17
EMPTY_HI = -1e17

_QUANT_BITS = 15  # 4 * 15 = 60 bits used, keeping codes positive in int64.


class STBVH:
    """Flat tensor representation of the spatio-temporal BVH.

    All node arrays are in heap order over a complete binary tree with
    ``num_leaves`` (a power of two) leaves: the root is node 0, the children
    of node ``i`` are ``2i + 1`` and ``2i + 2``, and the leaves occupy nodes
    ``[num_leaves - 1, 2 * num_leaves - 1)``.

    Attributes
    ----------
    node_lo, node_hi : Tensor[num_nodes, 3] (float32)
        Spatial bounds of each node (union over the node's frame interval).
    node_tmin, node_tmax : Tensor[num_nodes] (int32)
        Inclusive frame-interval bounds of each node. A ray belonging to
        frame ``f`` may only enter nodes with ``tmin <= f <= tmax``.
    node_miss : Tensor[num_nodes] (int32)
        Stackless traversal links: the next node in depth-first order when a
        node is skipped or a leaf has been processed (-1 terminates).
    leaf_prim : Tensor[num_leaves] (int32)
        Primitive index for each leaf, -1 for padding leaves.
    """

    def __init__(self, node_lo, node_hi, node_tmin, node_tmax, node_miss, leaf_prim):
        self.node_lo = node_lo
        self.node_hi = node_hi
        self.node_tmin = node_tmin
        self.node_tmax = node_tmax
        self.node_miss = node_miss
        self.leaf_prim = leaf_prim
        self.num_leaves = leaf_prim.shape[0]
        self.first_leaf = self.num_leaves - 1

    @property
    def num_nodes(self):
        return self.node_lo.shape[0]

    def get_memory_used(self):
        return sum(
            t.numel() * t.element_size()
            for t in (self.node_lo, self.node_hi, self.node_tmin, self.node_tmax,
                      self.node_miss, self.leaf_prim)
        )


def _spread_bits_4(x):
    """Spread the low 16 bits of int64 ``x`` so bit i moves to bit 4*i."""
    x = x & 0xFFFF
    x = (x | (x << 24)) & 0x000000FF000000FF
    x = (x | (x << 12)) & 0x000F000F000F000F
    x = (x | (x << 6)) & 0x0303030303030303
    x = (x | (x << 3)) & 0x1111111111111111
    return x


def morton_code_4d(x, y, z, t):
    """Interleave four int64 coordinate tensors (each < 2**16) into one code."""
    return (
        _spread_bits_4(x)
        | (_spread_bits_4(y) << 1)
        | (_spread_bits_4(z) << 2)
        | (_spread_bits_4(t) << 3)
    )


def _quantize(c, lo, hi):
    scale = (2 ** _QUANT_BITS - 1) / (hi - lo).clamp_min(1e-12)
    q = ((c - lo) * scale).long()
    return q.clamp_(min=0, max=2 ** _QUANT_BITS - 1)


def _box_cost(lo, hi):
    """Half-perimeter cost of boxes; empty boxes cost 0."""
    return (hi - lo).clamp_min(0).sum(-1)


def segment_primitives_in_time(frame_lo, frame_hi, tightness=2.0):
    """Adaptively partition each primitive's timeline into tight instances.

    Parameters
    ----------
    frame_lo, frame_hi : Tensor[T, N, 3]
        Per-frame spatial AABB of each primitive. Frames where a primitive is
        absent/invisible should be marked empty (``lo=EMPTY_LO, hi=EMPTY_HI``);
        they are excluded from the output instances entirely.
    tightness : float
        An interval is kept whole while the half-perimeter of its union box is
        at most ``tightness`` times the mean per-frame half-perimeter within
        it. Lower values produce more, tighter instances.

    Returns
    -------
    (prim_id [M], t0 [M], t1 [M], inst_lo [M, 3], inst_hi [M, 3])
        ``t0``/``t1`` are inclusive frame bounds (int64).
    """
    T, N, _ = frame_lo.shape
    device = frame_lo.device

    # Segmentation is independent per primitive; chunk wide inputs so the
    # interval pyramid's peak memory stays bounded.
    chunk = max(1, int(4e6) // max(T, 1))
    if chunk < N:
        parts = [
            segment_primitives_in_time(frame_lo[:, s:s + chunk],
                                       frame_hi[:, s:s + chunk], tightness)
            for s in range(0, N, chunk)
        ]
        prim_id = torch.cat([p[0] + s for p, s in
                             zip(parts, range(0, N, chunk))])
        t0 = torch.cat([p[1] for p in parts])
        t1 = torch.cat([p[2] for p in parts])
        inst_lo = torch.cat([p[3] for p in parts], 0)
        inst_hi = torch.cat([p[4] for p in parts], 0)
        return prim_id, t0, t1, inst_lo, inst_hi

    # Pad the time axis to a power of two with empty boxes so every dyadic
    # level halves evenly.
    Tp = 1 << max(T - 1, 0).bit_length() if T > 1 else 1
    if Tp != T:
        pad_lo = torch.full((Tp - T, N, 3), EMPTY_LO, device=device)
        pad_hi = torch.full((Tp - T, N, 3), EMPTY_HI, device=device)
        frame_lo = torch.cat((frame_lo, pad_lo), 0)
        frame_hi = torch.cat((frame_hi, pad_hi), 0)

    valid = (frame_hi >= frame_lo).all(-1)
    cost = _box_cost(frame_lo, frame_hi) * valid

    # Bottom-up pyramid over dyadic time intervals. levels[k] covers
    # intervals of length 2**k.
    levels = [(frame_lo, frame_hi, cost, valid.float())]
    while levels[-1][0].shape[0] > 1:
        lo, hi, c, v = levels[-1]
        lo2 = torch.minimum(lo[0::2], lo[1::2])
        hi2 = torch.maximum(hi[0::2], hi[1::2])
        levels.append((lo2, hi2, c[0::2] + c[1::2], v[0::2] + v[1::2]))

    # Top-down sweep: emit an instance where the union is tight, otherwise
    # descend into the two half-intervals.
    out_prim, out_t0, out_t1, out_lo, out_hi = [], [], [], [], []
    active = torch.ones((1, N), dtype=torch.bool, device=device)
    for k in range(len(levels) - 1, -1, -1):
        lo, hi, c, v = levels[k]
        nonempty = v > 0
        union_cost = _box_cost(lo, hi)
        mean_cost = c / v.clamp_min(1)
        tight = union_cost <= tightness * mean_cost + 1e-12
        if k == 0:
            tight = torch.ones_like(tight)
        emit = active & nonempty & tight
        if emit.any():
            t_idx, n_idx = emit.nonzero(as_tuple=True)
            out_prim.append(n_idx)
            out_t0.append(t_idx << k)
            out_t1.append(torch.clamp(((t_idx + 1) << k) - 1, max=T - 1))
            out_lo.append(lo[t_idx, n_idx])
            out_hi.append(hi[t_idx, n_idx])
        if k > 0:
            active = (active & nonempty & ~tight).repeat_interleave(2, dim=0)

    if len(out_prim) == 0:
        empty_l = torch.empty((0,), dtype=torch.long, device=device)
        empty_f = torch.empty((0, 3), device=device)
        return empty_l, empty_l, empty_l.clone(), empty_f, empty_f.clone()
    return (
        torch.cat(out_prim),
        torch.cat(out_t0),
        torch.cat(out_t1),
        torch.cat(out_lo, 0),
        torch.cat(out_hi, 0),
    )


def _compute_miss_links(num_leaves, device):
    """Miss links for stackless DFS over the implicit complete binary tree."""
    num_nodes = 2 * num_leaves - 1
    miss = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
    start, width = 1, 2
    while start < num_nodes:
        idx = torch.arange(start, start + width, device=device)
        left = (idx % 2) == 1
        parent_miss = miss[(idx - 1) >> 1]
        miss[idx] = torch.where(left, idx + 1, parent_miss)
        start += width
        width *= 2
    return miss


def build_stbvh(frame_lo, frame_hi, num_frames=None, tightness=2.0):
    """Build a spatio-temporal BVH from per-frame primitive bounds.

    Parameters
    ----------
    frame_lo, frame_hi : Tensor[Tc, N, 3]
        Per-frame spatial AABBs. ``Tc`` may be 1 for static geometry, in which
        case every primitive becomes a single instance spanning
        ``[0, num_frames - 1]``.
    num_frames : int
        Number of frames in the render batch (defaults to ``Tc``).
    tightness : float
        See :func:`segment_primitives_in_time`.
    """
    Tc, N, _ = frame_lo.shape
    device = frame_lo.device
    if num_frames is None:
        num_frames = Tc

    if Tc == 1:
        valid = (frame_hi[0] >= frame_lo[0]).all(-1)
        prim_id = valid.nonzero(as_tuple=True)[0]
        t0 = torch.zeros_like(prim_id)
        t1 = torch.full_like(prim_id, num_frames - 1)
        inst_lo = frame_lo[0, prim_id]
        inst_hi = frame_hi[0, prim_id]
    else:
        if Tc != num_frames:
            raise ValueError(
                f"frame bounds have {Tc} frames but the batch has {num_frames}"
            )
        prim_id, t0, t1, inst_lo, inst_hi = segment_primitives_in_time(
            frame_lo, frame_hi, tightness
        )

    M = prim_id.shape[0]
    if M > 0:
        # Sort instances along a 4D Morton curve so the implicit tree gets
        # spatio-temporally coherent subtrees.
        center = (inst_lo + inst_hi) * 0.5
        t_center = (t0 + t1).float() * 0.5
        smin = inst_lo.amin(0)
        smax = inst_hi.amax(0)
        q = _quantize(center, smin, smax)
        qt = _quantize(t_center, torch.zeros((), device=device),
                       torch.full((), float(max(num_frames - 1, 1)), device=device))
        codes = morton_code_4d(q[:, 0], q[:, 1], q[:, 2], qt)
        order = torch.argsort(codes)
        prim_id, t0, t1 = prim_id[order], t0[order], t1[order]
        inst_lo, inst_hi = inst_lo[order], inst_hi[order]

    P = 1 << max(M - 1, 0).bit_length() if M > 1 else 1
    num_nodes = 2 * P - 1

    node_lo = torch.full((num_nodes, 3), EMPTY_LO, device=device)
    node_hi = torch.full((num_nodes, 3), EMPTY_HI, device=device)
    node_tmin = torch.full((num_nodes,), 1 << 30, dtype=torch.long, device=device)
    node_tmax = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
    leaf_prim = torch.full((P,), -1, dtype=torch.long, device=device)

    first_leaf = P - 1
    if M > 0:
        node_lo[first_leaf:first_leaf + M] = inst_lo
        node_hi[first_leaf:first_leaf + M] = inst_hi
        node_tmin[first_leaf:first_leaf + M] = t0
        node_tmax[first_leaf:first_leaf + M] = t1
        leaf_prim[:M] = prim_id

    # Bottom-up union of children into parents, one vectorized level at a time.
    offset, width = first_leaf, P
    while width > 1:
        child_lo = node_lo[offset:offset + width]
        child_hi = node_hi[offset:offset + width]
        child_t0 = node_tmin[offset:offset + width]
        child_t1 = node_tmax[offset:offset + width]
        parent_offset = offset - width // 2
        node_lo[parent_offset:offset] = torch.minimum(child_lo[0::2], child_lo[1::2])
        node_hi[parent_offset:offset] = torch.maximum(child_hi[0::2], child_hi[1::2])
        node_tmin[parent_offset:offset] = torch.minimum(child_t0[0::2], child_t0[1::2])
        node_tmax[parent_offset:offset] = torch.maximum(child_t1[0::2], child_t1[1::2])
        offset, width = parent_offset, width // 2

    miss = _compute_miss_links(P, device)

    return STBVH(
        node_lo.contiguous(),
        node_hi.contiguous(),
        node_tmin.to(torch.int32).contiguous(),
        node_tmax.to(torch.int32).contiguous(),
        miss.to(torch.int32).contiguous(),
        leaf_prim.to(torch.int32).contiguous(),
    )
