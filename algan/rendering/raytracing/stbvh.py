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
   Morton curve so that nodes are coherent in space *and* time, then grouped
   ``LEAF_SIZE`` at a time into the leaves of an implicit complete
   ``BVH_ARITY``-ary tree (heap order), so children are found by index
   arithmetic. The kernels traverse *sibling blocks*: each internal node
   stores the bounds + frame intervals of its ``BVH_ARITY`` children
   contiguously (``blocks``, one aligned 128-byte fetch -- or 64 bytes
   f16-compressed -- per visit), so one dependent memory round tests a whole
   sibling group instead of one box, and the walk needs no per-node miss
   links (a small packed node/mask stack in the kernel replaces them).

Intersection tests remain exact: a leaf slot stores only the primitive index
and its frame interval (packed into one int32); the traversal kernel fetches
the primitive's geometry at the ray's exact frame and skips slots whose
instance does not cover that frame.

Everything in this module is implemented with vectorized PyTorch ops (the
per-ray traversal lives in the Taichi kernel modules --
``raytrace_kernels_taichi.py`` holds the shared block-walk funcs, consumed by
the wavefront and Monte Carlo kernels alike).
"""
from __future__ import annotations

import os

import torch

# Bounds used to mark an empty/invisible AABB. Unions and costs are computed
# with clamping so that empty boxes behave as the identity element. The
# magnitude is kept small enough that the float32 slab test in the traversal
# kernel ((bound - origin) * inv_dir, with inv_dir clamped to 1e12) stays
# finite.
EMPTY_LO = 1e17
EMPTY_HI = -1e17

_QUANT_BITS = 15  # 4 * 15 = 60 bits used, keeping codes positive in int64.

# Instances per leaf. Grouping shrinks the tree (depth and node memory both
# divide by LEAF_SIZE) at the cost of testing up to LEAF_SIZE primitives per
# leaf visit. Measured on animated scenes, grouping is counterproductive:
# the 4D Morton order places the *same* primitive at adjacent frames next to
# each other, so grouped leaves get time-swept union boxes whose slots are
# then mostly rejected by their frame intervals. Default to one instance per
# leaf; the env knob remains for experiments. Read once at import (the
# traversal kernels specialize on it).
LEAF_SIZE = max(1, int(os.environ.get("ALGAN_STBVH_LEAF_SIZE", "1")))

# Branching factor of the implicit tree. A wider tree is shallower -- depth
# divides by log2(BVH_ARITY) -- which shortens the serial chain of dependent
# node reads that dominates traversal latency, without changing which
# primitives a leaf holds (so renders are byte-for-byte identical to a binary
# tree). 4 (BVH4) is the measured sweet spot: depth halves vs binary for a
# ~10-16% trace-kernel speedup, while 8 over-widens (8 sibling-box tests per
# level outweigh the further depth cut). 2 reproduces the original binary
# layout. Read once at import; the traversal kernels specialize on it. Must
# be >= 2.
BVH_ARITY = max(2, int(os.environ.get("ALGAN_BVH_ARITY", "4")))

# Relative weight of the (normalized) time axis in the median-split builder's
# widest-axis choice. > 1 makes time splits happen higher in the tree, so
# subtrees become frame-pure sooner and the traversal's frame gate rejects
# them wholesale for rays in other frames. Purely a build-quality knob: the
# traversal is arrangement-invariant, so renders are byte-identical.
SPLIT_TIME_WEIGHT = float(os.environ.get("ALGAN_SPLIT_TIME_WEIGHT", "1"))

# Store the sibling-block child bounds as conservatively rounded float16
# (64-byte blocks) instead of exact float32 (128-byte blocks). Lower bounds
# round toward -inf and upper bounds toward +inf, so the f16 boxes strictly
# contain the exact ones and can never falsely cull a hit. NOT byte-identical
# though: hits routinely lie exactly on their (exact) box faces, and the
# looser boxes admit candidates within a float ulp of the traversal's
# DEPTH_TIE_EPSILON window boundaries that the exact boxes cull -- measured
# as epsilon-level image changes (few % of pixels by a few LSB), the same
# class of deviation as changing ``tightness``. Default on;
# ALGAN_BVH_BLOCK_F16=0 opts out (exact f32 blocks). Read once at import by
# both this module (build) and the traversal kernels (block decode + ndarray
# element type).
BLOCK_F16 = os.environ.get("ALGAN_BVH_BLOCK_F16", "1") == "1"

# Smallest normal float16. Conservative rounding pushes would-be-subnormal
# magnitudes outward to 0 or +-this, so a flush-to-zero f16->f32 conversion
# in the kernel could never shrink a box.
_F16_MIN_NORMAL = 6.103515625e-05
_F16_MAX = 65504.0


def _half_bits_directed(x, up):
    """float16 bit patterns (int16) of ``x`` rounded toward +inf (``up``) or
    -inf, with subnormal results pushed outward to 0 / +-min-normal. The
    decoded f16 is guaranteed ``>= x`` (``up``) or ``<= x`` (down).
    """
    x = x.float().clamp(-_F16_MAX, _F16_MAX)
    h = x.half()
    dec = h.float()
    # Map bit patterns to a monotone integer line so +-1 is nextafter.
    b = (h.view(torch.int16).to(torch.int32)) & 0xFFFF
    m = torch.where(b < 0x8000, b + 0x8000, 0xFFFF - b)
    wrong = (dec < x) if up else (dec > x)
    m = torch.where(wrong, m + (1 if up else -1), m)
    b2 = torch.where(m >= 0x8000, m - 0x8000, 0xFFFF - m)
    h2 = (b2 - ((b2 & 0x8000) << 1)).to(torch.int16).view(torch.float16)
    # Outward-flush subnormals (see _F16_MIN_NORMAL note).
    v = h2.float()
    sub = (v != 0) & (v.abs() < _F16_MIN_NORMAL)
    if up:
        v2 = torch.where(v > 0, torch.full_like(v, _F16_MIN_NORMAL),
                         torch.zeros_like(v))
    else:
        v2 = torch.where(v < 0, torch.full_like(v, -_F16_MIN_NORMAL),
                         torch.zeros_like(v))
    h2 = torch.where(sub, v2.half(), h2)
    return h2.view(torch.int16)


def _build_blocks(nodes, first_leaf):
    """Fuse each internal node's children into one kernel-facing sibling
    block.

    ``nodes`` is the builders' unpacked ``[num_nodes, 8]`` float32 rows
    ``(lo.xyz, hi.xyz, tmin, tmax)`` in heap order, so the children of the
    ``first_leaf`` internal nodes are exactly rows ``1 .. ARITY*first_leaf``
    in order. Returns ``[first_leaf, 8, BVH_ARITY]`` -- SoA across the
    sibling group: lanes 0-5 hold the children's ``lo.x/lo.y/lo.z/hi.x/hi.y/
    hi.z``, lanes 6-7 their packed frame interval ``tmin | (tmax << 16)``.
    float32 blocks bit-cast the int32 tspan into lane 6 (lane 7 pads the
    block to an aligned 128 bytes); float16 blocks (``BLOCK_F16``) store
    conservatively rounded bounds plus the tspan's low/high u16 halves as
    lanes 6/7 (64 bytes).
    """
    a = BVH_ARITY
    device = nodes.device
    child = nodes[1:1 + a * first_leaf].view(first_leaf, a, 8)
    t0 = child[..., 6].to(torch.int32).clamp(0, (1 << 15) - 1)
    t1 = child[..., 7].to(torch.int32).clamp(0, (1 << 15) - 1)
    tspan = (t0 | (t1 << 16)).contiguous()
    if BLOCK_F16:
        blk = torch.zeros((first_leaf, 8, a), dtype=torch.int16,
                          device=device)
        for d in range(3):
            blk[:, d] = _half_bits_directed(child[..., d], up=False)
            blk[:, 3 + d] = _half_bits_directed(child[..., 3 + d], up=True)
        halves = tspan.view(torch.int16).view(first_leaf, a, 2)
        blk[:, 6] = halves[..., 0]
        blk[:, 7] = halves[..., 1]
        return blk.view(torch.float16).contiguous()
    blk = torch.zeros((first_leaf, 8, a), dtype=torch.float32, device=device)
    for d in range(6):
        blk[:, d] = child[..., d]
    blk[:, 6] = tspan.view(torch.float32)
    return blk.contiguous()


class STBVH:
    """Flat tensor representation of the spatio-temporal BVH.

    Node data is in heap order over a complete ``BVH_ARITY``-ary tree with
    ``num_leaves`` (a power of ``BVH_ARITY``) leaves: the root is node 0, the
    children of node ``i`` are ``BVH_ARITY*i + 1 .. BVH_ARITY*i + BVH_ARITY``,
    and the ``num_leaves`` leaves occupy the last nodes (from ``first_leaf``).
    Each leaf holds ``LEAF_SIZE`` instance slots.

    Attributes
    ----------
    nodes : Tensor[num_nodes, 8] (float32)
        Unpacked per-node data: spatial bounds ``lo.xyz, hi.xyz`` (union over
        the node's frame interval) and the inclusive frame interval
        ``tmin, tmax`` stored as floats (exact for < 2**24 frames). Host-side
        source of truth (block construction, debugging); the traversal
        kernels read ``blocks``.
    blocks : Tensor[first_leaf, 8, BVH_ARITY] (float32 or float16)
        Kernel-facing sibling blocks: the bounds + packed frame interval of
        internal node ``i``'s children, SoA across the group (see
        :func:`_build_blocks`). One aligned fetch per node visit tests the
        whole sibling group; a ray belonging to frame ``f`` may only enter
        children whose interval satisfies ``tmin <= f <= tmax``. float16
        blocks store conservatively out-rounded bounds (``BLOCK_F16``, the
        default, epsilon-level non-identical to f32 -- see its comment).
    node_miss : Tensor[num_nodes] (int32)
        Stackless DFS miss links (next node when a node is skipped or a leaf
        has been processed, -1 terminates). Host-side/debug only: the block
        walk keeps a small in-kernel stack instead of following miss links.
    leaf_prim : Tensor[num_leaves * LEAF_SIZE] (int32)
        Primitive index for each leaf slot, -1 for padding slots. Slot ``j``
        of leaf ``l`` is at index ``l * LEAF_SIZE + j``.
    leaf_tspan : Tensor[num_leaves * LEAF_SIZE] (int32)
        Frame interval of each slot's instance, packed as
        ``tmin | (tmax << 16)`` (requires frame batches < 2**15 frames,
        enforced by the renderer's chunking). Bit 31 (the sign bit) flags
        instances that are fully *opaque* over their interval: the renderer
        can prune everything behind such a hit while gathering.
    """

    def __init__(self, nodes, node_miss, leaf_prim, leaf_tspan):
        self.nodes = nodes
        self.node_miss = node_miss
        self.leaf_prim = leaf_prim
        self.leaf_tspan = leaf_tspan
        # Implicit complete BVH_ARITY-ary tree with P leaves has
        # num_nodes = (A*P - 1)/(A - 1) nodes, the P leaves last in heap order.
        num_nodes = nodes.shape[0]
        self.num_leaves = (num_nodes * (BVH_ARITY - 1) + 1) // BVH_ARITY
        self.first_leaf = num_nodes - self.num_leaves
        self.blocks = _build_blocks(nodes, self.first_leaf)

    @classmethod
    def from_prebuilt(cls, nodes, node_miss, leaf_prim, leaf_tspan, blocks,
                      like=None):
        """Construct an STBVH whose kernel-facing blocks are already built.
        ``like`` (the source object) is accepted for interface parity with
        :meth:`refit_bvh.RefitBVH.from_prebuilt` and ignored -- this class
        derives its layout fields from the tensor shapes.

        Scene preparation can build a BVH on the CPU and later upload its
        finished tensors into the render arena.  Calling :class:`STBVH`
        normally at that boundary would run :func:`_build_blocks` again on the
        rendering device, creating both redundant work and destination-side
        temporary allocations.  This constructor only attaches the supplied
        tensors and derives the two scalar layout fields from their shapes.
        """
        self = cls.__new__(cls)
        self.nodes = nodes
        self.node_miss = node_miss
        self.leaf_prim = leaf_prim
        self.leaf_tspan = leaf_tspan
        num_nodes = nodes.shape[0]
        self.num_leaves = (num_nodes * (BVH_ARITY - 1) + 1) // BVH_ARITY
        self.first_leaf = num_nodes - self.num_leaves
        self.blocks = blocks
        return self

    @property
    def num_nodes(self):
        return self.nodes.shape[0]

    def get_memory_used(self):
        return sum(
            t.numel() * t.element_size()
            for t in (self.nodes, self.blocks, self.node_miss, self.leaf_prim,
                      self.leaf_tspan)
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


# Instance-ordering builder: "morton" (4D-Morton sort), "split" (recursive
# longest-axis median split) or "sah" (experimental explicit-DFS SAH tree). A
# space-filling curve is cheap but packs spatially-distant instances into the
# same balanced subtree at its discontinuities, leaving loose internal node
# boxes; a top-down median split bisects each node along its longest
# (normalized) axis, giving tighter boxes and ~20-25% fewer traversal steps.
# Both are pure reorderings -- same instances, same opaque flags, same tree
# shape -- so the traversal code is untouched and the set of intersections
# found is unchanged.
#
# The default is per geometry type (chosen by the caller of build_stbvh):
# "split" for triangles, whose depth-peel is provably arrangement-invariant
# (verified byte-identical), "morton" for PN patches / bezier circuits, whose
# seam de-duplication is discovery-order sensitive (split changes output at
# the epsilon level there -- faster, but kept off to preserve baselines).
# Setting ALGAN_BVH_BUILD forces one builder for every type (A/B escape
# hatch).
_BVH_BUILD = os.environ.get("ALGAN_BVH_BUILD")


def _median_split_slots(centers, P):
    """Assign ``M`` instances to the ``P`` leaf slots of the implicit balanced
    tree by recursive longest-axis median bisection.

    ``centers`` is ``[M, 4]`` (spatial centre xyz + temporal centre), assumed
    already normalized so the four axes are comparable. Returns ``slot_src``
    ``[P]`` mapping each leaf slot to its instance index, or ``-1`` for the
    ``P - M`` padding slots (which sink to the end of every subtree). Each
    bisection sorts a node's instances along its widest axis and cuts at the
    slot-capacity midpoint, so subtrees map onto contiguous heap-ordered slots.
    """
    M = centers.shape[0]
    device = centers.device
    BIG = 1e30
    cpad = torch.full((P, 4), BIG, device=device)
    cpad[:M] = centers
    valid = torch.arange(P, device=device) < M
    slot = torch.arange(P, device=device)
    n_levels = P.bit_length() - 1  # P is a power of the (power-of-two) arity
    for level in range(n_levels):
        ng = 1 << level
        gs = P >> level
        sg = slot.view(ng, gs)
        cg = cpad[sg]                       # [ng, gs, 4]
        vg = valid[sg].unsqueeze(-1)        # [ng, gs, 1]
        lo = torch.where(vg, cg, torch.full_like(cg, BIG)).amin(1)
        hi = torch.where(vg, cg, torch.full_like(cg, -BIG)).amax(1)
        axis = (hi - lo).argmax(1)          # [ng] widest axis per group
        key = torch.gather(
            cg, 2, axis.view(ng, 1, 1).expand(ng, gs, 1)).squeeze(2)
        # Padding slots carry key == BIG, so they sort to the end of each group.
        perm = key.argsort(1)
        slot = torch.gather(sg, 1, perm).reshape(-1)
    return torch.where(slot < M, slot, torch.full_like(slot, -1))


def _build_sah_dfs(inst_lo, inst_hi, t0, t1, prim_id, inst_opaque, num_frames,
                   device, nbins=16):
    """Top-down binned-SAH BVH (binary, unbalanced), laid out in DFS preorder
    with stackless skip pointers -- the explicit-tree counterpart of the implicit
    balanced layout. Returns the same flat arrays the traversal consumes, but
    addressed per-node: ``leaf_prim[node] >= 0`` marks a leaf (one instance), an
    internal node's first child is ``node + 1``, and the node's miss link (the
    subtree-escape "skip" index, -1 to terminate) is packed into its row by
    ``_pack_nodes`` like every other builder's.

    SAH cost is the surface-area heuristic on the 3D union boxes; the temporal
    dimension rides along via each node's frame interval (checked by the kernel's
    ``tspan`` test) plus the upstream temporal segmentation. Built on the CPU
    (NumPy) -- a from-scratch unbalanced build is awkward to vectorize; this is
    the correctness/validation version (see notes on build cost).

    Currently *not consumed by the kernels*: the skip-pointer row walk was
    retired with the sibling-block traversal (``build_stbvh`` raises on
    ``builder="sah"``). Kept as the reference SAH build for a future explicit
    child-link block layout.
    """
    import sys

    import numpy as np

    M = int(prim_id.shape[0])
    if M == 0:
        nodes = torch.zeros((1, 8), dtype=torch.float32)
        nodes[0, 0:3] = EMPTY_LO
        nodes[0, 3:6] = EMPTY_HI
        nodes[0, 6] = (1 << 15) - 1
        nodes[0, 7] = 0.0
        return STBVH(
            nodes.contiguous().to(device),
            torch.tensor([-1], dtype=torch.int32, device=device),
            torch.tensor([-1], dtype=torch.int32, device=device),
            torch.tensor([(1 << 15) - 1], dtype=torch.int32, device=device),
        )
    lo = inst_lo.detach().cpu().numpy().astype(np.float64)
    hi = inst_hi.detach().cpu().numpy().astype(np.float64)
    cent = (lo + hi) * 0.5
    pid = prim_id.detach().cpu().numpy().astype(np.int64)
    a0 = t0.detach().cpu().numpy().astype(np.int64)
    a1 = t1.detach().cpu().numpy().astype(np.int64)
    opq = (inst_opaque.detach().cpu().numpy().astype(bool)
           if inst_opaque is not None else np.zeros(M, bool))

    maxN = max(2 * M - 1, 1)
    n_lo = np.zeros((maxN, 3))
    n_hi = np.zeros((maxN, 3))
    n_t0 = np.zeros(maxN, np.int64)
    n_t1 = np.zeros(maxN, np.int64)
    n_prim = np.full(maxN, -1, np.int64)
    n_opq = np.zeros(maxN, bool)
    n_skip = np.zeros(maxN, np.int64)
    ctr = [0]

    def half_area(blo, bhi):
        d = np.maximum(bhi - blo, 0.0)
        return d[0] * d[1] + d[1] * d[2] + d[0] * d[2]

    sys.setrecursionlimit(max(10000, 4 * M))

    def build(ids):
        ni = ctr[0]
        ctr[0] += 1
        blo = lo[ids].min(0)
        bhi = hi[ids].max(0)
        n_lo[ni] = blo
        n_hi[ni] = bhi
        n_t0[ni] = a0[ids].min()
        n_t1[ni] = a1[ids].max()
        if ids.shape[0] == 1:
            n_prim[ni] = pid[ids[0]]
            n_opq[ni] = opq[ids[0]]
            n_skip[ni] = ctr[0]
            return
        n = ids.shape[0]
        best_cost = np.inf
        best_left = None
        diag = bhi - blo
        # Spatio-temporal SAH: search splits along x/y/z centroid AND the time
        # centroid (axis 3); weight each child's box cost by its frame-interval
        # extent so time-mixed nodes are penalised. A ray (frame f, spatial dir)
        # only enters a node when f is in its interval, so the expected cost
        # scales with surface_area * temporal_extent -- this recovers the
        # temporal coherence the 4D Morton curve gets for free.
        for ax in range(4):
            cax = cent[ids, ax] if ax < 3 else (a0[ids] + a1[ids]) * 0.5
            cmin, cmax = cax.min(), cax.max()
            if cmax - cmin <= 1e-12:
                continue
            b = np.minimum(((cax - cmin) / (cmax - cmin) * nbins).astype(int),
                           nbins - 1)
            for s in range(1, nbins):
                lmask = b < s
                nl = int(lmask.sum())
                if nl == 0 or nl == n:
                    continue
                lids = ids[lmask]
                rids = ids[~lmask]
                te_l = float(a1[lids].max() - a0[lids].min() + 1)
                te_r = float(a1[rids].max() - a0[rids].min() + 1)
                cost = (te_l * half_area(lo[lids].min(0), hi[lids].max(0)) * nl
                        + te_r * half_area(lo[rids].min(0), hi[rids].max(0))
                        * (n - nl))
                if cost < best_cost:
                    best_cost = cost
                    best_left = lmask
        if best_left is None:
            ax = int(np.argmax(diag))
            order = np.argsort(cent[ids, ax], kind="stable")
            best_left = np.zeros(ids.shape[0], bool)
            best_left[order[:ids.shape[0] // 2]] = True
        build(ids[best_left])
        build(ids[~best_left])
        n_skip[ni] = ctr[0]

    if M > 0:
        build(np.arange(M))
    N = max(ctr[0], 1)

    nodes = torch.zeros((N, 8), dtype=torch.float32)
    nodes[:, 0:3] = torch.from_numpy(n_lo[:N]).float()
    nodes[:, 3:6] = torch.from_numpy(n_hi[:N]).float()
    nodes[:, 6] = torch.from_numpy(n_t0[:N]).float()
    nodes[:, 7] = torch.from_numpy(n_t1[:N]).float()
    skip = torch.from_numpy(n_skip[:N]).long()
    skip = torch.where(skip >= N, torch.full_like(skip, -1), skip)
    tspan = (torch.from_numpy(np.clip(n_t0[:N], 0, (1 << 15) - 1)).long()
             | (torch.from_numpy(np.clip(n_t1[:N], 0, (1 << 15) - 1)).long()
                << 16)).to(torch.int32)
    opaque_t = torch.from_numpy(n_opq[:N])
    tspan = torch.where(opaque_t, tspan | torch.tensor(-2147483648,
                        dtype=torch.int32), tspan)
    leaf_prim = torch.from_numpy(n_prim[:N]).to(torch.int32)
    return STBVH(
        nodes.contiguous().to(device),
        skip.to(torch.int32).contiguous().to(device),
        leaf_prim.contiguous().to(device),
        tspan.contiguous().to(device),
    )


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
        # The emitted instance's [t0, t1] dyadic interval is the leaf's ``tspan``:
        # the trace kernel treats the primitive as present -- and tests its
        # (still-real) geometry -- on *every* frame of that range. So a block may
        # only be emitted whole when *all* its frames are valid; a block with any
        # empty frame (the primitive invisible / zero-opacity / pre-spawn there)
        # must be descended instead. This trims leading/trailing empty frames off
        # the tspan and splits around interior gaps (e.g. a surface that fades to
        # opacity 0 for a stretch mid-batch becomes two instances with the gap
        # culled) -- otherwise that invisible geometry stays spuriously hittable
        # and z-fights whatever is coincident with it.
        full = v >= float(1 << k) - 0.5
        union_cost = _box_cost(lo, hi)
        mean_cost = c / v.clamp_min(1)
        tight = union_cost <= tightness * mean_cost + 1e-12
        if k == 0:
            tight = torch.ones_like(tight)
        emit = active & nonempty & tight & full
        if emit.any():
            t_idx, n_idx = emit.nonzero(as_tuple=True)
            out_prim.append(n_idx)
            out_t0.append(t_idx << k)
            out_t1.append(torch.clamp(((t_idx + 1) << k) - 1, max=T - 1))
            out_lo.append(lo[t_idx, n_idx])
            out_hi.append(hi[t_idx, n_idx])
        if k > 0:
            active = (active & nonempty & ~(tight & full)).repeat_interleave(2, dim=0)

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
    """Miss links for stackless DFS over the implicit complete BVH_ARITY-ary
    tree: a skipped or finished node jumps to its next sibling, or -- for the
    last sibling in a group -- to its parent's miss target. Levels are filled
    top-down so each parent's link is already set when its children read it.
    """
    a = BVH_ARITY
    num_internal = (num_leaves - 1) // (a - 1)
    num_nodes = num_internal + num_leaves
    miss = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
    start, width = 1, a
    while start < num_nodes:
        idx = torch.arange(start, start + width, device=device)
        pos = (idx - 1) % a  # position within the sibling group (0..a-1)
        parent_miss = miss[(idx - 1) // a]
        miss[idx] = torch.where(pos == (a - 1), parent_miss, idx + 1)
        start += width
        width *= a
    return miss


def build_stbvh(frame_lo, frame_hi, num_frames=None, tightness=2.0,
                opaque=None, builder="morton"):
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
    opaque : Tensor[To, N] (bool), optional
        Per-frame full-opacity flags (``To`` is 1 or ``Tc``). An instance is
        marked opaque (``leaf_tspan`` bit 31) when the primitive is opaque on
        *every* frame of the instance's interval, allowing the renderer to
        prune hits behind it during gathering.
    builder : str
        Instance-ordering strategy: "morton", "split" or "sah" (see the
        ``_BVH_BUILD`` comment above for the trade-offs and per-geometry-type
        defaults). Overridden globally by env ALGAN_BVH_BUILD when set.
    """
    if _BVH_BUILD is not None:
        builder = _BVH_BUILD
    Tc, N, _ = frame_lo.shape
    device = frame_lo.device
    if num_frames is None:
        num_frames = Tc
    if num_frames >= 1 << 15:
        raise ValueError(
            f"STBVH leaf frame intervals are packed into 16-bit halves; "
            f"render batches must stay below {1 << 15} frames "
            f"(got {num_frames}).")

    if Tc == 1:
        valid = (frame_hi[0] >= frame_lo[0]).all(-1)
        prim_id = valid.nonzero(as_tuple=True)[0]
        t0 = torch.zeros_like(prim_id)
        t1 = torch.full_like(prim_id, num_frames - 1)
        inst_lo = frame_lo[0, prim_id]
        inst_hi = frame_hi[0, prim_id]
        inst_opaque = opaque.all(0)[prim_id] if opaque is not None else None
    else:
        if Tc != num_frames:
            raise ValueError(
                f"frame bounds have {Tc} frames but the batch has {num_frames}"
            )
        prim_id, t0, t1, inst_lo, inst_hi = segment_primitives_in_time(
            frame_lo, frame_hi, tightness
        )
        if opaque is not None:
            if opaque.shape[0] == 1:
                inst_opaque = opaque[0, prim_id]
            else:
                # Opaque over [t0, t1] iff no non-opaque frame in between
                # (prefix sums of the negated mask).
                prefix = torch.zeros((Tc + 1, N), dtype=torch.long,
                                     device=device)
                prefix[1:] = (~opaque).long().cumsum(0)
                inst_opaque = (prefix[t1 + 1, prim_id]
                               - prefix[t0, prim_id]) == 0
        else:
            inst_opaque = None

    if builder == "sah":
        # The explicit-DFS SAH tree's in-kernel walk (skip pointers over
        # per-node rows) was retired with the sibling-block traversal; the
        # builder is kept below as the reference for a future ST-SAH layout
        # with explicit per-block child links (an unbalanced tree cannot use
        # the implicit-heap child indexing the block walk relies on).
        raise NotImplementedError(
            "The SAH DFS traversal was removed with the sibling-block "
            "traversal rework; unset ALGAN_BVH_BUILD=sah (see "
            "stbvh._build_sah_dfs).")

    M = prim_id.shape[0]
    L = LEAF_SIZE
    a = BVH_ARITY
    num_groups = max((M + L - 1) // L, 1)
    # Leaf count: the smallest power of the arity that holds all groups, and
    # at least one full sibling group -- the block walk always fetches
    # ARITY-wide child blocks starting at the root, so even empty or
    # single-instance trees keep one internal root + ARITY leaves (padding
    # slots carry an impossible frame interval and never pass the gate).
    P = a
    while num_groups > P:
        P *= a
    num_nodes = (a * P - 1) // (a - 1)
    first_leaf = num_nodes - P

    use_split = (builder == "split") and (L == 1) and (M > 0)
    if use_split:
        # Recursive longest-axis median split: tighter internal boxes than the
        # Morton curve, fewer traversal steps. The four axes are normalized so
        # space and time are comparable, then instances are bisected into the
        # P leaf slots (padding slots map to -1). Same instances/opaque flags as
        # Morton, just a different sibling grouping.
        center = (inst_lo + inst_hi) * 0.5
        t_center = (t0 + t1).float() * 0.5
        smin = inst_lo.amin(0)
        smax = inst_hi.amax(0)
        cn = (center - smin) / (smax - smin).clamp_min(1e-12)
        tn = (t_center / float(max(num_frames - 1, 1))).unsqueeze(-1)
        tn = tn * SPLIT_TIME_WEIGHT
        slot_src = _median_split_slots(torch.cat((cn, tn), -1), P)  # [P]
        real = slot_src >= 0
        src = slot_src.clamp_min(0)
        slot_lo = torch.where(real.unsqueeze(-1), inst_lo[src],
                              torch.tensor(EMPTY_LO, device=device))
        slot_hi = torch.where(real.unsqueeze(-1), inst_hi[src],
                              torch.tensor(EMPTY_HI, device=device))
        slot_t0 = torch.where(real, t0[src],
                              torch.full_like(t0[src], (1 << 15) - 1))
        slot_t1 = torch.where(real, t1[src], torch.zeros_like(t1[src]))
        leaf_prim = torch.where(real, prim_id[src],
                                torch.full_like(prim_id[src], -1))
        if inst_opaque is not None:
            slot_opaque = real & inst_opaque[src]
        else:
            slot_opaque = torch.zeros((P,), dtype=torch.bool, device=device)
    else:
        if M > 0:
            # Sort instances along a 4D Morton curve so the implicit tree gets
            # spatio-temporally coherent subtrees.
            center = (inst_lo + inst_hi) * 0.5
            t_center = (t0 + t1).float() * 0.5
            smin = inst_lo.amin(0)
            smax = inst_hi.amax(0)
            q = _quantize(center, smin, smax)
            qt = _quantize(t_center, torch.zeros((), device=device),
                           torch.full((), float(max(num_frames - 1, 1)),
                                      device=device))
            codes = morton_code_4d(q[:, 0], q[:, 1], q[:, 2], qt)
            order = torch.argsort(codes)
            prim_id, t0, t1 = prim_id[order], t0[order], t1[order]
            inst_lo, inst_hi = inst_lo[order], inst_hi[order]
            if inst_opaque is not None:
                inst_opaque = inst_opaque[order]

        # Consecutive Morton-ordered instances share leaves, LEAF_SIZE at a time.
        # Padding slots get an impossible frame interval (tmin > tmax) and empty
        # bounds so they are never visited.
        slot_lo = torch.full((P * L, 3), EMPTY_LO, device=device)
        slot_hi = torch.full((P * L, 3), EMPTY_HI, device=device)
        slot_t0 = torch.full((P * L,), (1 << 15) - 1, dtype=torch.long,
                             device=device)
        slot_t1 = torch.zeros((P * L,), dtype=torch.long, device=device)
        slot_opaque = torch.zeros((P * L,), dtype=torch.bool, device=device)
        leaf_prim = torch.full((P * L,), -1, dtype=torch.long, device=device)
        if M > 0:
            slot_lo[:M] = inst_lo
            slot_hi[:M] = inst_hi
            slot_t0[:M] = t0
            slot_t1[:M] = t1
            leaf_prim[:M] = prim_id
            if inst_opaque is not None:
                slot_opaque[:M] = inst_opaque
    leaf_tspan = (slot_t0.clamp(0, (1 << 15) - 1)
                  | (slot_t1.clamp(0, (1 << 15) - 1) << 16)).to(torch.int32)
    # Bit 31 (sign bit) flags interval-opaque instances.
    leaf_tspan = torch.where(
        slot_opaque, leaf_tspan | torch.tensor(-2147483648, dtype=torch.int32,
                                               device=device), leaf_tspan)

    nodes = torch.empty((num_nodes, 8), device=device)
    nodes[first_leaf:, 0:3] = slot_lo.view(P, L, 3).amin(1)
    nodes[first_leaf:, 3:6] = slot_hi.view(P, L, 3).amax(1)
    nodes[first_leaf:, 6] = slot_t0.view(P, L).amin(1).float()
    nodes[first_leaf:, 7] = slot_t1.view(P, L).amax(1).float()

    # Bottom-up union of children into parents, one vectorized level at a time.
    # Each parent unions its `a` consecutive children (strided slices of the
    # level), so this works for any arity (a == 2 reproduces the binary union).
    offset, width = first_leaf, P
    while width > 1:
        child = nodes[offset:offset + width]
        parent_width = width // a
        parent_offset = offset - parent_width
        lo = torch.minimum(child[0::a, 0:3], child[1::a, 0:3])
        hi = torch.maximum(child[0::a, 3:6], child[1::a, 3:6])
        tlo = torch.minimum(child[0::a, 6], child[1::a, 6])
        thi = torch.maximum(child[0::a, 7], child[1::a, 7])
        for c in range(2, a):
            lo = torch.minimum(lo, child[c::a, 0:3])
            hi = torch.maximum(hi, child[c::a, 3:6])
            tlo = torch.minimum(tlo, child[c::a, 6])
            thi = torch.maximum(thi, child[c::a, 7])
        nodes[parent_offset:offset, 0:3] = lo
        nodes[parent_offset:offset, 3:6] = hi
        nodes[parent_offset:offset, 6] = tlo
        nodes[parent_offset:offset, 7] = thi
        offset, width = parent_offset, parent_width

    miss = _compute_miss_links(P, device)

    return STBVH(
        nodes.contiguous(),
        miss.to(torch.int32).contiguous(),
        leaf_prim.to(torch.int32).contiguous(),
        leaf_tspan.contiguous(),
    )
