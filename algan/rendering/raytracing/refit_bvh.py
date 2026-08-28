"""Shared-topology binned-SAH refit BVH (raytracer-v2 design doc, section 9).

The classic :mod:`stbvh` builds a spatio-temporal tree over *primitive
instances*: moving geometry segments into near-per-frame instances at the
confirmed-optimal ``tightness=1.0``, so the tree is up to ~10x larger than the
primitive count and every ray wades through mostly other-frames' instances
gated out by frame-interval tests. The phase-1 measurements
(``benchmarks/_rt2_refit_sah.py``) showed a *refit* topology -- ONE tree over
the N primitives whose node bounds are recomputed per frame -- costs
1.37-2.33x fewer expected node visits, with negligible staleness (<= 1.04 vs
a per-frame rebuild) across a render batch, and makes a real SAH build
affordable because topology is built once per batch.

This module implements that structure:

* **Topology**: a top-down *binned SAH* build over the batch-union boxes of
  the ever-visible primitives -- binary splits, ``log2(BVH_ARITY)`` of them
  per emitted node level, collapsed directly into ``BVH_ARITY``-wide sibling
  blocks. The tree is *unbalanced* (a leaf child can sit beside an internal
  child), which the implicit-heap layout of the classic tree cannot express,
  so every sibling block carries **explicit per-child links**.
* **Refit**: per-frame node bounds computed bottom-up as one vectorized
  ``[T, blocks, 8, ARITY]`` reduction per tree level; static geometry
  (single-frame input bounds) dedupes to ``T = 1``. Boxes are exactly tight
  per frame -- the thing the tightness A/B proved dominates traversal cost --
  and the frame-interval gates of the classic walk disappear.

Kernel-facing layout (consumed by the ``refit`` compile-time branch of the
walks in ``raytrace_kernels_taichi.py``):

``blocks [Tb * num_blocks, 8, BVH_ARITY]`` (f32, or conservatively
out-rounded f16 under ``stbvh.BLOCK_F16``): frame ``t``'s sibling block for
internal node ``i`` is row ``t * num_blocks + i``. Lanes 0-5 hold each
child's per-frame bounds exactly like the classic blocks; lanes 6-7 hold a
packed **per-(frame, child) int32 link word** (f32 blocks bit-cast it into
lane 6; f16 blocks store its low/high u16 halves in lanes 6/7, the same
scheme the classic tspan uses):

* ``-1``                       -- no child in this slot, or the child's whole
                                  subtree is invisible at this frame. The
                                  walk skips it. (An *empty* box cannot do
                                  this job: the slab test min/max-normalizes
                                  each axis, so inverted bounds still pass.)
* sign bit set (``< 0``, != -1) -- leaf child: bits 0-29 the primitive index,
                                  bit 30 the primitive's *per-frame* full-
                                  opacity flag (exact, unlike the classic
                                  per-interval flag; efficiency-only either
                                  way).
* ``>= 0``                     -- internal child: its sibling-block index.

The object intentionally quacks like :class:`stbvh.STBVH` -- same five tensor
field names, with ``first_leaf`` carrying ``num_blocks`` and ``nodes`` /
``node_miss`` / ``leaf_prim`` / ``leaf_tspan`` one-element placeholders -- so
every launch site's ``(blocks, node_miss, leaf_prim, leaf_tspan, first_leaf)``
quintuple, the merged-scene arena upload and the memory accounting work
unchanged; the kernels select the walk with a compile-time ``refit`` template.

Everything here is vectorized PyTorch (level-synchronous: one batched binned-
SAH split pass per binary level across every node of that level), so the
build runs on the render device under ``merge_on_gpu`` like the classic
builders.
"""

from __future__ import annotations

import warnings

import torch

from algan.environment import env_int
from algan.rendering.raytracing.stbvh import (
    BLOCK_F16,
    BVH_ARITY,
    EMPTY_HI,
    EMPTY_LO,
    STBVH,
    _half_bits_directed,
)

# Number of centroid bins per axis for the SAH split search. 16 is the
# standard sweet spot: finer binning stops improving tree quality while the
# histogram/prefix cost grows linearly. Purely a build-quality/build-cost knob
# -- the traversal is arrangement-invariant -- so it is exposed for tuning
# rather than fixed.
SAH_BINS = max(2, env_int("ALGAN_SAH_BINS", 16))

# Depth budget of the emitted ARITY-ary tree. Must not exceed the traversal
# kernels' fixed sibling-stack depth (raytrace_kernels_taichi._GROUP_STACK,
# 16): the walk pushes at most one entry per level. The builder force-switches
# a node to median splits whenever free SAH splitting could no longer resolve
# its subtree within the remaining budget (see ``_forced_median``), so the cap
# holds for any input.
MAX_DEPTH = 16

# Link-word encoding (see module docstring). Bits 0-29 carry a primitive
# index for leaf children, so a batch is limited to 2**30 - 1 primitives per
# geometry type; block indices are stored raw, and the walk packs
# ``block << BVH_ARITY`` into its int32 stack entries, capping blocks at
# 2**(31 - BVH_ARITY).
LINK_INVALID = -1
LINK_LEAF_BIT = -2147483648  # 1 << 31 as int32
LINK_OPAQUE_BIT = 1 << 30
# Bit 29: set when the leaf's primitive declared that it casts no shadow
# (``Mob.casts_shadows`` False). The link word is what the traversal already
# loads to find the leaf's primitive, so a non-caster is rejected here for no
# additional memory traffic -- the same argument as ``stbvh.LEAF_NOCAST_BIT``,
# which is that tree kind's spelling of this flag. Unlike the STBVH's word this
# one had no spare bit, so the primitive index gives one up: 2^29 primitives is
# far beyond any batch the rest of the renderer can hold, and the builder's
# existing range guard below now enforces the narrower bound.
LINK_NOCAST_BIT = 1 << 29
LINK_PRIM_MASK = (1 << 29) - 1


class RefitBVH(STBVH):
    """Flat tensor form of the shared-topology refit BVH.

    Attributes mirror :class:`stbvh.STBVH` so scene plumbing (arena upload,
    aliased-field memory accounting, kernel launch quintuples) is type-blind:

    ``blocks``    -- ``[Tb * num_blocks, 8, BVH_ARITY]`` per-frame sibling
                     blocks (see module docstring).
    ``first_leaf``-- carries ``num_blocks`` (the walk derives the frame count
                     as ``blocks.shape[0] // num_blocks`` and the row base as
                     ``(f % Tb) * num_blocks``).
    ``nodes`` / ``node_miss`` / ``leaf_prim`` / ``leaf_tspan``
                  -- one-element placeholders: the refit walk reads only
                     ``blocks``, but the tensors keep their argument slots in
                     every kernel signature.
    """

    def __init__(self, blocks, num_blocks, num_time, device):
        self.blocks = blocks
        self.num_blocks = int(num_blocks)
        self.num_time = int(num_time)
        self.first_leaf = int(num_blocks)
        self.nodes = torch.zeros((1, 8), dtype=torch.float32, device=device)
        self.node_miss = torch.full((1,), -1, dtype=torch.int32, device=device)
        self.leaf_prim = torch.full((1,), -1, dtype=torch.int32, device=device)
        self.leaf_tspan = torch.zeros((1,), dtype=torch.int32, device=device)
        # STBVH derived field, kept coherent for type-blind consumers.
        self.num_leaves = 0

    @property
    def num_nodes(self):
        return self.num_blocks

    @classmethod
    def from_prebuilt(cls, nodes, node_miss, leaf_prim, leaf_tspan, blocks, like=None):
        """Reattach already-built tensors (arena upload path). ``like`` is the
        source object whose scalar layout fields are carried over -- unlike
        the classic tree they cannot be derived from the tensor shapes alone
        (``blocks`` has ``Tb`` frames of ``num_blocks`` rows flattened
        together).
        """
        if like is None:
            raise ValueError("RefitBVH.from_prebuilt requires like=<source>")
        self = cls.__new__(cls)
        self.blocks = blocks
        self.num_blocks = like.num_blocks
        self.num_time = like.num_time
        self.first_leaf = like.num_blocks
        self.num_leaves = 0
        self.nodes = nodes
        self.node_miss = node_miss
        self.leaf_prim = leaf_prim
        self.leaf_tspan = leaf_tspan
        return self


def _half_area(lo, hi):
    """Surface-area-heuristic box cost: half surface area, 0 for empty."""
    d = (hi - lo).clamp_min(0)
    return d[..., 0] * d[..., 1] + d[..., 1] * d[..., 2] + d[..., 0] * d[..., 2]


def _binary_split(order, starts, counts, forced, cent, ulo, uhi):
    """One vectorized binned-SAH binary split pass over a set of node ranges.

    ``order`` is the global tree-primitive permutation; each range ``i`` owns
    ``order[starts[i] : starts[i] + counts[i]]`` (``counts[i] >= 2``).
    ``forced`` flags ranges that must median-split (degenerate SAH input or
    depth-budget pressure). Partitions each range's slice of ``order`` in
    place into [left | right] and returns ``(left_counts)``.
    """
    device = order.device
    K = starts.shape[0]
    S = int(counts.sum())
    seg = torch.repeat_interleave(torch.arange(K, device=device), counts)
    base = torch.zeros(K, dtype=torch.long, device=device)
    base[1:] = counts.cumsum(0)[:-1]
    rank = torch.arange(S, device=device) - base[seg]  # pos in range
    pos = starts[seg] + rank  # pos in order
    tp = order[pos]
    pc = cent[tp]  # [S, 3]

    # Per-range centroid bounds -> per-primitive bin index on each axis.
    cmin = torch.full((K, 3), float("inf"), device=device)
    cmax = torch.full((K, 3), float("-inf"), device=device)
    # ``index_reduce_`` remains the supported vectorized operation for this
    # reduction, but PyTorch intentionally warns that its API is beta. Keep
    # that implementation warning local to these internal uses.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"index_reduce\(\) is in beta and the API may change at any time\.",
            category=UserWarning,
        )
        cmin.index_reduce_(0, seg, pc, "amin")
        cmax.index_reduce_(0, seg, pc, "amax")
    ext = cmax - cmin
    nb = SAH_BINS
    t = (pc - cmin[seg]) / ext[seg].clamp_min(1e-30)
    bins = (t * nb).long().clamp_(0, nb - 1)  # [S, 3]

    # Histograms + per-bin box unions, per (range, axis, bin).
    ax_off = torch.arange(3, device=device).view(1, 3)
    idx = (seg.view(-1, 1) * 3 + ax_off) * nb + bins  # [S, 3]
    cnt = torch.zeros(K * 3 * nb, dtype=torch.long, device=device)
    cnt.index_add_(
        0, idx.reshape(-1), torch.ones(S * 3, dtype=torch.long, device=device)
    )
    blo = torch.full((K * 3 * nb, 3), float("inf"), device=device)
    bhi = torch.full((K * 3 * nb, 3), float("-inf"), device=device)
    plo = ulo[tp]
    phi = uhi[tp]
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"index_reduce\(\) is in beta and the API may change at any time\.",
            category=UserWarning,
        )
        for a in range(3):
            blo.index_reduce_(0, idx[:, a], plo, "amin")
            bhi.index_reduce_(0, idx[:, a], phi, "amax")
    cnt = cnt.view(K, 3, nb)
    blo = blo.view(K, 3, nb, 3)
    bhi = bhi.view(K, 3, nb, 3)

    # Prefix (left) and suffix (right) sweeps over the bins; split s puts
    # bins [0, s] left and (s, nb) right, s in [0, nb - 2].
    lcnt = cnt.cumsum(-1)[..., :-1]  # [K, 3, nb-1]
    rcnt = counts.view(K, 1, 1) - lcnt
    llo = blo.cummin(2).values[:, :, :-1]
    lhi = bhi.cummax(2).values[:, :, :-1]
    rlo = blo.flip(2).cummin(2).values.flip(2)[:, :, 1:]
    rhi = bhi.flip(2).cummax(2).values.flip(2)[:, :, 1:]
    cost = _half_area(llo, lhi) * lcnt + _half_area(rlo, rhi) * rcnt
    invalid = (lcnt == 0) | (rcnt == 0)
    cost = torch.where(invalid, torch.full_like(cost, float("inf")), cost)
    flat = cost.view(K, 3 * (nb - 1))
    best = flat.argmin(1)  # [K]
    best_axis = best // (nb - 1)
    best_bin = best % (nb - 1)
    no_valid = torch.isinf(flat.gather(1, best.view(K, 1)).squeeze(1))
    use_median = forced | no_valid

    # Side assignment: SAH ranges split by bin > best_bin on the best axis;
    # median ranges split at floor(count / 2) in current order.
    pb = bins.gather(1, best_axis[seg].view(-1, 1)).squeeze(1)
    side_sah = pb > best_bin[seg]
    side_med = rank >= (counts[seg] // 2)
    side = torch.where(use_median[seg], side_med, side_sah)

    # Stable partition of each range's slice of ``order`` into [left|right].
    key = seg * 2 + side.long()
    perm = torch.argsort(key, stable=True)
    order[pos] = tp[perm]
    nl = torch.zeros(K, dtype=torch.long, device=device)
    nl.index_add_(0, seg, (~side).long())
    return nl


def build_refit_bvh(
    frame_lo,
    frame_hi,
    num_frames=None,
    opaque=None,
    tightness=None,
    builder=None,
    casts=None,
):
    """Build a shared-topology binned-SAH refit BVH from per-frame bounds.

    Parameters mirror :func:`stbvh.build_stbvh` (``tightness`` / ``builder``
    are accepted and ignored so call sites can dispatch on a toggle without
    reshaping their arguments): ``frame_lo`` / ``frame_hi`` are
    ``[Tc, N, 3]`` per-frame AABBs with invisible frames marked empty
    (``lo = EMPTY_LO, hi = EMPTY_HI``); ``Tc`` may be 1 for static geometry
    (the refit dedupes to one time slice). ``opaque`` is an optional
    ``[To, N]`` bool mask (``To`` in {1, Tc}) feeding the per-frame leaf
    opacity flag; a static tree (``Tc == 1``) also accepts a full ``[T, N]``
    mask and reduces it conservatively over frames, like ``build_stbvh``.
    """
    Tc, N, _ = frame_lo.shape
    device = frame_lo.device
    if num_frames is None:
        num_frames = Tc
    if Tc not in (1, num_frames):
        raise ValueError(
            f"frame bounds have {Tc} frames but the batch has {num_frames}"
        )

    valid = (frame_hi >= frame_lo).all(-1)  # [Tc, N]
    ever = valid.any(0)
    pids = ever.nonzero(as_tuple=True)[0]  # tree -> orig
    M = int(pids.shape[0])
    if M >= LINK_PRIM_MASK:
        raise ValueError(
            f"refit BVH link words carry 29-bit primitive indices; got {M}"
        )

    # Batch-union boxes + centroids of the ever-visible primitives (the SAH
    # metric of the validated "union" topology in _rt2_refit_sah.py).
    vlo = torch.where(
        valid.unsqueeze(-1), frame_lo, torch.full_like(frame_lo, EMPTY_LO)
    )
    vhi = torch.where(
        valid.unsqueeze(-1), frame_hi, torch.full_like(frame_hi, EMPTY_HI)
    )
    ulo = vlo.amin(0)[pids]  # [M, 3]
    uhi = vhi.amax(0)[pids]
    cent = (ulo + uhi) * 0.5

    a = BVH_ARITY
    s_rounds = a.bit_length() - 1  # log2(arity)

    # ------------------------------------------------------------------
    # Topology: level-synchronous build. Each level splits every current
    # node range into up to ARITY sub-ranges (log2(ARITY) binary passes);
    # count-1 sub-ranges become leaf children, larger ones become the next
    # level's nodes. Blocks are numbered in BFS order.
    # ------------------------------------------------------------------
    order = torch.arange(M, dtype=torch.long, device=device)
    kind_rows = []  # per level: [K, A] uint8 (0 absent, 1 leaf, 2 internal)
    ref_rows = []  # per level: [K, A] long (leaf: TREE prim; internal: block)
    levels = []  # (block_start, block_end) per level
    next_block = 0
    if M <= 1:
        kind = torch.zeros((1, a), dtype=torch.uint8, device=device)
        ref = torch.full((1, a), -1, dtype=torch.long, device=device)
        if M == 1:
            kind[0, 0] = 1
            ref[0, 0] = 0
        kind_rows.append(kind)
        ref_rows.append(ref)
        levels.append((0, 1))
        next_block = 1
    else:
        r_start = torch.zeros(1, dtype=torch.long, device=device)
        r_count = torch.full((1,), M, dtype=torch.long, device=device)
        depth = 0
        while r_start.numel():
            K = r_start.shape[0]
            lv_start = next_block
            next_block += K
            levels.append((lv_start, next_block))
            # Depth-budget pressure: free SAH splitting guarantees only one
            # primitive off per level in the worst case, so once a node's
            # count could exceed what balanced splitting resolves in the
            # remaining levels, force median splits (children <= ~count/ARITY
            # per level, resolving within the budget).
            remaining = MAX_DEPTH - depth
            forced_blk = r_count > (a ** max(remaining - 1, 0))

            sub_start = r_start
            sub_count = r_count
            sub_block = torch.arange(K, device=device)
            for _ in range(s_rounds):
                splittable = sub_count >= 2
                if not bool(splittable.any()):
                    break
                ss = sub_start[splittable]
                sc = sub_count[splittable]
                sb = sub_block[splittable]
                nl = _binary_split(order, ss, sc, forced_blk[sb], cent, ulo, uhi)
                keep_start = sub_start[~splittable]
                keep_count = sub_count[~splittable]
                keep_block = sub_block[~splittable]
                sub_start = torch.cat([keep_start, ss, ss + nl])
                sub_count = torch.cat([keep_count, nl, sc - nl])
                sub_block = torch.cat([keep_block, sb, sb])
            # Canonical child order: left-to-right within each block.
            skey = torch.argsort(sub_block * (M + 1) + sub_start)
            sub_start = sub_start[skey]
            sub_count = sub_count[skey]
            sub_block = sub_block[skey]
            slot = torch.arange(sub_block.shape[0], device=device) - torch.searchsorted(
                sub_block, sub_block, right=False
            )
            is_leaf = sub_count == 1
            kind = torch.zeros((K, a), dtype=torch.uint8, device=device)
            ref = torch.full((K, a), -1, dtype=torch.long, device=device)
            kind[sub_block, slot] = torch.where(
                is_leaf,
                torch.ones_like(sub_block, dtype=torch.uint8),
                torch.full_like(sub_block, 2, dtype=torch.uint8),
            )
            leaf_ref = order[sub_start]
            child_ids = torch.cumsum((~is_leaf).long(), 0) - 1 + next_block
            ref[sub_block, slot] = torch.where(is_leaf, leaf_ref, child_ids)
            kind_rows.append(kind)
            ref_rows.append(ref)
            r_start = sub_start[~is_leaf]
            r_count = sub_count[~is_leaf]
            depth += 1
            if depth > MAX_DEPTH:
                raise RuntimeError("refit BVH exceeded its depth budget (builder bug)")

    B = next_block
    if B > (1 << (31 - BVH_ARITY)):
        raise ValueError(
            f"refit BVH has {B} sibling blocks; the traversal stack packs "
            f"block << {BVH_ARITY} into int32 entries"
        )
    child_kind = torch.cat(kind_rows, 0)  # [B, A]
    child_ref = torch.cat(ref_rows, 0)  # [B, A]
    # Leaf refs: tree-primitive -> original primitive index.
    if M:
        child_ref = torch.where(
            child_kind == 1, pids[child_ref.clamp_min(0)], child_ref
        )

    # ------------------------------------------------------------------
    # Refit: per-frame child boxes + links per level, bottom-up, all frames
    # in one [Tb, blocks-in-level, ARITY, 3] reduction per level.
    # ------------------------------------------------------------------
    Tb = Tc
    if opaque is None:
        opq = torch.zeros((1, N), dtype=torch.bool, device=device)
    elif opaque.shape[0] in (1, Tb):
        opq = opaque
    elif Tb == 1:
        # Static geometry: the bounds deduped to one time slice while the
        # opacity mask still carries the batch's frames -- the merge collapses
        # temporally-constant tables one at a time, so a batch whose geometry
        # holds still while a mob fades arrives here as Tc == 1, To == T. One
        # tree covers every frame, so the flag may only be set where it holds
        # on ALL of them: the same conservative reduction build_stbvh applies
        # to its Tc == 1 instances. Under-claiming is free -- the bit is
        # efficiency-only (it prunes hits behind a proven-opaque one), while
        # over-claiming would delete geometry behind a translucent frame.
        opq = opaque.all(0, keepdim=True)
    else:
        raise ValueError(
            f"refit BVH opacity mask has {opaque.shape[0]} frames; expected "
            f"1 or {Tb} to match the frame bounds"
        )
    # True where the primitive casts no shadow, [N]. Reduced with ``all`` over
    # frames like the STBVH's stamp: the flag is fixed for the render, so this
    # is exact. All-casting (the common case) leaves ``leaf_w`` bit-for-bit what
    # it was before the flag existed, because the OR term is then a constant 0.
    if casts is None:
        nocast = torch.zeros((N,), dtype=torch.int32, device=device)
    else:
        c = casts.all(0) if casts.dim() == 2 else casts
        nocast = (~c).to(torch.int32).to(device)
    nb_lo = torch.empty((Tb, B, 3), device=device)
    nb_hi = torch.empty((Tb, B, 3), device=device)
    ch_lo = torch.empty((Tb, B, a, 3), device=device)
    ch_hi = torch.empty((Tb, B, a, 3), device=device)
    link = torch.empty((Tb, B, a), dtype=torch.int32, device=device)
    for lv_s, lv_e in reversed(levels):
        kind = child_kind[lv_s:lv_e]  # [k, A]
        ref = child_ref[lv_s:lv_e]
        # Per-kind clamped gather indices: a slot's ref is a primitive index
        # for leaves and a block index for internal children, so each gather
        # must be bounded by its own array.
        safe_prim = torch.where(kind == 1, ref, torch.zeros_like(ref)).reshape(-1)
        safe_blk = torch.where(kind == 2, ref, torch.zeros_like(ref)).reshape(-1)
        leaf_lo = vlo[:, safe_prim].view(Tb, -1, a, 3)
        leaf_hi = vhi[:, safe_prim].view(Tb, -1, a, 3)
        int_lo = nb_lo[:, safe_blk].view(Tb, -1, a, 3)
        int_hi = nb_hi[:, safe_blk].view(Tb, -1, a, 3)
        km = kind.view(1, -1, a, 1)
        c_lo = torch.where(
            km == 1,
            leaf_lo,
            torch.where(km == 2, int_lo, torch.full_like(leaf_lo, EMPTY_LO)),
        )
        c_hi = torch.where(
            km == 1,
            leaf_hi,
            torch.where(km == 2, int_hi, torch.full_like(leaf_hi, EMPTY_HI)),
        )
        ch_lo[:, lv_s:lv_e] = c_lo
        ch_hi[:, lv_s:lv_e] = c_hi
        nb_lo[:, lv_s:lv_e] = c_lo.amin(2)
        nb_hi[:, lv_s:lv_e] = c_hi.amax(2)
        # Link words: invalid whenever the child's subtree is empty at t.
        alive = (c_hi >= c_lo).all(-1)  # [Tb, k, A]
        leaf_opq = opq[:, safe_prim].view(opq.shape[0], -1, a).to(torch.int32)
        if leaf_opq.shape[0] != Tb:
            leaf_opq = leaf_opq.expand(Tb, -1, -1)
        leaf_nocast = nocast[safe_prim].view(1, -1, a)
        leaf_w = (
            ref.to(torch.int32)
            | LINK_LEAF_BIT
            | (leaf_opq * LINK_OPAQUE_BIT)
            | (leaf_nocast * LINK_NOCAST_BIT)
        )
        w = torch.where(
            kind.view(1, -1, a) == 1,
            leaf_w,
            ref.to(torch.int32).view(1, -1, a).expand(Tb, -1, -1),
        )
        link[:, lv_s:lv_e] = torch.where(
            alive & (kind.view(1, -1, a) > 0), w, torch.full_like(w, LINK_INVALID)
        )

    # ------------------------------------------------------------------
    # Pack the kernel-facing blocks: [Tb * B, 8, ARITY].
    # ------------------------------------------------------------------
    rows = Tb * B
    lo_flat = ch_lo.view(rows, a, 3)
    hi_flat = ch_hi.view(rows, a, 3)
    link_flat = link.view(rows, a)
    if BLOCK_F16:
        blk = torch.zeros((rows, 8, a), dtype=torch.int16, device=device)
        for d in range(3):
            blk[:, d] = _half_bits_directed(lo_flat[..., d], up=False)
            blk[:, 3 + d] = _half_bits_directed(hi_flat[..., d], up=True)
        halves = link_flat.contiguous().view(torch.int16).view(rows, a, 2)
        blk[:, 6] = halves[..., 0]
        blk[:, 7] = halves[..., 1]
        blocks = blk.view(torch.float16).contiguous()
    else:
        blk = torch.zeros((rows, 8, a), dtype=torch.float32, device=device)
        for d in range(3):
            blk[:, d] = lo_flat[..., d]
            blk[:, 3 + d] = hi_flat[..., d]
        blk[:, 6] = link_flat.view(torch.float32)
        blocks = blk.contiguous()
    return RefitBVH(blocks, B, Tb, device)
