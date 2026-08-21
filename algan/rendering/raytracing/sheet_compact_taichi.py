"""Kernels for the fragment/sheet compaction's multi-pass host loops.

``DESIGN_optimization_targets.md`` T5 and ``DESIGN_sheet_resolve.md`` §10.4.
The compaction between the emission and the resolve is host torch, and three
of its passes are shaped so that torch has to walk the fragment stream many
times to compute something a kernel computes in registers in one:

``gather_fragment_arrays``
    The sorted fragment stream is materialized by six ``index_select`` calls
    that share one permutation, so the permutation is read six times and each
    output is a separate launch. Measured at 3840x2160 (3.66 M fragments):
    26.1 ms for ~106 bytes of traffic per fragment against the 66 the fused
    gather moves. Used twice -- once for the sort, once for the opaque-prefix
    truncation's keep-mask compaction.

``sheet_band_reduce``
    The sample-mask union and the §6.2 fusion detector were one
    ``scatter_add_`` per sample lane (eight passes over the stream and eight
    over the sheet array) to learn, per lane, whether the count is 0, 1, or
    more. 41.6 ms per call at 4K, called twice a frame. The exact-area sum
    rides along in the same pass: it walks the identical stream, and in torch
    it could not share it because ``scatter_add_`` wanted an f64 copy of the
    whole fragment array first.

``mask_popcount``
    Eight shift/and/add passes over the sheet array to count at most eight
    bits. 15.7 ms per call at 4K, called once or twice a frame.

``sheet_conflict_rank``
    The conflict-rank scan was eight ``torch.cumsum`` passes over the stream
    plus, per lane, an ``index_select``, a ``maximum`` and two ``where``s,
    with five live ``[n]`` arrays at the peak (RENDERER_WORK_QUEUE.md item 11;
    DESIGN_sheet_resolve.md §10.4 names it the compaction's one genuine
    remaining scan). One thread per BAND walks its fragments forward instead,
    holding the eight per-lane counters in registers and reading each before
    its own increment -- which is exactly the exclusive prefix the cumsums
    computed, because a lane's counter is only ever touched by its own band's
    walk. The sorted+masked copy the torch loop materialized never exists:
    the kernel gathers ``msk[order[j]]`` itself.

Bit-identity
------------
Every kernel here is integer-only or a verbatim copy of float bits, so
Taichi's ``fast_math`` -- which is what makes ``logical_pn_taichi``'s criteria
*not* bit-identical to their torch originals -- has nothing to act on. The
gather is a permutation. The popcount is exact. The union is an OR, and the
fusion detector is the one non-obvious case:

    ``prev = atomic_or(union[b], bits)`` returns the value before this
    fragment's contribution, so for a lane set in exactly ``k`` fragments of
    the band, exactly ``k - 1`` of them observe it already set -- whichever
    order the atomics serialize in. ``prev & bits`` is therefore non-empty for
    some fragment iff some lane has ``k >= 2``, which is precisely
    ``lane > 1`` in the loop this replaces.

So the aggregate is a pure function of the band's fragment SET, not of the
order the hardware happens to visit it in -- the property
``DESIGN_sheet_resolve.md`` §2.2 demands.

``sheet_conflict_rank`` needs no such argument, and that is the point of its
shape: both arms are integer and visit the stream in the SAME order -- the
kernel's serial band walk reads fragments exactly as the torch cumsums do --
so agreement is by construction, not by an order-independence proof.

The area sum is the one exception, and it is handled the way the torch code
handled it rather than by a fixed tree: an f64 atomic add, rounded to f32 by
the caller. That is not order-independent in principle, only far enough below
an f32 ulp that the cast absorbs it -- which is measured, not assumed. Two
things pin it: the kernel agrees with the torch ``scatter_add_`` it replaces
BITWISE at a 4K frame's shapes, and six consecutive runs of the kernel agree
with each other bitwise. Do not narrow it to f32: a real frame's sheets are
81% one fragment and 17% two (both order-independent whatever the width), but
the remaining 1.6% run to eleven, and ``sheet_cov`` feeds thresholds.

``benchmarks/_sheet_kernel_check.py`` is the parity harness; every kernel is
gated (``RASTER_FUSED_GATHER``, ``SHEET_MASK_KERNEL``, ``SHEET_RANK_KERNEL``)
so the torch passes stay runnable as the A/B arm.
"""

import taichi as ti

from algan.rendering.raytracing.raster_taichi import _AA_NUM_SAMPLES
from algan.rendering.taichi_runtime import init_taichi

init_taichi()


@ti.kernel
def gather_fragment_arrays(
    idx: ti.types.ndarray(),  # [M] i64 -- source row of each output row
    n: ti.i32,  # M
    in_key: ti.types.ndarray(),  # [N] i64
    in_ref: ti.types.ndarray(),  # [N] i32
    in_ab: ti.types.ndarray(),  # [N, 2] f32
    in_cov: ti.types.ndarray(),  # [N] f32
    in_msk: ti.types.ndarray(),  # [N] i32
    in_opq: ti.types.ndarray(),  # [N] u8 (a bool tensor viewed as bytes)
    out_key: ti.types.ndarray(),  # [M] i64
    out_ref: ti.types.ndarray(),  # [M] i32
    out_ab: ti.types.ndarray(),  # [M, 2] f32
    out_cov: ti.types.ndarray(),  # [M] f32
    out_msk: ti.types.ndarray(),  # [M] i32
    out_opq: ti.types.ndarray(),  # [M] u8
):
    """One pass of the six-array fragment gather ``idx`` drives.

    A fragment count never approaches 2**31 (the arrays would not fit in any
    device), so the source row narrows to i32 for the indexing.
    """
    for i in range(n):
        s = ti.cast(idx[i], ti.i32)
        out_key[i] = in_key[s]
        out_ref[i] = in_ref[s]
        out_ab[i, 0] = in_ab[s, 0]
        out_ab[i, 1] = in_ab[s, 1]
        out_cov[i] = in_cov[s]
        out_msk[i] = in_msk[s]
        out_opq[i] = in_opq[s]


@ti.kernel
def sheet_band_reduce(
    band: ti.types.ndarray(),  # [n] i64 -- band/sheet index of each fragment
    msk: ti.types.ndarray(),  # [n] i32 -- fragment mask word
    cov: ti.types.ndarray(),  # [n] f32 -- fragment exact area
    n: ti.i32,
    mask_all: ti.i32,  # _AA_MASK_ALL
    sliver_bit: ti.i32,  # _AA_SLIVER_BIT
    area: ti.types.ndarray(),  # [nb] f64, PRE-ZEROED
    union: ti.types.ndarray(),  # [nb] i32, PRE-ZEROED
    dup: ti.types.ndarray(),  # [nb] i32, PRE-ZEROED
    sliver: ti.types.ndarray(),  # [nb] i32 PRE-ZEROED, or a [1] dummy
    want_sliver: ti.template(),  # compile-time: is `sliver` real?
):
    """One pass: per-band exact area, sample union, doubly-claimed lanes, sliver.

    ``dup`` ends up holding exactly the lanes two or more of the band's
    fragments claimed -- see the module docstring for why the atomic order
    cannot change that -- so the caller's fusion flag is ``dup != 0``.

    ``area`` is the one FLOAT reduction here, and it accumulates in f64 for
    the reason the torch version did (§6.6.4): a float32 atomic add is not
    order-reproducible, and this value feeds thresholds. Widening happens in
    a register off an f32 read, which is the point of folding it in -- the
    torch form needed an f64 copy of the whole fragment stream just to give
    ``scatter_add_`` matching dtypes, 29 MB on a 4K frame for a value that is
    read once.
    """
    for i in range(n):
        b = ti.cast(band[i], ti.i32)
        word = msk[i]
        bits = word & mask_all
        shared = ti.atomic_or(union[b], bits) & bits
        if shared != 0:
            ti.atomic_or(dup[b], shared)
        ti.atomic_add(area[b], ti.cast(cov[i], ti.f64))
        if ti.static(want_sliver):
            if (word & sliver_bit) != 0:
                ti.atomic_or(sliver[b], 1)


@ti.kernel
def mask_popcount(
    bits: ti.types.ndarray(),  # [n] i64 -- mask words
    n: ti.i32,
    pop: ti.types.ndarray(),  # [n] i32
):
    """Set sample bits per word, all lanes in registers in one pass."""
    for i in range(n):
        word = bits[i]
        count = 0
        for b in ti.static(range(_AA_NUM_SAMPLES)):
            count += ti.cast((word >> b) & 1, ti.i32)
        pop[i] = count


@ti.kernel
def sheet_conflict_rank(
    band_start: ti.types.ndarray(),  # [n] u8 (a bool tensor viewed as bytes)
    order: ti.types.ndarray(),  # [n] i64 -- emission->sorted permutation
    msk: ti.types.ndarray(),  # [N] i32 -- UNSORTED fragment mask words
    n: ti.i32,
    mask_all: ti.i32,  # _AA_MASK_ALL
    rank: ti.types.ndarray(),  # [n] i32 -- output, UNCLAMPED
):
    """One pass: each sorted fragment's conflict rank within its band.

    ``rank[j]`` is the largest, over the sample lanes sorted fragment ``j``
    claims, of the number of EARLIER fragments of its band claiming that same
    lane -- the quantity ``sheets.compact_sheets``' eight-cumsum torch loop
    computed lane by lane as (global exclusive prefix sum) - (prefix at the
    band's first index). Bands are contiguous runs beginning at a set
    ``band_start``, so the thread whose own flag is set walks its band forward
    once with the eight per-lane counters in registers; each counter is read
    before the fragment's own increment (the exclusive prefix), and no lane's
    counter is touched by any other band's walk. The caller owns the
    ``max=15`` clamp; this returns the raw counts.

    Row 0 always starts a band, whether or not its flag is set -- the torch
    arm's cummax gives any leading run of clear flags band-first 0, which is
    that same band -- and that is what makes every output row written exactly
    once, which is what licenses the caller's ``torch.empty`` output.

    A fragment count never approaches 2**31 (the arrays would not fit in any
    device), so the source row narrows to i32 for the indexing.
    """
    for i in range(n):
        if band_start[i] != 0 or i == 0:
            cnt = ti.Vector([0] * _AA_NUM_SAMPLES)
            j = i
            while j < n:
                if j > i and band_start[j] != 0:
                    break
                bits = msk[ti.cast(order[j], ti.i32)] & mask_all
                r = 0
                for b in ti.static(range(_AA_NUM_SAMPLES)):
                    if ((bits >> b) & 1) != 0:
                        r = ti.max(r, cnt[b])
                        cnt[b] += 1
                rank[j] = r
                j += 1
