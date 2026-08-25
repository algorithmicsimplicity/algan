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

The one-mesh reduction (``one_mesh_pixel_reduce``) carries the same-shaped
float contract one stage earlier in the pipeline: its ``front``/``back``
per-pixel coverage sums accumulate in f64 registers and are rounded through
f32 by the caller, exactly as the torch ``scatter_add_`` pair it replaces was
-- and for the same reason (a ceiling that wobbles in its low bits flips
borderline fragments in and out of being clipped). Unlike the area sum it
accumulates SERIALLY, one thread per pixel walking the pixel's own CSR run,
so it is order-reproducible run to run by construction; agreement with the
torch arm is again by measurement, at 4K shapes, bitwise after the round.

``benchmarks/_sheet_kernel_check.py`` is the parity harness; every kernel is
gated (``RASTER_FUSED_GATHER``, ``SHEET_MASK_KERNEL``, ``SHEET_RANK_KERNEL``,
``RASTER_OPAQUE_TRUNC_KERNEL``, ``SHEET_ONE_MESH_KERNEL``,
``SHEET_SAMPLE_DEPTH_KERNEL``) so the torch passes stay runnable as the A/B
arm.
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


@ti.kernel
def opaque_prefix_keep(
    opaque: ti.types.ndarray(),  # [n] u8 (a bool tensor viewed as bytes)
    counts: ti.types.ndarray(),  # [num_cov] i64 -- fragments per covered pixel
    starts: ti.types.ndarray(),  # [num_cov] i64 -- CSR row start per pixel
    num_cov: ti.i32,
    keep: ti.types.ndarray(),  # [n] u8 OUT -- 1 where the fragment is kept
):
    """One pass: the opaque-prefix truncation's keep mask.

    A pixel keeps the prefix through its FIRST proven-opaque fragment, so
    ``keep[j]`` holds exactly when no opaque fragment of j's pixel lies
    strictly before j. One thread per covered pixel walks its CSR run twice --
    once to find that first opaque fragment, once to write the flags -- which
    replaces the torch chain (an arange, a repeat_interleave over the whole
    stream, a nonzero + index_select + scatter_reduce amin, and two full-length
    elementwise passes) with one launch and no [n] intermediate but the output.

    Integer flags compared identically in both arms, so agreement is exact by
    construction.
    """
    for p in range(num_cov):
        s = ti.cast(starts[p], ti.i32)
        e = s + ti.cast(counts[p], ti.i32)
        ke = e - 1
        j = s
        while j < e:
            if opaque[j] != 0:
                ke = j
                break
            j += 1
        for jj in range(s, e):
            keep[jj] = 1 if jj <= ke else 0


@ti.kernel
def one_mesh_pixel_reduce(
    key: ti.types.ndarray(),  # [n] i64 -- sorted fragment keys
    ref: ti.types.ndarray(),  # [n] i32 -- primitive refs (<0: circuit)
    mat_opaque: ti.types.ndarray(),  # [n] u8 (a bool tensor viewed as bytes)
    msk: ti.types.ndarray(),  # [n] i32 -- mask words (backface bit read here)
    cov: ti.types.ndarray(),  # [n] f32 -- exact areas
    counts: ti.types.ndarray(),  # [num_cov] i64 -- fragments per covered pixel
    starts: ti.types.ndarray(),  # [num_cov] i64
    num_cov: ti.i32,
    ppf: ti.i32,  # pixels per frame
    time_start: ti.i32,
    obj_rows: ti.i32,  # tri_obj.shape[0]
    backface_bit: ti.i32,  # _AA_BACKFACE_BIT
    tri_obj: ti.types.ndarray(),  # [obj_rows, N] -- fragment row -> surface id
    lo: ti.types.ndarray(),  # [num_cov] i32 OUT, PRE-FILLED with 2**31 - 1
    hi: ti.types.ndarray(),  # [num_cov] i32 OUT, PRE-FILLED with -1
    front: ti.types.ndarray(),  # [num_cov] f64 OUT, PRE-ZEROED
    back: ti.types.ndarray(),  # [num_cov] f64 OUT, PRE-ZEROED
):
    """One pass: per-pixel surface-id spread + facing-split coverage sums.

    The torch block this replaces scattered four reductions over a
    ``repeat_interleave`` segment map (amin/amax of usable surface ids, then
    two f64 ``scatter_add_``s splitting coverage by backface bit). Here one
    thread per covered pixel walks its own CSR run once, keeping all four
    aggregates in registers -- so the segment map never exists and the float
    sums accumulate in a fixed serial order (the torch atomics had none).

    ``lo``/``hi`` are integer min/max, exact under any visit order; their fill
    values differ from the torch arm's (i32 max vs 1<<40) only where a pixel
    would have NO fragment, which cannot happen -- every covered pixel holds
    at least one, and every usable surface id is < 2**31 - 1 -- so the values
    are identical wherever they can be observed.

    ``front``/``back`` carry the module-docstring float contract: f64
    accumulation rounded through f32 by the caller, bitwise-equal to the torch
    ``scatter_add_`` pair by measurement rather than by construction.
    """
    for p in range(num_cov):
        s = ti.cast(starts[p], ti.i32)
        e = s + ti.cast(counts[p], ti.i32)
        lo_v = 2147483647
        hi_v = -1
        # f64 explicitly: a bare 0.0 infers f32 and the sum would silently
        # narrow (Taichi warns "atomic add may lose precision").
        fr = ti.f64(0.0)
        bk = ti.f64(0.0)
        for j in range(s, e):
            r = ref[j]
            sid = -1
            if r >= 0 and mat_opaque[j] != 0:
                row = ((key[j] >> 32) // ppf + time_start) % obj_rows
                sid = ti.cast(tri_obj[row, r], ti.i32)
            if sid < lo_v:
                lo_v = sid
            if sid > hi_v:
                hi_v = sid
            c = ti.cast(cov[j], ti.f64)
            if (msk[j] & backface_bit) != 0:
                bk += c
            else:
                fr += c
        lo[p] = lo_v
        hi[p] = hi_v
        front[p] = fr
        back[p] = bk


@ti.kernel
def one_mesh_pixel_apply(
    counts: ti.types.ndarray(),  # [num_cov] i64
    starts: ti.types.ndarray(),  # [num_cov] i64
    num_cov: ti.i32,
    one_mesh: ti.types.ndarray(),  # [num_cov] u8 (a bool viewed as bytes)
    cap_pix: ti.types.ndarray(),  # [num_cov] f32 -- per-pixel ceiling
    msk: ti.types.ndarray(),  # [n] i32 INOUT -- ONE_MESH bit folded in place
    cap_s: ti.types.ndarray(),  # [n] f32 OUT -- 2.0 sentinel / ceiling
    one_mesh_bit: ti.i32,  # _AA_ONE_MESH_BIT
):
    """One pass: fold the one-mesh flag and per-fragment caps into the stream.

    Every row of ``msk``/``cap_s`` is written by exactly one thread (its own
    pixel's walk, over the CSR ranges the counts sum to exactly once), so
    plain read-modify-write needs no atomics. Values match the torch arm
    exactly: flagged pixels take ``cap_pix``, everything else keeps the 2.0
    no-ceiling sentinel.
    """
    for p in range(num_cov):
        s = ti.cast(starts[p], ti.i32)
        e = s + ti.cast(counts[p], ti.i32)
        om = one_mesh[p] != 0
        cp = cap_pix[p]
        for j in range(s, e):
            if om:
                msk[j] = msk[j] | one_mesh_bit
                cap_s[j] = cp
            else:
                cap_s[j] = 2.0


@ti.kernel
def sheet_lane_first_owner(
    band: ti.types.ndarray(),  # [n] i64 -- sheet index of each SORTED fragment
    msk: ti.types.ndarray(),  # [n] i32 -- mask words
    n: ti.i32,
    mask_all: ti.i32,  # _AA_MASK_ALL
    first_lane: ti.types.ndarray(),  # [nb * _AA_NUM_SAMPLES] i32, PRE-FILLED n
):
    """One pass: per (sheet, sample lane), the earliest owner's sorted index.

    The SHEET_SAMPLE_DEPTH lane loop asked, eight times over the stream, for
    the minimum sorted position among the fragments of each sheet owning one
    sample lane -- an amin scatter per lane over a full-length ``where`` copy.
    Here one thread per fragment does all eight lanes' atomic mins into one
    pre-initialised table; integer min is order-independent, so the result is
    exactly the torch loop's whatever order the threads land in. Lanes a
    sheet does not own keep the fill value, which the caller reads as "no
    owner" exactly as it read the torch arm's sentinel.
    """
    for i in range(n):
        b = ti.cast(band[i], ti.i32)
        bits = msk[i] & mask_all
        base = b * _AA_NUM_SAMPLES
        for lane in ti.static(range(_AA_NUM_SAMPLES)):
            if ((bits >> lane) & 1) != 0:
                ti.atomic_min(first_lane[base + lane], i)
