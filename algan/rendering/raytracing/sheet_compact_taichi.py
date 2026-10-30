"""Kernels for the fragment/sheet compaction's multi-pass host loops.

``DESIGN_sheet_resolve.md`` §10.4 describes the compaction contract. The host
orchestrates PyTorch operations and optional fused kernels between emission and
resolve. The historical measurements below explain why these passes were fused;
they are not current end-to-end performance predictions:

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
The mask, rank and gather operations are integer arithmetic or copies of float
bits, so floating-point reassociation cannot change them. Area/coverage
reductions are separate floating-point operations, discussed below. The
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

The one exception is not a loophole in that rule, it is the absence of the
alternative: **Metal has no f64 at all**, so MPS-friendly mode passes
``ti.f32`` for every ``acc_t`` here and accepts what the paragraph above says
it costs (``mps_compat``; ``DESIGN_mps_support.md`` §1.2). It is a
``ti.template()`` argument rather than a ``ti.static`` gate so that Taichi
compiles a variant per width -- a static gate resolves once, at the first
compile, and the second arm would silently reuse the first arm's code. The
same argument carries ``idx_t`` for ``band_stats_reduce``'s atomics, which
Metal aborts on at i64 and which lose nothing at i32 because every value they
reduce is a position or a count.

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
gated (``raster_fused_gather``, ``sheet_mask_kernel``, ``sheet_rank_kernel``,
``raster_opaque_trunc_kernel``, ``sheet_one_mesh_kernel``,
``sheet_sample_depth_kernel``, ``sheet_shell_ceiling_kernel``,
``sheet_band_stats_kernel``, ``raster_pair_expand_kernel``) so the torch
passes stay runnable as the A/B arm.
"""

from algan.rendering.raytracing.bezier_acceleration import (
    BEZIER_ACCEL_HEADER_SIZE,
    BEZIER_EDGE_END,
    BEZIER_EDGE_START,
)
from algan.rendering.raytracing.raster_taichi import _AA_FILTER_RADIUS, _AA_NUM_SAMPLES
from algan.rendering.raytracing.raytrace_kernels_taichi import (
    _M_BASIS_U,
    _M_BASIS_V,
    _M_BORDER_CENTERED,
    _M_BORDER_W,
    _M_CENTER,
    _M_FILLED,
    CameraProjection,
    _axis_cos,
    _circuit_query_radius,
    _generate_ray,
    _projected,
)
from algan.taichi_compat import ti

# Taichi is NOT initialized here. The arch depends on
# ``SETTINGS.computing.render_device``, which a script may still change at
# this point, so the program is created at the start of a render by
# ``taichi_runtime.ensure_taichi_for_render()``. Defining a kernel needs no
# program -- ``@ti.kernel`` only registers it; materialization at first launch
# is what needs one, and by then a render has selected the arch. Anything that
# launches these kernels outside a render (a benchmark, a unit test) must call
# ``init_taichi()`` itself.


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

    Source and destination row indices must fit signed int32. The kernel
    narrows source indices for indexing; callers must respect that bound.
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
def split_mixed_pixels(
    run_offsets: ti.types.ndarray(),  # [nc + 1] -- the stream's CSR
    frag_ref: ti.types.ndarray(),  # [n] i32 -- negative for a circuit fragment
    nc: ti.i32,
    mixed: ti.types.ndarray(),  # [nc] u8 OUT -- 1 where the pixel holds a triangle
    tally: ti.types.ndarray(),  # [nc, 2] i32 OUT -- (mixed, its fragments if mixed)
):
    """``_split_stroke_pixels``' classification: which covered pixels hold a
    triangle fragment, and how many fragments each such pixel brings.

    One thread per covered pixel scans its run until the first triangle
    fragment. ``tally``'s two columns are what the caller's single ``cumsum``
    turns into each pixel's triangle-pixel rank and its run's end in the
    triangle pixels' stream -- and, in its last row, the two totals.
    """
    for p in range(nc):
        s = ti.cast(run_offsets[p], ti.i32)
        e = ti.cast(run_offsets[p + 1], ti.i32)
        hit = 0
        i = s
        while i < e:
            if frag_ref[i] >= 0:
                hit = 1
                break
            i += 1
        mixed[p] = ti.cast(hit, ti.u8)
        tally[p, 0] = hit
        tally[p, 1] = hit * (e - s)


@ti.kernel
def split_gather_pixels(
    mixed: ti.types.ndarray(),  # [nc] u8 -- ``split_mixed_pixels``' flags
    ends: ti.types.ndarray(),  # [nc, 2] i32 -- ``tally``'s inclusive prefix
    run_offsets: ti.types.ndarray(),  # [nc + 1] -- the stream's CSR
    nc: ti.i32,
    in_key: ti.types.ndarray(),  # [n] i64
    in_ref: ti.types.ndarray(),  # [n] i32
    in_ab: ti.types.ndarray(),  # [n, 2] f32
    in_cov: ti.types.ndarray(),  # [n] f32
    in_msk: ti.types.ndarray(),  # [n] i32
    in_cap: ti.types.ndarray(),  # [n] f32
    out_key: ti.types.ndarray(),  # [m_frags] i64
    out_ref: ti.types.ndarray(),  # [m_frags] i32
    out_ab: ti.types.ndarray(),  # [m_frags, 2] f32
    out_cov: ti.types.ndarray(),  # [m_frags] f32
    out_msk: ti.types.ndarray(),  # [m_frags] i32
    out_cap: ti.types.ndarray(),  # [m_frags] f32
):
    """``_split_stroke_pixels``' triangle-pixel stream: the six fragment arrays
    of every covered pixel holding a triangle fragment, copied run by run.

    One thread per covered pixel; a stroke pixel copies nothing, and a
    triangle pixel's run lands so that it ends at ``ends[p, 1]``. Copies of
    the stored bits, so exact. Replaces a per-fragment pixel map, its gather,
    a ``nonzero`` and six ``index_select`` s.
    """
    for p in range(nc):
        if mixed[p] != 0:
            s = ti.cast(run_offsets[p], ti.i32)
            e = ti.cast(run_offsets[p + 1], ti.i32)
            d = ends[p, 1] - e
            for i in range(s, e):
                out_key[d + i] = in_key[i]
                out_ref[d + i] = in_ref[i]
                out_ab[d + i, 0] = in_ab[i, 0]
                out_ab[d + i, 1] = in_ab[i, 1]
                out_cov[d + i] = in_cov[i]
                out_msk[d + i] = in_msk[i]
                out_cap[d + i] = in_cap[i]


@ti.kernel
def split_merge_sheets(
    mixed: ti.types.ndarray(),  # [nc] u8 -- ``split_mixed_pixels``' flags
    ends: ti.types.ndarray(),  # [nc, 2] i32 -- ``tally``'s inclusive prefix
    run_offsets: ti.types.ndarray(),  # [nc + 1] -- the stream's CSR
    sub_frags: ti.types.ndarray(),  # [m + 1] i32 -- the triangle pixels' CSR
    sub_sheets: ti.types.ndarray(),  # [m + 1] i64 -- their sheets' CSR
    nc: ti.i32,
    mask_all: ti.i32,  # _AA_MASK_ALL
    sliver_bit: ti.i32,  # _AA_SLIVER_BIT
    frag_key: ti.types.ndarray(),  # [n] i64
    frag_ref: ti.types.ndarray(),  # [n] i32
    frag_ab: ti.types.ndarray(),  # [n, 2] f32
    frag_cov: ti.types.ndarray(),  # [n] f32
    frag_msk: ti.types.ndarray(),  # [n] i32
    frag_cap: ti.types.ndarray(),  # [n] f32
    sub_key: ti.types.ndarray(),  # [sub_nb] i64
    sub_ref: ti.types.ndarray(),  # [sub_nb] i32
    sub_ab: ti.types.ndarray(),  # [sub_nb, 2] f32
    sub_wgt: ti.types.ndarray(),  # [sub_nb] f32
    sub_wmsk: ti.types.ndarray(),  # [sub_nb] i32
    sub_cap: ti.types.ndarray(),  # [sub_nb] f32
    out_key: ti.types.ndarray(),  # [total] i64
    out_ref: ti.types.ndarray(),  # [total] i32
    out_ab: ti.types.ndarray(),  # [total, 2] f32
    out_wgt: ti.types.ndarray(),  # [total] f32
    out_wmsk: ti.types.ndarray(),  # [total] i32
    out_cap: ti.types.ndarray(),  # [total] f32
    sheet_offsets: ti.types.ndarray(),  # [nc + 1] i64 OUT -- all but the last
):
    """``_split_stroke_pixels``' six resolve fields, interleaved back into
    pixel order in one pass over the covered pixels, with the output's CSR.

    A pixel's sheets start where its fragments did, moved by how many more
    sheets than fragments the triangle pixels before it made: with ``k`` of
    them, ``sub_sheets[k] - sub_frags[k]``. A triangle pixel then copies its
    sheets from the general path's result; a stroke pixel writes its
    fragments one for one as ``_stroke_sheets`` does: the area as
    ``(cov + 0.0).clamp_min_(0.0)`` and the mask with the sliver bit forced
    on for an empty union. The area rule is spelled on the bits, so fast_math
    has no float arithmetic to rewrite: a set sign bit (-0.0, a negative,
    -inf) becomes +0.0, and a NaN comes back quieted, as the CPU's add
    returns it. That is the torch arm bit for bit on the CPU, with two
    exceptions no real area reaches: a GPU's add returns its one canonical
    NaN, and under flush-to-zero (which Taichi's initialization turns on for
    the CPU thread) the torch add turns a denormal into a zero of its sign,
    where this keeps a positive one and zeroes a negative one like any other.
    """
    for p in range(nc):
        hit = ti.cast(mixed[p], ti.i32)
        k = ends[p, 0] - hit
        s = ti.cast(run_offsets[p], ti.i32)
        e = ti.cast(run_offsets[p + 1], ti.i32)
        d = s + ti.cast(sub_sheets[k], ti.i32) - sub_frags[k]
        sheet_offsets[p] = d
        if hit != 0:
            ss = ti.cast(sub_sheets[k], ti.i32)
            se = ti.cast(sub_sheets[k + 1], ti.i32)
            for i in range(ss, se):
                o = d + i - ss
                out_key[o] = sub_key[i]
                out_ref[o] = sub_ref[i]
                out_ab[o, 0] = sub_ab[i, 0]
                out_ab[o, 1] = sub_ab[i, 1]
                out_wgt[o] = sub_wgt[i]
                out_wmsk[o] = sub_wmsk[i]
                out_cap[o] = sub_cap[i]
        else:
            for i in range(s, e):
                o = d + i - s
                out_key[o] = frag_key[i]
                out_ref[o] = frag_ref[i]
                out_ab[o, 0] = frag_ab[i, 0]
                out_ab[o, 1] = frag_ab[i, 1]
                bits = ti.bit_cast(frag_cov[i], ti.u32)
                if (bits & ti.u32(0x7FFFFFFF)) > ti.u32(0x7F800000):
                    bits = bits | ti.u32(0x00400000)
                elif (bits >> 31) != 0:
                    bits = ti.u32(0)
                out_wgt[o] = ti.bit_cast(bits, ti.f32)
                m = frag_msk[i]
                if (m & mask_all) == 0:
                    m = m | sliver_bit
                out_wmsk[o] = m
                out_cap[o] = frag_cap[i]


@ti.kernel
def sheet_band_reduce(
    band: ti.types.ndarray(),  # [n] i64 -- band/sheet index of each fragment
    msk: ti.types.ndarray(),  # [n] i32 -- fragment mask word
    cov: ti.types.ndarray(),  # [n] f32 -- fragment exact area
    n: ti.i32,
    mask_all: ti.i32,  # _AA_MASK_ALL
    sliver_bit: ti.i32,  # _AA_SLIVER_BIT
    area: ti.types.ndarray(),  # [nb] `acc_t`, PRE-ZEROED
    union: ti.types.ndarray(),  # [nb] i32, PRE-ZEROED
    dup: ti.types.ndarray(),  # [nb] i32, PRE-ZEROED
    sliver: ti.types.ndarray(),  # [nb] i32 PRE-ZEROED, or a [1] dummy
    want_sliver: ti.template(),  # compile-time: is `sliver` real?
    acc_t: ti.template(),  # compile-time: `area`'s element type
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

    ``acc_t`` is that width, and it is ``ti.f32`` in MPS-friendly mode because
    Metal has no f64 to widen into (``mps_compat``). It must match ``area``'s
    own dtype: a narrower array with a wider add warns and silently rounds
    every atomic.
    """
    for i in range(n):
        b = ti.cast(band[i], ti.i32)
        word = msk[i]
        bits = word & mask_all
        shared = ti.atomic_or(union[b], bits) & bits
        if shared != 0:
            ti.atomic_or(dup[b], shared)
        ti.atomic_add(area[b], ti.cast(cov[i], acc_t))
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

    Source and destination row indices must fit signed int32. The kernel
    narrows source indices for indexing; callers must respect that bound.
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
            # Cast spelled out because ``keep`` is u8 and the conditional is
            # typed i32, which Taichi warns about on every cold compile. The
            # values are 0 and 1, so nothing was ever lost -- but this is the
            # renderer's only implicit narrowing store, every other u8 write
            # being an explicit ``ti.cast``, and a warning that fires on a
            # harmless line is how a real one later reads as more of the same.
            keep[jj] = ti.cast(1 if jj <= ke else 0, ti.u8)


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
    front: ti.types.ndarray(),  # [num_cov] `acc_t` OUT, PRE-ZEROED
    back: ti.types.ndarray(),  # [num_cov] `acc_t` OUT, PRE-ZEROED
    acc_t: ti.template(),  # compile-time: `front`/`back`'s element type
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
    ``scatter_add_`` pair by measurement rather than by construction. ``acc_t``
    is that width and must match the two arrays' dtype; MPS-friendly mode
    passes ``ti.f32``, which makes the caller's round a no-op and gives up the
    contract, because Metal has no f64 to keep it in (``mps_compat``).
    """
    for p in range(num_cov):
        s = ti.cast(starts[p], ti.i32)
        e = s + ti.cast(counts[p], ti.i32)
        lo_v = 2147483647
        hi_v = -1
        # Typed explicitly: a bare 0.0 infers f32 and the f64 sum would
        # silently narrow (Taichi warns "atomic add may lose precision").
        fr = ti.cast(0.0, acc_t)
        bk = ti.cast(0.0, acc_t)
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
            c = ti.cast(cov[j], acc_t)
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
    band: ti.types.ndarray(),  # [n] -- table row of each SORTED fragment, < 0 to skip
    msk: ti.types.ndarray(),  # [n] i32 -- mask words
    n: ti.i32,
    mask_all: ti.i32,  # _AA_MASK_ALL
    first_lane: ti.types.ndarray(),  # [rows * _AA_NUM_SAMPLES] i32, PRE-FILLED n
):
    """One pass: per (sheet, sample lane), the earliest owner's sorted index.

    The sheet_sample_depth lane loop asked, eight times over the stream, for
    the minimum sorted position among the fragments of each sheet owning one
    sample lane -- an amin scatter per lane over a full-length ``where`` copy.
    Here one thread per fragment does all eight lanes' atomic mins into one
    pre-initialised table; integer min is order-independent, so the result is
    exactly the torch loop's whatever order the threads land in. Lanes a
    sheet does not own keep the fill value, which the caller reads as "no
    owner" exactly as it read the torch arm's sentinel. A fragment whose row
    is negative belongs to a sheet the table does not hold, and is skipped.
    """
    for i in range(n):
        b = ti.cast(band[i], ti.i32)
        if b >= 0:
            bits = msk[i] & mask_all
            base = b * _AA_NUM_SAMPLES
            for lane in ti.static(range(_AA_NUM_SAMPLES)):
                if ((bits >> lane) & 1) != 0:
                    ti.atomic_min(first_lane[base + lane], i)


#: The shell ceiling's spend is summed in this fixed point: a fragment's
#: coverage ``c`` (its exact area, in pixels) counts as the integer
#: ``int(c * SHELL_SPEND_ONE)``, truncated toward zero. See
#: ``solid_shell_ceiling``.
SHELL_SPEND_ONE = 4294967296.0  # 2**32
SHELL_SPEND_QUANTUM = 1.0 / SHELL_SPEND_ONE


@ti.kernel
def solid_shell_ceiling(
    key: ti.types.ndarray(),  # [n] i64 -- (pixel, surface) segment keys, sorted order
    o2: ti.types.ndarray(),  # [n] i64 -- permutation key order -> depth order
    back: ti.types.ndarray(),  # [n] u8 (a bool tensor viewed as bytes)
    n: ti.i32,
    cov: ti.types.ndarray(),  # [n] f32 INOUT -- clamped in place
    acc_t: ti.template(),  # compile-time: the accumulator width
):
    """One pass: the solid-shell opacity ceiling, applied to the fragments.

    The torch block this replaced walked its depth-ordered segments three
    more times after the sort that orders them: a ``nonzero`` + ``scatter_``
    for the segment starts, two facing-split f64 ``scatter_add_``s routed
    through a whole-stream segment map, and a clone + ``index_copy_`` to write
    the clamped areas back. Here one thread per segment -- detected in-kernel
    as a change of ``key`` along ``o2``, so neither the segment map nor the
    start indices exist -- walks its run twice with everything else in
    registers: once to accumulate the two shells' full-coverage sums (the
    cap), once to spend the allowance in depth order and write each
    fragment's scaled coverage back through ``cov[o2[j]]``.

    Exactness, in two parts. The facing-split sums are plain f64 adds of a
    segment's own values: the torch arm's atomic ``scatter_add_`` had no
    summation order at all and the kernel adds serially in stream order, so
    agreement with the torch arm is the same measure-not-assume contract as
    ``one_mesh_pixel_reduce``'s (verified bitwise at 4K shapes AND on a real
    captured stream).

    The SPEND -- what each fragment's in-segment predecessors have already
    consumed -- is integer arithmetic, so it needs no such argument. Each
    area counts as ``int(c * SHELL_SPEND_ONE)`` (Q32 fixed point, truncated),
    the walk adds those integers, and the total converts back exactly
    (``spent_q * SHELL_SPEND_QUANTUM``: below 2**53, times a power of two).
    Integer addition is associative, so the torch arm reaches the same
    integer by differencing a whole-stream ``cumsum`` at the segment's base,
    and so does any other grouping of the stream: the clamp of one segment
    is a function of that segment's fragments alone, never of how many
    fragments, pixels or frames precede it in the chunk. That is what lets
    ``compact_sheets`` compact a stream's triangle pixels apart from its
    stroke pixels (``_split_stroke_pixels``) whatever the scene declares.
    (The spend used to be a difference of two rows of a GLOBAL f64 cub scan,
    which made every fragment's low bits depend on the whole chunk and kept
    any stream with a closed solid in it off that split -- about half the
    compaction's time on such scenes.) The quantum, 2.3e-10, is below what
    that difference rounded to once the prefix passed ~1e6, and far below the
    f32 output's spacing; the bound is the stream's total coverage under
    2**31, which an arena-limited fragment count never approaches.

    Every remaining step is the same arithmetic on the same values in the
    same order in both arms: clamp/div/clamp chain, the f64 -> f32 -> f64
    rounding of the cap, and the ``1e-12`` denominator floor -- which the
    torch side clamps INTO ITS OWN COPY, so the closing multiply writes the
    floored value, not the raw area. Each output slot is written by exactly
    one walk because ``o2`` is a permutation, which is what made the torch
    side's clone + ``index_copy_`` equivalent to in-place.

    A fragment count never approaches 2**31, so rows narrow to i32 for
    indexing.

    ``acc_t`` is the accumulator width. MPS-friendly mode passes ``ti.f32``,
    because Metal has no f64 (``mps_compat``), and there the exactness
    argument above lapses: the cap's round through f32 is a no-op and nothing
    can be bitwise anyway. That arm keeps the float spend Metal has always
    run: it adds each segment's areas up in the same serial walk that spends
    them, which rounds at the scale of one pixel's coverage (4e-7 off the f64
    arm on a synthetic 3.2M-fragment stream) and, like the fixed point, never
    reads another segment. The torch arm does the same with a segmented scan
    (``sheets._segment_exclusive_prefix``).
    """
    for i in range(n):
        ii = ti.cast(o2[i], ti.i32)
        ki = key[ii]
        if i == 0 or ki != key[ti.cast(o2[i - 1], ti.i32)]:
            front = ti.cast(0.0, acc_t)
            bk = ti.cast(0.0, acc_t)
            j = i
            while j < n:
                jj = ti.cast(o2[j], ti.i32)
                if j > i and key[jj] != key[ti.cast(o2[j - 1], ti.i32)]:
                    break
                c = ti.cast(cov[jj], acc_t)
                if back[jj] != 0:
                    bk += c
                else:
                    front += c
                j += 1
            # The cap rounds through f32 (ss6.6.4): a ceiling that wobbles in
            # its low bits flips borderline fragments in and out of being
            # clipped. The f64 value came from an f32 array, but the SUM did
            # not -- only the round makes it well-defined.
            cap = ti.cast(ti.cast(ti.max(front, bk), ti.f32), acc_t)
            # The spend so far: Q32 integers at f64 (see the docstring), the
            # plain running sum in the f32 arm. Only one of them is used.
            spent_q = ti.cast(0, ti.i64)
            spent = ti.cast(0.0, acc_t)
            j = i
            while j < n:
                jj = ti.cast(o2[j], ti.i32)
                if j > i and key[jj] != key[ti.cast(o2[j - 1], ti.i32)]:
                    break
                # ``denom`` is written back, not ``c``: the torch arm's
                # ``div_(c2.clamp_min_(1e-12))`` clamps ITS OWN COPY in place,
                # so the closing multiply reads the floored value. Identical
                # wherever an area exceeds the floor, which is everywhere the
                # emission produces; below it both sides write ~zero ink, and
                # byte identity still demands they write the SAME zero.
                c = ti.cast(cov[jj], acc_t)
                denom = ti.max(c, ti.cast(1e-12, acc_t))
                # Declared outside the static branches: a name first bound
                # inside one is out of scope after it.
                scale = ti.cast(0.0, acc_t)
                if ti.static(acc_t != ti.f32):
                    scale = ti.max(
                        cap
                        - ti.cast(spent_q, acc_t)
                        * ti.cast(SHELL_SPEND_QUANTUM, acc_t),
                        ti.cast(0.0, acc_t),
                    )
                    spent_q += ti.cast(c * ti.cast(SHELL_SPEND_ONE, acc_t), ti.i64)
                else:
                    scale = ti.max(cap - spent, ti.cast(0.0, acc_t))
                    spent += c
                scale = scale / denom
                scale = ti.min(scale, ti.cast(1.0, acc_t))
                cov[jj] = ti.cast(denom * scale, ti.f32)
                j += 1


@ti.kernel
def band_stats_reduce(
    band: ti.types.ndarray(),  # [n] i64 -- sheet/band index of each sorted fragment
    msk: ti.types.ndarray(),  # [n] i32 -- mask words
    pos_o: ti.types.ndarray(),  # [n] i64 -- original stream position (the order)
    cov: ti.types.ndarray(),  # [n] f32 -- exact areas
    n: ti.i32,
    mask_all: ti.i32,  # _AA_MASK_ALL
    first_sorted: ti.types.ndarray(),  # [nb] `idx_t` OUT, PRE-FILLED n
    min_pos: ti.types.ndarray(),  # [nb] `idx_t` OUT, PRE-FILLED n
    first_sorted_p: ti.types.ndarray(),  # [nb] `idx_t` OUT, PRE-FILLED n
    min_pos_p: ti.types.ndarray(),  # [nb] `idx_t` OUT, PRE-FILLED n
    cmax: ti.types.ndarray(),  # [nb] f32 OUT, PRE-ZEROED
    nfrag: ti.types.ndarray(),  # [nb] `idx_t` OUT, PRE-ZEROED
    positioned: ti.template(),  # compile-time: sheet_positioned_depth
    idx_t: ti.template(),  # compile-time: the five OUT arrays' element type
):
    """One pass: the compaction's five per-band scatters, fused.

    The torch arms ran one ``scatter_reduce_``/``scatter_add_`` per output --
    five passes over the stream plus, for the positioned-depth restriction,
    two full-length ``where`` copies to mask it -- for values a single visit
    per fragment can all update: nearest sorted/original positions (amin),
    the same restricted to POSITIONED fragments (owning a sample bit), the
    band's largest exact area (amax), and its fragment count. Integer mins
    and maxes are exact under any atomics order; the f32 amax is exact too
    (max has no association). The caller gathers ``nearest_orig``/``sheet_pix``
    off ``first_sorted_p``/``first_sorted`` afterwards, unchanged.

    ``band_stats_rep_orig`` runs AFTER this one: it compares each fragment
    against its band's completed maximum.

    Every value reduced here is a position or a count, so ``idx_t`` may be
    ``ti.i32`` without changing an answer -- a fragment count never approaches
    2**31. MPS-friendly mode passes exactly that, because Metal's int64
    atomics abort the process rather than declining to compile (§1.2); the
    operands are cast to ``idx_t`` here rather than left to the atomic, so a
    narrowed run is a narrowed *compare*, not a rounded one. ``pos_o`` itself
    stays i64 either way -- Metal reads and packs 64-bit integers fine.
    """
    for i in range(n):
        b = ti.cast(band[i], ti.i32)
        p = ti.cast(pos_o[i], idx_t)
        ti.atomic_min(first_sorted[b], ti.cast(i, idx_t))
        ti.atomic_min(min_pos[b], p)
        if ti.static(positioned):
            if (msk[i] & mask_all) != 0:
                ti.atomic_min(first_sorted_p[b], ti.cast(i, idx_t))
                ti.atomic_min(min_pos_p[b], p)
        ti.atomic_max(cmax[b], cov[i])
        ti.atomic_add(nfrag[b], ti.cast(1, idx_t))


@ti.kernel
def band_stats_rep_orig(
    band: ti.types.ndarray(),  # [n] i64
    pos_o: ti.types.ndarray(),  # [n] i64
    cov: ti.types.ndarray(),  # [n] f32
    cmax: ti.types.ndarray(),  # [nb] f32 -- completed by band_stats_reduce
    n: ti.i32,
    rep_orig: ti.types.ndarray(),  # [nb] `idx_t` OUT, PRE-FILLED n
    idx_t: ti.template(),  # compile-time: `rep_orig`'s element type
):
    """One pass: the dominant fragment's original position, per band.

    Largest exact area wins; the earliest original position breaks ties
    (``cand_pos``/amin in the torch arm). Comparing against the completed
    ``cmax`` needs the previous kernel's finish, which is the only reason
    this is a second launch. Integer min over identical candidates, so the
    result is the torch arm's whatever order the atomics land in. ``idx_t`` is
    ``band_stats_reduce``'s, and narrows for the same reason.
    """
    for i in range(n):
        b = ti.cast(band[i], ti.i32)
        if cov[i] >= cmax[b]:
            ti.atomic_min(rep_orig[b], ti.cast(pos_o[i], idx_t))


@ti.func
def _span_row_extent(screen: ti.template(), fr, prim, y):
    """x-extent of the projected triangle ``(fr, prim)`` over the pixel row
    band ``[y, y + 1]``: ``(ok, xmin, xmax)`` in continuous screen pixels,
    ``ok == 0`` when no edge crosses the band (the triangle has no point in
    that row). Each edge is clipped to the band and its clipped endpoints'
    x taken; a triangle's x-extent over a horizontal band is attained on its
    edges, so the three clipped segments bound it exactly.
    """
    ok = 0
    xmin = 1e30
    xmax = -1e30
    ylo = ti.cast(y, ti.f32)
    yhi = ylo + 1.0
    for i in ti.static(range(3)):
        j = ti.static((i + 1) % 3)
        ax = screen[fr, prim, i]
        ay = screen[fr, prim, 3 + i]
        bx = screen[fr, prim, j]
        by = screen[fr, prim, 3 + j]
        if (ti.max(ay, by) >= ylo) and (ti.min(ay, by) <= yhi):
            t0 = 0.0
            t1 = 1.0
            dy = by - ay
            if ti.abs(dy) > 1e-12:
                ta = (ylo - ay) / dy
                tb = (yhi - ay) / dy
                t0 = ti.max(0.0, ti.min(ta, tb))
                t1 = ti.min(1.0, ti.max(ta, tb))
            xa = ax + (bx - ax) * t0
            xb = ax + (bx - ax) * t1
            xmin = ti.min(xmin, ti.min(xa, xb))
            xmax = ti.max(xmax, ti.max(xa, xb))
            ok = 1
    return ok, xmin, xmax


@ti.func
def _span_mode(mask_e, bw, bh, span_min_area, screen: ti.template(), fr, prim,
               spans: ti.template()):
    """Whether candidate ``e`` expands row by row: on, box at least
    ``span_min_area`` pixels, and a valid screen projection to follow.
    """
    use_span = 0
    if ti.static(spans):
        if (mask_e != 0) and (bw * bh >= span_min_area):
            if screen[fr, prim, 9] > 0.5:
                use_span = 1
    return use_span


@ti.func
def _span_chunks(screen: ti.template(), fr, prim, y, bx0, bx1, chunk):
    """Chunks the row ``y`` of a span candidate expands to, and the span."""
    ok, xmin, xmax = _span_row_extent(screen, fr, prim, y)
    xs = bx0
    xe = bx0 - 1
    nch = 0
    if ok == 1:
        xs = ti.max(ti.cast(ti.floor(xmin), ti.i32) - 1, bx0)
        xe = ti.min(ti.cast(ti.floor(xmax), ti.i32) + 1, bx1)
        if xe >= xs:
            nch = (xe - xs + 1 + chunk - 1) // chunk
    return nch, xs, xe


@ti.kernel
def pair_expand_count(
    mask: ti.types.ndarray(),  # [N] u8 (a bool tensor viewed as bytes), flat
    x0f: ti.types.ndarray(),  # [N] i64 -- flat x0 column
    x1f: ti.types.ndarray(),  # [N] i64
    y0f: ti.types.ndarray(),  # [N] i64
    y1f: ti.types.ndarray(),  # [N] i64
    n: ti.i32,  # N = frames * primitives
    chunk: ti.i32,  # raster_chunk
    counts: ti.types.ndarray(),  # [N] i64 OUT -- chunks per candidate, 0 elsewhere
    screen: ti.types.ndarray(),  # [F, C, >= 10] f32 projection table (spans)
    f_abs: ti.types.ndarray(),  # [Ft] -- absolute frame per window row
    ncirc: ti.i32,
    span_min_area: ti.i32,
    spans: ti.template(),
):
    """One pass: how many ``raster_chunk`` rows each candidate expands to --
    its box's chunks, or under ``spans`` (raster_span_candidates) the sum over
    its pixel rows of the chunks its projection reaches in that row.
    """
    for e in range(n):
        cnt = ti.i64(0)
        if mask[e] != 0:
            bx0 = ti.cast(x0f[e], ti.i32)
            bx1 = ti.cast(x1f[e], ti.i32)
            by0 = ti.cast(y0f[e], ti.i32)
            by1 = ti.cast(y1f[e], ti.i32)
            bw = bx1 - bx0 + 1
            bh = by1 - by0 + 1
            fr = 0
            prim = e % ncirc
            if ti.static(spans):
                fr = ti.cast(f_abs[e // ncirc] % screen.shape[0], ti.i32)
            if _span_mode(mask[e], bw, bh, span_min_area, screen, fr, prim, spans) == 1:
                for y in range(by0, by1 + 1):
                    nch, _xs, _xe = _span_chunks(screen, fr, prim, y, bx0, bx1, chunk)
                    cnt += nch
            else:
                cnt = (ti.cast(bw, ti.i64) * bh + chunk - 1) // chunk
        counts[e] = cnt


@ti.kernel
def pair_expand_write(
    mask: ti.types.ndarray(),  # [N] u8
    x0f: ti.types.ndarray(),  # [N] i64
    x1f: ti.types.ndarray(),  # [N] i64
    y0f: ti.types.ndarray(),  # [N] i64
    y1f: ti.types.ndarray(),  # [N] i64
    f_abs: ti.types.ndarray(),  # [Ft] -- absolute frame per window row
    offs: ti.types.ndarray(),  # [N] i64 -- EXCLUSIVE prefix of the counts
    n: ti.i32,  # N
    ncirc: ti.i32,
    chunk: ti.i32,
    rows: ti.types.ndarray(),  # [total, 8] i32 OUT
    screen: ti.types.ndarray(),  # [F, C, >= 10] f32 projection table (spans)
    span_min_area: ti.i32,
    spans: ti.template(),
):
    """One pass: the expanded ``(primitive, frame, bbox, offset)`` pair rows.

    One thread per candidate writes its rows from its prefix offset:
    candidates ascending in flattened order, and within one candidate its
    box's chunks ascending -- or under ``spans`` its pixel rows ascending and
    the chunks of each row's span ascending, each row's span emitted as a
    one-row-high box so the geometry kernels decode it unchanged. All eight
    columns carry the values the box expansion always did (i64 narrowed to
    i32 in the store; every value fits an i32 or the old rows already
    wrapped).
    """
    for e in range(n):
        if mask[e] != 0:
            bx0 = ti.cast(x0f[e], ti.i32)
            bx1 = ti.cast(x1f[e], ti.i32)
            by0 = ti.cast(y0f[e], ti.i32)
            by1 = ti.cast(y1f[e], ti.i32)
            bw = bx1 - bx0 + 1
            bh = by1 - by0 + 1
            prim = e % ncirc
            f = ti.cast(f_abs[e // ncirc], ti.i32)
            fr = 0
            if ti.static(spans):
                fr = ti.cast(f_abs[e // ncirc] % screen.shape[0], ti.i32)
            j = offs[e]
            if _span_mode(mask[e], bw, bh, span_min_area, screen, fr, prim, spans) == 1:
                for y in range(by0, by1 + 1):
                    nch, xs, xe = _span_chunks(screen, fr, prim, y, bx0, bx1, chunk)
                    for c in range(nch):
                        rows[j, 0] = prim
                        rows[j, 1] = f
                        rows[j, 2] = xs
                        rows[j, 3] = y
                        rows[j, 4] = xe - xs + 1
                        rows[j, 5] = 1
                        rows[j, 6] = c * chunk
                        rows[j, 7] = 0
                        j += 1
            else:
                nch = (ti.cast(bw, ti.i64) * bh + chunk - 1) // chunk
                for c in range(ti.cast(nch, ti.i32)):
                    rows[j, 0] = prim
                    rows[j, 1] = f
                    rows[j, 2] = bx0
                    rows[j, 3] = by0
                    rows[j, 4] = bw
                    rows[j, 5] = bh
                    rows[j, 6] = c * chunk
                    rows[j, 7] = 0
                    j += 1


# ---- Row-span candidates for circuits (raster_circuit_span_candidates) --------
#
# The circuit twin of the triangle spans above. A circuit is a planar outline --
# its plane in ``circuit_meta``, its flattened edges in plane coordinates in
# ``edges_2d`` -- and ``raster_bez_count`` accepts a pixel only when the point
# its centre ray hits on that plane lies within the circuit's query radius of
# an edge, or (filled) inside the outline. The radius is a fixed number of
# pixels scaled by ``pixel_size``, so it is known in pixels up front; see
# ``_circuit_reach`` for its bound on screen. A row's candidates are then the
# pixels whose centres lie within that reach of the PROJECTED edges: each edge
# is clipped to the row's centre line widened by the reach, and the clipped
# ends bound the row. A filled interior needs nothing more, since along the
# row it lies between boundary crossings, which lie on edges crossing the row.


@ti.func
def _camera_terms(f, cam_origin: ti.template(), screen_point: ti.template(),
                  pixel_basis_x: ti.template(), pixel_basis_y: ti.template()):
    """Frame ``f``'s camera, in the form :func:`_screen_point_of` reads."""
    ro = ti.math.vec3(cam_origin[f, 0], cam_origin[f, 1], cam_origin[f, 2])
    sp = ti.math.vec3(screen_point[f, 0], screen_point[f, 1], screen_point[f, 2])
    pbx = ti.math.vec3(pixel_basis_x[f, 0], pixel_basis_x[f, 1], pixel_basis_x[f, 2])
    pby = ti.math.vec3(pixel_basis_y[f, 0], pixel_basis_y[f, 1], pixel_basis_y[f, 2])
    nvec = pbx.cross(pby)
    big_d = (sp - ro).dot(nvec)
    dsq = ti.max(nvec.dot(nvec), 1e-30)
    return ro, sp, pbx, pby, nvec, big_d, dsq


@ti.func
def _screen_point_of(w, ro, sp, pbx, pby, nvec, big_d, dsq, half_w, half_h,
                     f, cam_proj: ti.template()):
    """``(front, px, py)``: world point ``w`` in screen pixel coordinates.

    The inverse of ``_generate_ray``, exactly as the host's
    ``raster_pipeline._project_points``; ``front`` is 0 for a point not in
    front of the camera plane. ``cam_proj`` holds the frames' ``(persp,
    spread)``: a pinhole camera (``spread == 0``) divides by the point's
    depth, any other by ``spread * D + persp * depth``
    (``algan.rendering.projection``), a constant for an orthographic camera.
    """
    front = 0
    px = 0.0
    py = 0.0
    d = w - ro
    wpn = d.dot(nvec)
    if ti.abs(wpn) >= 1e-12:
        td = big_d / wpn
        if td > 0.0:
            front = 1
            rel = ro + td * d - sp
            if ti.static(_projected(cam_proj)):
                if cam_proj[f, 1] != 0.0:
                    rel = (d - (wpn / dsq) * nvec) * (
                        big_d / (cam_proj[f, 1] * big_d + cam_proj[f, 0] * wpn))
            u = rel.cross(pby).dot(nvec) / dsq
            v = pbx.cross(rel).dot(nvec) / dsq
            px = u * half_h + half_w
            py = v * half_h + half_h
    return front, px, py


@ti.func
def _circuit_plane(tm, c, circuit_meta: ti.template()):
    center = ti.math.vec3(circuit_meta[tm, c, _M_CENTER],
                          circuit_meta[tm, c, _M_CENTER + 1],
                          circuit_meta[tm, c, _M_CENTER + 2])
    bu = ti.math.vec3(circuit_meta[tm, c, _M_BASIS_U],
                      circuit_meta[tm, c, _M_BASIS_U + 1],
                      circuit_meta[tm, c, _M_BASIS_U + 2])
    bv = ti.math.vec3(circuit_meta[tm, c, _M_BASIS_V],
                      circuit_meta[tm, c, _M_BASIS_V + 1],
                      circuit_meta[tm, c, _M_BASIS_V + 2])
    return center, bu, bv


@ti.func
def _circuit_reach(f, tm, c, half_w, half_h, aa_min_half_width,
                   cam_origin: ti.template(), screen_point: ti.template(),
                   pixel_basis_x: ti.template(), pixel_basis_y: ti.template(),
                   cam_proj: ti.template(), circuit_meta: ti.template()):
    """How far from its projected outline, in screen pixels, a pixel centre
    can still be accepted.

    ``_bez_pixel_hit`` accepts a centre within its coverage filter's reach of
    the drawn region, and the drawn region reaches ``R_d * pixel_size`` past
    the outline in the plane, ``R_d`` being the stroke's reach and the hairline
    dilation in pixels. ``pixel_size`` is world-per-pixel at the point's
    perpendicular depth, and a pinhole camera magnifies a world displacement
    there by at most ``1 / (pixel_size * cos(theta))`` pixels, ``theta`` the
    ray's angle off the optical axis -- whatever the plane's tilt. The filter's
    reach is ``_AA_FILTER_RADIUS`` pixels measured on screen (or ``pixel_size``
    multiples of it in the plane, where coverage is measured isotropically,
    which the same magnification bounds). So with ``R = R_d +
    _AA_FILTER_RADIUS``, ``R / cos(theta)`` over the frame's widest ray bounds
    it, with slack for the depth change across the reach itself (a fraction of
    a percent: the reach is a few pixels). The plane-space query
    ``_bez_pixel_hit`` makes is wider than ``R * pixel_size`` on a tilted plane
    -- the filter's screen reach spans more of the plane there -- but where on
    screen a centre can be accepted is what bounds the candidates.
    Any other projection is a pinhole with its eye pushed back (see
    ``algan.rendering.projection``), so the same bound holds with its own
    rays; an orthographic camera's are all on the axis, ``cos == 1``.
    """
    border_w = ti.abs(circuit_meta[tm, c, _M_BORDER_W])
    outline_w = ti.max(aa_min_half_width, 0.6)
    filled = circuit_meta[tm, c, _M_FILLED] > 0.5
    centered = circuit_meta[tm, c, _M_BORDER_CENTERED] > 0.5
    r = _circuit_query_radius(border_w, outline_w, filled, centered) + _AA_FILTER_RADIUS
    lowest = 1.0
    for k in ti.static(range(4)):
        ro, rd = _generate_ray(f, 0, 0, 2.0 * half_w * (k % 2),
                               2.0 * half_h * (k // 2), half_w, half_h,
                               cam_origin, screen_point, pixel_basis_x,
                               pixel_basis_y, cam_proj)
        lowest = ti.min(lowest, _axis_cos(f, ro, rd, screen_point,
                                          cam_origin, cam_proj))
    return r / ti.max(lowest, 0.05) * 1.02 + 0.5


@ti.func
def _circuit_span_setup(mask_e, bw, bh, span_min_area, f, c, half_w, half_h,
                        aa_min_half_width, cam_origin: ti.template(),
                        screen_point: ti.template(), pixel_basis_x: ti.template(),
                        pixel_basis_y: ti.template(), cam_proj: ti.template(),
                        circuit_meta: ti.template(),
                        edges_2d: ti.template(), edge_accel: ti.template()):
    """``(use_span, e0, e1, reach)`` for candidate circuit ``c`` of frame ``f``.

    Row by row only for a box of at least ``span_min_area`` pixels whose
    outline has edges and lies wholly in front of the camera; anything else
    keeps its box.
    """
    use_span = 0
    e0 = 0
    e1 = 0
    reach = 0.0
    if (mask_e != 0) and (bw * bh >= span_min_area):
        tm = f % circuit_meta.shape[0]
        te = f % edges_2d.shape[0]
        header = (te * circuit_meta.shape[1] + c) * BEZIER_ACCEL_HEADER_SIZE
        e0 = edge_accel[header + BEZIER_EDGE_START]
        e1 = edge_accel[header + BEZIER_EDGE_END]
        if e1 > e0:
            ro, sp, pbx, pby, nvec, big_d, dsq = _camera_terms(
                f, cam_origin, screen_point, pixel_basis_x, pixel_basis_y)
            center, bu, bv = _circuit_plane(tm, c, circuit_meta)
            front = 1
            for k in range(e0, e1):
                for end in ti.static(range(2)):
                    w = (center + edges_2d[te, k, 2 * end] * bu
                         + edges_2d[te, k, 2 * end + 1] * bv)
                    ok, _px, _py = _screen_point_of(w, ro, sp, pbx, pby, nvec,
                                                    big_d, dsq, 0.0, 0.0, f,
                                                    cam_proj)
                    front = ti.min(front, ok)
            if front == 1:
                use_span = 1
                reach = _circuit_reach(f, tm, c, half_w, half_h,
                                       aa_min_half_width, cam_origin,
                                       screen_point, pixel_basis_x,
                                       pixel_basis_y, cam_proj, circuit_meta)
    return use_span, e0, e1, reach


@ti.func
def _circuit_span_chunks(f, c, y, bx0, bx1, chunk, e0, e1, reach, half_w, half_h,
                         cam_origin: ti.template(), screen_point: ti.template(),
                         pixel_basis_x: ti.template(), pixel_basis_y: ti.template(),
                         cam_proj: ti.template(),
                         circuit_meta: ti.template(), edges_2d: ti.template()):
    """Chunks the row ``y`` of a circuit span candidate expands to, and the span."""
    tm = f % circuit_meta.shape[0]
    te = f % edges_2d.shape[0]
    ro, sp, pbx, pby, nvec, big_d, dsq = _camera_terms(
        f, cam_origin, screen_point, pixel_basis_x, pixel_basis_y)
    center, bu, bv = _circuit_plane(tm, c, circuit_meta)
    ylo = ti.cast(y, ti.f32) + 0.5 - reach
    yhi = ti.cast(y, ti.f32) + 0.5 + reach
    ok = 0
    xmin = 1e30
    xmax = -1e30
    for k in range(e0, e1):
        wa = center + edges_2d[te, k, 0] * bu + edges_2d[te, k, 1] * bv
        wb = center + edges_2d[te, k, 2] * bu + edges_2d[te, k, 3] * bv
        _fa, ax, ay = _screen_point_of(wa, ro, sp, pbx, pby, nvec, big_d, dsq,
                                       half_w, half_h, f, cam_proj)
        _fb, bx, by = _screen_point_of(wb, ro, sp, pbx, pby, nvec, big_d, dsq,
                                       half_w, half_h, f, cam_proj)
        if (ti.max(ay, by) >= ylo) and (ti.min(ay, by) <= yhi):
            t0 = 0.0
            t1 = 1.0
            dy = by - ay
            if ti.abs(dy) > 1e-12:
                ta = (ylo - ay) / dy
                tb = (yhi - ay) / dy
                t0 = ti.max(0.0, ti.min(ta, tb))
                t1 = ti.min(1.0, ti.max(ta, tb))
            xa = ax + (bx - ax) * t0
            xb = ax + (bx - ax) * t1
            xmin = ti.min(xmin, ti.min(xa, xb))
            xmax = ti.max(xmax, ti.max(xa, xb))
            ok = 1
    xs = bx0
    xe = bx0 - 1
    nch = 0
    if ok == 1:
        lo = ti.max(xmin - reach, -1e6)
        hi = ti.min(xmax + reach, 1e6)
        xs = ti.max(ti.cast(ti.floor(lo), ti.i32) - 1, bx0)
        xe = ti.min(ti.cast(ti.floor(hi), ti.i32) + 1, bx1)
        if xe >= xs:
            nch = (xe - xs + 1 + chunk - 1) // chunk
    return nch, xs, xe


@ti.kernel
def pair_expand_count_circuits(
    mask: ti.types.ndarray(),  # [N] u8
    x0f: ti.types.ndarray(),  # [N] i64
    x1f: ti.types.ndarray(),  # [N] i64
    y0f: ti.types.ndarray(),  # [N] i64
    y1f: ti.types.ndarray(),  # [N] i64
    n: ti.i32,
    chunk: ti.i32,
    counts: ti.types.ndarray(),  # [N] i64 OUT
    f_abs: ti.types.ndarray(),  # [Ft]
    ncirc: ti.i32,
    span_min_area: ti.i32,
    cam_origin: ti.types.ndarray(),
    screen_point: ti.types.ndarray(),
    pixel_basis_x: ti.types.ndarray(),
    pixel_basis_y: ti.types.ndarray(),
    cam_proj_table: ti.types.ndarray(),
    projected: ti.template(),
    circuit_meta: ti.types.ndarray(),
    edges_2d: ti.types.ndarray(),
    edge_accel: ti.types.ndarray(),
    half_w: ti.f32,
    half_h: ti.f32,
    aa_min_half_width: ti.f32,
):
    """:func:`pair_expand_count` for circuits: each candidate's chunk count,
    row by row along its projected outline where it qualifies.
    """
    cam_proj = ti.static(CameraProjection(cam_proj_table, projected))
    for e in range(n):
        cnt = ti.i64(0)
        if mask[e] != 0:
            bx0 = ti.cast(x0f[e], ti.i32)
            bx1 = ti.cast(x1f[e], ti.i32)
            by0 = ti.cast(y0f[e], ti.i32)
            by1 = ti.cast(y1f[e], ti.i32)
            bw = bx1 - bx0 + 1
            bh = by1 - by0 + 1
            c = e % ncirc
            f = ti.cast(f_abs[e // ncirc], ti.i32)
            use_span, e0, e1, reach = _circuit_span_setup(
                mask[e], bw, bh, span_min_area, f, c, half_w, half_h,
                aa_min_half_width, cam_origin, screen_point, pixel_basis_x,
                pixel_basis_y, cam_proj, circuit_meta, edges_2d, edge_accel)
            if use_span == 1:
                for y in range(by0, by1 + 1):
                    nch, _xs, _xe = _circuit_span_chunks(
                        f, c, y, bx0, bx1, chunk, e0, e1, reach, half_w, half_h,
                        cam_origin, screen_point, pixel_basis_x, pixel_basis_y,
                        cam_proj, circuit_meta, edges_2d)
                    cnt += nch
            else:
                cnt = (ti.cast(bw, ti.i64) * bh + chunk - 1) // chunk
        counts[e] = cnt


@ti.kernel
def pair_expand_write_circuits(
    mask: ti.types.ndarray(),  # [N] u8
    x0f: ti.types.ndarray(),  # [N] i64
    x1f: ti.types.ndarray(),  # [N] i64
    y0f: ti.types.ndarray(),  # [N] i64
    y1f: ti.types.ndarray(),  # [N] i64
    f_abs: ti.types.ndarray(),  # [Ft]
    offs: ti.types.ndarray(),  # [N] i64 -- EXCLUSIVE prefix of the counts
    n: ti.i32,
    ncirc: ti.i32,
    chunk: ti.i32,
    rows: ti.types.ndarray(),  # [total, 8] i32 OUT
    span_min_area: ti.i32,
    cam_origin: ti.types.ndarray(),
    screen_point: ti.types.ndarray(),
    pixel_basis_x: ti.types.ndarray(),
    pixel_basis_y: ti.types.ndarray(),
    cam_proj_table: ti.types.ndarray(),
    projected: ti.template(),
    circuit_meta: ti.types.ndarray(),
    edges_2d: ti.types.ndarray(),
    edge_accel: ti.types.ndarray(),
    half_w: ti.f32,
    half_h: ti.f32,
    aa_min_half_width: ti.f32,
):
    """:func:`pair_expand_write` for circuits, in the same row layout: each
    row's span a one-row-high box, so ``raster_bez_count`` decodes it unchanged,
    and rows ascending within a candidate, as its box's chunks were.
    """
    cam_proj = ti.static(CameraProjection(cam_proj_table, projected))
    for e in range(n):
        if mask[e] != 0:
            bx0 = ti.cast(x0f[e], ti.i32)
            bx1 = ti.cast(x1f[e], ti.i32)
            by0 = ti.cast(y0f[e], ti.i32)
            by1 = ti.cast(y1f[e], ti.i32)
            bw = bx1 - bx0 + 1
            bh = by1 - by0 + 1
            c = e % ncirc
            f = ti.cast(f_abs[e // ncirc], ti.i32)
            j = offs[e]
            use_span, e0, e1, reach = _circuit_span_setup(
                mask[e], bw, bh, span_min_area, f, c, half_w, half_h,
                aa_min_half_width, cam_origin, screen_point, pixel_basis_x,
                pixel_basis_y, cam_proj, circuit_meta, edges_2d, edge_accel)
            if use_span == 1:
                for y in range(by0, by1 + 1):
                    nch, xs, xe = _circuit_span_chunks(
                        f, c, y, bx0, bx1, chunk, e0, e1, reach, half_w, half_h,
                        cam_origin, screen_point, pixel_basis_x, pixel_basis_y,
                        cam_proj, circuit_meta, edges_2d)
                    for k in range(nch):
                        rows[j, 0] = c
                        rows[j, 1] = f
                        rows[j, 2] = xs
                        rows[j, 3] = y
                        rows[j, 4] = xe - xs + 1
                        rows[j, 5] = 1
                        rows[j, 6] = k * chunk
                        rows[j, 7] = 0
                        j += 1
            else:
                nch = (ti.cast(bw, ti.i64) * bh + chunk - 1) // chunk
                for k in range(ti.cast(nch, ti.i32)):
                    rows[j, 0] = c
                    rows[j, 1] = f
                    rows[j, 2] = bx0
                    rows[j, 3] = by0
                    rows[j, 4] = bw
                    rows[j, 5] = bh
                    rows[j, 6] = k * chunk
                    rows[j, 7] = 0
                    j += 1
