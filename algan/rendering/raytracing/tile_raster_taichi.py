"""Portable screen-bin construction and conservative primary-visibility proofs.

The bins contain primitive references, never a bounded fragment/K buffer. Count,
scan and write allocate the complete lists before any fragment is emitted. The
existing triangle/circuit coverage kernels remain the acceptance oracle.

Proof arithmetic is deliberately separate from shading arithmetic. Float32
interval endpoints are rounded outwards through integer bit operations, including
flush-to-zero-sized results. Unbounded/ill-conditioned inputs simply fail a
certificate; they are not grounds to discard a candidate.
"""

from algan.rendering.raytracing.raster_taichi import (
    _AA_FIXED_SCALE,
    _AA_MASK_ALL,
    _AA_SIMPLE_INTERIOR_BIT,
    _AA_SLIVER_BIT,
    raster_chunk,
)
from algan.rendering.raytracing.raytrace_kernels_taichi import (
    depth_tie_epsilon,
    min_hit_distance,
)
from algan.rendering.raytracing.sheet_compact_taichi import _span_chunks, _span_mode
from algan.rendering.raytracing.sheet_sort_taichi import _sort_run
from algan.rendering.raytracing.wavefront_kernels_taichi import (
    _tri_color_g,
    _tri_ior_transmission_g,
)
from algan.taichi_compat import ti

# Fixed implementation sizes, not claims about an optimal backend-specific size.
# Four fine tiles on each coarse axis. These partition work, never visibility.
FINE_TILE = 16
COARSE_TILE = 64
# This bit is SHEET data only. It is set for every sheet of a certified pixel,
# never for a subset of a pixel. Bits 20..27 are the sample-depth lose mask.
SIMPLE_INTERIOR_BIT = _AA_SIMPLE_INTERIOR_BIT

# Candidate ``flags`` values 0/1/2 are keep / geometrically separated / proven
# full-footprint opaque occluder. This bit rides beside them and says the COUNT
# pass chose the row-span form for that candidate, so the WRITE pass emits the
# rows it was sized for instead of re-deciding.
_SPAN_FORM_BIT = 4


@ti.func
def _finite(x):
    return (ti.bit_cast(x, ti.u32) & ti.u32(0x7fffffff)) < ti.u32(0x7f800000)


@ti.func
def _next_up(x):
    bits = ti.bit_cast(x, ti.u32)
    mag = bits & ti.u32(0x7fffffff)
    out = x
    if mag < ti.u32(0x00800000):
        # A GPU may flush a subnormal intermediate to signed zero. Enclose the
        # entire flushed range rather than stepping to the least subnormal.
        out = ti.bit_cast(ti.u32(0x00800000), ti.f32)
    elif mag < ti.u32(0x7f800000):
        if x > 0.0:
            out = ti.bit_cast(bits + ti.u32(1), ti.f32)
        else:
            out = ti.bit_cast(bits - ti.u32(1), ti.f32)
    return out


@ti.func
def _next_down(x):
    return -_next_up(-x)


@ti.func
def _outward(lo, hi):
    out = ti.Vector([-float('inf'), float('inf')])
    if _finite(lo) and _finite(hi):
        out = ti.Vector([_next_down(lo), _next_up(hi)])
    return out


@ti.func
def _ival(x):
    return ti.Vector([x, x])


@ti.func
def _iadd(a, b):
    return _outward(a[0] + b[0], a[1] + b[1])


@ti.func
def _isub(a, b):
    return _outward(a[0] - b[1], a[1] - b[0])


@ti.func
def _imul(a, b):
    v = ti.Vector([a[0] * b[0], a[0] * b[1],
                   a[1] * b[0], a[1] * b[1]])
    out = ti.Vector([-float('inf'), float('inf')])
    if _finite(v[0]) and _finite(v[1]) and _finite(v[2]) and _finite(v[3]):
        out = _outward(v.min(), v.max())
    return out


@ti.func
def _reciprocal_interval(x):
    # Backends may lower division to an approximate reciprocal under fast
    # math. Do not assume its error is one ulp: widen the estimate, then
    # VERIFY the enclosure with outward-rounded products. A failed check
    # returns unknown, never an optimistic bound.
    out = ti.Vector([-float('inf'), float('inf')])
    if x != 0.0 and _finite(x):
        q = 1.0 / x
        lo, hi = q, q
        for _ in ti.static(range(4)):
            lo, hi = _next_down(lo), _next_up(hi)
        lp = _imul(_ival(lo), _ival(x))
        hp = _imul(_ival(hi), _ival(x))
        verified = False
        if x > 0.0:
            verified = lp[1] <= 1.0 and hp[0] >= 1.0
        else:
            verified = lp[0] >= 1.0 and hp[1] <= 1.0
        if verified and _finite(lo) and _finite(hi):
            out = ti.Vector([lo, hi])
    return out


@ti.func
def _idiv(a, b):
    out = ti.Vector([-float('inf'), float('inf')])
    if b[0] > 0.0 or b[1] < 0.0:
        low = _reciprocal_interval(b[1])
        high = _reciprocal_interval(b[0])
        out = _imul(a, ti.Vector([low[0], high[1]]))
    return out


@ti.func
def _sqrt_interval(x):
    out = ti.Vector([0.0, float('inf')])
    if x == 0.0:
        out = ti.Vector([0.0, 0.0])
    elif x > 0.0 and _finite(x):
        q = ti.sqrt(x)
        lo, hi = q, q
        for _ in ti.static(range(4)):
            lo, hi = ti.max(0.0, _next_down(lo)), _next_up(hi)
        lp = _imul(_ival(lo), _ival(lo))
        hp = _imul(_ival(hi), _ival(hi))
        if lp[1] <= x and hp[0] >= x and _finite(hi):
            out = ti.Vector([lo, hi])
    return out


@ti.func
def _isquare(a):
    lo = ti.min(ti.abs(a[0]), ti.abs(a[1]))
    hi = ti.max(ti.abs(a[0]), ti.abs(a[1]))
    if a[0] <= 0.0 and a[1] >= 0.0:
        lo = 0.0
    return _outward(lo * lo, hi * hi)


@ti.func
def _edge_interval(ax, ay, bx, by, x, y):
    return _isub(_imul(_isub(_ival(bx), _ival(ax)), _isub(y, _ival(ay))),
                 _imul(_isub(_ival(by), _ival(ay)), _isub(x, _ival(ax))))


@ti.func
def _rect_proof(prim, f, x0, y0, x1, y1,
                screen: ti.template(), pos: ti.template(), camera: ti.template()):
    """Return (full, outside, near, far) for a continuous closed rectangle.

    ``full`` proves containment in BOTH the unsnapped triangle used for area
    integration and the fixed-point triangle used for sample ownership. Strict
    edges deliberately leave boundary equality to the general path. ``outside``
    similarly requires separation in both representations. Callers expand a
    candidate's rectangle before using outside to reject coverage-filter work.

    Distance encloses the screen-space intersection computation over the ENTIRE
    rectangle, not just its centre, corners or ownership samples. Interval
    dependence can make it wide; rejecting a certificate is always harmless.
    """
    full = 0
    outside = 0
    near = 0.0
    far = float('inf')
    sf = f % screen.shape[0]
    valid = screen[sf, prim, 9] > 0.5
    sx = ti.Vector.zero(ti.f32, 3)
    sy = ti.Vector.zero(ti.f32, 3)
    iw = ti.Vector.zero(ti.f32, 3)
    # The range gate precedes every integer conversion. Differences fit 30
    # bits after snapping, hence the two-product edge fits signed int64.
    for k in ti.static(range(3)):
        sx[k] = screen[sf, prim, k]
        sy[k] = screen[sf, prim, 3 + k]
        iw[k] = screen[sf, prim, 6 + k]
        valid = valid and _finite(sx[k]) and _finite(sy[k]) and _finite(iw[k])
        valid = valid and ti.abs(sx[k]) < 65536.0 and ti.abs(sy[k]) < 65536.0
    valid = valid and ti.abs(x0) < 65536.0 and ti.abs(x1) < 65536.0
    valid = valid and ti.abs(y0) < 65536.0 and ti.abs(y1) < 65536.0
    if valid:
        area = _edge_interval(sx[0], sy[0], sx[1], sy[1], _ival(sx[2]), _ival(sy[2]))
        orient = 0
        if area[0] > 0.0:
            orient = 1
        elif area[1] < 0.0:
            orient = -1
        ix = ti.Vector.zero(ti.i64, 3)
        iy = ti.Vector.zero(ti.i64, 3)
        for k in ti.static(range(3)):
            ix[k] = ti.cast(ti.round(sx[k] * _AA_FIXED_SCALE), ti.i64)
            iy[k] = ti.cast(ti.round(sy[k] * _AA_FIXED_SCALE), ti.i64)
        ia = (ix[1] - ix[0]) * (iy[2] - iy[0]) - (iy[1] - iy[0]) * (ix[2] - ix[0])
        if orient != 0 and ia * ti.cast(orient, ti.i64) > 0:
            full = 1
            e = ti.Matrix.zero(ti.f32, 3, 2)
            for k in ti.static(range(3)):
                a, b = (k + 1) % 3, (k + 2) % 3
                v = _edge_interval(sx[a], sy[a], sx[b], sy[b],
                                   ti.Vector([x0, x1]), ti.Vector([y0, y1]))
                e[k, 0], e[k, 1] = v[0], v[1]
                vlo, vhi = v[0], v[1]
                if orient < 0:
                    vlo, vhi = -v[1], -v[0]
                ilo = ti.i64(0x7fffffffffffffff)
                ihi = -ti.i64(0x7fffffffffffffff)
                for corner in ti.static(range(4)):
                    xx = x0 if corner % 2 == 0 else x1
                    yy = y0 if corner // 2 == 0 else y1
                    xx_i = ti.cast(ti.round(xx * _AA_FIXED_SCALE), ti.i64)
                    yy_i = ti.cast(ti.round(yy * _AA_FIXED_SCALE), ti.i64)
                    ev = ((ix[b] - ix[a]) * (yy_i - iy[a])
                          - (iy[b] - iy[a]) * (xx_i - ix[a])) * ti.cast(orient, ti.i64)
                    ilo, ihi = ti.min(ilo, ev), ti.max(ihi, ev)
                if vlo <= 0.0 or ilo <= 0:
                    full = 0
                if vhi < 0.0 and ihi < 0:
                    outside = 1
            # Exactly the rational interpolation used by _ss_pixel, enclosed
            # over x/y ranges rather than evaluating four representative rays.
            n = ti.Matrix.zero(ti.f32, 3, 2)
            denom = _ival(0.0)
            for k in ti.static(range(3)):
                nk = _imul(ti.Vector([e[k, 0], e[k, 1]]), _ival(iw[k]))
                n[k, 0], n[k, 1] = nk[0], nk[1]
                denom = _iadd(denom, nk)
            if (denom[0] > 1e-20 or denom[1] < -1e-20) and _finite(denom[0]) and _finite(denom[1]):
                bary = ti.Matrix.zero(ti.f32, 3, 2)
                for k in ti.static(range(3)):
                    bk = _idiv(ti.Vector([n[k, 0], n[k, 1]]), denom)
                    bary[k, 0], bary[k, 1] = bk[0], bk[1]
                dist2 = _ival(0.0)
                pf = f % pos.shape[0]
                cf = f % camera.shape[0]
                for axis in ti.static(range(3)):
                    hp = _ival(0.0)
                    for k in ti.static(range(3)):
                        hp = _iadd(hp, _imul(ti.Vector([bary[k, 0], bary[k, 1]]),
                                            _ival(pos[pf, prim, 3 * k + axis])))
                    delta = _isub(hp, _ival(camera[cf, axis]))
                    dist2 = _iadd(dist2, _isquare(delta))
                if _finite(dist2[0]) and _finite(dist2[1]):
                    near = _sqrt_interval(ti.max(dist2[0], 0.0))[0]
                    far = _sqrt_interval(ti.max(dist2[1], 0.0))[1]
    return full, outside, near, far


@ti.func
def _depth_range_safe(far):
    # Also excludes float-to-integer overflow in the legacy depth-bin
    # expression at enormous distances. Such records keep that exact legacy
    # ordering rather than being geometrically culled ahead of its sort.
    return _finite(far) and far < 2147483645.0 * depth_tie_epsilon


@ti.func
def _strictly_behind(near, far):
    # No equality/tie pruning, including the renderer's saturated depth bins.
    # Four epsilons cover the division/reciprocal binning variants as well as
    # the endpoints' outward rounding. Farther layers in a saturated bin must
    # still be ordered by their layer, so saturation explicitly fails proof.
    return (_depth_range_safe(far) and _depth_range_safe(near)
            and _next_up(far + 4.0 * depth_tie_epsilon) < near)


@ti.kernel
def bbox_records(bounds_f: ti.types.ndarray(), bounds_x: ti.types.ndarray(),
                 bounds_m: ti.types.ndarray(), records: ti.types.ndarray(),
                 base: int, nprim: int, time_start: int, frames: int,
                 width: int, height: int, kind: ti.template()):
    # [primitive, absolute frame, x0, y0, x1, y1, kind, material-opaque].
    # Inactive rows have x1 < x0; no nonfinite value is converted to integer.
    for i in range(frames * nprim):
        prim, f = i % nprim, time_start + i // nprim
        row = f % bounds_f.shape[0]
        j = base + i
        records[j, 0], records[j, 1] = prim, f
        records[j, 2], records[j, 3] = 0, 0
        records[j, 4], records[j, 5] = -1, -1
        records[j, 6] = kind
        records[j, 7] = ti.cast(bounds_m[row, prim, 3] != 0, ti.i32)
        on_y = bounds_f[row, prim, 3] >= -1.0 and bounds_f[row, prim, 2] <= height
        reach = (bounds_m[row, prim, 1] != 0 and on_y) or bounds_m[row, prim, 2] != 0
        active = reach and (bounds_m[row, prim, 3] != 0 or bounds_m[row, prim, 4] != 0)
        if active:
            y0, y1 = 0, height - 1
            if bounds_m[row, prim, 0] != 0:
                y0 = ti.cast(ti.math.clamp(bounds_f[row, prim, 0], 0.0, height - 1.0), ti.i32)
                y1 = ti.cast(ti.math.clamp(bounds_f[row, prim, 1], 0.0, height - 1.0), ti.i32)
            records[j, 2] = ti.max(0, bounds_x[row, prim, 0])
            records[j, 3] = y0
            records[j, 4] = ti.min(width - 1, bounds_x[row, prim, 1])
            records[j, 5] = y1


@ti.kernel
def coarse_counts(records: ti.types.ndarray(), counts: ti.types.ndarray(), n: int):
    for i in range(n):
        x0, y0, x1, y1 = records[i, 2], records[i, 3], records[i, 4], records[i, 5]
        count = ti.cast(0, ti.i64)
        if x1 >= x0 and y1 >= y0:
            count = ti.cast(x1 // COARSE_TILE - x0 // COARSE_TILE + 1, ti.i64) * (y1 // COARSE_TILE - y0 // COARSE_TILE + 1)
        counts[i] = count


@ti.kernel
def coarse_write(records: ti.types.ndarray(), offsets: ti.types.ndarray(),
                 bin_ids: ti.types.ndarray(), refs: ti.types.ndarray(),
                 n: int, time_start: int, coarse_w: int, coarse_h: int):
    for i in range(n):
        x0, y0, x1, y1 = records[i, 2], records[i, 3], records[i, 4], records[i, 5]
        dst = ti.cast(offsets[i], ti.i32)
        if x1 >= x0 and y1 >= y0:
            for y in range(y0 // COARSE_TILE, y1 // COARSE_TILE + 1):
                for x in range(x0 // COARSE_TILE, x1 // COARSE_TILE + 1):
                    bin_ids[dst] = ((records[i, 1] - time_start) * coarse_h + y) * coarse_w + x
                    refs[dst] = i
                    dst += 1


@ti.func
def _fine_rect(coarse_id, child, coarse_w, coarse_h, width, height):
    fr = coarse_id // (coarse_w * coarse_h)
    local = coarse_id % (coarse_w * coarse_h)
    x0 = (local % coarse_w) * COARSE_TILE + (child % 4) * FINE_TILE
    y0 = (local // coarse_w) * COARSE_TILE + (child // 4) * FINE_TILE
    return fr, x0, y0, ti.min(x0 + FINE_TILE - 1, width - 1), ti.min(y0 + FINE_TILE - 1, height - 1)


@ti.func
def _bbox_overlaps(records: ti.template(), r, x0, y0, x1, y1):
    return (x0 <= x1 and y0 <= y1 and records[r, 2] <= x1 and records[r, 4] >= x0
            and records[r, 3] <= y1 and records[r, 5] >= y0)


@ti.kernel
def fine_counts(records: ti.types.ndarray(), coarse_ids: ti.types.ndarray(),
                coarse_offsets: ti.types.ndarray(), coarse_refs: ti.types.ndarray(),
                counts: ti.types.ndarray(), nfine: int, coarse_w: int, coarse_h: int,
                width: int, height: int):
    for i in range(nfine):
        parent, child = i // 16, i % 16
        _fr, x0, y0, x1, y1 = _fine_rect(coarse_ids[parent], child, coarse_w, coarse_h, width, height)
        count = ti.cast(0, ti.i64)
        for j in range(coarse_offsets[parent], coarse_offsets[parent + 1]):
            r = coarse_refs[j]
            if _bbox_overlaps(records, r, x0, y0, x1, y1):
                count += 1
        counts[i] = count


@ti.kernel
def fine_write(records: ti.types.ndarray(), coarse_ids: ti.types.ndarray(),
               coarse_offsets: ti.types.ndarray(), coarse_refs: ti.types.ndarray(),
               offsets: ti.types.ndarray(), candidates: ti.types.ndarray(),
               tile_ids: ti.types.ndarray(), nfine: int, coarse_w: int, coarse_h: int,
               width: int, height: int):
    tw, th = (width + FINE_TILE - 1) // FINE_TILE, (height + FINE_TILE - 1) // FINE_TILE
    for i in range(nfine):
        parent, child = i // 16, i % 16
        fr, x0, y0, x1, y1 = _fine_rect(coarse_ids[parent], child, coarse_w, coarse_h, width, height)
        dst = ti.cast(offsets[i], ti.i32)
        tile_ids[i] = (fr * th + y0 // FINE_TILE) * tw + x0 // FINE_TILE
        for j in range(coarse_offsets[parent], coarse_offsets[parent + 1]):
            r = coarse_refs[j]
            if _bbox_overlaps(records, r, x0, y0, x1, y1):
                # The fine candidate owns exactly this tile/bbox intersection.
                candidates[dst, 0], candidates[dst, 1] = records[r, 0], records[r, 1]
                candidates[dst, 2] = ti.max(x0, records[r, 2])
                candidates[dst, 3] = ti.max(y0, records[r, 3])
                candidates[dst, 4] = ti.min(x1, records[r, 4])
                candidates[dst, 5] = ti.min(y1, records[r, 5])
                candidates[dst, 6], candidates[dst, 7] = records[r, 6], records[r, 7]
                candidates[dst, 8] = i
                dst += 1


@ti.kernel
def candidate_proofs(candidates: ti.types.ndarray(), intervals: ti.types.ndarray(),
                     flags: ti.types.ndarray(), screen: ti.types.ndarray(),
                     pos: ti.types.ndarray(), camera: ti.types.ndarray(),
                     opaque_proven: ti.types.ndarray(), n: int,
                     width: int, height: int, time_start: int, cull: ti.template()):
    for i in range(n):
        flags[i] = 0
        intervals[i, 0], intervals[i, 1] = 0.0, float('inf')
        if candidates[i, 6] == 1:
            prim, f = candidates[i, 0], candidates[i, 1]
            x0 = candidates[i, 2] // FINE_TILE * FINE_TILE
            y0 = candidates[i, 3] // FINE_TILE * FINE_TILE
            x1, y1 = ti.min(x0 + FINE_TILE, width), ti.min(y0 + FINE_TILE, height)
            # The coverage kernel's filter reaches outside the geometric
            # footprint. Separation of the expanded tile encloses that reach,
            # including sample-less area donors and its distance-test margin.
            _full, outside, _near, _far = _rect_proof(prim, f, x0 - 1.0, y0 - 1.0,
                                                    x1 + 1.0, y1 + 1.0, screen, pos, camera)
            if outside:
                flags[i] = 1  # geometric reject
            elif ti.static(cull):
                full, _outside, near, far = _rect_proof(prim, f, ti.cast(x0, ti.f32), ti.cast(y0, ti.f32),
                                                       ti.cast(x1, ti.f32), ti.cast(y1, ti.f32), screen, pos, camera)
                intervals[i, 0], intervals[i, 1] = near, far
                if full and near > min_hit_distance and candidates[i, 7] != 0:
                    if opaque_proven[f - time_start, prim] != 0:
                        flags[i] = 2  # full-footprint, materially opaque


@ti.kernel
def tile_occluders(offsets: ti.types.ndarray(), intervals: ti.types.ndarray(),
                   flags: ti.types.ndarray(), bound: ti.types.ndarray(), ntiles: int):
    for tile in range(ntiles):
        far = float('inf')
        for j in range(offsets[tile], offsets[tile + 1]):
            if flags[j] == 2:
                far = ti.min(far, intervals[j, 1])
        bound[tile] = far


@ti.func
def _tile_span_mode(candidates: ti.template(), i, screen: ti.template(),
                    span_min_area, spans: ti.template()):
    """Whether this tile candidate expands row by row, and its screen row.

    Exactly the reference frontend's gate (``_span_mode``) applied to the
    tile-clipped box instead of the whole primitive bbox: triangles only,
    all-front projections only, boxes of at least ``span_min_area`` pixels.
    A straddler keeps its box, because its projection is not a bound.
    """
    use = 0
    fr = 0
    if ti.static(spans):
        if candidates[i, 6] == 1:
            bw = candidates[i, 4] - candidates[i, 2] + 1
            bh = candidates[i, 5] - candidates[i, 3] + 1
            fr = candidates[i, 1] % screen.shape[0]
            use = _span_mode(1, bw, bh, span_min_area, screen, fr,
                             candidates[i, 0], spans)
    return use, fr


@ti.kernel
def tile_pair_counts(candidates: ti.types.ndarray(), flags: ti.types.ndarray(),
                     intervals: ti.types.ndarray(), bound: ti.types.ndarray(),
                     screen: ti.types.ndarray(), counts: ti.types.ndarray(),
                     stats: ti.types.ndarray(), n: int, span_min_area: int,
                     spans: ti.template()):
    """Chunks per candidate, and which form it emits them in.

    A tile candidate's box is the triangle's bbox clipped to the tile, and how
    well that box is FILLED is what decides the form. A tile the triangle
    crosses diagonally is mostly empty, and emitting the box hands COUNT the
    whole tile: measured on the nn scene at HD, 11.4M candidate pixels against
    the reference frontend's 2.9M for the same frame, which is where this
    frontend's COUNT/WRITE regression came from. A tile the triangle covers
    outright is the opposite case -- its box is already perfectly packed at 32
    pixels a chunk, and row-splitting it only doubles the chunk count for no
    saving (measured: the overdraw scene went 3.08s -> 3.99s warm on CPU when
    every eligible candidate took spans).

    So take spans only where they at least HALVE the candidate pixels. The
    decision rides in flags bit 2 so the write pass reads it back rather than
    re-deriving it and risking a different answer than the counts it fills.
    """
    for i in range(n):
        rejected = flags[i] == 1
        keep = not rejected
        hidden = False
        if keep and candidates[i, 6] == 1:
            hidden = (_depth_range_safe(intervals[i, 1])
                      and _strictly_behind(intervals[i, 0], bound[candidates[i, 8]]))
            keep = not hidden
        count = ti.cast(0, ti.i64)
        use_span = 0
        if keep:
            x0, y0 = candidates[i, 2], candidates[i, 3]
            x1, y1 = candidates[i, 4], candidates[i, 5]
            box_px = (x1 - x0 + 1) * (y1 - y0 + 1)
            count = (box_px + raster_chunk - 1) // raster_chunk
            eligible, fr = _tile_span_mode(candidates, i, screen, span_min_area, spans)
            if eligible == 1:
                span_px = 0
                span_chunks = ti.cast(0, ti.i64)
                for y in range(y0, y1 + 1):
                    nch, xs, xe = _span_chunks(screen, fr, candidates[i, 0], y,
                                               x0, x1, raster_chunk)
                    span_chunks += nch
                    if nch > 0:
                        span_px += xe - xs + 1
                if span_px * 2 <= box_px:
                    use_span = 1
                    count = span_chunks
        counts[i] = count
        if use_span != 0:
            flags[i] = flags[i] | _SPAN_FORM_BIT
        if rejected:
            ti.atomic_add(stats[0], 1)
        if hidden:
            ti.atomic_add(stats[1], 1)


@ti.kernel
def tile_pair_write(candidates: ti.types.ndarray(), offsets: ti.types.ndarray(),
                    flags: ti.types.ndarray(), screen: ti.types.ndarray(),
                    pairs: ti.types.ndarray(), classes: ti.types.ndarray(),
                    n: int, spans: ti.template()):
    for i in range(n):
        # The COUNT pass's slice is the authority on how many rows this
        # candidate owns -- and on whether it owns any at all. A candidate the
        # proofs rejected counted ZERO chunks, so it must write none: walking
        # its geometry here anyway would write over the next candidate's slice
        # and off the end of the buffer, which is why the emission is driven
        # by the prefix range rather than by recomputing acceptance.
        start, end = offsets[i], offsets[i + 1]
        if end > start:
            prim, f = candidates[i, 0], candidates[i, 1]
            x0, y0 = candidates[i, 2], candidates[i, 3]
            x1, y1 = candidates[i, 4], candidates[i, 5]
            # Same class order as the reference frontend: bez opaque/trans,
            # tri opaque/trans. Within a pixel a primitive occurs only once,
            # whichever form its rows take -- a span is clipped to this tile's
            # x-range, so two tiles never claim the same pixel.
            cls = 2 * candidates[i, 6] + (1 - candidates[i, 7])
            j = start
            use = 0
            fr = 0
            if ti.static(spans):
                use = ti.cast((flags[i] & _SPAN_FORM_BIT) != 0, ti.i32)
                fr = candidates[i, 1] % screen.shape[0]
            if use == 1:
                # The COUNT pass chose this form and sized the slice from the
                # SAME row walk, so it lands exactly inside; the bound below
                # is belt and braces.
                for y in range(y0, y1 + 1):
                    nch, xs, xe = _span_chunks(screen, fr, prim, y, x0, x1, raster_chunk)
                    for c in range(nch):
                        if j < end:
                            pairs[j, 0], pairs[j, 1] = prim, f
                            pairs[j, 2], pairs[j, 3] = xs, y
                            pairs[j, 4], pairs[j, 5] = xe - xs + 1, 1
                            pairs[j, 6], pairs[j, 7] = c * raster_chunk, 0
                            classes[j] = cls
                            j += 1
            else:
                bw, bh = x1 - x0 + 1, y1 - y0 + 1
                for c in range((bw * bh + raster_chunk - 1) // raster_chunk):
                    if j < end:
                        pairs[j, 0], pairs[j, 1] = prim, f
                        pairs[j, 2], pairs[j, 3] = x0, y0
                        pairs[j, 4], pairs[j, 5] = bw, bh
                        pairs[j, 6], pairs[j, 7] = c * raster_chunk, 0
                        classes[j] = cls
                        j += 1


@ti.func
def _lower_bound(values: ti.template(), value, n):
    lo, hi = 0, n
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if values[mid] < value:
            lo = mid + 1
        else:
            hi = mid
    return lo


@ti.func
def _pixel_bucket(pixel, active_tiles: ti.template(), width, height):
    ppf = width * height
    fr, local = pixel // ppf, pixel % ppf
    x, y = local % width, local // width
    tw, th = (width + FINE_TILE - 1) // FINE_TILE, (height + FINE_TILE - 1) // FINE_TILE
    tile = (fr * th + y // FINE_TILE) * tw + x // FINE_TILE
    ordinal = _lower_bound(active_tiles, tile, active_tiles.shape[0])
    return ordinal * (FINE_TILE * FINE_TILE) + (y % FINE_TILE) * FINE_TILE + x % FINE_TILE


@ti.kernel
def fragment_bucket_counts(keys: ti.types.ndarray(), active_tiles: ti.types.ndarray(),
                           counts: ti.types.ndarray(), n: int, width: int, height: int):
    for i in range(n):
        pixel = ti.cast(keys[i] >> 32, ti.i32)
        bucket = _pixel_bucket(pixel, active_tiles, width, height)
        ti.atomic_add(counts[bucket], 1)


@ti.kernel
def bucket_pixels(buckets: ti.types.ndarray(), tiles: ti.types.ndarray(),
                  pixels: ti.types.ndarray(), n: int, width: int, height: int):
    tw, th = (width + FINE_TILE - 1) // FINE_TILE, (height + FINE_TILE - 1) // FINE_TILE
    for i in range(n):
        b = buckets[i]
        tile = tiles[b // (FINE_TILE * FINE_TILE)]
        fr, local = tile // (tw * th), tile % (tw * th)
        x = (local % tw) * FINE_TILE + b % FINE_TILE
        y = (local // tw) * FINE_TILE + (b // FINE_TILE) % FINE_TILE
        pixels[i] = fr * width * height + y * width + x


@ti.kernel
def bucket_starts(buckets: ti.types.ndarray(), offsets: ti.types.ndarray(),
                  starts: ti.types.ndarray(), n: int):
    for i in range(n):
        starts[buckets[i]] = offsets[i]


@ti.kernel
def scatter_fragment_order(keys: ti.types.ndarray(), tiles: ti.types.ndarray(),
                           starts: ti.types.ndarray(), cursors: ti.types.ndarray(),
                           order: ti.types.ndarray(), n: int, width: int, height: int):
    for i in range(n):
        bucket = _pixel_bucket(ti.cast(keys[i] >> 32, ti.i32), tiles, width, height)
        local = ti.atomic_add(cursors[bucket], 1)
        order[starts[bucket] + local] = i


@ti.kernel
def primary_pixel_order(offsets: ti.types.ndarray(), key: ti.types.ndarray(),
                        order: ti.types.ndarray(), npixels: int):
    for p in range(npixels):
        # Existing insertion/heapsort, now on primary depth-bin/layer keys.
        # Original emission index is the final tie key, not atomic arrival.
        _sort_run(offsets[p], offsets[p + 1], order, key, key, False)


@ti.kernel
def interior_fragment_proofs(keys: ti.types.ndarray(), refs: ti.types.ndarray(),
                             cov: ti.types.ndarray(), masks: ti.types.ndarray(),
                             screen: ti.types.ndarray(), pos: ti.types.ndarray(),
                             camera: ti.types.ndarray(), objects: ti.types.ndarray(),
                             uncertain: ti.types.ndarray(), closed: ti.types.ndarray(),
                             eligible: ti.types.ndarray(), distances: ti.types.ndarray(),
                             surface: ti.types.ndarray(), n: int, time_start: int,
                             width: int, height: int):
    for i in range(n):
        eligible[i] = 0
        surface[i] = -1
        distances[i, 0], distances[i, 1] = 0.0, float('inf')
        prim = refs[i]
        if prim >= 0 and cov[i] == 1.0 and (masks[i] & _AA_MASK_ALL) == _AA_MASK_ALL and (masks[i] & _AA_SLIVER_BIT) == 0:
            pixel = ti.cast(keys[i] >> 32, ti.i32)
            f = time_start + pixel // (width * height)
            surface[i] = objects[f % objects.shape[0], prim]
            if uncertain[f % uncertain.shape[0], prim] == 0 and closed[f % closed.shape[0], prim] == 0:
                x, y = pixel % width, (pixel // width) % height
                full, _out, near, far = _rect_proof(prim, f, ti.cast(x, ti.f32), ti.cast(y, ti.f32),
                                                   x + 1.0, y + 1.0, screen, pos, camera)
                if full and near > min_hit_distance and _depth_range_safe(far):
                    eligible[i] = 1
                    distances[i, 0], distances[i, 1] = near, far


@ti.kernel
def interior_pixels(offsets: ti.types.ndarray(), eligible: ti.types.ndarray(),
                    distances: ti.types.ndarray(), surface: ti.types.ndarray(),
                    scratch_order: ti.types.ndarray(), simple: ti.types.ndarray(), npixels: int):
    for p in range(npixels):
        start, end = offsets[p], offsets[p + 1]
        valid = end > start
        prev_far = 0.0
        for j in range(start, end):
            valid = valid and eligible[j] != 0
            if j > start:
                valid = valid and _strictly_behind(distances[j, 0], prev_far)
            prev_far = distances[j, 1]
        if valid:
            # Distinct source surfaces are mandatory. Repeated faces/facing
            # bands can invoke shell/rank/sibling rules even when full-covering.
            # Sorting keeps this O(n log n) for deep transparent pixels.
            for j in range(start, end):
                scratch_order[j] = j
            _sort_run(start, end, scratch_order, surface, surface, False)
            for j in range(start + 1, end):
                if surface[scratch_order[j]] == surface[scratch_order[j - 1]]:
                    valid = False
        simple[p] = ti.cast(valid, ti.i32)


@ti.kernel
def select_pixel_fragments(pixel_indices: ti.types.ndarray(), src_offsets: ti.types.ndarray(),
                           dst_offsets: ti.types.ndarray(), indices: ti.types.ndarray(), npixels: int):
    for p in range(npixels):
        source = pixel_indices[p]
        start, end = src_offsets[source], src_offsets[source + 1]
        for j in range(start, end):
            indices[dst_offsets[p] + j - start] = j


@ti.kernel
def merge_interior_sheets(simple: ti.types.ndarray(), general_row: ti.types.ndarray(),
                          frag_offsets: ti.types.ndarray(), general_offsets: ti.types.ndarray(),
                          out_offsets: ti.types.ndarray(),
                          fk: ti.types.ndarray(), fr: ti.types.ndarray(), fa: ti.types.ndarray(),
                          fc: ti.types.ndarray(), fm: ti.types.ndarray(), fcap: ti.types.ndarray(),
                          gk: ti.types.ndarray(), gr: ti.types.ndarray(), ga: ti.types.ndarray(),
                          gc: ti.types.ndarray(), gm: ti.types.ndarray(), gcap: ti.types.ndarray(),
                          ok: ti.types.ndarray(), oref: ti.types.ndarray(), oa: ti.types.ndarray(),
                          oc: ti.types.ndarray(), om: ti.types.ndarray(), ocap: ti.types.ndarray(),
                          npixels: int):
    for p in range(npixels):
        out = out_offsets[p]
        if simple[p] != 0:
            for j in range(frag_offsets[p], frag_offsets[p + 1]):
                ok[out], oref[out] = fk[j], fr[j]
                oa[out, 0], oa[out, 1] = fa[j, 0], fa[j, 1]
                oc[out] = 1.0
                # Full, distinct single-fragment bands need no lose, sliver,
                # material-enforcer or sibling metadata. Keep the facing and
                # one-mesh bits where they are informational/identity data.
                om[out] = fm[j] | SIMPLE_INTERIOR_BIT
                ocap[out] = fcap[j]
                out += 1
        else:
            gp = general_row[p]
            for j in range(general_offsets[gp], general_offsets[gp + 1]):
                ok[out], oref[out] = gk[j], gr[j]
                oa[out, 0], oa[out, 1] = ga[j, 0], ga[j, 1]
                oc[out], om[out], ocap[out] = gc[j], gm[j], gcap[j]
                out += 1


@ti.kernel
def opaque_material_proofs(colors: ti.types.ndarray(), extra: ti.types.ndarray(),
                           col_row: ti.types.ndarray(), uvs: ti.types.ndarray(),
                           meta: ti.types.ndarray(), textures: ti.types.ndarray(),
                           proven: ti.types.ndarray(), num_colored: ti.template(),
                           nprim: int, time_start: int, frames: int):
    """Stricter than the legacy near-one/near-zero visibility flags.

    Nonconstant opacity or transmission maps stay unproven. Constant promoted
    materials use 1x1 maps and are just as certifiable as per-vertex materials.
    """
    for i in range(frames * nprim):
        prim, fr = i % nprim, i // nprim
        f = time_start + fr
        constant_maps = True
        if prim >= num_colored:
            idx = prim - num_colored
            if meta[idx, 0] >= 0:
                constant_maps = meta[idx, 1] == 1 and meta[idx, 2] == 1
            if meta[idx, 3] >= 0 and (meta[idx, 9] & 8) != 0:
                constant_maps = constant_maps and meta[idx, 4] == 1 and meta[idx, 5] == 1
        ok = constant_maps
        if constant_maps:
            for corner in ti.static(range(3)):
                w0, w1, w2 = ti.cast(corner == 0, ti.f32), ti.cast(corner == 1, ti.f32), ti.cast(corner == 2, ti.f32)
                _color, alpha = _tri_color_g(0, f, prim, w0, w1, w2,
                                             colors, col_row, uvs, meta, textures, num_colored)
                _ior, transmission = _tri_ior_transmission_g(0, f, prim, w0, w1, w2,
                                                            extra, col_row, uvs, meta, textures, num_colored)
                ok = ok and alpha == 1.0 and transmission == 0.0
        proven[fr, prim] = ti.cast(ok, ti.i32)
