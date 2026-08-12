"""Hybrid raster front-end kernels for deterministic primary visibility.

The frontend replaces the first classic wavefront iteration for flat triangles
and Bezier circuits.  Primitive/chunk pairs enumerate candidate pixels, exact
intersection tests reject bbox misses, a typed opaque visibility buffer removes
hidden work, and compact transparent fragment records are ordered by the same
transitive ``(depth bin, descending layer)`` relation as the classic tracer.
One thread then resolves each pixel's ordered straight-ray list serially.

Important implementation properties:

* Triangle projection data is precomputed once per frame/primitive by
  :func:`raster_pipeline.precompute_triangle_projection`.  Straddling cases use
  exact per-pixel ray casting.
* Proven-opaque triangles and Bezier circuits share a typed int64 visibility
  buffer.  The packed key contains depth bin and inverted layer, so coplanar
  layer semantics are respected before transparent culling.
* Transparent triangle/circuit COUNT and WRITE kernels fetch sampled alpha and
  discard alpha-zero texels before sorting.  Records contain only exact-distance
  key, typed primitive reference (including the circuit border/fill blend
  weight), two intersection parameters, and an analytic coverage lane.
* Analytic anti-aliasing (``ALGAN_ANALYTIC_AA``, see DESIGN_analytic_aa.md):
  each circuit fragment carries the fraction of the pixel square its drawn
  region covers -- a box filter of the outline signed-distance field that
  ``_bezier_point_metrics`` already computes -- and the resolve folds that into
  the fragment's alpha, so circuit silhouettes resolve continuously at
  ``anti_alias_level = 1`` instead of all-or-nothing.  Flat triangles keep
  coverage 1.0 (phase 2), so the coverage lane is host-pre-filled and only the
  circuit kernels write it.
* :func:`raster_shadow_event_build` performs the same ordered transport/seam
  decisions as final resolve and emits only accepted triangle lighting events.
  :func:`raster_shadow_trace` traces that separate sparse any-hit queue and
  stores one visibility value per event/light, with no fixed fragment-slot or
  packed-light limit. Point/spot emitter radii and directional angular radii
  use the same deterministic golden-angle fan as the classic wavefront path.
* :func:`raster_first_shade` resolves at most ``MAX_SURFACES_PER_RAY`` accepted
  primary surfaces, commits retired pixels, and writes reflection/refraction
  continuations to the compact primary queue.  The host later copies only
  surviving continuations into the classic secondary-wavefront state.

The current host tile is a contiguous linear ray range, usually a row band,
not a fixed square raster tile.  Each pair covers up to ``RASTER_CHUNK`` pixels.
Future work should benchmark square block bins and candidate-parallel block
kernels. PN patches, custom scatter, near clipping, and in-place supersampling
still route to the classic frontend without changing geometry construction.
"""
import taichi as ti

from algan.rendering.raytracing.raytrace_kernels_taichi import (
    _M_BASIS_U,
    _M_BASIS_V,
    _M_BORDER_W,
    _M_CENTER,
    _M_FILLED,
    _M_IOR,
    _M_NORMAL,
    _M_REFLECTIVITY,
    _M_ROUGHNESS,
    _M_TRANSMISSION,
    BARYCENTRIC_EPSILON,
    DEPTH_TIE_EPSILON,
    INV_DEPTH_TIE_EPSILON,
    MAX_SHADOW_LIGHTS,
    MAX_SURFACES_PER_RAY,
    MIN_ALPHA,
    MIN_HIT_DISTANCE,
    MIN_WEIGHT,
    NODE_ARG,
    TRIANGLE_EDGE_EPSILON,
    _bezier_normal,
    _bezier_point_metrics,
    _circuit_point_region,
    _circuit_query_radius,
    _generate_ray,
    _sample_circuit_color_blend,
    _shade_tri_hit,
    _shadow_occluded,
)
from algan.rendering.raytracing.shading_taichi import (
    _MID_UNLIT,
    _orient_hit_normals,
    _reflect_frame,
)
from algan.settings._startup import _SOFT_SHADOW_SAMPLES as SOFT_SHADOW_SAMPLES
from algan.rendering.raytracing.wavefront_kernels_taichi import (
    _ACTIVE,
    _DONE,
    _LT_AMBIENT,
    _LT_DIRECTIONAL,
    _LT_ENV_SH,
    _LT_HEMISPHERE,
    _GOLDEN_ANGLE,
    _material_reflectance,
    _offset_transmitted_origin,
    _refract_ray,
    _reserve_continuation_slot,
    _sample_env_map,
    _tri_color_g,
    _tri_extra_g,
    _tri_ior_g,
    _tri_normal_g,
    _tri_transmission_g,
)

# Candidate pixels per (prim, chunk) pair: one fine-raster thread tests at
# most this many pixels, bounding load imbalance for large bboxes.
RASTER_CHUNK = 32

# Empty typed visibility-buffer entry. Real hits pack the same strict ordering
# used by the classic deterministic tracer:
#
#   high 32 bits: floor(t / DEPTH_TIE_EPSILON)
#   low  32 bits: bitwise-inverted layer index (higher layer sorts first)
#
# Triangle layers are ``layer_offset_triangles + prim``; bezier layers are the
# circuit index.  This makes the atomic minimum layer-aware and lets the host
# recover the winning geometry type/id from the layer without a second winner
# buffer. Exact t/barycentrics are recomputed only for the terminal winner.
Z_SENTINEL = 0x7FFFFFFFFFFFFFFF

# Reach of the analytic-coverage box filter, in pixels: half the diagonal of the
# pixel square, so a nearest-edge query centred on the pixel centre is
# guaranteed to find every edge that can touch the square. (The filter itself
# uses the perpendicular half-pixel; this is the conservative query radius.)
_AA_FILTER_RADIUS = 0.7071067811865476

# A fragment whose analytic coverage is at least this is treated as covering the
# whole pixel: it may occupy the opaque z-prepass and truncate a sorted run.
AA_FULL_COVERAGE = 1.0 - 1e-6

# Mask word layout: bits 0..N-1 the sample set, then the flags below, which sit
# at a FIXED shift above the widest supported sample set so that changing the
# sample count cannot silently collide with them. One sample set suffices for
# both claiming and occluding because the fill rule makes the partition exact.
_AA_FLAG_SHIFT = 16

# Facing bit. Adjacent triangles of a manifold mesh wind consistently, so the
# sign of the projected screen area separates a closed mesh's two sheets exactly
# and for free -- it is already computed for the coverage orientation. (A depth
# window was tried first and rejected: no threshold in pixel footprints survives
# grazing incidence, where it splits genuine shared edges.)
#
# Now informational only: per-sample transmittance (ss18) needs no facing to keep
# a mesh's back faces out of its own silhouette, because each sample composites
# in true depth order. It is still written, so the resolve can distinguish the
# sheets if a future rule wants to.
_AA_BACKFACE_BIT = 1 << _AA_FLAG_SHIFT

# Marks a fragment whose coverage came from the sliver rule rather than from a
# sample it actually contains: an AREAL fraction of the pixel with no position in
# it, so it attenuates every sample uniformly rather than a subset exactly (the
# same treatment a circuit's SDF coverage gets). Only reachable when
# ANALYTIC_AA_SLIVER is not the default ``drop``.
_AA_SLIVER_BIT = 2 << _AA_FLAG_SHIFT

# Sample-less-triangle policies, matching ``settings.ANALYTIC_AA_SLIVER_MODES``.
# The live mode reaches ``_ss_pixel`` inside the ``aa`` template value (the
# geometry kernels are launched with ``1 + mode``) so that each policy gets its
# own compiled variant and its own offline-cache entry.
_AA_SLIVER_AREA = 0
_AA_SLIVER_EXACT = 1
_AA_SLIVER_DROP = 2
_AA_SLIVER_EXACT_OCC = 3

# Sub-pixel sample set. TRIANGLE coverage is the fraction of these points the
# triangle contains -- set arithmetic, not an area formula.
#
# A continuous area estimate was tried first and does not survive contact with a
# mesh, because an area alone cannot say WHERE in the pixel a fragment lies and
# the seam rule has to ADD sub-areas. Two failures follow directly. At a
# silhouette, several near-edge-on triangles each dilate half a pixel outward
# over the same sliver and their areas sum to a bright halo, dilating every mesh
# by a pixel. At a vertex, several wedges meet and their areas do not sum to the
# pixel, leaving a dark notch. Both are set-membership questions, so a set
# answers them: a fragment claims exactly the samples it contains that its group
# has not already taken. Coincident slivers claim the same sample once (no
# halo); the two halves of a shared edge claim complementary samples that sum to
# the whole pixel (no notch). Attempts to patch the area formula -- scaling by
# the fresh-sample ratio, bounding the group total by the sample count -- each
# fixed one failure and worsened the other; the set formulation has neither.
#
# The cost is real but small: the edge distances are affine in the sample
# position, so each sample is a few fused multiply-adds and no memory traffic --
# and still ONE shading per fragment, not one per sample, which is the whole
# point versus supersampling.
#
# Bezier circuits keep their continuous SDF coverage: they never group, so the
# set machinery buys them nothing, and the SDF is exact.

# Sub-pixel lattice for the exact coverage test: 1/4096 of a pixel. An edge
# function is a product of two coordinate differences, so at a 4096-wide screen
# (~2^24 lattice units) it reaches ~2^50 and the accumulator must be 64-bit --
# which is also the ceiling on how fine this can go.
#
# Resolution matters because a triangle whose projected area rounds to ZERO on
# the lattice has no usable edge functions at all. Near a silhouette, where the
# surface turns edge-on, triangles get arbitrarily foreshortened; at 1/256 that
# left visible holes around every rim.
_AA_FIXED_SHIFT = 12
_AA_FIXED_SCALE = float(1 << _AA_FIXED_SHIFT)
_AA_FIXED_HALF = 1 << (_AA_FIXED_SHIFT - 1)
_AA_Q_INF = 1 << 62

# The standard sparse sample positions (the D3D 8x and 16x patterns), in 16ths
# of a pixel from its centre: one sample per row and per column of an 8x8 / 16x16
# grid, which resolves edges of every orientation evenly.
#
# Which one is live is a COMPILE-TIME constant, deliberately not a setting: the
# count is baked into every kernel that rasterizes but is part of no template
# argument, so the offline cache would happily serve an 8-sample kernel to a
# 16-sample build. Changing it means editing this line and clearing the cache.
# 16 matches the sampling density of the anti_alias_level=4 reference; the
# measured difference is in DESIGN_analytic_aa.md ss16.
_AA_PATTERN_8 = (
    (1, -3), (-1, 3), (5, 1), (-3, -5), (-5, 5), (-7, -1), (3, 7), (7, -7),
)
_AA_PATTERN_16 = (
    (1, 1), (-1, -3), (-3, 2), (4, -1), (-5, -2), (2, 5), (5, 3), (3, -5),
    (-2, 6), (0, -7), (-4, -6), (-6, 4), (-8, 0), (7, -4), (6, 7), (-7, -8),
)
_AA_SAMPLES = tuple(
    (x * (1 << _AA_FIXED_SHIFT) // 16, y * (1 << _AA_FIXED_SHIFT) // 16)
    for x, y in _AA_PATTERN_8
)
_AA_NUM_SAMPLES = len(_AA_SAMPLES)
_AA_MASK_ALL = (1 << _AA_NUM_SAMPLES) - 1
_AA_SAMPLE_WEIGHT = 1.0 / _AA_NUM_SAMPLES


# Compare-swap network that sorts four values with five comparisons.
_AA_SORT4 = ((0, 1), (2, 3), (0, 2), (1, 3), (1, 2))


# Per-fragment debug dump (ALGAN_AA_DUMP="px,py,frame"; DESIGN_analytic_aa_v2.md
# ss7.1). Both walk kernels write one row per fragment they process at the
# requested pixel into a host-provided float buffer, so a golden host-side walk
# can recompute the pixel from the exact inputs the kernel saw and diff every
# column -- and the two walks' dumps can be diffed against each other, which is
# what catches a resolve/shadow-event desync on day one. Compiled out (a
# ``ti.static`` template flag) unless requested; the buffer's row 0 carries the
# target (px, py, frame) and the atomic row counter, so the feature costs one
# kernel argument.
#
# Row layout (_AA_DUMP_COLS wide):
#   [0] q      fragment index in the walk (the terminal row uses -1)
#   [1] kind   0 tri, 1 bez, 2 z-tri, 3 z-bez
#   [2] note   0 committed, 1 eff-skip, 2 bounce break, 3 occluded-glass
#              break, 4 far-clip break, 5 invalid, 6 seam skip
#   [3] ref    primitive / circuit id
#   [4] sid    surface id: tri_obj[ref] for a triangle, -1 - circuit for bez
#   [5] facing 1 when the fragment's backface bit is set
#   [6] msk    sample-mask low bits
#   [7] cov    frag_cov as emitted
#   [8] pop    popcount(msk)
#   [9] corr   the run correction applied (1.0 outside run mode)
#   [10] eff   effective coverage after per-sample transmittance (and corr)
#   [11] mat_alpha  material alpha before coverage
#   [12] alpha      committed alpha (mat_alpha * eff)
#   [13] trans_share
#   [14] refl_max
#   [15] t_hit
#   [16:24] svis after this fragment committed
# Terminal row: [-1, bounced, done, processed, vis_all, acc0, acc1, acc2,
#                acc3, w0, w1, w2, 0, 0, 0, 0, svis...]
_AA_DUMP_COLS = 24


@ti.func
def _aa_dump_match(dump_out: ti.template(), px, py, f):
    """Whether this thread's pixel is the one the dump targets (row 0)."""
    return ((px == ti.cast(dump_out[0, 0] + 0.5, ti.i32))
            and (py == ti.cast(dump_out[0, 1] + 0.5, ti.i32))
            and (f == ti.cast(dump_out[0, 2] + 0.5, ti.i32)))


@ti.func
def _aa_dump_reserve(dump_out: ti.template()):
    """Atomically reserve the next dump row; -1 when the buffer is full."""
    r = ti.cast(ti.atomic_add(dump_out[0, 3], 1.0) + 0.5, ti.i32) + 1
    if r >= dump_out.shape[0]:
        r = -1
    return r


@ti.func
def _aa_dump_frag(dump_out: ti.template(), q, kind, note, ref, sid, facing,
                  msk, cov, pop, corr, eff, mat_alpha, alpha, trans_share,
                  refl_max, t_hit, svis):
    r = _aa_dump_reserve(dump_out)
    if r >= 0:
        dump_out[r, 0] = ti.cast(q, ti.f32)
        dump_out[r, 1] = ti.cast(kind, ti.f32)
        dump_out[r, 2] = ti.cast(note, ti.f32)
        dump_out[r, 3] = ti.cast(ref, ti.f32)
        dump_out[r, 4] = ti.cast(sid, ti.f32)
        dump_out[r, 5] = ti.cast(facing, ti.f32)
        dump_out[r, 6] = ti.cast(msk & _AA_MASK_ALL, ti.f32)
        dump_out[r, 7] = cov
        dump_out[r, 8] = ti.cast(pop, ti.f32)
        dump_out[r, 9] = corr
        dump_out[r, 10] = eff
        dump_out[r, 11] = mat_alpha
        dump_out[r, 12] = alpha
        dump_out[r, 13] = trans_share
        dump_out[r, 14] = refl_max
        dump_out[r, 15] = t_hit
        for s in ti.static(range(_AA_NUM_SAMPLES)):
            dump_out[r, 16 + s] = svis[s]


@ti.func
def _aa_dump_terminal(dump_out: ti.template(), bounced, done, processed,
                      vis_all, acc, weight, svis):
    r = _aa_dump_reserve(dump_out)
    if r >= 0:
        dump_out[r, 0] = -1.0
        dump_out[r, 1] = 1.0 if bounced else 0.0
        dump_out[r, 2] = 1.0 if done else 0.0
        dump_out[r, 3] = ti.cast(processed, ti.f32)
        dump_out[r, 4] = vis_all
        for k in ti.static(range(4)):
            dump_out[r, 5 + k] = acc[k]
        for k in ti.static(range(3)):
            dump_out[r, 9 + k] = weight[k]
        for k in ti.static(range(4)):
            dump_out[r, 12 + k] = 0.0
        for s in ti.static(range(_AA_NUM_SAMPLES)):
            dump_out[r, 16 + s] = svis[s]


def _sliver_mode(aa):
    """Sample-less-triangle policy carried in the ``aa`` template value.

    ``aa`` is 0 (coverage off) or ``1 + mode + 4 * exact``; a plain int at
    kernel-compile time, so every use of this sits inside ``ti.static``.
    """
    return max(int(aa) - 1, 0) % 4


def _tri_cells(aa_tri):
    """Whether the resolve reads packed CELLS rather than a sample mask.

    Rides in the resolve's own ``aa_tri`` template (1 points, 2 cells, 3
    run-corrected) so representations cannot share a compiled kernel or an
    offline-cache entry.
    """
    return int(aa_tri) == 2


def _tri_run_mode(aa_tri):
    """Whether the resolve applies the RUN rule (DESIGN_analytic_aa_v2.md ss4):
    ownership and per-fragment magnitude from the mask, exact areas read only
    at run level. 3 = corr > 1 by scale-and-clamp (rule A), 4 = by
    redistribution onto the run's unowned samples (rule B) -- the ss4.4 open
    question, kept as separate compiled variants so the harness can decide."""
    return int(aa_tri) == 3 or int(aa_tri) == 4


def _tri_run_rule_b(aa_tri):
    """Whether the run rule redistributes clamped write residue (ss4.4 B)."""
    return int(aa_tri) == 4


def _tri_repr(aa):
    """Triangle representation carried in the geometry kernels' ``aa`` value:
    0 sampled points, 1 the parked cells emission, 2 run-corrected exact areas
    (``1 + sliver_mode + 4 * repr``, see :func:`_sliver_mode`)."""
    return max(int(aa) - 1, 0) // 4


def _tri_exact(aa):
    """Whether triangle coverage is the parked CELLS emission (repr 1).

    Rides in the same ``aa`` template value as the sliver policy (see
    :func:`_sliver_mode`) so that each combination compiles its own kernel and
    gets its own offline-cache entry.
    """
    return _tri_repr(aa) == 1


def _tri_run(aa):
    """Whether the geometry kernels emit run-corrected payloads (repr 2):
    exact clipped areas in ``frag_cov`` beside the untouched sample masks,
    slivers emitted as area donors at their clipped centroid."""
    return _tri_repr(aa) == 2


@ti.func
def _pixel_clip_area(vx, vy):
    """Exact area of (triangle n pixel square), the pixel centre at the origin.

    ``vx``/``vy`` are the triangle's projected vertices in pixels, already
    translated so the pixel square is [-0.5, 0.5]^2. Returns a value in [0, 1]
    that is ZERO for a triangle disjoint from the square and that SUMS EXACTLY
    over a tiling -- the two properties the continuous product-of-edge-distances
    form does not have. That form is a reconstruction filter: it deliberately
    spreads coverage half a pixel past the geometry, so a silhouette rim of
    foreshortened triangles, each tiling its neighbour, sums to far more than the
    area they actually cover and dilates the whole mesh (DESIGN_analytic_aa.md
    ss15.3).

    Method: the area is the boundary integral (1/2) o (x dy - y dx) taken around
    the triangle's outline PROJECTED ONTO THE SQUARE by a componentwise clamp.
    The clamp is the nearest-point map onto the square, so the part of the
    outline inside maps to itself while everything outside collapses onto the
    square's border, contributing nothing but the border runs that close the
    intersection -- the enclosed area is exactly (triangle n square). Sampled at
    the crossings of the four border lines, where the clamp changes linear
    branch, so each edge needs its four crossing parameters sorted (out-of-range
    ones clamp to an endpoint and drop out on their own, no branching).

    This is equivalent to Sutherland-Hodgman clipping followed by a shoelace,
    but keeps no vertex list -- hence no dynamically indexed local array, which
    Taichi handles poorly -- only a running accumulator.
    """
    acc = 0.0
    for k in ti.static(range(3)):
        ax = vx[ti.static(k)]
        ay = vy[ti.static(k)]
        bx = vx[ti.static((k + 1) % 3)]
        by = vy[ti.static((k + 1) % 3)]
        dx = bx - ax
        dy = by - ay
        # A segment parallel to a border line has no crossing with it; the zero
        # reciprocal sends both of that axis' parameters to an endpoint, which
        # the walk below skips over for free.
        invx = 0.0
        if ti.abs(dx) > 1e-20:
            invx = 1.0 / dx
        invy = 0.0
        if ti.abs(dy) > 1e-20:
            invy = 1.0 / dy
        tv = ti.math.vec4(
            ti.math.clamp((-0.5 - ax) * invx, 0.0, 1.0),
            ti.math.clamp((0.5 - ax) * invx, 0.0, 1.0),
            ti.math.clamp((-0.5 - ay) * invy, 0.0, 1.0),
            ti.math.clamp((0.5 - ay) * invy, 0.0, 1.0))
        for c in ti.static(range(len(_AA_SORT4))):
            i = ti.static(_AA_SORT4[c][0])
            j = ti.static(_AA_SORT4[c][1])
            if tv[i] > tv[j]:
                sw = tv[i]
                tv[i] = tv[j]
                tv[j] = sw
        cx = ti.math.clamp(ax, -0.5, 0.5)
        cy = ti.math.clamp(ay, -0.5, 0.5)
        for s in ti.static(range(4)):
            tt = tv[ti.static(s)]
            nx = ti.math.clamp(ax + dx * tt, -0.5, 0.5)
            ny = ti.math.clamp(ay + dy * tt, -0.5, 0.5)
            acc += cx * ny - nx * cy
            cx = nx
            cy = ny
        ex = ti.math.clamp(bx, -0.5, 0.5)
        ey = ti.math.clamp(by, -0.5, 0.5)
        acc += cx * ey - ex * cy
    # Sign follows the projected winding, which the caller knows from the exact
    # integer area; the magnitude is the same either way.
    return ti.min(ti.abs(acc) * 0.5, 1.0)


@ti.func
def _halfplane_clip_area(nx, ny, d):
    """Exact area of (half-plane n pixel square), the pixel centre at the origin.

    The covered side is ``{p : dot(n, p) + d >= 0}`` with ``n`` pointing INTO
    it, and ``d`` is the signed perpendicular distance from the pixel centre to
    the boundary line, in pixels, positive when the centre is covered.
    ``(nx, ny)`` needs no normalization -- only its direction is read.

    This is the one-crossing-edge case of :func:`_pixel_clip_area` in closed
    form.  A single straight boundary is what a pixel almost always contains, so
    it is the exact answer for a lone triangle edge and for a flattened outline
    segment, at a fraction of the cost of the full clip.

    Writing ``a >= b`` for the sorted components of the unit normal (so
    ``a^2 + b^2 = 1``, ``a >= 1/sqrt(2)``), the covered area is

        1                      d >= (a + b) / 2      the boundary clears the
        0                      d <= -(a + b) / 2     square entirely
        0.5 + d / a            |d| <= (a - b) / 2    a trapezoid: two opposite
                                                     sides of the square are cut
        dd^2 / (2 a b)         d < -(a - b) / 2      a corner triangle, dd being
        1 - dd^2 / (2 a b)     d >  (a - b) / 2      the normal distance from the
                                                     nearer corner to the line

    The branches meet continuously: at ``|d| = (a - b) / 2`` both forms give
    ``b / (2 a)``, and at ``|d| = (a + b) / 2`` the corner has shrunk to nothing.

    **This is the correction the circuit path needs.** For an AXIS-ALIGNED
    boundary ``b`` is zero, the trapezoid branch spans the whole square, and the
    result collapses to ``clamp(d + 0.5, 0, 1)`` -- which is the box filter
    :func:`_bez_pixel_hit` applies at EVERY orientation.  That filter is
    therefore exact for one orientation out of all of them: at 45 degrees it
    reports full coverage at ``d = 0.5``, where the truth is ``1 - (0.7071 -
    d)^2 = 0.957``.  The error peaks around 0.043 and is systematic in the edge's
    ANGLE, so it presents as diagonal edges carrying visibly different weight
    from horizontal ones rather than as noise.

    Verified against brute force, against :func:`_pixel_clip_area`, and for the
    complement property below by ``benchmarks/_aa_clip_area_check.py``.

    The property the seam rule needs: the two sides of one boundary sum to the
    whole pixel, ``A(n, d) + A(-n, -d) == 1``, and in the corner branches that
    holds BIT-EXACTLY (the pair computes an identical ``dd^2 / (2 a b)``, one
    returning it and the other returning one minus it).
    """
    ax = ti.abs(nx)
    ay = ti.abs(ny)
    inv_len = 1.0 / ti.max(ti.sqrt(ax * ax + ay * ay), 1e-30)
    a = ti.max(ax, ay) * inv_len
    b = ti.min(ax, ay) * inv_len
    # Half-width of the square along the normal, and the |d| below which the
    # boundary cuts two OPPOSITE sides rather than a corner.
    reach = 0.5 * (a + b)
    flat = 0.5 * (a - b)
    area = 0.0
    if d >= reach:
        area = 1.0
    elif d > -reach:
        if (d >= -flat) and (d <= flat):
            area = 0.5 + d / ti.max(a, 1e-30)
        else:
            # The cut corner's legs are dd/a and dd/b, so its area is
            # dd^2/(2ab). The division is unreachable at b == 0 -- there ``flat``
            # and ``reach`` coincide at 0.5 and the branches above take every
            # d -- but it is guarded anyway, since a predicated lane may still
            # evaluate it. Underflow is harmless for the same reason: dd <= b, so
            # a b small enough to flush dd^2 to zero has a true area of zero too.
            dd = reach - ti.abs(d)
            corner = dd * dd / ti.max(2.0 * a * b, 1e-30)
            if d < 0.0:
                area = corner
            else:
                area = 1.0 - corner
    return area


# Pixel CELLS: a 2x2 tiling of the pixel square, each cell half a pixel across.
#
# This is the representation that lets an exact area drive the resolve at all.
# Eight sample POINTS force a fragment's claim and its occlusion to be the same
# |M|/N -- consistent, but quantized to eighths, and an exact area cannot be
# substituted for either without breaking the other (ss21.3). A tiling of cells
# carrying exact clipped AREAS has all three properties at once: the claim sums
# to the true area exactly, no cell can exceed 1, and two triangles sharing an
# edge still partition every cell between them.
_AA_NUM_CELLS = 4
_AA_CELL_CENTRES = ((-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25))
_AA_CELL_HALF = 0.25
_AA_CELL_AREA = 0.25


@ti.func
def _cell_clip_area(vx, vy, k: ti.template()):
    """Exact area of (triangle n cell k), as a FRACTION of the cell.

    Needs no clipping code of its own: a cell is the pixel square translated to
    its centre and scaled by two, so mapping the triangle through that inverse
    turns the query straight back into :func:`_pixel_clip_area` -- already exact,
    already validated, and already known to sum over a tiling, which is the
    property the whole cell representation rests on.
    """
    cx = ti.static(_AA_CELL_CENTRES[k][0])
    cy = ti.static(_AA_CELL_CENTRES[k][1])
    return _pixel_clip_area(
        ti.math.vec3((vx[0] - cx) * 2.0, (vx[1] - cx) * 2.0, (vx[2] - cx) * 2.0),
        ti.math.vec3((vy[0] - cy) * 2.0, (vy[1] - cy) * 2.0, (vy[2] - cy) * 2.0))


@ti.func
def _pixel_clip_centroid(vx, vy):
    """Area and CENTROID of (triangle n pixel square), the centre at the origin.

    The centroid is what replaces the centroid-of-owned-samples that the point
    representation used to evaluate a fragment's depth and barycentrics at
    (ss15). That point has to lie INSIDE the triangle: the pixel centre lies
    outside a partially covering triangle about half the time, and there the
    plane intersection is an extrapolation past the geometry, which mis-sorts
    the two sheets of a closed mesh at every silhouette. An intersection of two
    convex sets is convex, so its centroid is inside both -- unlike an
    area-weighted average of CELL CENTRES, which can easily fall outside the
    triangle and would put the bug straight back.

    Same clamped boundary integral as :func:`_pixel_clip_area`, carrying the two
    first moments alongside the area; returns ``(area, cx, cy)``, the centroid
    falling back to the pixel centre for a degenerate (zero-area) clip.
    """
    acc = 0.0
    mx = 0.0
    my = 0.0
    for k in ti.static(range(3)):
        ax = vx[ti.static(k)]
        ay = vy[ti.static(k)]
        bx = vx[ti.static((k + 1) % 3)]
        by = vy[ti.static((k + 1) % 3)]
        dx = bx - ax
        dy = by - ay
        invx = 0.0
        if ti.abs(dx) > 1e-20:
            invx = 1.0 / dx
        invy = 0.0
        if ti.abs(dy) > 1e-20:
            invy = 1.0 / dy
        tv = ti.math.vec4(
            ti.math.clamp((-0.5 - ax) * invx, 0.0, 1.0),
            ti.math.clamp((0.5 - ax) * invx, 0.0, 1.0),
            ti.math.clamp((-0.5 - ay) * invy, 0.0, 1.0),
            ti.math.clamp((0.5 - ay) * invy, 0.0, 1.0))
        for c in ti.static(range(len(_AA_SORT4))):
            i = ti.static(_AA_SORT4[c][0])
            j = ti.static(_AA_SORT4[c][1])
            if tv[i] > tv[j]:
                sw = tv[i]
                tv[i] = tv[j]
                tv[j] = sw
        cx = ti.math.clamp(ax, -0.5, 0.5)
        cy = ti.math.clamp(ay, -0.5, 0.5)
        for s in ti.static(range(4)):
            tt = tv[ti.static(s)]
            nx = ti.math.clamp(ax + dx * tt, -0.5, 0.5)
            ny = ti.math.clamp(ay + dy * tt, -0.5, 0.5)
            cr = cx * ny - nx * cy
            acc += cr
            mx += (cx + nx) * cr
            my += (cy + ny) * cr
            cx = nx
            cy = ny
        ex = ti.math.clamp(bx, -0.5, 0.5)
        ey = ti.math.clamp(by, -0.5, 0.5)
        cr = cx * ey - ex * cy
        acc += cr
        mx += (cx + ex) * cr
        my += (cy + ey) * cr
    area = ti.abs(acc) * 0.5
    ox = 0.0
    oy = 0.0
    if area > 1e-9:
        # The 1/(6A) of the polygon centroid, with acc = 2A carrying the sign,
        # so the winding cancels instead of needing to be known.
        inv6 = 1.0 / (3.0 * acc)
        ox = mx * inv6
        oy = my * inv6
    return ti.min(area, 1.0), ox, oy


@ti.func
def _two_halfplane_area(n1x, n1y, d1, n2x, n2y, d2, a1, a2):
    """Exact area of (pixel square n H1 n H2), both in (unit normal, distance) form.

    ``a1``/``a2`` are the caller's already-computed single half-plane areas.

    Two regimes, because the apex of the wedge runs off to infinity as the lines
    approach parallel:

      * NEAR-PARALLEL -- the two half-planes' complements are disjoint, so
        inclusion-exclusion collapses to ``a1 + a2 - 1`` with no clipping at all.
        Verified exact to 2.8e-16 for antiparallel lines; the error grows with
        tilt (0.022 at 10 degrees, 0.049 at 20), which is what sets the cutoff.
      * OTHERWISE -- the wedge is a convex cone, and a cone is a TRIANGLE once
        its two rays are truncated far enough out that the truncation cannot
        reach the pixel. So the exact clipped area is :func:`_pixel_clip_area`,
        which is already exact and already validated, on a big triangle.
    """
    out = 0.0
    cross = n1x * n2y - n1y * n2x
    # Trivial containment FIRST, and not merely as an optimization. A second
    # segment somewhere else in the shape is usually tens of pixels away, which
    # puts the apex of the wedge tens of pixels away too; truncating its rays at
    # any fixed length then misses the pixel entirely and reports zero coverage
    # where the answer is "the first half-plane, unchanged". Short-circuiting
    # also bounds everything below: past here both lines genuinely cross the
    # square, so |d| < 0.71 and the apex is within 0.71*2/0.2 ~ 7 pixels.
    if a2 >= 1.0 - 1e-6:
        out = a1
    elif a1 >= 1.0 - 1e-6:
        out = a2
    elif (a1 <= 1e-6) or (a2 <= 1e-6):
        out = 0.0
    elif ti.abs(cross) < 0.2:
        # Near-parallel splits in two by the normals' SIGN, and conflating them
        # is wrong by up to 0.42. Facing each other (antiparallel) the two
        # complements are disjoint and inclusion-exclusion collapses; pointing
        # the same way they are NESTED, and the intersection is simply whichever
        # half-plane is the more restrictive.
        if (n1x * n2x + n1y * n2y) < 0.0:
            out = ti.math.clamp(a1 + a2 - 1.0, 0.0, 1.0)
        else:
            out = ti.min(a1, a2)
    else:
        inv = 1.0 / cross
        # Apex: the point on both lines.
        apx = (-d1 * n2y + d2 * n1y) * inv
        apy = (-n1x * d2 + n2x * d1) * inv
        # Each boundary ray runs along its own line, in the direction that lies
        # inside the OTHER half-plane; both signs follow from the cross product.
        s = 1.0
        if cross < 0.0:
            s = -1.0
        # Long enough that truncation cannot reach the square, short enough that
        # the shoelace keeps its precision in f32.
        r = 4.0 * (ti.abs(apx) + ti.abs(apy) + 2.0)
        t1x = -n1y * (s * r)
        t1y = n1x * (s * r)
        t2x = n2y * (s * r)
        t2y = -n2x * (s * r)
        out = _pixel_clip_area(
            ti.math.vec3(apx, apx + t1x, apx + t2x),
            ti.math.vec3(apy, apy + t1y, apy + t2y))
    return out


@ti.func
def _boundary_coverage(nx, ny, d, aa: ti.template()):
    """Pixel coverage by the drawn side of one boundary, at signed distance ``d``.

    ``aa == 2`` (``ANALYTIC_AA_EXACT``) takes the exact angle-aware area;
    anything else keeps the box filter ``clamp(d + 0.5, 0, 1)``, which IS that
    area for an axis-aligned boundary and an approximation at every other angle
    (see :func:`_halfplane_clip_area`).  The choice is a compile-time template
    value, so the two forms never share a compiled kernel or a cache entry.

    A boundary with no direction has nothing to orient: no outline segment came
    within the query radius, so the pixel is deep inside the shape or far outside
    it, and ``|d| >= 0.5`` makes both forms agree on 1 or 0 anyway.
    """
    c = 0.0
    if ti.static(int(aa) >= 2):
        if (nx == 0.0) and (ny == 0.0):
            c = ti.math.clamp(d + 0.5, 0.0, 1.0)
        else:
            c = _halfplane_clip_area(nx, ny, d)
    else:
        c = ti.math.clamp(d + 0.5, 0.0, 1.0)
    return c


@ti.func
def _popcount_samples(x):
    """Number of sub-pixel samples set, i.e. set bits in the low sample bits."""
    v = ti.cast(x, ti.u32) & ti.u32(_AA_MASK_ALL)
    v = v - ((v >> 1) & ti.u32(0x5555))
    v = (v & ti.u32(0x3333)) + ((v >> 2) & ti.u32(0x3333))
    v = (v + (v >> 4)) & ti.u32(0x0F0F)
    return ti.cast((v + (v >> 8)) & ti.u32(0x1F), ti.i32)


# Cell coverage packs into the SAME int32 lane the sample mask used: four cells
# at eight bits each, which is 1/255 per cell against the point set's 1/8 for the
# whole pixel. No new per-fragment storage.
_AA_CELL_QUANT = 255.0
_AA_CELLS_FULL = -1  # 0xFFFFFFFF: every cell fully covered


@ti.func
def _unpack_cells(msk):
    """Four packed cell coverages as a vec4 in [0, 1]."""
    u = ti.cast(msk, ti.u32)
    return ti.math.vec4(
        ti.cast(u & ti.u32(0xFF), ti.f32),
        ti.cast((u >> 8) & ti.u32(0xFF), ti.f32),
        ti.cast((u >> 16) & ti.u32(0xFF), ti.f32),
        ti.cast((u >> 24) & ti.u32(0xFF), ti.f32),
    ) * ti.static(1.0 / _AA_CELL_QUANT)


@ti.func
def _coverage_slots(cov, msk, areal, cells: ti.template(),
                    run: ti.template()):
    """A fragment's coverage as one value per resolve slot, both modes at once.

    Returns ``(slots, nsm)``: how much of each of the ``_AA_NUM_SAMPLES`` slots
    this fragment covers, and the sample population the tint term still needs.

    The resolve then reads every fragment the same way, with no branch::

        eff       = sum(slots[s] * svis[s]) / N
        svis[s]  *= 1 - alpha * slots[s]

    POINT mode fills a slot with the fragment's density where its mask is set and
    zero elsewhere, which is exactly what the masked and areal branches computed.

    CELL mode has only four values but writes each into TWO slots. That is not a
    hack for its own sake: it keeps every loop bound and the ``1/N`` in the
    formulas above untouched, so switching representations cannot perturb the
    point path, and the duplicated pair stays equal forever because both copies
    are always attenuated by the same factor. ``sum(slots)/N`` then works out to
    ``sum(cells)/4``, the fragment's true area.
    """
    slots = ti.Vector([0.0 for _ in range(_AA_NUM_SAMPLES)])
    nsm = 0
    dens = 1.0
    if ti.static(run and not cells):
        # RUN mode (DESIGN_analytic_aa_v2.md ss3): a triangle fragment's
        # ownership, occlusion AND per-fragment magnitude all come from the
        # mask at literal density 1 -- the exact area in ``cov`` is read only
        # by the run scan, never reconciled per fragment (the ss21.3 failure,
        # 5920 notches). An empty mask (a sliver donor) covers no slot and so
        # commits nothing here; circuits keep their exact areal scalar.
        if areal:
            for s in ti.static(range(_AA_NUM_SAMPLES)):
                slots[s] = 1.0
            nsm = _AA_NUM_SAMPLES
            dens = cov
        else:
            for s in ti.static(range(_AA_NUM_SAMPLES)):
                if (msk >> s) & 1:
                    slots[s] = 1.0
            nsm = _popcount_samples(msk)
    elif ti.static(cells):
        if areal:
            # A circuit writes no mask lane at all, and a scalar coverage has no
            # position in the pixel, so it spreads over every cell -- the same
            # reading it has always had.
            for s in ti.static(range(_AA_NUM_SAMPLES)):
                slots[s] = cov
        else:
            cv = _unpack_cells(msk)
            for k in ti.static(range(_AA_NUM_CELLS)):
                slots[ti.static(2 * k)] = cv[ti.static(k)]
                slots[ti.static(2 * k + 1)] = cv[ti.static(k)]
        nsm = _AA_NUM_SAMPLES
    else:
        # An INDICATOR here, with the magnitude left in ``dens``, which is what
        # keeps the point path bit-identical: ``eff`` stays "sum the visible
        # samples, then scale once", and the slots a fragment does not own
        # contribute an exact 0.0 to that sum rather than reordering it.
        cmsk, n, d = _coverage_density(cov, msk, areal)
        nsm = n
        dens = d
        for s in ti.static(range(_AA_NUM_SAMPLES)):
            if (cmsk >> s) & 1:
                slots[s] = 1.0
    return slots, nsm, dens


@ti.func
def _coverage_density(cov, msk, areal):
    """Split a fragment's coverage into WHERE it sits and HOW MUCH covers it.

    Returns ``(cmsk, nsm, dens)``: the sub-pixel samples this fragment
    attenuates, how many that is, and the fraction of each one it covers.  The
    resolve then reads a fragment as ``dens`` on ``cmsk`` and zero elsewhere::

        eff = dens * sum(svis[s] for s in cmsk) / N
        for s in cmsk: svis[s] *= 1 - alpha * dens

    The two quantities are INDEPENDENT, which is the whole point.  A mask can
    only express coverage in multiples of 1/N, so as long as it carried the
    magnitude too, an exact area had nowhere to go; splitting them lets the mask
    answer the set question (which part of the pixel, hence which fragment
    occludes which) while ``cov`` answers the measure question exactly.

    It also unifies the two fragment kinds that used to need separate branches
    at five sites each:

      * A mask whose popcount IS the coverage gives ``dens == 1`` -- what every
        triangle produces today, and what the masked branch assumed.
      * An AREAL fragment -- a circuit's SDF coverage, or a sliver's clipped
        area, both a fraction of the pixel with no position in it -- spreads over
        every sample, giving ``cmsk == all`` and ``dens == cov``, which is what
        the areal branch computed.

    Both come out BIT-IDENTICAL to the branches they replace (``cov`` is a
    multiple of 1/N in the first case and the mask is already full in the
    second), so this is a pure refactor until the geometry paths start emitting
    exact areas.
    """
    cmsk = msk
    if areal:
        cmsk = _AA_MASK_ALL
    nsm = _popcount_samples(cmsk)
    dens = 0.0
    if nsm > 0:
        # (cov * N) / nsm, and the grouping is load-bearing: when the mask IS
        # the coverage, cov is a multiple of 1/N, so cov * N is exact and the
        # quotient is exactly 1.0. Written cov * (N / nsm) it would round 8/3
        # first and land at 0.99999997, which is not the same render.
        dens = ti.min(
            cov * ti.static(float(_AA_NUM_SAMPLES)) / ti.cast(nsm, ti.f32), 1.0)
    # The clamp is inert while coverage comes from the mask, and becomes load
    # bearing with exact areas: a triangle can cover more of the pixel than its
    # share of the samples suggests, and a sample cannot be covered twice.
    return cmsk, nsm, dens


# Cap on the run scan's lookahead (DESIGN_analytic_aa_v2.md ss4.2). Runs are a
# few fragments at ordinary silhouettes; past the cap the remainder simply
# stays uncorrected -- shipped behavior, graceful. A module constant, not a
# setting: it is baked into the compiled kernels and part of no template
# argument (the _AA_SAMPLES cache-trap rule).
_AA_MAX_RUN_SCAN = 16


@ti.func
def _aa_run_scan(j0, nrun, start, sid0, face0, to_row,
                 frag_ref: ti.template(), frag_cov: ti.template(),
                 frag_msk: ti.template(), tri_obj: ti.template()):
    """Scan the RUN starting at fragment ``j0``: consecutive triangle
    fragments sharing (source surface, facing). Returns ``(E, U, end)`` --
    the exact-area sum (sliver donors included), the union of sample masks
    (disjoint within a sheet by the fill rule; OR is robust to mis-sorts),
    and the exclusive end index. Index arithmetic plus coherent loads; the
    z-winner (j == nrun), any circuit fragment, and any (sid, facing) change
    terminate it.
    """
    E = 0.0
    U = 0
    j = j0
    cnt = 0
    going = True
    while going and (j < nrun) and (cnt < _AA_MAX_RUN_SCAN):
        rf = frag_ref[start + j]
        going = rf >= 0
        if going:
            going = tri_obj[to_row, rf] == sid0
        if going:
            mj = frag_msk[start + j]
            fj = 0
            if (mj & _AA_BACKFACE_BIT) != 0:
                fj = 1
            going = fj == face0
            if going:
                E += frag_cov[start + j]
                U |= mj & _AA_MASK_ALL
                j += 1
                cnt += 1
    return E, U, j


@ti.func
def _run_svis_write(svis: ti.template(), slots, a_s, trans_share, cfac,
                    rule_b: ti.template()):
    """The walk's per-sample write with the run correction folded in.

    ``svis[s] *= (1 - ak) + ak * trans_share`` with ``ak = cfac * a_s *
    slots[s]`` -- at ``cfac == 1`` this is bit-for-bit the shipped write for
    both the opaque (``trans_share == 0``) and transmitting forms. corr > 1
    can push a factor negative; rule A clamps it at zero (the claim stays
    exact, the leftover keeps a bounded residual), rule B additionally
    returns the clamped-away amount so the caller can push it onto the run's
    unowned samples at run end (ss4.4).
    """
    resid = 0.0
    for s in ti.static(range(_AA_NUM_SAMPLES)):
        ak = cfac * a_s * slots[s]
        fct = (1.0 - ak) + ak * trans_share
        if ti.static(rule_b):
            if fct < 0.0:
                resid -= fct * svis[s]
                fct = 0.0
        else:
            fct = ti.max(fct, 0.0)
        svis[s] *= fct
    return resid


@ti.func
def _run_redistribute(svis: ti.template(), run_U, resid):
    """Rule B's run-end step: remove the clamped residue from the samples the
    run did NOT own, capping at zero (residue beyond their total is dropped
    -- the owned samples are already at zero and cannot give more)."""
    if resid > 0.0:
        tot = 0.0
        for s in ti.static(range(_AA_NUM_SAMPLES)):
            if ((run_U >> s) & 1) == 0:
                tot += svis[s]
        if tot > 1e-12:
            sc = ti.max(1.0 - resid / tot, 0.0)
            for s in ti.static(range(_AA_NUM_SAMPLES)):
                if ((run_U >> s) & 1) == 0:
                    svis[s] *= sc


@ti.func
def _order_key(t, layer):
    """Packed transitive depth-bin / descending-layer key."""
    # Keep bit 63 clear so signed ``atomic_min`` preserves unsigned depth
    # order even for pathological distances.  Layers share the 32-bit packed
    # representation used by the classic renderer.
    depth_f = ti.floor(t * INV_DEPTH_TIE_EPSILON)
    depth_bin = ti.cast(ti.math.clamp(depth_f, 0.0, 2147483647.0), ti.u64)
    layer_u = ti.cast(ti.min(ti.max(layer, 0), 2147483647), ti.u64)
    return ti.cast((depth_bin << 32) | (ti.u64(0xFFFFFFFF) - layer_u), ti.i64)


@ti.func
def _frag_t(key):
    """Recover exact positive f32 distance from a fragment's low key bits."""
    return ti.bit_cast(ti.cast(ti.cast(key, ti.u64) & ti.u64(0xFFFFFFFF), ti.u32), ti.f32)


# Quantization of a circuit fragment's border/fill blend weight, which rides in
# the low bits of its packed ref rather than in a per-fragment lane of its own:
# 8 bits resolve the blend finer than the framebuffer it lands in, and the
# circuit id has 23 bits left over, far more than a batch can hold.
_BEZ_BORDER_LEVELS = 255
_BEZ_BORDER_BITS = 8


@ti.func
def _pack_bez_ref(circuit, border_frac):
    """Negative typed fragment ref with the border weight folded into the id."""
    q = ti.cast(ti.round(ti.math.clamp(border_frac, 0.0, 1.0)
                         * _BEZ_BORDER_LEVELS), ti.i32)
    return -((circuit << _BEZ_BORDER_BITS) + q + 1)


@ti.func
def _decode_bez_ref(ref):
    """Inverse of :func:`_pack_bez_ref`: ``(circuit, border_frac)``."""
    code = -ref - 1
    return (code >> _BEZ_BORDER_BITS,
            ti.cast(code & _BEZ_BORDER_LEVELS, ti.f32)
            * (1.0 / _BEZ_BORDER_LEVELS))


@ti.func
def _decode_z_layer(zkey, layer_offset_triangles):
    inv_layer = ti.cast(zkey, ti.u64) & ti.u64(0xFFFFFFFF)
    layer = ti.cast(ti.u64(0xFFFFFFFF) - inv_layer, ti.i32)
    is_bez = layer < ti.cast(layer_offset_triangles, ti.i32)
    prim = layer
    if not is_bez:
        prim = layer - ti.cast(layer_offset_triangles, ti.i32)
    return is_bez, prim


@ti.func
def _ss_setup(f, prim, ss_enabled: ti.template(), tri_pos: ti.template(),
              tri_screen: ti.template(), cam_origin: ti.template(),
              aa: ti.template()):
    """Load the per-(frame, triangle) projection prepared once by the host.

    ``tri_screen[..., 0:3]`` are sx, ``3:6`` sy, ``6:9`` reciprocal
    perspective divisors, and column 9 is a validity flag.  World vertices are
    still read from ``tri_pos`` for hit reconstruction and the ray-cast
    fallback.  This removes repeated camera projection setup from every z,
    count, write, shadow-event, and resolve chunk.

    Under analytic coverage (``aa``) columns 10:13 additionally hold the
    reciprocal screen edge lengths, returned as ``il``; the host only allocates
    them when the feature is on, so ``aa`` must be derived from the table width
    the host produced rather than the live setting.
    """
    tp = f % tri_pos.shape[0]
    ts = f % tri_screen.shape[0]
    v0 = ti.math.vec3(tri_pos[tp, prim, 0], tri_pos[tp, prim, 1],
                      tri_pos[tp, prim, 2])
    v1 = ti.math.vec3(tri_pos[tp, prim, 3], tri_pos[tp, prim, 4],
                      tri_pos[tp, prim, 5])
    v2 = ti.math.vec3(tri_pos[tp, prim, 6], tri_pos[tp, prim, 7],
                      tri_pos[tp, prim, 8])
    cam_o = ti.math.vec3(cam_origin[f, 0], cam_origin[f, 1], cam_origin[f, 2])
    vm = ti.Matrix([[v0[0], v0[1], v0[2]],
                    [v1[0], v1[1], v1[2]],
                    [v2[0], v2[1], v2[2]]])
    sm = ti.Matrix.zero(ti.f32, 3, 3)
    il = ti.math.vec3(0.0, 0.0, 0.0)
    use_ss = 0
    if ti.static(ss_enabled):
        if tri_screen[ts, prim, 9] > 0.5:
            use_ss = 1
            for i in ti.static(range(3)):
                sm[0, i] = tri_screen[ts, prim, i]
                sm[1, i] = tri_screen[ts, prim, 3 + i]
                sm[2, i] = tri_screen[ts, prim, 6 + i]
            if ti.static(aa):
                for i in ti.static(range(3)):
                    il[i] = tri_screen[ts, prim, 10 + i]
    return use_ss, sm, vm, cam_o, il


@ti.func
def _ss_pixel(px, py, sm, vm, cam_o, il, aa: ti.template(),
              store_exact: ti.template()):
    """Screen-space test of one pixel against the pre-projected triangle.

    Edge functions give the 2D barycentric weights; the perspective-correct 3D
    weights are ``w_i = (E_i / d_i) / sum(E_j / d_j)`` (the shared screen-space
    area cancels). The 3D hit point ``H = sum w_i V_i`` gives the exact ray
    distance ``t = |H - cam_o|`` and barycentrics, matching Moller-Trumbore to
    float epsilon. Returns ``(ok, t, w1, w2, cov)``.

    Analytic coverage (``aa``, DESIGN_analytic_aa.md ss3). ``e_i`` is the cross
    product of edge ``i`` with the vector to the pixel centre, so ``e_i / |edge
    i|`` is the perpendicular distance in pixels, signed by the winding.  The
    winding is ``sign(e0 + e1 + e2)``, which is twice the signed screen area and
    therefore constant over the triangle -- NOT ``sign(s)``, whose perspective
    weights only agree for interior points.  Coverage is the product of the
    three clamped edge distances: exact whenever a single edge crosses the
    pixel, which is the overwhelmingly common case, and -- the property the
    seam rule depends on -- the two coverages either side of a shared edge sum
    to exactly 1.

    The hit is then accepted whenever coverage is non-zero, so the pixel CENTRE
    may lie outside the triangle.  ``t`` and the barycentrics are therefore
    evaluated at the CENTROID OF THE OWNED SAMPLES instead -- a point that is
    inside the triangle by construction, so neither the depth nor the
    barycentrics are ever extrapolations of the plane past its own edges (see
    the comment at that step for the silhouette-ordering failure this exists to
    prevent).  A fully covered fragment's centroid IS the pixel centre, so only
    partially covering ones move.  The returned barycentrics are still projected
    onto the simplex, because they go on to index colours, normals and UVs and
    must not sample outside the triangle.
    """
    qx = ti.cast(px, ti.f32) + 0.5
    qy = ti.cast(py, ti.f32) + 0.5
    sx0, sx1, sx2 = sm[0, 0], sm[0, 1], sm[0, 2]
    sy0, sy1, sy2 = sm[1, 0], sm[1, 1], sm[1, 2]
    e0 = (sx2 - sx1) * (qy - sy1) - (sy2 - sy1) * (qx - sx1)
    e1 = (sx0 - sx2) * (qy - sy2) - (sy0 - sy2) * (qx - sx2)
    e2 = (sx1 - sx0) * (qy - sy0) - (sy1 - sy0) * (qx - sx0)
    n0 = e0 * sm[2, 0]
    n1 = e1 * sm[2, 1]
    n2 = e2 * sm[2, 2]
    s = n0 + n1 + n2
    ok = 0
    t = 0.0
    w1 = 0.0
    w2 = 0.0
    cov = 1.0
    msk = _AA_MASK_ALL
    if ti.abs(s) > 1e-30:
        inv = 1.0 / s
        b0 = n0 * inv
        b1 = n1 * inv
        b2 = n2 * inv
        accept = ((b1 >= -BARYCENTRIC_EPSILON) and (b2 >= -BARYCENTRIC_EPSILON)
                  and (b0 >= -BARYCENTRIC_EPSILON))
        c = 1.0
        m = _AA_MASK_ALL
        if ti.static(aa):
            # Cheap conservative reject in float first: no sample can be inside
            # unless every edge is within half a pixel diagonal of the centre.
            # This keeps the exact path off the overwhelming majority of
            # candidates, which are plain misses. Two-sided because the winding
            # is not known yet -- it comes from the exact integer area below,
            # the float one being untrustworthy -- so reject only when the
            # centre is far outside under BOTH readings.
            d0 = e0 * il[0]
            d1 = e1 * il[1]
            d2 = e2 * il[2]
            accept = (
                ((d0 > -0.7072) and (d1 > -0.7072) and (d2 > -0.7072))
                or ((d0 < 0.7072) and (d1 < 0.7072) and (d2 < 0.7072)))
            if accept:
                # Keep the exact path for every accepted candidate. A
                # full-interior branch adds enough edge-support arithmetic to
                # regress dense moving meshes despite skipping sample tests.
                if ti.static(aa):
                    # EXACT fixed-point coverage (DESIGN_analytic_aa.md ss15).
                #
                # Snap the projected vertices to a 1/256-pixel integer lattice
                # and evaluate the edge functions in int64. Two triangles that
                # share an edge traverse it in opposite directions, and in
                # exact integer arithmetic their edge functions are then exact
                # negatives -- because E = D x (Q - V1) and the reversed edge
                # gives -D x (Q - V2) = -(D x (Q - V1)) once D x D = 0 drops
                # out. In FLOAT they are merely near-negatives, which is what
                # made a sample lying on a shared edge belong to whichever
                # triangle rounding happened to favour, and forced a choice
                # between a silhouette halo and speckle along every shared edge.
                #
                # Exactness lets the classic top-left fill rule settle it: a
                # sample exactly on an edge counts only for the triangle whose
                # traversal of that edge runs "down", or runs "left" when
                # horizontal. The neighbour traverses it the other way, so
                # exactly one of the pair takes the sample -- the masks
                # partition the pixel with no epsilon anywhere, and the same
                # mask can serve both as what a fragment claims and as what it
                # occludes.
                #
                # This relies on adjacent triangles carrying bit-identical
                # coordinates for a shared vertex. They do: the merged soup
                # stores each triangle's own copy, but from the same source
                # vertex, and the projection is one elementwise torch
                # expression. If that ever stopped holding, the failure is
                # graceful -- back to the float-era ambiguity on those edges.
                    fx0 = ti.cast(ti.round(sx0 * _AA_FIXED_SCALE), ti.i64)
                    fx1 = ti.cast(ti.round(sx1 * _AA_FIXED_SCALE), ti.i64)
                    fx2 = ti.cast(ti.round(sx2 * _AA_FIXED_SCALE), ti.i64)
                    fy0 = ti.cast(ti.round(sy0 * _AA_FIXED_SCALE), ti.i64)
                    fy1 = ti.cast(ti.round(sy1 * _AA_FIXED_SCALE), ti.i64)
                    fy2 = ti.cast(ti.round(sy2 * _AA_FIXED_SCALE), ti.i64)
                    qxf = ((ti.cast(px, ti.i64) << _AA_FIXED_SHIFT)
                           + _AA_FIXED_HALF)
                    qyf = ((ti.cast(py, ti.i64) << _AA_FIXED_SHIFT)
                           + _AA_FIXED_HALF)
                    ex0 = fx2 - fx1
                    ey0 = fy2 - fy1
                    ex1 = fx0 - fx2
                    ey1 = fy0 - fy2
                    ex2 = fx1 - fx0
                    ey2 = fy1 - fy0
                    r0 = ex0 * (qyf - fy1) - ey0 * (qxf - fx1)
                    r1 = ex1 * (qyf - fy2) - ey1 * (qxf - fx2)
                    r2 = ex2 * (qyf - fy0) - ey2 * (qxf - fx0)
                # Orientation from the EXACT integer sum -- which is twice the
                # lattice signed area, and is the ONLY safe source for it. The
                # float sum is the same quantity but formed from three large
                # cancelling products, so its sign is unreliable for the small
                # thin triangles a dense mesh is made of. Two neighbours
                # disagreeing about their winding is exactly what stops the
                # fill rule partitioning a shared edge, and it was measured
                # doing so (scratch fill-rule harness: every double-claimed
                # sample traced back to an orientation disagreement).
                    area2 = r0 + r1 + r2
                    oi = ti.i64(1)
                    if area2 < 0:
                        oi = ti.i64(-1)
                    ec0 = oi * r0
                    ec1 = oi * r1
                    ec2 = oi * r2
                    gx0 = oi * ex0
                    gy0 = oi * ey0
                    gx1 = oi * ex1
                    gy1 = oi * ey1
                    gx2 = oi * ex2
                    gy2 = oi * ey2
                # Top-left rule as a +1 bias: a sample exactly on an owned edge
                # (q == 0) then tests strictly positive, and on a disowned one
                # it does not.
                # NOTE the names: b0/b1/b2 above are the float barycentrics and
                # must not be touched -- assigning an integer into them would
                # keep them f32, silently dragging every "exact" value below
                # into float and corrupting the barycentrics on the way out.
                    tl0 = ti.i64(0)
                    if (gy0 > 0) or ((gy0 == 0) and (gx0 < 0)):
                        tl0 = ti.i64(1)
                    tl1 = ti.i64(0)
                    if (gy1 > 0) or ((gy1 == 0) and (gx1 < 0)):
                        tl1 = ti.i64(1)
                    tl2 = ti.i64(0)
                    if (gy2 > 0) or ((gy2 == 0) and (gx2 < 0)):
                        tl2 = ti.i64(1)
                    m = 0
                    best_k = 0
                    best_q = ti.i64(-_AA_Q_INF)
                    # Running centroid of the OWNED samples, in lattice units
                    # from the pixel centre. It is what the fragment's depth and
                    # barycentrics are evaluated at (see below); zero for a fully
                    # covered pixel, because the sample pattern is centred.
                    sox = 0.0
                    soy = 0.0
                    nsm = 0
                    for k in ti.static(range(len(_AA_SAMPLES))):  # noqa: B007
                        ox = ti.static(_AA_SAMPLES[k][0])
                        oy = ti.static(_AA_SAMPLES[k][1])
                        q0 = ec0 + gx0 * oy - gy0 * ox + tl0
                        q1 = ec1 + gx1 * oy - gy1 * ox + tl1
                        q2 = ec2 + gx2 * oy - gy2 * ox + tl2
                        qq = ti.min(q0, ti.min(q1, q2))
                        if qq > 0:
                            m |= 1 << k
                            sox += ti.static(float(ox))
                            soy += ti.static(float(oy))
                            nsm += 1
                    # Only the sliver policies need to know which sample the
                    # triangle came closest to.
                        if ti.static(_sliver_mode(aa) != _AA_SLIVER_DROP):
                            if qq > best_q:
                                best_q = qq
                                best_k = k
                    if area2 == 0:
                    # Foreshortened to zero area on the lattice. Its edge
                    # functions are all zero, so the top-left bias alone would
                    # otherwise let it claim the entire pixel. Clearing the set
                    # hands it to the sample-less policy below, which under the
                    # default drops it -- an error bounded by one lattice unit
                    # (1/4096 px), not a hole.
                        m = 0
                        sox = 0.0
                        soy = 0.0
                        nsm = 0
                    c = (ti.cast(_popcount_samples(m), ti.f32)
                         * _AA_SAMPLE_WEIGHT)
                    if ti.static(_tri_run(aa)):
                    # RUN-CORRECTED payload (DESIGN_analytic_aa_v2.md ss4.1).
                    # MASKED fragments leave here with their SAMPLED coverage:
                    # the keep decisions key on it in count and write alike,
                    # and only the WRITE kernel -- after its keep/z/alpha
                    # culls, on actually-stored fragments -- computes the
                    # exact lane (_run_exact_cov; clipping per CANDIDATE here
                    # cost count +36% / write +42% device on the meshes A/B).
                    # Sample-less DONORS are the exception: their acceptance
                    # IS "exact area > 0", so both kernels clip them, behind
                    # the one-sided oriented reject.
                    # the mask, fill rule and owned-sample centroid stay
                    # exactly as shipped -- ownership and occlusion never
                    # change -- while frag_cov carries the EXACT clipped area
                    # for the resolve's run scan to sum.
                    #
                    #   * A FULL mask whose edges all clear the half-diagonal
                    #     is exactly 1.0 with no clip at all (the hot path);
                    #     otherwise the exact area, snapped to 1.0 from
                    #     0.9999 so interior fragments stay bit-clean for the
                    #     cov-keyed full-coverage gates.
                    #   * A PARTIAL mask takes the exact area, falling back to
                    #     the sampled count if the float clip degenerates to
                    #     zero against the lattice (a mask-owning fragment
                    #     must never be dropped -- its samples would show the
                    #     background through the surface).
                    #   * An EMPTY mask (a sliver, including the
                    #     lattice-degenerate area2 == 0 case) is EMITTED as an
                    #     area donor: exact area, depth and barycentrics at
                    #     the centroid of (triangle n pixel), which is inside
                    #     the triangle, so the ss15 ordering argument holds.
                        if (m == _AA_MASK_ALL) and ti.static(store_exact):
                            ffl = 1.0
                            if oi < 0:
                                ffl = -1.0
                            if ((ffl * d0 >= 0.7072)
                                    and (ffl * d1 >= 0.7072)
                                    and (ffl * d2 >= 0.7072)):
                                c = 1.0
                            else:
                                ca = _pixel_clip_area(
                                    ti.math.vec3(
                                        sx0 - qx, sx1 - qx, sx2 - qx),
                                    ti.math.vec3(
                                        sy0 - qy, sy1 - qy, sy2 - qy))
                                c = ca
                                if (c >= 0.9999) or (c <= 0.0):
                                    c = 1.0
                        elif (m != 0) and ti.static(store_exact):
                            # Exact area of (triangle n pixel), stored for the
                            # run scan, cheapest-first: a microtriangle FULLY
                            # INSIDE the square (the sub-pixel-diced majority)
                            # is a bare shoelace; a fragment cut by ONE edge
                            # with the other two clear (the big-triangle
                            # silhouette case) is the half-plane closed form
                            # riding the edge distances already computed; only
                            # genuine corner/straddle fragments pay the clip.
                            ffl = 1.0
                            if oi < 0:
                                ffl = -1.0
                            od0 = ffl * d0
                            od1 = ffl * d1
                            od2 = ffl * d2
                            ca = -1.0
                            ax_ = sx0 - qx
                            ay_ = sy0 - qy
                            bx_ = sx1 - qx
                            by_ = sy1 - qy
                            cx2 = sx2 - qx
                            cy2 = sy2 - qy
                            if ((ti.abs(ax_) <= 0.5) and (ti.abs(ay_) <= 0.5)
                                    and (ti.abs(bx_) <= 0.5)
                                    and (ti.abs(by_) <= 0.5)
                                    and (ti.abs(cx2) <= 0.5)
                                    and (ti.abs(cy2) <= 0.5)):
                                ca = 0.5 * ti.abs(
                                    (bx_ - ax_) * (cy2 - ay_)
                                    - (by_ - ay_) * (cx2 - ax_))
                            elif ((od0 < 0.7072) and (od1 >= 0.7072)
                                    and (od2 >= 0.7072)):
                                ca = _halfplane_clip_area(
                                    -ffl * (sy2 - sy1), ffl * (sx2 - sx1),
                                    od0)
                            elif ((od1 < 0.7072) and (od0 >= 0.7072)
                                    and (od2 >= 0.7072)):
                                ca = _halfplane_clip_area(
                                    -ffl * (sy0 - sy2), ffl * (sx0 - sx2),
                                    od1)
                            elif ((od2 < 0.7072) and (od0 >= 0.7072)
                                    and (od1 >= 0.7072)):
                                ca = _halfplane_clip_area(
                                    -ffl * (sy1 - sy0), ffl * (sx1 - sx0),
                                    od2)
                            if ca < 0.0:
                                ca = _pixel_clip_area(
                                    ti.math.vec3(ax_, bx_, cx2),
                                    ti.math.vec3(ay_, by_, cy2))
                            if ca > 0.0:
                                c = ca
                        elif m == 0:
                            # One-sided oriented reject BEFORE any clipping:
                            # the conservative pre-test above is two-sided
                            # (winding unknown there), so on a dense mesh most
                            # sample-less candidates reaching here are plain
                            # misses on the correct winding -- and they were
                            # the bulk of the run representation's emission
                            # cost (tri_count +36% on the meshes A/B before
                            # this gate, +9% after).
                            # (The lattice-degenerate area2 == 0 case has no
                            # trustworthy winding, so it keeps the two-sided
                            # acceptance it already passed.)
                            ffl2 = 1.0
                            if oi < 0:
                                ffl2 = -1.0
                            if (area2 == 0) or ((ffl2 * d0 > -0.7072)
                                                and (ffl2 * d1 > -0.7072)
                                                and (ffl2 * d2 > -0.7072)):
                                if ti.static(store_exact):
                                    # WRITE needs the centroid (the donor's
                                    # depth/barycentric point); COUNT only
                                    # needs the acceptance bit, and the
                                    # moment-free clip keeps its register
                                    # footprint down.
                                    ca, cx_, cy_ = _pixel_clip_centroid(
                                        ti.math.vec3(
                                            sx0 - qx, sx1 - qx, sx2 - qx),
                                        ti.math.vec3(
                                            sy0 - qy, sy1 - qy, sy2 - qy))
                                    if ca > 0.0:
                                        c = ca
                                        nsm = 1
                                        sox = cx_ * ti.static(_AA_FIXED_SCALE)
                                        soy = cy_ * ti.static(_AA_FIXED_SCALE)
                                else:
                                    ca = _pixel_clip_area(
                                        ti.math.vec3(
                                            sx0 - qx, sx1 - qx, sx2 - qx),
                                        ti.math.vec3(
                                            sy0 - qy, sy1 - qy, sy2 - qy))
                                    if ca > 0.0:
                                        c = ca
                    if ti.static(_tri_exact(aa)):
                    # CELLS instead of points (DESIGN_analytic_aa.md ss21.4).
                    #
                    # The exact area cannot simply replace the sample count: a
                    # fragment's claim and its occlusion have to be the same
                    # quantity, and with point samples both are |M|/N, so
                    # substituting an area for one of them makes a boundary
                    # pixel stop summing to 1 (ss21.3, measured at 6000 interior
                    # notches). Cells carry an exact area PER SLOT, so claim and
                    # occlusion are the same number again, no slot can exceed 1,
                    # and two triangles sharing an edge still partition every
                    # cell between them.
                    #
                    # The top-left fill rule above is dead weight in this mode --
                    # it exists to make binary point ownership exact at a shared
                    # edge, which clipped areas summing over a tiling give for
                    # free -- but it is left running rather than compiled out
                    # until the mode is the default, so the two paths stay
                    # comparable.
                        vxx = ti.math.vec3(sx0 - qx, sx1 - qx, sx2 - qx)
                        vyy = ti.math.vec3(sy0 - qy, sy1 - qy, sy2 - qy)
                        # The EXACT clipped area, and nothing else: the resolve
                        # sums these within a surface, so no sub-pixel position
                        # is needed. The mask lane carries only the facing bit,
                        # which is what keeps a closed mesh's two sheets apart --
                        # packing anything else into it (an earlier revision
                        # packed per-cell areas) collides with the flag bit and
                        # scrambles the sheet grouping.
                        m = 0
                        if oi < 0:
                            m = _AA_BACKFACE_BIT
                        # Depth and barycentrics move to the centroid of the
                        # CLIPPED REGION, which is inside the triangle because an
                        # intersection of convex sets is convex. The sample
                        # centroid did the same job for the same reason (ss15);
                        # an area-weighted average of cell CENTRES would not, and
                        # would put the silhouette mis-sort back.
                        #
                        # Fed through the existing re-evaluation below by posing
                        # as a one-sample centroid in lattice units.
                        _ca, ox, oy = _pixel_clip_centroid(vxx, vyy)
                        c = _ca
                        accept = c > 0.0
                        nsm = 1
                        sox = ox * ti.static(_AA_FIXED_SCALE)
                        soy = oy * ti.static(_AA_FIXED_SCALE)
                # A triangle thinner than the sample spacing contains no sample,
                # so the set says nothing about it. The DEFAULT policy is to let
                # it contribute nothing, exactly as supersampling does -- sound
                # because the fill rule PARTITIONS the samples, so any sample
                # this triangle misses is contained by whichever neighbour of the
                # tiling does contain it, and dropping cannot open a hole. It
                # also measures better than every alternative on every config
                # (ss16.2), including sub-pixel rods.
                #
                # The other policies exist because they were the plausible ones:
                # give it the sample it comes closest to, weighted by how much of
                # the pixel it covers, and mark it a sliver so the resolve treats
                # that claim as provisional.
                    if ti.static(_sliver_mode(aa) != _AA_SLIVER_DROP
                                 and not _tri_exact(aa)
                                 and not _tri_run(aa)):
                        if m == 0:
                        # The weight must be the EXACT clipped area: the
                        # continuous product form spreads half a pixel past the
                        # geometry, so a silhouette rim of tiling slivers sums to
                        # a halo and dilates the whole mesh (ss15.3).
                            if ti.static(
                                    _sliver_mode(aa) == _AA_SLIVER_AREA):
                                ofl = 1.0
                                if oi < 0:
                                    ofl = -1.0
                                od0 = ofl * d0
                                od1 = ofl * d1
                                od2 = ofl * d2
                                if ((od0 > -0.5) and (od1 > -0.5)
                                        and (od2 > -0.5)):
                                    m = (1 << best_k) | _AA_SLIVER_BIT
                                    c = (
                                        ti.math.clamp(
                                            od0 + 0.5, 0.0, 1.0)
                                        * ti.math.clamp(
                                            od1 + 0.5, 0.0, 1.0)
                                        * ti.math.clamp(
                                            od2 + 0.5, 0.0, 1.0)
                                    )
                            else:
                                ca = _pixel_clip_area(
                                    ti.math.vec3(
                                        sx0 - qx, sx1 - qx, sx2 - qx),
                                    ti.math.vec3(
                                        sy0 - qy, sy1 - qy, sy2 - qy))
                                if ca > 0.0:
                                # exact and exact_occ coincide under per-sample
                                # transmittance: there is no separate occlusion
                                # set to opt into, since attenuating a sample IS
                                # occluding it (ss18).
                                    m = (1 << best_k) | _AA_SLIVER_BIT
                                    c = ca
                    if ti.static(_tri_exact(aa)):
                        accept = c > 0.0
                    elif ti.static(_tri_run(aa)):
                        # Acceptance widens to "any exact area": a sample-less
                        # sliver is an area donor now, not a drop.
                        accept = c > 0.0
                        if oi < 0:
                            m |= _AA_BACKFACE_BIT
                    else:
                        accept = ((m & _AA_MASK_ALL) != 0) and (c > 0.0)
                        if oi < 0:
                            m |= _AA_BACKFACE_BIT
                # Re-evaluate the fragment AT THE CENTROID OF THE SAMPLES IT
                # OWNS rather than at the pixel centre, which is the only point
                # the fragment is known to be visible at.
                #
                # The pixel centre lies OUTSIDE a partially covering triangle
                # roughly half the time, and there the plane intersection is an
                # EXTRAPOLATION -- past the geometry, on the wrong side of every
                # silhouette edge. Two faces meeting at a silhouette (any closed
                # mesh: their near and far sheets share that edge exactly) then
                # extrapolate in opposite directions, so beyond the edge the far
                # sheet's extrapolated distance is the SMALLER one and the rim
                # pixel sorts the two faces back to front. The far face won the
                # per-sample transmittance, occluded the near one entirely, and
                # the silhouette was drawn in the colour of the geometry behind
                # it -- brighter than the surface it outlines whenever the far
                # sheet is the better lit one. Whether a given rim pixel's centre
                # falls inside or outside is pure sub-pixel phase, which is why
                # it appeared on one edge of an octahedron at one size and not at
                # another.
                #
                # The owned samples are inside the triangle by construction and a
                # triangle is convex, so their centroid is too: no extrapolation,
                # and two faces sharing a silhouette edge order by true depth.
                # The sample pattern sums to zero, so a fully covered fragment
                # re-evaluates at the pixel centre exactly and is unchanged --
                # only rim fragments move, by less than half a pixel.
                    if nsm > 0:
                        inv_n = 1.0 / ti.cast(nsm, ti.f32)
                        dqx = sox * inv_n * ti.static(1.0 / _AA_FIXED_SCALE)
                        dqy = soy * inv_n * ti.static(1.0 / _AA_FIXED_SCALE)
                        if (dqx != 0.0) or (dqy != 0.0):
                            # Edge functions are affine in the sample point, so
                            # the shift is an update rather than a re-derivation.
                            ce0 = e0 + (sx2 - sx1) * dqy - (sy2 - sy1) * dqx
                            ce1 = e1 + (sx0 - sx2) * dqy - (sy0 - sy2) * dqx
                            ce2 = e2 + (sx1 - sx0) * dqy - (sy1 - sy0) * dqx
                            cn0 = ce0 * sm[2, 0]
                            cn1 = ce1 * sm[2, 1]
                            cn2 = ce2 * sm[2, 2]
                            cs = cn0 + cn1 + cn2
                            if ti.abs(cs) > 1e-30:
                                cinv = 1.0 / cs
                                b0 = cn0 * cinv
                                b1 = cn1 * cinv
                                b2 = cn2 * cinv
        if accept:
            v0 = ti.math.vec3(vm[0, 0], vm[0, 1], vm[0, 2])
            v1 = ti.math.vec3(vm[1, 0], vm[1, 1], vm[1, 2])
            v2 = ti.math.vec3(vm[2, 0], vm[2, 1], vm[2, 2])
            hp = b0 * v0 + b1 * v1 + b2 * v2
            tt = (hp - cam_o).norm()
            if tt > MIN_HIT_DISTANCE:
                ok = 1
                t = tt
                cov = c
                msk = m
                if ti.static(aa):
                    cb0 = ti.max(b0, 0.0)
                    cb1 = ti.max(b1, 0.0)
                    cb2 = ti.max(b2, 0.0)
                    bsum = cb0 + cb1 + cb2
                    if bsum > 1e-20:
                        inv_b = 1.0 / bsum
                        b1 = cb1 * inv_b
                        b2 = cb2 * inv_b
                w1 = b1
                w2 = b2
    return ok, t, w1, w2, cov, msk


@ti.func
def _raycast_pixel(px, py, f, vm, half_w, half_h,
                   cam_origin: ti.template(), screen_point: ti.template(),
                   pixel_basis_x: ti.template(), pixel_basis_y: ti.template(),
                   aa: ti.template()):
    """Per-pixel ray-cast fallback (Moller-Trumbore), used when a triangle
    straddles the camera plane so screen-space projection is invalid. Returns
    ``(ok, t, w1, w2, cov, msk)``.

    Under analytic coverage this does NOT report full coverage. It cannot use the
    screen-space edge functions -- the projection that would supply them is the
    thing that is invalid -- so it answers the same set-membership question
    directly instead: cast one ray per sub-pixel sample and test each against the
    triangle. That is the definition of the sample mask rather than an
    approximation of it, at the cost of N intersections on these pixels.

    This case is NOT rare, which is why it is worth the cost: any ground plane
    large enough to reach the horizon straddles the camera plane, and while it
    reported coverage 1 its edges were the only geometry in a scene left
    completely un-antialiased -- byte-identical to no AA at all, and the whole
    residual against anti_alias_level=2 on such a scene (ss19).

    A sample exactly on an edge shared by two straddling triangles (a quad's
    diagonal) is claimed by both, because the barycentric test has an epsilon
    rather than the exact fill rule the projected path uses. Per-sample
    transmittance makes that harmless: the nearer fragment takes the sample and
    the second one finds nothing left of it.
    """
    v0 = ti.math.vec3(vm[0, 0], vm[0, 1], vm[0, 2])
    v1 = ti.math.vec3(vm[1, 0], vm[1, 1], vm[1, 2])
    v2 = ti.math.vec3(vm[2, 0], vm[2, 1], vm[2, 2])
    e1 = v1 - v0
    e2 = v2 - v0
    ok = 0
    t = 0.0
    w1 = 0.0
    w2 = 0.0
    cov = 1.0
    msk = _AA_MASK_ALL
    ro, rd = _generate_ray(f, px, py, 0.5, 0.5, half_w, half_h,
                           cam_origin, screen_point,
                           pixel_basis_x, pixel_basis_y)
    pv = rd.cross(e2)
    det = e1.dot(pv)
    if ti.abs(det) > 1e-12:
        inv_det = 1.0 / det
        tvec = ro - v0
        b1 = tvec.dot(pv) * inv_det
        qv = tvec.cross(e1)
        b2 = rd.dot(qv) * inv_det
        inside = ((b1 >= -BARYCENTRIC_EPSILON)
                  and (b2 >= -BARYCENTRIC_EPSILON)
                  and (b1 + b2 <= 1.0 + BARYCENTRIC_EPSILON))
        th = e2.dot(qv) * inv_det
        if ti.static(aa):
            m = 0
            sox = 0.0
            soy = 0.0
            nsm = 0
            for k in ti.static(range(_AA_NUM_SAMPLES)):
                jx = ti.static(0.5 + _AA_SAMPLES[k][0] / _AA_FIXED_SCALE)
                jy = ti.static(0.5 + _AA_SAMPLES[k][1] / _AA_FIXED_SCALE)
                ros, rds = _generate_ray(f, px, py, jx, jy, half_w, half_h,
                                         cam_origin, screen_point,
                                         pixel_basis_x, pixel_basis_y)
                pvs = rds.cross(e2)
                dets = e1.dot(pvs)
                if ti.abs(dets) > 1e-12:
                    ivs = 1.0 / dets
                    tvs = ros - v0
                    c1 = tvs.dot(pvs) * ivs
                    qvs = tvs.cross(e1)
                    c2 = rds.dot(qvs) * ivs
                    if ((c1 >= -BARYCENTRIC_EPSILON)
                            and (c2 >= -BARYCENTRIC_EPSILON)
                            and (c1 + c2 <= 1.0 + BARYCENTRIC_EPSILON)
                            and (e2.dot(qvs) * ivs > MIN_HIT_DISTANCE)):
                        m |= 1 << k
                        sox += ti.static(float(_AA_SAMPLES[k][0]))
                        soy += ti.static(float(_AA_SAMPLES[k][1]))
                        nsm += 1
            # Same rule as the projected path: a partially covering fragment is
            # re-cast through the CENTROID OF THE SAMPLES IT OWNS, so its depth
            # is a real intersection with the triangle rather than the centre
            # ray's extrapolation past its edges (which sorts the two sheets of
            # a closed mesh backwards along every silhouette). A fully covered
            # fragment's centroid is the pixel centre, so it keeps the centre
            # ray untouched.
            if (m != 0) and (m != _AA_MASK_ALL):
                inv_n = 1.0 / ti.cast(nsm, ti.f32)
                jxc = 0.5 + sox * inv_n * ti.static(1.0 / _AA_FIXED_SCALE)
                jyc = 0.5 + soy * inv_n * ti.static(1.0 / _AA_FIXED_SCALE)
                roc, rdc = _generate_ray(f, px, py, jxc, jyc, half_w, half_h,
                                         cam_origin, screen_point,
                                         pixel_basis_x, pixel_basis_y)
                pvc = rdc.cross(e2)
                detc = e1.dot(pvc)
                if ti.abs(detc) > 1e-12:
                    ivc = 1.0 / detc
                    tvc = roc - v0
                    qvc = tvc.cross(e1)
                    b1 = tvc.dot(pvc) * ivc
                    b2 = rdc.dot(qvc) * ivc
                    th = e2.dot(qvc) * ivc
            if (m != 0) and (th > MIN_HIT_DISTANCE):
                ok = 1
                t = th
                cov = ti.cast(_popcount_samples(m), ti.f32) * _AA_SAMPLE_WEIGHT
                msk = m
                # The centre may lie outside the triangle when only some samples
                # are inside, so project its barycentrics onto the simplex
                # before they index colours, normals and UVs (as _ss_pixel does).
                cb1 = ti.max(b1, 0.0)
                cb2 = ti.max(b2, 0.0)
                cb0 = ti.max(1.0 - b1 - b2, 0.0)
                bsum = cb0 + cb1 + cb2
                if bsum > 1e-20:
                    inv_b = 1.0 / bsum
                    w1 = cb1 * inv_b
                    w2 = cb2 * inv_b
        else:
            if inside and (th > MIN_HIT_DISTANCE):
                ok = 1
                t = th
                w1 = b1
                w2 = b2
    return ok, t, w1, w2, cov, msk


@ti.func
def _pair_pixel(prim, f, x0, y0, bw, bh, off, j,
                time_start, width, height, tile_start, tile_pixels,
                half_w, half_h, use_ss, sm, vm, cam_o, il,
                cam_origin: ti.template(), screen_point: ti.template(),
                pixel_basis_x: ti.template(), pixel_basis_y: ti.template(),
                aa: ti.template(), store_exact: ti.template()):
    """Test candidate ``off + j`` of a pair at its pixel center, dispatching to
    the screen-space path (``use_ss``) or the ray-cast fallback. Returns
    ``(ok, local_pixel, t, w1, w2, cov, msk)``; both paths supply analytic
    coverage under ``aa``, the fallback by casting one ray per sub-pixel sample
    (see :func:`_raycast_pixel`).
    """
    ok = 0
    lp = 0
    t = 0.0
    w1 = 0.0
    w2 = 0.0
    cov = 1.0
    msk = _AA_MASK_ALL
    o = off + j
    if o < bw * bh:
        px = x0 + o % bw
        py = y0 + o // bw
        lpi = ((f - time_start) * (width * height) + py * width + px
               - tile_start)
        if (lpi >= 0) and (lpi < tile_pixels):
            hit = 0
            th = 0.0
            b1 = 0.0
            b2 = 0.0
            cv = 1.0
            mk = _AA_MASK_ALL
            if use_ss != 0:
                hit, th, b1, b2, cv, mk = _ss_pixel(
                    px, py, sm, vm, cam_o, il, aa, store_exact)
            else:
                hit, th, b1, b2, cv, mk = _raycast_pixel(
                    px, py, f, vm, half_w, half_h, cam_origin, screen_point,
                    pixel_basis_x, pixel_basis_y, aa)
            if hit != 0:
                ok = 1
                lp = lpi
                t = th
                w1 = b1
                w2 = b2
                cov = cv
                msk = mk
    return ok, lp, t, w1, w2, cov, msk


@ti.func
def _bez_pixel_hit(circuit, f, px, py, half_w, half_h,
                   cam_origin: ti.template(), screen_point: ti.template(),
                   pixel_basis_x: ti.template(), pixel_basis_y: ti.template(),
                   pixel_world_scale: ti.template(),
                   circuit_meta: ti.template(), circuit_colors: ti.template(),
                   edges_2d: ti.template(), edge_accel: ti.template(),
                   aa: ti.template(), aa_min_half_width: ti.f32,
                   border_aa: ti.template()):
    """Exact primary camera-ray/circuit hit for one known pixel.

    Returns ``(ok, t, u, v, border_frac, cov)``.  ``cov`` is 1.0 unless the
    compile-time ``aa`` template selects analytic coverage, in which case it is
    the fraction of the pixel square the circuit's drawn region covers, and the
    hit is accepted whenever that fraction is non-zero -- including pixels whose
    CENTRE is outside the circuit but whose square is not.  ``border_frac`` is
    the share of that covered area lying in the border band: 0 or 1 unless the
    ``border_aa`` template is set, where it is continuous across the border's
    inner edge.

    Analytic coverage (see DESIGN_analytic_aa.md ss4).  ``_bezier_point_metrics``
    returns the distance to the nearest outline segment plus a crossing parity,
    so ``d = +/- sqrt(min_dist_sq)`` is a signed distance (positive inside) in
    plane units, and ``pixel_size`` converts it to pixels.  The drawn region is

        filled:    d > -min_half_width
        unfilled:  |d| < border_w / 2                     (a band)

    and its coverage is the box filter ``clamp(distance_to_boundary + 0.5, 0, 1)``
    -- exact for a straight boundary crossing the pixel, which after flattening
    every boundary is.  The band form fades a sub-pixel-wide stroke by its width
    instead of dilating it, and both forms reach exactly 1 half a pixel inside,
    which is what lets the opaque z-prepass keep culling (``raster_bez_z``).

    A filled circuit's border is an INNER boundary at ``d = border_w`` cutting
    the same drawn region in two (:func:`_circuit_point_region`), so it needs its
    own box filter: the coverage of the fill-only part is subtracted from the
    total, and the remainder is the border's area share.  Without that the outer
    silhouette resolves continuously while the border/fill edge -- which can be
    the only visible edge, e.g. an outlined glyph over an invisible fill --
    stays a hard per-pixel classification.
    """
    ok = 0
    t = 0.0
    u = 0.0
    v = 0.0
    border_frac = 0.0
    cov = 1.0
    ro, rd = _generate_ray(f, px, py, 0.5, 0.5, half_w, half_h,
                           cam_origin, screen_point,
                           pixel_basis_x, pixel_basis_y)
    tm = f % circuit_meta.shape[0]
    n = ti.math.vec3(circuit_meta[tm, circuit, _M_NORMAL],
                     circuit_meta[tm, circuit, _M_NORMAL + 1],
                     circuit_meta[tm, circuit, _M_NORMAL + 2])
    denom = rd.dot(n)
    if ti.abs(denom) > 1e-9:
        center = ti.math.vec3(circuit_meta[tm, circuit, _M_CENTER],
                              circuit_meta[tm, circuit, _M_CENTER + 1],
                              circuit_meta[tm, circuit, _M_CENTER + 2])
        th = (center - ro).dot(n) / denom
        if th > MIN_HIT_DISTANCE:
            hit = ro + th * rd - center
            bu = ti.math.vec3(circuit_meta[tm, circuit, _M_BASIS_U],
                              circuit_meta[tm, circuit, _M_BASIS_U + 1],
                              circuit_meta[tm, circuit, _M_BASIS_U + 2])
            bv = ti.math.vec3(circuit_meta[tm, circuit, _M_BASIS_V],
                              circuit_meta[tm, circuit, _M_BASIS_V + 1],
                              circuit_meta[tm, circuit, _M_BASIS_V + 2])
            uu = hit.dot(bu)
            vv = hit.dot(bv)
            pixel_size = pixel_world_scale[f] * th
            border_w = ti.abs(
                circuit_meta[tm, circuit, _M_BORDER_W]) * pixel_size
            outline_w = 0.6 * pixel_size
            if ti.static(aa):
                outline_w = aa_min_half_width * pixel_size
            filled = circuit_meta[tm, circuit, _M_FILLED] > 0.5
            query_radius = _circuit_query_radius(border_w, outline_w, filled)
            if ti.static(aa or border_aa):
                # The filter reaches half a pixel past each drawn boundary in
                # any direction, so the nearest-edge query must too -- a pixel
                # whose centre is OUTSIDE now needs a real distance, where the
                # classic path was content with "no edge within the radius".
                query_radius += _AA_FILTER_RADIUS * pixel_size
            te = f % edges_2d.shape[0]
            (crossings, min_dist_sq, ccu, ccv, e1x, e1y, sg1,
             sec_dist_sq, scu, scv, e2x, e2y, sg2) = _bezier_point_metrics(
                circuit, te, uu, vv, query_radius,
                circuit_meta.shape[1], edges_2d, edge_accel)
            inside, is_border = _circuit_point_region(
                border_w, outline_w, filled, crossings, min_dist_sq)
            bf = 1.0 if is_border else 0.0
            c = 1.0
            if ti.static(aa or border_aa):
                inv_px = 1.0 / ti.max(pixel_size, 1e-30)
                # Signed distance, positive inside the outline. min_dist_sq is
                # left at 1e30 when no edge is within the query radius, which
                # correctly reads as "deep inside" / "far outside".
                d = ti.sqrt(ti.min(min_dist_sq, 1e30))
                if (crossings % 2) == 0:
                    d = -d
                # Direction in which the signed distance INCREASES, i.e. into the
                # shape. The closest-point vector runs from the query to the
                # outline, so it points out of the shape from inside and into it
                # from outside; negating it in the first case makes one gradient.
                # Plane units, but only its direction is read and the plane-to-
                # pixel map is isotropic (the anisotropy caveat of ss4 is
                # unchanged by this), so it needs no conversion.
                gu = ccu
                gv = ccv
                if d > 0.0:
                    gu = -gu
                    gv = -gv
                signed = d + outline_w
                # Which way the DRAWN region lies from the active boundary. For a
                # filled circuit that is always "deeper in", but an unfilled one
                # is a band with a wall on each side, and past its middle the
                # nearer wall is the inner one, whose covered side is outward.
                bnu = gu
                bnv = gv
                if not filled:
                    # Unfilled: the drawn band is bounded on the inside too, so
                    # a stroke thinner than a pixel fades instead of vanishing.
                    half = 0.5 * border_w
                    if (half - d) < (d + half):
                        signed = half - d
                        bnu = -gu
                        bnv = -gv
                    else:
                        signed = d + half
                c = _boundary_coverage(bnu, bnv, signed * inv_px, aa)
                if ti.static(int(aa) == 3):
                    # THE ORIENTED WEDGE (DESIGN_analytic_aa_v2.md ss5). A thin
                    # stroke is a STRIP and a corner is a WEDGE; one distance
                    # describes an edge running to infinity, so a 1px glyph
                    # stem reads as solid past its near wall (ss21.2) and a
                    # corner renders as its vertex's distance CIRCLE. Both
                    # walls' inward sides come from STORAGE (edges_2d column 5,
                    # written at flatten time where the contour is known) --
                    # recovering the second wall's side from the contour
                    # handedness was the ss21.6 failure: at a corner the SDF
                    # gradient points at the vertex, so the calibration sign
                    # was arbitrary exactly where the model mattered.
                    #
                    # Validated standalone by benchmarks/_aa_wedge_check.py:
                    # worst coverage error 0.0017 (convex) / 0.0010 (reflex)
                    # over 600 random corners against brute-force polygons.
                    if filled and (sec_dist_sq < 1e29) and (sg1 != 0.0) \
                            and (sg2 != 0.0):
                        l1 = ti.sqrt(e1x * e1x + e1y * e1y)
                        l2 = ti.sqrt(e2x * e2x + e2y * e2y)
                        if (l1 > 1e-20) and (l2 > 1e-20):
                            # Wall normals: sigma times the leftward
                            # perpendicular of the stored contour direction.
                            n1x = -e1y / l1 * sg1
                            n1y = e1x / l1 * sg1
                            n2x = -e2y / l2 * sg2
                            n2y = e2x / l2 * sg2
                            # Signed distances to the wall LINES (valid whether
                            # the closest point is interior or an endpoint --
                            # the endpoint lies on the line), dilated like the
                            # single-plane path.
                            b1p = n1x * ccu + n1y * ccv
                            b2p = n2x * scu + n2y * scv
                            sd1 = (outline_w - b1p) * inv_px
                            sd2 = (outline_w - b2p) * inv_px
                            nd = n1x * n2x + n1y * n2y
                            if nd < 0.9:
                                # nd >= 0.9: the second segment is the next
                                # flattening chord of the SAME wall; folding it
                                # in would halve every plain edge's coverage.
                                a1 = _halfplane_clip_area(n1x, n1y, sd1)
                                a2 = _halfplane_clip_area(n2x, n2y, sd2)
                                inter = _two_halfplane_area(
                                    n1x, n1y, sd1, n2x, n2y, sd2, a1, a2)
                                cn = n1x * n2y - n1y * n2x
                                c = inter
                                if ti.abs(cn) >= 0.2:
                                    # CONVEX vs REFLEX is which RAY of its
                                    # line each wall segment occupies: the
                                    # intersection region's boundary rays
                                    # satisfy the OTHER constraint, the
                                    # union's violate it. Read it off the
                                    # closest points against the apex; when a
                                    # closest point IS the apex, the clamped
                                    # endpoint sign (cp . d) says which way
                                    # the segment leaves it. Parity at the
                                    # pixel centre is the last-resort arbiter
                                    # (undilated distances -- the dilation
                                    # would disagree with parity in a band
                                    # around every edge).
                                    inv_cn = 1.0 / cn
                                    apx = (b1p * n2y - b2p * n1y) * inv_cn
                                    apy = (n1x * b2p - n2x * b1p) * inv_cn
                                    scl = (ti.abs(ccu) + ti.abs(ccv)
                                           + ti.abs(scu) + ti.abs(scv)
                                           + pixel_size)
                                    r1x = ccu - apx
                                    r1y = ccv - apy
                                    s1 = 0.0
                                    if (ti.abs(r1x) + ti.abs(r1y)) \
                                            > 1e-4 * scl:
                                        s1 = n2x * r1x + n2y * r1y
                                    else:
                                        t1 = ccu * e1x + ccv * e1y
                                        if ti.abs(t1) > 1e-4 * scl * l1:
                                            sgn = 1.0 if t1 > 0.0 else -1.0
                                            s1 = sgn * (n2x * e1x
                                                        + n2y * e1y)
                                    r2x = scu - apx
                                    r2y = scv - apy
                                    s2 = 0.0
                                    if (ti.abs(r2x) + ti.abs(r2y)) \
                                            > 1e-4 * scl:
                                        s2 = n1x * r2x + n1y * r2y
                                    else:
                                        t2 = scu * e2x + scv * e2y
                                        if ti.abs(t2) > 1e-4 * scl * l2:
                                            sgn = 1.0 if t2 > 0.0 else -1.0
                                            s2 = sgn * (n1x * e2x
                                                        + n1y * e2y)
                                    uni = ti.math.clamp(
                                        a1 + a2 - inter, 0.0, 1.0)
                                    if (s1 < 0.0) and (s2 < 0.0):
                                        c = uni
                                    elif not ((s1 > 0.0) and (s2 > 0.0)):
                                        ci = (crossings % 2) == 1
                                        in_i = (b1p < 0.0) and (b2p < 0.0)
                                        in_u = (b1p < 0.0) or (b2p < 0.0)
                                        if (in_u == ci) and (in_i != ci):
                                            c = uni
                bf = 0.0
                if border_w > 0.0:
                    if filled:
                        # Everything covered, minus the part deeper than the
                        # stroke width, is border. That inner edge is a boundary
                        # like any other and gets the same exact treatment, which
                        # is the only edge visible at all on an outlined glyph
                        # whose fill is invisible.
                        fill_c = _boundary_coverage(
                            gu, gv, (d - border_w) * inv_px, aa)
                        bf = ti.math.clamp(
                            (c - fill_c) / ti.max(c, 1e-6), 0.0, 1.0)
                    else:
                        bf = 1.0
            if ti.static(not aa):
                if inside:
                    ok = 1
                    t = th
                    u = uu
                    v = vv
                    border_frac = bf
            else:
                if c > 0.0:
                    ok = 1
                    t = th
                    u = uu
                    v = vv
                    cov = c
                    border_frac = bf
    return ok, t, u, v, border_frac, cov


@ti.func
def _bez_pair_pixel(circuit, f, x0, y0, bw, bh, off, j,
                    time_start, width, height, tile_start, tile_pixels,
                    half_w, half_h,
                    cam_origin: ti.template(), screen_point: ti.template(),
                    pixel_basis_x: ti.template(), pixel_basis_y: ti.template(),
                    pixel_world_scale: ti.template(),
                    circuit_meta: ti.template(), circuit_colors: ti.template(),
                    edges_2d: ti.template(), edge_accel: ti.template(),
                    aa: ti.template(), aa_min_half_width: ti.f32):
    """Pair wrapper around :func:`_bez_pixel_hit`."""
    ok = 0
    lp = 0
    t = 0.0
    u = 0.0
    v = 0.0
    in_border = 0.0
    cov = 1.0
    o = off + j
    if o < bw * bh:
        px = x0 + o % bw
        py = y0 + o // bw
        lpi = ((f - time_start) * (width * height) + py * width + px
               - tile_start)
        if (lpi >= 0) and (lpi < tile_pixels):
            hit, th, uu, vv, ib, cv = _bez_pixel_hit(
                circuit, f, px, py, half_w, half_h, cam_origin, screen_point,
                pixel_basis_x, pixel_basis_y, pixel_world_scale, circuit_meta,
                circuit_colors, edges_2d, edge_accel, aa, aa_min_half_width,
                aa)
            if hit != 0:
                ok = 1
                lp = lpi
                t = th
                u = uu
                v = vv
                in_border = ib
                cov = cv
    return ok, lp, t, u, v, in_border, cov


@ti.kernel
def raster_tri_z(
        pairs: ti.types.ndarray(), num_pairs: int,
        tri_pos: ti.types.ndarray(), tri_screen: ti.types.ndarray(),
        cam_origin: ti.types.ndarray(), screen_point: ti.types.ndarray(),
        pixel_basis_x: ti.types.ndarray(), pixel_basis_y: ti.types.ndarray(),
        time_start: int, width: int, height: int,
        half_w: ti.f32, half_h: ti.f32,
        tile_start: int, tile_pixels: int,
        ss_enabled: ti.template(), layer_offset_triangles: ti.f32,
        aa: ti.template(),
        zbuf: ti.types.ndarray()):
    """Typed opaque visibility prepass for flat triangles.

    Under analytic coverage only a FULLY covering fragment may claim the pixel;
    partially covering ones are emitted into the ordered transparent stream by
    the host's ``partial_only`` count/write pass instead (same rule as
    ``raster_bez_z``).
    """
    for p in range(num_pairs):
        prim = pairs[p, 0]
        f = pairs[p, 1]
        x0 = pairs[p, 2]
        y0 = pairs[p, 3]
        bw = pairs[p, 4]
        bh = pairs[p, 5]
        off = pairs[p, 6]
        use_ss, sm, vm, cam_o, il = _ss_setup(
            f, prim, ss_enabled, tri_pos, tri_screen, cam_origin, aa)
        layer = ti.cast(layer_offset_triangles, ti.i32) + prim
        for j in range(RASTER_CHUNK):
            ok, lp, t, _w1, _w2, cov, msk = _pair_pixel(
                prim, f, x0, y0, bw, bh, off, j, time_start, width, height,
                tile_start, tile_pixels, half_w, half_h, use_ss, sm, vm, cam_o,
                il, cam_origin, screen_point, pixel_basis_x, pixel_basis_y, aa,
                0)
            if ok != 0:
                full = True
                if ti.static(_tri_run(aa)):
                    # Under the run representation the prepass keys on the
                    # SAMPLED claim, exactly as the resolve's magnitude does:
                    # a full-mask fragment whose exact area is a hair under 1
                    # claims 1.0 either way (it terminates runs and is never
                    # corrected), so keeping it out of the prepass would cost
                    # z-culling for nothing (v2 ss4.1).
                    full = (msk & _AA_MASK_ALL) == _AA_MASK_ALL
                elif ti.static(aa):
                    full = cov >= AA_FULL_COVERAGE
                if full:
                    ti.atomic_min(zbuf[lp], _order_key(t, layer))


@ti.kernel
def raster_bez_z(
        pairs: ti.types.ndarray(), num_pairs: int,
        cam_origin: ti.types.ndarray(), screen_point: ti.types.ndarray(),
        pixel_basis_x: ti.types.ndarray(), pixel_basis_y: ti.types.ndarray(),
        pixel_world_scale: ti.types.ndarray(),
        circuit_meta: ti.types.ndarray(), circuit_colors: ti.types.ndarray(),
        edges_2d: ti.types.ndarray(), edge_accel: ti.types.ndarray(),
        time_start: int, width: int, height: int,
        half_w: ti.f32, half_h: ti.f32,
        tile_start: int, tile_pixels: int,
        aa: ti.template(), aa_min_half_width: ti.f32,
        zbuf: ti.types.ndarray()):
    """Typed opaque visibility prepass for proven-opaque bezier circuits.

    Under analytic coverage (``aa``) only a FULLY covering fragment may claim
    the pixel: a partially covered one has something showing through it and is
    emitted into the ordered transparent stream instead (the host runs the
    count/write pair over the opaque candidates too, ``partial_only``).
    """
    for p in range(num_pairs):
        circuit = pairs[p, 0]
        f = pairs[p, 1]
        x0 = pairs[p, 2]
        y0 = pairs[p, 3]
        bw = pairs[p, 4]
        bh = pairs[p, 5]
        off = pairs[p, 6]
        for j in range(RASTER_CHUNK):
            ok, lp, t, _u, _v, _ib, cov = _bez_pair_pixel(
                circuit, f, x0, y0, bw, bh, off, j, time_start, width, height,
                tile_start, tile_pixels, half_w, half_h, cam_origin,
                screen_point, pixel_basis_x, pixel_basis_y, pixel_world_scale,
                circuit_meta, circuit_colors, edges_2d, edge_accel,
                aa, aa_min_half_width)
            if ok != 0:
                full = True
                if ti.static(aa):
                    full = cov >= AA_FULL_COVERAGE
                if full:
                    ti.atomic_min(zbuf[lp], _order_key(t, circuit))


@ti.kernel
def raster_tri_count(
        pairs: ti.types.ndarray(), num_pairs: int,
        tri_pos: ti.types.ndarray(), tri_screen: ti.types.ndarray(),
        tri_colors: ti.types.ndarray(), col_row: ti.types.ndarray(),
        tri_uvs: ti.types.ndarray(), tri_tex_meta: ti.types.ndarray(),
        textures: ti.types.ndarray(), num_colored_triangles: ti.i32,
        cam_origin: ti.types.ndarray(), screen_point: ti.types.ndarray(),
        pixel_basis_x: ti.types.ndarray(), pixel_basis_y: ti.types.ndarray(),
        time_start: int, width: int, height: int,
        half_w: ti.f32, half_h: ti.f32,
        tile_start: int, tile_pixels: int,
        ss_enabled: ti.template(), layer_offset_triangles: ti.f32,
        z_cull: ti.template(), zbuf: ti.types.ndarray(),
        aa: ti.template(), partial_only: ti.template(),
        pair_count: ti.types.ndarray()):
    """Count surviving nonzero-alpha transparent triangle fragments.

    ``partial_only`` (analytic coverage only): emit just the partially covered
    fragments, so the host can run this pass over the proven-opaque candidates
    as well -- their fully covered pixels already sit in the z-prepass and only
    their silhouette pixels need to blend.
    """
    for p in range(num_pairs):
        prim = pairs[p, 0]
        f = pairs[p, 1]
        x0 = pairs[p, 2]
        y0 = pairs[p, 3]
        bw = pairs[p, 4]
        bh = pairs[p, 5]
        off = pairs[p, 6]
        use_ss, sm, vm, cam_o, il = _ss_setup(
            f, prim, ss_enabled, tri_pos, tri_screen, cam_origin, aa)
        layer = ti.cast(layer_offset_triangles, ti.i32) + prim
        cnt = 0
        for j in range(RASTER_CHUNK):
            ok, lp, t, w1, w2, cov, msk = _pair_pixel(
                prim, f, x0, y0, bw, bh, off, j, time_start, width, height,
                tile_start, tile_pixels, half_w, half_h, use_ss, sm, vm, cam_o,
                il, cam_origin, screen_point, pixel_basis_x, pixel_basis_y, aa,
                0)
            if ok != 0:
                keep = True
                if ti.static(partial_only):
                    if ti.static(_tri_run(aa)):
                        # Mirror of raster_tri_z's sampled-claim keying: only
                        # full-MASK fragments sit in the prepass, so only they
                        # are excluded here.
                        keep = (msk & _AA_MASK_ALL) != _AA_MASK_ALL
                    else:
                        keep = cov < AA_FULL_COVERAGE
                before_z = True
                if ti.static(z_cull):
                    before_z = _order_key(t, layer) < zbuf[lp]
                if keep and before_z:
                    w0 = 1.0 - w1 - w2
                    _color, alpha = _tri_color_g(
                        0, f, prim, w0, w1, w2, tri_colors, col_row, tri_uvs,
                        tri_tex_meta, textures, num_colored_triangles)
                    if ti.static(aa):
                        alpha *= cov
                    if alpha > MIN_ALPHA:
                        cnt += 1
        pair_count[p] = cnt


@ti.kernel
def raster_tri_write(
        pairs: ti.types.ndarray(), num_pairs: int,
        pair_offset: ti.types.ndarray(),
        tri_pos: ti.types.ndarray(), tri_screen: ti.types.ndarray(),
        tri_colors: ti.types.ndarray(), col_row: ti.types.ndarray(),
        tri_uvs: ti.types.ndarray(), tri_tex_meta: ti.types.ndarray(),
        textures: ti.types.ndarray(), num_colored_triangles: ti.i32,
        cam_origin: ti.types.ndarray(), screen_point: ti.types.ndarray(),
        pixel_basis_x: ti.types.ndarray(), pixel_basis_y: ti.types.ndarray(),
        time_start: int, width: int, height: int,
        half_w: ti.f32, half_h: ti.f32,
        tile_start: int, tile_pixels: int,
        ss_enabled: ti.template(), layer_offset_triangles: ti.f32,
        z_cull: ti.template(), zbuf: ti.types.ndarray(),
        aa: ti.template(), partial_only: ti.template(),
        frag_key: ti.types.ndarray(),
        frag_ref: ti.types.ndarray(), frag_ab: ti.types.ndarray(),
        frag_cov: ti.types.ndarray(), frag_msk: ti.types.ndarray()):
    """Emit exact-distance triangle records; alpha-zero texels are discarded.

    Acceptance must mirror :func:`raster_tri_count` exactly -- its counts sized
    these slots.
    """
    for p in range(num_pairs):
        prim = pairs[p, 0]
        f = pairs[p, 1]
        x0 = pairs[p, 2]
        y0 = pairs[p, 3]
        bw = pairs[p, 4]
        bh = pairs[p, 5]
        off = pairs[p, 6]
        use_ss, sm, vm, cam_o, il = _ss_setup(
            f, prim, ss_enabled, tri_pos, tri_screen, cam_origin, aa)
        layer = ti.cast(layer_offset_triangles, ti.i32) + prim
        w = pair_offset[p]
        for j in range(RASTER_CHUNK):
            ok, lp, t, w1, w2, cov, msk = _pair_pixel(
                prim, f, x0, y0, bw, bh, off, j, time_start, width, height,
                tile_start, tile_pixels, half_w, half_h, use_ss, sm, vm, cam_o,
                il, cam_origin, screen_point, pixel_basis_x, pixel_basis_y, aa,
                1)
            if ok != 0:
                keep = True
                if ti.static(partial_only):
                    if ti.static(_tri_run(aa)):
                        # Mirror of raster_tri_z's sampled-claim keying: only
                        # full-MASK fragments sit in the prepass, so only they
                        # are excluded here.
                        keep = (msk & _AA_MASK_ALL) != _AA_MASK_ALL
                    else:
                        keep = cov < AA_FULL_COVERAGE
                before_z = True
                if ti.static(z_cull):
                    before_z = _order_key(t, layer) < zbuf[lp]
                if keep and before_z:
                    w0 = 1.0 - w1 - w2
                    _color, alpha = _tri_color_g(
                        0, f, prim, w0, w1, w2, tri_colors, col_row,
                        tri_uvs, tri_tex_meta, textures,
                        num_colored_triangles)
                    if ti.static(_tri_run(aa)):
                        # WRITE's cov lane carries the exact area for masked
                        # fragments; the keep decision recomputes the SAMPLED
                        # share so both kernels take IDENTICAL branches --
                        # count sized these slots.
                        keep_c = cov
                        if (msk & _AA_MASK_ALL) != 0:
                            keep_c = (ti.cast(
                                _popcount_samples(msk), ti.f32)
                                * _AA_SAMPLE_WEIGHT)
                        alpha *= keep_c
                    elif ti.static(aa):
                        alpha *= cov
                    if alpha > MIN_ALPHA:
                        tb = ti.cast(ti.bit_cast(t, ti.u32), ti.i64)
                        frag_key[w] = (ti.cast(lp, ti.i64) << 32) | tb
                        frag_ref[w] = prim
                        frag_ab[w, 0] = w1
                        frag_ab[w, 1] = w2
                        if ti.static(aa):
                            frag_cov[w] = cov
                            frag_msk[w] = msk
                        w += 1


@ti.kernel
def raster_bez_count(
        pairs: ti.types.ndarray(), num_pairs: int,
        cam_origin: ti.types.ndarray(), screen_point: ti.types.ndarray(),
        pixel_basis_x: ti.types.ndarray(), pixel_basis_y: ti.types.ndarray(),
        pixel_world_scale: ti.types.ndarray(),
        circuit_meta: ti.types.ndarray(), circuit_colors: ti.types.ndarray(),
        circuit_border_colors: ti.types.ndarray(),
        edges_2d: ti.types.ndarray(), edge_accel: ti.types.ndarray(),
        time_start: int, width: int, height: int,
        half_w: ti.f32, half_h: ti.f32,
        tile_start: int, tile_pixels: int,
        z_cull: ti.template(), zbuf: ti.types.ndarray(),
        aa: ti.template(), aa_min_half_width: ti.f32,
        partial_only: ti.template(),
        pair_count: ti.types.ndarray()):
    """Count surviving nonzero-alpha translucent circuit fragments.

    ``partial_only`` (compile-time, analytic coverage only): emit ONLY the
    partially covered fragments.  The host uses it to run this pass over the
    proven-opaque candidate pairs as well -- their fully covered pixels are
    already in the z-prepass, and their silhouette pixels are exactly the ones
    that need to blend.
    """
    for p in range(num_pairs):
        circuit = pairs[p, 0]
        f = pairs[p, 1]
        x0 = pairs[p, 2]
        y0 = pairs[p, 3]
        bw = pairs[p, 4]
        bh = pairs[p, 5]
        off = pairs[p, 6]
        cnt = 0
        for j in range(RASTER_CHUNK):
            ok, lp, t, u, v, ib, cov = _bez_pair_pixel(
                circuit, f, x0, y0, bw, bh, off, j, time_start, width, height,
                tile_start, tile_pixels, half_w, half_h, cam_origin,
                screen_point, pixel_basis_x, pixel_basis_y, pixel_world_scale,
                circuit_meta, circuit_colors, edges_2d, edge_accel,
                aa, aa_min_half_width)
            if ok != 0:
                keep = True
                if ti.static(partial_only):
                    keep = cov < AA_FULL_COVERAGE
                before_z = True
                if ti.static(z_cull):
                    before_z = _order_key(t, circuit) < zbuf[lp]
                if keep and before_z:
                    _color, alpha = _sample_circuit_color_blend(
                        circuit, f, u, v, ib, circuit_meta, circuit_colors,
                        circuit_border_colors)
                    if ti.static(aa):
                        alpha *= cov
                    if alpha > MIN_ALPHA:
                        cnt += 1
        pair_count[p] = cnt


@ti.kernel
def raster_bez_write(
        pairs: ti.types.ndarray(), num_pairs: int,
        pair_offset: ti.types.ndarray(),
        cam_origin: ti.types.ndarray(), screen_point: ti.types.ndarray(),
        pixel_basis_x: ti.types.ndarray(), pixel_basis_y: ti.types.ndarray(),
        pixel_world_scale: ti.types.ndarray(),
        circuit_meta: ti.types.ndarray(), circuit_colors: ti.types.ndarray(),
        circuit_border_colors: ti.types.ndarray(),
        edges_2d: ti.types.ndarray(), edge_accel: ti.types.ndarray(),
        time_start: int, width: int, height: int,
        half_w: ti.f32, half_h: ti.f32,
        tile_start: int, tile_pixels: int,
        z_cull: ti.template(), zbuf: ti.types.ndarray(),
        aa: ti.template(), aa_min_half_width: ti.f32,
        partial_only: ti.template(),
        frag_key: ti.types.ndarray(),
        frag_ref: ti.types.ndarray(), frag_ab: ti.types.ndarray(),
        frag_cov: ti.types.ndarray()):
    """Emit circuit records with the border weight packed into ``frag_ref``.

    Must mirror :func:`raster_bez_count`'s acceptance exactly -- the counts sized
    this pass's slots.  ``frag_cov`` is the analytic coverage lane, pre-filled
    with 1.0 by the host so geometry without analytic coverage (flat triangles
    today) needs no write of its own.
    """
    for p in range(num_pairs):
        circuit = pairs[p, 0]
        f = pairs[p, 1]
        x0 = pairs[p, 2]
        y0 = pairs[p, 3]
        bw = pairs[p, 4]
        bh = pairs[p, 5]
        off = pairs[p, 6]
        w = pair_offset[p]
        for j in range(RASTER_CHUNK):
            ok, lp, t, u, v, ib, cov = _bez_pair_pixel(
                circuit, f, x0, y0, bw, bh, off, j, time_start, width, height,
                tile_start, tile_pixels, half_w, half_h, cam_origin,
                screen_point, pixel_basis_x, pixel_basis_y, pixel_world_scale,
                circuit_meta, circuit_colors, edges_2d, edge_accel,
                aa, aa_min_half_width)
            if ok != 0:
                keep = True
                if ti.static(partial_only):
                    keep = cov < AA_FULL_COVERAGE
                before_z = True
                if ti.static(z_cull):
                    before_z = _order_key(t, circuit) < zbuf[lp]
                if keep and before_z:
                    _color, alpha = _sample_circuit_color_blend(
                        circuit, f, u, v, ib, circuit_meta, circuit_colors,
                        circuit_border_colors)
                    if ti.static(aa):
                        alpha *= cov
                    if alpha > MIN_ALPHA:
                        tb = ti.cast(ti.bit_cast(t, ti.u32), ti.i64)
                        frag_key[w] = (ti.cast(lp, ti.i64) << 32) | tb
                        frag_ref[w] = _pack_bez_ref(circuit, ib)
                        frag_ab[w, 0] = u
                        frag_ab[w, 1] = v
                        if ti.static(aa):
                            frag_cov[w] = cov
                        w += 1


# Sub-pixel positions for continuation-ray supersampling, by sample count. At 4
# these are exactly where anti_alias_level=2 puts its supersamples, so a
# reflected image sampled at these positions matches that reference arm's
# geometry rather than merely resembling it. Regular grids, not random jitter:
# the renderer's output must stay deterministic and frame-independent
# (DESIGN_analytic_aa.md ss12), and a random offset would also make a mirror
# hiss between frames.
_AA_SEC_JITTER = {
    1: ((0.5, 0.5),),
    2: ((0.25, 0.25), (0.75, 0.75)),
    4: ((0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)),
    8: ((0.1875, 0.3125), (0.6875, 0.1875), (0.3125, 0.8125),
        (0.8125, 0.6875), (0.0625, 0.5625), (0.5625, 0.4375),
        (0.4375, 0.9375), (0.9375, 0.0625)),
}


def _nearest_jitter(k, n):
    """Index of the sub-pixel continuation position nearest to sample ``k``."""
    sx = 0.5 + _AA_SAMPLES[k][0] / _AA_FIXED_SCALE
    sy = 0.5 + _AA_SAMPLES[k][1] / _AA_FIXED_SCALE
    best, bd = 0, 1e9
    for j, (jx, jy) in enumerate(_AA_SEC_JITTER[n]):
        d = (sx - jx) ** 2 + (sy - jy) ** 2
        if d < bd:
            bd, best = d, j
    return best


# Which continuation position each coverage sample belongs to, per sample count.
# Pure Python, evaluated at import: the kernel only ever indexes it statically.
_AA_SEC_OWNER = {
    n: tuple(_nearest_jitter(k, n) for k in range(_AA_NUM_SAMPLES))
    for n in _AA_SEC_JITTER
}


@ti.func
def _sec_positions(msk, n: ti.template()):
    """Which of the N continuation positions this fragment actually covers.

    A fragment spawns one secondary ray per position it owns, not N regardless
    -- which is exactly what supersampling does, one ray per sub-pixel the
    primitive covers. Gating on the fragment's total coverage instead was tried
    and is WRONG: in a dense mesh nearly every fragment partially covers its
    pixel (each triangle owns a few samples), so a "full coverage only" test
    switched secondary sampling off almost everywhere and cost a glass scene its
    entire refracted-image quality.

    Returns ``(position_mask, count)``; count is zero only for an empty mask.
    """
    pm = 0
    for k in ti.static(range(_AA_NUM_SAMPLES)):
        if (msk >> k) & 1:
            pm |= 1 << ti.static(_AA_SEC_OWNER[n][k])
    cnt = 0
    for j in ti.static(range(n)):
        if (pm >> j) & 1:
            cnt += 1
    return pm, cnt


# Roughness below this leaves the reflection specular-perfect, so a mirror is
# BYTE-IDENTICAL to the pre-glossy build (the test baselines depend on it) and
# so does anything a user authored as a mirror. At this value the GGX width is
# alpha = 1e-8, i.e. a deflection of ~1e-8 radians -- far below a pixel at any
# resolution -- so nothing visible is being gated away.
#
# Deliberately a module constant rather than a setting: it is baked into the
# compiled kernel and is NOT part of any template argument, so an env knob would
# let the offline cache serve a kernel built for a different threshold (the same
# trap ``_AA_SAMPLES`` carries).
_GLOSSY_MIN_ROUGHNESS = 100#1e-4
_GLOSSY_INV_16 = 1.0 / 16.0


@ti.func
def _bayer4(px, py):
    """The canonical 4x4 ordered-dither index (0..15) of a pixel.

    ``M4[y][x] = 4*m2(y&1, x&1) + m2(y>>1, x>>1)`` with the 2x2 base
    ``[[0, 2], [3, 1]]`` written as ``m2(a, b) = 2*(a^b) + a``. A permutation of
    0..15 over every 4x4 block, arranged so neighbours are maximally far apart
    in index -- which is the property wanted here, since the index chooses which
    part of the reflection lobe the pixel samples.
    """
    yl = py & 1
    xl = px & 1
    yh = (py >> 1) & 1
    xh = (px >> 1) & 1
    return 4 * (2 * (yl ^ xl) + yl) + (2 * (yh ^ xh) + yh)


@ti.func
def _glossy_rotation(px, py, interleave: ti.template()):
    """Per-pixel offsets that rotate a fragment's lobe fan.

    Returns ``(radial_offset in (0,1), azimuth_offset in radians)``. With
    ``interleave`` off both are fixed, so every pixel samples the same few lobe
    directions and a four-tap fan reads as four ghost copies of the reflected
    image. With it on, a 4x4 Bayer index scatters the taps: the radial stratum
    is offset by a Cranley-Patterson rotation (16 sub-strata inside each of the
    fragment's own strata) and the azimuth by a golden-angle multiple of the
    same index, which decorrelates the two dimensions from each other.

    This is not decoration. The radial coordinate a tap ends up with is
    ``(j + (b + 0.5)/16) / k``, i.e. ``(16j + b + 0.5) / 16k``, and ``16j + b``
    runs over ``0 .. 16k-1`` exactly once -- so a 4x4 block's taps ARE the
    ``16k`` quantiles of the lobe, the optimal stratified set, for no extra
    rays. Verified bit-exact for k = 2, 3, 4 and 8, which matters because ``k``
    is how many sub-pixel positions the FRAGMENT covers, not the setting: a
    silhouette fragment owning two positions still samples its lobe at a perfect
    32 quantiles. Without the rotation every pixel repeats the same ``k``
    strata, so the whole image is a k-point quadrature -- measurably so
    (``benchmarks/_glossy_ggx_check.py`` Part A: KS 0.125 against the analytic
    CDF at k=4, exactly the 1/(2k) floor, against 0.008 with it).

    Fixed in SCREEN space, so it is a function of the pixel and nothing else:
    the same frame renders identically every time, and across an animation the
    pattern is stationary -- it cannot twinkle, which a time- or object-varying
    pattern would.
    """
    r_off = 0.5
    ang_off = 0.0
    if ti.static(interleave):
        b = ti.cast(_bayer4(px, py), ti.f32)
        r_off = (b + 0.5) * _GLOSSY_INV_16
        ang_off = _GOLDEN_ANGLE * b
    return r_off, ang_off


@ti.func
def _glossy_reflect(rd, n, roughness, j, k, r_off, ang_off):
    """Tap ``j`` of ``k`` on the GGX reflection lobe about normal ``n``.

    Samples the GGX / Trowbridge-Reitz normal distribution for a MICROFACET
    NORMAL and reflects ``rd`` about it, which is the same lobe
    ``shading_taichi._stage_standard`` uses for the direct highlight, at the
    same ``alpha = roughness^2``. Matching it is the point: a reflection blurred
    by a different width than the highlight beside it describes two different
    materials. (The Monte Carlo megakernel's ``rd + roughness * random_unit``
    is a NORMAL PERTURBATION, a visibly wider and differently shaped lobe --
    ~2.8x the angular width at roughness 0.18 -- and is not what this follows.)

    Deterministic: the radial coordinate is the stratum ``(j + r_off) / k`` of
    the inverted GGX CDF and the azimuth is ``GOLDEN_ANGLE * j + ang_off``, the
    same fixed low-discrepancy construction the soft-shadow fan uses. Nothing
    here reads ``ti.random``, so an animation cannot hiss between frames.

    Every tap carries an equal share of the reflected throughput and the caller
    divides by ``k``, so this redistributes the mirror ray's energy over the
    lobe without creating or destroying any: the NDF is sampled for DIRECTIONS
    only, and the per-tap Fresnel/geometry reweighting a full importance-sampled
    GGX estimator would apply is deliberately omitted (it would change the total
    at grazing, and the surrounding four-way split has already spent the
    Fresnel term).

    A tap that lands below the horizon is reflected back across ``n`` -- the
    same repair the Monte Carlo path makes -- so the returned direction always
    agrees with the ``n``-facing origin offset the caller pairs it with.
    """
    a = roughness * roughness
    u1 = (ti.cast(j, ti.f32) + r_off) / ti.cast(k, ti.f32)
    # Inverted GGX radial CDF: tan^2(theta) = a^2 * u / (1 - u).
    tan2 = (a * a) * u1 / ti.max(1.0 - u1, 1e-6)
    cos_t = 1.0 / ti.sqrt(1.0 + tan2)
    sin_t = ti.sqrt(ti.max(1.0 - cos_t * cos_t, 0.0))
    ang = _GOLDEN_ANGLE * ti.cast(j, ti.f32) + ang_off
    aref = ti.math.vec3(1.0, 0.0, 0.0)
    if ti.abs(n[0]) > 0.9:
        aref = ti.math.vec3(0.0, 1.0, 0.0)
    t1 = n.cross(aref).normalized()
    t2 = n.cross(t1)
    h = (t1 * (ti.cos(ang) * sin_t) + t2 * (ti.sin(ang) * sin_t)
         + n * cos_t).normalized()
    out = (rd - 2.0 * rd.dot(h) * h).normalized()
    if out.dot(n) <= 0.0:
        out = (out - 2.0 * out.dot(n) * n).normalized()
    return out


@ti.func
def _jittered_surface_sample(f, px, py, jx, jy, gen_meta: ti.template(),
                             is_bez, prim, hit_point, nrm,
                             tri_pos: ti.template(), tri_norm: ti.template(),
                             tri_uvs: ti.template(),
                             tri_tex_meta: ti.template(),
                             textures: ti.template(), num_colored_triangles,
                             cam_origin: ti.template(),
                             screen_point: ti.template(),
                             pixel_basis_x: ti.template(),
                             pixel_basis_y: ti.template()):
    """Re-sample one hit at a different sub-pixel position within its pixel.

    Regenerates the primary ray through ``(jx, jy)`` and asks where it would
    have met the SAME primitive. Both flat triangles and bezier circuits are
    planes, so one ray-plane solve is exact -- no re-traversal.

    For a triangle it also returns the SHADING NORMAL re-interpolated at that
    point, which on a curved mirror matters more than the shifted origin does:
    the reflected direction turns by twice the normal's change across the pixel,
    and curvature amplifies it, so reflecting several sub-samples off one shared
    normal blurs the reflection rather than resolving it. A circuit's normal is
    constant over its plane, so only its origin moves.

    Barycentrics are projected onto the simplex before they index normals or
    UVs, exactly as ``_ss_pixel`` does for a partially covering hit: a sub-sample
    of a silhouette pixel can fall outside the triangle, and extrapolated vertex
    normals are not safe on a coarse mesh.

    The jitter is an offset from whatever base sub-pixel position the pass uses
    (``gen_meta[0:2]``), so it composes with in-place supersampling rather than
    overriding it. A grazing or degenerate solve keeps the un-jittered hit --
    the same ray the un-supersampled path would have spawned.

    Returns ``(rd, hit_point, normal, b1, b2)``; the barycentrics are zero for a
    circuit and let the caller re-shade a triangle at that sub-sample.
    """
    ro, rd = _generate_ray(
        f, px, py, gen_meta[0] + (jx - 0.5), gen_meta[1] + (jy - 0.5),
        gen_meta[2], gen_meta[3], cam_origin, screen_point,
        pixel_basis_x, pixel_basis_y)
    hp = hit_point
    nj = nrm
    bj1 = 0.0
    bj2 = 0.0
    if is_bez:
        den = rd.dot(nrm)
        if ti.abs(den) > 1e-9:
            ts = (hit_point - ro).dot(nrm) / den
            if ts > MIN_HIT_DISTANCE:
                hp = ro + ts * rd
    else:
        tp = f % tri_pos.shape[0]
        v0 = ti.math.vec3(tri_pos[tp, prim, 0], tri_pos[tp, prim, 1],
                          tri_pos[tp, prim, 2])
        v1 = ti.math.vec3(tri_pos[tp, prim, 3], tri_pos[tp, prim, 4],
                          tri_pos[tp, prim, 5])
        v2 = ti.math.vec3(tri_pos[tp, prim, 6], tri_pos[tp, prim, 7],
                          tri_pos[tp, prim, 8])
        e1 = v1 - v0
        e2 = v2 - v0
        gn = e1.cross(e2)
        den = rd.dot(gn)
        if ti.abs(den) > 1e-9:
            ts = (v0 - ro).dot(gn) / den
            if ts > MIN_HIT_DISTANCE:
                hp = ro + ts * rd
                nn = gn.dot(gn)
                if nn > 1e-30:
                    inv = 1.0 / nn
                    d = hp - v0
                    c1 = ti.max(gn.dot(d.cross(e2)) * inv, 0.0)
                    c2 = ti.max(gn.dot(e1.cross(d)) * inv, 0.0)
                    c0 = ti.max(1.0 - c1 - c2, 0.0)
                    s = c0 + c1 + c2
                    if s > 1e-20:
                        si = 1.0 / s
                        bj1 = c1 * si
                        bj2 = c2 * si
                        nj = _tri_normal_g(
                            0, f, prim, c0 * si, bj1, bj2, tri_norm,
                            tri_pos, tri_uvs, tri_tex_meta, textures,
                            num_colored_triangles).normalized()
    return rd, hp, nj, bj1, bj2


@ti.func
def _plane_pt(f, px, py, jx, jy, gen_meta: ti.template(), p0, nrm, fallback,
              cam_origin: ti.template(), screen_point: ti.template(),
              pixel_basis_x: ti.template(), pixel_basis_y: ti.template()):
    """Where the sub-pixel-(jx, jy) primary ray meets the plane (p0, nrm)."""
    ro, rd = _generate_ray(
        f, px, py, gen_meta[0] + (jx - 0.5), gen_meta[1] + (jy - 0.5),
        gen_meta[2], gen_meta[3], cam_origin, screen_point,
        pixel_basis_x, pixel_basis_y)
    den = rd.dot(nrm)
    out = fallback
    if ti.abs(den) > 1e-9:
        ts = (p0 - ro).dot(nrm) / den
        if ts > MIN_HIT_DISTANCE:
            out = ro + ts * rd
    return out


@ti.func
def _pixel_footprint(f, px, py, gen_meta: ti.template(), hp, nrm,
                     cam_origin: ti.template(), screen_point: ti.template(),
                     pixel_basis_x: ti.template(),
                     pixel_basis_y: ti.template()):
    """World-space vectors spanning one pixel ACROSS the hit surface.

    The surface point of sub-pixel ``(jx, jy)`` is
    ``hp + (jx - 0.5) * dpx + (jy - 0.5) * dpy``. Central differences of the
    ray-plane intersection, so it is exact for the plane the hit lies in (which a
    flat triangle is) and correctly ANISOTROPIC: a surface seen at a grazing
    angle has a long footprint along the direction it recedes in, which is
    exactly where its shadow edge needs the most sub-pixel resolution.

    Six floats per shadow event, independent of how many sub-samples the trace
    then takes -- versus storing the sub-positions themselves.
    """
    xp = _plane_pt(f, px, py, 1.0, 0.5, gen_meta, hp, nrm, hp, cam_origin,
                   screen_point, pixel_basis_x, pixel_basis_y)
    xm = _plane_pt(f, px, py, 0.0, 0.5, gen_meta, hp, nrm, hp, cam_origin,
                   screen_point, pixel_basis_x, pixel_basis_y)
    yp = _plane_pt(f, px, py, 0.5, 1.0, gen_meta, hp, nrm, hp, cam_origin,
                   screen_point, pixel_basis_x, pixel_basis_y)
    ym = _plane_pt(f, px, py, 0.5, 0.0, gen_meta, hp, nrm, hp, cam_origin,
                   screen_point, pixel_basis_x, pixel_basis_y)
    return xp - xm, yp - ym


@ti.func
def _sub_pixel_origin(spos, dpx, dpy, s):
    """One of four 2x2-grid sub-pixel positions on the surface, indexed at
    runtime.

    Computed arithmetically rather than from a table because the shadow loop's
    sample index is a runtime value. The four offsets are exactly where
    ``anti_alias_level = 2`` puts its supersamples; a larger sample count cycles
    through them with a different soft-shadow fan angle each time.
    """
    su = ti.cast(s & 1, ti.f32) * 0.5 - 0.25
    sv = ti.cast((s >> 1) & 1, ti.f32) * 0.5 - 0.25
    return spos + su * dpx + sv * dpy


@ti.func
def _tri_surface_point(f, prim, w0, a, b, tri_pos: ti.template()):
    """The fragment's hit point, rebuilt from its own barycentrics.

    Under analytic coverage a PARTIALLY covering fragment's depth is evaluated
    at the CENTROID OF THE SAMPLES IT OWNS, not at the pixel centre (see
    ``_ss_pixel``) -- so ``ro + t_hit * rd`` along the CENTRE ray is a point
    that lies on neither the triangle nor, in general, the surface: it is the
    centre ray advanced to a distance measured along a different ray. On a
    closed mesh that lands it up to a facet-depth INSIDE the geometry, past the
    shared edge and below the neighbouring facet, and the fixed
    ``10 * MIN_HIT_DISTANCE`` normal offset applied to every secondary origin is
    far too small to escape. The continuation then re-hits the surface it just
    left, at grazing incidence where Fresnel goes to one, and the pixel gets a
    bright desaturated spike -- speckle scattered over every smooth-shaded mesh
    with a reflective material.

    ``w0/a/b`` are the barycentrics ``_ss_pixel`` already projected onto the
    simplex, so this reproduces exactly the point whose distance became
    ``t_hit`` and is inside the triangle by construction. Bezier circuits keep
    the centre-ray form: their coverage is areal, with no sub-pixel position,
    and their ``t`` is a centre-ray intersection to begin with.
    """
    tp = f % tri_pos.shape[0]
    v0 = ti.math.vec3(tri_pos[tp, prim, 0], tri_pos[tp, prim, 1],
                      tri_pos[tp, prim, 2])
    v1 = ti.math.vec3(tri_pos[tp, prim, 3], tri_pos[tp, prim, 4],
                      tri_pos[tp, prim, 5])
    v2 = ti.math.vec3(tri_pos[tp, prim, 6], tri_pos[tp, prim, 7],
                      tri_pos[tp, prim, 8])
    return w0 * v0 + a * v1 + b * v2


@ti.func
def _spawn_pool_ray(rs_ro: ti.template(), rs_rd: ti.template(),
                    rs_acc: ti.template(), rs_sca: ti.template(),
                    rs_int: ti.template(), rs_pix: ti.template(),
                    rs_alloc: ti.template(),
                    orig, dirn, wt, dist, bounces_left, processed, pixel, r,
                    compact: ti.template()):
    """Append one continuation ray to the tile's shared ray pool.

    Overflow is not an error and not a per-pixel cap: the host retries the whole
    tile with half as many primary rays, which doubles the continuation headroom
    (``REFRACT_INITIAL_POOL_RATIO``). A dropped slot silently loses that
    branch's contribution, which is why the caller must not have already
    committed throughput to it.
    """
    c, have_slot = _reserve_continuation_slot(rs_alloc, rs_ro.shape[0])
    if have_slot:
        for k in ti.static(range(3)):
            rs_ro[c, k] = orig[k]
            rs_rd[c, k] = dirn[k]
        for k in ti.static(range(4)):
            rs_acc[c, k] = 0.0
        rs_sca[c, 0] = wt[0]
        rs_sca[c, 1] = 0.0
        rs_sca[c, 2] = 1e30
        rs_sca[c, 3] = -1e30
        rs_sca[c, 4] = dist
        rs_sca[c, 5] = wt[1]
        rs_sca[c, 6] = wt[2]
        rs_int[c, 0] = bounces_left
        rs_int[c, 1] = processed
        rs_int[c, 2] = _ACTIVE
        rs_int[c, 3] = 0
        rs_pix[c] = pixel
        if ti.static(compact):
            rs_int[c, 4] = r
    return have_slot


@ti.func
def _terminal_z_hit(zkey, f, px, py, layer_offset_triangles,
                    ss_enabled: ti.template(),
                    tri_pos: ti.template(), tri_screen: ti.template(),
                    cam_origin: ti.template(), screen_point: ti.template(),
                    pixel_basis_x: ti.template(), pixel_basis_y: ti.template(),
                    pixel_world_scale: ti.template(),
                    circuit_meta: ti.template(), circuit_colors: ti.template(),
                    edges_2d: ti.template(), edge_accel: ti.template(),
                    half_w, half_h, border_aa: ti.template()):
    """Recompute exact payload for the typed visibility-buffer winner.

    Always classic (non-analytic) acceptance, for both geometries: a primitive
    only reaches the visibility buffer when its analytic coverage was FULL
    (``raster_bez_z`` / ``raster_tri_z``), and full coverage means every edge is
    at least half a pixel away, i.e. strictly inside the classic drawn region.
    So the classic predicate accepts exactly the same winners, and the winner's
    coverage is 1 by construction -- no coverage needs to be returned.

    Full OUTER coverage says nothing about the border's INNER edge, though: a
    pixel straight through the middle of an opaque glyph's outline stroke is a
    z-winner and still straddles the border/fill boundary.  So ``border_aa``
    (the live circuit analytic-coverage setting) still applies there.
    """
    valid = 0
    is_bez = False
    prim = 0
    t = 0.0
    a = 0.0
    b = 0.0
    in_border = 0.0
    if zkey != ti.i64(Z_SENTINEL):
        is_bez, prim = _decode_z_layer(zkey, layer_offset_triangles)
        if is_bez:
            valid, t, a, b, in_border, _cov = _bez_pixel_hit(
                prim, f, px, py, half_w, half_h, cam_origin, screen_point,
                pixel_basis_x, pixel_basis_y, pixel_world_scale, circuit_meta,
                circuit_colors, edges_2d, edge_accel, 0, 0.0, border_aa)
        else:
            use_ss, sm, vm, cam_o, il = _ss_setup(
                f, prim, ss_enabled, tri_pos, tri_screen, cam_origin, 0)
            if use_ss != 0:
                valid, t, a, b, _c, _m = _ss_pixel(
                    px, py, sm, vm, cam_o, il, 0, 0)
            else:
                valid, t, a, b, _c, _m = _raycast_pixel(
                    px, py, f, vm, half_w, half_h, cam_origin, screen_point,
                    pixel_basis_x, pixel_basis_y, 0)
    return valid, is_bez, prim, t, a, b, in_border


@ti.func
def _tri_shadow_normals(f, prim, a, b, rd,
                        tri_pos: ti.template(), tri_norm: ti.template(),
                        tri_uvs: ti.template(), tri_tex_meta: ti.template(),
                        textures: ti.template(), num_colored_triangles):
    """Shading + geometric face normals of a triangle hit, oriented for a
    shadow-ray origin exactly as ``wavefront_shade``'s inline shadow block (both
    defer to :func:`_orient_hit_normals`).
    """
    w0 = 1.0 - a - b
    snrm = _tri_normal_g(0, f, prim, w0, a, b, tri_norm, tri_pos, tri_uvs,
                         tri_tex_meta, textures, num_colored_triangles)
    tp = f % tri_pos.shape[0]
    v0 = ti.math.vec3(tri_pos[tp, prim, 0], tri_pos[tp, prim, 1],
                      tri_pos[tp, prim, 2])
    v1 = ti.math.vec3(tri_pos[tp, prim, 3], tri_pos[tp, prim, 4],
                      tri_pos[tp, prim, 5])
    v2 = ti.math.vec3(tri_pos[tp, prim, 6], tri_pos[tp, prim, 7],
                      tri_pos[tp, prim, 8])
    fnrm = (v1 - v0).cross(v2 - v0)
    return _orient_hit_normals(snrm, fnrm, rd)


@ti.kernel
def raster_shadow_event_build(
        num_pixels: int,
        run_offsets: ti.types.ndarray(),
        frag_key: ti.types.ndarray(), frag_ref: ti.types.ndarray(),
        frag_ab: ti.types.ndarray(), frag_cov: ti.types.ndarray(),
        frag_msk: ti.types.ndarray(),
        zbuf: ti.types.ndarray(),
        tri_pos: ti.types.ndarray(), tri_screen: ti.types.ndarray(),
        tri_norm: ti.types.ndarray(), tri_extra: ti.types.ndarray(),
        tri_colors: ti.types.ndarray(), tri_uvs: ti.types.ndarray(),
        tri_tex_meta: ti.types.ndarray(), textures: ti.types.ndarray(),
        num_colored_triangles: ti.i32, col_row: ti.types.ndarray(),
        tri_obj: ti.types.ndarray(),
        circuit_meta: ti.types.ndarray(), circuit_colors: ti.types.ndarray(),
        circuit_border_colors: ti.types.ndarray(),
        edges_2d: ti.types.ndarray(), edge_accel: ti.types.ndarray(),
        pixel_world_scale: ti.types.ndarray(),
        layer_offset_triangles: ti.f32,
        refraction: ti.template(), ss_enabled: ti.template(),
        has_bez: ti.template(), aa_bez: ti.template(), aa_tri: ti.template(),
        aa_grp: ti.template(), sec_aa: ti.template(),
        covered: ti.template(),
        covered_idx: ti.types.ndarray(), num_covered: int,
        compact: ti.template(),
        time_start: int, width: int, height: int, tile_start: int,
        cam_origin: ti.types.ndarray(), screen_point: ti.types.ndarray(),
        pixel_basis_x: ti.types.ndarray(), pixel_basis_y: ti.types.ndarray(),
        gen_meta: ti.types.ndarray(), max_bounces: int,
        frag_shadow_id: ti.types.ndarray(), z_shadow_id: ti.types.ndarray(),
        event_pos: ti.types.ndarray(), event_snrm: ti.types.ndarray(),
        event_fnrm: ti.types.ndarray(), event_frame: ti.types.ndarray(),
        event_dp: ti.types.ndarray(), event_msk: ti.types.ndarray(),
        event_count: ti.types.ndarray(),
        dump: ti.template(), dump_out: ti.types.ndarray()):
    """Build an exact sparse queue of accepted primary triangle shade events.

    The ordered transport walk mirrors ``raster_first_shade`` through seam
    rejection, alpha evaluation, throughput termination, and path bending.
    Only triangle fragments that the resolve will actually shade reserve an
    event.  Their IDs are written back beside the raw fragment (or terminal
    z-winner) so the later resolve can fetch one exact per-light visibility
    row without position-based slot approximations.

    ``covered`` (compile-time): like ``raster_first_shade``, iterate only the
    compact covered-pixel list -- empty pixels have no fragment and no
    z-winner, so they reserve no events; skipping them changes only the
    (already order-independent) event numbering, not the output. ``compact``
    means the CSR/z arrays themselves are compact (row ``t``) rather than
    dense tile arrays (row ``covered_idx[t]``).

    ``aa_bez`` / ``aa_tri`` (compile-time): analytic coverage scales a
    fragment's alpha and groups same-object fragments, which changes ``weight``
    and therefore where the walk terminates.  Both MUST be applied here exactly
    as in ``raster_first_shade`` -- including the disabled seam de-duplication
    and the group bookkeeping -- or the two walks accept different triangle
    events and every shadow id desynchronizes from its fragment.
    """
    pixels_per_frame = width * height
    loop_n = num_pixels
    if ti.static(covered):
        loop_n = num_covered
    for t in range(loop_n):
        r = t
        pixel = r
        if ti.static(covered):
            pixel = covered_idx[t]
            if ti.static(not compact):
                r = pixel
        g = tile_start + pixel
        f_rel = g // pixels_per_frame
        p = g - f_rel * pixels_per_frame
        f = time_start + f_rel
        py = p // width
        px = p - py * width
        ro, rd = _generate_ray(f, px, py, gen_meta[0], gen_meta[1],
                               gen_meta[2], gen_meta[3], cam_origin,
                               screen_point, pixel_basis_x, pixel_basis_y)
        dmatch = False
        if ti.static(dump):
            dmatch = _aa_dump_match(dump_out, px, py, f)
        # Mirrors raster_first_shade's transport state exactly, including the
        # per-sample transmittance (see its docstring). Any divergence here
        # desynchronizes every shadow id from its fragment.
        weight = ti.math.vec3(1.0, 1.0, 1.0)
        cells = ti.static(_tri_cells(aa_tri))
        svis = ti.Vector([1.0 for _ in range(_AA_NUM_SAMPLES)])
        seam_t = -1e30
        cprev_t = -1e30
        # RUN state (DESIGN_analytic_aa_v2.md ss4.2), dead outside run mode:
        # the exclusive end of the scanned run, its kind (0 uncorrected, 1
        # corrected, 2 pristine all-sliver), the magnitude correction, and the
        # pristine claim bookkeeping (area scale, svis at run start, fraction
        # of it already claimed). Rule B adds the run's owned-sample union,
        # the clamped write residue, and a pending flag for its run-end step.
        run_end = 0
        run_mode = 0
        run_corr = 1.0
        run_pscale = 0.0
        run_vstart = 0.0
        run_claimed = 0.0
        run_U = 0
        run_resid = 0.0
        run_pending = 0
        # SURFACE accounting (DESIGN ss21.9). Four scalars replace the whole
        # per-sample array: the pixel's remaining transmittance in front of the
        # current object, how much of the pixel the current SHEET has covered,
        # the largest sheet the current object has managed, and what the object
        # has absorbed so far.
        rem = 1.0
        sheet_cov = 0.0
        obj_cov = 0.0
        obj_absorb = 0.0
        cur_obj = -2147483647
        cur_face = 0
        bounces_left = max_bounces
        processed = 0
        start = run_offsets[r]
        end = run_offsets[r + 1]
        nrun = end - start
        has_z = 1 if zbuf[r] != ti.i64(Z_SENTINEL) else 0
        total = nrun + has_z
        q = 0
        while q < total and processed < MAX_SURFACES_PER_RAY:
            from_z = q >= nrun
            idx = start + q
            t_hit = 0.0
            ref = 0
            a = 0.0
            b = 0.0
            in_border = 0.0
            is_bez = False
            valid = 1
            # The z-winner is a fully covering hit by construction, so it
            # claims and occludes every sub-pixel sample.
            cov = 1.0
            msk = _AA_MASK_ALL
            slots = ti.Vector([1.0 for _ in range(_AA_NUM_SAMPLES)])
            nsm = _AA_NUM_SAMPLES
            dens = 1.0
            sliver = False
            tie = False
            contrib = 0.0
            a_s = 0.0
            # Dump-only locals; see raster_first_shade for the scoping note.
            d_mat = 0.0
            d_face = 0
            d_kind = 0
            d_sid = 0
            # Per-fragment run-correction factor and pristine area share
            # (dead outside run mode, like the dump locals above).
            cfac = 1.0
            run_pd = 0.0
            if not from_z:
                t_hit = _frag_t(frag_key[idx])
                ref = frag_ref[idx]
                if ti.static(has_bez):
                    is_bez = ref < 0
                if is_bez:
                    circuit, in_border = _decode_bez_ref(ref)
                    ref = circuit
                a = frag_ab[idx, 0]
                b = frag_ab[idx, 1]
                if ti.static(aa_bez or aa_tri):
                    cov = frag_cov[idx]
                    msk = frag_msk[idx]
            else:
                valid, is_bez, ref, t_hit, a, b, in_border = _terminal_z_hit(
                    zbuf[r], f, px, py, layer_offset_triangles, ss_enabled,
                    tri_pos, tri_screen, cam_origin, screen_point,
                    pixel_basis_x, pixel_basis_y, pixel_world_scale,
                    circuit_meta, circuit_colors, edges_2d, edge_accel,
                    gen_meta[2], gen_meta[3], aa_bez)
            q += 1
            if ti.static(dump):
                if (msk & _AA_BACKFACE_BIT) != 0:
                    d_face = 1
            if valid == 0:
                if ti.static(dump):
                    if dmatch:
                        _aa_dump_frag(dump_out, q - 1, 2, 5, ref, 0, 0, 0,
                                      cov, 0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      t_hit, svis)
                continue
            processed += 1
            w0 = 1.0 - a - b
            if ti.static(not aa_tri):
                edge_hit = 0
                if not is_bez:
                    if ti.min(w0, ti.min(a, b)) < TRIANGLE_EDGE_EPSILON:
                        edge_hit = 1
                if (edge_hit == 1) and (t_hit - seam_t <= DEPTH_TIE_EPSILON):
                    if ti.static(dump):
                        if dmatch:
                            _aa_dump_frag(dump_out, q - 1, 0, 6, ref, 0,
                                          d_face, msk, cov,
                                          _popcount_samples(msk), 1.0, 0.0,
                                          0.0, 0.0, 0.0, 0.0, t_hit, svis)
                    continue
                seam_t = t_hit if edge_hit == 1 else -1e30

            if ti.static(dump):
                if is_bez:
                    d_kind = 1
                if from_z:
                    d_kind += 2
                d_sid = -1 - ref
                if not is_bez:
                    d_sid = tri_obj[f % tri_obj.shape[0], ref]

            eff = cov
            if ti.static(aa_grp):
                sliver = False
                if ti.static(not cells):
                    sliver = (msk & _AA_SLIVER_BIT) != 0
                    msk &= _AA_MASK_ALL
                if ti.static(cells):
                    # WITHIN a surface, coverage ADDS; BETWEEN surfaces it
                    # composites. That is the whole idea: a mesh's triangles are
                    # pieces of one shape, and a 2D rasterizer never has this
                    # problem because it rasterizes a whole closed path at once.
                    # Summing exact clipped areas over a tiling is exact (it is
                    # property 3 of _aa_clip_area_check), so an interior edge
                    # stops existing rather than having to be partitioned.
                    #
                    # A closed mesh needs its two SHEETS kept apart, though. The
                    # front and back of a sphere both cover a silhouette pixel,
                    # and they cover the SAME part of it, so their areas must not
                    # add -- the object's coverage is the larger sheet, not the
                    # sum. The facing bit separates them exactly and for free.
                    sid = -1 - ref
                    fce = 0
                    if not is_bez:
                        sid = tri_obj[f % tri_obj.shape[0], ref]
                        if (msk & _AA_BACKFACE_BIT) != 0:
                            fce = 1
                    if sid != cur_obj:
                        rem *= 1.0 - obj_absorb
                        obj_absorb = 0.0
                        obj_cov = 0.0
                        sheet_cov = 0.0
                        cur_obj = sid
                        cur_face = fce
                    else:
                        if fce != cur_face:
                            obj_cov = ti.max(obj_cov, sheet_cov)
                            sheet_cov = 0.0
                            cur_face = fce
                    old_o = ti.max(obj_cov, sheet_cov)
                    sheet_cov = ti.min(1.0, sheet_cov + cov)
                    new_o = ti.max(obj_cov, sheet_cov)
                    # What this fragment newly reveals of the pixel.
                    contrib = new_o - old_o
                    eff = contrib * rem
                else:
                    # The run rule, in lockstep with raster_first_shade (any
                    # divergence desynchronizes every shadow id; see the
                    # kernel docstring).
                    if ti.static(_tri_run_mode(aa_tri)):
                        if (not is_bez) and (not from_z) \
                                and ((q - 1) >= run_end):
                            if ti.static(_tri_run_rule_b(aa_tri)):
                                if run_pending != 0:
                                    _run_redistribute(svis, run_U, run_resid)
                                    run_pending = 0
                                    run_resid = 0.0
                            run_mode = 0
                            run_end = q
                            if (msk & _AA_MASK_ALL) != _AA_MASK_ALL:
                                v0 = svis[0]
                                uni_v = v0 > 0.0
                                for s in ti.static(
                                        range(1, _AA_NUM_SAMPLES)):
                                    if svis[s] != v0:
                                        uni_v = False
                                if uni_v:
                                    to_row = f % tri_obj.shape[0]
                                    sid0 = tri_obj[to_row, ref]
                                    face0 = 0
                                    if (frag_msk[idx]
                                            & _AA_BACKFACE_BIT) != 0:
                                        face0 = 1
                                    rE, rU, rj = _aa_run_scan(
                                        q - 1, nrun, start, sid0, face0,
                                        to_row, frag_ref, frag_cov,
                                        frag_msk, tri_obj)
                                    run_end = rj
                                    if rU == _AA_MASK_ALL:
                                        run_mode = 1
                                        run_corr = 1.0
                                    elif rU == 0:
                                        run_mode = 2
                                        run_pscale = (ti.min(rE, 1.0)
                                                      / ti.max(rE, 1e-9))
                                        run_vstart = v0
                                        run_claimed = 0.0
                                    else:
                                        run_mode = 1
                                        qq_r = (ti.cast(
                                            _popcount_samples(rU), ti.f32)
                                            * _AA_SAMPLE_WEIGHT)
                                        # Capped by the tiling bound alone:
                                        # within one sheet exact areas sum to
                                        # <= 1 over the pixel, so E above 1 is
                                        # a mis-scan (overlap double-count)
                                        # and is capped, while E/Q well above
                                        # 1 is REAL for a sub-pixel rod that
                                        # owns one sample but covers several
                                        # samples' worth of area -- the
                                        # measured case that killed the
                                        # designed [0.5, 2] clamp (thin ink
                                        # stalled at 0.88). Rule B's
                                        # redistribution keeps the occlusion
                                        # side exact under large corr.
                                        run_corr = ti.min(rE, 1.0) / qq_r
                                    if ti.static(
                                            _tri_run_rule_b(aa_tri)):
                                        if run_mode == 1:
                                            run_U = rU
                                            run_resid = 0.0
                                            run_pending = 1
                    slots, nsm, dens = _coverage_slots(
                        cov, msk, is_bez or sliver, cells,
                        ti.static(_tri_run_mode(aa_tri)))
                    vis = 0.0
                    for s in ti.static(range(_AA_NUM_SAMPLES)):
                        vis += slots[s] * svis[s]
                    eff = vis * _AA_SAMPLE_WEIGHT * dens
                    if ti.static(_tri_run_mode(aa_tri)):
                        if (not is_bez) and (not from_z) \
                                and ((q - 1) < run_end):
                            if run_mode == 1:
                                cfac = run_corr
                                eff *= run_corr
                            elif run_mode == 2:
                                run_pd = run_pscale * cov
                                eff = run_pd * run_vstart
                                dens = run_pd / ti.max(
                                    1.0 - run_claimed, 1e-6)
                                for s in ti.static(range(_AA_NUM_SAMPLES)):
                                    slots[s] = 1.0
                                nsm = _AA_NUM_SAMPLES
                if eff <= MIN_ALPHA:
                    if ti.static(dump):
                        if dmatch:
                            _aa_dump_frag(dump_out, q - 1, d_kind, 1, ref,
                                          d_sid, d_face, msk, cov,
                                          _popcount_samples(msk), 1.0, eff,
                                          0.0, 0.0, 0.0, 0.0, t_hit, svis)
                    continue

            alpha = 0.0
            reflectivity = 0.0
            ior = 0.0
            transmission = 0.0
            albedo = ti.math.vec3(0.0, 0.0, 0.0)
            normal = ti.math.vec3(0.0, 0.0, 0.0)
            if is_bez:
                color, alpha = _sample_circuit_color_blend(
                    ref, f, a, b, in_border, circuit_meta, circuit_colors,
                    circuit_border_colors)
                albedo = ti.math.vec3(color[0], color[1], color[2])
                cm = f % circuit_meta.shape[0]
                reflectivity = circuit_meta[cm, ref, _M_REFLECTIVITY]
                ior = circuit_meta[cm, ref, _M_IOR]
                transmission = circuit_meta[cm, ref, _M_TRANSMISSION]
                if (reflectivity >= 0.0) or (transmission > 1e-4):
                    normal = _bezier_normal(f, ref, circuit_meta).normalized()
            else:
                color, alpha = _tri_color_g(
                    0, f, ref, w0, a, b, tri_colors, col_row, tri_uvs,
                    tri_tex_meta, textures, num_colored_triangles)
                albedo = ti.math.vec3(color[0], color[1], color[2])
                reflectivity, _rough = _tri_extra_g(
                    0, f, ref, w0, a, b, tri_extra, col_row, tri_uvs,
                    tri_tex_meta, textures, num_colored_triangles)
                ior = _tri_ior_g(
                    0, f, ref, w0, a, b, tri_extra, col_row, tri_uvs,
                    tri_tex_meta, textures, num_colored_triangles)
                transmission = _tri_transmission_g(
                    0, f, ref, w0, a, b, tri_extra, col_row, tri_uvs,
                    tri_tex_meta, textures, num_colored_triangles)
                # This exact accepted triangle event receives one sparse shadow
                # queue entry, irrespective of raw list position/depth.
                eid = ti.atomic_add(event_count[0], 1)
                if from_z:
                    z_shadow_id[r] = eid
                else:
                    frag_shadow_id[idx] = eid
                # Rebuilt from the fragment's own barycentrics, so a partially
                # covering fragment's shadow origin sits on ITS triangle rather
                # than wherever the centre ray reaches its centroid-measured
                # depth (see _tri_surface_point). Off analytic coverage every
                # fragment is a full centre hit and the two agree exactly, so
                # that path stays byte-identical.
                hp = ro + t_hit * rd
                srd = rd
                if ti.static(aa_tri):
                    hp = _tri_surface_point(f, ref, w0, a, b, tri_pos)
                    partial = cov < AA_FULL_COVERAGE
                    if ti.static(_tri_run_mode(aa_tri)):
                        partial = (msk & _AA_MASK_ALL) != _AA_MASK_ALL
                    if partial:
                        srd = (hp - ro).normalized()
                # The facing test must use the SAME ray the resolve does
                # (``surf_rd`` in raster_first_shade). Both kernels decide which
                # side of this fragment the viewer is on -- the resolve to
                # orient the shading and reflection normal, this one to pick the
                # side the shadow origin is offset towards. Inside the ~0.04
                # degree band where the two rays straddle the horizon they can
                # disagree, and then the shadow origin is pushed THROUGH the
                # surface and the ray self-shadows the pixel it came from.
                snrm, fnrm = _tri_shadow_normals(
                    f, ref, a, b, srd, tri_pos, tri_norm, tri_uvs,
                    tri_tex_meta, textures, num_colored_triangles)
                for k in ti.static(range(3)):
                    event_pos[eid, k] = hp[k]
                    event_snrm[eid, k] = snrm[k]
                    event_fnrm[eid, k] = fnrm[k]
                event_frame[eid] = f
                # Shadow visibility is sampled only at sub-pixel positions the
                # triangle fragment owns. A terminal z-winner owns all four
                # positions; an areal sliver has no discrete position and uses
                # all four as the least-biased representation of its area.
                shadow_msk = 0xF
                if ti.static(aa_tri):
                    # An empty-mask run-mode sliver is areal like the old
                    # sliver policies: no discrete position, so all four
                    # sub-pixel shadow positions represent its area.
                    if (not from_z) and (not sliver) \
                            and ((msk & _AA_MASK_ALL) != 0):
                        shadow_msk, _shadow_n = _sec_positions(
                            msk & _AA_MASK_ALL, 4)
                event_msk[eid] = shadow_msk
                if ti.static(sec_aa > 1):
                    # Footprint for sub-pixel shadow sampling: coverage cannot
                    # antialias a shadow EDGE, because visibility is a binary
                    # query at one point per fragment (ss7). The trace resolves
                    # it by moving that point over the pixel.
                    dpx, dpy = _pixel_footprint(
                        f, px, py, gen_meta, hp, fnrm, cam_origin,
                        screen_point, pixel_basis_x, pixel_basis_y)
                    for k in ti.static(range(3)):
                        event_dp[eid, k] = dpx[k]
                        event_dp[eid, 3 + k] = dpy[k]
                if (reflectivity >= 0.0) or (transmission > 1e-4):
                    normal = snrm

            if ti.static(aa_bez or aa_tri):
                mat_alpha = ti.math.clamp(alpha, 0.0, 1.0)
                alpha = mat_alpha * eff
                if ti.static(aa_grp):
                    a_s = mat_alpha * dens
                if ti.static(dump):
                    d_mat = mat_alpha
            alpha = ti.math.clamp(alpha, 0.0, 1.0)
            transmission = ti.math.clamp(transmission, 0.0, 1.0)
            R, diel_pass = _material_reflectance(
                rd, normal, reflectivity, ior, albedo)
            if bounces_left <= 0:
                R = ti.math.vec3(0.0, 0.0, 0.0)
            is_glass = False
            is_pane = False
            if ti.static(refraction != 0):
                if (transmission > 1e-4) and (bounces_left > 0) \
                        and (ior > 1.0 + 1e-4):
                    if is_bez:
                        is_pane = True
                    else:
                        is_glass = True
            tint = ti.math.clamp(albedo, 0.0, 1.0)
            trans_share = diel_pass * transmission
            refl_energy = alpha * R
            refl_max = ti.max(refl_energy[0],
                              ti.max(refl_energy[1], refl_energy[2]))
            trans_energy = alpha * trans_share
            cover_pass = 1.0 - alpha
            cover3 = ti.math.vec3(cover_pass, cover_pass, cover_pass)
            split_refl = False
            if ti.static(refraction != 0):
                if (refl_max > MIN_ALPHA) and (cover_pass > MIN_ALPHA) \
                        and (bounces_left > 0):
                    split_refl = True
            one3 = ti.math.vec3(1.0, 1.0, 1.0)
            if is_glass:
                if (refl_max > MIN_ALPHA) and (refl_max >= cover_pass):
                    if ti.static(dump):
                        if dmatch:
                            _aa_dump_frag(dump_out, q - 1, d_kind, 2, ref,
                                          d_sid, d_face, msk, cov,
                                          _popcount_samples(msk), 1.0, eff,
                                          d_mat, alpha, trans_share,
                                          refl_max, t_hit, svis)
                    break
                if ti.static(aa_grp):
                    for s in ti.static(range(_AA_NUM_SAMPLES)):
                        svis[s] *= 1.0 - a_s * slots[s]
                else:
                    weight *= cover_pass
            elif is_pane or split_refl:
                if ti.static(aa_grp):
                    # Pass-through, per covered sample. The MAGNITUDE is
                    # per-sample; the tint is not (that would need a vec3 per
                    # sample), so a chromatic transmitter applies its colour
                    # ratio to the whole pixel scaled by how much of it the
                    # fragment covers -- exact for a fully covering pane, which
                    # is what tinted glass almost always is.
                    ts_s = a_s * trans_share
                    pm = (1.0 - a_s) + ts_s
                    if ti.static(cells):
                        obj_absorb += a_s * (1.0 - trans_share) * contrib
                    elif ti.static(_tri_run_mode(aa_tri)):
                        rr = _run_svis_write(
                            svis, slots, a_s, trans_share, cfac,
                            ti.static(_tri_run_rule_b(aa_tri)))
                        if ti.static(_tri_run_rule_b(aa_tri)):
                            run_resid += rr
                        if run_pd > 0.0:
                            run_claimed += (a_s * (1.0 - run_claimed)
                                            * (1.0 - trans_share))
                    else:
                        for s in ti.static(range(_AA_NUM_SAMPLES)):
                            ak = a_s * slots[s]
                            svis[s] *= (1.0 - ak) + ak * trans_share
                    if ts_s > 1e-6:
                        frac = cfac * ti.cast(nsm, ti.f32) * _AA_SAMPLE_WEIGHT
                        if ti.static(cells):
                            fsum = 0.0
                            for s in ti.static(range(_AA_NUM_SAMPLES)):
                                fsum += slots[s]
                            frac = fsum * _AA_SAMPLE_WEIGHT
                        num = (ti.math.vec3(1.0, 1.0, 1.0) * (1.0 - a_s)
                               + ts_s * tint)
                        weight *= one3 + (num / ti.max(pm, 1e-6) - one3) * frac
                else:
                    weight *= cover3 + trans_energy * tint
            elif (refl_max > MIN_ALPHA) and (refl_max >= cover_pass):
                if ti.static(dump):
                    if dmatch:
                        _aa_dump_frag(dump_out, q - 1, d_kind, 2, ref,
                                      d_sid, d_face, msk, cov,
                                      _popcount_samples(msk), 1.0, eff,
                                      d_mat, alpha, trans_share, refl_max,
                                      t_hit, svis)
                break
            else:
                if ti.static(aa_grp):
                    # Pass-through, per covered sample. The MAGNITUDE is
                    # per-sample; the tint is not (that would need a vec3 per
                    # sample), so a chromatic transmitter applies its colour
                    # ratio to the whole pixel scaled by how much of it the
                    # fragment covers -- exact for a fully covering pane, which
                    # is what tinted glass almost always is.
                    ts_s = a_s * trans_share
                    pm = (1.0 - a_s) + ts_s
                    if ti.static(cells):
                        obj_absorb += a_s * (1.0 - trans_share) * contrib
                    elif ti.static(_tri_run_mode(aa_tri)):
                        rr = _run_svis_write(
                            svis, slots, a_s, trans_share, cfac,
                            ti.static(_tri_run_rule_b(aa_tri)))
                        if ti.static(_tri_run_rule_b(aa_tri)):
                            run_resid += rr
                        if run_pd > 0.0:
                            run_claimed += (a_s * (1.0 - run_claimed)
                                            * (1.0 - trans_share))
                    else:
                        for s in ti.static(range(_AA_NUM_SAMPLES)):
                            ak = a_s * slots[s]
                            svis[s] *= (1.0 - ak) + ak * trans_share
                    if ts_s > 1e-6:
                        frac = cfac * ti.cast(nsm, ti.f32) * _AA_SAMPLE_WEIGHT
                        if ti.static(cells):
                            fsum = 0.0
                            for s in ti.static(range(_AA_NUM_SAMPLES)):
                                fsum += slots[s]
                            frac = fsum * _AA_SAMPLE_WEIGHT
                        num = (ti.math.vec3(1.0, 1.0, 1.0) * (1.0 - a_s)
                               + ts_s * tint)
                        weight *= one3 + (num / ti.max(pm, 1e-6) - one3) * frac
                else:
                    weight *= cover3 + trans_energy * tint
            if ti.static(dump):
                if dmatch:
                    _aa_dump_frag(dump_out, q - 1, d_kind, 0, ref, d_sid,
                                  d_face, msk, cov, _popcount_samples(msk),
                                  cfac, eff, d_mat, alpha, trans_share,
                                  refl_max, t_hit, svis)
            cur_w = weight
            if ti.static(aa_grp):
                vis_all = 0.0
                if ti.static(cells):
                    vis_all = rem * (1.0 - obj_absorb) * _AA_NUM_SAMPLES
                else:
                    for s in ti.static(range(_AA_NUM_SAMPLES)):
                        vis_all += svis[s]
                cur_w = weight * (vis_all * _AA_SAMPLE_WEIGHT)
            if ti.max(cur_w[0], ti.max(cur_w[1], cur_w[2])) < MIN_WEIGHT:
                break
        if ti.static(_tri_run_mode(aa_tri) and _tri_run_rule_b(aa_tri)):
            if run_pending != 0:
                _run_redistribute(svis, run_U, run_resid)
                run_pending = 0
        if ti.static(dump):
            if dmatch:
                d_vis = 0.0
                for s in ti.static(range(_AA_NUM_SAMPLES)):
                    d_vis += svis[s]
                _aa_dump_terminal(dump_out, False, processed >= total,
                                  processed, d_vis * _AA_SAMPLE_WEIGHT,
                                  ti.math.vec4(0.0, 0.0, 0.0, 0.0), weight,
                                  svis)


@ti.kernel
def raster_shadow_trace(
        num_events: int,
        event_pos: ti.types.ndarray(), event_snrm: ti.types.ndarray(),
        event_fnrm: ti.types.ndarray(), event_frame: ti.types.ndarray(),
        event_msk: ti.types.ndarray(),
        t_nodes: NODE_ARG, t_node_miss: ti.types.ndarray(),
        t_leaf_prim: ti.types.ndarray(), t_leaf_tspan: ti.types.ndarray(),
        t_first_leaf: int,
        tri_pos: ti.types.ndarray(), tri_colors: ti.types.ndarray(),
        tri_uvs: ti.types.ndarray(), tri_tex_meta: ti.types.ndarray(),
        textures: ti.types.ndarray(), num_colored_triangles: ti.i32,
        p_nodes: NODE_ARG, p_node_miss: ti.types.ndarray(),
        p_leaf_prim: ti.types.ndarray(), p_leaf_tspan: ti.types.ndarray(),
        p_first_leaf: int,
        pn_ctrl: ti.types.ndarray(), pn_obb: ti.types.ndarray(),
        pn_colors: ti.types.ndarray(),
        b_nodes: NODE_ARG, b_node_miss: ti.types.ndarray(),
        b_leaf_prim: ti.types.ndarray(), b_leaf_tspan: ti.types.ndarray(),
        b_first_leaf: int,
        circuit_meta: ti.types.ndarray(), circuit_colors: ti.types.ndarray(),
        circuit_border_colors: ti.types.ndarray(),
        edges_2d: ti.types.ndarray(), edge_accel: ti.types.ndarray(),
        light_pos: ti.types.ndarray(), light_col: ti.types.ndarray(),
        num_lights: int, pixel_world_scale: ti.types.ndarray(),
        layer_offset_triangles: ti.f32, layer_offset_pn: ti.f32,
        refit: ti.template(),
        has_tri: ti.template(), has_pn: ti.template(), has_bez: ti.template(),
        event_dp: ti.types.ndarray(), sec_aa: ti.template(),
        shadow_vis: ti.types.ndarray()):
    """Trace the dedicated sparse any-hit shadow queue exactly.

    A zero-radius point/spot/directional light emits one hard-shadow ray.
    Non-zero emitter radii use the same fixed golden-angle fan as the classic
    wavefront shader; area lights are already expanded into packed sample rows
    and therefore naturally obtain soft visibility by averaging those rows in
    the material shader.

    ``sec_aa`` (``ANALYTIC_AA_SECONDARY_SAMPLES``): visibility is a BINARY query,
    so analytic coverage cannot antialias a shadow edge however exact the
    geometry is -- the stair steps just move from the silhouette to the shadow
    (ss7). With it on, the query point moves over the pixel instead: four
    sub-pixel positions on the shading surface, from the world-space footprint
    the event build stored (``event_dp``), averaged. For a HARD light that is 4
    rays instead of 1 at exactly the positions ``anti_alias_level = 2`` samples;
    for a soft one the existing fan is simply spread over those positions too,
    which costs nothing.
    """
    for e in range(num_events):
        f = event_frame[e]
        ff = ti.cast(f, ti.f32)
        spos = ti.math.vec3(event_pos[e, 0], event_pos[e, 1], event_pos[e, 2])
        snrm = ti.math.vec3(event_snrm[e, 0], event_snrm[e, 1],
                            event_snrm[e, 2])
        fnrm = ti.math.vec3(event_fnrm[e, 0], event_fnrm[e, 1],
                            event_fnrm[e, 2])
        sorigin = spos + fnrm * (10.0 * MIN_HIT_DISTANCE)
        dpx = ti.math.vec3(0.0, 0.0, 0.0)
        dpy = ti.math.vec3(0.0, 0.0, 0.0)
        if ti.static(sec_aa > 1):
            dpx = ti.math.vec3(event_dp[e, 0], event_dp[e, 1], event_dp[e, 2])
            dpy = ti.math.vec3(event_dp[e, 3], event_dp[e, 4], event_dp[e, 5])
        tl = f % light_pos.shape[0]
        for li in range(num_lights):
            visibility = 1.0
            lp = ti.math.vec3(light_pos[tl, li, 0], light_pos[tl, li, 1],
                              light_pos[tl, li, 2])
            ltype = 0
            radius = 0.0
            if light_col.shape[2] > 3:
                ltype = ti.cast(light_col[tl, li, 3] + 0.5, ti.i32)
                if light_col.shape[2] > 11:
                    radius = light_col[tl, li, 11]
            to_light = lp - spos
            ldist = to_light.norm()
            wi = ti.math.vec3(0.0, 0.0, 0.0)
            valid = 0
            if ltype == _LT_DIRECTIONAL:
                wi = -ti.math.vec3(light_col[tl, li, 6],
                                   light_col[tl, li, 7],
                                   light_col[tl, li, 8])
                ldist = 1e7
                valid = 1
            elif (ltype != _LT_AMBIENT) and (ltype != _LT_HEMISPHERE) \
                    and (ltype != _LT_ENV_SH) and (ldist > 1e-5):
                wi = to_light / ldist
                valid = 1
            if valid == 1:
                ns = 1
                b1 = ti.math.vec3(0.0, 0.0, 0.0)
                b2 = ti.math.vec3(0.0, 0.0, 0.0)
                if radius > 0.0:
                    ns = SOFT_SHADOW_SAMPLES
                    aref = ti.math.vec3(1.0, 0.0, 0.0)
                    if ti.abs(wi[0]) > 0.9:
                        aref = ti.math.vec3(0.0, 1.0, 0.0)
                    b1 = wi.cross(aref).normalized()
                    b2 = wi.cross(b1)
                if ti.static(sec_aa > 1):
                    # A hard light needs the sub-pixel positions to be separate
                    # rays; a soft one already has enough rays and just spreads
                    # its fan over them.
                    ns = ti.max(ns, 4)

                occ_sum = 0.0
                n_valid = 0.0
                for s in range(ns):
                    wis = wi
                    ldn = ldist
                    ok = 1
                    sorg = sorigin
                    if ti.static(sec_aa > 1):
                        sorg = _sub_pixel_origin(sorigin, dpx, dpy, s)
                        if ((event_msk[e] >> (s & 3)) & 1) == 0:
                            ok = 0
                    off = ti.math.vec3(0.0, 0.0, 0.0)
                    if radius > 0.0:
                        ang = _GOLDEN_ANGLE * s
                        rr = radius * ti.sqrt(
                            (ti.cast(s, ti.f32) + 0.5)
                            / ti.cast(ns, ti.f32))
                        off = (ti.cos(ang) * b1 + ti.sin(ang) * b2) * rr
                        if ltype == _LT_DIRECTIONAL:
                            wis = (wi + off).normalized()
                    if ltype != _LT_DIRECTIONAL:
                        # Moving the origin over the pixel changes both the
                        # direction and finite distance to a point/spot/area
                        # emitter. Retaining the centre ray here makes the
                        # samples non-convergent and can trace past the light.
                        tls = lp + off - sorg
                        ldn = tls.norm()
                        if ldn > 1e-5:
                            wis = tls / ldn
                        else:
                            ok = 0
                    if (ok == 1) and (fnrm.dot(wis) > 1e-3) \
                            and (snrm.dot(wis) > 1e-4):
                        n_valid += 1.0
                        occ_sum += _shadow_occluded(
                            refit, sorg, wis, f, ff,
                            ldn - 20.0 * MIN_HIT_DISTANCE,
                            pixel_world_scale[
                                f % pixel_world_scale.shape[0]], 0.0,
                            layer_offset_triangles, layer_offset_pn,
                            has_tri, has_pn, has_bez,
                            t_nodes, t_node_miss, t_leaf_prim, t_leaf_tspan,
                            t_first_leaf, tri_pos, tri_colors, tri_uvs,
                            tri_tex_meta, textures, num_colored_triangles,
                            p_nodes, p_node_miss, p_leaf_prim, p_leaf_tspan,
                            p_first_leaf, pn_ctrl, pn_obb, pn_colors,
                            b_nodes, b_node_miss, b_leaf_prim, b_leaf_tspan,
                            b_first_leaf, circuit_meta, circuit_colors,
                            circuit_border_colors, edges_2d, edge_accel)
                if n_valid > 0.0:
                    visibility = 1.0 - occ_sum / n_valid
            shadow_vis[e, li] = visibility


@ti.kernel
def raster_first_shade(
        num_pixels: int,
        run_offsets: ti.types.ndarray(),
        frag_key: ti.types.ndarray(), frag_ref: ti.types.ndarray(),
        frag_ab: ti.types.ndarray(), frag_cov: ti.types.ndarray(),
        frag_msk: ti.types.ndarray(),
        zbuf: ti.types.ndarray(),
        tri_pos: ti.types.ndarray(), tri_screen: ti.types.ndarray(),
        tri_norm: ti.types.ndarray(),
        tri_extra: ti.types.ndarray(), tri_colors: ti.types.ndarray(),
        tri_uvs: ti.types.ndarray(), tri_tex_meta: ti.types.ndarray(),
        textures: ti.types.ndarray(), num_colored_triangles: ti.i32,
        col_row: ti.types.ndarray(),
        tri_obj: ti.types.ndarray(),
        tri_mat_id: ti.types.ndarray(), tri_mat: ti.types.ndarray(),
        # Bezier circuit shading data (used when has_bez; 1x1 placeholders
        # otherwise). Circuits route entirely through the transparent fragment
        # stream or typed opaque visibility buffer, tagged by a negative ref
        # when it appears in the translucent fragment stream.
        circuit_meta: ti.types.ndarray(), circuit_colors: ti.types.ndarray(),
        circuit_border_colors: ti.types.ndarray(),
        pixel_world_scale: ti.types.ndarray(), edges_2d: ti.types.ndarray(),
        edge_accel: ti.types.ndarray(),
        light_pos: ti.types.ndarray(), light_col: ti.types.ndarray(),
        num_lights: int,
        layer_offsets: ti.types.ndarray(),
        frag_shading: ti.template(), frag_pipelines: ti.template(),
        refraction: ti.template(), skip_unlit_normal: ti.template(),
        ss_enabled: ti.template(), has_bez: ti.template(),
        aa_bez: ti.template(), aa_tri: ti.template(), aa_grp: ti.template(),
        sec_aa: ti.template(), sec_min_energy: ti.f32,
        glossy: ti.template(),
        shadows: ti.template(), prefill: ti.template(),
        covered: ti.template(),
        covered_idx: ti.types.ndarray(), num_covered: int,
        compact: ti.template(),
        time_start: int, width: int, height: int, tile_start: int,
        cam_origin: ti.types.ndarray(), screen_point: ti.types.ndarray(),
        pixel_basis_x: ti.types.ndarray(), pixel_basis_y: ti.types.ndarray(),
        gen_meta: ti.types.ndarray(),
        rs_ro: ti.types.ndarray(), rs_rd: ti.types.ndarray(),
        rs_acc: ti.types.ndarray(), rs_sca: ti.types.ndarray(),
        rs_int: ti.types.ndarray(), rs_pix: ti.types.ndarray(),
        pix_accum: ti.types.ndarray(), rs_alloc: ti.types.ndarray(),
        frag_shadow_id: ti.types.ndarray(), z_shadow_id: ti.types.ndarray(),
        shadow_vis: ti.types.ndarray(),
        dump: ti.template(), dump_out: ti.types.ndarray()):
    """Resolve + shade each pixel's complete straight-line hit list.

    One thread per tile pixel walks its sorted transparent run front-to-back
    (the z-prepass winner logically appended as the final, terminal hit) with
    the same per-hit logic as ``wavefront_shade``'s drain loop -- triangle
    colour/alpha, fragment-pipeline shading, and the built-in four-way
    reflect/transmit continuation, including split spawns into the shared
    pool. A reflected continuation is written into the pixel's own ray slot
    (status ACTIVE) for the classic wavefront iterations; every other pixel
    retires here, committing colour + leftover background weight into
    ``pix_accum``. Free pool slots must be pre-marked DONE by the host.

    ``prefill`` (compile-time): the host pre-filled every primary's
    ``pix_accum`` row with the retired-empty result ``[0,0,0,0, 1,1,1]``
    (status already DONE), so empty pixels exit before ray generation with
    zero writes, retiring pixels *store* their leftover weight into cols
    4-6, and bouncing pixels zero those columns back out. Byte-identical to
    the accumulate-onto-zero path (``ALGAN_RASTER_EMPTY_SKIP``).

    ``covered`` (compile-time, ``ALGAN_RASTER_COVERED_SHADE``): the host
    passed a compact ascending list ``covered_idx[0:num_covered]`` of the
    only pixels with a fragment or z-winner, so the loop runs one thread per
    covered pixel (``r = covered_idx[t]``) instead of one per tile pixel.
    Empty pixels are simply never launched -- their retired-empty pre-fill
    already holds their final state -- turning the resolve from O(tile
    pixels) into O(covered pixels). Requires ``prefill``; the ascending
    order preserves the original relative shading order (byte-identical).
    ``compact`` additionally means all CSR, z, ray-state, and accumulator rows
    are indexed by covered ordinal ``t``; ``covered_idx[t]`` is then used only
    for the actual frame/pixel coordinate.

    ``aa_bez`` / ``aa_tri`` (compile-time, ``ALGAN_ANALYTIC_AA``): each fragment
    carries the fraction of the pixel square its primitive covers in
    ``frag_cov``, and that scales its alpha -- so a silhouette pixel blends
    against whatever is behind it instead of being all-or-nothing. A z-prepass
    winner is fully covering by construction, hence coverage 1.
    ``raster_shadow_event_build`` applies the identical scaling; the two walks
    must stay in lockstep or its shadow ids desynchronize from the fragments.

    PER-SAMPLE TRANSMITTANCE (``aa_grp``, DESIGN_analytic_aa.md ss5 and ss18).
    Scaling alpha by coverage alone is only correct for ONE covering fragment.
    Two triangles of a mesh that split a pixel 40/60 at a shared edge would
    composite to 24% background showing through a fully covered pixel: a
    background-coloured lattice on every internal edge of every mesh. Their
    sub-areas are DISJOINT, not independent.

    So this walk keeps ``svis[s]``, the transmittance of each sub-pixel sample,
    and a fragment's effective coverage is how much light still reaches the
    samples it covers::

        eff = sum(svis[s] for s in mask) / N
        for s in mask: svis[s] *= pass_through

    That one array subsumes three earlier mechanisms and is exact where each of
    them was not: shared edges (disjoint masks sum to the pixel, no lattice),
    opaque occlusion (an opaque fragment zeroes its samples, so a mesh's back
    faces cannot add to its front faces), and stacked or INTERPENETRATING
    translucent surfaces (each sample composites in true depth order, so it no
    longer matters whether one object's fragments are consecutive -- the
    same-object grouping this replaced assumed they were, and two
    interpenetrating translucent meshes made it fail badly).

    ``weight`` carries only what cannot be per-sample: the CHROMATIC ratio of a
    tinted transmitter, applied to the whole pixel scaled by the fragment's
    coverage. Exact for a fully covering pane, which tinted glass almost always
    is. The pixel's throughput is ``weight * mean(svis)``.

    ``aa_grp`` off (``ALGAN_ANALYTIC_AA_SEAM=0``) restores the plain
    multiplicative walk -- the lattice -- and exists only so the parity script
    can measure what the rule is worth. The classic seam de-duplication is
    disabled under ``aa_tri``: it exists to drop the second of two hits on a
    shared edge, which is precisely the fragment coverage needs (and clamped
    barycentrics would make every partial fragment look like an edge hit
    anyway).

    ``sec_aa`` (compile-time, ``ANALYTIC_AA_SECONDARY_SAMPLES``): coverage
    antialiases a mirror's OUTLINE but not the image inside it, which is sampled
    by the continuation ray. At N > 1 a reflective or refractive hit spawns N
    continuations instead of one -- the primary ray re-generated through N
    sub-pixel positions and re-intersected with that hit's own plane
    (``_jittered_plane_hit``), each carrying 1/N of the throughput. For a bounce
    that continues in place, sub-sample 0 keeps the pixel's own slot (and its
    accumulated colour) and the rest go to the shared pool; since every branch
    commits both its colour and its leftover background weight when it retires,
    the pixel's totals are unchanged. The split happens only here, at the primary
    hit, so deeper bounces do not multiply. N == 1 compiles to exactly the
    single-ray code and is byte-identical.

    ``glossy`` (compile-time, ``GLOSSY_REFLECTION``; 1 fan, 2 fan + per-pixel
    rotation): those N continuations vary in LOBE DIRECTION as well as sub-pixel
    position, spread over the material's GGX lobe (``_glossy_reflect``), so a
    rough reflector's reflected image blurs with its roughness instead of
    staying razor-sharp beside a broad direct highlight. It reuses the taps that
    already exist, so it costs no extra rays and no extra pool slots -- and it
    therefore reaches only a fragment taking the secondary-sampling branch; one
    tap stays specular-perfect, because a single deterministic sample of a lobe
    is not a blur. Roughness below ``_GLOSSY_MIN_ROUGHNESS`` takes the untouched
    mirror expression, so a mirror is byte-identical. The blur applies to
    REFLECTED continuations only; a refracted ray through rough glass stays
    unbent-sharp (a separate lobe, and the transmitted branch has no `placed`
    in-slot ray to keep coherent).

    ``layer_offsets`` is the tracer's 8-wide variant: [2..5] environment map
    placement, [6] far clip (0 = off), [7] max_bounces.
    """
    pixels_per_frame = width * height
    env_off = ti.cast(layer_offsets[2] + 0.5, ti.i32)
    env_w = ti.cast(layer_offsets[3] + 0.5, ti.i32)
    env_h = ti.cast(layer_offsets[4] + 0.5, ti.i32)
    env_intensity = layer_offsets[5]
    far_clip = layer_offsets[6]
    max_bounces = ti.cast(layer_offsets[7] + 0.5, ti.i32)
    loop_n = num_pixels
    if ti.static(covered):
        loop_n = num_covered
    for t in range(loop_n):
        r = t
        pixel = r
        if ti.static(covered):
            pixel = covered_idx[t]
            if ti.static(not compact):
                r = pixel
        start = run_offsets[r]
        end = run_offsets[r + 1]
        nrun = end - start
        zk = zbuf[r]
        has_z = 1 if zk != ti.i64(Z_SENTINEL) else 0
        total = nrun + has_z
        if ti.static(prefill):
            # The host pre-filled every primary's committed state with the
            # retired-empty result (pix_accum row [0,0,0,0, 1,1,1], pool
            # status DONE), so a pixel with no fragments, no z-prepass
            # winner and no environment map to sample is already complete:
            # exit with zero writes, skipping ray generation entirely.
            if total == 0 and env_w <= 0:
                continue

        g = tile_start + pixel
        f_rel = g // pixels_per_frame
        p = g - f_rel * pixels_per_frame
        f = time_start + f_rel
        py = p // width
        px = p - py * width
        ro, rd = _generate_ray(f, px, py, gen_meta[0], gen_meta[1],
                               gen_meta[2], gen_meta[3],
                               cam_origin, screen_point,
                               pixel_basis_x, pixel_basis_y)
        dmatch = False
        if ti.static(dump):
            dmatch = _aa_dump_match(dump_out, px, py, f)
        # Which part of the reflection lobe this pixel's taps sample. Hoisted
        # out of the fragment walk because it depends on the PIXEL alone.
        g_roff = 0.5
        g_aoff = 0.0
        if ti.static(glossy != 0):
            g_roff, g_aoff = _glossy_rotation(px, py, ti.static(glossy == 2))

        acc = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
        # Under analytic coverage ``weight`` carries only the CHROMATIC part of
        # the transmittance (tints); the achromatic part is per-sample in
        # ``svis``, and the pixel's actual throughput is their product. With
        # coverage off, ``weight`` is the classic running transmittance and
        # ``svis`` stays at one.
        weight = ti.math.vec3(1.0, 1.0, 1.0)
        # Per-sample transmittance: how much light still reaches each sub-pixel
        # sample. This single array replaces the coverage-group rule, the opaque
        # sample mask and the object ids all at once -- see the docstring.
        cells = ti.static(_tri_cells(aa_tri))
        svis = ti.Vector([1.0 for _ in range(_AA_NUM_SAMPLES)])
        seam_t = -1e30
        cprev_t = -1e30
        # RUN state (DESIGN_analytic_aa_v2.md ss4.2), dead outside run mode:
        # the exclusive end of the scanned run, its kind (0 uncorrected, 1
        # corrected, 2 pristine all-sliver), the magnitude correction, and the
        # pristine claim bookkeeping (area scale, svis at run start, fraction
        # of it already claimed). Rule B adds the run's owned-sample union,
        # the clamped write residue, and a pending flag for its run-end step.
        run_end = 0
        run_mode = 0
        run_corr = 1.0
        run_pscale = 0.0
        run_vstart = 0.0
        run_claimed = 0.0
        run_U = 0
        run_resid = 0.0
        run_pending = 0
        # SURFACE accounting (DESIGN ss21.9). Four scalars replace the whole
        # per-sample array: the pixel's remaining transmittance in front of the
        # current object, how much of the pixel the current SHEET has covered,
        # the largest sheet the current object has managed, and what the object
        # has absorbed so far.
        rem = 1.0
        sheet_cov = 0.0
        obj_cov = 0.0
        obj_absorb = 0.0
        cur_obj = -2147483647
        cur_face = 0
        base_dist = 0.0
        bounces_left = max_bounces
        processed = 0
        bounced = False
        done = False

        q = 0
        while q < total and processed < MAX_SURFACES_PER_RAY:
            t_hit = 0.0
            prim_raw = 0
            a = 0.0
            b = 0.0
            in_border = 0.0
            from_z = q >= nrun
            idx = start + q
            valid = 1
            # The z-winner is a fully covering hit by construction, so it
            # claims and occludes every sub-pixel sample.
            cov = 1.0
            msk = _AA_MASK_ALL
            slots = ti.Vector([1.0 for _ in range(_AA_NUM_SAMPLES)])
            nsm = _AA_NUM_SAMPLES
            dens = 1.0
            sliver = False
            tie = False
            contrib = 0.0
            a_s = 0.0
            # Dump-only locals, declared unconditionally so Taichi's block
            # scoping accepts the assignments inside static-dump regions; the
            # constants are dead (and eliminated) in non-dump builds.
            d_mat = 0.0
            d_face = 0
            d_kind = 0
            d_sid = 0
            # Per-fragment run-correction factor and pristine area share
            # (dead outside run mode, like the dump locals above).
            cfac = 1.0
            run_pd = 0.0
            if q < nrun:
                t_hit = _frag_t(frag_key[idx])
                prim_raw = frag_ref[idx]
                a = frag_ab[idx, 0]
                b = frag_ab[idx, 1]
                if ti.static(aa_bez or aa_tri):
                    cov = frag_cov[idx]
                    msk = frag_msk[idx]
            else:
                valid, is_z_bez, zprim, t_hit, a, b, in_border = \
                    _terminal_z_hit(
                        zk, f, px, py, layer_offsets[0], ss_enabled,
                        tri_pos, tri_screen, cam_origin, screen_point,
                        pixel_basis_x, pixel_basis_y, pixel_world_scale,
                        circuit_meta, circuit_colors, edges_2d, edge_accel,
                        gen_meta[2], gen_meta[3], aa_bez)
                prim_raw = zprim
                if is_z_bez:
                    prim_raw = _pack_bez_ref(zprim, in_border)
            q += 1
            if ti.static(dump):
                if (msk & _AA_BACKFACE_BIT) != 0:
                    d_face = 1
            if valid == 0:
                if ti.static(dump):
                    if dmatch:
                        _aa_dump_frag(dump_out, q - 1, 2, 5, prim_raw, 0, 0,
                                      0, cov, 0, 1.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, t_hit, svis)
                continue
            if (far_clip > 0.0) and (base_dist + t_hit > far_clip):
                if ti.static(dump):
                    if dmatch:
                        _aa_dump_frag(dump_out, q - 1, 0, 4, prim_raw, 0,
                                      d_face, msk, cov,
                                      _popcount_samples(msk), 1.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, t_hit, svis)
                done = True
                break
            processed += 1

            # A negative packed id tags a bezier circuit fragment; triangles
            # (and the z-winner) are non-negative.
            is_bez = False
            if ti.static(has_bez):
                is_bez = prim_raw < 0
            if is_bez:
                circuit_decoded, border_decoded = _decode_bez_ref(prim_raw)
                prim_raw = circuit_decoded
                in_border = border_decoded

            # Seam de-duplication is triangle-only (shared-edge crossings);
            # edge_hit stays 0 for a circuit, so a bezier hit never skips and
            # simply resets the seam window. Disabled under triangle analytic
            # coverage: the duplicate it drops is exactly the fragment the
            # coverage union needs, and clamped barycentrics put every partial
            # fragment on the simplex boundary, so the test would fire on all
            # of them.
            w0 = 1.0 - a - b
            if ti.static(not aa_tri):
                edge_hit = 0
                if not is_bez:
                    if ti.min(w0, ti.min(a, b)) < TRIANGLE_EDGE_EPSILON:
                        edge_hit = 1
                if (edge_hit == 1) and (t_hit - seam_t <= DEPTH_TIE_EPSILON):
                    if ti.static(dump):
                        if dmatch:
                            _aa_dump_frag(dump_out, q - 1, 0, 6, prim_raw, 0,
                                          d_face, msk, cov,
                                          _popcount_samples(msk), 1.0, 0.0,
                                          0.0, 0.0, 0.0, 0.0, t_hit, svis)
                    continue
                seam_t = t_hit if edge_hit == 1 else -1e30

            if ti.static(dump):
                if is_bez:
                    d_kind = 1
                if from_z:
                    d_kind += 2
                d_sid = -1 - prim_raw
                if not is_bez:
                    d_sid = tri_obj[f % tri_obj.shape[0], prim_raw]

            # PER-SAMPLE TRANSMITTANCE (see the docstring). ``eff`` is how much
            # of the pixel's light actually reaches this fragment: the sum, over
            # the sub-pixel samples it covers, of what is still getting through
            # each of them.
            eff = cov
            if ti.static(aa_grp):
                sliver = False
                if ti.static(not cells):
                    sliver = (msk & _AA_SLIVER_BIT) != 0
                    msk &= _AA_MASK_ALL
                # Where the fragment sits (``cmsk``) and how much of each sample
                # it covers (``dens``), which are independent -- see
                # _coverage_density. A circuit's SDF coverage, and a sliver's
                # clipped area, are fractions of the pixel with no POSITION in
                # them, so they spread over every sample instead of covering a
                # subset exactly; that is what circuits already did, and it is
                # why they need no mask of their own.
                if ti.static(cells):
                    # WITHIN a surface, coverage ADDS; BETWEEN surfaces it
                    # composites. That is the whole idea: a mesh's triangles are
                    # pieces of one shape, and a 2D rasterizer never has this
                    # problem because it rasterizes a whole closed path at once.
                    # Summing exact clipped areas over a tiling is exact (it is
                    # property 3 of _aa_clip_area_check), so an interior edge
                    # stops existing rather than having to be partitioned.
                    #
                    # A closed mesh needs its two SHEETS kept apart, though. The
                    # front and back of a sphere both cover a silhouette pixel,
                    # and they cover the SAME part of it, so their areas must not
                    # add -- the object's coverage is the larger sheet, not the
                    # sum. The facing bit separates them exactly and for free.
                    sid = -1 - prim_raw
                    fce = 0
                    if not is_bez:
                        sid = tri_obj[f % tri_obj.shape[0], prim_raw]
                        if (msk & _AA_BACKFACE_BIT) != 0:
                            fce = 1
                    if sid != cur_obj:
                        rem *= 1.0 - obj_absorb
                        obj_absorb = 0.0
                        obj_cov = 0.0
                        sheet_cov = 0.0
                        cur_obj = sid
                        cur_face = fce
                    else:
                        if fce != cur_face:
                            obj_cov = ti.max(obj_cov, sheet_cov)
                            sheet_cov = 0.0
                            cur_face = fce
                    old_o = ti.max(obj_cov, sheet_cov)
                    sheet_cov = ti.min(1.0, sheet_cov + cov)
                    new_o = ti.max(obj_cov, sheet_cov)
                    # What this fragment newly reveals of the pixel.
                    contrib = new_o - old_o
                    eff = contrib * rem
                else:
                    if ti.static(_tri_run_mode(aa_tri)):
                        # THE RUN RULE (v2 ss4.2). At the first triangle
                        # fragment past the previous run, if its mask is
                        # partial and every per-sample transmittance is equal
                        # (the "nothing is contended" predicate -- where it
                        # fails, everything below stays shipped bit-for-bit),
                        # scan the run and derive one scalar correction:
                        # corr = E / Q, E the exact-area sum, Q the sampled
                        # union's share. A full union means the surface tiles
                        # the pixel: corr is exactly 1 and interior edges
                        # cannot seam by construction. An empty union is a
                        # pristine all-sliver run (a rod between the samples):
                        # it CLAIMS min(E, 1) of the run-start transmittance,
                        # distributed by area, with uniform areal writes.
                        if (not is_bez) and (not from_z) \
                                and ((q - 1) >= run_end):
                            if ti.static(_tri_run_rule_b(aa_tri)):
                                if run_pending != 0:
                                    _run_redistribute(svis, run_U, run_resid)
                                    run_pending = 0
                                    run_resid = 0.0
                            run_mode = 0
                            run_end = q
                            if (msk & _AA_MASK_ALL) != _AA_MASK_ALL:
                                v0 = svis[0]
                                uni_v = v0 > 0.0
                                for s in ti.static(
                                        range(1, _AA_NUM_SAMPLES)):
                                    if svis[s] != v0:
                                        uni_v = False
                                if uni_v:
                                    to_row = f % tri_obj.shape[0]
                                    sid0 = tri_obj[to_row, prim_raw]
                                    face0 = 0
                                    if (frag_msk[idx]
                                            & _AA_BACKFACE_BIT) != 0:
                                        face0 = 1
                                    rE, rU, rj = _aa_run_scan(
                                        q - 1, nrun, start, sid0, face0,
                                        to_row, frag_ref, frag_cov,
                                        frag_msk, tri_obj)
                                    run_end = rj
                                    if rU == _AA_MASK_ALL:
                                        run_mode = 1
                                        run_corr = 1.0
                                    elif rU == 0:
                                        run_mode = 2
                                        run_pscale = (ti.min(rE, 1.0)
                                                      / ti.max(rE, 1e-9))
                                        run_vstart = v0
                                        run_claimed = 0.0
                                    else:
                                        run_mode = 1
                                        qq_r = (ti.cast(
                                            _popcount_samples(rU), ti.f32)
                                            * _AA_SAMPLE_WEIGHT)
                                        # Capped by the tiling bound alone:
                                        # within one sheet exact areas sum to
                                        # <= 1 over the pixel, so E above 1 is
                                        # a mis-scan (overlap double-count)
                                        # and is capped, while E/Q well above
                                        # 1 is REAL for a sub-pixel rod that
                                        # owns one sample but covers several
                                        # samples' worth of area -- the
                                        # measured case that killed the
                                        # designed [0.5, 2] clamp (thin ink
                                        # stalled at 0.88). Rule B's
                                        # redistribution keeps the occlusion
                                        # side exact under large corr.
                                        run_corr = ti.min(rE, 1.0) / qq_r
                                    if ti.static(
                                            _tri_run_rule_b(aa_tri)):
                                        if run_mode == 1:
                                            run_U = rU
                                            run_resid = 0.0
                                            run_pending = 1
                    slots, nsm, dens = _coverage_slots(
                        cov, msk, is_bez or sliver, cells,
                        ti.static(_tri_run_mode(aa_tri)))
                    vis = 0.0
                    for s in ti.static(range(_AA_NUM_SAMPLES)):
                        vis += slots[s] * svis[s]
                    eff = vis * _AA_SAMPLE_WEIGHT * dens
                    if ti.static(_tri_run_mode(aa_tri)):
                        if (not is_bez) and (not from_z) \
                                and ((q - 1) < run_end):
                            if run_mode == 1:
                                cfac = run_corr
                                eff *= run_corr
                            elif run_mode == 2:
                                # Pristine claim: exact against the run-start
                                # transmittance, occlusion written areally
                                # with sequential renormalization so the
                                # leftover lands at vstart * (1 - sum) exactly
                                # (energy conservation; ss4.5).
                                run_pd = run_pscale * cov
                                eff = run_pd * run_vstart
                                dens = run_pd / ti.max(
                                    1.0 - run_claimed, 1e-6)
                                for s in ti.static(range(_AA_NUM_SAMPLES)):
                                    slots[s] = 1.0
                                nsm = _AA_NUM_SAMPLES
                if eff <= MIN_ALPHA:
                    # Nothing still reaches the samples this fragment covers:
                    # something opaque in front of it already has them.
                    if ti.static(dump):
                        if dmatch:
                            _aa_dump_frag(dump_out, q - 1, d_kind, 1,
                                          prim_raw, d_sid, d_face, msk, cov,
                                          _popcount_samples(msk), 1.0, eff,
                                          0.0, 0.0, 0.0, 0.0, t_hit, svis)
                    continue

            color = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
            alpha = 0.0
            reflectivity = 0.0
            # Surface roughness, which under ``glossy`` spreads the reflection
            # over a GGX lobe instead of a single mirror direction.
            rough = 0.0
            ior = 0.0
            T = 0.0
            albedo3 = ti.math.vec3(0.0, 0.0, 0.0)
            prim = 0
            circuit = 0
            fetched_bez = False
            # Where this fragment actually sits on the surface -- the point that
            # shades and that its secondary rays leave from. A partially
            # covering triangle measured its depth at the centroid of the
            # samples it owns, so advancing the PIXEL-CENTRE ray to that depth
            # names a point on neither the triangle nor the surface (see
            # _tri_surface_point); rebuilding it from the fragment's own
            # barycentrics puts it back on the triangle. Circuits and the
            # non-analytic path keep the centre-ray form and stay
            # byte-identical.
            surf_pos = ro + t_hit * rd
            # ...and the ray that REACHES it. A partially covering fragment is
            # represented at its sample centroid, so the pixel-centre direction
            # no longer points from the camera to the point being shaded: view
            # vector, Fresnel, and the reflected/refracted continuation would
            # all be evaluated for a ray that does not pass through the origin
            # they start from. That is the same mismatch as the position bug
            # above, one order smaller, and it is exactly recoverable -- the
            # centroid ray is the one whose intersection produced surf_pos.
            # Fully covered fragments and circuits keep the generated direction
            # bit-for-bit: their centroid IS the pixel centre, so recomputing
            # would only add rounding noise where nothing was wrong.
            surf_rd = rd
            if ti.static(has_bez):
                if is_bez:
                    fetched_bez = True
                    circuit = prim_raw
                    cm = f % circuit_meta.shape[0]
                    # Circuits keep their sampled colour (never material-shaded).
                    color, alpha = _sample_circuit_color_blend(
                        circuit, f, a, b, in_border, circuit_meta,
                        circuit_colors, circuit_border_colors)
                    albedo3 = ti.math.vec3(color[0], color[1], color[2])
                    reflectivity = circuit_meta[cm, circuit, _M_REFLECTIVITY]
                    rough = circuit_meta[cm, circuit, _M_ROUGHNESS]
                    ior = circuit_meta[cm, circuit, _M_IOR]
                    T = circuit_meta[cm, circuit, _M_TRANSMISSION]
            if not fetched_bez:
                # Built-in triangle shading + continuation (custom scatter is
                # excluded by the raster gate); port of the drain loop's
                # htype == 1 branch.
                prim = prim_raw
                if ti.static(aa_tri):
                    surf_pos = _tri_surface_point(f, prim, w0, a, b, tri_pos)
                    partial = cov < AA_FULL_COVERAGE
                    if ti.static(_tri_run_mode(aa_tri)):
                        # Keyed on the mask: a full-mask fragment's centroid
                        # IS the pixel centre whatever its exact area, and
                        # interior fragments must stay bit-clean (v2 ss4.1).
                        partial = (msk & _AA_MASK_ALL) != _AA_MASK_ALL
                    if partial:
                        surf_rd = (surf_pos - ro).normalized()
                color, alpha = _tri_color_g(0, f, prim, w0, a, b, tri_colors,
                                            col_row, tri_uvs, tri_tex_meta,
                                            textures, num_colored_triangles)
                reflectivity, rough = _tri_extra_g(
                    0, f, prim, w0, a, b, tri_extra, col_row, tri_uvs,
                    tri_tex_meta, textures, num_colored_triangles)
                # Raw albedo, saved before fragment shading replaces ``color``.
                albedo3 = ti.math.vec3(color[0], color[1], color[2])
                if ti.static(frag_shading != 0):
                    # Exact sparse shadow-event queue: accepted triangle events
                    # carry an id beside their fragment/terminal winner.  The
                    # dedicated any-hit queue writes one visibility value per
                    # light with no fixed depth or light-count packing limit.
                    vis = ti.Vector([1.0] * MAX_SHADOW_LIGHTS)
                    if ti.static(shadows != 0):
                        event_id = -1
                        if from_z:
                            event_id = z_shadow_id[r]
                        else:
                            event_id = frag_shadow_id[idx]
                        if event_id >= 0:
                            for li in range(num_lights):
                                if li < MAX_SHADOW_LIGHTS:
                                    vis[li] = shadow_vis[event_id, li]
                    sn = ti.math.vec3(0.0, 0.0, 0.0)
                    if ti.static(skip_unlit_normal != 0):
                        if tri_mat_id[f % tri_mat_id.shape[0], prim] \
                                != _MID_UNLIT:
                            sn = _tri_normal_g(
                                0, f, prim, w0, a, b, tri_norm, tri_pos,
                                tri_uvs, tri_tex_meta, textures,
                                num_colored_triangles)
                    else:
                        sn = _tri_normal_g(
                            0, f, prim, w0, a, b, tri_norm, tri_pos, tri_uvs,
                            tri_tex_meta, textures, num_colored_triangles)
                    # Shading is evaluated ONCE per fragment. Supersampling it
                    # at grazing incidence was built and measured (the theory
                    # being that a near-edge-on pixel spans enough world
                    # distance for the light term to swing inside it) and it
                    # earned nothing on any config while costing 4x the shading
                    # -- the residual it was aimed at turned out to be the
                    # un-antialiased ray-cast fallback instead (ss19).
                    color = _shade_tri_hit(frag_pipelines, f, prim, a, b,
                                           surf_rd, surf_pos,
                                           tri_pos, sn,
                                           tri_mat_id, tri_mat,
                                           light_pos, light_col,
                                           num_lights, color, shadows, vis)
                ior = _tri_ior_g(0, f, prim, w0, a, b, tri_extra, col_row,
                                 tri_uvs, tri_tex_meta, textures,
                                 num_colored_triangles)
                T = _tri_transmission_g(0, f, prim, w0, a, b, tri_extra,
                                        col_row, tri_uvs, tri_tex_meta,
                                        textures, num_colored_triangles)

            if ti.static(aa_bez or aa_tri):
                # Coverage is a partial-occlusion factor, so it rides on alpha:
                # the four-way share split below (shade / reflect / transmit /
                # miss) then routes the uncovered fraction to the miss lane for
                # free, and every continuation weight inherits it.
                mat_alpha = ti.math.clamp(alpha, 0.0, 1.0)
                alpha = mat_alpha * eff
                if ti.static(aa_grp):
                    a_s = mat_alpha * dens
                if ti.static(dump):
                    d_mat = mat_alpha
            alpha = ti.math.clamp(alpha, 0.0, 1.0)
            T = ti.math.clamp(T, 0.0, 1.0)

            normal = ti.math.vec3(0.0, 0.0, 0.0)
            # The GEOMETRIC normal beside the shading one: what the secondary
            # rays have to leave (see shading_taichi._reflect_frame -- a shading
            # normal tipped past the silhouette aims the mirror ray into the
            # solid). A circuit is flat, so its two normals coincide.
            geo_normal = ti.math.vec3(0.0, 0.0, 0.0)
            if (reflectivity >= 0.0) or (T > 1e-4):
                if fetched_bez:
                    normal = _bezier_normal(
                        f, circuit, circuit_meta).normalized()
                    geo_normal = normal
                else:
                    normal = _tri_normal_g(
                        0, f, prim, w0, a, b, tri_norm, tri_pos, tri_uvs,
                        tri_tex_meta, textures, num_colored_triangles
                    ).normalized()
                    gp = f % tri_pos.shape[0]
                    g0 = ti.math.vec3(tri_pos[gp, prim, 0], tri_pos[gp, prim, 1],
                                      tri_pos[gp, prim, 2])
                    g1 = ti.math.vec3(tri_pos[gp, prim, 3], tri_pos[gp, prim, 4],
                                      tri_pos[gp, prim, 5])
                    g2 = ti.math.vec3(tri_pos[gp, prim, 6], tri_pos[gp, prim, 7],
                                      tri_pos[gp, prim, 8])
                    geo_normal = (g1 - g0).cross(g2 - g0)

            R, diel_pass = _material_reflectance(surf_rd, normal,
                                                 reflectivity,
                                                 ior, albedo3)
            if bounces_left <= 0:
                R = ti.math.vec3(0.0, 0.0, 0.0)

            # Solid geometry (triangle) refracts (is_glass); a zero-thickness
            # bezier pane transmits unbent and folds the transmitted share into
            # the pass-through (is_pane), same as wavefront_shade.
            is_glass = False
            is_pane = False
            if ti.static(refraction != 0):
                if (T > 1e-4) and (bounces_left > 0) and (ior > 1.0 + 1e-4):
                    if fetched_bez:
                        is_pane = True
                    else:
                        is_glass = True

            one3 = ti.math.vec3(1.0, 1.0, 1.0)
            tint = ti.math.clamp(albedo3, 0.0, 1.0)
            trans_share = diel_pass * T
            r_glow = ti.max(R[0], ti.max(R[1], R[2]))
            w_glow = ti.max(weight[0], ti.max(weight[1], weight[2]))
            share = (weight * alpha) * (one3 - R - trans_share)
            acc += ti.math.vec4(
                share[0], share[1], share[2],
                w_glow * alpha * (1.0 - r_glow - trans_share)) * color
            refl_energy = alpha * R
            refl_max = ti.max(refl_energy[0],
                              ti.max(refl_energy[1], refl_energy[2]))
            trans_energy = alpha * trans_share
            cover_pass = 1.0 - alpha
            cover3 = ti.math.vec3(cover_pass, cover_pass, cover_pass)

            split_refl = False
            if ti.static(refraction != 0):
                if (refl_max > MIN_ALPHA) and (cover_pass > MIN_ALPHA) \
                        and (bounces_left > 0):
                    split_refl = True

            # Which sub-pixel positions a secondary branch samples: the ones this
            # fragment actually covers, one ray each (see _sec_positions).
            sec_pm = 0
            sec_n = 0
            if ti.static(sec_aa > 1):
                sec_pm, sec_n = _sec_positions(msk & _AA_MASK_ALL, sec_aa)

            if is_glass:
                wt = weight * trans_energy * tint
                wt_max = ti.max(wt[0], ti.max(wt[1], wt[2]))
                if wt_max > MIN_WEIGHT:
                    hp = surf_pos
                    tp = f % tri_pos.shape[0]
                    v0 = ti.math.vec3(tri_pos[tp, prim, 0],
                                      tri_pos[tp, prim, 1],
                                      tri_pos[tp, prim, 2])
                    v1 = ti.math.vec3(tri_pos[tp, prim, 3],
                                      tri_pos[tp, prim, 4],
                                      tri_pos[tp, prim, 5])
                    v2 = ti.math.vec3(tri_pos[tp, prim, 6],
                                      tri_pos[tp, prim, 7],
                                      tri_pos[tp, prim, 8])
                    face_normal = (v1 - v0).cross(v2 - v0)
                    if ti.static(sec_aa > 1) and (wt_max > sec_min_energy) \
                            and (sec_n > 1):
                        # Supersample what is seen THROUGH the glass: the same
                        # refracted branch from each sub-pixel position this
                        # fragment covers, sharing its throughput between them.
                        # Only for a branch carrying a real share of the pixel
                        # (ANALYTIC_AA_SECONDARY_MIN_ENERGY).
                        wsub = wt * (1.0 / ti.cast(sec_n, ti.f32))
                        for s in ti.static(range(sec_aa)):
                            if (sec_pm >> s) & 1:
                                rdj, hpj, nj, _b1, _b2 = \
                                    _jittered_surface_sample(
                                        f, px, py,
                                        ti.static(_AA_SEC_JITTER[sec_aa][s][0]),
                                        ti.static(_AA_SEC_JITTER[sec_aa][s][1]),
                                        gen_meta, fetched_bez, prim, hp, normal,
                                        tri_pos, tri_norm, tri_uvs,
                                        tri_tex_meta, textures,
                                        num_colored_triangles,
                                        cam_origin, screen_point,
                                        pixel_basis_x, pixel_basis_y)
                                rdt = _refract_ray(rdj, nj, ior)
                                _spawn_pool_ray(
                                    rs_ro, rs_rd, rs_acc, rs_sca, rs_int,
                                    rs_pix, rs_alloc,
                                    _offset_transmitted_origin(
                                        hpj, rdt, face_normal, nj),
                                    rdt, wsub, base_dist + t_hit,
                                    bounces_left - 1, processed, pixel, r,
                                    compact)
                    else:
                        rdt = _refract_ray(surf_rd, normal, ior)
                        _spawn_pool_ray(
                            rs_ro, rs_rd, rs_acc, rs_sca, rs_int, rs_pix,
                            rs_alloc,
                            _offset_transmitted_origin(
                                hp, rdt, face_normal, normal),
                            rdt, wt, base_dist + t_hit,
                            bounces_left - 1, processed, pixel, r, compact)
                if (refl_max > MIN_ALPHA) and (refl_max >= cover_pass):
                    refl_rd, nref = _reflect_frame(surf_rd, normal, geo_normal)
                    hit_point = surf_pos
                    if ti.static(sec_aa > 1) and (refl_max > sec_min_energy) \
                            and (sec_n > 1):
                        # Supersample the reflected image: one ray per sub-pixel
                        # position this fragment covers. The FIRST of them
                        # continues in the pixel's own ray slot (it carries the
                        # accumulated colour) and the rest go to the shared pool,
                        # each taking an equal share of the reflected throughput
                        # -- so the radiance and the leftover background weight,
                        # which every branch commits on retirement, still sum to
                        # what the single ray carried.
                        weight *= refl_energy * (1.0 / ti.cast(sec_n, ti.f32))
                        placed = False
                        jtap = 0
                        for s in ti.static(range(sec_aa)):
                            if (sec_pm >> s) & 1:
                                rdj, hpj, nj, _b1, _b2 = \
                                    _jittered_surface_sample(
                                        f, px, py,
                                        ti.static(_AA_SEC_JITTER[sec_aa][s][0]),
                                        ti.static(_AA_SEC_JITTER[sec_aa][s][1]),
                                        gen_meta, fetched_bez, prim, hit_point,
                                        nref, tri_pos, tri_norm, tri_uvs,
                                        tri_tex_meta, textures,
                                        num_colored_triangles,
                                        cam_origin, screen_point,
                                        pixel_basis_x, pixel_basis_y)
                                rdr, nj = _reflect_frame(rdj, nj, geo_normal)
                                if ti.static(glossy != 0):
                                    if rough > _GLOSSY_MIN_ROUGHNESS:
                                        rdr = _glossy_reflect(
                                            rdj, nj, rough, jtap, sec_n,
                                            g_roff, g_aoff)
                                jtap += 1
                                org = hpj + nj * (10.0 * MIN_HIT_DISTANCE)
                                if placed:
                                    _spawn_pool_ray(
                                        rs_ro, rs_rd, rs_acc, rs_sca, rs_int,
                                        rs_pix, rs_alloc, org, rdr, weight,
                                        base_dist + t_hit, bounces_left - 1,
                                        processed, pixel, r, compact)
                                else:
                                    rd = rdr
                                    ro = org
                                    placed = True
                    else:
                        rd = refl_rd
                        ro = hit_point + nref * (10.0 * MIN_HIT_DISTANCE)
                        weight *= refl_energy
                    base_dist += t_hit
                    seam_t = -1e30
                    bounces_left -= 1
                    bounced = True
                    if ti.static(dump):
                        if dmatch:
                            _aa_dump_frag(dump_out, q - 1, d_kind, 2,
                                          prim_raw, d_sid, d_face, msk, cov,
                                          _popcount_samples(msk), 1.0, eff,
                                          d_mat, alpha, trans_share,
                                          refl_max, t_hit, svis)
                    break
                else:
                    # The pass-through outweighs the reflection, so it keeps the
                    # primary ray -- but the reflection is NOT therefore
                    # droppable. Under analytic coverage ``cover_pass`` is the
                    # share of the pixel this fragment does not cover, which the
                    # walk picks up from the fragments behind it; it is not a
                    # competing continuation the way material transparency is.
                    # A dielectric reflects ~4%, so every fragment covering less
                    # than ~96% of its pixel failed the test above and lost its
                    # Fresnel reflection entirely -- on a mesh diced finer than
                    # a pixel that is most of the surface, and it rendered as a
                    # dark lattice at the tessellation's own period across
                    # anything seen through the glass. The pool is shared and
                    # append-only, so the reflection simply takes a slot of its
                    # own, exactly as ``split_refl`` does for a semi-transparent
                    # reflector below.
                    rwt = weight * refl_energy
                    rwt_max = ti.max(rwt[0], ti.max(rwt[1], rwt[2]))
                    if rwt_max > MIN_WEIGHT:
                        refl_rd, nref = _reflect_frame(surf_rd, normal,
                                                       geo_normal)
                        rhp = surf_pos
                        if ti.static(sec_aa > 1) \
                                and (rwt_max > sec_min_energy) and (sec_n > 1):
                            rwsub = rwt * (1.0 / ti.cast(sec_n, ti.f32))
                            jtap = 0
                            for s in ti.static(range(sec_aa)):
                                if (sec_pm >> s) & 1:
                                    rdj, hpj, nj, _b1, _b2 = \
                                        _jittered_surface_sample(
                                            f, px, py,
                                            ti.static(
                                                _AA_SEC_JITTER[sec_aa][s][0]),
                                            ti.static(
                                                _AA_SEC_JITTER[sec_aa][s][1]),
                                            gen_meta, fetched_bez, prim, rhp,
                                            nref, tri_pos, tri_norm, tri_uvs,
                                            tri_tex_meta, textures,
                                            num_colored_triangles,
                                            cam_origin, screen_point,
                                            pixel_basis_x, pixel_basis_y)
                                    rdr, nj = _reflect_frame(rdj, nj,
                                                             geo_normal)
                                    if ti.static(glossy != 0):
                                        if rough > _GLOSSY_MIN_ROUGHNESS:
                                            rdr = _glossy_reflect(
                                                rdj, nj, rough, jtap, sec_n,
                                                g_roff, g_aoff)
                                    jtap += 1
                                    _spawn_pool_ray(
                                        rs_ro, rs_rd, rs_acc, rs_sca, rs_int,
                                        rs_pix, rs_alloc,
                                        hpj + nj * (10.0 * MIN_HIT_DISTANCE),
                                        rdr, rwsub, base_dist + t_hit,
                                        bounces_left - 1, processed, pixel, r,
                                        compact)
                        else:
                            _spawn_pool_ray(
                                rs_ro, rs_rd, rs_acc, rs_sca, rs_int, rs_pix,
                                rs_alloc,
                                rhp + nref * (10.0 * MIN_HIT_DISTANCE),
                                refl_rd, rwt, base_dist + t_hit,
                                bounces_left - 1, processed, pixel, r, compact)
                    if ti.static(aa_grp):
                        if ti.static(_tri_run_mode(aa_tri)):
                            rr = _run_svis_write(
                                svis, slots, a_s, 0.0, cfac,
                                ti.static(_tri_run_rule_b(aa_tri)))
                            if ti.static(_tri_run_rule_b(aa_tri)):
                                run_resid += rr
                            if run_pd > 0.0:
                                run_claimed += a_s * (1.0 - run_claimed)
                        else:
                            for s in ti.static(range(_AA_NUM_SAMPLES)):
                                svis[s] *= 1.0 - a_s * slots[s]
                    else:
                        weight *= cover_pass
            elif is_pane or split_refl:
                # Thin pane (bezier) or semi-transparent reflector: reflection
                # into a split slot, pass-through (incl. any unbent transmitted
                # share) continues in place.
                wt = weight * refl_energy
                wt_max = ti.max(wt[0], ti.max(wt[1], wt[2]))
                if wt_max > MIN_WEIGHT:
                    refl_rd, nref = _reflect_frame(surf_rd, normal, geo_normal)
                    hp = surf_pos
                    if ti.static(sec_aa > 1) and (wt_max > sec_min_energy) \
                            and (sec_n > 1):
                        wsub = wt * (1.0 / ti.cast(sec_n, ti.f32))
                        jtap = 0
                        for s in ti.static(range(sec_aa)):
                            if (sec_pm >> s) & 1:
                                rdj, hpj, nj, _b1, _b2 = \
                                    _jittered_surface_sample(
                                        f, px, py,
                                        ti.static(_AA_SEC_JITTER[sec_aa][s][0]),
                                        ti.static(_AA_SEC_JITTER[sec_aa][s][1]),
                                        gen_meta, fetched_bez, prim, hp, nref,
                                        tri_pos, tri_norm, tri_uvs,
                                        tri_tex_meta, textures,
                                        num_colored_triangles,
                                        cam_origin, screen_point,
                                        pixel_basis_x, pixel_basis_y)
                                rdr, nj = _reflect_frame(rdj, nj, geo_normal)
                                if ti.static(glossy != 0):
                                    if rough > _GLOSSY_MIN_ROUGHNESS:
                                        rdr = _glossy_reflect(
                                            rdj, nj, rough, jtap, sec_n,
                                            g_roff, g_aoff)
                                jtap += 1
                                _spawn_pool_ray(
                                    rs_ro, rs_rd, rs_acc, rs_sca, rs_int,
                                    rs_pix, rs_alloc,
                                    hpj + nj * (10.0 * MIN_HIT_DISTANCE),
                                    rdr,
                                    wsub, base_dist + t_hit, bounces_left - 1,
                                    processed, pixel, r, compact)
                    else:
                        _spawn_pool_ray(
                            rs_ro, rs_rd, rs_acc, rs_sca, rs_int, rs_pix,
                            rs_alloc,
                            hp + nref * (10.0 * MIN_HIT_DISTANCE),
                            refl_rd,
                            wt, base_dist + t_hit, bounces_left - 1,
                            processed, pixel, r, compact)
                if ti.static(aa_grp):
                    # Pass-through, per covered sample. The MAGNITUDE is
                    # per-sample; the tint is not (that would need a vec3 per
                    # sample), so a chromatic transmitter applies its colour
                    # ratio to the whole pixel scaled by how much of it the
                    # fragment covers -- exact for a fully covering pane, which
                    # is what tinted glass almost always is.
                    ts_s = a_s * trans_share
                    pm = (1.0 - a_s) + ts_s
                    if ti.static(cells):
                        obj_absorb += a_s * (1.0 - trans_share) * contrib
                    elif ti.static(_tri_run_mode(aa_tri)):
                        rr = _run_svis_write(
                            svis, slots, a_s, trans_share, cfac,
                            ti.static(_tri_run_rule_b(aa_tri)))
                        if ti.static(_tri_run_rule_b(aa_tri)):
                            run_resid += rr
                        if run_pd > 0.0:
                            run_claimed += (a_s * (1.0 - run_claimed)
                                            * (1.0 - trans_share))
                    else:
                        for s in ti.static(range(_AA_NUM_SAMPLES)):
                            ak = a_s * slots[s]
                            svis[s] *= (1.0 - ak) + ak * trans_share
                    if ts_s > 1e-6:
                        frac = cfac * ti.cast(nsm, ti.f32) * _AA_SAMPLE_WEIGHT
                        if ti.static(cells):
                            fsum = 0.0
                            for s in ti.static(range(_AA_NUM_SAMPLES)):
                                fsum += slots[s]
                            frac = fsum * _AA_SAMPLE_WEIGHT
                        num = (ti.math.vec3(1.0, 1.0, 1.0) * (1.0 - a_s)
                               + ts_s * tint)
                        weight *= one3 + (num / ti.max(pm, 1e-6) - one3) * frac
                else:
                    weight *= cover3 + trans_energy * tint
            elif (refl_max > MIN_ALPHA) and (refl_max >= cover_pass):
                refl_rd, nref = _reflect_frame(surf_rd, normal, geo_normal)
                hit_point = surf_pos
                if ti.static(sec_aa > 1) and (refl_max > sec_min_energy) \
                        and (sec_n > 1):
                    # Mirror: one reflected ray per sub-pixel position this
                    # fragment covers, the first continuing in place and the rest
                    # pooled, sharing the throughput (see the glass case above
                    # for why that preserves the pixel's totals).
                    weight *= refl_energy * (1.0 / ti.cast(sec_n, ti.f32))
                    placed = False
                    jtap = 0
                    for s in ti.static(range(sec_aa)):
                        if (sec_pm >> s) & 1:
                            rdj, hpj, nj, _b1, _b2 = _jittered_surface_sample(
                                f, px, py,
                                ti.static(_AA_SEC_JITTER[sec_aa][s][0]),
                                ti.static(_AA_SEC_JITTER[sec_aa][s][1]),
                                gen_meta, fetched_bez, prim, hit_point, nref,
                                tri_pos, tri_norm, tri_uvs, tri_tex_meta,
                                textures, num_colored_triangles,
                                cam_origin, screen_point,
                                pixel_basis_x, pixel_basis_y)
                            rdr, nj = _reflect_frame(rdj, nj, geo_normal)
                            if ti.static(glossy != 0):
                                if rough > _GLOSSY_MIN_ROUGHNESS:
                                    rdr = _glossy_reflect(
                                        rdj, nj, rough, jtap, sec_n,
                                        g_roff, g_aoff)
                            jtap += 1
                            org = hpj + nj * (10.0 * MIN_HIT_DISTANCE)
                            if placed:
                                _spawn_pool_ray(
                                    rs_ro, rs_rd, rs_acc, rs_sca, rs_int,
                                    rs_pix, rs_alloc, org, rdr, weight,
                                    base_dist + t_hit, bounces_left - 1,
                                    processed, pixel, r, compact)
                            else:
                                rd = rdr
                                ro = org
                                placed = True
                else:
                    rd = refl_rd
                    ro = hit_point + nref * (10.0 * MIN_HIT_DISTANCE)
                    weight *= refl_energy
                base_dist += t_hit
                seam_t = -1e30
                bounces_left -= 1
                bounced = True
                if ti.static(dump):
                    if dmatch:
                        _aa_dump_frag(dump_out, q - 1, d_kind, 2, prim_raw,
                                      d_sid, d_face, msk, cov,
                                      _popcount_samples(msk), 1.0, eff,
                                      d_mat, alpha, trans_share, refl_max,
                                      t_hit, svis)
                break
            else:
                if ti.static(aa_grp):
                    # Pass-through, per covered sample. The MAGNITUDE is
                    # per-sample; the tint is not (that would need a vec3 per
                    # sample), so a chromatic transmitter applies its colour
                    # ratio to the whole pixel scaled by how much of it the
                    # fragment covers -- exact for a fully covering pane, which
                    # is what tinted glass almost always is.
                    ts_s = a_s * trans_share
                    pm = (1.0 - a_s) + ts_s
                    if ti.static(cells):
                        obj_absorb += a_s * (1.0 - trans_share) * contrib
                    elif ti.static(_tri_run_mode(aa_tri)):
                        rr = _run_svis_write(
                            svis, slots, a_s, trans_share, cfac,
                            ti.static(_tri_run_rule_b(aa_tri)))
                        if ti.static(_tri_run_rule_b(aa_tri)):
                            run_resid += rr
                        if run_pd > 0.0:
                            run_claimed += (a_s * (1.0 - run_claimed)
                                            * (1.0 - trans_share))
                    else:
                        for s in ti.static(range(_AA_NUM_SAMPLES)):
                            ak = a_s * slots[s]
                            svis[s] *= (1.0 - ak) + ak * trans_share
                    if ts_s > 1e-6:
                        frac = cfac * ti.cast(nsm, ti.f32) * _AA_SAMPLE_WEIGHT
                        if ti.static(cells):
                            fsum = 0.0
                            for s in ti.static(range(_AA_NUM_SAMPLES)):
                                fsum += slots[s]
                            frac = fsum * _AA_SAMPLE_WEIGHT
                        num = (ti.math.vec3(1.0, 1.0, 1.0) * (1.0 - a_s)
                               + ts_s * tint)
                        weight *= one3 + (num / ti.max(pm, 1e-6) - one3) * frac
                else:
                    weight *= cover3 + trans_energy * tint

            if ti.static(dump):
                if dmatch:
                    _aa_dump_frag(dump_out, q - 1, d_kind, 0, prim_raw,
                                  d_sid, d_face, msk, cov,
                                  _popcount_samples(msk), cfac, eff, d_mat,
                                  alpha, trans_share, refl_max, t_hit, svis)
            cur_w = weight
            if ti.static(aa_grp):
                vis_all = 0.0
                if ti.static(cells):
                    vis_all = rem * (1.0 - obj_absorb) * _AA_NUM_SAMPLES
                else:
                    for s in ti.static(range(_AA_NUM_SAMPLES)):
                        vis_all += svis[s]
                cur_w = weight * (vis_all * _AA_SAMPLE_WEIGHT)
            if ti.max(cur_w[0], ti.max(cur_w[1], cur_w[2])) < MIN_WEIGHT:
                done = True
                break

        if ti.static(_tri_run_mode(aa_tri) and _tri_run_rule_b(aa_tri)):
            # A run interrupted by a bounce or the walk's end still owes its
            # redistribution before the leftover is read.
            if run_pending != 0:
                _run_redistribute(svis, run_U, run_resid)
                run_pending = 0
        if ti.static(aa_grp):
            # Fold the per-sample transmittance into the pixel's leftover
            # throughput. NOT after a bounce: the reflected ray's weight already
            # went through ``refl_energy``, which carries the transmittance of
            # the samples that reflected, and the rest of the pixel ends here.
            if not bounced:
                vis_all = 0.0
                if ti.static(cells):
                    vis_all = rem * (1.0 - obj_absorb) * _AA_NUM_SAMPLES
                else:
                    for s in ti.static(range(_AA_NUM_SAMPLES)):
                        vis_all += svis[s]
                weight *= vis_all * _AA_SAMPLE_WEIGHT

        if processed >= MAX_SURFACES_PER_RAY:
            done = True

        if ti.static(dump):
            if dmatch:
                d_vis = 0.0
                for s in ti.static(range(_AA_NUM_SAMPLES)):
                    d_vis += svis[s]
                _aa_dump_terminal(dump_out, bounced, done, processed,
                                  d_vis * _AA_SAMPLE_WEIGHT, acc, weight,
                                  svis)

        if bounced and not done:
            if ti.static(prefill):
                # Undo the host's retired-empty pre-fill: an ACTIVE pixel's
                # leftover background weight is committed by its continuation
                # when that retires, so the base must return to zero.
                for k in ti.static(range(3)):
                    pix_accum[r, 4 + k] = 0.0
            for k in ti.static(range(3)):
                rs_ro[r, k] = ro[k]
                rs_rd[r, k] = rd[k]
            for k in ti.static(range(4)):
                rs_acc[r, k] = acc[k]
            rs_sca[r, 0] = weight[0]
            rs_sca[r, 1] = 0.0
            rs_sca[r, 2] = 1e30
            rs_sca[r, 3] = -1e30
            rs_sca[r, 4] = base_dist
            rs_sca[r, 5] = weight[1]
            rs_sca[r, 6] = weight[2]
            rs_int[r, 0] = bounces_left
            rs_int[r, 1] = processed
            rs_int[r, 2] = _ACTIVE
            rs_int[r, 3] = 0
            rs_pix[r] = pixel
            if ti.static(compact):
                rs_int[r, 4] = r
        else:
            # Retired: the straight-line list is complete, so a pixel that
            # neither bounced nor ran out of throughput has simply seen every
            # surface -- commit and show the background through the leftover.
            if (env_w > 0) and (ti.max(weight[0], ti.max(
                    weight[1], weight[2])) > 0.0):
                ec = _sample_env_map(f, rd, env_off, env_w, env_h,
                                     env_intensity, textures)
                for k in ti.static(range(3)):
                    acc[k] += weight[k] * ec[k]
                weight = ti.math.vec3(0.0, 0.0, 0.0)
            for k in ti.static(range(4)):
                ti.atomic_add(pix_accum[r, k], acc[k])
            if ti.static(prefill):
                # Cols 4-6 were pre-filled with 1.0, so the leftover weight
                # is stored rather than accumulated (this thread is the
                # pixel's only writer during iteration zero; the colour adds
                # above keep their zero base).
                for k in ti.static(range(3)):
                    pix_accum[r, 4 + k] = weight[k]
            else:
                for k in ti.static(range(3)):
                    ti.atomic_add(pix_accum[r, 4 + k], weight[k])
            rs_int[r, 2] = _DONE
