"""Hybrid raster front-end kernels for deterministic primary visibility.

The frontend replaces the first classic wavefront iteration for flat triangles
and Bezier circuits.  Primitive/chunk pairs enumerate candidate pixels, exact
intersection tests reject bbox misses, and compact fragment records are
ordered by the same transitive ``(depth bin, descending layer)`` relation as
the classic tracer.  The host then compacts them into per-pixel sheets
(``sheets.compact_sheets``) and one kernel resolves, builds shadow events for,
and shades those (``sheet_resolve_taichi.sheet_resolve_shade``).

Important implementation properties:

* Triangle projection data is precomputed once per frame/primitive by
  :func:`raster_pipeline.precompute_triangle_projection`.  Straddling cases use
  exact per-pixel ray casting.
* Triangle/circuit COUNT and WRITE kernels fetch sampled alpha and discard
  alpha-zero texels before sorting.  Records contain only exact-distance
  key, typed primitive reference (including the circuit border/fill blend
  weight), two intersection parameters, and an analytic coverage lane.
* Analytic anti-aliasing (``ALGAN_ANALYTIC_AA``, see DESIGN_analytic_aa.md):
  each circuit fragment carries the fraction of the pixel square its drawn
  region covers -- a box filter of the outline signed-distance field that
  ``_bezier_point_metrics`` already computes -- and the resolve folds that into
  the fragment's alpha, so circuit silhouettes resolve continuously at
  ``anti_alias_level = 1`` instead of all-or-nothing.  The coverage lane is
  host-pre-filled to 1.0 and written by the circuit kernels and -- under
  ``ANALYTIC_AA_TRI`` -- by ``raster_tri_write`` too, which carries each
  triangle fragment's exact clipped area for the sheet claims.
* :func:`raster_shadow_trace` traces the sheet resolve's sparse any-hit event
  queue and stores one visibility value per event/light, with no fixed
  fragment-slot or packed-light limit. Point/spot emitter radii and
  directional angular radii use the same deterministic golden-angle fan as
  the classic wavefront path.

Emission covers the whole prepared frame window at once; each pair covers up
to ``RASTER_CHUNK`` pixels.  Future work should benchmark square block bins
and candidate-parallel block kernels. PN patches, custom scatter, near
clipping, and in-place supersampling still route to the classic frontend
without changing geometry construction.
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
    _axis_cos,
    _bezier_point_metrics,
    _circuit_point_region,
    _circuit_query_radius,
    _generate_ray,
    _sample_circuit_color_blend,
    _shade_tri_hit,
    _shadow_occluded,
    _tri_hit,
)
from algan.rendering.raytracing.shading_taichi import (
    _MID_DEFAULT,
    _MID_UNLIT,
    _USER_PIPELINE_BASE,
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
    _light_zero_radiance,
    _material_reflectance,
    _offset_transmitted_origin,
    _refract_ray,
    _reserve_continuation_slot,
    _sample_env_map,
    _tri_color_g,
    _tri_extra_g,
    _tri_ior_transmission_g,
    _tri_material_g,
    _tri_normal_g,
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

# Marks every fragment of a pixel whose fragments are ALL opaque triangles of
# ONE surface (set by prepare_sparse_raster_coverage, which has the CSR to do it
# as a segment reduction). There the mesh's coverage is its near sheet's exact
# area and nothing else -- both sheets project to the same silhouette -- so the
# far sheet must not add coverage on top of it (the sheet resolve's
# per-pixel ceiling clamp reads it via the sheet records).
_AA_ONE_MESH_BIT = 4 << _AA_FLAG_SHIFT

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














def _aa_run_full(aa_grp):
    """Whether the relaxed gate admits a FULL-mask fragment that covers less
    than the whole pixel (DESIGN_mesh_identity.md ss6.3.2).

    A diced mesh's silhouette produces full-mask fragments: one triangle
    owning all eight sub-pixel samples while covering a fraction of the
    pixel's area. Without the gate those are painted at 1.0 with their exact
    area unread, and on a fine Sphere they are 52% of the silhouette pixels.
    An interior fragment's ``cov`` is within dust of 1, so admitting
    ``cov < 1 - dust`` picks up exactly the silhouette and leaves the
    interior alone. Its one surviving reader is the emission truncation in
    ``prepare_sparse_raster_coverage``, which must keep a sheet's area donors
    in the stream whenever the relaxed semantics will read them.

    Carried as ``aa_grp == 2``; 3 is the one-mesh rule, which implies this
    gate.
    """
    return int(aa_grp) >= 2


#: How far below 1 a full-mask fragment's exact area must sit before the run
#: scan treats it as a silhouette rather than an interior tiling. Float dust in
#: the exact-area arithmetic is far below this; a silhouette pixel's shortfall
#: is orders of magnitude above it. Also the band inside which a full-union run
#: keeps ``corr = 1`` exactly, so a genuine interior tiling stays bit-identical.
_AA_FULL_DUST = 1e-3


def _tri_repr(aa):
    """Triangle representation carried in the geometry kernels' ``aa`` value:
    0 sampled points, 1 retired (the deleted cells emission; the value is not
    reissued), 2 run-corrected exact areas (``1 + sliver_mode + 4 * repr``,
    see :func:`_sliver_mode`)."""
    return max(int(aa) - 1, 0) // 4


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
                    # Snap the projected vertices to a 1/4096-pixel integer lattice
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
                    # exact lane (clipping per CANDIDATE here cost count +36%
                    # / write +42% device on the meshes A/B).
                    # Sample-less DONORS are the exception: their acceptance
                    # IS "exact area > 0", so both kernels clip them, behind
                    # the one-sided oriented reject.
                    # the mask, fill rule and owned-sample centroid stay
                    # exactly as shipped -- ownership and occlusion never
                    # change -- while frag_cov carries the EXACT clipped area
                    # for the sheet compaction to sum.
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
                            # sheet claims, cheapest-first: a microtriangle FULLY
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
                                # BOTH count and write take the centroid form:
                                # a donor's barycentrics move to the clipped
                                # centroid, and the keep decision samples the
                                # texture ALPHA at those barycentrics -- a
                                # count-only moment-free clip left the count
                                # kernel sampling at the pixel centre instead,
                                # the two kernels' keep decisions diverged on
                                # textured prims, and the write pass left
                                # UNINITIALIZED fragment rows (the
                                # text_and_media CUDA_ERROR_ILLEGAL_ADDRESS:
                                # a float bit-pattern walked as a prim id).
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
                    if ti.static(_tri_run(aa)):
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
                # THE SAME INTERSECTION THE RAY PATH USES (ss3.3). This is a
                # set-membership question -- is this sub-pixel sample inside
                # this triangle -- and it used to be answered by an inline
                # Moller-Trumbore dilated by BARYCENTRIC_EPSILON. The dilation
                # was not a fudge: with a float test and no exact tie-break, a
                # sample lying on an edge shared by two straddling triangles
                # must be erred one way, and double-claiming is harmless here
                # (per-sample transmittance gives the sample to the nearer
                # fragment) while dropping it is a crack.
                #
                # _tri_hit answers it without needing to choose. Under
                # WATERTIGHT_TRI the shared edge's function is computed from the
                # same two projected vertices in both triangles and comes out as
                # the exact negative, so exactly one neighbour accepts: no
                # dilation, no duplicate, no crack. With the gate off this is
                # bit-identical to the code it replaces -- the same dilated
                # Moller-Trumbore with the same epsilon and the same three
                # comparisons.
                #
                # The projected path (_ss_pixel) reaches the same place by a
                # different route, its exact fixed-point fill rule; this one
                # cannot use that, because the projection it would need is
                # precisely what straddling the camera plane invalidates.
                hit_s, _c1, _c2, ts = _tri_hit(ros, rds, v0, v1, v2)
                if hit_s and (ts > MIN_HIT_DISTANCE):
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
    which is what lets the emission's opaque truncation keep culling.

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
            # th is the slant range; pixel_world_scale wants perpendicular
            # depth (see _axis_cos).
            pixel_size = (pixel_world_scale[f] * th
                          * _axis_cos(f, ro, rd, screen_point))
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
        pair_count: ti.types.ndarray(),
        pair_accept: ti.types.ndarray()):
    """Count surviving nonzero-alpha transparent triangle fragments.

    ``partial_only`` (analytic coverage only): emit just the partially covered
    fragments, so the host can run this pass over the proven-opaque candidates
    as well -- their fully covered pixels already sit in the z-prepass and only
    their silhouette pixels need to blend.

    ``pair_accept[p]`` records the per-pixel acceptance decisions as a bitmask
    (bit ``j`` = chunk pixel ``j`` survived; ``RASTER_CHUNK`` is 32, one i32).
    The write pass replays these bits instead of recomputing the whole
    acceptance chain -- most importantly the texture-sampling alpha fetch --
    so this kernel IS the acceptance authority the write contract points at.
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
        bits = 0
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
                        bits |= 1 << j
        pair_count[p] = cnt
        pair_accept[p] = bits


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
        frag_cov: ti.types.ndarray(), frag_msk: ti.types.ndarray(),
        pair_accept: ti.types.ndarray()):
    """Emit exact-distance triangle records; alpha-zero texels are discarded.

    Acceptance replays ``pair_accept`` -- the per-pixel decision bits the
    count pass recorded -- rather than recomputing the acceptance chain, so
    the two passes cannot diverge and this pass never touches the colour /
    texture arrays at all. Emission order per pair is ascending chunk pixel,
    exactly the order the recompute produced. The record's geometry lanes
    (lp/t/w1/w2/cov/msk) still come from the exact ``_pair_pixel`` evaluation
    (mode 1: the cov lane carries the exact area for masked fragments; the
    count's acceptance used the sampled share, which the bits already bake
    in).
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
        w = pair_offset[p]
        bits = pair_accept[p]
        for j in range(RASTER_CHUNK):
            if ((bits >> j) & 1) != 0:
                ok, lp, t, w1, w2, cov, msk = _pair_pixel(
                    prim, f, x0, y0, bw, bh, off, j, time_start, width,
                    height, tile_start, tile_pixels, half_w, half_h, use_ss,
                    sm, vm, cam_o, il, cam_origin, screen_point,
                    pixel_basis_x, pixel_basis_y, aa, 1)
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
        pair_count: ti.types.ndarray(),
        pair_accept: ti.types.ndarray()):
    """Count surviving nonzero-alpha translucent circuit fragments.

    ``partial_only`` (compile-time, analytic coverage only): emit ONLY the
    partially covered fragments.  The host uses it to run this pass over the
    proven-opaque candidate pairs as well -- their fully covered pixels are
    already in the z-prepass, and their silhouette pixels are exactly the ones
    that need to blend.

    ``pair_accept[p]`` records the per-pixel acceptance bits the write pass
    replays (see :func:`raster_tri_count`).
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
        bits = 0
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
                        bits |= 1 << j
        pair_count[p] = cnt
        pair_accept[p] = bits


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
        frag_cov: ti.types.ndarray(),
        pair_accept: ti.types.ndarray()):
    """Emit circuit records with the border weight packed into ``frag_ref``.

    Acceptance replays the count pass's ``pair_accept`` bits (see
    :func:`raster_tri_write`) -- the passes cannot diverge, and this one
    skips the colour sampling entirely; the exact ``_bez_pair_pixel``
    evaluation still supplies the record's geometry lanes.  ``frag_cov`` is
    the analytic coverage lane, pre-filled with 1.0 by the host so geometry
    without analytic coverage (flat triangles today) needs no write of its
    own.
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
        bits = pair_accept[p]
        for j in range(RASTER_CHUNK):
            if ((bits >> j) & 1) != 0:
                ok, lp, t, u, v, ib, cov = _bez_pair_pixel(
                    circuit, f, x0, y0, bw, bh, off, j, time_start, width,
                    height, tile_start, tile_pixels, half_w, half_h,
                    cam_origin, screen_point, pixel_basis_x, pixel_basis_y,
                    pixel_world_scale, circuit_meta, circuit_colors,
                    edges_2d, edge_accel, aa, aa_min_half_width)
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
        b_nodes: NODE_ARG, b_node_miss: ti.types.ndarray(),
        b_leaf_prim: ti.types.ndarray(), b_leaf_tspan: ti.types.ndarray(),
        b_first_leaf: int,
        circuit_meta: ti.types.ndarray(), circuit_colors: ti.types.ndarray(),
        circuit_border_colors: ti.types.ndarray(),
        edges_2d: ti.types.ndarray(), edge_accel: ti.types.ndarray(),
        light_pos: ti.types.ndarray(), light_col: ti.types.ndarray(),
        num_lights: int, pixel_world_scale: ti.types.ndarray(),
        layer_offset_triangles: ti.f32,
        refit: ti.template(),
        has_tri: ti.template(), has_bez: ti.template(),
        event_dp: ti.types.ndarray(), sec_aa: ti.template(),
        shadow_vis: ti.types.ndarray(), shadow_anyhit: ti.template()):
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
    # One thread per (event, light) cell: every cell's fan is independent
    # (its result lands in its own ``shadow_vis[e, li]`` and all per-light
    # state initializes inside the body), so flattening the light loop into
    # the launch grid multiplies parallelism by ``num_lights`` while keeping
    # each cell's arithmetic -- including the serial golden-angle sample loop,
    # whose float accumulation order is part of the output contract --
    # bit-for-bit identical. The per-event setup is recomputed per cell
    # (pure loads, negligible next to a single occlusion march).
    for idx in range(num_events * num_lights):
        e = idx // num_lights
        li = idx - e * num_lights
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
        # The event's material pipeline id (packed above the 4-bit sub-pixel
        # position mask at build time). In every built-in stage a zero-colour
        # light row (not yet spawned, or despawned) contributes nothing
        # whatever its visibility -- the lit stages' terms all carry the light
        # colour as a factor, and the default stage skips zero-colour rows
        # outright -- so such rows keep their all-lit default without tracing.
        # Only user pipelines, which may read visibility arbitrarily, keep the
        # exact fan for every light.
        pid_e = event_msk[e] >> 8
        fan_exact = 1
        fan_geom = 0
        if pid_e < _USER_PIPELINE_BASE:
            fan_exact = 0
            # Geometric zero-radiance culling is only valid for stages whose
            # vis terms all carry lc (see _light_zero_radiance); the default
            # stage's base fade is not one of them.
            if pid_e != _MID_DEFAULT:
                fan_geom = 1
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
        # A light past its range, a fragment outside a spot cone, an
        # area sample's backface: exactly zero radiance here, so the
        # fan's result multiplies zero. Skipping leaves the event's
        # all-lit default, exactly like the zero-colour skip below.
        if (valid == 1) and (fan_geom == 1):
            if _light_zero_radiance(light_col, tl, li, ltype, to_light,
                                    ldist) == 1:
                valid = 0
        if (valid == 1) and ((fan_exact == 1)
                             or (light_col[tl, li, 0] != 0.0)
                             or (light_col[tl, li, 1] != 0.0)
                             or (light_col[tl, li, 2] != 0.0)):
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
                        refit, shadow_anyhit, sorg, wis, f, ff,
                        ldn - 20.0 * MIN_HIT_DISTANCE,
                        pixel_world_scale[
                            f % pixel_world_scale.shape[0]], 0.0,
                        layer_offset_triangles,
                        has_tri, has_bez,
                        t_nodes, t_node_miss, t_leaf_prim, t_leaf_tspan,
                        t_first_leaf, tri_pos, tri_colors, tri_uvs,
                        tri_tex_meta, textures, num_colored_triangles,
                        b_nodes, b_node_miss, b_leaf_prim, b_leaf_tspan,
                        b_first_leaf, circuit_meta, circuit_colors,
                        circuit_border_colors, edges_2d, edge_accel)
            if n_valid > 0.0:
                visibility = 1.0 - occ_sum / n_valid
        shadow_vis[e, li] = visibility


