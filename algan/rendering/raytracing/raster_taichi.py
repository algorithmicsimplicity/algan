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
  key, typed primitive reference (including the circuit-border bit), two
  intersection parameters, and an analytic coverage lane.
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
    _generate_ray,
    _sample_circuit_color,
    _shade_tri_hit,
    _shadow_occluded,
)
from algan.rendering.raytracing.shading_taichi import _MID_UNLIT
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


def _sliver_mode(aa):
    """Sample-less-triangle policy carried in the ``aa`` template value.

    ``aa`` is 0 (coverage off) or ``1 + mode``; a plain int at kernel-compile
    time, so every use of this sits inside ``ti.static``.
    """
    return max(int(aa) - 1, 0)


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
def _popcount_samples(x):
    """Number of sub-pixel samples set, i.e. set bits in the low sample bits."""
    v = ti.cast(x, ti.u32) & ti.u32(_AA_MASK_ALL)
    v = v - ((v >> 1) & ti.u32(0x5555))
    v = (v & ti.u32(0x3333)) + ((v >> 2) & ti.u32(0x3333))
    v = (v + (v >> 4)) & ti.u32(0x0F0F)
    return ti.cast((v + (v >> 8)) & ti.u32(0x1F), ti.i32)


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


@ti.func
def _pack_bez_ref(circuit, in_border):
    """Negative typed fragment ref with the border bit folded into the id."""
    return -((circuit << 1) + (in_border & 1) + 1)


@ti.func
def _decode_bez_ref(ref):
    code = -ref - 1
    return code >> 1, code & 1


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
def _ss_pixel(px, py, sm, vm, cam_o, il, aa: ti.template()):
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
                    if ti.static(_sliver_mode(aa) != _AA_SLIVER_DROP):
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
                aa: ti.template()):
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
                    px, py, sm, vm, cam_o, il, aa)
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
                   aa: ti.template(), aa_min_half_width: ti.f32):
    """Exact primary camera-ray/circuit hit for one known pixel.

    Returns ``(ok, t, u, v, in_border, cov)``.  ``cov`` is 1.0 unless the
    compile-time ``aa`` template selects analytic coverage, in which case it is
    the fraction of the pixel square the circuit's drawn region covers, and the
    hit is accepted whenever that fraction is non-zero -- including pixels whose
    CENTRE is outside the circuit but whose square is not.

    Analytic coverage (see DESIGN_analytic_aa.md ss4).  ``_bezier_point_metrics``
    returns the distance to the nearest outline segment plus a crossing parity,
    so ``d = +/- sqrt(min_dist_sq)`` is a signed distance (positive inside) in
    plane units, and ``pixel_size`` converts it to pixels.  The drawn region is

        filled:    d > -max(|border_w|, min_half_width)
        unfilled:  |d| < |border_w|                       (a band)

    and its coverage is the box filter ``clamp(distance_to_boundary + 0.5, 0, 1)``
    -- exact for a straight boundary crossing the pixel, which after flattening
    every boundary is.  The band form fades a sub-pixel-wide stroke by its width
    instead of dilating it, and both forms reach exactly 1 half a pixel inside,
    which is what lets the opaque z-prepass keep culling (``raster_bez_z``).
    """
    ok = 0
    t = 0.0
    u = 0.0
    v = 0.0
    in_border = 0
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
            border_w = circuit_meta[tm, circuit, _M_BORDER_W] * pixel_size
            outline_w = 0.6 * pixel_size
            if ti.static(aa):
                outline_w = aa_min_half_width * pixel_size
            filled = circuit_meta[tm, circuit, _M_FILLED] > 0.5
            query_radius = ti.abs(border_w)
            if filled:
                query_radius = ti.max(query_radius, outline_w)
            if ti.static(aa):
                # The filter reaches half a pixel past the drawn boundary in
                # any direction, so the nearest-edge query must too -- a pixel
                # whose centre is OUTSIDE now needs a real distance, where the
                # classic path was content with "no edge within the radius".
                query_radius += _AA_FILTER_RADIUS * pixel_size
            te = f % edges_2d.shape[0]
            crossings, min_dist_sq = _bezier_point_metrics(
                circuit, te, uu, vv, query_radius,
                circuit_meta.shape[1], edges_2d, edge_accel)
            is_border = min_dist_sq < border_w * border_w
            inside = False
            if filled:
                inside = ((crossings % 2) == 1) or (
                    min_dist_sq < outline_w * outline_w)
            if ti.static(not aa):
                if inside or is_border:
                    ok = 1
                    t = th
                    u = uu
                    v = vv
                    in_border = 1 if is_border else 0
            else:
                inv_px = 1.0 / ti.max(pixel_size, 1e-30)
                # Signed distance, positive inside the outline. min_dist_sq is
                # left at 1e30 when no edge is within the query radius, which
                # correctly reads as "deep inside" / "far outside".
                d = ti.sqrt(ti.min(min_dist_sq, 1e30))
                if (crossings % 2) == 0:
                    d = -d
                outer_w = ti.abs(border_w)
                if filled:
                    outer_w = ti.max(outer_w, outline_w)
                signed = d + outer_w
                if not filled:
                    # Unfilled: the drawn band is bounded on the inside too, so
                    # a stroke thinner than a pixel fades instead of vanishing.
                    signed = ti.min(signed, ti.abs(border_w) - d)
                c = ti.math.clamp(signed * inv_px + 0.5, 0.0, 1.0)
                if c > 0.0:
                    ok = 1
                    t = th
                    u = uu
                    v = vv
                    cov = c
                    # Colour classification must widen with the region: a pixel
                    # in the half-pixel band OUTSIDE a bordered circuit is
                    # border-coloured, not fill-coloured (|d| < border_w alone
                    # would hand it the fill).
                    in_border = 1 if (is_border or (
                        (ti.abs(border_w) > 0.0) and (d <= -ti.abs(border_w))
                    )) else 0
    return ok, t, u, v, in_border, cov


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
    in_border = 0
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
                circuit_colors, edges_2d, edge_accel, aa, aa_min_half_width)
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
            ok, lp, t, _w1, _w2, cov, _msk = _pair_pixel(
                prim, f, x0, y0, bw, bh, off, j, time_start, width, height,
                tile_start, tile_pixels, half_w, half_h, use_ss, sm, vm, cam_o,
                il, cam_origin, screen_point, pixel_basis_x, pixel_basis_y, aa)
            if ok != 0:
                full = True
                if ti.static(aa):
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
                il, cam_origin, screen_point, pixel_basis_x, pixel_basis_y, aa)
            if ok != 0:
                keep = True
                if ti.static(partial_only):
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
                il, cam_origin, screen_point, pixel_basis_x, pixel_basis_y, aa)
            if ok != 0:
                keep = True
                if ti.static(partial_only):
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
                    if ti.static(aa):
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
                    _color, alpha = _sample_circuit_color(
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
    """Emit circuit records with the border flag packed into ``frag_ref``.

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
                    _color, alpha = _sample_circuit_color(
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
                    half_w, half_h):
    """Recompute exact payload for the typed visibility-buffer winner.

    Always classic (non-analytic) acceptance, for both geometries: a primitive
    only reaches the visibility buffer when its analytic coverage was FULL
    (``raster_bez_z`` / ``raster_tri_z``), and full coverage means every edge is
    at least half a pixel away, i.e. strictly inside the classic drawn region.
    So the classic predicate accepts exactly the same winners, and the winner's
    coverage is 1 by construction -- no coverage needs to be returned.
    """
    valid = 0
    is_bez = False
    prim = 0
    t = 0.0
    a = 0.0
    b = 0.0
    in_border = 0
    if zkey != ti.i64(Z_SENTINEL):
        is_bez, prim = _decode_z_layer(zkey, layer_offset_triangles)
        if is_bez:
            valid, t, a, b, in_border, _cov = _bez_pixel_hit(
                prim, f, px, py, half_w, half_h, cam_origin, screen_point,
                pixel_basis_x, pixel_basis_y, pixel_world_scale, circuit_meta,
                circuit_colors, edges_2d, edge_accel, 0, 0.0)
        else:
            use_ss, sm, vm, cam_o, il = _ss_setup(
                f, prim, ss_enabled, tri_pos, tri_screen, cam_origin, 0)
            if use_ss != 0:
                valid, t, a, b, _c, _m = _ss_pixel(
                    px, py, sm, vm, cam_o, il, 0)
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
    shadow-ray origin exactly as ``wavefront_shade``'s inline shadow block: the
    shading normal faces the viewer, the geometric normal shares its hemisphere
    (so a grazing shadow ray does not self-shadow on an adjacent uphill facet).
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
    if snrm.norm() > 1e-9:
        snrm = snrm.normalized()
    if snrm.dot(rd) > 0.0:
        snrm = -snrm
    if fnrm.norm() > 1e-9:
        fnrm = fnrm.normalized()
    if fnrm.dot(snrm) < 0.0:
        fnrm = -fnrm
    return snrm, fnrm


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
        event_count: ti.types.ndarray()):
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
        # Mirrors raster_first_shade's transport state exactly, including the
        # per-sample transmittance (see its docstring). Any divergence here
        # desynchronizes every shadow id from its fragment.
        weight = ti.math.vec3(1.0, 1.0, 1.0)
        svis = ti.Vector([1.0 for _ in range(_AA_NUM_SAMPLES)])
        seam_t = -1e30
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
            in_border = 0
            is_bez = False
            valid = 1
            # The z-winner is a fully covering hit by construction, so it
            # claims and occludes every sub-pixel sample.
            cov = 1.0
            msk = _AA_MASK_ALL
            sliver = False
            areal = False
            a_s = 0.0
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
                    gen_meta[2], gen_meta[3])
            q += 1
            if valid == 0:
                continue
            processed += 1
            w0 = 1.0 - a - b
            if ti.static(not aa_tri):
                edge_hit = 0
                if not is_bez:
                    if ti.min(w0, ti.min(a, b)) < TRIANGLE_EDGE_EPSILON:
                        edge_hit = 1
                if (edge_hit == 1) and (t_hit - seam_t <= DEPTH_TIE_EPSILON):
                    continue
                seam_t = t_hit if edge_hit == 1 else -1e30

            eff = cov
            if ti.static(aa_grp):
                sliver = (msk & _AA_SLIVER_BIT) != 0
                msk &= _AA_MASK_ALL
                areal = is_bez or sliver
                vis = 0.0
                if areal:
                    for s in ti.static(range(_AA_NUM_SAMPLES)):
                        vis += svis[s]
                    vis *= _AA_SAMPLE_WEIGHT * cov
                else:
                    for s in ti.static(range(_AA_NUM_SAMPLES)):
                        if (msk >> s) & 1:
                            vis += svis[s]
                    vis *= _AA_SAMPLE_WEIGHT
                eff = vis
                if eff <= MIN_ALPHA:
                    continue

            alpha = 0.0
            reflectivity = 0.0
            ior = 0.0
            transmission = 0.0
            albedo = ti.math.vec3(0.0, 0.0, 0.0)
            normal = ti.math.vec3(0.0, 0.0, 0.0)
            if is_bez:
                color, alpha = _sample_circuit_color(
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
                snrm, fnrm = _tri_shadow_normals(
                    f, ref, a, b, rd, tri_pos, tri_norm, tri_uvs,
                    tri_tex_meta, textures, num_colored_triangles)
                hp = ro + t_hit * rd
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
                    if (not from_z) and (not sliver):
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
                    a_s = mat_alpha
                    if areal:
                        a_s = mat_alpha * cov
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
                    break
                if ti.static(aa_grp):
                    pm = 1.0 - a_s
                    for s in ti.static(range(_AA_NUM_SAMPLES)):
                        if areal or ((msk >> s) & 1):
                            svis[s] *= pm
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
                    for s in ti.static(range(_AA_NUM_SAMPLES)):
                        if areal or ((msk >> s) & 1):
                            svis[s] *= pm
                    if ts_s > 1e-6:
                        frac = 1.0
                        if not areal:
                            frac = (ti.cast(_popcount_samples(msk), ti.f32)
                                    * _AA_SAMPLE_WEIGHT)
                        num = (ti.math.vec3(1.0, 1.0, 1.0) * (1.0 - a_s)
                               + ts_s * tint)
                        weight *= one3 + (num / ti.max(pm, 1e-6) - one3) * frac
                else:
                    weight *= cover3 + trans_energy * tint
            elif (refl_max > MIN_ALPHA) and (refl_max >= cover_pass):
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
                    for s in ti.static(range(_AA_NUM_SAMPLES)):
                        if areal or ((msk >> s) & 1):
                            svis[s] *= pm
                    if ts_s > 1e-6:
                        frac = 1.0
                        if not areal:
                            frac = (ti.cast(_popcount_samples(msk), ti.f32)
                                    * _AA_SAMPLE_WEIGHT)
                        num = (ti.math.vec3(1.0, 1.0, 1.0) * (1.0 - a_s)
                               + ts_s * tint)
                        weight *= one3 + (num / ti.max(pm, 1e-6) - one3) * frac
                else:
                    weight *= cover3 + trans_energy * tint
            cur_w = weight
            if ti.static(aa_grp):
                vis_all = 0.0
                for s in ti.static(range(_AA_NUM_SAMPLES)):
                    vis_all += svis[s]
                cur_w = weight * (vis_all * _AA_SAMPLE_WEIGHT)
            if ti.max(cur_w[0], ti.max(cur_w[1], cur_w[2])) < MIN_WEIGHT:
                break


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
        shadow_vis: ti.types.ndarray()):
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
        svis = ti.Vector([1.0 for _ in range(_AA_NUM_SAMPLES)])
        seam_t = -1e30
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
            in_border = 0
            from_z = q >= nrun
            idx = start + q
            valid = 1
            # The z-winner is a fully covering hit by construction, so it
            # claims and occludes every sub-pixel sample.
            cov = 1.0
            msk = _AA_MASK_ALL
            sliver = False
            areal = False
            a_s = 0.0
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
                        gen_meta[2], gen_meta[3])
                prim_raw = zprim
                if is_z_bez:
                    prim_raw = _pack_bez_ref(zprim, in_border)
            q += 1
            if valid == 0:
                continue
            if (far_clip > 0.0) and (base_dist + t_hit > far_clip):
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
                    continue
                seam_t = t_hit if edge_hit == 1 else -1e30

            # PER-SAMPLE TRANSMITTANCE (see the docstring). ``eff`` is how much
            # of the pixel's light actually reaches this fragment: the sum, over
            # the sub-pixel samples it covers, of what is still getting through
            # each of them.
            eff = cov
            if ti.static(aa_grp):
                sliver = (msk & _AA_SLIVER_BIT) != 0
                msk &= _AA_MASK_ALL
                # A circuit's SDF coverage -- and a sliver's clipped area -- is a
                # fraction of the pixel with no POSITION in it, so it attenuates
                # every sample uniformly instead of a subset exactly. That is
                # what circuits already did, and it is why they need no mask.
                areal = is_bez or sliver
                # Per-covered-sample opacity, filled in once the material alpha
                # is known: a masked fragment covers each of its samples fully,
                # an areal one attenuates all of them by its coverage.
                vis = 0.0
                if areal:
                    for s in ti.static(range(_AA_NUM_SAMPLES)):
                        vis += svis[s]
                    vis *= _AA_SAMPLE_WEIGHT * cov
                else:
                    for s in ti.static(range(_AA_NUM_SAMPLES)):
                        if (msk >> s) & 1:
                            vis += svis[s]
                    vis *= _AA_SAMPLE_WEIGHT
                eff = vis
                if eff <= MIN_ALPHA:
                    # Nothing still reaches the samples this fragment covers:
                    # something opaque in front of it already has them.
                    continue

            color = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
            alpha = 0.0
            reflectivity = 0.0
            ior = 0.0
            T = 0.0
            albedo3 = ti.math.vec3(0.0, 0.0, 0.0)
            prim = 0
            circuit = 0
            fetched_bez = False
            if ti.static(has_bez):
                if is_bez:
                    fetched_bez = True
                    circuit = prim_raw
                    cm = f % circuit_meta.shape[0]
                    # Circuits keep their sampled colour (never material-shaded).
                    color, alpha = _sample_circuit_color(
                        circuit, f, a, b, in_border, circuit_meta,
                        circuit_colors, circuit_border_colors)
                    albedo3 = ti.math.vec3(color[0], color[1], color[2])
                    reflectivity = circuit_meta[cm, circuit, _M_REFLECTIVITY]
                    ior = circuit_meta[cm, circuit, _M_IOR]
                    T = circuit_meta[cm, circuit, _M_TRANSMISSION]
            if not fetched_bez:
                # Built-in triangle shading + continuation (custom scatter is
                # excluded by the raster gate); port of the drain loop's
                # htype == 1 branch.
                prim = prim_raw
                color, alpha = _tri_color_g(0, f, prim, w0, a, b, tri_colors,
                                            col_row, tri_uvs, tri_tex_meta,
                                            textures, num_colored_triangles)
                reflectivity, _rough = _tri_extra_g(
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
                                           rd, t_hit, ro, tri_pos, sn,
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
                    a_s = mat_alpha
                    if areal:
                        a_s = mat_alpha * cov
            alpha = ti.math.clamp(alpha, 0.0, 1.0)
            T = ti.math.clamp(T, 0.0, 1.0)

            normal = ti.math.vec3(0.0, 0.0, 0.0)
            if (reflectivity >= 0.0) or (T > 1e-4):
                if fetched_bez:
                    normal = _bezier_normal(
                        f, circuit, circuit_meta).normalized()
                else:
                    normal = _tri_normal_g(
                        0, f, prim, w0, a, b, tri_norm, tri_pos, tri_uvs,
                        tri_tex_meta, textures, num_colored_triangles
                    ).normalized()

            R, diel_pass = _material_reflectance(rd, normal, reflectivity,
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
                    hp = ro + t_hit * rd
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
                        rdt = _refract_ray(rd, normal, ior)
                        _spawn_pool_ray(
                            rs_ro, rs_rd, rs_acc, rs_sca, rs_int, rs_pix,
                            rs_alloc,
                            _offset_transmitted_origin(
                                hp, rdt, face_normal, normal),
                            rdt, wt, base_dist + t_hit,
                            bounces_left - 1, processed, pixel, r, compact)
                if (refl_max > MIN_ALPHA) and (refl_max >= cover_pass):
                    nref = normal
                    if nref.dot(rd) > 0.0:
                        nref = -nref
                    hit_point = ro + t_hit * rd
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
                                if nj.dot(rdj) > 0.0:
                                    nj = -nj
                                rdr = (rdj - 2.0 * rdj.dot(nj)
                                       * nj).normalized()
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
                        rd = (rd - 2.0 * rd.dot(nref) * nref).normalized()
                        ro = hit_point + nref * (10.0 * MIN_HIT_DISTANCE)
                        weight *= refl_energy
                    base_dist += t_hit
                    seam_t = -1e30
                    bounces_left -= 1
                    bounced = True
                    break
                else:
                    if ti.static(aa_grp):
                        pm = 1.0 - a_s
                        for s in ti.static(range(_AA_NUM_SAMPLES)):
                            if areal or ((msk >> s) & 1):
                                svis[s] *= pm
                    else:
                        weight *= cover_pass
            elif is_pane or split_refl:
                # Thin pane (bezier) or semi-transparent reflector: reflection
                # into a split slot, pass-through (incl. any unbent transmitted
                # share) continues in place.
                wt = weight * refl_energy
                wt_max = ti.max(wt[0], ti.max(wt[1], wt[2]))
                if wt_max > MIN_WEIGHT:
                    nref = normal
                    if nref.dot(rd) > 0.0:
                        nref = -nref
                    hp = ro + t_hit * rd
                    if ti.static(sec_aa > 1) and (wt_max > sec_min_energy) \
                            and (sec_n > 1):
                        wsub = wt * (1.0 / ti.cast(sec_n, ti.f32))
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
                                if nj.dot(rdj) > 0.0:
                                    nj = -nj
                                _spawn_pool_ray(
                                    rs_ro, rs_rd, rs_acc, rs_sca, rs_int,
                                    rs_pix, rs_alloc,
                                    hpj + nj * (10.0 * MIN_HIT_DISTANCE),
                                    (rdj - 2.0 * rdj.dot(nj)
                                     * nj).normalized(),
                                    wsub, base_dist + t_hit, bounces_left - 1,
                                    processed, pixel, r, compact)
                    else:
                        _spawn_pool_ray(
                            rs_ro, rs_rd, rs_acc, rs_sca, rs_int, rs_pix,
                            rs_alloc,
                            hp + nref * (10.0 * MIN_HIT_DISTANCE),
                            (rd - 2.0 * rd.dot(nref) * nref).normalized(),
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
                    for s in ti.static(range(_AA_NUM_SAMPLES)):
                        if areal or ((msk >> s) & 1):
                            svis[s] *= pm
                    if ts_s > 1e-6:
                        frac = 1.0
                        if not areal:
                            frac = (ti.cast(_popcount_samples(msk), ti.f32)
                                    * _AA_SAMPLE_WEIGHT)
                        num = (ti.math.vec3(1.0, 1.0, 1.0) * (1.0 - a_s)
                               + ts_s * tint)
                        weight *= one3 + (num / ti.max(pm, 1e-6) - one3) * frac
                else:
                    weight *= cover3 + trans_energy * tint
            elif (refl_max > MIN_ALPHA) and (refl_max >= cover_pass):
                nref = normal
                if nref.dot(rd) > 0.0:
                    nref = -nref
                hit_point = ro + t_hit * rd
                if ti.static(sec_aa > 1) and (refl_max > sec_min_energy) \
                        and (sec_n > 1):
                    # Mirror: one reflected ray per sub-pixel position this
                    # fragment covers, the first continuing in place and the rest
                    # pooled, sharing the throughput (see the glass case above
                    # for why that preserves the pixel's totals).
                    weight *= refl_energy * (1.0 / ti.cast(sec_n, ti.f32))
                    placed = False
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
                            if nj.dot(rdj) > 0.0:
                                nj = -nj
                            rdr = (rdj - 2.0 * rdj.dot(nj) * nj).normalized()
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
                    rd = (rd - 2.0 * rd.dot(nref) * nref).normalized()
                    ro = hit_point + nref * (10.0 * MIN_HIT_DISTANCE)
                    weight *= refl_energy
                base_dist += t_hit
                seam_t = -1e30
                bounces_left -= 1
                bounced = True
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
                    for s in ti.static(range(_AA_NUM_SAMPLES)):
                        if areal or ((msk >> s) & 1):
                            svis[s] *= pm
                    if ts_s > 1e-6:
                        frac = 1.0
                        if not areal:
                            frac = (ti.cast(_popcount_samples(msk), ti.f32)
                                    * _AA_SAMPLE_WEIGHT)
                        num = (ti.math.vec3(1.0, 1.0, 1.0) * (1.0 - a_s)
                               + ts_s * tint)
                        weight *= one3 + (num / ti.max(pm, 1e-6) - one3) * frac
                else:
                    weight *= cover3 + trans_energy * tint

            cur_w = weight
            if ti.static(aa_grp):
                vis_all = 0.0
                for s in ti.static(range(_AA_NUM_SAMPLES)):
                    vis_all += svis[s]
                cur_w = weight * (vis_all * _AA_SAMPLE_WEIGHT)
            if ti.max(cur_w[0], ti.max(cur_w[1], cur_w[2])) < MIN_WEIGHT:
                done = True
                break

        if ti.static(aa_grp):
            # Fold the per-sample transmittance into the pixel's leftover
            # throughput. NOT after a bounce: the reflected ray's weight already
            # went through ``refl_energy``, which carries the transmittance of
            # the samples that reflected, and the rest of the pixel ends here.
            if not bounced:
                vis_all = 0.0
                for s in ti.static(range(_AA_NUM_SAMPLES)):
                    vis_all += svis[s]
                weight *= vis_all * _AA_SAMPLE_WEIGHT

        if processed >= MAX_SURFACES_PER_RAY:
            done = True

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
