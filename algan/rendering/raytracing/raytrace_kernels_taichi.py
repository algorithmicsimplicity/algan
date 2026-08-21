"""Shared Taichi ray-tracing library + the Monte Carlo path-tracing kernels.

This module holds the ``@ti.func`` building blocks every renderer uses --
sibling-block STBVH traversal, triangle / bezier-circuit
intersection and colour/material sampling, batched hit gathering
(``_collect_hits``), shadow occlusion and tonemapping -- plus the Monte Carlo
megakernels used when ``samples_per_pixel > 1``: ``path_trace_scene_stbvh``
and the physical-mode ``path_trace_physical_stbvh``, each launching one
thread per (frame, pixel, sample) path and accumulating atomically into a
float buffer that ``finalize_samples`` averages. The deterministic
(``samples_per_pixel == 1``) renderer lives in ``wavefront_kernels_taichi``
and imports these helpers.

Hits along a ray are processed strictly front-to-back by *batched depth
peeling*: each BVH traversal gathers the ``KBUF`` nearest hits beyond the
previous ones into registers (``_collect_hits``), and the batch is then
consumed in order, alpha-compositing each surface in place
(``acc += weight * a * color; weight *= 1 - a``) until the remaining
transmittance is negligible or the ray escapes to the (pre-filled)
background. A batch that comes back not full ends the ray without a
confirming traversal, and surfaces flagged fully opaque in the BVH
(``leaf_tspan`` bit 31) stop the gather at their depth, so simple pixels
still cost a single traversal. Coplanar surfaces -- ubiquitous in 2D scenes
-- are ordered deterministically by a per-primitive layer index (higher
layer on top; triangles layer above bezier circuits).

When a surface has reflectivity ``r > 0`` the ray is mirror-reflected about
the (interpolated) surface normal and marching continues with throughput
``weight * a * r``, up to ``max_bounces`` reflections.

Traversal data is laid out for one-cache-line node visits (see ``stbvh.py``):
``nodes [num_nodes, 8]`` packs bounds + frame interval per node, leaves hold
``LEAF_SIZE`` primitive slots (``leaf_prim`` plus a packed per-slot frame
interval ``leaf_tspan`` so out-of-frame instances are skipped exactly).

Geometry comes in three packed forms, each fetched at the ray's exact frame
(frame index modulo each array's own time length, so constant data can stay
single-frame). Hot data (what every candidate intersection touches) is kept
separate from cold data (what only confirmed hits touch):

* triangles: positions ``tri_pos [Tp, N, 9]`` (hot); shading normals
  ``tri_norm [Tn, N, 9]`` (cold: fetched only for mirror bounces or Monte
  Carlo scattering), ``tri_extra [Te, N, 9]`` (per-corner reflectivity +
  roughness pairs, then per-corner IOR, then per-corner transmission;
  usually single-frame) and
  ``tri_colors [Tc, N, 3, 5]``
  (RGB, glow, alpha per corner);
* planar bezier circuits: ``circuit_meta [Tm, C, 24]`` (plane frame, border
  width, fill flag, texture grid transform and four surface-transport
  channels), 2D polyline ``edges_2d`` with packed scanline/spatial tables
  ``edge_accel``, fill/texture colors ``circuit_colors [Tf, C, P, 5]``
  and border/texture colors ``circuit_border_colors [Tb, C, P, 5]``
  (both bilinearly sampled; P = 1 for plain colors).

Coplanar-surface layer order is bezier circuits < triangles, with each type's
primitive index breaking ties within the type.
"""

import taichi as ti

from algan.environment import env_flag, env_int
from algan.rendering.raytracing.bezier_acceleration import (
    BEZIER_ACCEL_HEADER_SIZE,
    BEZIER_GRID_INV_U,
    BEZIER_GRID_INV_V,
    BEZIER_MAX_U,
    BEZIER_MAX_V,
    BEZIER_MIN_U,
    BEZIER_MIN_V,
    BEZIER_SCAN_BINS,
    BEZIER_SCAN_INV_V,
    BEZIER_SCAN_OFFSET_BASE,
    BEZIER_SPATIAL_GRID,
    BEZIER_SPATIAL_OFFSET_BASE,
)
from algan.rendering.raytracing.shading_taichi import (
    # Re-exported: wavefront_kernels_taichi imports MAX_SHADOW_LIGHTS from
    # here rather than from shading_taichi, so this hop is load-bearing even
    # though nothing in this module reads the name.
    MAX_SHADOW_LIGHTS,  # noqa: F401
    _run_frag_pipeline,
)
from algan.rendering.raytracing.stbvh import BLOCK_F16, BVH_ARITY, LEAF_SIZE
from algan.rendering.taichi_runtime import init_taichi

init_taichi()

# Sibling-block traversal stack. The walk descends into one intersected
# child at a time and pushes the sibling group's *remaining* mask; a complete
# BVH_ARITY-ary tree over P leaves is log_ARITY(P) levels deep with at most
# one push per level, so 16 covers 4^16 leaves (the largest practical build
# is ~4^12). Entries pack ``node << BVH_ARITY | mask``.
_GROUP_STACK = 16
_GROUP_MASK = (1 << BVH_ARITY) - 1

# Minimum hit distance along a ray (also the self-intersection guard for
# reflected rays, together with a normal offset at the bounce origin).
MIN_HIT_DISTANCE = 1e-4
# Hits closer together than this along a ray are considered coplanar and are
# ordered by layer index instead of by distance.
DEPTH_TIE_EPSILON = 1e-4
# Reciprocal, used to bin distances into coplanarity buckets in _comes_after.
INV_DEPTH_TIE_EPSILON = 1.0 / DEPTH_TIE_EPSILON
# Surfaces more transparent than this neither reflect nor terminate peeling.
MIN_ALPHA = 1e-3
# Marching stops once the remaining transmittance drops below this.
MIN_WEIGHT = 1e-3
# Hard cap on blended surfaces per ray, to bound worst-case stacked geometry.
MAX_SURFACES_PER_RAY = 256
# Tolerance of the point-in-triangle test, in barycentric units. Adjacent
# triangles sharing an edge (e.g. the diagonals of a triangulated image
# grid) must overlap slightly rather than exclude each other: with exact
# tests, floating-point noise can make a ray on the shared edge miss *both*
# triangles, leaving crack pixels. The overlap is ~1e-4 of a triangle's
# size (sub-pixel), and the duplicate hit on the seam is discarded by the
# edge-merging rule below.
BARYCENTRIC_EPSILON = 1e-4
# A triangle hit whose smallest barycentric coordinate is below this counts
# as an *edge hit*. When two consecutive edge hits land within
# DEPTH_TIE_EPSILON of each other along a ray, they are the two triangles
# adjacent to a shared mesh edge reporting the same surface point: the
# second is discarded so the mesh behaves as one cohesive surface (in
# particular, a partially transparent mesh must not blend twice on seams).
TRIANGLE_EDGE_EPSILON = 2e-4

# WATERTIGHT ray/triangle intersection (Woop-Benthin-Wald), gated off.
# DESIGN_mesh_identity.md ss3.2.
#
# The two epsilons above are a matched pair and neither means anything alone:
# BARYCENTRIC_EPSILON dilates every triangle so a ray on a shared edge cannot
# miss BOTH neighbours and leave a crack, and TRIANGLE_EDGE_EPSILON removes the
# duplicate hit that the dilation then manufactures.
#
# WBW removes the need for both. It transforms the ray into a space where it is
# the +z axis, so a shared edge's edge function is computed from the SAME two
# projected vertices in both triangles and comes out as the exact negative --
# exactly one neighbour accepts, with no dilation and no duplicate to discard.
# It is the same argument the raster path's exact fixed-point rule already makes
# (``_ss_pixel`` in raster_taichi.py).
#
# Import-time, not a live setting: it changes the compiled kernel body, so a
# runtime toggle would silently reuse a cached kernel (the _AA_SAMPLES
# cache-trap rule). Clear the Taichi cache when flipping it.
#
# DEFAULT ON. Correctness was qualified on CUDA (ss3.2/ss4.7): zero enclosed
# background pixels leaked on two scenes built to provoke a crack, no double
# blend introduced, byte-identical on opaque geometry, and 2 pixels of 419904
# differing on a translucent scene -- where both arms are already correct and
# what differs is which neighbour owns the seam pixel.
#
# The cost was left unresolved for a while and the honest reading of the numbers
# is that there is nothing to resolve. The measured deltas were +8.5% to +10.7%
# on the kernels this flag can reach -- but ``raster_tri_count``, which it
# CANNOT reach (the rasterizer has no _tri_hit), moved +8.6% in the same runs.
# A control that moves with the target is ss7.15's definition of thermal drift
# rather than a cost, so the flag is below this machine's noise floor and no
# amount of re-running it here will say otherwise.
#
# Against that, the dilation is a known, if small, correctness defect: every
# triangle is tested slightly WIDER than it is, so a ray that should miss can
# hit. Trading an unmeasurable cost against a real defect is the trade taken
# here.
WATERTIGHT_TRI = env_flag("ALGAN_WATERTIGHT_TRI", True)


@ti.func
def _edge_is_canonical(px, py, qx, qy) -> bool:
    """Deterministic owner of an edge a ray passes through EXACTLY.

    A zero edge function means the ray is on the edge, and there the sign test
    alone accepts in both neighbours -- exact negation makes zero zero either
    way. Consistently wound neighbours traverse a shared edge in opposite
    directions, so any strict total order on the projected endpoints picks
    exactly one of them. This is the analogue of the raster path's top-left
    fill rule, and it is why that rule PARTITIONS its samples.
    """
    return (py < qy) or ((py == qy) and (px < qx))


@ti.func
def _tri_hit(ro, rd, v0, v1, v2):
    """Ray/triangle intersection: ``(ok, w1, w2, t)``.

    ``w1``/``w2`` are the barycentric weights of ``v1``/``v2`` (so
    ``w0 = 1 - w1 - w2``) and ``t`` the ray parameter, matching what the three
    call sites used to compute inline. Under ``WATERTIGHT_TRI`` this is
    Woop-Benthin-Wald; otherwise it is the shipped dilated Moller-Trumbore,
    unchanged and bit-for-bit.
    """
    ok = 0
    w1 = 0.0
    w2 = 0.0
    t = 0.0
    if ti.static(WATERTIGHT_TRI):
        a = v0 - ro
        b = v1 - ro
        c = v2 - ro
        # Permute so the ray's dominant axis becomes z, written as explicit
        # cases: Taichi indexes a vector by a runtime value only under a global
        # flag, and codegens it poorly in the hottest loop in the renderer.
        ax = ti.abs(rd[0])
        ay = ti.abs(rd[1])
        az = ti.abs(rd[2])
        a_x, a_y, a_z = a[0], a[1], a[2]
        b_x, b_y, b_z = b[0], b[1], b[2]
        c_x, c_y, c_z = c[0], c[1], c[2]
        d_x, d_y, d_z = rd[0], rd[1], rd[2]
        if (ax >= ay) and (ax >= az):
            a_x, a_y, a_z = a[1], a[2], a[0]
            b_x, b_y, b_z = b[1], b[2], b[0]
            c_x, c_y, c_z = c[1], c[2], c[0]
            d_x, d_y, d_z = rd[1], rd[2], rd[0]
        elif ay >= az:
            a_x, a_y, a_z = a[2], a[0], a[1]
            b_x, b_y, b_z = b[2], b[0], b[1]
            c_x, c_y, c_z = c[2], c[0], c[1]
            d_x, d_y, d_z = rd[2], rd[0], rd[1]
        if d_z < 0.0:
            # Preserve winding when the dominant component points the other way.
            a_x, a_y = a_y, a_x
            b_x, b_y = b_y, b_x
            c_x, c_y = c_y, c_x
            d_x, d_y = d_y, d_x
        if d_z != 0.0:
            inv_dz = 1.0 / d_z
            sx = d_x * inv_dz
            sy = d_y * inv_dz
            axs = a_x - sx * a_z
            ays = a_y - sy * a_z
            bxs = b_x - sx * b_z
            bys = b_y - sy * b_z
            cxs = c_x - sx * c_z
            cys = c_y - sy * c_z
            # Edge functions, one per opposite vertex.
            u = cxs * bys - cys * bxs
            v = axs * cys - ays * cxs
            w = bxs * ays - bys * axs
            lo = ti.min(u, ti.min(v, w))
            hi = ti.max(u, ti.max(v, w))
            inside = (lo >= 0.0) or (hi <= 0.0)
            if inside:
                # An exactly-zero edge function is the shared-edge case; break
                # the tie so one neighbour owns it.
                if u == 0.0:
                    inside = inside and _edge_is_canonical(bxs, bys, cxs, cys)
                if v == 0.0:
                    inside = inside and _edge_is_canonical(cxs, cys, axs, ays)
                if w == 0.0:
                    inside = inside and _edge_is_canonical(axs, ays, bxs, bys)
            if inside:
                det = u + v + w
                if det != 0.0:
                    inv_det = 1.0 / det
                    t = (u * a_z + v * b_z + w * c_z) * inv_dz * inv_det
                    w1 = v * inv_det
                    w2 = w * inv_det
                    ok = 1
    else:
        e1 = v1 - v0
        e2 = v2 - v0
        pv = rd.cross(e2)
        det = e1.dot(pv)
        if ti.abs(det) > 1e-12:
            inv_det = 1.0 / det
            tvec = ro - v0
            w1 = tvec.dot(pv) * inv_det
            qv = tvec.cross(e1)
            w2 = rd.dot(qv) * inv_det
            if ((w1 >= -BARYCENTRIC_EPSILON)
                    and (w2 >= -BARYCENTRIC_EPSILON)
                    and (w1 + w2 <= 1.0 + BARYCENTRIC_EPSILON)):
                t = e2.dot(qv) * inv_det
                ok = 1
    return ok, w1, w2, t

# Hits gathered per BVH traversal by the deterministic renderer. Depth
# peeling consumes hits strictly front-to-back; collecting a small batch of
# nearest hits per traversal lets a ray crossing several translucent
# surfaces re-traverse the scene once per KBUF surfaces instead of once per
# surface (and skip the final "anything left?" traversal whenever a batch
# comes back not full).
# KBUF is efficiency-only (the peel's transitive order makes the composite
# KBUF-invariant; verified byte-identical across 1/4/8), so it is exposed for
# per-scene tuning: small KBUF closes the traversal's depth window as soon as
# the buffer fills (tighter pruning for low-depth-complexity scenes), large
# KBUF re-traverses less on deep translucent stacks.
KBUF = max(1, env_int("ALGAN_KBUF", 4))

# Kernel-argument annotation for STBVH sibling-block arrays (see
# stbvh._build_blocks): entry [i, lane] holds one attribute of internal node
# i's BVH_ARITY children -- lanes 0-5 the box dims lo.x/lo.y/lo.z/hi.x/hi.y/
# hi.z, lanes 6(-7) their packed frame intervals -- so one aligned 128-byte
# (f32) or 64-byte (f16, conservatively out-rounded bounds) fetch tests a
# whole sibling group per dependent memory round. The vector element type
# makes Taichi issue the lanes as vector loads.
if BLOCK_F16:
    NODE_ARG = ti.types.ndarray(dtype=ti.types.vector(BVH_ARITY, ti.f16),
                                ndim=2)
else:
    NODE_ARG = ti.types.ndarray(dtype=ti.types.vector(BVH_ARITY, ti.f32),
                                ndim=2)

# tri_extra surface-transport block (see ``_pack_surface_extra``):
# per-corner (reflectivity, roughness) pairs in 0-5, per-corner IOR in 6-8,
# per-corner transmission in 9-11.
_EXTRA_W = 12

# circuit_meta channel layout.
_M_CENTER = 0      # 0-2   plane origin
_M_NORMAL = 3      # 3-5   unit plane normal
_M_BASIS_U = 6     # 6-8   plane frame u axis (unit)
_M_BASIS_V = 9     # 9-11  plane frame v axis (unit)
_M_BORDER_W = 12   # border stroke width in screen pixels (see _circuit_point_region)
_M_FILLED = 13     # > 0.5 if the circuit interior is filled
_M_GRID_W = 14     # texture grid width  (1 for plain fills)
_M_GRID_H = 15     # texture grid height (1 for plain fills)
_M_TEX = 16        # 16-19 2x2 map from plane (u, v) to texture axes
# Surface transport, mirroring tri_extra's channels for flat triangles: material
# metalness (-1 = non-PBR), roughness, the unsigned IOR magnitude (dielectric
# F0), and transmission. A circuit transmits as a thin pane rather than
# refracting (see ``circuit_scatter``).
_M_REFLECTIVITY = 20
_M_ROUGHNESS = 21
_M_IOR = 22
_M_TRANSMISSION = 23
_M_WIDTH = 24


@ti.func
def _safe_inverse(x: ti.f32) -> ti.f32:
    r = 1e12
    if x < 0.0:
        r = -1e12
    if ti.abs(x) > 1e-12:
        r = 1.0 / x
    return r


@ti.func
def _circuit_query_radius(border_w, outline_w, filled):
    """Nearest-edge search radius that can classify one point of a circuit.

    ``border_w`` is the circuit's full stroke width in plane units. A FILLED
    circuit draws its border INWARD from the outline, so classification needs
    distances out to the whole width -- or to the hairline dilation
    ``outline_w``, whichever reaches further. An UNFILLED one centres the stroke
    on the path and only needs half the width.
    """
    r = 0.5 * ti.abs(border_w)
    if filled:
        r = ti.max(ti.abs(border_w), outline_w)
    return r


@ti.func
def _circuit_point_region(border_w, outline_w, filled, crossings, min_dist_sq):
    """Classify one point of a circuit as ``(drawn, in_border)``.

    ``border_w`` is the full stroke width and ``crossings``/``min_dist_sq`` come
    from :func:`_bezier_point_metrics`, so the signed distance to the outline is
    ``d = +/- sqrt(min_dist_sq)``, positive inside.

    A FILLED circuit's border runs INWARD -- the drawn region is the fill itself
    (dilated by ``outline_w`` so hairlines and degenerate fills survive) and the
    border is the part of it within ``border_w`` of the outline, i.e. ``d <=
    border_w``. Raising ``border_width`` therefore eats into the shape instead
    of dilating it, which is what keeps neighbouring glyphs from fusing.

    An UNFILLED circuit has no interior to eat into, so its stroke stays centred
    on the path: the band ``|d| < border_w / 2``, the same total width.
    """
    drawn = False
    in_border = False
    if filled:
        drawn = ((crossings % 2) == 1) or (min_dist_sq < outline_w * outline_w)
        in_border = drawn and (ti.abs(border_w) > 0.0) and (
            ((crossings % 2) == 0)
            or (min_dist_sq < border_w * border_w))
    else:
        half = 0.5 * ti.abs(border_w)
        drawn = min_dist_sq < half * half
        in_border = drawn
    return drawn, in_border


@ti.func
def _bezier_point_metrics(circuit, te, u, v, query_radius, num_circuits,
                          edges_2d: ti.template(),
                          edge_accel: ti.template()):
    """Return even/odd crossings, nearest visible-edge distance, and its direction.

    Crossing candidates come from the circuit's local-y scanline bin. Border
    candidates come from every 2D cell touched by the radius query square.
    Both candidate sets are conservative; the original exact predicates are
    still evaluated here, so the acceleration changes only the number of edges
    inspected. For the border query the exact predicate is the RADIUS ITSELF:
    a grid cell is coarser than the query square (a glyph's cell is a pixel or
    two across), so it hands back segments several pixels away, and a candidate
    is only a result when its own distance is inside ``query_radius``. Without
    that test ``min_dist_sq`` reported the nearest segment in the NEIGHBOURHOOD
    rather than the nearest within reach, which the oriented wedge in
    :func:`_bez_pixel_hit` then modelled as the local boundary: a pixel in a
    glyph's empty interior notch -- the gap between the arms of an ``A`` -- sat
    on the drawn side of both long diagonals' extended LINES and was painted at
    full coverage, a lone opaque speck in the middle of nothing. The plain
    half-plane path is unaffected either way: ``query_radius`` is the widest
    distance the coverage filter can still read (the drawn width plus half a
    pixel's diagonal), so every candidate this drops was already resolving to a
    flat 0 or 1, which is what the ``1e30`` sentinel gives.

    ``(ccu, ccv)`` is the vector from the query point to the closest point on
    the nearest segment -- the closest-point vector this already forms to get
    ``min_dist_sq``, and used to discard. It is the local direction of the
    signed distance field's gradient (pointing OUT of the shape from an interior
    query, IN from an exterior one), which is what turns the distance into a
    boundary LINE and so into an exact, angle-aware coverage
    (``_halfplane_clip_area``). Zero when no edge is within the query radius, in
    which case there is no boundary near the pixel to orient.
    """
    header = ((te * num_circuits + circuit)
              * BEZIER_ACCEL_HEADER_SIZE)
    min_u = ti.bit_cast(edge_accel[header + BEZIER_MIN_U], ti.f32)
    min_v = ti.bit_cast(edge_accel[header + BEZIER_MIN_V], ti.f32)
    max_u = ti.bit_cast(edge_accel[header + BEZIER_MAX_U], ti.f32)
    max_v = ti.bit_cast(edge_accel[header + BEZIER_MAX_V], ti.f32)

    crossings = 0
    if (v >= min_v) and (v <= max_v):
        scan_inv_v = ti.bit_cast(
            edge_accel[header + BEZIER_SCAN_INV_V], ti.f32)
        scan_bin = ti.cast(ti.floor((v - min_v) * scan_inv_v), ti.i32)
        scan_bin = ti.math.clamp(scan_bin, 0, BEZIER_SCAN_BINS - 1)
        begin = edge_accel[header + BEZIER_SCAN_OFFSET_BASE + scan_bin]
        end = edge_accel[header + BEZIER_SCAN_OFFSET_BASE + scan_bin + 1]
        for ptr in range(begin, end):
            e = edge_accel[ptr]
            x0 = edges_2d[te, e, 0]
            y0 = edges_2d[te, e, 1]
            x1 = edges_2d[te, e, 2]
            y1 = edges_2d[te, e, 3]
            if (y0 > v) != (y1 > v):
                x_cross = x0 + (v - y0) * (x1 - x0) / (y1 - y0)
                if x_cross > u:
                    crossings += 1

    min_dist_sq = 1e30
    ccu = 0.0
    ccv = 0.0
    e1x = 0.0
    e1y = 0.0
    # Flatten-time inward sign of the nearest / second-nearest edge (edges_2d
    # column 5, DESIGN_analytic_aa_v2.md ss5.2): +-1 says which perpendicular
    # of the edge's direction points into the drawn region, 0 unknown. Only
    # the wedge branch consumes it; the column exists on every build.
    sg1 = 0.0
    # Second-nearest candidate, which is what turns a thin stroke from a
    # half-plane into a STRIP: one distance can only describe an edge running to
    # infinity, and a glyph stem is two walls a fraction of a pixel apart.
    sec_dist_sq = 1e30
    scu = 0.0
    scv = 0.0
    e2x = 0.0
    e2y = 0.0
    sg2 = 0.0
    if ((query_radius > 0.0) and (u + query_radius >= min_u)
            and (u - query_radius <= max_u)
            and (v + query_radius >= min_v)
            and (v - query_radius <= max_v)):
        # The cell walk below is a superset filter; this is the exact one.
        radius_sq = query_radius * query_radius
        grid_inv_u = ti.bit_cast(
            edge_accel[header + BEZIER_GRID_INV_U], ti.f32)
        grid_inv_v = ti.bit_cast(
            edge_accel[header + BEZIER_GRID_INV_V], ti.f32)
        cell_x0 = ti.cast(ti.floor(
            (u - query_radius - min_u) * grid_inv_u), ti.i32)
        cell_y0 = ti.cast(ti.floor(
            (v - query_radius - min_v) * grid_inv_v), ti.i32)
        cell_x1 = ti.cast(ti.floor(
            (u + query_radius - min_u) * grid_inv_u), ti.i32)
        cell_y1 = ti.cast(ti.floor(
            (v + query_radius - min_v) * grid_inv_v), ti.i32)
        cell_x0 = ti.math.clamp(cell_x0, 0, BEZIER_SPATIAL_GRID - 1)
        cell_y0 = ti.math.clamp(cell_y0, 0, BEZIER_SPATIAL_GRID - 1)
        cell_x1 = ti.math.clamp(cell_x1, 0, BEZIER_SPATIAL_GRID - 1)
        cell_y1 = ti.math.clamp(cell_y1, 0, BEZIER_SPATIAL_GRID - 1)
        for cell_y in range(cell_y0, cell_y1 + 1):
            for cell_x in range(cell_x0, cell_x1 + 1):
                cell = cell_y * BEZIER_SPATIAL_GRID + cell_x
                begin = edge_accel[
                    header + BEZIER_SPATIAL_OFFSET_BASE + cell]
                end = edge_accel[
                    header + BEZIER_SPATIAL_OFFSET_BASE + cell + 1]
                for ptr in range(begin, end):
                    e = edge_accel[ptr]
                    x0 = edges_2d[te, e, 0]
                    y0 = edges_2d[te, e, 1]
                    x1 = edges_2d[te, e, 2]
                    y1 = edges_2d[te, e, 3]
                    dx = x1 - x0
                    dy = y1 - y0
                    seg_t = ((u - x0) * dx + (v - y0) * dy) / ti.max(
                        dx * dx + dy * dy, 1e-12)
                    seg_t = ti.math.clamp(seg_t, 0.0, 1.0)
                    cx = x0 + seg_t * dx - u
                    cy = y0 + seg_t * dy - v
                    dsq = cx * cx + cy * cy
                    if dsq < radius_sq:
                        if dsq < min_dist_sq:
                            sec_dist_sq = min_dist_sq
                            scu = ccu
                            scv = ccv
                            e2x = e1x
                            e2y = e1y
                            sg2 = sg1
                            min_dist_sq = dsq
                            ccu = cx
                            ccv = cy
                            e1x = dx
                            e1y = dy
                            sg1 = edges_2d[te, e, 5]
                        elif dsq < sec_dist_sq:
                            sec_dist_sq = dsq
                            scu = cx
                            scv = cy
                            e2x = dx
                            e2y = dy
                            sg2 = edges_2d[te, e, 5]

    return (crossings, min_dist_sq, ccu, ccv, e1x, e1y, sg1,
            sec_dist_sq, scu, scv, e2x, e2y, sg2)


@ti.func
def _generate_ray(f, px, py, jitter_x, jitter_y, half_screen_w, half_screen_h,
                  cam_origin: ti.template(), screen_point: ti.template(),
                  pixel_basis_x: ti.template(), pixel_basis_y: ti.template()):
    """Build the world-space ray through a point inside a pixel
    (``jitter = 0.5`` is the pixel center; random jitter gives sub-pixel
    anti-aliasing when averaging many samples).

    Inverts the projection used by Algan's camera: a world point projects to
    screen coordinate ``u = dot(p - screen_point, b)`` which maps to pixel
    ``u * half_screen_h + half_screen`` -- so a pixel position corresponds to
    the world point ``screen_point + u * d_x + v * d_y`` where ``d_*`` is the
    reciprocal screen basis (precomputed into ``pixel_basis_*``).
    """
    x = ti.cast(px, ti.f32) + jitter_x
    y = ti.cast(py, ti.f32) + jitter_y
    u = (x - half_screen_w) / half_screen_h
    v = (y - half_screen_h) / half_screen_h
    ro = ti.math.vec3(cam_origin[f, 0], cam_origin[f, 1], cam_origin[f, 2])
    pix = ti.math.vec3(
        screen_point[f, 0] + u * pixel_basis_x[f, 0] + v * pixel_basis_y[f, 0],
        screen_point[f, 1] + u * pixel_basis_x[f, 1] + v * pixel_basis_y[f, 1],
        screen_point[f, 2] + u * pixel_basis_x[f, 2] + v * pixel_basis_y[f, 2],
    )
    rd = (pix - ro).normalized()
    return ro, rd


@ti.func
def _axis_cos(f, ro, rd, screen_point: ti.template()):
    """Cosine between a primary ray and the camera's optical axis.

    ``pixel_world_scale`` is world-per-pixel PER UNIT OF PERPENDICULAR DEPTH,
    calibrated on the optical axis from the screen plane's own distance. The
    camera is a pinhole with a FLAT image plane, so the Jacobian of
    world-to-screen on a surface facing the camera depends on that
    perpendicular depth alone and is constant across the frame -- but
    ``_generate_ray`` returns a NORMALISED direction, so a ray parameter ``t``
    is the SLANT RANGE, which is ``depth / cos(theta)``.

    Multiplying by this cosine converts one into the other. Without it every
    width derived from ``pixel_size`` (a circuit's stroke, its border band, the
    anti-crack outline dilation) is drawn ``1 / cos(theta)`` too wide, which at
    the default 53-degree fov reaches +35% at the left and right edges
    of a 16:9 frame: a horizontal ``Line`` whose geometry is identical in every
    column is drawn 9.09 px wide at the centre of a 720p frame and 12.18 px at
    its edges. ``benchmarks/_stroke_width_check.py`` is the regression check.
    """
    fwd = (ti.math.vec3(screen_point[f, 0], screen_point[f, 1],
                        screen_point[f, 2]) - ro).normalized()
    return rd.dot(fwd)


@ti.func
def _random_unit_vector():
    """Uniformly distributed point on the unit sphere."""
    z = 2.0 * ti.random(ti.f32) - 1.0
    phi = 6.283185307179586 * ti.random(ti.f32)
    r_xy = ti.sqrt(ti.max(0.0, 1.0 - z * z))
    return ti.math.vec3(r_xy * ti.cos(phi), r_xy * ti.sin(phi), z)


@ti.func
def _cosine_hemisphere_direction(normal):
    """Cosine-weighted random direction in the hemisphere around ``normal``
    (the Lambertian scattering distribution).
    """
    d = normal + _random_unit_vector()
    if d.norm() < 1e-6:
        d = normal
    return d.normalized()


@ti.func
def _test_children(blk, f, ro, inv_rd, t_lo, t_hi, blocks: ti.template()):
    """Spatio-temporal test of internal node ``blk``'s whole sibling group:
    per-child frame containment + slab test restricted to the parametric
    window [t_lo, t_hi] of still-relevant hits. The children's bounds and
    packed frame intervals live in one aligned SoA block (see
    stbvh._build_blocks) -- a single fetch feeds ``BVH_ARITY`` independent
    box tests, so a whole sibling group costs one dependent memory round.
    Returns ``(mask, near)`` where ``mask`` is the bitmask of intersected
    children (bit c = child ``BVH_ARITY * blk + 1 + c``) and ``near[c]`` is
    the slab-test entry distance for each passing child. Missed children get
    a large sentinel and are never selected. Traversals use these distances
    for deterministic near-to-far selection and recompute them when a saved
    sibling group is restored after the depth window tightens.

    The per-child predicate is float-for-float the retired per-node walk's
    box test, and callers re-validate pending masks whenever the depth window
    tightens (see the walk skeletons), so the set of nodes whose leaf slots
    get tested is *bit-identical* to the old walk's -- this matters because
    triangle hits routinely lie exactly on their bounding boxes' faces, where
    a test evaluated against a stale (looser) window can admit hits within a
    float ulp of the ``DEPTH_TIE_EPSILON`` acceptance boundary that the old
    walk never tested (observed as epsilon-level image changes).

    f16 blocks decode exactly (f16 -> f32 casts are lossless) to the
    conservatively out-rounded bounds baked at build time: never a false
    cull, but the *looser* boxes shift those same boundary cracks, so f16 is
    epsilon-level non-identical and stays opt-in (``ALGAN_BVH_BLOCK_F16``).
    """
    lox = blocks[blk, 0]
    loy = blocks[blk, 1]
    loz = blocks[blk, 2]
    hix = blocks[blk, 3]
    hiy = blocks[blk, 4]
    hiz = blocks[blk, 5]
    ts_a = blocks[blk, 6]
    ts_b = blocks[blk, 7]
    mask = 0
    near = ti.Vector([1e30] * BVH_ARITY)
    for c in ti.static(range(BVH_ARITY)):
        tspan = 0
        if ti.static(BLOCK_F16):
            tspan = ti.cast(ti.bit_cast(ts_a[c], ti.u16), ti.i32) | (
                ti.cast(ti.bit_cast(ts_b[c], ti.u16), ti.i32) << 16)
        else:
            tspan = ti.bit_cast(ts_a[c], ti.i32)
        if ((tspan & 0xFFFF) <= f) and (f <= ((tspan >> 16) & 0x7FFF)):
            tx0 = (ti.cast(lox[c], ti.f32) - ro[0]) * inv_rd[0]
            tx1 = (ti.cast(hix[c], ti.f32) - ro[0]) * inv_rd[0]
            t_near = ti.min(tx0, tx1)
            t_far = ti.max(tx0, tx1)
            ty0 = (ti.cast(loy[c], ti.f32) - ro[1]) * inv_rd[1]
            ty1 = (ti.cast(hiy[c], ti.f32) - ro[1]) * inv_rd[1]
            t_near = ti.max(t_near, ti.min(ty0, ty1))
            t_far = ti.min(t_far, ti.max(ty0, ty1))
            tz0 = (ti.cast(loz[c], ti.f32) - ro[2]) * inv_rd[2]
            tz1 = (ti.cast(hiz[c], ti.f32) - ro[2]) * inv_rd[2]
            t_near = ti.max(t_near, ti.min(tz0, tz1))
            t_far = ti.min(t_far, ti.max(tz0, tz1))
            if ((t_far >= ti.max(t_near, 0.0)) and (t_near <= t_hi)
                    and (t_far >= t_lo)):
                mask |= 1 << c
                near[c] = t_near
    return mask, near


@ti.func
def _test_children_refit(row, ro, inv_rd, t_lo, t_hi, blocks: ti.template()):
    """Refit-tree counterpart of :func:`_test_children` (see refit_bvh.py):
    ``row`` is the frame's row of internal node ``blk``
    (``_refit_row0(...) + blk``) in the ``[Tb * num_blocks, 8, ARITY]`` block
    array. The per-child gate is the link word -- ``-1`` marks an absent
    child or one whose whole subtree is invisible at this frame -- instead of
    a frame-interval test; the bounds are already this frame's exact boxes,
    so no temporal decode remains. The slab test itself is float-for-float
    the classic one.
    """
    lox = blocks[row, 0]
    loy = blocks[row, 1]
    loz = blocks[row, 2]
    hix = blocks[row, 3]
    hiy = blocks[row, 4]
    hiz = blocks[row, 5]
    ts_a = blocks[row, 6]
    ts_b = blocks[row, 7]
    mask = 0
    near = ti.Vector([1e30] * BVH_ARITY)
    for c in ti.static(range(BVH_ARITY)):
        w = 0
        if ti.static(BLOCK_F16):
            w = ti.cast(ti.bit_cast(ts_a[c], ti.u16), ti.i32) | (
                ti.cast(ti.bit_cast(ts_b[c], ti.u16), ti.i32) << 16)
        else:
            w = ti.bit_cast(ts_a[c], ti.i32)
        if w != -1:
            tx0 = (ti.cast(lox[c], ti.f32) - ro[0]) * inv_rd[0]
            tx1 = (ti.cast(hix[c], ti.f32) - ro[0]) * inv_rd[0]
            t_near = ti.min(tx0, tx1)
            t_far = ti.max(tx0, tx1)
            ty0 = (ti.cast(loy[c], ti.f32) - ro[1]) * inv_rd[1]
            ty1 = (ti.cast(hiy[c], ti.f32) - ro[1]) * inv_rd[1]
            t_near = ti.max(t_near, ti.min(ty0, ty1))
            t_far = ti.min(t_far, ti.max(ty0, ty1))
            tz0 = (ti.cast(loz[c], ti.f32) - ro[2]) * inv_rd[2]
            tz1 = (ti.cast(hiz[c], ti.f32) - ro[2]) * inv_rd[2]
            t_near = ti.max(t_near, ti.min(tz0, tz1))
            t_far = ti.min(t_far, ti.max(tz0, tz1))
            if ((t_far >= ti.max(t_near, 0.0)) and (t_near <= t_hi)
                    and (t_far >= t_lo)):
                mask |= 1 << c
                near[c] = t_near
    return mask, near


@ti.func
def _refit_row0(f, num_blocks, blocks: ti.template()):
    """Row base of frame ``f``'s sibling blocks in a refit tree: ``blocks``
    stores ``Tb`` frames of ``num_blocks`` rows flattened together (``Tb`` is
    1 for static geometry), mirroring the ``f % shape[0]`` convention of the
    geometry arrays. ``num_blocks`` rides in the walk's ``first_leaf``
    argument slot.
    """
    return (f % (blocks.shape[0] // num_blocks)) * num_blocks


@ti.func
def _refit_link(row, c, blocks: ti.template()):
    """Per-(frame, child) link word of a refit sibling block ``row``
    (see refit_bvh.py): ``-1`` invalid, ``< 0`` a leaf child (bits 0-29 the
    primitive index, bit 30 its per-frame full-opacity flag), ``>= 0`` the
    child's own block index. Reads the lanes the caller's group test just
    fetched, so this is a same-cache-line load.
    """
    ts_a = blocks[row, 6]
    ts_b = blocks[row, 7]
    w = 0
    for cc in ti.static(range(BVH_ARITY)):
        if cc == c:
            if ti.static(BLOCK_F16):
                w = ti.cast(ti.bit_cast(ts_a[cc], ti.u16), ti.i32) | (
                    ti.cast(ti.bit_cast(ts_b[cc], ti.u16), ti.i32) << 16)
            else:
                w = ti.bit_cast(ts_a[cc], ti.i32)
    return w


# Refit link-word decode masks (must match refit_bvh.LINK_*).
_REFIT_PRIM_MASK = (1 << 30) - 1


@ti.func
def _group_test(refit: ti.template(), row0, blk, f, ro, inv_rd, t_lo, t_hi,
                blocks: ti.template()):
    """Sibling-group test dispatch: the classic frame-gated implicit-heap
    block (``refit == 0``) or the per-frame link-gated refit block. ``row0``
    is dead in classic mode.
    """
    mask = 0
    near = ti.Vector([1e30] * BVH_ARITY)
    if ti.static(refit != 0):
        mask, near = _test_children_refit(row0 + blk, ro, inv_rd, t_lo, t_hi,
                                          blocks)
    else:
        mask, near = _test_children(blk, f, ro, inv_rd, t_lo, t_hi, blocks)
    return mask, near


@ti.func
def _nearest_pending_child(mask, near):
    """Return the pending child with the smallest slab entry distance.

    Iteration is low-to-high and replacement is strict, so equal entry
    distances retain the lower child index as a deterministic tie-break.
    """
    best_c = 0
    best_t = 1e30
    found = 0
    for c in ti.static(range(BVH_ARITY)):
        if mask & (1 << c) != 0:
            if (found == 0) or (near[c] < best_t):
                best_c = c
                best_t = near[c]
                found = 1
    return best_c


@ti.func
def _test_root(f, ro, inv_rd, t_lo, t_hi, blocks: ti.template()):
    """Box test of the tree's root, reconstructed from block 0: the retired
    per-node walk tested the root's stored row, which the builders computed
    as the (min/max) union of its children -- min/max are exact, so unioning
    block 0's lanes here recovers the very same floats and the test is
    bit-identical to the old root visit. Matters only for rays grazing the
    scene bound (a hit exactly on the root box face can sit within an ulp of
    the test boundary), but bit-parity needs it.
    """
    lox = blocks[0, 0]
    loy = blocks[0, 1]
    loz = blocks[0, 2]
    hix = blocks[0, 3]
    hiy = blocks[0, 4]
    hiz = blocks[0, 5]
    ts_a = blocks[0, 6]
    ts_b = blocks[0, 7]
    lo_x = ti.cast(lox[0], ti.f32)
    lo_y = ti.cast(loy[0], ti.f32)
    lo_z = ti.cast(loz[0], ti.f32)
    hi_x = ti.cast(hix[0], ti.f32)
    hi_y = ti.cast(hiy[0], ti.f32)
    hi_z = ti.cast(hiz[0], ti.f32)
    t0 = 0
    t1 = 0
    if ti.static(BLOCK_F16):
        t0 = ti.cast(ti.bit_cast(ts_a[0], ti.u16), ti.i32)
        t1 = ti.cast(ti.bit_cast(ts_b[0], ti.u16), ti.i32)
    else:
        ts = ti.bit_cast(ts_a[0], ti.i32)
        t0 = ts & 0xFFFF
        t1 = (ts >> 16) & 0x7FFF
    for c in ti.static(range(1, BVH_ARITY)):
        lo_x = ti.min(lo_x, ti.cast(lox[c], ti.f32))
        lo_y = ti.min(lo_y, ti.cast(loy[c], ti.f32))
        lo_z = ti.min(lo_z, ti.cast(loz[c], ti.f32))
        hi_x = ti.max(hi_x, ti.cast(hix[c], ti.f32))
        hi_y = ti.max(hi_y, ti.cast(hiy[c], ti.f32))
        hi_z = ti.max(hi_z, ti.cast(hiz[c], ti.f32))
        tc0 = 0
        tc1 = 0
        if ti.static(BLOCK_F16):
            tc0 = ti.cast(ti.bit_cast(ts_a[c], ti.u16), ti.i32)
            tc1 = ti.cast(ti.bit_cast(ts_b[c], ti.u16), ti.i32)
        else:
            ts = ti.bit_cast(ts_a[c], ti.i32)
            tc0 = ts & 0xFFFF
            tc1 = (ts >> 16) & 0x7FFF
        t0 = ti.min(t0, tc0)
        t1 = ti.max(t1, tc1)
    hit = False
    if (t0 <= f) and (f <= t1):
        tx0 = (lo_x - ro[0]) * inv_rd[0]
        tx1 = (hi_x - ro[0]) * inv_rd[0]
        t_near = ti.min(tx0, tx1)
        t_far = ti.max(tx0, tx1)
        ty0 = (lo_y - ro[1]) * inv_rd[1]
        ty1 = (hi_y - ro[1]) * inv_rd[1]
        t_near = ti.max(t_near, ti.min(ty0, ty1))
        t_far = ti.min(t_far, ti.max(ty0, ty1))
        tz0 = (lo_z - ro[2]) * inv_rd[2]
        tz1 = (hi_z - ro[2]) * inv_rd[2]
        t_near = ti.max(t_near, ti.min(tz0, tz1))
        t_far = ti.min(t_far, ti.max(tz0, tz1))
        hit = ((t_far >= ti.max(t_near, 0.0)) and (t_near <= t_hi)
               and (t_far >= t_lo))
    return hit


@ti.func
def _comes_after(t, layer, t_prev, layer_prev) -> bool:
    """Strict, transitive total order along the ray: by distance, with
    near-coplanar hits ordered by descending layer index.

    Distances are floored into ``DEPTH_TIE_EPSILON``-wide bins so hits in the
    same bin compare equal on distance and fall back to ``layer``. Binning
    (rather than the old symmetric ``t +/- EPS`` window) keeps the comparison
    transitive: the window version could rank A<B, B<C yet C<A, so the order in
    which the depth-peel consumed near-coplanar hits -- and thus the composite
    -- depended on how the hits were grouped, i.e. on KBUF (and on the BVH
    build). With a transitive order, the peel visits hits in one fixed sequence
    regardless of how many are gathered per traversal, so KBUF is efficiency-only.
    """
    bt = ti.floor(t * INV_DEPTH_TIE_EPSILON)
    bp = ti.floor(t_prev * INV_DEPTH_TIE_EPSILON)
    return (bt > bp) or ((bt == bp) and (layer < layer_prev))


@ti.func
def _shadow_identity_t_min(f, prim, src_sid, tri_obj: ti.template(),
                           ident: ti.template()) -> ti.f32:
    """Acceptance floor along a shadow ray for one candidate triangle hit.

    ``ident != 0`` (compile-time) engages self-shadow rejection by identity
    (DESIGN_mesh_identity_open.md ssI): a hit on the ray's OWN surface keeps
    the ``MIN_HIT_DISTANCE`` guard, while any OTHER mesh's threshold is zero,
    so contact shadows survive where the absolute epsilon used to erase them.
    The rejection is per hit -- "same mesh AND near-zero t", never "same
    mesh": a concave solid legitimately shadows itself.

    ``src_sid < 0`` (per-ray runtime) disables the test for that ray: callers
    without a source identity -- the megakernel's camera, secondary and
    shadow rays, and shadow events whose source is a bezier circuit or whose
    id did not fit the event_msk packing -- keep exactly the old epsilon.
    """
    t_min = MIN_HIT_DISTANCE
    if ti.static(ident != 0):
        if src_sid >= 0:
            hit_obj = ti.cast(tri_obj[f % tri_obj.shape[0], prim], ti.i32)
            if hit_obj != src_sid:
                t_min = 0.0
    return t_min


@ti.func
def _nearest_triangle_hit(refit: ti.template(), ro, rd, inv_rd, f, ff,
                          t_prev, layer_prev,
                          t_cap, layer_offset,
                          nodes: ti.template(), node_miss: ti.template(),
                          leaf_prim: ti.template(), leaf_tspan: ti.template(),
                          first_leaf, tri_pos: ti.template(),
                          src_sid, tri_obj: ti.template(),
                          ident: ti.template()):
    """Nearest triangle intersection strictly after (t_prev, layer_prev).

    ``refit != 0`` walks a refit tree instead (see refit_bvh.py): ``nodes``
    is the per-frame block array, ``first_leaf`` carries num_blocks, children
    are followed through per-(frame, child) link words and the leaf-slot
    arrays are unused.
    """
    best_t = 1e30
    best_layer = -1e30
    best_prim = -1
    best_w1 = 0.0
    best_w2 = 0.0
    tp = f % tri_pos.shape[0]
    row0 = 0
    if ti.static(refit != 0):
        row0 = _refit_row0(f, first_leaf, nodes)
    g_sp = 0
    g_st = ti.Vector([0] * _GROUP_STACK)
    g_cur = 0
    g_pend, g_near = _group_test(
        refit, row0, 0, f, ro, inv_rd, t_prev - DEPTH_TIE_EPSILON,
        ti.min(best_t + DEPTH_TIE_EPSILON,
               t_cap + DEPTH_TIE_EPSILON), nodes)
    while True:
        if g_pend == 0:
            if g_sp == 0:
                break
            g_sp -= 1
            saved = g_st[g_sp]
            g_cur = saved >> BVH_ARITY
            saved_mask = saved & _GROUP_MASK
            fresh_mask, g_near = _group_test(
                refit, row0, g_cur, f, ro, inv_rd,
                t_prev - DEPTH_TIE_EPSILON,
                ti.min(best_t + DEPTH_TIE_EPSILON,
                       t_cap + DEPTH_TIE_EPSILON), nodes)
            g_pend = saved_mask & fresh_mask
        else:
            g_c = _nearest_pending_child(g_pend, g_near)
            g_pend &= ~(1 << g_c)
            descend = 0
            child_blk = 0
            l_prim = -1
            l_base = 0
            if ti.static(refit != 0):
                w = _refit_link(row0 + g_cur, g_c, nodes)
                if w >= 0:
                    descend = 1
                    child_blk = w
                else:
                    l_prim = w & _REFIT_PRIM_MASK
            else:
                g_child = BVH_ARITY * g_cur + 1 + g_c
                if g_child >= first_leaf:
                    l_base = (g_child - first_leaf) * LEAF_SIZE
                else:
                    descend = 1
                    child_blk = g_child
            if descend == 0:
                for j in ti.static(range(1 if refit != 0 else LEAF_SIZE)):
                    prim = l_prim
                    if ti.static(refit == 0):
                        prim = -1
                        p0 = leaf_prim[l_base + j]
                        tspan = leaf_tspan[l_base + j]
                        if ((p0 >= 0) and ((tspan & 0xFFFF) <= f)
                                and (f <= ((tspan >> 16) & 0x7FFF))):
                            prim = p0
                    if prim >= 0:
                        v0 = ti.math.vec3(tri_pos[tp, prim, 0],
                                          tri_pos[tp, prim, 1],
                                          tri_pos[tp, prim, 2])
                        v1 = ti.math.vec3(tri_pos[tp, prim, 3],
                                          tri_pos[tp, prim, 4],
                                          tri_pos[tp, prim, 5])
                        v2 = ti.math.vec3(tri_pos[tp, prim, 6],
                                          tri_pos[tp, prim, 7],
                                          tri_pos[tp, prim, 8])
                        hit_ok, w1, w2, t = _tri_hit(ro, rd, v0, v1, v2)
                        if hit_ok != 0:
                            layer = layer_offset + ti.cast(prim, ti.f32)
                            if ((t > _shadow_identity_t_min(
                                    f, prim, src_sid, tri_obj, ident))
                                    and _comes_after(t, layer, t_prev,
                                                     layer_prev)
                                    and _comes_after(best_t, best_layer,
                                                     t, layer)):
                                best_t = t
                                best_layer = layer
                                best_prim = prim
                                best_w1 = w1
                                best_w2 = w2
            else:
                if g_pend != 0:
                    g_st[g_sp] = (g_cur << BVH_ARITY) | g_pend
                    g_sp += 1
                g_cur = child_blk
                g_pend, g_near = _group_test(
                    refit, row0, g_cur, f, ro, inv_rd,
                    t_prev - DEPTH_TIE_EPSILON,
                    ti.min(best_t + DEPTH_TIE_EPSILON,
                           t_cap + DEPTH_TIE_EPSILON), nodes)
    return best_t, best_prim, best_w1, best_w2, best_layer


@ti.func
def _nearest_bezier_hit(refit: ti.template(), ro, rd, inv_rd, f, ff, t_prev,
                        layer_prev, t_cap,
                        pixel_size_per_t, base_dist,
                        nodes: ti.template(), node_miss: ti.template(),
                        leaf_prim: ti.template(), leaf_tspan: ti.template(),
                        first_leaf, circuit_meta: ti.template(),
                        edges_2d: ti.template(), edge_accel: ti.template()):
    """Nearest bezier-circuit intersection strictly after (t_prev, layer_prev).

    A circuit hit is the ray/plane intersection classified against the
    circuit's 2D polyline: inside the even-odd fill (when filled) or within
    the screen-constant border width of the curve. Returns the plane (u, v)
    coordinates for color sampling and whether the border color applies.
    """
    best_t = 1e30
    best_layer = -1e30
    best_circuit = -1
    best_border = 0
    best_u = 0.0
    best_v = 0.0
    num_meta_frames = circuit_meta.shape[0]
    num_edge_frames = edges_2d.shape[0]
    row0 = 0
    if ti.static(refit != 0):
        row0 = _refit_row0(f, first_leaf, nodes)
    g_sp = 0
    g_st = ti.Vector([0] * _GROUP_STACK)
    g_cur = 0
    g_pend, g_near = _group_test(
        refit, row0, 0, f, ro, inv_rd, t_prev - DEPTH_TIE_EPSILON,
        ti.min(best_t + DEPTH_TIE_EPSILON,
               t_cap + DEPTH_TIE_EPSILON), nodes)
    while True:
        if g_pend == 0:
            if g_sp == 0:
                break
            g_sp -= 1
            saved = g_st[g_sp]
            g_cur = saved >> BVH_ARITY
            saved_mask = saved & _GROUP_MASK
            fresh_mask, g_near = _group_test(
                refit, row0, g_cur, f, ro, inv_rd,
                t_prev - DEPTH_TIE_EPSILON,
                ti.min(best_t + DEPTH_TIE_EPSILON,
                       t_cap + DEPTH_TIE_EPSILON), nodes)
            g_pend = saved_mask & fresh_mask
        else:
            g_c = _nearest_pending_child(g_pend, g_near)
            g_pend &= ~(1 << g_c)
            descend = 0
            child_blk = 0
            l_prim = -1
            l_base = 0
            if ti.static(refit != 0):
                w = _refit_link(row0 + g_cur, g_c, nodes)
                if w >= 0:
                    descend = 1
                    child_blk = w
                else:
                    l_prim = w & _REFIT_PRIM_MASK
            else:
                g_child = BVH_ARITY * g_cur + 1 + g_c
                if g_child >= first_leaf:
                    l_base = (g_child - first_leaf) * LEAF_SIZE
                else:
                    descend = 1
                    child_blk = g_child
            if descend == 0:
                for j in ti.static(range(1 if refit != 0 else LEAF_SIZE)):
                    circuit = l_prim
                    if ti.static(refit == 0):
                        circuit = -1
                        p0 = leaf_prim[l_base + j]
                        tspan = leaf_tspan[l_base + j]
                        if ((p0 >= 0) and ((tspan & 0xFFFF) <= f)
                                and (f <= ((tspan >> 16) & 0x7FFF))):
                            circuit = p0
                    if circuit >= 0:
                        tm = f % num_meta_frames
                        n = ti.math.vec3(circuit_meta[tm, circuit, _M_NORMAL],
                                         circuit_meta[tm, circuit, _M_NORMAL + 1],
                                         circuit_meta[tm, circuit, _M_NORMAL + 2])
                        denom = rd.dot(n)
                        layer = ti.cast(circuit, ti.f32)
                        if ti.abs(denom) > 1e-9:
                            center = ti.math.vec3(
                                circuit_meta[tm, circuit, _M_CENTER],
                                circuit_meta[tm, circuit, _M_CENTER + 1],
                                circuit_meta[tm, circuit, _M_CENTER + 2])
                            t = (center - ro).dot(n) / denom
                            if ((t > MIN_HIT_DISTANCE)
                                    and _comes_after(t, layer, t_prev,
                                                     layer_prev)
                                    and _comes_after(best_t, best_layer,
                                                     t, layer)):
                                hit = ro + t * rd - center
                                bu = ti.math.vec3(
                                    circuit_meta[tm, circuit, _M_BASIS_U],
                                    circuit_meta[tm, circuit, _M_BASIS_U + 1],
                                    circuit_meta[tm, circuit, _M_BASIS_U + 2])
                                bv = ti.math.vec3(
                                    circuit_meta[tm, circuit, _M_BASIS_V],
                                    circuit_meta[tm, circuit, _M_BASIS_V + 1],
                                    circuit_meta[tm, circuit, _M_BASIS_V + 2])
                                u = hit.dot(bu)
                                v = hit.dot(bv)

                                # World size of one screen pixel at this hit,
                                # for screen-constant border/outline widths.
                                pixel_size = pixel_size_per_t * (base_dist + t)
                                border_w = (circuit_meta[tm, circuit, _M_BORDER_W]
                                            * pixel_size)
                                outline_w = 0.6 * pixel_size
                                filled = circuit_meta[tm, circuit, _M_FILLED] > 0.5
                                query_radius = _circuit_query_radius(
                                    border_w, outline_w, filled)
                                te = f % num_edge_frames
                                (crossings, min_dist_sq, _ccu, _ccv, _e1x,
                                     _e1y, _sg1, _s2, _s2u, _s2v, _e2x, _e2y,
                                     _sg2) = _bezier_point_metrics(
                                    circuit, te, u, v, query_radius,
                                    circuit_meta.shape[1], edges_2d, edge_accel)
                                inside, in_border = _circuit_point_region(
                                    border_w, outline_w, filled, crossings,
                                    min_dist_sq)
                                if inside:
                                    best_t = t
                                    best_layer = layer
                                    best_circuit = circuit
                                    best_border = 1 if in_border else 0
                                    best_u = u
                                    best_v = v
            else:
                if g_pend != 0:
                    g_st[g_sp] = (g_cur << BVH_ARITY) | g_pend
                    g_sp += 1
                g_cur = child_blk
                g_pend, g_near = _group_test(
                    refit, row0, g_cur, f, ro, inv_rd,
                    t_prev - DEPTH_TIE_EPSILON,
                    ti.min(best_t + DEPTH_TIE_EPSILON,
                           t_cap + DEPTH_TIE_EPSILON), nodes)
    return best_t, best_circuit, best_border, best_u, best_v, best_layer


@ti.func
def _circuit_texture_color(circuit, f, u, v,
                           circuit_meta: ti.template(),
                           texture_colors: ti.template()):
    """Bilinearly sample one of a circuit's color texture grids."""
    color = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
    alpha = 0.0
    tm = f % circuit_meta.shape[0]
    tc = f % texture_colors.shape[0]
    grid_w = circuit_meta[tm, circuit, _M_GRID_W]
    grid_h = circuit_meta[tm, circuit, _M_GRID_H]
    # Map plane (u, v) to texture-grid coordinates via the precomputed
    # 2x2 transform (mirrors the rasterizer's texture lookup).
    c1 = 0.5 * (u * circuit_meta[tm, circuit, _M_TEX]
                + v * circuit_meta[tm, circuit, _M_TEX + 1]) + 0.5
    c2 = 0.5 * (u * circuit_meta[tm, circuit, _M_TEX + 2]
                + v * circuit_meta[tm, circuit, _M_TEX + 3]) + 0.5
    x = ti.math.clamp(c2 * grid_h, 0.0, ti.max(grid_h - 1.0, 0.0))
    y = ti.math.clamp(c1 * grid_w, 0.0, ti.max(grid_w - 1.0, 0.0))
    num_points = texture_colors.shape[2]
    x_floor = ti.floor(x)
    y_floor = ti.floor(y)
    xr = x - x_floor
    yr = y - y_floor
    sum_w = 0.0
    for corner in ti.static(range(4)):
        cx = x_floor + (corner % 2)
        cy = y_floor + (corner // 2)
        w = (xr if (corner % 2) == 1 else 1.0 - xr) * (
            yr if (corner // 2) == 1 else 1.0 - yr)
        p = ti.cast(cx + cy * grid_h, ti.i32)
        p = ti.math.clamp(p, 0, num_points - 1)
        color += w * ti.math.vec4(texture_colors[tc, circuit, p, 0],
                                  texture_colors[tc, circuit, p, 1],
                                  texture_colors[tc, circuit, p, 2],
                                  texture_colors[tc, circuit, p, 3])
        alpha += w * texture_colors[tc, circuit, p, 4]
        sum_w += w
    color /= ti.max(sum_w, 1e-6)
    alpha /= ti.max(sum_w, 1e-6)
    return color, alpha


@ti.func
def _circuit_border_color(circuit, f, u, v,
                          circuit_meta: ti.template(),
                          circuit_border_colors: ti.template()):
    """A circuit's border color/alpha sampled from its texture grid."""
    return _circuit_texture_color(
        circuit, f, u, v, circuit_meta, circuit_border_colors)


@ti.func
def _circuit_fill_color(circuit, f, u, v,
                        circuit_meta: ti.template(),
                        circuit_colors: ti.template()):
    """A circuit's fill color/alpha sampled from its texture grid."""
    return _circuit_texture_color(
        circuit, f, u, v, circuit_meta, circuit_colors)


@ti.func
def _sample_circuit_color(circuit, f, u, v, in_border,
                          circuit_meta: ti.template(),
                          circuit_colors: ti.template(),
                          circuit_border_colors: ti.template()):
    """Color of a circuit at plane coordinates (u, v): the border color, or
    the fill color sampled from the circuit's texture grid.
    """
    color = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
    alpha = 0.0
    if in_border == 1:
        color, alpha = _circuit_border_color(
            circuit, f, u, v, circuit_meta, circuit_border_colors)
    else:
        color, alpha = _circuit_fill_color(
            circuit, f, u, v, circuit_meta, circuit_colors)
    return color, alpha


@ti.func
def _sample_circuit_color_blend(circuit, f, u, v, border_frac,
                                circuit_meta: ti.template(),
                                circuit_colors: ti.template(),
                                circuit_border_colors: ti.template()):
    """Colour of a circuit point whose pixel straddles the border's inner edge.

    ``border_frac`` is the share of the pixel's COVERED area lying in the border
    band. 0 and 1 reduce to :func:`_sample_circuit_color`'s fill and border
    branches exactly (and skip the other branch's work); in between, the two
    regions are composited by area-weighted alpha -- the premultiplied average a
    supersampled render converges to -- so the border/fill boundary resolves
    continuously instead of snapping at the pixel centre. Glow (channel 3) is a
    separate additive channel and is weighted by area alone.
    """
    cb = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
    ab = 0.0
    cf = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
    af = 0.0
    if border_frac > 0.0:
        cb, ab = _circuit_border_color(
            circuit, f, u, v, circuit_meta, circuit_border_colors)
    if border_frac < 1.0:
        cf, af = _circuit_fill_color(
            circuit, f, u, v, circuit_meta, circuit_colors)
    color = cb
    alpha = ab
    if border_frac <= 0.0:
        color = cf
        alpha = af
    elif border_frac < 1.0:
        wb = border_frac * ab
        wf = (1.0 - border_frac) * af
        alpha = wb + wf
        inv = 1.0 / ti.max(alpha, 1e-6)
        color = ti.math.vec4(
            (wb * cb[0] + wf * cf[0]) * inv,
            (wb * cb[1] + wf * cf[1]) * inv,
            (wb * cb[2] + wf * cf[2]) * inv,
            border_frac * cb[3] + (1.0 - border_frac) * cf[3])
    return color, alpha


@ti.func
def _circuit_texture_alpha(circuit, f, u, v,
                           circuit_meta: ti.template(),
                           texture_colors: ti.template()) -> ti.f32:
    """Alpha-only bilinear sample from one circuit color texture grid."""
    alpha = 0.0
    tm = f % circuit_meta.shape[0]
    tc = f % texture_colors.shape[0]
    grid_w = circuit_meta[tm, circuit, _M_GRID_W]
    grid_h = circuit_meta[tm, circuit, _M_GRID_H]
    c1 = 0.5 * (u * circuit_meta[tm, circuit, _M_TEX]
                + v * circuit_meta[tm, circuit, _M_TEX + 1]) + 0.5
    c2 = 0.5 * (u * circuit_meta[tm, circuit, _M_TEX + 2]
                + v * circuit_meta[tm, circuit, _M_TEX + 3]) + 0.5
    x = ti.math.clamp(c2 * grid_h, 0.0, ti.max(grid_h - 1.0, 0.0))
    y = ti.math.clamp(c1 * grid_w, 0.0, ti.max(grid_w - 1.0, 0.0))
    num_points = texture_colors.shape[2]
    x_floor = ti.floor(x)
    y_floor = ti.floor(y)
    xr = x - x_floor
    yr = y - y_floor
    sum_w = 0.0
    for corner in ti.static(range(4)):
        cx = x_floor + (corner % 2)
        cy = y_floor + (corner // 2)
        w = (xr if (corner % 2) == 1 else 1.0 - xr) * (
            yr if (corner // 2) == 1 else 1.0 - yr)
        p = ti.cast(cx + cy * grid_h, ti.i32)
        p = ti.math.clamp(p, 0, num_points - 1)
        alpha += w * texture_colors[tc, circuit, p, 4]
        sum_w += w
    return alpha / ti.max(sum_w, 1e-6)


@ti.func
def _circuit_alpha(circuit, f, u, v, in_border,
                   circuit_meta: ti.template(),
                   circuit_colors: ti.template(),
                   circuit_border_colors: ti.template()) -> ti.f32:
    """Alpha-only variant of :func:`_sample_circuit_color` for opacity tests
    (shadow rays, stochastic transparency) that never read the RGB channels.
    """
    alpha = 0.0
    if in_border == 1:
        alpha = _circuit_texture_alpha(
            circuit, f, u, v, circuit_meta, circuit_border_colors)
    else:
        alpha = _circuit_texture_alpha(
            circuit, f, u, v, circuit_meta, circuit_colors)
    return alpha


@ti.func
def _triangle_color(f, prim, w0, w1, w2, tri_colors: ti.template()):
    """Barycentric color (RGB + glow) and alpha of a confirmed triangle hit."""
    tc = f % tri_colors.shape[0]
    color = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
    for ci in ti.static(range(4)):
        color[ci] = (w0 * tri_colors[tc, prim, 0, ci]
                     + w1 * tri_colors[tc, prim, 1, ci]
                     + w2 * tri_colors[tc, prim, 2, ci])
    alpha = (w0 * tri_colors[tc, prim, 0, 4]
             + w1 * tri_colors[tc, prim, 1, 4]
             + w2 * tri_colors[tc, prim, 2, 4])
    return color, alpha


@ti.func
def _triangle_alpha(f, prim, w0, w1, w2, tri_colors: ti.template()) -> ti.f32:
    tc = f % tri_colors.shape[0]
    return (w0 * tri_colors[tc, prim, 0, 4]
            + w1 * tri_colors[tc, prim, 1, 4]
            + w2 * tri_colors[tc, prim, 2, 4])


@ti.func
def _sample_texture(f, u, v, prim_uv_index, tri_tex_meta: ti.template(), textures: ti.template()):
    offset = tri_tex_meta[prim_uv_index, 0]
    width = tri_tex_meta[prim_uv_index, 1]
    height = tri_tex_meta[prim_uv_index, 2]

    px = u * (width - 1.0)
    py = v * (height - 1.0)

    px = ti.math.clamp(px, 0.0, ti.max(width - 1.0, 0.0))
    py = ti.math.clamp(py, 0.0, ti.max(height - 1.0, 0.0))

    x_floor = ti.floor(px)
    y_floor = ti.floor(py)
    xr = px - x_floor
    yr = py - y_floor

    color = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
    alpha = 0.0
    sum_w = 0.0

    tc = f % textures.shape[0]
    num_points = textures.shape[1]

    for corner in ti.static(range(4)):
        cx = ti.cast(x_floor + (corner % 2), ti.i32)
        cy = ti.cast(y_floor + (corner // 2), ti.i32)
        w = (xr if (corner % 2) == 1 else 1.0 - xr) * (
            yr if (corner // 2) == 1 else 1.0 - yr)

        cx = ti.math.clamp(cx, 0, ti.cast(width - 1.0, ti.i32))
        cy = ti.math.clamp(cy, 0, ti.cast(height - 1.0, ti.i32))

        local_idx = cx * ti.cast(height, ti.i32) + cy
        abs_idx = offset + local_idx
        abs_idx = ti.math.clamp(abs_idx, 0, num_points - 1)

        color += w * ti.math.vec4(textures[tc, abs_idx, 0],
                                  textures[tc, abs_idx, 1],
                                  textures[tc, abs_idx, 2],
                                  textures[tc, abs_idx, 3])
        alpha += w * textures[tc, abs_idx, 4]
        sum_w += w

    color /= ti.max(sum_w, 1e-6)
    alpha /= ti.max(sum_w, 1e-6)
    return color, alpha


@ti.func
def _flat_triangle_color(f, prim, w0, w1, w2, tri_colors: ti.template(),
                         tri_uvs: ti.template(), tri_tex_meta: ti.template(),
                         textures: ti.template(), num_colored_triangles: ti.i32):
    color = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
    alpha = 0.0
    # A "textured" triangle (prim >= num_colored_triangles) may carry only
    # material/normal maps and no color map (meta offset -1); it then falls
    # back to its per-vertex colors, which the scene merge keeps for every
    # triangle. (The meta index is clamped because `or` does not
    # short-circuit in Taichi.)
    if (prim < num_colored_triangles) or (
            tri_tex_meta[ti.max(prim - num_colored_triangles, 0), 0] < 0):
        tc = f % tri_colors.shape[0]
        for ci in ti.static(range(4)):
            color[ci] = (w0 * tri_colors[tc, prim, 0, ci]
                         + w1 * tri_colors[tc, prim, 1, ci]
                         + w2 * tri_colors[tc, prim, 2, ci])
        alpha = (w0 * tri_colors[tc, prim, 0, 4]
                 + w1 * tri_colors[tc, prim, 1, 4]
                 + w2 * tri_colors[tc, prim, 2, 4])
    else:
        prim_uv_index = prim - num_colored_triangles
        tu = f % tri_uvs.shape[0]
        u = (w0 * tri_uvs[tu, prim_uv_index, 0]
             + w1 * tri_uvs[tu, prim_uv_index, 2]
             + w2 * tri_uvs[tu, prim_uv_index, 4])
        v = (w0 * tri_uvs[tu, prim_uv_index, 1]
             + w1 * tri_uvs[tu, prim_uv_index, 3]
             + w2 * tri_uvs[tu, prim_uv_index, 5])
        color, alpha = _sample_texture(f, u, v, prim_uv_index, tri_tex_meta, textures)
    return color, alpha


@ti.func
def _flat_triangle_alpha(f, prim, w0, w1, w2, tri_colors: ti.template(),
                         tri_uvs: ti.template(), tri_tex_meta: ti.template(),
                         textures: ti.template(), num_colored_triangles: ti.i32) -> ti.f32:
    alpha = 0.0
    # Same per-vertex fallback as _flat_triangle_color for textured triangles
    # without a color map (meta offset -1).
    if (prim < num_colored_triangles) or (
            tri_tex_meta[ti.max(prim - num_colored_triangles, 0), 0] < 0):
        tc = f % tri_colors.shape[0]
        alpha = (w0 * tri_colors[tc, prim, 0, 4]
                 + w1 * tri_colors[tc, prim, 1, 4]
                 + w2 * tri_colors[tc, prim, 2, 4])
    else:
        prim_uv_index = prim - num_colored_triangles
        tu = f % tri_uvs.shape[0]
        u = (w0 * tri_uvs[tu, prim_uv_index, 0]
             + w1 * tri_uvs[tu, prim_uv_index, 2]
             + w2 * tri_uvs[tu, prim_uv_index, 4])
        v = (w0 * tri_uvs[tu, prim_uv_index, 1]
             + w1 * tri_uvs[tu, prim_uv_index, 3]
             + w2 * tri_uvs[tu, prim_uv_index, 5])

        offset = tri_tex_meta[prim_uv_index, 0]
        width = tri_tex_meta[prim_uv_index, 1]
        height = tri_tex_meta[prim_uv_index, 2]

        px = u * (width - 1.0)
        py = v * (height - 1.0)

        px = ti.math.clamp(px, 0.0, ti.max(width - 1.0, 0.0))
        py = ti.math.clamp(py, 0.0, ti.max(height - 1.0, 0.0))

        x_floor = ti.floor(px)
        y_floor = ti.floor(py)
        xr = px - x_floor
        yr = py - y_floor

        sum_w = 0.0
        tc = f % textures.shape[0]
        num_points = textures.shape[1]

        for corner in ti.static(range(4)):
            cx = ti.cast(x_floor + (corner % 2), ti.i32)
            cy = ti.cast(y_floor + (corner // 2), ti.i32)
            w = (xr if (corner % 2) == 1 else 1.0 - xr) * (
                yr if (corner // 2) == 1 else 1.0 - yr)

            cx = ti.math.clamp(cx, 0, ti.cast(width - 1.0, ti.i32))
            cy = ti.math.clamp(cy, 0, ti.cast(height - 1.0, ti.i32))

            local_idx = cx * ti.cast(height, ti.i32) + cy
            abs_idx = offset + local_idx
            abs_idx = ti.math.clamp(abs_idx, 0, num_points - 1)

            alpha += w * textures[tc, abs_idx, 4]
            sum_w += w
        alpha /= ti.max(sum_w, 1e-6)
    return alpha


@ti.func
def _triangle_extra(f, prim, w0, w1, w2, tri_extra: ti.template()):
    """Barycentric (reflectivity, roughness) of a confirmed triangle hit.
    ``tri_extra`` rows hold per-corner (reflectivity, roughness) pairs.
    """
    te = f % tri_extra.shape[0]
    reflectivity = (w0 * tri_extra[te, prim, 0]
                    + w1 * tri_extra[te, prim, 2]
                    + w2 * tri_extra[te, prim, 4])
    roughness = (w0 * tri_extra[te, prim, 1]
                 + w1 * tri_extra[te, prim, 3]
                 + w2 * tri_extra[te, prim, 5])
    return reflectivity, roughness


@ti.func
def _triangle_normal(f, prim, w0, w1, w2, tri_norm: ti.template(),
                     tri_pos: ti.template()):
    """Interpolated shading normal of a triangle hit, falling back to the
    geometric normal when the shading normals are degenerate. Only fetched
    for hits that actually scatter or reflect.
    """
    tn = f % tri_norm.shape[0]
    normal = ti.math.vec3(0.0, 0.0, 0.0)
    for ci in ti.static(range(3)):
        normal[ci] = (w0 * tri_norm[tn, prim, ci]
                      + w1 * tri_norm[tn, prim, 3 + ci]
                      + w2 * tri_norm[tn, prim, 6 + ci])
    if normal.norm() < 1e-6:
        tp = f % tri_pos.shape[0]
        v0 = ti.math.vec3(tri_pos[tp, prim, 0], tri_pos[tp, prim, 1],
                          tri_pos[tp, prim, 2])
        v1 = ti.math.vec3(tri_pos[tp, prim, 3], tri_pos[tp, prim, 4],
                          tri_pos[tp, prim, 5])
        v2 = ti.math.vec3(tri_pos[tp, prim, 6], tri_pos[tp, prim, 7],
                          tri_pos[tp, prim, 8])
        normal = (v1 - v0).cross(v2 - v0)
    return normal


@ti.func
def _bezier_normal(f, circuit, circuit_meta: ti.template()):
    tm = f % circuit_meta.shape[0]
    return ti.math.vec3(circuit_meta[tm, circuit, _M_NORMAL],
                        circuit_meta[tm, circuit, _M_NORMAL + 1],
                        circuit_meta[tm, circuit, _M_NORMAL + 2])


@ti.func
def _shade_tri_hit(frag_pipelines: ti.template(), pids_present: ti.template(),
                   f, prim, a, b, rd, pos,
                   tri_pos: ti.template(), shade_normal,
                   tri_mat_id: ti.template(), tri_mat: ti.template(),
                   light_pos: ti.template(), light_col: ti.template(),
                   num_lights, albedo, shadows: ti.template(), vis):
    """Per-fragment material shading of a confirmed flat-triangle hit: feeds the
    caller-supplied shading normal ``shade_normal``, the geometric face normal,
    the hit position and the per-primitive parameter block into
    :func:`_run_frag_pipeline`. The caller passes the *normal-mapped* shading
    normal (``_flat_triangle_normal``), so a tangent-space normal map perturbs
    the lighting -- with no map that equals the plain interpolated vertex
    normal, so unmapped surfaces are byte-identical to the previous vertex-only
    shading. ``albedo`` is the interpolated (raw) base RGB + glow;
    ``tri_mat_id``/``tri_mat`` carry the per-primitive pipeline id and parameter
    block; returns the shaded RGB + glow. ``vis`` holds the caller's per-light
    shadow visibilities (used iff ``shadows``).

    ``pos`` is the WORLD HIT POSITION, passed in rather than rebuilt here as
    ``ro + t_hit * rd``: under analytic coverage a partially covering raster
    fragment's ``t_hit`` is measured along the sample-centroid ray, not the
    pixel-centre one, so that expression does not name a point on the triangle
    (see ``raster_taichi._tri_surface_point``). Ray-traced callers pass exactly
    ``ro + t_hit * rd`` and are unchanged. ``rd`` is still the view ray and only
    supplies the view DIRECTION, whose sub-pixel error is ~0.05 degrees.

    ``pids_present`` is the compile-time bitmask of the material pipeline ids
    the batch's TRIANGLES carry (see ``shading_taichi._run_frag_pipeline``).
    """
    tp = f % tri_pos.shape[0]
    v0 = ti.math.vec3(tri_pos[tp, prim, 0], tri_pos[tp, prim, 1],
                      tri_pos[tp, prim, 2])
    v1 = ti.math.vec3(tri_pos[tp, prim, 3], tri_pos[tp, prim, 4],
                      tri_pos[tp, prim, 5])
    v2 = ti.math.vec3(tri_pos[tp, prim, 6], tri_pos[tp, prim, 7],
                      tri_pos[tp, prim, 8])
    face_n = (v1 - v0).cross(v2 - v0)
    rgb = ti.math.vec3(albedo[0], albedo[1], albedo[2])
    return _run_frag_pipeline(frag_pipelines, pids_present,
                              prim, f, pos, -rd, shade_normal,
                              face_n, rgb,
                              albedo[3], light_pos, light_col, num_lights,
                              tri_mat_id, tri_mat, shadows, vis)


@ti.func
def _nearest_surface_g(refit: ti.template(),
                     has_tri: ti.template(),
                     has_bez: ti.template(),
                     ro, rd, inv_rd, f, ff, t_prev, layer_prev,
                     t_cap,
                     pixel_size_per_t, base_dist, layer_offset_triangles,
                     t_nodes: ti.template(), t_node_miss: ti.template(),
                     t_leaf_prim: ti.template(), t_leaf_tspan: ti.template(),
                     t_first_leaf, tri_pos: ti.template(),
                     b_nodes: ti.template(), b_node_miss: ti.template(),
                     b_leaf_prim: ti.template(), b_leaf_tspan: ti.template(),
                     b_first_leaf, circuit_meta: ti.template(),
                     edges_2d: ti.template(), edge_accel: ti.template(),
                     src_sid, tri_obj: ti.template(), ident: ti.template()):
    """Nearest surface of any geometry type strictly after
    (t_prev, layer_prev) along the ray. Geometry only -- shading data is
    fetched by the caller for the hits it actually uses.

    Returns ``(found, t_hit, layer, prim, hit_type, a, b, border,
    edge_hit)`` where ``hit_type`` is 0 for bezier circuits and 1 for
    triangles, and ``(a, b)`` are the barycentric ``(w1, w2)`` for triangle
    hits or the plane ``(u, v)`` for bezier hits; ``found == 0`` means the ray
    escapes the scene, ``edge_hit == 1`` flags a triangle hit on/near one of
    its edges (used to merge the duplicate hits of mesh seams).

    ``(src_sid, tri_obj, ident)`` carry the shadow ray's source-surface
    identity for :func:`_shadow_identity_t_min`; see that function for the
    sentinel convention.
    """
    found = 0
    t_hit = 1e30
    hit_layer = -1e30
    hit_prim = -1
    hit_type = 0
    a = 0.0
    b = 0.0
    border = 0
    edge_hit = 0

    tt = 1e30
    t_prim = -1
    w1 = 0.0
    w2 = 0.0
    t_layer = -1e30
    if ti.static(has_tri != 0):
        tt, t_prim, w1, w2, t_layer = _nearest_triangle_hit(
            refit, ro, rd, inv_rd, f, ff, t_prev, layer_prev, t_cap,
            layer_offset_triangles,
            t_nodes, t_node_miss, t_leaf_prim, t_leaf_tspan, t_first_leaf,
            tri_pos, src_sid, tri_obj, ident)
    bt = 1e30
    b_circ = -1
    b_border = 0
    b_u = 0.0
    b_v = 0.0
    b_layer = -1e30
    if ti.static(has_bez != 0):
        bez_cap = t_cap
        if t_prim >= 0:
            bez_cap = ti.min(bez_cap, tt + DEPTH_TIE_EPSILON)
        bt, b_circ, b_border, b_u, b_v, b_layer = _nearest_bezier_hit(
            refit, ro, rd, inv_rd, f, ff, t_prev, layer_prev, bez_cap,
            pixel_size_per_t, base_dist, b_nodes, b_node_miss, b_leaf_prim,
            b_leaf_tspan, b_first_leaf, circuit_meta, edges_2d, edge_accel)

    if t_prim >= 0:
        found = 1
        t_hit = tt
        hit_layer = t_layer
        hit_prim = t_prim
        hit_type = 1
        a = w1
        b = w2
    if (b_circ >= 0) and ((found == 0)
                          or (not _comes_after(bt, b_layer, t_hit,
                                               hit_layer))):
        found = 1
        t_hit = bt
        hit_layer = b_layer
        hit_prim = b_circ
        hit_type = 0
        a = b_u
        b = b_v
        border = b_border
    if (found == 1) and (hit_type >= 1):
        w0 = 1.0 - a - b
        if ti.min(w0, ti.min(a, b)) < TRIANGLE_EDGE_EPSILON:
            edge_hit = 1
    return (found, t_hit, hit_layer, hit_prim, hit_type, a, b, border,
            edge_hit)


@ti.func
def _barycentric_coords(p, v0, v1, v2):
    e0 = v1 - v0
    e1 = v2 - v0
    e2 = p - v0
    d00 = e0.dot(e0)
    d01 = e0.dot(e1)
    d11 = e1.dot(e1)
    d20 = e2.dot(e0)
    d21 = e2.dot(e1)
    denom = d00 * d11 - d01 * d01
    w0 = 1.0
    w1 = 0.0
    w2 = 0.0
    if ti.abs(denom) > 1e-9:
        w1 = (d11 * d20 - d01 * d21) / denom
        w2 = (d00 * d21 - d01 * d20) / denom
        w0 = 1.0 - w1 - w2
    return w0, w1, w2


@ti.func
def agx_default_contrast_approx(x: ti.f32) -> ti.f32:
    x2 = x * x
    x4 = x2 * x2
    return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232


@ti.func
def agx_tonemap(color: ti.math.vec3) -> ti.math.vec3:
    r_rec2020 = 0.627409 * color[0] + 0.329282 * color[1] + 0.043309 * color[2]
    g_rec2020 = 0.069055 * color[0] + 0.919540 * color[1] + 0.011405 * color[2]
    b_rec2020 = 0.016390 * color[0] + 0.088013 * color[1] + 0.895597 * color[2]
    
    r_inset = 0.856627153315983 * r_rec2020 + 0.0951212405381588 * g_rec2020 + 0.0482516061458583 * b_rec2020
    g_inset = 0.137318972929847 * r_rec2020 + 0.761241990602591 * g_rec2020 + 0.101439036467562 * b_rec2020
    b_inset = 0.11189821299995 * r_rec2020 + 0.0767994186031903 * g_rec2020 + 0.811302368396859 * b_rec2020
    
    r_log = ti.math.clamp(ti.log(ti.max(r_inset, 1e-10)) / 0.6931471805599453, -12.47393, 4.026069)
    g_log = ti.math.clamp(ti.log(ti.max(g_inset, 1e-10)) / 0.6931471805599453, -12.47393, 4.026069)
    b_log = ti.math.clamp(ti.log(ti.max(b_inset, 1e-10)) / 0.6931471805599453, -12.47393, 4.026069)
    
    r_norm = (r_log - (-12.47393)) / (4.026069 - (-12.47393))
    g_norm = (g_log - (-12.47393)) / (4.026069 - (-12.47393))
    b_norm = (b_log - (-12.47393)) / (4.026069 - (-12.47393))
    
    r_curve = agx_default_contrast_approx(r_norm)
    g_curve = agx_default_contrast_approx(g_norm)
    b_curve = agx_default_contrast_approx(b_norm)
    
    r_out = 1.1271005818144368 * r_curve - 0.11060664309660323 * g_curve - 0.016493938717834573 * b_curve
    g_out = -0.1413297634984383 * r_curve + 1.157823702216272 * g_curve - 0.016493938717834257 * b_curve
    b_out = -0.14132976349843826 * r_curve - 0.11060664309660294 * g_curve + 1.2519364065950405 * b_curve
    
    r_srgb = 1.6605 * r_out - 0.1246 * g_out - 0.0182 * b_out
    g_srgb = -0.5876 * r_out + 1.1329 * g_out - 0.1006 * b_out
    b_srgb = -0.0728 * r_out - 0.0083 * g_out + 1.1187 * b_out
    
    return ti.math.clamp(ti.math.vec3(r_srgb, g_srgb, b_srgb), 0.0, 1.0)


@ti.func
def pbr_neutral_tonemap(color: ti.math.vec3) -> ti.math.vec3:
    startCompression = 0.76
    desaturation = 0.15

    x = ti.min(color[0], ti.min(color[1], color[2]))
    offset = 0.04
    if x < 0.08:
        offset = x - 6.25 * x * x
    
    color_offset = color - offset

    peak = ti.max(color_offset[0], ti.max(color_offset[1], color_offset[2]))
    out = color_offset
    if peak >= startCompression:
        d = 1.0 - startCompression
        newPeak = 1.0 - d * d / (peak + d - startCompression)
        color_offset *= newPeak / peak
        
        g = 1.0 - 1.0 / (desaturation * (peak - newPeak) + 1.0)
        out[0] = ti.math.mix(color_offset[0], newPeak, g)
        out[1] = ti.math.mix(color_offset[1], newPeak, g)
        out[2] = ti.math.mix(color_offset[2], newPeak, g)

    return ti.math.clamp(out, 0.0, 1.0)


@ti.func
def finalize_pixel_color(csum: ti.math.vec4, inv_samples: ti.f32, tonemapping: ti.template(), tonemap_exposure: ti.f32) -> ti.math.vec4:
    color_hdr = ti.math.vec3(csum[0], csum[1], csum[2]) * inv_samples
    if ti.static(tonemapping == 1):
        color_hdr = pbr_neutral_tonemap(color_hdr * (tonemap_exposure / 255.0)) * 255.0
    elif ti.static(tonemapping == 2):
        color_hdr = agx_tonemap(color_hdr * (tonemap_exposure / 255.0)) * 255.0
    elif ti.static(tonemapping == 3):
        color_hdr = ti.math.max(color_hdr, 0.0)
    else:
        color_hdr = ti.math.clamp(color_hdr, 0.0, 255.0)
    
    glow_val = csum[3] * inv_samples
    if ti.static(tonemapping == 3):
        glow_val = ti.math.max(glow_val, 0.0)
        return ti.math.vec4(color_hdr[0], color_hdr[1], color_hdr[2], glow_val)
    else:
        glow_val = ti.math.clamp(glow_val, 0.0, 255.0)
        return ti.math.clamp(
            ti.math.vec4(color_hdr[0] + 0.5, color_hdr[1] + 0.5, color_hdr[2] + 0.5, glow_val + 0.5),
            0.0, 255.0
        )


@ti.func
def _closest_point_segment_segment(ro, rd, t_max, p0, p1):
    u = rd
    v = p1 - p0
    w = ro - p0
    a = 1.0
    b = u.dot(v)
    c = v.dot(v)
    d = u.dot(w)
    e = v.dot(w)
    D = a * c - b * b
    
    s = 0.0
    t = 0.0
    if D < 1e-8:
        s = 0.0
        t = 0.0 if e < 0.0 else (1.0 if e > c else e / c)
    else:
        t = (a * e - b * d) / D
        t = ti.math.clamp(t, 0.0, 1.0)
        s = (b * t - d) / a
        s = ti.math.clamp(s, 0.0, t_max)
        t = (b * s + e) / c
        t = ti.math.clamp(t, 0.0, 1.0)
        s = (b * t - d) / a
        s = ti.math.clamp(s, 0.0, t_max)
        
    p_ray = ro + s * rd
    p_seg = p0 + t * v
    return p_ray, p_seg


@ti.func
def _distance_sq_to_point(ro, rd, t_max, p):
    t_proj = (p - ro).dot(rd)
    t_proj = ti.math.clamp(t_proj, 0.0, t_max)
    p_ray = ro + t_proj * rd
    diff = p_ray - p
    return diff.dot(diff)


@ti.func
def _nearest_surface(refit: ti.template(),
                     ro, rd, inv_rd, f, ff, t_prev, layer_prev,
                     pixel_size_per_t, base_dist, layer_offset_triangles,
                     t_nodes: ti.template(), t_node_miss: ti.template(),
                     t_leaf_prim: ti.template(), t_leaf_tspan: ti.template(),
                     t_first_leaf, tri_pos: ti.template(),
                     b_nodes: ti.template(), b_node_miss: ti.template(),
                     b_leaf_prim: ti.template(), b_leaf_tspan: ti.template(),
                     b_first_leaf, circuit_meta: ti.template(),
                     edges_2d: ti.template(), edge_accel: ti.template()):
    """All-geometry-present wrapper of :func:`_nearest_surface_g` for callers
    (Monte-Carlo path tracers + gbuffer) that don't specialize on which geometry
    types are present. Byte-identical to the pre-gating ``_nearest_surface``.

    These rays carry no source identity, so the identity-aware acceptance
    floor compiles out (sentinel ``(src_sid, ident)`` = ``(-1, 0)``; the
    forwarded ``tri_pos`` is never read).
    """
    return _nearest_surface_g(
        refit, 1, 1,
        ro, rd, inv_rd, f, ff, t_prev, layer_prev,
        1e30,
        pixel_size_per_t, base_dist, layer_offset_triangles,
        t_nodes, t_node_miss, t_leaf_prim, t_leaf_tspan, t_first_leaf, tri_pos,
        b_nodes, b_node_miss, b_leaf_prim, b_leaf_tspan, b_first_leaf,
        circuit_meta, edges_2d, edge_accel,
        -1, tri_pos, 0)


@ti.func
def _collect_hits(refit: ti.template(),
                  ro, rd, inv_rd, f, ff, t_prev, layer_prev,
                  pixel_size_per_t, base_dist, layer_offset_triangles,
                  hit_t: ti.template(), hit_layer: ti.template(),
                  hit_prim: ti.template(), hit_flags: ti.template(),
                  hit_a: ti.template(), hit_b: ti.template(),
                  t_nodes: ti.template(), t_node_miss: ti.template(),
                  t_leaf_prim: ti.template(), t_leaf_tspan: ti.template(),
                  t_first_leaf, tri_pos: ti.template(),
                  b_nodes: ti.template(), b_node_miss: ti.template(),
                  b_leaf_prim: ti.template(), b_leaf_tspan: ti.template(),
                   b_first_leaf, circuit_meta: ti.template(),
                   edges_2d: ti.template(), edge_accel: ti.template(),
                   has_tri: ti.template(),
                   has_bez: ti.template(), initial_opq_t: ti.f32,
                   initial_opq_layer: ti.f32,
                   src_sid, tri_obj: ti.template(), ident: ti.template()) -> ti.i32:
    """Gather the up-to-``KBUF`` nearest hits strictly after
    (t_prev, layer_prev) into the caller's buffers, in one traversal of each
    BVH. Triangles are traversed first; the bezier traversal then prunes
    against the hits already gathered.

    ``has_tri``/``has_bez`` flag which geometry types are present;
    a type absent from the whole batch has only a placeholder (empty) BVH, so
    its traversal is skipped outright (a launch-uniform branch, no divergence).
    ``refit != 0`` selects the refit-tree walk for BOTH trees (see
    refit_bvh.py): per-frame link-gated blocks, explicit child indices, the
    per-frame opacity flag in the leaf link word, and unused leaf-slot arrays.

    Buffers hold geometry only (the consumer fetches shading data):
    ``hit_flags`` packs the hit type (0 = bezier circuit, 1 = triangle)
    in bits 0-1, plus ``edge_hit << 2`` and ``border << 3``.
    ``(src_sid, tri_obj, ident)`` carry the shadow ray's source-surface
    identity for :func:`_shadow_identity_t_min` (triangle arm only; the
    bezier arm keeps the classic epsilon). See that function for the sentinel
    convention.
    Returns the number of hits gathered. When the return value is smaller
    than ``KBUF``, the buffer provably contains *every* remaining hit along
    the ray, so the consumer never needs another traversal.
    """
    count = 0
    worst_idx = 0
    worst_t = 1e30
    worst_layer = -1e30
    # Earliest known fully-opaque hit (leaf_tspan bit 31): everything peeling
    # after it is absorbed, so gathering and traversal can stop at its depth.
    opq_t = initial_opq_t
    opq_layer = initial_opq_layer

    # --- Triangle BVH ---
    if ti.static(has_tri != 0):
        tp = f % tri_pos.shape[0]
        t_row0 = 0
        if ti.static(refit != 0):
            t_row0 = _refit_row0(f, t_first_leaf, t_nodes)
        window_hi = worst_t + DEPTH_TIE_EPSILON if count == KBUF else 1e30
        window_hi = ti.min(window_hi, opq_t + DEPTH_TIE_EPSILON)
        g_sp = 0
        g_st = ti.Vector([0] * _GROUP_STACK)
        g_cur = 0
        g_pend, g_near = _group_test(
            refit, t_row0, 0, f, ro, inv_rd, t_prev - DEPTH_TIE_EPSILON,
            window_hi, t_nodes)
        while True:
            if g_pend == 0:
                if g_sp == 0:
                    break
                g_sp -= 1
                saved = g_st[g_sp]
                g_cur = saved >> BVH_ARITY
                saved_mask = saved & _GROUP_MASK
                window_hi = (worst_t + DEPTH_TIE_EPSILON
                             if count == KBUF else 1e30)
                window_hi = ti.min(window_hi, opq_t + DEPTH_TIE_EPSILON)
                fresh_mask, g_near = _group_test(
                    refit, t_row0, g_cur, f, ro, inv_rd,
                    t_prev - DEPTH_TIE_EPSILON, window_hi, t_nodes)
                g_pend = saved_mask & fresh_mask
            else:
                g_c = _nearest_pending_child(g_pend, g_near)
                g_pend &= ~(1 << g_c)
                descend = 0
                child_blk = 0
                l_prim = -1
                l_opq = 0
                l_base = 0
                if ti.static(refit != 0):
                    w = _refit_link(t_row0 + g_cur, g_c, t_nodes)
                    if w >= 0:
                        descend = 1
                        child_blk = w
                    else:
                        l_prim = w & _REFIT_PRIM_MASK
                        l_opq = (w >> 30) & 1
                else:
                    g_child = BVH_ARITY * g_cur + 1 + g_c
                    if g_child >= t_first_leaf:
                        l_base = (g_child - t_first_leaf) * LEAF_SIZE
                    else:
                        descend = 1
                        child_blk = g_child
                if descend == 0:
                    for j in ti.static(
                            range(1 if refit != 0 else LEAF_SIZE)):
                        prim = l_prim
                        opq = l_opq
                        if ti.static(refit == 0):
                            prim = -1
                            p0 = t_leaf_prim[l_base + j]
                            tspan = t_leaf_tspan[l_base + j]
                            if ((p0 >= 0) and ((tspan & 0xFFFF) <= f)
                                    and (f <= ((tspan >> 16) & 0x7FFF))):
                                prim = p0
                                opq = 1 if tspan < 0 else 0
                        if prim >= 0:
                            v0 = ti.math.vec3(tri_pos[tp, prim, 0],
                                              tri_pos[tp, prim, 1],
                                              tri_pos[tp, prim, 2])
                            v1 = ti.math.vec3(tri_pos[tp, prim, 3],
                                              tri_pos[tp, prim, 4],
                                              tri_pos[tp, prim, 5])
                            v2 = ti.math.vec3(tri_pos[tp, prim, 6],
                                              tri_pos[tp, prim, 7],
                                              tri_pos[tp, prim, 8])
                            hit_ok, w1, w2, t = _tri_hit(ro, rd, v0, v1, v2)
                            if hit_ok != 0:
                                layer = (layer_offset_triangles
                                         + ti.cast(prim, ti.f32))
                                accept = ((t > _shadow_identity_t_min(
                                    f, prim, src_sid, tri_obj, ident))
                                          and _comes_after(
                                              t, layer, t_prev,
                                              layer_prev)
                                          and not _comes_after(
                                              t, layer, opq_t,
                                              opq_layer))
                                if accept and (count == KBUF):
                                    accept = _comes_after(
                                        worst_t, worst_layer, t, layer)
                                if accept:
                                    slot = worst_idx
                                    if count < KBUF:
                                        slot = count
                                        count += 1
                                    hit_t[slot] = t
                                    hit_layer[slot] = layer
                                    hit_prim[slot] = prim
                                    w0 = 1.0 - w1 - w2
                                    eh = 1 if (ti.min(w0,
                                                      ti.min(w1, w2))
                                               < TRIANGLE_EDGE_EPSILON) \
                                        else 0
                                    hit_flags[slot] = 1 | (eh << 2)
                                    hit_a[slot] = w1
                                    hit_b[slot] = w2
                                    if (opq != 0) and _comes_after(
                                            opq_t, opq_layer, t, layer):
                                        opq_t = t
                                        opq_layer = layer
                                    if count == KBUF:
                                        worst_idx = 0
                                        worst_t = hit_t[0]
                                        worst_layer = hit_layer[0]
                                        for q in ti.static(
                                                range(1, KBUF)):
                                            if _comes_after(
                                                    hit_t[q],
                                                    hit_layer[q],
                                                    worst_t,
                                                    worst_layer):
                                                worst_idx = q
                                                worst_t = hit_t[q]
                                                worst_layer = \
                                                    hit_layer[q]
                else:
                    if g_pend != 0:
                        g_st[g_sp] = (g_cur << BVH_ARITY) | g_pend
                        g_sp += 1
                    g_cur = child_blk
                    window_hi = worst_t + DEPTH_TIE_EPSILON \
                        if count == KBUF else 1e30
                    window_hi = ti.min(window_hi, opq_t + DEPTH_TIE_EPSILON)
                    g_pend, g_near = _group_test(
                        refit, t_row0, g_cur, f, ro, inv_rd,
                        t_prev - DEPTH_TIE_EPSILON, window_hi, t_nodes)

    # --- Bezier BVH (window tightened by the triangle hits) ---
    if ti.static(has_bez != 0):
        num_meta_frames = circuit_meta.shape[0]
        num_edge_frames = edges_2d.shape[0]
        b_row0 = 0
        if ti.static(refit != 0):
            b_row0 = _refit_row0(f, b_first_leaf, b_nodes)
        window_hi = worst_t + DEPTH_TIE_EPSILON if count == KBUF else 1e30
        window_hi = ti.min(window_hi, opq_t + DEPTH_TIE_EPSILON)
        g_sp = 0
        g_st = ti.Vector([0] * _GROUP_STACK)
        g_cur = 0
        g_pend, g_near = _group_test(
            refit, b_row0, 0, f, ro, inv_rd, t_prev - DEPTH_TIE_EPSILON,
            window_hi, b_nodes)
        while True:
            if g_pend == 0:
                if g_sp == 0:
                    break
                g_sp -= 1
                saved = g_st[g_sp]
                g_cur = saved >> BVH_ARITY
                saved_mask = saved & _GROUP_MASK
                window_hi = (worst_t + DEPTH_TIE_EPSILON
                             if count == KBUF else 1e30)
                window_hi = ti.min(window_hi, opq_t + DEPTH_TIE_EPSILON)
                fresh_mask, g_near = _group_test(
                    refit, b_row0, g_cur, f, ro, inv_rd,
                    t_prev - DEPTH_TIE_EPSILON, window_hi, b_nodes)
                g_pend = saved_mask & fresh_mask
            else:
                g_c = _nearest_pending_child(g_pend, g_near)
                g_pend &= ~(1 << g_c)
                descend = 0
                child_blk = 0
                l_prim = -1
                l_opq = 0
                l_base = 0
                if ti.static(refit != 0):
                    w = _refit_link(b_row0 + g_cur, g_c, b_nodes)
                    if w >= 0:
                        descend = 1
                        child_blk = w
                    else:
                        l_prim = w & _REFIT_PRIM_MASK
                        l_opq = (w >> 30) & 1
                else:
                    g_child = BVH_ARITY * g_cur + 1 + g_c
                    if g_child >= b_first_leaf:
                        l_base = (g_child - b_first_leaf) * LEAF_SIZE
                    else:
                        descend = 1
                        child_blk = g_child
                if descend == 0:
                    for j in ti.static(
                            range(1 if refit != 0 else LEAF_SIZE)):
                        circuit = l_prim
                        opq = l_opq
                        if ti.static(refit == 0):
                            circuit = -1
                            p0 = b_leaf_prim[l_base + j]
                            tspan = b_leaf_tspan[l_base + j]
                            if ((p0 >= 0) and ((tspan & 0xFFFF) <= f)
                                    and (f <= ((tspan >> 16) & 0x7FFF))):
                                circuit = p0
                                opq = 1 if tspan < 0 else 0
                        if circuit >= 0:
                            tm = f % num_meta_frames
                            n = ti.math.vec3(circuit_meta[tm, circuit, _M_NORMAL],
                                             circuit_meta[tm, circuit, _M_NORMAL + 1],
                                             circuit_meta[tm, circuit, _M_NORMAL + 2])
                            denom = rd.dot(n)
                            layer = ti.cast(circuit, ti.f32)
                            if ti.abs(denom) > 1e-9:
                                center = ti.math.vec3(
                                    circuit_meta[tm, circuit, _M_CENTER],
                                    circuit_meta[tm, circuit, _M_CENTER + 1],
                                    circuit_meta[tm, circuit, _M_CENTER + 2])
                                t = (center - ro).dot(n) / denom
                                accept = ((t > MIN_HIT_DISTANCE)
                                          and _comes_after(t, layer, t_prev,
                                                           layer_prev)
                                          and not _comes_after(
                                              t, layer, opq_t, opq_layer))
                                if accept and (count == KBUF):
                                    accept = _comes_after(worst_t, worst_layer,
                                                          t, layer)
                                if accept:
                                    hit = ro + t * rd - center
                                    bu = ti.math.vec3(
                                        circuit_meta[tm, circuit, _M_BASIS_U],
                                        circuit_meta[tm, circuit, _M_BASIS_U + 1],
                                        circuit_meta[tm, circuit, _M_BASIS_U + 2])
                                    bv = ti.math.vec3(
                                        circuit_meta[tm, circuit, _M_BASIS_V],
                                        circuit_meta[tm, circuit, _M_BASIS_V + 1],
                                        circuit_meta[tm, circuit, _M_BASIS_V + 2])
                                    u = hit.dot(bu)
                                    v = hit.dot(bv)

                                    pixel_size = pixel_size_per_t * (base_dist + t)
                                    border_w = (circuit_meta[tm, circuit, _M_BORDER_W]
                                                * pixel_size)
                                    outline_w = 0.6 * pixel_size
                                    filled = circuit_meta[tm, circuit, _M_FILLED] > 0.5
                                    query_radius = _circuit_query_radius(
                                        border_w, outline_w, filled)
                                    te = f % num_edge_frames
                                    (crossings, min_dist_sq, _ccu, _ccv, _e1x,
                                     _e1y, _sg1, _s2, _s2u, _s2v, _e2x, _e2y,
                                     _sg2) = _bezier_point_metrics(
                                        circuit, te, u, v, query_radius,
                                        circuit_meta.shape[1], edges_2d, edge_accel)
                                    inside, in_border = _circuit_point_region(
                                        border_w, outline_w, filled, crossings,
                                        min_dist_sq)
                                    if inside:
                                        slot = worst_idx
                                        if count < KBUF:
                                            slot = count
                                            count += 1
                                        hit_t[slot] = t
                                        hit_layer[slot] = layer
                                        hit_prim[slot] = circuit
                                        hit_flags[slot] = (
                                            (1 if in_border else 0) << 3)
                                        hit_a[slot] = u
                                        hit_b[slot] = v
                                        if (opq != 0) and _comes_after(
                                                opq_t, opq_layer, t, layer):
                                            opq_t = t
                                            opq_layer = layer
                                        if count == KBUF:
                                            worst_idx = 0
                                            worst_t = hit_t[0]
                                            worst_layer = hit_layer[0]
                                            for q in ti.static(range(1, KBUF)):
                                                if _comes_after(hit_t[q],
                                                                hit_layer[q],
                                                                worst_t,
                                                                worst_layer):
                                                    worst_idx = q
                                                    worst_t = hit_t[q]
                                                    worst_layer = hit_layer[q]
                else:
                    if g_pend != 0:
                        g_st[g_sp] = (g_cur << BVH_ARITY) | g_pend
                        g_sp += 1
                    g_cur = child_blk
                    window_hi = worst_t + DEPTH_TIE_EPSILON \
                        if count == KBUF else 1e30
                    window_hi = ti.min(window_hi, opq_t + DEPTH_TIE_EPSILON)
                    g_pend, g_near = _group_test(
                        refit, b_row0, g_cur, f, ro, inv_rd,
                        t_prev - DEPTH_TIE_EPSILON, window_hi, b_nodes)
    return count


@ti.func
def _anyhit_opaque_tri(refit: ti.template(), ro, rd, inv_rd, f, t_lo, max_t,
                       nodes: ti.template(), leaf_prim: ti.template(),
                       leaf_tspan: ti.template(), first_leaf,
                       tri_pos: ti.template(),
                       src_sid, tri_obj: ti.template(),
                       ident: ti.template()) -> ti.i32:
    """1 if ANY interval-opaque triangle (classic ``leaf_tspan`` bit 31 /
    refit link bit 30) is hit with the identity-aware floor (see
    :func:`_shadow_identity_t_min`) ``< t < max_t``.

    Any-hit: no ordering state, near-first descent, exits on the first
    accepted hit. Non-opaque leaves are skipped before intersection, so the
    walk is cheap even when the scene stacks translucent surfaces. The
    intersection predicate is exactly the nearest-walk's, so an accepted hit
    here is one the ordered shadow march would (eventually) consume.

    ``t_lo`` is purely a traversal-pruning hint (the node-visit window's
    lower edge) -- acceptance deliberately stays ``(floor, max_t)``, so a
    caller that already marched a prefix of the ray can pass the marched
    depth without changing which answer is correct.
    """
    hit = 0
    tp = f % tri_pos.shape[0]
    row0 = 0
    if ti.static(refit != 0):
        row0 = _refit_row0(f, first_leaf, nodes)
    g_sp = 0
    g_st = ti.Vector([0] * _GROUP_STACK)
    g_cur = 0
    g_pend, g_near = _group_test(
        refit, row0, 0, f, ro, inv_rd, t_lo,
        max_t + DEPTH_TIE_EPSILON, nodes)
    while hit == 0:
        if g_pend == 0:
            if g_sp == 0:
                break
            g_sp -= 1
            saved = g_st[g_sp]
            g_cur = saved >> BVH_ARITY
            saved_mask = saved & _GROUP_MASK
            fresh_mask, g_near = _group_test(
                refit, row0, g_cur, f, ro, inv_rd, t_lo,
                max_t + DEPTH_TIE_EPSILON, nodes)
            g_pend = saved_mask & fresh_mask
        else:
            g_c = _nearest_pending_child(g_pend, g_near)
            g_pend &= ~(1 << g_c)
            descend = 0
            child_blk = 0
            l_prim = -1
            l_base = 0
            if ti.static(refit != 0):
                w = _refit_link(row0 + g_cur, g_c, nodes)
                if w >= 0:
                    descend = 1
                    child_blk = w
                elif ((w >> 30) & 1) != 0:
                    l_prim = w & _REFIT_PRIM_MASK
            else:
                g_child = BVH_ARITY * g_cur + 1 + g_c
                if g_child >= first_leaf:
                    l_base = (g_child - first_leaf) * LEAF_SIZE
                else:
                    descend = 1
                    child_blk = g_child
            if descend == 0:
                for j in ti.static(range(1 if refit != 0 else LEAF_SIZE)):
                    prim = l_prim
                    if ti.static(refit == 0):
                        prim = -1
                        p0 = leaf_prim[l_base + j]
                        tspan = leaf_tspan[l_base + j]
                        # Bit 31 (sign) flags interval-opaque instances.
                        if ((p0 >= 0) and (tspan < 0)
                                and ((tspan & 0xFFFF) <= f)
                                and (f <= ((tspan >> 16) & 0x7FFF))):
                            prim = p0
                    if (hit == 0) and (prim >= 0):
                        v0 = ti.math.vec3(tri_pos[tp, prim, 0],
                                          tri_pos[tp, prim, 1],
                                          tri_pos[tp, prim, 2])
                        v1 = ti.math.vec3(tri_pos[tp, prim, 3],
                                          tri_pos[tp, prim, 4],
                                          tri_pos[tp, prim, 5])
                        v2 = ti.math.vec3(tri_pos[tp, prim, 6],
                                          tri_pos[tp, prim, 7],
                                          tri_pos[tp, prim, 8])
                        hit_ok, w1, w2, t = _tri_hit(ro, rd, v0, v1, v2)
                        if hit_ok != 0:
                            if (t > _shadow_identity_t_min(
                                    f, prim, src_sid, tri_obj, ident)) \
                                    and (t < max_t):
                                hit = 1
            else:
                if g_pend != 0:
                    g_st[g_sp] = (g_cur << BVH_ARITY) | g_pend
                    g_sp += 1
                g_cur = child_blk
                g_pend, g_near = _group_test(
                    refit, row0, g_cur, f, ro, inv_rd, t_lo,
                    max_t + DEPTH_TIE_EPSILON, nodes)
    return hit


@ti.func
def _anyhit_opaque_bez(refit: ti.template(), ro, rd, inv_rd, f, t_lo, max_t,
                       pixel_size_per_t, base_dist,
                       nodes: ti.template(), leaf_prim: ti.template(),
                       leaf_tspan: ti.template(), first_leaf,
                       circuit_meta: ti.template(),
                       edges_2d: ti.template(),
                       edge_accel: ti.template()) -> ti.i32:
    """Bezier-circuit arm of the opaque any-hit walk (see
    :func:`_anyhit_opaque_tri`). A circuit's opaque flag already requires the
    fill AND (when shown) the border to be fully opaque, so any drawn-region
    hit on a flagged circuit absorbs the ray.
    """
    hit = 0
    num_meta_frames = circuit_meta.shape[0]
    num_edge_frames = edges_2d.shape[0]
    row0 = 0
    if ti.static(refit != 0):
        row0 = _refit_row0(f, first_leaf, nodes)
    g_sp = 0
    g_st = ti.Vector([0] * _GROUP_STACK)
    g_cur = 0
    g_pend, g_near = _group_test(
        refit, row0, 0, f, ro, inv_rd, t_lo,
        max_t + DEPTH_TIE_EPSILON, nodes)
    while hit == 0:
        if g_pend == 0:
            if g_sp == 0:
                break
            g_sp -= 1
            saved = g_st[g_sp]
            g_cur = saved >> BVH_ARITY
            saved_mask = saved & _GROUP_MASK
            fresh_mask, g_near = _group_test(
                refit, row0, g_cur, f, ro, inv_rd, t_lo,
                max_t + DEPTH_TIE_EPSILON, nodes)
            g_pend = saved_mask & fresh_mask
        else:
            g_c = _nearest_pending_child(g_pend, g_near)
            g_pend &= ~(1 << g_c)
            descend = 0
            child_blk = 0
            l_prim = -1
            l_base = 0
            if ti.static(refit != 0):
                w = _refit_link(row0 + g_cur, g_c, nodes)
                if w >= 0:
                    descend = 1
                    child_blk = w
                elif ((w >> 30) & 1) != 0:
                    l_prim = w & _REFIT_PRIM_MASK
            else:
                g_child = BVH_ARITY * g_cur + 1 + g_c
                if g_child >= first_leaf:
                    l_base = (g_child - first_leaf) * LEAF_SIZE
                else:
                    descend = 1
                    child_blk = g_child
            if descend == 0:
                for j in ti.static(range(1 if refit != 0 else LEAF_SIZE)):
                    circuit = l_prim
                    if ti.static(refit == 0):
                        circuit = -1
                        p0 = leaf_prim[l_base + j]
                        tspan = leaf_tspan[l_base + j]
                        if ((p0 >= 0) and (tspan < 0)
                                and ((tspan & 0xFFFF) <= f)
                                and (f <= ((tspan >> 16) & 0x7FFF))):
                            circuit = p0
                    if (hit == 0) and (circuit >= 0):
                        tm = f % num_meta_frames
                        n = ti.math.vec3(
                            circuit_meta[tm, circuit, _M_NORMAL],
                            circuit_meta[tm, circuit, _M_NORMAL + 1],
                            circuit_meta[tm, circuit, _M_NORMAL + 2])
                        denom = rd.dot(n)
                        if ti.abs(denom) > 1e-9:
                            center = ti.math.vec3(
                                circuit_meta[tm, circuit, _M_CENTER],
                                circuit_meta[tm, circuit, _M_CENTER + 1],
                                circuit_meta[tm, circuit, _M_CENTER + 2])
                            t = (center - ro).dot(n) / denom
                            if (t > MIN_HIT_DISTANCE) and (t < max_t):
                                hp = ro + t * rd - center
                                bu = ti.math.vec3(
                                    circuit_meta[tm, circuit, _M_BASIS_U],
                                    circuit_meta[tm, circuit, _M_BASIS_U + 1],
                                    circuit_meta[tm, circuit, _M_BASIS_U + 2])
                                bv = ti.math.vec3(
                                    circuit_meta[tm, circuit, _M_BASIS_V],
                                    circuit_meta[tm, circuit, _M_BASIS_V + 1],
                                    circuit_meta[tm, circuit, _M_BASIS_V + 2])
                                u = hp.dot(bu)
                                v = hp.dot(bv)
                                pixel_size = pixel_size_per_t * (base_dist + t)
                                border_w = (
                                    circuit_meta[tm, circuit, _M_BORDER_W]
                                    * pixel_size)
                                outline_w = 0.6 * pixel_size
                                filled = (circuit_meta[tm, circuit, _M_FILLED]
                                          > 0.5)
                                query_radius = _circuit_query_radius(
                                    border_w, outline_w, filled)
                                te = f % num_edge_frames
                                (crossings, min_dist_sq, _ccu, _ccv, _e1x,
                                     _e1y, _sg1, _s2, _s2u, _s2v, _e2x, _e2y,
                                     _sg2) = _bezier_point_metrics(
                                    circuit, te, u, v, query_radius,
                                    circuit_meta.shape[1], edges_2d,
                                    edge_accel)
                                inside, in_border = _circuit_point_region(
                                    border_w, outline_w, filled, crossings,
                                    min_dist_sq)
                                if inside:
                                    hit = 1
            else:
                if g_pend != 0:
                    g_st[g_sp] = (g_cur << BVH_ARITY) | g_pend
                    g_sp += 1
                g_cur = child_blk
                g_pend, g_near = _group_test(
                    refit, row0, g_cur, f, ro, inv_rd, t_lo,
                    max_t + DEPTH_TIE_EPSILON, nodes)
    return hit


@ti.func
def _shadow_anyhit_opaque(refit: ti.template(),
                          has_tri: ti.template(),
                          has_bez: ti.template(),
                          ro, rd, inv_rd, f, t_lo, max_t,
                          pixel_size_per_t, base_dist,
                          t_nodes: ti.template(), t_leaf_prim: ti.template(),
                          t_leaf_tspan: ti.template(), t_first_leaf,
                          tri_pos: ti.template(),
                          b_nodes: ti.template(), b_leaf_prim: ti.template(),
                          b_leaf_tspan: ti.template(), b_first_leaf,
                          circuit_meta: ti.template(),
                          edges_2d: ti.template(),
                          edge_accel: ti.template(),
                          src_sid, tri_obj: ti.template(),
                          ident: ti.template()) -> ti.i32:
    """1 if any interval-opaque primitive of any geometry type blocks the
    shadow ray before ``max_t``. Trees are tried triangle -> bezier, the
    second skipped entirely on a hit in the first. ``t_lo`` prunes the
    node-visit window only (see :func:`_anyhit_opaque_tri`).

    ``(src_sid, tri_obj, ident)`` carry the shadow ray's source-surface
    identity into the triangle arm; a circuit blocker keeps the classic
    epsilon (circuits have no per-triangle identity).
    """
    hit = 0
    if ti.static(has_tri != 0):
        hit = _anyhit_opaque_tri(refit, ro, rd, inv_rd, f, t_lo, max_t,
                                 t_nodes, t_leaf_prim, t_leaf_tspan,
                                 t_first_leaf, tri_pos,
                                 src_sid, tri_obj, ident)
    if ti.static(has_bez != 0):
        if hit == 0:
            hit = _anyhit_opaque_bez(refit, ro, rd, inv_rd, f, t_lo, max_t,
                                     pixel_size_per_t, base_dist,
                                     b_nodes, b_leaf_prim, b_leaf_tspan,
                                     b_first_leaf, circuit_meta, edges_2d,
                                     edge_accel)
    return hit


@ti.func
def _shadow_occluded(refit: ti.template(), anyhit: ti.template(),
                     ro, rd, f, ff, max_t,
                     pixel_size_per_t, base_dist, layer_offset_triangles,
                     has_tri: ti.template(),
                     has_bez: ti.template(),
                     t_nodes: ti.template(), t_node_miss: ti.template(),
                     t_leaf_prim: ti.template(), t_leaf_tspan: ti.template(),
                     t_first_leaf, tri_pos: ti.template(),
                     tri_colors: ti.template(), tri_uvs: ti.template(),
                     tri_tex_meta: ti.template(), textures: ti.template(),
                     num_colored_triangles: ti.i32,
                     b_nodes: ti.template(), b_node_miss: ti.template(),
                     b_leaf_prim: ti.template(), b_leaf_tspan: ti.template(),
                     b_first_leaf, circuit_meta: ti.template(),
                     circuit_colors: ti.template(),
                     circuit_border_colors: ti.template(),
                     edges_2d: ti.template(), edge_accel: ti.template(),
                     src_sid, tri_obj: ti.template(), ident: ti.template()):
    """Fraction of light occluded along a deterministic shadow ray.

    Every surface between the shaded point and the light attenuates the
    remaining light by its opacity, matching the physical path tracer's
    transmittance calculation. A fully opaque hit exits immediately. Mesh
    seams still merge their duplicate edge hit so a thin surface cannot
    attenuate twice along a shared edge.

    ``(src_sid, tri_obj, ident)`` carry the shadow ray's source-surface
    identity for :func:`_shadow_identity_t_min` (see it for the sentinel
    convention); with ``ident == 0`` every acceptance test is exactly the
    pre-identity one.

    ``anyhit`` (compile-time, from the host's shadow mode) engages the
    opaque any-hit early-out: 3 (chosen when the batch provably contains no
    translucent geometry, so any march hit would be fully opaque and a miss
    proves the ray lit) runs ONLY :func:`_shadow_anyhit_opaque` -- the march
    is compiled out entirely. 2 (mixed batches) marches normally and, after
    the FIRST partially transparent surface, spends one any-hit walk over
    the remaining range: an opaque blocker further along forces the final
    occlusion to exactly 1.0 no matter what lies between, so a hit retires
    the ray without peeling the translucent stack, while lit rays and rays
    blocked by their first surface never pay for the walk at all. Not
    strictly byte-identical to the plain march in two corner cases the
    early-out deliberately overrules: an opaque edge hit the seam merge
    would have folded into an earlier translucent edge hit within
    ``DEPTH_TIE_EPSILON``, and an opaque blocker past
    ``MAX_SURFACES_PER_RAY`` peeled surfaces -- in both the any-hit's full
    occlusion is the physically correct answer. 4 replaces the march with
    :func:`_shadow_gather_occluded`, the same peel rebuilt on the KBUF
    gather (one traversal per KBUF surfaces instead of per surface).
    """
    inv_rd = ti.math.vec3(_safe_inverse(rd[0]), _safe_inverse(rd[1]),
                          _safe_inverse(rd[2]))
    occluded = 0.0
    if ti.static(anyhit == 3):
        occluded = ti.cast(
            _shadow_anyhit_opaque(
                refit, has_tri, has_bez, ro, rd, inv_rd, f,
                -DEPTH_TIE_EPSILON, max_t,
                pixel_size_per_t, base_dist,
                t_nodes, t_leaf_prim, t_leaf_tspan, t_first_leaf, tri_pos,
                b_nodes, b_leaf_prim, b_leaf_tspan, b_first_leaf,
                circuit_meta, edges_2d, edge_accel,
                src_sid, tri_obj, ident),
            ti.f32)
    else:
        if ti.static(anyhit == 4):
            occluded = _shadow_gather_occluded(
                refit, ro, rd, inv_rd, f, ff, max_t,
                pixel_size_per_t, base_dist, layer_offset_triangles,
                has_tri, has_bez,
                t_nodes, t_node_miss, t_leaf_prim, t_leaf_tspan,
                t_first_leaf, tri_pos, tri_colors, tri_uvs, tri_tex_meta,
                textures, num_colored_triangles,
                b_nodes, b_node_miss, b_leaf_prim, b_leaf_tspan,
                b_first_leaf, circuit_meta, circuit_colors,
                circuit_border_colors, edges_2d, edge_accel,
                src_sid, tri_obj, ident)
        else:
            occluded = _shadow_march_occluded(
                refit, anyhit, ro, rd, inv_rd, f, ff, max_t,
                pixel_size_per_t, base_dist, layer_offset_triangles,
                has_tri, has_bez,
                t_nodes, t_node_miss, t_leaf_prim, t_leaf_tspan,
                t_first_leaf, tri_pos, tri_colors, tri_uvs, tri_tex_meta,
                textures, num_colored_triangles,
                b_nodes, b_node_miss, b_leaf_prim, b_leaf_tspan,
                b_first_leaf, circuit_meta, circuit_colors,
                circuit_border_colors, edges_2d, edge_accel,
                src_sid, tri_obj, ident)
    return occluded


@ti.func
def _shadow_march_occluded(refit: ti.template(), anyhit: ti.template(),
                           ro, rd, inv_rd, f, ff,
                           max_t,
                           pixel_size_per_t, base_dist,
                           layer_offset_triangles,
                           has_tri: ti.template(),
                           has_bez: ti.template(),
                           t_nodes: ti.template(), t_node_miss: ti.template(),
                           t_leaf_prim: ti.template(),
                           t_leaf_tspan: ti.template(),
                           t_first_leaf, tri_pos: ti.template(),
                           tri_colors: ti.template(), tri_uvs: ti.template(),
                           tri_tex_meta: ti.template(),
                           textures: ti.template(),
                           num_colored_triangles: ti.i32,
                           b_nodes: ti.template(), b_node_miss: ti.template(),
                           b_leaf_prim: ti.template(),
                           b_leaf_tspan: ti.template(),
                           b_first_leaf, circuit_meta: ti.template(),
                           circuit_colors: ti.template(),
                           circuit_border_colors: ti.template(),
                           edges_2d: ti.template(),
                           edge_accel: ti.template(),
                           src_sid, tri_obj: ti.template(),
                           ident: ti.template()):
    """The classic ordered closest-hit shadow march (the pre-any-hit body of
    :func:`_shadow_occluded`, byte-identical at ``anyhit`` 0/1; 2 adds the
    deferred opaque any-hit early-out documented there).
    """
    transmitted = 1.0
    t_prev = 0.0
    layer_prev = 1e30
    seam_t = -1e30
    step = 0
    behind_checked = 0
    while step < MAX_SURFACES_PER_RAY:
        step += 1
        # Cap the walk at the light: t_cap only tightens the node-visit
        # window to min(best_t, t_cap) + DEPTH_TIE_EPSILON (hit acceptance is
        # not capped), so every subtree beyond the light is pruned while any
        # hit the t_hit >= max_t break below would have consumed is still
        # found -- candidates that differ between capped and uncapped walks
        # all lie beyond max_t, where the caller breaks either way.
        # Byte-identical; directional lights pass max_t = 1e7 and lose only
        # the (empty) beyond-horizon descent.
        (found, t_hit, hit_layer, prim, hit_type, a, b, border,
         edge_hit) = _nearest_surface_g(
            refit, has_tri, has_bez,
            ro, rd, inv_rd, f, ff, t_prev, layer_prev,
            max_t,
            pixel_size_per_t, base_dist, layer_offset_triangles,
            t_nodes, t_node_miss, t_leaf_prim, t_leaf_tspan, t_first_leaf,
            tri_pos,
            b_nodes, b_node_miss, b_leaf_prim, b_leaf_tspan, b_first_leaf,
            circuit_meta, edges_2d, edge_accel,
            src_sid, tri_obj, ident)
        if (found == 0) or (t_hit >= max_t):
            break
        seam_eps = DEPTH_TIE_EPSILON
        if (edge_hit == 1) and (t_hit - seam_t <= seam_eps):
            t_prev = t_hit
            layer_prev = hit_layer
            continue
        seam_t = t_hit if edge_hit == 1 else -1e30
        alpha = 0.0
        if hit_type == 1:
            alpha = _flat_triangle_alpha(f, prim, 1.0 - a - b, a, b, tri_colors,
                                         tri_uvs, tri_tex_meta, textures, num_colored_triangles)
        else:
            alpha = _circuit_alpha(prim, f, a, b, border, circuit_meta,
                                   circuit_colors, circuit_border_colors)
        alpha = ti.math.clamp(alpha, 0.0, 1.0)
        transmitted *= 1.0 - alpha
        if alpha >= 1.0:
            break
        if ti.static(anyhit == 2):
            # Deferred opaque any-hit: the ray just peeled a partially
            # transparent surface, committing the march to the whole
            # translucent stack. One unordered walk over the remaining
            # range answers "is any opaque blocker back there" -- a hit
            # forces the final occlusion to exactly 1.0 (everything in
            # between only multiplies onto a factor that ends at zero) and
            # retires the ray; a miss proves the stack all-translucent.
            # Runs at most once per ray, and never for rays that are lit
            # or blocked by their first surface. ``t_hit`` only prunes the
            # walk's node window; acceptance is unchanged, so coincident
            # same-depth-bin opaque hits ordered after this one by layer
            # are still found.
            if behind_checked == 0:
                behind_checked = 1
                if _shadow_anyhit_opaque(
                        refit, has_tri, has_bez, ro, rd, inv_rd, f,
                        t_hit - DEPTH_TIE_EPSILON, max_t,
                        pixel_size_per_t, base_dist,
                        t_nodes, t_leaf_prim, t_leaf_tspan, t_first_leaf,
                        tri_pos,
                        b_nodes, b_leaf_prim, b_leaf_tspan, b_first_leaf,
                        circuit_meta, edges_2d, edge_accel,
                        src_sid, tri_obj, ident) == 1:
                    transmitted = 0.0
                    break
        t_prev = t_hit
        layer_prev = hit_layer
    return 1.0 - transmitted


@ti.func
def _shadow_gather_occluded(refit: ti.template(),
                            ro, rd, inv_rd, f, ff,
                            max_t,
                            pixel_size_per_t, base_dist,
                            layer_offset_triangles,
                            has_tri: ti.template(),
                            has_bez: ti.template(),
                            t_nodes: ti.template(),
                            t_node_miss: ti.template(),
                            t_leaf_prim: ti.template(),
                            t_leaf_tspan: ti.template(),
                            t_first_leaf, tri_pos: ti.template(),
                            tri_colors: ti.template(),
                            tri_uvs: ti.template(),
                            tri_tex_meta: ti.template(),
                            textures: ti.template(),
                            num_colored_triangles: ti.i32,
                            b_nodes: ti.template(),
                            b_node_miss: ti.template(),
                            b_leaf_prim: ti.template(),
                            b_leaf_tspan: ti.template(),
                            b_first_leaf, circuit_meta: ti.template(),
                            circuit_colors: ti.template(),
                            circuit_border_colors: ti.template(),
                            edges_2d: ti.template(),
                            edge_accel: ti.template(),
                            src_sid, tri_obj: ti.template(),
                            ident: ti.template()):
    """The ordered shadow march rebuilt on the KBUF gather (shadow mode 4).

    Where :func:`_shadow_march_occluded` restarts a full three-tree
    traversal per peeled surface, each traversal here gathers the up-to-
    ``KBUF`` nearest hits with :func:`_collect_hits` and drains them in the
    same transitive :func:`_comes_after` order the march peels in, with the
    identical seam merge, alpha accumulation and early exits. A k-surface
    translucent stack therefore costs ``ceil((k+1)/KBUF)`` traversals
    instead of ``k+1``, while an all-opaque blocked ray stays at one (its
    first buffer opens with an interval-opaque hit whose alpha is 1).

    The light cap rides in as the gather's initial opaque window
    (``initial_opq_t = max_t``): node-visit windows close at ``max_t`` +
    ``DEPTH_TIE_EPSILON`` exactly like the march's ``t_cap``, and
    acceptance drops precisely the hits whose depth bin lies beyond the
    light's -- hits the march breaks on before applying any of them. Hits
    inside the light's own depth bin are still gathered and terminate the
    drain through the same ``t >= max_t`` test as the march.

    Divergence from the march is confined to a corner shared with the
    camera peel (which composites from the same gather + opaque window): a
    surface behind an interval-opaque edge hit that the drain seam-merges
    into an earlier coincident edge hit was pruned by the opaque window,
    where the march would have peeled on through the merged surface.
    """
    transmitted = 1.0
    t_prev = 0.0
    layer_prev = 1e30
    seam_t = -1e30
    step = 0
    alive = 1
    while (alive == 1) and (step < MAX_SURFACES_PER_RAY):
        kb_t = ti.Vector([0.0] * KBUF)
        kb_layer = ti.Vector([0.0] * KBUF)
        kb_prim = ti.Vector([0] * KBUF)
        kb_flags = ti.Vector([0] * KBUF)
        kb_a = ti.Vector([0.0] * KBUF)
        kb_b = ti.Vector([0.0] * KBUF)
        num_hits = _collect_hits(
            refit, ro, rd, inv_rd, f, ff, t_prev, layer_prev,
            pixel_size_per_t, base_dist, layer_offset_triangles,
            kb_t, kb_layer, kb_prim, kb_flags, kb_a, kb_b,
            t_nodes, t_node_miss, t_leaf_prim, t_leaf_tspan, t_first_leaf,
            tri_pos,
            b_nodes, b_node_miss, b_leaf_prim, b_leaf_tspan, b_first_leaf,
            circuit_meta, edges_2d, edge_accel, has_tri, has_bez,
            max_t, -1e30,
            src_sid, tri_obj, ident)
        if num_hits == 0:
            alive = 0
        drained = 0
        while (alive == 1) and (drained < num_hits) \
                and (step < MAX_SURFACES_PER_RAY):
            step += 1
            # Nearest unconsumed slot, scalar-tracked with ti.static
            # selects so the kb_* vectors are never dynamically indexed
            # (a dynamic vector index spills the whole vector to local
            # memory -- see the wavefront_shade drain).
            sel = 0
            sel_found = 0
            t_hit = 0.0
            hit_layer = 0.0
            for q in ti.static(range(KBUF)):
                if (q < num_hits) and (kb_prim[q] >= 0):
                    if sel_found == 0:
                        sel = q
                        t_hit = kb_t[q]
                        hit_layer = kb_layer[q]
                        sel_found = 1
                    elif _comes_after(t_hit, hit_layer,
                                      kb_t[q], kb_layer[q]):
                        sel = q
                        t_hit = kb_t[q]
                        hit_layer = kb_layer[q]
            prim = 0
            flags = 0
            a = 0.0
            b = 0.0
            for q in ti.static(range(KBUF)):
                if q == sel:
                    prim = kb_prim[q]
                    flags = kb_flags[q]
                    a = kb_a[q]
                    b = kb_b[q]
                    kb_prim[q] = -1
            drained += 1
            if t_hit >= max_t:
                alive = 0
            else:
                hit_type = flags & 3
                edge_hit = (flags >> 2) & 1
                border = (flags >> 3) & 1
                seam_eps = DEPTH_TIE_EPSILON
                if (edge_hit == 1) and (t_hit - seam_t <= seam_eps):
                    t_prev = t_hit
                    layer_prev = hit_layer
                else:
                    seam_t = t_hit if edge_hit == 1 else -1e30
                    alpha = 0.0
                    if hit_type == 1:
                        alpha = _flat_triangle_alpha(
                            f, prim, 1.0 - a - b, a, b, tri_colors, tri_uvs,
                            tri_tex_meta, textures, num_colored_triangles)
                    else:
                        alpha = _circuit_alpha(prim, f, a, b, border,
                                               circuit_meta, circuit_colors,
                                               circuit_border_colors)
                    alpha = ti.math.clamp(alpha, 0.0, 1.0)
                    transmitted *= 1.0 - alpha
                    if alpha >= 1.0:
                        alive = 0
                    else:
                        t_prev = t_hit
                        layer_prev = hit_layer
        # A short buffer proves every remaining hit inside the light's
        # depth window was gathered and drained; the march's next step
        # would find nothing (or only beyond-light hits it breaks on).
        if (alive == 1) and (num_hits < KBUF):
            alive = 0
    return 1.0 - transmitted


@ti.kernel
def path_trace_scene_stbvh(
        # Classic STBVH vs refit-tree walk (compile-time; see refit_bvh.py).
        refit: ti.template(),
        # Triangle STBVH + packed geometry.
        t_nodes: NODE_ARG, t_node_miss: ti.types.ndarray(),
        t_leaf_prim: ti.types.ndarray(), t_leaf_tspan: ti.types.ndarray(),
        t_first_leaf: int,
        tri_pos: ti.types.ndarray(), tri_norm: ti.types.ndarray(),
        tri_extra: ti.types.ndarray(), tri_colors: ti.types.ndarray(),
        tri_uvs: ti.types.ndarray(), tri_tex_meta: ti.types.ndarray(),
        textures: ti.types.ndarray(), num_colored_triangles: ti.i32,
        # Bezier STBVH + packed geometry.
        b_nodes: NODE_ARG, b_node_miss: ti.types.ndarray(),
        b_leaf_prim: ti.types.ndarray(), b_leaf_tspan: ti.types.ndarray(),
        b_first_leaf: int,
        circuit_meta: ti.types.ndarray(), circuit_colors: ti.types.ndarray(),
        circuit_border_colors: ti.types.ndarray(),
        edges_2d: ti.types.ndarray(), edge_accel: ti.types.ndarray(),
        # Per-frame camera and pixel scale.
        cam_origin: ti.types.ndarray(), screen_point: ti.types.ndarray(),
        pixel_basis_x: ti.types.ndarray(), pixel_basis_y: ti.types.ndarray(),
        pixel_world_scale: ti.types.ndarray(),
        # Render parameters.
        time_start: int, time_end: int, width: int, height: int,
        half_screen_w: float, half_screen_h: float,
        layer_offset_triangles: float,
        max_bounces: int, transparent: int,
        samples_per_pixel: int, indirect_strength: float,
        # Background buffer [time_end - time_start, width * height,
        # channels] (u8), read by paths that escape the scene.
        out: ti.types.ndarray(),
        # Per-pixel sample accumulator [time_end - time_start,
        # width * height, 5] (f32, zero-filled by the caller); converted to
        # u8 means by ``finalize_samples``.
        accum: ti.types.ndarray()):
    """Monte Carlo estimator of the same light transport as the deterministic
    wavefront tracer, generalized with random scattering.

    The parallel loop is flattened over (frame, pixel, sample): one thread
    traces one *path*, so the GPU stays saturated even for single-frame
    renders at high sample counts, and a pixel's samples occupy adjacent
    threads (their nearly identical primary rays keep warps coherent).
    Contributions are accumulated atomically into ``accum`` and averaged by
    ``finalize_samples``. At every surface a path makes stochastic decisions
    instead of deterministic splits:

    * with probability ``1 - alpha`` it passes straight through (stochastic
      transparency -- the expectation equals alpha blending);
    * otherwise, with probability ``reflectivity`` it reflects specularly,
      jittered into a glossy lobe of the surface's ``roughness``;
    * otherwise it is a diffuse interaction: the surface's (vertex-shaded)
      color is emitted into the sample and, when ``indirect_strength > 0``,
      the path continues in a cosine-weighted random hemisphere direction
      with throughput scaled by ``albedo * indirect_strength`` (color
      bleeding / one-bounce-per-step global illumination).

    Paths that escape the scene pick up the background.
    """
    pixels_per_frame = width * height
    paths_per_frame = pixels_per_frame * samples_per_pixel
    num_paths = (time_end - time_start) * paths_per_frame

    for path_id in range(num_paths):
        f_rel = path_id // paths_per_frame
        rem = path_id - f_rel * paths_per_frame
        p = rem // samples_per_pixel
        f = time_start + f_rel
        ff = ti.cast(f, ti.f32)
        py = p // width
        px = p - py * width
        pixel_size_per_t = pixel_world_scale[f]

        background = ti.math.vec4(ti.cast(out[f_rel, p, 0], ti.f32),
                                  ti.cast(out[f_rel, p, 1], ti.f32),
                                  ti.cast(out[f_rel, p, 2], ti.f32),
                                  ti.cast(out[f_rel, p, 3], ti.f32)) / 255.0
        background_alpha = 1.0
        if transparent != 0:
            background_alpha = ti.cast(out[f_rel, p, 4], ti.f32) / 255.0

        acc = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
        ro, rd = _generate_ray(f, px, py, ti.random(ti.f32),
                               ti.random(ti.f32),
                               half_screen_w, half_screen_h,
                               cam_origin, screen_point,
                               pixel_basis_x, pixel_basis_y)
        inv_rd = ti.math.vec3(_safe_inverse(rd[0]), _safe_inverse(rd[1]),
                              _safe_inverse(rd[2]))
        # Distances fed to pixel_size_per_t are slant ranges; fold in the
        # primary ray's cosine so the first (and dominant) segment converts
        # from perpendicular depth (see _axis_cos).
        pixel_size_per_t *= _axis_cos(f, ro, rd, screen_point)
        throughput = ti.math.vec4(1.0, 1.0, 1.0, 1.0)
        t_prev = 0.0
        layer_prev = 1e30
        base_dist = 0.0
        seam_t = -1e30
        bounces_left = max_bounces
        interacted = False
        escaped = False

        step = 0
        while step < MAX_SURFACES_PER_RAY:
            step += 1
            (found, t_hit, hit_layer, prim, hit_type, a, b, border,
             edge_hit) = _nearest_surface(
                refit, ro, rd, inv_rd, f, ff, t_prev, layer_prev,
                pixel_size_per_t, base_dist, layer_offset_triangles,
                t_nodes, t_node_miss, t_leaf_prim, t_leaf_tspan,
                t_first_leaf, tri_pos,
                b_nodes, b_node_miss, b_leaf_prim, b_leaf_tspan,
                b_first_leaf, circuit_meta, edges_2d, edge_accel)
            
            t_seg_end = 1e30
            if found != 0:
                t_seg_end = t_hit
            if found == 0:
                escaped = True
                break

            # Mesh seams: skip the duplicate edge hit of the adjacent
            # triangle so the surface scatters/transmits exactly once.
            seam_eps = DEPTH_TIE_EPSILON
            if (edge_hit == 1) and (t_hit - seam_t <= seam_eps):
                t_prev = t_hit
                layer_prev = hit_layer
                continue
            seam_t = t_hit if edge_hit == 1 else -1e30

            w0 = 1.0 - a - b
            color = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
            alpha = 0.0
            reflectivity = 0.0
            roughness = 0.0
            if hit_type == 1:
                color, alpha = _flat_triangle_color(f, prim, w0, a, b, tri_colors,
                                                    tri_uvs, tri_tex_meta, textures, num_colored_triangles)
                reflectivity, roughness = _triangle_extra(
                    f, prim, w0, a, b, tri_extra)
            else:
                color, alpha = _sample_circuit_color(
                    prim, f, a, b, border,
                    circuit_meta, circuit_colors, circuit_border_colors)
                cm = f % circuit_meta.shape[0]
                reflectivity = circuit_meta[cm, prim, _M_REFLECTIVITY]
                roughness = circuit_meta[cm, prim, _M_ROUGHNESS]

            alpha = ti.math.clamp(alpha, 0.0, 1.0)
            if ti.random(ti.f32) >= alpha:
                # Pass straight through the (partially) transparent
                # surface; advance the peel state along the same ray.
                t_prev = t_hit
                layer_prev = hit_layer
                continue
            interacted = True

            metalness = reflectivity

            normal = ti.math.vec3(0.0, 0.0, 0.0)
            if hit_type == 1:
                normal = _triangle_normal(f, prim, w0, a, b, tri_norm,
                                          tri_pos)
            else:
                normal = _bezier_normal(f, prim, circuit_meta)
            if normal.norm() > 1e-9:
                normal = normal.normalized()
            if normal.dot(rd) > 0.0:
                normal = -normal
            hit_point = ro + t_hit * rd

            # The packed surface value is material metalness, not an
            # independent mirror probability.  Derive a Schlick reflectance
            # (4% dielectric F0, rising to a conductor lobe at metalness=1).
            reflectivity = 0.0
            if (metalness >= 0.0) and (bounces_left > 0):
                m = ti.math.clamp(metalness, 0.0, 1.0)
                f0 = 0.04 * (1.0 - m) + m
                cos_view = ti.math.clamp(normal.dot(-rd), 0.0, 1.0)
                reflectivity = f0 + (1.0 - f0) \
                    * ti.pow(1.0 - cos_view, 5.0)

            if ti.random(ti.f32) < reflectivity:
                # Specular bounce, jittered into a glossy lobe.
                rd_new = (rd - 2.0 * rd.dot(normal) * normal).normalized()
                if roughness > 1e-4:
                    rd_new = (rd_new + roughness
                              * _random_unit_vector()).normalized()
                    if rd_new.dot(normal) < 0.0:
                        rd_new = rd_new - 2.0 * rd_new.dot(normal) * normal
                rd = rd_new
            else:
                # Diffuse interaction: emit the surface's color.
                acc += throughput * color
                if (indirect_strength <= 0.0) or (bounces_left <= 0):
                    break  # absorbed
                albedo_mean = (color[0] + color[1] + color[2]) / 3.0
                throughput *= ti.math.vec4(
                    color[0], color[1], color[2], albedo_mean
                ) * indirect_strength
                if (ti.max(throughput[0],
                           ti.max(throughput[1], throughput[2]))
                        < MIN_WEIGHT):
                    break  # absorbed
                rd = _cosine_hemisphere_direction(normal)

            ro = hit_point + normal * (10.0 * MIN_HIT_DISTANCE)
            inv_rd = ti.math.vec3(_safe_inverse(rd[0]),
                                  _safe_inverse(rd[1]),
                                  _safe_inverse(rd[2]))
            base_dist += t_hit
            t_prev = 0.0
            layer_prev = 1e30
            seam_t = -1e30
            bounces_left -= 1

        sample_alpha = 1.0
        if escaped:
            acc += throughput * background
            if not interacted:
                sample_alpha = background_alpha

        for ci in ti.static(range(4)):
            ti.atomic_add(accum[f_rel, p, ci], acc[ci])
        if transparent != 0:
            ti.atomic_add(accum[f_rel, p, 4], sample_alpha)


@ti.kernel
def finalize_samples(samples_per_pixel: int, transparent: int,
                     tonemapping: ti.template(), tonemap_exposure: ti.f32,
                     accum: ti.types.ndarray(), out: ti.types.ndarray()):
    """Convert the Monte Carlo kernels' per-pixel sample sums into the u8
    output: ``out = clamp(accum / samples_per_pixel)``.
    """
    num_frames = accum.shape[0]
    num_pixels = accum.shape[1]
    inv_spp = 1.0 / ti.cast(samples_per_pixel, ti.f32)
    for cell in range(num_frames * num_pixels):
        f_rel = cell // num_pixels
        p = cell - f_rel * num_pixels
        csum = ti.math.vec4(accum[f_rel, p, 0] * 255.0,
                            accum[f_rel, p, 1] * 255.0,
                            accum[f_rel, p, 2] * 255.0,
                            accum[f_rel, p, 3] * 255.0)
        color_final = finalize_pixel_color(csum, inv_spp, tonemapping, tonemap_exposure)
        for ci in ti.static(range(4)):
            if ti.static(tonemapping == 3):
                out[f_rel, p, ci] = color_final[ci]
            else:
                out[f_rel, p, ci] = ti.cast(color_final[ci], ti.u8)
        if transparent != 0:
            val = accum[f_rel, p, 4] * inv_spp * 255.0
            out[f_rel, p, 4] = ti.cast(ti.math.clamp(val + 0.5, 0.0, 255.0),
                                       ti.u8)


@ti.func
def _transmittance(refit: ti.template(), ro, rd, f, ff, max_t,
                   pixel_size_per_t, base_dist, layer_offset_triangles,
                   t_nodes: ti.template(), t_node_miss: ti.template(),
                   t_leaf_prim: ti.template(), t_leaf_tspan: ti.template(),
                   t_first_leaf, tri_pos: ti.template(),
                   tri_colors: ti.template(),
                   b_nodes: ti.template(), b_node_miss: ti.template(),
                   b_leaf_prim: ti.template(), b_leaf_tspan: ti.template(),
                   b_first_leaf, circuit_meta: ti.template(),
                   circuit_colors: ti.template(),
                   circuit_border_colors: ti.template(),
                   edges_2d: ti.template(), edge_accel: ti.template()):
    """Fraction of light transmitted along a shadow ray of length ``max_t``:
    every surface crossed attenuates by its transparency ``1 - alpha``
    (only the alpha channel of the surfaces is ever fetched).
    """
    inv_rd = ti.math.vec3(_safe_inverse(rd[0]), _safe_inverse(rd[1]),
                          _safe_inverse(rd[2]))
    transmitted = 1.0
    t_prev = 0.0
    layer_prev = 1e30
    seam_t = -1e30
    step = 0
    while step < MAX_SURFACES_PER_RAY:
        step += 1
        (found, t_hit, hit_layer, prim, hit_type, a, b, border,
         edge_hit) = _nearest_surface(
            refit, ro, rd, inv_rd, f, ff, t_prev, layer_prev,
            pixel_size_per_t, base_dist, layer_offset_triangles,
            t_nodes, t_node_miss, t_leaf_prim, t_leaf_tspan, t_first_leaf,
            tri_pos,
            b_nodes, b_node_miss, b_leaf_prim, b_leaf_tspan, b_first_leaf,
            circuit_meta, edges_2d, edge_accel)
        if (found == 0) or (t_hit >= max_t):
            break
        # Skip the duplicate edge hit of mesh seams (attenuate once).
        seam_eps = DEPTH_TIE_EPSILON
        if (edge_hit == 1) and (t_hit - seam_t <= seam_eps):
            t_prev = t_hit
            layer_prev = hit_layer
            continue
        seam_t = t_hit if edge_hit == 1 else -1e30
        alpha = 0.0
        if hit_type == 1:
            alpha = _triangle_alpha(f, prim, 1.0 - a - b, a, b, tri_colors)
        else:
            alpha = _circuit_alpha(prim, f, a, b, border, circuit_meta,
                                   circuit_colors, circuit_border_colors)
        transmitted *= 1.0 - ti.math.clamp(alpha, 0.0, 1.0)
        if transmitted < 1e-3:
            transmitted = 0.0
            break
        t_prev = t_hit
        layer_prev = hit_layer
    return transmitted


@ti.kernel
def path_trace_physical_stbvh(
        # Classic STBVH vs refit-tree walk (compile-time; see refit_bvh.py).
        refit: ti.template(),
        # Triangle STBVH + packed geometry.
        t_nodes: NODE_ARG, t_node_miss: ti.types.ndarray(),
        t_leaf_prim: ti.types.ndarray(), t_leaf_tspan: ti.types.ndarray(),
        t_first_leaf: int,
        tri_pos: ti.types.ndarray(), tri_norm: ti.types.ndarray(),
        tri_extra: ti.types.ndarray(), tri_colors: ti.types.ndarray(),
        tri_uvs: ti.types.ndarray(), tri_tex_meta: ti.types.ndarray(),
        textures: ti.types.ndarray(), num_colored_triangles: ti.i32,
        # Bezier STBVH + packed geometry.
        b_nodes: NODE_ARG, b_node_miss: ti.types.ndarray(),
        b_leaf_prim: ti.types.ndarray(), b_leaf_tspan: ti.types.ndarray(),
        b_first_leaf: int,
        circuit_meta: ti.types.ndarray(), circuit_colors: ti.types.ndarray(),
        circuit_border_colors: ti.types.ndarray(),
        edges_2d: ti.types.ndarray(), edge_accel: ti.types.ndarray(),
        # Per-frame camera and pixel scale.
        cam_origin: ti.types.ndarray(), screen_point: ti.types.ndarray(),
        pixel_basis_x: ti.types.ndarray(), pixel_basis_y: ti.types.ndarray(),
        pixel_world_scale: ti.types.ndarray(),
        # Render parameters.
        time_start: int, time_end: int, width: int, height: int,
        half_screen_w: float, half_screen_h: float,
        layer_offset_triangles: float,
        max_bounces: int, transparent: int,
        samples_per_pixel: int,
        # Explicit point lights [Tl, L, 3] and lighting controls.
        light_pos: ti.types.ndarray(), light_col: ti.types.ndarray(),
        num_lights: int, light_intensity: float, ambient: float,
        # Background/environment buffer (u8), read by escaping paths.
        out: ti.types.ndarray(),
        # Per-pixel sample accumulator [frames, pixels, 5] (f32,
        # zero-filled); converted to u8 means by ``finalize_samples``.
        accum: ti.types.ndarray()):
    """Physically based Monte Carlo path tracer with explicit lights.

    Vertex colors are treated as raw *albedo* (vertex shading is skipped in
    this mode) and all illumination is computed by the integrator:

    * **Point lights** are sampled explicitly at every interaction
      (next-event estimation): a shadow ray accumulates the transmittance
      through partially transparent occluders, and the surface responds with
      a Lambertian diffuse lobe plus a Fresnel-weighted glossy specular lobe
      (Schlick ``F0 = lerp(0.04, albedo, metallic)``, normalized
      Blinn-Phong with exponent derived from ``roughness``). The
      packed surface channel carries material ``metalness``; legacy materials
      use a negative sentinel and therefore have no PBR specular lobe.
    * **Emissive surfaces**: the ``glow`` channel emits
      ``albedo * glow`` radiance, picked up by paths that hit the surface
      (point lights are never hit by chance, so nothing is double counted).
    * **The background acts as the environment light** for escaping paths,
      and ``ambient`` adds a constant ambient term per diffuse interaction.
    * Continuation rays importance-sample the BRDF: a Fresnel-proportional
      coin chooses the specular lobe (mirror direction jittered by
      ``roughness``) or the cosine-weighted diffuse lobe, with throughput
      weights that keep the estimator unbiased. Transparency passes rays
      straight through with probability ``1 - alpha``.

    Like ``path_trace_scene_stbvh``, the parallel loop is flattened over
    (frame, pixel, sample) -- one thread per path -- with contributions
    accumulated atomically into ``accum`` and averaged by
    ``finalize_samples``.
    """
    pixels_per_frame = width * height
    paths_per_frame = pixels_per_frame * samples_per_pixel
    num_paths = (time_end - time_start) * paths_per_frame
    num_light_frames = ti.max(light_pos.shape[0], 1)

    for path_id in range(num_paths):
        f_rel = path_id // paths_per_frame
        rem = path_id - f_rel * paths_per_frame
        p = rem // samples_per_pixel
        f = time_start + f_rel
        ff = ti.cast(f, ti.f32)
        py = p // width
        px = p - py * width
        pixel_size_per_t = pixel_world_scale[f]

        background = ti.math.vec3(ti.cast(out[f_rel, p, 0], ti.f32),
                                  ti.cast(out[f_rel, p, 1], ti.f32),
                                  ti.cast(out[f_rel, p, 2], ti.f32)) / 255.0
        background_alpha = 1.0
        if transparent != 0:
            background_alpha = ti.cast(out[f_rel, p, 4], ti.f32) / 255.0

        acc = ti.math.vec4(0.0, 0.0, 0.0, 0.0)  # radiance rgb + bloom glow
        ro, rd = _generate_ray(f, px, py, ti.random(ti.f32),
                               ti.random(ti.f32),
                               half_screen_w, half_screen_h,
                               cam_origin, screen_point,
                               pixel_basis_x, pixel_basis_y)
        inv_rd = ti.math.vec3(_safe_inverse(rd[0]), _safe_inverse(rd[1]),
                              _safe_inverse(rd[2]))
        # See the matching note in path_trace_scene_stbvh.
        pixel_size_per_t *= _axis_cos(f, ro, rd, screen_point)
        throughput = ti.math.vec3(1.0, 1.0, 1.0)
        t_prev = 0.0
        layer_prev = 1e30
        base_dist = 0.0
        seam_t = -1e30
        bounces_left = max_bounces
        interacted = False
        escaped = False

        step = 0
        while step < MAX_SURFACES_PER_RAY:
            step += 1
            (found, t_hit, hit_layer, prim, hit_type, a, b, border,
             edge_hit) = _nearest_surface(
                refit, ro, rd, inv_rd, f, ff, t_prev, layer_prev,
                pixel_size_per_t, base_dist, layer_offset_triangles,
                t_nodes, t_node_miss, t_leaf_prim, t_leaf_tspan,
                t_first_leaf, tri_pos,
                b_nodes, b_node_miss, b_leaf_prim, b_leaf_tspan,
                b_first_leaf, circuit_meta, edges_2d, edge_accel)

            t_seg_end = 1e30
            if found != 0:
                t_seg_end = t_hit

            if found == 0:
                escaped = True
                break

            # Mesh seams: skip the duplicate edge hit of the adjacent
            # triangle (one interaction per surface crossing).
            seam_eps = DEPTH_TIE_EPSILON
            if (edge_hit == 1) and (t_hit - seam_t <= seam_eps):
                t_prev = t_hit
                layer_prev = hit_layer
                continue
            seam_t = t_hit if edge_hit == 1 else -1e30

            w0 = 1.0 - a - b
            color = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
            alpha = 0.0
            reflectivity = 0.0
            roughness = 0.0
            if hit_type == 1:
                color, alpha = _flat_triangle_color(f, prim, w0, a, b, tri_colors,
                                                    tri_uvs, tri_tex_meta, textures, num_colored_triangles)
                reflectivity, roughness = _triangle_extra(
                    f, prim, w0, a, b, tri_extra)
            else:
                color, alpha = _sample_circuit_color(
                    prim, f, a, b, border,
                    circuit_meta, circuit_colors, circuit_border_colors)
                cm = f % circuit_meta.shape[0]
                reflectivity = circuit_meta[cm, prim, _M_REFLECTIVITY]
                roughness = circuit_meta[cm, prim, _M_ROUGHNESS]

            alpha = ti.math.clamp(alpha, 0.0, 1.0)
            if ti.random(ti.f32) >= alpha:
                t_prev = t_hit
                layer_prev = hit_layer
                continue
            interacted = True

            albedo = ti.math.vec3(color[0], color[1], color[2])
            glow = ti.max(color[3], 0.0)
            has_pbr_material = reflectivity >= 0.0
            metallic = ti.math.clamp(reflectivity, 0.0, 1.0)
            if not has_pbr_material:
                metallic = 0.0
                roughness = 1.0
            normal = ti.math.vec3(0.0, 0.0, 0.0)
            if hit_type == 1:
                normal = _triangle_normal(f, prim, w0, a, b, tri_norm,
                                          tri_pos)
            else:
                normal = _bezier_normal(f, prim, circuit_meta)
            if normal.norm() > 1e-9:
                normal = normal.normalized()
            if normal.dot(rd) > 0.0:
                normal = -normal
            hit_point = ro + t_hit * rd
            shadow_origin = hit_point + normal * (10.0 * MIN_HIT_DISTANCE)

            f0 = ti.math.vec3(0.0, 0.0, 0.0)
            if has_pbr_material:
                f0 = ti.math.vec3(0.04, 0.04, 0.04) \
                    * (1.0 - metallic) + albedo * metallic
            cos_view = ti.max(normal.dot(-rd), 0.0)
            fresnel = f0 + (ti.math.vec3(1.0, 1.0, 1.0) - f0) \
                * ti.pow(1.0 - cos_view, 5.0)
            if not has_pbr_material:
                fresnel = ti.math.vec3(0.0, 0.0, 0.0)

            # Emission (glow) and constant ambient.
            acc += ti.math.vec4(
                throughput[0] * albedo[0] * glow,
                throughput[1] * albedo[1] * glow,
                throughput[2] * albedo[2] * glow,
                (throughput[0] + throughput[1] + throughput[2])
                / 3.0 * glow)
            if ambient > 0.0:
                amb = ambient * (1.0 - metallic)
                acc += ti.math.vec4(throughput[0] * albedo[0] * amb,
                                    throughput[1] * albedo[1] * amb,
                                    throughput[2] * albedo[2] * amb, 0.0)

            # Next-event estimation: sample every point light.
            phong_n = ti.math.clamp(
                2.0 / ti.max(roughness * roughness, 5e-4) - 2.0,
                1.0, 4096.0)
            tl = f % num_light_frames
            for li in range(num_lights):
                lp = ti.math.vec3(light_pos[tl, li, 0],
                                  light_pos[tl, li, 1],
                                  light_pos[tl, li, 2])
                to_light = lp - hit_point
                light_dist = to_light.norm()
                if light_dist > 1e-5:
                    wi = to_light / light_dist
                    cos_i = normal.dot(wi)
                    if cos_i > 1e-4:
                        visible = _transmittance(
                            refit, shadow_origin, wi, f, ff,
                            light_dist - 20.0 * MIN_HIT_DISTANCE,
                            pixel_size_per_t, base_dist,
                            layer_offset_triangles,
                            t_nodes, t_node_miss, t_leaf_prim,
                            t_leaf_tspan, t_first_leaf, tri_pos,
                            tri_colors,
                            b_nodes, b_node_miss, b_leaf_prim,
                            b_leaf_tspan, b_first_leaf, circuit_meta,
                            circuit_colors, circuit_border_colors,
                            edges_2d, edge_accel)
                        if visible > 1e-4:
                            half_v = (wi - rd).normalized()
                            cos_h = ti.max(normal.dot(half_v), 0.0)
                            spec = fresnel * ((phong_n + 2.0)
                                              / 6.283185307179586
                                              * ti.pow(cos_h, phong_n))
                            diff = albedo * ((1.0 - metallic)
                                             / 3.141592653589793)
                            radiance = ti.math.vec3(
                                light_col[tl, li, 0],
                                light_col[tl, li, 1],
                                light_col[tl, li, 2]) * light_intensity
                            lit = throughput * (diff + spec) \
                                * (cos_i * visible) * radiance
                            acc += ti.math.vec4(lit[0], lit[1], lit[2],
                                                0.0)

            # Importance-sample the BRDF for the continuation ray.
            if bounces_left <= 0:
                break  # absorbed
            spec_prob = ti.math.clamp(
                (fresnel[0] + fresnel[1] + fresnel[2]) / 3.0, 0.0, 0.95)
            if not has_pbr_material:
                spec_prob = 0.0
            if ti.random(ti.f32) < spec_prob:
                rd_new = (rd - 2.0 * rd.dot(normal) * normal).normalized()
                if roughness > 1e-4:
                    rd_new = (rd_new + roughness
                              * _random_unit_vector()).normalized()
                    if rd_new.dot(normal) < 0.0:
                        rd_new = rd_new - 2.0 * rd_new.dot(normal) * normal
                rd = rd_new
                throughput *= fresnel / spec_prob
            else:
                rd = _cosine_hemisphere_direction(normal)
                throughput *= albedo * ((1.0 - metallic)
                                        / (1.0 - spec_prob))
            if (ti.max(throughput[0],
                       ti.max(throughput[1], throughput[2]))
                    < MIN_WEIGHT):
                break  # absorbed
            ro = hit_point + normal * (10.0 * MIN_HIT_DISTANCE)
            inv_rd = ti.math.vec3(_safe_inverse(rd[0]),
                                  _safe_inverse(rd[1]),
                                  _safe_inverse(rd[2]))
            base_dist += t_hit
            t_prev = 0.0
            layer_prev = 1e30
            seam_t = -1e30
            bounces_left -= 1

        sample_alpha = 1.0
        if escaped:
            acc += ti.math.vec4(throughput[0] * background[0],
                                throughput[1] * background[1],
                                throughput[2] * background[2], 0.0)
            if not interacted:
                sample_alpha = background_alpha

        for ci in ti.static(range(4)):
            ti.atomic_add(accum[f_rel, p, ci], acc[ci])
        if transparent != 0:
            ti.atomic_add(accum[f_rel, p, 4], sample_alpha)
