"""Taichi kernel for ray tracing Algan scenes through spatio-temporal BVHs.

One GPU thread is launched per (frame, pixel). Each thread owns exactly one
cell of the output buffer ``[num_frames, num_pixels, channels]`` and performs
the whole render for its pixel -- visibility, alpha blending and mirror
bounces -- entirely in registers before writing the final color once. There
are no atomics and no intermediate fragment storage, so memory use is
independent of depth complexity and of the number of ray bounces.

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
* PN (curved point-normal) triangles, rendered as quadratic Bezier
  (Steiner) triangle patches: monomial coefficients ``pn_ctrl [Tp, N, 18]``
  packing ``S(u, v) = K0 + Ku u + Kv v + Kuu u^2 + Kvv v^2 + Kuv uv`` over
  the barycentric domain (hot); shading data ``pn_norm/pn_extra/pn_colors``
  with the same per-corner layouts as triangles, interpolated at vertex
  weights ``(1 - u - v, u, v)``. A ray can pierce one patch up to four
  times; ``_pn_intersect`` returns every hit so depth peeling stays exact;
* planar bezier circuits: ``circuit_meta [Tm, C, 20]`` (plane frame, border
  width, fill flag, texture grid transform), 2D polyline ``edges_2d`` with
  packed scanline/spatial tables ``edge_accel``, fill/texture colors
  ``circuit_colors [Tf, C, P, 5]`` (bilinearly sampled; P = 1 for plain
  fills) and ``circuit_border_colors [Tb, C, 5]``.

Coplanar-surface layer order is bezier circuits < triangles < PN patches,
with each type's primitive index breaking ties within the type.
"""
import os

import taichi as ti

from algan.rendering.raytracing.stbvh import BLOCK_F16, BVH_ARITY, LEAF_SIZE
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
    MAX_SHADOW_LIGHTS, _run_frag_pipeline)


from algan.rendering.taichi_runtime import init_taichi

init_taichi()

# Cull PN-patch candidates against their tight oriented box before the
# matrix-pencil solve (default on; env ALGAN_PN_OBB=0 disables for A/B). Output
# is identical -- the OBB conservatively bounds the patch -- and it removes the
# ~98% of solver invocations whose ray pierces a patch's loose axis-aligned leaf
# box but misses the (thin, often diagonal) patch itself.
_PN_OBB_ON = os.environ.get("ALGAN_PN_OBB", "1") == "1"

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
# A surface this opaque (or more) casts a deterministic hard shadow; more
# transparent surfaces are ignored by the binary shadow test (no glass/soft
# shadows -- those need the physical path tracer). Picked at one-half so a
# surface shadows exactly when it covers most of the light it occludes.
SHADOW_ALPHA_THRESHOLD = 0.5
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
# Slightly larger tolerances for the point-normal (PN) triangle patch
# numerical intersection solver and edge/seam de-duplication, to cover
# larger floating-point / solver noise.
PN_BARYCENTRIC_EPSILON = 1e-4
# Two G0-adjacent PN patches share their boundary curve but have different
# tangent planes there, so near a seam their curved surfaces overlap in a thin
# band and a ray can pierce *both* a small distance inside their shared edge.
# A PN hit counts as an edge hit (a seam-merge candidate) when its smallest
# barycentric coordinate is below this -- wide enough to cover that overlap
# band, so a translucent seam is blended once rather than twice.
PN_EDGE_EPSILON = 8e-3
# A ray grazing or near-edge piercing a patch makes two intersections nearly
# coincide, and the solver can recover them as several candidate hits at
# essentially the same surface point (via the two split lines, or the linear
# fallback landing on a pencil hit); Newton leaves them a few
# thousandths apart in (u, v) -- and, where the surface is steep, more than
# DEPTH_TIE_EPSILON apart in depth -- so they survive the depth tie-break and
# would blend two or three times (a bright seam on a translucent patch). Hits
# whose patch parameters agree to within this are treated as the same hit.
PN_DEDUP_UV_EPSILON = 5e-3
# Depth window for merging the duplicate edge hits of a *shared PN-patch seam*
# (the two curved patches meeting along a boundary curve). Across the overlap
# band the two near-edge hits sit a few thousandths apart in depth -- past
# DEPTH_TIE_EPSILON -- so the seam-merge needs a looser window than the
# flat-triangle case or a translucent seam blends twice. Still far below any
# visible surface separation (sub-pixel at typical scene scales), and only the
# extreme silhouette could merge a front/back pair this close, over a band far
# narrower than a pixel.
PN_SEAM_DEPTH_EPSILON = 8e-3
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
KBUF = max(1, int(os.environ.get("ALGAN_KBUF", "4")))

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

# tri_extra / pn_extra surface-transport block (see ``_pack_surface_extra``):
# per-corner (reflectivity, roughness) pairs in 0-5, per-corner IOR in 6-8,
# per-corner transmission in 9-11.
_EXTRA_W = 12
# ``pn_extra`` appends the per-corner UVs and then the per-patch texture
# metadata after that block (see ``_merge_scene``), so both are addressed
# relative to its width -- deriving them keeps the kernel in step with the
# packer instead of hard-coding offsets that silently rot when it changes.
_PN_UV = _EXTRA_W          # 6 cols: (u, v) per corner
_PN_META = _EXTRA_W + 6    # 10 cols: color(3), material(3), normal(3), flags

# circuit_meta channel layout.
_M_CENTER = 0      # 0-2   plane origin
_M_NORMAL = 3      # 3-5   unit plane normal
_M_BASIS_U = 6     # 6-8   plane frame u axis (unit)
_M_BASIS_V = 9     # 9-11  plane frame v axis (unit)
_M_BORDER_W = 12   # border half-width in screen pixels
_M_FILLED = 13     # > 0.5 if the circuit interior is filled
_M_GRID_W = 14     # texture grid width  (1 for plain fills)
_M_GRID_H = 15     # texture grid height (1 for plain fills)
_M_TEX = 16        # 16-19 2x2 map from plane (u, v) to texture axes
_M_GLOW_RADIUS = 20
# Surface transport, mirroring tri_extra's channels for flat triangles: material
# metalness (-1 = non-PBR), roughness, the unsigned IOR magnitude (dielectric
# F0), and transmission. A circuit transmits as a thin pane rather than
# refracting (see ``circuit_scatter``).
_M_REFLECTIVITY = 21
_M_ROUGHNESS = 22
_M_IOR = 23
_M_TRANSMISSION = 24
_M_WIDTH = 25


@ti.func
def _safe_inverse(x: ti.f32) -> ti.f32:
    r = 1e12
    if x < 0.0:
        r = -1e12
    if ti.abs(x) > 1e-12:
        r = 1.0 / x
    return r


@ti.func
def _bezier_point_metrics(circuit, te, u, v, query_radius, num_circuits,
                          edges_2d: ti.template(),
                          edge_accel: ti.template()):
    """Return even/odd crossings and nearest visible-edge distance.

    Crossing candidates come from the circuit's local-y scanline bin. Border
    candidates come from every 2D cell touched by the radius query square.
    Both candidate sets are conservative; the original exact predicates are
    still evaluated here, so the acceleration changes only the number of edges
    inspected.
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
    if ((query_radius > 0.0) and (u + query_radius >= min_u)
            and (u - query_radius <= max_u)
            and (v + query_radius >= min_v)
            and (v - query_radius <= max_v)):
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
                    min_dist_sq = ti.min(min_dist_sq, cx * cx + cy * cy)

    return crossings, min_dist_sq


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
def _obb_misses(ro, rd, po, prim, pn_obb: ti.template(), t_lo, t_hi) -> bool:
    """Conservative ray/OBB cull for a PN patch: True when the ray misses the
    patch's tight oriented box within ``[t_lo, t_hi]``. The box (built in
    :func:`pn_obb`) bounds the patch's control hull, so a miss here is a
    guaranteed miss of the patch and the matrix-pencil solve can be skipped --
    this rejects the bulk of false-positive candidates whose loose axis-aligned
    leaf box the ray pierces but whose (thin, often diagonal) patch it does not.
    The three packed axes are the frame directions scaled by their half-extents;
    a zero-extent axis (a perfectly flat patch) is left unconstrained.
    """
    cen = ti.math.vec3(pn_obb[po, prim, 0], pn_obb[po, prim, 1],
                       pn_obb[po, prim, 2])
    d = ro - cen
    tnear = -1e30
    tfar = 1e30
    miss = False
    for ax in ti.static(range(3)):
        a = ti.math.vec3(pn_obb[po, prim, 3 + ax * 3],
                         pn_obb[po, prim, 4 + ax * 3],
                         pn_obb[po, prim, 5 + ax * 3])
        l2 = a.dot(a)
        if l2 > 1e-30:
            inv = 1.0 / l2
            e = d.dot(a) * inv      # ray-origin coord in [-1, 1] slab units
            g = rd.dot(a) * inv
            if ti.abs(g) > 1e-12:
                ta = (-1.0 - e) / g
                tb = (1.0 - e) / g
                tnear = ti.max(tnear, ti.min(ta, tb))
                tfar = ti.min(tfar, ti.max(ta, tb))
            elif ti.abs(e) > 1.0:
                miss = True
    if (tnear > tfar) or (tfar < t_lo) or (tnear > t_hi):
        miss = True
    return miss


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
def _cbrt(x):
    """Real cube root, valid for negative arguments."""
    r = ti.pow(ti.abs(x), 1.0 / 3.0)
    if x < 0.0:
        r = -r
    return r


@ti.func
def _cubic_real_roots(a3, a2, a1, a0):
    """Real roots of ``a3 x^3 + a2 x^2 + a1 x + a0`` in closed form.

    Degenerate-degree aware: a vanishing leading coefficient falls back to
    the stable quadratic / linear formula, so the same routine serves the
    flat-patch pencil (whose cubic collapses to a quadratic) and the curved
    case. Returns ``(count, roots)`` with the real roots in ``roots[:count]``
    (``count`` in 0..3); the order is unspecified -- callers select a root by
    conditioning, not by magnitude.
    """
    roots = ti.math.vec3(0.0, 0.0, 0.0)
    count = 0
    scale = ti.max(ti.max(ti.abs(a3), ti.abs(a2)),
                   ti.max(ti.abs(a1), ti.max(ti.abs(a0), 1e-30)))
    if ti.abs(a3) <= 1e-7 * scale:
        # Effectively quadratic a2 x^2 + a1 x + a0 (or linear).
        if ti.abs(a2) <= 1e-7 * scale:
            if ti.abs(a1) > 1e-30:
                roots[0] = -a0 / a1
                count = 1
        else:
            disc = a1 * a1 - 4.0 * a2 * a0
            if disc >= 0.0:
                sq = ti.sqrt(disc)
                w = -0.5 * (a1 + sq)
                if a1 < 0.0:
                    w = -0.5 * (a1 - sq)
                if ti.abs(w) > 1e-30:
                    roots[0] = w / a2
                    roots[1] = a0 / w
                else:
                    roots[0] = -0.5 * a1 / a2
                    roots[1] = roots[0]
                count = 2
    else:
        # Depressed cubic y^3 + p y + q via x = y - a2 / (3 a3).
        b = a2 / a3
        c = a1 / a3
        d = a0 / a3
        shift = b * (1.0 / 3.0)
        p = c - b * b * (1.0 / 3.0)
        q = (2.0 / 27.0) * b * b * b - (1.0 / 3.0) * b * c + d
        half_q = 0.5 * q
        disc = half_q * half_q + p * p * p * (1.0 / 27.0)
        if disc >= 0.0:
            # One real root (Cardano); the other two are complex.
            sq = ti.sqrt(disc)
            roots[0] = _cbrt(-half_q + sq) + _cbrt(-half_q - sq) - shift
            count = 1
        else:
            # Three distinct real roots (p < 0): trigonometric form.
            m = 2.0 * ti.sqrt(-p * (1.0 / 3.0))
            arg = ti.math.clamp(3.0 * q / (p * m), -1.0, 1.0)
            theta = ti.acos(arg) * (1.0 / 3.0)
            roots[0] = m * ti.cos(theta) - shift
            roots[1] = m * ti.cos(theta - 2.0943951023931953) - shift
            roots[2] = m * ti.cos(theta - 4.1887902047863905) - shift
            count = 3
    return count, roots


@ti.func
def _pn_intersect(ro, rd, tp, prim, pn_ctrl: ti.template()):
    """Every intersection (up to four) of a ray with a quadratic Bezier
    (Steiner) triangle patch, packed as monomial coefficients
    ``S(u, v) = K0 + Ku u + Kv v + Kuu u^2 + Kvv v^2 + Kuv uv`` over the
    barycentric domain ``u, v >= 0, u + v <= 1``.

    Sederberg & Anderson's two-plane method: the patch is projected onto two
    orthogonal planes containing the ray, giving two bivariate quadratics
    ``f(u, v) = g(u, v) = 0`` whose common roots are the hits. Rather than
    eliminating a variable into a resultant quartic (numerically fragile --
    catastrophic cancellation near grazing rays), the pair is solved with the
    *matrix pencil*: each quadratic is a symmetric 3x3 conic matrix
    (``MF``, ``MG``), and ``det(x MF + MG)`` is a cubic in ``x`` whose real
    roots make ``M = x MF + MG`` a degenerate conic -- a product of two
    straight lines through all four common roots of ``f`` and ``g``. The root
    whose member splits into a *real* line pair (adjugate with a negative
    diagonal -- the squared homogeneous intersection point) is chosen, ``M``
    is factored into the two lines via ``M + [p]_x`` (rank 1), and each line
    is intersected with ``f`` through a single stable 1D quadratic. A flat
    patch makes f and g linear, which collapses the pencil (``det`` vanishes
    identically); that case falls back to the 2x2 linear solve. Every
    candidate is polished with three Newton steps on ``(f, g)`` to f32
    accuracy and kept only if it lands in the barycentric domain.

    Returns ``(count, t, u, v)`` with per-hit values in the vec4 slots
    ``[0, count)``; hits closer than DEPTH_TIE_EPSILON along the ray
    (tangential double roots) are merged.
    """
    k0 = ti.math.vec3(pn_ctrl[tp, prim, 0], pn_ctrl[tp, prim, 1],
                      pn_ctrl[tp, prim, 2]) - ro
    ku = ti.math.vec3(pn_ctrl[tp, prim, 3], pn_ctrl[tp, prim, 4],
                      pn_ctrl[tp, prim, 5])
    kv = ti.math.vec3(pn_ctrl[tp, prim, 6], pn_ctrl[tp, prim, 7],
                      pn_ctrl[tp, prim, 8])
    kuu = ti.math.vec3(pn_ctrl[tp, prim, 9], pn_ctrl[tp, prim, 10],
                       pn_ctrl[tp, prim, 11])
    kvv = ti.math.vec3(pn_ctrl[tp, prim, 12], pn_ctrl[tp, prim, 13],
                       pn_ctrl[tp, prim, 14])
    kuv = ti.math.vec3(pn_ctrl[tp, prim, 15], pn_ctrl[tp, prim, 16],
                       pn_ctrl[tp, prim, 17])

    helper = ti.math.vec3(1.0, 0.0, 0.0)
    if ti.abs(rd[0]) > 0.9:
        helper = ti.math.vec3(0.0, 1.0, 0.0)
    n1 = rd.cross(helper).normalized()
    n2 = rd.cross(n1)

    # f and g, each normalized so its largest coefficient is 1 (the
    # geometry is already relative to the ray origin via k0).
    A1 = n1.dot(kuu)
    B1 = n1.dot(kuv)
    C1 = n1.dot(kvv)
    D1 = n1.dot(ku)
    E1 = n1.dot(kv)
    F1 = n1.dot(k0)
    sf = ti.max(ti.max(ti.max(ti.abs(A1), ti.abs(B1)),
                       ti.max(ti.abs(C1), ti.abs(D1))),
                ti.max(ti.abs(E1), ti.max(ti.abs(F1), 1e-30)))
    A1 /= sf
    B1 /= sf
    C1 /= sf
    D1 /= sf
    E1 /= sf
    F1 /= sf
    A2 = n2.dot(kuu)
    B2 = n2.dot(kuv)
    C2 = n2.dot(kvv)
    D2 = n2.dot(ku)
    E2 = n2.dot(kv)
    F2 = n2.dot(k0)
    sg = ti.max(ti.max(ti.max(ti.abs(A2), ti.abs(B2)),
                       ti.max(ti.abs(C2), ti.abs(D2))),
                ti.max(ti.abs(E2), ti.max(ti.abs(F2), 1e-30)))
    A2 /= sg
    B2 /= sg
    C2 /= sg
    D2 /= sg
    E2 /= sg
    F2 /= sg

    # Symmetric conic matrices of f and g: (u, v, 1) M (u, v, 1)^T equals the
    # quadratic, so the cross/linear terms carry the standard factor 1/2.
    mf00 = A1
    mf11 = C1
    mf22 = F1
    mf01 = 0.5 * B1
    mf02 = 0.5 * D1
    mf12 = 0.5 * E1
    mg00 = A2
    mg11 = C2
    mg22 = F2
    mg01 = 0.5 * B2
    mg02 = 0.5 * D2
    mg12 = 0.5 * E2

    # Adjugates (symmetric) of MF and MG.
    af00 = mf11 * mf22 - mf12 * mf12
    af11 = mf00 * mf22 - mf02 * mf02
    af22 = mf00 * mf11 - mf01 * mf01
    af01 = mf02 * mf12 - mf01 * mf22
    af02 = mf01 * mf12 - mf02 * mf11
    af12 = mf01 * mf02 - mf00 * mf12
    ag00 = mg11 * mg22 - mg12 * mg12
    ag11 = mg00 * mg22 - mg02 * mg02
    ag22 = mg00 * mg11 - mg01 * mg01
    ag01 = mg02 * mg12 - mg01 * mg22
    ag02 = mg01 * mg12 - mg02 * mg11
    ag12 = mg01 * mg02 - mg00 * mg12

    # det(x MF + MG) = c3 x^3 + c2 x^2 + c1 x + c0 (the matrix pencil):
    # c3 = det(MF), c0 = det(MG), c2 = tr(adj(MF) MG), c1 = tr(adj(MG) MF).
    c3 = mf00 * af00 + mf01 * af01 + mf02 * af02
    c0 = mg00 * ag00 + mg01 * ag01 + mg02 * ag02
    c2 = (af00 * mg00 + af11 * mg11 + af22 * mg22
          + 2.0 * (af01 * mg01 + af02 * mg02 + af12 * mg12))
    c1 = (ag00 * mf00 + ag11 * mf11 + ag22 * mf22
          + 2.0 * (ag01 * mf01 + ag02 * mf02 + ag12 * mf12))

    nx, xr = _cubic_real_roots(c3, c2, c1, c0)

    # Pick the real pencil root whose member conic splits into a real line
    # pair. det(M) = 0 makes M two lines; the pair is real iff M's adjugate
    # has a negative diagonal entry (= -|intersection point|^2 in homogeneous
    # coordinates). Comparing on the scale-normalized M keeps a large root
    # (the member near the line at infinity) from looking spuriously
    # well-conditioned.
    best_q = 0.0
    x0 = 0.0
    found = 0
    for i in ti.static(range(3)):
        if i < nx:
            xroot = xr[i]
            a00 = xroot * mf00 + mg00
            a11 = xroot * mf11 + mg11
            a22 = xroot * mf22 + mg22
            a01 = xroot * mf01 + mg01
            a02 = xroot * mf02 + mg02
            a12 = xroot * mf12 + mg12
            ms = ti.max(ti.max(ti.max(ti.abs(a00), ti.abs(a11)),
                               ti.max(ti.abs(a22), ti.abs(a01))),
                        ti.max(ti.abs(a02), ti.max(ti.abs(a12), 1e-30)))
            inv = 1.0 / ms
            a00 *= inv
            a11 *= inv
            a22 *= inv
            a01 *= inv
            a02 *= inv
            a12 *= inv
            d00 = a11 * a22 - a12 * a12
            d11 = a00 * a22 - a02 * a02
            d22 = a00 * a11 - a01 * a01
            q = ti.max(ti.max(-d00, -d11), -d22)
            if q > best_q:
                best_q = q
                x0 = xroot
                found = 1

    count = 0
    out_t = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
    out_u = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
    out_v = ti.math.vec4(0.0, 0.0, 0.0, 0.0)

    # Linear fallback. A flat patch (or any ray whose two planes both cut the
    # patch in a straight line) makes f and g linear, so MF and MG share the
    # line at infinity and det(x MF + MG) vanishes for every x -- the pencil
    # has no usable root (found == 0). The lone hit is then the solution of
    # the 2x2 linear system, processed through the same path as the curved
    # candidates below.
    ldet = D1 * E2 - E1 * D2
    lin_u = 0.0
    lin_v = 0.0
    lin_ok = 0
    if (found == 0) and (ti.abs(ldet) > 1e-9):
        lin_u = (E1 * F2 - E2 * F1) / ldet
        lin_v = (D2 * F1 - D1 * F2) / ldet
        lin_ok = 1

    # The two split lines (defaulted so the loop skips them when found == 0).
    lines_a = ti.math.vec2(0.0, 0.0)
    lines_b = ti.math.vec2(0.0, 0.0)
    lines_c = ti.math.vec2(0.0, 0.0)
    if found == 1:
        # Rebuild the chosen degenerate member (scale-normalized) and split it
        # into the two lines L0 (largest-norm row of M + [p]_x) and L1
        # (largest-norm column): M + [p]_x is the rank-1 outer product of the
        # two line vectors, p being their intersection point.
        a00 = x0 * mf00 + mg00
        a11 = x0 * mf11 + mg11
        a22 = x0 * mf22 + mg22
        a01 = x0 * mf01 + mg01
        a02 = x0 * mf02 + mg02
        a12 = x0 * mf12 + mg12
        ms = ti.max(ti.max(ti.max(ti.abs(a00), ti.abs(a11)),
                           ti.max(ti.abs(a22), ti.abs(a01))),
                    ti.max(ti.abs(a02), ti.max(ti.abs(a12), 1e-30)))
        inv = 1.0 / ms
        a00 *= inv
        a11 *= inv
        a22 *= inv
        a01 *= inv
        a02 *= inv
        a12 *= inv
        b00 = a11 * a22 - a12 * a12
        b11 = a00 * a22 - a02 * a02
        b22 = a00 * a11 - a01 * a01
        b01 = a02 * a12 - a01 * a22
        b02 = a01 * a12 - a02 * a11
        b12 = a01 * a02 - a00 * a12
        px = 0.0
        py = 0.0
        pz = 0.0
        if (-b00 >= -b11) and (-b00 >= -b22):
            s = ti.sqrt(ti.max(-b00, 1e-30))
            px = b00 / s
            py = b01 / s
            pz = b02 / s
        elif -b11 >= -b22:
            s = ti.sqrt(ti.max(-b11, 1e-30))
            px = b01 / s
            py = b11 / s
            pz = b12 / s
        else:
            s = ti.sqrt(ti.max(-b22, 1e-30))
            px = b02 / s
            py = b12 / s
            pz = b22 / s
        c00 = a00
        c01 = a01 - pz
        c02 = a02 + py
        c10 = a01 + pz
        c11 = a11
        c12 = a12 - px
        c20 = a02 - py
        c21 = a12 + px
        c22 = a22
        r0n = c00 * c00 + c01 * c01 + c02 * c02
        r1n = c10 * c10 + c11 * c11 + c12 * c12
        r2n = c20 * c20 + c21 * c21 + c22 * c22
        la0 = c00
        lb0 = c01
        lc0 = c02
        if (r1n >= r0n) and (r1n >= r2n):
            la0 = c10
            lb0 = c11
            lc0 = c12
        elif r2n >= r0n:
            la0 = c20
            lb0 = c21
            lc0 = c22
        k0n = c00 * c00 + c10 * c10 + c20 * c20
        k1n = c01 * c01 + c11 * c11 + c21 * c21
        k2n = c02 * c02 + c12 * c12 + c22 * c22
        la1 = c00
        lb1 = c10
        lc1 = c20
        if (k1n >= k0n) and (k1n >= k2n):
            la1 = c01
            lb1 = c11
            lc1 = c21
        elif k2n >= k0n:
            la1 = c02
            lb1 = c12
            lc1 = c22
        lines_a = ti.math.vec2(la0, la1)
        lines_b = ti.math.vec2(lb0, lb1)
        lines_c = ti.math.vec2(lc0, lc1)

    # Candidate (u, v) sources: the two split lines (li 0, 1) intersected with
    # f, plus the linear-fallback point (li 2). Each is Newton-polished on
    # (f, g), domain-tested and de-duplicated.
    for li in ti.static(range(3)):
        u_c0 = 0.0
        v_c0 = 0.0
        u_c1 = 0.0
        v_c1 = 0.0
        num_uv = 0
        if ti.static(li < 2):
            la = lines_a[li]
            lb = lines_b[li]
            lc = lines_c[li]
            if ti.max(ti.abs(la), ti.abs(lb)) > 1e-12:
                # Substitute the line into f -> a stable 1D quadratic, solved
                # along whichever axis the line is least parallel to.
                qa = 0.0
                qb = 0.0
                qc = 0.0
                use_u = ti.abs(la) >= ti.abs(lb)
                al = 0.0
                be = 0.0
                if use_u:
                    al = -lb / la
                    be = -lc / la
                    qa = A1 * al * al + B1 * al + C1
                    qb = 2.0 * A1 * al * be + B1 * be + D1 * al + E1
                    qc = A1 * be * be + D1 * be + F1
                else:
                    al = -la / lb
                    be = -lc / lb
                    qa = C1 * al * al + B1 * al + A1
                    qb = 2.0 * C1 * al * be + B1 * be + E1 * al + D1
                    qc = C1 * be * be + E1 * be + F1
                t0 = 0.0
                t1 = 0.0
                if ti.abs(qa) <= 1e-12 * ti.max(ti.abs(qb), 1e-30):
                    if ti.abs(qb) > 1e-30:
                        t0 = -qc / qb
                        t1 = t0
                        num_uv = 1
                else:
                    disc = qb * qb - 4.0 * qa * qc
                    qsc = ti.max(qb * qb, ti.abs(4.0 * qa * qc)) + 1e-30
                    # A line through a real common root meets f; a slightly
                    # negative discriminant (a near-tangent rounded below zero)
                    # is clamped so the grazing double root is still recovered.
                    if disc >= -1e-6 * qsc:
                        sq = ti.sqrt(ti.max(disc, 0.0))
                        w = -0.5 * (qb + sq)
                        if qb < 0.0:
                            w = -0.5 * (qb - sq)
                        if ti.abs(w) > 1e-30:
                            t0 = w / qa
                            t1 = qc / w
                        else:
                            t0 = -0.5 * qb / qa
                            t1 = t0
                        num_uv = 2
                if use_u:
                    v_c0 = t0
                    u_c0 = al * t0 + be
                    v_c1 = t1
                    u_c1 = al * t1 + be
                else:
                    u_c0 = t0
                    v_c0 = al * t0 + be
                    u_c1 = t1
                    v_c1 = al * t1 + be
        else:
            if lin_ok == 1:
                u_c0 = lin_u
                v_c0 = lin_v
                num_uv = 1
        for vc in ti.static(range(2)):
            if vc < num_uv:
                u = u_c0 if ti.static(vc == 0) else u_c1
                v = v_c0 if ti.static(vc == 0) else v_c1
                for _ in ti.static(range(3)):
                    fval = (A1 * u + B1 * v + D1) * u + (C1 * v + E1) * v + F1
                    gval = (A2 * u + B2 * v + D2) * u + (C2 * v + E2) * v + F2
                    fu = 2.0 * A1 * u + B1 * v + D1
                    fv = B1 * u + 2.0 * C1 * v + E1
                    gu = 2.0 * A2 * u + B2 * v + D2
                    gv = B2 * u + 2.0 * C2 * v + E2
                    det = fu * gv - fv * gu
                    if ti.abs(det) > 1e-12:
                        du = (gv * fval - fv * gval) / det
                        dv = (fu * gval - gu * fval) / det
                        u -= ti.math.clamp(du, -0.2, 0.2)
                        v -= ti.math.clamp(dv, -0.2, 0.2)
                fval = (A1 * u + B1 * v + D1) * u + (C1 * v + E1) * v + F1
                gval = (A2 * u + B2 * v + D2) * u + (C2 * v + E2) * v + F2
                if ((u >= -PN_BARYCENTRIC_EPSILON)
                        and (v >= -PN_BARYCENTRIC_EPSILON)
                        and (u + v <= 1.0 + PN_BARYCENTRIC_EPSILON)
                        and (ti.abs(fval) < 2e-3) and (ti.abs(gval) < 2e-3)):
                    x = (k0 + u * ku + v * kv + (u * u) * kuu
                         + (v * v) * kvv + (u * v) * kuv)
                    t = x.dot(rd)
                    dup = 0
                    for c in ti.static(range(4)):
                        if (c < count) and (
                                (ti.abs(out_t[c] - t) <= DEPTH_TIE_EPSILON)
                                or ((ti.abs(out_u[c] - u)
                                     <= PN_DEDUP_UV_EPSILON)
                                    and (ti.abs(out_v[c] - v)
                                         <= PN_DEDUP_UV_EPSILON))):
                            dup = 1
                    if (dup == 0) and (count < 4):
                        out_t[count] = t
                        out_u[count] = u
                        out_v[count] = v
                        count += 1
    return count, out_t, out_u, out_v


@ti.func
def _nearest_triangle_hit(ro, rd, inv_rd, f, ff, t_prev, layer_prev,
                          t_cap, layer_offset,
                          nodes: ti.template(), node_miss: ti.template(),
                          leaf_prim: ti.template(), leaf_tspan: ti.template(),
                          first_leaf, tri_pos: ti.template()):
    """Nearest triangle intersection strictly after (t_prev, layer_prev)."""
    best_t = 1e30
    best_layer = -1e30
    best_prim = -1
    best_w1 = 0.0
    best_w2 = 0.0
    tp = f % tri_pos.shape[0]
    g_sp = 0
    g_st = ti.Vector([0] * _GROUP_STACK)
    g_cur = 0
    g_pend, g_near = _test_children(
        0, f, ro, inv_rd, t_prev - DEPTH_TIE_EPSILON,
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
            fresh_mask, g_near = _test_children(
                g_cur, f, ro, inv_rd, t_prev - DEPTH_TIE_EPSILON,
                ti.min(best_t + DEPTH_TIE_EPSILON,
                       t_cap + DEPTH_TIE_EPSILON), nodes)
            g_pend = saved_mask & fresh_mask
        else:
            g_c = _nearest_pending_child(g_pend, g_near)
            g_pend &= ~(1 << g_c)
            g_child = BVH_ARITY * g_cur + 1 + g_c
            if g_child >= first_leaf:
                base = (g_child - first_leaf) * LEAF_SIZE
                for j in ti.static(range(LEAF_SIZE)):
                    prim = leaf_prim[base + j]
                    tspan = leaf_tspan[base + j]
                    if ((prim >= 0) and ((tspan & 0xFFFF) <= f)
                            and (f <= ((tspan >> 16) & 0x7FFF))):
                        v0 = ti.math.vec3(tri_pos[tp, prim, 0],
                                          tri_pos[tp, prim, 1],
                                          tri_pos[tp, prim, 2])
                        v1 = ti.math.vec3(tri_pos[tp, prim, 3],
                                          tri_pos[tp, prim, 4],
                                          tri_pos[tp, prim, 5])
                        v2 = ti.math.vec3(tri_pos[tp, prim, 6],
                                          tri_pos[tp, prim, 7],
                                          tri_pos[tp, prim, 8])
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
                                layer = layer_offset + ti.cast(prim, ti.f32)
                                if ((t > MIN_HIT_DISTANCE)
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
                g_cur = g_child
                g_pend, g_near = _test_children(
                    g_cur, f, ro, inv_rd, t_prev - DEPTH_TIE_EPSILON,
                    ti.min(best_t + DEPTH_TIE_EPSILON,
                           t_cap + DEPTH_TIE_EPSILON), nodes)
    return best_t, best_prim, best_w1, best_w2, best_layer


@ti.func
def _nearest_pn_hit(ro, rd, inv_rd, f, ff, t_prev, layer_prev, t_cap,
                    layer_offset,
                    nodes: ti.template(), node_miss: ti.template(),
                    leaf_prim: ti.template(), leaf_tspan: ti.template(),
                    first_leaf, pn_ctrl: ti.template(), pn_obb: ti.template()):
    """Nearest PN-patch intersection strictly after (t_prev, layer_prev).
    Every root of each candidate patch is considered (a ray can pierce a
    curved patch several times) and the patch parameters (u, v) of the
    winning hit double as its color/normal interpolation weights.

    Each candidate is first culled against its tight oriented box
    (:func:`_obb_misses`) within the still-useful window, skipping the
    matrix-pencil solve for the many rays that pierce a patch's loose leaf AABB
    but miss the patch itself -- the same conservative (output-preserving) cull
    the primary depth-peel uses.
    """
    best_t = 1e30
    best_layer = -1e30
    best_prim = -1
    best_u = 0.0
    best_v = 0.0
    tp = f % pn_ctrl.shape[0]
    po = f % pn_obb.shape[0]
    g_sp = 0
    g_st = ti.Vector([0] * _GROUP_STACK)
    g_cur = 0
    g_pend, g_near = _test_children(
        0, f, ro, inv_rd, t_prev - DEPTH_TIE_EPSILON,
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
            fresh_mask, g_near = _test_children(
                g_cur, f, ro, inv_rd, t_prev - DEPTH_TIE_EPSILON,
                ti.min(best_t + DEPTH_TIE_EPSILON,
                       t_cap + DEPTH_TIE_EPSILON), nodes)
            g_pend = saved_mask & fresh_mask
        else:
            g_c = _nearest_pending_child(g_pend, g_near)
            g_pend &= ~(1 << g_c)
            g_child = BVH_ARITY * g_cur + 1 + g_c
            if g_child >= first_leaf:
                base = (g_child - first_leaf) * LEAF_SIZE
                for j in ti.static(range(LEAF_SIZE)):
                    prim = leaf_prim[base + j]
                    tspan = leaf_tspan[base + j]
                    if ((prim >= 0) and ((tspan & 0xFFFF) <= f)
                            and (f <= ((tspan >> 16) & 0x7FFF))
                            and (ti.static(not _PN_OBB_ON) or not _obb_misses(
                                ro, rd, po, prim, pn_obb,
                                t_prev - DEPTH_TIE_EPSILON,
                                best_t + DEPTH_TIE_EPSILON))):
                        cnt, ts, us, vs = _pn_intersect(ro, rd, tp, prim,
                                                        pn_ctrl)
                        layer = layer_offset + ti.cast(prim, ti.f32)
                        for r in ti.static(range(4)):
                            if r < cnt:
                                t = ts[r]
                                if ((t > MIN_HIT_DISTANCE)
                                        and _comes_after(t, layer, t_prev,
                                                         layer_prev)
                                        and _comes_after(best_t, best_layer,
                                                         t, layer)):
                                    best_t = t
                                    best_layer = layer
                                    best_prim = prim
                                    best_u = us[r]
                                    best_v = vs[r]
            else:
                if g_pend != 0:
                    g_st[g_sp] = (g_cur << BVH_ARITY) | g_pend
                    g_sp += 1
                g_cur = g_child
                g_pend, g_near = _test_children(
                    g_cur, f, ro, inv_rd, t_prev - DEPTH_TIE_EPSILON,
                    ti.min(best_t + DEPTH_TIE_EPSILON,
                           t_cap + DEPTH_TIE_EPSILON), nodes)
    return best_t, best_prim, best_u, best_v, best_layer


@ti.func
def _nearest_bezier_hit(ro, rd, inv_rd, f, ff, t_prev, layer_prev, t_cap,
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
    g_sp = 0
    g_st = ti.Vector([0] * _GROUP_STACK)
    g_cur = 0
    g_pend, g_near = _test_children(
        0, f, ro, inv_rd, t_prev - DEPTH_TIE_EPSILON,
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
            fresh_mask, g_near = _test_children(
                g_cur, f, ro, inv_rd, t_prev - DEPTH_TIE_EPSILON,
                ti.min(best_t + DEPTH_TIE_EPSILON,
                       t_cap + DEPTH_TIE_EPSILON), nodes)
            g_pend = saved_mask & fresh_mask
        else:
            g_c = _nearest_pending_child(g_pend, g_near)
            g_pend &= ~(1 << g_c)
            g_child = BVH_ARITY * g_cur + 1 + g_c
            if g_child >= first_leaf:
                base = (g_child - first_leaf) * LEAF_SIZE
                for j in ti.static(range(LEAF_SIZE)):
                    circuit = leaf_prim[base + j]
                    tspan = leaf_tspan[base + j]
                    if ((circuit >= 0) and ((tspan & 0xFFFF) <= f)
                            and (f <= ((tspan >> 16) & 0x7FFF))):
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
                                query_radius = ti.abs(border_w)
                                if filled:
                                    query_radius = ti.max(query_radius, outline_w)
                                te = f % num_edge_frames
                                crossings, min_dist_sq = _bezier_point_metrics(
                                    circuit, te, u, v, query_radius,
                                    circuit_meta.shape[1], edges_2d, edge_accel)
                                in_border = min_dist_sq < border_w * border_w
                                inside = False
                                if filled:
                                    inside = ((crossings % 2) == 1) or (
                                        min_dist_sq < outline_w * outline_w)
                                if inside or in_border:
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
                g_cur = g_child
                g_pend, g_near = _test_children(
                    g_cur, f, ro, inv_rd, t_prev - DEPTH_TIE_EPSILON,
                    ti.min(best_t + DEPTH_TIE_EPSILON,
                           t_cap + DEPTH_TIE_EPSILON), nodes)
    return best_t, best_circuit, best_border, best_u, best_v, best_layer


@ti.func
def _sample_circuit_color(circuit, f, u, v, in_border,
                          circuit_meta: ti.template(),
                          circuit_colors: ti.template(),
                          circuit_border_colors: ti.template()):
    """Color of a circuit at plane coordinates (u, v): the border color, or
    the fill color bilinearly sampled from the circuit's texture grid (plain
    fills are 1x1 grids, which degenerate to the single fill color).
    """
    color = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
    alpha = 0.0
    if in_border == 1:
        tb = f % circuit_border_colors.shape[0]
        color = ti.math.vec4(circuit_border_colors[tb, circuit, 0],
                             circuit_border_colors[tb, circuit, 1],
                             circuit_border_colors[tb, circuit, 2],
                             circuit_border_colors[tb, circuit, 3])
        alpha = circuit_border_colors[tb, circuit, 4]
    else:
        tm = f % circuit_meta.shape[0]
        tc = f % circuit_colors.shape[0]
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
        num_points = circuit_colors.shape[2]
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
            color += w * ti.math.vec4(circuit_colors[tc, circuit, p, 0],
                                      circuit_colors[tc, circuit, p, 1],
                                      circuit_colors[tc, circuit, p, 2],
                                      circuit_colors[tc, circuit, p, 3])
            alpha += w * circuit_colors[tc, circuit, p, 4]
            sum_w += w
        color /= ti.max(sum_w, 1e-6)
        alpha /= ti.max(sum_w, 1e-6)
    return color, alpha


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
        tb = f % circuit_border_colors.shape[0]
        alpha = circuit_border_colors[tb, circuit, 4]
    else:
        tm = f % circuit_meta.shape[0]
        tc = f % circuit_colors.shape[0]
        grid_w = circuit_meta[tm, circuit, _M_GRID_W]
        grid_h = circuit_meta[tm, circuit, _M_GRID_H]
        c1 = 0.5 * (u * circuit_meta[tm, circuit, _M_TEX]
                    + v * circuit_meta[tm, circuit, _M_TEX + 1]) + 0.5
        c2 = 0.5 * (u * circuit_meta[tm, circuit, _M_TEX + 2]
                    + v * circuit_meta[tm, circuit, _M_TEX + 3]) + 0.5
        x = ti.math.clamp(c2 * grid_h, 0.0, ti.max(grid_h - 1.0, 0.0))
        y = ti.math.clamp(c1 * grid_w, 0.0, ti.max(grid_w - 1.0, 0.0))
        num_points = circuit_colors.shape[2]
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
            alpha += w * circuit_colors[tc, circuit, p, 4]
            sum_w += w
        alpha /= ti.max(sum_w, 1e-6)
    return alpha


@ti.func
def _triangle_color(f, prim, w0, w1, w2, tri_colors: ti.template()):
    """Barycentric color (RGB + glow) and alpha of a confirmed triangle hit.
    PN-patch hits share the per-corner layout and pass ``pn_colors`` with
    weights ``(1 - u - v, u, v)``; likewise for the alpha/extra variants.
    """
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
def _pn_normal(f, prim, u, v, pn_norm: ti.template(), pn_ctrl: ti.template()):
    """Interpolated shading normal of a PN-patch hit, at the flat-triangle
    vertex weights (1 - u - v, u, v): continuous across patch seams, exactly
    like Phong shading on the source mesh. Falls back to the patch's
    geometric normal (the cross product of the parametric tangents) when the
    vertex normals are degenerate. Only fetched for hits that scatter or
    reflect.
    """
    tn = f % pn_norm.shape[0]
    w0 = 1.0 - u - v
    normal = ti.math.vec3(0.0, 0.0, 0.0)
    for ci in ti.static(range(3)):
        normal[ci] = (w0 * pn_norm[tn, prim, ci]
                      + u * pn_norm[tn, prim, 3 + ci]
                      + v * pn_norm[tn, prim, 6 + ci])
    if normal.norm() < 1e-6:
        tp = f % pn_ctrl.shape[0]
        su = ti.math.vec3(0.0, 0.0, 0.0)
        sv = ti.math.vec3(0.0, 0.0, 0.0)
        for ci in ti.static(range(3)):
            su[ci] = (pn_ctrl[tp, prim, 3 + ci]
                      + 2.0 * u * pn_ctrl[tp, prim, 9 + ci]
                      + v * pn_ctrl[tp, prim, 15 + ci])
            sv[ci] = (pn_ctrl[tp, prim, 6 + ci]
                      + 2.0 * v * pn_ctrl[tp, prim, 12 + ci]
                      + u * pn_ctrl[tp, prim, 15 + ci])
        normal = su.cross(sv)
    return normal


@ti.func
def _bezier_normal(f, circuit, circuit_meta: ti.template()):
    tm = f % circuit_meta.shape[0]
    return ti.math.vec3(circuit_meta[tm, circuit, _M_NORMAL],
                        circuit_meta[tm, circuit, _M_NORMAL + 1],
                        circuit_meta[tm, circuit, _M_NORMAL + 2])


@ti.func
def _shade_tri_hit(frag_pipelines: ti.template(), f, prim, a, b, rd, t_hit, ro,
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
    shadow visibilities (used iff ``shadows``)."""
    tp = f % tri_pos.shape[0]
    v0 = ti.math.vec3(tri_pos[tp, prim, 0], tri_pos[tp, prim, 1],
                      tri_pos[tp, prim, 2])
    v1 = ti.math.vec3(tri_pos[tp, prim, 3], tri_pos[tp, prim, 4],
                      tri_pos[tp, prim, 5])
    v2 = ti.math.vec3(tri_pos[tp, prim, 6], tri_pos[tp, prim, 7],
                      tri_pos[tp, prim, 8])
    face_n = (v1 - v0).cross(v2 - v0)
    pos = ro + t_hit * rd
    rgb = ti.math.vec3(albedo[0], albedo[1], albedo[2])
    return _run_frag_pipeline(frag_pipelines, prim, f, pos, -rd, shade_normal,
                              face_n, rgb,
                              albedo[3], light_pos, light_col, num_lights,
                              tri_mat_id, tri_mat, shadows, vis)


@ti.func
def _shade_pn_hit(frag_pipelines: ti.template(), f, prim, a, b, rd, t_hit, ro,
                  pn_ctrl: ti.template(), shade_normal,
                  pn_mat_id: ti.template(), pn_mat: ti.template(),
                  light_pos: ti.template(), light_col: ti.template(),
                  num_lights, albedo, shadows: ti.template(), vis):
    """Per-fragment material shading of a confirmed PN-patch hit. Like
    :func:`_shade_tri_hit` (caller-supplied normal-mapped ``shade_normal``) but
    the geometric face normal is the cross product of the patch's parametric
    tangents at (u, v) = (a, b)."""
    tp = f % pn_ctrl.shape[0]
    su = ti.math.vec3(0.0, 0.0, 0.0)
    sv = ti.math.vec3(0.0, 0.0, 0.0)
    for ci in ti.static(range(3)):
        su[ci] = (pn_ctrl[tp, prim, 3 + ci]
                  + 2.0 * a * pn_ctrl[tp, prim, 9 + ci]
                  + b * pn_ctrl[tp, prim, 15 + ci])
        sv[ci] = (pn_ctrl[tp, prim, 6 + ci]
                  + 2.0 * b * pn_ctrl[tp, prim, 12 + ci]
                  + a * pn_ctrl[tp, prim, 15 + ci])
    face_n = su.cross(sv)
    pos = ro + t_hit * rd
    rgb = ti.math.vec3(albedo[0], albedo[1], albedo[2])
    return _run_frag_pipeline(frag_pipelines, prim, f, pos, -rd, shade_normal,
                              face_n, rgb,
                              albedo[3], light_pos, light_col, num_lights,
                              pn_mat_id, pn_mat, shadows, vis)


@ti.func
def _nearest_surface_g(has_tri: ti.template(), has_pn: ti.template(),
                     has_bez: ti.template(),
                     ro, rd, inv_rd, f, ff, t_prev, layer_prev,
                     t_cap,
                     pixel_size_per_t, base_dist, layer_offset_triangles,
                     layer_offset_pn,
                     t_nodes: ti.template(), t_node_miss: ti.template(),
                     t_leaf_prim: ti.template(), t_leaf_tspan: ti.template(),
                     t_first_leaf, tri_pos: ti.template(),
                     p_nodes: ti.template(), p_node_miss: ti.template(),
                     p_leaf_prim: ti.template(), p_leaf_tspan: ti.template(),
                     p_first_leaf, pn_ctrl: ti.template(),
                     pn_obb: ti.template(),
                     b_nodes: ti.template(), b_node_miss: ti.template(),
                     b_leaf_prim: ti.template(), b_leaf_tspan: ti.template(),
                     b_first_leaf, circuit_meta: ti.template(),
                     edges_2d: ti.template(), edge_accel: ti.template()):
    """Nearest surface of any geometry type strictly after
    (t_prev, layer_prev) along the ray. Geometry only -- shading data is
    fetched by the caller for the hits it actually uses.

    Returns ``(found, t_hit, layer, prim, hit_type, a, b, border,
    edge_hit)`` where ``hit_type`` is 0 for bezier circuits, 1 for
    triangles and 2 for PN patches, and ``(a, b)`` are the barycentric
    ``(w1, w2)`` for triangle hits, the patch parameters ``(u, v)`` for PN
    hits (their vertex weights are ``(1 - u - v, u, v)``) or the plane
    ``(u, v)`` for bezier hits; ``found == 0`` means the ray escapes the
    scene, ``edge_hit == 1`` flags a triangle/patch hit on/near one of its
    edges (used to merge the duplicate hits of mesh seams).
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
            ro, rd, inv_rd, f, ff, t_prev, layer_prev, t_cap,
            layer_offset_triangles,
            t_nodes, t_node_miss, t_leaf_prim, t_leaf_tspan, t_first_leaf,
            tri_pos)
    pt = 1e30
    p_prim = -1
    p_u = 0.0
    p_v = 0.0
    p_layer = -1e30
    if ti.static(has_pn != 0):
        pn_cap = t_cap
        if t_prim >= 0:
            pn_cap = ti.min(pn_cap, tt + DEPTH_TIE_EPSILON)
        pt, p_prim, p_u, p_v, p_layer = _nearest_pn_hit(
            ro, rd, inv_rd, f, ff, t_prev, layer_prev, pn_cap,
            layer_offset_pn,
            p_nodes, p_node_miss, p_leaf_prim, p_leaf_tspan, p_first_leaf,
            pn_ctrl, pn_obb)
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
        if p_prim >= 0:
            bez_cap = ti.min(bez_cap, pt + DEPTH_TIE_EPSILON)
        bt, b_circ, b_border, b_u, b_v, b_layer = _nearest_bezier_hit(
            ro, rd, inv_rd, f, ff, t_prev, layer_prev, bez_cap,
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
    if (p_prim >= 0) and ((found == 0)
                          or (not _comes_after(pt, p_layer, t_hit,
                                               hit_layer))):
        found = 1
        t_hit = pt
        hit_layer = p_layer
        hit_prim = p_prim
        hit_type = 2
        a = p_u
        b = p_v
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
        eps = TRIANGLE_EDGE_EPSILON
        if hit_type == 2:
            eps = PN_EDGE_EPSILON
        if ti.min(w0, ti.min(a, b)) < eps:
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
def _nearest_surface(ro, rd, inv_rd, f, ff, t_prev, layer_prev,
                     pixel_size_per_t, base_dist, layer_offset_triangles,
                     layer_offset_pn,
                     t_nodes: ti.template(), t_node_miss: ti.template(),
                     t_leaf_prim: ti.template(), t_leaf_tspan: ti.template(),
                     t_first_leaf, tri_pos: ti.template(),
                     p_nodes: ti.template(), p_node_miss: ti.template(),
                     p_leaf_prim: ti.template(), p_leaf_tspan: ti.template(),
                     p_first_leaf, pn_ctrl: ti.template(),
                     pn_obb: ti.template(),
                     b_nodes: ti.template(), b_node_miss: ti.template(),
                     b_leaf_prim: ti.template(), b_leaf_tspan: ti.template(),
                     b_first_leaf, circuit_meta: ti.template(),
                     edges_2d: ti.template(), edge_accel: ti.template()):
    """All-geometry-present wrapper of :func:`_nearest_surface_g` for callers
    (Monte-Carlo path tracers + gbuffer) that don't specialize on which geometry
    types are present. Byte-identical to the pre-gating ``_nearest_surface``."""
    return _nearest_surface_g(
        1, 1, 1,
        ro, rd, inv_rd, f, ff, t_prev, layer_prev,
        1e30,
        pixel_size_per_t, base_dist, layer_offset_triangles, layer_offset_pn,
        t_nodes, t_node_miss, t_leaf_prim, t_leaf_tspan, t_first_leaf, tri_pos,
        p_nodes, p_node_miss, p_leaf_prim, p_leaf_tspan, p_first_leaf,
        pn_ctrl, pn_obb,
        b_nodes, b_node_miss, b_leaf_prim, b_leaf_tspan, b_first_leaf,
        circuit_meta, edges_2d, edge_accel)


@ti.func
def _collect_hits(ro, rd, inv_rd, f, ff, t_prev, layer_prev,
                  pixel_size_per_t, base_dist, layer_offset_triangles,
                  layer_offset_pn,
                  hit_t: ti.template(), hit_layer: ti.template(),
                  hit_prim: ti.template(), hit_flags: ti.template(),
                  hit_a: ti.template(), hit_b: ti.template(),
                  t_nodes: ti.template(), t_node_miss: ti.template(),
                  t_leaf_prim: ti.template(), t_leaf_tspan: ti.template(),
                  t_first_leaf, tri_pos: ti.template(),
                  p_nodes: ti.template(), p_node_miss: ti.template(),
                  p_leaf_prim: ti.template(), p_leaf_tspan: ti.template(),
                  p_first_leaf, pn_ctrl: ti.template(),
                  pn_obb: ti.template(),
                  b_nodes: ti.template(), b_node_miss: ti.template(),
                  b_leaf_prim: ti.template(), b_leaf_tspan: ti.template(),
                  b_first_leaf, circuit_meta: ti.template(),
                  edges_2d: ti.template(), edge_accel: ti.template(),
                  has_tri: ti.template(), has_pn: ti.template(),
                  has_bez: ti.template(), initial_opq_t: ti.f32,
                  initial_opq_layer: ti.f32) -> ti.i32:
    """Gather the up-to-``KBUF`` nearest hits strictly after
    (t_prev, layer_prev) into the caller's buffers, in one traversal of each
    BVH. Triangles are traversed first; the PN-patch and bezier traversals
    then prune against the hits already gathered.

    ``has_tri``/``has_pn``/``has_bez`` flag which geometry types are present;
    a type absent from the whole batch has only a placeholder (empty) BVH, so
    its traversal is skipped outright (a launch-uniform branch, no divergence).

    Buffers hold geometry only (the consumer fetches shading data):
    ``hit_flags`` packs the hit type (0 = bezier circuit, 1 = triangle,
    2 = PN patch) in bits 0-1, plus ``edge_hit << 2`` and ``border << 3``.
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
        window_hi = worst_t + DEPTH_TIE_EPSILON if count == KBUF else 1e30
        window_hi = ti.min(window_hi, opq_t + DEPTH_TIE_EPSILON)
        g_sp = 0
        g_st = ti.Vector([0] * _GROUP_STACK)
        g_cur = 0
        g_pend, g_near = _test_children(
            0, f, ro, inv_rd, t_prev - DEPTH_TIE_EPSILON, window_hi,
            t_nodes)
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
                fresh_mask, g_near = _test_children(
                    g_cur, f, ro, inv_rd,
                    t_prev - DEPTH_TIE_EPSILON, window_hi, t_nodes)
                g_pend = saved_mask & fresh_mask
            else:
                g_c = _nearest_pending_child(g_pend, g_near)
                g_pend &= ~(1 << g_c)
                g_child = BVH_ARITY * g_cur + 1 + g_c
                if g_child >= t_first_leaf:
                    base = (g_child - t_first_leaf) * LEAF_SIZE
                    for j in ti.static(range(LEAF_SIZE)):
                        prim = t_leaf_prim[base + j]
                        tspan = t_leaf_tspan[base + j]
                        if ((prim >= 0) and ((tspan & 0xFFFF) <= f)
                                and (f <= ((tspan >> 16) & 0x7FFF))):
                            v0 = ti.math.vec3(tri_pos[tp, prim, 0],
                                              tri_pos[tp, prim, 1],
                                              tri_pos[tp, prim, 2])
                            v1 = ti.math.vec3(tri_pos[tp, prim, 3],
                                              tri_pos[tp, prim, 4],
                                              tri_pos[tp, prim, 5])
                            v2 = ti.math.vec3(tri_pos[tp, prim, 6],
                                              tri_pos[tp, prim, 7],
                                              tri_pos[tp, prim, 8])
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
                                        and (w1 + w2
                                             <= 1.0 + BARYCENTRIC_EPSILON)):
                                    t = e2.dot(qv) * inv_det
                                    layer = (layer_offset_triangles
                                             + ti.cast(prim, ti.f32))
                                    accept = ((t > MIN_HIT_DISTANCE)
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
                                        if (tspan < 0) and _comes_after(
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
                    g_cur = g_child
                    window_hi = worst_t + DEPTH_TIE_EPSILON \
                        if count == KBUF else 1e30
                    window_hi = ti.min(window_hi, opq_t + DEPTH_TIE_EPSILON)
                    g_pend, g_near = _test_children(
                        g_cur, f, ro, inv_rd,
                        t_prev - DEPTH_TIE_EPSILON, window_hi, t_nodes)

    # --- PN patch BVH (window already tightened by the triangle hits) ---
    if ti.static(has_pn != 0):
        pp = f % pn_ctrl.shape[0]
        po = f % pn_obb.shape[0]
        window_hi = worst_t + DEPTH_TIE_EPSILON if count == KBUF else 1e30
        window_hi = ti.min(window_hi, opq_t + DEPTH_TIE_EPSILON)
        g_sp = 0
        g_st = ti.Vector([0] * _GROUP_STACK)
        g_cur = 0
        g_pend, g_near = _test_children(
            0, f, ro, inv_rd, t_prev - DEPTH_TIE_EPSILON, window_hi,
            p_nodes)
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
                fresh_mask, g_near = _test_children(
                    g_cur, f, ro, inv_rd,
                    t_prev - DEPTH_TIE_EPSILON, window_hi, p_nodes)
                g_pend = saved_mask & fresh_mask
            else:
                g_c = _nearest_pending_child(g_pend, g_near)
                g_pend &= ~(1 << g_c)
                g_child = BVH_ARITY * g_cur + 1 + g_c
                if g_child >= p_first_leaf:
                    # Refresh the depth window at the leaf so the OBB cull
                    # prunes with the hits gathered since the parent's test.
                    window_hi = worst_t + DEPTH_TIE_EPSILON \
                        if count == KBUF else 1e30
                    window_hi = ti.min(window_hi, opq_t + DEPTH_TIE_EPSILON)
                    base = (g_child - p_first_leaf) * LEAF_SIZE
                    for j in ti.static(range(LEAF_SIZE)):
                        prim = p_leaf_prim[base + j]
                        tspan = p_leaf_tspan[base + j]
                        if ((prim >= 0) and ((tspan & 0xFFFF) <= f)
                                and (f <= ((tspan >> 16) & 0x7FFF))
                                and (ti.static(not _PN_OBB_ON) or not _obb_misses(
                                    ro, rd, po, prim, pn_obb,
                                    t_prev - DEPTH_TIE_EPSILON, window_hi))):
                            cnt, ts, us, vs = _pn_intersect(ro, rd, pp, prim,
                                                            pn_ctrl)
                            layer = layer_offset_pn + ti.cast(prim, ti.f32)
                            for r in ti.static(range(4)):
                                if r < cnt:
                                    t = ts[r]
                                    accept = ((t > MIN_HIT_DISTANCE)
                                              and _comes_after(t, layer, t_prev,
                                                               layer_prev)
                                              and not _comes_after(
                                                  t, layer, opq_t, opq_layer))
                                    if accept and (count == KBUF):
                                        accept = _comes_after(worst_t, worst_layer,
                                                              t, layer)
                                    if accept:
                                        slot = worst_idx
                                        if count < KBUF:
                                            slot = count
                                            count += 1
                                        hit_t[slot] = t
                                        hit_layer[slot] = layer
                                        hit_prim[slot] = prim
                                        u = us[r]
                                        v = vs[r]
                                        w0 = 1.0 - u - v
                                        eh = 1 if (ti.min(w0, ti.min(u, v))
                                                   < PN_EDGE_EPSILON) else 0
                                        hit_flags[slot] = 2 | (eh << 2)
                                        hit_a[slot] = u
                                        hit_b[slot] = v
                                        if (tspan < 0) and _comes_after(
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
                    g_cur = g_child
                    window_hi = worst_t + DEPTH_TIE_EPSILON \
                        if count == KBUF else 1e30
                    window_hi = ti.min(window_hi, opq_t + DEPTH_TIE_EPSILON)
                    g_pend, g_near = _test_children(
                        g_cur, f, ro, inv_rd,
                        t_prev - DEPTH_TIE_EPSILON, window_hi, p_nodes)

    # --- Bezier BVH (window tightened by the triangle and patch hits) ---
    if ti.static(has_bez != 0):
        num_meta_frames = circuit_meta.shape[0]
        num_edge_frames = edges_2d.shape[0]
        window_hi = worst_t + DEPTH_TIE_EPSILON if count == KBUF else 1e30
        window_hi = ti.min(window_hi, opq_t + DEPTH_TIE_EPSILON)
        g_sp = 0
        g_st = ti.Vector([0] * _GROUP_STACK)
        g_cur = 0
        g_pend, g_near = _test_children(
            0, f, ro, inv_rd, t_prev - DEPTH_TIE_EPSILON, window_hi,
            b_nodes)
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
                fresh_mask, g_near = _test_children(
                    g_cur, f, ro, inv_rd,
                    t_prev - DEPTH_TIE_EPSILON, window_hi, b_nodes)
                g_pend = saved_mask & fresh_mask
            else:
                g_c = _nearest_pending_child(g_pend, g_near)
                g_pend &= ~(1 << g_c)
                g_child = BVH_ARITY * g_cur + 1 + g_c
                if g_child >= b_first_leaf:
                    base = (g_child - b_first_leaf) * LEAF_SIZE
                    for j in ti.static(range(LEAF_SIZE)):
                        circuit = b_leaf_prim[base + j]
                        tspan = b_leaf_tspan[base + j]
                        if ((circuit >= 0) and ((tspan & 0xFFFF) <= f)
                                and (f <= ((tspan >> 16) & 0x7FFF))):
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
                                    query_radius = ti.abs(border_w)
                                    if filled:
                                        query_radius = ti.max(query_radius, outline_w)
                                    te = f % num_edge_frames
                                    crossings, min_dist_sq = _bezier_point_metrics(
                                        circuit, te, u, v, query_radius,
                                        circuit_meta.shape[1], edges_2d, edge_accel)
                                    in_border = min_dist_sq < border_w * border_w
                                    inside = False
                                    if filled:
                                        inside = ((crossings % 2) == 1) or (
                                            min_dist_sq < outline_w * outline_w)
                                    if inside or in_border:
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
                                        if (tspan < 0) and _comes_after(
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
                    g_cur = g_child
                    window_hi = worst_t + DEPTH_TIE_EPSILON \
                        if count == KBUF else 1e30
                    window_hi = ti.min(window_hi, opq_t + DEPTH_TIE_EPSILON)
                    g_pend, g_near = _test_children(
                        g_cur, f, ro, inv_rd,
                        t_prev - DEPTH_TIE_EPSILON, window_hi, b_nodes)
    return count


@ti.func
def _shadow_occluded(ro, rd, f, ff, max_t,
                     pixel_size_per_t, base_dist, layer_offset_triangles,
                     layer_offset_pn,
                     has_tri: ti.template(), has_pn: ti.template(),
                     has_bez: ti.template(),
                     t_nodes: ti.template(), t_node_miss: ti.template(),
                     t_leaf_prim: ti.template(), t_leaf_tspan: ti.template(),
                     t_first_leaf, tri_pos: ti.template(),
                     tri_colors: ti.template(), tri_uvs: ti.template(),
                     tri_tex_meta: ti.template(), textures: ti.template(),
                     num_colored_triangles: ti.i32,
                     p_nodes: ti.template(), p_node_miss: ti.template(),
                     p_leaf_prim: ti.template(), p_leaf_tspan: ti.template(),
                     p_first_leaf, pn_ctrl: ti.template(),
                     pn_obb: ti.template(),
                     pn_colors: ti.template(),
                     b_nodes: ti.template(), b_node_miss: ti.template(),
                     b_leaf_prim: ti.template(), b_leaf_tspan: ti.template(),
                     b_first_leaf, circuit_meta: ti.template(),
                     circuit_colors: ti.template(),
                     circuit_border_colors: ti.template(),
                     edges_2d: ti.template(), edge_accel: ti.template()):
    """Binary hard-shadow test for the deterministic renderer: returns 1.0 if
    a sufficiently opaque surface lies between the shaded point and the light
    (within ``max_t`` along ``rd``), else 0.0.

    Cheaper than the physical kernel's :func:`_transmittance`: it stops at the
    *first* blocker whose alpha reaches :data:`SHADOW_ALPHA_THRESHOLD` (so a lit
    point usually costs a single BVH traversal) and ignores more transparent
    surfaces entirely -- no transmittance accumulation, so no glass/soft
    shadows (use the physical path tracer for those). Mesh seams still merge
    their duplicate edge hit so a thin opaque seam can't double-count.
    """
    inv_rd = ti.math.vec3(_safe_inverse(rd[0]), _safe_inverse(rd[1]),
                          _safe_inverse(rd[2]))
    occluded = 0.0
    t_prev = 0.0
    layer_prev = 1e30
    seam_t = -1e30
    step = 0
    while step < MAX_SURFACES_PER_RAY:
        step += 1
        (found, t_hit, hit_layer, prim, hit_type, a, b, border,
         edge_hit) = _nearest_surface_g(
            has_tri, has_pn, has_bez,
            ro, rd, inv_rd, f, ff, t_prev, layer_prev,
            1e30,
            pixel_size_per_t, base_dist, layer_offset_triangles,
            layer_offset_pn,
            t_nodes, t_node_miss, t_leaf_prim, t_leaf_tspan, t_first_leaf,
            tri_pos,
            p_nodes, p_node_miss, p_leaf_prim, p_leaf_tspan, p_first_leaf,
            pn_ctrl, pn_obb,
            b_nodes, b_node_miss, b_leaf_prim, b_leaf_tspan, b_first_leaf,
            circuit_meta, edges_2d, edge_accel)
        if (found == 0) or (t_hit >= max_t):
            break
        seam_eps = PN_SEAM_DEPTH_EPSILON if hit_type == 2             else DEPTH_TIE_EPSILON
        if (edge_hit == 1) and (t_hit - seam_t <= seam_eps):
            t_prev = t_hit
            layer_prev = hit_layer
            continue
        seam_t = t_hit if edge_hit == 1 else -1e30
        alpha = 0.0
        if hit_type == 1:
            alpha = _flat_triangle_alpha(f, prim, 1.0 - a - b, a, b, tri_colors,
                                         tri_uvs, tri_tex_meta, textures, num_colored_triangles)
        elif hit_type == 2:
            alpha = _triangle_alpha(f, prim, 1.0 - a - b, a, b, pn_colors)
        else:
            alpha = _circuit_alpha(prim, f, a, b, border, circuit_meta,
                                   circuit_colors, circuit_border_colors)
        if alpha >= SHADOW_ALPHA_THRESHOLD:
            occluded = 1.0
            break
        t_prev = t_hit
        layer_prev = hit_layer
    return occluded


@ti.kernel
def path_trace_scene_stbvh(
        # Triangle STBVH + packed geometry.
        t_nodes: NODE_ARG, t_node_miss: ti.types.ndarray(),
        t_leaf_prim: ti.types.ndarray(), t_leaf_tspan: ti.types.ndarray(),
        t_first_leaf: int,
        tri_pos: ti.types.ndarray(), tri_norm: ti.types.ndarray(),
        tri_extra: ti.types.ndarray(), tri_colors: ti.types.ndarray(),
        tri_uvs: ti.types.ndarray(), tri_tex_meta: ti.types.ndarray(),
        textures: ti.types.ndarray(), num_colored_triangles: ti.i32,
        # PN patch STBVH + packed geometry.
        p_nodes: NODE_ARG, p_node_miss: ti.types.ndarray(),
        p_leaf_prim: ti.types.ndarray(), p_leaf_tspan: ti.types.ndarray(),
        p_first_leaf: int,
        pn_ctrl: ti.types.ndarray(), pn_norm: ti.types.ndarray(),
        pn_extra: ti.types.ndarray(), pn_colors: ti.types.ndarray(),
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
        layer_offset_triangles: float, layer_offset_pn: float,
        max_bounces: int, transparent: int,
        samples_per_pixel: int, indirect_strength: float,
        # Per-PN-patch oriented bounding box for the pre-solve cull.
        pn_obb: ti.types.ndarray(),
        # Background buffer [time_end - time_start, width * height,
        # channels] (u8), read by paths that escape the scene.
        out: ti.types.ndarray(),
        # Per-pixel sample accumulator [time_end - time_start,
        # width * height, 5] (f32, zero-filled by the caller); converted to
        # u8 means by ``finalize_samples``.
        accum: ti.types.ndarray()):
    """Monte Carlo estimator of the same light transport as
    ``render_scene_stbvh``, generalized with random scattering.

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
                ro, rd, inv_rd, f, ff, t_prev, layer_prev,
                pixel_size_per_t, base_dist, layer_offset_triangles,
                layer_offset_pn,
                t_nodes, t_node_miss, t_leaf_prim, t_leaf_tspan,
                t_first_leaf, tri_pos,
                p_nodes, p_node_miss, p_leaf_prim, p_leaf_tspan,
                p_first_leaf, pn_ctrl, pn_obb,
                b_nodes, b_node_miss, b_leaf_prim, b_leaf_tspan,
                b_first_leaf, circuit_meta, edges_2d, edge_accel)
            
            t_seg_end = 1e30
            if found != 0:
                t_seg_end = t_hit
            if found == 0:
                escaped = True
                break

            # Mesh seams: skip the duplicate edge hit of the adjacent
            # triangle/patch so the surface scatters/transmits exactly once
            # (PN seams need a looser depth window than flat triangles).
            seam_eps = PN_SEAM_DEPTH_EPSILON if hit_type == 2 \
                else DEPTH_TIE_EPSILON
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
            elif hit_type == 2:
                color, alpha = _triangle_color(f, prim, w0, a, b,
                                               pn_colors)
                reflectivity, roughness = _triangle_extra(
                    f, prim, w0, a, b, pn_extra)
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
            elif hit_type == 2:
                normal = _pn_normal(f, prim, a, b, pn_norm, pn_ctrl)
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
def _transmittance(ro, rd, f, ff, max_t,
                   pixel_size_per_t, base_dist, layer_offset_triangles,
                   layer_offset_pn,
                   t_nodes: ti.template(), t_node_miss: ti.template(),
                   t_leaf_prim: ti.template(), t_leaf_tspan: ti.template(),
                   t_first_leaf, tri_pos: ti.template(),
                   tri_colors: ti.template(),
                   p_nodes: ti.template(), p_node_miss: ti.template(),
                   p_leaf_prim: ti.template(), p_leaf_tspan: ti.template(),
                   p_first_leaf, pn_ctrl: ti.template(),
                   pn_obb: ti.template(),
                   pn_colors: ti.template(),
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
            ro, rd, inv_rd, f, ff, t_prev, layer_prev,
            pixel_size_per_t, base_dist, layer_offset_triangles,
            layer_offset_pn,
            t_nodes, t_node_miss, t_leaf_prim, t_leaf_tspan, t_first_leaf,
            tri_pos,
            p_nodes, p_node_miss, p_leaf_prim, p_leaf_tspan, p_first_leaf,
            pn_ctrl, pn_obb,
            b_nodes, b_node_miss, b_leaf_prim, b_leaf_tspan, b_first_leaf,
            circuit_meta, edges_2d, edge_accel)
        if (found == 0) or (t_hit >= max_t):
            break
        # Skip the duplicate edge hit of mesh seams (attenuate once); PN
        # seams need a looser depth window than flat triangles.
        seam_eps = PN_SEAM_DEPTH_EPSILON if hit_type == 2 \
            else DEPTH_TIE_EPSILON
        if (edge_hit == 1) and (t_hit - seam_t <= seam_eps):
            t_prev = t_hit
            layer_prev = hit_layer
            continue
        seam_t = t_hit if edge_hit == 1 else -1e30
        alpha = 0.0
        if hit_type == 1:
            alpha = _triangle_alpha(f, prim, 1.0 - a - b, a, b, tri_colors)
        elif hit_type == 2:
            alpha = _triangle_alpha(f, prim, 1.0 - a - b, a, b, pn_colors)
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
        # Triangle STBVH + packed geometry.
        t_nodes: NODE_ARG, t_node_miss: ti.types.ndarray(),
        t_leaf_prim: ti.types.ndarray(), t_leaf_tspan: ti.types.ndarray(),
        t_first_leaf: int,
        tri_pos: ti.types.ndarray(), tri_norm: ti.types.ndarray(),
        tri_extra: ti.types.ndarray(), tri_colors: ti.types.ndarray(),
        tri_uvs: ti.types.ndarray(), tri_tex_meta: ti.types.ndarray(),
        textures: ti.types.ndarray(), num_colored_triangles: ti.i32,
        # PN patch STBVH + packed geometry.
        p_nodes: NODE_ARG, p_node_miss: ti.types.ndarray(),
        p_leaf_prim: ti.types.ndarray(), p_leaf_tspan: ti.types.ndarray(),
        p_first_leaf: int,
        pn_ctrl: ti.types.ndarray(), pn_norm: ti.types.ndarray(),
        pn_extra: ti.types.ndarray(), pn_colors: ti.types.ndarray(),
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
        layer_offset_triangles: float, layer_offset_pn: float,
        max_bounces: int, transparent: int,
        samples_per_pixel: int,
        # Explicit point lights [Tl, L, 3] and lighting controls.
        light_pos: ti.types.ndarray(), light_col: ti.types.ndarray(),
        num_lights: int, light_intensity: float, ambient: float,
        # Per-PN-patch oriented bounding box for the pre-solve cull.
        pn_obb: ti.types.ndarray(),
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
                ro, rd, inv_rd, f, ff, t_prev, layer_prev,
                pixel_size_per_t, base_dist, layer_offset_triangles,
                layer_offset_pn,
                t_nodes, t_node_miss, t_leaf_prim, t_leaf_tspan,
                t_first_leaf, tri_pos,
                p_nodes, p_node_miss, p_leaf_prim, p_leaf_tspan,
                p_first_leaf, pn_ctrl, pn_obb,
                b_nodes, b_node_miss, b_leaf_prim, b_leaf_tspan,
                b_first_leaf, circuit_meta, edges_2d, edge_accel)
            
            t_seg_end = 1e30
            if found != 0:
                t_seg_end = t_hit

            if found == 0:
                escaped = True
                break

            # Mesh seams: skip the duplicate edge hit of the adjacent
            # triangle/patch (one interaction per surface crossing); PN seams
            # need a looser depth window than flat triangles.
            seam_eps = PN_SEAM_DEPTH_EPSILON if hit_type == 2 \
                else DEPTH_TIE_EPSILON
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
            elif hit_type == 2:
                color, alpha = _triangle_color(f, prim, w0, a, b,
                                               pn_colors)
                reflectivity, roughness = _triangle_extra(
                    f, prim, w0, a, b, pn_extra)
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
            elif hit_type == 2:
                normal = _pn_normal(f, prim, a, b, pn_norm, pn_ctrl)
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
                            shadow_origin, wi, f, ff,
                            light_dist - 20.0 * MIN_HIT_DISTANCE,
                            pixel_size_per_t, base_dist,
                            layer_offset_triangles, layer_offset_pn,
                            t_nodes, t_node_miss, t_leaf_prim,
                            t_leaf_tspan, t_first_leaf, tri_pos,
                            tri_colors,
                            p_nodes, p_node_miss, p_leaf_prim,
                            p_leaf_tspan, p_first_leaf, pn_ctrl,
                            pn_obb,
                            pn_colors,
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
