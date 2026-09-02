"""Shared Taichi ray-tracing library.

This module holds the ``@ti.func`` building blocks every renderer uses --
sibling-block STBVH traversal, triangle / bezier-circuit
intersection and color/material sampling, batched hit gathering
(``_collect_hits``), shadow occlusion and tonemapping -- plus
``finalize_samples``, which averages the path tracer's per-pixel sample sums
into the frame buffer. The deterministic (``samples_per_pixel == 1``)
renderer lives in ``wavefront_kernels_taichi``, the path tracer
(``samples_per_pixel > 1``) in ``path_tracer_taichi``; both import these
helpers.

Hits along a ray are processed strictly front-to-back by *batched depth
peeling*: each BVH traversal gathers the ``kbuf`` nearest hits beyond the
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
``bvh_leaf_size`` primitive slots (``leaf_prim`` plus a packed per-slot frame
interval ``leaf_tspan`` so out-of-frame instances are skipped exactly).

Geometry comes in three packed forms, each fetched at the ray's exact frame
(frame index modulo each array's own time length, so constant data can stay
single-frame). Hot data (what every candidate intersection touches) is kept
separate from cold data (what only confirmed hits touch):

* triangles: positions ``tri_pos [Tp, N, 9]`` (hot); shading normals
  ``tri_norm [Tn, N, 9]`` (cold: fetched only for mirror bounces or Monte
  Carlo scattering), ``tri_extra`` (per-corner reflectivity +
  roughness pairs, then per-corner IOR, then per-corner transmission, then
  the per-primitive RGB absorption coefficient; usually single-frame) and
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

from algan.environment import env_float, env_int
from algan.rendering.raytracing.bezier_acceleration import (
    BEZIER_ACCEL_HEADER_SIZE,
    BEZIER_GRID_INV_U,
    BEZIER_GRID_INV_V,
    BEZIER_MAX_U,
    BEZIER_MAX_V,
    BEZIER_MIN_U,
    BEZIER_MIN_V,
    BEZIER_SCAN_INV_V,
    BEZIER_SCAN_OFFSET_BASE,
    BEZIER_SPATIAL_OFFSET_BASE,
    bezier_scan_bins,
    bezier_spatial_grid,
)
from algan.rendering.raytracing.color_space_taichi import srgb_to_linear_f
from algan.rendering.raytracing.shading_taichi import (
    _run_frag_pipeline,
    _vis_max_component,
    # Re-exported: wavefront_kernels_taichi imports max_shadow_lights from
    # here rather than from shading_taichi, so this hop is load-bearing even
    # though nothing in this module reads the name.
    max_shadow_lights,  # noqa: F401
)
from algan.rendering.raytracing.stbvh import bvh_arity, bvh_block_f16, bvh_leaf_size


def rgb_shadow_tint():
    """Whether shadow rays carry colored payloads end to end (the
    ``ALGAN_RGB_SHADOW_TINT`` gate, default on): a transmissive surface tints
    the light it passes with its albedo and absorbs over its interior chord,
    instead of passing an achromatic fraction.

    Read live through the module object -- never import the value by value --
    because every use is behind ``ti.static``: the branch is resolved when the
    kernel COMPILES, so flipping the setting mid-process does nothing for any
    kernel already compiled. An A/B between the two arms must therefore be one
    process per arm. The variable is declared import-time in
    ``algan/environment.py`` precisely so a warm daemon refuses a client whose
    value differs instead of silently reusing the first arm's kernels.

    The gate covers only the TINTING and the ABSORPTION. The payload itself is
    RGB unconditionally: with the gate off each channel carries today's scalar
    value unchanged, which keeps the render byte-identical while the plumbing
    stays exercised.
    """
    from algan.rendering.raytracing import settings as rt_settings

    return bool(rt_settings.rgb_shadow_tint)

# Taichi is NOT initialized here. The arch depends on
# ``SETTINGS.computing.render_device``, which a script may still change at
# this point, so the program is created at the start of a render by
# ``taichi_runtime.ensure_taichi_for_render()``. Defining a kernel needs no
# program -- ``@ti.kernel`` only registers it; materialization at first launch
# is what needs one, and by then a render has selected the arch. Anything that
# launches these kernels outside a render (a benchmark, a unit test) must call
# ``init_taichi()`` itself.

# Sibling-block traversal stack. The walk descends into one intersected
# child at a time and pushes the sibling group's *remaining* mask; a complete
# bvh_arity-ary tree over P leaves is log_ARITY(P) levels deep with at most
# one push per level, so 16 covers 4^16 leaves (the largest practical build
# is ~4^12). Entries pack ``node << bvh_arity | mask``.
_GROUP_STACK = 16
_GROUP_MASK = (1 << bvh_arity) - 1

# The five numbers below are absolute WORLD-SPACE quantities, so unlike the
# barycentric pair further down they are only right for scenes built at
# Algan's own scale (a unit Square is 1 unit across). A scene authored at, say,
# 1000x that scale wants them scaled with it, which is why each takes an
# environment default rather than being a bare literal. Each is folded into the
# trace kernels when they compile, so set the variable before importing algan.

# Minimum hit distance along a ray (also the self-intersection guard for
# reflected rays, together with a normal offset at the bounce origin).
min_hit_distance = env_float("ALGAN_MIN_HIT_DISTANCE", 1e-4)
# Hits closer together than this along a ray are considered coplanar and are
# ordered by layer index instead of by distance. Also the quantum of the
# raster route's packed depth key (``floor(t / depth_tie_epsilon)`` in the high
# 32 bits of ``raster_taichi.Z_SENTINEL``'s layout), so shrinking it narrows
# the ray depth that key can address.
depth_tie_epsilon = env_float("ALGAN_DEPTH_TIE_EPSILON", 1e-4)
# Reciprocal, used to bin distances into coplanarity buckets in _comes_after.
INV_DEPTH_TIE_EPSILON = 1.0 / depth_tie_epsilon
# Surfaces more transparent than this neither reflect nor terminate peeling.
min_alpha = env_float("ALGAN_MIN_ALPHA", 1e-3)
# Marching stops once the remaining transmittance drops below this.
min_weight = env_float("ALGAN_MIN_WEIGHT", 1e-3)
# Hard cap on blended surfaces per ray, to bound worst-case stacked geometry.
# Raising it is what a deep translucent stack needs: the cap is one of the four
# ceilings ``truncation.py`` counts, and hitting it drops the surfaces past it
# out of the composite, which moves the image.
max_surfaces_per_ray = max(1, env_int("ALGAN_MAX_SURFACES_PER_RAY", 256))
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
# depth_tie_epsilon of each other along a ray, they are the two triangles
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
# That argument needs the edge functions to be exact negations, which held on
# x64 and did NOT hold on CUDA: NVVM contracts ``x * y - z * w`` into an FMA,
# and a contracted edge function is no longer antisymmetric. Both neighbours of
# a shared edge then returned the SAME tiny value where the true one is zero --
# negative, and each rejected (a black crack pixel down a quad's diagonal);
# positive, and each accepted. ``_edge_fn`` below is what restores it, by
# ordering the endpoints so both neighbours evaluate one identical expression;
# read its docstring before touching the arithmetic. A 64x64 render of a lit
# plane showed 16 crack pixels on CUDA and none on CPU before it.
#
# The switch itself lives on ``rt_settings`` as ``watertight_tri`` (reachable
# as ``SETTINGS.raytracing.experimental.watertight_tri``) and is read here
# through :func:`watertight_tri`, exactly as ``rgb_shadow_tint`` is: both are
# ``ti.static`` gates over a kernel body and there is no reason for one to be a
# setting and the other not.
#
# This comment used to say a runtime toggle "would silently reuse a cached
# kernel". The offline half of that is not true on Taichi 1.7.4 -- a folded
# gate reaches the IR the cache key is computed from, so each arm gets its own
# entry (see the corrected note at ``raster_taichi._GLOSSY_MIN_ROUGHNESS``).
# The in-process half is true and is what the setter documents: the branch is
# resolved when the kernel COMPILES, so a flip takes effect only for kernels
# compiled after it, and an A/B needs one process per arm.
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
def watertight_tri():
    """Whether ``_tri_hit`` uses the watertight (Woop-Benthin-Wald) test.

    Read live through the module object -- never import the value by value --
    for the same reason as :func:`rgb_shadow_tint`: the use is behind
    ``ti.static``, so whatever the setting holds when a kernel compiles is what
    that kernel keeps.
    """
    from algan.rendering.raytracing import settings as rt_settings

    return bool(rt_settings.watertight_tri)


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
def _edge_fn(px, py, qx, qy):
    """Signed area of (ray, P, Q): the WBW edge function, made antisymmetric.

    Watertightness rests on neighbours agreeing EXACTLY about a shared edge:
    the two triangles see it as (P, Q) and (Q, P), and the sign test partitions
    the ray between them only if their edge functions are exact negations.
    Written as ``px * qy - py * qx`` that holds in plain IEEE -- multiplication
    commutes and ``fl(x - y) == -fl(y - x)`` -- but NOT once the backend
    contracts the expression into an FMA, which CUDA does and x64 does not.
    ``fma(px, qy, -fl(py * qx))`` rounds only the SECOND product, so where the
    true value is zero (both products equal) each neighbour returns minus its
    own rounding error: the same nonzero value with the same SIGN, not
    negatives. Both then reject (a crack pixel) or both accept.

    So order the endpoints first, by the same total order that breaks the
    exact-zero tie above, and let the second neighbour negate the first's
    result. Both evaluate one bit-identical expression, so whatever the backend
    does to it they agree, and negation is exact. On a target that does not
    contract this is bit-for-bit what the direct form computed.
    """
    ax, ay, bx, by = px, py, qx, qy
    swap = not _edge_is_canonical(px, py, qx, qy)
    if swap:
        ax, ay, bx, by = qx, qy, px, py
    e = ax * by - ay * bx
    if swap:
        e = -e
    return e


@ti.func
def _tri_hit(ro, rd, v0, v1, v2):
    """Ray/triangle intersection: ``(ok, w1, w2, t)``.

    ``w1``/``w2`` are the barycentric weights of ``v1``/``v2`` (so
    ``w0 = 1 - w1 - w2``) and ``t`` the ray parameter, matching what the three
    call sites used to compute inline. Under ``watertight_tri`` this is
    Woop-Benthin-Wald; otherwise it is the shipped dilated Moller-Trumbore,
    unchanged and bit-for-bit.
    """
    ok = 0
    w1 = 0.0
    w2 = 0.0
    t = 0.0
    if ti.static(watertight_tri()):
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
            u = _edge_fn(cxs, cys, bxs, bys)
            v = _edge_fn(axs, ays, cxs, cys)
            w = _edge_fn(bxs, bys, axs, ays)
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
# surfaces re-traverse the scene once per kbuf surfaces instead of once per
# surface (and skip the final "anything left?" traversal whenever a batch
# comes back not full).
# kbuf is efficiency-only (the peel's transitive order makes the composite
# kbuf-invariant; verified byte-identical across 1/4/8), so it is exposed for
# per-scene tuning: small kbuf closes the traversal's depth window as soon as
# the buffer fills (tighter pruning for low-depth-complexity scenes), large
# kbuf re-traverses less on deep translucent stacks.
kbuf = max(1, env_int("ALGAN_KBUF", 4))

# Kernel-argument annotation for STBVH sibling-block arrays (see
# stbvh._build_blocks): entry [i, lane] holds one attribute of internal node
# i's bvh_arity children -- lanes 0-5 the box dims lo.x/lo.y/lo.z/hi.x/hi.y/
# hi.z, lanes 6(-7) their packed frame intervals -- so one aligned 128-byte
# (f32) or 64-byte (f16, conservatively out-rounded bounds) fetch tests a
# whole sibling group per dependent memory round. The vector element type
# makes Taichi issue the lanes as vector loads.
if bvh_block_f16:
    NODE_ARG = ti.types.ndarray(dtype=ti.types.vector(bvh_arity, ti.f16),
                                ndim=2)
else:
    NODE_ARG = ti.types.ndarray(dtype=ti.types.vector(bvh_arity, ti.f32),
                                ndim=2)

# tri_extra surface-transport block (see ``_pack_surface_extra``):
# per-corner (reflectivity, roughness) pairs in 0-5, per-corner IOR in 6-8,
# per-corner transmission in 9-11, and the per-primitive Beer-Lambert
# absorption coefficient (RGB) in 12-14 -- per-primitive rather than
# per-corner because one primitive is one material, so there is nothing to
# interpolate across the face.
_EXTRA_W = 15

# tri_extra columns holding that absorption coefficient (see _shadow_hit_sigma).
_EXTRA_SIGMA = 12

# Coverage floor for "this hit is a solid's surface", used by the shadow
# march's interior-absorption pairing.
#
# It is NOT ``alpha >= 1.0``, and the difference is visible in the frame. A
# hit's alpha is the barycentric blend ``w0*a0 + w1*a1 + w2*a2`` with
# ``w0 = 1 - a - b``, and ``(1-a-b) + a + b`` is not associative in f32, so the
# sum can land one ulp BELOW 1.0 even when all three corner alphas are exactly
# 1.0. A hit that misses the floor does not OPEN the medium (or does not CLOSE
# it), and that ray then loses its whole interior absorption -- never part of
# it, and never in the other direction, so every affected pixel comes out
# BRIGHTER than its neighbours.
#
# What that measured, on calib_absorption's three glass spheres: salt-and-
# pepper speckle across every umbra, at 5.8% / 14.5% / 28.2% relative
# deviation in green, worsening with the chord being dropped, where the
# pre-RGB renderer was uniform to std exactly 0. Widening the floor to the
# tolerance below took those to 0.7% / 0.6% / 1.0% AND moved the mean onto the
# Beer-Lambert line (large sphere 0.142 -> 0.132 against 0.122 predicted) --
# which is what identifies the bright pixels as dropped chords rather than as
# a sampling artefact. That A/B is the evidence the floor needs slack; the
# exact share of barycentrics affected depends on how the backend rounds and
# contracts the blend, so it is not worth quoting a percentage here.
#
# 1e-6 is the same slack ``_pack_frame_visibility`` already uses to decide a
# primitive is opaque, so "fully covering" means one thing across the package.
_SOLID_COVERAGE_MIN = 1.0 - 1e-6

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
# > 0.5 if a FILLED circuit's border straddles the outline instead of running
# inward from it (``SETTINGS.style.border_placement``, see
# ``_circuit_point_region``). Carried per circuit rather than baked into the
# kernels so the placement is switchable between renders in one process, and
# separate from _M_BORDER_W rather than riding its sign for the same reason
# _M_TRANSMISSION is separate from _M_IOR. Inert on an unfilled circuit, whose
# stroke is centred either way.
_M_BORDER_CENTERED = 24
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
def _circuit_outer_dilation(border_w, outline_w, filled, centered):
    """How far OUTSIDE the outline a circuit's drawn region reaches.

    Normally just the hairline dilation ``outline_w``, which is what keeps a
    sub-pixel or degenerate fill from vanishing. A FILLED circuit drawing a
    CENTRED border also spills half the stroke past the outline, so its drawn
    region reaches whichever of the two is further.
    """
    r = outline_w
    if filled and centered:
        r = ti.max(outline_w, 0.5 * ti.abs(border_w))
    return r


@ti.func
def _circuit_inner_distance(border_w, filled, centered):
    """Signed distance (positive inside) at which the border gives way to fill.

    The whole stroke width for an INWARD border, since it is laid entirely
    inside the outline; half of it for a CENTRED one, which spends the other
    half outside. An unfilled circuit is a centred band either way.
    """
    r = ti.abs(border_w)
    if centered or (not filled):
        r = 0.5 * ti.abs(border_w)
    return r


@ti.func
def _circuit_query_radius(border_w, outline_w, filled, centered):
    """Nearest-edge search radius that can classify one point of a circuit.

    ``border_w`` is the circuit's full stroke width in plane units. The radius
    has to reach both boundaries of the drawn region: the inner border/fill edge
    (:func:`_circuit_inner_distance`) and the outer silhouette
    (:func:`_circuit_outer_dilation`). An UNFILLED circuit is the band ``|d| <
    border_w / 2`` and needs only half the width.
    """
    r = 0.5 * ti.abs(border_w)
    if filled:
        r = ti.max(_circuit_inner_distance(border_w, filled, centered),
                   _circuit_outer_dilation(border_w, outline_w, filled,
                                           centered))
    return r


@ti.func
def _circuit_point_region(border_w, outline_w, filled, centered, crossings,
                          min_dist_sq):
    """Classify one point of a circuit as ``(drawn, in_border)``.

    ``border_w`` is the full stroke width and ``crossings``/``min_dist_sq`` come
    from :func:`_bezier_point_metrics`, so the signed distance to the outline is
    ``d = +/- sqrt(min_dist_sq)``, positive inside.

    A FILLED circuit's border runs INWARD by default -- the drawn region is the
    fill itself (dilated by ``outline_w`` so hairlines and degenerate fills
    survive) and the border is the part of it within ``border_w`` of the
    outline, i.e. ``d <= border_w``. Raising ``stroke_width`` therefore eats
    into the shape instead of dilating it, which is what keeps neighbouring
    glyphs from fusing.

    With ``centered`` set (``SETTINGS.style.border_placement == "centered"``,
    which is Manim's convention) the same stroke straddles the outline instead:
    the drawn region grows outward to ``d > -border_w / 2`` and the border is
    ``|d| < border_w / 2``. The shape then dilates with its stroke width, which
    is the whole difference -- everything downstream (the border/fill blend, the
    coverage filter, the packed ref) reads the same two booleans.

    An UNFILLED circuit has no interior to eat into, so its stroke is centred on
    the path whichever mode is in force: the band ``|d| < border_w / 2``.
    """
    drawn = False
    in_border = False
    if filled:
        dil = _circuit_outer_dilation(border_w, outline_w, filled, centered)
        inner = _circuit_inner_distance(border_w, filled, centered)
        drawn = ((crossings % 2) == 1) or (min_dist_sq < dil * dil)
        in_border = drawn and (ti.abs(border_w) > 0.0) and (
            ((crossings % 2) == 0)
            or (min_dist_sq < inner * inner))
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
        scan_bin = ti.math.clamp(scan_bin, 0, bezier_scan_bins - 1)
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
        cell_x0 = ti.math.clamp(cell_x0, 0, bezier_spatial_grid - 1)
        cell_y0 = ti.math.clamp(cell_y0, 0, bezier_spatial_grid - 1)
        cell_x1 = ti.math.clamp(cell_x1, 0, bezier_spatial_grid - 1)
        cell_y1 = ti.math.clamp(cell_y1, 0, bezier_spatial_grid - 1)
        for cell_y in range(cell_y0, cell_y1 + 1):
            for cell_x in range(cell_x0, cell_x1 + 1):
                cell = cell_y * bezier_spatial_grid + cell_x
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
    stbvh._build_blocks) -- a single fetch feeds ``bvh_arity`` independent
    box tests, so a whole sibling group costs one dependent memory round.
    Returns ``(mask, near)`` where ``mask`` is the bitmask of intersected
    children (bit c = child ``bvh_arity * blk + 1 + c``) and ``near[c]`` is
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
    float ulp of the ``depth_tie_epsilon`` acceptance boundary that the old
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
    near = ti.Vector([1e30] * bvh_arity)
    for c in ti.static(range(bvh_arity)):
        tspan = 0
        if ti.static(bvh_block_f16):
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
    near = ti.Vector([1e30] * bvh_arity)
    for c in ti.static(range(bvh_arity)):
        w = 0
        if ti.static(bvh_block_f16):
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
    for cc in ti.static(range(bvh_arity)):
        if cc == c:
            if ti.static(bvh_block_f16):
                w = ti.cast(ti.bit_cast(ts_a[cc], ti.u16), ti.i32) | (
                    ti.cast(ti.bit_cast(ts_b[cc], ti.u16), ti.i32) << 16)
            else:
                w = ti.bit_cast(ts_a[cc], ti.i32)
    return w


# Refit link-word decode masks (must match refit_bvh.LINK_*).
_REFIT_PRIM_MASK = (1 << 29) - 1

# "This primitive casts no shadow" (Mob.casts_shadows False), in the two leaf
# words the traversal already loads: bit 29 of a refit link word
# (refit_bvh.LINK_NOCAST_BIT) and bit 15 of an STBVH leaf's packed frame
# interval (stbvh.LEAF_NOCAST_BIT). Tested only where a SHADOW ray accepts a
# leaf, so the same geometry stays visible to camera, reflection and refraction
# rays; the two spellings exist because the two tree kinds pack their leaves
# differently, not because they mean different things. Setting bit 15 is why
# every leaf-interval read below masks t0 with 0x7FFF rather than 0xFFFF -- the
# halves were always clipped to 15 bits, so that narrowing changes nothing for
# a tree with no non-casters in it.
_REFIT_NOCAST_BIT = 1 << 29
_LEAF_NOCAST_BIT = 1 << 15


@ti.func
def _group_test(refit: ti.template(), row0, blk, f, ro, inv_rd, t_lo, t_hi,
                blocks: ti.template()):
    """Sibling-group test dispatch: the classic frame-gated implicit-heap
    block (``refit == 0``) or the per-frame link-gated refit block. ``row0``
    is dead in classic mode.
    """
    mask = 0
    near = ti.Vector([1e30] * bvh_arity)
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
    for c in ti.static(range(bvh_arity)):
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
    if ti.static(bvh_block_f16):
        t0 = ti.cast(ti.bit_cast(ts_a[0], ti.u16), ti.i32)
        t1 = ti.cast(ti.bit_cast(ts_b[0], ti.u16), ti.i32)
    else:
        ts = ti.bit_cast(ts_a[0], ti.i32)
        t0 = ts & 0xFFFF
        t1 = (ts >> 16) & 0x7FFF
    for c in ti.static(range(1, bvh_arity)):
        lo_x = ti.min(lo_x, ti.cast(lox[c], ti.f32))
        lo_y = ti.min(lo_y, ti.cast(loy[c], ti.f32))
        lo_z = ti.min(lo_z, ti.cast(loz[c], ti.f32))
        hi_x = ti.max(hi_x, ti.cast(hix[c], ti.f32))
        hi_y = ti.max(hi_y, ti.cast(hiy[c], ti.f32))
        hi_z = ti.max(hi_z, ti.cast(hiz[c], ti.f32))
        tc0 = 0
        tc1 = 0
        if ti.static(bvh_block_f16):
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

    Distances are floored into ``depth_tie_epsilon``-wide bins so hits in the
    same bin compare equal on distance and fall back to ``layer``. Binning
    (rather than the old symmetric ``t +/- EPS`` window) keeps the comparison
    transitive: the window version could rank A<B, B<C yet C<A, so the order in
    which the depth-peel consumed near-coplanar hits -- and thus the composite
    -- depended on how the hits were grouped, i.e. on kbuf (and on the BVH
    build). With a transitive order, the peel visits hits in one fixed sequence
    regardless of how many are gathered per traversal, so kbuf is efficiency-only.
    """
    bt = ti.floor(t * INV_DEPTH_TIE_EPSILON)
    bp = ti.floor(t_prev * INV_DEPTH_TIE_EPSILON)
    return (bt > bp) or ((bt == bp) and (layer < layer_prev))


@ti.func
def _shadow_identity_t_min(f, prim, src_sid, src_prim, eps_self, eps_near,
                           tri_obj: ti.template(),
                           ident: ti.template()) -> ti.f32:
    """Acceptance floor along a shadow ray for one candidate triangle hit.

    ``ident != 0`` (compile-time) engages identity-aware rejection
    (DESIGN_mesh_identity_open.md ssI, extended). The floor is chosen per hit
    from how plausibly that hit could be an artifact of the ray's own origin,
    in three tiers:

    * the ray's OWN primitive -- the only surface a reconstructed origin can
      re-hit spuriously by construction -- keeps the full floor ``eps_self``;
    * another primitive of the SAME mesh keeps ``eps_near``, which exists for
      mesh seams (a facet's reconstructed point can land under its
      neighbour's plane) and is 0 by default, i.e. primitive-precise;
    * any OTHER mesh gets 0. A different mesh cannot be an artifact of this
      ray's origin, so no floor is defensible there, and this is what lets a
      contact shadow survive.

    Rejecting a whole mesh would be wrong for a concave solid that
    legitimately shadows itself; rejecting only near-zero ``t`` on the source
    primitive is the narrowest test that still covers the artifact.

    Both epsilons arrive already scaled to the batch's scene size (see
    ``shadow_eps_relative``), which is what retires the absolute
    ``min_hit_distance`` from this path.

    ``src_prim < 0`` (per-ray runtime) disables the test for that ray:
    callers without a source identity -- the megakernel's camera, secondary
    and shadow rays, and shadow events whose source is a bezier circuit --
    keep exactly the old absolute epsilon.
    """
    t_min = min_hit_distance
    if ti.static(ident != 0):
        if src_prim >= 0:
            if prim == src_prim:
                t_min = eps_self
            else:
                hit_obj = ti.cast(tri_obj[f % tri_obj.shape[0], prim], ti.i32)
                t_min = 0.0
                if hit_obj == src_sid:
                    t_min = eps_near
    return t_min


@ti.func
def _nearest_triangle_hit(refit: ti.template(), ro, rd, inv_rd, f, ff,
                          t_prev, layer_prev,
                          t_cap, layer_offset,
                          nodes: ti.template(), node_miss: ti.template(),
                          leaf_prim: ti.template(), leaf_tspan: ti.template(),
                          first_leaf, tri_pos: ti.template(),
                          src_sid, src_prim, eps_self, eps_near,
                          tri_obj: ti.template(),
                          ident: ti.template(),
                          # 1 rejects a leaf whose primitive declared
                          # Mob.casts_shadows False; 0 (every non-shadow ray)
                          # compiles the test out.
                          nocast: ti.template()):
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
        refit, row0, 0, f, ro, inv_rd, t_prev - depth_tie_epsilon,
        ti.min(best_t + depth_tie_epsilon,
               t_cap + depth_tie_epsilon), nodes)
    while True:
        if g_pend == 0:
            if g_sp == 0:
                break
            g_sp -= 1
            saved = g_st[g_sp]
            g_cur = saved >> bvh_arity
            saved_mask = saved & _GROUP_MASK
            fresh_mask, g_near = _group_test(
                refit, row0, g_cur, f, ro, inv_rd,
                t_prev - depth_tie_epsilon,
                ti.min(best_t + depth_tie_epsilon,
                       t_cap + depth_tie_epsilon), nodes)
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
                    if ti.static(nocast != 0):
                        if (w & _REFIT_NOCAST_BIT) != 0:
                            l_prim = -1
            else:
                g_child = bvh_arity * g_cur + 1 + g_c
                if g_child >= first_leaf:
                    l_base = (g_child - first_leaf) * bvh_leaf_size
                else:
                    descend = 1
                    child_blk = g_child
            if descend == 0:
                for j in ti.static(range(1 if refit != 0 else bvh_leaf_size)):
                    prim = l_prim
                    if ti.static(refit == 0):
                        prim = -1
                        p0 = leaf_prim[l_base + j]
                        tspan = leaf_tspan[l_base + j]
                        if ((p0 >= 0) and ((tspan & 0x7FFF) <= f)
                                and (f <= ((tspan >> 16) & 0x7FFF))):
                            prim = p0
                            if ti.static(nocast != 0):
                                if (tspan & _LEAF_NOCAST_BIT) != 0:
                                    prim = -1
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
                                    f, prim, src_sid, src_prim, eps_self, eps_near, tri_obj, ident))
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
                    g_st[g_sp] = (g_cur << bvh_arity) | g_pend
                    g_sp += 1
                g_cur = child_blk
                g_pend, g_near = _group_test(
                    refit, row0, g_cur, f, ro, inv_rd,
                    t_prev - depth_tie_epsilon,
                    ti.min(best_t + depth_tie_epsilon,
                           t_cap + depth_tie_epsilon), nodes)
    return best_t, best_prim, best_w1, best_w2, best_layer


@ti.func
def _nearest_bezier_hit(refit: ti.template(), ro, rd, inv_rd, f, ff, t_prev,
                        layer_prev, t_cap,
                        pixel_size_per_t, base_dist,
                        nodes: ti.template(), node_miss: ti.template(),
                        leaf_prim: ti.template(), leaf_tspan: ti.template(),
                        first_leaf, circuit_meta: ti.template(),
                        edges_2d: ti.template(), edge_accel: ti.template(),
                        nocast: ti.template()):
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
        refit, row0, 0, f, ro, inv_rd, t_prev - depth_tie_epsilon,
        ti.min(best_t + depth_tie_epsilon,
               t_cap + depth_tie_epsilon), nodes)
    while True:
        if g_pend == 0:
            if g_sp == 0:
                break
            g_sp -= 1
            saved = g_st[g_sp]
            g_cur = saved >> bvh_arity
            saved_mask = saved & _GROUP_MASK
            fresh_mask, g_near = _group_test(
                refit, row0, g_cur, f, ro, inv_rd,
                t_prev - depth_tie_epsilon,
                ti.min(best_t + depth_tie_epsilon,
                       t_cap + depth_tie_epsilon), nodes)
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
                    if ti.static(nocast != 0):
                        if (w & _REFIT_NOCAST_BIT) != 0:
                            l_prim = -1
            else:
                g_child = bvh_arity * g_cur + 1 + g_c
                if g_child >= first_leaf:
                    l_base = (g_child - first_leaf) * bvh_leaf_size
                else:
                    descend = 1
                    child_blk = g_child
            if descend == 0:
                for j in ti.static(range(1 if refit != 0 else bvh_leaf_size)):
                    circuit = l_prim
                    if ti.static(refit == 0):
                        circuit = -1
                        p0 = leaf_prim[l_base + j]
                        tspan = leaf_tspan[l_base + j]
                        if ((p0 >= 0) and ((tspan & 0x7FFF) <= f)
                                and (f <= ((tspan >> 16) & 0x7FFF))):
                            circuit = p0
                            if ti.static(nocast != 0):
                                if (tspan & _LEAF_NOCAST_BIT) != 0:
                                    circuit = -1
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
                            if ((t > min_hit_distance)
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
                                centered = (circuit_meta[
                                    tm, circuit, _M_BORDER_CENTERED] > 0.5)
                                query_radius = _circuit_query_radius(
                                    border_w, outline_w, filled, centered)
                                te = f % num_edge_frames
                                (crossings, min_dist_sq, _ccu, _ccv, _e1x,
                                     _e1y, _sg1, _s2, _s2u, _s2v, _e2x, _e2y,
                                     _sg2) = _bezier_point_metrics(
                                    circuit, te, u, v, query_radius,
                                    circuit_meta.shape[1], edges_2d, edge_accel)
                                inside, in_border = _circuit_point_region(
                                    border_w, outline_w, filled, centered,
                                    crossings, min_dist_sq)
                                if inside:
                                    best_t = t
                                    best_layer = layer
                                    best_circuit = circuit
                                    best_border = 1 if in_border else 0
                                    best_u = u
                                    best_v = v
            else:
                if g_pend != 0:
                    g_st[g_sp] = (g_cur << bvh_arity) | g_pend
                    g_sp += 1
                g_cur = child_blk
                g_pend, g_near = _group_test(
                    refit, row0, g_cur, f, ro, inv_rd,
                    t_prev - depth_tie_epsilon,
                    ti.min(best_t + depth_tie_epsilon,
                           t_cap + depth_tie_epsilon), nodes)
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
    """Color of a circuit point whose pixel straddles the border's inner edge.

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
def _color_map_texel(tc, base_row, lut_base, texel_idx, num_points,
                     textures: ti.template()):
    """One texel of a color map: ``(rgb+glow vec4, alpha)``.

    ``lut_base < 0`` is a plain f32 map (five channels at row ``base_row +
    texel_idx``, the historical layout, byte for byte). ``lut_base >= 0`` is a
    u8-packed map (texture_u8_storage): one RGBA texel bit-packed into ONE f32
    lane of the shared bank -- lane ``base_row * 5 + texel_idx``, bytes
    little-endian r|g<<8|b<<16|a<<24 -- decoded through the per-map 256-entry
    LUT at rows ``lut_base..lut_base+255`` (col 0 = the value the f32 path
    would have stored for color byte k, col 1 = k/255 for the coverage
    byte). The host scatters the LUT from the map's own direct decode
    (``scene_builder._append_u8_lut``), so both layouts hand this function's
    callers the same bits up to torch-CPU's one-ulp SIMD-tail residue. Glow
    is 0 by the u8 map's admission rule (``texture_u8_ok``).
    """
    color = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
    alpha = 0.0
    if lut_base < 0:
        abs_idx = ti.math.clamp(base_row + texel_idx, 0, num_points - 1)
        color = ti.math.vec4(textures[tc, abs_idx, 0],
                             textures[tc, abs_idx, 1],
                             textures[tc, abs_idx, 2],
                             textures[tc, abs_idx, 3])
        alpha = textures[tc, abs_idx, 4]
    else:
        lane = base_row * 5 + texel_idx
        row = ti.math.clamp(lane // 5, 0, num_points - 1)
        ch = lane - 5 * (lane // 5)
        bits = ti.bit_cast(textures[tc, row, ch], ti.u32)
        rb = ti.cast(bits & ti.u32(0xFF), ti.i32)
        gb = ti.cast((bits >> 8) & ti.u32(0xFF), ti.i32)
        bb = ti.cast((bits >> 16) & ti.u32(0xFF), ti.i32)
        ab = ti.cast((bits >> 24) & ti.u32(0xFF), ti.i32)
        color = ti.math.vec4(textures[tc, lut_base + rb, 0],
                             textures[tc, lut_base + gb, 0],
                             textures[tc, lut_base + bb, 0],
                             0.0)
        alpha = textures[tc, lut_base + ab, 1]
    return color, alpha


@ti.func
def _authored_texel(tc, offset, frame_texel_base, u8_packed, texel_idx,
                    num_points, textures: ti.template()):
    """One texel of an ENDPOINT stack, in authored space (texture_time_lerp).

    Endpoint stacks are stored pre-decode so the time lerp runs on authored
    values -- the order the dense path applies them (the timeline lerps
    authored texels, then the merge decodes). ``u8_packed`` says the stack is
    RGBA bytes bit-packed one texel per f32 lane (meta col 15 == -2): the
    bytes ARE the authored ``k / 255`` values, recovered by an IEEE division
    (bit-equal to the host's ``q / 255``), with glow 0 by the u8 admission
    rule. Otherwise five plain f32 channels at ``offset + frame_texel_base +
    texel_idx``, byte for byte as authored.
    """
    color = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
    alpha = 0.0
    if u8_packed == 0:
        abs_idx = ti.math.clamp(offset + frame_texel_base + texel_idx, 0,
                                num_points - 1)
        color = ti.math.vec4(textures[tc, abs_idx, 0],
                             textures[tc, abs_idx, 1],
                             textures[tc, abs_idx, 2],
                             textures[tc, abs_idx, 3])
        alpha = textures[tc, abs_idx, 4]
    else:
        lane = offset * 5 + frame_texel_base + texel_idx
        row = ti.math.clamp(lane // 5, 0, num_points - 1)
        ch = lane - 5 * (lane // 5)
        bits = ti.bit_cast(textures[tc, row, ch], ti.u32)
        rb = ti.cast(bits & ti.u32(0xFF), ti.f32)
        gb = ti.cast((bits >> 8) & ti.u32(0xFF), ti.f32)
        bb = ti.cast((bits >> 16) & ti.u32(0xFF), ti.f32)
        ab = ti.cast((bits >> 24) & ti.u32(0xFF), ti.f32)
        color = ti.math.vec4(rb / 255.0, gb / 255.0, bb / 255.0, 0.0)
        alpha = ab / 255.0
    return color, alpha


@ti.func
def _sample_texture(f, u, v, prim_uv_index, tri_tex_meta: ti.template(), textures: ti.template()):
    offset = tri_tex_meta[prim_uv_index, 0]
    width = tri_tex_meta[prim_uv_index, 1]
    height = tri_tex_meta[prim_uv_index, 2]
    # Per-map time length (meta col 10): under texture_time_flat a map's
    # frames sit consecutively along the texel axis, so frame f of the map
    # starts at ``offset + (f % t) * w * h`` while the buffer's own time axis
    # stays at length 1. With the legacy shared axis t == 1 and this is the
    # old addressing exactly.
    tmap = ti.max(tri_tex_meta[prim_uv_index, 10], 1)
    # u8-packed storage marker (meta col 15, -1 = plain f32 rows); see
    # ``_color_map_texel``.
    lut_base = tri_tex_meta[prim_uv_index, 15]
    # Endpoint-interpolation region (meta cols 16-17, texture_time_lerp):
    # the map is a stack of authored endpoint images and frame f blends
    # endpoints i0 and i1 by w, all read from one tiny bank row. Column 3 of
    # the row says whether the blended rgb still needs the linear-light
    # decode the merge skipped (linear_color_space).
    lerp_off = tri_tex_meta[prim_uv_index, 16]
    lerp_i0 = 0
    lerp_i1 = 0
    lerp_w = 0.0
    lerp_dec = 0.0

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
    wi = ti.cast(width, ti.i32)
    hi = ti.cast(height, ti.i32)
    frame_base = (f % tmap) * (wi * hi)
    lerp_u8 = 0
    if lerp_off >= 0:
        lerp_len = ti.max(tri_tex_meta[prim_uv_index, 17], 1)
        lrow = lerp_off + (f % lerp_len)
        lerp_i0 = ti.cast(textures[tc, lrow, 0], ti.i32)
        lerp_i1 = ti.cast(textures[tc, lrow, 1], ti.i32)
        lerp_w = textures[tc, lrow, 2]
        lerp_dec = textures[tc, lrow, 3]
        if lut_base != -1:
            # -2 = packed bytes with no LUT (see _authored_texel).
            lerp_u8 = 1

    for corner in ti.static(range(4)):
        cx = ti.cast(x_floor + (corner % 2), ti.i32)
        cy = ti.cast(y_floor + (corner // 2), ti.i32)
        w = (xr if (corner % 2) == 1 else 1.0 - xr) * (
            yr if (corner // 2) == 1 else 1.0 - yr)

        cx = ti.math.clamp(cx, 0, ti.cast(width - 1.0, ti.i32))
        cy = ti.math.clamp(cy, 0, ti.cast(height - 1.0, ti.i32))

        local_idx = cx * hi + cy
        c = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
        a = 0.0
        if lerp_off < 0:
            c, a = _color_map_texel(tc, offset + frame_base, lut_base,
                                    local_idx, num_points, textures)
        else:
            # Endpoint blend in AUTHORED space, then decode, then the
            # bilinear accumulate below -- the dense path's own order
            # (timeline lerp, merge decode, per-texel bilinear).
            c0, a0 = _authored_texel(tc, offset, lerp_i0 * (wi * hi),
                                     lerp_u8, local_idx, num_points,
                                     textures)
            c1, a1 = _authored_texel(tc, offset, lerp_i1 * (wi * hi),
                                     lerp_u8, local_idx, num_points,
                                     textures)
            c = c0 + lerp_w * (c1 - c0)
            a = a0 + lerp_w * (a1 - a0)
            if lerp_dec > 0.5:
                c[0] = srgb_to_linear_f(c[0])
                c[1] = srgb_to_linear_f(c[1])
                c[2] = srgb_to_linear_f(c[2])

        color += w * c
        alpha += w * a
        sum_w += w

    color /= ti.max(sum_w, 1e-6)
    alpha /= ti.max(sum_w, 1e-6)
    # In-sampler opacity multiply (texture_opacity_in_kernel): the mob's
    # animated opacity rides the bank as a tiny per-map region (meta col 13 =
    # its base row, col 14 = its frame count) instead of being premultiplied
    # into the map's coverage channel on the host -- which is what lets a fade
    # of a static image keep a one-frame map. Legacy premultiplied maps carry
    # op_off = -1 and skip the multiply, restoring the old values byte for
    # byte.
    op_off = tri_tex_meta[prim_uv_index, 13]
    if op_off >= 0:
        op_len = ti.max(tri_tex_meta[prim_uv_index, 14], 1)
        alpha *= textures[tc, op_off + (f % op_len), 0]
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
def _shadow_pass_through(f, prim, hit_type, w0, w1, w2,
                         tri_extra: ti.template(),
                         circuit_meta: ti.template(), tint):
    """Fraction of the light a *covered* surface still passes to a shadow ray,
    per RGB channel.

    The march multiplies by ``1 - alpha`` for the part of the pixel a surface
    does not cover; this is what the covered part does with the light, which
    for anything transmissive is most of it. Without it a wine glass and a
    brick are the same object to a shadow ray -- measured against a path
    tracer, a clear glass sphere's shadow was as dark as an opaque sphere's
    (1% of the open floor, where the reference passes ~100%).

        pass = transmission * (1 - metalness) * (1 - F0)   [* albedo]

    ``metalness < 0`` is the non-PBR sentinel and passes nothing, matching the
    old behaviour exactly. The metal share never transmits, the same gate
    ``_material_reflectance`` applies via ``diel_pass``.

    F0 is the *normal-incidence* dielectric reflectance, not an angle-dependent
    Fresnel: the march has no normals and fetching them would cost the hottest
    loop in the renderer for a second-order effect. It is also less of an
    approximation than it looks -- a solid presents the march two surfaces
    (entry and exit), each taking its own ``1 - F0``, so a glass ball attenuates
    by 0.96^2 on its own.

    The albedo tint (the bracketed factor above) is new with the RGB payload
    and gated behind :func:`rgb_shadow_tint`: it matches what the bounce loop
    does to its transmitted share (``trans_w = trans_energy * tint`` in
    ``_scatter_impl``, with ``tint = clamp(albedo, 0, 1)``), so light through
    green glass now arrives green instead of grey. ``tint`` arrives from the
    caller's color fetch -- the march reads the surface color row anyway for
    its alpha, so this reuses that fetch rather than issuing a second one.
    With the gate off the scalar pass-through is broadcast to all three
    channels unchanged, byte-identical to the pre-RGB renderer.

    One thing this still deliberately does not model:

    * **Refraction.** Real glass bends the light it passes, which is why a
      glass ball's shadow has a bright caustic core; a shadow ray that
      continues straight cannot produce one.

    So the honest description is "a transmissive surface stops blocking light
    and tints what it passes", not "glass casts a correct shadow".
    """
    metalness = -1.0
    ior = 0.0
    transmission = 0.0
    if hit_type == 1:
        # Same guard as ``_flat_triangle_extra``: a promoted constant-material
        # triangle sits past the shrunk ``tri_extra``, and has no per-vertex row.
        if prim < tri_extra.shape[1]:
            te = f % tri_extra.shape[0]
            metalness = (w0 * tri_extra[te, prim, 0]
                         + w1 * tri_extra[te, prim, 2]
                         + w2 * tri_extra[te, prim, 4])
            ior = (w0 * tri_extra[te, prim, 6]
                   + w1 * tri_extra[te, prim, 7]
                   + w2 * tri_extra[te, prim, 8])
            transmission = (w0 * tri_extra[te, prim, 9]
                            + w1 * tri_extra[te, prim, 10]
                            + w2 * tri_extra[te, prim, 11])
    else:
        cm = f % circuit_meta.shape[0]
        metalness = circuit_meta[cm, prim, _M_REFLECTIVITY]
        ior = circuit_meta[cm, prim, _M_IOR]
        transmission = circuit_meta[cm, prim, _M_TRANSMISSION]
    out = ti.math.vec3(0.0)
    if (metalness >= 0.0) and (transmission > 1e-4):
        m = ti.math.clamp(metalness, 0.0, 1.0)
        io = ti.abs(ior)
        f0 = 0.0
        if io > 1.0 + 1e-4:
            r0 = (1.0 - io) / (1.0 + io)
            f0 = r0 * r0
        # The scalar arithmetic is today's, unchanged; the payload just carries
        # it per channel, so an equal-channel input reduces exactly (and the
        # gate-off arm never touches ``tint``, keeping every render it gates
        # byte-identical).
        p = ti.math.clamp(transmission, 0.0, 1.0) * (1.0 - m) * (1.0 - f0)
        if ti.static(rgb_shadow_tint()):
            out = p * tint
        else:
            out = ti.math.vec3(p)
    return ti.math.clamp(out, 0.0, 1.0)


@ti.func
def _shadow_hit_sigma(f, prim, hit_type, tri_extra: ti.template()):
    """Per-primitive Beer-Lambert absorption coefficient at a shadow hit
    (tri_extra columns 12..14; see ``_pack_surface_extra``). Zero for circuits,
    which are zero-thickness panes with no interior to absorb over, and zero
    for anything whose material does not attenuate.

    Called only from the gated medium-pairing blocks in the shadow march and
    gather, so the fetch compiles out entirely when :func:`rgb_shadow_tint`
    is off.
    """
    sigma = ti.math.vec3(0.0, 0.0, 0.0)
    if hit_type == 1:
        # Same promoted-triangle guard as ``_shadow_pass_through``.
        if prim < tri_extra.shape[1]:
            te = f % tri_extra.shape[0]
            sigma = ti.math.vec3(
                tri_extra[te, prim, _EXTRA_SIGMA],
                tri_extra[te, prim, _EXTRA_SIGMA + 1],
                tri_extra[te, prim, _EXTRA_SIGMA + 2])
    return sigma


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
                   num_lights, albedo, shadows: ti.template(), vis,
                   cam_origin: ti.template()):
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
    shadow visibilities (used iff ``shadows``). ``cam_origin`` is the batch's
    per-frame camera-position table, read at the hit's frame and handed on as
    the stages' ``cam_pos`` (depth-style materials measure from the CAMERA,
    which no ray argument names once the ray has bounced).

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
    cam_pos = ti.math.vec3(cam_origin[f, 0], cam_origin[f, 1], cam_origin[f, 2])
    return _run_frag_pipeline(frag_pipelines, pids_present,
                              prim, f, pos, -rd, shade_normal,
                              face_n, rgb,
                              albedo[3], light_pos, light_col, num_lights,
                              tri_mat_id, tri_mat, shadows, vis, cam_pos)


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
                     src_sid, src_prim, eps_self, eps_near,
                     tri_obj: ti.template(), ident: ti.template(),
                     nocast: ti.template()):
    """Nearest surface of any geometry type strictly after
    (t_prev, layer_prev) along the ray. Geometry only -- shading data is
    fetched by the caller for the hits it actually uses.

    Returns ``(found, t_hit, layer, prim, hit_type, a, b, border,
    edge_hit)`` where ``hit_type`` is 0 for bezier circuits and 1 for
    triangles, and ``(a, b)`` are the barycentric ``(w1, w2)`` for triangle
    hits or the plane ``(u, v)`` for bezier hits; ``found == 0`` means the ray
    escapes the scene, ``edge_hit == 1`` flags a triangle hit on/near one of
    its edges (used to merge the duplicate hits of mesh seams).

    ``(src_sid, src_prim, eps_self, eps_near, tri_obj, ident)`` carry the shadow ray's source-surface
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
            tri_pos, src_sid, src_prim, eps_self, eps_near, tri_obj, ident,
            nocast)
    bt = 1e30
    b_circ = -1
    b_border = 0
    b_u = 0.0
    b_v = 0.0
    b_layer = -1e30
    if ti.static(has_bez != 0):
        bez_cap = t_cap
        if t_prim >= 0:
            bez_cap = ti.min(bez_cap, tt + depth_tie_epsilon)
        bt, b_circ, b_border, b_u, b_v, b_layer = _nearest_bezier_hit(
            refit, ro, rd, inv_rd, f, ff, t_prev, layer_prev, bez_cap,
            pixel_size_per_t, base_dist, b_nodes, b_node_miss, b_leaf_prim,
            b_leaf_tspan, b_first_leaf, circuit_meta, edges_2d, edge_accel,
            nocast)

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
    
    # Linear Rec.2020 -> linear Rec.709. Both spaces share the D65 white point,
    # so this must map white to white and every row must therefore sum to 1.
    # It was written transposed until 2026-08-22, giving row sums of
    # 1.5177 / 0.4447 / 1.0376 -- a fixed +52% red, -56% green on any neutral,
    # which rendered authored grey (128,128,128) as magenta (255,77,180).
    r_srgb = 1.6605 * r_out - 0.5876 * g_out - 0.0728 * b_out
    g_srgb = -0.1246 * r_out + 1.1329 * g_out - 0.0083 * b_out
    b_srgb = -0.0182 * r_out - 0.1006 * g_out + 1.1187 * b_out
    
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
        # Scale at the read rather than in place. `color_offset *= newPeak / peak`
        # here is dropped outright when Taichi compiles this func with
        # advanced_optimization on -- the mixes below then read the unscaled
        # value and an authored white tonemaps to 244 instead of 222. Algan runs
        # with that pass off so released renders never saw it, but ALGAN_ADV_OPT=1
        # and any bare `ti.init` do turn it on. Bit-identical to the in-place
        # form under Algan's own config, measured over 250k random colors.
        scale = newPeak / peak

        g = 1.0 - 1.0 / (desaturation * (peak - newPeak) + 1.0)
        out[0] = ti.math.mix(color_offset[0] * scale, newPeak, g)
        out[1] = ti.math.mix(color_offset[1] * scale, newPeak, g)
        out[2] = ti.math.mix(color_offset[2] * scale, newPeak, g)

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
        -1, -1, 0.0, 0.0, tri_pos, 0, 0)


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
                   src_sid, src_prim, eps_self, eps_near,
                   tri_obj: ti.template(), ident: ti.template(),
                   nocast: ti.template()) -> ti.i32:
    """Gather the up-to-``kbuf`` nearest hits strictly after
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
    ``(src_sid, src_prim, eps_self, eps_near, tri_obj, ident)`` carry the shadow ray's source-surface
    identity for :func:`_shadow_identity_t_min` (triangle arm only; the
    bezier arm keeps the classic epsilon). See that function for the sentinel
    convention.
    Returns the number of hits gathered. When the return value is smaller
    than ``kbuf``, the buffer provably contains *every* remaining hit along
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
        window_hi = worst_t + depth_tie_epsilon if count == kbuf else 1e30
        window_hi = ti.min(window_hi, opq_t + depth_tie_epsilon)
        g_sp = 0
        g_st = ti.Vector([0] * _GROUP_STACK)
        g_cur = 0
        g_pend, g_near = _group_test(
            refit, t_row0, 0, f, ro, inv_rd, t_prev - depth_tie_epsilon,
            window_hi, t_nodes)
        while True:
            if g_pend == 0:
                if g_sp == 0:
                    break
                g_sp -= 1
                saved = g_st[g_sp]
                g_cur = saved >> bvh_arity
                saved_mask = saved & _GROUP_MASK
                window_hi = (worst_t + depth_tie_epsilon
                             if count == kbuf else 1e30)
                window_hi = ti.min(window_hi, opq_t + depth_tie_epsilon)
                fresh_mask, g_near = _group_test(
                    refit, t_row0, g_cur, f, ro, inv_rd,
                    t_prev - depth_tie_epsilon, window_hi, t_nodes)
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
                        if ti.static(nocast != 0):
                            if (w & _REFIT_NOCAST_BIT) != 0:
                                l_prim = -1
                        l_opq = (w >> 30) & 1
                else:
                    g_child = bvh_arity * g_cur + 1 + g_c
                    if g_child >= t_first_leaf:
                        l_base = (g_child - t_first_leaf) * bvh_leaf_size
                    else:
                        descend = 1
                        child_blk = g_child
                if descend == 0:
                    for j in ti.static(
                            range(1 if refit != 0 else bvh_leaf_size)):
                        prim = l_prim
                        opq = l_opq
                        if ti.static(refit == 0):
                            prim = -1
                            p0 = t_leaf_prim[l_base + j]
                            tspan = t_leaf_tspan[l_base + j]
                            if ((p0 >= 0) and ((tspan & 0x7FFF) <= f)
                                    and (f <= ((tspan >> 16) & 0x7FFF))):
                                prim = p0
                                if ti.static(nocast != 0):
                                    if (tspan & _LEAF_NOCAST_BIT) != 0:
                                        prim = -1
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
                                    f, prim, src_sid, src_prim, eps_self, eps_near, tri_obj, ident))
                                          and _comes_after(
                                              t, layer, t_prev,
                                              layer_prev)
                                          and not _comes_after(
                                              t, layer, opq_t,
                                              opq_layer))
                                if accept and (count == kbuf):
                                    accept = _comes_after(
                                        worst_t, worst_layer, t, layer)
                                if accept:
                                    slot = worst_idx
                                    if count < kbuf:
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
                                    if count == kbuf:
                                        worst_idx = 0
                                        worst_t = hit_t[0]
                                        worst_layer = hit_layer[0]
                                        for q in ti.static(
                                                range(1, kbuf)):
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
                        g_st[g_sp] = (g_cur << bvh_arity) | g_pend
                        g_sp += 1
                    g_cur = child_blk
                    window_hi = worst_t + depth_tie_epsilon \
                        if count == kbuf else 1e30
                    window_hi = ti.min(window_hi, opq_t + depth_tie_epsilon)
                    g_pend, g_near = _group_test(
                        refit, t_row0, g_cur, f, ro, inv_rd,
                        t_prev - depth_tie_epsilon, window_hi, t_nodes)

    # --- Bezier BVH (window tightened by the triangle hits) ---
    if ti.static(has_bez != 0):
        num_meta_frames = circuit_meta.shape[0]
        num_edge_frames = edges_2d.shape[0]
        b_row0 = 0
        if ti.static(refit != 0):
            b_row0 = _refit_row0(f, b_first_leaf, b_nodes)
        window_hi = worst_t + depth_tie_epsilon if count == kbuf else 1e30
        window_hi = ti.min(window_hi, opq_t + depth_tie_epsilon)
        g_sp = 0
        g_st = ti.Vector([0] * _GROUP_STACK)
        g_cur = 0
        g_pend, g_near = _group_test(
            refit, b_row0, 0, f, ro, inv_rd, t_prev - depth_tie_epsilon,
            window_hi, b_nodes)
        while True:
            if g_pend == 0:
                if g_sp == 0:
                    break
                g_sp -= 1
                saved = g_st[g_sp]
                g_cur = saved >> bvh_arity
                saved_mask = saved & _GROUP_MASK
                window_hi = (worst_t + depth_tie_epsilon
                             if count == kbuf else 1e30)
                window_hi = ti.min(window_hi, opq_t + depth_tie_epsilon)
                fresh_mask, g_near = _group_test(
                    refit, b_row0, g_cur, f, ro, inv_rd,
                    t_prev - depth_tie_epsilon, window_hi, b_nodes)
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
                        if ti.static(nocast != 0):
                            if (w & _REFIT_NOCAST_BIT) != 0:
                                l_prim = -1
                        l_opq = (w >> 30) & 1
                else:
                    g_child = bvh_arity * g_cur + 1 + g_c
                    if g_child >= b_first_leaf:
                        l_base = (g_child - b_first_leaf) * bvh_leaf_size
                    else:
                        descend = 1
                        child_blk = g_child
                if descend == 0:
                    for j in ti.static(
                            range(1 if refit != 0 else bvh_leaf_size)):
                        circuit = l_prim
                        opq = l_opq
                        if ti.static(refit == 0):
                            circuit = -1
                            p0 = b_leaf_prim[l_base + j]
                            tspan = b_leaf_tspan[l_base + j]
                            if ((p0 >= 0) and ((tspan & 0x7FFF) <= f)
                                    and (f <= ((tspan >> 16) & 0x7FFF))):
                                circuit = p0
                                if ti.static(nocast != 0):
                                    if (tspan & _LEAF_NOCAST_BIT) != 0:
                                        circuit = -1
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
                                accept = ((t > min_hit_distance)
                                          and _comes_after(t, layer, t_prev,
                                                           layer_prev)
                                          and not _comes_after(
                                              t, layer, opq_t, opq_layer))
                                if accept and (count == kbuf):
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
                                    centered = (circuit_meta[
                                        tm, circuit, _M_BORDER_CENTERED] > 0.5)
                                    query_radius = _circuit_query_radius(
                                        border_w, outline_w, filled, centered)
                                    te = f % num_edge_frames
                                    (crossings, min_dist_sq, _ccu, _ccv, _e1x,
                                     _e1y, _sg1, _s2, _s2u, _s2v, _e2x, _e2y,
                                     _sg2) = _bezier_point_metrics(
                                        circuit, te, u, v, query_radius,
                                        circuit_meta.shape[1], edges_2d, edge_accel)
                                    inside, in_border = _circuit_point_region(
                                        border_w, outline_w, filled, centered,
                                        crossings, min_dist_sq)
                                    if inside:
                                        slot = worst_idx
                                        if count < kbuf:
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
                                        if count == kbuf:
                                            worst_idx = 0
                                            worst_t = hit_t[0]
                                            worst_layer = hit_layer[0]
                                            for q in ti.static(range(1, kbuf)):
                                                if _comes_after(hit_t[q],
                                                                hit_layer[q],
                                                                worst_t,
                                                                worst_layer):
                                                    worst_idx = q
                                                    worst_t = hit_t[q]
                                                    worst_layer = hit_layer[q]
                else:
                    if g_pend != 0:
                        g_st[g_sp] = (g_cur << bvh_arity) | g_pend
                        g_sp += 1
                    g_cur = child_blk
                    window_hi = worst_t + depth_tie_epsilon \
                        if count == kbuf else 1e30
                    window_hi = ti.min(window_hi, opq_t + depth_tie_epsilon)
                    g_pend, g_near = _group_test(
                        refit, b_row0, g_cur, f, ro, inv_rd,
                        t_prev - depth_tie_epsilon, window_hi, b_nodes)
    return count


@ti.func
def _anyhit_opaque_tri(refit: ti.template(), ro, rd, inv_rd, f, t_lo, max_t,
                       nodes: ti.template(), leaf_prim: ti.template(),
                       leaf_tspan: ti.template(), first_leaf,
                       tri_pos: ti.template(),
                       src_sid, src_prim, eps_self, eps_near,
                       tri_obj: ti.template(),
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
        max_t + depth_tie_epsilon, nodes)
    while hit == 0:
        if g_pend == 0:
            if g_sp == 0:
                break
            g_sp -= 1
            saved = g_st[g_sp]
            g_cur = saved >> bvh_arity
            saved_mask = saved & _GROUP_MASK
            fresh_mask, g_near = _group_test(
                refit, row0, g_cur, f, ro, inv_rd, t_lo,
                max_t + depth_tie_epsilon, nodes)
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
                    if (w & _REFIT_NOCAST_BIT) != 0:
                        l_prim = -1
            else:
                g_child = bvh_arity * g_cur + 1 + g_c
                if g_child >= first_leaf:
                    l_base = (g_child - first_leaf) * bvh_leaf_size
                else:
                    descend = 1
                    child_blk = g_child
            if descend == 0:
                for j in ti.static(range(1 if refit != 0 else bvh_leaf_size)):
                    prim = l_prim
                    if ti.static(refit == 0):
                        prim = -1
                        p0 = leaf_prim[l_base + j]
                        tspan = leaf_tspan[l_base + j]
                        # Bit 31 (sign) flags interval-opaque instances.
                        if ((p0 >= 0) and (tspan < 0)
                                and ((tspan & 0x7FFF) <= f)
                                and (f <= ((tspan >> 16) & 0x7FFF))):
                            prim = p0
                            if (tspan & _LEAF_NOCAST_BIT) != 0:
                                prim = -1
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
                                    f, prim, src_sid, src_prim, eps_self, eps_near, tri_obj, ident)) \
                                    and (t < max_t):
                                hit = 1
            else:
                if g_pend != 0:
                    g_st[g_sp] = (g_cur << bvh_arity) | g_pend
                    g_sp += 1
                g_cur = child_blk
                g_pend, g_near = _group_test(
                    refit, row0, g_cur, f, ro, inv_rd, t_lo,
                    max_t + depth_tie_epsilon, nodes)
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
        max_t + depth_tie_epsilon, nodes)
    while hit == 0:
        if g_pend == 0:
            if g_sp == 0:
                break
            g_sp -= 1
            saved = g_st[g_sp]
            g_cur = saved >> bvh_arity
            saved_mask = saved & _GROUP_MASK
            fresh_mask, g_near = _group_test(
                refit, row0, g_cur, f, ro, inv_rd, t_lo,
                max_t + depth_tie_epsilon, nodes)
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
                    if (w & _REFIT_NOCAST_BIT) != 0:
                        l_prim = -1
            else:
                g_child = bvh_arity * g_cur + 1 + g_c
                if g_child >= first_leaf:
                    l_base = (g_child - first_leaf) * bvh_leaf_size
                else:
                    descend = 1
                    child_blk = g_child
            if descend == 0:
                for j in ti.static(range(1 if refit != 0 else bvh_leaf_size)):
                    circuit = l_prim
                    if ti.static(refit == 0):
                        circuit = -1
                        p0 = leaf_prim[l_base + j]
                        tspan = leaf_tspan[l_base + j]
                        if ((p0 >= 0) and (tspan < 0)
                                and ((tspan & 0x7FFF) <= f)
                                and (f <= ((tspan >> 16) & 0x7FFF))):
                            circuit = p0
                            if (tspan & _LEAF_NOCAST_BIT) != 0:
                                circuit = -1
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
                            if (t > min_hit_distance) and (t < max_t):
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
                                centered = (circuit_meta[
                                    tm, circuit, _M_BORDER_CENTERED] > 0.5)
                                query_radius = _circuit_query_radius(
                                    border_w, outline_w, filled, centered)
                                te = f % num_edge_frames
                                (crossings, min_dist_sq, _ccu, _ccv, _e1x,
                                     _e1y, _sg1, _s2, _s2u, _s2v, _e2x, _e2y,
                                     _sg2) = _bezier_point_metrics(
                                    circuit, te, u, v, query_radius,
                                    circuit_meta.shape[1], edges_2d,
                                    edge_accel)
                                inside, in_border = _circuit_point_region(
                                    border_w, outline_w, filled, centered,
                                    crossings, min_dist_sq)
                                if inside:
                                    hit = 1
            else:
                if g_pend != 0:
                    g_st[g_sp] = (g_cur << bvh_arity) | g_pend
                    g_sp += 1
                g_cur = child_blk
                g_pend, g_near = _group_test(
                    refit, row0, g_cur, f, ro, inv_rd, t_lo,
                    max_t + depth_tie_epsilon, nodes)
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
                          src_sid, src_prim, eps_self, eps_near,
                          tri_obj: ti.template(),
                          ident: ti.template()) -> ti.i32:
    """1 if any interval-opaque primitive of any geometry type blocks the
    shadow ray before ``max_t``. Trees are tried triangle -> bezier, the
    second skipped entirely on a hit in the first. ``t_lo`` prunes the
    node-visit window only (see :func:`_anyhit_opaque_tri`).

    ``(src_sid, src_prim, eps_self, eps_near, tri_obj, ident)`` carry the shadow ray's source-surface
    identity into the triangle arm; a circuit blocker keeps the classic
    epsilon (circuits have no per-triangle identity).
    """
    hit = 0
    if ti.static(has_tri != 0):
        hit = _anyhit_opaque_tri(refit, ro, rd, inv_rd, f, t_lo, max_t,
                                 t_nodes, t_leaf_prim, t_leaf_tspan,
                                 t_first_leaf, tri_pos,
                                 src_sid, src_prim, eps_self, eps_near, tri_obj, ident)
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
                     tri_extra: ti.template(),
                     num_colored_triangles: ti.i32,
                     b_nodes: ti.template(), b_node_miss: ti.template(),
                     b_leaf_prim: ti.template(), b_leaf_tspan: ti.template(),
                     b_first_leaf, circuit_meta: ti.template(),
                     circuit_colors: ti.template(),
                     circuit_border_colors: ti.template(),
                     edges_2d: ti.template(), edge_accel: ti.template(),
                     src_sid, src_prim, eps_self, eps_near,
                     tri_obj: ti.template(), ident: ti.template()):
    """Fraction of light occluded along a deterministic shadow ray, per RGB
    channel (a transmissive blocker can dim them unequally; an opaque one is
    1 in all three).

    Every surface between the shaded point and the light attenuates the
    remaining light by its opacity, matching the physical path tracer's
    transmittance calculation. A fully opaque hit exits immediately. Mesh
    seams still merge their duplicate edge hit so a thin surface cannot
    attenuate twice along a shared edge.

    ``(src_sid, src_prim, eps_self, eps_near, tri_obj, ident)`` carry the shadow ray's source-surface
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
    ``depth_tie_epsilon``, and an opaque blocker past
    ``max_surfaces_per_ray`` peeled surfaces -- in both the any-hit's full
    occlusion is the physically correct answer. 4 replaces the march with
    :func:`_shadow_gather_occluded`, the same peel rebuilt on the kbuf
    gather (one traversal per kbuf surfaces instead of per surface).
    """
    inv_rd = ti.math.vec3(_safe_inverse(rd[0]), _safe_inverse(rd[1]),
                          _safe_inverse(rd[2]))
    occluded = ti.math.vec3(0.0)
    if ti.static(anyhit == 3):
        # The any-hit answer is binary, so its RGB payload is that one bit
        # broadcast: a blocker blocks every channel (which is exactly what the
        # old scalar 0/1 meant once a payload existed to carry it).
        opaque_hit = _shadow_anyhit_opaque(
            refit, has_tri, has_bez, ro, rd, inv_rd, f,
            -depth_tie_epsilon, max_t,
            pixel_size_per_t, base_dist,
            t_nodes, t_leaf_prim, t_leaf_tspan, t_first_leaf, tri_pos,
            b_nodes, b_leaf_prim, b_leaf_tspan, b_first_leaf,
            circuit_meta, edges_2d, edge_accel,
            src_sid, src_prim, eps_self, eps_near, tri_obj, ident)
        occluded = ti.math.vec3(ti.cast(opaque_hit, ti.f32))
    else:
        if ti.static(anyhit == 4):
            occluded = _shadow_gather_occluded(
                refit, ro, rd, inv_rd, f, ff, max_t,
                pixel_size_per_t, base_dist, layer_offset_triangles,
                has_tri, has_bez,
                t_nodes, t_node_miss, t_leaf_prim, t_leaf_tspan,
                t_first_leaf, tri_pos, tri_colors, tri_uvs, tri_tex_meta,
                textures, tri_extra, num_colored_triangles,
                b_nodes, b_node_miss, b_leaf_prim, b_leaf_tspan,
                b_first_leaf, circuit_meta, circuit_colors,
                circuit_border_colors, edges_2d, edge_accel,
                src_sid, src_prim, eps_self, eps_near, tri_obj, ident)
        else:
            occluded = _shadow_march_occluded(
                refit, anyhit, ro, rd, inv_rd, f, ff, max_t,
                pixel_size_per_t, base_dist, layer_offset_triangles,
                has_tri, has_bez,
                t_nodes, t_node_miss, t_leaf_prim, t_leaf_tspan,
                t_first_leaf, tri_pos, tri_colors, tri_uvs, tri_tex_meta,
                textures, tri_extra, num_colored_triangles,
                b_nodes, b_node_miss, b_leaf_prim, b_leaf_tspan,
                b_first_leaf, circuit_meta, circuit_colors,
                circuit_border_colors, edges_2d, edge_accel,
                src_sid, src_prim, eps_self, eps_near, tri_obj, ident)
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
                           tri_extra: ti.template(),
                           num_colored_triangles: ti.i32,
                           b_nodes: ti.template(), b_node_miss: ti.template(),
                           b_leaf_prim: ti.template(),
                           b_leaf_tspan: ti.template(),
                           b_first_leaf, circuit_meta: ti.template(),
                           circuit_colors: ti.template(),
                           circuit_border_colors: ti.template(),
                           edges_2d: ti.template(),
                           edge_accel: ti.template(),
                           src_sid, src_prim, eps_self, eps_near,
                           tri_obj: ti.template(),
                           ident: ti.template()):
    """The classic ordered closest-hit shadow march (the pre-any-hit body of
    :func:`_shadow_occluded`, byte-identical at ``anyhit`` 0/1; 2 adds the
    deferred opaque any-hit early-out documented there).

    The payload is RGB: each channel carries its own transmittance, so a
    colored transmissive blocker tints what it passes. With every channel
    equal -- which is exactly the pre-RGB world, and always the case while
    :func:`rgb_shadow_tint` is off -- the per-channel arithmetic is today's
    scalar arithmetic, operation for operation.

    Under the same gate, the march applies Beer-Lambert absorption over the
    chord a ray spends inside a solid, from the per-primitive sigma in
    ``tri_extra`` (see :func:`_shadow_hit_sigma`). The march has no normals,
    so it cannot use the bounce loop's ``rd . n > 0`` exit test; it pairs
    hits instead: the first fully-covering attenuating hit OPENS a medium
    (recording sigma and entry depth) and the next one CLOSES it, multiplying
    the payload by ``exp(-sigma * (t_exit - t_entry))``. That is exact for a
    single convex solid -- entry then exit, one chord -- which is the same
    guarantee the view path's absorption makes. Two solids along one ray pair
    nearest-entry with next-hit (the second solid's entry closes the first's
    medium), and a medium still open when the ray retires loses its trailing
    absorption; both are approximations, stated here rather than hidden.
    Only FULL-COVERAGE hits participate (``_SOLID_COVERAGE_MIN``): coverage
    and transmission are independent channels -- the glass ball this is for is
    alpha 1 with transmission 1 -- so requiring full coverage keeps half-faded
    panes, which are not solids, from opening a medium their far side may never
    close. That floor carries a one-ulp tolerance for a reason the constant's
    own comment gives: an exact ``>= 1.0`` speckles every colored shadow,
    because barycentric alpha misses 1.0 by an ulp often enough to drop whole
    chords at random.
    """
    transmitted = ti.math.vec3(1.0)
    one3 = ti.math.vec3(1.0)
    # Interior-absorption state: three floats plus the open flag. Every READ
    # sits behind the compile-time rgb_shadow_tint gate below, so with the
    # gate off nothing here is ever touched and the values compile away;
    # declared unconditionally because Taichi does not carry a name CREATED
    # inside one ti.static block into another.
    medium_open = 0
    medium_sigma = ti.math.vec3(0.0)
    medium_t_entry = 0.0
    t_prev = 0.0
    layer_prev = 1e30
    seam_t = -1e30
    step = 0
    behind_checked = 0
    while step < max_surfaces_per_ray:
        step += 1
        # Cap the walk at the light: t_cap only tightens the node-visit
        # window to min(best_t, t_cap) + depth_tie_epsilon (hit acceptance is
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
            src_sid, src_prim, eps_self, eps_near, tri_obj, ident,
            # A shadow ray: reject leaves whose primitive casts no shadow.
            1)
        if (found == 0) or (t_hit >= max_t):
            break
        seam_eps = depth_tie_epsilon
        if (edge_hit == 1) and (t_hit - seam_t <= seam_eps):
            t_prev = t_hit
            layer_prev = hit_layer
            continue
        seam_t = t_hit if edge_hit == 1 else -1e30
        # Color + alpha from ONE fetch: alpha gates the covered share, and
        # the RGB (clamped, like ``_scatter_impl``'s tint) is what a
        # transmissive surface tints the light it passes with. This replaces
        # the alpha-only helpers, whose loads this subsumes -- no extra array
        # access, wider rows on the textured paths (see the report).
        color4 = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
        alpha = 0.0
        if hit_type == 1:
            color4, alpha = _flat_triangle_color(
                f, prim, 1.0 - a - b, a, b, tri_colors, tri_uvs, tri_tex_meta,
                textures, num_colored_triangles)
        else:
            color4, alpha = _sample_circuit_color(
                prim, f, a, b, border, circuit_meta, circuit_colors,
                circuit_border_colors)
        alpha = ti.math.clamp(alpha, 0.0, 1.0)
        tint = ti.math.clamp(
            ti.math.vec3(color4[0], color4[1], color4[2]), 0.0, 1.0)
        if ti.static(rgb_shadow_tint()):
            # Beer-Lambert over the interior chord (docstring): a hit with
            # full coverage AND a non-zero sigma either closes the open
            # medium or opens one. Absorption multiplies before this hit's
            # own interface factor, matching ray order -- the chord ends AT
            # the exit surface, whose pass-through applies to what leaves it.
            sigma = _shadow_hit_sigma(f, prim, hit_type, tri_extra)
            if (alpha >= _SOLID_COVERAGE_MIN) \
                    and (_vis_max_component(sigma) > 0.0):
                if medium_open == 1:
                    seg = ti.max(t_hit - medium_t_entry, 0.0)
                    transmitted *= ti.math.vec3(
                        ti.exp(-medium_sigma[0] * seg),
                        ti.exp(-medium_sigma[1] * seg),
                        ti.exp(-medium_sigma[2] * seg))
                    medium_open = 0
                else:
                    medium_open = 1
                    medium_sigma = sigma
                    medium_t_entry = t_hit
        # What the covered part of the surface passes -- 0 for anything
        # opaque, so this is byte-identical wherever nothing transmits (see
        # ``_shadow_pass_through``, and the host gate in ``tracer.py`` that
        # keeps a transmissive batch off the any-hit modes, which answer a
        # question this makes no longer equivalent).
        passed = _shadow_pass_through(f, prim, hit_type, 1.0 - a - b, a, b,
                                      tri_extra, circuit_meta, tint)
        transmitted *= one3 - alpha * (one3 - passed)
        # Scalar early-outs become max-component tests: reducing a color
        # weight to its maximum component is this codebase's convention
        # (``_scatter_impl``, see shading_taichi._vis_max_component), and the
        # max of equal channels IS the old scalar, so both tests fire at the
        # exact same steps they always did.
        if _vis_max_component(transmitted) <= min_alpha:
            transmitted = ti.math.vec3(0.0)
            break
        if (alpha >= 1.0) and (_vis_max_component(passed) <= 0.0):
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
                        t_hit - depth_tie_epsilon, max_t,
                        pixel_size_per_t, base_dist,
                        t_nodes, t_leaf_prim, t_leaf_tspan, t_first_leaf,
                        tri_pos,
                        b_nodes, b_leaf_prim, b_leaf_tspan, b_first_leaf,
                        circuit_meta, edges_2d, edge_accel,
                        src_sid, src_prim, eps_self, eps_near, tri_obj, ident) == 1:
                    transmitted = ti.math.vec3(0.0)
                    break
        t_prev = t_hit
        layer_prev = hit_layer
    return one3 - transmitted


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
                            tri_extra: ti.template(),
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
                            src_sid, src_prim, eps_self, eps_near,
                            tri_obj: ti.template(),
                            ident: ti.template()):
    """The ordered shadow march rebuilt on the kbuf gather (shadow mode 4).

    Where :func:`_shadow_march_occluded` restarts a full two-tree
    (triangle + Bezier) traversal per peeled surface, each traversal here
    gathers the up-to-
    ``kbuf`` nearest hits with :func:`_collect_hits` and drains them in the
    same transitive :func:`_comes_after` order the march peels in, with the
    identical seam merge, alpha accumulation and early exits. A k-surface
    translucent stack therefore costs ``ceil((k+1)/kbuf)`` traversals
    instead of ``k+1``, while an all-opaque blocked ray stays at one (its
    first buffer opens with an interval-opaque hit whose alpha is 1).

    The light cap rides in as the gather's initial opaque window
    (``initial_opq_t = max_t``): node-visit windows close at ``max_t`` +
    ``depth_tie_epsilon`` exactly like the march's ``t_cap``, and
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
    transmitted = ti.math.vec3(1.0)
    one3 = ti.math.vec3(1.0)
    # The march's interior-absorption state, declared OUTSIDE both loops: a
    # medium may open in one gather window and close in a later one, so it
    # must survive across kbuf refills (see the march docstring for the
    # pairing rule and its single-convex-solid guarantee). Every READ sits
    # behind the compile-time gate; see the march's note on why the
    # declaration itself is unconditional.
    medium_open = 0
    medium_sigma = ti.math.vec3(0.0)
    medium_t_entry = 0.0
    t_prev = 0.0
    layer_prev = 1e30
    seam_t = -1e30
    step = 0
    alive = 1
    while (alive == 1) and (step < max_surfaces_per_ray):
        kb_t = ti.Vector([0.0] * kbuf)
        kb_layer = ti.Vector([0.0] * kbuf)
        kb_prim = ti.Vector([0] * kbuf)
        kb_flags = ti.Vector([0] * kbuf)
        kb_a = ti.Vector([0.0] * kbuf)
        kb_b = ti.Vector([0.0] * kbuf)
        num_hits = _collect_hits(
            refit, ro, rd, inv_rd, f, ff, t_prev, layer_prev,
            pixel_size_per_t, base_dist, layer_offset_triangles,
            kb_t, kb_layer, kb_prim, kb_flags, kb_a, kb_b,
            t_nodes, t_node_miss, t_leaf_prim, t_leaf_tspan, t_first_leaf,
            tri_pos,
            b_nodes, b_node_miss, b_leaf_prim, b_leaf_tspan, b_first_leaf,
            circuit_meta, edges_2d, edge_accel, has_tri, has_bez,
            max_t, -1e30,
            src_sid, src_prim, eps_self, eps_near, tri_obj, ident,
            # A shadow ray: reject leaves whose primitive casts no shadow.
            1)
        if num_hits == 0:
            alive = 0
        drained = 0
        while (alive == 1) and (drained < num_hits) \
                and (step < max_surfaces_per_ray):
            step += 1
            # Nearest unconsumed slot, scalar-tracked with ti.static
            # selects so the kb_* vectors are never dynamically indexed
            # (a dynamic vector index spills the whole vector to local
            # memory -- see the wavefront_shade drain).
            sel = 0
            sel_found = 0
            t_hit = 0.0
            hit_layer = 0.0
            for q in ti.static(range(kbuf)):
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
            for q in ti.static(range(kbuf)):
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
                seam_eps = depth_tie_epsilon
                if (edge_hit == 1) and (t_hit - seam_t <= seam_eps):
                    t_prev = t_hit
                    layer_prev = hit_layer
                else:
                    seam_t = t_hit if edge_hit == 1 else -1e30
                    # Color + alpha from one fetch, exactly like the march.
                    color4 = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
                    alpha = 0.0
                    if hit_type == 1:
                        color4, alpha = _flat_triangle_color(
                            f, prim, 1.0 - a - b, a, b, tri_colors, tri_uvs,
                            tri_tex_meta, textures, num_colored_triangles)
                    else:
                        color4, alpha = _sample_circuit_color(
                            prim, f, a, b, border, circuit_meta,
                            circuit_colors, circuit_border_colors)
                    alpha = ti.math.clamp(alpha, 0.0, 1.0)
                    tint = ti.math.clamp(
                        ti.math.vec3(color4[0], color4[1], color4[2]), 0.0, 1.0)
                    if ti.static(rgb_shadow_tint()):
                        # The march's Beer-Lambert pairing; state persists
                        # across gather windows (declared above).
                        sigma = _shadow_hit_sigma(f, prim, hit_type, tri_extra)
                        if (alpha >= _SOLID_COVERAGE_MIN) \
                                and (_vis_max_component(sigma) > 0.0):
                            if medium_open == 1:
                                seg = ti.max(t_hit - medium_t_entry, 0.0)
                                transmitted *= ti.math.vec3(
                                    ti.exp(-medium_sigma[0] * seg),
                                    ti.exp(-medium_sigma[1] * seg),
                                    ti.exp(-medium_sigma[2] * seg))
                                medium_open = 0
                            else:
                                medium_open = 1
                                medium_sigma = sigma
                                medium_t_entry = t_hit
                    # See the march: a transmissive surface passes light
                    # rather than blocking it, tinted by its albedo under the
                    # rgb_shadow_tint gate.
                    passed = _shadow_pass_through(
                        f, prim, hit_type, 1.0 - a - b, a, b,
                        tri_extra, circuit_meta, tint)
                    transmitted *= one3 - alpha * (one3 - passed)
                    # Max-component early-outs, same reasoning as the march.
                    if _vis_max_component(transmitted) <= min_alpha:
                        transmitted = ti.math.vec3(0.0)
                        alive = 0
                    elif (alpha >= 1.0) \
                            and (_vis_max_component(passed) <= 0.0):
                        alive = 0
                    else:
                        t_prev = t_hit
                        layer_prev = hit_layer
        # A short buffer proves every remaining hit inside the light's
        # depth window was gathered and drained; the march's next step
        # would find nothing (or only beyond-light hits it breaks on).
        if (alive == 1) and (num_hits < kbuf):
            alive = 0
    return one3 - transmitted


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

