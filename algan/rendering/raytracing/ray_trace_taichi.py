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
  Carlo scattering), ``tri_extra [Te, N, 6]`` (per-corner reflectivity +
  roughness pairs, usually single-frame) and ``tri_colors [Tc, N, 3, 5]``
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
  per-circuit ranges ``edge_offsets``, fill/texture colors
  ``circuit_colors [Tf, C, P, 5]`` (bilinearly sampled; P = 1 for plain
  fills) and ``circuit_border_colors [Tb, C, 5]``.

Coplanar-surface layer order is bezier circuits < triangles < PN patches,
with each type's primitive index breaking ties within the type.
"""
import taichi as ti

from algan.rendering.raytracing.stbvh import LEAF_SIZE


def _ensure_taichi_initialized():
    """Initialize Taichi unless another module (e.g. the rasterizer) already
    has; re-initializing would invalidate previously compiled kernels.
    """
    initialized = False
    try:
        initialized = ti.lang.impl.get_runtime().prog is not None
    except Exception:
        initialized = False
    if not initialized:
        ti.init(arch=ti.gpu)


_ensure_taichi_initialized()

# Minimum hit distance along a ray (also the self-intersection guard for
# reflected rays, together with a normal offset at the bounce origin).
MIN_HIT_DISTANCE = 1e-4
# Hits closer together than this along a ray are considered coplanar and are
# ordered by layer index instead of by distance.
DEPTH_TIE_EPSILON = 1e-4
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
# Slightly larger tolerances for the point-normal (PN) triangle patch
# numerical intersection solver and edge/seam de-duplication, to cover
# larger floating-point / solver noise.
PN_BARYCENTRIC_EPSILON = 1e-4
PN_EDGE_EPSILON = 2e-4
# Hits gathered per BVH traversal by the deterministic renderer. Depth
# peeling consumes hits strictly front-to-back; collecting a small batch of
# nearest hits per traversal lets a ray crossing several translucent
# surfaces re-traverse the scene once per KBUF surfaces instead of once per
# surface (and skip the final "anything left?" traversal whenever a batch
# comes back not full).
KBUF = 4

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


@ti.func
def _safe_inverse(x: ti.f32) -> ti.f32:
    r = 1e12
    if x < 0.0:
        r = -1e12
    if ti.abs(x) > 1e-12:
        r = 1.0 / x
    return r


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
def _node_intersected(node, ff, ro, inv_rd, t_lo, t_hi,
                      nodes: ti.template()) -> bool:
    """Spatio-temporal node test: frame containment + slab test restricted to
    the parametric window [t_lo, t_hi] of still-relevant hits. The node's
    bounds and frame interval live in one packed 8-float row.
    """
    hit = False
    if (nodes[node, 6] <= ff) and (ff <= nodes[node, 7]):
        tx0 = (nodes[node, 0] - ro[0]) * inv_rd[0]
        tx1 = (nodes[node, 3] - ro[0]) * inv_rd[0]
        t_near = ti.min(tx0, tx1)
        t_far = ti.max(tx0, tx1)
        ty0 = (nodes[node, 1] - ro[1]) * inv_rd[1]
        ty1 = (nodes[node, 4] - ro[1]) * inv_rd[1]
        t_near = ti.max(t_near, ti.min(ty0, ty1))
        t_far = ti.min(t_far, ti.max(ty0, ty1))
        tz0 = (nodes[node, 2] - ro[2]) * inv_rd[2]
        tz1 = (nodes[node, 5] - ro[2]) * inv_rd[2]
        t_near = ti.max(t_near, ti.min(tz0, tz1))
        t_far = ti.min(t_far, ti.max(tz0, tz1))
        hit = (t_far >= ti.max(t_near, 0.0)) and (t_near <= t_hi) and (t_far >= t_lo)
    return hit


@ti.func
def _comes_after(t, layer, t_prev, layer_prev) -> bool:
    """Strict ordering along the ray: by distance, with near-coplanar hits
    (within DEPTH_TIE_EPSILON) ordered by descending layer index.
    """
    return (t > t_prev + DEPTH_TIE_EPSILON) or (
        (t > t_prev - DEPTH_TIE_EPSILON) and (layer < layer_prev))


@ti.func
def _quartic_roots(q4, q3, q2, q1, q0, lo, hi):
    """Real roots of ``q4 x^4 + q3 x^3 + q2 x^2 + q1 x + q0`` in [lo, hi],
    tolerant of degenerate (effectively lower-degree) coefficient sets.

    Roots are isolated, not solved in closed form: [lo, hi] is split into
    monotone pieces at the polynomial's critical points -- the derivative
    cubic's roots, themselves isolated by *its* critical points from the
    closed-form quadratic -- and each piece with a sign change is bisected.
    This finds every simple root in the interval regardless of the actual
    degree; near-zero values at the split points are also accepted as roots
    so tangential (even-multiplicity) contacts are kept.

    Returns ``(count, roots)`` with the roots ascending in ``roots[:count]``.
    """
    scale = ti.max(ti.max(ti.abs(q4), ti.abs(q3)),
                   ti.max(ti.max(ti.abs(q2), ti.abs(q1)),
                          ti.max(ti.abs(q0), 1e-30)))
    c4 = q4 / scale
    c3 = q3 / scale
    c2 = q2 / scale
    c1 = q1 / scale
    c0 = q0 / scale

    # Critical points of the derivative cubic: roots of 12c4 x^2 + 6c3 x
    # + 2c2 (stable quadratic formula, degree-degenerate aware).
    s0 = lo
    s1 = lo
    qa = 12.0 * c4
    qb = 6.0 * c3
    qc = 2.0 * c2
    if ti.abs(qa) > 1e-12:
        disc = qb * qb - 4.0 * qa * qc
        if disc > 0.0:
            sq = ti.sqrt(disc)
            qq = -0.5 * (qb + sq)
            if qb < 0.0:
                qq = -0.5 * (qb - sq)
            if ti.abs(qq) > 1e-30:
                s0 = qq / qa
                s1 = qc / qq
    elif ti.abs(qb) > 1e-12:
        s0 = -qc / qb
    s0 = ti.math.clamp(s0, lo, hi)
    s1 = ti.math.clamp(s1, lo, hi)
    if s1 < s0:
        swap = s0
        s0 = s1
        s1 = swap

    # Roots of the derivative cubic on its (up to 3) monotone pieces: the
    # quartic's critical points.
    d3 = 4.0 * c4
    d2 = 3.0 * c3
    d1 = 2.0 * c2
    d0 = c1
    crit = ti.math.vec3(hi, hi, hi)
    ncrit = 0
    xa = lo
    ya = ((d3 * xa + d2) * xa + d1) * xa + d0
    for k in ti.static(range(3)):
        xb = hi
        if ti.static(k == 0):
            xb = s0
        if ti.static(k == 1):
            xb = s1
        if xb > xa:
            yb = ((d3 * xb + d2) * xb + d1) * xb + d0
            if (ya > 0.0) != (yb > 0.0):
                ra = xa
                rb = xb
                fa = ya
                for _ in range(24):
                    m = 0.5 * (ra + rb)
                    fm = ((d3 * m + d2) * m + d1) * m + d0
                    if (fm > 0.0) == (fa > 0.0):
                        ra = m
                        fa = fm
                    else:
                        rb = m
                crit[ncrit] = 0.5 * (ra + rb)
                ncrit += 1
            xa = xb
            ya = yb

    # Roots of the quartic on its (up to 4) monotone pieces.
    roots = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
    count = 0
    xa = lo
    ya = (((c4 * xa + c3) * xa + c2) * xa + c1) * xa + c0
    if ti.abs(ya) < 1e-6:
        roots[count] = xa
        count += 1
    for k in ti.static(range(4)):
        xb = hi
        if ti.static(k < 3):
            if k < ncrit:
                xb = crit[k]
        if xb > xa:
            yb = (((c4 * xb + c3) * xb + c2) * xb + c1) * xb + c0
            root = hi + 1.0
            if (ya > 0.0) != (yb > 0.0):
                ra = xa
                rb = xb
                fa = ya
                for _ in range(26):
                    m = 0.5 * (ra + rb)
                    fm = (((c4 * m + c3) * m + c2) * m + c1) * m + c0
                    if (fm > 0.0) == (fa > 0.0):
                        ra = m
                        fa = fm
                    else:
                        rb = m
                root = 0.5 * (ra + rb)
            elif ti.abs(yb) < 1e-3:
                # Tangential contact at a critical point (or the interval
                # end): the monotone piece's single root is the endpoint.
                root = xb
            if root <= hi:
                dup = 0
                for c in ti.static(range(4)):
                    if (c < count) and (ti.abs(roots[c] - root) < 1e-5):
                        dup = 1
                if (dup == 0) and (count < 4):
                    roots[count] = root
                    count += 1
            xa = xb
            ya = yb
    return count, roots


@ti.func
def _pn_intersect(ro, rd, tp, prim, pn_ctrl: ti.template()):
    """Every intersection (up to four) of a ray with a quadratic Bezier
    (Steiner) triangle patch, packed as monomial coefficients
    ``S(u, v) = K0 + Ku u + Kv v + Kuu u^2 + Kvv v^2 + Kuv uv`` over the
    barycentric domain ``u, v >= 0, u + v <= 1``.

    Sederberg & Anderson's two-plane method: the patch is projected onto
    two orthogonal planes containing the ray, giving two bivariate
    quadratics ``f(u, v) = g(u, v) = 0`` whose common roots are the hits.
    The v-resultant of the pair is a quartic in u (the cubic
    ``b1 a2 - b2 a1`` when both projections are linear in v, which covers
    flat patches exactly); its real roots in the domain are isolated by
    :func:`_quartic_roots`, v is recovered from the linear pencil
    ``c2 f - c1 g`` (falling back to the better-conditioned quadratic, whose
    negative discriminant rejects complex-pair phantoms), and each root is
    polished with two Newton steps on (f, g) to f32 accuracy.

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

    # Quadratics in v: f = C1 v^2 + b1(u) v + a1(u) with b1 = B1 u + E1,
    # a1 = A1 u^2 + D1 u + F1 (same for g). The cubic gm = b1 a2 - b2 a1 is
    # both the v-eliminated system when C1 = C2 = 0 and one factor of the
    # general resultant (c1 a2 - c2 a1)^2 - (c1 b2 - c2 b1)(b1 a2 - b2 a1).
    g3 = B1 * A2 - B2 * A1
    g2 = B1 * D2 + E1 * A2 - B2 * D1 - E2 * A1
    g1 = B1 * F2 + E1 * D2 - B2 * F1 - E2 * D1
    g0 = E1 * F2 - E2 * F1
    q4 = 0.0
    q3 = g3
    q2 = g2
    q1 = g1
    q0 = g0
    if ti.max(ti.abs(C1), ti.abs(C2)) > 1e-6:
        al2 = C1 * A2 - C2 * A1
        al1 = C1 * D2 - C2 * D1
        al0 = C1 * F2 - C2 * F1
        be1 = C1 * B2 - C2 * B1
        be0 = C1 * E2 - C2 * E1
        q4 = al2 * al2 - be1 * g3
        q3 = 2.0 * al2 * al1 - be1 * g2 - be0 * g3
        q2 = al1 * al1 + 2.0 * al2 * al0 - be1 * g1 - be0 * g2
        q1 = 2.0 * al1 * al0 - be1 * g0 - be0 * g1
        q0 = al0 * al0 - be0 * g0

    nu, ru = _quartic_roots(q4, q3, q2, q1, q0, -1e-2, 1.0 + 1e-2)

    count = 0
    out_t = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
    out_u = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
    out_v = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
    for ri in ti.static(range(4)):
        if ri < nu:
            u = ru[ri]
            a1 = (A1 * u + D1) * u + F1
            b1 = B1 * u + E1
            a2 = (A2 * u + D2) * u + F2
            b2 = B2 * u + E2
            denom = C2 * b1 - C1 * b2
            v = 0.0
            ok = 0
            if ti.abs(denom) > 1e-8:
                v = (C1 * a2 - C2 * a1) / denom
                ok = 1
            elif ti.max(ti.abs(C1), ti.abs(C2)) > 1e-7:
                # The two v-quadratics are near-proportional at this u:
                # solve the better-conditioned one directly. A negative
                # discriminant means the shared root pair is complex (the
                # resultant vanishes without a real intersection).
                cc = C1
                bb = b1
                aa = a1
                co = C2
                bo = b2
                ao = a2
                if ti.abs(C2) > ti.abs(C1):
                    cc = C2
                    bb = b2
                    aa = a2
                    co = C1
                    bo = b1
                    ao = a1
                disc = bb * bb - 4.0 * cc * aa
                if disc >= 0.0:
                    sq = ti.sqrt(disc)
                    qq = -0.5 * (bb + sq)
                    if bb < 0.0:
                        qq = -0.5 * (bb - sq)
                    v0 = -0.5 * bb / cc
                    v1 = v0
                    if ti.abs(qq) > 1e-30:
                        v0 = qq / cc
                        v1 = aa / qq
                    v = v0
                    if (ti.abs((co * v1 + bo) * v1 + ao)
                            < ti.abs((co * v0 + bo) * v0 + ao)):
                        v = v1
                    ok = 1
            else:
                # Both projections linear in v.
                if ti.abs(b1) >= ti.abs(b2):
                    if ti.abs(b1) > 1e-8:
                        v = -a1 / b1
                        ok = 1
                elif ti.abs(b2) > 1e-8:
                    v = -a2 / b2
                    ok = 1
            if ok == 1:
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
                if ((u >= -PN_BARYCENTRIC_EPSILON) and (v >= -PN_BARYCENTRIC_EPSILON)
                        and (u + v <= 1.0 + PN_BARYCENTRIC_EPSILON)
                        and (ti.abs(fval) < 2e-3) and (ti.abs(gval) < 2e-3)):
                    x = (k0 + u * ku + v * kv + (u * u) * kuu
                         + (v * v) * kvv + (u * v) * kuv)
                    t = x.dot(rd)
                    dup = 0
                    for c in ti.static(range(4)):
                        if ((c < count)
                                and (ti.abs(out_t[c] - t) <= DEPTH_TIE_EPSILON)):
                            dup = 1
                    if dup == 0:
                        out_t[count] = t
                        out_u[count] = u
                        out_v[count] = v
                        count += 1
    return count, out_t, out_u, out_v


@ti.func
def _nearest_triangle_hit(ro, rd, inv_rd, f, ff, t_prev, layer_prev,
                          layer_offset,
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
    node = 0
    while node != -1:
        if _node_intersected(node, ff, ro, inv_rd,
                             t_prev - DEPTH_TIE_EPSILON,
                             best_t + DEPTH_TIE_EPSILON, nodes):
            if node >= first_leaf:
                base = (node - first_leaf) * LEAF_SIZE
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
                node = node_miss[node]
            else:
                node = 2 * node + 1
        else:
            node = node_miss[node]
    return best_t, best_prim, best_w1, best_w2, best_layer


@ti.func
def _nearest_pn_hit(ro, rd, inv_rd, f, ff, t_prev, layer_prev, layer_offset,
                    nodes: ti.template(), node_miss: ti.template(),
                    leaf_prim: ti.template(), leaf_tspan: ti.template(),
                    first_leaf, pn_ctrl: ti.template()):
    """Nearest PN-patch intersection strictly after (t_prev, layer_prev).
    Every root of each candidate patch is considered (a ray can pierce a
    curved patch several times) and the patch parameters (u, v) of the
    winning hit double as its color/normal interpolation weights.
    """
    best_t = 1e30
    best_layer = -1e30
    best_prim = -1
    best_u = 0.0
    best_v = 0.0
    tp = f % pn_ctrl.shape[0]
    node = 0
    while node != -1:
        if _node_intersected(node, ff, ro, inv_rd,
                             t_prev - DEPTH_TIE_EPSILON,
                             best_t + DEPTH_TIE_EPSILON, nodes):
            if node >= first_leaf:
                base = (node - first_leaf) * LEAF_SIZE
                for j in ti.static(range(LEAF_SIZE)):
                    prim = leaf_prim[base + j]
                    tspan = leaf_tspan[base + j]
                    if ((prim >= 0) and ((tspan & 0xFFFF) <= f)
                            and (f <= ((tspan >> 16) & 0x7FFF))):
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
                node = node_miss[node]
            else:
                node = 2 * node + 1
        else:
            node = node_miss[node]
    return best_t, best_prim, best_u, best_v, best_layer


@ti.func
def _nearest_bezier_hit(ro, rd, inv_rd, f, ff, t_prev, layer_prev,
                        pixel_size_per_t, base_dist,
                        nodes: ti.template(), node_miss: ti.template(),
                        leaf_prim: ti.template(), leaf_tspan: ti.template(),
                        first_leaf, circuit_meta: ti.template(),
                        edges_2d: ti.template(), edge_offsets: ti.template()):
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
    node = 0
    while node != -1:
        if _node_intersected(node, ff, ro, inv_rd,
                             t_prev - DEPTH_TIE_EPSILON,
                             best_t + DEPTH_TIE_EPSILON, nodes):
            if node >= first_leaf:
                base = (node - first_leaf) * LEAF_SIZE
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

                                te = f % num_edge_frames
                                crossings = 0
                                min_dist_sq = 1e30
                                for e in range(edge_offsets[circuit],
                                               edge_offsets[circuit + 1]):
                                    x0 = edges_2d[te, e, 0]
                                    y0 = edges_2d[te, e, 1]
                                    x1 = edges_2d[te, e, 2]
                                    y1 = edges_2d[te, e, 3]
                                    if (y0 > v) != (y1 > v):
                                        x_cross = x0 + (v - y0) * (x1 - x0) / (y1 - y0)
                                        if x_cross > u:
                                            crossings += 1
                                    dx = x1 - x0
                                    dy = y1 - y0
                                    seg_t = ((u - x0) * dx + (v - y0) * dy) / ti.max(
                                        dx * dx + dy * dy, 1e-12)
                                    seg_t = ti.math.clamp(seg_t, 0.0, 1.0)
                                    cx = x0 + seg_t * dx - u
                                    cy = y0 + seg_t * dy - v
                                    min_dist_sq = ti.min(min_dist_sq,
                                                         cx * cx + cy * cy)

                                # World size of one screen pixel at this hit,
                                # for screen-constant border/outline widths.
                                pixel_size = pixel_size_per_t * (base_dist + t)
                                border_w = (circuit_meta[tm, circuit, _M_BORDER_W]
                                            * pixel_size)
                                in_border = min_dist_sq < border_w * border_w
                                outline_w = 0.6 * pixel_size
                                inside = False
                                if circuit_meta[tm, circuit, _M_FILLED] > 0.5:
                                    inside = ((crossings % 2) == 1) or (
                                        min_dist_sq < outline_w * outline_w)
                                if inside or in_border:
                                    best_t = t
                                    best_layer = layer
                                    best_circuit = circuit
                                    best_border = 1 if in_border else 0
                                    best_u = u
                                    best_v = v
                node = node_miss[node]
            else:
                node = 2 * node + 1
        else:
            node = node_miss[node]
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
def _nearest_surface(ro, rd, inv_rd, f, ff, t_prev, layer_prev,
                     pixel_size_per_t, base_dist, layer_offset_triangles,
                     layer_offset_pn,
                     t_nodes: ti.template(), t_node_miss: ti.template(),
                     t_leaf_prim: ti.template(), t_leaf_tspan: ti.template(),
                     t_first_leaf, tri_pos: ti.template(),
                     p_nodes: ti.template(), p_node_miss: ti.template(),
                     p_leaf_prim: ti.template(), p_leaf_tspan: ti.template(),
                     p_first_leaf, pn_ctrl: ti.template(),
                     b_nodes: ti.template(), b_node_miss: ti.template(),
                     b_leaf_prim: ti.template(), b_leaf_tspan: ti.template(),
                     b_first_leaf, circuit_meta: ti.template(),
                     edges_2d: ti.template(), edge_offsets: ti.template()):
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

    tt, t_prim, w1, w2, t_layer = _nearest_triangle_hit(
        ro, rd, inv_rd, f, ff, t_prev, layer_prev, layer_offset_triangles,
        t_nodes, t_node_miss, t_leaf_prim, t_leaf_tspan, t_first_leaf,
        tri_pos)
    pt, p_prim, p_u, p_v, p_layer = _nearest_pn_hit(
        ro, rd, inv_rd, f, ff, t_prev, layer_prev, layer_offset_pn,
        p_nodes, p_node_miss, p_leaf_prim, p_leaf_tspan, p_first_leaf,
        pn_ctrl)
    bt, b_circ, b_border, b_u, b_v, b_layer = _nearest_bezier_hit(
        ro, rd, inv_rd, f, ff, t_prev, layer_prev, pixel_size_per_t,
        base_dist, b_nodes, b_node_miss, b_leaf_prim, b_leaf_tspan,
        b_first_leaf, circuit_meta, edges_2d, edge_offsets)

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
                  b_nodes: ti.template(), b_node_miss: ti.template(),
                  b_leaf_prim: ti.template(), b_leaf_tspan: ti.template(),
                  b_first_leaf, circuit_meta: ti.template(),
                  edges_2d: ti.template(), edge_offsets: ti.template()) -> ti.i32:
    """Gather the up-to-``KBUF`` nearest hits strictly after
    (t_prev, layer_prev) into the caller's buffers, in one traversal of each
    BVH. Triangles are traversed first; the PN-patch and bezier traversals
    then prune against the hits already gathered.

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
    opq_t = 1e30
    opq_layer = -1e30

    # --- Triangle BVH ---
    tp = f % tri_pos.shape[0]
    node = 0
    while node != -1:
        window_hi = worst_t + DEPTH_TIE_EPSILON if count == KBUF else 1e30
        window_hi = ti.min(window_hi, opq_t + DEPTH_TIE_EPSILON)
        if _node_intersected(node, ff, ro, inv_rd,
                             t_prev - DEPTH_TIE_EPSILON, window_hi, t_nodes):
            if node >= t_first_leaf:
                base = (node - t_first_leaf) * LEAF_SIZE
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
                                    and (w1 + w2 <= 1.0 + BARYCENTRIC_EPSILON)):
                                t = e2.dot(qv) * inv_det
                                layer = layer_offset_triangles + ti.cast(
                                    prim, ti.f32)
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
                                    w0 = 1.0 - w1 - w2
                                    eh = 1 if (ti.min(w0, ti.min(w1, w2))
                                               < TRIANGLE_EDGE_EPSILON) else 0
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
                                        for q in ti.static(range(1, KBUF)):
                                            if _comes_after(hit_t[q],
                                                            hit_layer[q],
                                                            worst_t,
                                                            worst_layer):
                                                worst_idx = q
                                                worst_t = hit_t[q]
                                                worst_layer = hit_layer[q]
                node = t_node_miss[node]
            else:
                node = 2 * node + 1
        else:
            node = t_node_miss[node]

    # --- PN patch BVH (window already tightened by the triangle hits) ---
    pp = f % pn_ctrl.shape[0]
    node = 0
    while node != -1:
        window_hi = worst_t + DEPTH_TIE_EPSILON if count == KBUF else 1e30
        window_hi = ti.min(window_hi, opq_t + DEPTH_TIE_EPSILON)
        if _node_intersected(node, ff, ro, inv_rd,
                             t_prev - DEPTH_TIE_EPSILON, window_hi, p_nodes):
            if node >= p_first_leaf:
                base = (node - p_first_leaf) * LEAF_SIZE
                for j in ti.static(range(LEAF_SIZE)):
                    prim = p_leaf_prim[base + j]
                    tspan = p_leaf_tspan[base + j]
                    if ((prim >= 0) and ((tspan & 0xFFFF) <= f)
                            and (f <= ((tspan >> 16) & 0x7FFF))):
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
                node = p_node_miss[node]
            else:
                node = 2 * node + 1
        else:
            node = p_node_miss[node]

    # --- Bezier BVH (window tightened by the triangle and patch hits) ---
    num_meta_frames = circuit_meta.shape[0]
    num_edge_frames = edges_2d.shape[0]
    node = 0
    while node != -1:
        window_hi = worst_t + DEPTH_TIE_EPSILON if count == KBUF else 1e30
        window_hi = ti.min(window_hi, opq_t + DEPTH_TIE_EPSILON)
        if _node_intersected(node, ff, ro, inv_rd,
                             t_prev - DEPTH_TIE_EPSILON, window_hi, b_nodes):
            if node >= b_first_leaf:
                base = (node - b_first_leaf) * LEAF_SIZE
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

                                te = f % num_edge_frames
                                crossings = 0
                                min_dist_sq = 1e30
                                for e in range(edge_offsets[circuit],
                                               edge_offsets[circuit + 1]):
                                    x0 = edges_2d[te, e, 0]
                                    y0 = edges_2d[te, e, 1]
                                    x1 = edges_2d[te, e, 2]
                                    y1 = edges_2d[te, e, 3]
                                    if (y0 > v) != (y1 > v):
                                        x_cross = x0 + (v - y0) * (x1 - x0) / (y1 - y0)
                                        if x_cross > u:
                                            crossings += 1
                                    dx = x1 - x0
                                    dy = y1 - y0
                                    seg_t = ((u - x0) * dx + (v - y0) * dy) / ti.max(
                                        dx * dx + dy * dy, 1e-12)
                                    seg_t = ti.math.clamp(seg_t, 0.0, 1.0)
                                    cx = x0 + seg_t * dx - u
                                    cy = y0 + seg_t * dy - v
                                    min_dist_sq = ti.min(min_dist_sq,
                                                         cx * cx + cy * cy)

                                pixel_size = pixel_size_per_t * (base_dist + t)
                                border_w = (circuit_meta[tm, circuit, _M_BORDER_W]
                                            * pixel_size)
                                in_border = min_dist_sq < border_w * border_w
                                outline_w = 0.6 * pixel_size
                                inside = False
                                if circuit_meta[tm, circuit, _M_FILLED] > 0.5:
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
                node = b_node_miss[node]
            else:
                node = 2 * node + 1
        else:
            node = b_node_miss[node]
    return count


@ti.kernel
def render_scene_stbvh(
        # Triangle STBVH + packed geometry.
        t_nodes: ti.types.ndarray(), t_node_miss: ti.types.ndarray(),
        t_leaf_prim: ti.types.ndarray(), t_leaf_tspan: ti.types.ndarray(),
        t_first_leaf: int,
        tri_pos: ti.types.ndarray(), tri_norm: ti.types.ndarray(),
        tri_extra: ti.types.ndarray(), tri_colors: ti.types.ndarray(),
        # PN patch STBVH + packed geometry.
        p_nodes: ti.types.ndarray(), p_node_miss: ti.types.ndarray(),
        p_leaf_prim: ti.types.ndarray(), p_leaf_tspan: ti.types.ndarray(),
        p_first_leaf: int,
        pn_ctrl: ti.types.ndarray(), pn_norm: ti.types.ndarray(),
        pn_extra: ti.types.ndarray(), pn_colors: ti.types.ndarray(),
        # Bezier STBVH + packed geometry.
        b_nodes: ti.types.ndarray(), b_node_miss: ti.types.ndarray(),
        b_leaf_prim: ti.types.ndarray(), b_leaf_tspan: ti.types.ndarray(),
        b_first_leaf: int,
        circuit_meta: ti.types.ndarray(), circuit_colors: ti.types.ndarray(),
        circuit_border_colors: ti.types.ndarray(),
        edges_2d: ti.types.ndarray(), edge_offsets: ti.types.ndarray(),
        # Per-frame camera and pixel scale.
        cam_origin: ti.types.ndarray(), screen_point: ti.types.ndarray(),
        pixel_basis_x: ti.types.ndarray(), pixel_basis_y: ti.types.ndarray(),
        pixel_world_scale: ti.types.ndarray(),
        # Render parameters.
        time_start: int, time_end: int, width: int, height: int,
        half_screen_w: float, half_screen_h: float,
        layer_offset_triangles: float, layer_offset_pn: float,
        max_bounces: int, transparent: int,
        # Output buffer [time_end - time_start, width * height, channels],
        # pre-filled with the background; blended in place.
        out: ti.types.ndarray()):
    pixels_per_frame = width * height
    num_rays = (time_end - time_start) * pixels_per_frame

    for ray_id in range(num_rays):
        f_rel = ray_id // pixels_per_frame
        p = ray_id - f_rel * pixels_per_frame
        f = time_start + f_rel
        ff = ti.cast(f, ti.f32)
        py = p // width
        px = p - py * width

        ro, rd = _generate_ray(f, px, py, 0.5, 0.5,
                               half_screen_w, half_screen_h,
                               cam_origin, screen_point,
                               pixel_basis_x, pixel_basis_y)
        inv_rd = ti.math.vec3(_safe_inverse(rd[0]), _safe_inverse(rd[1]),
                              _safe_inverse(rd[2]))
        pixel_size_per_t = pixel_world_scale[f]

        acc = ti.math.vec4(0.0, 0.0, 0.0, 0.0)  # premultiplied RGB + glow
        weight = 1.0       # remaining transmittance * reflection throughput
        t_prev = 0.0
        layer_prev = 1e30  # accept any first hit
        base_dist = 0.0    # distance accumulated over previous bounces
        bounces_left = max_bounces
        seam_t = -1e30  # depth of the last processed triangle edge hit

        # Batched depth peeling: each traversal gathers up to KBUF nearest
        # hits, which are then consumed strictly front-to-back. A batch that
        # comes back not full contained every remaining hit, so peeling ends
        # without a confirming traversal.
        kb_t = ti.Vector([0.0] * KBUF)
        kb_layer = ti.Vector([0.0] * KBUF)
        kb_prim = ti.Vector([0] * KBUF)
        kb_flags = ti.Vector([0] * KBUF)
        kb_a = ti.Vector([0.0] * KBUF)
        kb_b = ti.Vector([0.0] * KBUF)

        processed = 0
        done = False
        while (not done) and (processed < MAX_SURFACES_PER_RAY):
            num_hits = _collect_hits(
                ro, rd, inv_rd, f, ff, t_prev, layer_prev,
                pixel_size_per_t, base_dist, layer_offset_triangles,
                layer_offset_pn,
                kb_t, kb_layer, kb_prim, kb_flags, kb_a, kb_b,
                t_nodes, t_node_miss, t_leaf_prim, t_leaf_tspan,
                t_first_leaf, tri_pos,
                p_nodes, p_node_miss, p_leaf_prim, p_leaf_tspan,
                p_first_leaf, pn_ctrl,
                b_nodes, b_node_miss, b_leaf_prim, b_leaf_tspan,
                b_first_leaf, circuit_meta, edges_2d, edge_offsets)
            if num_hits == 0:
                break

            bounced = False
            drained = 0
            while drained < num_hits:
                # Select the earliest unconsumed hit, with the same pairwise
                # (distance, layer) rule the traversal itself applies.
                sel = 0
                sel_found = 0
                for q in ti.static(range(KBUF)):
                    if (q < num_hits) and (kb_prim[q] >= 0):
                        if sel_found == 0:
                            sel = q
                            sel_found = 1
                        elif _comes_after(kb_t[sel], kb_layer[sel],
                                          kb_t[q], kb_layer[q]):
                            sel = q
                t_hit = kb_t[sel]
                hit_layer = kb_layer[sel]
                prim = kb_prim[sel]
                flags = kb_flags[sel]
                a = kb_a[sel]
                b = kb_b[sel]
                kb_prim[sel] = -1  # consume
                drained += 1
                processed += 1
                htype = flags & 3
                edge_hit = (flags >> 2) & 1
                border = (flags >> 3) & 1

                # Mesh seams: the two triangles adjacent to a shared edge can
                # both report the crossing ray; the second edge hit at the
                # same depth is the same surface point, so skip it (one
                # cohesive surface, blended exactly once).
                if (edge_hit == 1) and (t_hit - seam_t <= DEPTH_TIE_EPSILON):
                    t_prev = t_hit
                    layer_prev = hit_layer
                    continue
                seam_t = t_hit if edge_hit == 1 else -1e30

                color = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
                alpha = 0.0
                reflectivity = 0.0
                if htype == 1:
                    w0 = 1.0 - a - b
                    color, alpha = _triangle_color(f, prim, w0, a, b,
                                                   tri_colors)
                    reflectivity, _rough = _triangle_extra(f, prim, w0, a, b,
                                                           tri_extra)
                elif htype == 2:
                    w0 = 1.0 - a - b
                    color, alpha = _triangle_color(f, prim, w0, a, b,
                                                   pn_colors)
                    reflectivity, _rough = _triangle_extra(f, prim, w0, a, b,
                                                           pn_extra)
                else:
                    color, alpha = _sample_circuit_color(
                        prim, f, a, b, border,
                        circuit_meta, circuit_colors, circuit_border_colors)

                alpha = ti.math.clamp(alpha, 0.0, 1.0)
                reflectivity = ti.math.clamp(reflectivity, 0.0, 1.0)
                if bounces_left <= 0:
                    reflectivity = 0.0

                acc += (weight * alpha * (1.0 - reflectivity)) * color

                if (reflectivity > MIN_ALPHA) and (alpha > MIN_ALPHA):
                    # Mirror bounce: reflect about the face-forward normal
                    # and restart peeling along the new ray (the rest of the
                    # gathered batch belongs to the old ray and is dropped).
                    # The normal is only fetched here, on the bounce path.
                    normal = ti.math.vec3(0.0, 0.0, 0.0)
                    if htype == 1:
                        normal = _triangle_normal(f, prim, 1.0 - a - b, a, b,
                                                  tri_norm, tri_pos)
                    elif htype == 2:
                        normal = _pn_normal(f, prim, a, b, pn_norm, pn_ctrl)
                    else:
                        normal = _bezier_normal(f, prim, circuit_meta)
                    normal = normal.normalized()
                    if normal.dot(rd) > 0.0:
                        normal = -normal
                    hit_point = ro + t_hit * rd
                    rd = (rd - 2.0 * rd.dot(normal) * normal).normalized()
                    ro = hit_point + normal * (10.0 * MIN_HIT_DISTANCE)
                    inv_rd = ti.math.vec3(_safe_inverse(rd[0]),
                                          _safe_inverse(rd[1]),
                                          _safe_inverse(rd[2]))
                    weight *= alpha * reflectivity
                    base_dist += t_hit
                    t_prev = 0.0
                    layer_prev = 1e30
                    seam_t = -1e30
                    bounces_left -= 1
                    bounced = True
                    break
                else:
                    weight *= 1.0 - alpha
                    t_prev = t_hit
                    layer_prev = hit_layer
                if weight < MIN_WEIGHT:
                    done = True
                    break
            if (not done) and (not bounced) and (num_hits < KBUF):
                done = True  # the batch held every remaining hit

        # Composite over the pre-filled background and write the pixel.
        for ci in ti.static(range(4)):
            bg = ti.cast(out[f_rel, p, ci], ti.f32)
            val = acc[ci] * 255.0 + weight * bg
            out[f_rel, p, ci] = ti.cast(ti.math.clamp(val + 0.5, 0.0, 255.0),
                                        ti.u8)
        if transparent != 0:
            bg_a = ti.cast(out[f_rel, p, 4], ti.f32)
            val = (1.0 - weight) * 255.0 + weight * bg_a
            out[f_rel, p, 4] = ti.cast(ti.math.clamp(val + 0.5, 0.0, 255.0),
                                       ti.u8)


@ti.kernel
def path_trace_scene_stbvh(
        # Triangle STBVH + packed geometry.
        t_nodes: ti.types.ndarray(), t_node_miss: ti.types.ndarray(),
        t_leaf_prim: ti.types.ndarray(), t_leaf_tspan: ti.types.ndarray(),
        t_first_leaf: int,
        tri_pos: ti.types.ndarray(), tri_norm: ti.types.ndarray(),
        tri_extra: ti.types.ndarray(), tri_colors: ti.types.ndarray(),
        # PN patch STBVH + packed geometry.
        p_nodes: ti.types.ndarray(), p_node_miss: ti.types.ndarray(),
        p_leaf_prim: ti.types.ndarray(), p_leaf_tspan: ti.types.ndarray(),
        p_first_leaf: int,
        pn_ctrl: ti.types.ndarray(), pn_norm: ti.types.ndarray(),
        pn_extra: ti.types.ndarray(), pn_colors: ti.types.ndarray(),
        # Bezier STBVH + packed geometry.
        b_nodes: ti.types.ndarray(), b_node_miss: ti.types.ndarray(),
        b_leaf_prim: ti.types.ndarray(), b_leaf_tspan: ti.types.ndarray(),
        b_first_leaf: int,
        circuit_meta: ti.types.ndarray(), circuit_colors: ti.types.ndarray(),
        circuit_border_colors: ti.types.ndarray(),
        edges_2d: ti.types.ndarray(), edge_offsets: ti.types.ndarray(),
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
                p_first_leaf, pn_ctrl,
                b_nodes, b_node_miss, b_leaf_prim, b_leaf_tspan,
                b_first_leaf, circuit_meta, edges_2d, edge_offsets)
            if found == 0:
                escaped = True
                break

            # Mesh seams: skip the duplicate edge hit of the adjacent
            # triangle so the surface scatters/transmits exactly once.
            if (edge_hit == 1) and (t_hit - seam_t <= DEPTH_TIE_EPSILON):
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
                color, alpha = _triangle_color(f, prim, w0, a, b,
                                               tri_colors)
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

            alpha = ti.math.clamp(alpha, 0.0, 1.0)
            if ti.random(ti.f32) >= alpha:
                # Pass straight through the (partially) transparent
                # surface; advance the peel state along the same ray.
                t_prev = t_hit
                layer_prev = hit_layer
                continue
            interacted = True

            reflectivity = ti.math.clamp(reflectivity, 0.0, 1.0)
            if bounces_left <= 0:
                reflectivity = 0.0

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
        for ci in ti.static(range(4)):
            val = accum[f_rel, p, ci] * inv_spp * 255.0
            out[f_rel, p, ci] = ti.cast(ti.math.clamp(val + 0.5, 0.0, 255.0),
                                        ti.u8)
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
                   pn_colors: ti.template(),
                   b_nodes: ti.template(), b_node_miss: ti.template(),
                   b_leaf_prim: ti.template(), b_leaf_tspan: ti.template(),
                   b_first_leaf, circuit_meta: ti.template(),
                   circuit_colors: ti.template(),
                   circuit_border_colors: ti.template(),
                   edges_2d: ti.template(), edge_offsets: ti.template()):
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
            pn_ctrl,
            b_nodes, b_node_miss, b_leaf_prim, b_leaf_tspan, b_first_leaf,
            circuit_meta, edges_2d, edge_offsets)
        if (found == 0) or (t_hit >= max_t):
            break
        # Skip the duplicate edge hit of mesh seams (attenuate once).
        if (edge_hit == 1) and (t_hit - seam_t <= DEPTH_TIE_EPSILON):
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
        t_nodes: ti.types.ndarray(), t_node_miss: ti.types.ndarray(),
        t_leaf_prim: ti.types.ndarray(), t_leaf_tspan: ti.types.ndarray(),
        t_first_leaf: int,
        tri_pos: ti.types.ndarray(), tri_norm: ti.types.ndarray(),
        tri_extra: ti.types.ndarray(), tri_colors: ti.types.ndarray(),
        # PN patch STBVH + packed geometry.
        p_nodes: ti.types.ndarray(), p_node_miss: ti.types.ndarray(),
        p_leaf_prim: ti.types.ndarray(), p_leaf_tspan: ti.types.ndarray(),
        p_first_leaf: int,
        pn_ctrl: ti.types.ndarray(), pn_norm: ti.types.ndarray(),
        pn_extra: ti.types.ndarray(), pn_colors: ti.types.ndarray(),
        # Bezier STBVH + packed geometry.
        b_nodes: ti.types.ndarray(), b_node_miss: ti.types.ndarray(),
        b_leaf_prim: ti.types.ndarray(), b_leaf_tspan: ti.types.ndarray(),
        b_first_leaf: int,
        circuit_meta: ti.types.ndarray(), circuit_colors: ti.types.ndarray(),
        circuit_border_colors: ti.types.ndarray(),
        edges_2d: ti.types.ndarray(), edge_offsets: ti.types.ndarray(),
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
      Blinn-Phong with exponent derived from ``roughness``). ``reflectivity``
      doubles as the metallicness.
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
                p_first_leaf, pn_ctrl,
                b_nodes, b_node_miss, b_leaf_prim, b_leaf_tspan,
                b_first_leaf, circuit_meta, edges_2d, edge_offsets)
            if found == 0:
                escaped = True
                break

            # Mesh seams: skip the duplicate edge hit of the adjacent
            # triangle (one interaction per surface crossing).
            if (edge_hit == 1) and (t_hit - seam_t <= DEPTH_TIE_EPSILON):
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
                color, alpha = _triangle_color(f, prim, w0, a, b,
                                               tri_colors)
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

            alpha = ti.math.clamp(alpha, 0.0, 1.0)
            if ti.random(ti.f32) >= alpha:
                t_prev = t_hit
                layer_prev = hit_layer
                continue
            interacted = True

            albedo = ti.math.vec3(color[0], color[1], color[2])
            glow = ti.max(color[3], 0.0)
            metallic = ti.math.clamp(reflectivity, 0.0, 1.0)
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

            f0 = ti.math.vec3(0.04, 0.04, 0.04) * (1.0 - metallic) \
                + albedo * metallic
            cos_view = ti.max(normal.dot(-rd), 0.0)
            fresnel = f0 + (ti.math.vec3(1.0, 1.0, 1.0) - f0) \
                * ti.pow(1.0 - cos_view, 5.0)

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
                            pn_colors,
                            b_nodes, b_node_miss, b_leaf_prim,
                            b_leaf_tspan, b_first_leaf, circuit_meta,
                            circuit_colors, circuit_border_colors,
                            edges_2d, edge_offsets)
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
            if metallic < 1e-3:
                spec_prob = 0.0  # skip glints on plain dielectrics
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
