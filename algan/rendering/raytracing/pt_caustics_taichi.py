"""Manifold next-event estimation (MNEE): shadow caustics for the path tracer.

Hanika, Droske & Fascione, "Manifold Next Event Estimation" (EGSR 2015), the
method Blender Cycles ships as "shadow caustics". A caustic is an ``L S+ D``
connection: light reaching a diffuse vertex THROUGH specular bending. The
straight shadow ray of ordinary next-event estimation cannot make it, and a
BSDF-sampled path finds it with probability zero for a delta light. MNEE makes
the connection deterministically: the straight segment from the shading point
to the light sample is probed for caster crossings, which seed a chain of
specular vertices; Newton iterations then move those vertices across the
caster's triangles until every one of them satisfies Snell's law, so that the
BENT chain connects the shading point to the same light sample.

This module is the geometry half -- vertex frames, the constraint and its
derivatives, the block solve, the manifold walk, the generalized geometry
term, and the caster-aware walks along straight segments. The radiometry and
the orchestration live in ``pt_shade`` (``path_tracer_taichi``), which owns the
materials, the light rows and the shadow march.

Scope (``DESIGN_path_tracer_roadmap.md`` section 1): refraction only, through
at most ``MNEE_MAX_VERTICES`` caster interfaces -- one interface (a water
surface, the boundary of a subsurface interior) or one interface PAIR (the
entry and exit of a glass solid). Reflective chains and longer chains are not
attempted; a straight segment that crosses more caster interfaces than that is
fail-closed and counted. Each interface separates its material (behind the
geometric normal) from index 1 in front of it, as in Cycles: nested
dielectrics are not modelled along a caustic chain.

Notation, for a chain ``x_0 -> x_1 -> ... -> x_k -> y``: ``x_0`` is the
shading point (fixed), ``x_1 .. x_k`` the caster vertices (free), and ``y``
the light end -- a point (point, spot and area lights, emissive triangles) or
a fixed direction (directional lights, the environment map). At vertex ``i``
``wi`` points back toward ``x_{i-1}`` and ``wo`` on toward ``x_{i+1}``.

The constraint is the generalized half vector (Walter et al. 2007; Jakob &
Marschner 2012): with ``eta = eta_o / eta_i`` the ratio of the indices on the
``wo`` and ``wi`` sides, ``h = normalize(-(wi + eta * wo))`` is parallel to
the shading normal exactly when the vertex refracts. ``C_i = (h . s, h . t)``
projects it onto an orthonormal frame perpendicular to the shading normal, so
the chain is a solution when every ``C_i`` vanishes (and every vertex has its
two neighbours on opposite sides, which rules out the reflection-like
solutions of the projected constraint). A vertex moves in the plane of its
triangle along an orthonormal in-plane frame ``(dp_du, dp_dv)``; the
curvature the walk sees comes from the interpolated normals' derivatives
``(dn_du, dn_dv)``, which is why a caster needs smooth normals. The Jacobian
of the whole chain is block tridiagonal: ``A_i`` (with respect to
``x_{i-1}``), ``B_i`` (``x_i``) and ``C_i`` (``x_{i+1}``), each 2x2.

2x2 matrices travel as ``vec4`` rows ``(m00, m01, m10, m11)``: row 0 is the
``s`` component of the constraint, row 1 the ``t`` component; column 0 is a
displacement along the first tangent, column 1 along the second.

There are no random decisions here: a solve is a pure function of the path
and its light sample, so MNEE draws no sampler dimension.
"""

from algan.rendering.raytracing.raytrace_kernels_taichi import (
    _flat_triangle_color,
    _nearest_surface_g,
    _safe_inverse,
    _shadow_pass_through,
    depth_tie_epsilon,
    max_surfaces_per_ray,
)
from algan.rendering.raytracing.transport_taichi import (
    _offset_ray_origin,
    _shadow_tmax,
)
from algan.rendering.raytracing.wavefront_kernels_taichi import (
    _flat_triangle_material,
)
from algan.taichi_compat import ti

#: Caster interfaces one chain may hold: one interface, or the entry and exit
#: of one solid. A straight segment crossing more is fail-closed and counted.
MNEE_MAX_VERTICES = 2
#: Newton iterations per solve, counting the rejected (halved) steps.
MNEE_MAX_ITERATIONS = 64
#: Converged when every vertex's half vector is this close to its normal
#: (the tangential components of a unit vector: roughly radians). Cycles uses
#: 1e-3; Newton's quadratic convergence makes the extra digit about one step.
MNEE_THRESHOLD = 1e-4
#: Rounding of one f32 chain coordinate relative to its magnitude (two ulps).
#: A leg is a difference of such coordinates, so its direction is uncertain by
#: about this times the coordinates' magnitude over the leg's length: from a
#: receiver a hundredth of a unit under its caster that already exceeds
#: ``MNEE_THRESHOLD``, and no walk could converge there. The convergence test
#: and the coverage test's tolerance rise to that floor instead
#: (``_pt_mnee_chain``'s ``c_floor``); where legs are long it is far below
#: both and changes nothing.
MNEE_ROUNDING = 2.0**-22
#: The first step's fraction of the full Newton step. Far from the seed the
#: linearised normals (``dn_du``, ``dn_dv``) overshoot, so the walk starts
#: gently and doubles back up to 1 on every accepted step (as Cycles does).
MNEE_INITIAL_STEP = 0.1
#: The step-size floor of the backtracking line search.
MNEE_MIN_STEP = 1e-4
#: A segment shorter than this has no direction worth differentiating.
MNEE_MIN_DISTANCE = 1e-6
#: Singular pivot floor of the block solve: a determinant below this is a
#: chain at its own fold (where the caustic's irradiance diverges).
MNEE_MIN_DETERMINANT = 1e-12
#: Hits a projection ray may peel before it reaches the caster it is after.
MNEE_PROJECTION_PEELS = 8
#: Barycentric slack before a tentative point is treated as having left its
#: triangle (it is then projected through the BVH onto the neighbour).
MNEE_BARY_SLACK = 1e-5
#: How much brighter than the unobstructed light (at the same path length) a
#: caustic may get. A point light focused by a lens has a divergent fold; the
#: cap keeps one sample's weight finite there. The focus of a glass ball under
#: a point light concentrates by a few hundred, which this leaves alone.
MNEE_MAX_CONCENTRATION = 1e4

#: Two first-leg directions closer than this (a chord of unit vectors, so
#: about radians) are the same chain: the walk converges to ``MNEE_THRESHOLD``,
#: an order finer, and distinct solutions of one light end sit much further
#: apart everywhere but at the fold where two of them merge. Raised to four
#: times the chain's f32 noise floor where its legs are too short for either
#: (``MNEE_ROUNDING``).
MNEE_COVER_TOLERANCE = 1e-3

#: ``pt_stats`` columns this module's failures are tallied in (see
#: ``truncation.py``): straight segments with more caster crossings than a
#: chain may hold, and solves that ran out of iterations still walking.
PT_STAT_CAUSTIC_CHAIN = 5
PT_STAT_CAUSTIC_SOLVER = 6

# The coverage test's per-path record (``pt_cau``, ``PT_CAU_WIDTH`` f32
# columns per path slot; a one-row dummy in a kernel compiled without
# caustics). A BSDF path that scatters at a receiver and then finds an emitter
# through casters is the same light transport a light sample's chain may have
# delivered at that receiver, so ``pt_shade`` holds its radiance back, and
# ``pt_caustic_cover`` weights it against that chain: by the power heuristic
# where the walk from that receiver to that very light end converges to that
# very path (``_pt_mnee_covered``), at weight 1 everywhere else. Culling every
# such path instead (Cycles' rule) loses all the light the walk cannot find --
# a solution outside the straight seed's basin, or a light end whose straight
# segment misses the caster -- which darkened the floor under a glass ball
# resting on it by 36% under a uniform sky; culling only the covered ones
# conserves that light but leaves the chain alone where it is a poor
# strategy (a light the glass magnifies), whose rare huge samples are
# fireflies.
#
# Columns 0-13 describe the receiver, written at its scatter (the one that
# sets ``_PT_CAU_COUNT`` to 1): its point, its geometric normal on the side
# the path arrived from (zero at a medium vertex, which has no side), its
# ray-facing shading normal (the caustic block's horizon), the direction the
# BSDF sampled there and that direction's density, and whether the path had
# scattered before it (0) or reaches it straight from the camera (1; its
# caustics are direct light, which the firefly clamp must not see). Columns
# 14-29 are one pending emitter: its kind (0 none, 1 a point on an emissive
# triangle, 2 an environment direction), that light end, its radiance at
# weight 1, the MIS weight it has with caustics off, the caster interfaces
# the path crossed since the receiver, the frame, the density next-event
# estimation samples that light end with (``n_ls`` light samples' worth: per
# unit area on a triangle, per unit solid angle for a direction), the
# emitter's front normal and whether it emits from that side only, and the
# firefly clamp the radiance takes outside a direct caustic (0: none).
# ``pt_caustic_cover`` clears the kind after every ``pt_shade`` launch, so a
# launch starts with none pending. Column 30 is the path's caustic count
# (``_PT_CAU_COUNT``), which outlives the launches. The host zeroes the record
# with the rest of a path's state before ``pt_generate``.
_PT_CAU_X0 = 0
_PT_CAU_NSIDE = 3
_PT_CAU_SPEC = 6
_PT_CAU_WB = 9
_PT_CAU_PB = 12
_PT_CAU_DIRECT = 13
_PT_CAU_KIND = 14
_PT_CAU_Y = 15
_PT_CAU_VAL = 18
_PT_CAU_WOLD = 21
_PT_CAU_K = 22
_PT_CAU_F = 23
_PT_CAU_PY = 24
_PT_CAU_NE = 25
_PT_CAU_ONE_SIDED = 28
_PT_CAU_CLAMP = 29
#: The caster interfaces a path has crossed since its last receiver. The BSDF
#: path that makes a light sample's chain journey -- a scatter at a receiver,
#: then nothing but smooth transmissions through casters, then an emitter --
#: carries the same light, and the two must share it. A blurring scatter at a
#: vertex that ran the caustic block writes 1, each smooth caster transmission
#: after it adds one, and anything else (another scatter, a mirror bounce,
#: one transmission more than a chain may hold) writes 0; a straight
#: pass-through, an index-matched boundary and a smooth transmission through a
#: surface that is not a caster keep it. An emissive triangle or environment
#: direction the next-event table could have sampled, found at 2 or more, is
#: held back for the coverage test: where a light sample's chain from the
#: same receiver is this very path it keeps the BSDF path's power-heuristic
#: share (``_pt_caustic_mis``), and everywhere else its whole value.
_PT_CAU_COUNT = 30
PT_CAU_WIDTH = 31


@ti.func
def _m2_det(m):
    return m[0] * m[3] - m[1] * m[2]


@ti.func
def _m2_inv(m, det):
    inv = 1.0 / det
    return ti.math.vec4(m[3] * inv, -m[1] * inv, -m[2] * inv, m[0] * inv)


@ti.func
def _m2_mul(a, b):
    return ti.math.vec4(
        a[0] * b[0] + a[1] * b[2], a[0] * b[1] + a[1] * b[3],
        a[2] * b[0] + a[3] * b[2], a[2] * b[1] + a[3] * b[3])


@ti.func
def _m2_vec(m, v):
    return ti.math.vec2(m[0] * v[0] + m[1] * v[1], m[2] * v[0] + m[3] * v[1])


@ti.func
def _pt_caster_id(shell: ti.template(), f, prim, offset):
    """The caustic-caster object id of triangle ``prim`` (-1: not a caster),
    from the ``tri_shell`` section the host appended at ``offset`` (0 when no
    section exists, which is the gate every caller tests first).
    """
    cid = -1
    if (offset > 0) and (prim >= 0):
        cid = shell[f % shell.shape[0], offset + prim]
    return cid


@ti.func
def _pt_mnee_surface(f, prim, a, b, tri_extra: ti.template(),
                     tri_uvs: ti.template(), tri_tex_meta: ti.template(),
                     textures: ti.template(), num_colored_triangles):
    """``(ior, refracts)`` of a caster triangle at barycentrics ``(a, b)``.

    ``refracts`` is the path tracer's own test for a SMOOTH dielectric
    crossing (``pt_shade``'s ``glass`` with a delta lobe): transmissive, not
    wholly metal, an index off 1 and a GGX width on its 1e-4 floor. A caster
    whose material fails it at this point (a texel with no transmission, a
    rough patch) is not bent here; the straight shadow march tints or blocks
    it as it always has.
    """
    met, rough, ior, trans = _flat_triangle_material(
        f, prim, 1.0 - a - b, a, b, tri_extra, tri_uvs, tri_tex_meta, textures,
        num_colored_triangles)
    refracts = 0
    if (trans > 1e-4) and (met < 1.0 - 1e-6) and (ti.abs(ior - 1.0) >= 1e-4) \
            and (rough * rough < 1e-4):
        refracts = 1
    return ior, refracts


@ti.func
def _pt_mnee_point(f, prim, a, b, tri_pos: ti.template()):
    """Point with barycentrics ``(a, b)`` on triangle ``prim`` and the
    triangle's unit geometric normal.
    """
    tp = f % tri_pos.shape[0]
    v0 = ti.math.vec3(tri_pos[tp, prim, 0], tri_pos[tp, prim, 1],
                      tri_pos[tp, prim, 2])
    v1 = ti.math.vec3(tri_pos[tp, prim, 3], tri_pos[tp, prim, 4],
                      tri_pos[tp, prim, 5])
    v2 = ti.math.vec3(tri_pos[tp, prim, 6], tri_pos[tp, prim, 7],
                      tri_pos[tp, prim, 8])
    e1 = v1 - v0
    e2 = v2 - v0
    ng = e1.cross(e2)
    return v0 + e1 * a + e2 * b, ng / ti.max(ng.norm(), 1e-30)


@ti.func
def _pt_mnee_extent(f, prim, tri_pos: ti.template()):
    """The magnitude a point on triangle ``prim`` rounds at: three times its
    largest corner coordinate (``v0 + e1 a + e2 b`` sums three terms that
    large).
    """
    tp = f % tri_pos.shape[0]
    m = 0.0
    for k in ti.static(range(9)):
        m = ti.max(m, ti.abs(tri_pos[tp, prim, k]))
    return 3.0 * m


@ti.func
def _pt_mnee_vertex(f, prim, a, b, flat, tri_pos: ti.template(),
                    tri_norm: ti.template()):
    """Differential geometry of the point with barycentrics ``(a, b)`` (the
    weights of the triangle's second and third corners) on triangle ``prim``.

    Returns ``(p, n, ng, dp_du, dp_dv, dn_du, dn_dv, valid)``: the point, the
    unit shading normal the camera path refracts with (``_triangle_normal``'s
    interpolation, blended toward the face normal by the material's ``flat``
    as ``_shading_normal`` blends it; normal maps are not followed) oriented
    to agree with the unit geometric normal ``ng``, an orthonormal frame in
    the triangle's plane (``dp_du`` along the first edge, Gram-Schmidt for
    the second), and the shading normal's derivatives along that frame.
    """
    tp = f % tri_pos.shape[0]
    v0 = ti.math.vec3(tri_pos[tp, prim, 0], tri_pos[tp, prim, 1],
                      tri_pos[tp, prim, 2])
    v1 = ti.math.vec3(tri_pos[tp, prim, 3], tri_pos[tp, prim, 4],
                      tri_pos[tp, prim, 5])
    v2 = ti.math.vec3(tri_pos[tp, prim, 6], tri_pos[tp, prim, 7],
                      tri_pos[tp, prim, 8])
    e1 = v1 - v0
    e2 = v2 - v0
    p = v0 + e1 * a + e2 * b
    ng = e1.cross(e2)
    ng_len = ng.norm()
    valid = 1
    if ng_len <= 1e-20:
        valid = 0
        ng = ti.math.vec3(0.0, 0.0, 1.0)
    else:
        ng = ng / ng_len
    tn = f % tri_norm.shape[0]
    n0 = ti.math.vec3(tri_norm[tn, prim, 0], tri_norm[tn, prim, 1],
                      tri_norm[tn, prim, 2])
    n1 = ti.math.vec3(tri_norm[tn, prim, 3], tri_norm[tn, prim, 4],
                      tri_norm[tn, prim, 5])
    n2 = ti.math.vec3(tri_norm[tn, prim, 6], tri_norm[tn, prim, 7],
                      tri_norm[tn, prim, 8])
    w0 = 1.0 - a - b
    big_n = n0 * w0 + n1 * a + n2 * b
    big_len = big_n.norm()
    n = ng
    dn_da = ti.math.vec3(0.0, 0.0, 0.0)
    dn_db = ti.math.vec3(0.0, 0.0, 0.0)
    if big_len > 1e-6:
        # d/da (N / |N|) = (dN - n (n . dN)) / |N|, dN = n1 - n0 (and n2 - n0).
        n = big_n / big_len
        d1 = n1 - n0
        d2 = n2 - n0
        dn_da = (d1 - n * n.dot(d1)) / big_len
        dn_db = (d2 - n * n.dot(d2)) / big_len
    if n.dot(ng) < 0.0:
        n = -n
        dn_da = -dn_da
        dn_db = -dn_db
    if flat > 1e-4:
        m = n * (1.0 - flat) + ng * flat
        m_len = ti.max(m.norm(), 1e-12)
        nb = m / m_len
        dm_a = dn_da * (1.0 - flat)
        dm_b = dn_db * (1.0 - flat)
        dn_da = (dm_a - nb * nb.dot(dm_a)) / m_len
        dn_db = (dm_b - nb * nb.dot(dm_b)) / m_len
        n = nb
    # The same linear change of variables takes (a, b) to the orthonormal
    # frame for positions and normals alike: p = v0 + e1 a + e2 b, and
    # e2 = dp_du * e2s + dp_dv * l2.
    l1 = e1.norm()
    dp_du = e1 / ti.max(l1, 1e-30)
    e2s = e2.dot(dp_du)
    tq = e2 - dp_du * e2s
    l2 = tq.norm()
    if (l1 <= 1e-20) or (l2 <= 1e-20):
        valid = 0
    dp_dv = tq / ti.max(l2, 1e-30)
    dn_du = dn_da / ti.max(l1, 1e-30)
    dn_dv = (dn_db - dn_du * e2s) / ti.max(l2, 1e-30)
    return p, n, ng, dp_du, dp_dv, dn_du, dn_dv, valid


@ti.func
def _pt_mnee_constraint(xp, p, xn, fixed, n, ng, dp_du, dp_dv, dn_du, dn_dv,
                        ior, ep_u, ep_v, en_u, en_v):
    """Half-vector constraint of one chain vertex and its three 2x2 blocks.

    ``xp`` / ``xn`` are the previous and next chain points; with ``fixed``
    nonzero ``xn`` is instead a unit DIRECTION toward the light (a
    directional light or an environment sample), which then does not move
    with ``p``. ``ior`` is the caster's index behind its geometric normal
    ``ng`` (index 1 in front). ``(ep_u, ep_v)`` and ``(en_u, en_v)`` are the
    displacements the A and C blocks differentiate along: the previous and
    next vertices' frames, or for the light end two unit vectors
    perpendicular to the last leg (a point end then moves across the plane
    perpendicular to the emitted ray; a fixed direction turns by exactly that
    vector, so its C block is a solid-angle derivative).

    Returns ``(c, a_blk, b_blk, c_blk, eta, valid)``. ``valid == 0`` for a
    degenerate configuration: a zero-length segment, a vanishing half vector
    (index-matched and straight), or a vertex whose two neighbours lie on the
    SAME side of its plane -- the reflection-like solutions of the projected
    constraint, which are not refractions.
    """
    valid = 1
    vi = xp - p
    li = vi.norm()
    wi = ti.math.vec3(0.0, 0.0, 1.0)
    if li <= MNEE_MIN_DISTANCE:
        valid = 0
    else:
        wi = vi / li
    wo = xn
    lo = 1.0
    if fixed == 0:
        vo = xn - p
        lo = vo.norm()
        if lo <= MNEE_MIN_DISTANCE:
            valid = 0
            wo = ti.math.vec3(0.0, 0.0, 1.0)
        else:
            wo = vo / lo
    if wi.dot(ng) * wo.dot(ng) >= 0.0:
        valid = 0
    eta = ior
    if wi.dot(ng) < 0.0:
        eta = 1.0 / ti.max(ior, 1e-6)
    big_h = -(wi + wo * eta)
    hl = big_h.norm()
    if hl <= 1e-7:
        valid = 0
        hl = 1.0
    h = big_h / hl
    sig = dp_du - n * dp_du.dot(n)
    sl = sig.norm()
    if sl <= 1e-12:
        valid = 0
        sl = 1.0
    s = sig / sl
    t = n.cross(s)
    c = ti.math.vec2(h.dot(s), h.dot(t))
    li_s = ti.max(li, MNEE_MIN_DISTANCE)
    lo_s = ti.max(lo, MNEE_MIN_DISTANCE)
    # A block: moving the previous point moves ``wi`` only.
    dhp_u = -(ep_u - wi * wi.dot(ep_u)) / li_s
    dhp_v = -(ep_v - wi * wi.dot(ep_v)) / li_s
    dhp_u = (dhp_u - h * h.dot(dhp_u)) / hl
    dhp_v = (dhp_v - h * h.dot(dhp_v)) / hl
    a_blk = ti.math.vec4(dhp_u.dot(s), dhp_v.dot(s), dhp_u.dot(t), dhp_v.dot(t))
    # B block: moving this vertex moves ``wi``, ``wo`` (unless the light end
    # is a fixed direction) and the frame the half vector is projected on.
    dwi_u = -(dp_du - wi * wi.dot(dp_du)) / li_s
    dwi_v = -(dp_dv - wi * wi.dot(dp_dv)) / li_s
    dwo_u = ti.math.vec3(0.0, 0.0, 0.0)
    dwo_v = ti.math.vec3(0.0, 0.0, 0.0)
    if fixed == 0:
        dwo_u = -(dp_du - wo * wo.dot(dp_du)) / lo_s
        dwo_v = -(dp_dv - wo * wo.dot(dp_dv)) / lo_s
    dh_u = -(dwi_u + dwo_u * eta)
    dh_v = -(dwi_v + dwo_v * eta)
    dh_u = (dh_u - h * h.dot(dh_u)) / hl
    dh_v = (dh_v - h * h.dot(dh_v)) / hl
    dsig_u = -(n * dp_du.dot(dn_du) + dn_du * dp_du.dot(n))
    dsig_v = -(n * dp_du.dot(dn_dv) + dn_dv * dp_du.dot(n))
    ds_u = (dsig_u - s * s.dot(dsig_u)) / sl
    ds_v = (dsig_v - s * s.dot(dsig_v)) / sl
    dt_u = dn_du.cross(s) + n.cross(ds_u)
    dt_v = dn_dv.cross(s) + n.cross(ds_v)
    b_blk = ti.math.vec4(
        dh_u.dot(s) + h.dot(ds_u), dh_v.dot(s) + h.dot(ds_v),
        dh_u.dot(t) + h.dot(dt_u), dh_v.dot(t) + h.dot(dt_v))
    # C block: moving the next point (or turning the fixed direction) moves
    # ``wo`` only.
    dhn_u = en_u
    dhn_v = en_v
    if fixed == 0:
        dhn_u = (en_u - wo * wo.dot(en_u)) / lo_s
        dhn_v = (en_v - wo * wo.dot(en_v)) / lo_s
    dhn_u = -dhn_u * eta
    dhn_v = -dhn_v * eta
    dhn_u = (dhn_u - h * h.dot(dhn_u)) / hl
    dhn_v = (dhn_v - h * h.dot(dhn_v)) / hl
    c_blk = ti.math.vec4(dhn_u.dot(s), dhn_v.dot(s), dhn_u.dot(t), dhn_v.dot(t))
    return c, a_blk, b_blk, c_blk, eta, valid


@ti.func
def _pt_mnee_orthonormal(w):
    """Two unit vectors perpendicular to unit ``w`` and to each other."""
    axis = ti.math.vec3(1.0, 0.0, 0.0)
    if ti.abs(w[0]) > 0.9:
        axis = ti.math.vec3(0.0, 1.0, 0.0)
    b1 = w.cross(axis).normalized()
    b2 = w.cross(b1)
    return b1, b2


@ti.func
def _pt_mnee_next_tri(refit: ti.template(), ro, rd, f, t_prev, layer_prev, max_t,
                      layer_offset, nodes: ti.template(), miss: ti.template(),
                      leaf: ti.template(), span: ti.template(), first_leaf,
                      tri_pos: ti.template()):
    """Nearest SHADOW-CASTING triangle strictly after ``(t_prev, layer_prev)``
    -- ``_pt_medium_next``'s query with the caster test switched on, because
    every walk here stands in for a shadow ray: a mob that declared
    ``casts_shadows = False`` neither blocks nor bends one. The Bezier arm is
    compile-time absent, so its placeholder arguments are never read.
    """
    inv_rd = ti.math.vec3(_safe_inverse(rd[0]), _safe_inverse(rd[1]),
                          _safe_inverse(rd[2]))
    found, t, layer, prim, _kind, a, b, _border, edge = _nearest_surface_g(
        refit, 1, 0, ro, rd, inv_rd, f, ti.cast(f, ti.f32), t_prev, layer_prev,
        max_t, 0.0, 0.0, None, layer_offset,
        nodes, miss, leaf, span, first_leaf, tri_pos,
        nodes, miss, leaf, span, 0, tri_pos, tri_pos, tri_pos,
        -1, -1, 0.0, 0.0, tri_pos, 0, 1)
    return found, t, layer, prim, a, b, edge


@ti.func
def _pt_mnee_probe(refit: ti.template(), ro, rd, max_t, f, layer_offset,
                   nodes: ti.template(), miss: ti.template(), leaf: ti.template(),
                   span: ti.template(), first_leaf, tri_pos: ti.template(),
                   tri_colors: ti.template(), tri_uvs: ti.template(),
                   tri_tex_meta: ti.template(), textures: ti.template(),
                   tri_extra: ti.template(), circuit_meta: ti.template(),
                   num_colored_triangles, shell: ti.template(), caster_offset):
    """Caster crossings along one straight shadow segment ``[0, max_t)``.

    Peels the segment in the shadow march's order (seam duplicates merged),
    recording up to ``MNEE_MAX_VERTICES`` crossings of covered, refracting
    caster surfaces -- the seed chain of a manifold walk -- and the factor
    the STRAIGHT connection keeps through them: a caster bends what it
    passes, so only its uncovered fraction ``1 - alpha`` lets the straight
    ray by. The shadow march, which the caller still runs for everything
    else, instead passes a covered transmissive surface tinted; ``straight``
    is the per-channel ratio that turns the march's factor at each caster
    crossing into ``1 - alpha``, so ``march * straight`` is the straight
    connection with casters bending rather than tinting. (The march's chord
    absorption pairs only full-coverage hits, whose straight share is zero
    either way.)

    Every other surface is skipped, opaque ones included: like Cycles' seed
    ray this collects the casters along the whole segment, because the bent
    chain leaves the straight line and may pass what blocks it. Returns
    ``(count, prims, as, bs, straight, overflow)``; ``count`` is the number
    of recorded crossings and ``overflow`` is 1 when the segment crossed more
    refracting casters than a chain may hold.
    """
    count = 0
    prims = ti.Vector([-1] * MNEE_MAX_VERTICES)
    bary_a = ti.Vector([0.0] * MNEE_MAX_VERTICES)
    bary_b = ti.Vector([0.0] * MNEE_MAX_VERTICES)
    straight = ti.math.vec3(1.0, 1.0, 1.0)
    overflow = 0
    t_prev = 0.0
    layer_prev = 1e30
    seam = -1e30
    steps = 0
    while steps < max_surfaces_per_ray:
        found, t, layer, prim, a, b, edge = _pt_mnee_next_tri(
            refit, ro, rd, f, t_prev, layer_prev, max_t, layer_offset,
            nodes, miss, leaf, span, first_leaf, tri_pos)
        if (found == 0) or (t >= max_t):
            break
        steps += 1
        duplicate = (edge != 0) and (t - seam <= depth_tie_epsilon)
        seam = t if edge != 0 else -1e30
        t_prev = t
        layer_prev = layer
        if (not duplicate) and (_pt_caster_id(shell, f, prim, caster_offset) >= 0):
            _ior, refracts = _pt_mnee_surface(
                f, prim, a, b, tri_extra, tri_uvs, tri_tex_meta, textures,
                num_colored_triangles)
            w0 = 1.0 - a - b
            color4, alpha = _flat_triangle_color(
                f, prim, w0, a, b, tri_colors, tri_uvs, tri_tex_meta, textures,
                num_colored_triangles)
            alpha = ti.math.clamp(alpha, 0.0, 1.0)
            if (refracts == 1) and (alpha > 1e-6):
                if count < MNEE_MAX_VERTICES:
                    for q in ti.static(range(MNEE_MAX_VERTICES)):
                        if q == count:
                            prims[q] = prim
                            bary_a[q] = a
                            bary_b[q] = b
                    count += 1
                else:
                    overflow = 1
                tint = ti.math.clamp(
                    ti.math.vec3(color4[0], color4[1], color4[2]), 0.0, 1.0)
                passed = _shadow_pass_through(f, prim, 1, w0, a, b, tri_extra,
                                              circuit_meta, tint)
                for k in ti.static(range(3)):
                    marched = 1.0 - alpha * (1.0 - passed[k])
                    if marched > 1e-12:
                        straight[k] *= (1.0 - alpha) / marched
                    else:
                        straight[k] = 0.0
    return count, prims, bary_a, bary_b, straight, overflow


@ti.func
def _pt_mnee_segment_clear(refit: ti.template(), ro, rd, max_t, f, layer_offset,
                           nodes: ti.template(), miss: ti.template(),
                           leaf: ti.template(), span: ti.template(), first_leaf,
                           tri_pos: ti.template(), shell: ti.template(),
                           caster_offset):
    """1 when no caster surface lies on the segment ``[0, max_t)``.

    A bent chain's straight legs must not cross a caster anywhere but at
    their own vertices (which the callers' spawn offsets and pulled-back ends
    exclude): a second crossing would bend the leg too -- a non-convex
    caster, or another caster in the way. Everything else on a leg is the
    shadow march's, which the caller runs separately.
    """
    clear = 1
    t_prev = 0.0
    layer_prev = 1e30
    steps = 0
    while steps < max_surfaces_per_ray:
        found, t, layer, prim, _a, _b, _edge = _pt_mnee_next_tri(
            refit, ro, rd, f, t_prev, layer_prev, max_t, layer_offset,
            nodes, miss, leaf, span, first_leaf, tri_pos)
        if (found == 0) or (t >= max_t):
            break
        steps += 1
        if _pt_caster_id(shell, f, prim, caster_offset) >= 0:
            clear = 0
            break
        t_prev = t
        layer_prev = layer
    return clear


@ti.func
def _pt_mnee_project(refit: ti.template(), origin, direction, max_t, f, cid,
                     layer_offset, nodes: ti.template(), miss: ti.template(),
                     leaf: ti.template(), span: ti.template(), first_leaf,
                     tri_pos: ti.template(), shell: ti.template(),
                     caster_offset):
    """First crossing of caster ``cid`` along ``origin + t * direction``,
    ``t < max_t``: the walk's projection of a tentative point back onto its
    caster once it has left its triangle. Returns ``(found, prim, a, b)``.
    """
    hit = 0
    prim_out = -1
    a_out = 0.0
    b_out = 0.0
    t_prev = 0.0
    layer_prev = 1e30
    steps = 0
    while steps < MNEE_PROJECTION_PEELS:
        found, t, layer, prim, a, b, _edge = _pt_mnee_next_tri(
            refit, origin, direction, f, t_prev, layer_prev, max_t,
            layer_offset, nodes, miss, leaf, span, first_leaf, tri_pos)
        if (found == 0) or (t >= max_t):
            break
        steps += 1
        if _pt_caster_id(shell, f, prim, caster_offset) == cid:
            hit = 1
            prim_out = prim
            a_out = a
            b_out = b
            break
        t_prev = t
        layer_prev = layer
    return hit, prim_out, a_out, b_out


@ti.func
def _pt_mnee_barycentric(f, prim, p, tri_pos: ti.template()):
    """Barycentrics ``(a, b)`` of a point in triangle ``prim``'s plane, and
    whether it lies inside the triangle (with ``MNEE_BARY_SLACK``).
    """
    tp = f % tri_pos.shape[0]
    v0 = ti.math.vec3(tri_pos[tp, prim, 0], tri_pos[tp, prim, 1],
                      tri_pos[tp, prim, 2])
    v1 = ti.math.vec3(tri_pos[tp, prim, 3], tri_pos[tp, prim, 4],
                      tri_pos[tp, prim, 5])
    v2 = ti.math.vec3(tri_pos[tp, prim, 6], tri_pos[tp, prim, 7],
                      tri_pos[tp, prim, 8])
    e1 = v1 - v0
    e2 = v2 - v0
    d = p - v0
    g11 = e1.dot(e1)
    g12 = e1.dot(e2)
    g22 = e2.dot(e2)
    r1 = e1.dot(d)
    r2 = e2.dot(d)
    det = g11 * g22 - g12 * g12
    a = 0.0
    b = 0.0
    inside = 0
    if ti.abs(det) > 1e-30:
        a = (g22 * r1 - g12 * r2) / det
        b = (g11 * r2 - g12 * r1) / det
        if (a >= -MNEE_BARY_SLACK) and (b >= -MNEE_BARY_SLACK) \
                and (a + b <= 1.0 + MNEE_BARY_SLACK):
            inside = 1
    return a, b, inside


@ti.func
def _pt_mnee_chain(x0, y, fixed, count, prims, bary_a, bary_b, iors, flats, f,
                   tri_pos: ti.template(), tri_norm: ti.template()):
    """Constraint, blocks and frames of a whole chain at its current vertices.

    Returns ``(c_norm, c_sq, cs, a_blks, b_blks, c_blks, ps, nss, ngs, us,
    vs, etas, valid, c_floor)`` with one row per vertex (rows past ``count``
    unused): ``c_norm`` is the largest vertex's constraint length (the
    convergence test) and ``c_sq`` the chain's sum of squares (what a Newton
    step must decrease). ``c_floor`` is the largest vertex's constraint noise
    in f32 (``MNEE_ROUNDING``): each leg's direction error, the rounding of
    its two ends over its length, through the half vector, whose length a
    refraction keeps at least ``|1 - eta|``.
    The last vertex's C block is with respect to the light end, taken along
    two unit vectors perpendicular to the last leg: for a fixed direction
    that is the solid-angle measure itself, and for a point it is the area
    perpendicular to the emitted ray (``_pt_mnee_geometry`` says what the
    caller converts).
    """
    ps = ti.Matrix.zero(ti.f32, MNEE_MAX_VERTICES, 3)
    nss = ti.Matrix.zero(ti.f32, MNEE_MAX_VERTICES, 3)
    ngs = ti.Matrix.zero(ti.f32, MNEE_MAX_VERTICES, 3)
    us = ti.Matrix.zero(ti.f32, MNEE_MAX_VERTICES, 3)
    vs = ti.Matrix.zero(ti.f32, MNEE_MAX_VERTICES, 3)
    dus = ti.Matrix.zero(ti.f32, MNEE_MAX_VERTICES, 3)
    dvs = ti.Matrix.zero(ti.f32, MNEE_MAX_VERTICES, 3)
    exts = ti.Vector([0.0] * MNEE_MAX_VERTICES)
    valid = 1
    for q in ti.static(range(MNEE_MAX_VERTICES)):
        if q < count:
            p, n, ng, du, dv, dndu, dndv, ok = _pt_mnee_vertex(
                f, prims[q], bary_a[q], bary_b[q], flats[q], tri_pos, tri_norm)
            exts[q] = _pt_mnee_extent(f, prims[q], tri_pos)
            if ok == 0:
                valid = 0
            for k in ti.static(range(3)):
                ps[q, k] = p[k]
                nss[q, k] = n[k]
                ngs[q, k] = ng[k]
                us[q, k] = du[k]
                vs[q, k] = dv[k]
                dus[q, k] = dndu[k]
                dvs[q, k] = dndv[k]
    cs = ti.Matrix.zero(ti.f32, MNEE_MAX_VERTICES, 2)
    a_blks = ti.Matrix.zero(ti.f32, MNEE_MAX_VERTICES, 4)
    b_blks = ti.Matrix.zero(ti.f32, MNEE_MAX_VERTICES, 4)
    c_blks = ti.Matrix.zero(ti.f32, MNEE_MAX_VERTICES, 4)
    etas = ti.Vector([1.0] * MNEE_MAX_VERTICES)
    c_norm = 0.0
    c_sq = 0.0
    c_floor = 0.0
    zero3 = ti.math.vec3(0.0, 0.0, 0.0)
    for q in ti.static(range(MNEE_MAX_VERTICES)):
        if q < count:
            p = ti.math.vec3(ps[q, 0], ps[q, 1], ps[q, 2])
            xp = x0
            m_prev = ti.max(ti.abs(x0[0]), ti.max(ti.abs(x0[1]),
                                                  ti.abs(x0[2])))
            ep_u = zero3
            ep_v = zero3
            if ti.static(q > 0):
                xp = ti.math.vec3(ps[q - 1, 0], ps[q - 1, 1], ps[q - 1, 2])
                m_prev = exts[q - 1]
                ep_u = ti.math.vec3(us[q - 1, 0], us[q - 1, 1], us[q - 1, 2])
                ep_v = ti.math.vec3(vs[q - 1, 0], vs[q - 1, 1], vs[q - 1, 2])
            xn = y
            m_next = ti.max(ti.abs(y[0]), ti.max(ti.abs(y[1]), ti.abs(y[2])))
            fx = fixed
            en_u = zero3
            en_v = zero3
            if q < count - 1:
                if ti.static(q + 1 < MNEE_MAX_VERTICES):
                    xn = ti.math.vec3(ps[q + 1, 0], ps[q + 1, 1], ps[q + 1, 2])
                    m_next = exts[q + 1]
                    en_u = ti.math.vec3(us[q + 1, 0], us[q + 1, 1], us[q + 1, 2])
                    en_v = ti.math.vec3(vs[q + 1, 0], vs[q + 1, 1], vs[q + 1, 2])
                fx = 0
            else:
                w_l = y
                if fixed == 0:
                    w_l = (y - p).normalized()
                en_u, en_v = _pt_mnee_orthonormal(w_l)
            c, a_blk, b_blk, c_blk, eta, ok = _pt_mnee_constraint(
                xp, p, xn, fx,
                ti.math.vec3(nss[q, 0], nss[q, 1], nss[q, 2]),
                ti.math.vec3(ngs[q, 0], ngs[q, 1], ngs[q, 2]),
                ti.math.vec3(us[q, 0], us[q, 1], us[q, 2]),
                ti.math.vec3(vs[q, 0], vs[q, 1], vs[q, 2]),
                ti.math.vec3(dus[q, 0], dus[q, 1], dus[q, 2]),
                ti.math.vec3(dvs[q, 0], dvs[q, 1], dvs[q, 2]),
                iors[q], ep_u, ep_v, en_u, en_v)
            if ok == 0:
                valid = 0
            etas[q] = eta
            # A fixed light end is an exact direction: only its incoming leg
            # rounds.
            noise = (m_prev + exts[q]) / ti.max((xp - p).norm(),
                                                MNEE_MIN_DISTANCE)
            if fx == 0:
                noise += eta * (m_next + exts[q]) / ti.max(
                    (xn - p).norm(), MNEE_MIN_DISTANCE)
            c_floor = ti.max(c_floor, MNEE_ROUNDING * noise
                             / ti.max(ti.abs(1.0 - eta), 1e-3))
            cs[q, 0] = c[0]
            cs[q, 1] = c[1]
            for k in ti.static(range(4)):
                a_blks[q, k] = a_blk[k]
                b_blks[q, k] = b_blk[k]
                c_blks[q, k] = c_blk[k]
            c_norm = ti.max(c_norm, c.norm())
            c_sq += c.dot(c)
    return c_norm, c_sq, cs, a_blks, b_blks, c_blks, ps, nss, ngs, us, vs, \
        etas, valid, c_floor


@ti.func
def _pt_mnee_step(count, cs, a_blks, b_blks, c_blks):
    """Newton direction of the block-tridiagonal system ``J delta = c``
    (block Thomas algorithm). Returns ``(deltas, ok)``; ``ok == 0`` when a
    pivot block is singular.
    """
    ok = 1
    deltas = ti.Matrix.zero(ti.f32, MNEE_MAX_VERTICES, 2)
    ys = ti.Matrix.zero(ti.f32, MNEE_MAX_VERTICES, 2)
    us = ti.Matrix.zero(ti.f32, MNEE_MAX_VERTICES, 4)
    u_prev = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
    y_prev = ti.math.vec2(0.0, 0.0)
    for q in ti.static(range(MNEE_MAX_VERTICES)):
        if q < count:
            b_q = ti.math.vec4(b_blks[q, 0], b_blks[q, 1], b_blks[q, 2],
                               b_blks[q, 3])
            rhs = ti.math.vec2(cs[q, 0], cs[q, 1])
            if ti.static(q > 0):
                a_q = ti.math.vec4(a_blks[q, 0], a_blks[q, 1], a_blks[q, 2],
                                   a_blks[q, 3])
                b_q = b_q - _m2_mul(a_q, u_prev)
                rhs = rhs - _m2_vec(a_q, y_prev)
            det = _m2_det(b_q)
            if ti.abs(det) <= MNEE_MIN_DETERMINANT:
                ok = 0
                det = 1.0
            l_inv = _m2_inv(b_q, det)
            c_q = ti.math.vec4(c_blks[q, 0], c_blks[q, 1], c_blks[q, 2],
                               c_blks[q, 3])
            u_prev = _m2_mul(l_inv, c_q)
            y_prev = _m2_vec(l_inv, rhs)
            for k in ti.static(range(4)):
                us[q, k] = u_prev[k]
            ys[q, 0] = y_prev[0]
            ys[q, 1] = y_prev[1]
    d_next = ti.math.vec2(0.0, 0.0)
    for qq in ti.static(range(MNEE_MAX_VERTICES)):
        q = MNEE_MAX_VERTICES - 1 - qq
        if q < count:
            d_q = ti.math.vec2(ys[q, 0], ys[q, 1])
            if q < count - 1:
                u_q = ti.math.vec4(us[q, 0], us[q, 1], us[q, 2], us[q, 3])
                d_q = d_q - _m2_vec(u_q, d_next)
            deltas[q, 0] = d_q[0]
            deltas[q, 1] = d_q[1]
            d_next = d_q
    return deltas, ok


@ti.func
def _pt_mnee_transfer(count, a_blks, b_blks, c_blks):
    """``T = d(x_1)/d(light)``: how the first vertex slides, in its own
    in-plane frame, per unit displacement of the light end along the frame
    the last vertex's C block was taken in (implicit function theorem on the
    solved chain, ``J dX + dC/dy = 0``). Returns ``(T, ok)``.
    """
    ok = 1
    # Forward elimination, as in ``_pt_mnee_step``, keeping
    # ``U_q = L_q^-1 C_q`` so the last column can be solved for.
    us = ti.Matrix.zero(ti.f32, MNEE_MAX_VERTICES, 4)
    u_prev = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
    for q in ti.static(range(MNEE_MAX_VERTICES)):
        if q < count:
            b_q = ti.math.vec4(b_blks[q, 0], b_blks[q, 1], b_blks[q, 2],
                               b_blks[q, 3])
            if ti.static(q > 0):
                a_q = ti.math.vec4(a_blks[q, 0], a_blks[q, 1], a_blks[q, 2],
                                   a_blks[q, 3])
                b_q = b_q - _m2_mul(a_q, u_prev)
            det = _m2_det(b_q)
            if ti.abs(det) <= MNEE_MIN_DETERMINANT:
                ok = 0
                det = 1.0
            l_inv = _m2_inv(b_q, det)
            c_q = ti.math.vec4(c_blks[q, 0], c_blks[q, 1], c_blks[q, 2],
                               c_blks[q, 3])
            u_prev = _m2_mul(l_inv, c_q)
            for k in ti.static(range(4)):
                us[q, k] = u_prev[k]
    # The light enters only the LAST constraint, through its C block D:
    # J X = -[0 .. 0 D]^T. Forward substitution leaves Y_last = -L_last^-1 D
    # (the earlier rows of Y are zero) -- which is ``-U_last`` -- and back
    # substitution X_q = -U_q X_{q+1}.
    x = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
    for qq in ti.static(range(MNEE_MAX_VERTICES)):
        q = MNEE_MAX_VERTICES - 1 - qq
        if q < count:
            u_q = ti.math.vec4(us[q, 0], us[q, 1], us[q, 2], us[q, 3])
            if q == count - 1:
                x = -u_q
            else:
                x = -_m2_mul(u_q, x)
    return x, ok


@ti.func
def _pt_mnee_geometry(x0, count, ps, ngs, a_blks, b_blks, c_blks):
    """Generalized geometry term of a solved chain: the solid angle the
    chain's first leg subtends at ``x0`` per unit measure at the light end,
    ``|w0 . ng_1| / l0^2 * |det T|``.

    That measure is the one the last vertex's C block was taken in
    (``_pt_mnee_chain``): area perpendicular to the emitted ray for a point
    end, solid angle for a fixed direction. A point end on a surface (an
    emissive triangle) converts with the emitter's ``|cos|``; an index
    change along the chain with ``(n_0 / n_L)^2``. Both are the caller's.
    Unobstructed, the term is ``1 / l0^2`` (a point) or 1 (a direction), so
    their ratio to it is the caustic's concentration. Returns ``(g, ok)``.
    """
    t_mat, ok = _pt_mnee_transfer(count, a_blks, b_blks, c_blks)
    p1 = ti.math.vec3(ps[0, 0], ps[0, 1], ps[0, 2])
    n1 = ti.math.vec3(ngs[0, 0], ngs[0, 1], ngs[0, 2])
    leg = p1 - x0
    l2 = leg.dot(leg)
    g = 0.0
    if (ok == 1) and (l2 > MNEE_MIN_DISTANCE * MNEE_MIN_DISTANCE):
        cos1 = ti.abs(leg.dot(n1)) / ti.sqrt(l2)
        g = cos1 / l2 * ti.abs(_m2_det(t_mat))
    else:
        ok = 0
    return g, ok


@ti.func
def _pt_mnee_sides(x0, y, fixed, count, prims, bary_a, bary_b, f,
                   tri_pos: ti.template()):
    """1 when every vertex of the chain has its two neighbours on opposite
    sides of its triangle: a geometric crossing, the camera path's own
    requirement of a refraction (``pt_shade``'s ``geometric_cross``).
    """
    ok = 1
    pts = ti.Matrix.zero(ti.f32, MNEE_MAX_VERTICES, 3)
    nrms = ti.Matrix.zero(ti.f32, MNEE_MAX_VERTICES, 3)
    for q in ti.static(range(MNEE_MAX_VERTICES)):
        if q < count:
            p, ng = _pt_mnee_point(f, prims[q], bary_a[q], bary_b[q], tri_pos)
            for k in ti.static(range(3)):
                pts[q, k] = p[k]
                nrms[q, k] = ng[k]
    for q in ti.static(range(MNEE_MAX_VERTICES)):
        if q < count:
            p = ti.math.vec3(pts[q, 0], pts[q, 1], pts[q, 2])
            ng = ti.math.vec3(nrms[q, 0], nrms[q, 1], nrms[q, 2])
            wi = x0 - p
            if ti.static(q > 0):
                wi = ti.math.vec3(pts[q - 1, 0], pts[q - 1, 1],
                                  pts[q - 1, 2]) - p
            wo = y - p
            if fixed != 0:
                wo = y
            if q < count - 1:
                if ti.static(q + 1 < MNEE_MAX_VERTICES):
                    wo = ti.math.vec3(pts[q + 1, 0], pts[q + 1, 1],
                                      pts[q + 1, 2]) - p
            if wi.dot(ng) * wo.dot(ng) >= 0.0:
                ok = 0
    return ok


@ti.func
def _pt_mnee_materials(count, prims, bary_a, bary_b, f, tri_mat: ti.template(),
                       tri_extra: ti.template(), tri_uvs: ti.template(),
                       tri_tex_meta: ti.template(), textures: ti.template(),
                       num_colored_triangles):
    """Per-vertex ``(iors, flats, refracts)`` at the chain's current points:
    the index the walk bends with, the material's flat-shading blend (slot
    10, as ``pt_shade`` reads it for a built-in pipeline), and whether every
    vertex still refracts (``_pt_mnee_surface``).
    """
    iors = ti.Vector([1.0] * MNEE_MAX_VERTICES)
    flats = ti.Vector([0.0] * MNEE_MAX_VERTICES)
    refracts = 1
    tm = f % tri_mat.shape[0]
    for q in ti.static(range(MNEE_MAX_VERTICES)):
        if q < count:
            ior, ok = _pt_mnee_surface(
                f, prims[q], bary_a[q], bary_b[q], tri_extra, tri_uvs,
                tri_tex_meta, textures, num_colored_triangles)
            iors[q] = ior
            flats[q] = tri_mat[tm, prims[q], 10]
            if ok == 0:
                refracts = 0
    return iors, flats, refracts


@ti.func
def _pt_mnee_solve(refit: ti.template(), x0, n_recv, y, fixed, count, prims,
                   bary_a, bary_b, cids, f, layer_offset, nodes: ti.template(),
                   miss: ti.template(), leaf: ti.template(), span: ti.template(),
                   first_leaf, tri_pos: ti.template(), tri_norm: ti.template(),
                   tri_mat: ti.template(), tri_extra: ti.template(),
                   tri_uvs: ti.template(), tri_tex_meta: ti.template(),
                   textures: ti.template(), num_colored_triangles,
                   shell: ti.template(), caster_offset):
    """Walk the specular manifold from the seed chain to a solution.

    Damped Newton with backtracking, after Cycles' ``mnee_newton_solver``:
    each step moves every vertex along its tangent plane by ``beta`` times
    the full Newton direction and puts it back on the caster -- exactly, by
    its barycentrics, while it stays inside its triangle, otherwise along a
    ray from the previous chain point through it, keeping the first crossing
    of the same caster (``_pt_mnee_project``). A step is accepted only when
    every vertex still refracts as a geometric crossing AND the chain's sum
    of squared constraints went down; a rejected step halves ``beta``, an
    accepted one doubles it, up to 1. The decrease test is this walk's
    addition to Cycles': the half vector saturates far from a solution, so
    an unchecked Newton step from a wide-angle seed can fly off the caster
    altogether. ``n_recv`` is the receiver's geometric normal on the side the
    light arrives from (zero for a medium vertex): a first vertex stepped
    below the receiver's plane is a rejected step too, since the receiver
    could not see it.

    ``x0`` and ``y`` are the chain's fixed ends (``y`` a unit direction when
    ``fixed``), ``prims`` / ``bary_a`` / ``bary_b`` the seed vertices from
    ``_pt_mnee_probe`` and ``cids`` their caster ids. Returns ``(ok, capped,
    prims, bary_a, bary_b, ps, ngs, etas, a_blks, b_blks, c_blks, c_floor)``:
    whether the walk converged, whether it gave up at ``MNEE_MAX_ITERATIONS``
    still walking (counted by the caller), the solved vertices, their points
    and geometric normals, the index ratios across them, the Jacobian blocks
    at the solution (``_pt_mnee_geometry``'s input) and the f32 noise floor
    of its constraint. Converged is every constraint below
    ``MNEE_THRESHOLD`` or, where the chain's legs are too short to resolve
    that, below its own noise floor (``MNEE_ROUNDING``).
    """
    ok = 0
    capped = 0
    iors, flats, refracts = _pt_mnee_materials(
        count, prims, bary_a, bary_b, f, tri_mat, tri_extra, tri_uvs,
        tri_tex_meta, textures, num_colored_triangles)
    c_norm, c_sq, cs, a_blks, b_blks, c_blks, ps, _nss, ngs, us, vs, etas, \
        valid, c_floor = _pt_mnee_chain(x0, y, fixed, count, prims, bary_a,
                                        bary_b, iors, flats, f, tri_pos,
                                        tri_norm)
    deltas = ti.Matrix.zero(ti.f32, MNEE_MAX_VERTICES, 2)
    step_ok = 0
    if (valid == 1) and (refracts == 1):
        if c_norm < ti.max(MNEE_THRESHOLD, c_floor):
            ok = 1
        else:
            deltas, step_ok = _pt_mnee_step(count, cs, a_blks, b_blks, c_blks)
    # The accepted chain: what the next tentative step starts from.
    ps_cur = ps
    ngs_cur = ngs
    us_cur = us
    vs_cur = vs
    c_sq_cur = c_sq
    c_floor_t = c_floor
    recv_side = n_recv.norm() > 0.0
    beta = MNEE_INITIAL_STEP
    it = 0
    walking = (ok == 0) and (step_ok == 1)
    while walking and (it < MNEE_MAX_ITERATIONS):
        it += 1
        t_prims = prims
        t_a = bary_a
        t_b = bary_b
        reject = 0
        for q in ti.static(range(MNEE_MAX_VERTICES)):
            if (q < count) and (reject == 0):
                p = ti.math.vec3(ps_cur[q, 0], ps_cur[q, 1], ps_cur[q, 2])
                move = ti.math.vec3(us_cur[q, 0], us_cur[q, 1], us_cur[q, 2]) \
                    * deltas[q, 0] \
                    + ti.math.vec3(vs_cur[q, 0], vs_cur[q, 1], vs_cur[q, 2]) \
                    * deltas[q, 1]
                tp = p - move * beta
                if ti.static(q == 0):
                    if recv_side and (n_recv.dot(tp - x0) <= 0.0):
                        reject = 1
                if reject == 0:
                    a_n, b_n, inside = _pt_mnee_barycentric(f, prims[q], tp,
                                                            tri_pos)
                    if inside == 1:
                        t_a[q] = a_n
                        t_b[q] = b_n
                    else:
                        origin = x0
                        if ti.static(q > 0):
                            prev = ti.math.vec3(ps_cur[q - 1, 0],
                                                ps_cur[q - 1, 1],
                                                ps_cur[q - 1, 2])
                            ng_prev = ti.math.vec3(ngs_cur[q - 1, 0],
                                                   ngs_cur[q - 1, 1],
                                                   ngs_cur[q - 1, 2])
                            toward = tp - prev
                            origin = _offset_ray_origin(
                                prev, ng_prev if ng_prev.dot(toward) >= 0.0
                                else -ng_prev)
                        span_v = tp - origin
                        dist = span_v.norm()
                        hit = 0
                        if dist > MNEE_MIN_DISTANCE:
                            hit, prim_h, a_h, b_h = _pt_mnee_project(
                                refit, origin, span_v / dist, 2.0 * dist, f,
                                cids[q], layer_offset, nodes, miss, leaf, span,
                                first_leaf, tri_pos, shell, caster_offset)
                            if hit == 1:
                                t_prims[q] = prim_h
                                t_a[q] = a_h
                                t_b[q] = b_h
                        if hit == 0:
                            reject = 1
        if reject == 0:
            reject = 1 - _pt_mnee_sides(x0, y, fixed, count, t_prims, t_a, t_b,
                                        f, tri_pos)
        if reject == 0:
            iors, flats, refracts = _pt_mnee_materials(
                count, t_prims, t_a, t_b, f, tri_mat, tri_extra, tri_uvs,
                tri_tex_meta, textures, num_colored_triangles)
            c_norm, c_sq, cs, a_blks, b_blks, c_blks, ps, _nss, ngs, us, vs, \
                etas, valid, c_floor_t = _pt_mnee_chain(
                    x0, y, fixed, count, t_prims, t_a, t_b, iors, flats, f,
                    tri_pos, tri_norm)
            if (valid == 0) or (refracts == 0) or (c_sq >= c_sq_cur):
                reject = 1
        if reject == 1:
            beta *= 0.5
            if beta < MNEE_MIN_STEP:
                walking = False
        else:
            prims = t_prims
            bary_a = t_a
            bary_b = t_b
            ps_cur = ps
            ngs_cur = ngs
            us_cur = us
            vs_cur = vs
            c_sq_cur = c_sq
            c_floor = c_floor_t
            if c_norm < ti.max(MNEE_THRESHOLD, c_floor):
                ok = 1
                walking = False
            else:
                deltas, step_ok = _pt_mnee_step(count, cs, a_blks, b_blks,
                                                c_blks)
                if step_ok == 0:
                    walking = False
                beta = ti.min(1.0, 2.0 * beta)
    if walking:
        capped = 1
    return ok, capped, prims, bary_a, bary_b, ps, ngs, etas, a_blks, b_blks, \
        c_blks, c_floor


@ti.func
def _pt_mnee_turn(q, count, x0, y, fixed, prims, bary_a, bary_b, ps, f,
                  tri_mat: ti.template(), tri_pos: ti.template(),
                  tri_norm: ti.template()):
    """Vertex ``q`` of a solved chain as the camera path's delta lobe meets
    it: ``(prim, a, b, rd, spec, ci, ok)``.

    ``rd`` is the direction the chain arrives along, ``spec`` the shading
    normal turned to face it (the camera path's ``spec_n``; a refraction
    leaves through its other side) and ``ci`` the cosine there; ``ok`` is 0
    when the chain arrives grazing or leaves on the arriving side, which a
    refraction never does.
    """
    prim_q = -1
    a_q = 0.0
    b_q = 0.0
    p_q = x0
    prev = x0
    nxt = y
    nxt_fixed = fixed
    for qq in ti.static(range(MNEE_MAX_VERTICES)):
        if qq == q:
            prim_q = prims[qq]
            a_q = bary_a[qq]
            b_q = bary_b[qq]
            p_q = ti.math.vec3(ps[qq, 0], ps[qq, 1], ps[qq, 2])
            if ti.static(qq > 0):
                prev = ti.math.vec3(ps[qq - 1, 0], ps[qq - 1, 1], ps[qq - 1, 2])
            if ti.static(qq + 1 < MNEE_MAX_VERTICES):
                if qq + 1 < count:
                    nxt = ti.math.vec3(ps[qq + 1, 0], ps[qq + 1, 1],
                                       ps[qq + 1, 2])
                    nxt_fixed = 0
    rd_q = (p_q - prev).normalized()
    wo_q = nxt
    if nxt_fixed == 0:
        wo_q = (nxt - p_q).normalized()
    tm_q = f % tri_mat.shape[0]
    _p_q, n_q, _ng_q, _du_q, _dv_q, _dndu_q, _dndv_q, _ok_q = _pt_mnee_vertex(
        f, prim_q, a_q, b_q, tri_mat[tm_q, prim_q, 10], tri_pos, tri_norm)
    spec_q = n_q
    if n_q.dot(rd_q) > 0.0:
        spec_q = -n_q
    ci_q = spec_q.dot(-rd_q)
    ok = 1
    if (ci_q <= 1e-6) or (spec_q.dot(wo_q) >= -1e-6):
        ok = 0
    return prim_q, a_q, b_q, rd_q, spec_q, ci_q, ok


@ti.func
def _pt_mnee_leg(q, count, x0, sorigin, n_recv, y, fixed, prims, ps, ngs):
    """Leg ``q`` of the solved chain ``x0 -> x_1 .. x_count -> y`` and the
    straight shadow ray that tests it: ``(origin, ray_d, reach, leg_d,
    leg_len, inner, b_prim, b_ng, b_fixed)``.

    The ray leaves from ``sorigin`` (the receiver's own shadow-ray origin) on
    the first leg and from the near vertex offset to the leg's side of its
    triangle on the others, and aims at the far vertex itself: parallel to
    the leg from the offset origin it met the vertex's own triangle beside
    the vertex, short of the end by about the offset over the leg's cosine
    there -- past the end's pull-back on a grazing leg, so the leg read as
    blocked by its own glass. ``leg_d`` / ``leg_len`` are the leg's own
    direction and length (``1e7`` toward a light at infinity). ``inner`` is
    the caster triangle whose solid the leg runs inside -- it leaves through
    its far vertex, or (a light inside the solid) entered at its near one --
    else -1; ``b_prim`` / ``b_ng`` / ``b_fixed`` are the far end's triangle
    (-1 at the light end), geometric normal and fixedness.
    """
    a_pt = x0
    a_ng = n_recv
    a_prim = -1
    b_pt = y
    b_ng = ti.math.vec3(0.0, 0.0, 0.0)
    b_prim = -1
    b_fixed = fixed
    for qq in ti.static(range(MNEE_MAX_VERTICES + 1)):
        if qq == q:
            if ti.static(qq > 0):
                a_pt = ti.math.vec3(ps[qq - 1, 0], ps[qq - 1, 1], ps[qq - 1, 2])
                a_ng = ti.math.vec3(ngs[qq - 1, 0], ngs[qq - 1, 1],
                                    ngs[qq - 1, 2])
                a_prim = prims[qq - 1]
            if ti.static(qq < MNEE_MAX_VERTICES):
                if qq < count:
                    b_pt = ti.math.vec3(ps[qq, 0], ps[qq, 1], ps[qq, 2])
                    b_ng = ti.math.vec3(ngs[qq, 0], ngs[qq, 1], ngs[qq, 2])
                    b_prim = prims[qq]
                    b_fixed = 0
    leg_d = b_pt
    leg_len = 1e7
    if b_fixed == 0:
        span_l = b_pt - a_pt
        leg_len = span_l.norm()
        leg_d = span_l / ti.max(leg_len, 1e-30)
    origin = sorigin
    if q > 0:
        origin = _offset_ray_origin(
            a_pt, a_ng if a_ng.dot(leg_d) >= 0.0 else -a_ng)
    ray_d = leg_d
    reach = leg_len - (origin - a_pt).dot(leg_d)
    if b_fixed == 0:
        aim = b_pt - origin
        reach = aim.norm()
        ray_d = aim / ti.max(reach, 1e-30)
    inner = -1
    if b_prim >= 0:
        if leg_d.dot(b_ng) > 0.0:
            inner = b_prim
    elif (a_prim >= 0) and (leg_d.dot(a_ng) < 0.0):
        inner = a_prim
    return origin, ray_d, reach, leg_d, leg_len, inner, b_prim, b_ng, b_fixed


@ti.func
def _pt_mnee_admissible(refit: ti.template(), x0, sorigin, n_recv, y, fixed,
                        count, prims, bary_a, bary_b, ps, ngs, f, layer_offset,
                        nodes: ti.template(), miss: ti.template(),
                        leaf: ti.template(), span: ti.template(), first_leaf,
                        tri_pos: ti.template(), tri_norm: ti.template(),
                        tri_mat: ti.template(), shell: ti.template(),
                        caster_offset):
    """1 when a solved chain is one a light sample delivers.

    What a converged walk does not already guarantee: the receiver sees its
    first vertex (the walk keeps every step above its plane, but not the
    seed it started from), every vertex refracts as the camera path's delta
    lobe would (``_pt_mnee_turn``), no leg crosses a caster but at its own
    ends (``_pt_mnee_segment_clear``: a second crossing would bend it too),
    and no light at infinity lies inside a solid. The caustic block
    (``_pt_caustic_chain``) and the coverage test (``_pt_mnee_covered``) both
    decide by this, so that the BSDF path keeps all of what no chain
    delivers.
    """
    ok = 1
    leg0 = ti.math.vec3(ps[0, 0], ps[0, 1], ps[0, 2]) - x0
    if (n_recv.norm() > 0.0) and (n_recv.dot(leg0) <= 0.0):
        ok = 0
    if ok == 1:
        for q in range(count):
            _prim, _a, _b, _rd, _spec, _ci, turn_ok = _pt_mnee_turn(
                q, count, x0, y, fixed, prims, bary_a, bary_b, ps, f, tri_mat,
                tri_pos, tri_norm)
            if turn_ok == 0:
                ok = 0
    if ok == 1:
        for q in range(count + 1):
            if ok == 1:
                origin, ray_d, reach, _leg_d, _leg_len, inner, _b_prim, \
                    _b_ng, b_fixed = _pt_mnee_leg(q, count, x0, sorigin,
                                                  n_recv, y, fixed, prims, ps,
                                                  ngs)
                if (inner >= 0) and (b_fixed == 1):
                    # A light at infinity inside a solid.
                    ok = 0
                elif _pt_mnee_segment_clear(
                        refit, origin, ray_d,
                        _shadow_tmax(origin, ray_d, reach), f, layer_offset,
                        nodes, miss, leaf, span, first_leaf, tri_pos, shell,
                        caster_offset) == 0:
                    ok = 0
    return ok


@ti.func
def _pt_caustic_mis(p_y, jac, p_b):
    """Power-heuristic weight of a light sample's chain against the BSDF path
    that makes the same journey (``1 -`` it is the path's).

    The chain's density in the receiver's solid angle is the light end's
    sampling density ``p_y`` (``n_ls`` samples' worth) over ``jac``, the
    solid angle the chain's first leg sweeps per unit measure at the light
    end (``_pt_mnee_geometry``, times the emitter's ``|cos|`` at a point on a
    triangle); the BSDF path's is ``p_b``, its receiver's sampling density
    toward that leg. A light the glass magnifies (``jac`` large: the focus of
    a caustic seen from the receiver, a sky the ball spreads over a wide cone
    near its contact point) is the BSDF path's to find; a small bright one it
    shrinks is the chain's. Without a light-end density (a delta light, which
    no BSDF path hits) the chain keeps all of it.
    """
    w = 1.0
    if p_y > 0.0:
        ratio = p_b * jac / p_y
        w = 1.0 / (1.0 + ratio * ratio)
    return w


@ti.func
def _pt_cau_receiver(pt_cau: ti.template(), r, x0, n_side, spec, w_b, p_b,
                     direct):
    """Record path ``r``'s receiver for the coverage test (``PT_CAU_WIDTH``)."""
    for c in ti.static(range(3)):
        pt_cau[r, _PT_CAU_X0 + c] = x0[c]
        pt_cau[r, _PT_CAU_NSIDE + c] = n_side[c]
        pt_cau[r, _PT_CAU_SPEC + c] = spec[c]
        pt_cau[r, _PT_CAU_WB + c] = w_b[c]
    pt_cau[r, _PT_CAU_PB] = p_b
    pt_cau[r, _PT_CAU_DIRECT] = ti.cast(direct, ti.f32)


@ti.func
def _pt_cau_pend(pt_cau: ti.template(), r, kind, y, value, w_old, k, f, p_y,
                 n_e, one_sided, clamp):
    """Hold back one emitter path ``r`` found through casters for the coverage
    test; returns whether it did. One per ``pt_shade`` launch: the caller
    weights a second (an emitter the path passes through, then an escape) as
    it would with caustics off.
    """
    held = 0
    if pt_cau[r, _PT_CAU_KIND] < 0.5:
        held = 1
        pt_cau[r, _PT_CAU_KIND] = ti.cast(kind, ti.f32)
        for c in ti.static(range(3)):
            pt_cau[r, _PT_CAU_Y + c] = y[c]
            pt_cau[r, _PT_CAU_VAL + c] = value[c]
            pt_cau[r, _PT_CAU_NE + c] = n_e[c]
        pt_cau[r, _PT_CAU_WOLD] = w_old
        pt_cau[r, _PT_CAU_K] = ti.cast(k, ti.f32)
        pt_cau[r, _PT_CAU_F] = ti.cast(f, ti.f32)
        pt_cau[r, _PT_CAU_PY] = p_y
        pt_cau[r, _PT_CAU_ONE_SIDED] = ti.cast(one_sided, ti.f32)
        pt_cau[r, _PT_CAU_CLAMP] = clamp
    return held


@ti.func
def _pt_mnee_covered(refit: ti.template(), x0, n_side, spec, w_b, y, n_e,
                     one_sided, fixed, k, f, layer_offset,
                     nodes: ti.template(), miss: ti.template(),
                     leaf: ti.template(), span: ti.template(), first_leaf,
                     tri_pos: ti.template(), tri_norm: ti.template(),
                     tri_colors: ti.template(), tri_uvs: ti.template(),
                     tri_tex_meta: ti.template(), textures: ti.template(),
                     tri_extra: ti.template(), tri_mat: ti.template(),
                     circuit_meta: ti.template(), num_colored_triangles,
                     shell: ti.template(), caster_offset):
    """Whether a light sample at receiver ``x0`` would have delivered a BSDF
    path's chain: ``(covered, overflow, jac)``.

    The path left ``x0`` along ``w_b`` and reached the light end ``y`` (a point
    on an emitter whose front normal is ``n_e``, or with ``fixed`` a
    direction) through ``k`` caster interfaces. The caustic block, handed the
    light sample ``y``, makes the same decisions this does -- its horizon
    test on ``spec``, the probe of the straight segment, the walk from the
    probe's seed, the chain's admissibility (``_pt_mnee_admissible``) -- and
    a walk is a pure function of its ends, so the chain it would have
    delivered is the one this walk converges to. The path is covered when
    that chain crosses ``k`` interfaces, leaves ``x0`` along ``w_b`` and
    carries light (not to the back of a ``one_sided`` emitter); then the
    light sample counted it too, and ``jac`` is the solid angle its first leg
    sweeps per unit measure at the light end, which the two strategies'
    weights read (``_pt_caustic_mis``). Anything else -- no caster on the
    straight segment, a walk that fails or finds another solution, a chain
    the light sample rejects -- is light only the BSDF path carries. (The
    one decision not repeated here is the caustic block dropping a chain
    whose legs' medium walk was truncated, ``PT_STAT_MEDIA_*``: the path's
    medium stack at its receiver is not kept, and such a render is already
    reported as truncated.) ``overflow`` is the straight segment crossing
    more casters than a chain holds: the caustic block then left the
    straight connection tinted as with caustics off, and the caller weights
    the path as it was weighted then.
    """
    covered = 0
    overflow = 0
    jac = 0.0
    medium = n_side.norm() <= 0.0
    wi = y
    ldist = 1e7
    if fixed == 0:
        to_y = y - x0
        ldist = to_y.norm()
        wi = to_y / ti.max(ldist, 1e-30)
    if ((fixed == 1) or (ldist > 1e-4)) and (medium or (spec.dot(wi) > 1e-4)):
        sorigin = x0
        if not medium:
            sorigin = _offset_ray_origin(x0, n_side)
        count, prims, bary_a, bary_b, _straight, over = _pt_mnee_probe(
            refit, sorigin, wi,
            _shadow_tmax(sorigin, wi, ldist - (sorigin - x0).dot(wi)), f,
            layer_offset, nodes, miss, leaf, span, first_leaf, tri_pos,
            tri_colors, tri_uvs, tri_tex_meta, textures, tri_extra,
            circuit_meta, num_colored_triangles, shell, caster_offset)
        overflow = over
        if (over == 0) and (count > 0) and (count == k):
            cids = ti.Vector([-1] * MNEE_MAX_VERTICES)
            for q in ti.static(range(MNEE_MAX_VERTICES)):
                if q < count:
                    cids[q] = _pt_caster_id(shell, f, prims[q], caster_offset)
            solved, _capped, prims, bary_a, bary_b, ps, ngs, _etas, a_blks, \
                b_blks, c_blks, c_floor = _pt_mnee_solve(
                    refit, x0, n_side, y, fixed, count, prims, bary_a, bary_b,
                    cids, f, layer_offset, nodes, miss, leaf, span,
                    first_leaf, tri_pos, tri_norm, tri_mat, tri_extra,
                    tri_uvs, tri_tex_meta, textures, num_colored_triangles,
                    shell, caster_offset)
            g = 0.0
            ok = 0
            if solved == 1:
                g, ok = _pt_mnee_geometry(x0, count, ps, ngs, a_blks, b_blks,
                                          c_blks)
            if ok == 1:
                leg0 = ti.math.vec3(ps[0, 0], ps[0, 1], ps[0, 2]) - x0
                w0 = leg0 / ti.max(leg0.norm(), 1e-30)
                # The solved first leg is known to its constraint's noise
                # floor and its own rounding, each at most ``c_floor``
                # (``_pt_mnee_chain``).
                if (w0 - w_b).norm() < ti.max(MNEE_COVER_TOLERANCE,
                                              4.0 * c_floor):
                    covered = _pt_mnee_admissible(
                        refit, x0, sorigin, n_side, y, fixed, count, prims,
                        bary_a, bary_b, ps, ngs, f, layer_offset, nodes, miss,
                        leaf, span, first_leaf, tri_pos, tri_norm, tri_mat,
                        shell, caster_offset)
                    jac = g
                    if fixed == 0:
                        # The caustic block's emitter cosine, toward the
                        # chain's last vertex.
                        last = ti.math.vec3(ps[0, 0], ps[0, 1], ps[0, 2])
                        for qq in ti.static(range(1, MNEE_MAX_VERTICES)):
                            if qq < count:
                                last = ti.math.vec3(ps[qq, 0], ps[qq, 1],
                                                    ps[qq, 2])
                        cos_e = (last - y).normalized().dot(n_e)
                        if (one_sided == 1) and (cos_e <= 1e-6):
                            covered = 0
                        jac = g * ti.abs(cos_e)
    return covered, overflow, jac
