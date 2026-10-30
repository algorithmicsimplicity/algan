"""The prefiltered-radiance half of the split-sum glossy route.

``DESIGN_glossy_prefilter.md`` is the design; the renderer audit's REPORT.md
§4.5/§4.5.1 is the measurement it answers. In one line: a rough reflector spawns
ONE deterministic ray in the mirror direction with throughput 1, its radiance
lands in a per-pixel reflection buffer instead of in the pixel, and after the
frame's rays drain that buffer is prefiltered by the lobe's screen footprint and
composited back with the analytic split-sum energy that was factored out of it.

Three passes live here, all per FRAME (never per batch -- a batch is many
frames and these buffers would otherwise be the render's dominant allocation):

``gloss_scatter``      compact tile rows -> the frame's ``gl_main``/``gl_pyr``
``gloss_pyramid_level``one 2x2 weighted box reduction, launched per level
``gloss_composite``    trilinear prefilter + ``finalize(csum + W * refl)``

``gloss_finish`` is the last two plus the buffers' reset for the next frame,
in ONE launch (``ALGAN_OPT_DISABLE=glossfuse`` restores the separate ones), and
``gloss_clear`` is the reset alone. Every kernel here computes a texel or a
pixel through the same ``ti.func``, so the fused launch does the same
arithmetic, in the same order, as the launches it replaces. It does it only
inside the frame's glossy bounding box (``gl_box``), on a pyramid that is zero
everywhere else -- which is what the separate launches compute there.

Nothing here runs unless ``glossy_reflection_mode() == 3``; the host does not
allocate the buffers otherwise.
"""

from algan.rendering.raytracing.raytrace_kernels_taichi import (
    CameraProjection,
    _projected,
    finalize_pixel_color,
)
from algan.taichi_compat import ti

# A box filter of width w has standard deviation w / sqrt(12) = 0.2887 w, so
# mip level L -- a box of 2^L level-0 pixels -- prefilters at sigma =
# 0.2887 * 2^L. Inverting that is how a pixel's blur radius picks its level.
_BOX_SIGMA = 0.28867513459481287
_LOG2_INV_BOX_SIGMA = 1.7924812503605778  # log2(1 / _BOX_SIGMA)

# A GGX lobe is NOT a Gaussian: it has a sharp core and a long tail, and a
# single blur at its median radius reproduces neither. Inverting the NDF's
# radial CDF gives the reflected deflection at quantile ``u`` as
# ``R50 * sqrt(u / (1 - u))``, so in units of the median radius the lobe puts
# its 10th/25th/75th/90th percentiles at 0.333 / 0.577 / 1.732 / 3.0 -- where a
# 2-D Gaussian matched at the median would put them at 0.645 / 0.759(ish) /
# 1.414 / 1.82. The tail is more than 1.6x too short and the core far too flat.
#
# Two Gaussians fix it for one extra pyramid fetch. Weights 0.6/0.4 at 0.6 and
# 1.8 times the median radius reproduce that quantile list to within 0.01 of
# probability at every one of those five points (0.0925 / 0.242 / 0.507 /
# 0.739 / 0.900 against 0.1 / 0.25 / 0.5 / 0.75 / 0.9). That is what makes the
# rendered reflection read as a bright core inside a wide halo -- the shape the
# path tracer draws -- instead of a flat disc of the right total energy.
_LOBE_CORE_SIGMA = 0.6
_LOBE_HALO_SIGMA = 1.8
_LOBE_CORE_WEIGHT = 0.6

# Column of a PLAIN accumulator row (``pix_accum[r]``) holding the pixel's
# geometric background residual: the fraction of the pixel area no sheet
# covered, deposited only by a primary that retired without bouncing (see
# sheet_resolve_taichi's retire). Columns 0..6 are the premultiplied color and
# the total leftover weight; this is the one column past them, which is why the
# plain row is eight wide. On the glossy layout the row is GL_ROW_WIDTH wide
# and column 7 of the plain row is still this -- ``GL_ROW_DIST`` below is
# column 7 of the *gloss* row, a different row of the same buffer.
ACC_GEO = 7

# Columns of a glossy accumulator row (``pix_accum[r + gloss_base]``). The
# drain owns 0..7 and the resolve 8..13, except that a rough reflector met in
# the drain adds to ``W`` and claims a pixel the resolve left unclaimed; see
# DESIGN_glossy_prefilter.md §4.2-§4.3.
GL_ROW_DIST = 7
GL_ROW_W = 8
GL_ROW_SIGMA_SCALE = 11
GL_ROW_DP = 12
GL_ROW_MAG = 13
GL_ROW_WIDTH = 14

# Columns of ``gl_main``: the pixel's linear pre-finalize color (with the
# background already folded in exactly as ``wf_composite_accum_sparse`` folds
# it), the factored-out reflection energy, and the blur radius. A NEGATIVE
# radius is the "not a glossy pixel" flag -- the host initialises the column to
# -1 per frame, and only ``gloss_scatter`` ever clears it. ``G`` is the radius
# as a fraction of the full lobe's (the cone factor, before the clamp) and
# ``SCALE`` the full lobe's radius, so a neighbour's fraction can be turned
# into the radius THIS pixel's lobe gives the same content.
GL_MAIN_CSUM = 0
GL_MAIN_W = 4
GL_MAIN_SIGMA = 7
GL_MAIN_G = 8
GL_MAIN_SCALE = 9
GL_MAIN_WIDTH = 10

# ``gl_box``: the bounding box of the pixels the frame's ``gloss_scatter``
# wrote, inclusive, as four i32s. ``gloss_finish`` works inside it alone and
# puts it back to empty: x0 and y0 at ``GL_BOX_EMPTY``, x1 and y1 at -1, so
# every extent ``x1 - x0 + 1`` is negative at every pyramid level.
GL_BOX_X0 = 0
GL_BOX_Y0 = 1
GL_BOX_X1 = 2
GL_BOX_Y1 = 3
GL_BOX_WIDTH = 4
GL_BOX_EMPTY = 1 << 30

# A ``gl_pyr`` texel holds two layers of the reflection, FAR then NEAR: each
# has reflected radiance, reflected glow, the validity weight that normalises
# them, and that weight times ``G``, and the near layer also that weight times
# ``log2 SCALE`` and its square. The glow lane is ``out``'s column 3, which is bloom coverage
# rather than alpha, and it has to be prefiltered alongside the color or a
# blurred reflection would carry a sharp bloom mask. A texel is split between
# the layers by ``_near_weight`` of its own ``G``; see ``gloss_composite`` for
# why there are two, and ``_near_same_surface`` for the near layer's scale.
GL_PYR_FAR = 0
GL_PYR_NEAR = 6
GL_PYR_WIDTH = 14
_PYR_V = 4
_PYR_GV = 5
_PYR_LV = 6
_PYR_LLV = 7

# The ramp, in ``G``, from a fully near reflection to a fully far one. Content
# a tenth of the way from the reflector to the eye (the sphere in the logo's
# satin cone, ``G`` about 0.07) is near; the environment (1) is far. Linear
# rather than a hard split, so a reflection whose distance varies smoothly --
# an object standing on a rough floor -- has no seam where it crosses over.
_NEAR_G_LO = 0.15
_NEAR_G_HI = 0.35

# The ramp, in ``|log2|`` of the ratio between the near content's full-lobe
# radius and this pixel's, from near content this pixel's surface could have
# written (it keeps its share) to content another surface wrote (it loses it).
# ``SCALE`` is ``2 * roughness^2`` times the frame's pixels per radian, so
# 0.5 and 1.5 are roughnesses within a factor of 1.19 and beyond 1.68.
_SAME_SURFACE_LO = 0.5
_SAME_SURFACE_HI = 1.5


@ti.func
def _edge_curvature(pa, pb, na, nb):
    """Normal curvature along the edge ``pa -> pb`` of a surface with unit
    normals ``na``, ``nb`` at its ends: ``1/R`` along any edge of a sphere of
    radius ``R`` with outward normals, 0 on a plane.
    """
    e = pb - pa
    ee = e.dot(e)
    k = 0.0
    if ee > 1e-20:
        k = (nb - na).dot(e) / ee
    return k


@ti.func
def _least_curvature(e01, e12, e20, k01, k12, k20):
    """The least normal curvature of a surface, in any direction, from the
    normal curvatures ``k01``, ``k12``, ``k20`` measured along a triangle's
    three edges.

    Normal curvature is a quadratic form in the direction (Euler), and three
    directions fix its three coefficients, so the form's smaller eigenvalue
    is the least curvature over every direction in the triangle's plane. The
    least of the three edges is only that when an edge happens to lie along
    the flattest direction. A cone is flat along its generators, but a
    triangle of its subdivided surface need not have an edge along one, and
    those triangles came out curved, magnified 1.44 against the 1 of their
    neighbours on the logo's cone: thin, sharper wedges across the satin cone
    that moved with the camera. The eigenvalue is never more than any edge's value, so
    the result is the least of the edges and the form's, and a sliver
    triangle, whose edges pin the form down poorly, keeps the edges alone.
    """
    k_lo = ti.min(k01, ti.min(k12, k20))
    w = e01.cross(-e20)
    w_len = w.norm()
    e_len = e01.norm()
    if w_len > 1e-20 and e_len > 1e-20:
        u = e01 / e_len
        v = (w / w_len).cross(u)
        m = ti.Matrix.zero(ti.f32, 3, 3)
        for i in ti.static(range(3)):
            e = e01
            if ti.static(i == 1):
                e = e12
            elif ti.static(i == 2):
                e = e20
            x = e.dot(u)
            y = e.dot(v)
            ll = ti.max(x * x + y * y, 1e-30)
            m[i, 0] = x * x / ll
            m[i, 1] = 2.0 * x * y / ll
            m[i, 2] = y * y / ll
        if ti.abs(m.determinant()) > 1e-3:
            abc = m.inverse() @ ti.math.vec3(k01, k12, k20)
            half_diff = 0.5 * (abc[0] - abc[2])
            lam = 0.5 * (abc[0] + abc[2]) - ti.sqrt(
                half_diff * half_diff + abc[1] * abc[1])
            k_lo = ti.min(k_lo, lam)
    return k_lo


@ti.func
def _reflector_magnification(f, prim, rd, tri_pos: ti.template(),
                             tri_norm: ti.template(), d_p):
    """How much faster than the view ray a curved mirror swings its reflection.

    The screen footprint in ``DESIGN_glossy_prefilter.md`` §3 is a FLAT
    mirror's: turning the view ray by ``delta`` turns the reflected ray by
    ``delta``, so a lobe of angular width ``sigma`` covers ``sigma / theta_px``
    pixels. A convex mirror also tilts its normal under the moving hit point,
    by ``kappa * d_p * delta`` at curvature ``kappa`` and camera distance
    ``d_p``, and the reflection swings twice that on top -- so the same lobe
    covers ``1 + 2 * kappa * d_p`` times fewer pixels. On the logo's chrome
    sphere that factor is about 47, and leaving it out blurred a roughness-0.04
    mirror into a grey haze whose coarse pyramid texels showed along its rim,
    while the same sphere seen in the floor (a deeper bounce, never
    prefiltered) stayed sharp.

    The curvature comes from the hit triangle's own vertex normals: along each
    edge ``(n_j - n_i) . e / |e|^2`` is the normal curvature in that
    direction. The footprint is an ellipse, narrowed most along the most
    curved direction; the one isotropic blur that stands for it is sized by
    the LEAST curved direction (``_least_curvature``), so it never undercuts
    the footprint in any direction. That is exact on a sphere and leaves a
    surface that is flat one way -- a cylinder, a cone along its generators
    -- at the flat footprint. It is also what keeps the blur continuous: the
    estimate is constant over a triangle, and the largest curvature jumps
    between neighbours wherever the curvature varies -- a cone's ring nearer
    the apex is twice as curved as the ring below it -- which drew sawtooth
    stripes and seams across a satin cone. Normals are first turned to the
    viewer's side of the triangle, so a surface bulging towards the camera is
    convex whichever way its mesh is wound. A concave direction counts as
    flat: a concave mirror can magnify without bound near its focus, and the
    flat footprint is the widest blur this route ever drew, so the change
    only ever narrows it.

    A flat triangle -- every polyhedron face, every Bezier circuit, which
    never comes here -- returns exactly 1.
    """
    tp = f % tri_pos.shape[0]
    tn = f % tri_norm.shape[0]
    p0 = ti.math.vec3(tri_pos[tp, prim, 0], tri_pos[tp, prim, 1],
                      tri_pos[tp, prim, 2])
    p1 = ti.math.vec3(tri_pos[tp, prim, 3], tri_pos[tp, prim, 4],
                      tri_pos[tp, prim, 5])
    p2 = ti.math.vec3(tri_pos[tp, prim, 6], tri_pos[tp, prim, 7],
                      tri_pos[tp, prim, 8])
    n0 = ti.math.vec3(tri_norm[tn, prim, 0], tri_norm[tn, prim, 1],
                      tri_norm[tn, prim, 2])
    n1 = ti.math.vec3(tri_norm[tn, prim, 3], tri_norm[tn, prim, 4],
                      tri_norm[tn, prim, 5])
    n2 = ti.math.vec3(tri_norm[tn, prim, 6], tri_norm[tn, prim, 7],
                      tri_norm[tn, prim, 8])
    mag = 1.0
    if n0.norm() > 1e-12 and n1.norm() > 1e-12 and n2.norm() > 1e-12:
        facing = (p1 - p0).cross(p2 - p0)
        if facing.dot(rd) > 0.0:
            facing = -facing
        n0 = n0.normalized()
        n1 = n1.normalized()
        n2 = n2.normalized()
        if n0.dot(facing) < 0.0:
            n0 = -n0
        if n1.dot(facing) < 0.0:
            n1 = -n1
        if n2.dot(facing) < 0.0:
            n2 = -n2
        k01 = _edge_curvature(p0, p1, n0, n1)
        k12 = _edge_curvature(p1, p2, n1, n2)
        k20 = _edge_curvature(p2, p0, n2, n0)
        k_lo = _least_curvature(p1 - p0, p2 - p1, p0 - p2, k01, k12, k20)
        if k_lo > 0.0:
            mag = 1.0 + 2.0 * d_p * k_lo
    return mag


@ti.func
def _sample_pyramid_level(gl_pyr: ti.template(), level_meta: ti.template(),
                          lvl, px, py, width, height):
    """Bilinear fetch of one pyramid level at level-0 pixel ``(px, py)``.

    Returns the raw (unnormalised) texel; the caller divides each layer's
    radiance by that layer's validity weight. Sampling in level-0 coordinates
    and mapping into the level is what keeps the two levels of a trilinear
    fetch registered with each other and with the pixel being composited.
    """
    off = level_meta[lvl, 0]
    lw = level_meta[lvl, 1]
    lh = level_meta[lvl, 2]
    fx = (ti.cast(px, ti.f32) + 0.5) * (
        ti.cast(lw, ti.f32) / ti.cast(width, ti.f32)) - 0.5
    fy = (ti.cast(py, ti.f32) + 0.5) * (
        ti.cast(lh, ti.f32) / ti.cast(height, ti.f32)) - 0.5
    x0f = ti.floor(fx)
    y0f = ti.floor(fy)
    tx = fx - x0f
    ty = fy - y0f
    # Both taps clamp from the UNCLAMPED index. Deriving the second from the
    # clamped first is the classic edge bug and it is not a fringe effect at
    # the coarse end of a pyramid, where "the edge" is half the level: a
    # sample left of texel centre 0 has an unclamped index of -1, so the two
    # taps must both be texel 0 and the weight must not matter -- clamping
    # 0 + 1 instead reads texel 1 and interpolates the wrong way across the
    # whole outer half of every level.
    x0i = ti.cast(x0f, ti.i32)
    y0i = ti.cast(y0f, ti.i32)
    x0 = ti.math.clamp(x0i, 0, lw - 1)
    y0 = ti.math.clamp(y0i, 0, lh - 1)
    x1 = ti.math.clamp(x0i + 1, 0, lw - 1)
    y1 = ti.math.clamp(y0i + 1, 0, lh - 1)
    i00 = off + y0 * lw + x0
    i10 = off + y0 * lw + x1
    i01 = off + y1 * lw + x0
    i11 = off + y1 * lw + x1
    w00 = (1.0 - tx) * (1.0 - ty)
    w10 = tx * (1.0 - ty)
    w01 = (1.0 - tx) * ty
    w11 = tx * ty
    out = ti.Vector([0.0] * GL_PYR_WIDTH)
    for k in ti.static(range(GL_PYR_WIDTH)):
        out[k] = (gl_pyr[i00, k] * w00 + gl_pyr[i10, k] * w10
                  + gl_pyr[i01, k] * w01 + gl_pyr[i11, k] * w11)
    return out


@ti.func
def _near_weight(g):
    """How much of a reflection blurred at fraction ``g`` of its lobe is
    near.
    """
    return ti.math.clamp((_NEAR_G_HI - g) / (_NEAR_G_HI - _NEAR_G_LO),
                         0.0, 1.0)


@ti.func
def _pyramid_taps(gl_pyr: ti.template(), level_meta: ti.template(),
                  sigma, num_levels, px, py, width, height):
    """The raw texels of one trilinear fetch at Gaussian width ``sigma`` px.

    The two levels whose box widths bracket ``sigma``, each fetched
    bilinearly, and the fractional level between them. Normalising is left to
    the caller (``_layer_at``), because each layer has its own validity.
    """
    lvl = 0.0
    if sigma > _BOX_SIGMA:
        lvl = ti.log(sigma) * 1.4426950408889634 + _LOG2_INV_BOX_SIGMA
    lvl = ti.math.clamp(lvl, 0.0, ti.cast(num_levels - 1, ti.f32))
    l0 = ti.cast(lvl, ti.i32)
    l1 = ti.min(l0 + 1, num_levels - 1)
    frac = lvl - ti.cast(l0, ti.f32)

    s0 = _sample_pyramid_level(gl_pyr, level_meta, l0, px, py, width, height)
    s1 = s0
    if l1 != l0:
        s1 = _sample_pyramid_level(gl_pyr, level_meta, l1, px, py,
                                   width, height)
    return s0, s1, frac


@ti.func
def _layer_at(s0, s1, frac, base: ti.template()):
    """One layer of a trilinear fetch: its radiance + glow, each level divided
    by its own validity weight and then lerped, and the lerped weight.

    A footprint that found no valid texel of the layer returns zero radiance
    and zero weight.
    """
    out = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
    for k in ti.static(range(4)):
        a = 0.0
        if s0[base + _PYR_V] > 1e-6:
            a = s0[base + k] / s0[base + _PYR_V]
        b = 0.0
        if s1[base + _PYR_V] > 1e-6:
            b = s1[base + k] / s1[base + _PYR_V]
        out[k] = a * (1.0 - frac) + b * frac
    v = s0[base + _PYR_V] * (1.0 - frac) + s1[base + _PYR_V] * frac
    return out, v


@ti.func
def _layer_mean_g(s0, s1, frac, base: ti.template(), fallback):
    """The validity-weighted mean ``G`` of one layer over a fetch's footprint,
    or ``fallback`` where the footprint holds none of it.
    """
    v = s0[base + _PYR_V] * (1.0 - frac) + s1[base + _PYR_V] * frac
    gv = s0[base + _PYR_GV] * (1.0 - frac) + s1[base + _PYR_GV] * frac
    g = fallback
    if v > 1e-6:
        g = gv / v
    return g


@ti.func
def _near_same_surface(s0, s1, frac, scale):
    """The share of a fetch's near content this pixel's own surface could have
    written, judged by the full-lobe radius it was written at.

    The pyramid is screen space and does not know where one surface ends. A
    rough surface in front of a glossy one writes, along its silhouette, what
    its own lobe meets just behind it: the glossy surface, up close, near. The
    glossy surface's pixels beside that silhouette then found it in their
    near footprint and took it as part of their own reflection, which drew a
    bright outline round the wave in the logo's chrome A. A surface's near
    reflection comes in at its own roughness, so near content written at a
    very different ``SCALE`` is another surface's.

    The footprint can hold both, so a mean scale is not enough: half this
    surface's own near content and half another's averages to a scale
    neither wrote at. Read as two populations, one at this pixel's
    ``log2 SCALE`` and one ``d`` away from it, the first two moments of the
    offset ``e`` fix both: the other's share is ``E[e]^2 / E[e^2]`` and
    ``d`` is ``E[e^2] / E[e]``. Its share counts against this pixel by how
    far ``d`` is past the ``_SAME_SURFACE`` ramp.
    """
    v = (s0[GL_PYR_NEAR + _PYR_V] * (1.0 - frac)
         + s1[GL_PYR_NEAR + _PYR_V] * frac)
    lv = (s0[GL_PYR_NEAR + _PYR_LV] * (1.0 - frac)
          + s1[GL_PYR_NEAR + _PYR_LV] * frac)
    llv = (s0[GL_PYR_NEAR + _PYR_LLV] * (1.0 - frac)
           + s1[GL_PYR_NEAR + _PYR_LLV] * frac)
    same = 1.0
    if v > 1e-6 and scale > 0.0:
        own = ti.log(scale) * 1.4426950408889634
        m1 = lv / v
        e1 = m1 - own
        # E[e^2] about this pixel's own scale. The floor keeps float
        # cancellation in a footprint of one scale from reading as a second
        # population.
        e2 = llv / v - 2.0 * own * m1 + own * own
        if e2 > 1e-4:
            other = ti.min(e1 * e1 / e2, 1.0)
            offset = e2 / ti.max(ti.abs(e1), 1e-6)
            past = ti.math.clamp(
                (offset - _SAME_SURFACE_LO)
                / (_SAME_SURFACE_HI - _SAME_SURFACE_LO), 0.0, 1.0)
            same = 1.0 - other * past
    return same


@ti.func
def _near_share(s0, s1, frac):
    """The near layer's share of the valid texels in a fetch's footprint."""
    a = 0.0
    t0 = s0[GL_PYR_FAR + _PYR_V] + s0[GL_PYR_NEAR + _PYR_V]
    if t0 > 1e-6:
        a = s0[GL_PYR_NEAR + _PYR_V] / t0
    b = 0.0
    t1 = s1[GL_PYR_FAR + _PYR_V] + s1[GL_PYR_NEAR + _PYR_V]
    if t1 > 1e-6:
        b = s1[GL_PYR_NEAR + _PYR_V] / t1
    return a * (1.0 - frac) + b * frac


@ti.kernel
def gloss_scatter(
        num_covered: int, gloss_base: int,
        frame_base: int, frame_rel: int, width: int,
        sigma_max: ti.f32,
        covered_idx: ti.types.ndarray(),
        pix_accum: ti.types.ndarray(),
        gl_main: ti.types.ndarray(), gl_pyr: ti.types.ndarray(),
        gl_box: ti.types.ndarray(),
        out: ti.types.ndarray(), row_base: int,
        cam_proj_table: ti.types.ndarray(), projected: ti.template(),
        frame: int):
    """Move one drained tile's glossy pixels into the frame's buffers.

    Must run BEFORE ``wf_composite_accum_sparse`` for the same tile: both read
    the frame buffer's raw prefilled background, and the composite overwrites
    it with the finalized pixel.

    Every pixel written here grows the frame's ``gl_box``, which is all of the
    frame ``gloss_finish`` then touches. Each bound is read before its atomic
    and the atomic skipped when the pixel is already inside: the bounds only
    ever move outwards, so a stale read can cost a redundant atomic but never
    a missed one, and the frame's pixels do not all queue on four words.

    ``row_base`` is where this call's pixels start in the tile's accumulator
    rows: a tile may span several frames while the buffers written here hold
    one, so the host scatters a tile one frame-part at a time, handing over
    that part's slice of ``covered_idx`` and its row offset into the
    tile-wide ``pix_accum`` (whose reflection rows sit ``gloss_base`` further
    on, as before). A tile lying within one frame passes 0.

    ``csum`` here is deliberately the composite's own arithmetic, repeated
    rather than shared, because ``wf_composite_accum_sparse`` finalizes in the
    same expression and this pass must stop one step short of that -- the
    glossy pixel's final value is not known until the frame's prefilter has
    run, and a tonemap is not something you can add to afterwards.

    ``cam_proj_table[frame]`` is the frame's projection, read only when
    ``projected`` (:class:`CameraProjection`; ``algan.rendering
    .projection``). Off the pinhole a pixel at camera path length ``s`` is
    ``B + C * s`` world units across, ``C`` zero for an orthographic camera,
    so the resolve stores the lobe's scale per unit of ``B`` instead of per
    unit of ``C`` (``sheet_resolve_taichi``), and the cone factor becomes
    ``(d_tot - d_p) / (1 + d_tot * C / B)``: a blur that grows with the
    distance past the reflector and, with no perspective, never shrinks
    with distance from the camera.
    """
    cam_proj = ti.static(CameraProjection(cam_proj_table, projected))
    for ri in range(num_covered):
        r = ri + row_base
        gr = r + gloss_base
        sigma_scale = pix_accum[gr, GL_ROW_SIGMA_SCALE]
        if sigma_scale <= 0.0:
            # No prefiltered glossy branch at this pixel: it is an ordinary
            # covered pixel and the tile composite's value for it is final.
            continue
        p = covered_idx[ri] - frame_base
        py = p // width
        px = p - py * width
        if px < gl_box[GL_BOX_X0]:
            ti.atomic_min(gl_box[GL_BOX_X0], px)
        if py < gl_box[GL_BOX_Y0]:
            ti.atomic_min(gl_box[GL_BOX_Y0], py)
        if px > gl_box[GL_BOX_X1]:
            ti.atomic_max(gl_box[GL_BOX_X1], px)
        if py > gl_box[GL_BOX_Y1]:
            ti.atomic_max(gl_box[GL_BOX_Y1], py)

        # The pixel's own color, background folded in (wf_composite_accum
        # _sparse's expression, stopped before finalize_pixel_color).
        w_main = ti.math.vec4(pix_accum[r, 4], pix_accum[r, 5],
                              pix_accum[r, 6], 0.0)
        w_main[3] = ti.max(w_main[0], ti.max(w_main[1], w_main[2]))
        # ... and the reflection's, which retired through the same background.
        w_refl = ti.math.vec4(pix_accum[gr, 4], pix_accum[gr, 5],
                              pix_accum[gr, 6], 0.0)
        w_refl[3] = ti.max(w_refl[0], ti.max(w_refl[1], w_refl[2]))
        refl = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
        for ci in ti.static(range(4)):
            bg = ti.cast(out[frame_rel, p, ci], ti.f32)
            gl_main[p, GL_MAIN_CSUM + ci] = (
                pix_accum[r, ci] * 255.0 + w_main[ci] * bg)
            refl[ci] = pix_accum[gr, ci] * 255.0 + w_refl[ci] * bg

        for k in ti.static(range(3)):
            gl_main[p, GL_MAIN_W + k] = pix_accum[gr, GL_ROW_W + k]

        # Contact hardening (DESIGN_glossy_prefilter.md §3): the lobe's screen
        # footprint scales with how far past the reflector the reflected
        # content sits. The drain sums each surface the glossy ray met as its
        # coverage over its total camera path length, so d_tot is their
        # coverage-weighted harmonic distance and ``1 - d_p / d_tot`` the
        # coverage-weighted mean cone factor, a miss counting as 1. A ray that
        # found nothing left the sum at zero, which is k = 1 -- a reflection
        # of the sky, fully blurred.
        d_inv = pix_accum[gr, GL_ROW_DIST]
        d_tot = 1e30
        if d_inv > 1e-30:
            d_tot = 1.0 / d_inv
        d_p = pix_accum[gr, GL_ROW_DP]
        k_cone = 1.0
        if d_tot < 1e29:
            k_cone = ti.math.clamp(1.0 - d_p / ti.max(d_tot, 1e-6), 0.0, 1.0)
        # A curved reflector (``_reflector_magnification``) sees its reflected
        # content ``d_r`` away through a reflection that swings ``mag`` times
        # faster than the view, so the lobe's view angle is
        # ``sigma * d_r / (d_p + mag * d_r)``: ``k / (1 + (mag - 1) k)`` in
        # terms of the flat cone factor. A flat mirror's ``mag`` is exactly 1,
        # and the expression is then exactly ``k``.
        mag = ti.max(pix_accum[gr, GL_ROW_MAG], 1.0)
        k_cone = k_cone / (1.0 + (mag - 1.0) * k_cone)
        g = k_cone
        gl_main[p, GL_MAIN_SIGMA] = ti.min(k_cone * sigma_scale,
                                           sigma_max)
        if ti.static(_projected(cam_proj)):
            spread = cam_proj[frame, 1]
            if spread != 0.0:
                sd = ti.math.vec3(cam_proj[frame, 3], cam_proj[frame, 4],
                                  cam_proj[frame, 5])
                inv_virtual = cam_proj[frame, 0] / (
                    spread * ti.max(sd.norm(), 1e-30))
                sigma = sigma_max
                if d_tot < 1e29:
                    sigma = (ti.max(d_tot - d_p, 0.0) * sigma_scale
                             / (1.0 + inv_virtual * d_tot))
                elif inv_virtual > 0.0:
                    sigma = sigma_scale / inv_virtual
                gl_main[p, GL_MAIN_SIGMA] = ti.min(sigma, sigma_max)
                g = sigma / sigma_scale
        gl_main[p, GL_MAIN_G] = g
        gl_main[p, GL_MAIN_SCALE] = sigma_scale

        # Level 0 of both layers. A texel wholly in one layer writes exact
        # zeros to the other, so a frame with no near reflection anywhere
        # leaves the near lanes zero through every level of the pyramid.
        w_near = _near_weight(g)
        w_far = 1.0 - w_near
        for ci in ti.static(range(4)):
            gl_pyr[p, GL_PYR_FAR + ci] = refl[ci] * w_far
            gl_pyr[p, GL_PYR_NEAR + ci] = refl[ci] * w_near
        gl_pyr[p, GL_PYR_FAR + _PYR_V] = w_far
        gl_pyr[p, GL_PYR_FAR + _PYR_GV] = w_far * g
        gl_pyr[p, GL_PYR_NEAR + _PYR_V] = w_near
        gl_pyr[p, GL_PYR_NEAR + _PYR_GV] = w_near * g
        log_scale = ti.log(sigma_scale) * 1.4426950408889634
        gl_pyr[p, GL_PYR_NEAR + _PYR_LV] = w_near * log_scale
        gl_pyr[p, GL_PYR_NEAR + _PYR_LLV] = w_near * log_scale * log_scale


@ti.func
def _gloss_reduce_texel(src, dst, x, y, level_meta: ti.template(),
                        gl_pyr: ti.template()):
    """Texel ``(x, y)`` of level ``dst``: the 2x2 weighted box of level ``src``.

    Weighted, not plain: a texel's validity lane rides through the reduction
    with its radiance, so a level averages only the pixels that actually have a
    reflection and a wide lobe beside a reflector's silhouette does not pull in
    the zeros of the background.

    The clamp handles an odd source dimension by repeating its last row or
    column, which is only sound because the level table halves by CEILING --
    see ``tracer._gloss_pyramid_levels``, where the comment records what a
    floor-halved chain silently did to the top of the pyramid.
    """
    off_s = level_meta[src, 0]
    ws = level_meta[src, 1]
    hs = level_meta[src, 2]
    acc = ti.Vector([0.0] * GL_PYR_WIDTH)
    for dy in ti.static(range(2)):
        for dx in ti.static(range(2)):
            sx = ti.min(2 * x + dx, ws - 1)
            sy = ti.min(2 * y + dy, hs - 1)
            s = off_s + sy * ws + sx
            for k in ti.static(range(GL_PYR_WIDTH)):
                acc[k] += gl_pyr[s, k]
    d = level_meta[dst, 0] + y * level_meta[dst, 1] + x
    for k in ti.static(range(GL_PYR_WIDTH)):
        gl_pyr[d, k] = acc[k] * 0.25


@ti.kernel
def gloss_pyramid_level(src: int, dst: int, level_meta: ti.types.ndarray(),
                        gl_pyr: ti.types.ndarray()):
    """One 2x2 weighted box reduction of the reflection pyramid
    (``_gloss_reduce_texel``).
    """
    wd = level_meta[dst, 1]
    for i in range(wd * level_meta[dst, 2]):
        y = i // wd
        _gloss_reduce_texel(src, dst, i - y * wd, y, level_meta, gl_pyr)


@ti.kernel
def gloss_composite(
        num_pixels: int, width: int, height: int, num_levels: int,
        frame_rel: int,
        tonemapping: ti.template(), tonemap_exposure: ti.f32,
        level_meta: ti.types.ndarray(),
        gl_main: ti.types.ndarray(), gl_pyr: ti.types.ndarray(),
        out: ti.types.ndarray()):
    """``out = finalize(csum + W * prefilter(B))`` for every glossy pixel
    (``_gloss_composite_pixel``).
    """
    sigma_max = _BOX_SIGMA * ti.cast(1 << (num_levels - 1), ti.f32)
    for p in range(num_pixels):
        sigma = gl_main[p, GL_MAIN_SIGMA]
        if sigma < 0.0:
            continue
        _gloss_composite_pixel(p, sigma, sigma_max, width, height, num_levels,
                               frame_rel, tonemapping, tonemap_exposure,
                               level_meta, gl_main, gl_pyr, out)


@ti.func
def _box_extent(lo, hi, lvl):
    """How many texels of pyramid level ``lvl`` the level-0 range
    ``[lo, hi]`` reaches through the reduction, 0 for an empty range.

    Texel ``x`` of a level is reduced from texels ``2x`` and ``2x + 1`` of
    the one below (the clamp at an odd edge repeats ``2x``), so level-0
    column ``c`` reaches exactly column ``c >> lvl``.
    """
    n = 0
    if hi >= lo:
        n = (hi >> lvl) - (lo >> lvl) + 1
    return n


@ti.func
def _box_xy(i, lo_x, lo_y, box_w):
    """Texel ``i`` of a ``box_w`` wide box at ``(lo_x, lo_y)``, counted
    row-major, as its level's ``(x, y)``.
    """
    j = i // box_w
    return lo_x + i - j * box_w, lo_y + j


@ti.kernel
def gloss_finish(
        levels: ti.template(),
        width: int, height: int, num_levels: int,
        frame_rel: int,
        tonemapping: ti.template(), tonemap_exposure: ti.f32,
        level_meta: ti.types.ndarray(),
        gl_main: ti.types.ndarray(), gl_pyr: ti.types.ndarray(),
        gl_box: ti.types.ndarray(),
        out: ti.types.ndarray()):
    """A finished frame's whole prefilter in one launch, and the reset.

    ``levels`` is ``num_levels`` again, as a template: each unrolled level is
    its own top-level loop, so its own offloaded task, and tasks run in order
    -- a level reads the one below it complete, as it did when each level was
    a launch of ``gloss_pyramid_level``. Then ``gloss_composite``'s loop, then
    the reset ``gloss_clear`` would have done, cut to what the next frame
    reads.

    All of it inside ``gl_box``, the frame's glossy pixels' bounding box
    (``gloss_scatter``): a frame with no glossy pixel does nothing, and one
    with a small reflector does a small reflector's worth. Each level reduces
    only the texels the box reaches (``_box_extent``), which is exact because
    the pyramid is all zero when a frame starts but for the level-0 texels
    the frame's scatter wrote: every texel outside the box's reach is then a
    reduction of zeros, which is the zero already there, so the composite's
    taps read what the full-frame launches would have given them. The
    composite runs over the pixels from the box's first to its last in
    row-major order rather than over the box itself, so that the pixel index
    stays the loop's own, as in ``gloss_composite``: under ``fast_math`` the
    composite's arithmetic is fused differently when the index is computed
    from a box position, and measured that way it moved by an ulp or two on
    a few pixels a frame. The reset keeps the pyramid zero:

    * ``gl_main``'s blur radius back to -1, written by the pixel that just
      read it. ``gloss_scatter`` writes every column of the rows it marks
      glossy and nothing reads an unmarked row past its radius, so the other
      columns need no reset.
    * Every level of ``gl_pyr`` back to zero inside the box's reach, after
      the composite has read it; nothing outside it was written.
      ``gloss_clear`` zeroes the whole pyramid before a chunk's first frame.
      A texel to an iteration, all its lanes: a lane to an iteration put the
      division by the box's width on every lane, and on the CPU that made
      the reset of a frame glossy all over twenty times the full-frame
      launches' reset of level 0 alone.
    * ``gl_box`` back to empty.

    ``num_levels`` stays a runtime argument because the composite's
    arithmetic takes it as one in ``gloss_composite``.
    """
    x0 = gl_box[GL_BOX_X0]
    y0 = gl_box[GL_BOX_Y0]
    x1 = gl_box[GL_BOX_X1]
    y1 = gl_box[GL_BOX_Y1]
    for lvl in ti.static(range(1, levels)):
        fw = _box_extent(x0, x1, lvl)
        for i in range(fw * _box_extent(y0, y1, lvl)):
            x, y = _box_xy(i, x0 >> lvl, y0 >> lvl, fw)
            _gloss_reduce_texel(lvl - 1, lvl, x, y, level_meta, gl_pyr)
    sigma_max = _BOX_SIGMA * ti.cast(1 << (num_levels - 1), ti.f32)
    # The box's first and last pixel (an empty box's sentinels would
    # overflow the product).
    p_lo = 0
    p_hi = 0
    if x1 >= x0:
        p_lo = y0 * width + x0
        p_hi = y1 * width + x1 + 1
    for p in range(p_lo, p_hi):
        sigma = gl_main[p, GL_MAIN_SIGMA]
        if sigma < 0.0:
            continue
        _gloss_composite_pixel(p, sigma, sigma_max, width, height, num_levels,
                               frame_rel, tonemapping, tonemap_exposure,
                               level_meta, gl_main, gl_pyr, out)
        gl_main[p, GL_MAIN_SIGMA] = -1.0
    for lvl in ti.static(range(levels)):
        rw = _box_extent(x0, x1, lvl)
        for i in range(rw * _box_extent(y0, y1, lvl)):
            x, y = _box_xy(i, x0 >> lvl, y0 >> lvl, rw)
            t = level_meta[lvl, 0] + y * level_meta[lvl, 1] + x
            for k in ti.static(range(GL_PYR_WIDTH)):
                gl_pyr[t, k] = 0.0
    gl_box[GL_BOX_X0] = GL_BOX_EMPTY
    gl_box[GL_BOX_Y0] = GL_BOX_EMPTY
    gl_box[GL_BOX_X1] = -1
    gl_box[GL_BOX_Y1] = -1


@ti.kernel
def gloss_clear(num_pixels: int, gl_main: ti.types.ndarray(),
                gl_pyr: ti.types.ndarray(), gl_box: ti.types.ndarray()):
    """Reset the per-frame reflection buffers (``tracer._gloss_clear``).

    Every ``gl_main`` element, flattened so each store is coalesced: zero,
    and -1 in the blur radius column. A strided write of that one column
    rewrote only part of every DRAM sector it touched. Every level of
    ``gl_pyr``, and ``gl_box`` empty: ``gloss_finish``'s docstring has why
    the pyramid has to start a frame all zero, and how it stays that way.
    """
    for i in range(num_pixels * GL_MAIN_WIDTH):
        c = i % GL_MAIN_WIDTH
        v = 0.0
        if c == GL_MAIN_SIGMA:
            v = -1.0
        gl_main[i // GL_MAIN_WIDTH, c] = v
    for i in range(gl_pyr.shape[0] * GL_PYR_WIDTH):
        gl_pyr[i // GL_PYR_WIDTH, i % GL_PYR_WIDTH] = 0.0
    gl_box[GL_BOX_X0] = GL_BOX_EMPTY
    gl_box[GL_BOX_Y0] = GL_BOX_EMPTY
    gl_box[GL_BOX_X1] = -1
    gl_box[GL_BOX_Y1] = -1


@ti.func
def _gloss_composite_pixel(p, sigma, sigma_max, width, height, num_levels,
                           frame_rel, tonemapping: ti.template(),
                           tonemap_exposure, level_meta: ti.template(),
                           gl_main: ti.template(), gl_pyr: ti.template(),
                           out: ti.template()):
    """``out = finalize(csum + W * prefilter(B))`` for glossy pixel ``p``.

    The tile composite already wrote a finalized value at these pixels (its
    own, without the reflection); this overwrites it. Doing it that way rather
    than teaching ``wf_composite_accum_sparse`` to skip them keeps that kernel,
    and the ordering rules around it, exactly as they were -- the cost is one
    redundant finalize per glossy pixel, which is a handful of arithmetic on a
    small subset of the frame.

    The prefilter is two trilinear fetches of the mip pyramid, a narrow core
    and a wide halo, mixed to the GGX radial profile (see ``_LOBE_CORE_SIGMA``).
    A pyramid rather than a direct blur because the radii involved are enormous:
    at roughness 0.35 the lobe's median deflection is ~300 px on a PREVIEW
    frame, which no separable blur can afford and no strided approximation of
    one can sample without aliasing.

    Each pixel's radius comes from its own glossy ray, so a single blur drew
    a hard edge round every near reflection: the pixels whose ray met the
    nearby object were blurred a few pixels, the ones beside them whose ray
    went past it a hundred, and the object's reflection stopped dead where
    the radius jumped. In a rough reflector the lobe of a pixel beside it
    still sees part of the object, at the object's own small blur, in front
    of the distant content at the large one. So the reflection is prefiltered
    in two layers (``_near_weight``), each at its own radius:

        refl = a * near(sigma_near) + (1 - a) * far(sigma_far)

    with ``a`` the near layer's share of the footprint at ``sigma_near``,
    which is how much of the lobe the near content covers. A pixel knows its
    own layer's radius; the other layer's is the mean ``G`` of that layer
    within reach, at this pixel's own lobe scale. Far content excludes the
    near texels, as the near object occludes it in the reflection, and
    normalising by the far layer's own weight fills the hole behind it from
    around it. Near content another surface wrote at a different roughness
    takes no share (``_near_same_surface``). With no near texel within reach
    the near term is zero and the far layer is all there is, which is the
    single blur exactly: a frame with no near reflection anywhere renders as
    it did before.
    """
    py = p // width
    px = p - py * width

    g = gl_main[p, GL_MAIN_G]
    scale = gl_main[p, GL_MAIN_SCALE]
    w_near = _near_weight(g)
    # The widest halo a near reflection can have at this pixel's scale:
    # near content farther away than this does not reach its lobe.
    reach = ti.min(_NEAR_G_HI * _LOBE_HALO_SIGMA * scale, sigma_max)
    r0, r1, rf = _pyramid_taps(gl_pyr, level_meta, reach, num_levels,
                               px, py, width, height)
    near_in_reach = (r0[GL_PYR_NEAR + _PYR_V] * (1.0 - rf)
                     + r1[GL_PYR_NEAR + _PYR_V] * rf)
    refl = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
    if w_near <= 0.0 and near_in_reach <= 0.0:
        c0, c1, cf = _pyramid_taps(gl_pyr, level_meta,
                                   sigma * _LOBE_CORE_SIGMA, num_levels,
                                   px, py, width, height)
        core, _vc = _layer_at(c0, c1, cf, GL_PYR_FAR)
        h0, h1, hf = _pyramid_taps(gl_pyr, level_meta,
                                   sigma * _LOBE_HALO_SIGMA, num_levels,
                                   px, py, width, height)
        halo, _vh = _layer_at(h0, h1, hf, GL_PYR_FAR)
        refl = core * _LOBE_CORE_WEIGHT + halo * (1.0 - _LOBE_CORE_WEIGHT)
    else:
        g_far = _layer_mean_g(r0, r1, rf, GL_PYR_FAR, 1.0)
        g_near = _layer_mean_g(r0, r1, rf, GL_PYR_NEAR, g)
        sigma_far = sigma
        if w_near > 0.0:
            sigma_far = ti.min(
                ((1.0 - w_near) * g + w_near * g_far) * scale, sigma_max)
            # A near pixel's own ``G`` is not its near layer's radius
            # where the pixel is only partly near: on the outline of a
            # near reflection the glossy ray's coverage-weighted cone
            # factor mixes the object's with the sky's, and blurring the
            # near layer at that mix drew the outline as a bright thread.
            # The near layer's mean ``G`` over the pixel's own near
            # footprint is the object's alone, and is the pixel's own
            # wherever the near reflection is uniform.
            o0, o1, of = _pyramid_taps(
                gl_pyr, level_meta,
                ti.min(g * _LOBE_HALO_SIGMA * scale, sigma_max),
                num_levels, px, py, width, height)
            g_own = _layer_mean_g(o0, o1, of, GL_PYR_NEAR, g)
            g_near = w_near * g_own + (1.0 - w_near) * g_near
        sigma_near = ti.min(g_near * scale, sigma_max)
        for lobe in ti.static(range(2)):
            lobe_sigma = ti.static(
                _LOBE_CORE_SIGMA if lobe == 0 else _LOBE_HALO_SIGMA)
            lobe_weight = ti.static(
                _LOBE_CORE_WEIGHT if lobe == 0
                else 1.0 - _LOBE_CORE_WEIGHT)
            f0, f1, ff = _pyramid_taps(gl_pyr, level_meta,
                                       sigma_far * lobe_sigma, num_levels,
                                       px, py, width, height)
            far, v_far = _layer_at(f0, f1, ff, GL_PYR_FAR)
            n0, n1, nf = _pyramid_taps(gl_pyr, level_meta,
                                       sigma_near * lobe_sigma,
                                       num_levels, px, py, width, height)
            near, v_near = _layer_at(n0, n1, nf, GL_PYR_NEAR)
            # A near pixel's own near content is in its footprint, so
            # the footprint's mean scale says little about whose the rest
            # is, and only a far pixel is held to it.
            same = _near_same_surface(n0, n1, nf, scale)
            a = _near_share(n0, n1, nf) * (
                w_near + (1.0 - w_near) * same)
            # A layer with nothing in its footprint stands in for neither
            # colour: the other one fills it.
            if v_far <= 1e-6:
                far = near
            if v_near <= 1e-6:
                near = far
            refl += (a * near + (1.0 - a) * far) * lobe_weight

    w_energy = ti.math.vec3(gl_main[p, GL_MAIN_W],
                            gl_main[p, GL_MAIN_W + 1],
                            gl_main[p, GL_MAIN_W + 2])
    csum = ti.math.vec4(gl_main[p, GL_MAIN_CSUM],
                        gl_main[p, GL_MAIN_CSUM + 1],
                        gl_main[p, GL_MAIN_CSUM + 2],
                        gl_main[p, GL_MAIN_CSUM + 3])
    for k in ti.static(range(3)):
        csum[k] += w_energy[k] * refl[k]
    # The glow lane carries the reflection's bloom coverage scaled by the
    # strongest channel of the energy, the same reduction the resolve uses
    # to give a color weight a scalar glow.
    csum[3] += ti.max(w_energy[0],
                      ti.max(w_energy[1], w_energy[2])) * refl[3]

    color_final = finalize_pixel_color(
        csum, 1.0, tonemapping, tonemap_exposure)
    for ci in ti.static(range(4)):
        if ti.static(tonemapping == 3):
            out[frame_rel, p, ci] = color_final[ci]
        else:
            out[frame_rel, p, ci] = ti.cast(color_final[ci], ti.u8)
