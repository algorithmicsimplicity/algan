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
# -1 per frame, and only ``gloss_scatter`` ever clears it.
GL_MAIN_CSUM = 0
GL_MAIN_W = 4
GL_MAIN_SIGMA = 7
GL_MAIN_WIDTH = 8

# Columns of a ``gl_pyr`` texel: reflected radiance, reflected glow, and the
# validity weight that normalises them. The glow lane is ``out``'s column 3,
# which is bloom coverage rather than alpha, and it has to be prefiltered
# alongside the color or a blurred reflection would carry a sharp bloom mask.
GL_PYR_WIDTH = 5


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
    the LEAST curved edge, so it never undercuts the footprint in any
    direction. That is exact on a sphere and leaves a surface that is flat
    one way -- a cylinder, a cone along its generators -- at the flat
    footprint. It is also what keeps the blur continuous: the estimate is
    constant over a triangle, and the largest curvature jumps between
    neighbours wherever the curvature varies -- a cone's ring nearer the apex
    is twice as curved as the ring below it -- which drew sawtooth stripes
    and seams across a satin cone. Normals are first turned to the viewer's
    side of the triangle, so a surface bulging towards the camera is convex
    whichever way its mesh is wound. A concave direction counts as flat: a
    concave mirror can magnify without bound near its focus, and the flat
    footprint is the widest blur this route ever drew, so the change only
    ever narrows it.

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
        k_lo = ti.min(k01, ti.min(k12, k20))
        if k_lo > 0.0:
            mag = 1.0 + 2.0 * d_p * k_lo
    return mag


@ti.func
def _sample_pyramid_level(gl_pyr: ti.template(), level_meta: ti.template(),
                          lvl, px, py, width, height):
    """Bilinear fetch of one pyramid level at level-0 pixel ``(px, py)``.

    Returns the raw (unnormalised) 5-vector; the caller divides the radiance by
    the validity weight. Sampling in level-0 coordinates and mapping into the
    level is what keeps the two levels of a trilinear fetch registered with
    each other and with the pixel being composited.
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
def _prefilter_at(gl_pyr: ti.template(), level_meta: ti.template(),
                  sigma, num_levels, px, py, width, height):
    """One prefiltered radiance fetch at Gaussian width ``sigma`` pixels.

    Trilinear: the two levels whose box widths bracket ``sigma``, each fetched
    bilinearly, lerped by the fractional level. Returns a vec4 (radiance +
    glow) already divided by the validity weight; a footprint that found no
    valid texel at all returns zero.
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
    out = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
    for k in ti.static(range(4)):
        a = 0.0
        if s0[4] > 1e-6:
            a = s0[k] / s0[4]
        b = 0.0
        if s1[4] > 1e-6:
            b = s1[k] / s1[4]
        out[k] = a * (1.0 - frac) + b * frac
    return out


@ti.kernel
def gloss_scatter(
        num_covered: int, gloss_base: int,
        frame_base: int, frame_rel: int, width: int,
        sigma_max: ti.f32,
        covered_idx: ti.types.ndarray(),
        pix_accum: ti.types.ndarray(),
        gl_main: ti.types.ndarray(), gl_pyr: ti.types.ndarray(),
        out: ti.types.ndarray(), row_base: int,
        cam_proj_table: ti.types.ndarray(), projected: ti.template(),
        frame: int):
    """Move one drained tile's glossy pixels into the frame's buffers.

    Must run BEFORE ``wf_composite_accum_sparse`` for the same tile: both read
    the frame buffer's raw prefilled background, and the composite overwrites
    it with the finalized pixel.

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

        # The pixel's own color, background folded in (wf_composite_accum
        # _sparse's expression, stopped before finalize_pixel_color).
        w_main = ti.math.vec4(pix_accum[r, 4], pix_accum[r, 5],
                              pix_accum[r, 6], 0.0)
        w_main[3] = ti.max(w_main[0], ti.max(w_main[1], w_main[2]))
        # ... and the reflection's, which retired through the same background.
        w_refl = ti.math.vec4(pix_accum[gr, 4], pix_accum[gr, 5],
                              pix_accum[gr, 6], 0.0)
        w_refl[3] = ti.max(w_refl[0], ti.max(w_refl[1], w_refl[2]))
        for ci in ti.static(range(4)):
            bg = ti.cast(out[frame_rel, p, ci], ti.f32)
            gl_main[p, GL_MAIN_CSUM + ci] = (
                pix_accum[r, ci] * 255.0 + w_main[ci] * bg)
            gl_pyr[p, ci] = pix_accum[gr, ci] * 255.0 + w_refl[ci] * bg
        gl_pyr[p, 4] = 1.0

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


@ti.kernel
def gloss_pyramid_level(src: int, dst: int, level_meta: ti.types.ndarray(),
                        gl_pyr: ti.types.ndarray()):
    """One 2x2 weighted box reduction of the reflection pyramid.

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
    off_d = level_meta[dst, 0]
    wd = level_meta[dst, 1]
    hd = level_meta[dst, 2]
    for i in range(wd * hd):
        y = i // wd
        x = i - y * wd
        acc = ti.Vector([0.0] * GL_PYR_WIDTH)
        for dy in ti.static(range(2)):
            for dx in ti.static(range(2)):
                sx = ti.min(2 * x + dx, ws - 1)
                sy = ti.min(2 * y + dy, hs - 1)
                s = off_s + sy * ws + sx
                for k in ti.static(range(GL_PYR_WIDTH)):
                    acc[k] += gl_pyr[s, k]
        d = off_d + i
        for k in ti.static(range(GL_PYR_WIDTH)):
            gl_pyr[d, k] = acc[k] * 0.25


@ti.kernel
def gloss_composite(
        num_pixels: int, width: int, height: int, num_levels: int,
        frame_rel: int,
        tonemapping: ti.template(), tonemap_exposure: ti.f32,
        level_meta: ti.types.ndarray(),
        gl_main: ti.types.ndarray(), gl_pyr: ti.types.ndarray(),
        out: ti.types.ndarray()):
    """``out = finalize(csum + W * prefilter(B))`` for every glossy pixel.

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
    """
    for p in range(num_pixels):
        sigma = gl_main[p, GL_MAIN_SIGMA]
        if sigma < 0.0:
            continue
        py = p // width
        px = p - py * width

        core = _prefilter_at(gl_pyr, level_meta, sigma * _LOBE_CORE_SIGMA,
                             num_levels, px, py, width, height)
        halo = _prefilter_at(gl_pyr, level_meta, sigma * _LOBE_HALO_SIGMA,
                             num_levels, px, py, width, height)
        refl = core * _LOBE_CORE_WEIGHT + halo * (1.0 - _LOBE_CORE_WEIGHT)

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
