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

import taichi as ti

from algan.rendering.raytracing.raytrace_kernels_taichi import (
    finalize_pixel_color,
)

# A box filter of width w has standard deviation w / sqrt(12) = 0.2887 w, so
# mip level L -- a box of 2^L level-0 pixels -- prefilters at sigma =
# 0.2887 * 2^L. Inverting that is how a pixel's blur radius picks its level.
_BOX_SIGMA = 0.28867513459481287
_LOG2_INV_BOX_SIGMA = 1.7924812503605778  # log2(1 / _BOX_SIGMA)

# Columns of a glossy accumulator row (``pix_accum[r + gloss_base]``). The
# drain owns 0..7, the resolve owns 8..12; see DESIGN_glossy_prefilter.md §4.2.
GL_ROW_DIST = 7
GL_ROW_W = 8
GL_ROW_SIGMA_SCALE = 11
GL_ROW_DP = 12
GL_ROW_WIDTH = 13

# Columns of ``gl_main``: the pixel's linear pre-finalize colour (with the
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
# alongside the colour or a blurred reflection would carry a sharp bloom mask.
GL_PYR_WIDTH = 5


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
    x0 = ti.math.clamp(ti.cast(x0f, ti.i32), 0, lw - 1)
    y0 = ti.math.clamp(ti.cast(y0f, ti.i32), 0, lh - 1)
    x1 = ti.math.clamp(x0 + 1, 0, lw - 1)
    y1 = ti.math.clamp(y0 + 1, 0, lh - 1)
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


@ti.kernel
def gloss_scatter(
        num_covered: int, gloss_base: int,
        frame_base: int, frame_rel: int, width: int,
        sigma_max: ti.f32,
        covered_idx: ti.types.ndarray(),
        pix_accum: ti.types.ndarray(),
        gl_main: ti.types.ndarray(), gl_pyr: ti.types.ndarray(),
        out: ti.types.ndarray()):
    """Move one drained tile's glossy pixels into the frame's buffers.

    Must run BEFORE ``wf_composite_accum_sparse`` for the same tile: both read
    the frame buffer's raw prefilled background, and the composite overwrites
    it with the finalized pixel.

    ``csum`` here is deliberately the composite's own arithmetic, repeated
    rather than shared, because ``wf_composite_accum_sparse`` finalizes in the
    same expression and this pass must stop one step short of that -- the
    glossy pixel's final value is not known until the frame's prefilter has
    run, and a tonemap is not something you can add to afterwards.
    """
    for r in range(num_covered):
        gr = r + gloss_base
        sigma_scale = pix_accum[gr, GL_ROW_SIGMA_SCALE]
        if sigma_scale <= 0.0:
            # No prefiltered glossy branch at this pixel: it is an ordinary
            # covered pixel and the tile composite's value for it is final.
            continue
        p = covered_idx[r] - frame_base

        # The pixel's own colour, background folded in (wf_composite_accum
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
        # content sits. d_tot is the total camera path length to the nearest
        # thing the glossy ray found; a ray that found nothing left it at the
        # host's +inf, which is k = 1 -- a reflection of the sky, fully blurred.
        d_tot = pix_accum[gr, GL_ROW_DIST]
        d_p = pix_accum[gr, GL_ROW_DP]
        k_cone = 1.0
        if d_tot < 1e29:
            k_cone = ti.math.clamp(1.0 - d_p / ti.max(d_tot, 1e-6), 0.0, 1.0)
        gl_main[p, GL_MAIN_SIGMA] = ti.min(k_cone * sigma_scale,
                                           sigma_max)


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

    The prefilter is a trilinear fetch of the mip pyramid at the level whose
    box width matches the pixel's blur radius. A box mip is the shape of the
    FOOTPRINT, not of the GGX lobe: that is the standard approximation for the
    radiance half of split-sum, and it is what makes the cost O(pixels) instead
    of O(pixels * radius^2) -- at roughness 0.35 the radius is ~300 px on a
    PREVIEW frame, which no direct blur can afford.
    """
    for p in range(num_pixels):
        sigma = gl_main[p, GL_MAIN_SIGMA]
        if sigma < 0.0:
            continue
        py = p // width
        px = p - py * width

        lvl = 0.0
        if sigma > _BOX_SIGMA:
            lvl = ti.log(sigma) * 1.4426950408889634 + _LOG2_INV_BOX_SIGMA
        lvl = ti.math.clamp(lvl, 0.0, ti.cast(num_levels - 1, ti.f32))
        l0 = ti.cast(lvl, ti.i32)
        l1 = ti.min(l0 + 1, num_levels - 1)
        frac = lvl - ti.cast(l0, ti.f32)

        s0 = _sample_pyramid_level(gl_pyr, level_meta, l0, px, py,
                                   width, height)
        s1 = s0
        if l1 != l0:
            s1 = _sample_pyramid_level(gl_pyr, level_meta, l1, px, py,
                                       width, height)
        refl = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
        for k in ti.static(range(4)):
            a = 0.0
            if s0[4] > 1e-6:
                a = s0[k] / s0[4]
            b = 0.0
            if s1[4] > 1e-6:
                b = s1[k] / s1[4]
            refl[k] = a * (1.0 - frac) + b * frac

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
        # to give a colour weight a scalar glow.
        csum[3] += ti.max(w_energy[0],
                          ti.max(w_energy[1], w_energy[2])) * refl[3]

        color_final = finalize_pixel_color(
            csum, 1.0, tonemapping, tonemap_exposure)
        for ci in ti.static(range(4)):
            if ti.static(tonemapping == 3):
                out[frame_rel, p, ci] = color_final[ci]
            else:
                out[frame_rel, p, ci] = ti.cast(color_final[ci], ti.u8)
