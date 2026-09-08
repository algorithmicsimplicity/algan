"""Reciprocal two-port compensation for the neutral fraction of rough glass."""

from algan.rendering.raytracing.glass_energy import (
    GLASS_COSINE_SIZE,
    GLASS_IOR_SIZE,
    GLASS_ROUGHNESS_SIZE,
)
from algan.taichi_compat import ti


@ti.func
def _pt_glass_energy_lookup(table: ti.template(), rough, eta, cosine):
    """Return incident loss, incident mean loss, and opposite mean loss."""
    r = ti.math.clamp(rough, 0.0, 1.0) * (GLASS_ROUGHNESS_SIZE - 1)
    e = ti.sqrt(ti.abs((eta - 1.0) / (eta + 1.0))) * (GLASS_IOR_SIZE - 1)
    c = ti.sqrt(ti.math.clamp(cosine, 0.0, 1.0)) * (GLASS_COSINE_SIZE - 1)
    ir = ti.min(ti.cast(r, ti.i32), GLASS_ROUGHNESS_SIZE - 2)
    ie = ti.min(ti.cast(e, ti.i32), GLASS_IOR_SIZE - 2)
    ic = ti.min(ti.cast(c, ti.i32), GLASS_COSINE_SIZE - 2)
    fr, fe, fc = r - ir, e - ie, c - ic
    side = 0
    if eta > 1.0:
        side = 1
    result = ti.math.vec3(0.0, 0.0, 0.0)
    for dr, de in ti.static(ti.ndrange(2, 2)):
        wr = fr if dr else 1.0 - fr
        we = fe if de else 1.0 - fe
        base = ((ir + dr) * GLASS_IOR_SIZE + ie + de) * (GLASS_COSINE_SIZE + 1)
        loss = (1.0 - fc) * table[base + ic, side] \
            + fc * table[base + ic + 1, side]
        result += (wr * we) * ti.math.vec3(
            loss, table[base + GLASS_COSINE_SIZE, side],
            table[base + GLASS_COSINE_SIZE, 1 - side])
    return result


@ti.func
def _pt_glass_neutral_weight(f0, eta, metalness, albedo, transmission):
    """Extract a lossless neutral-glass fraction without restoring absorption.

    Subtracting w*F from reflection and w*(1-F) from transmission leaves
    nonnegative facet responses totalling at most 1-w. These angle-independent
    bounds also cover boosted specular and mixed metalness. They are symmetric
    under eta -> 1/eta, so tinting does not break compensation reciprocity.
    """
    m = ti.math.clamp(metalness, 0.0, 1.0)
    base = ((eta - 1.0) / (eta + 1.0)) ** 2
    d = f0 - m * albedo
    w = ti.min((1.0 - m) * ti.math.clamp(transmission, 0.0, 1.0)
               * ti.math.clamp(albedo, 0.0, 1.0), 1.0 - m)
    w = ti.min(w, d / ti.max(base, 1e-12))
    w = ti.min(w, ((1.0 - m) - d) / ti.max(1.0 - base, 1e-12))
    return ti.math.clamp(w, 0.0, 1.0)


@ti.func
def _pt_glass_ms_parameters(table: ti.template(), f0, rough, eta, metalness,
                             albedo, transmission, cosine):
    """Radiance coefficient, mixture probability, and reflected-side mass.

    Let L_i = 1-E_i be lost single-scatter POWER, and barL its cosine mean.
    D = n_i^2 barL_i + n_t^2 barL_t. The added radiance BSDF is
    w * n_i^2 L_i(view) L_j(light) / (pi D), for either destination j.
    Reflection plus eta^-2 transmission integrates to w*L_i(view), not twice
    that energy. Swapping endpoints gives the required eta^2 reciprocity.
    """
    coeff = ti.math.vec3(0.0, 0.0, 0.0)
    p_ms, p_reflect = 0.0, 0.5
    if (rough * rough >= 1e-4) and (cosine > 1e-6) \
            and (eta > 0.0) and (ti.abs(eta - 1.0) >= 1e-4):
        w = _pt_glass_neutral_weight(f0, eta, metalness, albedo, transmission)
        peak = ti.max(w[0], ti.max(w[1], w[2]))
        if peak > 0.0:
            loss = _pt_glass_energy_lookup(table, rough, eta, cosine)
            # Normalise n_i^2 and n_t^2 before using them, avoiding overflow
            # or subtractive cancellation at extreme relative indices.
            ni2, nt2 = eta * eta, 1.0
            if eta > 1.0:
                ni2, nt2 = 1.0, (1.0 / eta) ** 2
            denom = ni2 * loss[1] + nt2 * loss[2]
            if denom > 1e-12:
                coeff = w * (ni2 * loss[0] / (3.141592653589793 * denom))
                p_ms = ti.math.clamp(peak * loss[0], 0.0, 0.95)
                p_reflect = ni2 * loss[1] / denom
    return coeff, p_ms, p_reflect
