"""Reciprocal two-sided loss lookup and conservative dielectric colour budget.

This is a separable energy closure, not an exact microfacet random walk.
See DESIGN_path_tracer_roadmap.md, "Coupled rough-glass compensation".
"""

from algan.taichi_compat import ti

from .glass_energy import GLASS_ETA_SIZE, GLASS_MU_SIZE, GLASS_ROUGH_SIZE


@ti.func
def _glass_losses(data: ti.template(), base, rough, eta, mu):
    """Return incident/opposite directional loss and their cosine averages."""
    x = ti.math.clamp(ti.abs(eta - 1.0) / (eta + 1.0), 0.0, 1.0)
    xe = x * (GLASS_ETA_SIZE - 1)
    xr = ti.math.clamp(rough, 0.0, 1.0) * (GLASS_ROUGH_SIZE - 1)
    xm = ti.math.clamp(mu, 0.0, 1.0) * (GLASS_MU_SIZE - 1)
    ie = ti.min(ti.cast(xe, ti.i32), GLASS_ETA_SIZE - 2)
    ir = ti.min(ti.cast(xr, ti.i32), GLASS_ROUGH_SIZE - 2)
    im = ti.min(ti.cast(xm, ti.i32), GLASS_MU_SIZE - 2)
    fe, fr, fm = xe - ie, xr - ir, xm - im
    values = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
    # Four eta/roughness corners, two interface sides. Average rows use the
    # exact cosine integral of the same piecewise-linear directional profile.
    for de, dr, side in ti.static(ti.ndrange(2, 2, 2)):
        weight = (fe if de else 1.0 - fe) * (fr if dr else 1.0 - fr)
        off = base + (((ie + de) * GLASS_ROUGH_SIZE + ir + dr) * 2 + side) * (GLASS_MU_SIZE + 1)
        values[side] += weight * (data[off + im] * (1.0 - fm) + data[off + im + 1] * fm)
        values[side + 2] += weight * data[off + GLASS_MU_SIZE]
    if eta > 1.0:
        values = ti.math.vec4(values[1], values[0], values[3], values[2])
    return values


@ti.func
def _glass_compensated_fraction(f0, eta, metalness, albedo, transmission):
    """Common ideal dielectric fraction present in BOTH authored outcomes.

    Compensating 1 - (authored R + T) would turn absorption into white light.
    These bounds instead split off c*[F, 1-F], with a nonnegative passive
    single-scatter remainder. This is conservative for tinted/partial glass.
    """
    m = ti.math.clamp(metalness, 0.0, 1.0)
    d = 1.0 - m
    b = ti.pow((eta - 1.0) / (eta + 1.0), 2.0)
    r0 = f0 - m * albedo
    c = ti.math.vec3(0.0, 0.0, 0.0)
    for k in ti.static(range(3)):
        if (albedo[k] >= 0.0) and (albedo[k] <= 1.0):
            c[k] = ti.max(0.0, ti.min(d, ti.min(
                r0[k] / ti.max(b, 1e-12), ti.min(
                (d - r0[k]) / ti.max(1.0 - b, 1e-12),
                d * transmission * albedo[k]))))
    return c
