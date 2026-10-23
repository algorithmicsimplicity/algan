"""Heterogeneous (density-grid) media for the path tracer.

A density volume scales a closed physical interior's coefficients by a scalar
field: ``sigma_a(x) = d(x) * sigma_a`` and ``sigma_s(x) = d(x) * sigma_s``, so
the single-scattering albedo and the phase function stay the material's while
the extinction varies. ``d`` is a voxel grid sampled trilinearly (clamped to
its edge samples) over a box in the owning Mob's local frame, and zero outside
that box. The host (``pt_volume.prepare_volumes``) uploads the grids and a
per-frame world-to-box transform for each, so the field moves, turns and
stretches with the Mob.

Transport uses the null-collision formulation over a piecewise-constant
majorant. Each grid carries a coarse majorant grid (the maximum density any
point of a coarse cell can interpolate to); a 3-D DDA walks it, and within a
cell tentative collisions are an exponential process at
``majorant = cell_max * max_c(sigma_t)``. Restarting that process at every cell
boundary is exact, because it is memoryless.

* **Free flights** (``scatter != 0``) use spectral tracking (Kutz et al. 2017,
  "Spectral and Decomposition Tracking for Rendering Heterogeneous Volumes"):
  at a tentative collision the path scatters with probability ``P_s``
  proportional to the throughput-weighted mean of ``sigma_s(x)``, or continues
  as a null collision with ``1 - P_s`` proportional to that of
  ``sigma_n(x) = majorant - sigma_t(x)``, and the RGB weights
  ``sigma_s / (majorant * P_s)`` and ``sigma_n / (majorant * P_n)`` keep the
  estimator unbiased per channel. There is no emission, so absorption is not a
  sampled event: its mass is simply absent from both weights, which for a grey
  medium reduces every tentative collision to the factor
  ``1 - sigma_a(x) / majorant`` -- ratio tracking for the absorbed part and
  delta tracking for the scattered part.
* **Transmittance** (``scatter == 0``: shadow connections, and flights that may
  no longer scatter) is ratio tracking, ``prod(1 - sigma_t(x_i) / majorant)``
  per channel, with Russian roulette once the estimate falls below
  ``_PT_VOL_RR_THRESHOLD`` (Novak et al. 2014; the same rule as PBRT-v4).

Randomness. The first tentative collision of a free flight draws from the
flight's Sobol pair (``2 + 6b + 4``: y distance, x event), exactly as the
homogeneous flight does, so thin media keep its stratification. Every later
draw -- the count is unbounded -- is a hash of a seed the caller derives from
the path identity, the flight's segment ordinal and the sample index, plus the
draw ordinal: a pure function of path identity, never shared mutable state
(roadmap contract 1).

Bounds. One tracked segment takes at most ``PT_VOL_MAX_STEPS`` loop steps
(coarse-cell crossings plus tentative collisions). A segment that exhausts it
is fail-closed -- its weight is zeroed -- and counted in
``pt_stats[PT_STAT_VOLUME_STEPS]`` (``RenderPlan.truncations.medium_tracking``),
like the medium stack's other ceilings.
"""

from algan.rendering.raytracing.shading_taichi import _MAT_DENSITY_VOLUME
from algan.taichi_compat import ti

#: ``vol_i`` header words: ``[V, T]`` -- the volume count and the number of
#: transform rows (1, or the batch's frame count; read as ``f % T``).
PT_VOL_HEADER = 2
#: ``vol_i`` words per volume: ``key, nx, ny, nz, density offset, mx, my, mz``.
#: The majorant grid follows the density samples in ``vol_f``.
PT_VOL_STRIDE = 8
#: ``vol_f`` floats per (frame, volume) transform: the 3x3 world-to-box matrix,
#: row-major, then its translation. Box coordinates run over ``[0, 1]^3``.
PT_VOL_XFORM = 12
#: Loop steps one tracked segment may take before it is absorbed and counted.
PT_VOL_MAX_STEPS = 4096
#: ``pt_stats`` slot of segments that exhausted ``PT_VOL_MAX_STEPS``.
PT_STAT_VOLUME_STEPS = 4

# Ratio tracking's roulette: below this estimate, survive with probability
# ``_PT_VOL_RR_SURVIVE`` and divide by it.
_PT_VOL_RR_THRESHOLD = 0.05
_PT_VOL_RR_SURVIVE = 0.25


@ti.func
def _pt_vol_hash(x: ti.u32) -> ti.u32:
    # Wellons' lowbias32, the path tracer's own finalizer (``_pt_hash``),
    # repeated here because that module imports this one.
    x ^= x >> 16
    x = x * ti.u32(0x7FEB352D)
    x ^= x >> 15
    x = x * ti.u32(0x846CA68B)
    x ^= x >> 16
    return x


@ti.func
def _pt_vol_seed(seed: ti.u32, value) -> ti.u32:
    """Fold one more identity word into a tracking seed."""
    v = ti.cast(value, ti.u32)
    return _pt_vol_hash(seed ^ (v + ti.u32(0x9E3779B9) + (seed << 6) + (seed >> 2)))


@ti.func
def _pt_vol_rand(seed: ti.u32, k) -> ti.f32:
    """Uniform in ``[0, 1)``: draw ``k`` of the stream ``seed`` names."""
    h = _pt_vol_seed(seed ^ ti.u32(0x632BE5AB), k)
    return ti.cast(h >> 8, ti.f32) * (1.0 / 16777216.0)


@ti.func
def _pt_medium_volume(vol_i: ti.template(), tri_mat: ti.template(), f, prim):
    """Index of the density volume a medium triangle names, or -1.

    The material block's ``_MAT_DENSITY_VOLUME`` slot holds the volume's key
    (0.0, the padding rule's "as before", is a homogeneous interior). A key the
    host table does not hold is also -1; ``prepare_volumes`` refuses to build a
    table that misses a key the batch uses, so that is never a silent fallback.
    """
    v = -1
    if (prim >= 0) and (tri_mat.shape[2] > _MAT_DENSITY_VOLUME):
        key = ti.cast(
            tri_mat[f % tri_mat.shape[0], prim, _MAT_DENSITY_VOLUME] + 0.5, ti.i32
        )
        if key > 0:
            for q in range(vol_i[0]):
                if vol_i[PT_VOL_HEADER + PT_VOL_STRIDE * q] == key:
                    v = q
    return v


@ti.func
def _pt_vol_density(vol_f: ti.template(), off, n, u):
    """Trilinear density at box coordinates ``u``, clamped to the edge samples.

    Samples sit at cell centres, ``(i + 0.5) / n``, stored C-order over
    ``[nx, ny, nz]``. The interpolant is a convex combination of eight
    samples, so it never exceeds their maximum -- what the majorant grid bounds.
    """
    lo = ti.math.ivec3(0, 0, 0)
    hi = ti.math.ivec3(0, 0, 0)
    w = ti.math.vec3(0.0, 0.0, 0.0)
    for k in ti.static(range(3)):
        top = ti.cast(n[k] - 1, ti.f32)
        g = ti.math.clamp(u[k] * ti.cast(n[k], ti.f32) - 0.5, 0.0, top)
        i = ti.min(ti.cast(g, ti.i32), n[k] - 1)
        lo[k] = i
        hi[k] = ti.min(i + 1, n[k] - 1)
        w[k] = g - ti.cast(i, ti.f32)
    ny = n[1]
    nz = n[2]
    c000 = vol_f[off + (lo[0] * ny + lo[1]) * nz + lo[2]]
    c001 = vol_f[off + (lo[0] * ny + lo[1]) * nz + hi[2]]
    c010 = vol_f[off + (lo[0] * ny + hi[1]) * nz + lo[2]]
    c011 = vol_f[off + (lo[0] * ny + hi[1]) * nz + hi[2]]
    c100 = vol_f[off + (hi[0] * ny + lo[1]) * nz + lo[2]]
    c101 = vol_f[off + (hi[0] * ny + lo[1]) * nz + hi[2]]
    c110 = vol_f[off + (hi[0] * ny + hi[1]) * nz + lo[2]]
    c111 = vol_f[off + (hi[0] * ny + hi[1]) * nz + hi[2]]
    c00 = c000 + (c001 - c000) * w[2]
    c01 = c010 + (c011 - c010) * w[2]
    c10 = c100 + (c101 - c100) * w[2]
    c11 = c110 + (c111 - c110) * w[2]
    c0 = c00 + (c01 - c00) * w[1]
    c1 = c10 + (c11 - c10) * w[1]
    return ti.max(c0 + (c1 - c0) * w[0], 0.0)


@ti.func
def _pt_vol_to_box(vol_i: ti.template(), vol_f: ti.template(), v, f, ro, rd):
    """The ray ``ro + t * rd`` in box coordinates: ``(origin, direction)``.

    The map is affine, so ``t`` keeps measuring WORLD distance along the ray
    and the world-space coefficients apply unchanged.
    """
    xo = ((f % vol_i[1]) * vol_i[0] + v) * PT_VOL_XFORM
    o_u = ti.math.vec3(vol_f[xo + 9], vol_f[xo + 10], vol_f[xo + 11])
    d_u = ti.math.vec3(0.0, 0.0, 0.0)
    for j in ti.static(range(3)):
        row = ti.math.vec3(
            vol_f[xo + 3 * j], vol_f[xo + 3 * j + 1], vol_f[xo + 3 * j + 2]
        )
        o_u[j] += row.dot(ro)
        d_u[j] = row.dot(rd)
    return o_u, d_u


@ti.func
def _pt_vol_track(
    vol_i: ti.template(),
    vol_f: ti.template(),
    v,
    f,
    ro,
    rd,
    t_start,
    t_end,
    sigma_a,
    sigma_s,
    history,
    scatter,
    seed: ti.u32,
    u_first,
):
    """Track ``[t_start, t_end)`` of the ray through density volume ``v``.

    Returns ``(collision, t_collision, weight, overflow)``. ``weight`` is the
    RGB throughput factor in the convention of ``_pt_medium_sample``: with
    ``scatter == 0`` it is a transmittance estimate and ``collision`` is 0;
    with ``scatter != 0`` a collision's weight already carries
    ``sigma_s(x) / (majorant * P_s)``, so the scatter vertex multiplies by the
    phase function only. ``history`` is the path throughput the spectral
    tracking probabilities weight the channels by. ``u_first`` is the first
    tentative collision's pair (y distance, x event); later draws hash
    ``seed``. ``overflow`` is 1 when the segment exhausted
    ``PT_VOL_MAX_STEPS``; the weight is zero then (fail-closed).
    """
    weight = ti.math.vec3(1.0, 1.0, 1.0)
    collision = 0
    t_coll = t_end
    overflow = 0
    base = PT_VOL_HEADER + PT_VOL_STRIDE * v
    n = ti.math.ivec3(vol_i[base + 1], vol_i[base + 2], vol_i[base + 3])
    d_off = vol_i[base + 4]
    m = ti.math.ivec3(vol_i[base + 5], vol_i[base + 6], vol_i[base + 7])
    m_off = d_off + n[0] * n[1] * n[2]
    o_u, d_u = _pt_vol_to_box(vol_i, vol_f, v, f, ro, rd)
    # Clip to the box: outside it the density is zero, so the interval there
    # is vacuum and costs nothing.
    ta = t_start
    tb = t_end
    for k in ti.static(range(3)):
        if ti.abs(d_u[k]) < 1e-30:
            if (o_u[k] < 0.0) or (o_u[k] > 1.0):
                tb = -1e30
        else:
            inv = 1.0 / d_u[k]
            t0 = -o_u[k] * inv
            t1 = (1.0 - o_u[k]) * inv
            ta = ti.max(ta, ti.min(t0, t1))
            tb = ti.min(tb, ti.max(t0, t1))
    sigma_t = sigma_a + sigma_s
    s_bar = ti.max(sigma_t[0], ti.max(sigma_t[1], sigma_t[2]))
    if (ta < tb) and (s_bar > 0.0):
        p0 = o_u + ta * d_u
        cell = ti.math.ivec3(0, 0, 0)
        step = ti.math.ivec3(0, 0, 0)
        t_next = ti.math.vec3(1e30, 1e30, 1e30)
        t_delta = ti.math.vec3(1e30, 1e30, 1e30)
        for k in ti.static(range(3)):
            mk = ti.cast(m[k], ti.f32)
            ck = ti.math.clamp(ti.cast(ti.floor(p0[k] * mk), ti.i32), 0, m[k] - 1)
            cell[k] = ck
            if d_u[k] > 1e-30:
                step[k] = 1
                t_next[k] = ta + (ti.cast(ck + 1, ti.f32) / mk - p0[k]) / d_u[k]
                t_delta[k] = 1.0 / (mk * d_u[k])
            elif d_u[k] < -1e-30:
                step[k] = -1
                t_next[k] = ta + (ti.cast(ck, ti.f32) / mk - p0[k]) / d_u[k]
                t_delta[k] = -1.0 / (mk * d_u[k])
        t = ta
        steps = 0
        draws = 0
        done = 0
        while done == 0:
            if steps >= PT_VOL_MAX_STEPS:
                overflow = 1
                done = 1
                weight = ti.math.vec3(0.0, 0.0, 0.0)
                collision = 0
            else:
                steps += 1
                t_cell = ti.min(ti.min(t_next[0], t_next[1]), ti.min(t_next[2], tb))
                maj = (
                    vol_f[m_off + (cell[0] * m[1] + cell[1]) * m[2] + cell[2]] * s_bar
                )
                advance = 1
                if (maj > 0.0) and (t_cell > t):
                    xi = u_first[1]
                    if draws > 0:
                        xi = _pt_vol_rand(seed, 2 * draws)
                    dt = -ti.log(ti.max(1.0 - xi, 1e-30)) / maj
                    if t + dt < t_cell:
                        advance = 0
                        t += dt
                        dens = _pt_vol_density(vol_f, d_off, n, o_u + t * d_u)
                        ss = dens * sigma_s
                        sn = ti.math.vec3(0.0, 0.0, 0.0)
                        for k in ti.static(range(3)):
                            sn[k] = ti.max(maj - dens * sigma_t[k], 0.0)
                        if scatter != 0:
                            hw = ti.abs(history * weight)
                            cs = (ss * hw).sum()
                            cn = (sn * hw).sum()
                            if cs + cn <= 0.0:
                                cs = ss.sum()
                                cn = sn.sum()
                            ps = 0.0
                            if cs + cn > 0.0:
                                ps = cs / (cs + cn)
                            xe = u_first[0]
                            if draws > 0:
                                xe = _pt_vol_rand(seed, 2 * draws + 1)
                            if xe < ps:
                                weight *= ss / (maj * ps)
                                collision = 1
                                t_coll = t
                                done = 1
                            else:
                                weight *= sn / (maj * (1.0 - ps))
                        else:
                            weight *= sn / maj
                            w_max = ti.max(weight[0], ti.max(weight[1], weight[2]))
                            if w_max <= 0.0:
                                done = 1
                            elif w_max < _PT_VOL_RR_THRESHOLD:
                                if _pt_vol_rand(seed, 2 * draws + 1) < (
                                    1.0 - _PT_VOL_RR_SURVIVE
                                ):
                                    weight = ti.math.vec3(0.0, 0.0, 0.0)
                                    done = 1
                                else:
                                    weight /= _PT_VOL_RR_SURVIVE
                    draws += 1
                if (advance == 1) and (done == 0):
                    t = ti.max(t, t_cell)
                    if t >= tb:
                        done = 1
                    else:
                        axis = 0
                        t_min = t_next[0]
                        if t_next[1] < t_min:
                            axis = 1
                            t_min = t_next[1]
                        if t_next[2] < t_min:
                            axis = 2
                        for k in ti.static(range(3)):
                            if k == axis:
                                cell[k] += step[k]
                                t_next[k] += t_delta[k]
                                if (cell[k] < 0) or (cell[k] >= m[k]):
                                    done = 1
    return collision, t_coll, weight, overflow
