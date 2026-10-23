"""Small compiled probes for density-grid tracking, independent of render noise."""

from algan.rendering.raytracing.pt_volume_taichi import (
    PT_VOL_HEADER,
    _pt_vol_density,
    _pt_vol_seed,
    _pt_vol_track,
)
from algan.taichi_compat import ti


@ti.kernel
def density_probe(
    vol_i: ti.types.ndarray(),
    vol_f: ti.types.ndarray(),
    points: ti.types.ndarray(),
    out: ti.types.ndarray(),
):
    # Volume 0's samples at box coordinates ``points``.
    n = ti.math.ivec3(
        vol_i[PT_VOL_HEADER + 1], vol_i[PT_VOL_HEADER + 2], vol_i[PT_VOL_HEADER + 3]
    )
    off = vol_i[PT_VOL_HEADER + 4]
    for i in range(points.shape[0]):
        u = ti.math.vec3(points[i, 0], points[i, 1], points[i, 2])
        out[i] = _pt_vol_density(vol_f, off, n, u)


@ti.kernel
def track_probe(
    vol_i: ti.types.ndarray(),
    vol_f: ti.types.ndarray(),
    volume: ti.i32,
    frame: ti.i32,
    ray: ti.types.ndarray(),
    coeff: ti.types.ndarray(),
    scatter: ti.i32,
    samples: ti.types.ndarray(),
    out: ti.types.ndarray(),
):
    """One tracked segment per sample: ``ray`` is ``ro, rd, t_start, t_end``,
    ``coeff`` is ``sigma_a, sigma_s``. Rows of ``out`` are ``collision, t,
    weight (3), overflow``.
    """
    for i in range(samples.shape[0]):
        ro = ti.math.vec3(ray[0], ray[1], ray[2])
        rd = ti.math.vec3(ray[3], ray[4], ray[5])
        sa = ti.math.vec3(coeff[0], coeff[1], coeff[2])
        ss = ti.math.vec3(coeff[3], coeff[4], coeff[5])
        seed = _pt_vol_seed(ti.u32(0x2545F491), i)
        hit, t, weight, overflow = _pt_vol_track(
            vol_i,
            vol_f,
            volume,
            frame,
            ro,
            rd,
            ray[6],
            ray[7],
            sa,
            ss,
            ti.math.vec3(1.0, 1.0, 1.0),
            scatter,
            seed,
            ti.math.vec2(samples[i, 0], samples[i, 1]),
        )
        out[i, 0] = ti.cast(hit, ti.f32)
        out[i, 1] = t
        for k in ti.static(range(3)):
            out[i, 2 + k] = weight[k]
        out[i, 5] = ti.cast(overflow, ti.f32)
