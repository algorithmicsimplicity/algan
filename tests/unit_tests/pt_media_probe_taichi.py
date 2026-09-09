"""Small compiled probes for homogeneous transport, independent of render noise."""

from algan.rendering.raytracing.pt_media_taichi import (
    _pt_hg_pdf,
    _pt_hg_sample,
    _pt_medium_cross,
    _pt_medium_sample,
)
from algan.taichi_compat import ti


@ti.kernel
def phase_probe(g: ti.f32, samples: ti.types.ndarray(), out: ti.types.ndarray()):
    for i in range(samples.shape[0]):
        u = ti.math.vec2(samples[i, 0], samples[i, 1])
        wi, pdf = _pt_hg_sample(ti.math.vec3(0.0, 0.0, 1.0), g, u)
        out[i, 0] = wi[2]
        out[i, 1] = wi.norm()
        out[i, 2] = pdf
        out[i, 3] = _pt_hg_pdf(wi[2], g)


@ti.kernel
def flight_probe(
    sa: ti.types.ndarray(),
    ss: ti.types.ndarray(),
    length: ti.f32,
    samples: ti.types.ndarray(),
    out: ti.types.ndarray(),
):
    for i in range(samples.shape[0]):
        a = ti.math.vec3(sa[0], sa[1], sa[2])
        s = ti.math.vec3(ss[0], ss[1], ss[2])
        hit, distance, weight = _pt_medium_sample(
            a, s, length, ti.math.vec2(samples[i, 0], samples[i, 1])
        )
        out[i, 0] = ti.cast(hit, ti.f32)
        out[i, 1] = distance
        for k in ti.static(range(3)):
            out[i, 2 + k] = weight[k]


@ti.kernel
def stack_probe(
    shell: ti.types.ndarray(), events: ti.types.ndarray(), out: ti.types.ndarray()
):
    # One event stream threads a single stack, so the walk must be serial: the
    # OUTERMOST range-for is the parallel one on every backend, and sharing the
    # stack across its threads races (on a GPU it does, visibly). A one-trip
    # outer loop parks the walk in an inner loop, which is always serial.
    for _ in range(1):
        stack = ti.Vector([-1, -1, -1, -1])
        for i in range(events.shape[0]):
            stack, overflow = _pt_medium_cross(
                stack, shell, 0, events[i, 0], shell.shape[1] // 2, events[i, 1] != 0
            )
            for k in ti.static(range(4)):
                out[i, k] = stack[k]
            out[i, 4] = overflow
