"""Scale-aware secondary-ray origin placement shared by both ray tracers.

The helper implements Wächter & Binder's floating-point-aware origin offset
from *Ray Tracing Gems* (2019, ch. 6).  Secondary rays must leave a hit point
on the side of the surface they travel into, but a fixed world-space epsilon
cannot do that robustly across scene scales: it rounds away at large
coordinates and skips nearby geometry at small ones.
"""

from algan.taichi_compat import ti

# Wächter/Binder constants.  Below ``_OFS_ORIGIN`` in magnitude a coordinate
# is offset by an absolute ``_OFS_FLOAT``; above it, the bit pattern is moved by
# ``_OFS_INT`` ULPs so the displacement follows the local f32 spacing.
_OFS_ORIGIN = 1.0 / 32.0
_OFS_FLOAT = 1.0 / 65536.0
_OFS_INT = 256.0


@ti.func
def _offset_ray_origin(p, n):
    """Return ``p`` moved robustly toward the side indicated by ``n``.

    ``n`` may be a geometric normal oriented toward the outgoing ray, or the
    outgoing direction itself for a zero-thickness surface.  The operation is
    component-wise in f32 bit space so it remains representable far from the
    world origin while retaining a small absolute fallback near zero.
    """
    out = ti.math.vec3(0.0, 0.0, 0.0)
    for k in ti.static(range(3)):
        off_i = ti.cast(_OFS_INT * n[k], ti.i32)
        if p[k] < 0.0:
            off_i = -off_i
        p_i = ti.bit_cast(ti.bit_cast(p[k], ti.i32) + off_i, ti.f32)
        if ti.abs(p[k]) < _OFS_ORIGIN:
            out[k] = p[k] + _OFS_FLOAT * n[k]
        else:
            out[k] = p_i
    return out
