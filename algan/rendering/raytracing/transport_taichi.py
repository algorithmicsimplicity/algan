"""Shared ray spawning and dielectric-interface arithmetic."""

from algan.taichi_compat import ti

# Self-intersection offsetting (Wachter & Binder, "A Fast and Robust Method
# for Avoiding Self-Intersection", Ray Tracing Gems 2019 ch. 6). The constants
# are theirs: below ``_OFS_ORIGIN`` in magnitude a coordinate is offset by an
# absolute ``_OFS_FLOAT`` (float spacing near zero is finer than any useful
# world epsilon), above it by ``_OFS_INT`` ULPs, which scales with the point's
# own magnitude exactly as the representable spacing does.
_OFS_ORIGIN = 1.0 / 32.0
_OFS_FLOAT = 1.0 / 65536.0
_OFS_INT = 256.0


@ti.func
def _offset_ray_origin(p, n):
    """Move hit point ``p`` off the surface along ``n`` by a SCALE-AWARE
    epsilon, and return the spawn origin.

    The fixed ``10 * min_hit_distance`` (1e-3 world units) this replaces was
    wrong in both directions: acne on a scene authored at large coordinates,
    where 1e-3 is below the float spacing of the hit point, and light leaking
    through thin geometry on one authored at small coordinates, where 1e-3 is
    a visible distance. Offsetting in INTEGER float space instead ties the
    step to the representable spacing at ``p``. This is a practical offset,
    not a proof of error bounds for every intersection kernel; it does not
    replace the traversal's separate minimum-hit tolerances.

    ``n`` points to the side the ray leaves from; each call site keeps its own
    convention (the geometric normal flipped toward the outgoing direction,
    or the ray direction itself for a zero-thickness pane).
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


@ti.func
def _shadow_tmax(sorigin, wi, ldist):
    """Shadow-ray max distance: the emitter end pulled back by the SAME
    scale-aware offset ``_offset_ray_origin`` applies at the surface end,
    so a light sitting on geometry is not occluded by its own emitter and the
    pull-back scales with the scene the way the spawn offset does (it was a
    fixed ``20 * min_hit_distance``).

    Measure the pull-back between nearby endpoint coordinates rather than
    subtracting large distances, including the 1e7 directional-light sentinel.
    """
    lp = sorigin + wi * ldist
    back = (lp - _offset_ray_origin(lp, -wi)).dot(wi)
    return ldist - ti.max(back, 0.0)


@ti.func
def _transmission_normal(rd, shade_n, face_n):
    """Use an outward interface normal whose side agrees with the geometry.

    Interpolation/normal mapping can tip a shading normal past the silhouette.
    Fresnel and Snell must make the same entry/exit decision as the medium stack.
    """
    n = shade_n
    if n.dot(n) <= 1e-18:
        n = face_n
    if face_n.dot(face_n) > 1e-18:
        if (rd.dot(n) < 0.0) != (rd.dot(face_n) < 0.0):
            n = face_n
    if n.dot(n) <= 1e-18:
        n = -rd
    return n.normalized()


@ti.func
def _dielectric_schlick(cos_signed, inside_over_outside, transmission):
    """Dielectric Schlick reflectance using the actual interface index ratio.

    The signed cosine is negative on entry. For transmission from the denser
    side, evaluate Schlick at the refracted cosine and test total internal
    reflection. Ratios below one are valid (e.g. an air bubble in water).
    Non-transmitting two-sided surfaces keep the ordinary incident-angle lobe.
    """
    ratio = ti.max(inside_over_outside, 1e-6)
    result = 0.0
    if ti.abs(ratio - 1.0) > 1e-4:
        cos_i = ti.math.clamp(ti.abs(cos_signed), 0.0, 1.0)
        r0 = (1.0 - ratio) / (1.0 + ratio)
        f0 = r0 * r0
        eta = 1.0 / ratio
        if cos_signed >= 0.0:
            eta = ratio
        cos_s = cos_i
        tir = False
        if (transmission > 1e-4) and (eta > 1.0):
            sin2_t = eta * eta * ti.max(1.0 - cos_i * cos_i, 0.0)
            if sin2_t >= 1.0:
                tir = True
            else:
                cos_s = ti.sqrt(1.0 - sin2_t)
        if tir:
            result = 1.0
        else:
            result = f0 + (1.0 - f0) * ti.pow(1.0 - cos_s, 5.0)
    return result
