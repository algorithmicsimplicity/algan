"""Homogeneous RGB transport and shell-identity tracking for the path tracer.

No path splits and no per-scene specialization. A medium stack holds at most
four representative triangle indices; identity comparisons use tri_obj, NOT
triangle indices or the opacity ceiling's transmission-exempt shell ids.
Heterogeneous density fields and diffusion-profile SSS are deliberately absent.

Distance sampling uses the RGB channel-mixture estimator (PBRT, 3e, 15.2):
choose a channel uniformly, sample its exponential, and divide the spectral
transmittance by the marginal density, not the selected channel's density.
The HG convention here uses *travel* directions: positive g is forward.
"""

from algan.rendering.raytracing.raytrace_kernels_taichi import (
    _nearest_surface_g,
    _safe_inverse,
    depth_tie_epsilon,
    max_surfaces_per_ray,
)
from algan.rendering.raytracing.shading_taichi import (
    _MAT_ATTENUATION_SIGMA,
    _MAT_PHASE_G,
    _MAT_SIGMA_S,
)
from algan.taichi_compat import ti

PT_MEDIA_SLOTS = 4
PT_STAT_MEDIA_STACK = 2
PT_STAT_MEDIA_QUERY = 3


@ti.func
def _pt_hg_pdf(cosine, g):
    c = ti.math.clamp(cosine, -1.0, 1.0)
    ag = ti.abs(g)
    oriented = c if g >= 0.0 else -c
    denom = ti.max((1.0 - ag) * (1.0 - ag) + 2.0 * ag * (1.0 - oriented), 1e-30)
    return ((1.0 - g) * (1.0 + g)) / (12.566370614359172 * denom * ti.sqrt(denom))


@ti.func
def _pt_hg_sample(rd, g, u):
    # Algebraically expanded inverse CDF near g=0, avoiding both division
    # by g and the biased isotropic approximation sometimes used there.
    v = 2.0 * u[0] - 1.0
    cosine = (2.0 * v * (1.0 + g * g) + g * (3.0 - g * g + v * v * (1.0 + g * g))) / (
        2.0 * (1.0 + g * v) * (1.0 + g * v)
    )
    if ti.abs(g) > 1e-3:
        ratio = (1.0 - g * g) / (1.0 - g + 2.0 * g * u[0])
        cosine = (1.0 + g * g - ratio * ratio) / (2.0 * g)
    cosine = ti.math.clamp(cosine, -1.0, 1.0)
    sine = ti.sqrt(ti.max(0.0, 1.0 - cosine * cosine))
    axis = ti.math.vec3(0.0, 0.0, 1.0)
    if ti.abs(rd[2]) > 0.9:
        axis = ti.math.vec3(0.0, 1.0, 0.0)
    tangent = axis.cross(rd).normalized()
    bitangent = rd.cross(tangent)
    phi = 6.283185307179586 * u[1]
    wi = (
        cosine * rd + sine * (ti.cos(phi) * tangent + ti.sin(phi) * bitangent)
    ).normalized()
    return wi, _pt_hg_pdf(cosine, g)


@ti.func
def _pt_medium_sample(sigma_a, sigma_s, length, u):
    """Return (collision, distance, RGB throughput / marginal pdf).

    Zero-rate channels have a point mass at infinity. In particular, a zero
    coefficient in one RGB channel must NOT become opaque at long distances.
    The purely absorbing case is integrated analytically by the caller.
    """
    sigma_t = sigma_a + sigma_s
    channel = ti.min(ti.cast(u[0] * 3.0, ti.i32), 2)
    distance = 1e30
    rate = sigma_t[channel]
    if rate > 0.0:
        distance = -ti.log(ti.max(1.0 - u[1], 1e-30)) / rate
    collision = distance < length
    distance = ti.min(distance, length)
    transmittance = ti.exp(-sigma_t * distance)
    density = transmittance
    weight = transmittance
    if collision:
        density *= sigma_t
        weight *= sigma_s
    pdf = (density[0] + density[1] + density[2]) / 3.0
    if pdf > 0.0:
        weight /= pdf
    else:
        weight = ti.math.vec3(0.0, 0.0, 0.0)
    return collision, distance, weight


@ti.func
def _pt_medium_coeff(tri_mat: ti.template(), f, prim):
    sigma_a = ti.math.vec3(0.0, 0.0, 0.0)
    sigma_s = ti.math.vec3(0.0, 0.0, 0.0)
    g = 0.0
    if prim >= 0:
        row = f % tri_mat.shape[0]
        for k in ti.static(range(3)):
            sigma_a[k] = ti.max(tri_mat[row, prim, _MAT_ATTENUATION_SIGMA + k], 0.0)
            sigma_s[k] = ti.max(tri_mat[row, prim, _MAT_SIGMA_S + k], 0.0)
        g = tri_mat[row, prim, _MAT_PHASE_G]
    return sigma_a, sigma_s, g


@ti.func
def _pt_medium_id(shell: ti.template(), f, prim, offset):
    sid = -1
    if prim >= 0:
        sid = shell[f % shell.shape[0], offset + prim]
    return sid


@ti.func
def _pt_medium_top(stack):
    prim = -1
    for k in ti.static(range(PT_MEDIA_SLOTS)):
        if stack[k] >= 0:
            prim = stack[k]
    return prim


@ti.func
def _pt_medium_find(stack, shell: ti.template(), f, prim, offset):
    found = -1
    sid = _pt_medium_id(shell, f, prim, offset)
    for k in ti.static(range(PT_MEDIA_SLOTS)):
        if stack[k] >= 0:
            if _pt_medium_id(shell, f, stack[k], offset) == sid:
                found = k
    return found


@ti.func
def _pt_medium_cross(stack, shell: ti.template(), f, prim, offset, entering):
    """Push on entry; remove the MATCHING object on exit, preserving others.

    Overlapping shells use last-entered priority. A repeated edge of a shell
    cannot push it twice. Overflow is fail-closed, never silently truncated.
    """
    out = stack
    overflow = 0
    sid = _pt_medium_id(shell, f, prim, offset)
    if sid >= 0:
        found = _pt_medium_find(out, shell, f, prim, offset)
        if entering:
            if found < 0:
                inserted = 0
                for k in ti.static(range(PT_MEDIA_SLOTS)):
                    if (inserted == 0) and (out[k] < 0):
                        out[k] = prim
                        inserted = 1
                overflow = 1 - inserted
        elif found >= 0:
            for k in ti.static(range(PT_MEDIA_SLOTS - 1)):
                if k >= found:
                    out[k] = out[k + 1]
            out[PT_MEDIA_SLOTS - 1] = -1
    return out, overflow


@ti.func
def _pt_medium_normal(tri_pos: ti.template(), f, prim):
    row = f % tri_pos.shape[0]
    p0 = ti.math.vec3(
        tri_pos[row, prim, 0], tri_pos[row, prim, 1], tri_pos[row, prim, 2]
    )
    p1 = ti.math.vec3(
        tri_pos[row, prim, 3], tri_pos[row, prim, 4], tri_pos[row, prim, 5]
    )
    p2 = ti.math.vec3(
        tri_pos[row, prim, 6], tri_pos[row, prim, 7], tri_pos[row, prim, 8]
    )
    return (p1 - p0).cross(p2 - p0)


@ti.func
def _pt_medium_next(
    refit: ti.template(),
    ro,
    rd,
    f,
    t_prev,
    layer_prev,
    max_t,
    layer_offset,
    nodes: ti.template(),
    miss: ti.template(),
    leaf: ti.template(),
    span: ti.template(),
    first_leaf,
    tri_pos: ti.template(),
):
    inv_rd = ti.math.vec3(
        _safe_inverse(rd[0]), _safe_inverse(rd[1]), _safe_inverse(rd[2])
    )
    # The Bezier arm is compile-time absent, so its placeholder arguments are
    # never dereferenced. Volume boundaries ignore casts_shadows: extinction
    # is transport through matter, not an optional surface shadow effect.
    found, t, layer, prim, _kind, _a, _b, _border, edge = _nearest_surface_g(
        refit,
        1,
        0,
        ro,
        rd,
        inv_rd,
        f,
        ti.cast(f, ti.f32),
        t_prev,
        layer_prev,
        max_t,
        0.0,
        0.0,
        layer_offset,
        nodes,
        miss,
        leaf,
        span,
        first_leaf,
        tri_pos,
        nodes,
        miss,
        leaf,
        span,
        0,
        tri_pos,
        tri_pos,
        tri_pos,
        -1,
        -1,
        0.0,
        0.0,
        tri_pos,
        0,
        0,
    )
    return found, t, layer, prim, edge


@ti.func
def _pt_initial_media(
    refit: ti.template(),
    ro,
    f,
    layer_offset,
    nodes: ti.template(),
    miss: ti.template(),
    leaf: ti.template(),
    span: ti.template(),
    first_leaf,
    tri_pos: ti.template(),
    shell: ti.template(),
    offset,
):
    """Classify the actual near-clipped camera point, including nested shells.

    In a forward probe, unmatched exits identify shells containing the origin.
    Reversing their order puts the innermost shell on top. Entry/exit matching
    uses object identity, so a nearer unrelated object cannot hide containment.
    """
    stack = ti.Vector([-1, -1, -1, -1])
    entered = ti.Vector([-1, -1, -1, -1])
    origin = ti.Vector([-1, -1, -1, -1])
    rd = ti.math.vec3(0.37139067, 0.55708601, 0.74278135)
    t_prev = 0.0
    layer_prev = 1e30
    seam = -1e30
    count = 0
    n_origin = 0
    overflow = 0
    finished = 0
    while count < max_surfaces_per_ray:
        found, t, layer, prim, edge = _pt_medium_next(
            refit,
            ro,
            rd,
            f,
            t_prev,
            layer_prev,
            1e30,
            layer_offset,
            nodes,
            miss,
            leaf,
            span,
            first_leaf,
            tri_pos,
        )
        if found == 0:
            finished = 1
            break
        count += 1
        duplicate = (edge != 0) and (t - seam <= depth_tie_epsilon)
        seam = t if edge != 0 else -1e30
        t_prev = t
        layer_prev = layer
        if not duplicate:
            if _pt_medium_id(shell, f, prim, offset) >= 0:
                entering = rd.dot(_pt_medium_normal(tri_pos, f, prim)) < 0.0
                known = _pt_medium_find(entered, shell, f, prim, offset)
                if (not entering) and (known < 0):
                    if n_origin < PT_MEDIA_SLOTS:
                        origin[n_origin] = prim
                        n_origin += 1
                    else:
                        overflow = 1
                else:
                    entered, failed = _pt_medium_cross(
                        entered, shell, f, prim, offset, entering
                    )
                    overflow = ti.max(overflow, failed)
    for k in ti.static(range(PT_MEDIA_SLOTS)):
        if k < n_origin:
            stack[k] = origin[n_origin - k - 1]
    return stack, overflow, 1 - finished


@ti.func
def _pt_medium_transmittance(
    refit: ti.template(),
    ro,
    rd,
    f,
    max_t,
    stack,
    layer_offset,
    nodes: ti.template(),
    miss: ti.template(),
    leaf: ti.template(),
    span: ti.template(),
    first_leaf,
    tri_pos: ti.template(),
    tri_mat: ti.template(),
    shell: ti.template(),
    offset,
):
    """Straight-connection transport through null interfaces and nested media.

    A refracting boundary is not a null collision: this NEE strategy cannot
    sample the bent path or its directional Jacobian. Stop the connection
    there and let the ordinary Fresnel/BSDF walk sample it. Allowing the legacy
    straight transparent shadow approximation here double-counts paths whose
    specular boundary resets prev_pdf, and creates energy in dense glass.
    """
    media = stack
    result = ti.math.vec3(1.0, 1.0, 1.0)
    t_prev = 0.0
    layer_prev = 1e30
    seam = -1e30
    count = 0
    overflow = 0
    finished = 0
    while count < max_surfaces_per_ray:
        found, t, layer, prim, edge = _pt_medium_next(
            refit,
            ro,
            rd,
            f,
            t_prev,
            layer_prev,
            max_t,
            layer_offset,
            nodes,
            miss,
            leaf,
            span,
            first_leaf,
            tri_pos,
        )
        end = max_t
        if found != 0:
            end = ti.min(t, max_t)
        sa, ss, _g = _pt_medium_coeff(tri_mat, f, _pt_medium_top(media))
        result *= ti.exp(-(sa + ss) * ti.max(end - t_prev, 0.0))
        if (found == 0) or (t >= max_t):
            finished = 1
            break
        count += 1
        duplicate = (edge != 0) and (t - seam <= depth_tie_epsilon)
        seam = t if edge != 0 else -1e30
        t_prev = t
        layer_prev = layer
        if not duplicate:
            if _pt_medium_id(shell, f, prim, offset) >= 0:
                entering = rd.dot(_pt_medium_normal(tri_pos, f, prim)) < 0.0
                previous = _pt_medium_top(media)
                ni = 1.0
                if previous >= 0:
                    ni = tri_mat[f % tri_mat.shape[0], previous, 12]
                media, failed = _pt_medium_cross(
                    media, shell, f, prim, offset, entering
                )
                overflow = ti.max(overflow, failed)
                following = _pt_medium_top(media)
                nt = 1.0
                if following >= 0:
                    nt = tri_mat[f % tri_mat.shape[0], following, 12]
                if ti.abs(ni / ti.max(nt, 1e-6) - 1.0) >= 1e-4:
                    result = ti.math.vec3(0.0, 0.0, 0.0)
                    finished = 1
                    break
    if (overflow != 0) or (finished == 0):
        result = ti.math.vec3(0.0, 0.0, 0.0)
    return result, overflow, 1 - finished
