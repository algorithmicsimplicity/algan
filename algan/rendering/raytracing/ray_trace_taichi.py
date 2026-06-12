"""Taichi kernel for ray tracing Algan scenes through spatio-temporal BVHs.

One GPU thread is launched per (frame, pixel). Each thread owns exactly one
cell of the output buffer ``[num_frames, num_pixels, channels]`` and performs
the whole render for its pixel -- visibility, alpha blending and mirror
bounces -- entirely in registers before writing the final color once. There
are no atomics and no intermediate fragment storage, so memory use is
independent of depth complexity and of the number of ray bounces.

Hits along a ray are processed strictly front-to-back by *depth peeling*: the
scene is repeatedly queried for the nearest hit beyond the previous one, and
each surface is alpha-composited in place (``acc += weight * a * color;
weight *= 1 - a``) until the remaining transmittance is negligible or the ray
escapes to the (pre-filled) background. Coplanar surfaces -- ubiquitous in 2D
scenes -- are ordered deterministically by a per-primitive layer index
(higher layer on top; triangles layer above bezier circuits).

When a surface has reflectivity ``r > 0`` the ray is mirror-reflected about
the (interpolated) surface normal and marching continues with throughput
``weight * a * r``, up to ``max_bounces`` reflections.

Geometry comes in two packed forms, each with its own STBVH (see
``stbvh.py``); leaves store primitive index + frame interval and geometry is
fetched at the ray's exact frame (frame index modulo each array's own time
length, so constant data can stay single-frame):

* triangles: ``tri_verts [Tv, N, 3, 8]`` (position, normal, reflectivity,
  roughness per corner) and ``tri_colors [Tc, N, 3, 5]`` (RGB, glow, alpha
  per corner);
* planar bezier circuits: ``circuit_meta [Tm, C, 20]`` (plane frame, border
  width, fill flag, texture grid transform), 2D polyline ``edges_2d`` with
  per-circuit ranges ``edge_offsets``, fill/texture colors
  ``circuit_colors [Tf, C, P, 5]`` (bilinearly sampled; P = 1 for plain
  fills) and ``circuit_border_colors [Tb, C, 5]``.
"""
import taichi as ti


def _ensure_taichi_initialized():
    """Initialize Taichi unless another module (e.g. the rasterizer) already
    has; re-initializing would invalidate previously compiled kernels.
    """
    initialized = False
    try:
        initialized = ti.lang.impl.get_runtime().prog is not None
    except Exception:
        initialized = False
    if not initialized:
        ti.init(arch=ti.gpu)


_ensure_taichi_initialized()

# Minimum hit distance along a ray (also the self-intersection guard for
# reflected rays, together with a normal offset at the bounce origin).
MIN_HIT_DISTANCE = 1e-4
# Hits closer together than this along a ray are considered coplanar and are
# ordered by layer index instead of by distance.
DEPTH_TIE_EPSILON = 1e-4
# Surfaces more transparent than this neither reflect nor terminate peeling.
MIN_ALPHA = 1e-3
# Marching stops once the remaining transmittance drops below this.
MIN_WEIGHT = 1e-3
# Hard cap on blended surfaces per ray, to bound worst-case stacked geometry.
MAX_SURFACES_PER_RAY = 256

# circuit_meta channel layout.
_M_CENTER = 0      # 0-2   plane origin
_M_NORMAL = 3      # 3-5   unit plane normal
_M_BASIS_U = 6     # 6-8   plane frame u axis (unit)
_M_BASIS_V = 9     # 9-11  plane frame v axis (unit)
_M_BORDER_W = 12   # border half-width in screen pixels
_M_FILLED = 13     # > 0.5 if the circuit interior is filled
_M_GRID_W = 14     # texture grid width  (1 for plain fills)
_M_GRID_H = 15     # texture grid height (1 for plain fills)
_M_TEX = 16        # 16-19 2x2 map from plane (u, v) to texture axes


@ti.func
def _safe_inverse(x: ti.f32) -> ti.f32:
    r = 1e12
    if x < 0.0:
        r = -1e12
    if ti.abs(x) > 1e-12:
        r = 1.0 / x
    return r


@ti.func
def _generate_ray(f, px, py, jitter_x, jitter_y, half_screen_w, half_screen_h,
                  cam_origin: ti.template(), screen_point: ti.template(),
                  pixel_basis_x: ti.template(), pixel_basis_y: ti.template()):
    """Build the world-space ray through a point inside a pixel
    (``jitter = 0.5`` is the pixel center; random jitter gives sub-pixel
    anti-aliasing when averaging many samples).

    Inverts the projection used by Algan's camera: a world point projects to
    screen coordinate ``u = dot(p - screen_point, b)`` which maps to pixel
    ``u * half_screen_h + half_screen`` -- so a pixel position corresponds to
    the world point ``screen_point + u * d_x + v * d_y`` where ``d_*`` is the
    reciprocal screen basis (precomputed into ``pixel_basis_*``).
    """
    x = ti.cast(px, ti.f32) + jitter_x
    y = ti.cast(py, ti.f32) + jitter_y
    u = (x - half_screen_w) / half_screen_h
    v = (y - half_screen_h) / half_screen_h
    ro = ti.math.vec3(cam_origin[f, 0], cam_origin[f, 1], cam_origin[f, 2])
    pix = ti.math.vec3(
        screen_point[f, 0] + u * pixel_basis_x[f, 0] + v * pixel_basis_y[f, 0],
        screen_point[f, 1] + u * pixel_basis_x[f, 1] + v * pixel_basis_y[f, 1],
        screen_point[f, 2] + u * pixel_basis_x[f, 2] + v * pixel_basis_y[f, 2],
    )
    rd = (pix - ro).normalized()
    return ro, rd


@ti.func
def _random_unit_vector():
    """Uniformly distributed point on the unit sphere."""
    z = 2.0 * ti.random(ti.f32) - 1.0
    phi = 6.283185307179586 * ti.random(ti.f32)
    r_xy = ti.sqrt(ti.max(0.0, 1.0 - z * z))
    return ti.math.vec3(r_xy * ti.cos(phi), r_xy * ti.sin(phi), z)


@ti.func
def _cosine_hemisphere_direction(normal):
    """Cosine-weighted random direction in the hemisphere around ``normal``
    (the Lambertian scattering distribution).
    """
    d = normal + _random_unit_vector()
    if d.norm() < 1e-6:
        d = normal
    return d.normalized()


@ti.func
def _node_intersected(node, f, ro, inv_rd, t_lo, t_hi,
                      node_lo: ti.template(), node_hi: ti.template(),
                      node_tmin: ti.template(), node_tmax: ti.template()) -> bool:
    """Spatio-temporal node test: frame containment + slab test restricted to
    the parametric window [t_lo, t_hi] of still-relevant hits.
    """
    hit = False
    if (node_tmin[node] <= f) and (f <= node_tmax[node]):
        tx0 = (node_lo[node, 0] - ro[0]) * inv_rd[0]
        tx1 = (node_hi[node, 0] - ro[0]) * inv_rd[0]
        t_near = ti.min(tx0, tx1)
        t_far = ti.max(tx0, tx1)
        ty0 = (node_lo[node, 1] - ro[1]) * inv_rd[1]
        ty1 = (node_hi[node, 1] - ro[1]) * inv_rd[1]
        t_near = ti.max(t_near, ti.min(ty0, ty1))
        t_far = ti.min(t_far, ti.max(ty0, ty1))
        tz0 = (node_lo[node, 2] - ro[2]) * inv_rd[2]
        tz1 = (node_hi[node, 2] - ro[2]) * inv_rd[2]
        t_near = ti.max(t_near, ti.min(tz0, tz1))
        t_far = ti.min(t_far, ti.max(tz0, tz1))
        hit = (t_far >= ti.max(t_near, 0.0)) and (t_near <= t_hi) and (t_far >= t_lo)
    return hit


@ti.func
def _comes_after(t, layer, t_prev, layer_prev) -> bool:
    """Strict ordering along the ray: by distance, with near-coplanar hits
    (within DEPTH_TIE_EPSILON) ordered by descending layer index.
    """
    return (t > t_prev + DEPTH_TIE_EPSILON) or (
        (t > t_prev - DEPTH_TIE_EPSILON) and (layer < layer_prev))


@ti.func
def _nearest_triangle_hit(ro, rd, inv_rd, f, t_prev, layer_prev, layer_offset,
                          node_lo: ti.template(), node_hi: ti.template(),
                          node_tmin: ti.template(), node_tmax: ti.template(),
                          node_miss: ti.template(), leaf_prim: ti.template(),
                          first_leaf, tri_verts: ti.template()):
    """Nearest triangle intersection strictly after (t_prev, layer_prev)."""
    best_t = 1e30
    best_layer = -1e30
    best_prim = -1
    best_w1 = 0.0
    best_w2 = 0.0
    num_vert_frames = tri_verts.shape[0]
    node = 0
    while node != -1:
        if _node_intersected(node, f, ro, inv_rd,
                             t_prev - DEPTH_TIE_EPSILON,
                             best_t + DEPTH_TIE_EPSILON,
                             node_lo, node_hi, node_tmin, node_tmax):
            if node >= first_leaf:
                prim = leaf_prim[node - first_leaf]
                if prim >= 0:
                    tv = f % num_vert_frames
                    v0 = ti.math.vec3(tri_verts[tv, prim, 0, 0],
                                      tri_verts[tv, prim, 0, 1],
                                      tri_verts[tv, prim, 0, 2])
                    v1 = ti.math.vec3(tri_verts[tv, prim, 1, 0],
                                      tri_verts[tv, prim, 1, 1],
                                      tri_verts[tv, prim, 1, 2])
                    v2 = ti.math.vec3(tri_verts[tv, prim, 2, 0],
                                      tri_verts[tv, prim, 2, 1],
                                      tri_verts[tv, prim, 2, 2])
                    e1 = v1 - v0
                    e2 = v2 - v0
                    pv = rd.cross(e2)
                    det = e1.dot(pv)
                    if ti.abs(det) > 1e-12:
                        inv_det = 1.0 / det
                        tvec = ro - v0
                        w1 = tvec.dot(pv) * inv_det
                        qv = tvec.cross(e1)
                        w2 = rd.dot(qv) * inv_det
                        if (w1 >= 0.0) and (w2 >= 0.0) and (w1 + w2 <= 1.0):
                            t = e2.dot(qv) * inv_det
                            layer = layer_offset + ti.cast(prim, ti.f32)
                            if ((t > MIN_HIT_DISTANCE)
                                    and _comes_after(t, layer, t_prev, layer_prev)
                                    and _comes_after(best_t, best_layer, t, layer)):
                                best_t = t
                                best_layer = layer
                                best_prim = prim
                                best_w1 = w1
                                best_w2 = w2
                node = node_miss[node]
            else:
                node = 2 * node + 1
        else:
            node = node_miss[node]
    return best_t, best_prim, best_w1, best_w2, best_layer


@ti.func
def _nearest_bezier_hit(ro, rd, inv_rd, f, t_prev, layer_prev,
                        pixel_size_per_t, base_dist,
                        node_lo: ti.template(), node_hi: ti.template(),
                        node_tmin: ti.template(), node_tmax: ti.template(),
                        node_miss: ti.template(), leaf_prim: ti.template(),
                        first_leaf, circuit_meta: ti.template(),
                        edges_2d: ti.template(), edge_offsets: ti.template()):
    """Nearest bezier-circuit intersection strictly after (t_prev, layer_prev).

    A circuit hit is the ray/plane intersection classified against the
    circuit's 2D polyline: inside the even-odd fill (when filled) or within
    the screen-constant border width of the curve. Returns the plane (u, v)
    coordinates for color sampling and whether the border color applies.
    """
    best_t = 1e30
    best_layer = -1e30
    best_circuit = -1
    best_border = 0
    best_u = 0.0
    best_v = 0.0
    num_meta_frames = circuit_meta.shape[0]
    num_edge_frames = edges_2d.shape[0]
    node = 0
    while node != -1:
        if _node_intersected(node, f, ro, inv_rd,
                             t_prev - DEPTH_TIE_EPSILON,
                             best_t + DEPTH_TIE_EPSILON,
                             node_lo, node_hi, node_tmin, node_tmax):
            if node >= first_leaf:
                circuit = leaf_prim[node - first_leaf]
                if circuit >= 0:
                    tm = f % num_meta_frames
                    n = ti.math.vec3(circuit_meta[tm, circuit, _M_NORMAL],
                                     circuit_meta[tm, circuit, _M_NORMAL + 1],
                                     circuit_meta[tm, circuit, _M_NORMAL + 2])
                    denom = rd.dot(n)
                    layer = ti.cast(circuit, ti.f32)
                    if ti.abs(denom) > 1e-9:
                        center = ti.math.vec3(
                            circuit_meta[tm, circuit, _M_CENTER],
                            circuit_meta[tm, circuit, _M_CENTER + 1],
                            circuit_meta[tm, circuit, _M_CENTER + 2])
                        t = (center - ro).dot(n) / denom
                        if ((t > MIN_HIT_DISTANCE)
                                and _comes_after(t, layer, t_prev, layer_prev)
                                and _comes_after(best_t, best_layer, t, layer)):
                            hit = ro + t * rd - center
                            bu = ti.math.vec3(
                                circuit_meta[tm, circuit, _M_BASIS_U],
                                circuit_meta[tm, circuit, _M_BASIS_U + 1],
                                circuit_meta[tm, circuit, _M_BASIS_U + 2])
                            bv = ti.math.vec3(
                                circuit_meta[tm, circuit, _M_BASIS_V],
                                circuit_meta[tm, circuit, _M_BASIS_V + 1],
                                circuit_meta[tm, circuit, _M_BASIS_V + 2])
                            u = hit.dot(bu)
                            v = hit.dot(bv)

                            te = f % num_edge_frames
                            crossings = 0
                            min_dist_sq = 1e30
                            for e in range(edge_offsets[circuit],
                                           edge_offsets[circuit + 1]):
                                x0 = edges_2d[te, e, 0]
                                y0 = edges_2d[te, e, 1]
                                x1 = edges_2d[te, e, 2]
                                y1 = edges_2d[te, e, 3]
                                if (y0 > v) != (y1 > v):
                                    x_cross = x0 + (v - y0) * (x1 - x0) / (y1 - y0)
                                    if x_cross > u:
                                        crossings += 1
                                dx = x1 - x0
                                dy = y1 - y0
                                seg_t = ((u - x0) * dx + (v - y0) * dy) / ti.max(
                                    dx * dx + dy * dy, 1e-12)
                                seg_t = ti.math.clamp(seg_t, 0.0, 1.0)
                                cx = x0 + seg_t * dx - u
                                cy = y0 + seg_t * dy - v
                                min_dist_sq = ti.min(min_dist_sq,
                                                     cx * cx + cy * cy)

                            # World size of one screen pixel at this hit, for
                            # screen-constant border/outline widths.
                            pixel_size = pixel_size_per_t * (base_dist + t)
                            border_w = (circuit_meta[tm, circuit, _M_BORDER_W]
                                        * pixel_size)
                            in_border = min_dist_sq < border_w * border_w
                            outline_w = 0.6 * pixel_size
                            inside = False
                            if circuit_meta[tm, circuit, _M_FILLED] > 0.5:
                                inside = ((crossings % 2) == 1) or (
                                    min_dist_sq < outline_w * outline_w)
                            if inside or in_border:
                                best_t = t
                                best_layer = layer
                                best_circuit = circuit
                                best_border = 1 if in_border else 0
                                best_u = u
                                best_v = v
                node = node_miss[node]
            else:
                node = 2 * node + 1
        else:
            node = node_miss[node]
    return best_t, best_circuit, best_border, best_u, best_v, best_layer


@ti.func
def _sample_circuit_color(circuit, f, u, v, in_border,
                          circuit_meta: ti.template(),
                          circuit_colors: ti.template(),
                          circuit_border_colors: ti.template()):
    """Color of a circuit at plane coordinates (u, v): the border color, or
    the fill color bilinearly sampled from the circuit's texture grid (plain
    fills are 1x1 grids, which degenerate to the single fill color).
    """
    color = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
    alpha = 0.0
    if in_border == 1:
        tb = f % circuit_border_colors.shape[0]
        color = ti.math.vec4(circuit_border_colors[tb, circuit, 0],
                             circuit_border_colors[tb, circuit, 1],
                             circuit_border_colors[tb, circuit, 2],
                             circuit_border_colors[tb, circuit, 3])
        alpha = circuit_border_colors[tb, circuit, 4]
    else:
        tm = f % circuit_meta.shape[0]
        tc = f % circuit_colors.shape[0]
        grid_w = circuit_meta[tm, circuit, _M_GRID_W]
        grid_h = circuit_meta[tm, circuit, _M_GRID_H]
        # Map plane (u, v) to texture-grid coordinates via the precomputed
        # 2x2 transform (mirrors the rasterizer's texture lookup).
        c1 = 0.5 * (u * circuit_meta[tm, circuit, _M_TEX]
                    + v * circuit_meta[tm, circuit, _M_TEX + 1]) + 0.5
        c2 = 0.5 * (u * circuit_meta[tm, circuit, _M_TEX + 2]
                    + v * circuit_meta[tm, circuit, _M_TEX + 3]) + 0.5
        x = ti.math.clamp(c2 * grid_h, 0.0, ti.max(grid_h - 1.0, 0.0))
        y = ti.math.clamp(c1 * grid_w, 0.0, ti.max(grid_w - 1.0, 0.0))
        num_points = circuit_colors.shape[2]
        x_floor = ti.floor(x)
        y_floor = ti.floor(y)
        xr = x - x_floor
        yr = y - y_floor
        sum_w = 0.0
        for corner in ti.static(range(4)):
            cx = x_floor + (corner % 2)
            cy = y_floor + (corner // 2)
            w = (xr if (corner % 2) == 1 else 1.0 - xr) * (
                yr if (corner // 2) == 1 else 1.0 - yr)
            p = ti.cast(cx + cy * grid_h, ti.i32)
            p = ti.math.clamp(p, 0, num_points - 1)
            color += w * ti.math.vec4(circuit_colors[tc, circuit, p, 0],
                                      circuit_colors[tc, circuit, p, 1],
                                      circuit_colors[tc, circuit, p, 2],
                                      circuit_colors[tc, circuit, p, 3])
            alpha += w * circuit_colors[tc, circuit, p, 4]
            sum_w += w
        color /= ti.max(sum_w, 1e-6)
        alpha /= ti.max(sum_w, 1e-6)
    return color, alpha


@ti.func
def _nearest_surface(ro, rd, inv_rd, f, t_prev, layer_prev,
                     pixel_size_per_t, base_dist, layer_offset_triangles,
                     t_node_lo: ti.template(), t_node_hi: ti.template(),
                     t_node_tmin: ti.template(), t_node_tmax: ti.template(),
                     t_node_miss: ti.template(), t_leaf_prim: ti.template(),
                     t_first_leaf,
                     tri_verts: ti.template(), tri_colors: ti.template(),
                     b_node_lo: ti.template(), b_node_hi: ti.template(),
                     b_node_tmin: ti.template(), b_node_tmax: ti.template(),
                     b_node_miss: ti.template(), b_leaf_prim: ti.template(),
                     b_first_leaf,
                     circuit_meta: ti.template(), circuit_colors: ti.template(),
                     circuit_border_colors: ti.template(),
                     edges_2d: ti.template(), edge_offsets: ti.template()):
    """Nearest surface of either geometry type strictly after
    (t_prev, layer_prev) along the ray, with its shading properties fetched.

    Returns (found, t_hit, layer, color[rgb+glow], alpha, reflectivity,
    roughness, normal); ``found == 0`` means the ray escapes the scene.
    """
    found = 0
    t_hit = 1e30
    hit_layer = -1e30
    color = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
    alpha = 0.0
    reflectivity = 0.0
    roughness = 0.0
    normal = ti.math.vec3(0.0, 0.0, 0.0)

    tt, t_prim, w1, w2, t_layer = _nearest_triangle_hit(
        ro, rd, inv_rd, f, t_prev, layer_prev, layer_offset_triangles,
        t_node_lo, t_node_hi, t_node_tmin, t_node_tmax,
        t_node_miss, t_leaf_prim, t_first_leaf, tri_verts)
    bt, b_circ, b_border, b_u, b_v, b_layer = _nearest_bezier_hit(
        ro, rd, inv_rd, f, t_prev, layer_prev, pixel_size_per_t, base_dist,
        b_node_lo, b_node_hi, b_node_tmin, b_node_tmax,
        b_node_miss, b_leaf_prim, b_first_leaf,
        circuit_meta, edges_2d, edge_offsets)

    if (t_prim >= 0) or (b_circ >= 0):
        found = 1
        use_triangle = (t_prim >= 0) and (
            (b_circ < 0) or (not _comes_after(tt, t_layer, bt, b_layer)))
        if use_triangle:
            t_hit = tt
            hit_layer = t_layer
            w0 = 1.0 - w1 - w2
            tc = f % tri_colors.shape[0]
            tv = f % tri_verts.shape[0]
            for ci in ti.static(range(4)):
                color[ci] = (w0 * tri_colors[tc, t_prim, 0, ci]
                             + w1 * tri_colors[tc, t_prim, 1, ci]
                             + w2 * tri_colors[tc, t_prim, 2, ci])
            alpha = (w0 * tri_colors[tc, t_prim, 0, 4]
                     + w1 * tri_colors[tc, t_prim, 1, 4]
                     + w2 * tri_colors[tc, t_prim, 2, 4])
            reflectivity = (w0 * tri_verts[tv, t_prim, 0, 6]
                            + w1 * tri_verts[tv, t_prim, 1, 6]
                            + w2 * tri_verts[tv, t_prim, 2, 6])
            roughness = (w0 * tri_verts[tv, t_prim, 0, 7]
                         + w1 * tri_verts[tv, t_prim, 1, 7]
                         + w2 * tri_verts[tv, t_prim, 2, 7])
            for ci in ti.static(range(3)):
                normal[ci] = (w0 * tri_verts[tv, t_prim, 0, 3 + ci]
                              + w1 * tri_verts[tv, t_prim, 1, 3 + ci]
                              + w2 * tri_verts[tv, t_prim, 2, 3 + ci])
            if normal.norm() < 1e-6:
                v0 = ti.math.vec3(tri_verts[tv, t_prim, 0, 0],
                                  tri_verts[tv, t_prim, 0, 1],
                                  tri_verts[tv, t_prim, 0, 2])
                v1 = ti.math.vec3(tri_verts[tv, t_prim, 1, 0],
                                  tri_verts[tv, t_prim, 1, 1],
                                  tri_verts[tv, t_prim, 1, 2])
                v2 = ti.math.vec3(tri_verts[tv, t_prim, 2, 0],
                                  tri_verts[tv, t_prim, 2, 1],
                                  tri_verts[tv, t_prim, 2, 2])
                normal = (v1 - v0).cross(v2 - v0)
        else:
            t_hit = bt
            hit_layer = b_layer
            color, alpha = _sample_circuit_color(
                b_circ, f, b_u, b_v, b_border,
                circuit_meta, circuit_colors, circuit_border_colors)
            tm = f % circuit_meta.shape[0]
            normal = ti.math.vec3(circuit_meta[tm, b_circ, _M_NORMAL],
                                  circuit_meta[tm, b_circ, _M_NORMAL + 1],
                                  circuit_meta[tm, b_circ, _M_NORMAL + 2])
    return found, t_hit, hit_layer, color, alpha, reflectivity, roughness, normal


@ti.kernel
def render_scene_stbvh(
        # Triangle STBVH + packed geometry.
        t_node_lo: ti.types.ndarray(), t_node_hi: ti.types.ndarray(),
        t_node_tmin: ti.types.ndarray(), t_node_tmax: ti.types.ndarray(),
        t_node_miss: ti.types.ndarray(), t_leaf_prim: ti.types.ndarray(),
        t_first_leaf: int,
        tri_verts: ti.types.ndarray(), tri_colors: ti.types.ndarray(),
        # Bezier STBVH + packed geometry.
        b_node_lo: ti.types.ndarray(), b_node_hi: ti.types.ndarray(),
        b_node_tmin: ti.types.ndarray(), b_node_tmax: ti.types.ndarray(),
        b_node_miss: ti.types.ndarray(), b_leaf_prim: ti.types.ndarray(),
        b_first_leaf: int,
        circuit_meta: ti.types.ndarray(), circuit_colors: ti.types.ndarray(),
        circuit_border_colors: ti.types.ndarray(),
        edges_2d: ti.types.ndarray(), edge_offsets: ti.types.ndarray(),
        # Per-frame camera and pixel scale.
        cam_origin: ti.types.ndarray(), screen_point: ti.types.ndarray(),
        pixel_basis_x: ti.types.ndarray(), pixel_basis_y: ti.types.ndarray(),
        pixel_world_scale: ti.types.ndarray(),
        # Render parameters.
        time_start: int, time_end: int, width: int, height: int,
        half_screen_w: float, half_screen_h: float,
        layer_offset_triangles: float, max_bounces: int, transparent: int,
        # Output buffer [time_end - time_start, width * height, channels],
        # pre-filled with the background; blended in place.
        out: ti.types.ndarray()):
    num_color_frames = tri_colors.shape[0]
    num_vert_frames = tri_verts.shape[0]
    pixels_per_frame = width * height
    num_rays = (time_end - time_start) * pixels_per_frame

    for ray_id in range(num_rays):
        f_rel = ray_id // pixels_per_frame
        p = ray_id - f_rel * pixels_per_frame
        f = time_start + f_rel
        py = p // width
        px = p - py * width

        ro, rd = _generate_ray(f, px, py, 0.5, 0.5,
                               half_screen_w, half_screen_h,
                               cam_origin, screen_point,
                               pixel_basis_x, pixel_basis_y)
        inv_rd = ti.math.vec3(_safe_inverse(rd[0]), _safe_inverse(rd[1]),
                              _safe_inverse(rd[2]))
        pixel_size_per_t = pixel_world_scale[f]

        acc = ti.math.vec4(0.0, 0.0, 0.0, 0.0)  # premultiplied RGB + glow
        weight = 1.0       # remaining transmittance * reflection throughput
        t_prev = 0.0
        layer_prev = 1e30  # accept any first hit
        base_dist = 0.0    # distance accumulated over previous bounces
        bounces_left = max_bounces

        step = 0
        while step < MAX_SURFACES_PER_RAY:
            step += 1
            tt, t_prim, w1, w2, t_layer = _nearest_triangle_hit(
                ro, rd, inv_rd, f, t_prev, layer_prev, layer_offset_triangles,
                t_node_lo, t_node_hi, t_node_tmin, t_node_tmax,
                t_node_miss, t_leaf_prim, t_first_leaf, tri_verts)
            bt, b_circ, b_border, b_u, b_v, b_layer = _nearest_bezier_hit(
                ro, rd, inv_rd, f, t_prev, layer_prev,
                pixel_size_per_t, base_dist,
                b_node_lo, b_node_hi, b_node_tmin, b_node_tmax,
                b_node_miss, b_leaf_prim, b_first_leaf,
                circuit_meta, edges_2d, edge_offsets)

            use_triangle = (t_prim >= 0) and (
                (b_circ < 0) or (not _comes_after(tt, t_layer, bt, b_layer)))
            if (t_prim < 0) and (b_circ < 0):
                break

            color = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
            alpha = 0.0
            reflectivity = 0.0
            t_hit = 0.0
            hit_layer = 0.0
            normal = ti.math.vec3(0.0, 0.0, 0.0)
            if use_triangle:
                t_hit = tt
                hit_layer = t_layer
                w0 = 1.0 - w1 - w2
                tc = f % num_color_frames
                tv = f % num_vert_frames
                for ci in ti.static(range(4)):
                    color[ci] = (w0 * tri_colors[tc, t_prim, 0, ci]
                                 + w1 * tri_colors[tc, t_prim, 1, ci]
                                 + w2 * tri_colors[tc, t_prim, 2, ci])
                alpha = (w0 * tri_colors[tc, t_prim, 0, 4]
                         + w1 * tri_colors[tc, t_prim, 1, 4]
                         + w2 * tri_colors[tc, t_prim, 2, 4])
                reflectivity = (w0 * tri_verts[tv, t_prim, 0, 6]
                                + w1 * tri_verts[tv, t_prim, 1, 6]
                                + w2 * tri_verts[tv, t_prim, 2, 6])
                if reflectivity > MIN_ALPHA:
                    # Interpolated shading normal, falling back to the
                    # geometric normal when vertex normals are absent.
                    for ci in ti.static(range(3)):
                        normal[ci] = (
                            w0 * tri_verts[tv, t_prim, 0, 3 + ci]
                            + w1 * tri_verts[tv, t_prim, 1, 3 + ci]
                            + w2 * tri_verts[tv, t_prim, 2, 3 + ci])
                    if normal.norm() < 1e-6:
                        v0 = ti.math.vec3(tri_verts[tv, t_prim, 0, 0],
                                          tri_verts[tv, t_prim, 0, 1],
                                          tri_verts[tv, t_prim, 0, 2])
                        v1 = ti.math.vec3(tri_verts[tv, t_prim, 1, 0],
                                          tri_verts[tv, t_prim, 1, 1],
                                          tri_verts[tv, t_prim, 1, 2])
                        v2 = ti.math.vec3(tri_verts[tv, t_prim, 2, 0],
                                          tri_verts[tv, t_prim, 2, 1],
                                          tri_verts[tv, t_prim, 2, 2])
                        normal = (v1 - v0).cross(v2 - v0)
            else:
                t_hit = bt
                hit_layer = b_layer
                color, alpha = _sample_circuit_color(
                    b_circ, f, b_u, b_v, b_border,
                    circuit_meta, circuit_colors, circuit_border_colors)

            alpha = ti.math.clamp(alpha, 0.0, 1.0)
            reflectivity = ti.math.clamp(reflectivity, 0.0, 1.0)
            if bounces_left <= 0:
                reflectivity = 0.0

            acc += (weight * alpha * (1.0 - reflectivity)) * color

            if (reflectivity > MIN_ALPHA) and (alpha > MIN_ALPHA):
                # Mirror bounce: reflect about the face-forward normal and
                # restart peeling along the new ray. The transmitted
                # remainder of a semi-transparent mirror is dropped.
                normal = normal.normalized()
                if normal.dot(rd) > 0.0:
                    normal = -normal
                hit_point = ro + t_hit * rd
                rd = (rd - 2.0 * rd.dot(normal) * normal).normalized()
                ro = hit_point + normal * (10.0 * MIN_HIT_DISTANCE)
                inv_rd = ti.math.vec3(_safe_inverse(rd[0]),
                                      _safe_inverse(rd[1]),
                                      _safe_inverse(rd[2]))
                weight *= alpha * reflectivity
                base_dist += t_hit
                t_prev = 0.0
                layer_prev = 1e30
                bounces_left -= 1
            else:
                weight *= 1.0 - alpha
                t_prev = t_hit
                layer_prev = hit_layer
            if weight < MIN_WEIGHT:
                break

        # Composite over the pre-filled background and write the pixel.
        for ci in ti.static(range(4)):
            bg = ti.cast(out[f_rel, p, ci], ti.f32)
            val = acc[ci] * 255.0 + weight * bg
            out[f_rel, p, ci] = ti.cast(ti.math.clamp(val + 0.5, 0.0, 255.0),
                                        ti.u8)
        if transparent != 0:
            bg_a = ti.cast(out[f_rel, p, 4], ti.f32)
            val = (1.0 - weight) * 255.0 + weight * bg_a
            out[f_rel, p, 4] = ti.cast(ti.math.clamp(val + 0.5, 0.0, 255.0),
                                       ti.u8)


@ti.kernel
def path_trace_scene_stbvh(
        # Triangle STBVH + packed geometry.
        t_node_lo: ti.types.ndarray(), t_node_hi: ti.types.ndarray(),
        t_node_tmin: ti.types.ndarray(), t_node_tmax: ti.types.ndarray(),
        t_node_miss: ti.types.ndarray(), t_leaf_prim: ti.types.ndarray(),
        t_first_leaf: int,
        tri_verts: ti.types.ndarray(), tri_colors: ti.types.ndarray(),
        # Bezier STBVH + packed geometry.
        b_node_lo: ti.types.ndarray(), b_node_hi: ti.types.ndarray(),
        b_node_tmin: ti.types.ndarray(), b_node_tmax: ti.types.ndarray(),
        b_node_miss: ti.types.ndarray(), b_leaf_prim: ti.types.ndarray(),
        b_first_leaf: int,
        circuit_meta: ti.types.ndarray(), circuit_colors: ti.types.ndarray(),
        circuit_border_colors: ti.types.ndarray(),
        edges_2d: ti.types.ndarray(), edge_offsets: ti.types.ndarray(),
        # Per-frame camera and pixel scale.
        cam_origin: ti.types.ndarray(), screen_point: ti.types.ndarray(),
        pixel_basis_x: ti.types.ndarray(), pixel_basis_y: ti.types.ndarray(),
        pixel_world_scale: ti.types.ndarray(),
        # Render parameters.
        time_start: int, time_end: int, width: int, height: int,
        half_screen_w: float, half_screen_h: float,
        layer_offset_triangles: float, max_bounces: int, transparent: int,
        samples_per_pixel: int, indirect_strength: float,
        # Output buffer [time_end - time_start, width * height, channels],
        # pre-filled with the background.
        out: ti.types.ndarray()):
    """Monte Carlo estimator of the same light transport as
    ``render_scene_stbvh``, generalized with random scattering.

    Each thread averages ``samples_per_pixel`` independent paths through a
    random sub-pixel position. At every surface a path makes stochastic
    decisions instead of deterministic splits:

    * with probability ``1 - alpha`` it passes straight through (stochastic
      transparency -- the expectation equals alpha blending);
    * otherwise, with probability ``reflectivity`` it reflects specularly,
      jittered into a glossy lobe of the surface's ``roughness``;
    * otherwise it is a diffuse interaction: the surface's (vertex-shaded)
      color is emitted into the sample and, when ``indirect_strength > 0``,
      the path continues in a cosine-weighted random hemisphere direction
      with throughput scaled by ``albedo * indirect_strength`` (color
      bleeding / one-bounce-per-step global illumination).

    Paths that escape the scene pick up the background; the sample mean is
    written over the pre-filled background buffer.
    """
    pixels_per_frame = width * height
    num_rays = (time_end - time_start) * pixels_per_frame

    for ray_id in range(num_rays):
        f_rel = ray_id // pixels_per_frame
        p = ray_id - f_rel * pixels_per_frame
        f = time_start + f_rel
        py = p // width
        px = p - py * width
        pixel_size_per_t = pixel_world_scale[f]

        background = ti.math.vec4(ti.cast(out[f_rel, p, 0], ti.f32),
                                  ti.cast(out[f_rel, p, 1], ti.f32),
                                  ti.cast(out[f_rel, p, 2], ti.f32),
                                  ti.cast(out[f_rel, p, 3], ti.f32)) / 255.0
        background_alpha = 1.0
        if transparent != 0:
            background_alpha = ti.cast(out[f_rel, p, 4], ti.f32) / 255.0

        acc = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
        acc_alpha = 0.0
        for _sample in range(samples_per_pixel):
            ro, rd = _generate_ray(f, px, py, ti.random(ti.f32),
                                   ti.random(ti.f32),
                                   half_screen_w, half_screen_h,
                                   cam_origin, screen_point,
                                   pixel_basis_x, pixel_basis_y)
            inv_rd = ti.math.vec3(_safe_inverse(rd[0]), _safe_inverse(rd[1]),
                                  _safe_inverse(rd[2]))
            throughput = ti.math.vec4(1.0, 1.0, 1.0, 1.0)
            t_prev = 0.0
            layer_prev = 1e30
            base_dist = 0.0
            bounces_left = max_bounces
            interacted = False
            escaped = False

            step = 0
            while step < MAX_SURFACES_PER_RAY:
                step += 1
                (found, t_hit, hit_layer, color, alpha, reflectivity,
                 roughness, normal) = _nearest_surface(
                    ro, rd, inv_rd, f, t_prev, layer_prev,
                    pixel_size_per_t, base_dist, layer_offset_triangles,
                    t_node_lo, t_node_hi, t_node_tmin, t_node_tmax,
                    t_node_miss, t_leaf_prim, t_first_leaf,
                    tri_verts, tri_colors,
                    b_node_lo, b_node_hi, b_node_tmin, b_node_tmax,
                    b_node_miss, b_leaf_prim, b_first_leaf,
                    circuit_meta, circuit_colors, circuit_border_colors,
                    edges_2d, edge_offsets)
                if found == 0:
                    escaped = True
                    break

                alpha = ti.math.clamp(alpha, 0.0, 1.0)
                if ti.random(ti.f32) >= alpha:
                    # Pass straight through the (partially) transparent
                    # surface; advance the peel state along the same ray.
                    t_prev = t_hit
                    layer_prev = hit_layer
                    continue
                interacted = True

                reflectivity = ti.math.clamp(reflectivity, 0.0, 1.0)
                if bounces_left <= 0:
                    reflectivity = 0.0

                if normal.norm() > 1e-9:
                    normal = normal.normalized()
                if normal.dot(rd) > 0.0:
                    normal = -normal
                hit_point = ro + t_hit * rd

                if ti.random(ti.f32) < reflectivity:
                    # Specular bounce, jittered into a glossy lobe.
                    rd_new = (rd - 2.0 * rd.dot(normal) * normal).normalized()
                    if roughness > 1e-4:
                        rd_new = (rd_new + roughness
                                  * _random_unit_vector()).normalized()
                        if rd_new.dot(normal) < 0.0:
                            rd_new = rd_new - 2.0 * rd_new.dot(normal) * normal
                    rd = rd_new
                else:
                    # Diffuse interaction: emit the surface's color.
                    acc += throughput * color
                    if (indirect_strength <= 0.0) or (bounces_left <= 0):
                        break  # absorbed
                    albedo_mean = (color[0] + color[1] + color[2]) / 3.0
                    throughput *= ti.math.vec4(
                        color[0], color[1], color[2], albedo_mean
                    ) * indirect_strength
                    if (ti.max(throughput[0],
                               ti.max(throughput[1], throughput[2]))
                            < MIN_WEIGHT):
                        break  # absorbed
                    rd = _cosine_hemisphere_direction(normal)

                ro = hit_point + normal * (10.0 * MIN_HIT_DISTANCE)
                inv_rd = ti.math.vec3(_safe_inverse(rd[0]),
                                      _safe_inverse(rd[1]),
                                      _safe_inverse(rd[2]))
                base_dist += t_hit
                t_prev = 0.0
                layer_prev = 1e30
                bounces_left -= 1

            if escaped:
                acc += throughput * background
                acc_alpha += 1.0 if interacted else background_alpha
            else:
                acc_alpha += 1.0

        inv_spp = 1.0 / ti.cast(samples_per_pixel, ti.f32)
        for ci in ti.static(range(4)):
            val = acc[ci] * inv_spp * 255.0
            out[f_rel, p, ci] = ti.cast(ti.math.clamp(val + 0.5, 0.0, 255.0),
                                        ti.u8)
        if transparent != 0:
            val = acc_alpha * inv_spp * 255.0
            out[f_rel, p, 4] = ti.cast(ti.math.clamp(val + 0.5, 0.0, 255.0),
                                       ti.u8)


@ti.func
def _transmittance(ro, rd, f, max_t,
                   pixel_size_per_t, base_dist, layer_offset_triangles,
                   t_node_lo: ti.template(), t_node_hi: ti.template(),
                   t_node_tmin: ti.template(), t_node_tmax: ti.template(),
                   t_node_miss: ti.template(), t_leaf_prim: ti.template(),
                   t_first_leaf,
                   tri_verts: ti.template(), tri_colors: ti.template(),
                   b_node_lo: ti.template(), b_node_hi: ti.template(),
                   b_node_tmin: ti.template(), b_node_tmax: ti.template(),
                   b_node_miss: ti.template(), b_leaf_prim: ti.template(),
                   b_first_leaf,
                   circuit_meta: ti.template(), circuit_colors: ti.template(),
                   circuit_border_colors: ti.template(),
                   edges_2d: ti.template(), edge_offsets: ti.template()):
    """Fraction of light transmitted along a shadow ray of length ``max_t``:
    every surface crossed attenuates by its transparency ``1 - alpha``.
    """
    inv_rd = ti.math.vec3(_safe_inverse(rd[0]), _safe_inverse(rd[1]),
                          _safe_inverse(rd[2]))
    transmitted = 1.0
    t_prev = 0.0
    layer_prev = 1e30
    step = 0
    while step < MAX_SURFACES_PER_RAY:
        step += 1
        (found, t_hit, hit_layer, _color, alpha, _refl, _rough,
         _normal) = _nearest_surface(
            ro, rd, inv_rd, f, t_prev, layer_prev,
            pixel_size_per_t, base_dist, layer_offset_triangles,
            t_node_lo, t_node_hi, t_node_tmin, t_node_tmax,
            t_node_miss, t_leaf_prim, t_first_leaf,
            tri_verts, tri_colors,
            b_node_lo, b_node_hi, b_node_tmin, b_node_tmax,
            b_node_miss, b_leaf_prim, b_first_leaf,
            circuit_meta, circuit_colors, circuit_border_colors,
            edges_2d, edge_offsets)
        if (found == 0) or (t_hit >= max_t):
            break
        transmitted *= 1.0 - ti.math.clamp(alpha, 0.0, 1.0)
        if transmitted < 1e-3:
            transmitted = 0.0
            break
        t_prev = t_hit
        layer_prev = hit_layer
    return transmitted


@ti.kernel
def path_trace_physical_stbvh(
        # Triangle STBVH + packed geometry.
        t_node_lo: ti.types.ndarray(), t_node_hi: ti.types.ndarray(),
        t_node_tmin: ti.types.ndarray(), t_node_tmax: ti.types.ndarray(),
        t_node_miss: ti.types.ndarray(), t_leaf_prim: ti.types.ndarray(),
        t_first_leaf: int,
        tri_verts: ti.types.ndarray(), tri_colors: ti.types.ndarray(),
        # Bezier STBVH + packed geometry.
        b_node_lo: ti.types.ndarray(), b_node_hi: ti.types.ndarray(),
        b_node_tmin: ti.types.ndarray(), b_node_tmax: ti.types.ndarray(),
        b_node_miss: ti.types.ndarray(), b_leaf_prim: ti.types.ndarray(),
        b_first_leaf: int,
        circuit_meta: ti.types.ndarray(), circuit_colors: ti.types.ndarray(),
        circuit_border_colors: ti.types.ndarray(),
        edges_2d: ti.types.ndarray(), edge_offsets: ti.types.ndarray(),
        # Per-frame camera and pixel scale.
        cam_origin: ti.types.ndarray(), screen_point: ti.types.ndarray(),
        pixel_basis_x: ti.types.ndarray(), pixel_basis_y: ti.types.ndarray(),
        pixel_world_scale: ti.types.ndarray(),
        # Render parameters.
        time_start: int, time_end: int, width: int, height: int,
        half_screen_w: float, half_screen_h: float,
        layer_offset_triangles: float, max_bounces: int, transparent: int,
        samples_per_pixel: int,
        # Explicit point lights [Tl, L, 3] and lighting controls.
        light_pos: ti.types.ndarray(), light_col: ti.types.ndarray(),
        num_lights: int, light_intensity: float, ambient: float,
        # Output buffer pre-filled with the background (the environment).
        out: ti.types.ndarray()):
    """Physically based Monte Carlo path tracer with explicit lights.

    Vertex colors are treated as raw *albedo* (vertex shading is skipped in
    this mode) and all illumination is computed by the integrator:

    * **Point lights** are sampled explicitly at every interaction
      (next-event estimation): a shadow ray accumulates the transmittance
      through partially transparent occluders, and the surface responds with
      a Lambertian diffuse lobe plus a Fresnel-weighted glossy specular lobe
      (Schlick ``F0 = lerp(0.04, albedo, metallic)``, normalized
      Blinn-Phong with exponent derived from ``roughness``). ``reflectivity``
      doubles as the metallicness.
    * **Emissive surfaces**: the ``glow`` channel emits
      ``albedo * glow`` radiance, picked up by paths that hit the surface
      (point lights are never hit by chance, so nothing is double counted).
    * **The background acts as the environment light** for escaping paths,
      and ``ambient`` adds a constant ambient term per diffuse interaction.
    * Continuation rays importance-sample the BRDF: a Fresnel-proportional
      coin chooses the specular lobe (mirror direction jittered by
      ``roughness``) or the cosine-weighted diffuse lobe, with throughput
      weights that keep the estimator unbiased. Transparency passes rays
      straight through with probability ``1 - alpha``.
    """
    pixels_per_frame = width * height
    num_rays = (time_end - time_start) * pixels_per_frame
    num_light_frames = ti.max(light_pos.shape[0], 1)

    for ray_id in range(num_rays):
        f_rel = ray_id // pixels_per_frame
        p = ray_id - f_rel * pixels_per_frame
        f = time_start + f_rel
        py = p // width
        px = p - py * width
        pixel_size_per_t = pixel_world_scale[f]

        background = ti.math.vec3(ti.cast(out[f_rel, p, 0], ti.f32),
                                  ti.cast(out[f_rel, p, 1], ti.f32),
                                  ti.cast(out[f_rel, p, 2], ti.f32)) / 255.0
        background_alpha = 1.0
        if transparent != 0:
            background_alpha = ti.cast(out[f_rel, p, 4], ti.f32) / 255.0

        acc = ti.math.vec4(0.0, 0.0, 0.0, 0.0)  # radiance rgb + bloom glow
        acc_alpha = 0.0
        for _sample in range(samples_per_pixel):
            ro, rd = _generate_ray(f, px, py, ti.random(ti.f32),
                                   ti.random(ti.f32),
                                   half_screen_w, half_screen_h,
                                   cam_origin, screen_point,
                                   pixel_basis_x, pixel_basis_y)
            inv_rd = ti.math.vec3(_safe_inverse(rd[0]), _safe_inverse(rd[1]),
                                  _safe_inverse(rd[2]))
            throughput = ti.math.vec3(1.0, 1.0, 1.0)
            t_prev = 0.0
            layer_prev = 1e30
            base_dist = 0.0
            bounces_left = max_bounces
            interacted = False
            escaped = False

            step = 0
            while step < MAX_SURFACES_PER_RAY:
                step += 1
                (found, t_hit, hit_layer, color, alpha, reflectivity,
                 roughness, normal) = _nearest_surface(
                    ro, rd, inv_rd, f, t_prev, layer_prev,
                    pixel_size_per_t, base_dist, layer_offset_triangles,
                    t_node_lo, t_node_hi, t_node_tmin, t_node_tmax,
                    t_node_miss, t_leaf_prim, t_first_leaf,
                    tri_verts, tri_colors,
                    b_node_lo, b_node_hi, b_node_tmin, b_node_tmax,
                    b_node_miss, b_leaf_prim, b_first_leaf,
                    circuit_meta, circuit_colors, circuit_border_colors,
                    edges_2d, edge_offsets)
                if found == 0:
                    escaped = True
                    break

                alpha = ti.math.clamp(alpha, 0.0, 1.0)
                if ti.random(ti.f32) >= alpha:
                    t_prev = t_hit
                    layer_prev = hit_layer
                    continue
                interacted = True

                albedo = ti.math.vec3(color[0], color[1], color[2])
                glow = ti.max(color[3], 0.0)
                metallic = ti.math.clamp(reflectivity, 0.0, 1.0)
                if normal.norm() > 1e-9:
                    normal = normal.normalized()
                if normal.dot(rd) > 0.0:
                    normal = -normal
                hit_point = ro + t_hit * rd
                shadow_origin = hit_point + normal * (10.0 * MIN_HIT_DISTANCE)

                f0 = ti.math.vec3(0.04, 0.04, 0.04) * (1.0 - metallic) \
                    + albedo * metallic
                cos_view = ti.max(normal.dot(-rd), 0.0)
                fresnel = f0 + (ti.math.vec3(1.0, 1.0, 1.0) - f0) \
                    * ti.pow(1.0 - cos_view, 5.0)

                # Emission (glow) and constant ambient.
                acc += ti.math.vec4(
                    throughput[0] * albedo[0] * glow,
                    throughput[1] * albedo[1] * glow,
                    throughput[2] * albedo[2] * glow,
                    (throughput[0] + throughput[1] + throughput[2])
                    / 3.0 * glow)
                if ambient > 0.0:
                    amb = ambient * (1.0 - metallic)
                    acc += ti.math.vec4(throughput[0] * albedo[0] * amb,
                                        throughput[1] * albedo[1] * amb,
                                        throughput[2] * albedo[2] * amb, 0.0)

                # Next-event estimation: sample every point light.
                phong_n = ti.math.clamp(
                    2.0 / ti.max(roughness * roughness, 5e-4) - 2.0,
                    1.0, 4096.0)
                tl = f % num_light_frames
                for li in range(num_lights):
                    lp = ti.math.vec3(light_pos[tl, li, 0],
                                      light_pos[tl, li, 1],
                                      light_pos[tl, li, 2])
                    to_light = lp - hit_point
                    light_dist = to_light.norm()
                    if light_dist > 1e-5:
                        wi = to_light / light_dist
                        cos_i = normal.dot(wi)
                        if cos_i > 1e-4:
                            visible = _transmittance(
                                shadow_origin, wi, f,
                                light_dist - 20.0 * MIN_HIT_DISTANCE,
                                pixel_size_per_t, base_dist,
                                layer_offset_triangles,
                                t_node_lo, t_node_hi, t_node_tmin,
                                t_node_tmax, t_node_miss, t_leaf_prim,
                                t_first_leaf, tri_verts, tri_colors,
                                b_node_lo, b_node_hi, b_node_tmin,
                                b_node_tmax, b_node_miss, b_leaf_prim,
                                b_first_leaf, circuit_meta, circuit_colors,
                                circuit_border_colors, edges_2d,
                                edge_offsets)
                            if visible > 1e-4:
                                half_v = (wi - rd).normalized()
                                cos_h = ti.max(normal.dot(half_v), 0.0)
                                spec = fresnel * ((phong_n + 2.0)
                                                  / 6.283185307179586
                                                  * ti.pow(cos_h, phong_n))
                                diff = albedo * ((1.0 - metallic)
                                                 / 3.141592653589793)
                                radiance = ti.math.vec3(
                                    light_col[tl, li, 0],
                                    light_col[tl, li, 1],
                                    light_col[tl, li, 2]) * light_intensity
                                lit = throughput * (diff + spec) \
                                    * (cos_i * visible) * radiance
                                acc += ti.math.vec4(lit[0], lit[1], lit[2],
                                                    0.0)

                # Importance-sample the BRDF for the continuation ray.
                if bounces_left <= 0:
                    break  # absorbed
                spec_prob = ti.math.clamp(
                    (fresnel[0] + fresnel[1] + fresnel[2]) / 3.0, 0.0, 0.95)
                if metallic < 1e-3:
                    spec_prob = 0.0  # skip glints on plain dielectrics
                if ti.random(ti.f32) < spec_prob:
                    rd_new = (rd - 2.0 * rd.dot(normal) * normal).normalized()
                    if roughness > 1e-4:
                        rd_new = (rd_new + roughness
                                  * _random_unit_vector()).normalized()
                        if rd_new.dot(normal) < 0.0:
                            rd_new = rd_new - 2.0 * rd_new.dot(normal) * normal
                    rd = rd_new
                    throughput *= fresnel / spec_prob
                else:
                    rd = _cosine_hemisphere_direction(normal)
                    throughput *= albedo * ((1.0 - metallic)
                                            / (1.0 - spec_prob))
                if (ti.max(throughput[0],
                           ti.max(throughput[1], throughput[2]))
                        < MIN_WEIGHT):
                    break  # absorbed
                ro = hit_point + normal * (10.0 * MIN_HIT_DISTANCE)
                inv_rd = ti.math.vec3(_safe_inverse(rd[0]),
                                      _safe_inverse(rd[1]),
                                      _safe_inverse(rd[2]))
                base_dist += t_hit
                t_prev = 0.0
                layer_prev = 1e30
                bounces_left -= 1

            if escaped:
                acc += ti.math.vec4(throughput[0] * background[0],
                                    throughput[1] * background[1],
                                    throughput[2] * background[2], 0.0)
                acc_alpha += 1.0 if interacted else background_alpha
            else:
                acc_alpha += 1.0

        inv_spp = 1.0 / ti.cast(samples_per_pixel, ti.f32)
        for ci in ti.static(range(4)):
            val = acc[ci] * inv_spp * 255.0
            out[f_rel, p, ci] = ti.cast(ti.math.clamp(val + 0.5, 0.0, 255.0),
                                        ti.u8)
        if transparent != 0:
            val = acc_alpha * inv_spp * 255.0
            out[f_rel, p, 4] = ti.cast(ti.math.clamp(val + 0.5, 0.0, 255.0),
                                       ti.u8)
