"""Hybrid raster front-end kernels for deterministic primary visibility.

The frontend replaces the first classic wavefront iteration for flat triangles
and Bezier circuits.  Primitive/chunk pairs enumerate candidate pixels, exact
intersection tests reject bbox misses, a typed opaque visibility buffer removes
hidden work, and compact transparent fragment records are ordered by the same
transitive ``(depth bin, descending layer)`` relation as the classic tracer.
One thread then resolves each pixel's ordered straight-ray list serially.

Important implementation properties:

* Triangle projection data is precomputed once per frame/primitive by
  :func:`raster_pipeline.precompute_triangle_projection`.  Straddling cases use
  exact per-pixel ray casting.
* Proven-opaque triangles and Bezier circuits share a typed int64 visibility
  buffer.  The packed key contains depth bin and inverted layer, so coplanar
  layer semantics are respected before transparent culling.
* Transparent triangle/circuit COUNT and WRITE kernels fetch sampled alpha and
  discard alpha-zero texels before sorting.  Records contain only exact-distance
  key, typed primitive reference (including the circuit-border bit), and two
  intersection parameters.
* :func:`raster_shadow_event_build` performs the same ordered transport/seam
  decisions as final resolve and emits only accepted triangle lighting events.
  :func:`raster_shadow_trace` traces that separate sparse any-hit queue and
  stores one visibility value per event/light, with no fixed fragment-slot or
  packed-light limit. Point/spot emitter radii and directional angular radii
  use the same deterministic golden-angle fan as the classic wavefront path.
* :func:`raster_first_shade` resolves at most ``MAX_SURFACES_PER_RAY`` accepted
  primary surfaces, commits retired pixels, and writes reflection/refraction
  continuations to the compact primary queue.  The host later copies only
  surviving continuations into the classic secondary-wavefront state.

The current host tile is a contiguous linear ray range, usually a row band,
not a fixed square raster tile.  Each pair covers up to ``RASTER_CHUNK`` pixels.
Future work should benchmark square block bins and candidate-parallel block
kernels. PN patches, custom scatter, near clipping, and in-place supersampling
still route to the classic frontend without changing geometry construction.
"""
import taichi as ti

from algan.rendering.raytracing.raytrace_kernels_taichi import (
    _M_BASIS_U,
    _M_BASIS_V,
    _M_BORDER_W,
    _M_CENTER,
    _M_FILLED,
    _M_IOR,
    _M_NORMAL,
    _M_REFLECTIVITY,
    _M_TRANSMISSION,
    BARYCENTRIC_EPSILON,
    DEPTH_TIE_EPSILON,
    INV_DEPTH_TIE_EPSILON,
    MAX_SHADOW_LIGHTS,
    MAX_SURFACES_PER_RAY,
    MIN_ALPHA,
    MIN_HIT_DISTANCE,
    MIN_WEIGHT,
    NODE_ARG,
    TRIANGLE_EDGE_EPSILON,
    _bezier_normal,
    _bezier_point_metrics,
    _generate_ray,
    _sample_circuit_color,
    _shade_tri_hit,
    _shadow_occluded,
)
from algan.rendering.raytracing.shading_taichi import _MID_UNLIT
from algan.rendering.raytracing.settings import SOFT_SHADOW_SAMPLES
from algan.rendering.raytracing.wavefront_kernels_taichi import (
    _ACTIVE,
    _DONE,
    _LT_AMBIENT,
    _LT_DIRECTIONAL,
    _LT_ENV_SH,
    _LT_HEMISPHERE,
    _GOLDEN_ANGLE,
    _material_reflectance,
    _offset_transmitted_origin,
    _refract_ray,
    _reserve_continuation_slot,
    _sample_env_map,
    _tri_color_g,
    _tri_extra_g,
    _tri_ior_g,
    _tri_normal_g,
    _tri_transmission_g,
)

# Candidate pixels per (prim, chunk) pair: one fine-raster thread tests at
# most this many pixels, bounding load imbalance for large bboxes.
RASTER_CHUNK = 256

# Empty typed visibility-buffer entry. Real hits pack the same strict ordering
# used by the classic deterministic tracer:
#
#   high 32 bits: floor(t / DEPTH_TIE_EPSILON)
#   low  32 bits: bitwise-inverted layer index (higher layer sorts first)
#
# Triangle layers are ``layer_offset_triangles + prim``; bezier layers are the
# circuit index.  This makes the atomic minimum layer-aware and lets the host
# recover the winning geometry type/id from the layer without a second winner
# buffer. Exact t/barycentrics are recomputed only for the terminal winner.
Z_SENTINEL = 0x7FFFFFFFFFFFFFFF


@ti.func
def _order_key(t, layer):
    """Packed transitive depth-bin / descending-layer key."""
    # Keep bit 63 clear so signed ``atomic_min`` preserves unsigned depth
    # order even for pathological distances.  Layers share the 32-bit packed
    # representation used by the classic renderer.
    depth_f = ti.floor(t * INV_DEPTH_TIE_EPSILON)
    depth_bin = ti.cast(ti.math.clamp(depth_f, 0.0, 2147483647.0), ti.u64)
    layer_u = ti.cast(ti.min(ti.max(layer, 0), 2147483647), ti.u64)
    return ti.cast((depth_bin << 32) | (ti.u64(0xFFFFFFFF) - layer_u), ti.i64)


@ti.func
def _frag_t(key):
    """Recover exact positive f32 distance from a fragment's low key bits."""
    return ti.bit_cast(ti.cast(ti.cast(key, ti.u64) & ti.u64(0xFFFFFFFF), ti.u32), ti.f32)


@ti.func
def _pack_bez_ref(circuit, in_border):
    """Negative typed fragment ref with the border bit folded into the id."""
    return -((circuit << 1) + (in_border & 1) + 1)


@ti.func
def _decode_bez_ref(ref):
    code = -ref - 1
    return code >> 1, code & 1


@ti.func
def _decode_z_layer(zkey, layer_offset_triangles):
    inv_layer = ti.cast(zkey, ti.u64) & ti.u64(0xFFFFFFFF)
    layer = ti.cast(ti.u64(0xFFFFFFFF) - inv_layer, ti.i32)
    is_bez = layer < ti.cast(layer_offset_triangles, ti.i32)
    prim = layer
    if not is_bez:
        prim = layer - ti.cast(layer_offset_triangles, ti.i32)
    return is_bez, prim


@ti.func
def _ss_setup(f, prim, ss_enabled: ti.template(), tri_pos: ti.template(),
              tri_screen: ti.template(), cam_origin: ti.template()):
    """Load the per-(frame, triangle) projection prepared once by the host.

    ``tri_screen[..., 0:3]`` are sx, ``3:6`` sy, ``6:9`` reciprocal
    perspective divisors, and column 9 is a validity flag.  World vertices are
    still read from ``tri_pos`` for hit reconstruction and the ray-cast
    fallback.  This removes repeated camera projection setup from every z,
    count, write, shadow-event, and resolve chunk.
    """
    tp = f % tri_pos.shape[0]
    ts = f % tri_screen.shape[0]
    v0 = ti.math.vec3(tri_pos[tp, prim, 0], tri_pos[tp, prim, 1],
                      tri_pos[tp, prim, 2])
    v1 = ti.math.vec3(tri_pos[tp, prim, 3], tri_pos[tp, prim, 4],
                      tri_pos[tp, prim, 5])
    v2 = ti.math.vec3(tri_pos[tp, prim, 6], tri_pos[tp, prim, 7],
                      tri_pos[tp, prim, 8])
    cam_o = ti.math.vec3(cam_origin[f, 0], cam_origin[f, 1], cam_origin[f, 2])
    vm = ti.Matrix([[v0[0], v0[1], v0[2]],
                    [v1[0], v1[1], v1[2]],
                    [v2[0], v2[1], v2[2]]])
    sm = ti.Matrix.zero(ti.f32, 3, 3)
    use_ss = 0
    if ti.static(ss_enabled):
        if tri_screen[ts, prim, 9] > 0.5:
            use_ss = 1
            for i in ti.static(range(3)):
                sm[0, i] = tri_screen[ts, prim, i]
                sm[1, i] = tri_screen[ts, prim, 3 + i]
                sm[2, i] = tri_screen[ts, prim, 6 + i]
    return use_ss, sm, vm, cam_o


@ti.func
def _ss_pixel(px, py, sm, vm, cam_o):
    """Screen-space test of one pixel against the pre-projected triangle.

    Edge functions give the 2D barycentric weights; the perspective-correct 3D
    weights are ``w_i = (E_i / d_i) / sum(E_j / d_j)`` (the shared screen-space
    area cancels). The 3D hit point ``H = sum w_i V_i`` gives the exact ray
    distance ``t = |H - cam_o|`` and barycentrics, matching Moller-Trumbore to
    float epsilon. Returns ``(ok, t, w1, w2)``.
    """
    qx = ti.cast(px, ti.f32) + 0.5
    qy = ti.cast(py, ti.f32) + 0.5
    sx0, sx1, sx2 = sm[0, 0], sm[0, 1], sm[0, 2]
    sy0, sy1, sy2 = sm[1, 0], sm[1, 1], sm[1, 2]
    e0 = (sx2 - sx1) * (qy - sy1) - (sy2 - sy1) * (qx - sx1)
    e1 = (sx0 - sx2) * (qy - sy2) - (sy0 - sy2) * (qx - sx2)
    e2 = (sx1 - sx0) * (qy - sy0) - (sy1 - sy0) * (qx - sx0)
    n0 = e0 * sm[2, 0]
    n1 = e1 * sm[2, 1]
    n2 = e2 * sm[2, 2]
    s = n0 + n1 + n2
    ok = 0
    t = 0.0
    w1 = 0.0
    w2 = 0.0
    if ti.abs(s) > 1e-30:
        inv = 1.0 / s
        b0 = n0 * inv
        b1 = n1 * inv
        b2 = n2 * inv
        if ((b1 >= -BARYCENTRIC_EPSILON) and (b2 >= -BARYCENTRIC_EPSILON)
                and (b0 >= -BARYCENTRIC_EPSILON)):
            v0 = ti.math.vec3(vm[0, 0], vm[0, 1], vm[0, 2])
            v1 = ti.math.vec3(vm[1, 0], vm[1, 1], vm[1, 2])
            v2 = ti.math.vec3(vm[2, 0], vm[2, 1], vm[2, 2])
            hp = b0 * v0 + b1 * v1 + b2 * v2
            tt = (hp - cam_o).norm()
            if tt > MIN_HIT_DISTANCE:
                ok = 1
                t = tt
                w1 = b1
                w2 = b2
    return ok, t, w1, w2


@ti.func
def _raycast_pixel(px, py, f, vm, half_w, half_h,
                   cam_origin: ti.template(), screen_point: ti.template(),
                   pixel_basis_x: ti.template(), pixel_basis_y: ti.template()):
    """Per-pixel ray-cast fallback (Moller-Trumbore), used when a triangle
    straddles the camera plane so screen-space projection is invalid. Returns
    ``(ok, t, w1, w2)``.
    """
    ro, rd = _generate_ray(f, px, py, 0.5, 0.5, half_w, half_h,
                           cam_origin, screen_point,
                           pixel_basis_x, pixel_basis_y)
    v0 = ti.math.vec3(vm[0, 0], vm[0, 1], vm[0, 2])
    v1 = ti.math.vec3(vm[1, 0], vm[1, 1], vm[1, 2])
    v2 = ti.math.vec3(vm[2, 0], vm[2, 1], vm[2, 2])
    e1 = v1 - v0
    e2 = v2 - v0
    pv = rd.cross(e2)
    det = e1.dot(pv)
    ok = 0
    t = 0.0
    w1 = 0.0
    w2 = 0.0
    if ti.abs(det) > 1e-12:
        inv_det = 1.0 / det
        tvec = ro - v0
        b1 = tvec.dot(pv) * inv_det
        qv = tvec.cross(e1)
        b2 = rd.dot(qv) * inv_det
        if ((b1 >= -BARYCENTRIC_EPSILON) and (b2 >= -BARYCENTRIC_EPSILON)
                and (b1 + b2 <= 1.0 + BARYCENTRIC_EPSILON)):
            th = e2.dot(qv) * inv_det
            if th > MIN_HIT_DISTANCE:
                ok = 1
                t = th
                w1 = b1
                w2 = b2
    return ok, t, w1, w2


@ti.func
def _pair_pixel(prim, f, x0, y0, bw, bh, off, j,
                time_start, width, height, tile_start, tile_pixels,
                half_w, half_h, use_ss, sm, vm, cam_o,
                cam_origin: ti.template(), screen_point: ti.template(),
                pixel_basis_x: ti.template(), pixel_basis_y: ti.template()):
    """Test candidate ``off + j`` of a pair at its pixel center, dispatching to
    the screen-space path (``use_ss``) or the ray-cast fallback. Returns
    ``(ok, local_pixel, t, w1, w2)``.
    """
    ok = 0
    lp = 0
    t = 0.0
    w1 = 0.0
    w2 = 0.0
    o = off + j
    if o < bw * bh:
        px = x0 + o % bw
        py = y0 + o // bw
        lpi = ((f - time_start) * (width * height) + py * width + px
               - tile_start)
        if (lpi >= 0) and (lpi < tile_pixels):
            hit = 0
            th = 0.0
            b1 = 0.0
            b2 = 0.0
            if use_ss != 0:
                hit, th, b1, b2 = _ss_pixel(px, py, sm, vm, cam_o)
            else:
                hit, th, b1, b2 = _raycast_pixel(
                    px, py, f, vm, half_w, half_h, cam_origin, screen_point,
                    pixel_basis_x, pixel_basis_y)
            if hit != 0:
                ok = 1
                lp = lpi
                t = th
                w1 = b1
                w2 = b2
    return ok, lp, t, w1, w2


@ti.func
def _bez_pixel_hit(circuit, f, px, py, half_w, half_h,
                   cam_origin: ti.template(), screen_point: ti.template(),
                   pixel_basis_x: ti.template(), pixel_basis_y: ti.template(),
                   pixel_world_scale: ti.template(),
                   circuit_meta: ti.template(), circuit_colors: ti.template(),
                   edges_2d: ti.template(), edge_accel: ti.template()):
    """Exact primary camera-ray/circuit hit for one known pixel."""
    ok = 0
    t = 0.0
    u = 0.0
    v = 0.0
    in_border = 0
    ro, rd = _generate_ray(f, px, py, 0.5, 0.5, half_w, half_h,
                           cam_origin, screen_point,
                           pixel_basis_x, pixel_basis_y)
    tm = f % circuit_meta.shape[0]
    n = ti.math.vec3(circuit_meta[tm, circuit, _M_NORMAL],
                     circuit_meta[tm, circuit, _M_NORMAL + 1],
                     circuit_meta[tm, circuit, _M_NORMAL + 2])
    denom = rd.dot(n)
    if ti.abs(denom) > 1e-9:
        center = ti.math.vec3(circuit_meta[tm, circuit, _M_CENTER],
                              circuit_meta[tm, circuit, _M_CENTER + 1],
                              circuit_meta[tm, circuit, _M_CENTER + 2])
        th = (center - ro).dot(n) / denom
        if th > MIN_HIT_DISTANCE:
            hit = ro + th * rd - center
            bu = ti.math.vec3(circuit_meta[tm, circuit, _M_BASIS_U],
                              circuit_meta[tm, circuit, _M_BASIS_U + 1],
                              circuit_meta[tm, circuit, _M_BASIS_U + 2])
            bv = ti.math.vec3(circuit_meta[tm, circuit, _M_BASIS_V],
                              circuit_meta[tm, circuit, _M_BASIS_V + 1],
                              circuit_meta[tm, circuit, _M_BASIS_V + 2])
            uu = hit.dot(bu)
            vv = hit.dot(bv)
            pixel_size = pixel_world_scale[f] * th
            border_w = circuit_meta[tm, circuit, _M_BORDER_W] * pixel_size
            outline_w = 0.6 * pixel_size
            filled = circuit_meta[tm, circuit, _M_FILLED] > 0.5
            query_radius = ti.abs(border_w)
            if filled:
                query_radius = ti.max(query_radius, outline_w)
            te = f % edges_2d.shape[0]
            crossings, min_dist_sq = _bezier_point_metrics(
                circuit, te, uu, vv, query_radius,
                circuit_meta.shape[1], edges_2d, edge_accel)
            is_border = min_dist_sq < border_w * border_w
            inside = False
            if filled:
                inside = ((crossings % 2) == 1) or (
                    min_dist_sq < outline_w * outline_w)
            if inside or is_border:
                ok = 1
                t = th
                u = uu
                v = vv
                in_border = 1 if is_border else 0
    return ok, t, u, v, in_border


@ti.func
def _bez_pair_pixel(circuit, f, x0, y0, bw, bh, off, j,
                    time_start, width, height, tile_start, tile_pixels,
                    half_w, half_h,
                    cam_origin: ti.template(), screen_point: ti.template(),
                    pixel_basis_x: ti.template(), pixel_basis_y: ti.template(),
                    pixel_world_scale: ti.template(),
                    circuit_meta: ti.template(), circuit_colors: ti.template(),
                    edges_2d: ti.template(), edge_accel: ti.template()):
    """Pair wrapper around :func:`_bez_pixel_hit`."""
    ok = 0
    lp = 0
    t = 0.0
    u = 0.0
    v = 0.0
    in_border = 0
    o = off + j
    if o < bw * bh:
        px = x0 + o % bw
        py = y0 + o // bw
        lpi = ((f - time_start) * (width * height) + py * width + px
               - tile_start)
        if (lpi >= 0) and (lpi < tile_pixels):
            hit, th, uu, vv, ib = _bez_pixel_hit(
                circuit, f, px, py, half_w, half_h, cam_origin, screen_point,
                pixel_basis_x, pixel_basis_y, pixel_world_scale, circuit_meta,
                circuit_colors, edges_2d, edge_accel)
            if hit != 0:
                ok = 1
                lp = lpi
                t = th
                u = uu
                v = vv
                in_border = ib
    return ok, lp, t, u, v, in_border


@ti.kernel
def raster_tri_z(
        pairs: ti.types.ndarray(), num_pairs: int,
        tri_pos: ti.types.ndarray(), tri_screen: ti.types.ndarray(),
        cam_origin: ti.types.ndarray(), screen_point: ti.types.ndarray(),
        pixel_basis_x: ti.types.ndarray(), pixel_basis_y: ti.types.ndarray(),
        time_start: int, width: int, height: int,
        half_w: ti.f32, half_h: ti.f32,
        tile_start: int, tile_pixels: int,
        ss_enabled: ti.template(), layer_offset_triangles: ti.f32,
        zbuf: ti.types.ndarray()):
    """Typed opaque visibility prepass for flat triangles."""
    for p in range(num_pairs):
        prim = pairs[p, 0]
        f = pairs[p, 1]
        x0 = pairs[p, 2]
        y0 = pairs[p, 3]
        bw = pairs[p, 4]
        bh = pairs[p, 5]
        off = pairs[p, 6]
        use_ss, sm, vm, cam_o = _ss_setup(
            f, prim, ss_enabled, tri_pos, tri_screen, cam_origin)
        layer = ti.cast(layer_offset_triangles, ti.i32) + prim
        for j in range(RASTER_CHUNK):
            ok, lp, t, _w1, _w2 = _pair_pixel(
                prim, f, x0, y0, bw, bh, off, j, time_start, width, height,
                tile_start, tile_pixels, half_w, half_h, use_ss, sm, vm, cam_o,
                cam_origin, screen_point, pixel_basis_x, pixel_basis_y)
            if ok != 0:
                ti.atomic_min(zbuf[lp], _order_key(t, layer))


@ti.kernel
def raster_bez_z(
        pairs: ti.types.ndarray(), num_pairs: int,
        cam_origin: ti.types.ndarray(), screen_point: ti.types.ndarray(),
        pixel_basis_x: ti.types.ndarray(), pixel_basis_y: ti.types.ndarray(),
        pixel_world_scale: ti.types.ndarray(),
        circuit_meta: ti.types.ndarray(), circuit_colors: ti.types.ndarray(),
        edges_2d: ti.types.ndarray(), edge_accel: ti.types.ndarray(),
        time_start: int, width: int, height: int,
        half_w: ti.f32, half_h: ti.f32,
        tile_start: int, tile_pixels: int,
        zbuf: ti.types.ndarray()):
    """Typed opaque visibility prepass for proven-opaque bezier circuits."""
    for p in range(num_pairs):
        circuit = pairs[p, 0]
        f = pairs[p, 1]
        x0 = pairs[p, 2]
        y0 = pairs[p, 3]
        bw = pairs[p, 4]
        bh = pairs[p, 5]
        off = pairs[p, 6]
        for j in range(RASTER_CHUNK):
            ok, lp, t, _u, _v, _ib = _bez_pair_pixel(
                circuit, f, x0, y0, bw, bh, off, j, time_start, width, height,
                tile_start, tile_pixels, half_w, half_h, cam_origin,
                screen_point, pixel_basis_x, pixel_basis_y, pixel_world_scale,
                circuit_meta, circuit_colors, edges_2d, edge_accel)
            if ok != 0:
                ti.atomic_min(zbuf[lp], _order_key(t, circuit))


@ti.kernel
def raster_tri_count(
        pairs: ti.types.ndarray(), num_pairs: int,
        tri_pos: ti.types.ndarray(), tri_screen: ti.types.ndarray(),
        tri_colors: ti.types.ndarray(), col_row: ti.types.ndarray(),
        tri_uvs: ti.types.ndarray(), tri_tex_meta: ti.types.ndarray(),
        textures: ti.types.ndarray(), num_colored_triangles: ti.i32,
        cam_origin: ti.types.ndarray(), screen_point: ti.types.ndarray(),
        pixel_basis_x: ti.types.ndarray(), pixel_basis_y: ti.types.ndarray(),
        time_start: int, width: int, height: int,
        half_w: ti.f32, half_h: ti.f32,
        tile_start: int, tile_pixels: int,
        ss_enabled: ti.template(), layer_offset_triangles: ti.f32,
        zbuf: ti.types.ndarray(), pair_count: ti.types.ndarray()):
    """Count surviving nonzero-alpha transparent triangle fragments."""
    for p in range(num_pairs):
        prim = pairs[p, 0]
        f = pairs[p, 1]
        x0 = pairs[p, 2]
        y0 = pairs[p, 3]
        bw = pairs[p, 4]
        bh = pairs[p, 5]
        off = pairs[p, 6]
        use_ss, sm, vm, cam_o = _ss_setup(
            f, prim, ss_enabled, tri_pos, tri_screen, cam_origin)
        layer = ti.cast(layer_offset_triangles, ti.i32) + prim
        cnt = 0
        for j in range(RASTER_CHUNK):
            ok, lp, t, w1, w2 = _pair_pixel(
                prim, f, x0, y0, bw, bh, off, j, time_start, width, height,
                tile_start, tile_pixels, half_w, half_h, use_ss, sm, vm, cam_o,
                cam_origin, screen_point, pixel_basis_x, pixel_basis_y)
            if ok != 0 and _order_key(t, layer) < zbuf[lp]:
                w0 = 1.0 - w1 - w2
                _color, alpha = _tri_color_g(
                    0, f, prim, w0, w1, w2, tri_colors, col_row, tri_uvs,
                    tri_tex_meta, textures, num_colored_triangles)
                if alpha > MIN_ALPHA:
                    cnt += 1
        pair_count[p] = cnt


@ti.kernel
def raster_tri_write(
        pairs: ti.types.ndarray(), num_pairs: int,
        pair_offset: ti.types.ndarray(),
        tri_pos: ti.types.ndarray(), tri_screen: ti.types.ndarray(),
        tri_colors: ti.types.ndarray(), col_row: ti.types.ndarray(),
        tri_uvs: ti.types.ndarray(), tri_tex_meta: ti.types.ndarray(),
        textures: ti.types.ndarray(), num_colored_triangles: ti.i32,
        cam_origin: ti.types.ndarray(), screen_point: ti.types.ndarray(),
        pixel_basis_x: ti.types.ndarray(), pixel_basis_y: ti.types.ndarray(),
        time_start: int, width: int, height: int,
        half_w: ti.f32, half_h: ti.f32,
        tile_start: int, tile_pixels: int,
        ss_enabled: ti.template(), layer_offset_triangles: ti.f32,
        zbuf: ti.types.ndarray(), frag_key: ti.types.ndarray(),
        frag_ref: ti.types.ndarray(), frag_ab: ti.types.ndarray()):
    """Emit exact-distance triangle records; alpha-zero texels are discarded."""
    for p in range(num_pairs):
        prim = pairs[p, 0]
        f = pairs[p, 1]
        x0 = pairs[p, 2]
        y0 = pairs[p, 3]
        bw = pairs[p, 4]
        bh = pairs[p, 5]
        off = pairs[p, 6]
        use_ss, sm, vm, cam_o = _ss_setup(
            f, prim, ss_enabled, tri_pos, tri_screen, cam_origin)
        layer = ti.cast(layer_offset_triangles, ti.i32) + prim
        w = pair_offset[p]
        for j in range(RASTER_CHUNK):
            ok, lp, t, w1, w2 = _pair_pixel(
                prim, f, x0, y0, bw, bh, off, j, time_start, width, height,
                tile_start, tile_pixels, half_w, half_h, use_ss, sm, vm, cam_o,
                cam_origin, screen_point, pixel_basis_x, pixel_basis_y)
            if ok != 0 and _order_key(t, layer) < zbuf[lp]:
                w0 = 1.0 - w1 - w2
                _color, alpha = _tri_color_g(
                    0, f, prim, w0, w1, w2, tri_colors, col_row, tri_uvs,
                    tri_tex_meta, textures, num_colored_triangles)
                if alpha > MIN_ALPHA:
                    tb = ti.cast(ti.bit_cast(t, ti.u32), ti.i64)
                    frag_key[w] = (ti.cast(lp, ti.i64) << 32) | tb
                    frag_ref[w] = prim
                    frag_ab[w, 0] = w1
                    frag_ab[w, 1] = w2
                    w += 1


@ti.kernel
def raster_bez_count(
        pairs: ti.types.ndarray(), num_pairs: int,
        cam_origin: ti.types.ndarray(), screen_point: ti.types.ndarray(),
        pixel_basis_x: ti.types.ndarray(), pixel_basis_y: ti.types.ndarray(),
        pixel_world_scale: ti.types.ndarray(),
        circuit_meta: ti.types.ndarray(), circuit_colors: ti.types.ndarray(),
        circuit_border_colors: ti.types.ndarray(),
        edges_2d: ti.types.ndarray(), edge_accel: ti.types.ndarray(),
        time_start: int, width: int, height: int,
        half_w: ti.f32, half_h: ti.f32,
        tile_start: int, tile_pixels: int,
        zbuf: ti.types.ndarray(), pair_count: ti.types.ndarray()):
    """Count surviving nonzero-alpha translucent circuit fragments."""
    for p in range(num_pairs):
        circuit = pairs[p, 0]
        f = pairs[p, 1]
        x0 = pairs[p, 2]
        y0 = pairs[p, 3]
        bw = pairs[p, 4]
        bh = pairs[p, 5]
        off = pairs[p, 6]
        cnt = 0
        for j in range(RASTER_CHUNK):
            ok, lp, t, u, v, ib = _bez_pair_pixel(
                circuit, f, x0, y0, bw, bh, off, j, time_start, width, height,
                tile_start, tile_pixels, half_w, half_h, cam_origin,
                screen_point, pixel_basis_x, pixel_basis_y, pixel_world_scale,
                circuit_meta, circuit_colors, edges_2d, edge_accel)
            if ok != 0 and _order_key(t, circuit) < zbuf[lp]:
                _color, alpha = _sample_circuit_color(
                    circuit, f, u, v, ib, circuit_meta, circuit_colors,
                    circuit_border_colors)
                if alpha > MIN_ALPHA:
                    cnt += 1
        pair_count[p] = cnt


@ti.kernel
def raster_bez_write(
        pairs: ti.types.ndarray(), num_pairs: int,
        pair_offset: ti.types.ndarray(),
        cam_origin: ti.types.ndarray(), screen_point: ti.types.ndarray(),
        pixel_basis_x: ti.types.ndarray(), pixel_basis_y: ti.types.ndarray(),
        pixel_world_scale: ti.types.ndarray(),
        circuit_meta: ti.types.ndarray(), circuit_colors: ti.types.ndarray(),
        circuit_border_colors: ti.types.ndarray(),
        edges_2d: ti.types.ndarray(), edge_accel: ti.types.ndarray(),
        time_start: int, width: int, height: int,
        half_w: ti.f32, half_h: ti.f32,
        tile_start: int, tile_pixels: int,
        zbuf: ti.types.ndarray(), frag_key: ti.types.ndarray(),
        frag_ref: ti.types.ndarray(), frag_ab: ti.types.ndarray()):
    """Emit circuit records with the border flag packed into ``frag_ref``."""
    for p in range(num_pairs):
        circuit = pairs[p, 0]
        f = pairs[p, 1]
        x0 = pairs[p, 2]
        y0 = pairs[p, 3]
        bw = pairs[p, 4]
        bh = pairs[p, 5]
        off = pairs[p, 6]
        w = pair_offset[p]
        for j in range(RASTER_CHUNK):
            ok, lp, t, u, v, ib = _bez_pair_pixel(
                circuit, f, x0, y0, bw, bh, off, j, time_start, width, height,
                tile_start, tile_pixels, half_w, half_h, cam_origin,
                screen_point, pixel_basis_x, pixel_basis_y, pixel_world_scale,
                circuit_meta, circuit_colors, edges_2d, edge_accel)
            if ok != 0 and _order_key(t, circuit) < zbuf[lp]:
                _color, alpha = _sample_circuit_color(
                    circuit, f, u, v, ib, circuit_meta, circuit_colors,
                    circuit_border_colors)
                if alpha > MIN_ALPHA:
                    tb = ti.cast(ti.bit_cast(t, ti.u32), ti.i64)
                    frag_key[w] = (ti.cast(lp, ti.i64) << 32) | tb
                    frag_ref[w] = _pack_bez_ref(circuit, ib)
                    frag_ab[w, 0] = u
                    frag_ab[w, 1] = v
                    w += 1


@ti.func
def _terminal_z_hit(zkey, f, px, py, layer_offset_triangles,
                    ss_enabled: ti.template(),
                    tri_pos: ti.template(), tri_screen: ti.template(),
                    cam_origin: ti.template(), screen_point: ti.template(),
                    pixel_basis_x: ti.template(), pixel_basis_y: ti.template(),
                    pixel_world_scale: ti.template(),
                    circuit_meta: ti.template(), circuit_colors: ti.template(),
                    edges_2d: ti.template(), edge_accel: ti.template(),
                    half_w, half_h):
    """Recompute exact payload for the typed visibility-buffer winner."""
    valid = 0
    is_bez = False
    prim = 0
    t = 0.0
    a = 0.0
    b = 0.0
    in_border = 0
    if zkey != ti.i64(Z_SENTINEL):
        is_bez, prim = _decode_z_layer(zkey, layer_offset_triangles)
        if is_bez:
            valid, t, a, b, in_border = _bez_pixel_hit(
                prim, f, px, py, half_w, half_h, cam_origin, screen_point,
                pixel_basis_x, pixel_basis_y, pixel_world_scale, circuit_meta,
                circuit_colors, edges_2d, edge_accel)
        else:
            use_ss, sm, vm, cam_o = _ss_setup(
                f, prim, ss_enabled, tri_pos, tri_screen, cam_origin)
            if use_ss != 0:
                valid, t, a, b = _ss_pixel(px, py, sm, vm, cam_o)
            else:
                valid, t, a, b = _raycast_pixel(
                    px, py, f, vm, half_w, half_h, cam_origin, screen_point,
                    pixel_basis_x, pixel_basis_y)
    return valid, is_bez, prim, t, a, b, in_border


@ti.func
def _tri_shadow_normals(f, prim, a, b, rd,
                        tri_pos: ti.template(), tri_norm: ti.template(),
                        tri_uvs: ti.template(), tri_tex_meta: ti.template(),
                        textures: ti.template(), num_colored_triangles):
    """Shading + geometric face normals of a triangle hit, oriented for a
    shadow-ray origin exactly as ``wavefront_shade``'s inline shadow block: the
    shading normal faces the viewer, the geometric normal shares its hemisphere
    (so a grazing shadow ray does not self-shadow on an adjacent uphill facet).
    """
    w0 = 1.0 - a - b
    snrm = _tri_normal_g(0, f, prim, w0, a, b, tri_norm, tri_pos, tri_uvs,
                         tri_tex_meta, textures, num_colored_triangles)
    tp = f % tri_pos.shape[0]
    v0 = ti.math.vec3(tri_pos[tp, prim, 0], tri_pos[tp, prim, 1],
                      tri_pos[tp, prim, 2])
    v1 = ti.math.vec3(tri_pos[tp, prim, 3], tri_pos[tp, prim, 4],
                      tri_pos[tp, prim, 5])
    v2 = ti.math.vec3(tri_pos[tp, prim, 6], tri_pos[tp, prim, 7],
                      tri_pos[tp, prim, 8])
    fnrm = (v1 - v0).cross(v2 - v0)
    if snrm.norm() > 1e-9:
        snrm = snrm.normalized()
    if snrm.dot(rd) > 0.0:
        snrm = -snrm
    if fnrm.norm() > 1e-9:
        fnrm = fnrm.normalized()
    if fnrm.dot(snrm) < 0.0:
        fnrm = -fnrm
    return snrm, fnrm


@ti.kernel
def raster_shadow_event_build(
        num_pixels: int,
        run_offsets: ti.types.ndarray(),
        frag_key: ti.types.ndarray(), frag_ref: ti.types.ndarray(),
        frag_ab: ti.types.ndarray(), zbuf: ti.types.ndarray(),
        tri_pos: ti.types.ndarray(), tri_screen: ti.types.ndarray(),
        tri_norm: ti.types.ndarray(), tri_extra: ti.types.ndarray(),
        tri_colors: ti.types.ndarray(), tri_uvs: ti.types.ndarray(),
        tri_tex_meta: ti.types.ndarray(), textures: ti.types.ndarray(),
        num_colored_triangles: ti.i32, col_row: ti.types.ndarray(),
        circuit_meta: ti.types.ndarray(), circuit_colors: ti.types.ndarray(),
        circuit_border_colors: ti.types.ndarray(),
        edges_2d: ti.types.ndarray(), edge_accel: ti.types.ndarray(),
        pixel_world_scale: ti.types.ndarray(), layer_offset_triangles: ti.f32,
        refraction: ti.template(), ss_enabled: ti.template(),
        has_bez: ti.template(),
        time_start: int, width: int, height: int, tile_start: int,
        cam_origin: ti.types.ndarray(), screen_point: ti.types.ndarray(),
        pixel_basis_x: ti.types.ndarray(), pixel_basis_y: ti.types.ndarray(),
        gen_meta: ti.types.ndarray(), max_bounces: int,
        frag_shadow_id: ti.types.ndarray(), z_shadow_id: ti.types.ndarray(),
        event_pos: ti.types.ndarray(), event_snrm: ti.types.ndarray(),
        event_fnrm: ti.types.ndarray(), event_frame: ti.types.ndarray(),
        event_count: ti.types.ndarray()):
    """Build an exact sparse queue of accepted primary triangle shade events.

    The ordered transport walk mirrors ``raster_first_shade`` through seam
    rejection, alpha evaluation, throughput termination, and path bending.
    Only triangle fragments that the resolve will actually shade reserve an
    event.  Their IDs are written back beside the raw fragment (or terminal
    z-winner) so the later resolve can fetch one exact per-light visibility
    row without position-based slot approximations.
    """
    pixels_per_frame = width * height
    for r in range(num_pixels):
        g = tile_start + r
        f_rel = g // pixels_per_frame
        p = g - f_rel * pixels_per_frame
        f = time_start + f_rel
        py = p // width
        px = p - py * width
        ro, rd = _generate_ray(f, px, py, gen_meta[0], gen_meta[1],
                               gen_meta[2], gen_meta[3], cam_origin,
                               screen_point, pixel_basis_x, pixel_basis_y)
        weight = ti.math.vec3(1.0, 1.0, 1.0)
        seam_t = -1e30
        bounces_left = max_bounces
        processed = 0
        start = run_offsets[r]
        end = run_offsets[r + 1]
        nrun = end - start
        has_z = 1 if zbuf[r] != ti.i64(Z_SENTINEL) else 0
        total = nrun + has_z
        q = 0
        while q < total and processed < MAX_SURFACES_PER_RAY:
            from_z = q >= nrun
            idx = start + q
            t_hit = 0.0
            ref = 0
            a = 0.0
            b = 0.0
            in_border = 0
            is_bez = False
            valid = 1
            if not from_z:
                t_hit = _frag_t(frag_key[idx])
                ref = frag_ref[idx]
                if ti.static(has_bez):
                    is_bez = ref < 0
                if is_bez:
                    circuit, in_border = _decode_bez_ref(ref)
                    ref = circuit
                a = frag_ab[idx, 0]
                b = frag_ab[idx, 1]
            else:
                valid, is_bez, ref, t_hit, a, b, in_border = _terminal_z_hit(
                    zbuf[r], f, px, py, layer_offset_triangles, ss_enabled,
                    tri_pos, tri_screen, cam_origin, screen_point,
                    pixel_basis_x, pixel_basis_y, pixel_world_scale,
                    circuit_meta, circuit_colors, edges_2d, edge_accel,
                    gen_meta[2], gen_meta[3])
            q += 1
            if valid == 0:
                continue
            processed += 1
            w0 = 1.0 - a - b
            edge_hit = 0
            if not is_bez:
                if ti.min(w0, ti.min(a, b)) < TRIANGLE_EDGE_EPSILON:
                    edge_hit = 1
            if (edge_hit == 1) and (t_hit - seam_t <= DEPTH_TIE_EPSILON):
                continue
            seam_t = t_hit if edge_hit == 1 else -1e30

            alpha = 0.0
            reflectivity = 0.0
            ior = 0.0
            transmission = 0.0
            albedo = ti.math.vec3(0.0, 0.0, 0.0)
            normal = ti.math.vec3(0.0, 0.0, 0.0)
            if is_bez:
                color, alpha = _sample_circuit_color(
                    ref, f, a, b, in_border, circuit_meta, circuit_colors,
                    circuit_border_colors)
                albedo = ti.math.vec3(color[0], color[1], color[2])
                cm = f % circuit_meta.shape[0]
                reflectivity = circuit_meta[cm, ref, _M_REFLECTIVITY]
                ior = circuit_meta[cm, ref, _M_IOR]
                transmission = circuit_meta[cm, ref, _M_TRANSMISSION]
                if (reflectivity >= 0.0) or (transmission > 1e-4):
                    normal = _bezier_normal(f, ref, circuit_meta).normalized()
            else:
                color, alpha = _tri_color_g(
                    0, f, ref, w0, a, b, tri_colors, col_row, tri_uvs,
                    tri_tex_meta, textures, num_colored_triangles)
                albedo = ti.math.vec3(color[0], color[1], color[2])
                reflectivity, _rough = _tri_extra_g(
                    0, f, ref, w0, a, b, tri_extra, col_row, tri_uvs,
                    tri_tex_meta, textures, num_colored_triangles)
                ior = _tri_ior_g(
                    0, f, ref, w0, a, b, tri_extra, col_row, tri_uvs,
                    tri_tex_meta, textures, num_colored_triangles)
                transmission = _tri_transmission_g(
                    0, f, ref, w0, a, b, tri_extra, col_row, tri_uvs,
                    tri_tex_meta, textures, num_colored_triangles)
                # This exact accepted triangle event receives one sparse shadow
                # queue entry, irrespective of raw list position/depth.
                eid = ti.atomic_add(event_count[0], 1)
                if from_z:
                    z_shadow_id[r] = eid
                else:
                    frag_shadow_id[idx] = eid
                snrm, fnrm = _tri_shadow_normals(
                    f, ref, a, b, rd, tri_pos, tri_norm, tri_uvs,
                    tri_tex_meta, textures, num_colored_triangles)
                hp = ro + t_hit * rd
                for k in ti.static(range(3)):
                    event_pos[eid, k] = hp[k]
                    event_snrm[eid, k] = snrm[k]
                    event_fnrm[eid, k] = fnrm[k]
                event_frame[eid] = f
                if (reflectivity >= 0.0) or (transmission > 1e-4):
                    normal = snrm

            alpha = ti.math.clamp(alpha, 0.0, 1.0)
            transmission = ti.math.clamp(transmission, 0.0, 1.0)
            R, diel_pass = _material_reflectance(
                rd, normal, reflectivity, ior, albedo)
            if bounces_left <= 0:
                R = ti.math.vec3(0.0, 0.0, 0.0)
            is_glass = False
            is_pane = False
            if ti.static(refraction != 0):
                if (transmission > 1e-4) and (bounces_left > 0) \
                        and (ior > 1.0 + 1e-4):
                    if is_bez:
                        is_pane = True
                    else:
                        is_glass = True
            tint = ti.math.clamp(albedo, 0.0, 1.0)
            trans_share = diel_pass * transmission
            refl_energy = alpha * R
            refl_max = ti.max(refl_energy[0],
                              ti.max(refl_energy[1], refl_energy[2]))
            trans_energy = alpha * trans_share
            cover_pass = 1.0 - alpha
            cover3 = ti.math.vec3(cover_pass, cover_pass, cover_pass)
            split_refl = False
            if ti.static(refraction != 0):
                if (refl_max > MIN_ALPHA) and (cover_pass > MIN_ALPHA) \
                        and (bounces_left > 0):
                    split_refl = True
            if is_glass:
                if (refl_max > MIN_ALPHA) and (refl_max >= cover_pass):
                    break
                weight *= cover_pass
            elif is_pane or split_refl:
                weight *= cover3 + trans_energy * tint
            elif (refl_max > MIN_ALPHA) and (refl_max >= cover_pass):
                break
            else:
                weight *= cover3 + trans_energy * tint
            if ti.max(weight[0], ti.max(weight[1], weight[2])) < MIN_WEIGHT:
                break


@ti.kernel
def raster_shadow_trace(
        num_events: int,
        event_pos: ti.types.ndarray(), event_snrm: ti.types.ndarray(),
        event_fnrm: ti.types.ndarray(), event_frame: ti.types.ndarray(),
        t_nodes: NODE_ARG, t_node_miss: ti.types.ndarray(),
        t_leaf_prim: ti.types.ndarray(), t_leaf_tspan: ti.types.ndarray(),
        t_first_leaf: int,
        tri_pos: ti.types.ndarray(), tri_colors: ti.types.ndarray(),
        tri_uvs: ti.types.ndarray(), tri_tex_meta: ti.types.ndarray(),
        textures: ti.types.ndarray(), num_colored_triangles: ti.i32,
        p_nodes: NODE_ARG, p_node_miss: ti.types.ndarray(),
        p_leaf_prim: ti.types.ndarray(), p_leaf_tspan: ti.types.ndarray(),
        p_first_leaf: int,
        pn_ctrl: ti.types.ndarray(), pn_obb: ti.types.ndarray(),
        pn_colors: ti.types.ndarray(),
        b_nodes: NODE_ARG, b_node_miss: ti.types.ndarray(),
        b_leaf_prim: ti.types.ndarray(), b_leaf_tspan: ti.types.ndarray(),
        b_first_leaf: int,
        circuit_meta: ti.types.ndarray(), circuit_colors: ti.types.ndarray(),
        circuit_border_colors: ti.types.ndarray(),
        edges_2d: ti.types.ndarray(), edge_accel: ti.types.ndarray(),
        light_pos: ti.types.ndarray(), light_col: ti.types.ndarray(),
        num_lights: int, pixel_world_scale: ti.types.ndarray(),
        layer_offset_triangles: ti.f32, layer_offset_pn: ti.f32,
        refit: ti.template(),
        has_tri: ti.template(), has_pn: ti.template(), has_bez: ti.template(),
        shadow_vis: ti.types.ndarray()):
    """Trace the dedicated sparse any-hit shadow queue exactly.

    A zero-radius point/spot/directional light emits one hard-shadow ray.
    Non-zero emitter radii use the same fixed golden-angle fan as the classic
    wavefront shader; area lights are already expanded into packed sample rows
    and therefore naturally obtain soft visibility by averaging those rows in
    the material shader.
    """
    for e in range(num_events):
        f = event_frame[e]
        ff = ti.cast(f, ti.f32)
        spos = ti.math.vec3(event_pos[e, 0], event_pos[e, 1], event_pos[e, 2])
        snrm = ti.math.vec3(event_snrm[e, 0], event_snrm[e, 1],
                            event_snrm[e, 2])
        fnrm = ti.math.vec3(event_fnrm[e, 0], event_fnrm[e, 1],
                            event_fnrm[e, 2])
        sorigin = spos + fnrm * (10.0 * MIN_HIT_DISTANCE)
        tl = f % light_pos.shape[0]
        for li in range(num_lights):
            visibility = 1.0
            lp = ti.math.vec3(light_pos[tl, li, 0], light_pos[tl, li, 1],
                              light_pos[tl, li, 2])
            ltype = 0
            radius = 0.0
            if light_col.shape[2] > 3:
                ltype = ti.cast(light_col[tl, li, 3] + 0.5, ti.i32)
                if light_col.shape[2] > 11:
                    radius = light_col[tl, li, 11]
            to_light = lp - spos
            ldist = to_light.norm()
            wi = ti.math.vec3(0.0, 0.0, 0.0)
            valid = 0
            if ltype == _LT_DIRECTIONAL:
                wi = -ti.math.vec3(light_col[tl, li, 6],
                                   light_col[tl, li, 7],
                                   light_col[tl, li, 8])
                ldist = 1e7
                valid = 1
            elif (ltype != _LT_AMBIENT) and (ltype != _LT_HEMISPHERE) \
                    and (ltype != _LT_ENV_SH) and (ldist > 1e-5):
                wi = to_light / ldist
                valid = 1
            if valid == 1:
                ns = 1
                b1 = ti.math.vec3(0.0, 0.0, 0.0)
                b2 = ti.math.vec3(0.0, 0.0, 0.0)
                if radius > 0.0:
                    ns = SOFT_SHADOW_SAMPLES
                    aref = ti.math.vec3(1.0, 0.0, 0.0)
                    if ti.abs(wi[0]) > 0.9:
                        aref = ti.math.vec3(0.0, 1.0, 0.0)
                    b1 = wi.cross(aref).normalized()
                    b2 = wi.cross(b1)

                occ_sum = 0.0
                n_valid = 0.0
                for s in range(ns):
                    wis = wi
                    ldn = ldist
                    ok = 1
                    if radius > 0.0:
                        ang = _GOLDEN_ANGLE * s
                        rr = radius * ti.sqrt(
                            (ti.cast(s, ti.f32) + 0.5)
                            / ti.cast(ns, ti.f32))
                        off = (ti.cos(ang) * b1 + ti.sin(ang) * b2) * rr
                        if ltype == _LT_DIRECTIONAL:
                            wis = (wi + off).normalized()
                        else:
                            tls = lp + off - spos
                            ldn = tls.norm()
                            if ldn > 1e-5:
                                wis = tls / ldn
                            else:
                                ok = 0
                    if (ok == 1) and (fnrm.dot(wis) > 1e-3) \
                            and (snrm.dot(wis) > 1e-4):
                        n_valid += 1.0
                        occ_sum += _shadow_occluded(
                            refit, sorigin, wis, f, ff,
                            ldn - 20.0 * MIN_HIT_DISTANCE,
                            pixel_world_scale[
                                f % pixel_world_scale.shape[0]], 0.0,
                            layer_offset_triangles, layer_offset_pn,
                            has_tri, has_pn, has_bez,
                            t_nodes, t_node_miss, t_leaf_prim, t_leaf_tspan,
                            t_first_leaf, tri_pos, tri_colors, tri_uvs,
                            tri_tex_meta, textures, num_colored_triangles,
                            p_nodes, p_node_miss, p_leaf_prim, p_leaf_tspan,
                            p_first_leaf, pn_ctrl, pn_obb, pn_colors,
                            b_nodes, b_node_miss, b_leaf_prim, b_leaf_tspan,
                            b_first_leaf, circuit_meta, circuit_colors,
                            circuit_border_colors, edges_2d, edge_accel)
                if n_valid > 0.0:
                    visibility = 1.0 - occ_sum / n_valid
            shadow_vis[e, li] = visibility


@ti.kernel
def raster_first_shade(
        num_pixels: int,
        run_offsets: ti.types.ndarray(),
        frag_key: ti.types.ndarray(), frag_ref: ti.types.ndarray(),
        frag_ab: ti.types.ndarray(),
        zbuf: ti.types.ndarray(),
        tri_pos: ti.types.ndarray(), tri_screen: ti.types.ndarray(),
        tri_norm: ti.types.ndarray(),
        tri_extra: ti.types.ndarray(), tri_colors: ti.types.ndarray(),
        tri_uvs: ti.types.ndarray(), tri_tex_meta: ti.types.ndarray(),
        textures: ti.types.ndarray(), num_colored_triangles: ti.i32,
        col_row: ti.types.ndarray(),
        tri_mat_id: ti.types.ndarray(), tri_mat: ti.types.ndarray(),
        # Bezier circuit shading data (used when has_bez; 1x1 placeholders
        # otherwise). Circuits route entirely through the transparent fragment
        # stream or typed opaque visibility buffer, tagged by a negative ref
        # when it appears in the translucent fragment stream.
        circuit_meta: ti.types.ndarray(), circuit_colors: ti.types.ndarray(),
        circuit_border_colors: ti.types.ndarray(),
        pixel_world_scale: ti.types.ndarray(), edges_2d: ti.types.ndarray(),
        edge_accel: ti.types.ndarray(),
        light_pos: ti.types.ndarray(), light_col: ti.types.ndarray(),
        num_lights: int,
        layer_offsets: ti.types.ndarray(),
        frag_shading: ti.template(), frag_pipelines: ti.template(),
        refraction: ti.template(), skip_unlit_normal: ti.template(),
        ss_enabled: ti.template(), has_bez: ti.template(),
        shadows: ti.template(), prefill: ti.template(),
        covered: ti.template(),
        covered_idx: ti.types.ndarray(), num_covered: int,
        time_start: int, width: int, height: int, tile_start: int,
        cam_origin: ti.types.ndarray(), screen_point: ti.types.ndarray(),
        pixel_basis_x: ti.types.ndarray(), pixel_basis_y: ti.types.ndarray(),
        gen_meta: ti.types.ndarray(),
        rs_ro: ti.types.ndarray(), rs_rd: ti.types.ndarray(),
        rs_acc: ti.types.ndarray(), rs_sca: ti.types.ndarray(),
        rs_int: ti.types.ndarray(), rs_pix: ti.types.ndarray(),
        pix_accum: ti.types.ndarray(), rs_alloc: ti.types.ndarray(),
        frag_shadow_id: ti.types.ndarray(), z_shadow_id: ti.types.ndarray(),
        shadow_vis: ti.types.ndarray()):
    """Resolve + shade each pixel's complete straight-line hit list.

    One thread per tile pixel walks its sorted transparent run front-to-back
    (the z-prepass winner logically appended as the final, terminal hit) with
    the same per-hit logic as ``wavefront_shade``'s drain loop -- triangle
    colour/alpha, fragment-pipeline shading, and the built-in four-way
    reflect/transmit continuation, including split spawns into the shared
    pool. A reflected continuation is written into the pixel's own ray slot
    (status ACTIVE) for the classic wavefront iterations; every other pixel
    retires here, committing colour + leftover background weight into
    ``pix_accum``. Free pool slots must be pre-marked DONE by the host.

    ``prefill`` (compile-time): the host pre-filled every primary's
    ``pix_accum`` row with the retired-empty result ``[0,0,0,0, 1,1,1]``
    (status already DONE), so empty pixels exit before ray generation with
    zero writes, retiring pixels *store* their leftover weight into cols
    4-6, and bouncing pixels zero those columns back out. Byte-identical to
    the accumulate-onto-zero path (``ALGAN_RASTER_EMPTY_SKIP``).

    ``covered`` (compile-time, ``ALGAN_RASTER_COVERED_SHADE``): the host
    passed a compact ascending list ``covered_idx[0:num_covered]`` of the
    only pixels with a fragment or z-winner, so the loop runs one thread per
    covered pixel (``r = covered_idx[t]``) instead of one per tile pixel.
    Empty pixels are simply never launched -- their retired-empty pre-fill
    already holds their final state -- turning the resolve from O(tile
    pixels) into O(covered pixels). Requires ``prefill``; the ascending
    order preserves the original relative shading order (byte-identical).

    ``layer_offsets`` is the tracer's 8-wide variant: [2..5] environment map
    placement, [6] far clip (0 = off), [7] max_bounces.
    """
    pixels_per_frame = width * height
    env_off = ti.cast(layer_offsets[2] + 0.5, ti.i32)
    env_w = ti.cast(layer_offsets[3] + 0.5, ti.i32)
    env_h = ti.cast(layer_offsets[4] + 0.5, ti.i32)
    env_intensity = layer_offsets[5]
    far_clip = layer_offsets[6]
    max_bounces = ti.cast(layer_offsets[7] + 0.5, ti.i32)
    loop_n = num_pixels
    if ti.static(covered):
        loop_n = num_covered
    for t in range(loop_n):
        r = t
        if ti.static(covered):
            r = covered_idx[t]
        start = run_offsets[r]
        end = run_offsets[r + 1]
        nrun = end - start
        zk = zbuf[r]
        has_z = 1 if zk != ti.i64(Z_SENTINEL) else 0
        total = nrun + has_z
        if ti.static(prefill):
            # The host pre-filled every primary's committed state with the
            # retired-empty result (pix_accum row [0,0,0,0, 1,1,1], pool
            # status DONE), so a pixel with no fragments, no z-prepass
            # winner and no environment map to sample is already complete:
            # exit with zero writes, skipping ray generation entirely.
            if total == 0 and env_w <= 0:
                continue

        g = tile_start + r
        f_rel = g // pixels_per_frame
        p = g - f_rel * pixels_per_frame
        f = time_start + f_rel
        py = p // width
        px = p - py * width
        ro, rd = _generate_ray(f, px, py, gen_meta[0], gen_meta[1],
                               gen_meta[2], gen_meta[3],
                               cam_origin, screen_point,
                               pixel_basis_x, pixel_basis_y)

        acc = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
        weight = ti.math.vec3(1.0, 1.0, 1.0)
        seam_t = -1e30
        base_dist = 0.0
        bounces_left = max_bounces
        processed = 0
        bounced = False
        done = False

        q = 0
        while q < total and processed < MAX_SURFACES_PER_RAY:
            t_hit = 0.0
            prim_raw = 0
            a = 0.0
            b = 0.0
            in_border = 0
            from_z = q >= nrun
            idx = start + q
            valid = 1
            if q < nrun:
                t_hit = _frag_t(frag_key[idx])
                prim_raw = frag_ref[idx]
                a = frag_ab[idx, 0]
                b = frag_ab[idx, 1]
            else:
                valid, is_z_bez, zprim, t_hit, a, b, in_border = \
                    _terminal_z_hit(
                        zk, f, px, py, layer_offsets[0], ss_enabled,
                        tri_pos, tri_screen, cam_origin, screen_point,
                        pixel_basis_x, pixel_basis_y, pixel_world_scale,
                        circuit_meta, circuit_colors, edges_2d, edge_accel,
                        gen_meta[2], gen_meta[3])
                prim_raw = zprim
                if is_z_bez:
                    prim_raw = _pack_bez_ref(zprim, in_border)
            q += 1
            if valid == 0:
                continue
            if (far_clip > 0.0) and (base_dist + t_hit > far_clip):
                done = True
                break
            processed += 1

            # A negative packed id tags a bezier circuit fragment; triangles
            # (and the z-winner) are non-negative.
            is_bez = False
            if ti.static(has_bez):
                is_bez = prim_raw < 0
            if is_bez:
                circuit_decoded, border_decoded = _decode_bez_ref(prim_raw)
                prim_raw = circuit_decoded
                in_border = border_decoded

            # Seam de-duplication is triangle-only (shared-edge crossings);
            # edge_hit stays 0 for a circuit, so a bezier hit never skips and
            # simply resets the seam window.
            w0 = 1.0 - a - b
            edge_hit = 0
            if not is_bez:
                if ti.min(w0, ti.min(a, b)) < TRIANGLE_EDGE_EPSILON:
                    edge_hit = 1
            if (edge_hit == 1) and (t_hit - seam_t <= DEPTH_TIE_EPSILON):
                continue
            seam_t = t_hit if edge_hit == 1 else -1e30

            color = ti.math.vec4(0.0, 0.0, 0.0, 0.0)
            alpha = 0.0
            reflectivity = 0.0
            ior = 0.0
            T = 0.0
            albedo3 = ti.math.vec3(0.0, 0.0, 0.0)
            prim = 0
            circuit = 0
            fetched_bez = False
            if ti.static(has_bez):
                if is_bez:
                    fetched_bez = True
                    circuit = prim_raw
                    cm = f % circuit_meta.shape[0]
                    # Circuits keep their sampled colour (never material-shaded).
                    color, alpha = _sample_circuit_color(
                        circuit, f, a, b, in_border, circuit_meta,
                        circuit_colors, circuit_border_colors)
                    albedo3 = ti.math.vec3(color[0], color[1], color[2])
                    reflectivity = circuit_meta[cm, circuit, _M_REFLECTIVITY]
                    ior = circuit_meta[cm, circuit, _M_IOR]
                    T = circuit_meta[cm, circuit, _M_TRANSMISSION]
            if not fetched_bez:
                # Built-in triangle shading + continuation (custom scatter is
                # excluded by the raster gate); port of the drain loop's
                # htype == 1 branch.
                prim = prim_raw
                color, alpha = _tri_color_g(0, f, prim, w0, a, b, tri_colors,
                                            col_row, tri_uvs, tri_tex_meta,
                                            textures, num_colored_triangles)
                reflectivity, _rough = _tri_extra_g(
                    0, f, prim, w0, a, b, tri_extra, col_row, tri_uvs,
                    tri_tex_meta, textures, num_colored_triangles)
                # Raw albedo, saved before fragment shading replaces ``color``.
                albedo3 = ti.math.vec3(color[0], color[1], color[2])
                if ti.static(frag_shading != 0):
                    # Exact sparse shadow-event queue: accepted triangle events
                    # carry an id beside their fragment/terminal winner.  The
                    # dedicated any-hit queue writes one visibility value per
                    # light with no fixed depth or light-count packing limit.
                    vis = ti.Vector([1.0] * MAX_SHADOW_LIGHTS)
                    if ti.static(shadows != 0):
                        event_id = -1
                        if from_z:
                            event_id = z_shadow_id[r]
                        else:
                            event_id = frag_shadow_id[idx]
                        if event_id >= 0:
                            for li in range(num_lights):
                                if li < MAX_SHADOW_LIGHTS:
                                    vis[li] = shadow_vis[event_id, li]
                    sn = ti.math.vec3(0.0, 0.0, 0.0)
                    if ti.static(skip_unlit_normal != 0):
                        if tri_mat_id[f % tri_mat_id.shape[0], prim] \
                                != _MID_UNLIT:
                            sn = _tri_normal_g(
                                0, f, prim, w0, a, b, tri_norm, tri_pos,
                                tri_uvs, tri_tex_meta, textures,
                                num_colored_triangles)
                    else:
                        sn = _tri_normal_g(
                            0, f, prim, w0, a, b, tri_norm, tri_pos, tri_uvs,
                            tri_tex_meta, textures, num_colored_triangles)
                    color = _shade_tri_hit(frag_pipelines, f, prim, a, b, rd,
                                           t_hit, ro, tri_pos, sn,
                                           tri_mat_id, tri_mat,
                                           light_pos, light_col, num_lights,
                                           color, shadows, vis)
                ior = _tri_ior_g(0, f, prim, w0, a, b, tri_extra, col_row,
                                 tri_uvs, tri_tex_meta, textures,
                                 num_colored_triangles)
                T = _tri_transmission_g(0, f, prim, w0, a, b, tri_extra,
                                        col_row, tri_uvs, tri_tex_meta,
                                        textures, num_colored_triangles)

            alpha = ti.math.clamp(alpha, 0.0, 1.0)
            T = ti.math.clamp(T, 0.0, 1.0)

            normal = ti.math.vec3(0.0, 0.0, 0.0)
            if (reflectivity >= 0.0) or (T > 1e-4):
                if fetched_bez:
                    normal = _bezier_normal(
                        f, circuit, circuit_meta).normalized()
                else:
                    normal = _tri_normal_g(
                        0, f, prim, w0, a, b, tri_norm, tri_pos, tri_uvs,
                        tri_tex_meta, textures, num_colored_triangles
                    ).normalized()

            R, diel_pass = _material_reflectance(rd, normal, reflectivity,
                                                 ior, albedo3)
            if bounces_left <= 0:
                R = ti.math.vec3(0.0, 0.0, 0.0)

            # Solid geometry (triangle) refracts (is_glass); a zero-thickness
            # bezier pane transmits unbent and folds the transmitted share into
            # the pass-through (is_pane), same as wavefront_shade.
            is_glass = False
            is_pane = False
            if ti.static(refraction != 0):
                if (T > 1e-4) and (bounces_left > 0) and (ior > 1.0 + 1e-4):
                    if fetched_bez:
                        is_pane = True
                    else:
                        is_glass = True

            one3 = ti.math.vec3(1.0, 1.0, 1.0)
            tint = ti.math.clamp(albedo3, 0.0, 1.0)
            trans_share = diel_pass * T
            r_glow = ti.max(R[0], ti.max(R[1], R[2]))
            w_glow = ti.max(weight[0], ti.max(weight[1], weight[2]))
            share = (weight * alpha) * (one3 - R - trans_share)
            acc += ti.math.vec4(
                share[0], share[1], share[2],
                w_glow * alpha * (1.0 - r_glow - trans_share)) * color
            refl_energy = alpha * R
            refl_max = ti.max(refl_energy[0],
                              ti.max(refl_energy[1], refl_energy[2]))
            trans_energy = alpha * trans_share
            cover_pass = 1.0 - alpha
            cover3 = ti.math.vec3(cover_pass, cover_pass, cover_pass)

            split_refl = False
            if ti.static(refraction != 0):
                if (refl_max > MIN_ALPHA) and (cover_pass > MIN_ALPHA) \
                        and (bounces_left > 0):
                    split_refl = True

            if is_glass:
                wt = weight * trans_energy * tint
                wt_max = ti.max(wt[0], ti.max(wt[1], wt[2]))
                if wt_max > MIN_WEIGHT:
                    c, have_slot = _reserve_continuation_slot(
                        rs_alloc, rs_ro.shape[0])
                    if have_slot:
                        rdt = _refract_ray(rd, normal, ior)
                        hp = ro + t_hit * rd
                        tp = f % tri_pos.shape[0]
                        v0 = ti.math.vec3(tri_pos[tp, prim, 0],
                                          tri_pos[tp, prim, 1],
                                          tri_pos[tp, prim, 2])
                        v1 = ti.math.vec3(tri_pos[tp, prim, 3],
                                          tri_pos[tp, prim, 4],
                                          tri_pos[tp, prim, 5])
                        v2 = ti.math.vec3(tri_pos[tp, prim, 6],
                                          tri_pos[tp, prim, 7],
                                          tri_pos[tp, prim, 8])
                        face_normal = (v1 - v0).cross(v2 - v0)
                        rorig = _offset_transmitted_origin(
                            hp, rdt, face_normal, normal)
                        for k in ti.static(range(3)):
                            rs_ro[c, k] = rorig[k]
                            rs_rd[c, k] = rdt[k]
                        for k in ti.static(range(4)):
                            rs_acc[c, k] = 0.0
                        rs_sca[c, 0] = wt[0]
                        rs_sca[c, 1] = 0.0
                        rs_sca[c, 2] = 1e30
                        rs_sca[c, 3] = -1e30
                        rs_sca[c, 4] = base_dist + t_hit
                        rs_sca[c, 5] = wt[1]
                        rs_sca[c, 6] = wt[2]
                        rs_int[c, 0] = bounces_left - 1
                        rs_int[c, 1] = processed
                        rs_int[c, 2] = _ACTIVE
                        rs_int[c, 3] = 0
                        rs_pix[c] = r
                if (refl_max > MIN_ALPHA) and (refl_max >= cover_pass):
                    nref = normal
                    if nref.dot(rd) > 0.0:
                        nref = -nref
                    hit_point = ro + t_hit * rd
                    rd = (rd - 2.0 * rd.dot(nref) * nref).normalized()
                    ro = hit_point + nref * (10.0 * MIN_HIT_DISTANCE)
                    weight *= refl_energy
                    base_dist += t_hit
                    seam_t = -1e30
                    bounces_left -= 1
                    bounced = True
                    break
                else:
                    weight *= cover_pass
            elif is_pane or split_refl:
                # Thin pane (bezier) or semi-transparent reflector: reflection
                # into a split slot, pass-through (incl. any unbent transmitted
                # share) continues in place.
                wt = weight * refl_energy
                wt_max = ti.max(wt[0], ti.max(wt[1], wt[2]))
                if wt_max > MIN_WEIGHT:
                    c, have_slot = _reserve_continuation_slot(
                        rs_alloc, rs_ro.shape[0])
                    if have_slot:
                        nref = normal
                        if nref.dot(rd) > 0.0:
                            nref = -nref
                        rdr = (rd - 2.0 * rd.dot(nref) * nref).normalized()
                        hp = ro + t_hit * rd
                        for k in ti.static(range(3)):
                            rs_ro[c, k] = (hp[k] + nref[k]
                                           * (10.0 * MIN_HIT_DISTANCE))
                            rs_rd[c, k] = rdr[k]
                        for k in ti.static(range(4)):
                            rs_acc[c, k] = 0.0
                        rs_sca[c, 0] = wt[0]
                        rs_sca[c, 1] = 0.0
                        rs_sca[c, 2] = 1e30
                        rs_sca[c, 3] = -1e30
                        rs_sca[c, 4] = base_dist + t_hit
                        rs_sca[c, 5] = wt[1]
                        rs_sca[c, 6] = wt[2]
                        rs_int[c, 0] = bounces_left - 1
                        rs_int[c, 1] = processed
                        rs_int[c, 2] = _ACTIVE
                        rs_int[c, 3] = 0
                        rs_pix[c] = r
                weight *= cover3 + trans_energy * tint
            elif (refl_max > MIN_ALPHA) and (refl_max >= cover_pass):
                nref = normal
                if nref.dot(rd) > 0.0:
                    nref = -nref
                hit_point = ro + t_hit * rd
                rd = (rd - 2.0 * rd.dot(nref) * nref).normalized()
                ro = hit_point + nref * (10.0 * MIN_HIT_DISTANCE)
                weight *= refl_energy
                base_dist += t_hit
                seam_t = -1e30
                bounces_left -= 1
                bounced = True
                break
            else:
                weight *= cover3 + trans_energy * tint

            if ti.max(weight[0], ti.max(weight[1], weight[2])) < MIN_WEIGHT:
                done = True
                break

        if processed >= MAX_SURFACES_PER_RAY:
            done = True

        if bounced and not done:
            if ti.static(prefill):
                # Undo the host's retired-empty pre-fill: an ACTIVE pixel's
                # leftover background weight is committed by its continuation
                # when that retires, so the base must return to zero.
                for k in ti.static(range(3)):
                    pix_accum[r, 4 + k] = 0.0
            for k in ti.static(range(3)):
                rs_ro[r, k] = ro[k]
                rs_rd[r, k] = rd[k]
            for k in ti.static(range(4)):
                rs_acc[r, k] = acc[k]
            rs_sca[r, 0] = weight[0]
            rs_sca[r, 1] = 0.0
            rs_sca[r, 2] = 1e30
            rs_sca[r, 3] = -1e30
            rs_sca[r, 4] = base_dist
            rs_sca[r, 5] = weight[1]
            rs_sca[r, 6] = weight[2]
            rs_int[r, 0] = bounces_left
            rs_int[r, 1] = processed
            rs_int[r, 2] = _ACTIVE
            rs_int[r, 3] = 0
            rs_pix[r] = r
        else:
            # Retired: the straight-line list is complete, so a pixel that
            # neither bounced nor ran out of throughput has simply seen every
            # surface -- commit and show the background through the leftover.
            if (env_w > 0) and (ti.max(weight[0], ti.max(
                    weight[1], weight[2])) > 0.0):
                ec = _sample_env_map(f, rd, env_off, env_w, env_h,
                                     env_intensity, textures)
                for k in ti.static(range(3)):
                    acc[k] += weight[k] * ec[k]
                weight = ti.math.vec3(0.0, 0.0, 0.0)
            for k in ti.static(range(4)):
                ti.atomic_add(pix_accum[r, k], acc[k])
            if ti.static(prefill):
                # Cols 4-6 were pre-filled with 1.0, so the leftover weight
                # is stored rather than accumulated (this thread is the
                # pixel's only writer during iteration zero; the colour adds
                # above keep their zero base).
                for k in ti.static(range(3)):
                    pix_accum[r, 4 + k] = weight[k]
            else:
                for k in ti.static(range(3)):
                    ti.atomic_add(pix_accum[r, 4 + k], weight[k])
            rs_int[r, 2] = _DONE
