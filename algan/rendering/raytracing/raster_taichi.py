"""Hybrid raster front-end kernels (raytracer-v2; flat triangles + bezier).

These kernels replace the deterministic wavefront's *first* iteration
(primary-ray generate + traverse + shade) with primitive-side candidate
enumeration -- see ``settings.HYBRID_RASTER`` and
``raster_pipeline.raster_iteration_zero`` for the host orchestration:

Each (prim, chunk) pair projects its triangle to screen space once
(:func:`_ss_setup`) and tests candidate pixels with edge functions +
perspective-correct barycentric interpolation (:func:`_ss_pixel`), moving the
projection/normalization out of the per-pixel loop; a triangle straddling the
camera plane falls back to per-pixel ray casting (:func:`_raycast_pixel`).
Screen-space is numerically equivalent to the ray cast (verified worst
|dt| ~5e-5, |d_bary| ~6e-5) and toggled by ``settings.RASTER_SS``.

* :func:`raster_tri_z` -- opaque candidate pairs: exact per-pixel test,
  nearest opaque hit kept per pixel via a packed ``(depth_bits << 32 | prim)``
  int64 ``atomicMin`` (min is commutative, so the winner is deterministic
  regardless of thread order; equal-depth ties resolve to the lower primitive
  id).
* :func:`raster_tri_count` / :func:`raster_tri_write` -- transparent
  candidate pairs, culled against the finished z-buffer: a count pass sizes
  the fragment list exactly (deterministic layout, no atomic append), then a
  write pass emits ``(local_pixel << 32 | depth_bits)`` sort keys plus the
  hit payload (t, prim, barycentrics). The host sorts by key (torch/cub
  radix) -- raw f32 depth order, no DEPTH_TIE_EPSILON binning.
* :func:`raster_first_shade` -- one thread per tile pixel: walks its sorted
  fragment run front-to-back with the same shading/continuation logic as
  ``wavefront_shade``'s drain loop (colour/alpha fetch, fragment pipelines,
  built-in reflect/refract continuation incl. pool splits), appends the
  z-prepass winner as the terminal hit, commits colour + leftover background
  weight into ``pix_accum``, and writes a bounced continuation into the
  pixel's own ray slot for the classic wavefront iterations to pick up.

A pair row (see ``raster_pipeline``) is ``[prim, f, x0, y0, bw, bh, off, 0]``
(int32): a chunk of up to ``RASTER_CHUNK`` candidate pixels, row-major within
the primitive's clipped screen bbox starting at flattened offset ``off``.

Bezier circuits (2D shapes / text) are rasterized too (``raster_bez_count`` /
``raster_bez_write`` + ``_bez_pair_pixel``): they project their per-frame world
AABB to a screen bbox and, per candidate pixel, do a ray/plane intersection +
:func:`_bezier_point_metrics` inside test. Circuits route entirely through the
transparent (sorted) fragment stream -- never the z-prepass -- tagged by a
negative ``frag_prim`` (= ``-(circuit + 1)``) with plane coords in ``frag_ab``
and the in-border flag in ``frag_flags``; ``raster_first_shade`` decodes the
sign and shades circuits with their sampled colour + the thin-pane continuation.
PN patches are NOT rasterized (raster forces surfaces to flat triangles).

Not byte-identical to the classic wavefront (raw-depth hit order, strict
opaque culling); shadows/custom-scatter/mem-trim/PN excluded -- the tracer's
``use_raster`` gate enforces this.
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
from algan.rendering.raytracing.wavefront_kernels_taichi import (
    _ACTIVE,
    _DONE,
    _LT_AMBIENT,
    _LT_DIRECTIONAL,
    _LT_ENV_SH,
    _LT_HEMISPHERE,
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

# Empty z-buffer entry: any packed (depth_bits << 32 | prim) of a real hit
# compares below it (finite positive f32 bits < 0x7F800000 << 32).
Z_SENTINEL = 0x7FFFFFFFFFFFFFFF

# Deferred hard-shadow packing (see raster_shadow / raster_first_shade). One
# int32 ``rs_vis[r]`` per pixel holds occlusion bits for up to
# ``_RASTER_SHADOW_SLOTS`` shading points along the pixel's straight-line hit
# list x ``_RASTER_SHADOW_LIGHTS`` lights (bit ``slot * LIGHTS + li``). The
# terminal opaque z-hit -- the dominant shadow receiver -- always owns the last
# slot; the nearest transparent triangle fragments own slots 0.. below it, and
# any lit fragment past the budget is treated as unshadowed (a rare deep
# translucent stack). Only hard point/directional lights are packed; the tracer
# gate keeps soft-shadow and many-light scenes on the classic path.
_RASTER_SHADOW_SLOTS = 4
_RASTER_SHADOW_LIGHTS = max(1, min(MAX_SHADOW_LIGHTS, 32 // _RASTER_SHADOW_SLOTS))


@ti.func
def _raster_shadow_slot(q_pos, nrun):
    """Shadow-bit slot for the fragment at walk position ``q_pos`` (the
    terminal opaque z-hit is at ``q_pos == nrun``). Returns -1 for fragments
    outside the packed budget (left unshadowed). Shared by the pre-pass and the
    resolve so both index the same bits.
    """
    slot = -1
    if q_pos < nrun:
        if q_pos < _RASTER_SHADOW_SLOTS - 1:
            slot = q_pos
    else:
        slot = _RASTER_SHADOW_SLOTS - 1
    return slot


@ti.func
def _ss_setup(f, prim, ss_enabled: ti.template(), tri_pos: ti.template(),
              cam_origin: ti.template(), screen_point: ti.template(),
              pixel_basis_x: ti.template(), pixel_basis_y: ti.template(),
              half_w, half_h):
    """Per-triangle screen-space setup, computed once per (prim, chunk) pair.

    Projects the triangle's three vertices to continuous pixel coordinates
    (the exact forward of ``_generate_ray``) and records the perspective
    divisor ``1/d_i`` (``d_i = dot(V_i - cam_o, n)``, ``n = pbx x pby`` the
    screen plane normal). Returns ``(use_ss, sm, vm, cam_o)``:

    * ``use_ss`` -- 1 if all vertices are strictly in front of the camera plane
      (the projective rasterization is valid); 0 forces the per-pixel ray-cast
      fallback (a triangle straddling the camera plane).
    * ``sm`` -- 3x3: row 0 = the three ``sx``, row 1 = the three ``sy``, row 2
      = the three ``1/d_i``.
    * ``vm`` -- 3x3: row i = vertex ``V_i`` (world), shared by both paths.
    * ``cam_o`` -- camera origin for this frame.
    """
    tp = f % tri_pos.shape[0]
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
        sp = ti.math.vec3(screen_point[f, 0], screen_point[f, 1],
                          screen_point[f, 2])
        pbx = ti.math.vec3(pixel_basis_x[f, 0], pixel_basis_x[f, 1],
                           pixel_basis_x[f, 2])
        pby = ti.math.vec3(pixel_basis_y[f, 0], pixel_basis_y[f, 1],
                           pixel_basis_y[f, 2])
        n = pbx.cross(pby)
        n2 = n.dot(n)
        big_d = (sp - cam_o).dot(n)
        if (n2 > 1e-30) and (ti.abs(big_d) > 1e-20):
            inv_n2 = 1.0 / n2
            sd = 1.0 if big_d >= 0.0 else -1.0
            d0 = (v0 - cam_o).dot(n)
            d1 = (v1 - cam_o).dot(n)
            d2 = (v2 - cam_o).dot(n)
            # All vertices strictly on the screen side (same sign as big_d, not
            # on the camera plane) -> the projective map is finite everywhere
            # in the triangle. Otherwise fall back to ray casting.
            if (d0 * sd > 1e-9) and (d1 * sd > 1e-9) and (d2 * sd > 1e-9):
                use_ss = 1
                for i in ti.static(range(3)):
                    vi = ti.math.vec3(vm[i, 0], vm[i, 1], vm[i, 2])
                    di = (vi - cam_o).dot(n)
                    rel = (cam_o - sp) + (vi - cam_o) * (big_d / di)
                    u = rel.cross(pby).dot(n) * inv_n2
                    vv = pbx.cross(rel).dot(n) * inv_n2
                    sm[0, i] = u * half_h + half_w
                    sm[1, i] = vv * half_h + half_h
                    sm[2, i] = 1.0 / di
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
def _bez_pair_pixel(circuit, f, x0, y0, bw, bh, off, j,
                    time_start, width, height, tile_start, tile_pixels,
                    half_w, half_h,
                    cam_origin: ti.template(), screen_point: ti.template(),
                    pixel_basis_x: ti.template(), pixel_basis_y: ti.template(),
                    pixel_world_scale: ti.template(),
                    circuit_meta: ti.template(), circuit_colors: ti.template(),
                    edges_2d: ti.template(), edge_accel: ti.template()):
    """Test candidate ``off + j`` against one bezier circuit at its pixel
    center: ray/plane intersection (the circuit is planar) then the inside /
    border test via :func:`_bezier_point_metrics`. Ports the bezier branch of
    ``_collect_hits``; ``base_dist`` is 0 for primaries so the screen-constant
    border width uses ``t`` directly. Returns
    ``(ok, local_pixel, t, u, v, in_border)``.
    """
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
                    border_w = circuit_meta[tm, circuit, _M_BORDER_W] \
                        * pixel_size
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
                        lp = lpi
                        t = th
                        u = uu
                        v = vv
                        in_border = 1 if is_border else 0
    return ok, lp, t, u, v, in_border


@ti.kernel
def raster_tri_z(
        pairs: ti.types.ndarray(), num_pairs: int,
        tri_pos: ti.types.ndarray(),
        cam_origin: ti.types.ndarray(), screen_point: ti.types.ndarray(),
        pixel_basis_x: ti.types.ndarray(), pixel_basis_y: ti.types.ndarray(),
        time_start: int, width: int, height: int,
        half_w: ti.f32, half_h: ti.f32,
        tile_start: int, tile_pixels: int,
        ss_enabled: ti.template(),
        zbuf: ti.types.ndarray()):
    """Opaque z-prepass: nearest proven-opaque hit per pixel by atomicMin."""
    for p in range(num_pairs):
        prim = pairs[p, 0]
        f = pairs[p, 1]
        x0 = pairs[p, 2]
        y0 = pairs[p, 3]
        bw = pairs[p, 4]
        bh = pairs[p, 5]
        off = pairs[p, 6]
        use_ss, sm, vm, cam_o = _ss_setup(
            f, prim, ss_enabled, tri_pos, cam_origin, screen_point,
            pixel_basis_x, pixel_basis_y, half_w, half_h)
        for j in range(RASTER_CHUNK):
            ok, lp, t, w1, w2 = _pair_pixel(
                prim, f, x0, y0, bw, bh, off, j, time_start, width, height,
                tile_start, tile_pixels, half_w, half_h, use_ss, sm, vm, cam_o,
                cam_origin, screen_point, pixel_basis_x, pixel_basis_y)
            if ok != 0:
                tb = ti.cast(ti.bit_cast(t, ti.u32), ti.i64)
                key = (tb << 32) | ti.cast(prim, ti.i64)
                ti.atomic_min(zbuf[lp], key)


@ti.kernel
def raster_tri_count(
        pairs: ti.types.ndarray(), num_pairs: int,
        tri_pos: ti.types.ndarray(),
        cam_origin: ti.types.ndarray(), screen_point: ti.types.ndarray(),
        pixel_basis_x: ti.types.ndarray(), pixel_basis_y: ti.types.ndarray(),
        time_start: int, width: int, height: int,
        half_w: ti.f32, half_h: ti.f32,
        tile_start: int, tile_pixels: int,
        ss_enabled: ti.template(),
        zbuf: ti.types.ndarray(), pair_count: ti.types.ndarray()):
    """Count each transparent pair's surviving hits (strictly nearer than the
    pixel's opaque winner), sizing the fragment list exactly.
    """
    for p in range(num_pairs):
        prim = pairs[p, 0]
        f = pairs[p, 1]
        x0 = pairs[p, 2]
        y0 = pairs[p, 3]
        bw = pairs[p, 4]
        bh = pairs[p, 5]
        off = pairs[p, 6]
        use_ss, sm, vm, cam_o = _ss_setup(
            f, prim, ss_enabled, tri_pos, cam_origin, screen_point,
            pixel_basis_x, pixel_basis_y, half_w, half_h)
        cnt = 0
        for j in range(RASTER_CHUNK):
            ok, lp, t, w1, w2 = _pair_pixel(
                prim, f, x0, y0, bw, bh, off, j, time_start, width, height,
                tile_start, tile_pixels, half_w, half_h, use_ss, sm, vm, cam_o,
                cam_origin, screen_point, pixel_basis_x, pixel_basis_y)
            if ok != 0:
                tb = ti.cast(ti.bit_cast(t, ti.u32), ti.i64)
                if (zbuf[lp] >> 32) > tb:
                    cnt += 1
        pair_count[p] = cnt


@ti.kernel
def raster_tri_write(
        pairs: ti.types.ndarray(), num_pairs: int,
        pair_offset: ti.types.ndarray(),
        tri_pos: ti.types.ndarray(),
        cam_origin: ti.types.ndarray(), screen_point: ti.types.ndarray(),
        pixel_basis_x: ti.types.ndarray(), pixel_basis_y: ti.types.ndarray(),
        time_start: int, width: int, height: int,
        half_w: ti.f32, half_h: ti.f32,
        tile_start: int, tile_pixels: int,
        ss_enabled: ti.template(),
        zbuf: ti.types.ndarray(),
        frag_key: ti.types.ndarray(), frag_t: ti.types.ndarray(),
        frag_prim: ti.types.ndarray(), frag_ab: ti.types.ndarray(),
        frag_flags: ti.types.ndarray()):
    """Emit the surviving transparent triangle fragments at each pair's exact
    offset (same tests as the count pass, so the layout is deterministic).
    ``frag_prim >= 0`` marks a triangle fragment (bezier stores the negated
    circuit id -- see :func:`raster_bez_write`); ``frag_flags`` is 0 for
    triangles (their edge/seam flag is derived from the barycentrics).
    """
    for p in range(num_pairs):
        prim = pairs[p, 0]
        f = pairs[p, 1]
        x0 = pairs[p, 2]
        y0 = pairs[p, 3]
        bw = pairs[p, 4]
        bh = pairs[p, 5]
        off = pairs[p, 6]
        use_ss, sm, vm, cam_o = _ss_setup(
            f, prim, ss_enabled, tri_pos, cam_origin, screen_point,
            pixel_basis_x, pixel_basis_y, half_w, half_h)
        w = pair_offset[p]
        for j in range(RASTER_CHUNK):
            ok, lp, t, w1, w2 = _pair_pixel(
                prim, f, x0, y0, bw, bh, off, j, time_start, width, height,
                tile_start, tile_pixels, half_w, half_h, use_ss, sm, vm, cam_o,
                cam_origin, screen_point, pixel_basis_x, pixel_basis_y)
            if ok != 0:
                tb = ti.cast(ti.bit_cast(t, ti.u32), ti.i64)
                if (zbuf[lp] >> 32) > tb:
                    frag_key[w] = (ti.cast(lp, ti.i64) << 32) | tb
                    frag_t[w] = t
                    frag_prim[w] = prim
                    frag_ab[w, 0] = w1
                    frag_ab[w, 1] = w2
                    frag_flags[w] = 0
                    w += 1


@ti.kernel
def raster_bez_count(
        pairs: ti.types.ndarray(), num_pairs: int,
        cam_origin: ti.types.ndarray(), screen_point: ti.types.ndarray(),
        pixel_basis_x: ti.types.ndarray(), pixel_basis_y: ti.types.ndarray(),
        pixel_world_scale: ti.types.ndarray(),
        circuit_meta: ti.types.ndarray(), circuit_colors: ti.types.ndarray(),
        edges_2d: ti.types.ndarray(), edge_accel: ti.types.ndarray(),
        time_start: int, width: int, height: int,
        half_w: ti.f32, half_h: ti.f32,
        tile_start: int, tile_pixels: int,
        zbuf: ti.types.ndarray(), pair_count: ti.types.ndarray()):
    """Count each bezier pair's surviving hits (nearer than the pixel's opaque
    triangle winner in ``zbuf``). Bezier is never in the z-prepass, so every
    in/border hit that clears the triangle z-buffer becomes a fragment.
    """
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
            if ok != 0:
                tb = ti.cast(ti.bit_cast(t, ti.u32), ti.i64)
                if (zbuf[lp] >> 32) > tb:
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
        edges_2d: ti.types.ndarray(), edge_accel: ti.types.ndarray(),
        time_start: int, width: int, height: int,
        half_w: ti.f32, half_h: ti.f32,
        tile_start: int, tile_pixels: int,
        zbuf: ti.types.ndarray(),
        frag_key: ti.types.ndarray(), frag_t: ti.types.ndarray(),
        frag_prim: ti.types.ndarray(), frag_ab: ti.types.ndarray(),
        frag_flags: ti.types.ndarray()):
    """Emit surviving bezier fragments. ``frag_prim = -(circuit + 1)`` tags the
    fragment as a circuit (the resolve decodes the sign); ``frag_ab`` holds the
    plane coords (u, v) and ``frag_flags`` bit 0 the in-border flag.
    """
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
            if ok != 0:
                tb = ti.cast(ti.bit_cast(t, ti.u32), ti.i64)
                if (zbuf[lp] >> 32) > tb:
                    frag_key[w] = (ti.cast(lp, ti.i64) << 32) | tb
                    frag_t[w] = t
                    frag_prim[w] = -(circuit + 1)
                    frag_ab[w, 0] = u
                    frag_ab[w, 1] = v
                    frag_flags[w] = ib
                    w += 1


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


@ti.func
def _hard_shadow_bits(spos, snrm, fnrm, f, ff, pixel_size_per_t, base_dist,
                      layer_offset_triangles, layer_offset_pn,
                      has_tri: ti.template(), has_pn: ti.template(),
                      has_bez: ti.template(),
                      t_nodes: ti.template(), t_node_miss: ti.template(),
                      t_leaf_prim: ti.template(), t_leaf_tspan: ti.template(),
                      t_first_leaf, tri_pos: ti.template(),
                      tri_colors: ti.template(), tri_uvs: ti.template(),
                      tri_tex_meta: ti.template(), textures: ti.template(),
                      num_colored_triangles,
                      p_nodes: ti.template(), p_node_miss: ti.template(),
                      p_leaf_prim: ti.template(), p_leaf_tspan: ti.template(),
                      p_first_leaf, pn_ctrl: ti.template(),
                      pn_obb: ti.template(), pn_colors: ti.template(),
                      b_nodes: ti.template(), b_node_miss: ti.template(),
                      b_leaf_prim: ti.template(), b_leaf_tspan: ti.template(),
                      b_first_leaf, circuit_meta: ti.template(),
                      circuit_colors: ti.template(),
                      circuit_border_colors: ti.template(),
                      edges_2d: ti.template(), edge_accel: ti.template(),
                      light_pos: ti.template(), light_col: ti.template(),
                      num_lights):
    """Binary hard-shadow bits for one triangle shading point: bit ``li`` set
    when light ``li`` (``li < _RASTER_SHADOW_LIGHTS``) is occluded. Mirrors the
    classic ``wavefront_shade`` inline hard-shadow path (a single ray per light,
    directional/point handling identical); soft-shadow lights are kept on the
    classic path by the tracer gate, so this never fans samples.
    """
    bits = 0
    sorigin = spos + fnrm * (10.0 * MIN_HIT_DISTANCE)
    tl = f % light_pos.shape[0]
    for li in range(num_lights):
        if li < _RASTER_SHADOW_LIGHTS:
            lp = ti.math.vec3(light_pos[tl, li, 0], light_pos[tl, li, 1],
                              light_pos[tl, li, 2])
            ltype = 0
            if light_col.shape[2] > 3:
                ltype = ti.cast(light_col[tl, li, 3] + 0.5, ti.i32)
            to_light = lp - spos
            ldist = to_light.norm()
            wi = ti.math.vec3(0.0, 0.0, 0.0)
            valid = 0
            if ltype == _LT_DIRECTIONAL:
                wi = -ti.math.vec3(light_col[tl, li, 6], light_col[tl, li, 7],
                                   light_col[tl, li, 8])
                ldist = 1e7
                valid = 1
            elif (ltype != _LT_AMBIENT) and (ltype != _LT_HEMISPHERE) \
                    and (ltype != _LT_ENV_SH) and (ldist > 1e-5):
                wi = to_light / ldist
                valid = 1
            if valid == 1:
                if (fnrm.dot(wi) > 1e-3) and (snrm.dot(wi) > 1e-4):
                    occ = _shadow_occluded(
                        sorigin, wi, f, ff,
                        ldist - 20.0 * MIN_HIT_DISTANCE,
                        pixel_size_per_t, base_dist,
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
                    if occ > 0.5:
                        bits |= (1 << li)
    return bits


@ti.kernel
def raster_shadow(
        num_pixels: int,
        run_start: ti.types.ndarray(), run_len: ti.types.ndarray(),
        frag_t: ti.types.ndarray(), frag_prim: ti.types.ndarray(),
        frag_ab: ti.types.ndarray(),
        zbuf: ti.types.ndarray(),
        # Full (all-primitive) STBVHs + geometry for the any-hit shadow trace,
        # exactly as the classic wavefront_shade consumes for inline shadows.
        t_nodes: NODE_ARG, t_node_miss: ti.types.ndarray(),
        t_leaf_prim: ti.types.ndarray(), t_leaf_tspan: ti.types.ndarray(),
        t_first_leaf: int,
        tri_pos: ti.types.ndarray(), tri_norm: ti.types.ndarray(),
        tri_colors: ti.types.ndarray(), tri_uvs: ti.types.ndarray(),
        tri_tex_meta: ti.types.ndarray(), textures: ti.types.ndarray(),
        num_colored_triangles: ti.i32,
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
        num_lights: int,
        pixel_world_scale: ti.types.ndarray(),
        layer_offset_triangles: ti.f32, layer_offset_pn: ti.f32,
        has_tri: ti.template(), has_pn: ti.template(), has_bez: ti.template(),
        ss_enabled: ti.template(),
        time_start: int, width: int, height: int, tile_start: int,
        cam_origin: ti.types.ndarray(), screen_point: ti.types.ndarray(),
        pixel_basis_x: ti.types.ndarray(), pixel_basis_y: ti.types.ndarray(),
        gen_meta: ti.types.ndarray(),
        rs_vis: ti.types.ndarray()):
    """Deferred hard-shadow pre-pass: one thread per tile pixel casts shadow
    rays for the (up to ``_RASTER_SHADOW_SLOTS``) shading points on its
    straight-line hit list -- the nearest transparent triangle fragments plus
    the terminal opaque z-hit -- and packs per-light occlusion bits into
    ``rs_vis[r]`` for ``raster_first_shade`` to read. Bezier fragments never
    receive shadows (they keep their sampled colour) but bezier geometry still
    occludes shadow rays (``has_bez``). Base distance is 0 (primary iteration).
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
                               gen_meta[2], gen_meta[3],
                               cam_origin, screen_point,
                               pixel_basis_x, pixel_basis_y)
        ff = ti.cast(f, ti.f32)
        pixel_size_per_t = pixel_world_scale[f]
        bits = 0
        nrun = run_len[r]
        start = run_start[r]
        zk = zbuf[r]
        has_z = 0
        if (zk >> 32) < 0x7F800000:
            has_z = 1

        # Nearest transparent triangle fragments (slots 0 .. SLOTS-2).
        nt = ti.min(nrun, _RASTER_SHADOW_SLOTS - 1)
        for qi in range(nt):
            idx = start + qi
            prim = frag_prim[idx]
            if prim >= 0:
                a = frag_ab[idx, 0]
                b = frag_ab[idx, 1]
                t_hit = frag_t[idx]
                snrm, fnrm = _tri_shadow_normals(
                    f, prim, a, b, rd, tri_pos, tri_norm, tri_uvs,
                    tri_tex_meta, textures, num_colored_triangles)
                spos = ro + t_hit * rd
                sb = _hard_shadow_bits(
                    spos, snrm, fnrm, f, ff, pixel_size_per_t, 0.0,
                    layer_offset_triangles, layer_offset_pn,
                    has_tri, has_pn, has_bez,
                    t_nodes, t_node_miss, t_leaf_prim, t_leaf_tspan,
                    t_first_leaf, tri_pos, tri_colors, tri_uvs,
                    tri_tex_meta, textures, num_colored_triangles,
                    p_nodes, p_node_miss, p_leaf_prim, p_leaf_tspan,
                    p_first_leaf, pn_ctrl, pn_obb, pn_colors,
                    b_nodes, b_node_miss, b_leaf_prim, b_leaf_tspan,
                    b_first_leaf, circuit_meta, circuit_colors,
                    circuit_border_colors, edges_2d, edge_accel,
                    light_pos, light_col, num_lights)
                bits |= (sb << (qi * _RASTER_SHADOW_LIGHTS))

        # Terminal opaque z-hit (slot SLOTS-1): always a triangle. Recompute the
        # barycentrics via the same screen-space / ray-cast path that found the
        # winner, matching the resolve.
        if has_z == 1:
            prim = ti.cast(zk & 0x7FFFFFFF, ti.i32)
            t_hit = ti.bit_cast(ti.cast(zk >> 32, ti.u32), ti.f32)
            use_ss, sm, vm, cam_ow = _ss_setup(
                f, prim, ss_enabled, tri_pos, cam_origin, screen_point,
                pixel_basis_x, pixel_basis_y, gen_meta[2], gen_meta[3])
            a = 0.0
            b = 0.0
            if use_ss != 0:
                _hok, _ht, a, b = _ss_pixel(px, py, sm, vm, cam_ow)
            else:
                _hok, _ht, a, b = _raycast_pixel(
                    px, py, f, vm, gen_meta[2], gen_meta[3], cam_origin,
                    screen_point, pixel_basis_x, pixel_basis_y)
            snrm, fnrm = _tri_shadow_normals(
                f, prim, a, b, rd, tri_pos, tri_norm, tri_uvs,
                tri_tex_meta, textures, num_colored_triangles)
            spos = ro + t_hit * rd
            sb = _hard_shadow_bits(
                spos, snrm, fnrm, f, ff, pixel_size_per_t, 0.0,
                layer_offset_triangles, layer_offset_pn,
                has_tri, has_pn, has_bez,
                t_nodes, t_node_miss, t_leaf_prim, t_leaf_tspan,
                t_first_leaf, tri_pos, tri_colors, tri_uvs,
                tri_tex_meta, textures, num_colored_triangles,
                p_nodes, p_node_miss, p_leaf_prim, p_leaf_tspan,
                p_first_leaf, pn_ctrl, pn_obb, pn_colors,
                b_nodes, b_node_miss, b_leaf_prim, b_leaf_tspan,
                b_first_leaf, circuit_meta, circuit_colors,
                circuit_border_colors, edges_2d, edge_accel,
                light_pos, light_col, num_lights)
            bits |= (sb << ((_RASTER_SHADOW_SLOTS - 1) * _RASTER_SHADOW_LIGHTS))

        rs_vis[r] = bits


@ti.kernel
def raster_first_shade(
        num_pixels: int,
        run_start: ti.types.ndarray(), run_len: ti.types.ndarray(),
        frag_t: ti.types.ndarray(), frag_prim: ti.types.ndarray(),
        frag_ab: ti.types.ndarray(), frag_flags: ti.types.ndarray(),
        zbuf: ti.types.ndarray(),
        tri_pos: ti.types.ndarray(), tri_norm: ti.types.ndarray(),
        tri_extra: ti.types.ndarray(), tri_colors: ti.types.ndarray(),
        tri_uvs: ti.types.ndarray(), tri_tex_meta: ti.types.ndarray(),
        textures: ti.types.ndarray(), num_colored_triangles: ti.i32,
        col_row: ti.types.ndarray(),
        tri_mat_id: ti.types.ndarray(), tri_mat: ti.types.ndarray(),
        # Bezier circuit shading data (used when has_bez; 1x1 placeholders
        # otherwise). Circuits route entirely through the transparent fragment
        # stream (never the z-prepass), tagged by a negative frag_prim.
        circuit_meta: ti.types.ndarray(), circuit_colors: ti.types.ndarray(),
        circuit_border_colors: ti.types.ndarray(),
        light_pos: ti.types.ndarray(), light_col: ti.types.ndarray(),
        num_lights: int,
        layer_offsets: ti.types.ndarray(),
        frag_shading: ti.template(), frag_pipelines: ti.template(),
        refraction: ti.template(), skip_unlit_normal: ti.template(),
        ss_enabled: ti.template(), has_bez: ti.template(),
        shadows: ti.template(),
        time_start: int, width: int, height: int, tile_start: int,
        cam_origin: ti.types.ndarray(), screen_point: ti.types.ndarray(),
        pixel_basis_x: ti.types.ndarray(), pixel_basis_y: ti.types.ndarray(),
        gen_meta: ti.types.ndarray(),
        rs_ro: ti.types.ndarray(), rs_rd: ti.types.ndarray(),
        rs_acc: ti.types.ndarray(), rs_sca: ti.types.ndarray(),
        rs_int: ti.types.ndarray(), rs_pix: ti.types.ndarray(),
        pix_accum: ti.types.ndarray(), rs_alloc: ti.types.ndarray(),
        rs_vis: ti.types.ndarray()):
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
    for r in range(num_pixels):
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

        nrun = run_len[r]
        start = run_start[r]
        zk = zbuf[r]
        has_z = 0
        # The sentinel's high half is 0x7FFFFFFF; a real hit's high half is
        # its finite positive f32 depth bits, always below the +inf pattern.
        if (zk >> 32) < 0x7F800000:
            has_z = 1
        total = nrun + has_z

        q = 0
        while q < total:
            t_hit = 0.0
            prim_raw = 0
            a = 0.0
            b = 0.0
            in_border = 0
            if q < nrun:
                idx = start + q
                t_hit = frag_t[idx]
                prim_raw = frag_prim[idx]
                a = frag_ab[idx, 0]
                b = frag_ab[idx, 1]
                in_border = frag_flags[idx] & 1
            else:
                # Terminal opaque hit from the z-prepass (always a triangle --
                # bezier never enters the z-buffer). ``t_hit`` is the exact
                # packed depth; recompute only the barycentrics, via the same
                # screen-space (or ray-cast fallback) path that found the
                # winner so they are consistent.
                prim_raw = ti.cast(zk & 0x7FFFFFFF, ti.i32)
                t_hit = ti.bit_cast(ti.cast(zk >> 32, ti.u32), ti.f32)
                use_ss, sm, vm, cam_ow = _ss_setup(
                    f, prim_raw, ss_enabled, tri_pos, cam_origin, screen_point,
                    pixel_basis_x, pixel_basis_y, gen_meta[2], gen_meta[3])
                if use_ss != 0:
                    _hok, _ht, a, b = _ss_pixel(px, py, sm, vm, cam_ow)
                else:
                    _hok, _ht, a, b = _raycast_pixel(
                        px, py, f, vm, gen_meta[2], gen_meta[3], cam_origin,
                        screen_point, pixel_basis_x, pixel_basis_y)
            # This fragment's shadow-bit slot (the terminal z-hit is at
            # q == nrun); -1 for fragments outside the packed budget.
            q_pos = q
            q += 1
            if (far_clip > 0.0) and (base_dist + t_hit > far_clip):
                done = True
                break
            processed += 1

            # A negative packed id tags a bezier circuit fragment; triangles
            # (and the z-winner) are non-negative.
            is_bez = False
            if ti.static(has_bez):
                is_bez = prim_raw < 0

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
                    circuit = -prim_raw - 1
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
                    # Per-light shadow visibility for this fragment: all-lit
                    # unless the deferred ``raster_shadow`` pre-pass packed an
                    # occlusion bit for this fragment's slot into ``rs_vis[r]``
                    # (only the budgeted nearest fragments + the opaque z-hit
                    # carry bits; anything past the budget stays lit).
                    vis = ti.Vector([1.0] * MAX_SHADOW_LIGHTS)
                    if ti.static(shadows != 0):
                        sh_slot = _raster_shadow_slot(q_pos, nrun)
                        if sh_slot >= 0:
                            sbits = rs_vis[r]
                            for li in range(num_lights):
                                if li < _RASTER_SHADOW_LIGHTS:
                                    if ((sbits >> (sh_slot
                                                   * _RASTER_SHADOW_LIGHTS
                                                   + li)) & 1) != 0:
                                        vis[li] = 0.0
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
                                           color, 0, vis)
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
            for k in ti.static(range(3)):
                ti.atomic_add(pix_accum[r, 4 + k], weight[k])
            rs_int[r, 2] = _DONE
