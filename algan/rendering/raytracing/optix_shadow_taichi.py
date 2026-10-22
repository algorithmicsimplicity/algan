"""The Taichi half of the OptiX shadow path: transmittance from a hit list.

The OptiX shadow trace (``optix/shadows.py``) answers a shadow ray itself
when an opaque caster stops it or nothing touches it. Otherwise it hands over
the translucent casters it crossed -- up to ``SHADOW_HITS`` of them, first in
``_comes_after`` order -- and this kernel multiplies them exactly as
``_shadow_march_occluded`` does: the same order, seam merge, colour fetch,
pass-through, Beer-Lambert pairing, early exits and surface cap, through the
march's own material functions, so materials stay single-source. A ray whose
list ran past ``SHADOW_HITS`` and still had light left when the list ended is
marched again in software, from scratch, by the march itself.
"""
from algan.rendering.raytracing.arena_args_taichi import ArenaView, arena_packed
from algan.rendering.raytracing.raster_taichi import _RASTER_SHADOW_TRACE_ARENA
from algan.rendering.raytracing.raytrace_kernels_taichi import (
    _SOLID_COVERAGE_MIN,
    NODE_ARG,
    TRIANGLE_EDGE_EPSILON,
    CameraProjection,
    _comes_after,
    _flat_triangle_color,
    _pixel_size_base,
    _shadow_hit_sigma,
    _shadow_occluded,
    _shadow_pass_through,
    depth_tie_epsilon,
    max_surfaces_per_ray,
    min_alpha,
    rgb_shadow_tint,
)
from algan.rendering.raytracing.shading_taichi import _vis_max_component
from algan.taichi_compat import ti

#: Hits one OptiX shadow ray lists (``runtime.SHADOW_HITS``, the device
#: programs' ``AO_SHADOW_HITS``).
SHADOW_HITS = 4


@ti.func
def _march_list(r, f, max_t, n, hits: ti.template(), tri_colors: ti.template(),
                tri_uvs: ti.template(), tri_tex_meta: ti.template(),
                textures: ti.template(), tri_extra: ti.template(),
                circuit_meta: ti.template(), num_colored_triangles,
                layer_offset_triangles):
    """``_shadow_march_occluded`` over a listed hit set.

    Returns ``(settled, occluded)``. ``settled`` is 0 only when the list was
    cut short (``n > SHADOW_HITS``) and the march consumed every listed hit
    without ending: the hits past the list could still matter.
    """
    transmitted = ti.math.vec3(1.0)
    one3 = ti.math.vec3(1.0)
    medium_open = 0
    medium_sigma = ti.math.vec3(0.0)
    medium_t_entry = 0.0
    seam_t = -1e30
    step = 0
    settled = 0
    used = ti.min(n, SHADOW_HITS)
    consumed = 0
    t_prev = 0.0
    layer_prev = 1e30
    while (consumed < used) and (step < max_surfaces_per_ray):
        step += 1
        # The next hit in _comes_after order after (t_prev, layer_prev): the
        # list arrives sorted already, but the order is decided here, by the
        # march's own comparison, not by the device's.
        sel = -1
        t_hit = 0.0
        hit_layer = 0.0
        for k in ti.static(range(SHADOW_HITS)):
            if k < used:
                tk = hits[r, 4 * k]
                lk = layer_offset_triangles + ti.cast(
                    ti.bit_cast(hits[r, 4 * k + 1], ti.i32), ti.f32)
                if _comes_after(tk, lk, t_prev, layer_prev):
                    if (sel < 0) or _comes_after(t_hit, hit_layer, tk, lk):
                        sel = k
                        t_hit = tk
                        hit_layer = lk
        prim = 0
        a = 0.0
        b = 0.0
        for k in ti.static(range(SHADOW_HITS)):
            if k == sel:
                prim = ti.bit_cast(hits[r, 4 * k + 1], ti.i32)
                a = hits[r, 4 * k + 2]
                b = hits[r, 4 * k + 3]
        consumed += 1
        if t_hit >= max_t:
            settled = 1
            break
        w0 = 1.0 - a - b
        edge_hit = 1 if ti.min(w0, ti.min(a, b)) < TRIANGLE_EDGE_EPSILON else 0
        if (edge_hit == 1) and (t_hit - seam_t <= depth_tie_epsilon):
            t_prev = t_hit
            layer_prev = hit_layer
            continue
        seam_t = t_hit if edge_hit == 1 else -1e30
        color4, alpha = _flat_triangle_color(
            f, prim, w0, a, b, tri_colors, tri_uvs, tri_tex_meta,
            textures, num_colored_triangles)
        alpha = ti.math.clamp(alpha, 0.0, 1.0)
        tint = ti.math.clamp(
            ti.math.vec3(color4[0], color4[1], color4[2]), 0.0, 1.0)
        if ti.static(rgb_shadow_tint()):
            sigma = _shadow_hit_sigma(f, prim, 1, tri_extra)
            if (alpha >= _SOLID_COVERAGE_MIN) \
                    and (_vis_max_component(sigma) > 0.0):
                if medium_open == 1:
                    seg = ti.max(t_hit - medium_t_entry, 0.0)
                    transmitted *= ti.math.vec3(
                        ti.exp(-medium_sigma[0] * seg),
                        ti.exp(-medium_sigma[1] * seg),
                        ti.exp(-medium_sigma[2] * seg))
                    medium_open = 0
                else:
                    medium_open = 1
                    medium_sigma = sigma
                    medium_t_entry = t_hit
        passed = _shadow_pass_through(f, prim, 1, w0, a, b,
                                      tri_extra, circuit_meta, tint)
        transmitted *= one3 - alpha * (one3 - passed)
        if _vis_max_component(transmitted) <= min_alpha:
            transmitted = ti.math.vec3(0.0)
            settled = 1
            break
        if (alpha >= 1.0) and (_vis_max_component(passed) <= 0.0):
            settled = 1
            break
        t_prev = t_hit
        layer_prev = hit_layer
    if (settled == 0) and ((n <= SHADOW_HITS) or (step >= max_surfaces_per_ray)):
        # Every hit was listed, or the march's own cap ended it here too.
        settled = 1
    return settled, one3 - transmitted


@ti.kernel
def optix_shadow_resolve_arena(
        num_events: int, event_first: int, num_rays: int,
        event_frame: ti.types.ndarray(), event_src_prim: ti.types.ndarray(),
        t_nodes: NODE_ARG, t_first_leaf: int, num_colored_triangles: int,
        b_nodes: NODE_ARG, b_first_leaf: int, layer_offset_triangles: ti.f32,
        refit: ti.template(), has_tri: ti.template(), has_bez: ti.template(),
        shadow_anyhit: ti.template(), shadow_identity: ti.template(),
        eps_self: ti.f32, eps_near: ti.f32,
        queue_data: ti.types.ndarray(), status: ti.types.ndarray(),
        hits: ti.types.ndarray(), occ_out: ti.types.ndarray(),
        projected: ti.template(),
        arena_f32: ti.types.ndarray(), arena_i32: ti.types.ndarray(),
        aoff: ti.types.ndarray(), ashp: ti.types.ndarray()):
    """Turn the OptiX trace's hit lists into occlusion (``occ_out``).

    ``status[r]`` is 0 for a ray this phase did not trace and 1 for one the
    trace answered itself; ``2 + n`` means it crossed ``n`` translucent
    casters and listed the first ``min(n, SHADOW_HITS)`` in ``hits[r]``. The
    arena layout is ``raster_shadow_trace``'s: a ray the list cannot settle
    runs ``_shadow_occluded``, exactly as ``shadow_queue_trace`` would.
    """
    t_node_miss = ti.static(ArenaView(arena_i32, aoff[0], (ashp[0],)))
    t_leaf_prim = ti.static(ArenaView(arena_i32, aoff[1], (ashp[1],)))
    t_leaf_tspan = ti.static(ArenaView(arena_i32, aoff[2], (ashp[2],)))
    tri_pos = ti.static(ArenaView(arena_f32, aoff[3], (ashp[3], ashp[4], ashp[5])))
    tri_colors = ti.static(ArenaView(
        arena_f32, aoff[4], (ashp[6], ashp[7], ashp[8], ashp[9])))
    tri_uvs = ti.static(ArenaView(arena_f32, aoff[5], (ashp[10], ashp[11], ashp[12])))
    tri_tex_meta = ti.static(ArenaView(arena_i32, aoff[6], (ashp[13], ashp[14])))
    textures = ti.static(ArenaView(arena_f32, aoff[7], (ashp[15], ashp[16], ashp[17])))
    tri_extra = ti.static(ArenaView(arena_f32, aoff[8], (ashp[18], ashp[19], ashp[20])))
    b_node_miss = ti.static(ArenaView(arena_i32, aoff[9], (ashp[21],)))
    b_leaf_prim = ti.static(ArenaView(arena_i32, aoff[10], (ashp[22],)))
    b_leaf_tspan = ti.static(ArenaView(arena_i32, aoff[11], (ashp[23],)))
    circuit_meta = ti.static(ArenaView(
        arena_f32, aoff[12], (ashp[24], ashp[25], ashp[26])))
    circuit_colors = ti.static(ArenaView(
        arena_f32, aoff[13], (ashp[27], ashp[28], ashp[29], ashp[30])))
    circuit_border_colors = ti.static(ArenaView(
        arena_f32, aoff[14], (ashp[31], ashp[32], ashp[33], ashp[34])))
    edges_2d = ti.static(ArenaView(arena_f32, aoff[15], (ashp[35], ashp[36], ashp[37])))
    edge_accel = ti.static(ArenaView(arena_i32, aoff[16], (ashp[38],)))
    # Kept for the shared serial-shadow arena offset contract (lights).
    light_pos = ti.static(ArenaView(  # noqa: F841
        arena_f32, aoff[17], (ashp[39], ashp[40], ashp[41])))
    light_col = ti.static(ArenaView(  # noqa: F841
        arena_f32, aoff[18], (ashp[42], ashp[43], ashp[44])))
    pixel_world_scale = ti.static(ArenaView(arena_f32, aoff[19], (ashp[45],)))
    tri_obj = ti.static(ArenaView(arena_i32, aoff[20], (ashp[46], ashp[47])))
    cam_proj_table = ti.static(ArenaView(arena_f32, aoff[21], (ashp[48], ashp[49])))
    cam_proj = ti.static(CameraProjection(cam_proj_table, projected))

    for r in range(num_rays):
        s = status[r]
        if s >= 2:
            e = event_first + r % num_events
            f = event_frame[e]
            max_t = queue_data[r, 6]
            settled, occ = _march_list(
                r, f, max_t, s - 2, hits, tri_colors, tri_uvs, tri_tex_meta,
                textures, tri_extra, circuit_meta, num_colored_triangles,
                layer_offset_triangles)
            if settled == 0:
                src_prim = -1
                src_sid = -1
                if ti.static(shadow_identity):
                    src_prim = event_src_prim[e]
                    if src_prim >= 0:
                        src_sid = tri_obj[f % tri_obj.shape[0], src_prim]
                origin = ti.math.vec3(queue_data[r, 0], queue_data[r, 1],
                                      queue_data[r, 2])
                direction = ti.math.vec3(queue_data[r, 3], queue_data[r, 4],
                                         queue_data[r, 5])
                occ = _shadow_occluded(
                    refit, shadow_anyhit, origin, direction, f,
                    ti.cast(f, ti.f32), max_t,
                    pixel_world_scale[f % pixel_world_scale.shape[0]],
                    0.0, _pixel_size_base(cam_proj, f),
                    layer_offset_triangles, has_tri, has_bez,
                    t_nodes, t_node_miss, t_leaf_prim, t_leaf_tspan,
                    t_first_leaf, tri_pos, tri_colors, tri_uvs, tri_tex_meta,
                    textures, tri_extra, num_colored_triangles, b_nodes,
                    b_node_miss, b_leaf_prim, b_leaf_tspan, b_first_leaf,
                    circuit_meta, circuit_colors, circuit_border_colors,
                    edges_2d, edge_accel, src_sid, src_prim, eps_self,
                    eps_near, tri_obj, shadow_identity)
                status[r] = -1  # diagnostic: marched in software
            for k in ti.static(range(3)):
                occ_out[r, k] = occ[k]


_OPTIX_SHADOW_RESOLVE_ARENA = _RASTER_SHADOW_TRACE_ARENA
_OPTIX_SHADOW_RESOLVE_PARAMS = (
    "num_events", "event_first", "num_rays", "event_frame", "event_src_prim",
    "t_nodes", "t_first_leaf", "num_colored_triangles", "b_nodes", "b_first_leaf",
    "layer_offset_triangles", "refit", "has_tri", "has_bez", "shadow_anyhit",
    "shadow_identity", "eps_self", "eps_near", "queue_data", "status", "hits",
    "occ_out", "t_node_miss", "t_leaf_prim", "t_leaf_tspan", "tri_pos",
    "tri_colors", "tri_uvs", "tri_tex_meta", "textures", "tri_extra",
    "b_node_miss", "b_leaf_prim", "b_leaf_tspan", "circuit_meta",
    "circuit_colors", "circuit_border_colors", "edges_2d", "edge_accel",
    "light_pos", "light_col", "pixel_world_scale", "tri_obj", "cam_proj_table",
    "projected",
)
_optix_shadow_resolve_launch = arena_packed(
    __name__, "optix_shadow_resolve_arena",
    _OPTIX_SHADOW_RESOLVE_PARAMS, _OPTIX_SHADOW_RESOLVE_ARENA)


def optix_shadow_resolve(*args):
    """Launch ``optix_shadow_resolve_arena`` through the arena convention."""
    return _optix_shadow_resolve_launch(*args)
