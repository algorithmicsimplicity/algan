"""A ``RectAreaLight`` as emissive geometry, for the path tracer only.

The deterministic renderer needs a ``RectAreaLight`` expanded into ``K = k*k``
packed cell rows: its shadow fans integrate visibility over one cell each, and
its lighting stage sums rows. The path tracer inherited that packing and paid
for it three times over -- ``K`` next-event entries per light (a 4x4 light is
already 16), a per-cell jitter special case in ``_pt_light_sample_point``, and
two gaps it could not close while a light was a row rather than geometry: no
reflected image in a mirror, and no BSDF strategy to MIS against.

This module gives the path tracer its own view of the same light: **two
emissive triangles** covering the rectangle, inserted into the path-traced
merge before its acceleration structures are built. They ride the emissive-triangle path
that already exists end to end -- area sampling from the next-event table,
``_pt_lit_f_pdf`` at both ends of the MIS pair, power-heuristic weights, and a
BSDF continuation ray that can find them.

The triangles are ordinary opaque, shadow-casting surfaces, visible to camera,
reflection and refraction rays. Their front side emits; their back side is
black. They enter scene preparation before its BVH build and arena upload,
so no second tree or persistent copy of the triangle tables is needed.

The existing light falloff controls remain shared by the two MIS strategies.

``rt_settings.pt_area_light_quads`` (``ALGAN_PT_AREA_LIGHT_QUADS``) is the kill
switch: off, nothing here runs and the packed cell rows are the path tracer's
emitters exactly as before.
"""

from __future__ import annotations

import torch

from algan.logging.logger import PERF, get_logger
from algan.rendering.raytracing import settings as rt_settings
from algan.rendering.raytracing.shading_taichi import (
    _LT_AREA_SAMPLE,
    _MAT_NO_SHADOW_RECEIVE,
    _MAT_ONE_SIDED,
    _MID_LAMBERT,
)
from algan.rendering.raytracing.stbvh import EMPTY_HI, EMPTY_LO

logger = get_logger("raytracing")

#: ``nee_meta``'s "no synthetic quads in this render" sentinel: a primitive
#: index no batch can reach, and exactly representable in float32 (the meta
#: vector is f32).
NO_QUAD_BASE = 1 << 30

#: Aux columns of a packed light row (``lights.Light._build_aux``'s layout,
#: which is the packed row's own columns shifted by the three RGB ones).
_AUX_DECAY = 1
_AUX_DISTANCE = 2
_AUX_NORMAL = slice(3, 6)


def _bcast_time(x, frames):
    """Broadcast a merged table's leading (time) axis to ``frames`` rows.

    Merged tables collapse to one row when they are constant over the batch
    (``scene_builder._dedup_time``), so a group being concatenated has to be
    re-expanded against the widest member first.
    """
    if x.shape[0] == frames:
        return x
    if x.shape[0] == 1:
        return x.expand(frames, *x.shape[1:])
    raise ValueError(
        f"cannot broadcast a merged table with {x.shape[0]} time rows to {frames}"
    )


def _collapse_time(x):
    """Collapse a leading time axis that is constant across frames to one row.

    ``scene_builder._dedup_time``'s rule, repeated here rather than imported so
    this module does not depend on the merge's private helpers; the ``bool()``
    is one sync per array and these arrays are tiny.
    """
    if x.shape[0] > 1 and bool((x == x[:1]).all()):
        return x[:1].contiguous()
    return x.contiguous()


def _rect_axes_from_normal(n):
    """(right, up) unit axes of the rectangle whose facing direction is ``n``.

    ``lights.RectAreaLight._rect_axes``' rule, evaluated on the packed normal
    rather than on the light's location: the light snapshot the renderer is
    handed carries the normal, and reading ``light.location`` at render time
    would read animation state the prefetch worker may be rewriting.
    """
    up_ref = torch.tensor((0.0, 1.0, 0.0), device=n.device, dtype=n.dtype)
    ref = up_ref.expand_as(n)
    parallel = (n * ref).sum(-1).abs() > 0.99
    alt = torch.tensor((1.0, 0.0, 0.0), device=n.device, dtype=n.dtype).expand_as(n)
    ref = torch.where(parallel.unsqueeze(-1), alt, ref)
    right = torch.nn.functional.normalize(
        torch.linalg.cross(ref, n, dim=-1), p=2, dim=-1
    )
    up = torch.linalg.cross(n, right, dim=-1)
    return right, up


def area_light_quad_sources(light_sources):
    """The lights this module replaces with geometry, in packed-row order.

    Returns a list of ``(light, row_start, num_rows)``: the light object and
    the span of packed light rows it occupies (``scene_builder._pack_lights``
    emits every light's samples consecutively, in this order). An empty list
    means the render has no area light and nothing below runs.
    """
    out = []
    row = 0
    for light in light_sources or ():
        origin = getattr(light, "origin", None)
        rows = (
            1
            if origin is None
            else int(origin.reshape(origin.shape[0], -1, 3).shape[1])
        )
        aux = getattr(light, "_render_aux", None)
        is_area = (
            aux is not None
            and float(getattr(light, "light_type", -1.0)) == float(_LT_AREA_SAMPLE)
            and getattr(light, "width", None) is not None
            and getattr(light, "height", None) is not None
        )
        if is_area:
            out.append((light, row, rows))
        row += rows
    return out


def _quad_geometry(light, num_frames, device):
    """Per-frame vertices, normal, radiance and falloff of one light's quad.

    Returns ``(pos [T, 2, 9], normal [T, 3], radiance [T, 3], exponent,
    range)``: two triangles per frame wound so their geometric normal
    ``(v1-v0) x (v2-v0)`` is the light's own facing direction, which is the
    side ``_light_eval``'s one-sided cosine emits toward.
    """
    origin = light.origin.reshape(light.origin.shape[0], -1, 3).float().to(device)
    aux = (
        light._render_aux.reshape(light._render_aux.shape[0], -1, 13).float().to(device)
    )
    col = light.light_color
    col = col.reshape(col.shape[0], -1, col.shape[-1])[..., :3].float().to(device)
    frames = max(origin.shape[0], aux.shape[0], col.shape[0], 1)
    origin = _bcast_time(origin, frames)
    aux = _bcast_time(aux, frames)
    col = _bcast_time(col, frames)

    k = int(origin.shape[1])
    # The K cell centres are a symmetric grid about the rectangle's own centre,
    # so their mean is that centre. (The light's ``location`` is the exact
    # answer, but it lives on animation state this thread must not read.)
    centre = origin.mean(1)  # [T, 3]
    normal = torch.nn.functional.normalize(aux[:, 0, _AUX_NORMAL], p=2, dim=-1)
    right, up = _rect_axes_from_normal(normal)
    hw = 0.5 * float(light.width)
    hh = 0.5 * float(light.height)
    a = centre - right * hw - up * hh
    b = centre + right * hw - up * hh
    c = centre + right * hw + up * hh
    d = centre - right * hw + up * hh
    # Winding: (b - a) x (d - a) = (2hw right) x (2hh up) = 4 hw hh (right x up)
    # and ``_rect_axes_from_normal`` builds up = n x right, so right x up = n.
    tri0 = torch.cat((a, b, d), -1)
    tri1 = torch.cat((b, c, d), -1)
    pos = torch.stack((tri0, tri1), 1)  # [T, 2, 9]

    area = float(light.width) * float(light.height)
    # Each packed row carries C * I / K; the rectangle's emitted radiance is
    # the whole light's power spread over its area, which is the matching the
    # section-5 acceptance test does by hand.
    radiance = col[:, 0, :] * (float(k) / max(area, 1e-12))
    decay = float(aux[0, 0, _AUX_DECAY].item())
    rng = float(aux[0, 0, _AUX_DISTANCE].item())
    return pos, normal, radiance, 2.0 - decay, rng


def build_area_light_quads(merged, light_sources, num_frames, bvh_inputs):
    """Return the widened scene and triangle BVH inputs, before tree building.

    ``pt_quad_base`` identifies synthetic emitters; ``pt_quad_falloff`` holds
    their distance law and ``pt_quad_rows`` withdraws their packed rows from
    physical next-event estimation. Authored materials retain those rows.
    Existing build inputs are extended directly, without recomputing bounds.
    """
    if not rt_settings.pt_area_light_quads:
        return merged, bvh_inputs
    sources = area_light_quad_sources(light_sources)
    if not sources:
        return merged, bvh_inputs

    device = merged["tri_pos"].device
    n_old = int(merged.get("num_triangles") or 0)
    n_colored = int(merged.get("num_colored_triangles") or 0)
    if n_old > 0:
        # The two layouts whose per-prim tables are not dense over [0, N):
        # promotion shrinks tri_colors/tri_extra, the memory trim reorders and
        # compacts them behind a remap. Neither is produced for
        # samples_per_pixel > 1, so this is a guard, not a fallback path.
        if merged.get("tri_col_row") is not None:
            return merged, bvh_inputs
        if int(merged["tri_colors"].shape[1]) < n_old:
            return merged, bvh_inputs

    rows_replaced = []
    pos_parts, norm_parts, rad_parts, fall_parts, active_parts = [], [], [], [], []
    for light, row_start, row_count in sources:
        pos, normal, radiance, expo, rng = _quad_geometry(light, num_frames, device)
        pos_parts.append(pos)
        norm_parts.append(normal)
        rad_parts.append(radiance)
        active = getattr(light, "_render_active", None)
        if active is None:
            active = torch.ones(1, dtype=torch.bool, device=device)
        active_parts.append(active.to(device).reshape(-1, 1).expand(-1, 2))
        fall_parts.append([expo, rng])
        fall_parts.append([expo, rng])
        rows_replaced.extend(range(row_start, row_start + row_count))

    frames = max(p.shape[0] for p in pos_parts)
    if frames not in (1, int(num_frames)):
        # The merge's own time axes are 1 or the batch's frame count and the
        # BVH builders accept only those two; a light snapshot shaped any
        # other way is not something to guess at, so the render keeps its
        # rows rather than getting geometry that cannot be built.
        logger.log(
            PERF,
            "path tracer: area-light snapshot has %d frames against the "
            "batch's %d; keeping the packed cell rows.",
            frames,
            int(num_frames),
        )
        return merged, bvh_inputs
    quad_pos = torch.cat([_bcast_time(p, frames) for p in pos_parts], 1)
    n_new = int(quad_pos.shape[1])
    # Both triangles of a quad share the light's facing normal, at every corner.
    quad_norm = torch.cat(
        [
            _bcast_time(n, frames).unsqueeze(1).expand(frames, 2, 3).repeat(1, 1, 3)
            for n in norm_parts
        ],
        1,
    )
    quad_rad = torch.cat(
        [_bcast_time(r, frames).unsqueeze(1).expand(frames, 2, 3) for r in rad_parts],
        1,
    )

    f32 = torch.float32
    i32 = torch.int32
    new = dict(merged)

    def _grow(key, extra):
        """Concatenate ``extra`` onto the merged table's primitive axis."""
        old = merged.get(key)
        if old is None:
            return extra.contiguous()
        t = max(int(old.shape[0]), int(extra.shape[0]))
        return torch.cat((_bcast_time(old, t), _bcast_time(extra, t)), 1).contiguous()

    # A triangle-free batch carries [1, 1, ...] placeholders rather than empty
    # tables, so the quads REPLACE them instead of extending them.
    base_pos = None if n_old == 0 else merged["tri_pos"]

    def _cat(key, extra):
        return extra.contiguous() if base_pos is None else _grow(key, extra)

    new["tri_pos"] = _collapse_time(_cat("tri_pos", quad_pos))
    new["tri_norm"] = _collapse_time(_cat("tri_norm", quad_norm))

    # Colours: black albedo (all the radiance is emissive), zero glow, opaque.
    quad_colors = torch.zeros((frames, n_new, 3, 5), dtype=f32, device=device)
    quad_colors[..., 4] = 1.0
    new["tri_colors"] = _collapse_time(_cat("tri_colors", quad_colors))

    # Per-vertex material extras: reflectivity/roughness/transmission 0, IOR 1
    # (columns 6..8; see raytrace_kernels_taichi._EXTRA_W's layout).
    extra_w = int(merged["tri_extra"].shape[2])
    quad_extra = torch.zeros((frames, n_new, extra_w), dtype=f32, device=device)
    if extra_w > 8:
        quad_extra[..., 6:9] = 1.0
    new["tri_extra"] = _collapse_time(_cat("tri_extra", quad_extra))

    mat_w = int(merged["tri_mat"].shape[2])
    quad_mat = torch.zeros((frames, n_new, mat_w), dtype=f32, device=device)
    quad_mat[..., 0:3] = quad_rad
    quad_mat[..., 3] = 1.0
    if mat_w > _MAT_ONE_SIDED:
        quad_mat[..., _MAT_ONE_SIDED] = 1.0
    if mat_w > _MAT_NO_SHADOW_RECEIVE:
        quad_mat[..., _MAT_NO_SHADOW_RECEIVE] = 1.0
    new["tri_mat"] = _collapse_time(_cat("tri_mat", quad_mat))

    # Every appended block takes the merged table's own dtype: these are
    # concatenations, and torch refuses a mismatched one rather than casting.
    quad_ids = torch.full(
        (1, n_new), _MID_LAMBERT, dtype=merged["tri_mat_id"].dtype, device=device
    )
    new["tri_mat_id"] = _collapse_time(_cat("tri_mat_id", quad_ids))

    obj_base = 0
    obj_dtype = i32
    if base_pos is not None and merged.get("tri_obj") is not None:
        obj_base = int(merged["tri_obj"].max().item()) + 1
        obj_dtype = merged["tri_obj"].dtype
    quad_obj = torch.arange(
        obj_base, obj_base + n_new, dtype=obj_dtype, device=device
    ).view(1, n_new)
    new["tri_obj"] = _collapse_time(_cat("tri_obj", quad_obj))
    closed_dtype = f32
    if merged.get("tri_closed") is not None:
        closed_dtype = merged["tri_closed"].dtype
    quad_closed = torch.zeros((1, n_new), dtype=closed_dtype, device=device)
    new["tri_closed"] = _collapse_time(_cat("tri_closed", quad_closed))

    # The uv / tex-meta tier is indexed by ``prim - num_colored_triangles``, so
    # it has to be extended by exactly the number of prims past that boundary.
    # A batch with no textured primitive carries a single placeholder row that
    # stands for none of them, hence the slice to the real count.
    meta_real = max(0, (n_old - n_colored) if base_pos is not None else 0)
    tex_meta_w = int(merged["tri_tex_meta"].shape[1])
    quad_meta = torch.full((n_new, tex_meta_w), -1, dtype=i32, device=device)
    quad_meta[:, 10:13] = 1
    quad_meta[:, 14] = 1
    quad_meta[:, 17] = 1
    new["tri_tex_meta"] = torch.cat(
        (merged["tri_tex_meta"][:meta_real], quad_meta), 0
    ).contiguous()
    uv_w = int(merged["tri_uvs"].shape[2])
    old_uvs = merged["tri_uvs"][:, :meta_real]
    quad_uvs = torch.zeros((old_uvs.shape[0], n_new, uv_w), dtype=f32, device=device)
    new["tri_uvs"] = torch.cat((old_uvs, quad_uvs), 1).contiguous()

    # Extend the original build inputs; scene_builder builds the only tree.
    active_frames = max(a.shape[0] for a in active_parts)
    active = torch.cat([_bcast_time(a, active_frames) for a in active_parts], 1)
    lo, hi, opaque, casts = _tri_bvh_inputs(
        bvh_inputs, new, n_old, n_new, device, active
    )
    new["tri_frame_valid"] = _collapse_time((hi >= lo).all(-1))
    new["tri_frame_opaque"] = _collapse_time(opaque)
    new["tri_frame_casts"] = _collapse_time(casts)
    uncertain = torch.zeros((1, n_new), dtype=torch.bool, device=device)
    new["tri_alpha_uncertain"] = _collapse_time(_cat("tri_alpha_uncertain", uncertain))
    new["num_triangles"] = (n_old if base_pos is not None else 0) + n_new
    # ``tri_has_visible`` / ``has_any_visible`` are deliberately NOT set here:
    # _merge_scene re-runs _record_visibility over the widened bounds and
    # recomputes both from the per-prefix flags as soon as this returns, so a
    # write here would be dead and would read as authoritative to the next
    # person changing the visibility rule.

    new["pt_quad_base"] = n_old if base_pos is not None else 0
    new["pt_quad_falloff"] = torch.tensor(fall_parts, dtype=f32, device=device)
    new["pt_quad_rows"] = sorted(rows_replaced)
    logger.log(
        PERF,
        "path tracer: %d RectAreaLight(s) -> %d emissive triangles "
        "(%d packed cell rows withdrawn from the next-event table).",
        len(sources),
        n_new,
        len(rows_replaced),
    )
    return new, (lo, hi, opaque, casts)


def _tri_bvh_inputs(bvh_inputs, new, n_old, n_new, device, active):
    """Extend the merge's bounds and flags with opaque, casting emitters."""
    parts_lo, parts_hi, parts_op, parts_cast = [], [], [], []
    if bvh_inputs is not None:
        lo, hi, opaque, casts = bvh_inputs
        parts_lo.append(lo)
        parts_hi.append(hi)
        parts_op.append(opaque)
        parts_cast.append(casts)

    q = new["tri_pos"][:, n_old:]
    qv = q.reshape(q.shape[0], n_new, 3, 3)
    frames = max(qv.shape[0], active.shape[0])
    active = _bcast_time(active, frames).unsqueeze(-1)
    lo = _bcast_time(qv.amin(-2), frames)
    hi = _bcast_time(qv.amax(-2), frames)
    parts_lo.append(torch.where(active, lo, torch.full_like(lo, EMPTY_LO)))
    parts_hi.append(torch.where(active, hi, torch.full_like(hi, EMPTY_HI)))
    parts_op.append(torch.ones((1, n_new), dtype=torch.bool, device=device))
    parts_cast.append(torch.ones((1, n_new), dtype=torch.bool, device=device))

    def _join(parts):
        t = max(int(p.shape[0]) for p in parts)
        return torch.cat([_bcast_time(p, t) for p in parts], 1).contiguous()

    return _join(parts_lo), _join(parts_hi), _join(parts_op), _join(parts_cast)
