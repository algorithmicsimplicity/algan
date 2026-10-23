"""Host side of the path tracer's MNEE caustics (``pt_caustics_taichi``).

``_prepare_casters`` appends the caustic-caster section to ``pt_shade``'s
``tri_shell`` table. The kernel module holds everything else.

The same chains are how a point, spot or directional light reaches the
inside of a subsurface solid: no sampled ray can hit a delta light, and the
straight shadow connection stops at a scattering interior's refracting
boundary. So with ``pt_caustics`` off the section is still built for a batch
with a scattering medium in it (``pt_subsurface_delta_lights``), holding only
those media's smooth boundaries, and the kernel lets only medium vertices
receive through it: every surface keeps the straight, tinted shadow it has
without caustics.
"""

from __future__ import annotations

import torch

from algan.rendering.raytracing import settings as rt_settings
from algan.rendering.raytracing.shading_taichi import (
    _MAT_MEDIUM_CLOSED,
    _MAT_NO_CAUSTIC_CAST,
    _MAT_SIGMA_S,
    _MID_LAMBERT,
    _MID_PHYSICAL,
)


def _caster_mask(merged):
    """``[rows, n]`` bool: triangles a manifold walk may refract through.

    A superset of the kernel's per-point test (``_pt_mnee_surface``, which
    is ``pt_shade``'s own smooth-dielectric crossing): a lit material that
    transmits, is not wholly metal, has an index off 1 and a GGX width on
    its floor somewhere on the triangle -- each property's corner values
    interpolate barycentrically, so "somewhere" is a test on the corners'
    range -- or that takes the property from its material map, on a mob
    that casts shadows and has not opted out with ``Mob.casts_caustics``. A
    triangle without a per-vertex row (a promoted constant material, which
    never transmits) reads the kernel's defaults.
    """
    extra = merged["tri_extra"]
    n = int(merged["tri_obj"].shape[1])
    met = extra[..., 0:6:2]
    rough = extra[..., 1:6:2]
    corner = torch.stack(
        (
            (extra[..., 9:12] > 1e-4).any(-1),
            met.amin(-1) < 1.0 - 1e-6,
            ((extra[..., 6:9] - 1.0).abs() >= 1e-4).any(-1),
            (rough.amin(-1) < 1e-2) & (rough.amax(-1) > -1e-2),
        ),
        dim=-1,
    )  # [Te, ne, 4]: transmits, dielectric, bends, smooth
    ne = int(corner.shape[1])
    if ne < n:
        # The kernel's defaults past the per-vertex rows: reflectivity and
        # roughness 0, index 1, no transmission.
        tail = torch.tensor([False, True, False, True], device=extra.device)
        corner = torch.cat((corner, tail.expand(corner.shape[0], n - ne, 4)), dim=1)
    elif ne > n:
        corner = corner[:, :n]
    # Material-map overrides (``_flat_triangle_material``): a flagged
    # property is sampled per fragment, so it may hold anywhere.
    num_colored = int(merged["num_colored_triangles"])
    meta = merged["tri_tex_meta"]
    textured = max(0, min(n - num_colored, int(meta.shape[0])))
    if textured > 0:
        rows = meta[:textured]
        flags = torch.where(rows[:, 3] >= 0, rows[:, 9], torch.zeros_like(rows[:, 9]))
        bits = torch.tensor([8, 1, 4, 2], dtype=flags.dtype, device=flags.device)
        mapped = (flags.unsqueeze(-1) & bits) != 0  # [textured, 4]
        override = torch.zeros((n, 4), dtype=torch.bool, device=extra.device)
        override[num_colored : num_colored + textured] = mapped.to(extra.device)
        corner = corner | override
    refracts = corner.all(-1)  # [Te, n]
    mat_id = merged["tri_mat_id"]
    lit = (mat_id >= _MID_LAMBERT) & (mat_id <= _MID_PHYSICAL)
    mat = merged["tri_mat"]
    if int(mat.shape[-1]) > _MAT_NO_CAUSTIC_CAST:
        opted = mat[..., _MAT_NO_CAUSTIC_CAST] < 0.5
    else:
        opted = torch.ones_like(lit)
    # A triangle that casts no shadow bends no light sample either: the walks
    # skip it (``_pt_mnee_next_tri``), so counting it as a caster would hold
    # back the BSDF paths through it for a chain that never forms, while the
    # straight light sample passes it unshadowed. Not a caster, it keeps the
    # MIS weight it has with caustics off.
    casts = merged.get("tri_frame_casts")
    if casts is not None and int(casts.shape[-1]) == n:
        shadow = casts.to(device=extra.device, dtype=torch.bool)
    else:
        shadow = torch.ones((1, n), dtype=torch.bool, device=extra.device)
    others = {int(t.shape[0]) for t in (refracts, lit, opted)} - {1}
    if int(shadow.shape[0]) != 1 and others - {int(shadow.shape[0])}:
        # Time rows that do not broadcast: a caster only where it casts in
        # every frame.
        shadow = shadow.all(0, keepdim=True)
    refracts, lit, opted, shadow = torch.broadcast_tensors(refracts, lit, opted, shadow)
    return refracts & lit & opted & shadow


def _scattering_boundary_mask(merged):
    """``[rows, n]`` bool: triangles that bound a scattering interior.

    The physical, closed-shell triangles with a positive ``sigma_s`` -- the
    shells ``pt_media._prepare_media`` puts in its section because a path
    scatters inside them. Glass, zero-density cavities and an index-matched
    fog's neighbours are not among them.
    """
    mat = merged["tri_mat"]
    if int(mat.shape[-1]) <= _MAT_MEDIUM_CLOSED:
        return torch.zeros((1, int(mat.shape[1])), dtype=torch.bool, device=mat.device)
    physical = merged["tri_mat_id"] == _MID_PHYSICAL
    scattering = (mat[..., _MAT_SIGMA_S : _MAT_SIGMA_S + 3] > 0).any(-1)
    closed = mat[..., _MAT_MEDIUM_CLOSED] > 0.5
    physical, scattering, closed = torch.broadcast_tensors(physical, scattering, closed)
    return physical & scattering & closed


def _prepare_casters(memory, merged, shells, media_offset=0):
    """Return ``(tri_shell, caster offset, media only)``.

    ``shells`` is ``pt_shade``'s shell table as ``_prepare_media`` left it:
    the opacity ring's columns, then the media section's when there is one
    (``media_offset`` > 0), or the ``[1, 1]`` placeholder. The caster section
    appended after them holds each triangle's ``tri_obj`` surface id where
    ``_caster_mask`` keeps it and -1 elsewhere; the offset is its first
    column.

    With ``pt_caustics`` on every caster is in it and every receiver walks
    through it. With it off and ``pt_subsurface_delta_lights`` on, a batch
    with a scattering medium still gets a section, of that medium's casters
    only, and ``media only`` is True: the kernel then lets only medium
    vertices receive through it, so delta lights reach a subsurface interior
    and nothing else changes. Offset 0 -- neither applies, or nothing in the
    merge that could refract -- returns ``shells`` unchanged, and the caller
    compiles the kernel without caustics, so such a render is the renderer it
    was before they existed.
    """
    media_only = False
    if not rt_settings.pt_caustics:
        if not (rt_settings.pt_subsurface_delta_lights and int(media_offset) > 0):
            return shells, 0, False
        media_only = True
    if merged.get("has_refractive") is False:
        return shells, 0, False
    if int(merged.get("num_triangles", 0)) == 0 or merged.get("tri_mat") is None:
        return shells, 0, False
    caster = _caster_mask(merged)
    if media_only:
        caster, boundary = torch.broadcast_tensors(
            caster, _scattering_boundary_mask(merged).to(caster.device)
        )
        caster = caster & boundary
    if not bool(caster.any()):
        return shells, 0, False
    identities = torch.where(
        caster,
        merged["tri_obj"].to(torch.int32),
        torch.full((1, 1), -1, dtype=torch.int32, device=caster.device),
    )
    n = int(identities.shape[1])
    rows = max(int(identities.shape[0]), int(shells.shape[0]))
    # The placeholder widens to a full opacity section of -1 so the caster
    # columns start past every primitive index the opacity ring reads.
    head = n if int(shells.shape[1]) == 1 else int(shells.shape[1])
    combined = torch.cat(
        (shells.expand(rows, head), identities.expand(rows, n)), dim=1
    ).contiguous()
    from algan.rendering.raytracing.tracer import _arena_copy

    with memory.scope("pt_caster_shells", rows=rows):
        combined = _arena_copy(memory, combined)
    return combined, head, media_only
