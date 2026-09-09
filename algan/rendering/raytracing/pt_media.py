"""Scene preparation for homogeneous path-traced interiors (no kernel variants)."""

from __future__ import annotations

import torch

from algan.rendering.raytracing.shading_taichi import (
    _MAT_MEDIUM_CLOSED,
    _MAT_PHASE_G,
    _MAT_SIGMA_S,
    _MID_PHYSICAL,
)


def _prepare_media(memory, merged, opacity_shells):
    """Return (combined shell table, PT-only surface extras, medium offset).

    The first N columns retain the opacity ring's original semantics. The
    second N contain raw tri_obj identities for physical closed interiors,
    including transmissive objects and zero-density nested cavities. Ordinary
    scenes keep their original arrays and consume no extra scene memory.
    """
    mat = merged.get("tri_mat")
    extra = merged["tri_extra"]
    if merged.get("has_scattering_media") is False:
        return opacity_shells, extra, 0
    if mat is None or mat.shape[-1] <= _MAT_PHASE_G:
        return opacity_shells, extra, 0
    physical = merged["tri_mat_id"] == _MID_PHYSICAL
    scattering = mat[..., _MAT_SIGMA_S : _MAT_SIGMA_S + 3]
    physical, positive = torch.broadcast_tensors(physical, (scattering > 0).any(dim=-1))
    scattering = scattering.expand(*physical.shape, 3)
    active = physical & positive
    if not bool(torch.isfinite(scattering[physical]).all()):
        raise ValueError("sigma_s must contain finite, non-negative coefficients")
    if bool((scattering[physical] < 0).any()):
        raise ValueError("sigma_s must contain finite, non-negative coefficients")
    if not bool(active.any()):
        return opacity_shells, extra, 0
    anisotropy = mat[..., _MAT_PHASE_G].expand_as(active)
    if not bool((torch.isfinite(anisotropy) & (anisotropy.abs() < 1))[active].all()):
        raise ValueError("g must be finite and strictly between -1 and 1")
    closed = physical & (mat[..., _MAT_MEDIUM_CLOSED] > 0.5)
    if bool((active & ~closed).any()):
        raise ValueError(
            "Scattering media require a watertight triangle surface declared "
            "closed_shell=True; use a closed Sphere/Prism or declare a closed mesh."
        )
    identities = torch.where(closed, merged["tri_obj"].to(torch.int32), -1)
    n = identities.shape[1]
    rows = max(identities.shape[0], opacity_shells.shape[0])
    combined = torch.cat(
        (opacity_shells.expand(rows, n), identities.expand(rows, n)), dim=1
    ).contiguous()
    # The legacy shadow marcher treats absorption as paired surface chords.
    # The medium walk integrates exact partial/nested chords instead, so remove
    # those coefficients in a PT-private copy. Never mutate the shared merge:
    # deterministic rendering and retries can reuse it.
    rows_extra = max(extra.shape[0], closed.shape[0])
    shadow_extra = extra.expand(rows_extra, -1, -1).clone()
    shadow_extra[..., 12:15] = torch.where(
        closed.expand(rows_extra, n).unsqueeze(-1),
        0.0,
        shadow_extra[..., 12:15],
    )
    from algan.rendering.raytracing.tracer import _arena_copy

    with memory.scope("pt_media_shells", rows=rows):
        combined = _arena_copy(memory, combined)
    with memory.scope("pt_media_shadow_extra", rows=rows_extra):
        shadow_extra = _arena_copy(memory, shadow_extra.contiguous())
    return combined, shadow_extra, n
