"""Conservative host-side selection of shared shadow traversal modes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: Smallest ``min_alpha`` under which an opaque blocker still takes the march's
#: transmittance to exactly zero (``_shadow_opaque_first``): the primitives'
#: opacity flag allows alpha ``1 - 1e-6`` and pass-through ``1e-6``, which
#: leave at most ~2e-6 of the light.
_OPAQUE_FIRST_MIN_ALPHA = 1e-5
#: Smallest ``max_surfaces_per_ray`` the opaque-first answer is offered under:
#: a march capped before it reaches the blocker is the one corner the any-hit
#: pass cannot see, and the shipped 256 is the cap it was qualified at.
_OPAQUE_FIRST_MIN_SURFACES = 256


def _provably_opaque_shadow_batch(merged: Mapping[str, Any]) -> bool:
    """Missing metadata is uncertainty, never proof of opacity."""
    return not any(
        merged.get(name, True)
        for name in (
            "has_transmissive",
            "tri_has_translucent",
            "bez_has_translucent",
            "has_uncertain_texture_alpha",
        )
    )


def _opaque_first_applies(merged: Mapping[str, Any]) -> bool:
    """Whether shadow mode 5 can answer this batch exactly and has anything to
    answer early: some caster must be opaque, or every ray would march anyway.
    """
    from algan.rendering.raytracing import raytrace_kernels_taichi as rk

    return (
        rk.min_alpha >= _OPAQUE_FIRST_MIN_ALPHA
        and rk.max_surfaces_per_ray >= _OPAQUE_FIRST_MIN_SURFACES
        and bool(merged.get("tri_has_opaque") or merged.get("bez_has_opaque"))
    )


def _select_shadow_mode(
    shadows: bool, mode: bool | str, merged: Mapping[str, Any]
) -> int:
    """0 off, 1 ordered march, 2 mixed prepass, 3 opaque any-hit, 4 gather,
    5 opaque-first.

    Auto never uses the mixed prepass: on an uncertain batch it retains the
    ordered march. "opaquefirst" is auto plus, for a mixed batch it can serve,
    the opaque-first query (``_shadow_opaque_first``): an any-hit pass that
    settles blocked and untouched rays, with the ordered march behind it for
    the rest. Opt-in, on GPU measurement (``settings.shadow_anyhit``). Explicit
    True and "gather" retain their experimental modes; explicit False is the
    reference-path kill switch.
    """
    if not shadows:
        return 0
    if not mode:
        return 1
    if mode == "gather":
        return 4
    if _provably_opaque_shadow_batch(merged):
        return 3
    if mode == "opaquefirst":
        return 5 if _opaque_first_applies(merged) else 1
    if mode == "auto" or merged.get("has_transmissive", True):
        return 1
    return 2
