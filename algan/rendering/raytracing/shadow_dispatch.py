"""Conservative host-side selection of shared shadow traversal modes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


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


def _select_shadow_mode(
    shadows: bool, mode: bool | str, merged: Mapping[str, Any]
) -> int:
    """0 off, 1 ordered march, 2 mixed prepass, 3 opaque any-hit, 4 gather.

    Auto never uses the mixed prepass: on an uncertain batch it retains the
    ordered march. Explicit True and "gather" retain their experimental modes;
    explicit False is the reference-path kill switch.
    """
    if not shadows:
        return 0
    if not mode:
        return 1
    if mode == "gather":
        return 4
    if _provably_opaque_shadow_batch(merged):
        return 3
    if mode == "auto" or merged.get("has_transmissive", True):
        return 1
    return 2
