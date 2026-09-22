"""Validation shared by native and imported vector stroke styles."""

from __future__ import annotations

import math
import warnings

from algan.errors import AlganConfigurationError, UnsupportedFeatureWarning


def _style_name(value, choices, name):
    value = getattr(value, "name", value)
    if value is None or (isinstance(value, str) and value.lower() == "auto"):
        return "round"
    if isinstance(value, str) and value.lower() in choices:
        return value.lower()
    raise AlganConfigurationError(f"{name} must be one of {', '.join(choices)}")


def _stroke_style(cap_style, joint_type, miter_limit):
    cap = _style_name(cap_style, ("round", "butt", "square"), "cap_style")
    join = _style_name(joint_type, ("round", "bevel", "miter"), "joint_type")
    try:
        limit = float(miter_limit)
        if not math.isfinite(limit) or limit < 1:
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise AlganConfigurationError(
            "miter_limit must be finite and at least 1"
        ) from exc
    return cap, join, limit


def _warn_background_stroke(width, opacity=1):
    if float(width or 0) > 0 and float(opacity or 0) > 0:
        warnings.warn(
            "Background strokes are not rendered. Use a separate wider path behind "
            "the foreground path for a background stroke.",
            UnsupportedFeatureWarning,
            stacklevel=3,
        )
