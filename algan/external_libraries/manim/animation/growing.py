"""Inert stand-ins for Manim's growing animations."""

from __future__ import annotations

from .transform import Transform

__all__ = ["GrowFromCenter", "GrowFromPoint", "SpinInFromNothing"]


class GrowFromPoint(Transform):
    """See :mod:`~manim.animation`."""


class GrowFromCenter(GrowFromPoint):
    """See :mod:`~manim.animation`."""


class SpinInFromNothing(GrowFromPoint):
    """See :mod:`~manim.animation`."""
