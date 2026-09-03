"""Inert stand-ins for Manim's updater animations."""

from __future__ import annotations

from ..animation import Animation

__all__ = ["UpdateFromAlphaFunc", "UpdateFromFunc"]


class UpdateFromFunc(Animation):
    """See :mod:`~manim.animation`."""


class UpdateFromAlphaFunc(UpdateFromFunc):
    """See :mod:`~manim.animation`."""
