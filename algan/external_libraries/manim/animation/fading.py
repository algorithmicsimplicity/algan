"""Inert stand-ins for Manim's fading animations."""

from __future__ import annotations

from .animation import Animation

__all__ = ["FadeIn", "FadeOut"]


class _Fade(Animation):
    """See :mod:`~manim.animation`."""


class FadeIn(_Fade):
    """See :mod:`~manim.animation`."""


class FadeOut(_Fade):
    """See :mod:`~manim.animation`."""
