"""Inert stand-ins for Manim's creation animations."""

from __future__ import annotations

from .animation import Animation

__all__ = ["Create", "ShowPartial", "SpiralIn", "Uncreate", "Write"]


class ShowPartial(Animation):
    """See :mod:`~manim.animation`."""


class Create(ShowPartial):
    """See :mod:`~manim.animation`."""


class Uncreate(Create):
    """See :mod:`~manim.animation`."""


class Write(Create):
    """See :mod:`~manim.animation`."""


class SpiralIn(Animation):
    """See :mod:`~manim.animation`."""
