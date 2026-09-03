"""Inert stand-ins for Manim's transform animations."""

from __future__ import annotations

from .animation import Animation

__all__ = ["Transform", "_MethodAnimation"]


class Transform(Animation):
    """See :mod:`~manim.animation`."""


class _MethodAnimation(Transform):
    """What upstream builds behind ``mob.animate.shift(...)``.

    Reached only through ``Mobject.animate``, which needs a Manim ``Scene`` to
    play what it builds. Algan records the plain method call instead: call
    ``mob.shift(...)`` inside an Algan animation context.
    """
