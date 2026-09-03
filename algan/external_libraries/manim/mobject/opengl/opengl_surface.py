"""Inert stand-in for ``OpenGLSurface``."""

from __future__ import annotations

from .opengl_mobject import OpenGLMobject

__all__ = ["OpenGLSurface"]


class OpenGLSurface(OpenGLMobject):
    """See :mod:`~manim.mobject.opengl`."""
