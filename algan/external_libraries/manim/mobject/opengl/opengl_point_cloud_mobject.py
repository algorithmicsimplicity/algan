"""Inert stand-in for ``OpenGLPMobject``."""

from __future__ import annotations

from .opengl_mobject import OpenGLMobject

__all__ = ["OpenGLPMobject"]


class OpenGLPMobject(OpenGLMobject):
    """See :mod:`~manim.mobject.opengl`."""
