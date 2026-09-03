"""Inert stand-ins for ``OpenGLVMobject`` and ``OpenGLVGroup``."""

from __future__ import annotations

from .opengl_mobject import OpenGLMobject

__all__ = ["OpenGLVGroup", "OpenGLVMobject"]


class OpenGLVMobject(OpenGLMobject):
    """See :mod:`~manim.mobject.opengl`."""


class OpenGLVGroup(OpenGLVMobject):
    """See :mod:`~manim.mobject.opengl`."""
