"""Inert stand-ins for ``OpenGLMobject`` and ``OpenGLGroup``."""

from __future__ import annotations

from ._placeholder import _OpenGLPlaceholder

__all__ = ["OpenGLGroup", "OpenGLMobject"]


class OpenGLMobject(_OpenGLPlaceholder):
    """See :mod:`~manim.mobject.opengl`."""


class OpenGLGroup(OpenGLMobject):
    """See :mod:`~manim.mobject.opengl`."""
