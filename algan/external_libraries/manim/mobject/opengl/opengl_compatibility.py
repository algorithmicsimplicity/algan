"""The ``ConvertToOpenGL`` metaclass, minus the renderer swap."""

from __future__ import annotations

from abc import ABCMeta
from typing import Any

__all__ = ["ConvertToOpenGL"]


class ConvertToOpenGL(ABCMeta):
    """Upstream swaps a class's bases for their OpenGL counterparts here when
    ``config.renderer`` is OpenGL. The vendored subset ships no OpenGL
    renderer -- ``ManimConfig.renderer`` rejects ``"opengl"`` outright -- so
    this is the Cairo branch alone: an ``ABCMeta`` that records the classes
    built with it, because that registry is still walked by the config setter.
    """

    _converted_classes: list[type] = []

    def __init__(cls, name: str, bases: tuple[type, ...], namespace: dict[str, Any]):
        super().__init__(name, bases, namespace)
        cls._converted_classes.append(cls)
