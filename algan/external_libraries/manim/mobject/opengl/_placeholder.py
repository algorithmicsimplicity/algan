"""The base every OpenGL stand-in shares."""

from __future__ import annotations

from typing import Any, NoReturn

__all__ = ["_OpenGLPlaceholder"]


class _OpenGLPlaceholder:
    """A class no object is ever an instance of.

    The geometry modules reach for the OpenGL types two ways, and both survive:
    ``isinstance(x, (Mobject, OpenGLMobject))`` answers exactly as it would
    upstream under the Cairo renderer, and annotations are never evaluated.
    Construction is the one thing that cannot work, so it says so.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise NotImplementedError(
            f"{type(self).__name__} belongs to Manim's OpenGL renderer, which "
            "Algan's vendored Manim subset does not include. Algan renders "
            "Manim geometry with its own ray tracer; build the Cairo-side "
            "class instead."
        )
