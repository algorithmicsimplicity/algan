"""``Animation`` itself, and the ``override_animation`` decorator."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NoReturn

__all__ = ["Animation", "override_animation"]


class Animation:
    """A Manim animation, as far as the vendored geometry subset needs one.

    Manim's animations drive a Manim ``Scene`` through a Manim renderer, and
    this subset vendors neither. Algan's own animation system replaces them;
    see :mod:`algan.animations`.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise NotImplementedError(
            f"{type(self).__name__} is part of Manim's animation system, which "
            "Algan's vendored Manim subset does not include -- Manim geometry "
            "is converted to Algan Mobs and animated by Algan. Use Algan's "
            "animations on the converted Mob instead."
        )


def override_animation(animation_class: type) -> Callable[[Callable], Callable]:
    """Mark a Mobject method as the override for ``animation_class``.

    Upstream's implementation verbatim. It only tags the function;
    :class:`~manim.mobject.mobject.Mobject.__init_subclass__` is what collects
    the tags into ``animation_overrides``. Nothing here runs an animation, so
    the registration side works unchanged.
    """

    def decorator(func: Callable) -> Callable:
        func._override_animation = animation_class
        return func

    return decorator
