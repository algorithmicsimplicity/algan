"""Typing and NumPy shims, so the vendored subset runs on Algan's floor.

Upstream Manim requires Python 3.11 and NumPy 2.1; Algan supports Python 3.9
and NumPy 1.20 upwards. Three upstream spellings need a fallback, and the
vendoring script rewrites every use to come through here:

``Self``
    :data:`typing.Self` is 3.11. It is only ever used in an annotation, and
    every vendored module carries ``from __future__ import annotations``, so an
    older interpreter never evaluates it -- the name just has to exist.
``TypeAlias``
    :data:`typing.TypeAlias` is 3.10.
``trapezoid``
    ``np.trapezoid`` is NumPy 2.0's spelling of ``np.trapz``.
``zip_strict``
    ``zip``'s ``strict=`` keyword is 3.10.
"""

from __future__ import annotations

import itertools
import sys
from typing import Any

import numpy as np

__all__ = ["Self", "TypeAlias", "trapezoid", "zip_strict"]

if sys.version_info >= (3, 11):
    from typing import Self, TypeAlias
else:  # pragma: no cover - exercised only on Python 3.9 and 3.10
    if sys.version_info >= (3, 10):
        from typing import TypeAlias
    else:
        try:
            from typing_extensions import TypeAlias
        except ImportError:
            TypeAlias = Any  # type: ignore[misc, assignment]
    try:
        from typing_extensions import Self
    except ImportError:
        Self = Any  # type: ignore[misc, assignment]

#: ``np.trapz`` was renamed in NumPy 2.0 and removed from the main namespace.
trapezoid = getattr(np, "trapezoid", None) or np.trapz


if sys.version_info >= (3, 10):
    zip_strict = zip
else:  # pragma: no cover - exercised only on Python 3.9

    def _raise_on_ragged(iterables: tuple[Any, ...]) -> Any:
        sentinel = object()
        for row in itertools.zip_longest(*iterables, fillvalue=sentinel):
            if any(value is sentinel for value in row):
                raise ValueError("zip() arguments are of unequal length")
            yield row

    def zip_strict(*iterables: Any, strict: bool = False) -> Any:
        """``zip``, with 3.10's ``strict=`` keyword.

        Lazy either way, and ``strict=True`` raises ``ValueError`` on ragged
        input as the builtin does -- the message differs, the type does not.
        """
        if not strict:
            return zip(*iterables)
        return _raise_on_ragged(iterables)
