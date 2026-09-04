"""Typing and NumPy shims, so the vendored subset runs on Algan's floor.

Upstream Manim requires Python 3.11 and NumPy 2.1; Algan supports Python 3.10
and NumPy 1.20 upwards. Two upstream spellings need a fallback, and the
vendoring script rewrites every use to come through here:

``Self``
    :data:`typing.Self` is 3.11. It is only ever used in an annotation, and
    every vendored module carries ``from __future__ import annotations``, so an
    older interpreter never evaluates it -- the name just has to exist.
``trapezoid``
    ``np.trapezoid`` is NumPy 2.0's spelling of ``np.trapz``.

``TypeAlias`` and ``zip_strict`` are re-exported unchanged and need no
fallback at all: both are 3.10, which is the floor. They stay in ``__all__``
because the rewriter routes every use of either through this module, and a
name that is sometimes here and sometimes not would make that rewrite
conditional on the floor.
"""

from __future__ import annotations

import sys
from typing import TypeAlias

import numpy as np

__all__ = ["Self", "TypeAlias", "trapezoid", "zip_strict"]

if sys.version_info >= (3, 11):
    from typing import Self
else:  # pragma: no cover - exercised only on Python 3.10
    from typing import Any

    try:
        from typing_extensions import Self
    except ImportError:
        Self = Any  # type: ignore[misc, assignment]

#: ``np.trapz`` was renamed in NumPy 2.0 and removed from the main namespace.
trapezoid = getattr(np, "trapezoid", None) or np.trapz

#: ``zip``'s ``strict=`` keyword is 3.10, which is the floor.
zip_strict = zip
