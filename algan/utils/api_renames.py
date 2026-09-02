"""Superseded argument spellings, and the degrees-not-radians check beside them.

Algan carries no compatibility aliases: there is one spelling for each thing,
and Manim's spellings live in :mod:`algan.manim` (see
``agent_guidance/api_settings.md``). A reader arriving from Manim still writes
``mobject=`` and ``element_to_mobject=``, though, and a bare
``TypeError: got an unexpected keyword argument 'mobject'`` does not say that
``mob=`` is right there. These helpers word that error, in the same spirit as
``_MANIM_METHOD_HINTS`` in ``algan/animatable_base/mob.py``: the old name does
not exist and never will, and the message is the only thing it buys.

:func:`_warn_if_angle_looks_like_radians` is the other half of the same
migration problem. Algan states angles in degrees everywhere, so
``Arc(angle=PI / 2)`` is a 1.57 degree sliver rather than the quarter arc the
author meant -- a wrong picture with no error anywhere. A supplied angle that
is a non-integer float smaller than a full turn is almost certainly radians, so
it warns and says what the degree spelling would be. Whole numbers never warn:
``Arc(angle=5)`` is a legitimate five-degree arc.
"""

from __future__ import annotations

import functools
import math
import warnings

from algan.errors import AlganConfigurationError, ApproximationWarning

#: Manim's spelling -> Algan's, for arguments that reach the root namespace.
#: Both are Mobject-valued: ``mobject`` on the animations and the Mob-taking
#: constructors, ``element_to_mobject`` on the table classes.
_ROOT_KEYWORD_RENAMES: dict[str, str] = {
    "mobject": "mob",
    "element_to_mobject": "element_to_mob",
}

_FULL_TURN_RADIANS = 2 * math.pi
_RADIANS_TO_DEGREES = 180.0 / math.pi


def _reject_renamed_keywords(
    owner: str, kwargs, renames=None, *, manim_alternative: str | None = None
) -> None:
    """Raise if ``kwargs`` carries a superseded spelling, naming the new one.

    ``owner`` is the name to quote in the message -- the root spelling of the
    class or function the caller actually typed. ``manim_alternative`` is the
    ``algan.manim`` name to point at, for the root spellings of Manim classes
    that have one; Algan's own animations do not, and must not claim to.
    """
    for old, new in (renames or _ROOT_KEYWORD_RENAMES).items():
        if old in kwargs:
            pointer = (
                f", or reach for `algan.manim.{manim_alternative}`, where "
                "Manim's conventions -- and Manim's spellings -- apply"
                if manim_alternative is not None
                else ""
            )
            raise AlganConfigurationError(
                f"{owner}() has no `{old}` argument: Algan spells it `{new}`. "
                f"Pass `{new}=` instead{pointer}."
            )


def _renamed_keywords(**renames):
    """Decorate a root callable so a superseded keyword raises rather than TypeErrors.

    The wrapped function keeps its own signature, so ``help()`` and the
    reference show only the current spelling.
    """

    def decorate(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            _reject_renamed_keywords(func.__name__, kwargs, renames)
            return func(*args, **kwargs)

        return wrapper

    return decorate


def _looks_like_radians(value) -> bool:
    """Whether a supplied angle is more plausibly radians than Algan's degrees.

    True for a non-integer float strictly inside one turn. Integers are left
    alone in both directions: ``5`` is a real five-degree angle, and nobody
    writes a radian measure as a whole number on purpose.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    value = float(value)
    if not math.isfinite(value) or value == 0.0 or value.is_integer():
        return False
    return abs(value) < _FULL_TURN_RADIANS


def _format_number(value: float) -> str:
    degrees = round(value, 6)
    if float(degrees).is_integer():
        return f"{int(degrees)}"
    return f"{degrees:g}"


def _warn_if_angle_looks_like_radians(parameter: str, value) -> None:
    """Warn that ``parameter`` was probably written in radians, and say the fix."""
    if not _looks_like_radians(value):
        return
    warnings.warn(
        f"`{parameter}={float(value):g}` looks like radians; Algan takes "
        f"degrees (did you mean "
        f"`{parameter}={_format_number(float(value) * _RADIANS_TO_DEGREES)}`?)",
        ApproximationWarning,
        stacklevel=3,
    )
