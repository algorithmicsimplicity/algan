"""Rate functions: the easing curves animations are played through.

A rate function maps a normalized time ``t`` in ``[0, 1]`` to a normalized
progress, also usually in ``[0, 1]``. Every animation context takes one as
``easing``, and it decides the *feel* of a movement without changing its
runtime or its endpoints.

``smooth`` is the default: it eases in and out, so things start and stop gently.
``identity`` (and its alias ``linear``) is constant speed, which is what camera
moves and anything mechanical usually want. The standard Penner easing families
(sine, quad, cubic, quart, quint, expo, circ, back, elastic, bounce) provide
fine-grained control over acceleration and deceleration.

A rate function is just a callable on tensors, so writing your own is a
one-liner. See :doc:`/new_user_tutorials/combining_animations`.
"""

from __future__ import annotations

import math

import torch


def identity(t):
    """Linear progression with constant rate of change."""
    return t


linear = identity


#: ``smooth``'s three per-inflection constants, keyed by the inflection value.
#: The curve is the default easing, so this is evaluated once per animation
#: event per replay -- thousands of times in a batch of a mob-heavy scene --
#: and every one of those calls used to build a one-element tensor and run a
#: sigmoid, a negate, a divide and two more elementwise ops on it to arrive at
#: the same two numbers. Cached rather than folded into Python floats so the
#: values are bit-identical to what the per-call arithmetic produced.
_SMOOTH_CONSTANTS: dict[float, tuple] = {}


def _smooth_constants(inflection):
    """``(inflection_tensor, error, 1 - 2 * error)`` for ``smooth``."""
    cached = _SMOOTH_CONSTANTS.get(inflection)
    if cached is not None:
        return cached
    inflection_t = torch.tensor((inflection,))
    error = (-inflection_t / 2).sigmoid_()
    constants = (inflection_t, error, 1 - 2 * error)
    _SMOOTH_CONSTANTS[inflection] = constants
    return constants


def smooth(t, inflection=10.0):
    """Ease in and out along a rescaled sigmoid; the default easing.

    Maps 0 to 0 and 1 to 1, and is symmetric about ``t = 0.5``.

    Parameters
    ----------
    t
        Normalized time, a float or tensor in ``[0, 1]``.
    inflection
        Steepness of the sigmoid. Higher values linger longer at the ends
        and move faster through the middle. Defaults to ``10.0``.

    Returns
    -------
    torch.Tensor
        Progress, clamped to ``[0, 1]``.
    """
    if isinstance(inflection, (int, float)):
        inflection, error, scale = _smooth_constants(float(inflection))
    else:
        # Anything that is not a plain number is not a cache key; build the
        # constants exactly as this always did, including what that rejects.
        inflection = torch.tensor((inflection,))
        error = (-inflection / 2).sigmoid_()
        scale = 1 - 2 * error
    return (((inflection * (t - 0.5)).sigmoid_() - error) / scale).clamp_(min=0, max=1)


def delay_fade(t):
    """Rise linearly to halfway in the first 20% of the time, then accelerate to the end."""
    f = 0.2
    return ((t - f).clamp_min(0) / (1 - f)).pow(2) * 0.5 + (t / f).clamp_max(1) * 0.5


def pulse_fade(t):
    """Rise linearly to halfway in the first 20% of the time, then linearly to the end."""
    f = 0.2
    t_tensor = torch.as_tensor(t)
    return (t_tensor - f).clamp_min(0) * (0.5 / (1 - f)) + (t_tensor / f).clamp_max(
        1
    ) * 0.5


def inversed(f):
    """Mirror an easing curve, turning an ease-in into the matching ease-out.

    Returns ``lambda x: 1 - f(1 - x)``, the curve rotated half a turn about
    ``(0.5, 0.5)``; it keeps the endpoints 0 and 1.
    """
    return lambda x: 1 - f(1 - x)


# --- Sine ---
def ease_in_sine(t):
    """Start slowly and speed up, along a quarter cosine wave."""
    t_tensor = torch.as_tensor(t)
    return 1 - torch.cos((t_tensor * math.pi) / 2)


def ease_out_sine(t):
    """Start fast and slow down, along a quarter sine wave."""
    t_tensor = torch.as_tensor(t)
    return torch.sin((t_tensor * math.pi) / 2)


def ease_in_out_sine(t):
    """Ease in and out along a half cosine wave; the gentlest in-out curve."""
    t_tensor = torch.as_tensor(t)
    return -(torch.cos(t_tensor * math.pi) - 1) / 2


# --- Quad ---
def ease_in_quad(t):
    """Start slowly and speed up, as ``t ** 2``."""
    return t * t


def ease_out_quad(t):
    """Start fast and slow down; the mirror image of :func:`ease_in_quad`."""
    return 1 - (1 - t) * (1 - t)


def ease_in_out_quad(t):
    """Ease in and out with quadratic acceleration, meeting at ``t = 0.5``."""
    t_tensor = torch.as_tensor(t)
    m = (t_tensor < 0.5).float()
    return m * (2 * t_tensor * t_tensor) + (1 - m) * (
        1 - torch.pow(-2 * t_tensor + 2, 2) / 2
    )


# --- Cubic ---
def ease_in_cubic(t):
    """Start slowly and speed up, as ``t ** 3``."""
    return t * t * t


def ease_out_cubic(t):
    """Start fast and slow down; the mirror image of :func:`ease_in_cubic`."""
    return 1 - (1 - t) ** 3


def ease_in_out_cubic(t):
    """Ease in and out with cubic acceleration, meeting at ``t = 0.5``."""
    t_tensor = torch.as_tensor(t)
    m = (t_tensor < 0.5).float()
    return m * (4 * t_tensor**3) + (1 - m) * (1 - torch.pow(-2 * t_tensor + 2, 3) / 2)


# --- Quart ---
def ease_in_quart(t):
    """Start slowly and speed up, as ``t ** 4``."""
    return t**4


def ease_out_quart(t):
    """Start fast and slow down; the mirror image of :func:`ease_in_quart`."""
    return 1 - (1 - t) ** 4


def ease_in_out_quart(t):
    """Ease in and out with quartic acceleration, meeting at ``t = 0.5``."""
    t_tensor = torch.as_tensor(t)
    m = (t_tensor < 0.5).float()
    return m * (8 * t_tensor**4) + (1 - m) * (1 - torch.pow(-2 * t_tensor + 2, 4) / 2)


# --- Quint ---
def ease_in_quint(t):
    """Start slowly and speed up, as ``t ** 5``."""
    return t**5


def ease_out_quint(t):
    """Start fast and slow down; the mirror image of :func:`ease_in_quint`."""
    return 1 - (1 - t) ** 5


def ease_out_quintic(t):
    """Start fast and slow down; identical to :func:`ease_out_quint`."""
    return 1 - ((1 - t) ** 5)


def ease_in_out_quint(t):
    """Ease in and out with quintic acceleration, meeting at ``t = 0.5``."""
    t_tensor = torch.as_tensor(t)
    m = (t_tensor < 0.5).float()
    return m * (16 * t_tensor**5) + (1 - m) * (1 - torch.pow(-2 * t_tensor + 2, 5) / 2)


# --- Expo ---
def ease_out_exp(t, scale=4):
    """Start fast and slow down along a rescaled softplus curve.

    Gentler than :func:`ease_out_expo`. Maps 0 to 0 and 1 to 1; ``scale``
    (default ``4``) sets how sharply it decelerates.
    """

    def f(t):
        return -torch.nn.functional.softplus(-scale * (t - 0.5))

    s = f(torch.tensor((0.0,)))
    e = f(torch.tensor((1.0,)))
    return (f(t) - s) / (e - s)


def ease_out_exp_square(t):
    """Square of :func:`ease_out_exp`: a gentle start, then a soft landing.

    Maps 0 to 0 and 1 to 1.
    """
    o = ease_out_exp(t)
    return o**2


def ease_in_expo(t: float) -> float:
    """Start slowly and speed up exponentially, as a rescaled ``2 ** (5t - 5)``.

    Rescaled so it maps exactly 0 to 0 and 1 to 1. Milder than the
    standard Penner curve, which uses ``2 ** (10t - 10)``.
    """

    def f(t):
        s = 5
        return pow(2, s * t - s)

    s = f(torch.tensor((0.0,)))
    e = f(torch.tensor((1.0,)))
    return (f(t) - s) / (e - s)


def ease_out_expo(t: float) -> float:
    """Start fast and slow down exponentially; the mirror image of :func:`ease_in_expo`.

    Rescaled so it maps exactly 0 to 0 and 1 to 1.
    """

    def f(t):
        s = 5
        return 1 - pow(2, -s * t)

    s = f(torch.tensor((0.0,)))
    e = f(torch.tensor((1.0,)))
    return (f(t) - s) / (e - s)


# --- Circ ---
def ease_in_circ(t):
    """Start slowly and speed up along a quarter circle; very steep at the end."""
    t_tensor = torch.as_tensor(t).clamp(0, 1)
    return 1 - torch.sqrt((1 - torch.pow(t_tensor, 2)).clamp_min(0))


def ease_out_circ(t):
    """Start very fast and slow down along a quarter circle."""
    t_tensor = torch.as_tensor(t).clamp(0, 1)
    return torch.sqrt((1 - torch.pow(t_tensor - 1, 2)).clamp_min(0))


def ease_in_out_circ(t):
    """Ease in and out along two quarter circles, steepest at ``t = 0.5``."""
    t_tensor = torch.as_tensor(t).clamp(0, 1)
    m = (t_tensor < 0.5).float()
    return m * ((1 - torch.sqrt((1 - torch.pow(2 * t_tensor, 2)).clamp_min(0))) / 2) + (
        1 - m
    ) * ((torch.sqrt((1 - torch.pow(-2 * t_tensor + 2, 2)).clamp_min(0)) + 1) / 2)


# --- Back ---
def ease_in_back(t, s=1.70158):
    """Pull back slightly before speeding forward to the end.

    Overshoots below 0 near the start (to about ``-0.1`` with the default
    ``s = 1.70158``); larger ``s`` pulls back further. Ends exactly at 1.
    """
    t_tensor = torch.as_tensor(t)
    return (s + 1) * t_tensor**3 - s * t_tensor**2


def ease_out_back(t, s=1.70158):
    """Speed past the end, then settle back onto it.

    Overshoots above 1 near the end (to about ``1.1`` with the default
    ``s = 1.70158``); larger ``s`` overshoots further. Starts exactly at 0.
    """
    t_tensor = torch.as_tensor(t)
    return 1 + (s + 1) * (t_tensor - 1) ** 3 + s * (t_tensor - 1) ** 2


def ease_in_out_back(t, s=1.70158):
    """Pull back at the start and overshoot at the end.

    Dips to about ``-0.1`` and peaks at about ``1.1`` with the default
    ``s = 1.70158``, but starts at 0 and ends at 1.
    """
    c2 = s * 1.525
    t_tensor = torch.as_tensor(t)
    m = (t_tensor < 0.5).float()
    return m * (torch.pow(2 * t_tensor, 2) * ((c2 + 1) * 2 * t_tensor - c2) / 2) + (
        1 - m
    ) * (
        (torch.pow(2 * t_tensor - 2, 2) * ((c2 + 1) * (t_tensor * 2 - 2) + c2) + 2) / 2
    )


# --- Elastic ---
def ease_in_elastic(t):
    """Wind up with growing oscillations, then snap to the end.

    Swings well outside ``[0, 1]`` (down to about ``-0.37``) before t = 1;
    maps 0 to 0 and 1 to 1 exactly.
    """
    t_tensor = torch.as_tensor(t)
    c4 = (2 * math.pi) / 3
    return torch.where(
        t_tensor == 0,
        torch.zeros_like(t_tensor),
        torch.where(
            t_tensor == 1,
            torch.ones_like(t_tensor),
            -torch.pow(2.0, 10 * t_tensor - 10)
            * torch.sin((t_tensor * 10 - 10.75) * c4),
        ),
    )


def ease_out_elastic(t):
    """Snap past the end, then wobble into place like a spring.

    Overshoots to about ``1.37`` before settling; maps 0 to 0 and 1 to 1
    exactly.
    """
    t_tensor = torch.as_tensor(t)
    c4 = (2 * math.pi) / 3
    return torch.where(
        t_tensor == 0,
        torch.zeros_like(t_tensor),
        torch.where(
            t_tensor == 1,
            torch.ones_like(t_tensor),
            torch.pow(2.0, -10 * t_tensor) * torch.sin((t_tensor * 10 - 0.75) * c4) + 1,
        ),
    )


def ease_in_out_elastic(t):
    """Wobble out of the start and wobble into the end.

    Swings outside ``[0, 1]`` (about ``-0.12`` to ``1.12``); maps 0 to 0
    and 1 to 1 exactly.
    """
    t_tensor = torch.as_tensor(t)
    c5 = (2 * math.pi) / 4.5
    m = (t_tensor < 0.5).float()
    return torch.where(
        t_tensor == 0,
        torch.zeros_like(t_tensor),
        torch.where(
            t_tensor == 1,
            torch.ones_like(t_tensor),
            m
            * (
                -(
                    torch.pow(2.0, 20 * t_tensor - 10)
                    * torch.sin((20 * t_tensor - 11.125) * c5)
                )
                / 2
            )
            + (1 - m)
            * (
                (
                    torch.pow(2.0, -20 * t_tensor + 10)
                    * torch.sin((20 * t_tensor - 11.125) * c5)
                )
                / 2
                + 1
            ),
        ),
    )


# --- Bounce ---
def _ease_out_bounce_tensor(x):
    n1 = 7.5625
    d1 = 2.75
    cond1 = x < (1 / d1)
    cond2 = (~cond1) & (x < (2 / d1))
    cond3 = (~cond1) & (~cond2) & (x < (2.5 / d1))
    cond4 = (~cond1) & (~cond2) & (~cond3)

    res = torch.zeros_like(x)
    res = torch.where(cond1, n1 * x * x, res)
    x2 = x - (1.5 / d1)
    res = torch.where(cond2, n1 * x2 * x2 + 0.75, res)
    x3 = x - (2.25 / d1)
    res = torch.where(cond3, n1 * x3 * x3 + 0.9375, res)
    x4 = x - (2.625 / d1)
    res = torch.where(cond4, n1 * x4 * x4 + 0.984375, res)
    return res


def ease_out_bounce(t):
    """Fall to the end and bounce on it, with shrinking bounces. Stays in ``[0, 1]``."""
    t_tensor = torch.as_tensor(t)
    return _ease_out_bounce_tensor(t_tensor)


def ease_in_bounce(t):
    """Bounce on the start with growing bounces, then leave. Stays in ``[0, 1]``."""
    t_tensor = torch.as_tensor(t)
    return 1 - _ease_out_bounce_tensor(1 - t_tensor)


def ease_in_out_bounce(t):
    """Bounce away from the start and bounce into the end. Stays in ``[0, 1]``."""
    t_tensor = torch.as_tensor(t)
    m = (t_tensor < 0.5).float()
    return m * ((1 - _ease_out_bounce_tensor(1 - 2 * t_tensor)) / 2) + (1 - m) * (
        (1 + _ease_out_bounce_tensor(2 * t_tensor - 1)) / 2
    )


# --- Animation flow helpers ---
rush_into = ease_in_cubic
rush_from = ease_out_cubic
slow_into = ease_out_quad


def tan(t, scale=10):
    """Decelerate into the middle, then accelerate out to the end.

    Combines :func:`ease_out_expo` on the first half with
    :func:`ease_in_expo` on the second. Note that the curve jumps at
    ``t = 0.5`` (from about ``0.42`` to ``0.58``), and ``scale`` is
    currently ignored. Maps 0 to 0 and 1 to 1.
    """
    t_tensor = torch.as_tensor(t)
    m = (t_tensor < 0.5).float()
    return m * ease_out_expo(t_tensor) * 0.5 + (1 - m) * (
        ease_in_expo(t_tensor) * 0.5 + 0.5
    )


# Explicit, so ``from algan.easings import *`` brings in the curves and not
# this module's own ``math`` and ``torch`` imports, and so the documentation
# tooling has a definitive roster to render. (``dir()`` still lists the imports
# -- ``__all__`` governs star-imports and autodoc, not attribute listing.)
__all__ = [
    "delay_fade",
    "ease_in_back",
    "ease_in_bounce",
    "ease_in_circ",
    "ease_in_cubic",
    "ease_in_elastic",
    "ease_in_expo",
    "ease_in_out_back",
    "ease_in_out_bounce",
    "ease_in_out_circ",
    "ease_in_out_cubic",
    "ease_in_out_elastic",
    "ease_in_out_quad",
    "ease_in_out_quart",
    "ease_in_out_quint",
    "ease_in_out_sine",
    "ease_in_quad",
    "ease_in_quart",
    "ease_in_quint",
    "ease_in_sine",
    "ease_out_back",
    "ease_out_bounce",
    "ease_out_circ",
    "ease_out_cubic",
    "ease_out_elastic",
    "ease_out_exp",
    "ease_out_exp_square",
    "ease_out_expo",
    "ease_out_quad",
    "ease_out_quart",
    "ease_out_quint",
    "ease_out_quintic",
    "ease_out_sine",
    "identity",
    "inversed",
    "linear",
    "pulse_fade",
    "rush_from",
    "rush_into",
    "slow_into",
    "smooth",
    "tan",
]
