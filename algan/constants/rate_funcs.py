"""Rate functions: the easing curves animations are played through.

A rate function maps a normalized time ``t`` in ``[0, 1]`` to a normalized
progress, also usually in ``[0, 1]``. Every animation context takes one as
``rate_func``, and it decides the *feel* of a movement without changing its
duration or its endpoints.

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


def smooth(t, inflection=10.0):
    inflection = torch.tensor((inflection,))
    error = (-inflection / 2).sigmoid_()
    return (((inflection * (t - 0.5)).sigmoid_() - error) / (1 - 2 * error)).clamp_(
        min=0, max=1
    )


def delay_fade(t):
    """Fast initial rise followed by parabolic settling."""
    f = 0.2
    return ((t - f).clamp_min(0) / (1 - f)).pow(2) * 0.5 + (t / f).clamp_max(1) * 0.5


def pulse_fade(t):
    """Linear blend with rapid initial ramp."""
    f = 0.2
    t_tensor = torch.as_tensor(t)
    return (t_tensor - f).clamp_min(0) * (0.5 / (1 - f)) + (t_tensor / f).clamp_max(
        1
    ) * 0.5


def inversed(f):
    """Reverses an easing function."""
    return lambda x: 1 - f(1 - x)


# --- Sine ---
def ease_in_sine(t):
    t_tensor = torch.as_tensor(t)
    return 1 - torch.cos((t_tensor * math.pi) / 2)


def ease_out_sine(t):
    t_tensor = torch.as_tensor(t)
    return torch.sin((t_tensor * math.pi) / 2)


def ease_in_out_sine(t):
    t_tensor = torch.as_tensor(t)
    return -(torch.cos(t_tensor * math.pi) - 1) / 2


# --- Quad ---
def ease_in_quad(t):
    return t * t


def ease_out_quad(t):
    return 1 - (1 - t) * (1 - t)


def ease_in_out_quad(t):
    t_tensor = torch.as_tensor(t)
    m = (t_tensor < 0.5).float()
    return m * (2 * t_tensor * t_tensor) + (1 - m) * (
        1 - torch.pow(-2 * t_tensor + 2, 2) / 2
    )


# --- Cubic ---
def ease_in_cubic(t):
    return t * t * t


def ease_out_cubic(t):
    return 1 - (1 - t) ** 3


def ease_in_out_cubic(t):
    t_tensor = torch.as_tensor(t)
    m = (t_tensor < 0.5).float()
    return m * (4 * t_tensor**3) + (1 - m) * (1 - torch.pow(-2 * t_tensor + 2, 3) / 2)


# --- Quart ---
def ease_in_quart(t):
    return t**4


def ease_out_quart(t):
    return 1 - (1 - t) ** 4


def ease_in_out_quart(t):
    t_tensor = torch.as_tensor(t)
    m = (t_tensor < 0.5).float()
    return m * (8 * t_tensor**4) + (1 - m) * (1 - torch.pow(-2 * t_tensor + 2, 4) / 2)


# --- Quint ---
def ease_in_quint(t):
    return t**5


def ease_out_quint(t):
    return 1 - (1 - t) ** 5


def ease_out_quintic(t):
    return 1 - ((1 - t) ** 5)


def ease_in_out_quint(t):
    t_tensor = torch.as_tensor(t)
    m = (t_tensor < 0.5).float()
    return m * (16 * t_tensor**5) + (1 - m) * (1 - torch.pow(-2 * t_tensor + 2, 5) / 2)


# --- Expo ---
def ease_out_exp(t, scale=4):
    def f(t):
        return -torch.nn.functional.softplus(-scale * (t - 0.5))

    s = f(torch.tensor((0.0,)))
    e = f(torch.tensor((1.0,)))
    return (f(t) - s) / (e - s)


def ease_out_exp_square(t):
    o = ease_out_exp(t)
    return o**2


def ease_in_expo(t: float) -> float:
    def f(t):
        s = 5
        return pow(2, s * t - s)

    s = f(torch.tensor((0.0,)))
    e = f(torch.tensor((1.0,)))
    return (f(t) - s) / (e - s)


def ease_out_expo(t: float) -> float:
    def f(t):
        s = 5
        return 1 - pow(2, -s * t)

    s = f(torch.tensor((0.0,)))
    e = f(torch.tensor((1.0,)))
    return (f(t) - s) / (e - s)


# --- Circ ---
def ease_in_circ(t):
    t_tensor = torch.as_tensor(t).clamp(0, 1)
    return 1 - torch.sqrt((1 - torch.pow(t_tensor, 2)).clamp_min(0))


def ease_out_circ(t):
    t_tensor = torch.as_tensor(t).clamp(0, 1)
    return torch.sqrt((1 - torch.pow(t_tensor - 1, 2)).clamp_min(0))


def ease_in_out_circ(t):
    t_tensor = torch.as_tensor(t).clamp(0, 1)
    m = (t_tensor < 0.5).float()
    return m * ((1 - torch.sqrt((1 - torch.pow(2 * t_tensor, 2)).clamp_min(0))) / 2) + (
        1 - m
    ) * ((torch.sqrt((1 - torch.pow(-2 * t_tensor + 2, 2)).clamp_min(0)) + 1) / 2)


# --- Back ---
def ease_in_back(t, s=1.70158):
    t_tensor = torch.as_tensor(t)
    return (s + 1) * t_tensor**3 - s * t_tensor**2


def ease_out_back(t, s=1.70158):
    t_tensor = torch.as_tensor(t)
    return 1 + (s + 1) * (t_tensor - 1) ** 3 + s * (t_tensor - 1) ** 2


def ease_in_out_back(t, s=1.70158):
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
    t_tensor = torch.as_tensor(t)
    return _ease_out_bounce_tensor(t_tensor)


def ease_in_bounce(t):
    t_tensor = torch.as_tensor(t)
    return 1 - _ease_out_bounce_tensor(1 - t_tensor)


def ease_in_out_bounce(t):
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
    t_tensor = torch.as_tensor(t)
    m = (t_tensor < 0.5).float()
    return m * ease_out_expo(t_tensor) * 0.5 + (1 - m) * (
        ease_in_expo(t_tensor) * 0.5 + 0.5
    )
