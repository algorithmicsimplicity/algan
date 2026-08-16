"""Rate functions: the easing curves animations are played through.

A rate function maps a normalized time ``t`` in ``[0, 1]`` to a normalized
progress, also usually in ``[0, 1]``. Every animation context takes one as
``rate_func``, and it decides the *feel* of a movement without changing its
duration or its endpoints.

``smooth`` is the default: it eases in and out, so things start and stop gently.
``identity`` (and its alias ``linear``) is constant speed, which is what camera
moves and anything mechanical usually want. The ``ease_out_*`` family starts fast
and settles, ``delay_fade`` and ``pulse_fade`` are shaped for appearance and
attention effects, and ``inversed`` reverses any function you hand it.

A rate function is just a callable on tensors, so writing your own is a
one-liner. See :doc:`/new_user_tutorials/controlling_animations`.
"""

from __future__ import annotations

import torch


def identity(t):
    return t


linear = identity


def smooth(t, inflection=10.0):
    inflection = torch.tensor((inflection,))
    error = (-inflection / 2).sigmoid_()
    return (((inflection * (t - 0.5)).sigmoid_() - error) / (1 - 2 * error)).clamp_(
        min=0, max=1
    )


def delay_fade(t):
    f = 0.2
    return ((t - f).clamp_min(0) / (1 - f)).pow(2) * 0.5 + (t / f).clamp_max(1) * 0.5


def pulse_fade(t):
    f = 0.2
    return (t - f).clamp_min(0) * (0.5 / (1 - f)) + (t / f).clamp_max(1) * 0.5
    # return (t*0 + 0.25) * (t > 0.1) +
    t = 1 - t
    f = 0.00
    m = t < f
    t * m * 5 + (~m) * (((t - f) / (1 - f)) * 0.5 + 0.5)
    t = 1 - t
    return t


def ease_out_quintic(t):
    return 1 - ((1 - t) ** 5)


def ease_out_exp(t, scale=4):
    def f(t):
        return -torch.nn.functional.softplus(-scale * (t - 0.5))

    s = f(torch.tensor((0.0,)))
    e = f(torch.tensor((1.0,)))
    return (f(t) - s) / (e - s)


def inversed(f):
    return lambda x: 1 - f(1 - x)


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


def tan(t, scale=10):
    m = (t < 0.5).float()
    return m * ease_out_expo(t) * 0.5 + (1 - m) * (ease_in_expo(t) * 0.5 + 0.5)
