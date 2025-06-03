import math

import torch


def identity(t):
    return t


def smooth(t, inflection=10.0):
    inflection = torch.tensor((inflection,))
    error = (-inflection / 2).sigmoid_()
    return (((inflection * (t - 0.5)).sigmoid_() - error) / (1 - 2 * error)).clamp_(min=0,max=1)


def ease_out_quintic(t):
    return 1 - ((1 - t) ** 5)


def ease_out_exp(t, scale=10):
    def f(t):
        return -torch.nn.functional.softplus(-scale*(t-0.5))

    s = f(torch.tensor((0.0,)))
    e = f(torch.tensor((1.0,)))
    return (f(t) - s) / (e-s)


def ease_out_exp(t, scale=4):
    def f(t):
        return -torch.nn.functional.softplus(-scale*(t-0.5))

    s = f(torch.tensor((0.0,)))
    e = f(torch.tensor((1.0,)))
    return (f(t) - s) / (e-s)


def inversed(f):
    return lambda x: 1-f(1-x)


def ease_out_exp_square(t):
    o = ease_out_exp(t)
    return o**2


def ease_in_expo(t: float) -> float:
    def f(t):
        s = 2
        return pow(2, s * t - s)

    s = f(torch.tensor((0.0,)))
    e = f(torch.tensor((1.0,)))
    return (f(t) - s) / (e - s)


def ease_out_expo(t: float) -> float:
    def f(t):
        s = 2
        return 1 - pow(2, -s * t)

    s = f(torch.tensor((0.0,)))
    e = f(torch.tensor((1.0,)))
    return (f(t) - s) / (e - s)


def tan(t, scale=10):
    m = (t < 0.5).float()
    return m * ease_out_expo(t) * 0.5 + (1-m) * (ease_in_expo(t) * 0.5 + 0.5)
    #return eas
    #return torch.tan((t-0.5)*2 * 2 / math.pi)