from numbers import (Rational,
                     Real)
from typing import Type

from cfractions import Fraction

from .hints import (Point,)


def rationalize(value):
    try:
        return Fraction(value)
    except TypeError:
        return value


def square(value):
    return value * value


def to_rational_point(point,
                      point_cls):
    return point_cls(rationalize(point.x), rationalize(point.y))


def to_sign(value) -> int:
    return 1 if value > 0 else (-1 if value else 0)
