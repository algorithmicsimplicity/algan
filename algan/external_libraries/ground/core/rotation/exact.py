from typing import (Tuple,
                    Type)

from algan.external_libraries.ground.core.hints import (Point,)
from algan.external_libraries.ground.core.primitive import rationalize


def rotate_point_around_origin(point: Point,
                               cosine,
                               sine,
                               point_cls: Type[Point]) -> Point:
    cosine, sine = rationalize(cosine), rationalize(sine)
    x, y = rationalize(point.x), rationalize(point.y)
    return point_cls(cosine * x - sine * y, sine * x + cosine * y)


def rotate_translate_point(point: Point,
                           cosine,
                           sine,
                           step_x,
                           step_y,
                           point_cls: Type[Point]) -> Point:
    x, y = rationalize(point.x), rationalize(point.y)
    cosine, sine = rationalize(cosine), rationalize(sine)
    return point_cls(cosine * x - sine * y + rationalize(step_x),
                     sine * x + cosine * y + rationalize(step_y))


def point_to_step(point: Point,
                  cosine,
                  sine):
    x, y = rationalize(point.x), rationalize(point.y)
    cosine, sine = rationalize(cosine), rationalize(sine)
    return x - (cosine * x - sine * y), y - (sine * x + cosine * y)
