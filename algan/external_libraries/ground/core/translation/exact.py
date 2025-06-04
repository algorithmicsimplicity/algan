from typing import Type

from algan.external_libraries.ground.core.hints import (Point)
from algan.external_libraries.ground.core.primitive import rationalize


def translate_point(point: Point,
                    step_x,
                    step_y,
                    point_cls: Type[Point]) -> Point:
    return point_cls(rationalize(point.x) + rationalize(step_x),
                     rationalize(point.y) + rationalize(step_y))
