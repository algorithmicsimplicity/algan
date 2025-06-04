from typing import Type

from algan.external_libraries.ground.core.hints import (Point,
                               Scalar)
from algan.external_libraries.ground.core.primitive import rationalize


def scale_point(point: Point,
                factor_x: Scalar,
                factor_y: Scalar,
                point_cls: Type[Point]) -> Point:
    return point_cls(rationalize(point.x) * rationalize(factor_x),
                     rationalize(point.y) * rationalize(factor_y))
