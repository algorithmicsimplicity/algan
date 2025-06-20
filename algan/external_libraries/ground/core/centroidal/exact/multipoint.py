from typing import (Callable,
                    Type)

from fractions import Fraction

from algan.external_libraries.ground.core.hints import (Multipoint,
                               Point)
from algan.external_libraries.ground.core.primitive import rationalize


def centroid(multipoint: Multipoint,
             point_cls: Type[Point],
             inverse: Callable[[int], Fraction] = Fraction(1).__truediv__
             ) -> Point:
    result_x = result_y = 0
    for point in multipoint.points:
        result_x += rationalize(point.x)
        result_y += rationalize(point.y)
    inverted_points_count = inverse(len(multipoint.points))
    return point_cls(result_x * inverted_points_count,
                     result_y * inverted_points_count)
