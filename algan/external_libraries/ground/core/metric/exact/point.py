from algan.external_libraries.ground.core.hints import (Point)
from algan.external_libraries.ground.core.primitive import rationalize


def point_squared_distance(first: Point, second: Point):
    return ((rationalize(first.x) - rationalize(second.x)) ** 2
            + (rationalize(first.y) - rationalize(second.y)) ** 2)
