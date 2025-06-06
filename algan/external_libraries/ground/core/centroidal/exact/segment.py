from typing import Type

from algan.external_libraries.ground.core.hints import (Point,
                               Segment)
from algan.external_libraries.ground.core.primitive import rationalize


def centroid(segment: Segment, point_cls: Type[Point]) -> Point:
    return point_cls((rationalize(segment.start.x)
                      + rationalize(segment.end.x)) / 2,
                     (rationalize(segment.start.y)
                      + rationalize(segment.end.y)) / 2)
