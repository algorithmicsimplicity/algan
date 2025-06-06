from typing import Type

from algan.external_libraries.ground.core.hints import (Box,
                               Point,
                               QuaternaryPointFunction,
                               Segment)
from algan.external_libraries.ground.core.primitive import rationalize
from .segment import (point_squared_distance as segment_point_squared_distance,
                      segment_squared_distance
                      as segment_segment_squared_distance)


def point_squared_distance(box: Box, point: Point):
    return (_linear_interval_distance(rationalize(box.min_x),
                                      rationalize(box.max_x),
                                      rationalize(point.x)) ** 2
            + _linear_interval_distance(rationalize(box.min_y),
                                        rationalize(box.max_y),
                                        rationalize(point.y)) ** 2)


def segment_squared_distance(box: Box,
                             segment: Segment,
                             dot_producer: QuaternaryPointFunction[None],
                             segments_collision_detector
                             : QuaternaryPointFunction[bool],
                             point_cls: Type[Point]):
    segment_start, segment_end = segment.start, segment.end
    min_x, min_y, max_x, max_y = (rationalize(box.min_x),
                                  rationalize(box.min_y),
                                  rationalize(box.max_x),
                                  rationalize(box.max_y))
    return (0
            if ((min_x <= segment_start.x <= max_x
                 and min_y <= segment_start.y <= max_y)
                or (min_x <= segment_end.x <= max_x
                    and min_y <= segment_end.y <= max_y))
            else
            ((segment_point_squared_distance(segment_start, segment_end,
                                             point_cls(min_x, min_y),
                                             dot_producer)
              if min_y == max_y
              else segment_segment_squared_distance(
                    segment_start, segment_end, point_cls(min_x, min_y),
                    point_cls(min_x, max_y), dot_producer,
                    segments_collision_detector))
             if min_x == max_x
             else (segment_segment_squared_distance(
                    segment_start, segment_end, point_cls(min_x, min_y),
                    point_cls(max_x, min_y), dot_producer,
                    segments_collision_detector)
                   if min_y == max_y
                   else _non_degenerate_segment_squared_distance(
                    max_x, max_y, min_x, min_y, segment_start, segment_end,
                    dot_producer, segments_collision_detector, point_cls))))


def _linear_interval_distance(min_coordinate,
                              max_coordinate,
                              coordinate):
    return (min_coordinate - coordinate
            if coordinate < min_coordinate
            else (coordinate - max_coordinate
                  if coordinate > max_coordinate
                  else 0))


def _non_degenerate_segment_squared_distance(
        max_x,
        max_y,
        min_x,
        min_y,
        segment_start: Point,
        segment_end: Point,
        dot_producer: QuaternaryPointFunction[None],
        segments_collision_detector: QuaternaryPointFunction[bool],
        point_cls: Type[Point]):
    bottom_left, bottom_right = (point_cls(min_x, min_y),
                                 point_cls(max_x, min_y))
    bottom_side_distance = segment_segment_squared_distance(
            segment_start, segment_end, bottom_left, bottom_right,
            dot_producer, segments_collision_detector)
    if not bottom_side_distance:
        return bottom_side_distance
    top_right = point_cls(max_x, max_y)
    right_side_distance = segment_segment_squared_distance(
            segment_start, segment_end, bottom_right, top_right, dot_producer,
            segments_collision_detector)
    if not right_side_distance:
        return right_side_distance
    top_left = point_cls(min_x, max_y)
    top_side_distance = segment_segment_squared_distance(
            segment_start, segment_end, top_left, top_right, dot_producer,
            segments_collision_detector)
    if not top_side_distance:
        return top_side_distance
    left_side_distance = segment_segment_squared_distance(
            segment_start, segment_end, bottom_left, top_left, dot_producer,
            segments_collision_detector)
    return (left_side_distance
            and min(bottom_side_distance, right_side_distance,
                    top_side_distance, left_side_distance))
