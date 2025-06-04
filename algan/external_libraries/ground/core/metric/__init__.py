from typing import (Callable,
                    Type)

from reprit import serializers
from reprit.base import generate_repr

from algan.external_libraries.ground.core.hints import (Box,
                               Point,
                               QuaternaryPointFunction,
                               Segment)
from .exact import (box as exact_box,
                    point as exact_point,
                    segment as exact_segment)

BoxPointMetric = Callable[[Box, Point], None]
BoxSegmentMetric = Callable[[Box, Segment, QuaternaryPointFunction[None],
                             QuaternaryPointFunction[bool], Type[Point]],
                            None]
PointPointMetric = Callable[[Point, Point], None]
SegmentPointMetric = Callable[[Point, Point, Point,
                               QuaternaryPointFunction[None]],
                              None]
SegmentSegmentMetric = Callable[[Point, Point, Point, Point,
                                 QuaternaryPointFunction[None],
                                 QuaternaryPointFunction[bool]], None]


class Context:
    __slots__ = ('_box_point_squared_metric', '_box_segment_squared_metric',
                 '_point_point_squared_metric',
                 '_segment_point_squared_metric',
                 '_segment_segment_squared_metric')

    def __init__(self,
                 *,
                 box_point_squared_metric: BoxPointMetric,
                 box_segment_squared_metric: BoxSegmentMetric,
                 point_point_squared_metric: PointPointMetric,
                 segment_point_squared_metric: SegmentPointMetric,
                 segment_segment_squared_metric: SegmentSegmentMetric
                 ) -> None:
        self._box_point_squared_metric = box_point_squared_metric
        self._box_segment_squared_metric = box_segment_squared_metric
        self._point_point_squared_metric = point_point_squared_metric
        self._segment_point_squared_metric = segment_point_squared_metric
        self._segment_segment_squared_metric = segment_segment_squared_metric

    __repr__ = generate_repr(__init__,
                             argument_serializer=serializers.complex_,
                             with_module_name=True)

    @property
    def box_point_squared_metric(self) -> BoxPointMetric:
        return self._box_point_squared_metric

    @property
    def box_segment_squared_metric(self) -> BoxSegmentMetric:
        return self._box_segment_squared_metric

    @property
    def point_point_squared_metric(self) -> PointPointMetric:
        return self._point_point_squared_metric

    @property
    def segment_point_squared_metric(self) -> SegmentPointMetric:
        return self._segment_point_squared_metric

    @property
    def segment_segment_squared_metric(self) -> SegmentSegmentMetric:
        return self._segment_segment_squared_metric


exact_context = Context(
        box_point_squared_metric=exact_box.point_squared_distance,
        box_segment_squared_metric=exact_box.segment_squared_distance,
        point_point_squared_metric=exact_point.point_squared_distance,
        segment_point_squared_metric=exact_segment.point_squared_distance,
        segment_segment_squared_metric=exact_segment.segment_squared_distance)

