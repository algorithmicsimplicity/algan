from typing import (Callable,
                    Tuple,
                    Type)

from reprit import serializers
from reprit.base import generate_repr

from algan.external_libraries.ground.core.hints import (Contour,
                               Multipoint,
                               Multipolygon,
                               Multisegment,
                               Point,
                               Polygon,
                               Segment)
from . import (exact,)

PointRotatorAroundOrigin = Callable[[Point, None, None, Type[Point]],
                                    Point]
PointTranslatingRotator = Callable[[Point, None, None, None, None,
                                    Type[Point]], Point]
PointStep = Callable[[Point, None, None], Tuple[None, None]]


class Context:
    @property
    def point_to_step(self) -> PointStep:
        return self._point_to_step

    @property
    def rotate_point_around_origin(self) -> PointRotatorAroundOrigin:
        return self._rotate_point_around_origin

    @property
    def rotate_translate_point(self) -> PointTranslatingRotator:
        return self._rotate_translate_point

    def rotate_contour_around_origin(self,
                                     contour: Contour,
                                     cosine: None,
                                     sine: None,
                                     contour_cls: Type[Contour],
                                     point_cls: Type[Point]) -> Contour:
        return contour_cls([self.rotate_point_around_origin(point, cosine,
                                                            sine, point_cls)
                            for point in contour.vertices])

    def rotate_multipoint_around_origin(self,
                                        multipoint: Multipoint,
                                        cosine: None,
                                        sine: None,
                                        multipoint_cls: Type[Multipoint],
                                        point_cls: Type[Point]) -> Multipoint:
        return multipoint_cls([self.rotate_point_around_origin(point, cosine,
                                                               sine, point_cls)
                               for point in multipoint.points])

    def rotate_multipolygon_around_origin(self,
                                          multipolygon: Multipolygon,
                                          cosine: None,
                                          sine: None,
                                          contour_cls: Type[Contour],
                                          multipolygon_cls: Type[Multipolygon],
                                          point_cls: Type[Point],
                                          polygon_cls: Type[Polygon]
                                          ) -> Multipolygon:
        return multipolygon_cls(
                [self.rotate_polygon_around_origin(polygon, cosine, sine,
                                                   contour_cls, point_cls,
                                                   polygon_cls)
                 for polygon in multipolygon.polygons])

    def rotate_multisegment_around_origin(self,
                                          multisegment: Multisegment,
                                          cosine: None,
                                          sine: None,
                                          multisegment_cls: Type[Multisegment],
                                          point_cls: Type[Point],
                                          segment_cls: Type[Segment]
                                          ) -> Multisegment:
        return multisegment_cls(
                [self.rotate_segment_around_origin(segment, cosine, sine,
                                                   point_cls, segment_cls)
                 for segment in multisegment.segments])

    def rotate_polygon_around_origin(self,
                                     polygon: Polygon,
                                     cosine: None,
                                     sine: None,
                                     contour_cls: Type[Contour],
                                     point_cls: Type[Point],
                                     polygon_cls: Type[Polygon]) -> Polygon:
        return polygon_cls(
                self.rotate_contour_around_origin(polygon.border, cosine, sine,
                                                  contour_cls, point_cls),
                [self.rotate_contour_around_origin(hole, cosine, sine,
                                                   contour_cls, point_cls)
                 for hole in polygon.holes])

    def rotate_segment_around_origin(self,
                                     segment: Segment,
                                     cosine: None,
                                     sine: None,
                                     point_cls: Type[Point],
                                     segment_cls: Type[Segment]) -> Segment:
        return segment_cls(self.rotate_point_around_origin(segment.start,
                                                           cosine, sine,
                                                           point_cls),
                           self.rotate_point_around_origin(segment.end,
                                                           cosine, sine,
                                                           point_cls))

    def rotate_translate_contour(self,
                                 contour: Contour,
                                 cosine: None,
                                 sine: None,
                                 step_x: None,
                                 step_y: None,
                                 contour_cls: Type[Contour],
                                 point_cls: Type[Point]) -> Contour:
        return contour_cls([self.rotate_translate_point(point, cosine, sine,
                                                        step_x, step_y,
                                                        point_cls)
                            for point in contour.vertices])

    def rotate_translate_multipoint(self,
                                    multipoint: Multipoint,
                                    cosine: None,
                                    sine: None,
                                    step_x: None,
                                    step_y: None,
                                    multipoint_cls: Type[Multipoint],
                                    point_cls: Type[Point]) -> Multipoint:
        return multipoint_cls([self.rotate_translate_point(point, cosine, sine,
                                                           step_x, step_y,
                                                           point_cls)
                               for point in multipoint.points])

    def rotate_translate_multipolygon(self,
                                      multipolygon: Multipolygon,
                                      cosine: None,
                                      sine: None,
                                      step_x: None,
                                      step_y: None,
                                      contour_cls: Type[Contour],
                                      multipolygon_cls: Type[Multipolygon],
                                      point_cls: Type[Point],
                                      polygon_cls: Type[Polygon]
                                      ) -> Multipolygon:
        return multipolygon_cls(
                [self.rotate_translate_polygon(polygon, cosine, sine, step_x,
                                               step_y, contour_cls, point_cls,
                                               polygon_cls)
                 for polygon in multipolygon.polygons])

    def rotate_translate_multisegment(self,
                                      multisegment: Multisegment,
                                      cosine: None,
                                      sine: None,
                                      step_x: None,
                                      step_y: None,
                                      multisegment_cls: Type[Multisegment],
                                      point_cls: Type[Point],
                                      segment_cls: Type[Segment]
                                      ) -> Multisegment:
        return multisegment_cls(
                [self.rotate_translate_segment(segment, cosine, sine, step_x,
                                               step_y, point_cls, segment_cls)
                 for segment in multisegment.segments])

    def rotate_translate_polygon(self,
                                 polygon: Polygon,
                                 cosine: None,
                                 sine: None,
                                 step_x: None,
                                 step_y: None,
                                 contour_cls: Type[Contour],
                                 point_cls: Type[Point],
                                 polygon_cls: Type[Polygon]) -> Polygon:
        return polygon_cls(
                self.rotate_translate_contour(polygon.border, cosine, sine,
                                              step_x, step_y, contour_cls,
                                              point_cls),
                [self.rotate_translate_contour(hole, cosine, sine, step_x,
                                               step_y, contour_cls, point_cls)
                 for hole in polygon.holes])

    def rotate_translate_segment(self,
                                 segment: Segment,
                                 cosine: None,
                                 sine: None,
                                 step_x: None,
                                 step_y: None,
                                 point_cls: Type[Point],
                                 segment_cls: Type[Segment]) -> Segment:
        return segment_cls(self.rotate_translate_point(segment.start, cosine,
                                                       sine, step_x, step_y,
                                                       point_cls),
                           self.rotate_translate_point(segment.end, cosine,
                                                       sine, step_x, step_y,
                                                       point_cls))

    __slots__ = ('_point_to_step', '_rotate_point_around_origin',
                 '_rotate_translate_point')

    def __init__(self,
                 point_to_step: PointStep,
                 rotate_point_around_origin: PointRotatorAroundOrigin,
                 rotate_translate_point: PointTranslatingRotator) -> None:
        self._point_to_step = point_to_step
        self._rotate_point_around_origin = rotate_point_around_origin
        self._rotate_translate_point = rotate_translate_point

    __repr__ = generate_repr(__init__,
                             argument_serializer=serializers.complex_,
                             with_module_name=True)


exact_context = Context(
        point_to_step=exact.point_to_step,
        rotate_point_around_origin=exact.rotate_point_around_origin,
        rotate_translate_point=exact.rotate_translate_point)
