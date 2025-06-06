from typing import (Callable,
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
from .exact import (contour as exact_contour,
                    multipoint as exact_multipoint,
                    multipolygon as exact_multipolygon,
                    multisegment as exact_multisegment,
                    polygon as exact_polygon,
                    region as exact_region,
                    segment as exact_segment)


ContourCentroid = Callable[[Contour, Type[Point], None], Point]
MultipointCentroid = Callable[[Multipoint, Type[Point]], Point]
MultipolygonCentroid = Callable[[Multipolygon, Type[Point]], Point]
MultisegmentCentroid = Callable[[Multisegment, Type[Point], None],
                                Point]
PolygonCentroid = Callable[[Polygon, Type[Point]], Point]
RegionCentroid = Callable[[Contour, Type[Point]], Point]
SegmentCentroid = Callable[[Segment, Type[Point]], Point]


class Context:
    __slots__ = ('_contour_centroid', '_multipoint_centroid',
                 '_multipolygon_centroid', '_multisegment_centroid',
                 '_polygon_centroid', '_region_centroid', '_segment_centroid')

    def __init__(self,
                 contour_centroid: ContourCentroid,
                 multipoint_centroid: MultipointCentroid,
                 multipolygon_centroid: MultipolygonCentroid,
                 multisegment_centroid: MultisegmentCentroid,
                 polygon_centroid: PolygonCentroid,
                 region_centroid: RegionCentroid,
                 segment_centroid: SegmentCentroid) -> None:
        self._contour_centroid = contour_centroid
        self._multipoint_centroid = multipoint_centroid
        self._multipolygon_centroid = multipolygon_centroid
        self._multisegment_centroid = multisegment_centroid
        self._polygon_centroid = polygon_centroid
        self._region_centroid = region_centroid
        self._segment_centroid = segment_centroid

    __repr__ = generate_repr(__init__,
                             argument_serializer=serializers.complex_,
                             with_module_name=True)

    @property
    def contour_centroid(self) -> ContourCentroid:
        return self._contour_centroid

    @property
    def multipoint_centroid(self) -> MultipointCentroid:
        return self._multipoint_centroid

    @property
    def multipolygon_centroid(self) -> MultipolygonCentroid:
        return self._multipolygon_centroid

    @property
    def multisegment_centroid(self) -> MultisegmentCentroid:
        return self._multisegment_centroid

    @property
    def polygon_centroid(self) -> PolygonCentroid:
        return self._polygon_centroid

    @property
    def region_centroid(self) -> RegionCentroid:
        return self._region_centroid

    @property
    def segment_centroid(self) -> SegmentCentroid:
        return self._segment_centroid


exact_context = Context(contour_centroid=exact_contour.centroid,
                        multipoint_centroid=exact_multipoint.centroid,
                        multipolygon_centroid=exact_multipolygon.centroid,
                        multisegment_centroid=exact_multisegment.centroid,
                        polygon_centroid=exact_polygon.centroid,
                        region_centroid=exact_region.centroid,
                        segment_centroid=exact_segment.centroid)
