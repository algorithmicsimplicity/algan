from reprit import serializers
from reprit.base import generate_repr

from algan.external_libraries.ground.core.enums import Location
from algan.external_libraries.ground.core.hints import QuaternaryPointFunction
from .exact import point_point_point as exact_point_point_point

PointPointPointLocator = QuaternaryPointFunction[Location]


class Context:
    __slots__ = '_point_point_point_test',

    def __init__(self, point_point_point_test: PointPointPointLocator
                 ) -> None:
        self._point_point_point_test = point_point_point_test

    __repr__ = generate_repr(__init__,
                             argument_serializer=serializers.complex_,
                             with_module_name=True)

    @property
    def point_point_point_locator(self) -> PointPointPointLocator:
        return self._point_point_point_test


exact_context = Context(exact_point_point_point.test)
