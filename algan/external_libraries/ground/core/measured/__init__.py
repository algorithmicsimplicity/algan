from typing import Callable

from reprit import serializers
from reprit.base import generate_repr

from .exact import region as exact_region


class Context:
    @property
    def region_signed_area(self):
        return self._region_signed_area

    __slots__ = '_region_signed_area',

    def __init__(self,
                 region_signed_area) -> None:
        self._region_signed_area = region_signed_area

    __repr__ = generate_repr(__init__,
                             argument_serializer=serializers.complex_,
                             with_module_name=True)


exact_context = Context(exact_region.signed_area)
