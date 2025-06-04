from reprit import serializers
from reprit.base import generate_repr

from algan.external_libraries.ground.core.hints import (QuaternaryPointFunction)
from .exact import (cross as exact_cross,
                    dot as exact_dot)

QuaternaryFunction = QuaternaryPointFunction


class Context:
    __slots__ = '_cross_product', '_dot_product'

    def __init__(self,
                 cross_product: QuaternaryFunction,
                 dot_product: QuaternaryFunction) -> None:
        self._cross_product, self._dot_product = cross_product, dot_product

    __repr__ = generate_repr(__init__,
                             argument_serializer=serializers.complex_,
                             with_module_name=True)

    @property
    def cross_product(self) -> QuaternaryFunction:
        return self._cross_product

    @property
    def dot_product(self) -> QuaternaryFunction:
        return self._dot_product


exact_context = Context(cross_product=exact_cross.multiply,
                        dot_product=exact_dot.multiply)
