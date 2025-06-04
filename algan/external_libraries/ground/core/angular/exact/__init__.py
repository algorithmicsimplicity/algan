from algan.external_libraries.ground.core.enums import (Kind,
                               Orientation)
from algan.external_libraries.ground.core.hints import (Point,
                               QuaternaryPointFunction)
from algan.external_libraries.ground.core.primitive import to_sign
from algan.external_libraries.ground.core.vector.exact import (cross,
                                      dot)


def kind(vertex: Point,
         first_ray_point: Point,
         second_ray_point: Point,
         dot_producer: QuaternaryPointFunction = dot.multiply) -> Kind:
    return Kind(to_sign(dot_producer(vertex, first_ray_point, vertex,
                                     second_ray_point)))


def orientation(vertex: Point,
                first_ray_point: Point,
                second_ray_point: Point,
                cross_producer: QuaternaryPointFunction
                = cross.multiply) -> Orientation:
    return Orientation(to_sign(cross_producer(vertex, first_ray_point, vertex,
                                              second_ray_point)))
