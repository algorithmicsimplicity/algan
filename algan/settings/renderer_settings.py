from dataclasses import dataclass

from algan.settings.abstract_settings import Settings
from algan.rendering.raytracing import RayTracedBezierCircuitPrimitive, RayTracedTrianglePrimitive


@dataclass
class RendererSettings(Settings):
    triangle_primitive = RayTracedTrianglePrimitive
    bezier_circuit_primitive = RayTracedBezierCircuitPrimitive
    render_kernel = None

RENDERER_SETTINGS = RendererSettings()


def effective_triangle_primitive():
    """Triangle primitive class to build for new surfaces / meshes.

    Geometry construction must not depend on whether a later render batch is
    eligible for the hybrid raster front-end.  In particular, enabling raster
    must not silently flatten PN patches when another feature (near clipping,
    custom scatter, AA, etc.) routes the batch back to the classic tracer.

    The raster dispatcher therefore treats PN geometry as an unsupported
    frontend feature and falls back to classic primary traversal while keeping
    the configured primitive class intact.
    """
    return RENDERER_SETTINGS.triangle_primitive