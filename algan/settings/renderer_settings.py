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

    Forced to the flat class when the hybrid raster front-end is enabled: the
    raster path has no PN-patch rasterizer (by design -- see
    ``settings.HYBRID_RASTER``), so PN surfaces render as flat triangles under
    raster. When raster is off this is just the configured
    ``triangle_primitive`` (flat by default, PN when ``pn_triangles`` is set).
    """
    from algan.rendering.raytracing import settings as rt_settings
    if rt_settings.HYBRID_RASTER:
        return RayTracedTrianglePrimitive
    return RENDERER_SETTINGS.triangle_primitive