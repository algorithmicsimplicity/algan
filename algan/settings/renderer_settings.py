from dataclasses import dataclass

from algan.settings.abstract_settings import Settings
from algan.rendering.raytracing import RayTracedBezierCircuitPrimitive, RayTracedTrianglePrimitive


@dataclass
class RendererSettings(Settings):
    triangle_primitive = RayTracedTrianglePrimitive
    bezier_circuit_primitive = RayTracedBezierCircuitPrimitive
    render_kernel = None

RENDERER_SETTINGS = RendererSettings()