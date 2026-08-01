"""Renderer backend registry.

Backend classes and kernels are runtime services, not user configuration, so
this registry deliberately lives outside the global ``SETTINGS`` object.
"""
from __future__ import annotations

from algan.rendering.raytracing import (
    RayTracedBezierCircuitPrimitive,
    RayTracedTrianglePrimitive,
)


class RendererRegistry:
    def __init__(self):
        self.triangle_primitive = RayTracedTrianglePrimitive
        self.bezier_circuit_primitive = RayTracedBezierCircuitPrimitive


RENDERER_REGISTRY = RendererRegistry()
# Compatibility alias.
RENDERER_SETTINGS = RENDERER_REGISTRY


def effective_triangle_primitive():
    """Triangle primitive class to build for new surfaces / meshes."""
    return RENDERER_REGISTRY.triangle_primitive


__all__ = [
    "RendererRegistry",
    "RENDERER_REGISTRY",
    "RENDERER_SETTINGS",
    "effective_triangle_primitive",
]
