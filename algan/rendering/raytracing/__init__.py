"""Hybrid raster/ray-tracing and Monte Carlo transport for Algan.

``tracer.render_batch_raytraced`` dispatches explicitly on
``SETTINGS.raytracing.samples_per_pixel``:

* ``1`` uses the deterministic renderer. Eligible primary views use raster
  emission, per-pixel sheet compaction and sheet shading. Reflected/refracted
  continuations, and primary views that fail the raster route's gates, use
  staged wavefront traversal and shading.
* Values greater than ``1`` select ``path_tracer.py`` and
  ``path_tracer_taichi.py``. Sample waves share the wavefront traversal kernels,
  keep bounded non-splitting path state and accumulate jittered samples at
  output resolution. Alpha compositing is deterministic; scattering is sampled.

Scene preparation builds acceleration structures per geometry type. The default
shared-topology refit BVH stores per-frame bounds; the alternative STBVH stores
spatio-temporal primitive instances. Geometry, acceleration structures, hit
buffers, output images and transient state all contribute to memory use.

Usage::

    from algan import Scene, SETTINGS, Sphere, MeshStandardMaterial

    SETTINGS.raytracing.set(samples_per_pixel=64)
    Sphere().set_material(MeshStandardMaterial(metalness=1.0, roughness=0.2)).spawn()
    Scene.save_video()
"""

from __future__ import annotations

from algan.rendering.raytracing.primitives import (
    LogicalPNTrianglePrimitive,
    RayTracedBezierCircuitPrimitive,
    RayTracedTrianglePrimitive,
    is_post_process_tonemap_enabled,
    max_bounces,
    set_fragment_shading,
    set_samples_per_pixel,
    set_shadows,
    set_unsupported_feature_policy,
)
from algan.rendering.raytracing.stbvh import STBVH, build_stbvh
from algan.rendering.raytracing.tracer import RenderPlan
from algan.rendering.raytracing.truncation import TruncationCounts

__all__ = [
    "max_bounces",
    "STBVH",
    "build_stbvh",
    "LogicalPNTrianglePrimitive",
    "RayTracedTrianglePrimitive",
    "RayTracedBezierCircuitPrimitive",
    "is_post_process_tonemap_enabled",
    "set_samples_per_pixel",
    "set_fragment_shading",
    "set_shadows",
    "set_unsupported_feature_policy",
    "RenderPlan",
    "TruncationCounts",
]
