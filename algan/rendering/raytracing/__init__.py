"""Ray traced rendering backend for Algan.

Algan animates whole batches of frames in one pass, so this backend is built
around a single *spatio-temporal* BVH per primitive batch: time is treated as
a fourth dimension alongside x/y/z, primitives are adaptively segmented into
(frame interval, union bound) instances and ordered along a 4D Morton curve
(see :mod:`algan.rendering.raytracing.stbvh`). One tree therefore serves every
frame in the batch, with memory proportional to how much the scene *moves*
rather than ``num_frames * num_primitives``.

Rendering is dispatched by sample count (see ``tracer.render_batch_raytraced``,
the entry point). The default deterministic renderer
(``samples_per_pixel == 1``) is a *wavefront* tracer
(:mod:`~algan.rendering.raytracing.wavefront_kernels_taichi`): rays run in
bounded screen tiles through generate -> traverse -> shade -> composite kernel
stages, with per-ray state pool-allocated from the render arena and host-side
compaction between iterations. Each ray depth-peels its hits front-to-back and
alpha-blends every surface -- following reflections and refractive splits up
to ``MAX_BOUNCES`` -- into a fixed ``[frames, pixels, channels]`` output
buffer. ``samples_per_pixel > 1`` switches to the Monte Carlo path-tracing
megakernel (:mod:`~algan.rendering.raytracing.raytrace_kernels_taichi`), one
thread per (frame, pixel, sample) path. Tree and geometry preparation is
vectorized PyTorch.

Usage::

    import algan
    from algan.rendering.raytracing import set_samples_per_pixel

    # samples_per_pixel=1 (default) renders with the exact deterministic
    # wavefront tracer; > 1 enables Monte Carlo path tracing: jittered
    # sub-pixel rays, stochastic transparency, glossy (rough) reflections and
    # optional diffuse indirect lighting via indirect_bounce_strength.
    set_samples_per_pixel(64)  # before rendering

    mirror = algan.Sphere().set_material(
        algan.MeshStandardMaterial(metalness=1.0, roughness=0.2)
    )
    mirror.spawn()
    ...
    algan.Scene.save_video()
"""
from __future__ import annotations

from algan.rendering.raytracing.pn_patch import (
    evaluate_pn_patch,
    pn_control_points,
    pn_patch_coefficients,
)
from algan.rendering.raytracing.primitives import (
    MAX_BOUNCES,
    LogicalPNTrianglePrimitive,
    RayTracedBezierCircuitPrimitive,
    RayTracedPNTrianglePrimitive,
    RayTracedTrianglePrimitive,
    is_post_process_tonemap_enabled,
    set_ambient_light,
    set_fragment_shading,
    set_indirect_bounce_strength,
    set_light_intensity,
    set_ray_traced_shadows,
    set_samples_per_pixel,
    set_unsupported_feature_policy,
)
from algan.rendering.raytracing.stbvh import STBVH, build_stbvh
from algan.rendering.raytracing.tracer import RenderPlan

__all__ = [
    "MAX_BOUNCES",
    "STBVH",
    "build_stbvh",
    "LogicalPNTrianglePrimitive",
    "RayTracedTrianglePrimitive",
    "RayTracedPNTrianglePrimitive",
    "RayTracedBezierCircuitPrimitive",
    "is_post_process_tonemap_enabled",
    "set_samples_per_pixel",
    "set_indirect_bounce_strength",
    "set_light_intensity",
    "set_ambient_light",
    "set_fragment_shading",
    "set_ray_traced_shadows",
    "set_unsupported_feature_policy",
    "RenderPlan",
    "pn_control_points",
    "pn_patch_coefficients",
    "evaluate_pn_patch",
]
