"""Ray traced rendering backend for Algan.

Algan animates whole batches of frames in one pass, so this backend is built
around a single *spatio-temporal* BVH per primitive batch: time is treated as
a fourth dimension alongside x/y/z, primitives are adaptively segmented into
(frame interval, union bound) instances and ordered along a 4D Morton curve
(see :mod:`algan.rendering.raytracing.stbvh`). One tree therefore serves every
frame in the batch, with memory proportional to how much the scene *moves*
rather than ``num_frames * num_primitives``.

Rendering itself is a single Taichi kernel
(:mod:`algan.rendering.raytracing.ray_trace_taichi`) launching one thread per
(frame, pixel). Each thread depth-peels its ray front-to-back and
alpha-blends every surface -- following mirror reflections up to
``MAX_BOUNCES`` -- directly into its own cell of a fixed
``[frames, pixels, channels]`` output buffer, so memory use is independent of
depth complexity and bounce count, with no fragment buffers, sorting passes
or atomics. Tree and geometry preparation is vectorized PyTorch.

Usage::

    import algan
    from algan.rendering.raytracing import (
        enable_ray_tracing, set_reflectivity, set_roughness)

    # samples_per_pixel=1 (default) renders with the exact deterministic
    # kernel; > 1 enables Monte Carlo path tracing: jittered sub-pixel rays,
    # stochastic transparency, glossy (rough) reflections and optional
    # diffuse indirect lighting via indirect_bounce_strength.
    enable_ray_tracing(samples_per_pixel=64)  # before creating mobs

    mirror = algan.Sphere()
    set_reflectivity(mirror, 0.9)  # before spawning
    set_roughness(mirror, 0.2)     # glossy instead of sharp
    mirror.spawn()
    ...
    algan.render_to_file()
"""
from __future__ import annotations

from algan.rendering.raytracing.primitives import (
    MAX_BOUNCES,
    RayTracedBezierCircuitPrimitive,
    RayTracedTrianglePrimitive,
    disable_ray_tracing,
    enable_ray_tracing,
    set_indirect_bounce_strength,
    set_reflectivity,
    set_roughness,
    set_samples_per_pixel,
)
from algan.rendering.raytracing.stbvh import STBVH, build_stbvh

__all__ = [
    "MAX_BOUNCES",
    "STBVH",
    "build_stbvh",
    "RayTracedTrianglePrimitive",
    "RayTracedBezierCircuitPrimitive",
    "enable_ray_tracing",
    "disable_ray_tracing",
    "set_reflectivity",
    "set_roughness",
    "set_samples_per_pixel",
    "set_indirect_bounce_strength",
]
