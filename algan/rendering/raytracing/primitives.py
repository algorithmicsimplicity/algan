"""Ray traced drop-in replacements for Algan's rasterized render primitives.

``RayTracedTrianglePrimitive`` and ``RayTracedBezierCircuitPrimitive`` subclass
the rasterized primitives to keep their construction and batching, but render
through a self-contained ray tracing pipeline. ``RayTracedPNTrianglePrimitive``
additionally renders each triangle as a curved point-normal (PN) patch -- a
quadratic Bezier triangle bent to match the vertex normals (enable with
``enable_ray_tracing(pn_triangles=True)``):

* ``project_to_screen`` shades vertices and packs geometry + per-frame bounds
  for the whole batch of frames (the spatio-temporal BVH of ``stbvh.py``).
* ``render`` traces one ray per (frame, pixel) with the unified Taichi kernel
  (``ray_trace_taichi.py``), which alpha-blends every surface it encounters
  -- including mirror bounces -- directly into a fixed
  ``[num_frames, num_pixels, channels]`` output buffer. Memory use is
  independent of depth complexity and bounce count, and there is no fragment
  buffer or sorting pass. The Monte Carlo kernels flatten the parallel loop
  further to one thread per (frame, pixel, sample) path, accumulating into a
  float32 per-pixel buffer with atomic adds.

Mirrors: give a mob a reflectivity with :func:`set_reflectivity` (before
spawning); the value is per-vertex, animatable, and bounces up to
``MAX_BOUNCES`` times.

Call :func:`enable_ray_tracing` *before constructing any mobs* to make new
mobs render through this pipeline.
"""
from __future__ import annotations

import gc
import os

import taichi as ti
import torch
import torch.nn.functional as F

from algan.rendering.primitives.bezier_circuit_primitive import (
    BezierCircuitPrimitive,
    batch_arange,
)
from algan.rendering.primitives.primitive import OutOfRenderMemory
from algan.rendering.primitives.triangle_primitive import TrianglePrimitive
from algan.rendering.raytracing.pn_patch import (
    pn_control_points,
    pn_obb,
    pn_patch_coefficients,
)
from algan.rendering.raytracing.ray_trace_taichi import (
    KBUF,
    MAX_SURFACES_PER_RAY,
    MIN_ALPHA,
    _ensure_globals,
    finalize_samples,
    path_trace_physical_stbvh,
    path_trace_scene_stbvh,
    render_scene_stbvh,
    render_triangles_stbvh,
    render_triangles_knots_stbvh,
)
from algan.rendering.raytracing.no_pn_taichi import render_no_pn_stbvh
from algan.rendering.raytracing.gbuffer_taichi import (
    GB_HIT_W,
    gbuffer_nearest_general,
    shade_accumulate_wavefront,
    shade_gbuffer_torch,
    wf_drain_record_gbuffer,
    wf_traverse_gbuffer,
)
from algan.rendering.raytracing.shading_taichi import MAT_W
from algan.rendering.raytracing.wavefront_taichi import (
    wf_composite,
    wf_composite_aa,
    wf_composite_accum,
    wf_composite_accum_aa,
    wf_finalize_aa,
    wf_gen_general,
    wf_gen_triangle,
    wf_shade_general,
    wf_shade_triangle,
    wf_shadow_general,
    wf_traverse_general,
    wf_traverse_triangle,
    wf_traverse_triangle_knots,
)
from algan.rendering.raytracing.stbvh import EMPTY_HI, EMPTY_LO, build_stbvh
from algan.rendering.raytracing.time_compression import (
    compress_time, expand_time)
from algan.settings.defaults import COMPUTING_DEFAULTS
from algan.utils.memory_utils import InsufficientMemoryException, empty_cache
from algan.utils.tensor_utils import broadcast_all, cast_to_tensor, unsquish

# Maximum number of ray bounces (mirror reflections / diffuse scatters).
MAX_BOUNCES = 4
# Rays averaged per pixel. 1 renders with the exact deterministic kernel;
# > 1 switches to the Monte Carlo path tracer (stochastic transparency,
# glossy reflections, optional diffuse indirect lighting).
SAMPLES_PER_PIXEL = 1
# ACES Filmic Tonemapping settings:
TONEMAPPING = True
TONEMAP_EXPOSURE = 1.0
TONEMAP_METHOD = "neutral"
POST_PROCESS_TONEMAP = True
# 3D Raytraced Glow setting:
RAYTRACED_GLOW = False
# Strength of diffuse indirect bounces in the Monte Carlo renderer: 0 keeps
# surfaces purely (vertex-shader) lit, > 0 scatters paths on diffuse hits
# with throughput ``albedo * strength`` for color bleeding.
INDIRECT_BOUNCE_STRENGTH = 0.0
# Fully physical mode: vertex shading is skipped (colors are raw albedo) and
# the path tracer computes all lighting from the scene's explicit point
# lights (with shadow rays), glow emission and the background environment.
# Requires SAMPLES_PER_PIXEL > 1.
PHYSICAL_LIGHTING = False
# Radiance scale of explicit point lights in physical mode. The default of
# pi makes a white light produce roughly albedo-level Lambertian brightness.
LIGHT_INTENSITY = 3.141592653589793
# Constant ambient term added per diffuse interaction in physical mode.
AMBIENT_LIGHT = 0.0
# When True, the deterministic trace kernel is told which geometry types are
# actually present and skips the per-ray traversal of any type whose tree is
# just the empty placeholder (a launch-uniform branch, no divergence). Set
# False to force all three traversals -- used by the A/B benchmark to measure
# the gain in isolation.
GATE_EMPTY_TRAVERSALS = True
# Deferred shadows for the general wavefront: when True, binary hard-shadow
# rays are traced in a separate lean kernel (``wf_shadow_general``) between
# traverse and shade, and the shade kernel reads packed visibility bits instead
# of inlining the ``_shadow_occluded`` -> PN-solver call graph. Byte-identical;
# trades one extra kernel launch + a small bit buffer for lower shade-kernel
# register pressure (higher occupancy). Off by default (env ALGAN_WF_DEFERRED_SHADOWS=1).
DEFER_WF_SHADOWS = os.environ.get("ALGAN_WF_DEFERRED_SHADOWS", "0") == "1"
# When True, a batch containing only flat triangles is rendered by the
# specialized ``render_triangles_stbvh`` kernel instead of the general
# three-geometry-type ``render_scene_stbvh``. Output is identical; the
# specialized kernel just carries no PN/bezier code, so it has lower register
# pressure (higher GPU occupancy). Set False to force the general kernel (A/B).
USE_TRIANGLE_ONLY_KERNEL = True
# When True, a batch with NO PN patches but containing bezier circuits (text,
# 2D shapes), optionally mixed with flat triangles, is rendered by the
# specialized ``render_no_pn_stbvh`` kernel instead of the general
# ``render_scene_stbvh``. The general kernel always compiles the PN Matrix Pencil
# solver into its call graph (its dominant register cost); the no-PN kernel
# omits it, so a no-PN scene runs at higher occupancy. Output is byte-identical
# (the omitted PN paths are inert when no PN patches are present). The pure
# triangle-only case is handled by the even-leaner kernel above; this catches
# the remaining default (pn_triangles=False) scenes. Env ALGAN_NO_PN=0 forces
# the general kernel (A/B against this specialization).
USE_NO_PN_KERNEL = os.environ.get("ALGAN_NO_PN", "1") == "1"
# When True, ray-traced batches are rendered by the experimental wavefront
# (stage-split) path instead of the single-megakernel path: per-ray state lives
# in global memory and small per-stage kernels (gen, traverse, shade, composite)
# run in a host loop with PyTorch ray compaction between iterations, for higher
# occupancy and less divergence. Output is byte-identical. Covers both the
# triangle-only and the general (PN/bezier/mixed) cases. Off by default (env
# ALGAN_WAVEFRONT=1): measured SLOWER than the megakernels on the GTX 1050
# (global state-I/O cost exceeds the occupancy gain on a bandwidth-limited GPU),
# kept for validation and for higher-bandwidth hardware.
USE_WAVEFRONT = os.environ.get("ALGAN_WAVEFRONT", "1") == "1"
INPLACE_AA = os.environ.get("ALGAN_INPLACE_AA", "0") == "1"
# Rays per wavefront screen tile. The wavefront holds per-ray state for every
# ray it processes at once (~(18 + 6*KBUF) floats/ray); processing the chunk in
# tiles of this many rays bounds that state so it fits at any resolution / chunk
# length (a single HD frame is ~2M rays). ~2M rays * ~168 B ~= 350 MB of state.
WAVEFRONT_TILE_RAYS = int(os.environ.get("ALGAN_WAVEFRONT_TILE", str(1 << 21)))
# Pool over-allocation factor for the general wavefront when refraction is on:
# a glass (reflective+refractive) ray splits into a reflected + refracted pair,
# so the pool reserves this many slots per primary pixel for spawned split rays.
# Total per-tile state is unchanged (fewer pixels per tile, not bigger state);
# splits beyond the pool simply drop the refracted branch (graceful). Only the
# refraction path pays it (non-refractive renders use 1 slot per pixel).
REFRACT_SPLIT_SLOTS = max(2, int(os.environ.get("ALGAN_WAVEFRONT_SPLIT", "4")))
# Triangle-only batches automatically render through the (tiled, byte-exact)
# wavefront path when the frame has at least this many rays -- where its higher
# occupancy / lower divergence wins (measured ~1.41x wall at HD on high-divergence
# scenes, neutral otherwise). Smaller frames stay on the megakernel, whose per-ray
# state is in-register so it has no host-loop / launch overhead to amortize. Env
# ALGAN_WAVEFRONT=1 forces the wavefront on at any size; ALGAN_WAVEFRONT_MIN_PIXELS
# tunes the threshold (0 disables the auto path).
WAVEFRONT_MIN_PIXELS = int(
    os.environ.get("ALGAN_WAVEFRONT_MIN_PIXELS", "700000"))
# When True, the *deterministic* fragment-shaded path uses the experimental
# deferred-shading (G-buffer) prototype instead of the in-kernel ``_shade_fragment``:
# the trace kernel writes per-pixel surface attributes and a PyTorch pass shades
# the whole screen at once (see :mod:`algan.rendering.raytracing.gbuffer_taichi`).
# Prototype only -- nearest opaque hit, no transparency/bounces/shadows -- gated
# behind ALGAN_GBUFFER=1 for benchmarking against the megakernel.
USE_GBUFFER = os.environ.get("ALGAN_GBUFFER", "0") == "1"
# Which deferred path ALGAN_GBUFFER selects: "wavefront" (full transparency +
# reflection ping-pong, default) or "nearest" (the single-hit opaque prototype).
GBUFFER_MODE = os.environ.get("ALGAN_GBUFFER_MODE", "wavefront")
# Accumulated wall time (seconds) spent inside the per-chunk render dispatch
# (trace kernel + any deferred shade), summed across chunks when
# ALGAN_KERNEL_TIMING=1. Benchmarks read and reset this to isolate the render
# kernel cost from animation generation and video encoding.
_KERNEL_TIME_TOTAL = 0.0
# When True, the *deterministic* ray tracer (SAMPLES_PER_PIXEL == 1, non-physical)
# shades the core lit materials per fragment inside the trace kernel instead of
# baking per-vertex colours (Gouraud). Off by default so existing frame-comparison
# baselines are unchanged; enable via enable_ray_tracing(fragment_shading=True) or
# set_fragment_shading(True). Ignored by the Monte Carlo / physical paths.
FRAGMENT_SHADING = False

# Temporal compression of per-frame geometry (see time_compression.py). When
# > 0, animated geometry arrays are compressed to a per-primitive piecewise
# linear knot representation instead of a dense [T, N, D] tensor. Mode 1 is a
# validation harness: it round-trips the geometry through compress/expand so the
# existing dense kernels run unchanged, exposing only the (tiny) reconstruction
# error -- used to confirm the representation is render-safe before the kernels
# fetch knots directly. Env ALGAN_TIME_COMPRESS selects the mode.
TIME_COMPRESS = int(os.environ.get("ALGAN_TIME_COMPRESS", "0"))


def set_time_compress(mode):
    """Set the temporal-compression mode at runtime (0 off, 1 round-trip
    validation, 2 in-kernel knot path). Lets a benchmark alternate modes in one
    process for thermally-fair A/B timing."""
    global TIME_COMPRESS
    TIME_COMPRESS = int(mode)


def set_wavefront(enabled):
    """Toggle the wavefront (stage-split) trace path at runtime, for in-process
    A/B against the megakernel path."""
    global USE_WAVEFRONT
    USE_WAVEFRONT = bool(enabled)


def _maybe_roundtrip_time(x):
    """Validation shim: replace a dense [T, N, D] geometry array with
    ``expand_time(compress_time(x))`` so the existing kernels see the
    reconstructed geometry. A no-op unless ALGAN_TIME_COMPRESS == 1.
    """
    if TIME_COMPRESS == 1 and x is not None and x.dim() == 3 and x.shape[0] > 1:
        return expand_time(compress_time(x))
    return x


def set_fragment_shading(enabled):
    """Toggle per-fragment shading of the *deterministic* ray tracer.

    When enabled, triangle/PN hits whose material is one of the core lit
    shaders (the legacy diffuse default, ``MeshBasicMaterial``,
    ``MeshLambertMaterial``, ``MeshPhongMaterial``, ``MeshStandardMaterial``)
    are shaded per fragment in-kernel from the raw albedo, a per-primitive
    material block and the scene's point lights -- crisper specular highlights
    and smooth shading on coarse meshes. Other materials keep vertex shading.
    Only the deterministic renderer (``set_samples_per_pixel(1)``, non-physical)
    is affected. Set before rendering.
    """
    global FRAGMENT_SHADING
    FRAGMENT_SHADING = bool(enabled)


# When True, the deterministic ray tracer casts binary hard shadows: each
# shaded triangle/PN fragment fires one shadow ray per point light and an
# opaque occluder (alpha >= SHADOW_ALPHA_THRESHOLD) fully blocks that light's
# direct contribution. Implies per-fragment shading (shadows are evaluated in
# the lighting model) and forces the general kernel. No soft/transmissive
# shadows -- use the physical path tracer for those. Off by default.
SHADOWS = False


def set_ray_traced_shadows(enabled):
    """Toggle binary hard shadows in the *deterministic* ray tracer.

    When enabled, every shaded triangle/PN fragment traces one shadow ray per
    scene point light; a light is occluded (its direct diffuse/specular term
    dropped, ambient/emissive kept) when an opaque surface lies between the
    fragment and the light. Shadows are evaluated inside the per-fragment
    lighting model, so this implies :func:`set_fragment_shading` for the render
    and forces the general ``render_scene_stbvh`` kernel (the lean triangle-only
    and no-PN kernels are bypassed). Shadows are hard-edged and ignore
    transparency; for soft or glass shadows use the physical path tracer
    (``set_samples_per_pixel(n)`` with ``n > 1``). Only the deterministic
    renderer (``set_samples_per_pixel(1)``, non-physical) is affected. Set
    before rendering.
    """
    global SHADOWS
    SHADOWS = bool(enabled)


# --- Core lit material registry (shader function -> in-kernel material id) ----
# Ids must match shading_taichi: 0 default diffuse, 1 basic/unlit/passthrough,
# 2 lambert, 3 phong, 4 standard.
def _build_core_shader_ids():
    from algan.rendering.shaders.material_shaders import (
        basic_material_shader,
        lambert_shader,
        phong_shader,
        standard_shader,
    )
    from algan.rendering.shaders.pbr_shaders import default_shader, null_shader

    return {
        default_shader: 0,
        null_shader: 1,
        basic_material_shader: 1,
        lambert_shader: 2,
        phong_shader: 3,
        standard_shader: 4,
    }


_CORE_SHADER_IDS = None
# Per-material parameter defaults (canonical 12-slot block; see shading_taichi).
_MAT_DEFAULTS = [0.0, 0.0, 0.0, 1.0, 0.0666, 0.0666, 0.0666, 30.0, 1.0, 0.0,
                 0.0, 1.0]
# Material-property name -> (start slot, width) in the canonical block.
_MAT_SLOTS = {
    "emissive": (0, 3),
    "emissive_intensity": (3, 1),
    "specular": (4, 3),
    "shininess": (7, 1),
    "roughness": (8, 1),
    "metalness": (9, 1),
    "flat_shading": (10, 1),
    "env_map_intensity": (11, 1),
}


def _core_shader_ids():
    global _CORE_SHADER_IDS
    if _CORE_SHADER_IDS is None:
        _CORE_SHADER_IDS = _build_core_shader_ids()
    return _CORE_SHADER_IDS


def _shader_material_id(shader):
    """In-kernel material id for a shader function. Unknown / non-core shaders
    (and ``None``) map to 1 (unlit passthrough: the kernel returns the colour --
    raw or baked -- unchanged)."""
    if shader is None:
        return 1
    return _core_shader_ids().get(shader, 1)


def _shader_is_core(shader):
    """True if ``shader`` has an in-kernel port (so its hits can be fragment
    shaded rather than baked)."""
    return shader is not None and shader in _core_shader_ids()


def set_physical_lighting(enabled):
    """Toggle fully physical lighting for the Monte Carlo renderer: vertex
    shading is skipped and illumination comes from the scene's point lights
    (sampled with shadow rays), surface ``glow`` emission, and the background
    environment. Requires ``set_samples_per_pixel`` > 1. Set before
    rendering.
    """
    global PHYSICAL_LIGHTING
    PHYSICAL_LIGHTING = bool(enabled)


def set_light_intensity(intensity):
    """Radiance scale applied to explicit point lights in physical mode."""
    global LIGHT_INTENSITY
    LIGHT_INTENSITY = float(intensity)


def set_ambient_light(intensity):
    """Constant ambient lighting term used in physical mode."""
    global AMBIENT_LIGHT
    AMBIENT_LIGHT = float(intensity)


def set_samples_per_pixel(samples):
    """Set how many rays are averaged per pixel. 1 (the default) uses the
    exact deterministic renderer; larger values enable Monte Carlo path
    tracing with that many samples.
    """
    global SAMPLES_PER_PIXEL
    SAMPLES_PER_PIXEL = max(1, int(samples))


def set_indirect_bounce_strength(strength):
    """Set the diffuse indirect lighting strength of the Monte Carlo
    renderer (0 disables diffuse bounces).
    """
    global INDIRECT_BOUNCE_STRENGTH
    INDIRECT_BOUNCE_STRENGTH = float(strength)


def set_tonemapping(enabled):
    """Enable or disable ACES Filmic Tonemapping in the ray-tracing rendering kernels."""
    global TONEMAPPING
    TONEMAPPING = bool(enabled)


def set_tonemap_exposure(exposure):
    """Set the exposure multiplier for the ACES Filmic Tonemapper."""
    global TONEMAP_EXPOSURE
    TONEMAP_EXPOSURE = float(exposure)


def set_tonemap_method(method):
    """Set the tonemapping method ("neutral" or "agx")."""
    global TONEMAP_METHOD
    if method not in ("neutral", "agx"):
        raise ValueError("tonemap_method must be 'neutral' or 'agx'")
    TONEMAP_METHOD = str(method)


def set_post_process_tonemap(enabled):
    """Enable or disable post-process tonemapping instead of in-kernel tonemapping."""
    global POST_PROCESS_TONEMAP
    POST_PROCESS_TONEMAP = bool(enabled)


def is_post_process_tonemap_enabled():
    """Return whether post-process tonemapping is enabled."""
    return POST_PROCESS_TONEMAP


def _get_tonemap_t_val():
    if POST_PROCESS_TONEMAP:
        return 3
    if not TONEMAPPING:
        return 0
    return 2 if TONEMAP_METHOD == "agx" else 1


def set_raytraced_glow(enabled):
    """Enable or disable volumetric 3D raytraced glow.
    If disabled, switches back to the post-processing 2D bloom filter style glow.
    """
    global RAYTRACED_GLOW
    RAYTRACED_GLOW = bool(enabled)
    from algan.rendering.raytracing import ray_trace_taichi
    ray_trace_taichi._ensure_globals()
    ray_trace_taichi.global_raytraced_glow[None] = 1 if RAYTRACED_GLOW else 0


def is_raytraced_glow_enabled():
    """Return whether 3D raytraced glow is enabled."""
    return RAYTRACED_GLOW



def _set_surface_param(mob, name, value):
    value = cast_to_tensor(float(value)).view(1, 1)
    for descendant in reversed(mob.get_descendants()):
        setattr(descendant, name, value)
        names = list(getattr(descendant, "shader_specific_param_names", []))
        if name not in names:
            names.append(name)
        descendant.shader_specific_param_names = names
    return mob


def set_reflectivity(mob, reflectivity):
    """Make a mob a mirror under the ray traced renderer.

    Attaches a (static) ``reflectivity`` value (0 = matte, 1 = perfect
    mirror) to the mob and its descendants, exposed to the renderer as a
    shader parameter. Call before the mob is spawned; only the ray traced
    pipeline uses it (the rasterizer's shaders would reject the extra
    parameter).
    """
    return _set_surface_param(mob, "reflectivity", reflectivity)


def set_roughness(mob, roughness):
    """Set the glossiness of a mirror mob: 0 is a sharp mirror, larger
    values blur its reflections. Only used by the Monte Carlo renderer
    (``set_samples_per_pixel`` > 1); the deterministic renderer reflects
    sharply. Call before the mob is spawned.
    """
    return _set_surface_param(mob, "roughness", roughness)


def set_refractive_index(mob, ior):
    """Make a mob refract light (glass) under the ray traced renderer.

    Attaches a (static) index of refraction to the mob and its descendants:
    1.0 means no bending (the default for unset mobs is 0, treated as "not
    refractive"), ~1.33 water, ~1.5 glass, ~2.4 diamond. The *transmitted*
    fraction of each ray (the part not reflected or absorbed -- so give the mob
    some transparency via its colour's opacity) is bent at the surface by
    Snell's law and traced onward, with total internal reflection handled.

    Only the **general wavefront** tracer implements refraction, so it is used
    when that path renders the batch (``set_wavefront(True)`` /
    ``ALGAN_WAVEFRONT=1``, or automatically for any batch that contains a
    refractive mob). The megakernel and Monte Carlo paths ignore it. Reflection
    takes priority over refraction if both are set on the same mob. Call before
    the mob is spawned.
    """
    return _set_surface_param(mob, "refractive_index", ior)


def _flat_frames(x, last_dims):
    """Collapse camera tensors like [T, 1, 1, 3] to [T, *last_dims]."""
    return x.reshape(x.shape[0], *last_dims).float()


def _expand_frames(x, num_frames):
    if x.shape[0] == num_frames:
        return x
    return x.expand(num_frames, *x.shape[1:])


def _pixel_bases(screen_basis):
    """Per-frame world-space steps corresponding to one unit of normalized
    screen coordinate, matching the camera's projection exactly.

    The camera projects a world point by intersecting its view ray with the
    plane ``normal = basis_row_2`` through the screen center, then taking raw
    dot products with ``basis_row_0/1``. The screen basis is rotation x
    non-uniform scale, so under camera rotation its rows are *not* mutually
    orthogonal (``row0 . row2 != 0``) -- the projection is anisotropic and
    changes with orientation. The exact inverse image of screen coordinate
    (u, v) is ``screen_point + u * d0 + v * d1`` where ``d0, d1`` are the
    first two columns of the inverse basis matrix (the reciprocal basis:
    ``d_i . row_j = delta_ij``), which both lies on the projection plane and
    reproduces the dot products.
    """
    eye = torch.eye(3, device=screen_basis.device).unsqueeze(0) * 1e-12
    dual = torch.linalg.inv(screen_basis + eye)
    return dual[:, :, 0].contiguous(), dual[:, :, 1].contiguous()


def _unify_time(tensors, error_context):
    """Expand a set of tensors whose leading (time) dims are each 1 or T to a
    common T. Returns the expanded tensors and T.
    """
    T = max(t.shape[0] for t in tensors)
    for t in tensors:
        if t.shape[0] not in (1, T):
            raise ValueError(
                f"{error_context}: incompatible frame counts "
                f"{[tuple(t.shape) for t in tensors]}")
    return [_expand_frames(t, T) for t in tensors], T


def _cat_collections(tensors, dim, error_context):
    """Concatenate per-collection tensors along ``dim``, broadcasting their
    (possibly different) time dimensions to a common length first. A single
    collection is passed through without copying (the kernel indexes each
    array's time dimension independently, so no expansion is needed).
    """
    if len(tensors) == 1:
        return tensors[0]
    tensors, _ = _unify_time(tensors, error_context)
    return torch.cat(tensors, dim).contiguous()


class RayTracedTrianglePrimitive(TrianglePrimitive):
    """Triangle batch rendered by ray tracing a spatio-temporal BVH."""

    stbvh_tightness = float(os.environ.get("ALGAN_STBVH_TIGHTNESS", "1.0"))

    # Per-vertex surface parameters consumed by the trace kernels rather
    # than by a shader; popped from the shader kwargs (see set_reflectivity
    # and set_roughness).
    _surface_params = ("reflectivity", "roughness", "refractive_index", "glow", "glow_radius")

    def __init__(self, corners=None, colors=None, opacity=1, normals=None,
                 perimeter_points=None, reverse_perimeter=False,
                 triangle_collection=None, glow=0, shader=None,
                 uvs=None, texture_map=None,
                 material_texture_map=None, material_texture_flags=0,
                 normal_texture_map=None,
                 **shader_kwargs):
        if triangle_collection is not None:
            super().__init__(corners, colors, opacity, normals,
                             perimeter_points, reverse_perimeter,
                             triangle_collection, glow, shader,
                             uvs=uvs, texture_map=texture_map,
                             material_texture_map=material_texture_map,
                             material_texture_flags=material_texture_flags,
                             normal_texture_map=normal_texture_map,
                             **shader_kwargs)
            # Gather per-mob surface params with the same broadcast/cat
            # recipe the base class applies to corners/colors, so shapes
            # line up -- except along time: the references are sliced to a
            # single frame so a static parameter (the usual case) stays
            # single-frame instead of being expanded to the batch length.
            for name in self._surface_params:
                values = []
                for triangle in triangle_collection:
                    v = getattr(triangle, name, None)
                    if v is None:
                        v = torch.zeros_like(triangle.colors[:1, ..., :1])
                    v = broadcast_all(
                        [triangle.corners[:1], triangle.colors[:1],
                         triangle.normals[:1], v], ignored_dims=[-1]
                    )[-1][..., :1]
                    values.append(v)
                setattr(self, name, unsquish(torch.cat(values, 1), -2, 3
                                             ).to(COMPUTING_DEFAULTS.render_device))
        else:
            params = {name: shader_kwargs.pop(name, None)
                      for name in self._surface_params}
            if params["glow"] is None:
                params["glow"] = glow
            super().__init__(corners, colors, opacity, normals,
                             perimeter_points, reverse_perimeter,
                             triangle_collection, glow, shader=shader,
                             uvs=uvs, texture_map=texture_map,
                             material_texture_map=material_texture_map,
                             material_texture_flags=material_texture_flags,
                             normal_texture_map=normal_texture_map,
                             **shader_kwargs)
            for name, value in params.items():
                if value is None:
                    setattr(self, name,
                            torch.zeros_like(self.colors[:1, ..., :1]))
                else:
                    value = cast_to_tensor(value).to(self.colors.device)
                    setattr(self, name, broadcast_all(
                        [self.corners[:1], self.colors[:1], value],
                        ignored_dims=[-1])[-1][..., :1])

    def _shaded_per_fragment(self):
        """True when this primitive's hits are shaded per fragment in-kernel
        (deterministic renderer, fragment shading on, core lit material) rather
        than baked per vertex -- in which case ``colors`` stays raw albedo."""
        return (FRAGMENT_SHADING and not PHYSICAL_LIGHTING
                and SAMPLES_PER_PIXEL <= 1
                and _shader_is_core(getattr(self, "shader", None)))

    def _shade_vertex_colors(self, camera, light_sources):
        """Vertex shading, identical to the rasterized pipeline. Skipped in
        physical mode (raw albedo, the path tracer lights the scene) and when
        this primitive is shaded per fragment instead (see
        :meth:`_shaded_per_fragment`).
        """
        if PHYSICAL_LIGHTING or self._shaded_per_fragment():
            return
        d = -1
        if getattr(self, "shader", None) is not None:
            for light_source in light_sources:
                with self.memory.temp():
                    self.colors[..., :d] = self.shader(
                        self.memory,
                        self.corners,
                        self.normals,
                        self.colors[..., :d],
                        camera.ray_origin,
                        light_source.origin,
                        light_source.light_color,
                        1,
                        1,
                        *self.shader_param_values,
                    )

    def _pack_material(self):
        """Per-primitive material id ``[1, N]`` and the canonical material
        parameter block ``[Tm, N, MAT_W]`` consumed by the in-kernel fragment
        shader. Material properties are per-mob constants broadcast to
        vertices, so each triangle's value is taken from its first corner.
        Non-core (or absent) shaders get id 1 (passthrough) and default params.
        """
        colors = self.colors
        N = colors.shape[1]
        device = colors.device
        shader = getattr(self, "shader", None)
        mat_id = torch.full((1, N), _shader_material_id(shader),
                            dtype=torch.int32, device=device)
        def per_triangle(value):
            v = value.float().to(device)
            if v.dim() >= 4:              # [T, N, 3, w] -> per-triangle corner 0
                v = v[:, :, 0, :]
            return v

        pairs = []
        if _shader_is_core(shader):
            # The material's shader params, addressed by their real names (the
            # signature is not reliable: the ray tracer pops ``roughness`` out
            # of the shader kwargs into a surface param, see _surface_params).
            names = list(getattr(self, "shader_param_names", None) or [])
            values = list(getattr(self, "shader_param_values", None) or [])
            for name, value in zip(names, values):
                if name in _MAT_SLOTS and value is not None:
                    pairs.append((name, per_triangle(value)))
            # ``roughness`` (MeshStandardMaterial) was popped into self.roughness;
            # feed it into the roughness slot (ignored by the non-PBR branches).
            roughness = getattr(self, "roughness", None)
            if roughness is not None:
                pairs.append(("roughness", per_triangle(roughness)))
        Tm = max([1] + [v.shape[0] for _n, v in pairs])
        mat = torch.tensor(_MAT_DEFAULTS, device=device).view(
            1, 1, MAT_W).expand(Tm, N, MAT_W).contiguous()
        for name, v in pairs:
            start, width = _MAT_SLOTS[name]
            if v.shape[-1] != width:      # broadcast a scalar into a vector slot
                v = v.expand(*v.shape[:-1], width)
            mat[:, :, start:start + width] = v
        return mat_id.contiguous(), mat.contiguous()

    def _pack_surface_extra(self, error_context):
        """Per-primitive surface params [Te, N, 15]: the interleaved per-corner
        (reflectivity, roughness) pairs in columns 0-5 (unchanged; consumed by
        ``_triangle_extra`` in every kernel), followed by the per-corner
        refractive index in columns 6-8 (0 = not refractive; read by the
        wavefront's ``_corner_ior`` for the refraction path), followed by the
        per-corner glow strength in columns 9-11, followed by the per-corner
        glow radius in columns 12-14."""
        (reflectivity_e, roughness_e, ior_e, glow_e, glow_radius_e), _ = _unify_time(
            [self.reflectivity.float(), self.roughness.float(),
             self.refractive_index.float(), self.glow.float(),
             self.glow_radius.float()], error_context)
        n_t, n_p = reflectivity_e.shape[0], reflectivity_e.shape[1]
        refl_rough = torch.cat((reflectivity_e, roughness_e), -1).reshape(
            n_t, n_p, 6)
        ior = ior_e.reshape(n_t, n_p, 3)
        glow = glow_e.reshape(n_t, n_p, 3)
        glow_radius = glow_radius_e.reshape(n_t, n_p, 3)
        return torch.cat((refl_rough, ior, glow, glow_radius), -1).contiguous()

    def _pack_frame_visibility(self, lo, hi, colors, error_context):
        """Per-frame bounds; frames where a primitive is fully transparent
        and not glowing are marked empty so they never enter the BVH. Fully opaque frames
        are flagged so the trace kernel can prune hits behind them while
        gathering.
        """
        # Last channel is opacity. Indexing (rather than the Color.opacity
        # property) so this also works for textured surfaces (ImageMob), whose
        # per-vertex colors are plain tensors, not Color instances.
        alpha = colors[..., -1]
        glow_f = self.glow.squeeze(-1)
        glow_radius_f = self.glow_radius.squeeze(-1)

        visible = (alpha.amax(-1) > MIN_ALPHA) | (glow_f.amax(-1) > 0.0)
        opaque = alpha.amin(-1) >= 1.0 - 1e-6
        eff_rad = torch.where(glow_f.amax(-1, keepdim=True) > 0.0, glow_radius_f.amax(-1, keepdim=True), 0.0)

        (lo, hi, visible, opaque, eff_rad), _ = _unify_time(
            [lo, hi, visible.unsqueeze(-1), opaque.unsqueeze(-1), eff_rad],
            error_context)
        visible = visible.squeeze(-1)
        self._rt_frame_opaque = opaque.squeeze(-1).contiguous()
        self._rt_frame_lo = torch.where(
            visible.unsqueeze(-1), lo - eff_rad,
            torch.tensor(EMPTY_LO, device=lo.device)).contiguous()
        self._rt_frame_hi = torch.where(
            visible.unsqueeze(-1), hi + eff_rad,
            torch.tensor(EMPTY_HI, device=hi.device)).contiguous()

    def _set_frame_buffer_bytes(self, camera):
        """Per-frame buffer bytes: the u8 output, plus the f32 sample
        accumulator in Monte Carlo mode, plus the wavefront's per-ray global
        state when that path is active (so the chunk is sized to fit it instead
        of OOMing on a megakernel-sized chunk -- the wavefront holds state for
        every ray in the chunk at once, unlike the in-register megakernel).
        """
        mc = PHYSICAL_LIGHTING or SAMPLES_PER_PIXEL > 1
        self._rt_frame_bytes = int(
            camera.screen_width * camera.screen_height * 5 * 4
            * (2 if mc else 1))
        if USE_WAVEFRONT:
            # Per ray: rs_ro/rd/acc/sca/int (~18 floats) + 6 KBUF-wide hit
            # buffers, all float/int32 (4 bytes), held for the whole chunk.
            self._rt_frame_bytes += int(
                camera.screen_width * camera.screen_height
                * (18 + 6 * KBUF) * 4)

    def project_to_screen(self, camera, light_sources):
        self._shade_vertex_colors(camera, light_sources)

        corners = self.corners.float()
        normals = self.normals.float()
        # Hot/cold split, each array with its own (independent) time
        # dimension: positions are touched by every candidate
        # intersection, normals only by hits that bounce or scatter, and
        # reflectivity/roughness (usually static) only by confirmed hits.
        self._rt_tri_pos = _maybe_roundtrip_time(corners.reshape(
            corners.shape[0], corners.shape[1], 9).contiguous())
        self._rt_tri_norm = normals.reshape(
            normals.shape[0], normals.shape[1], 9).contiguous()
        self._rt_tri_extra = self._pack_surface_extra(
            "triangle surface params")
        self._rt_tri_colors = self.colors.float().contiguous()
        self._rt_tri_mat_id, self._rt_tri_mat = self._pack_material()
        self._rt_num_frames = camera.ray_origin.shape[0]

        if self.uvs is not None:
            self._rt_tri_uvs = self.uvs.float().reshape(self.uvs.shape[0], self.uvs.shape[1], 6).contiguous()
            self._rt_texture_map = self.texture_map.float().contiguous() if self.texture_map is not None else None
            mtex = getattr(self, "material_texture_map", None)
            self._rt_material_texture = (mtex.float().contiguous()
                                         if mtex is not None else None)
            self._rt_material_flags = int(
                getattr(self, "material_texture_flags", 0) or 0)
            ntex = getattr(self, "normal_texture_map", None)
            self._rt_normal_texture = (ntex.float().contiguous()
                                       if ntex is not None else None)
        else:
            self._rt_tri_uvs = None
            self._rt_texture_map = None
            self._rt_material_texture = None
            self._rt_material_flags = 0
            self._rt_normal_texture = None

        self._pack_frame_visibility(corners.amin(-2), corners.amax(-2),
                                    self._rt_tri_colors,
                                    "triangle bounds/colors")

        # Everything the renderer needs now lives in the packed arrays;
        # release the unpacked geometry to halve resident GPU memory.
        self.corners = self.normals = None
        self.reflectivity = self.roughness = self.refractive_index = None
        self.colors = self.shader_param_values = None
        self.uvs = self.texture_map = None
        self.material_texture_map = self.normal_texture_map = None

        self._set_frame_buffer_bytes(camera)

        # Ensure released geometry is actually freed before rendering.
        empty_cache(force_gc=False)
        return self

    def get_memory_used_per_timestep(self):
        return self._rt_frame_bytes

    def get_memory_used_for_blending(self, start_ind, end_ind):
        return 0  # Blending happens in-register inside the trace kernel.

    def render(self, primitives, scene, save_image, screen_width,
               screen_height, time_start, time_end, background_color,
               transparent_background=False, *args, **kwargs):
        return render_batch_ray_traced(
            primitives, scene, screen_width, screen_height, time_start,
            time_end, background_color, transparent_background, *args,
            **kwargs)


class RayTracedPNTrianglePrimitive(RayTracedTrianglePrimitive):
    """Curved point-normal (PN) triangle batch: each triangle is rendered as
    the quadratic Bezier (Steiner) triangle whose mid-edge control points
    bend the surface to respect the vertex normals (see
    :mod:`algan.rendering.raytracing.pn_patch`), so coarsely tessellated
    smooth surfaces keep smooth silhouettes. Construction, batching and
    vertex shading are inherited from the flat triangles; only the packed
    geometry differs (monomial patch coefficients instead of corner
    positions) and the trace kernels intersect rays with the curved patch
    (up to four hits per ray). Triangles with zero or face-constant normals
    stay exactly flat, and adjacent patches share boundary curves, so PN
    meshes stay watertight.
    """

    def project_to_screen(self, camera, light_sources):
        self._shade_vertex_colors(camera, light_sources)

        corners = self.corners.float()
        normals = self.normals.float()
        # Hot/cold split as for flat triangles, with the patch's
        # monomial coefficients as the hot geometry. corners and
        # normals share a time dimension by construction (the batching
        # constructor broadcasts them together).
        control_points = pn_control_points(corners, normals)
        self._rt_pn_ctrl = _maybe_roundtrip_time(pn_patch_coefficients(
            control_points).contiguous())
        # Tight oriented bounding box per patch: the trace kernel tests it
        # before the matrix-pencil solve to reject the (many) candidates
        # whose loose axis-aligned leaf box the ray pierces but whose actual
        # (often thin, diagonal) patch it misses.
        self._rt_pn_obb = pn_obb(control_points).contiguous()
        self._rt_pn_norm = normals.reshape(
            normals.shape[0], normals.shape[1], 9).contiguous()
        self._rt_pn_extra = self._pack_surface_extra("pn surface params")
        self._rt_pn_colors = self.colors.float().contiguous()
        self._rt_pn_mat_id, self._rt_pn_mat = self._pack_material()
        self._rt_num_frames = camera.ray_origin.shape[0]

        # Texture maps (color / material / normal). PN patches have no kernel
        # argument budget left (the general wavefront shade kernel is at
        # Taichi's 64-arg ceiling), so unlike flat triangles the UVs and the
        # per-patch texture metadata are folded into the cold pn_extra array at
        # merge time (see _merge_scene); here we just stash the raw maps + UVs.
        if self.uvs is not None:
            self._rt_pn_uvs = self.uvs.float().reshape(
                self.uvs.shape[0], self.uvs.shape[1], 6).contiguous()
            self._rt_texture_map = (self.texture_map.float().contiguous()
                                    if self.texture_map is not None else None)
            mtex = getattr(self, "material_texture_map", None)
            self._rt_material_texture = (mtex.float().contiguous()
                                         if mtex is not None else None)
            self._rt_material_flags = int(
                getattr(self, "material_texture_flags", 0) or 0)
            ntex = getattr(self, "normal_texture_map", None)
            self._rt_normal_texture = (ntex.float().contiguous()
                                       if ntex is not None else None)
        else:
            self._rt_pn_uvs = None
            self._rt_texture_map = None
            self._rt_material_texture = None
            self._rt_material_flags = 0
            self._rt_normal_texture = None

        # The patch lies in the convex hull of its control points, so
        # the control net bounds it.
        self._pack_frame_visibility(control_points.amin(-2),
                                    control_points.amax(-2),
                                    self._rt_pn_colors,
                                    "pn bounds/colors")

        self.corners = self.normals = None
        self.reflectivity = self.roughness = self.refractive_index = None
        self.colors = self.shader_param_values = None
        self.uvs = self.texture_map = None
        self.material_texture_map = self.normal_texture_map = None

        self._set_frame_buffer_bytes(camera)

        # Ensure released geometry is actually freed before rendering.
        empty_cache(force_gc=False)
        return self


def _evaluate_cubic_bezier_batch(p, t):
    """p: [..., 4, 3] control points, t: broadcastable parameter in [0, 1)."""
    mt = 1.0 - t
    return ((mt * mt * mt) * p[..., 0, :]
            + (3.0 * mt * mt * t) * p[..., 1, :]
            + (3.0 * mt * t * t) * p[..., 2, :]
            + (t * t * t) * p[..., 3, :])


class RayTracedBezierCircuitPrimitive(BezierCircuitPrimitive):
    """Planar bezier circuits rendered by ray tracing a spatio-temporal BVH.

    Circuits are sampled into polylines at a screen-space density (matching
    the rasterizer's sampling rule) but expressed in each circuit's own plane
    coordinates; the trace kernel intersects rays with the plane and
    classifies hits by an even-odd crossing test (fill) plus a min distance
    to the polyline (border). Texture-mapped circuits (``ImageMob`` etc.) are
    sampled bilinearly in-kernel from their texture grid.
    """

    stbvh_tightness = float(os.environ.get("ALGAN_STBVH_TIGHTNESS", "1.0"))
    max_samples_per_segment = 512

    def project_to_screen(self, camera, light_sources):
        corners = self.corners.float().contiguous()  # [Tc, S, 4, 3]
        num_frames = camera.ray_origin.shape[0]
        self._rt_num_frames = num_frames

        device = corners.device
        cam_o = _expand_frames(_flat_frames(camera.ray_origin, (3,)),
                               num_frames).to(device)
        sp = _expand_frames(_flat_frames(camera.screen_point, (3,)),
                            num_frames).to(device)
        sb = _expand_frames(_flat_frames(camera.screen_basis, (3, 3)),
                            num_frames).to(device)

        num_samples = self._compute_samples_per_segment(
            corners, cam_o, sp, sb, camera.screen_height)
        self._build_circuit_geometry(corners, num_samples)
        self._build_frame_bounds(corners, cam_o, sp, sb,
                                 camera.screen_height)

        # The polylines/metadata now carry everything the renderer needs;
        # release the control points to reduce resident GPU memory.
        self.corners = None

        # Per-frame buffer bytes: the u8 output, plus the f32 sample
        # accumulator in Monte Carlo mode.
        mc = PHYSICAL_LIGHTING or SAMPLES_PER_PIXEL > 1
        self._rt_frame_bytes = int(
            camera.screen_width * camera.screen_height * 5 * 4
            * (2 if mc else 1))

        # Ensure released geometry is actually freed before rendering.
        empty_cache(force_gc=False)
        return self

    def _compute_samples_per_segment(self, corners, cam_o, sp, sb, screen_h):
        """Choose the polyline density per bezier segment from the maximum
        projected control-net length over the batch (the rasterizer's rule:
        one sample every ``num_pixels_per_sample`` screen pixels).
        """
        device = corners.device
        T = cam_o.shape[0]
        Tc = corners.shape[0]
        S = corners.shape[1]
        net_max = torch.zeros((S,), device=device)
        chunk = max(1, int(2e6 // max(S * 4, 1)))
        for s in range(0, T, chunk):
            e = min(s + chunk, T)
            cor = corners if Tc == 1 else corners[s:e]
            cam_c = cam_o[s:e].view(-1, 1, 1, 3)
            sp_c = sp[s:e].view(-1, 1, 1, 3)
            n_c = sb[s:e, 2].view(-1, 1, 1, 3)
            rays = cor - cam_c
            denom = (rays * n_c).sum(-1, keepdim=True)
            t_plane = ((sp_c - cam_c) * n_c).sum(-1, keepdim=True) / denom
            proj = cam_c + t_plane * rays
            rel = proj - sp_c
            pts = torch.stack(((rel * sb[s:e, 0].view(-1, 1, 1, 3)).sum(-1),
                               (rel * sb[s:e, 1].view(-1, 1, 1, 3)).sum(-1)), -1)
            pts = pts.nan_to_num_() * (screen_h // 2)
            net = (pts[..., 1:, :] - pts[..., :-1, :]).norm(p=2, dim=-1).sum(-1)
            net_max = torch.maximum(net_max, net.amax(0))
        return (net_max / self.num_pixels_per_sample).ceil().long().clamp_(
            min=1, max=self.max_samples_per_segment)

    def _build_circuit_geometry(self, corners, num_samples):
        """Sample world-space polylines into per-circuit plane coordinates and
        pack the per-circuit metadata the trace kernel consumes.
        """
        device = corners.device
        S = corners.shape[1]
        num_segments = self.num_segments_per_object.to(device).view(-1).long()
        C = num_segments.shape[0]

        circuit_of_segment = torch.repeat_interleave(
            torch.arange(C, device=device), num_segments)
        vert_circuit = torch.repeat_interleave(circuit_of_segment, num_samples)
        V = int(num_samples.sum())

        t_params = (batch_arange(num_samples)
                    / torch.repeat_interleave(num_samples, num_samples))
        ctrl = torch.repeat_interleave(corners, num_samples, dim=1)
        verts = _evaluate_cubic_bezier_batch(ctrl, t_params.view(1, -1, 1))

        # Plane frame per circuit: normal + an arbitrary orthonormal basis.
        normals = F.normalize(self.normals.float(), p=2, dim=-1)
        centers = self.mob_center.float()
        (normals, centers), _ = _unify_time([normals, centers], "bezier planes")
        axis = torch.zeros_like(normals)
        axis[..., 0] = 1
        alt_axis = torch.zeros_like(normals)
        alt_axis[..., 1] = 1
        helper = torch.where(normals[..., :1].abs() < 0.9, axis, alt_axis)
        basis_u = F.normalize(torch.cross(normals, helper, dim=-1), p=2, dim=-1)
        basis_v = torch.cross(normals, basis_u, dim=-1)

        segment_lengths = (corners[..., 1:, :] - corners[..., :-1, :]).square().sum(-1).sum(-1)
        is_degenerate = segment_lengths < 1e-9
        edge_degenerate = torch.repeat_interleave(is_degenerate, num_samples, dim=1)

        # Absolute polyline index of the first sample of each segment, and of
        # the sample each segment's last sample connects to (closing each
        # subpath through next_segment_inds, exactly like the rasterizer).
        seg_starts = num_samples.cumsum(0) - num_samples
        seg_ends = seg_starts - 1
        seg_ends[0] = V - 1
        seg_ends = torch.roll(seg_ends, -1, 0)
        nsi = self.next_segment_inds.to(device).reshape(
            self.next_segment_inds.shape[0], S).long()
        next_start = seg_starts[nsi]  # [Tn, S]

        Tn = nsi.shape[0]
        border_visible = torch.ones((Tn, V), device=device, dtype=torch.float32)
        closing_mask = nsi <= torch.arange(S, device=device).view(1, -1)
        seg_ends_expanded = seg_ends.view(1, -1).expand(Tn, -1)
        border_visible.scatter_(1, seg_ends_expanded, torch.where(closing_mask, torch.tensor(0.0, device=device), torch.tensor(1.0, device=device)))

        (verts_e, centers_e, basis_u_e, basis_v_e, next_start_e, edge_degenerate_e, border_visible_e), T_geo = _unify_time(
            [verts, centers, basis_u, basis_v, next_start.unsqueeze(-1), edge_degenerate.unsqueeze(-1), border_visible.unsqueeze(-1)],
            "bezier geometry")
        next_start_e = next_start_e.squeeze(-1)
        edge_degenerate_e = edge_degenerate_e.squeeze(-1)
        border_visible_e = border_visible_e.squeeze(-1)

        rel = verts_e - centers_e[:, vert_circuit]
        u = (rel * basis_u_e[:, vert_circuit]).sum(-1)
        v = (rel * basis_v_e[:, vert_circuit]).sum(-1)
        locals_uv = torch.stack((u, v), -1)  # [T_geo, V, 2]
        next_uv = locals_uv.roll(-1, dims=1)
        gather_inds = next_start_e.unsqueeze(-1).expand(T_geo, -1, 2)
        next_uv[:, seg_ends] = torch.gather(locals_uv, 1, gather_inds)
        self._rt_edges = torch.cat((locals_uv, next_uv, border_visible_e.unsqueeze(-1)), -1).float().contiguous()
        self._rt_edges = torch.where(
            edge_degenerate_e.unsqueeze(-1),
            torch.tensor([1e9, 1e9, 1e9, 1e9, 0.0], device=device),
            self._rt_edges
        )

        samples_per_circuit = torch.zeros((C,), dtype=torch.long, device=device)
        samples_per_circuit.index_add_(0, circuit_of_segment, num_samples)
        edge_offsets = torch.zeros((C + 1,), dtype=torch.long, device=device)
        edge_offsets[1:] = samples_per_circuit.cumsum(0)
        self._rt_edge_offsets = edge_offsets.to(torch.int32).contiguous()
        self._rt_circuit_of_segment = circuit_of_segment

        # Texture-grid transform: maps plane (u, v) displacements to the
        # mob-basis coordinates used by the texture lookup.
        def scaled(basis):
            basis = basis.float()
            return basis / basis.norm(p=2, dim=-1, keepdim=True).square().clamp_min(1e-12)

        basis1, basis2 = scaled(self.basis1), scaled(self.basis2)
        border_width = self.border_width.float().reshape(
            self.border_width.shape[0], C)
        grid_w = self.grid_width.float().reshape(self.grid_width.shape[0], C)
        grid_h = self.grid_height.float().reshape(self.grid_height.shape[0], C)
        glow_radius = self.glow_radius.float().reshape(
            self.glow_radius.shape[0], C)
        (centers_m, normals_m, bu_m, bv_m, b1_m, b2_m, bw_m, gw_m, gh_m, glow_radius_m), Tm = _unify_time(
            [centers, normals, basis_u, basis_v, basis1, basis2,
             border_width.unsqueeze(-1), grid_w.unsqueeze(-1),
             grid_h.unsqueeze(-1), glow_radius.unsqueeze(-1)], "bezier metadata")
        filled = torch.full((Tm, C, 1), 1.0 if self.filled else 0.0,
                            device=device)
        tex = torch.stack((
            (b1_m * bu_m).sum(-1), (b1_m * bv_m).sum(-1),
            (b2_m * bu_m).sum(-1), (b2_m * bv_m).sum(-1)), -1).nan_to_num_()
        self._rt_circuit_meta = torch.cat(
            (centers_m, normals_m, bu_m, bv_m, bw_m, filled, gw_m, gh_m, tex, glow_radius_m),
            -1).contiguous()

        colors = self.colors.float()
        if colors.dim() == 3:  # plain fills: a 1x1 "texture" grid
            colors = colors.unsqueeze(-2)
        self._rt_circuit_colors = colors.contiguous()
        self._rt_circuit_border_colors = self.border_color.float().contiguous()
        self._rt_border_width = border_width

    def _build_frame_bounds(self, corners, cam_o, sp, sb, screen_h):
        """Per-frame circuit AABBs (from control-point hulls, inflated by the
        screen-space border width and glow radius), with invisible frames marked empty.
        """
        device = corners.device
        C = self._rt_edge_offsets.shape[0] - 1
        circuit_of_segment = self._rt_circuit_of_segment

        seg_lo = corners.amin(-2)
        seg_hi = corners.amax(-2)
        Tb = seg_lo.shape[0]
        idx = circuit_of_segment.view(1, -1, 1).expand(Tb, -1, 3)
        lo = torch.full((Tb, C, 3), EMPTY_LO, device=device).scatter_reduce_(
            1, idx, seg_lo, "amin", include_self=True)
        hi = torch.full((Tb, C, 3), EMPTY_HI, device=device).scatter_reduce_(
            1, idx, seg_hi, "amax", include_self=True)

        fill_alpha = self._rt_circuit_colors.opacity.squeeze(-1).amax(-1)  # over texture
        fill_min = self._rt_circuit_colors.opacity.squeeze(-1).amin(-1)
        if not self.filled:
            fill_alpha = torch.zeros_like(fill_alpha)
        border_alpha = self._rt_circuit_border_colors.opacity.squeeze(-1)
        border_on = self._rt_border_width > 1e-3
        glow_alpha = self._rt_circuit_colors[..., 3].amax(-1)
        visible = (fill_alpha > MIN_ALPHA) | (
            (border_alpha > MIN_ALPHA) & border_on) | (glow_alpha > 0.0)
        (lo, hi, visible, fill_min, border_alpha, border_on), _ = _unify_time(
            [lo, hi, visible.unsqueeze(-1), fill_min.unsqueeze(-1),
             border_alpha.unsqueeze(-1), border_on.unsqueeze(-1)],
            "bezier bounds/colors")
        visible = visible.squeeze(-1)
        # A circuit is opaque (prunes hits behind it while gathering) only if
        # every region a hit can land in -- the fill/texture and, when shown,
        # the border -- is fully opaque.
        opaque = (fill_min.squeeze(-1) >= 1.0 - 1e-6) & (
            (~border_on.squeeze(-1))
            | (border_alpha.squeeze(-1) >= 1.0 - 1e-6))
        if not self.filled:
            opaque = torch.zeros_like(opaque)
        self._rt_frame_opaque = opaque.contiguous()
        lo = torch.where(visible.unsqueeze(-1), lo,
                         torch.tensor(EMPTY_LO, device=device))
        hi = torch.where(visible.unsqueeze(-1), hi,
                         torch.tensor(EMPTY_HI, device=device))

        # Inflate by the border (+ anti-crack outline) width converted to
        # world units at each circuit's distance from the camera, plus the glow radius.
        b1_norm = sb[:, 1].norm(p=2, dim=-1)
        screen_dist = (sp - cam_o).norm(p=2, dim=-1)
        pixel_world_scale = 2.0 / (screen_h * b1_norm * screen_dist).clamp_min(1e-12)
        centers = self._rt_circuit_meta[..., :3]
        dist = (centers - cam_o.view(-1, 1, 3)).norm(p=2, dim=-1)
        world_per_px = (pixel_world_scale.view(-1, 1) * dist).amax(0)
        
        glow_rad = torch.where(glow_alpha > 0.0, self.glow_radius.squeeze(-1), 0.0)
        glow_rad_max = glow_rad.amax(0)
        
        inflate = (self._rt_border_width.amax(0) + 1.0) * world_per_px + glow_rad_max
        self._rt_frame_lo = (lo - inflate.view(1, -1, 1)).contiguous()
        self._rt_frame_hi = (hi + inflate.view(1, -1, 1)).contiguous()

    def get_memory_used_per_timestep(self):
        return self._rt_frame_bytes

    def get_memory_used_for_blending(self, start_ind, end_ind):
        return 0  # Blending happens in-register inside the trace kernel.

    def render(self, primitives, scene, save_image, screen_width,
               screen_height, time_start, time_end, background_color,
               transparent_background=False, *args, **kwargs):
        return render_batch_ray_traced(
            primitives, scene, screen_width, screen_height, time_start,
            time_end, background_color, transparent_background, *args,
            **kwargs)


def _empty_scene_part(device):
    """Placeholder STBVH + arrays for an absent geometry type."""
    lo = torch.full((1, 1, 3), EMPTY_LO, device=device)
    hi = torch.full((1, 1, 3), EMPTY_HI, device=device)
    return build_stbvh(lo, hi, num_frames=1)


def _merge_scene(primitives):
    """Merge the batch's collections into one set per geometry type --
    triangles, PN patches and bezier circuits, each with a single STBVH
    over all frames -- cached for the batch.
    """
    first = primitives[0]
    cached = getattr(first, "_rt_merged_scene", None)
    if cached is not None:
        return cached

    empty_cache(force_gc=False)
    device = COMPUTING_DEFAULTS.render_device
    pn_patches = [p for p in primitives
                  if isinstance(p, RayTracedPNTrianglePrimitive)]
    triangles = [p for p in primitives
                 if isinstance(p, RayTracedTrianglePrimitive)
                 and not isinstance(p, RayTracedPNTrianglePrimitive)]
    beziers = [p for p in primitives
               if isinstance(p, RayTracedBezierCircuitPrimitive)]
    unknown = [p for p in primitives
               if p not in triangles and p not in pn_patches
               and p not in beziers]
    if unknown:
        raise TypeError(
            "The ray traced renderer can only draw ray traced primitives; "
            f"got {[type(p).__name__ for p in unknown]}. Was "
            "enable_ray_tracing() called before the mobs were created?")
    num_frames = max(p._rt_num_frames for p in primitives)

    # Any PN patch carrying a texture map forces the whole batch onto the
    # general wavefront tracer (the only kernel that samples PN textures); the
    # megakernel's PN path has no UVs. Flags PN color maps too (unlike flat
    # colour maps, which the megakernel can sample).
    has_pn_textures = any(
        getattr(p, "_rt_pn_uvs", None) is not None for p in pn_patches)

    scene = {}
    scene["has_pn_textures"] = has_pn_textures
    # Shared flat texel buffer for *all* texture maps, flat-triangle and
    # PN-patch alike (color / material / normal). Each map is appended once,
    # padded to 5 channels and flattened to [T, W*H, 5]; its placement is a
    # (offset, w, h) triplet recorded in the consuming geometry's metadata
    # (offset -1 = no map). Flat triangles key those triplets by tri_tex_meta;
    # PN patches fold them into pn_extra (no kernel-arg budget left). Assembled
    # into scene["textures"] once both geometry blocks below have appended.
    _texture_tensors = []
    _texel_offset = [0]

    def _append_texture(tex):
        if tex is None:
            return (-1, 0, 0)
        if tex.dim() == 3:  # [W, H, C]
            tex = tex.unsqueeze(0)  # [1, W, H, C]
        w, h, c = tex.shape[-3], tex.shape[-2], tex.shape[-1]
        if c < 5:
            tex = torch.cat(
                (tex, tex.new_zeros((*tex.shape[:-1], 5 - c))), -1)
        # Flatten W and H (dimensions 1 and 2).
        _texture_tensors.append(tex.reshape(tex.shape[0], -1, 5))
        o = _texel_offset[0]
        _texel_offset[0] += w * h
        return (o, w, h)

    scene["tex_has_refractive"] = False
    if triangles:
        colored_triangles = [p for p in triangles if getattr(p, "_rt_tri_uvs", None) is None]
        textured_triangles = [p for p in triangles if getattr(p, "_rt_tri_uvs", None) is not None]
        all_triangles = colored_triangles + textured_triangles
        num_colored = sum(p._rt_tri_pos.shape[1] for p in colored_triangles)

        scene["num_colored_triangles"] = num_colored
        scene["tri_pos"] = _cat_collections(
            [p._rt_tri_pos for p in all_triangles], 1, "triangle merge")
        scene["tri_norm"] = _cat_collections(
            [p._rt_tri_norm for p in all_triangles], 1, "triangle merge")
        scene["tri_extra"] = _cat_collections(
            [p._rt_tri_extra for p in all_triangles], 1, "triangle merge")

        # Vertex colors are merged for *all* triangles (textured included):
        # a textured primitive may carry only material/normal maps, in which
        # case the kernel falls back to its per-vertex colors (color-map
        # meta offset -1).
        scene["tri_colors"] = _cat_collections(
            [p._rt_tri_colors for p in all_triangles], 1, "triangle merge")

        scene["has_material_textures"] = any(
            getattr(p, "_rt_material_texture", None) is not None
            or getattr(p, "_rt_normal_texture", None) is not None
            for p in textured_triangles)

        if textured_triangles:
            scene["tri_uvs"] = _cat_collections(
                [p._rt_tri_uvs for p in textured_triangles], 1, "triangle merge")

            # Each map's placement is a (offset, w, h) triplet in the
            # per-triangle meta row (offset -1 = no map -> per-vertex
            # fallback). Meta layout: cols 0-2 color map, 3-5 material map
            # (channels: reflectivity, roughness, refractive index), 6-8
            # normal map, 9 bitmask of texture-driven material properties.
            tex_meta_list = []
            for p in textured_triangles:
                color_meta = _append_texture(p._rt_texture_map)
                mtex = getattr(p, "_rt_material_texture", None)
                material_meta = _append_texture(mtex)
                normal_meta = _append_texture(
                    getattr(p, "_rt_normal_texture", None))
                flags = int(getattr(p, "_rt_material_flags", 0) or 0)
                if (mtex is not None and (flags & 4)
                        and bool((mtex[..., 2] > 1.0 + 1e-4).any())):
                    scene["tex_has_refractive"] = True
                num_tris = p._rt_tri_pos.shape[1]
                meta = torch.tensor(
                    [*color_meta, *material_meta, *normal_meta, flags],
                    dtype=torch.int32, device=device).view(1, 10).expand(
                        num_tris, 10)
                tex_meta_list.append(meta)
            scene["tri_tex_meta"] = torch.cat(tex_meta_list, 0).contiguous()
        else:
            scene["tri_uvs"] = torch.zeros((1, 1, 6), device=device)
            scene["tri_tex_meta"] = torch.full((1, 10), -1, dtype=torch.int32, device=device)

        scene["tri_mat_id"] = _cat_collections(
            [p._rt_tri_mat_id for p in all_triangles], 1, "triangle merge")
        scene["tri_mat"] = _cat_collections(
            [p._rt_tri_mat for p in all_triangles], 1, "triangle merge")

        lo = _cat_collections([p._rt_frame_lo for p in all_triangles], 1,
                              "triangle merge")
        hi = _cat_collections([p._rt_frame_hi for p in all_triangles], 1,
                              "triangle merge")
        opaque = _cat_collections([p._rt_frame_opaque for p in all_triangles], 1,
                                  "triangle merge")
        # Median-split ordering: ~25% faster traversal than Morton at ~0.2s
        # extra build per batch; byte-identical for triangles (the depth-peel
        # is arrangement-invariant). PN/bezier BVHs below stay Morton -- their
        # seam de-dup is discovery-order sensitive (see stbvh._BVH_BUILD).
        scene["tri_bvh"] = build_stbvh(
            lo, hi, num_frames=num_frames,
            tightness=RayTracedTrianglePrimitive.stbvh_tightness,
            opaque=opaque, builder="split")
    else:
        scene["tri_pos"] = torch.zeros((1, 1, 9), device=device)
        scene["tri_norm"] = torch.zeros((1, 1, 9), device=device)
        scene["tri_extra"] = torch.zeros((1, 1, 15), device=device)
        scene["tri_colors"] = torch.zeros((1, 1, 3, 5), device=device)
        scene["tri_uvs"] = torch.zeros((1, 1, 6), device=device)
        scene["tri_tex_meta"] = torch.full((1, 10), -1, dtype=torch.int32, device=device)
        scene["num_colored_triangles"] = 0
        scene["has_material_textures"] = False
        scene["tri_mat_id"] = torch.zeros((1, 1), dtype=torch.int32,
                                          device=device)
        scene["tri_mat"] = torch.zeros((1, 1, MAT_W), device=device)
        scene["tri_bvh"] = _empty_scene_part(device)
    scene["num_triangles"] = scene["tri_pos"].shape[1] if triangles else 0

    # Temporal compression of triangle positions (knot representation). The BVH
    # is already built from the per-frame bounds (independent of tri_pos), so the
    # dense positions can be dropped once compressed -- the knot kernel
    # reconstructs each frame's geometry in-register. Only flat triangles are
    # wired up; reflective/fragment-shaded/shadowed batches keep the dense path.
    scene["tri_tc"] = None
    scene["tri_has_reflective"] = False
    # Only mode 2 (in-kernel knot kernel) consumes the compressed positions.
    # Mode 3 (timeline-direct) already expanded geometry to dense upstream and
    # renders through the existing dense kernel, so it must not compress here.
    if (TIME_COMPRESS == 2 and triangles
            and not scene["has_material_textures"] and not has_pn_textures):
        tc = compress_time(scene["tri_pos"])
        scene["tri_tc"] = tc
        refl = scene["tri_extra"][..., 0:6:2]  # per-corner reflectivity columns
        scene["tri_has_reflective"] = bool((refl > MIN_ALPHA).any())
        if os.environ.get("ALGAN_TIME_COMPRESS_STATS", "0") == "1":
            dense_b = scene["tri_pos"].numel() * 4
            knot_b = 4 * sum(t.numel() for t in (
                tc.knot_val, tc.knot_base, tc.sched_id,
                tc.sched_seg, tc.sched_z, tc.sched_nknots))
            print(f"[time_compress] T={tc.T} N={tc.N} "
                  f"total_knots={tc.total_knots} schedules={tc.num_schedules} "
                  f"| tri_pos dense={dense_b / 1e6:.2f}MB "
                  f"knots={knot_b / 1e6:.2f}MB "
                  f"ratio={dense_b / max(knot_b, 1):.1f}x")

    if pn_patches:
        scene["pn_ctrl"] = _cat_collections(
            [p._rt_pn_ctrl for p in pn_patches], 1, "pn merge")
        scene["pn_obb"] = _cat_collections(
            [p._rt_pn_obb for p in pn_patches], 1, "pn merge")
        scene["pn_norm"] = _cat_collections(
            [p._rt_pn_norm for p in pn_patches], 1, "pn merge")
        # Fold per-patch UVs + texture metadata into the (cold, hit-only)
        # pn_extra array: PN has no kernel-arg budget for its own uv/meta/
        # texture arrays (the general wavefront shade kernel is at Taichi's
        # 64-arg cap), so it reads them from widened pn_extra. Layout appended
        # after the existing 15 material cols: cols 15-20 per-corner UV, 21-23
        # color map (offset, w, h) into the shared ``textures`` buffer, 24-26
        # material map, 27-29 normal map, 30 material bitmask. A color-map
        # offset of -1 means fall back to per-vertex pn_colors. The array is
        # widened unconditionally (even with no maps -> all -1) because the
        # default wavefront path shades every PN scene through this kernel, so
        # the texture-sampling code always executes and must find 31 columns.
        # Every patch keeps its slot (no colored/textured reorder -- the PN
        # morton BVH seam de-dup is discovery-order sensitive).
        pn_extra_list = []
        for p in pn_patches:
            extra = p._rt_pn_extra                # [Te, Np, 15]
            Np = extra.shape[1]
            uvs = getattr(p, "_rt_pn_uvs", None)
            if uvs is None:
                uvs = torch.zeros((1, Np, 6), device=device)
            if has_pn_textures:
                color_meta = _append_texture(getattr(p, "_rt_texture_map", None))
                mtex = getattr(p, "_rt_material_texture", None)
                material_meta = _append_texture(mtex)
                normal_meta = _append_texture(
                    getattr(p, "_rt_normal_texture", None))
                flags = int(getattr(p, "_rt_material_flags", 0) or 0)
                if (mtex is not None and (flags & 4)
                        and bool((mtex[..., 2] > 1.0 + 1e-4).any())):
                    scene["tex_has_refractive"] = True
                meta_vals = [*color_meta, *material_meta, *normal_meta, flags]
            else:
                meta_vals = [-1, 0, 0, -1, 0, 0, -1, 0, 0, 0]
            T = max(extra.shape[0], uvs.shape[0])
            # UVs inherit the (CPU) animation device from the per-mob build,
            # while extra/meta are on the render device -- unify before cat.
            extra_e = _expand_frames(extra, T).to(device)
            uvs_e = _expand_frames(uvs, T).to(device)
            meta_e = torch.tensor(
                meta_vals, dtype=torch.float32, device=device
            ).view(1, 1, 10).expand(T, Np, 10)
            pn_extra_list.append(torch.cat([extra_e, uvs_e, meta_e], -1))
        scene["pn_extra"] = _cat_collections(pn_extra_list, 1, "pn merge")
        scene["pn_colors"] = _cat_collections(
            [p._rt_pn_colors for p in pn_patches], 1, "pn merge")
        scene["pn_mat_id"] = _cat_collections(
            [p._rt_pn_mat_id for p in pn_patches], 1, "pn merge")
        scene["pn_mat"] = _cat_collections(
            [p._rt_pn_mat for p in pn_patches], 1, "pn merge")
        lo = _cat_collections([p._rt_frame_lo for p in pn_patches], 1,
                              "pn merge")
        hi = _cat_collections([p._rt_frame_hi for p in pn_patches], 1,
                              "pn merge")
        opaque = _cat_collections([p._rt_frame_opaque for p in pn_patches],
                                  1, "pn merge")
        scene["pn_bvh"] = build_stbvh(
            lo, hi, num_frames=num_frames,
            tightness=RayTracedPNTrianglePrimitive.stbvh_tightness,
            opaque=opaque)
    else:
        scene["pn_ctrl"] = torch.zeros((1, 1, 18), device=device)
        scene["pn_obb"] = torch.zeros((1, 1, 12), device=device)
        scene["pn_norm"] = torch.zeros((1, 1, 9), device=device)
        # 31 cols (15 material + 6 UV + 10 tex-meta) to match the real path, so
        # the wavefront's PN texture reads never run off the stub (see above).
        scene["pn_extra"] = torch.zeros((1, 1, 31), device=device)
        scene["pn_colors"] = torch.zeros((1, 1, 3, 5), device=device)
        scene["pn_mat_id"] = torch.zeros((1, 1), dtype=torch.int32,
                                         device=device)
        scene["pn_mat"] = torch.zeros((1, 1, MAT_W), device=device)
        scene["pn_bvh"] = _empty_scene_part(device)
    scene["num_pn"] = scene["pn_ctrl"].shape[1] if pn_patches else 0

    # Assemble the shared texel buffer now that both the flat-triangle and PN
    # blocks above have appended their maps (offsets recorded in tri_tex_meta /
    # pn_extra respectively).
    if _texture_tensors:
        scene["textures"] = _cat_collections(
            _texture_tensors, 1, "texture merge")
    else:
        scene["textures"] = torch.zeros((1, 1, 5), device=device)
    scene["has_pn_textures"] = has_pn_textures

    # Refraction is active iff some triangle/PN surface carries a meaningful
    # index of refraction (extra columns 6-8, per-corner; 0/1 = no bending).
    # Used to gate the wavefront's refraction template and to route refractive
    # batches to the general wavefront (the only path that refracts).
    def _extra_has_refractive(extra):
        return bool((extra[..., 6:9] > 1.0 + 1e-4).any())
    scene["has_refractive"] = (_extra_has_refractive(scene["tri_extra"])
                               or _extra_has_refractive(scene["pn_extra"])
                               or bool(scene.get("tex_has_refractive")))

    if beziers:
        scene["circuit_meta"] = _cat_collections(
            [p._rt_circuit_meta for p in beziers], 1, "bezier merge")
        scene["circuit_border_colors"] = _cat_collections(
            [p._rt_circuit_border_colors for p in beziers], 1, "bezier merge")
        max_points = max(p._rt_circuit_colors.shape[2] for p in beziers)
        padded = []
        for p in beziers:
            c = p._rt_circuit_colors
            if c.shape[2] < max_points:
                pad = torch.zeros((c.shape[0], c.shape[1],
                                   max_points - c.shape[2], c.shape[3]),
                                  device=c.device)
                c = torch.cat((c, pad), 2)
            padded.append(c)
        scene["circuit_colors"] = _cat_collections(padded, 1, "bezier merge")
        scene["edges_2d"] = _cat_collections(
            [p._rt_edges for p in beziers], 1, "bezier merge")
        offsets, shift = [torch.zeros((1,), dtype=torch.int32, device=device)], 0
        for p in beziers:
            offsets.append(p._rt_edge_offsets[1:].long() + shift)
            shift = shift + p._rt_edges.shape[1]
        scene["edge_offsets"] = torch.cat(
            [o.to(torch.int32) for o in offsets]).contiguous()
        lo = _cat_collections([p._rt_frame_lo for p in beziers], 1,
                              "bezier merge")
        hi = _cat_collections([p._rt_frame_hi for p in beziers], 1,
                              "bezier merge")
        opaque = _cat_collections([p._rt_frame_opaque for p in beziers], 1,
                                  "bezier merge")
        scene["bez_bvh"] = build_stbvh(
            lo, hi, num_frames=num_frames,
            tightness=RayTracedBezierCircuitPrimitive.stbvh_tightness,
            opaque=opaque)
        scene["num_circuits"] = scene["circuit_meta"].shape[1]
    else:
        scene["circuit_meta"] = torch.zeros((1, 1, 21), device=device)
        scene["circuit_colors"] = torch.zeros((1, 1, 1, 5), device=device)
        scene["circuit_border_colors"] = torch.zeros((1, 1, 5), device=device)
        scene["edges_2d"] = torch.zeros((1, 1, 5), device=device)
        scene["edge_offsets"] = torch.zeros((2,), dtype=torch.int32,
                                            device=device)
        scene["bez_bvh"] = _empty_scene_part(device)
        scene["num_circuits"] = 0

    scene["num_frames"] = num_frames
    # The merged tensors replace the per-collection ones; release the
    # originals so peak GPU memory stays close to one copy of the scene.
    for p in triangles:
        p._rt_tri_pos = p._rt_tri_norm = None
        p._rt_tri_extra = p._rt_tri_colors = None
        p._rt_tri_mat_id = p._rt_tri_mat = None
        p._rt_tri_uvs = p._rt_texture_map = None
        p._rt_material_texture = p._rt_normal_texture = None
        p._rt_frame_lo = p._rt_frame_hi = p._rt_frame_opaque = None
    for p in pn_patches:
        p._rt_pn_ctrl = p._rt_pn_norm = None
        p._rt_pn_obb = None
        p._rt_pn_extra = p._rt_pn_colors = None
        p._rt_pn_mat_id = p._rt_pn_mat = None
        p._rt_pn_uvs = p._rt_texture_map = None
        p._rt_material_texture = p._rt_normal_texture = None
        p._rt_frame_lo = p._rt_frame_hi = p._rt_frame_opaque = None
    for p in beziers:
        p._rt_circuit_meta = p._rt_circuit_colors = None
        p._rt_circuit_border_colors = p._rt_edges = None
        p._rt_frame_lo = p._rt_frame_hi = p._rt_frame_opaque = None

    empty_cache(force_gc=False)
    first._rt_merged_scene = scene
    return scene


def _pack_lights(light_sources, num_frames, device):
    """Per-frame positions [T, L, 3] and RGB radiances [T, L, 3] of the
    scene's point lights (as prepared by the scene before rendering).
    """
    positions, colors = [], []
    for light in light_sources or ():
        positions.append(_expand_frames(
            _flat_frames(light.origin, (3,)), num_frames))
        col = light.light_color.reshape(light.light_color.shape[0], -1)
        colors.append(_expand_frames(col[:, :3].float(), num_frames))
    if not positions:
        return (torch.zeros((1, 1, 3), device=device),
                torch.zeros((1, 1, 3), device=device), 0)
    light_pos = torch.stack(positions, 1).to(device).contiguous()
    light_col = torch.stack(colors, 1).to(device).contiguous()
    return light_pos, light_col, light_pos.shape[1]


def _prefill_background(out, background_color, frame_offset, device):
    """Fill the output buffer with the background. Solid colors arrive as a
    float [channels] tensor in [0, 1]; animated/image backgrounds arrive as a
    uint8 row tensor [1 + frames * pixels, channels] (leading padding row).
    """
    num_frames, num_pixels, C_out = out.shape
    bg = background_color.to(device)
    if bg.dim() <= 1 or bg.shape[0] == 1:  # solid color (in [0, 1] floats)
        vals = (bg.float().flatten()[:5] * 255).round_().clamp_(0, 255)
        k = min(vals.shape[0], C_out)
        out[..., :k] = vals[:k].to(out.dtype)
        if C_out > k:
            # Alpha (and any missing channel) defaults to the background's
            # last channel, matching opaque-by-default behavior.
            out[..., k:] = vals[-1].to(out.dtype)
    else:
        rows = bg.reshape(-1, bg.shape[-1])[1:]
        rows = rows[frame_offset * num_pixels:
                    (frame_offset + num_frames) * num_pixels]
        rows = rows.view(num_frames, num_pixels, -1)
        k = min(rows.shape[-1], C_out)
        out[..., :k] = rows[..., :k].to(out.dtype)
        if C_out > k:
            out[..., k:] = rows[..., -1:].to(out.dtype)


def _downsample_background(background_color, aa, num_frames, screen_height,
                           screen_width):
    """Average a super-sampled animated/image background down to the output
    resolution (box filter, matching ``post_process_frames``), so the in-place
    anti-aliased renderer -- which samples the background once per output pixel
    -- gets a background at the right resolution.

    Solid colors (resolution-free) and backgrounds that are not super-sampled
    (row count not ``num_frames * (screen_height*aa) * (screen_width*aa)``) are
    returned unchanged.
    """
    bg = background_color
    if not torch.is_tensor(bg) or bg.dim() <= 1 or bg.shape[0] == 1:
        return bg  # solid color
    C = bg.shape[-1]
    body = bg.reshape(-1, C)[1:]  # drop the leading padding row
    h_aa, w_aa = screen_height * aa, screen_width * aa
    if body.shape[0] != num_frames * h_aa * w_aa:
        return bg  # not a super-sampled image background; leave as-is
    img = body.view(num_frames, h_aa, w_aa, C).float().permute(0, 3, 1, 2)
    ds = F.avg_pool2d(img, aa).permute(0, 2, 3, 1).reshape(-1, C)
    ds = (ds + 0.5).clamp_(0, 255).to(bg.dtype)
    return torch.cat((ds[:1], ds), 0)


def _alloc_wavefront_state(memory, tn, sca_width):
    """Allocate the wavefront's per-ray global state from the render memory pool
    (a bump allocator) rather than fresh ``torch.empty`` tensors.

    The caller snapshots ``memory.get_pointers()`` before each tile and restores
    them after, so this ~hundreds-of-MB of state is released *deterministically*
    at the end of every iteration and the next tile reuses the same arena bytes.
    Previously these were ``torch.empty`` allocations that the CUDA caching
    allocator / Python GC didn't reclaim before the next tile asked for its own,
    so consecutive tiles' state piled up and OOMed the GPU at HD/AA>=2.
    """
    f32 = torch.float32
    i32 = torch.int32
    return (
        memory.get_tensor((tn, 3), f32),          # rs_ro
        memory.get_tensor((tn, 3), f32),          # rs_rd
        memory.get_tensor((tn, 4), f32),          # rs_acc
        memory.get_tensor((tn, sca_width), f32),  # rs_sca (4 tri / 5 general)
        memory.get_tensor((tn, 4), i32),          # rs_int
        memory.get_tensor((tn, KBUF), f32),       # rs_kt
        memory.get_tensor((tn, KBUF), f32),       # rs_kl
        memory.get_tensor((tn, KBUF), f32),       # rs_ka
        memory.get_tensor((tn, KBUF), f32),       # rs_kb
        memory.get_tensor((tn, KBUF), i32),       # rs_kp
        memory.get_tensor((tn, KBUF), i32),       # rs_kf
    )


def render_triangles_wavefront(
        t_nodes, t_node_miss, t_leaf_prim, t_leaf_tspan, t_first_leaf,
        tri_pos, tri_norm, tri_extra, tri_colors,
        tri_uvs, tri_tex_meta, textures, num_colored_triangles,
        cam_origin, screen_point, pixel_basis_x, pixel_basis_y,
        time_start, time_end, width, height, half_screen_w, half_screen_h,
        layer_offset_triangles, max_bounces, transparent, memory, out,
        aa_level=1):
    """Wavefront orchestration for a triangle-only batch: byte-identical to
    ``render_triangles_stbvh`` but split into per-stage kernels over per-ray
    global state, with PyTorch ray compaction between host iterations.

    When ``aa_level > 1``, runs ``aa^2`` sub-pixel passes at the output
    resolution (in-place MSAA), accumulating into a float buffer and averaging
    at the end -- no super-sampled frame buffer needed.

    Geometry/intersection/shading is the megakernel's (see
    :mod:`algan.rendering.raytracing.wavefront_taichi`); this just owns the
    state buffers and the gen -> (traverse -> shade -> compact)* -> composite
    loop.
    """
    device = out.device
    t_val = _get_tonemap_t_val()
    n = (time_end - time_start) * width * height
    i32 = torch.int32
    f32 = torch.float32
    max_iters = MAX_SURFACES_PER_RAY + max_bounces + 2
    aa = max(1, int(aa_level))
    do_aa = aa > 1
    inv_aa = 1.0 / aa

    # Allocate the per-pixel float accumulator for in-place AA (persistent
    # across tiles and sub-pixel passes; zeroed once before all passes).
    aa_accum = None
    if do_aa:
        aa_accum = memory.get_tensor((n, 5 if transparent else 4), f32)
        aa_accum.zero_()

    for si in range(aa):
        for sj in range(aa):
            jx = (si + 0.5) * inv_aa if do_aa else 0.5
            jy = (sj + 0.5) * inv_aa if do_aa else 0.5

            # Screen tiling: process the chunk's rays in bounded tiles so the
            # per-ray global state never exceeds one tile's worth, independent
            # of resolution or chunk length (the megakernel holds no such
            # state). State is tile-local; the global ray (frame/pixel + output
            # cell) is ``tile_start + r``. State is allocated from the render
            # pool and freed (set_pointers) after every tile.
            for tile_start in range(0, n, WAVEFRONT_TILE_RAYS):
                tn = min(WAVEFRONT_TILE_RAYS, n - tile_start)
                state_ptrs = memory.get_pointers()
                (rs_ro, rs_rd, rs_acc, rs_sca, rs_int,
                 rs_kt, rs_kl, rs_ka, rs_kb, rs_kp, rs_kf) = \
                    _alloc_wavefront_state(memory, tn, 4)

                wf_gen_triangle(
                    cam_origin, screen_point, pixel_basis_x, pixel_basis_y,
                    int(time_start), int(width), int(height),
                    float(half_screen_w), float(half_screen_h),
                    int(max_bounces),
                    int(tile_start), float(jx), float(jy),
                    rs_ro, rs_rd, rs_acc, rs_sca, rs_int)

                active = torch.arange(tn, dtype=i32, device=device)
                it = 0
                while active.numel() > 0 and it < max_iters:
                    na = int(active.numel())
                    wf_traverse_triangle(
                        active, na, t_nodes, t_node_miss, t_leaf_prim,
                        t_leaf_tspan,
                        int(t_first_leaf), tri_pos,
                        float(layer_offset_triangles),
                        int(time_start), int(width), int(height),
                        int(tile_start),
                        rs_ro, rs_rd, rs_sca, rs_int,
                        rs_kt, rs_kl, rs_ka, rs_kb, rs_kp, rs_kf)
                    wf_shade_triangle(
                        active, na, t_nodes, t_node_miss, t_leaf_prim,
                        t_leaf_tspan, int(t_first_leaf),
                        tri_pos, tri_norm, tri_extra, tri_colors,
                        tri_uvs, tri_tex_meta, textures,
                        num_colored_triangles,
                        int(time_start), int(width), int(height),
                        int(tile_start),
                        rs_ro, rs_rd, rs_acc, rs_sca, rs_int,
                        rs_kt, rs_kl, rs_ka, rs_kb, rs_kp, rs_kf)
                    active = (rs_int[:, 2] == 0).nonzero(
                        as_tuple=True)[0].to(i32)
                    it += 1

                if do_aa:
                    wf_composite_aa(
                        int(time_start), int(width), int(height),
                        1 if transparent else 0, int(tile_start),
                        rs_acc, rs_sca, out, aa_accum)
                else:
                    wf_composite(
                        int(time_start), int(width), int(height),
                        1 if transparent else 0, int(tile_start),
                        rs_acc, rs_sca, t_val, float(TONEMAP_EXPOSURE), out)
                # Release this tile's state back to the pool before the next
                # tile.
                memory.set_pointers(state_ptrs)

    if do_aa:
        wf_finalize_aa(int(width), int(height),
                       1 if transparent else 0,
                       float(inv_aa * inv_aa), t_val, float(TONEMAP_EXPOSURE), aa_accum, out)


def render_triangles_wavefront_knots(
        t_nodes, t_node_miss, t_leaf_prim, t_leaf_tspan, t_first_leaf,
        tc, tri_norm, tri_extra, tri_colors,
        tri_uvs, tri_tex_meta, textures, num_colored_triangles,
        cam_origin, screen_point, pixel_basis_x, pixel_basis_y,
        time_start, time_end, width, height, half_screen_w, half_screen_h,
        layer_offset_triangles, max_bounces, transparent, memory, out,
        aa_level=1):
    """Wavefront orchestration with knot (compressed) geometry: identical to
    ``render_triangles_wavefront`` but the traverse stage reconstructs positions
    from ``tc`` (a TimeCompressed) instead of a dense ``tri_pos``. Restricted to
    non-reflective batches (the shade stage's reflection normal -- the only use
    of positions there -- is inert, so a 1x1x9 dummy is passed).

    When ``aa_level > 1``, runs in-place MSAA (see
    ``render_triangles_wavefront``)."""
    device = out.device
    t_val = _get_tonemap_t_val()
    n = (time_end - time_start) * width * height
    f32 = torch.float32
    i32 = torch.int32
    max_iters = MAX_SURFACES_PER_RAY + max_bounces + 2
    dummy_pos = torch.zeros((1, 1, 9), dtype=f32, device=device)
    aa = max(1, int(aa_level))
    do_aa = aa > 1
    inv_aa = 1.0 / aa

    aa_accum = None
    if do_aa:
        aa_accum = memory.get_tensor((n, 5 if transparent else 4), f32)
        aa_accum.zero_()

    for si in range(aa):
        for sj in range(aa):
            jx = (si + 0.5) * inv_aa if do_aa else 0.5
            jy = (sj + 0.5) * inv_aa if do_aa else 0.5

            # Screen tiling (see render_triangles_wavefront): bounded per-tile
            # state, pool-allocated and freed after every tile.
            for tile_start in range(0, n, WAVEFRONT_TILE_RAYS):
                tn = min(WAVEFRONT_TILE_RAYS, n - tile_start)
                state_ptrs = memory.get_pointers()
                (rs_ro, rs_rd, rs_acc, rs_sca, rs_int,
                 rs_kt, rs_kl, rs_ka, rs_kb, rs_kp, rs_kf) = \
                    _alloc_wavefront_state(memory, tn, 4)

                wf_gen_triangle(
                    cam_origin, screen_point, pixel_basis_x, pixel_basis_y,
                    int(time_start), int(width), int(height),
                    float(half_screen_w), float(half_screen_h),
                    int(max_bounces),
                    int(tile_start), float(jx), float(jy),
                    rs_ro, rs_rd, rs_acc, rs_sca, rs_int)

                active = torch.arange(tn, dtype=i32, device=device)
                it = 0
                while active.numel() > 0 and it < max_iters:
                    na = int(active.numel())
                    wf_traverse_triangle_knots(
                        active, na, t_nodes, t_node_miss, t_leaf_prim,
                        t_leaf_tspan,
                        int(t_first_leaf), tc.knot_val, tc.knot_base,
                        tc.sched_id,
                        tc.sched_seg, tc.sched_z, tc.sched_nknots,
                        float(layer_offset_triangles),
                        int(time_start), int(width), int(height),
                        int(tile_start),
                        rs_ro, rs_rd, rs_sca, rs_int,
                        rs_kt, rs_kl, rs_ka, rs_kb, rs_kp, rs_kf)
                    wf_shade_triangle(
                        active, na, t_nodes, t_node_miss, t_leaf_prim,
                        t_leaf_tspan, int(t_first_leaf),
                        dummy_pos, tri_norm, tri_extra, tri_colors,
                        tri_uvs, tri_tex_meta, textures,
                        num_colored_triangles,
                        int(time_start), int(width), int(height),
                        int(tile_start),
                        rs_ro, rs_rd, rs_acc, rs_sca, rs_int,
                        rs_kt, rs_kl, rs_ka, rs_kb, rs_kp, rs_kf)
                    active = (rs_int[:, 2] == 0).nonzero(
                        as_tuple=True)[0].to(i32)
                    it += 1

                if do_aa:
                    wf_composite_aa(
                        int(time_start), int(width), int(height),
                        1 if transparent else 0, int(tile_start),
                        rs_acc, rs_sca, out, aa_accum)
                else:
                    wf_composite(
                        int(time_start), int(width), int(height),
                        1 if transparent else 0, int(tile_start),
                        rs_acc, rs_sca, t_val, float(TONEMAP_EXPOSURE), out)
                memory.set_pointers(state_ptrs)

    if do_aa:
        wf_finalize_aa(int(width), int(height),
                       1 if transparent else 0,
                       float(inv_aa * inv_aa), t_val, float(TONEMAP_EXPOSURE), aa_accum, out)


def render_general_wavefront(
        tri_bvh, pn_bvh, bez_bvh, merged,
        cam_origin, screen_point, pixel_basis_x, pixel_basis_y,
        pixel_world_scale, time_start, time_end, width, height,
        half_screen_w, half_screen_h, layer_offset_triangles, layer_offset_pn,
        has_tri, has_pn, has_bez, max_bounces,
        light_pos, light_col, num_lights, frag_flag, shadow_flag,
        refraction_flag, transparent, memory, out, aa_level=1):
    """Wavefront orchestration for the general (triangle + PN + bezier) case:
    byte-identical to ``render_scene_stbvh`` but stage-split over per-ray global
    state, with PyTorch ray compaction between host iterations. State carries a
    5th scalar (base_dist) for bezier border widths across bounces.

    ``frag_flag``/``shadow_flag`` select the deterministic per-fragment shading
    and binary hard-shadow paths (compile-time templates of the shade kernel,
    matching ``render_scene_stbvh``); ``light_pos``/``light_col`` feed both.

    ``refraction_flag`` enables simultaneous reflection + refraction (glass): the
    shade kernel SPLITS such a ray, continuing the reflected branch in place and
    spawning the refracted branch into a free pool slot. The pool is therefore
    over-allocated by ``split_k`` (only when refraction is on) -- it holds
    ``primary_per_tile`` one-per-pixel rays plus spare slots for split branches,
    at fixed total memory (fewer pixels per tile instead of bigger per-ray
    state). Each ray commits into a shared per-pixel accumulator (``pix_accum``)
    on termination, so a pixel's reflected and refracted branches sum.

    When ``aa_level > 1``, runs ``aa^2`` sub-pixel passes at the output
    resolution (in-place MSAA), accumulating into a float buffer and averaging
    at the end -- no super-sampled frame buffer needed."""
    device = out.device
    t_val = _get_tonemap_t_val()
    i32 = torch.int32
    f32 = torch.float32
    max_iters = MAX_SURFACES_PER_RAY + max_bounces * 2 + 4
    n = (time_end - time_start) * width * height
    aa = max(1, int(aa_level))
    do_aa = aa > 1
    inv_aa = 1.0 / aa

    # Pool over-allocation for ray splitting. Only glass (reflective+refractive)
    # surfaces split, so spare slots are reserved only when refraction is on; the
    # non-refractive path keeps split_k == 1 (one slot per pixel, as before).
    split_k = REFRACT_SPLIT_SLOTS if refraction_flag else 1
    primary_per_tile = max(1, WAVEFRONT_TILE_RAYS // split_k)
    # Deferred shadows: trace shadow rays in a separate lean kernel and have the
    # shade kernel read packed visibility bits (lower shade register pressure).
    # Only active when the toggle is on AND this render actually casts shadows.
    deferred_sh = 1 if (DEFER_WF_SHADOWS and shadow_flag) else 0

    aa_accum = None
    if do_aa:
        aa_accum = memory.get_tensor((n, 5 if transparent else 4), f32)
        aa_accum.zero_()

    for si in range(aa):
        for sj in range(aa):
            jx = (si + 0.5) * inv_aa if do_aa else 0.5
            jy = (sj + 0.5) * inv_aa if do_aa else 0.5

            # Ray-offset screen tiling (like render_triangles_wavefront):
            # process the chunk's pixels in bounded tiles, so per-tile state
            # stays bounded *regardless of frame size* -- a single UHD frame
            # just splits into several tiles. ``tile_start`` is the first
            # pixel's global ray index; the pool slot r (< primary count)
            # renders pixel ``tile_start + r``. rs_sca has a 5th column
            # (base_dist). State is pool-allocated and freed after every tile.
            for tile_start in range(0, n, primary_per_tile):
                tn_primary = min(primary_per_tile, n - tile_start)
                pool = tn_primary * split_k
                state_ptrs = memory.get_pointers()
                (rs_ro, rs_rd, rs_acc, rs_sca, rs_int,
                 rs_kt, rs_kl, rs_ka, rs_kb, rs_kp, rs_kf) = \
                    _alloc_wavefront_state(memory, pool, 5)
                rs_pix = memory.get_tensor((pool,), i32)
                pix_accum = memory.get_tensor((tn_primary, 5), f32)
                # Per-pixel spare-slot counter (zeroed by wf_gen_general): a
                # split ray bumps rs_used[its pixel], so distinct pixels touch
                # distinct addresses -- no single global atomic to serialise on.
                rs_used = memory.get_tensor((tn_primary,), i32)
                # Packed per-ray shadow visibility bits (deferred shadows only);
                # a 1-element placeholder otherwise (the reader compiles out).
                rs_vis = memory.get_tensor(
                    (pool if deferred_sh else 1,), i32)

                wf_gen_general(
                    cam_origin, screen_point, pixel_basis_x, pixel_basis_y,
                    int(time_start), int(width), int(height),
                    float(half_screen_w), float(half_screen_h),
                    int(max_bounces),
                    int(tile_start), int(tn_primary), float(jx), float(jy),
                    rs_ro, rs_rd, rs_acc, rs_sca, rs_int,
                    rs_pix, pix_accum, rs_used)

                active = torch.arange(tn_primary, dtype=i32, device=device)
                it = 0
                while active.numel() > 0 and it < max_iters:
                    na = int(active.numel())
                    wf_traverse_general(
                        active, na,
                        tri_bvh.nodes, tri_bvh.node_miss, tri_bvh.leaf_prim,
                        tri_bvh.leaf_tspan, int(tri_bvh.first_leaf),
                        merged["tri_pos"],
                        pn_bvh.nodes, pn_bvh.node_miss, pn_bvh.leaf_prim,
                        pn_bvh.leaf_tspan, int(pn_bvh.first_leaf),
                        merged["pn_ctrl"],
                        merged["pn_obb"],
                        bez_bvh.nodes, bez_bvh.node_miss, bez_bvh.leaf_prim,
                        bez_bvh.leaf_tspan, int(bez_bvh.first_leaf),
                        merged["circuit_meta"],
                        merged["edges_2d"], merged["edge_offsets"],
                        pixel_world_scale,
                        float(layer_offset_triangles), float(layer_offset_pn),
                        int(has_tri), int(has_pn), int(has_bez),
                        int(time_start), int(width), int(height),
                        int(tile_start),
                        rs_ro, rs_rd, rs_sca, rs_int,
                        rs_kt, rs_kl, rs_ka, rs_kb, rs_kp, rs_kf, rs_pix)
                    if deferred_sh:
                        wf_shadow_general(
                            active, na,
                            tri_bvh.nodes, tri_bvh.node_miss,
                            tri_bvh.leaf_prim, tri_bvh.leaf_tspan,
                            int(tri_bvh.first_leaf),
                            merged["tri_pos"], merged["tri_norm"],
                            merged["tri_colors"], merged["tri_uvs"],
                            merged["tri_tex_meta"], merged["textures"],
                            int(merged["num_colored_triangles"]),
                            pn_bvh.nodes, pn_bvh.node_miss, pn_bvh.leaf_prim,
                            pn_bvh.leaf_tspan, int(pn_bvh.first_leaf),
                            merged["pn_ctrl"], merged["pn_norm"],
                            merged["pn_extra"],
                            merged["pn_colors"], merged["pn_obb"],
                            bez_bvh.nodes, bez_bvh.node_miss,
                            bez_bvh.leaf_prim, bez_bvh.leaf_tspan,
                            int(bez_bvh.first_leaf),
                            merged["circuit_meta"], merged["circuit_colors"],
                            merged["circuit_border_colors"],
                            merged["edges_2d"], merged["edge_offsets"],
                            pixel_world_scale,
                            float(layer_offset_triangles),
                            float(layer_offset_pn),
                            int(has_tri), int(has_pn), int(has_bez),
                            light_pos, int(num_lights),
                            int(time_start), int(width), int(height),
                            int(tile_start),
                            rs_ro, rs_rd, rs_sca, rs_int,
                            rs_kt, rs_ka, rs_kb, rs_kp, rs_kf, rs_pix,
                            rs_vis)
                    wf_shade_general(
                        active, na,
                        tri_bvh.nodes, tri_bvh.node_miss, tri_bvh.leaf_prim,
                        tri_bvh.leaf_tspan, int(tri_bvh.first_leaf),
                        merged["tri_pos"], merged["tri_norm"],
                        merged["tri_extra"],
                        merged["tri_colors"], merged["tri_uvs"],
                        merged["tri_tex_meta"],
                        merged["textures"],
                        int(merged["num_colored_triangles"]),
                        pn_bvh.nodes, pn_bvh.node_miss, pn_bvh.leaf_prim,
                        pn_bvh.leaf_tspan, int(pn_bvh.first_leaf),
                        merged["pn_ctrl"], merged["pn_norm"],
                        merged["pn_extra"],
                        merged["pn_colors"], merged["pn_obb"],
                        bez_bvh.nodes, bez_bvh.node_miss, bez_bvh.leaf_prim,
                        bez_bvh.leaf_tspan, int(bez_bvh.first_leaf),
                        merged["circuit_meta"], merged["circuit_colors"],
                        merged["circuit_border_colors"],
                        merged["edges_2d"], merged["edge_offsets"],
                        pixel_world_scale,
                        float(layer_offset_triangles), float(layer_offset_pn),
                        int(frag_flag), int(shadow_flag),
                        int(refraction_flag),
                        int(has_tri), int(has_pn), int(has_bez),
                        int(deferred_sh),
                        merged["tri_mat_id"], merged["tri_mat"],
                        merged["pn_mat_id"], merged["pn_mat"],
                        light_pos, light_col, int(num_lights),
                        int(time_start), int(width), int(height),
                        int(tile_start),
                        rs_ro, rs_rd, rs_acc, rs_sca, rs_int,
                        rs_kt, rs_kl, rs_ka, rs_kb, rs_kp, rs_kf,
                        rs_pix, pix_accum, rs_used, rs_vis)
                    active = (rs_int[:, 2] == 0).nonzero(
                        as_tuple=True)[0].to(i32)
                    it += 1

                if do_aa:
                    wf_composite_accum_aa(
                        int(time_start), int(width), int(height),
                        1 if transparent else 0, int(tile_start),
                        pix_accum, out, aa_accum)
                else:
                    wf_composite_accum(
                        int(time_start), int(width), int(height),
                        1 if transparent else 0, int(tile_start),
                        pix_accum, t_val, float(TONEMAP_EXPOSURE), out)
                # Release this tile's state back to the pool before the next
                # tile.
                memory.set_pointers(state_ptrs)

    if do_aa:
        wf_finalize_aa(int(width), int(height),
                       1 if transparent else 0,
                       float(inv_aa * inv_aa), t_val, float(TONEMAP_EXPOSURE), aa_accum, out)


# Pixels per PyTorch shading block. The deferred shade pass runs over the
# G-buffer in fixed-size blocks so its temporaries stay small and bounded
# (independent of frame count), which matters on memory-tight GPUs where the
# render arena already reserves most of VRAM.
GBUFFER_SHADE_BLOCK = 1 << 18  # 262144 pixels


def render_gbuffer_general(
        merged, memory, cam_origin, screen_point, pixel_basis_x,
        pixel_basis_y, pixel_world_scale, time_start, time_end, width, height,
        half_screen_w, half_screen_h, layer_offset_triangles, layer_offset_pn,
        light_pos, light_col, num_lights, transparent, out):
    """Deferred-shading (G-buffer) prototype host driver.

    The trace kernel writes each primary ray's nearest-hit surface attributes
    into a per-pixel G-buffer; PyTorch then material-shades the screen (reusing
    ``_shade_fragment``'s math) in bounded pixel-blocks and composites into
    ``out``. Drop-in alternative to the megakernel's in-kernel fragment shading
    for the all-opaque deterministic path -- see
    :mod:`algan.rendering.raytracing.gbuffer_taichi`. Prototype: nearest opaque
    hit only (no transparent-layer compositing, mirror bounces or shadows).

    The G-buffer is allocated from the render memory pool (``memory``) rather
    than as fresh ``torch.empty`` tensors: the pool's arena is pre-reserved, so
    this adds no new allocation (leaving VRAM free for the Taichi launch) and a
    pool overflow raises ``InsufficientMemoryException``, hooking into the
    caller's existing frame-splitting OOM handler.
    """
    device = out.device
    n = (time_end - time_start) * width * height
    pixels_per_frame = width * height
    gb_f32 = memory.get_tensor((n, 16), torch.float32)
    gb_i32 = memory.get_tensor((n, 3), torch.int32)

    tri_bvh = merged["tri_bvh"]
    pn_bvh = merged["pn_bvh"]
    bez_bvh = merged["bez_bvh"]
    gbuffer_nearest_general(
        tri_bvh.nodes, tri_bvh.node_miss, tri_bvh.leaf_prim,
        tri_bvh.leaf_tspan, int(tri_bvh.first_leaf),
        merged["tri_pos"], merged["tri_norm"], merged["tri_colors"],
        merged["tri_uvs"], merged["tri_tex_meta"], merged["textures"],
        int(merged["num_colored_triangles"]),
        pn_bvh.nodes, pn_bvh.node_miss, pn_bvh.leaf_prim, pn_bvh.leaf_tspan,
        int(pn_bvh.first_leaf),
        merged["pn_ctrl"], merged["pn_norm"], merged["pn_colors"],
        merged["pn_obb"],
        bez_bvh.nodes, bez_bvh.node_miss, bez_bvh.leaf_prim, bez_bvh.leaf_tspan,
        int(bez_bvh.first_leaf),
        merged["circuit_meta"], merged["circuit_colors"],
        merged["circuit_border_colors"], merged["edges_2d"],
        merged["edge_offsets"],
        cam_origin, screen_point, pixel_basis_x, pixel_basis_y,
        pixel_world_scale,
        int(time_start), int(width), int(height),
        float(half_screen_w), float(half_screen_h),
        float(layer_offset_triangles), float(layer_offset_pn),
        gb_f32, gb_i32)

    out_v = out.view(n, -1)
    C = out_v.shape[1]
    k = min(4, C)
    mat_default = torch.tensor(_MAT_DEFAULTS, device=device,
                               dtype=torch.float32).view(1, MAT_W)

    for off in range(0, n, GBUFFER_SHADE_BLOCK):
        hi = min(off + GBUFFER_SHADE_BLOCK, n)
        bs = hi - off
        gi = gb_i32[off:hi]
        valid = gi[:, 0] == 1
        prim = gi[:, 1].long()
        htype = gi[:, 2]
        f_idx = (torch.arange(off, hi, device=device) // pixels_per_frame
                 + int(time_start))

        # Per-pixel material block (mat_id + 12 slots) gathered by hit type;
        # bezier hits and misses keep the unlit default (id 1), as the
        # megakernel leaves them.
        mat = mat_default.repeat(bs, 1)
        mat_id = torch.ones(bs, dtype=torch.long, device=device)
        for type_id, id_key, mat_key in ((1, "tri_mat_id", "tri_mat"),
                                         (2, "pn_mat_id", "pn_mat")):
            mask = valid & (htype == type_id)
            if not bool(mask.any()):
                continue
            mat_arr = merged[mat_key]    # [Tm, N, MAT_W]
            id_arr = merged[id_key]      # [Tmid, N]
            pm = prim[mask]
            fm = f_idx[mask]
            mat[mask] = mat_arr[fm % mat_arr.shape[0], pm].float()
            mat_id[mask] = id_arr[fm % id_arr.shape[0], pm].long()

        shaded = shade_gbuffer_torch(gb_f32[off:hi], mat, mat_id, f_idx,
                                     light_pos, light_col, int(num_lights))
        if TONEMAPPING and not POST_PROCESS_TONEMAP:
            color_exposed = shaded[:, :3] * TONEMAP_EXPOSURE
            if TONEMAP_METHOD == "neutral":
                # Khronos PBR Neutral
                x, _ = torch.min(color_exposed, dim=1, keepdim=True)
                offset = torch.where(x < 0.08, x - 6.25 * x * x, torch.tensor(0.04, device=device))
                color_offset = color_exposed - offset
                
                peak, _ = torch.max(color_offset, dim=1, keepdim=True)
                mask_compress = (peak >= 0.76).squeeze(1)
                
                if mask_compress.any():
                    color_c = color_offset[mask_compress]
                    peak_c = peak[mask_compress]
                    
                    d = 0.24
                    newPeak = 1.0 - d * d / (peak_c + d - 0.76)
                    color_c *= newPeak / peak_c
                    
                    g = 1.0 - 1.0 / (0.15 * (peak_c - newPeak) + 1.0)
                    color_offset[mask_compress] = color_c + g * (newPeak - color_c)
                
                shaded[:, :3] = torch.clamp(color_offset, 0.0, 1.0)
            elif TONEMAP_METHOD == "agx":
                # AgX
                r_rec2020 = 0.627409 * color_exposed[:, 0] + 0.329282 * color_exposed[:, 1] + 0.043309 * color_exposed[:, 2]
                g_rec2020 = 0.069055 * color_exposed[:, 0] + 0.919540 * color_exposed[:, 1] + 0.011405 * color_exposed[:, 2]
                b_rec2020 = 0.016390 * color_exposed[:, 0] + 0.088013 * color_exposed[:, 1] + 0.895597 * color_exposed[:, 2]
                
                r_inset = 0.856627153315983 * r_rec2020 + 0.0951212405381588 * g_rec2020 + 0.0482516061458583 * b_rec2020
                g_inset = 0.137318972929847 * r_rec2020 + 0.761241990602591 * g_rec2020 + 0.101439036467562 * b_rec2020
                b_inset = 0.11189821299995 * r_rec2020 + 0.0767994186031903 * g_rec2020 + 0.811302368396859 * b_rec2020
                
                r_log = torch.clamp(torch.log2(torch.clamp(r_inset, min=1e-10)), -12.47393, 4.026069)
                g_log = torch.clamp(torch.log2(torch.clamp(g_inset, min=1e-10)), -12.47393, 4.026069)
                b_log = torch.clamp(torch.log2(torch.clamp(b_inset, min=1e-10)), -12.47393, 4.026069)
                
                r_norm = (r_log - (-12.47393)) / (4.026069 - (-12.47393))
                g_norm = (g_log - (-12.47393)) / (4.026069 - (-12.47393))
                b_norm = (b_log - (-12.47393)) / (4.026069 - (-12.47393))
                
                def agx_curve(x):
                    x2 = x * x
                    x4 = x2 * x2
                    return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232
                    
                r_curve = agx_curve(r_norm)
                g_curve = agx_curve(g_norm)
                b_curve = agx_curve(b_norm)
                
                r_out = 1.1271005818144368 * r_curve - 0.11060664309660323 * g_curve - 0.016493938717834573 * b_curve
                g_out = -0.1413297634984383 * r_curve + 1.157823702216272 * g_curve - 0.016493938717834257 * b_curve
                b_out = -0.14132976349843826 * r_curve - 0.11060664309660294 * g_curve + 1.2519364065950405 * b_curve
                
                r_srgb = 1.6605 * r_out - 0.1246 * g_out - 0.0182 * b_out
                g_srgb = -0.5876 * r_out + 1.1329 * g_out - 0.1006 * b_out
                b_srgb = -0.0728 * r_out - 0.0083 * g_out + 1.1187 * b_out
                
                shaded[:, 0] = torch.clamp(r_srgb, 0.0, 1.0)
                shaded[:, 1] = torch.clamp(g_srgb, 0.0, 1.0)
                shaded[:, 2] = torch.clamp(b_srgb, 0.0, 1.0)
        if POST_PROCESS_TONEMAP:
            px = (shaded * 255.0).clamp_min(0.0)
            block = out_v[off:hi]
            block[valid, :k] = px[valid, :k].to(block.dtype)
            if transparent and C >= 5:
                block[valid, 4] = torch.tensor(255.0, dtype=block.dtype,
                                               device=device)
        else:
            px = (shaded * 255.0 + 0.5).clamp(0.0, 255.0).to(torch.uint8)
            block = out_v[off:hi]
            block[valid, :k] = px[valid, :k]
            if transparent and C >= 5:
                block[valid, 4] = torch.tensor(255, dtype=torch.uint8,
                                               device=device)


# Pixels per PyTorch shading block for the deferred wavefront path.
GBUFFER_WF_SHADE_BLOCK = 1 << 18


def render_gbuffer_wavefront_general(
        merged, memory, cam_origin, screen_point, pixel_basis_x,
        pixel_basis_y, pixel_world_scale, time_start, time_end, width, height,
        half_screen_w, half_screen_h, layer_offset_triangles, layer_offset_pn,
        has_tri, has_pn, has_bez, max_bounces,
        light_pos, light_col, num_lights, transparent, out):
    """Deferred wavefront with transparency + reflections.

    Stage-split ping-pong (gen -> (traverse -> drain-record -> PyTorch shade ->
    compact)* -> composite): the trace/drain/bounce state machine runs in Taichi
    exactly as the megakernel, but per-hit material shading is deferred to a
    PyTorch pass over a per-ray G-buffer (see
    :mod:`algan.rendering.raytracing.gbuffer_taichi`). All per-ray state and the
    G-buffer are allocated from the render pool so the caller's frame-splitting
    OOM handler bounds memory. No shadows or refractions.
    """
    device = out.device
    t_val = _get_tonemap_t_val()
    n = (time_end - time_start) * width * height
    f32 = torch.float32
    i32 = torch.int32
    g = memory.get_tensor
    rs_ro = g((n, 3), f32)
    rs_rd = g((n, 3), f32)
    rs_acc = g((n, 4), f32)
    rs_sca = g((n, 5), f32)
    rs_int = g((n, 4), i32)
    rs_kt = g((n, KBUF), f32)
    rs_kl = g((n, KBUF), f32)
    rs_ka = g((n, KBUF), f32)
    rs_kb = g((n, KBUF), f32)
    rs_kp = g((n, KBUF), i32)
    rs_kf = g((n, KBUF), i32)
    gb_f32 = g((n, KBUF, GB_HIT_W), f32)
    gb_i32 = g((n, KBUF, 2), i32)
    gb_count = g((n,), i32)

    tri_bvh = merged["tri_bvh"]
    pn_bvh = merged["pn_bvh"]
    bez_bvh = merged["bez_bvh"]

    wf_gen_general(
        cam_origin, screen_point, pixel_basis_x, pixel_basis_y,
        int(time_start), int(width), int(height),
        float(half_screen_w), float(half_screen_h), int(max_bounces),
        rs_ro, rs_rd, rs_acc, rs_sca, rs_int)

    active = torch.arange(n, dtype=i32, device=device)
    max_iters = MAX_SURFACES_PER_RAY + max_bounces + 2
    it = 0
    while active.numel() > 0 and it < max_iters:
        na = int(active.numel())
        wf_traverse_gbuffer(
            active, na,
            tri_bvh.nodes, tri_bvh.node_miss, tri_bvh.leaf_prim,
            tri_bvh.leaf_tspan, int(tri_bvh.first_leaf), merged["tri_pos"],
            pn_bvh.nodes, pn_bvh.node_miss, pn_bvh.leaf_prim,
            pn_bvh.leaf_tspan, int(pn_bvh.first_leaf), merged["pn_ctrl"],
            merged["pn_obb"],
            bez_bvh.nodes, bez_bvh.node_miss, bez_bvh.leaf_prim,
            bez_bvh.leaf_tspan, int(bez_bvh.first_leaf),
            merged["circuit_meta"], merged["edges_2d"], merged["edge_offsets"],
            pixel_world_scale,
            float(layer_offset_triangles), float(layer_offset_pn),
            int(has_tri), int(has_pn), int(has_bez),
            int(time_start), int(width), int(height),
            rs_ro, rs_rd, rs_sca, rs_int,
            rs_kt, rs_kl, rs_ka, rs_kb, rs_kp, rs_kf)
        wf_drain_record_gbuffer(
            active, na,
            merged["tri_pos"], merged["tri_norm"], merged["tri_extra"],
            merged["tri_colors"], merged["tri_uvs"], merged["tri_tex_meta"],
            merged["textures"], int(merged["num_colored_triangles"]),
            merged["pn_ctrl"], merged["pn_norm"], merged["pn_extra"],
            merged["pn_colors"],
            merged["circuit_meta"], merged["circuit_colors"],
            merged["circuit_border_colors"],
            int(time_start), int(width), int(height),
            rs_ro, rs_rd, rs_sca, rs_int,
            rs_kt, rs_kl, rs_ka, rs_kb, rs_kp, rs_kf,
            gb_f32, gb_i32, gb_count)
        shade_accumulate_wavefront(
            active, gb_f32, gb_i32, gb_count, rs_acc, merged,
            light_pos, light_col, int(num_lights), width * height,
            int(time_start), GBUFFER_WF_SHADE_BLOCK)
        active = (rs_int[:, 2] == 0).nonzero(as_tuple=True)[0].to(i32)
        it += 1

    wf_composite(int(time_start), int(width), int(height),
                 1 if transparent else 0, 0, rs_acc, rs_sca,
                 t_val, float(TONEMAP_EXPOSURE), out)


def render_batch_ray_traced(primitives, scene, screen_width, screen_height,
                            time_start, time_end, background_color,
                            transparent_background, ray_origin, screen_point,
                            screen_basis, anti_alias_level=1, light_sources=(),
                            memory=None, post_processes=(), **kwargs):
    """Render frames [time_start, time_end) of a primitive batch by ray
    tracing into a fixed [frames, pixels, channels] buffer.

    On out-of-memory the time window is halved and retried; per-frame memory
    is just the output buffer (plus post-processing), independent of scene
    depth complexity or bounce count.
    """
    # Lazily allocate module-level Taichi fields (the glow toggle) against the
    # live runtime, before any kernel launches -- so a runtime re-init before
    # the first render (e.g. the profiler enabling kernel_profiler) is safe.
    _ensure_globals()
    merged = _merge_scene(primitives)
    aa = max(1, int(anti_alias_level))
    # Refraction is only implemented by the general wavefront tracer, so a
    # deterministic batch that contains a refractive surface is routed there
    # regardless of USE_WAVEFRONT (the megakernel / Monte Carlo paths ignore the
    # refractive index). Computed before the AA strategy because it, like
    # USE_WAVEFRONT, forces the super-sampled (non-in-place) AA path.
    refractive_det = (bool(merged.get("has_refractive"))
                      and not bool(PHYSICAL_LIGHTING)
                      and int(SAMPLES_PER_PIXEL) <= 1)
    # Texture-mapped material properties (reflectivity/roughness/IOR/normal
    # maps) are likewise only sampled by the general wavefront tracer, so a
    # deterministic batch containing them is routed there too. The megakernel
    # and Monte Carlo paths fall back to the per-vertex values.
    # ``has_pn_textures`` folds in ANY textured PN patch (color included):
    # unlike flat colour maps the megakernel cannot sample a PN texture (no PN
    # UVs there), so any textured PN must render through the general wavefront.
    mat_tex_det = (bool(merged.get("has_material_textures"))
                   or bool(merged.get("has_pn_textures"))
                   ) and not bool(PHYSICAL_LIGHTING) and int(SAMPLES_PER_PIXEL) <= 1
    use_wavefront = USE_WAVEFRONT or refractive_det or mat_tex_det
    # Anti-aliasing strategy. All deterministic renderers (megakernels and
    # wavefront) average ``aa^2`` jittered sub-pixel rays *in place* at the
    # output resolution, so the frame buffer stays ``screen_width x
    # screen_height`` regardless of ``aa`` (aa^2× less render memory than
    # super-sampling). The wavefront path runs the full gen→traverse→shade→
    # compact→composite pipeline once per sub-pixel sample, accumulating into
    # a float buffer and averaging at the end.
    inplace_aa = INPLACE_AA
    if inplace_aa:
        width = screen_width
        height = screen_height
        kernel_aa = aa  # in-kernel sub-pixel averaging factor
        post_aa = 1  # post-processing does not down-sample
    else:
        width = screen_width * aa
        height = screen_height * aa
        kernel_aa = 1
        post_aa = aa

    C_out = 5 if transparent_background else 4
    device = COMPUTING_DEFAULTS.render_device
    num_frames = merged["num_frames"]

    cam_origin = _expand_frames(_flat_frames(ray_origin, (3,)),
                                num_frames).contiguous()
    sp = _expand_frames(_flat_frames(screen_point, (3,)),
                        num_frames).contiguous()
    sb = _expand_frames(_flat_frames(screen_basis, (3, 3)), num_frames)
    pbx, pby = _pixel_bases(sb)
    # World units per screen pixel per unit distance (for border widths). Border
    # widths are authored in *anti-aliased* pixels (see BezierCircuit), so this
    # always uses the super-sampled height (screen_height * aa), whether or not
    # the frame buffer itself is super-sampled.
    b1_norm = sb[:, 1].norm(p=2, dim=-1)
    screen_dist = (sp - cam_origin).norm(p=2, dim=-1)
    pixel_world_scale = (
        2.0 / (screen_height * aa * b1_norm * screen_dist).clamp_min(1e-12)
    ).contiguous()

    # In-place AA samples the background once per output pixel, so an
    # animated/image background that arrived super-sampled must be averaged
    # down to the output resolution first (solid colors are resolution-free).
    if aa > 1:
        background_color = _downsample_background(
            background_color, aa, time_end - time_start,
            screen_height, screen_width)

    tri_bvh = merged["tri_bvh"]
    pn_bvh = merged["pn_bvh"]
    bez_bvh = merged["bez_bvh"]
    # A geometry type absent from the whole batch has only a placeholder BVH;
    # tell the deterministic kernel so it skips that empty traversal per ray.
    if GATE_EMPTY_TRAVERSALS:
        has_tri = 1 if merged["num_triangles"] > 0 else 0
        has_pn = 1 if merged["num_pn"] > 0 else 0
        has_bez = 1 if merged["num_circuits"] > 0 else 0
    else:  # benchmarking escape hatch: traverse every (possibly empty) tree
        has_tri = has_pn = has_bez = 1
    first = primitives[0]
    first.memory = memory
    t_val = _get_tonemap_t_val()

    samples = max(1, int(SAMPLES_PER_PIXEL))
    # In-place AA folds the anti-alias super-sampling into the Monte Carlo
    # sample count: each of the ``aa^2`` sub-pixels would have drawn ``samples``
    # random rays jittered over its own cell and then been averaged down, which
    # is equivalent (same total, same expectation) to drawing ``samples * aa^2``
    # rays jittered over the whole output pixel. (The wavefront/super-sample
    # path keeps ``kernel_aa == 1``, so ``samples_eff == samples`` there.)
    samples_eff = samples * (kernel_aa * kernel_aa)
    physical = bool(PHYSICAL_LIGHTING)
    if physical and samples <= 1:
        raise ValueError(
            "Physical lighting is a Monte Carlo mode; call "
            "set_samples_per_pixel(n) with n > 1 (e.g. 32) to use it.")
    # Deterministic per-fragment shading is active for a single-sample,
    # non-physical render with the toggle on; it needs the scene's point lights
    # in the kernel. (Physical mode packs the same lights for its own path.)
    # Deterministic hard shadows are evaluated inside the per-fragment lighting
    # model, so enabling them implies fragment shading for this render.
    det_shadows = bool(SHADOWS) and not physical and samples <= 1
    det_frag = ((bool(FRAGMENT_SHADING) or det_shadows)
                and not physical and samples <= 1)
    frag_flag = 1 if det_frag else 0
    shadow_flag = 1 if det_shadows else 0
    # Refraction (general wavefront only; see refractive_det above).
    refraction_flag = 1 if refractive_det else 0
    if physical or det_frag:
        light_pos, light_col, num_lights = _pack_lights(
            light_sources, num_frames, device)
    elif samples > 1:
        light_pos = light_col = None
        num_lights = 0
    else:
        # Deterministic, fragment shading off: tiny placeholders for the
        # (compiled-out) material/light kernel args.
        light_pos = torch.zeros((1, 1, 3), device=device)
        light_col = torch.zeros((1, 1, 3), device=device)
        num_lights = 0

    def render_chunk(start, end):
        # The Monte Carlo kernels launch one thread per (frame, pixel,
        # sample) path; keep the flattened index within int32 range. (The
        # deterministic kernels loop the aa^2 sub-pixels serially per pixel, so
        # only the Monte Carlo path multiplies the thread count by the samples.)
        if (samples > 1 and
                (end - start) * width * height * samples_eff >= 1 << 31):
            print(f'Render OOM, splitting {start}:{end}')
            if end - start <= 1:
                raise OutOfRenderMemory(
                    "samples_per_pixel * resolution exceeds the ray tracer's "
                    "per-launch path budget (2^31). Please lower the sample "
                    "count, resolution or anti-alias level.")
            middle = (start + end) // 2
            return render_chunk(start, middle) + render_chunk(middle, end)
        entry_pointers = memory.get_pointers()
        try:
            out_dtype = torch.float32 if is_post_process_tonemap_enabled() else torch.uint8
            out = memory.get_tensor((end - start, width * height, C_out),
                                    out_dtype)
            _prefill_background(out, background_color, start - time_start,
                                device)
            accum = None
            if physical or samples > 1:
                # f32 per-pixel sample sums, averaged by finalize_samples.
                accum = memory.get_tensor((end - start, width * height, 5),
                                          torch.float32)
                accum.zero_()
            # Coplanar layer order: circuits < triangles < PN patches.
            layer_offset_triangles = float(merged["num_circuits"])
            layer_offset_pn = layer_offset_triangles + float(
                merged["num_triangles"])
            shared_args = (
                tri_bvh.nodes, tri_bvh.node_miss, tri_bvh.leaf_prim,
                tri_bvh.leaf_tspan, tri_bvh.first_leaf,
                merged["tri_pos"], merged["tri_norm"], merged["tri_extra"],
                merged["tri_colors"], merged["tri_uvs"], merged["tri_tex_meta"],
                merged["textures"], int(merged["num_colored_triangles"]),
                pn_bvh.nodes, pn_bvh.node_miss, pn_bvh.leaf_prim,
                pn_bvh.leaf_tspan, pn_bvh.first_leaf,
                merged["pn_ctrl"], merged["pn_norm"], merged["pn_extra"],
                merged["pn_colors"],
                bez_bvh.nodes, bez_bvh.node_miss, bez_bvh.leaf_prim,
                bez_bvh.leaf_tspan, bez_bvh.first_leaf,
                merged["circuit_meta"], merged["circuit_colors"],
                merged["circuit_border_colors"], merged["edges_2d"],
                merged["edge_offsets"],
                cam_origin, sp, pbx, pby, pixel_world_scale,
                int(start), int(end), int(width), int(height),
                float(width // 2), float(height // 2),
                layer_offset_triangles, layer_offset_pn, int(MAX_BOUNCES),
                1 if transparent_background else 0)
            if physical:
                path_trace_physical_stbvh(
                    *shared_args, samples_eff, light_pos, light_col,
                    int(num_lights), float(LIGHT_INTENSITY),
                    float(AMBIENT_LIGHT), merged["pn_obb"], out, accum)
                finalize_samples(samples_eff,
                                 1 if transparent_background else 0,
                                 t_val, float(TONEMAP_EXPOSURE),
                                 accum, out)
            elif samples > 1:
                path_trace_scene_stbvh(*shared_args, samples_eff,
                                       float(INDIRECT_BOUNCE_STRENGTH),
                                       merged["pn_obb"], out, accum)
                finalize_samples(samples_eff,
                                 1 if transparent_background else 0,
                                 t_val, float(TONEMAP_EXPOSURE),
                                 accum, out)
            elif (USE_GBUFFER and det_frag and not det_shadows
                  and not refractive_det and not mat_tex_det):
                # Deferred-shading (G-buffer) path: trace/drain in Taichi, shade
                # in PyTorch. Replaces the megakernel's in-kernel fragment
                # shading. "wavefront" supports transparency + reflections;
                # "nearest" is the single-hit opaque prototype. (Refraction is
                # not supported here, so refractive batches use the general
                # wavefront below instead.)
                if GBUFFER_MODE == "nearest":
                    render_gbuffer_general(
                        merged, memory, cam_origin, sp, pbx, pby,
                        pixel_world_scale,
                        int(start), int(end), int(width), int(height),
                        float(width // 2), float(height // 2),
                        layer_offset_triangles, layer_offset_pn,
                        light_pos, light_col, int(num_lights),
                        1 if transparent_background else 0, out)
                else:
                    render_gbuffer_wavefront_general(
                        merged, memory, cam_origin, sp, pbx, pby,
                        pixel_world_scale,
                        int(start), int(end), int(width), int(height),
                        float(width // 2), float(height // 2),
                        layer_offset_triangles, layer_offset_pn,
                        has_tri, has_pn, has_bez, int(MAX_BOUNCES),
                        light_pos, light_col, int(num_lights),
                        1 if transparent_background else 0, out)
            elif ((USE_WAVEFRONT
                   or (WAVEFRONT_MIN_PIXELS
                       and width * height >= WAVEFRONT_MIN_PIXELS))
                  and not det_frag and not det_shadows and not refractive_det
                  and not mat_tex_det
                  and has_tri and not has_pn and not has_bez
                  and merged.get("tri_tc") is None):
                # Triangle-only batch via the (tiled, stage-split) wavefront
                # path: the default for large frames (higher occupancy / lower
                # divergence; ~1.41x wall at HD on high-depth scenes, neutral
                # otherwise). Byte-identical to the megakernel (validated on
                # deterministic scenes incl. high depth complexity). The
                # megakernel handles small frames and the PN/bezier/fragment/
                # shadow cases.
                # byte-identical output, higher occupancy / less divergence.
                # (The wavefront path has no fragment shader, so a fragment
                # shaded render falls through to the megakernel below.)
                render_triangles_wavefront(
                    tri_bvh.nodes, tri_bvh.node_miss, tri_bvh.leaf_prim,
                    tri_bvh.leaf_tspan, tri_bvh.first_leaf,
                    merged["tri_pos"], merged["tri_norm"],
                    merged["tri_extra"], merged["tri_colors"],
                    merged["tri_uvs"], merged["tri_tex_meta"],
                    merged["textures"], int(merged["num_colored_triangles"]),
                    cam_origin, sp, pbx, pby,
                    int(start), int(end), int(width), int(height),
                    float(width // 2), float(height // 2),
                    layer_offset_triangles, int(MAX_BOUNCES),
                    1 if transparent_background else 0, memory, out,
                    kernel_aa)
            elif (use_wavefront and merged.get("tri_tc") is None):
                # General (PN/bezier/mixed) batch via the wavefront path, at
                # full megakernel parity: deterministic per-fragment shading,
                # binary hard shadows and refraction (frag/shadow/refraction
                # flags) when enabled, else the vertex-shaded path. All are
                # compile-time templates of the shade kernel, so the launched
                # kernel matches the corresponding render_scene_stbvh
                # specialization (plus refraction, which the megakernel lacks).
                render_general_wavefront(
                    tri_bvh, pn_bvh, bez_bvh, merged,
                    cam_origin, sp, pbx, pby, pixel_world_scale,
                    int(start), int(end), int(width), int(height),
                    float(width // 2), float(height // 2),
                    layer_offset_triangles, layer_offset_pn,
                    has_tri, has_pn, has_bez, int(MAX_BOUNCES),
                    light_pos, light_col, int(num_lights),
                    frag_flag, shadow_flag, refraction_flag,
                    1 if transparent_background else 0, memory, out,
                    kernel_aa)
            elif (merged.get("tri_tc") is not None and has_tri
                  and not has_pn and not has_bez and not det_shadows
                  and not det_frag and not merged["tri_has_reflective"]):
                # Triangle-only batch with temporally compressed positions:
                # reconstruct each frame's geometry from per-primitive knots
                # in-register so the dense [T, N, 9] position array is never
                # resident. Restricted to the non-reflective, vertex-shaded,
                # shadow-free case the knot kernel covers.
                tc = merged["tri_tc"]
                merged["tri_pos"] = None  # dense positions no longer needed
                if USE_WAVEFRONT:
                    # Knot geometry through the stage-split wavefront path
                    # (register-pressure A/B against the knot megakernel).
                    render_triangles_wavefront_knots(
                        tri_bvh.nodes, tri_bvh.node_miss, tri_bvh.leaf_prim,
                        tri_bvh.leaf_tspan, tri_bvh.first_leaf, tc,
                        merged["tri_norm"], merged["tri_extra"],
                        merged["tri_colors"], merged["tri_uvs"],
                        merged["tri_tex_meta"], merged["textures"],
                        int(merged["num_colored_triangles"]),
                        cam_origin, sp, pbx, pby,
                        int(start), int(end), int(width), int(height),
                        float(width // 2), float(height // 2),
                        layer_offset_triangles, int(MAX_BOUNCES),
                        1 if transparent_background else 0, memory, out,
                        kernel_aa)
                else:
                    render_triangles_knots_stbvh(
                        tri_bvh.nodes, tri_bvh.node_miss, tri_bvh.leaf_prim,
                        tri_bvh.leaf_tspan, tri_bvh.first_leaf,
                        tc.knot_val, tc.knot_base, tc.sched_id, tc.sched_seg,
                        tc.sched_z, tc.sched_nknots,
                        merged["tri_extra"], merged["tri_colors"],
                        merged["tri_uvs"], merged["tri_tex_meta"],
                        merged["textures"], int(merged["num_colored_triangles"]),
                        cam_origin, sp, pbx, pby,
                        int(start), int(end), int(width), int(height),
                        float(width // 2), float(height // 2),
                        layer_offset_triangles, int(MAX_BOUNCES),
                        1 if transparent_background else 0, kernel_aa,
                        t_val, float(TONEMAP_EXPOSURE), out)
            elif (USE_TRIANGLE_ONLY_KERNEL and has_tri
                  and not has_pn and not has_bez and not det_shadows):
                # Triangle-only batch: the lean kernel (no PN/bezier code)
                # gives identical output at lower register pressure. (Shadows
                # live only in the general kernel, so a shadowed render falls
                # through to render_scene_stbvh below.)
                render_triangles_stbvh(
                    tri_bvh.nodes, tri_bvh.node_miss, tri_bvh.leaf_prim,
                    tri_bvh.leaf_tspan, tri_bvh.first_leaf,
                    merged["tri_pos"], merged["tri_norm"],
                    merged["tri_extra"], merged["tri_colors"],
                    merged["tri_uvs"], merged["tri_tex_meta"],
                    merged["textures"], int(merged["num_colored_triangles"]),
                    cam_origin, sp, pbx, pby,
                    int(start), int(end), int(width), int(height),
                    float(width // 2), float(height // 2),
                    layer_offset_triangles, int(MAX_BOUNCES),
                    1 if transparent_background else 0,
                    frag_flag, merged["tri_mat_id"], merged["tri_mat"],
                    light_pos, light_col, int(num_lights), kernel_aa,
                    t_val, float(TONEMAP_EXPOSURE), out)
            elif USE_NO_PN_KERNEL and not has_pn and not det_shadows:
                # No PN patches (bezier circuits, optionally with flat
                # triangles): the no-PN kernel omits the Matrix Pencil solver from
                # its call graph for identical output at lower register
                # pressure. (Pure triangle-only was handled above; a shadowed
                # render uses the general kernel below.)
                render_no_pn_stbvh(
                    tri_bvh.nodes, tri_bvh.node_miss, tri_bvh.leaf_prim,
                    tri_bvh.leaf_tspan, tri_bvh.first_leaf,
                    merged["tri_pos"], merged["tri_norm"],
                    merged["tri_extra"], merged["tri_colors"],
                    merged["tri_uvs"], merged["tri_tex_meta"],
                    merged["textures"], int(merged["num_colored_triangles"]),
                    bez_bvh.nodes, bez_bvh.node_miss, bez_bvh.leaf_prim,
                    bez_bvh.leaf_tspan, bez_bvh.first_leaf,
                    merged["circuit_meta"], merged["circuit_colors"],
                    merged["circuit_border_colors"], merged["edges_2d"],
                    merged["edge_offsets"],
                    cam_origin, sp, pbx, pby, pixel_world_scale,
                    int(start), int(end), int(width), int(height),
                    float(width // 2), float(height // 2),
                    layer_offset_triangles, int(MAX_BOUNCES),
                    1 if transparent_background else 0,
                    has_tri, has_bez,
                    frag_flag, merged["tri_mat_id"], merged["tri_mat"],
                    light_pos, light_col, int(num_lights), kernel_aa,
                    t_val, float(TONEMAP_EXPOSURE), out)
            else:
                render_scene_stbvh(*shared_args, has_tri, has_pn, has_bez,
                                   frag_flag, merged["tri_mat_id"],
                                   merged["tri_mat"], merged["pn_mat_id"],
                                   merged["pn_mat"], light_pos, light_col,
                                   int(num_lights), shadow_flag, kernel_aa,
                                   merged["pn_obb"],
                                   t_val, float(TONEMAP_EXPOSURE), out)
            frames = out.view(end - start, height, width, C_out)
            frames = first.post_process_frames(
                frames, anti_alias_level=post_aa,
                post_processes=list(post_processes))
            memory.set_pointers(entry_pointers)
            return [frames]
        except (InsufficientMemoryException, torch.OutOfMemoryError):
            print(f'Render OOM, splitting {start}:{end}')
            memory.set_pointers(entry_pointers)
            # Release the failed allocation (e.g. the wavefront's large per-ray
            # state) so it doesn't fragment/block the smaller retry.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if end - start <= 1:
                raise OutOfRenderMemory(
                    "Insufficient memory to ray trace a single frame. "
                    "Please lower the resolution or anti-alias level.") from None
            middle = (start + end) // 2
            return render_chunk(start, middle) + render_chunk(middle, end)

    chunks = render_chunk(time_start, time_end)
    if len(chunks) == 1:
        return chunks[0]
    return torch.cat(chunks, 0)


_originals = {}


def enable_ray_tracing(samples_per_pixel=None, indirect_bounce_strength=None,
                       physical_lighting=None, pn_triangles=False,
                       fragment_shading=None, shadows=None,
                       tonemapping=None, tonemap_exposure=None,
                       tonemap_method=None, raytraced_glow=None,
                       post_process_tonemap=None):
    """Route newly created mobs through the ray traced render pipeline.

    Rebinds the primitive classes used by the mob modules; call this before
    constructing the mobs that should be ray traced (bezier mobs bind their
    primitive class at construction time).

    Parameters
    ----------
    samples_per_pixel
        Rays averaged per pixel. 1 (default) renders with the exact
        deterministic kernel; larger values enable the Monte Carlo path
        tracer (see :func:`set_samples_per_pixel`).
    indirect_bounce_strength
        Diffuse indirect lighting strength for the Monte Carlo renderer
        (see :func:`set_indirect_bounce_strength`).
    physical_lighting
        Fully physical mode: skip vertex shading and light the scene with
        the explicit point lights, glow emission and the background
        environment (see :func:`set_physical_lighting`).
    pn_triangles
        Render triangle mobs as curved point-normal (PN) patches --
        quadratic Bezier triangles bent to match the vertex normals -- so
        coarsely tessellated smooth surfaces (spheres, parametric surfaces)
        keep smooth silhouettes. Triangles whose normals are zero or
        constant across the face stay exactly flat.
    fragment_shading
        Shade the core lit materials per fragment in the *deterministic*
        renderer (``samples_per_pixel == 1``, non-physical) instead of baking
        per-vertex colours -- crisper specular highlights and smooth shading on
        coarse meshes (see :func:`set_fragment_shading`). Off by default.
    shadows
        Cast binary hard shadows in the *deterministic* renderer: each shaded
        fragment is darkened where an opaque surface occludes a point light
        (see :func:`set_ray_traced_shadows`). Implies ``fragment_shading``. For
        soft or transmissive shadows use physical lighting instead. Off by
        default.
    tonemapping
        Enable or disable Filmic Tonemapping (see :func:`set_tonemapping`).
        Defaults to True.
    tonemap_exposure
        Set the exposure multiplier for the Tonemapper (see :func:`set_tonemap_exposure`).
        Defaults to 1.0.
    tonemap_method
        Set the tonemapping method ("neutral" or "agx"). Defaults to "neutral".
    """
    if samples_per_pixel is not None:
        set_samples_per_pixel(samples_per_pixel)
    if indirect_bounce_strength is not None:
        set_indirect_bounce_strength(indirect_bounce_strength)
    if physical_lighting is not None:
        set_physical_lighting(physical_lighting)
    if fragment_shading is not None:
        set_fragment_shading(fragment_shading)
    if shadows is not None:
        set_ray_traced_shadows(shadows)
    if tonemapping is not None:
        set_tonemapping(tonemapping)
    if tonemap_exposure is not None:
        set_tonemap_exposure(tonemap_exposure)
    if tonemap_method is not None:
        set_tonemap_method(tonemap_method)
    if raytraced_glow is not None:
        set_raytraced_glow(raytraced_glow)
    set_post_process_tonemap(post_process_tonemap)

    triangle_cls = (RayTracedPNTrianglePrimitive if pn_triangles
                    else RayTracedTrianglePrimitive)
    targets = []
    import algan.mobs.bezier_circuit as bezier_circuit
    import algan.mobs.shapes_2d as shapes_2d
    import algan.mobs.surfaces.surface as surface
    targets.append((shapes_2d, "TrianglePrimitive", triangle_cls))
    targets.append((surface, "TrianglePrimitive", triangle_cls))
    targets.append((bezier_circuit, "BezierCircuitPrimitive",
                    RayTracedBezierCircuitPrimitive))
    try:
-        import algan.mobs.fbx.mesh as fbx_mesh
        targets.append((fbx_mesh, "TrianglePrimitive", triangle_cls))
    except Exception:
        pass  # fbx package optional (needs the mesh mob only for imports)
    try:
        import algan.mobs.plots as plots
        targets.append((plots, "TrianglePrimitive", triangle_cls))
    except Exception:
        pass  # plots has a broken legacy import; skip it.
    for module, name, cls in targets:
        _originals.setdefault((module, name), getattr(module, name))
        setattr(module, name, cls)


def disable_ray_tracing():
    """Restore the rasterized primitive classes for newly created mobs."""
    for (module, name), original in _originals.items():
        setattr(module, name, original)
    _originals.clear()
    set_post_process_tonemap(False)


def is_ray_tracing_enabled():
    """True if the ray traced primitive classes are currently active (i.e.
    :func:`enable_ray_tracing` has been called and not yet disabled)."""
    return bool(_originals)
