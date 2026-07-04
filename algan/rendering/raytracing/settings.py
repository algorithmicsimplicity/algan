import os

from algan.rendering.raytracing.shading_taichi import _USER_PIPELINE_BASE

# Maximum number of ray bounces (mirror reflections / diffuse scatters).
MAX_BOUNCES = 4
# Rays averaged per pixel. 1 renders with the exact deterministic kernel;
# > 1 switches to the Monte Carlo pathtracer (stochastic transparency,
# glossy reflections, optional diffuse indirect lighting).
SAMPLES_PER_PIXEL = 1

TONEMAPPING = True
TONEMAP_EXPOSURE = 1.0
TONEMAP_METHOD = "neutral"
POST_PROCESS_TONEMAP = False

# Strength of diffuse indirect bounces in the Monte Carlo renderer: 0 keeps
# surfaces purely (vertex-shader) lit, > 0 scatters paths on diffuse hits
# with throughput ``albedo * strength`` for color bleeding.
INDIRECT_BOUNCE_STRENGTH = 0.0

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
# When True, the *deterministic* raytracer (SAMPLES_PER_PIXEL == 1, non-physical)
# shades the core lit materials per fragment inside the trace kernel instead of
# baking per-vertex colours (Gouraud). Ignored by the Monte Carlo pathtracer.
FRAGMENT_SHADING = True
# Promote a mob whose colour AND material params (reflectivity/roughness/index
# of refraction) are constant across the whole surface to a 1x1 texture at merge
# time, dropping its per-vertex ``tri_colors``/``tri_extra`` rows, instead of
# broadcasting the constant to every vertex. The shared texel buffer keeps one
# copy per mob (and, when the colour is also constant across frames, one copy
# total) rather than [T, N, 3, 5] / [T, N, 15]. Only applied on the
# deterministic fragment-shading wavefront path -- the only path where a
# "constant colour" mob genuinely has constant per-fragment colour (vertex
# lighting bakes per-vertex variation, so a promoted mob would be wrong there).
# The trace kernels guard every per-vertex read with ``prim < array.shape[1]``,
# so the shrunk arrays are never indexed for a promoted prim and every other
# render path stays byte-identical. Sampling a 1x1 map reduces exactly to the
# stored constant, so a promoted render matches the per-vertex one to <=1 ULP
# (the barycentric sum ``w0+w1+w2`` is not exactly 1.0 in f32). Default on;
# ALGAN_PROMOTE_CONSTANTS=0 disables it (for A/B and validation).
PROMOTE_CONSTANTS = os.environ.get("ALGAN_PROMOTE_CONSTANTS", "1") == "1"


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

def _constant_promotion_active():
    """True when constant-property -> 1x1-texture promotion applies to this
    render: it is enabled, and the batch will render through the deterministic
    fragment-shading general wavefront (the only path where a mob's colours are
    raw albedo, so a "constant colour" is genuinely constant per fragment, and
    the only kernel whose per-vertex reads are guarded for shrunk arrays). The
    material maps the promotion adds set ``has_material_textures``, which routes
    the batch to that kernel (see render_batch_ray_traced)."""
    return PROMOTE_CONSTANTS and FRAGMENT_SHADING and SAMPLES_PER_PIXEL <= 1

def _scene_has_user_pipeline(merged):
    """True if any merged primitive carries a custom fragment-pipeline id
    (``>= _USER_PIPELINE_BASE``), so the render must enable fragment shading."""
    for key in ("tri_mat_id", "pn_mat_id"):
        arr = merged.get(key)
        if arr is not None and arr.numel() and int(arr.max()) >= _USER_PIPELINE_BASE:
            return True
    return False