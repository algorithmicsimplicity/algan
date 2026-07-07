"""Per-fragment (Taichi) shading for Algan's deterministic ray tracer.

The deterministic ray tracer normally shades *per vertex*: the PyTorch material
shader (:mod:`algan.rendering.shaders.material_shaders`) is evaluated at each
triangle corner before upload and the kernel only interpolates the baked colors
(Gouraud shading). When fragment shading is enabled
(:func:`algan.rendering.raytracing.primitives.set_fragment_shading`) the kernel
instead receives the *raw albedo* plus a compact per-primitive parameter block,
interpolates the surface normal at the hit, and evaluates the lighting model
here -- per fragment (Phong shading), so specular highlights stay crisp and
coarse meshes shade smoothly.

Shading is expressed as **stages** with a single uniform ``@ti.func`` contract
(see ``_stage_phong`` etc.). A per-primitive **pipeline** is an ordered list of
stages run left-to-right, each receiving the previous stage's output colour --
so a user recolour stage can feed a built-in lighting stage. The built-in *core
lit* materials are the first stages: the legacy diffuse
:func:`~algan.rendering.shaders.pbr_shaders.default_shader`, ``MeshBasicMaterial``
(unlit), ``MeshLambertMaterial``, ``MeshPhongMaterial`` and ``MeshStandardMaterial``.
Custom user stages (also ``@ti.func``) are composed into per-pipeline funcs by
:func:`make_pipeline_func` and injected into the shade kernel as a flat
``ti.template()`` tuple (see ``taichi-func-injection``).

Per-primitive **pipeline id** (``pid_arr``); ids 0-4 are the built-in
single-stage pipelines, ids >= ``_USER_PIPELINE_BASE`` index the injected user
pipeline tuple::

    0  default diffuse      3  phong  (Blinn-Phong diffuse + specular)
    1  basic / unlit / passthrough     4  standard (Cook-Torrance GGX PBR)
    2  lambert (diffuse)

Built-in material parameter block ``params[.., off:off+MAT_W]`` (per primitive),
canonical slot layout (``off`` is the stage's base offset, 0 for a built-in
single-stage pipeline)::

    0..2 emissive   3 emissive_intensity   4..6 specular   7 shininess
    8 roughness     9 metalness            10 flat_shading  11 env_map_intensity

The lighting math mirrors ``material_shaders.py`` exactly (same GGX/Smith/Schlick
terms, ``AMBIENT_STRENGTH``, ``light_intensity == ambient == 1``) and reproduces
its multi-light behaviour: each light is applied in sequence with the running
colour as the albedo (the renderer's vertex path overwrites the colour per
light), which is identical to a single light -- the common case.
"""
import os

import taichi as ti

# Width of the built-in per-primitive material parameter block (see slot map).
MAT_W = 12

# Built-in single-stage pipeline ids.
_MID_DEFAULT = 0
_MID_UNLIT = 1
_MID_LAMBERT = 2
_MID_PHONG = 3
_MID_STANDARD = 4

# Pipeline ids at or above this index address the injected user pipeline tuple
# (``frag_pipelines``): user pipeline k has id ``_USER_PIPELINE_BASE + k``.
_USER_PIPELINE_BASE = 5

# Base ambient coefficient (matches material_shaders.AMBIENT_STRENGTH).
AMBIENT_STRENGTH = 0.1

# Maximum number of lights that can cast deterministic ray-traced shadows.
# Each shaded fragment collects one visibility scalar per light (1 = lit,
# 0 = occluded) into a fixed-size ``ti.Vector`` -- Taichi vector lengths are
# compile-time, so this is a compile-time cap, not a runtime one. Lights past
# the cap are still *lit*, just never shadowed. The visibility vector is
# dead-code-eliminated when shadows are off (the default), so a larger cap
# only costs registers on opt-in shadow renders.
#
# Default 8. This is also how many samples of a soft area light actually cast
# shadows: a RectAreaLight with more samples still *lights* from all of them,
# but only the first 8 are shadow-tested. That is deliberate -- 8 gives a clean
# penumbra, and pushing this higher can (with the non-physical default diffuse
# shader, whose per-light contributions are summed unnormalised) over-brighten
# the umbra of a large area light. Raise ALGAN_MAX_SHADOW_LIGHTS if you have a
# rig of more than 8 distinct shadow-casters and have verified the look; a truly
# unbounded (runtime) count would need the per-fragment visibilities in a global
# scratch buffer instead of a stack vector (see the docs on soft shadows).
MAX_SHADOW_LIGHTS = max(1, int(os.environ.get("ALGAN_MAX_SHADOW_LIGHTS", "8")))


@ti.func
def _ggx_distribution(n_dot_h, roughness):
    """GGX / Trowbridge-Reitz normal distribution function."""
    a = ti.max(roughness * roughness, 1e-4)
    a2 = a * a
    d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0
    return a2 / ti.max(3.14159265 * d * d, 1e-7)


@ti.func
def _smith_geometry(n_dot_v, n_dot_l, roughness):
    """Smith geometry term with Schlick-GGX, direct-lighting k remapping."""
    r = roughness + 1.0
    k = (r * r) / 8.0
    gv = n_dot_v / ti.max(n_dot_v * (1.0 - k) + k, 1e-6)
    gl = n_dot_l / ti.max(n_dot_l * (1.0 - k) + k, 1e-6)
    return gv * gl


@ti.func
def _shading_normal(n_interp, face_n, flat):
    """Per-fragment shading normal, optionally blended toward the (geometric)
    face normal for flat shading -- the in-kernel analogue of
    ``material_shaders._shading_normal``."""
    n = n_interp.normalized()
    if flat > 1e-4:
        fn = face_n.normalized()
        # Align the face normal with the interpolated normal so flat shading
        # doesn't flip lighting on a smoothed mesh.
        if fn.dot(n) < 0.0:
            fn = -fn
        n = (n * (1.0 - flat) + fn * flat).normalized()
    return n


@ti.func
def _prep_normal(n_interp, face_n, flat, view_dir):
    """Shading normal (optionally flat-blended) flipped to the visible side.

    Two-sided shading: light the face the viewer actually sees. A coarsely
    tessellated mesh whose perpendicular frame was built from an arbitrary axis
    (e.g. a Cylinder via ``move_between_points``) can leave some patch normals
    pointing inward; without flipping them toward the viewer they shade as unlit
    backfaces (black), so a thin tube's apparent lighting would depend on its
    (incidental) frame orientation instead of just its shape.
    """
    n = _shading_normal(n_interp, face_n, flat)
    if n.dot(view_dir) < 0.0:
        n = -n
    return n


@ti.func
def _light(light_pos: ti.template(), light_col: ti.template(), f, li):
    """Point-light world position and RGB colour for light ``li`` at frame ``f``."""
    tl = f % light_pos.shape[0]
    lp = ti.math.vec3(light_pos[tl, li, 0], light_pos[tl, li, 1],
                      light_pos[tl, li, 2])
    lc = ti.math.vec3(light_col[tl, li, 0], light_col[tl, li, 1],
                      light_col[tl, li, 2])
    return lp, lc


# Light type ids (column 3 of an extended packed light row; mirrors
# algan.rendering.lights.LIGHT_*).
_LT_POINT = 0
_LT_DIRECTIONAL = 1
_LT_AMBIENT = 2
_LT_HEMISPHERE = 3
_LT_SPOT = 4
_LT_AREA_SAMPLE = 5
_LT_ENV_SH = 6


@ti.func
def _light_eval(light_pos: ti.template(), light_col: ti.template(),
                f, li, pos, n):
    """Evaluate light ``li`` for a surface point ``pos`` with shading normal
    ``n``: returns ``(ld, lc, spec_w)`` -- the unit direction toward the light,
    its effective RGB radiance (falloff / cone / hemisphere blending applied)
    and the specular gate (0 for the direction-less ambient-like types).

    Compact rows (``light_col`` width 3, the legacy packing used whenever the
    scene has only plain point lights) take the original point-light path with
    identical arithmetic. Extended rows (width 16) carry a type id + parameters
    (packed by ``scene_builder._pack_lights``; layout documented on
    :meth:`algan.rendering.lights.Light.build_aux`).

    Ambient-like types (ambient / hemisphere / env-SH) return ``ld = n`` so the
    material stages' ``n . ld`` diffuse factor becomes 1 -- they reuse the
    stages' diffuse term unchanged, with the specular term gated off.
    """
    tl = f % light_pos.shape[0]
    lp = ti.math.vec3(light_pos[tl, li, 0], light_pos[tl, li, 1],
                      light_pos[tl, li, 2])
    lc = ti.math.vec3(light_col[tl, li, 0], light_col[tl, li, 1],
                      light_col[tl, li, 2])
    ld = (lp - pos).normalized()
    spec_w = 1.0
    if light_col.shape[2] > 3:
        ltype = ti.cast(light_col[tl, li, 3] + 0.5, ti.i32)
        if ltype == _LT_DIRECTIONAL:
            ld = -ti.math.vec3(light_col[tl, li, 6], light_col[tl, li, 7],
                               light_col[tl, li, 8])
        elif ltype == _LT_AMBIENT:
            ld = n
            spec_w = 0.0
        elif ltype == _LT_HEMISPHERE:
            up = ti.math.vec3(light_col[tl, li, 6], light_col[tl, li, 7],
                              light_col[tl, li, 8])
            ground = ti.math.vec3(light_col[tl, li, 12],
                                  light_col[tl, li, 13],
                                  light_col[tl, li, 14])
            h = 0.5 + 0.5 * n.dot(up)
            lc = ground * (1.0 - h) + lc * h
            ld = n
            spec_w = 0.0
        elif ltype == _LT_ENV_SH:
            # Order-1 spherical-harmonics irradiance of the environment map,
            # as a linear form A + B . n (coefficients packed host-side).
            bx = ti.math.vec3(light_col[tl, li, 6], light_col[tl, li, 7],
                              light_col[tl, li, 8])
            by = ti.math.vec3(light_col[tl, li, 9], light_col[tl, li, 10],
                              light_col[tl, li, 11])
            bz = ti.math.vec3(light_col[tl, li, 12], light_col[tl, li, 13],
                              light_col[tl, li, 14])
            lc = ti.max(lc + bx * n[0] + by * n[1] + bz * n[2],
                        ti.math.vec3(0.0, 0.0, 0.0))
            ld = n
            spec_w = 0.0
        if (ltype == _LT_POINT) or (ltype == _LT_SPOT) \
                or (ltype == _LT_AREA_SAMPLE):
            d = (lp - pos).norm()
            decay = light_col[tl, li, 4]
            if decay > 0.0:
                lc = lc / ti.pow(ti.max(d, 1e-4), decay)
            rng = light_col[tl, li, 5]
            if rng > 0.0:
                q = ti.math.clamp(d / rng, 0.0, 1.0)
                q2 = q * q
                fade = ti.math.clamp(1.0 - q2 * q2, 0.0, 1.0)
                lc = lc * (fade * fade)
        if ltype == _LT_SPOT:
            sd = ti.math.vec3(light_col[tl, li, 6], light_col[tl, li, 7],
                              light_col[tl, li, 8])
            cos_outer = light_col[tl, li, 9]
            cos_inner = light_col[tl, li, 10]
            c = (-ld).dot(sd)
            t = ti.math.clamp((c - cos_outer)
                              / ti.max(cos_inner - cos_outer, 1e-6), 0.0, 1.0)
            lc = lc * (t * t * (3.0 - 2.0 * t))
        elif ltype == _LT_AREA_SAMPLE:
            # One-sided cosine emission of the rectangle sample.
            an = ti.math.vec3(light_col[tl, li, 6], light_col[tl, li, 7],
                              light_col[tl, li, 8])
            lc = lc * ti.max((-ld).dot(an), 0.0)
    return ld, lc, spec_w


@ti.func
def _light_vis(shadows: ti.template(), vis, li):
    """Per-light shadow visibility (1 lit / 0 occluded). Compiled out entirely
    when shadows are off, and falls back to fully lit beyond the shadow-ray cap."""
    v = 1.0
    if ti.static(shadows != 0):
        if li < MAX_SHADOW_LIGHTS:
            v = vis[li]
    return v


# ---------------------------------------------------------------------------
# Built-in core lit material stages.
#
# Stage contract (a ``@ti.func``): evaluate one shading pass for a surface hit
# and return the new RGB + glow as a ``vec4``. ``in_rgb`` is the running colour
# (the previous stage's output, or the interpolated raw albedo for the first
# stage); ``in_glow`` is the passthrough 4th channel; ``view_dir`` is the unit
# direction from the surface back toward the viewer. ``params`` is the
# per-primitive parameter ndarray and ``off`` this stage's base slot offset.
# When ``shadows`` is enabled, ``vis`` carries one visibility scalar per light;
# only the direct diffuse/specular response is gated by it (ambient/emissive
# stay lit). Stages loop the lights internally, exactly as the single-light
# vertex path overwrites the colour per light.
# ---------------------------------------------------------------------------

@ti.func
def _stage_unlit(pos, view_dir, n_interp, face_n, in_rgb, in_glow,
                 params: ti.template(), f, prim, off,
                 light_pos: ti.template(), light_col: ti.template(),
                 num_lights, shadows: ti.template(), vis):
    """MeshBasicMaterial / passthrough: returns the colour unchanged."""
    return ti.math.vec4(in_rgb[0], in_rgb[1], in_rgb[2], in_glow)


@ti.func
def _stage_default(pos, view_dir, n_interp, face_n, in_rgb, in_glow,
                   params: ti.template(), f, prim, off,
                   light_pos: ti.template(), light_col: ti.template(),
                   num_lights, shadows: ti.template(), vis):
    """default_shader: diffuse lerp of the colour toward each light colour.

    Additive over lights: gather every light's lerp weight, then blend once.
    For a single light this equals the legacy per-light lerp
    (``out*(1-w) + lc*w``); for many lights it stays stable (an area light's
    sample fan, or a key/fill/rim rig) instead of the old sequential lerp
    driving the colour toward the last light's."""
    flat = params[f % params.shape[0], prim, off + 10]
    n = _prep_normal(n_interp, face_n, flat, view_dir)
    out = in_rgb
    acc = ti.math.vec3(0.0, 0.0, 0.0)
    wsum = 0.0
    for li in range(num_lights):
        ld, lc, _spec_w = _light_eval(light_pos, light_col, f, li, pos, n)
        v = _light_vis(shadows, vis, li)
        d = ti.max(ld.dot(n), 0.0)
        w = d * d * d * d * d * 0.5 * v
        acc += lc * w
        wsum += w
    # Blend the base toward the lights' weighted-average colour (``acc/wsum``)
    # with total weight ``min(wsum, 1)``. Normalising ``acc`` by the weight sum
    # keeps many lights (an area light's sample fan, a large shadow-caster rig)
    # from summing past a single light's brightness -- the over-bright shadow
    # core we saw when raising the shadow-light cap. Dividing by ``max(wsum, 1)``
    # makes the single-/low-light case (``wsum <= 1``, which is *every* single
    # point light, since ``w <= 0.5``) bit-identical to the un-normalised form
    # (``x / 1.0 == x``); normalisation only engages once the summed weight
    # would otherwise blow out.
    out = out * (1.0 - ti.min(wsum, 1.0)) + acc / ti.max(wsum, 1.0)
    return ti.math.vec4(out[0], out[1], out[2], in_glow)


@ti.func
def _stage_lambert(pos, view_dir, n_interp, face_n, in_rgb, in_glow,
                   params: ti.template(), f, prim, off,
                   light_pos: ti.template(), light_col: ti.template(),
                   num_lights, shadows: ti.template(), vis):
    """MeshLambertMaterial: Lambertian (diffuse-only) lighting plus emissive."""
    tm = f % params.shape[0]
    emissive = ti.math.vec3(params[tm, prim, off + 0], params[tm, prim, off + 1],
                            params[tm, prim, off + 2])
    emissive_intensity = params[tm, prim, off + 3]
    flat = params[tm, prim, off + 10]
    env = params[tm, prim, off + 11]
    n = _prep_normal(n_interp, face_n, flat, view_dir)
    # Additive multi-light accumulation over a fixed albedo: ambient + emissive
    # once, then each light's direct diffuse. For a single light this equals
    # the legacy expression; for many lights it sums correctly (the old
    # per-light overwrite collapsed the colour and re-added ambient/emissive
    # per light -- e.g. an area light's sample fan came out wrong).
    acc = (in_rgb * (AMBIENT_STRENGTH * env)
           + emissive * emissive_intensity)
    for li in range(num_lights):
        ld, lc, _spec_w = _light_eval(light_pos, light_col, f, li, pos, n)
        v = _light_vis(shadows, vis, li)
        n_dot_l = ti.max(n.dot(ld), 0.0)
        acc += in_rgb * lc * (n_dot_l * v)
    return ti.math.vec4(acc[0], acc[1], acc[2], in_glow)


@ti.func
def _stage_phong(pos, view_dir, n_interp, face_n, in_rgb, in_glow,
                 params: ti.template(), f, prim, off,
                 light_pos: ti.template(), light_col: ti.template(),
                 num_lights, shadows: ti.template(), vis):
    """MeshPhongMaterial: Blinn-Phong diffuse + specular highlight + emissive."""
    tm = f % params.shape[0]
    emissive = ti.math.vec3(params[tm, prim, off + 0], params[tm, prim, off + 1],
                            params[tm, prim, off + 2])
    emissive_intensity = params[tm, prim, off + 3]
    specular = ti.math.vec3(params[tm, prim, off + 4], params[tm, prim, off + 5],
                            params[tm, prim, off + 6])
    shininess = params[tm, prim, off + 7]
    flat = params[tm, prim, off + 10]
    env = params[tm, prim, off + 11]
    n = _prep_normal(n_interp, face_n, flat, view_dir)
    # Additive over lights (see _stage_lambert): ambient + emissive once, then
    # each light's Blinn-Phong diffuse + specular.
    acc = (in_rgb * (AMBIENT_STRENGTH * env)
           + emissive * emissive_intensity)
    for li in range(num_lights):
        ld, lc, spec_w = _light_eval(light_pos, light_col, f, li, pos, n)
        v = _light_vis(shadows, vis, li)
        half = (ld + view_dir).normalized()
        n_dot_l = ti.max(n.dot(ld), 0.0)
        n_dot_h = ti.max(n.dot(half), 0.0)
        spec_term = ti.pow(ti.max(n_dot_h, 1e-4), ti.max(shininess, 1e-3))
        gate = spec_w if n_dot_l > 0.0 else 0.0
        acc += (in_rgb * lc * n_dot_l
                + specular * lc * spec_term * gate) * v
    return ti.math.vec4(acc[0], acc[1], acc[2], in_glow)


@ti.func
def _stage_standard(pos, view_dir, n_interp, face_n, in_rgb, in_glow,
                    params: ti.template(), f, prim, off,
                    light_pos: ti.template(), light_col: ti.template(),
                    num_lights, shadows: ti.template(), vis):
    """MeshStandardMaterial: metalness/roughness Cook-Torrance GGX PBR + emissive."""
    tm = f % params.shape[0]
    emissive = ti.math.vec3(params[tm, prim, off + 0], params[tm, prim, off + 1],
                            params[tm, prim, off + 2])
    emissive_intensity = params[tm, prim, off + 3]
    roughness = params[tm, prim, off + 8]
    metalness = params[tm, prim, off + 9]
    flat = params[tm, prim, off + 10]
    env = params[tm, prim, off + 11]
    n = _prep_normal(n_interp, face_n, flat, view_dir)
    # Additive over lights (see _stage_lambert): the metalness/F0 ambient +
    # emissive base once, then each light's Cook-Torrance direct term.
    one = ti.math.vec3(1.0, 1.0, 1.0)
    rgb = in_rgb
    f0 = ti.math.vec3(0.04, 0.04, 0.04) * (1.0 - metalness) + rgb * metalness
    acc = ((rgb * (1.0 - metalness) + f0 * metalness) * (AMBIENT_STRENGTH * env)
           + emissive * emissive_intensity)
    for li in range(num_lights):
        ld, lc, spec_w = _light_eval(light_pos, light_col, f, li, pos, n)
        v = _light_vis(shadows, vis, li)
        half = (ld + view_dir).normalized()
        n_dot_l = ti.max(n.dot(ld), 0.0)
        n_dot_v = ti.max(n.dot(view_dir), 1e-4)
        n_dot_h = ti.max(n.dot(half), 0.0)
        v_dot_h = ti.max(view_dir.dot(half), 0.0)
        fresnel = f0 + (one - f0) * ti.pow(ti.max(1.0 - v_dot_h, 0.0), 5.0)
        ndf = _ggx_distribution(n_dot_h, roughness)
        geom = _smith_geometry(n_dot_v, n_dot_l, roughness)
        spec = (ndf * geom) * fresnel / ti.max(4.0 * n_dot_v * n_dot_l, 1e-4)
        k_d = (one - fresnel) * (1.0 - metalness)
        diffuse = k_d * rgb * lc * n_dot_l
        acc += (diffuse + spec * lc * (n_dot_l * spec_w)) * v
    return ti.math.vec4(acc[0], acc[1], acc[2], in_glow)


def make_pipeline_func(stages, offsets):
    """Compose an ordered list of stage ``@ti.func``s into a single ``@ti.func``.

    Taichi cannot take a nested tuple as a ``ti.template()`` argument, so each
    distinct pipeline is baked into one func here (closing over its ``stages``
    and per-stage param ``offsets``); the shade kernel then receives just a flat
    tuple of these composed funcs (see ``taichi-func-injection``). Each stage's
    ``vec4`` output threads forward as the next stage's ``in_rgb``/``in_glow``.
    """
    stages = tuple(stages)
    offsets = tuple(int(o) for o in offsets)

    @ti.func
    def pipeline_fn(pos, view_dir, n_interp, face_n, in_rgb, in_glow,
                    params: ti.template(), f, prim,
                    light_pos: ti.template(), light_col: ti.template(),
                    num_lights, shadows: ti.template(), vis):
        out = in_rgb
        g = in_glow
        for si in ti.static(range(len(stages))):
            stage = ti.static(stages[si])
            off = ti.static(offsets[si])
            r = stage(pos, view_dir, n_interp, face_n, out, g,
                      params, f, prim, off,
                      light_pos, light_col, num_lights, shadows, vis)
            out = ti.math.vec3(r[0], r[1], r[2])
            g = r[3]
        return ti.math.vec4(out[0], out[1], out[2], g)

    return pipeline_fn


# ---------------------------------------------------------------------------
# Sorted material dispatch (Cycles-style) support.
#
# The sorted wavefront (see ``wavefront_sorted_kernels_taichi``) launches one
# small shade kernel per material bucket with that material's *pipeline func*
# injected as a ``ti.template()`` -- so the runtime pid switch of
# ``_run_frag_pipeline`` disappears and a warp never mixes materials. The five
# built-in single-stage materials are wrapped into composed pipeline funcs here
# (lazily, cached) so built-in and user pipelines share one injection contract.
#
# **Scatter contract** (user-customisable ray-bouncing): a scatter is a
# ``@ti.func`` deciding how a shaded surface event continues its ray::
#
#     scatter(rd, n_interp, face_n, hit_point, shaded, alpha, reflectivity,
#             ior, params: ti.template(), f, prim, bounces_left,
#             refraction: ti.template())
#         -> (contrib, pass_w,
#             refl_orig, refl_dir, refl_w,
#             trans_orig, trans_dir, trans_w)
#
# ``rd`` is the unit ray direction, ``shaded`` the pipeline's output colour
# (vec4: RGB + glow), ``contrib`` the premultiplied colour committed to the ray
# (the kernel adds ``weight * contrib``). ``pass_w`` is the throughput
# multiplier for continuing *through* the surface to the next depth layer
# (used only when ``refl_w == 0``). A positive ``refl_w`` bounces the ray from
# ``refl_orig`` along ``refl_dir`` with throughput ``weight * refl_w``; a
# positive ``trans_w`` additionally *splits* off a transmitted branch (glass)
# from ``trans_orig`` along ``trans_dir``. The default scatter
# (``wavefront_sorted_kernels_taichi.default_scatter``) reproduces the classic
# opacity/reflectivity/Fresnel-glass behaviour; attach a custom one to a
# :class:`~algan.rendering.shaders.fragment_shaders.FragmentStage` via its
# ``scatter=`` argument to override how rays bounce.
# ---------------------------------------------------------------------------

_BUILTIN_STAGE_FNS = (_stage_default, _stage_unlit, _stage_lambert,
                      _stage_phong, _stage_standard)
_BUILTIN_PIPELINE_FNS = {}


def builtin_pipeline_fn(pid):
    """Composed single-stage pipeline func for built-in material id ``pid``
    (0 default, 1 unlit, 2 lambert, 3 phong, 4 standard), for injection into a
    sorted per-material shade kernel. Lazily created and cached so every render
    reuses the same func objects (stable Taichi template instantiations)."""
    pid = int(pid)
    if pid not in _BUILTIN_PIPELINE_FNS:
        _BUILTIN_PIPELINE_FNS[pid] = make_pipeline_func(
            [_BUILTIN_STAGE_FNS[pid]], [0])
    return _BUILTIN_PIPELINE_FNS[pid]


@ti.func
def _run_frag_pipeline(frag_pipelines: ti.template(),
                       prim, f, pos, view_dir, n_interp, face_n, albedo, glow,
                       light_pos: ti.template(), light_col: ti.template(),
                       num_lights, pid_arr: ti.template(),
                       params: ti.template(), shadows: ti.template(), vis):
    """Evaluate a surface hit's per-primitive shading pipeline.

    ``pid_arr[f, prim]`` selects the pipeline: ids 0-4 are the built-in
    single-stage materials (dispatched directly for a transparently identical
    result to the pre-pipeline ``_shade_fragment``); ids >= ``_USER_PIPELINE_BASE``
    index the injected ``frag_pipelines`` tuple. ``albedo`` is the interpolated
    raw base RGB (``glow`` the passthrough 4th channel). Returns the shaded
    RGB + glow as a ``vec4``.
    """
    pid = pid_arr[f % pid_arr.shape[0], prim]
    out = ti.math.vec3(albedo[0], albedo[1], albedo[2])
    g = glow
    if pid == _MID_DEFAULT:
        r = _stage_default(pos, view_dir, n_interp, face_n, out, g,
                           params, f, prim, 0,
                           light_pos, light_col, num_lights, shadows, vis)
        out = ti.math.vec3(r[0], r[1], r[2])
        g = r[3]
    elif pid == _MID_LAMBERT:
        r = _stage_lambert(pos, view_dir, n_interp, face_n, out, g,
                           params, f, prim, 0,
                           light_pos, light_col, num_lights, shadows, vis)
        out = ti.math.vec3(r[0], r[1], r[2])
        g = r[3]
    elif pid == _MID_PHONG:
        r = _stage_phong(pos, view_dir, n_interp, face_n, out, g,
                         params, f, prim, 0,
                         light_pos, light_col, num_lights, shadows, vis)
        out = ti.math.vec3(r[0], r[1], r[2])
        g = r[3]
    elif pid == _MID_STANDARD:
        r = _stage_standard(pos, view_dir, n_interp, face_n, out, g,
                            params, f, prim, 0,
                            light_pos, light_col, num_lights, shadows, vis)
        out = ti.math.vec3(r[0], r[1], r[2])
        g = r[3]
    elif pid == _MID_UNLIT:
        pass  # passthrough: colour returned unchanged (raw or baked).
    else:
        for pi in ti.static(range(len(frag_pipelines))):
            if pid == _USER_PIPELINE_BASE + pi:
                fn = ti.static(frag_pipelines[pi])
                r = fn(pos, view_dir, n_interp, face_n, out, g,
                       params, f, prim,
                       light_pos, light_col, num_lights, shadows, vis)
                out = ti.math.vec3(r[0], r[1], r[2])
                g = r[3]
    #out = ti.math.clamp(out, 0.0, 1.0)
    return ti.math.vec4(out[0], out[1], out[2], g)
