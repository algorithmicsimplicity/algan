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

# Maximum number of point lights that can cast deterministic ray-traced
# shadows. The caller fires one shadow ray per light up to this cap and packs
# the per-light visibilities (1 = lit, 0 = occluded) into a fixed-size vector;
# lights past the cap are still shaded but never shadowed. Eight covers every
# realistic explanatory-video setup while keeping the visibility vector small.
MAX_SHADOW_LIGHTS = 8


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
    """default_shader: diffuse lerp of the colour toward each light colour."""
    flat = params[f % params.shape[0], prim, off + 10]
    n = _prep_normal(n_interp, face_n, flat, view_dir)
    out = in_rgb
    for li in range(num_lights):
        lp, lc = _light(light_pos, light_col, f, li)
        v = _light_vis(shadows, vis, li)
        inc = (pos - lp).normalized()
        d = ti.max(-inc.dot(n), 0.0)
        diffuse = d * d * d * d * d * 0.5 * v
        out = out * (1.0 - diffuse) + lc * diffuse
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
    out = in_rgb
    for li in range(num_lights):
        lp, lc = _light(light_pos, light_col, f, li)
        v = _light_vis(shadows, vis, li)
        ld = (lp - pos).normalized()
        n_dot_l = ti.max(n.dot(ld), 0.0)
        ambient = out * (AMBIENT_STRENGTH * env)
        out = (ambient + out * lc * n_dot_l * v
               + emissive * emissive_intensity)
    return ti.math.vec4(out[0], out[1], out[2], in_glow)


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
    out = in_rgb
    for li in range(num_lights):
        lp, lc = _light(light_pos, light_col, f, li)
        v = _light_vis(shadows, vis, li)
        ld = (lp - pos).normalized()
        half = (ld + view_dir).normalized()
        n_dot_l = ti.max(n.dot(ld), 0.0)
        n_dot_h = ti.max(n.dot(half), 0.0)
        ambient = out * (AMBIENT_STRENGTH * env)
        diffuse = out * lc * n_dot_l
        spec_term = ti.pow(ti.max(n_dot_h, 1e-4), ti.max(shininess, 1e-3))
        gate = 1.0 if n_dot_l > 0.0 else 0.0
        out = (ambient + (diffuse + specular * lc * spec_term * gate)
               * v + emissive * emissive_intensity)
    return ti.math.vec4(out[0], out[1], out[2], in_glow)


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
    out = in_rgb
    for li in range(num_lights):
        lp, lc = _light(light_pos, light_col, f, li)
        v = _light_vis(shadows, vis, li)
        rgb = out
        ld = (lp - pos).normalized()
        half = (ld + view_dir).normalized()
        n_dot_l = ti.max(n.dot(ld), 0.0)
        n_dot_v = ti.max(n.dot(view_dir), 1e-4)
        n_dot_h = ti.max(n.dot(half), 0.0)
        v_dot_h = ti.max(view_dir.dot(half), 0.0)
        one = ti.math.vec3(1.0, 1.0, 1.0)
        f0 = ti.math.vec3(0.04, 0.04, 0.04) * (1.0 - metalness) \
            + rgb * metalness
        fresnel = f0 + (one - f0) * ti.pow(
            ti.max(1.0 - v_dot_h, 0.0), 5.0)
        ndf = _ggx_distribution(n_dot_h, roughness)
        geom = _smith_geometry(n_dot_v, n_dot_l, roughness)
        spec = (ndf * geom) * fresnel / ti.max(
            4.0 * n_dot_v * n_dot_l, 1e-4)
        k_d = (one - fresnel) * (1.0 - metalness)
        diffuse = k_d * rgb * lc * n_dot_l
        direct = diffuse + spec * lc * n_dot_l
        ambient = (rgb * (1.0 - metalness) + f0 * metalness) * (
            AMBIENT_STRENGTH * env)
        out = ambient + direct * v + emissive * emissive_intensity
    return ti.math.vec4(out[0], out[1], out[2], in_glow)


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
    out = ti.math.clamp(out, 0.0, 1.0)
    return ti.math.vec4(out[0], out[1], out[2], g)
