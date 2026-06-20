"""Per-fragment (Taichi) ports of Algan's core lit material shaders.

The deterministic ray tracer normally shades *per vertex*: the PyTorch material
shader (:mod:`algan.rendering.shaders.material_shaders`) is evaluated at each
triangle corner before upload and the kernel only interpolates the baked colors
(Gouraud shading). When fragment shading is enabled
(:func:`algan.rendering.raytracing.primitives.set_fragment_shading`) the kernel
instead receives the *raw albedo* plus a compact per-primitive material block,
interpolates the surface normal at the hit, and evaluates the lighting model
here -- per fragment (Phong shading), so specular highlights stay crisp and
coarse meshes shade smoothly.

Only the *core lit* materials are ported (the ones where per-fragment normals
matter most): the legacy diffuse :func:`~algan.rendering.shaders.pbr_shaders.default_shader`,
``MeshBasicMaterial`` (unlit), ``MeshLambertMaterial``, ``MeshPhongMaterial`` and
``MeshStandardMaterial``. Other materials keep vertex shading and are tagged
``_MID_UNLIT`` so the kernel returns their (already baked) colour unchanged.

Material id (``mat_id`` array, per primitive)::

    0  default diffuse      3  phong  (Blinn-Phong diffuse + specular)
    1  basic / unlit / passthrough     4  standard (Cook-Torrance GGX PBR)
    2  lambert (diffuse)

Material parameter block ``mat[MAT_W]`` (per primitive), canonical slot layout::

    0..2 emissive   3 emissive_intensity   4..6 specular   7 shininess
    8 roughness     9 metalness            10 flat_shading  11 env_map_intensity

The lighting math mirrors ``material_shaders.py`` exactly (same GGX/Smith/Schlick
terms, ``AMBIENT_STRENGTH``, ``light_intensity == ambient == 1``) and reproduces
its multi-light behaviour: each light is applied in sequence with the running
colour as the albedo (the renderer's vertex path overwrites the colour per
light), which is identical to a single light -- the common case.
"""
import taichi as ti

# Width of the per-primitive material parameter block (see slot map above).
MAT_W = 12

# Material ids.
_MID_DEFAULT = 0
_MID_UNLIT = 1
_MID_LAMBERT = 2
_MID_PHONG = 3
_MID_STANDARD = 4

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
def _shade_fragment(prim, f, pos, view_dir, n_interp, face_n, albedo, glow,
                    light_pos: ti.template(), light_col: ti.template(),
                    num_lights,
                    mat_id_arr: ti.template(), mat_arr: ti.template(),
                    shadows: ti.template(), vis):
    """Evaluate the core lit lighting model for one surface hit.

    ``albedo`` is the interpolated base RGB (``glow`` is the passthrough 4th
    colour channel); ``view_dir`` is the unit direction from the surface back
    toward the viewer (``-ray_dir``, which equals ``normalize(camera - pos)``
    for a primary ray). Returns the shaded RGB + glow as a ``vec4``.

    When ``shadows`` is enabled, ``vis`` carries one visibility scalar per
    light (``vis[li]`` is 0 if that light is occluded at this point, else 1,
    as computed by the caller's binary shadow rays); only the *direct*
    diffuse/specular response is gated by it -- the ambient and emissive
    terms are unaffected, so shadowed regions stay lit by ambient.
    """
    mid = mat_id_arr[f % mat_id_arr.shape[0], prim]
    tm = f % mat_arr.shape[0]
    emissive = ti.math.vec3(mat_arr[tm, prim, 0], mat_arr[tm, prim, 1],
                            mat_arr[tm, prim, 2])
    emissive_intensity = mat_arr[tm, prim, 3]
    specular = ti.math.vec3(mat_arr[tm, prim, 4], mat_arr[tm, prim, 5],
                            mat_arr[tm, prim, 6])
    shininess = mat_arr[tm, prim, 7]
    roughness = mat_arr[tm, prim, 8]
    metalness = mat_arr[tm, prim, 9]
    flat = mat_arr[tm, prim, 10]
    env = mat_arr[tm, prim, 11]

    n = _shading_normal(n_interp, face_n, flat)

    out = albedo
    # mid == _MID_UNLIT: basic / passthrough -- colour returned unchanged.
    if mid != _MID_UNLIT:
        for li in range(num_lights):
            tl = f % light_pos.shape[0]
            lp = ti.math.vec3(light_pos[tl, li, 0], light_pos[tl, li, 1],
                              light_pos[tl, li, 2])
            lc = ti.math.vec3(light_col[tl, li, 0], light_col[tl, li, 1],
                              light_col[tl, li, 2])
            # Per-light shadow visibility (1 lit / 0 occluded); only the direct
            # diffuse + specular response is gated by it. Compiled out entirely
            # when shadows are off, and falls back to fully lit for any light
            # beyond the shadow-ray cap.
            v = 1.0
            if ti.static(shadows != 0):
                if li < MAX_SHADOW_LIGHTS:
                    v = vis[li]
            if mid == _MID_DEFAULT:
                # default_shader: diffuse lerp toward the light colour.
                inc = (pos - lp).normalized()
                d = ti.max(-inc.dot(n), 0.0)
                diffuse = d * d * d * d * d * 0.5 * v
                out = out * (1.0 - diffuse) + lc * diffuse
            elif mid == _MID_LAMBERT:
                ld = (lp - pos).normalized()
                n_dot_l = ti.max(n.dot(ld), 0.0)
                ambient = out * (AMBIENT_STRENGTH * env)
                out = (ambient + out * lc * n_dot_l * v
                       + emissive * emissive_intensity)
            elif mid == _MID_PHONG:
                ld = (lp - pos).normalized()
                half = (ld + view_dir).normalized()
                n_dot_l = ti.max(n.dot(ld), 0.0)
                n_dot_h = ti.max(n.dot(half), 0.0)
                ambient = out * (AMBIENT_STRENGTH * env)
                diffuse = out * lc * n_dot_l
                spec_term = ti.pow(ti.max(n_dot_h, 1e-4),
                                   ti.max(shininess, 1e-3))
                gate = 1.0 if n_dot_l > 0.0 else 0.0
                out = (ambient + (diffuse + specular * lc * spec_term * gate)
                       * v + emissive * emissive_intensity)
            else:  # _MID_STANDARD
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

    out = ti.math.clamp(out, 0.0, 1.0)
    return ti.math.vec4(out[0], out[1], out[2], glow)
