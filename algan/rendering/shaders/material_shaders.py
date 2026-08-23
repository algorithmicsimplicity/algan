"""Lighting-model shader functions backing the Three.js-style material system.

Each function follows the shader calling convention used throughout Algan (see
:data:`SHADER_FIXED_PARAM_COUNT` below): the first nine parameters are fixed
(``memory``, ``vertex_location``, ``vertex_normal``, ``albedo_color``,
``camera_location``, ``light_origin``, ``light_color``, ``light_intensity``,
``ambient_light_intensity``) and any further parameters are the material's
animatable properties. The renderer registers those extra parameters as
animatable attributes on the mob (see
:meth:`~algan.animatable_base.mob_materials.MobMaterialsMixin.set_shader`).

Channel layout
--------------
``albedo_color`` arrives as **4 channels** ``[R, G, B, glow]`` (the mob's colour
without its trailing opacity channel) and the return value is written back into
those same 4 channels, so every shader does its RGB maths on ``[..., :3]`` and
re-attaches the passthrough ``glow`` channel. ``light_color`` is likewise
4 channels (opacity pre-multiplied); only its RGB is used.

These are intentionally simplified, real-time-friendly approximations of the
Three.js GLSL materials -- enough to reproduce their look and respond to the same
properties, evaluated per vertex in PyTorch. Texture maps, image-based env maps,
matcap images and view-space depth packing are not sampled (Algan has no
per-fragment UV pipeline); the corresponding approximations are noted per
function.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from algan.utils.color_space import linear_to_srgb, srgb_to_linear
from algan.utils.tensor_utils import dot_product

#: Number of fixed leading parameters in the shader calling convention: the
#: ``memory, vertex_location, vertex_normal, albedo_color, camera_location,
#: light_origin, light_color, light_intensity, ambient_light_intensity``
#: prefix every lighting-model shader declares before its own material
#: parameters. This constant is the convention's reference -- the length
#: ``set_shader`` slices a shader's extra parameters from and the length the
#: renderer treats as shader-independent -- so it is pinned to a real
#: signature by tests/unit_tests/test_materials.py, which asserts it equals
#: ``len(inspect.signature(basic_material_shader).parameters)``.
SHADER_FIXED_PARAM_COUNT = 9

# Base ambient coefficient. The renderer always passes ``ambient_light_intensity``
# as 1, so we scale it down here to avoid washing surfaces out to white and to
# keep unlit sides dark (matching a Three.js scene lit by a single point light
# with no AmbientLight).
AMBIENT_STRENGTH = 0.1

# The same fill in linear light. 0.1 was chosen as a display-referred
# coefficient; carrying it unchanged into the linear working space would make
# the ambient nearly nine times brighter, because 0.1 of linear light encodes
# to byte 89 where 0.1 of an encoded value is byte 26. srgb_to_linear(0.1) =
# 0.01003, so 0.01 delivers the fill the old pipeline delivered -- the number
# changes because the units changed, not because the look was retuned. Twin of
# shading_taichi.AMBIENT_STRENGTH_LINEAR.
AMBIENT_STRENGTH_LINEAR = 0.01


def _ambient_strength():
    """The ambient coefficient for the active working space."""
    return AMBIENT_STRENGTH_LINEAR if _linear_color_space() else AMBIENT_STRENGTH


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _linear_color_space():
    """True when shading runs in the linear working colour space.

    Gates this module's two gamma-era compensations -- ``_energy_scale``'s
    illumination budget and ``_recombine``'s peak bound, the torch twins of
    ``shading_taichi._linear_color_space``'s gates. They normalise away an
    overshoot that only exists when sRGB-encoded values are summed; in linear
    light lights genuinely add, so both are off.

    The import is local and read is through the module object so the value is
    live at every call -- whatever the setting holds when a shader runs,
    never a value frozen into this module at import time.
    """
    from algan.rendering.raytracing import settings as rt_settings

    return bool(rt_settings.LINEAR_COLOR_SPACE)


def _split_albedo(albedo_color):
    """Split ``[..., 4]`` albedo into ``(rgb[..., 3], glow_tail[..., 1])``."""
    return albedo_color[..., :3], albedo_color[..., 3:]


def _energy_scale(weight):
    """Reciprocal of the illumination budget -- the torch twin of
    ``shading_taichi._energy_scale``.

    ``weight`` is the total illumination arriving at the surface: the ambient
    fill's coefficient plus this light's ``n.l``. A reflective surface cannot
    send out more light than arrives, so once the incident weight passes unity
    the reflected terms are scaled back by it. Exactly 1.0 below unity, so an
    under-lit surface is untouched.

    These shaders see one light at a time -- the vertex path loops over lights
    outside them (``primitives.py``) -- so this bounds the ambient-on-top-of-
    direct overshoot here, and the fragment path's twin is what bounds the
    multi-light sum.

    Off under ``_linear_color_space()``: there lights sum plainly and this
    returns exactly 1.0, since normalising would make them stop adding.
    """
    if _linear_color_space():
        return 1.0
    return 1.0 / weight.clamp_min(1.0)


def _recombine(rgb, glow_tail):
    """Bound the RGB result to ``[0, 1]`` and re-attach the glow channel.

    Lights accumulate without normalisation, so several of them drive a fully
    lit surface past 1.0. Scaling all three channels by the peak keeps the hue,
    where the per-channel clamp this replaced truncated each independently and
    slid an over-range saturated colour toward white. Identity below 1.0, so
    anything already in range is untouched; ``glow_tail`` is never bounded,
    which leaves glow the one route to above-1.0 output for bloom.

    Kept in step with ``_run_frag_pipeline`` in
    ``algan/rendering/raytracing/shading_taichi.py``, which does the same thing
    for the fragment path.

    The peak bound is off under ``_linear_color_space()``: in linear light the
    sum is physically additive and the sRGB OETF at the byte write owns the
    range, so scaling by the peak would make lights stop adding. The negative
    clamp stays either way -- it is not part of the bound; it stops a negative
    reaching the encoder's pow.
    """
    rgb = rgb.clamp_min(0.0)
    if not _linear_color_space():
        # clamp_min(1.0) makes the divisor exactly 1 whenever nothing is over
        # range, so the in-range case is a bit-identical no-op and there is no
        # divide-by-zero on black.
        rgb = rgb / rgb.amax(-1, keepdim=True).clamp_min(1.0)
    return torch.cat((rgb, glow_tail), -1)


def _normalize(v):
    return F.normalize(v, p=2, dim=-1)


def _shading_normal(vertex_location, vertex_normal, flat_shading):
    """Per-vertex normal, optionally blended toward the flat (per-face) normal.

    ``flat_shading`` is a ``[..., 1]`` tensor in ``[0, 1]``. The geometric face
    normal is only available when the corner axis is present (the renderer passes
    the triangle's three corners as ``vertex_location`` with shape
    ``[..., 3, 3]``); otherwise the smooth normal is returned unchanged.
    """
    n = _normalize(vertex_normal)
    if torch.is_tensor(flat_shading) and vertex_location.shape[-2] == 3:
        # No host sync here (perf-sensitive render path): when flat_shading is 0
        # the lerp below is a no-op, so we always compute the face normal rather
        # than reading the flag back to the CPU to branch.
        c = vertex_location
        face = torch.linalg.cross(
            c[..., 1, :] - c[..., 0, :], c[..., 2, :] - c[..., 0, :], dim=-1
        )
        face = _normalize(face).unsqueeze(-2)  # [..., 1, 3]
        # Align the face normal's orientation with the interpolated normals so
        # flat shading doesn't flip lighting on a smoothed mesh.
        sign = torch.sign(dot_product(face, n).sum(-2, keepdim=True) + 1e-9)
        face = (face * sign).expand_as(n)
        n = _normalize(torch.lerp(n, face, flat_shading))
    return n


def fresnel_schlick(cos_theta, f0):
    """Schlick's Fresnel approximation."""
    return f0 + (1.0 - f0) * (1.0 - cos_theta).clamp(0.0, 1.0) ** 5


def ggx_distribution(n_dot_h, roughness):
    """GGX / Trowbridge-Reitz normal distribution function."""
    a = (roughness * roughness).clamp_min(1e-4)
    a2 = a * a
    d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0
    return a2 / (torch.pi * d * d).clamp_min(1e-7)


def smith_geometry(n_dot_v, n_dot_l, roughness):
    """Smith geometry term with Schlick-GGX, direct-lighting k remapping."""
    r = roughness + 1.0
    k = (r * r) / 8.0
    gv = n_dot_v / (n_dot_v * (1.0 - k) + k).clamp_min(1e-6)
    gl = n_dot_l / (n_dot_l * (1.0 - k) + k).clamp_min(1e-6)
    return gv * gl


def _d_charlie(n_dot_h, sheen_roughness):
    """Charlie sheen distribution (Estevez and Kulla 2017, via Three.js's
    ``D_Charlie``): exponentiated-sine microfibre lobe. The ``sin2h`` floor is
    Three.js's, kept so ``pow(0, large)`` cannot appear.
    """
    alpha = (
        max(sheen_roughness, 1e-4)
        if not torch.is_tensor(sheen_roughness)
        else sheen_roughness.clamp_min(1e-4)
    )
    inv_alpha = 1.0 / (alpha * alpha)
    sin2h = (1.0 - n_dot_h * n_dot_h).clamp_min(0.0078125)
    return (2.0 + inv_alpha) * sin2h ** (inv_alpha * 0.5) / (2.0 * math.pi)


def _v_neubelt(n_dot_v, n_dot_l):
    """Neubelt and Pettineo 2013 sheen visibility, as in Three.js's
    ``V_Neubelt``, clamped to 1 like its ``saturate``.
    """
    return (
        1.0 / (4.0 * (n_dot_l + n_dot_v - n_dot_l * n_dot_v)).clamp_min(1e-6)
    ).clamp(0.0, 1.0)


def _ibl_sheen_brdf(cos_theta, sheen_roughness):
    """Three.js's ``IBLSheenBRDF``: a curve fit to the Charlie lobe integrated
    over the hemisphere, used for the base layer's energy compensation.
    """
    r = sheen_roughness
    r2 = r * r
    r_inv = 1.0 / (r + 0.1)
    a = -1.9362 + 1.0678 * r + 0.4573 * r2 - 0.8469 * r_inv
    b = -0.6014 + 0.5538 * r - 0.4670 * r2 - 0.1255 * r_inv
    dg = (
        torch.exp(a * cos_theta + b)
        if torch.is_tensor(cos_theta)
        else math.exp(a * cos_theta + b)
    )
    return dg.clamp(0.0, 1.0) if torch.is_tensor(dg) else min(max(dg, 0.0), 1.0)


def _light_geometry(vertex_location, normal, camera_location, light_origin):
    """Common normalized direction vectors and clamped dot products."""
    light_dir = _normalize(light_origin - vertex_location)
    view_dir = _normalize(camera_location - vertex_location)
    half_dir = _normalize(light_dir + view_dir)
    n_dot_l = dot_product(normal, light_dir).clamp_min(0.0)
    n_dot_v = dot_product(normal, view_dir).clamp_min(1e-4)
    n_dot_h = dot_product(normal, half_dir).clamp_min(0.0)
    v_dot_h = dot_product(view_dir, half_dir).clamp_min(0.0)
    return light_dir, view_dir, half_dir, n_dot_l, n_dot_v, n_dot_h, v_dot_h


# ---------------------------------------------------------------------------
# Material shaders
# ---------------------------------------------------------------------------


def basic_material_shader(
    memory,
    vertex_location,
    vertex_normal,
    albedo_color,
    camera_location,
    light_origin,
    light_color,
    light_intensity: float,
    ambient_light_intensity: float,
):
    """MeshBasicMaterial: unlit, returns the flat base colour unchanged."""
    return albedo_color


def lambert_shader(
    memory,
    vertex_location,
    vertex_normal,
    albedo_color,
    camera_location,
    light_origin,
    light_color,
    light_intensity: float,
    ambient_light_intensity: float,
    emissive=(0.0, 0.0, 0.0),
    emissive_intensity: float = 1.0,
    flat_shading: float = 0.0,
    env_map_intensity: float = 1.0,
):
    """MeshLambertMaterial: Lambertian (diffuse-only) lighting plus emissive."""
    rgb, glow = _split_albedo(albedo_color)
    n = _shading_normal(vertex_location, vertex_normal, flat_shading)
    light_dir = _normalize(light_origin - vertex_location)
    n_dot_l = dot_product(n, light_dir).clamp_min(0.0)
    radiance = light_color[..., :3] * light_intensity

    kA = _ambient_strength() * ambient_light_intensity * env_map_intensity
    ambient = rgb * kA
    diffuse = rgb * radiance * n_dot_l
    out = (ambient + diffuse) * _energy_scale(
        n_dot_l * radiance.amax(-1, keepdim=True) + kA
    ) + emissive * emissive_intensity
    return _recombine(out, glow)


def manim_shader(
    memory,
    vertex_location,
    vertex_normal,
    albedo_color,
    camera_location,
    light_origin,
    light_color,
    light_intensity: float,
    ambient_light_intensity: float,
    flat_shading: float = 0.0,
):
    """Shade a surface with Manim's default 3-D lighting model.

    Implements Manim's ``get_shaded_rgb``: each light contributes an offset of
    ``0.5 * (n . to_light) ** 3`` -- halved when the surface faces away from
    the light, so back-facing surfaces darken at half the rate front-facing
    ones brighten. There is no ambient term, no specular lobe and no distance
    falloff; ``ambient_light_intensity`` is accepted for signature parity but
    unused, because Manim's model has none.

    Under Manim's own rig -- the single white ``PointLight`` of intensity 1
    that :meth:`~.Scene.use_manim_defaults` installs, with decay 0 and
    distance 0 -- the light-colour factor below is exactly ``(1, 1, 1)`` and
    the offset reproduces Manim's scalar one exactly. Multiplying by
    ``light_color * light_intensity`` is a strict generalisation to coloured
    and multi-light rigs, which Manim does not have; the per-light offsets
    simply sum.

    Manim adds its offset to display-referred sRGB values. Under the default
    linear working space this shader therefore encodes the base colour to
    sRGB, adds the offsets there, clamps the sum to ``[0, 1]`` and decodes
    back to linear light; under the display-referred setting it adds and
    clamps directly. Exact Manim fidelity further assumes exposure 1 and
    tonemapping off -- which is what :meth:`~.Scene.use_manim_defaults` sets;
    any other exposure or tonemap curve maps the result as Manim never would.

    Parameters
    ----------
    memory
        Scratch-tensor provider supplied by the renderer. Unused here.
    vertex_location
        Location of the vertex to shade, shape ``(*, 3)``; the renderer's
        triangle path passes ``(*, 3, 3)`` corners.
    vertex_normal
        Surface normal at the vertex; need not be normalized. Shape ``(*, 3)``.
    albedo_color
        Base colour with its trailing glow channel, shape ``(*, 4)``, which is
        also the shape of the return value.
    camera_location
        Camera position, shape ``(*, 3)``. Accepted for signature parity;
        Manim's model is view-independent, so this shader ignores it.
    light_origin
        Position of the light source, shape ``(*, 3)``.
    light_color
        Colour of the light with its trailing opacity channel, shape
        ``(*, 4)``; only its RGB is used.
    light_intensity
        Multiplier on the light's contribution. Defaults to whatever the
        renderer passes (1 for Algan's stock rig).
    ambient_light_intensity
        Accepted for signature parity; unused, since the model has no ambient
        term.
    flat_shading
        Blend of the interpolated normal toward the flat per-face normal,
        from 0 (smooth, the default) to 1 (flat).
    """
    rgb, glow = _split_albedo(albedo_color)
    n = _shading_normal(vertex_location, vertex_normal, flat_shading)
    to_sun = _normalize(light_origin - vertex_location)
    w = 0.5 * dot_product(n, to_sun) ** 3
    w = torch.where(w < 0.0, 0.5 * w, w)
    offset = w * light_color[..., :3] * light_intensity
    if _linear_color_space():
        out = srgb_to_linear(linear_to_srgb(rgb) + offset).clamp(0.0, 1.0)
    else:
        out = (rgb + offset).clamp(0.0, 1.0)
    return _recombine(out, glow)


def phong_shader(
    memory,
    vertex_location,
    vertex_normal,
    albedo_color,
    camera_location,
    light_origin,
    light_color,
    light_intensity: float,
    ambient_light_intensity: float,
    emissive=(0.0, 0.0, 0.0),
    emissive_intensity: float = 1.0,
    specular=(0.0666, 0.0666, 0.0666),
    shininess: float = 30.0,
    flat_shading: float = 0.0,
    env_map_intensity: float = 1.0,
):
    """MeshPhongMaterial: Blinn-Phong diffuse + specular highlight + emissive.

    Twin of ``shading_taichi._stage_phong``; read its docstring for why the
    specular lobe carries three.js's ``0.25 * (shininess * 0.5 + 1)``
    normalization and its Fresnel term but not its ``1/pi`` (the diffuse lobe
    drops the same factor, so the ratio between them is three.js's exactly).
    """
    rgb, glow = _split_albedo(albedo_color)
    n = _shading_normal(vertex_location, vertex_normal, flat_shading)
    (_l, _v, _h, n_dot_l, _nv, n_dot_h, v_dot_h) = _light_geometry(
        vertex_location, n, camera_location, light_origin
    )
    radiance = light_color[..., :3] * light_intensity

    kA = _ambient_strength() * ambient_light_intensity * env_map_intensity
    ambient = rgb * kA
    diffuse = rgb * radiance * n_dot_l
    # Blinn-Phong specular: F_Schlick(specular, 1, V.H) * 0.25 * D, with
    # D = (shininess * 0.5 + 1) * (N.H)^shininess, scaled by N.L -- which
    # also keeps back faces dark without a separate gate.
    s = (
        shininess.clamp_min(1e-3)
        if torch.is_tensor(shininess)
        else max(shininess, 1e-3)
    )
    d_blinn = (s * 0.5 + 1.0) * n_dot_h.clamp_min(1e-4) ** s
    # ``specular`` reaches the live path as a tensor (Material.get_shader_param_values
    # runs it through _to_rgb); the signature default is a plain tuple, and
    # ``1 - tuple`` is not a thing, so normalise before the Fresnel.
    spec_rgb = specular if torch.is_tensor(specular) else torch.tensor(specular)
    f_spec = spec_rgb + (1.0 - spec_rgb) * (1.0 - v_dot_h).clamp(0.0, 1.0) ** 5
    specular_out = f_spec * radiance * (0.25 * d_blinn * n_dot_l)
    out = (ambient + diffuse + specular_out) * _energy_scale(
        n_dot_l * radiance.amax(-1, keepdim=True) + kA
    ) + emissive * emissive_intensity
    return _recombine(out, glow)


def standard_shader(
    memory,
    vertex_location,
    vertex_normal,
    albedo_color,
    camera_location,
    light_origin,
    light_color,
    light_intensity: float,
    ambient_light_intensity: float,
    roughness: float = 1.0,
    metalness: float = 0.0,
    emissive=(0.0, 0.0, 0.0),
    emissive_intensity: float = 1.0,
    env_map_intensity: float = 1.0,
    flat_shading: float = 0.0,
):
    """MeshStandardMaterial: metalness/roughness Cook-Torrance PBR + emissive.

    Implements the GGX NDF, Smith geometry and Schlick Fresnel terms. The
    image-based environment reflection of the GLSL material is approximated by a
    constant ambient term scaled by ``env_map_intensity``.
    """
    rgb, glow = _split_albedo(albedo_color)
    n = _shading_normal(vertex_location, vertex_normal, flat_shading)
    (_l, _v, _h, n_dot_l, n_dot_v, n_dot_h, v_dot_h) = _light_geometry(
        vertex_location, n, camera_location, light_origin
    )
    radiance = light_color[..., :3] * light_intensity

    # Base reflectivity: 4% for dielectrics, the albedo for metals.
    f0 = 0.04 * (1.0 - metalness) + metalness * rgb
    fresnel = fresnel_schlick(v_dot_h, f0)
    ndf = ggx_distribution(n_dot_h, roughness)
    geom = smith_geometry(n_dot_v, n_dot_l, roughness)
    specular = (ndf * geom * fresnel) / (4.0 * n_dot_v * n_dot_l).clamp_min(1e-4)

    k_d = (1.0 - fresnel) * (1.0 - metalness)
    diffuse = k_d * rgb * radiance * n_dot_l
    direct = diffuse + specular * radiance * n_dot_l

    # Ambient/environment approximation (diffuse for dielectrics, tinted for metals).
    kA = _ambient_strength() * ambient_light_intensity * env_map_intensity
    ambient = (rgb * (1.0 - metalness) + f0 * metalness) * kA
    out = (ambient + direct) * _energy_scale(
        n_dot_l * radiance.amax(-1, keepdim=True) + kA
    ) + emissive * emissive_intensity
    return _recombine(out, glow)


def physical_shader(
    memory,
    vertex_location,
    vertex_normal,
    albedo_color,
    camera_location,
    light_origin,
    light_color,
    light_intensity: float,
    ambient_light_intensity: float,
    roughness: float = 1.0,
    metalness: float = 0.0,
    emissive=(0.0, 0.0, 0.0),
    emissive_intensity: float = 1.0,
    env_map_intensity: float = 1.0,
    flat_shading: float = 0.0,
    ior: float = 1.5,
    specular_intensity: float = 1.0,
    specular_color=(1.0, 1.0, 1.0),
    clearcoat: float = 0.0,
    clearcoat_roughness: float = 0.0,
    sheen: float = 0.0,
    sheen_roughness: float = 1.0,
    sheen_color=(0.0, 0.0, 0.0),
    transmission: float = 0.0,
    iridescence: float = 0.0,
    # Registered so the packed name reaches the material block's
    # attenuation_sigma slots; deliberately unused HERE -- absorption acts on
    # the segment a ray travels inside the medium, which a per-vertex surface
    # pass does not see (the wavefront bounce loop applies it).
    attenuation_sigma=(0.0, 0.0, 0.0),
):
    """MeshPhysicalMaterial: MeshStandard plus clearcoat, sheen, ior-driven
    specular and (approximate) transmission.

    ``ior`` drives the dielectric base reflectivity
    ``F0 = ((ior - 1) / (ior + 1))^2``, scaled by ``specular_intensity`` and
    tinted by ``specular_color`` (the KHR specular workflow). A second GGX lobe
    adds the ``clearcoat``. ``sheen`` adds a soft inverted-Fresnel rim. The
    ``transmission`` and ``iridescence`` parameters are approximated (no
    refraction / thin-film spectral model in a per-vertex pass).
    Volumetric absorption is not approximated here at all: it is applied along
    the refracted path in the renderer, driven by this parameter's packed slot.
    """
    rgb, glow = _split_albedo(albedo_color)
    n = _shading_normal(vertex_location, vertex_normal, flat_shading)
    (_l, _v, _h, n_dot_l, n_dot_v, n_dot_h, v_dot_h) = _light_geometry(
        vertex_location, n, camera_location, light_origin
    )
    radiance = light_color[..., :3] * light_intensity

    dielectric_f0 = (
        (((ior - 1.0) / (ior + 1.0)) ** 2) * specular_intensity * specular_color
    )
    f0 = dielectric_f0 * (1.0 - metalness) + metalness * rgb
    fresnel = fresnel_schlick(v_dot_h, f0)
    ndf = ggx_distribution(n_dot_h, roughness)
    geom = smith_geometry(n_dot_v, n_dot_l, roughness)
    specular = (ndf * geom * fresnel) / (4.0 * n_dot_v * n_dot_l).clamp_min(1e-4)

    k_d = (1.0 - fresnel) * (1.0 - metalness) * (1.0 - transmission)

    # Sheen, and what it takes from the layer underneath -- the in-torch twin
    # of shading_taichi._stage_physical; see the long comment there for why
    # the base is scaled and the clearcoat is not.
    sheen_c = sheen_color * sheen
    sheen_max = sheen_c.max(-1, keepdim=True).values
    sheen_r = (
        sheen_roughness.clamp(1e-4, 1.0)
        if torch.is_tensor(sheen_roughness)
        else max(min(sheen_roughness, 1.0), 1e-4)
    )
    sheen_comp = 1.0 - sheen_max * torch.maximum(
        _ibl_sheen_brdf(n_dot_v, sheen_r), _ibl_sheen_brdf(n_dot_l, sheen_r)
    )

    diffuse = k_d * rgb * radiance * n_dot_l
    direct = (diffuse + specular * radiance * n_dot_l) * sheen_comp

    # Clearcoat: a thin dielectric GGX lobe (fixed F0 = 0.04) on top.
    cc_ndf = ggx_distribution(n_dot_h, clearcoat_roughness)
    cc_geom = smith_geometry(n_dot_v, n_dot_l, clearcoat_roughness)
    cc_fresnel = fresnel_schlick(v_dot_h, 0.04)
    clearcoat_spec = (
        clearcoat
        * (cc_ndf * cc_geom * cc_fresnel)
        / (4.0 * n_dot_v * n_dot_l).clamp_min(1e-4)
    )
    direct = direct + clearcoat_spec * radiance * n_dot_l

    # The Charlie fibre lobe itself (KHR_materials_sheen / Three.js BRDF_Sheen).
    sheen_brdf = _d_charlie(n_dot_h, sheen_r) * _v_neubelt(n_dot_v, n_dot_l)
    direct = direct + sheen_c * sheen_brdf * radiance * n_dot_l

    kA = _ambient_strength() * ambient_light_intensity * env_map_intensity
    ambient = (rgb * (1.0 - metalness) + f0 * metalness) * kA
    # No per-light transmission term: the transmitted share is carried by the
    # renderer's own continuation, and adding it here again double counted.
    # See shading_taichi._stage_physical.
    out = (ambient + direct) * _energy_scale(
        n_dot_l * radiance.amax(-1, keepdim=True) + kA
    ) + emissive * emissive_intensity
    return _recombine(out, glow)


def toon_shader(
    memory,
    vertex_location,
    vertex_normal,
    albedo_color,
    camera_location,
    light_origin,
    light_color,
    light_intensity: float,
    ambient_light_intensity: float,
    emissive=(0.0, 0.0, 0.0),
    emissive_intensity: float = 1.0,
    num_bands: float = 3.0,
    flat_shading: float = 0.0,
):
    """MeshToonMaterial: diffuse lighting quantized into flat bands (cel shading).

    The Three.js ``gradientMap`` is approximated by an even ``num_bands``-step
    ramp.
    """
    rgb, glow = _split_albedo(albedo_color)
    n = _shading_normal(vertex_location, vertex_normal, flat_shading)
    light_dir = _normalize(light_origin - vertex_location)
    n_dot_l = dot_product(n, light_dir).clamp_min(0.0)

    bands = (
        num_bands.clamp_min(1.0) if torch.is_tensor(num_bands) else max(num_bands, 1.0)
    )
    stepped = torch.ceil(n_dot_l * bands) / bands
    kA = _ambient_strength() * ambient_light_intensity
    ambient = rgb * kA
    diffuse = rgb * light_color[..., :3] * light_intensity * stepped
    out = (ambient + diffuse) * _energy_scale(
        stepped * (light_color[..., :3] * light_intensity).amax(-1, keepdim=True) + kA
    ) + emissive * emissive_intensity
    return _recombine(out, glow)


def normal_shader(
    memory,
    vertex_location,
    vertex_normal,
    albedo_color,
    camera_location,
    light_origin,
    light_color,
    light_intensity: float,
    ambient_light_intensity: float,
    flat_shading: float = 0.0,
):
    """MeshNormalMaterial: RGB encodes the surface normal (``n * 0.5 + 0.5``).

    Three.js uses *view-space* normals; only the camera location (not its
    orientation) is available here, so this uses world-space normals.
    """
    _rgb, glow = _split_albedo(albedo_color)
    n = _shading_normal(vertex_location, vertex_normal, flat_shading)
    out = n * 0.5 + 0.5
    return _recombine(out, glow)


def matcap_shader(
    memory,
    vertex_location,
    vertex_normal,
    albedo_color,
    camera_location,
    light_origin,
    light_color,
    light_intensity: float,
    ambient_light_intensity: float,
    flat_shading: float = 0.0,
):
    """MeshMatcapMaterial: the matcap image is not sampled (no UV pipeline); this
    approximates a default matcap with a view-facing diffuse term plus a rim
    highlight, tinted by the base colour.
    """
    rgb, glow = _split_albedo(albedo_color)
    n = _shading_normal(vertex_location, vertex_normal, flat_shading)
    view_dir = _normalize(camera_location - vertex_location)
    n_dot_v = dot_product(n, view_dir).clamp_min(0.0)
    rim = (1.0 - n_dot_v).clamp(0.0, 1.0) ** 3
    out = rgb * (0.3 + 0.7 * n_dot_v) + rim * 0.4
    return _recombine(out, glow)


def depth_shader(
    memory,
    vertex_location,
    vertex_normal,
    albedo_color,
    camera_location,
    light_origin,
    light_color,
    light_intensity: float,
    ambient_light_intensity: float,
    near: float = 0.1,
    far: float = 100.0,
):
    """MeshDepthMaterial: grayscale by camera distance, near=bright, far=dark.

    Approximates Three.js depth packing with a simple linear luminance ramp.
    """
    _rgb, glow = _split_albedo(albedo_color)
    distance = (vertex_location - camera_location).norm(p=2, dim=-1, keepdim=True)
    span = far - near
    span = span.clamp_min(1e-6) if torch.is_tensor(span) else max(span, 1e-6)
    normalized = ((distance - near) / span).clamp(0.0, 1.0)
    value = (1.0 - normalized).expand(*normalized.shape[:-1], 3)
    return _recombine(value, glow)
