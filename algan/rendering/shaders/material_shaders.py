"""Lighting-model shader functions backing the Three.js-style material system.

Each function follows the shader calling convention used throughout Algan (see
:func:`algan.rendering.shaders.pbr_shaders.default_shader`): the first nine
parameters are fixed (``memory``, ``vertex_location``, ``vertex_normal``,
``albedo_color``, ``camera_location``, ``light_origin``, ``light_color``,
``light_intensity``, ``ambient_light_intensity``) and any further parameters are
the material's animatable properties. The renderer registers those extra
parameters as animatable attributes on the mob (see
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

from algan.utils.tensor_utils import dot_product

# Base ambient coefficient. The renderer always passes ``ambient_light_intensity``
# as 1, so we scale it down here to avoid washing surfaces out to white and to
# keep unlit sides dark (matching a Three.js scene lit by a single point light
# with no AmbientLight).
AMBIENT_STRENGTH = 0.1


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


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
    """
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
    """
    rgb = rgb.clamp_min(0.0)
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

    kA = AMBIENT_STRENGTH * ambient_light_intensity * env_map_intensity
    ambient = rgb * kA
    diffuse = rgb * radiance * n_dot_l
    out = (ambient + diffuse) * _energy_scale(
        n_dot_l * radiance.amax(-1, keepdim=True) + kA
    ) + emissive * emissive_intensity
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
    """MeshPhongMaterial: Blinn-Phong diffuse + specular highlight + emissive."""
    rgb, glow = _split_albedo(albedo_color)
    n = _shading_normal(vertex_location, vertex_normal, flat_shading)
    (_l, _v, _h, n_dot_l, _nv, n_dot_h, _vh) = _light_geometry(
        vertex_location, n, camera_location, light_origin
    )
    radiance = light_color[..., :3] * light_intensity

    kA = AMBIENT_STRENGTH * ambient_light_intensity * env_map_intensity
    ambient = rgb * kA
    diffuse = rgb * radiance * n_dot_l
    # Blinn-Phong specular: (N.H)^shininess, gated by N.L so back faces stay dark.
    spec_term = n_dot_h.clamp_min(1e-4) ** shininess.clamp_min(1e-3)
    specular_out = specular * radiance * spec_term * (n_dot_l > 0)
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
    kA = AMBIENT_STRENGTH * ambient_light_intensity * env_map_intensity
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
):
    """MeshPhysicalMaterial: MeshStandard plus clearcoat, sheen, ior-driven
    specular and (approximate) transmission.

    ``ior`` drives the dielectric base reflectivity
    ``F0 = ((ior - 1) / (ior + 1))^2``, scaled by ``specular_intensity`` and
    tinted by ``specular_color`` (the KHR specular workflow). A second GGX lobe
    adds the ``clearcoat``. ``sheen`` adds a soft inverted-Fresnel rim. The
    ``transmission`` and ``iridescence`` parameters are approximated (no
    refraction / thin-film spectral model in a per-vertex pass).
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

    kA = AMBIENT_STRENGTH * ambient_light_intensity * env_map_intensity
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
    kA = AMBIENT_STRENGTH * ambient_light_intensity
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
