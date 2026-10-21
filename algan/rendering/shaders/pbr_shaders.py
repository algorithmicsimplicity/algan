"""Algan's built-in vertex shaders.

A shader here is a plain torch function: it receives per-vertex geometry, normals
and material parameters along with the Scene's lights, and returns per-vertex
color. It runs before upload, once per vertex, which is cheap -- the trade-off
is that highlights are interpolated across a triangle rather than evaluated at
each pixel.

A Mob that sets no material of its own does not come here: it is shaded by
:class:`~algan.rendering.shaders.materials.DiffuseMaterial`, Algan's default
material, whose lighting shaders live in
:mod:`algan.rendering.shaders.material_shaders`.
:func:`null_shader` skips lighting entirely, leaving a Mob its own flat color.
:func:`basic_pbr_shader` is the metallic/roughness model the
:class:`~algan.rendering.shaders.materials.MeshStandardMaterial` family builds on.

Each has an in-kernel counterpart in
:mod:`algan.rendering.shaders.fragment_shaders`; passing one of these functions
to :meth:`~algan.animatable_base.mob_materials.MobMaterialsMixin.set_fragment_shader`
resolves it to the matching per-fragment stage.

See :doc:`/advanced_user_tutorials/shaders_and_materials`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from algan.utils.tensor_utils import dot_product


def basic_pbr_shader(
    memory,
    vertex_location,
    vertex_normal,
    albedo_color,
    camera_location,
    light_origin,
    light_color,
    light_intensity: float,
    ambient_light_intensity: float,
    smoothness: float = 0.5,
    metallicness: float = 0.5,
):
    """Shade vertices with a simplified physically based (PBR) lighting model.

    Pass it to
    :meth:`~algan.animatable_base.mob_materials.MobMaterialsMixin.set_shader`
    rather than calling it: the renderer supplies the nine leading arguments,
    and ``smoothness`` and ``metallicness`` become animatable attributes of the
    Mob. Diffuse light is Lambertian; the highlight is a GGX lobe with a
    Schlick Fresnel term, and there is no distance falloff.

    Parameters
    ----------
    memory
        Scratch-tensor provider supplied by the renderer. Unused here.
    vertex_location
        Location of the vertex to shade, shape ``(*, 3)``.
    vertex_normal
        Surface normal at the vertex; need not be normalized. Shape ``(*, 3)``.
    albedo_color
        Base color with its trailing glow channel, shape ``(*, 4)``, which is
        also the shape of the return value.
    camera_location
        Camera position, shape ``(*, 3)``.
    light_origin
        Position of the light source, shape ``(*, 3)``.
    light_color
        Color of the light, shape ``(*, 4)``.
    light_intensity
        Multiplier on the light's contribution.
    ambient_light_intensity
        Strength of the ambient term, from 0 (none) to 1.
    smoothness
        How smooth the material is, from 0 (rough, a broad dim highlight) to 1
        (smooth, a tight bright one). Defaults to ``0.5``.
    metallicness
        How metallic the material is, from 0 (dielectric: a white highlight
        over a diffuse base) to 1 (metal: no diffuse, a highlight tinted by the
        base color). Defaults to ``0.5``.

    Returns
    -------
    torch.Tensor
        The shaded color, shape ``(*, 4)``, clamped to ``[0, 1]``.
    """
    # 1. Normalize inputs and calculate core direction vectors
    # Ensure all vectors are numpy arrays for vector operations
    vertex_normal = F.normalize(
        vertex_normal, p=2, dim=-1
    )  # vertex_normal / F.norm(nornp.linalg.norm(vertex_normal)

    # Direction from object to the light source
    light_dir = light_origin - vertex_location
    distance_to_light = (light_dir).norm(p=2, dim=-1, keepdim=True)
    light_dir /= distance_to_light

    # Direction from object to the camera
    view_dir = camera_location - vertex_location
    view_dir = F.normalize(view_dir, p=2, dim=-1)

    # Halfway vector, crucial for specular calculations
    half_dir = F.normalize(light_dir + view_dir, p=2, dim=-1)

    # 2. Calculate Light Attenuation (light gets dimmer with distance)
    # A simple 1/d^2 falloff
    attenuation = 1  # 1.0 / (distance_to_light ** 2 + 1)
    radiance = light_color * light_intensity * attenuation

    # 3. Ambient Component
    # The base amount of light the object receives from the environment
    ambient = albedo_color * ambient_light_intensity

    # --- PBR Specular and Diffuse Calculation ---

    # Dot products, clamped to be non-negative
    NdotL = dot_product(vertex_normal, light_dir).relu_()
    NdotH = dot_product(vertex_normal, half_dir).relu_()

    # 4. Fresnel Term (Schlick's Approximation)
    # Determines the ratio of reflection at different angles.
    # For non-metals (dielectrics), F0 is a constant (approx 4% reflectance)
    F0_dielectric = torch.tensor(
        [0.04, 0.04, 0.04, 0.04], device=vertex_location.device
    )
    # For metals, F0 is the base color of the metal
    F0 = F0_dielectric * (1 - metallicness) + metallicness * albedo_color

    fresnel = F0 + (1.0 - F0) * (1.0 - (dot_product(light_dir, half_dir).relu_())) ** 5

    # 5. Specular Term (NDF - Normal Distribution Function, GGX Trowbridge-Reitz)
    # Approximates how much the surface's microfacets are aligned with the halfway vector.
    # Controlled by "roughness".
    roughness = 1.0 - smoothness
    alpha = roughness**2

    # The math for the GGX NDF
    alpha_sq = alpha**2
    denom = NdotH**2 * (alpha_sq - 1.0) + 1.0
    denom = torch.pi * denom**2
    ndf = alpha_sq / (denom.clamp_min(0.0001))  # Avoid division by zero

    # For a "basic" version, we can skip the Geometry (G) term, which is more complex.
    # A simplified specular BRDF is then:
    specular_brdf = (ndf * fresnel) / (
        4.0 * (dot_product(vertex_normal, view_dir).relu_()) * NdotL
    ).clamp_min(0.001)
    specular = specular_brdf * radiance * NdotL

    # 6. Diffuse Term (Lambertian)
    # PBR conserves energy: light is either reflected (specular) or refracted (diffuse).
    # The Fresnel term tells us the ratio.
    kS = fresnel  # kS is the specular ratio
    kD = 1.0 - kS  # kD is the diffuse ratio

    # Metals have no diffuse reflection; their color comes from tinted specular highlights.
    # We use the metallicness property to fade out the diffuse component.
    kD *= 1.0 - metallicness

    # Lambertian diffuse model
    diffuse_brdf = albedo_color / torch.pi
    diffuse = kD * diffuse_brdf * radiance * NdotL

    # 7. Final Combination
    # The final color is the sum of ambient, diffuse, and specular light.
    # We clip the final color to prevent negative values, although PBR should not produce them.
    final_color = ambient + diffuse + specular
    return final_color.clamp_(min=0, max=1)


def null_shader(
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
    return albedo_color
