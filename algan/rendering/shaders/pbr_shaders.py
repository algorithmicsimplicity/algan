import torch
import torch.nn.functional as F

from algan.utils.tensor_utils import dot_product


def basic_pbr_shader(object_pos,
        normal,
        base_color,
        camera_pos,

        smoothness: float,
        metallic: float,

        light_pos,
        light_color,
        light_intensity: float,

        # --- Scene Ambient Light ---
        ambient_light_color, ambient_light_intensity):
    """Implements a simplified physics-based rendering shader, using smoothness and metallicness
    to define diffuse and specular lighting..

    Parameters
    ----------
    object_pos : (np.ndarray)
        The 3D position of the point on the object.
    normal : (np.ndarray)
        The surface normal vector at that point (must be normalized).
    base_color : (np.ndarray)
        The albedo/base RGB color of the material.
    camera_pos : (np.ndarray)
        The 3D position of the camera/viewer.
    smoothness : (float)
        The smoothness of the material [0, 1]. 0=rough, 1=smooth.
    metallic : (float)
        The metallic property of the material [0, 1]. 0=dielectric, 1=metal.
    light_pos : (np.ndarray)
        The 3D position of the point light source.
    light_color : (np.ndarray)
        The RGB color of the light.
    light_intensity : float
        The intensity/brightness of the light.
    ambient_light_color : np.ndarray
        The RGB color of the scene's ambient light.

    Returns
    -------
    np.ndarray
        The final computed RGB color for the point.
    """
    # 1. Normalize inputs and calculate core direction vectors
    # Ensure all vectors are numpy arrays for vector operations
    normal = F.normalize(normal, p=2, dim=-1)#normal / F.norm(nornp.linalg.norm(normal)

    # Direction from object to the light source
    light_dir = light_pos - object_pos
    distance_to_light = (light_dir).norm(p=2, dim=-1, keepdim=True)
    light_dir /= distance_to_light

    # Direction from object to the camera
    view_dir = camera_pos - object_pos
    view_dir = F.normalize(view_dir, p=2, dim=-1)

    # Halfway vector, crucial for specular calculations
    half_dir = F.normalize(light_dir + view_dir, p=2, dim=-1)

    # 2. Calculate Light Attenuation (light gets dimmer with distance)
    # A simple 1/d^2 falloff
    attenuation = 1#1.0 / (distance_to_light ** 2 + 1)
    radiance = light_color * light_intensity * attenuation

    # 3. Ambient Component
    # The base amount of light the object receives from the environment
    ambient = base_color * 1##ambient_light_color * base_color

    # --- PBR Specular and Diffuse Calculation ---

    # Dot products, clamped to be non-negative
    NdotL = dot_product(normal, light_dir).relu_()
    NdotH = dot_product(normal, half_dir).relu_()

    # 4. Fresnel Term (Schlick's Approximation)
    # Determines the ratio of reflection at different angles.
    # For non-metals (dielectrics), F0 is a constant (approx 4% reflectance)
    F0_dielectric = torch.tensor([0.04, 0.04, 0.04, 0.04], device=object_pos.device)
    # For metals, F0 is the base color of the metal
    F0 = F0_dielectric * (1-metallic) + metallic * base_color

    fresnel = F0 + (1.0 - F0) * (1.0 - (dot_product(light_dir, half_dir).relu_())) ** 5

    # 5. Specular Term (NDF - Normal Distribution Function, GGX Trowbridge-Reitz)
    # Approximates how much the surface's microfacets are aligned with the halfway vector.
    # Controlled by "roughness".
    roughness = 1.0 - smoothness
    alpha = roughness ** 2

    # The math for the GGX NDF
    alpha_sq = alpha ** 2
    denom = (NdotH ** 2 * (alpha_sq - 1.0) + 1.0)
    denom = torch.pi * denom ** 2
    ndf = alpha_sq / (denom.clamp_min(0.0001))  # Avoid division by zero

    # For a "basic" version, we can skip the Geometry (G) term, which is more complex.
    # A simplified specular BRDF is then:
    specular_brdf = (ndf * fresnel) / (4.0 * (dot_product(normal, view_dir).relu_()) * NdotL).clamp_min(0.001)
    specular = specular_brdf * radiance * NdotL

    # 6. Diffuse Term (Lambertian)
    # PBR conserves energy: light is either reflected (specular) or refracted (diffuse).
    # The Fresnel term tells us the ratio.
    kS = fresnel  # kS is the specular ratio
    kD = 1.0 - kS  # kD is the diffuse ratio

    # Metals have no diffuse reflection; their color comes from tinted specular highlights.
    # We use the metallic property to fade out the diffuse component.
    kD *= (1.0 - metallic)

    # Lambertian diffuse model
    diffuse_brdf = base_color# / torch.pi
    diffuse = kD * diffuse_brdf * radiance * NdotL

    # 7. Final Combination
    # The final color is the sum of ambient, diffuse, and specular light.
    # We clip the final color to prevent negative values, although PBR should not produce them.
    final_color = ambient + diffuse + specular
    return (final_color.clamp_(min=0, max=1))


def default_shader(object_pos,
        normal,
        base_color,
        camera_pos,

        smoothness: float,
        metallic: float,

        light_pos,
        light_color,
        light_intensity: float,

        # --- Scene Ambient Light ---
        ambient_light_color, ambient_light_intensity):


    incidences = F.normalize(object_pos - light_pos, p=2, dim=-1)
    #reflects = F.normalize(incidences - 2 * normals * (dot_product(normals, incidences)), p=2, dim=-1)
    #diffuse_factor = dot_product(views, reflects).relu_().pow_(0.5)
    #diffuse_factor = (dot_product(-incidences, normals) * ((dot_product(views, normals) < 0).float()*2-1)).abs().relu_().pow_(10)
    diffuse_factor = (dot_product(-incidences, normal)).relu_().pow_(5)
    diffuse_factor = diffuse_factor * 0.5
    self_colors = base_color.clone()
    self_colors[...,:-1] = self_colors[...,:-1] * (1-diffuse_factor) + diffuse_factor * light_color[:-1]
    return self_colors