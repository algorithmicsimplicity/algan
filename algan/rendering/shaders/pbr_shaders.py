import torch
import torch.nn.functional as F

from algan.utils.tensor_utils import dot_product


def basic_pbr_shader(vertex_location,
                     vertex_normal,
                     albedo_color,
                     camera_location,
                     light_origin,
                     light_color,
                     light_intensity: float,
                     ambient_light_intensity: float,
                     smoothness: float=0.5,
                     metallicness: float=0.5):
    """Implements a simplified physics-based rendering shader, using smoothness and metallicness
    to define diffuse and specular lighting.

    Parameters
    ----------
    vertex_location : (np.ndarray)
        The 3D location of the vertex to be shaded.
    vertex_normal : (np.ndarray)
        The surface normal vector at the vertex point (need not be normalized).
    albedo_color : (np.ndarray)
        The albedo/base RGBG color of the material.
    camera_location : (np.ndarray)
        The 3D location of the camera/viewer.
    light_origin : (np.ndarray)
        The 3D location of the point light source.
    light_color : (np.ndarray)
        The RGB color of the light.
    light_intensity : float
        The intensity/brightness of the light source.
    ambient_light_intensity : float
        The intensity of ambient scene lighting, ranges from 0 (no light) to 1.
    smoothness : (float)
        The smoothness of the material [0, 1]. 0=rough, 1=smooth.
    metallicness : (float)
        The metallicness property of the material [0, 1]. 0=dielectric, 1=metal.

    Returns
    -------
    np.ndarray
        The final computed RGB color for the vertex.

    """
    # 1. Normalize inputs and calculate core direction vectors
    # Ensure all vectors are numpy arrays for vector operations
    vertex_normal = F.normalize(vertex_normal, p=2, dim=-1)#vertex_normal / F.norm(nornp.linalg.norm(vertex_normal)

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
    attenuation = 1#1.0 / (distance_to_light ** 2 + 1)
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
    F0_dielectric = torch.tensor([0.04, 0.04, 0.04, 0.04], device=vertex_location.device)
    # For metals, F0 is the base color of the metal
    F0 = F0_dielectric * (1-metallicness) + metallicness * albedo_color

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
    specular_brdf = (ndf * fresnel) / (4.0 * (dot_product(vertex_normal, view_dir).relu_()) * NdotL).clamp_min(0.001)
    specular = specular_brdf * radiance * NdotL

    # 6. Diffuse Term (Lambertian)
    # PBR conserves energy: light is either reflected (specular) or refracted (diffuse).
    # The Fresnel term tells us the ratio.
    kS = fresnel  # kS is the specular ratio
    kD = 1.0 - kS  # kD is the diffuse ratio

    # Metals have no diffuse reflection; their color comes from tinted specular highlights.
    # We use the metallicness property to fade out the diffuse component.
    kD *= (1.0 - metallicness)

    # Lambertian diffuse model
    diffuse_brdf = albedo_color# / torch.pi
    diffuse = kD * diffuse_brdf * radiance * NdotL

    # 7. Final Combination
    # The final color is the sum of ambient, diffuse, and specular light.
    # We clip the final color to prevent negative values, although PBR should not produce them.
    final_color = ambient + diffuse + specular
    return final_color.clamp_(min=0, max=1)


def default_shader(vertex_location,
                   vertex_normal,
                   albedo_color,
                   camera_location,
                   light_origin,
                   light_color,
                   light_intensity: float,
                   ambient_light_intensity: float
                   ):
    """Implements just diffuse lighting.

    Parameters
    ----------
    vertex_location : (np.ndarray)
        The 3D location of the vertex to be shaded.
    vertex_normal : (np.ndarray)
        The surface normal vector at the vertex point (need not be normalized).
    albedo_color : (np.ndarray)
        The albedo/base RGBG color of the material.
    camera_location : (np.ndarray)
        The 3D location of the camera/viewer.
    light_origin : (np.ndarray)
        The 3D location of the point light source.
    light_color : (np.ndarray)
        The RGB color of the light.
    light_intensity : float
        The intensity/brightness of the light source.
    ambient_light_intensity : float
        The intensity of ambient scene lighting, ranges from 0 (no light) to 1.

    Returns
    -------
    np.ndarray
        The final computed RGB color for the vertex.

    """

    incidences = F.normalize(vertex_location - light_origin, p=2, dim=-1)
    vertex_normal = F.normalize(vertex_normal, p=2, dim=-1)
    diffuse_factor = (dot_product(-incidences, vertex_normal)).relu_().pow_(5) * 0.5
    return albedo_color * (1-diffuse_factor) + diffuse_factor * light_color


def null_shader(vertex_location,
                   vertex_normal,
                   albedo_color,
                   camera_location,
                   light_origin,
                   light_color,
                   light_intensity: float,
                   ambient_light_intensity: float
                   ):
    return albedo_color