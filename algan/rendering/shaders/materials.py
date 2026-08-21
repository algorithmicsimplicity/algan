"""Three.js-style material classes for Algan.

These mirror the Three.js *mesh* materials -- the same material types, property
names and default settings -- so a material can be configured the familiar way
and applied to a mob with
:meth:`~algan.animatable_base.mob_materials.MobMaterialsMixin.set_material`.
One deliberate deviation:
``color`` defaults to ``None``, meaning "keep the mob's existing colour",
whereas Three.js defaults it to white (which would silently repaint any mob
the material is applied to)::

    from algan import Sphere, MeshStandardMaterial

    Sphere().set_material(MeshStandardMaterial(metalness=1.0, roughness=0.2)).spawn()

Each :class:`~algan.rendering.shaders.materials.Material` is a thin configuration
object: it knows its lighting
:attr:`~algan.rendering.shaders.materials.Material.shader` (a function from :mod:`algan.rendering.shaders.material_shaders`)
and, via
:meth:`~algan.rendering.shaders.materials.Material.get_shader_param_values`,
the per-vertex shader parameters that ``set_material`` registers as animatable
attributes on the mob. So after applying
a material you can animate e.g. ``mob.roughness = 0.1`` or ``mob.emissive_intensity = 3``.

Limitations
-----------
Algan shades per vertex and has no UV / image-sampling pipeline, so every
texture / image-based property (``map``, ``normalMap``, ``roughnessMap``,
``envMap``, ``matcap``, ``gradientMap``, ...) is accepted for API parity but not
sampled; a one-time warning is emitted when such a slot is set. ``wireframe``,
``vertexColors`` and non-default ``side`` are likewise unsupported. The matcap,
normal and depth materials use documented approximations (see
:mod:`algan.rendering.shaders.material_shaders`).
"""

from __future__ import annotations

import math
import warnings

from algan.constants.color import Color
from algan.rendering.shaders import material_shaders as ms
from algan.utils.tensor_utils import cast_to_tensor

__all__ = [
    "FrontSide",
    "BackSide",
    "DoubleSide",
    "Material",
    "UnlitMaterial",
    "DiffuseMaterial",
    "SpecularMaterial",
    "PBRMaterial",
    "AdvancedPBRMaterial",
    "MeshBasicMaterial",
    "MeshLambertMaterial",
    "MeshPhongMaterial",
    "MeshStandardMaterial",
    "MeshPhysicalMaterial",
    "MeshToonMaterial",
    "MeshNormalMaterial",
    "MeshMatcapMaterial",
    "MeshDepthMaterial",
]

# Three.js Side constants.
FrontSide = 0
BackSide = 1
DoubleSide = 2

# Image-based property names accepted for API parity but never sampled.
_TEXTURE_SLOTS = frozenset(
    {
        "map",
        "alpha_map",
        "ao_map",
        "env_map",
        "light_map",
        "bump_map",
        "normal_map",
        "displacement_map",
        "roughness_map",
        "metalness_map",
        "emissive_map",
        "specular_map",
        "gradient_map",
        "matcap",
        "clearcoat_map",
        "clearcoat_roughness_map",
        "clearcoat_normal_map",
        "sheen_color_map",
        "sheen_roughness_map",
        "transmission_map",
        "thickness_map",
        "iridescence_map",
        "iridescence_thickness_map",
        "specular_intensity_map",
        "specular_color_map",
        "normal_scale",
        "displacement_scale",
        "displacement_bias",
    }
)


def _to_rgb(value):
    """Parse a Three.js-style colour into a 3-channel rgb tensor ``[..., 3]``."""
    return _to_color5(value)[..., :3]


def _to_color5(value):
    """Parse a colour into a 5-channel :class:`Color` ``[R, G, B, glow, opacity]``.

    Accepts a hex int (``0xff0000``), a hex string (``"#ff0000"``), an RGB tuple
    in ``[0, 1]``, or an existing :class:`Color` / tensor.
    """
    if isinstance(value, Color):
        return value
    if isinstance(value, bool):  # guard: bool is a subclass of int
        raise TypeError(f"invalid colour value: {value!r}")
    if isinstance(value, int):
        return Color("#%06X" % (value & 0xFFFFFF))
    if isinstance(value, str):
        return Color(value)
    t = cast_to_tensor(value).reshape(-1)
    return Color((float(t[0]), float(t[1]), float(t[2])))


class Material:
    """Base class holding the Three.js shared material properties + defaults.

    Subclasses set :attr:`shader`, list their animatable shader parameters and
    implement
    :meth:`~algan.rendering.shaders.materials.Material.get_shader_param_values`.
    """

    #: Lighting shader backing this material (a plain function).
    shader = staticmethod(ms.basic_material_shader)
    #: Whether the material's ``color`` should drive the mob's base colour.
    #: Even then, the default ``color=None`` means "keep the mob's existing
    #: colour" -- unlike Three.js, where an unset material colour is white --
    #: so ``Sphere(color=RED).set_material(MeshPhysicalMaterial())`` stays red.
    applies_color = True

    def __init__(
        self,
        color=None,
        *,
        opacity=1.0,
        transparent=False,
        visible=True,
        side=FrontSide,
        flat_shading=False,
        vertex_colors=False,
        wireframe=False,
        tone_mapped=True,
        **texture_kwargs,
    ):
        self.color = color
        self.opacity = opacity
        self.transparent = transparent
        self.visible = visible
        self.side = side
        self.flat_shading = flat_shading
        self.vertex_colors = vertex_colors
        self.wireframe = wireframe
        self.tone_mapped = tone_mapped
        # Stash any texture / unsupported slots so set_material can warn about them.
        self._textures = {k: v for k, v in texture_kwargs.items() if v is not None}
        unexpected = set(texture_kwargs) - _TEXTURE_SLOTS
        if unexpected:
            raise TypeError(
                f"{type(self).__name__} got unexpected keyword(s): {sorted(unexpected)}"
            )

    # -- shader parameters ------------------------------------------------
    def get_shader_param_values(self):
        """Map of ``{shader_param_name: value}`` matching this material's shader
        signature. Base materials expose no extra parameters.
        """
        return {}

    def _flat(self):
        return 1.0 if self.flat_shading else 0.0

    # -- warnings ---------------------------------------------------------
    def emit_warnings(self):
        """Warn (once per call) about properties Algan's renderer cannot honour."""
        msgs = []
        if self._textures:
            msgs.append(
                f"texture/image properties {sorted(self._textures)} are not "
                "sampled by Algan's per-vertex renderer and are ignored"
            )
        if self.wireframe:
            msgs.append("wireframe is not supported and is ignored")
        if self.vertex_colors:
            msgs.append("vertex_colors is not supported and is ignored")
        if self.side != FrontSide:
            msgs.append(
                "non-default 'side' (BackSide/DoubleSide) is not supported; "
                "Algan renders all faces. Whether a back-facing hit is LIT "
                "from the viewer's side is decided by the geometry instead, "
                "through Mob.two_sided"
            )
        for m in msgs:
            warnings.warn(f"{type(self).__name__}: {m}", stacklevel=2)

    def __repr__(self):
        return f"{type(self).__name__}()"


class UnlitMaterial(Material):
    """Unlit material: renders the flat base colour, ignoring lights."""

    shader = staticmethod(ms.basic_material_shader)


MeshBasicMaterial = UnlitMaterial


class DiffuseMaterial(Material):
    """Lambertian (diffuse-only) shading plus emissive."""

    shader = staticmethod(ms.lambert_shader)

    def __init__(
        self,
        color=None,
        *,
        emissive=0x000000,
        emissive_intensity=1.0,
        env_map_intensity=1.0,
        **kwargs,
    ):
        super().__init__(color, **kwargs)
        self.emissive = emissive
        self.emissive_intensity = emissive_intensity
        self.env_map_intensity = env_map_intensity

    def get_shader_param_values(self):
        return {
            "emissive": _to_rgb(self.emissive),
            "emissive_intensity": self.emissive_intensity,
            "flat_shading": self._flat(),
            "env_map_intensity": self.env_map_intensity,
        }


MeshLambertMaterial = DiffuseMaterial


class SpecularMaterial(Material):
    """Blinn-Phong shading: diffuse + specular highlight + emissive."""

    shader = staticmethod(ms.phong_shader)

    def __init__(
        self,
        color=None,
        *,
        emissive=0x000000,
        emissive_intensity=1.0,
        specular=0x111111,
        shininess=30.0,
        env_map_intensity=1.0,
        **kwargs,
    ):
        super().__init__(color, **kwargs)
        self.emissive = emissive
        self.emissive_intensity = emissive_intensity
        self.specular = specular
        self.shininess = shininess
        self.env_map_intensity = env_map_intensity

    def get_shader_param_values(self):
        return {
            "emissive": _to_rgb(self.emissive),
            "emissive_intensity": self.emissive_intensity,
            "specular": _to_rgb(self.specular),
            "shininess": self.shininess,
            "flat_shading": self._flat(),
            "env_map_intensity": self.env_map_intensity,
        }


MeshPhongMaterial = SpecularMaterial


class PBRMaterial(Material):
    """Metalness/roughness physically-based (Cook-Torrance) material."""

    shader = staticmethod(ms.standard_shader)

    def __init__(
        self,
        color=None,
        *,
        roughness=1.0,
        metalness=0.0,
        emissive=0x000000,
        emissive_intensity=1.0,
        env_map_intensity=1.0,
        **kwargs,
    ):
        super().__init__(color, **kwargs)
        self.roughness = roughness
        self.metalness = metalness
        self.emissive = emissive
        self.emissive_intensity = emissive_intensity
        self.env_map_intensity = env_map_intensity

    def get_shader_param_values(self):
        return {
            "roughness": self.roughness,
            "metalness": self.metalness,
            "emissive": _to_rgb(self.emissive),
            "emissive_intensity": self.emissive_intensity,
            "env_map_intensity": self.env_map_intensity,
            "flat_shading": self._flat(),
        }


MeshStandardMaterial = PBRMaterial


class AdvancedPBRMaterial(MeshStandardMaterial):
    """Extends :class:`MeshStandardMaterial` with clearcoat, sheen, ior-driven
    specular, ray-traced transmission, and approximate iridescence.
    """

    shader = staticmethod(ms.physical_shader)

    def __init__(
        self,
        color=None,
        *,
        clearcoat=0.0,
        clearcoat_roughness=0.0,
        ior=1.5,
        reflectivity=None,
        specular_intensity=1.0,
        specular_color=0xFFFFFF,
        sheen=0.0,
        sheen_roughness=1.0,
        sheen_color=0x000000,
        transmission=0.0,
        thickness=0.0,
        attenuation_color=0xFFFFFF,
        attenuation_distance=math.inf,
        iridescence=0.0,
        iridescence_ior=1.3,
        **kwargs,
    ):
        super().__init__(color, **kwargs)
        self.clearcoat = clearcoat
        self.clearcoat_roughness = clearcoat_roughness
        # Three.js exposes ``reflectivity`` as a backwards-compatible alias
        # for dielectric IOR rather than as an independent mirror control.
        # Preserve that API: an explicitly supplied reflectivity updates IOR;
        # otherwise the effective reflectivity is derived from IOR.
        if reflectivity is None:
            self.ior = ior
            self.reflectivity = 2.5 * (ior - 1.0) / (ior + 1.0)
        else:
            self.reflectivity = reflectivity
            self.ior = (1.0 + 0.4 * reflectivity) / (1.0 - 0.4 * reflectivity)
        self.specular_intensity = specular_intensity
        self.specular_color = specular_color
        self.sheen = sheen
        self.sheen_roughness = sheen_roughness
        self.sheen_color = sheen_color
        self.transmission = transmission
        # Stored for API parity; not used by the per-vertex approximation.
        self.thickness = thickness
        self.attenuation_color = attenuation_color
        self.attenuation_distance = attenuation_distance
        self.iridescence = iridescence
        self.iridescence_ior = iridescence_ior

    def get_shader_param_values(self):
        return {
            "roughness": self.roughness,
            "metalness": self.metalness,
            "emissive": _to_rgb(self.emissive),
            "emissive_intensity": self.emissive_intensity,
            "env_map_intensity": self.env_map_intensity,
            "flat_shading": self._flat(),
            "ior": self.ior,
            "specular_intensity": self.specular_intensity,
            "specular_color": _to_rgb(self.specular_color),
            "clearcoat": self.clearcoat,
            "clearcoat_roughness": self.clearcoat_roughness,
            "sheen": self.sheen,
            "sheen_roughness": self.sheen_roughness,
            "sheen_color": _to_rgb(self.sheen_color),
            "transmission": self.transmission,
            "iridescence": self.iridescence,
        }


MeshPhysicalMaterial = AdvancedPBRMaterial


class MeshToonMaterial(Material):
    """Cel-shaded (banded diffuse) material plus emissive.

    Three.js drives the bands with ``gradientMap``; since textures are not
    sampled, the band count is controlled by the Algan-specific ``bands``
    argument (default 3).
    """

    shader = staticmethod(ms.toon_shader)

    def __init__(
        self,
        color=None,
        *,
        emissive=0x000000,
        emissive_intensity=1.0,
        bands=3.0,
        **kwargs,
    ):
        super().__init__(color, **kwargs)
        self.emissive = emissive
        self.emissive_intensity = emissive_intensity
        self.bands = bands

    def get_shader_param_values(self):
        return {
            "emissive": _to_rgb(self.emissive),
            "emissive_intensity": self.emissive_intensity,
            "num_bands": self.bands,
            "flat_shading": self._flat(),
        }


class MeshNormalMaterial(Material):
    """Encodes the surface normal as RGB. Does not use a base colour."""

    shader = staticmethod(ms.normal_shader)
    applies_color = False

    def get_shader_param_values(self):
        return {"flat_shading": self._flat()}


class MeshMatcapMaterial(Material):
    """Material-capture shading. The matcap image is not sampled; a default
    view-facing approximation is used (tinted by the base colour).
    """

    shader = staticmethod(ms.matcap_shader)

    def get_shader_param_values(self):
        return {"flat_shading": self._flat()}


class MeshDepthMaterial(Material):
    """Renders camera distance as grayscale (near=bright, far=dark)."""

    shader = staticmethod(ms.depth_shader)
    applies_color = False

    def __init__(self, color=None, *, near=0.1, far=100.0, **kwargs):
        super().__init__(color, **kwargs)
        self.near = near
        self.far = far

    def get_shader_param_values(self):
        return {"near": self.near, "far": self.far}
