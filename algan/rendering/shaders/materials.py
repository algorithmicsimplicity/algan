"""Three.js-style material classes for Algan.

These mirror the Three.js *mesh* materials -- the same material types, property
names and default settings -- so a material can be configured the familiar way
and applied to a mob with :meth:`Mob.set_material`::

    from algan import Sphere, MeshStandardMaterial

    Sphere().set_material(MeshStandardMaterial(metalness=1.0, roughness=0.2)).spawn()

Each :class:`Material` is a thin configuration object: it knows its lighting
:attr:`shader` (a function from :mod:`algan.rendering.shaders.material_shaders`)
and, via :meth:`get_shader_param_values`, the per-vertex shader parameters that
``set_material`` registers as animatable attributes on the mob. So after applying
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

import math
import warnings

from algan.constants.color import Color, WHITE
from algan.utils.tensor_utils import cast_to_tensor
from algan.rendering.shaders import material_shaders as ms

__all__ = [
    "FrontSide",
    "BackSide",
    "DoubleSide",
    "Material",
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
_TEXTURE_SLOTS = frozenset({
    "map", "alphaMap", "aoMap", "envMap", "lightMap", "bumpMap", "normalMap",
    "displacementMap", "roughnessMap", "metalnessMap", "emissiveMap",
    "specularMap", "gradientMap", "matcap", "clearcoatMap",
    "clearcoatRoughnessMap", "clearcoatNormalMap", "sheenColorMap",
    "sheenRoughnessMap", "transmissionMap", "thicknessMap", "iridescenceMap",
    "iridescenceThicknessMap", "specularIntensityMap", "specularColorMap",
    "normalScale", "displacementScale", "displacementBias",
})


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
    implement :meth:`get_shader_param_values`.
    """

    #: Lighting shader backing this material (a plain function).
    shader = staticmethod(ms.basic_material_shader)
    #: Whether the material's ``color`` should drive the mob's base colour.
    applies_color = True

    def __init__(
        self,
        color=None,
        *,
        opacity=1.0,
        transparent=False,
        visible=True,
        side=FrontSide,
        flatShading=False,
        vertexColors=False,
        wireframe=False,
        toneMapped=True,
        **texture_kwargs,
    ):
        self.color = WHITE if color is None else color
        self.opacity = opacity
        self.transparent = transparent
        self.visible = visible
        self.side = side
        self.flatShading = flatShading
        self.vertexColors = vertexColors
        self.wireframe = wireframe
        self.toneMapped = toneMapped
        # Stash any texture / unsupported slots so set_material can warn about them.
        self._textures = {k: v for k, v in texture_kwargs.items() if v is not None}
        unexpected = set(texture_kwargs) - _TEXTURE_SLOTS
        if unexpected:
            raise TypeError(
                f"{type(self).__name__} got unexpected keyword(s): "
                f"{sorted(unexpected)}"
            )

    # -- shader parameters ------------------------------------------------
    def get_shader_param_values(self):
        """Map of ``{shader_param_name: value}`` matching this material's shader
        signature. Base materials expose no extra parameters."""
        return {}

    def _flat(self):
        return 1.0 if self.flatShading else 0.0

    # -- ray traced renderer (single source of truth for transport params) --
    def physical_surface_params(self):
        """``(reflectivity, roughness, refractive_index)`` -- the ray-transport
        surface parameters this material routes onto the ray traced renderer
        (see :meth:`Mob.set_material`), unifying the standalone
        :func:`~algan.rendering.raytracing.primitives.set_reflectivity` /
        :func:`~algan.rendering.raytracing.primitives.set_roughness` /
        :func:`~algan.rendering.raytracing.primitives.set_refractive_index`
        setters into the material.

        ``reflectivity`` (= metalness) and ``roughness`` are shaded per ray hit
        by the physical path tracer; ``refractive_index`` (> 0 only for a
        transmissive material) makes the surface refract (glass) in the general
        wavefront tracer. The base default is a non-metallic, fully rough,
        non-refractive (diffuse) surface; PBR materials override it. Emissive
        colour, clearcoat, sheen, etc. are not yet routed to the path tracer
        (they remain a vertex-shading feature)."""
        return 0.0, 1.0, 0.0

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
        if self.vertexColors:
            msgs.append("vertexColors is not supported and is ignored")
        if self.side != FrontSide:
            msgs.append(
                "non-default 'side' (BackSide/DoubleSide) is not supported; "
                "Algan renders all faces"
            )
        for m in msgs:
            warnings.warn(f"{type(self).__name__}: {m}", stacklevel=2)

    def __repr__(self):
        return f"{type(self).__name__}()"


class MeshBasicMaterial(Material):
    """Unlit material: renders the flat base colour, ignoring lights."""

    shader = staticmethod(ms.basic_material_shader)


class MeshLambertMaterial(Material):
    """Lambertian (diffuse-only) shading plus emissive."""

    shader = staticmethod(ms.lambert_shader)

    def __init__(
        self,
        color=None,
        *,
        emissive=0x000000,
        emissiveIntensity=1.0,
        envMapIntensity=1.0,
        **kwargs,
    ):
        super().__init__(color, **kwargs)
        self.emissive = emissive
        self.emissiveIntensity = emissiveIntensity
        self.envMapIntensity = envMapIntensity

    def get_shader_param_values(self):
        return {
            "emissive": _to_rgb(self.emissive),
            "emissive_intensity": self.emissiveIntensity,
            "flat_shading": self._flat(),
            "env_map_intensity": self.envMapIntensity,
        }


class MeshPhongMaterial(Material):
    """Blinn-Phong shading: diffuse + specular highlight + emissive."""

    shader = staticmethod(ms.phong_shader)

    def __init__(
        self,
        color=None,
        *,
        emissive=0x000000,
        emissiveIntensity=1.0,
        specular=0x111111,
        shininess=30.0,
        envMapIntensity=1.0,
        **kwargs,
    ):
        super().__init__(color, **kwargs)
        self.emissive = emissive
        self.emissiveIntensity = emissiveIntensity
        self.specular = specular
        self.shininess = shininess
        self.envMapIntensity = envMapIntensity

    def get_shader_param_values(self):
        return {
            "emissive": _to_rgb(self.emissive),
            "emissive_intensity": self.emissiveIntensity,
            "specular": _to_rgb(self.specular),
            "shininess": self.shininess,
            "flat_shading": self._flat(),
            "env_map_intensity": self.envMapIntensity,
        }

    def physical_surface_params(self):
        # Map the Blinn-Phong exponent to a GGX-like roughness so the glossy
        # highlight survives in the path tracer; dielectric (no metalness),
        # opaque (no refraction).
        return 0.0, float((2.0 / (float(self.shininess) + 2.0)) ** 0.5), 0.0


class MeshStandardMaterial(Material):
    """Metalness/roughness physically-based (Cook-Torrance) material."""

    shader = staticmethod(ms.standard_shader)

    def __init__(
        self,
        color=None,
        *,
        roughness=1.0,
        metalness=0.0,
        emissive=0x000000,
        emissiveIntensity=1.0,
        envMapIntensity=1.0,
        **kwargs,
    ):
        super().__init__(color, **kwargs)
        self.roughness = roughness
        self.metalness = metalness
        self.emissive = emissive
        self.emissiveIntensity = emissiveIntensity
        self.envMapIntensity = envMapIntensity

    def get_shader_param_values(self):
        return {
            "roughness": self.roughness,
            "metalness": self.metalness,
            "emissive": _to_rgb(self.emissive),
            "emissive_intensity": self.emissiveIntensity,
            "env_map_intensity": self.envMapIntensity,
            "flat_shading": self._flat(),
        }

    def physical_surface_params(self):
        # Metalness-driven mirror + GGX roughness; opaque (no refraction).
        return float(self.metalness), float(self.roughness), 0.0


class MeshPhysicalMaterial(MeshStandardMaterial):
    """Extends :class:`MeshStandardMaterial` with clearcoat, sheen, ior-driven
    specular and (approximate) transmission / iridescence."""

    shader = staticmethod(ms.physical_shader)

    def __init__(
        self,
        color=None,
        *,
        clearcoat=0.0,
        clearcoatRoughness=0.0,
        ior=1.5,
        reflectivity=0.5,
        specularIntensity=1.0,
        specularColor=0xFFFFFF,
        sheen=0.0,
        sheenRoughness=1.0,
        sheenColor=0x000000,
        transmission=0.0,
        thickness=0.0,
        attenuationColor=0xFFFFFF,
        attenuationDistance=math.inf,
        iridescence=0.0,
        iridescenceIOR=1.3,
        **kwargs,
    ):
        super().__init__(color, **kwargs)
        self.clearcoat = clearcoat
        self.clearcoatRoughness = clearcoatRoughness
        self.ior = ior
        self.reflectivity = reflectivity
        self.specularIntensity = specularIntensity
        self.specularColor = specularColor
        self.sheen = sheen
        self.sheenRoughness = sheenRoughness
        self.sheenColor = sheenColor
        self.transmission = transmission
        # Stored for API parity; not used by the per-vertex approximation.
        self.thickness = thickness
        self.attenuationColor = attenuationColor
        self.attenuationDistance = attenuationDistance
        self.iridescence = iridescence
        self.iridescenceIOR = iridescenceIOR

    def get_shader_param_values(self):
        return {
            "roughness": self.roughness,
            "metalness": self.metalness,
            "emissive": _to_rgb(self.emissive),
            "emissive_intensity": self.emissiveIntensity,
            "env_map_intensity": self.envMapIntensity,
            "flat_shading": self._flat(),
            "ior": self.ior,
            "specular_intensity": self.specularIntensity,
            "specular_color": _to_rgb(self.specularColor),
            "clearcoat": self.clearcoat,
            "clearcoat_roughness": self.clearcoatRoughness,
            "sheen": self.sheen,
            "sheen_roughness": self.sheenRoughness,
            "sheen_color": _to_rgb(self.sheenColor),
            "transmission": self.transmission,
            "iridescence": self.iridescence,
        }

    def physical_surface_params(self):
        # As MeshStandardMaterial, but a transmissive material also routes its
        # index of refraction so it renders as glass in the general wavefront
        # tracer. ior is only emitted when there is transmission to carry it
        # (an opaque physical material stays non-refractive, ior 0).
        refractive_index = float(self.ior) if float(self.transmission) > 0.0 else 0.0
        return float(self.metalness), float(self.roughness), refractive_index


class MeshToonMaterial(Material):
    """Cel-shaded (banded diffuse) material plus emissive.

    Three.js drives the bands with ``gradientMap``; since textures are not
    sampled, the band count is controlled by the Algan-specific ``bands``
    argument (default 3)."""

    shader = staticmethod(ms.toon_shader)

    def __init__(
        self,
        color=None,
        *,
        emissive=0x000000,
        emissiveIntensity=1.0,
        bands=3.0,
        **kwargs,
    ):
        super().__init__(color, **kwargs)
        self.emissive = emissive
        self.emissiveIntensity = emissiveIntensity
        self.bands = bands

    def get_shader_param_values(self):
        return {
            "emissive": _to_rgb(self.emissive),
            "emissive_intensity": self.emissiveIntensity,
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
    view-facing approximation is used (tinted by the base colour)."""

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
