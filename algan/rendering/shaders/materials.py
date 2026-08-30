"""Three.js-style material classes for Algan.

These mirror the Three.js *mesh* materials -- the same material types, property
names and default settings -- so a material can be configured the familiar way
and applied to a mob with
:meth:`~algan.animatable_base.mob_materials.MobMaterialsMixin.set_material`.
One deliberate deviation:
``color`` defaults to ``None``, meaning "keep the mob's existing color",
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

Texture maps
------------
``set_material`` forwards the image slots the renderer has a sampler for --
``map``, ``normal_map``, ``roughness_map`` and ``metalness_map`` -- onto the
geometry, which is where Algan's texture pipeline lives. Each one takes a path
or an ``[H, W, C]`` image and is sampled bilinearly per fragment in the trace
kernel::

    Sphere().set_material(
        MeshStandardMaterial(map="earth.png", roughness_map="ocean_gloss.png")
    )

Sampling needs per-vertex UVs, so this reaches a
:class:`~algan.mobs.surfaces.surface.Surface` (and its subclasses --
:class:`~.Sphere`, :class:`~.Cylinder`, :class:`~.Torus`, :class:`~.ImageMob`,
...) or a :class:`~algan.mobs.three_d_models.mesh.TriangleMesh` built with
``uvs``. On any other Mob -- a :class:`~.Polyhedron`, a
:class:`~.Cube` -- the maps are ignored, with a warning saying so.

A forwarded map is **static**: unlike the scalar properties a material installs
(``mob.roughness`` and the rest), it is not an animatable attribute, and
setting one warns to that effect. The one exception is ``map`` on a Surface,
which lands on the animatable
:attr:`~algan.mobs.surfaces.surface.Surface.color_texture` and so warns not at
all.

Limitations
-----------
The image slots with no channel in the renderer (``env_map``, ``matcap``,
``gradient_map``, ``ao_map``, ``transmission_map``, ...) are still accepted for
API parity and dropped, with a warning naming them. ``wireframe``,
``vertexColors`` and non-default ``side``
are likewise unsupported. The matcap, normal and depth materials use documented
approximations (see :mod:`algan.rendering.shaders.material_shaders`). Every
built-in material class shades per fragment in the render kernel; only a
*custom* per-vertex shader (``set_shader`` with a plain function) is baked into
vertex colors, which is what costs it every light beyond a plain
:class:`~.PointLight`, all shadows, and an environment map's diffuse
contribution. Combining one with a lighting rig that asks for any of those
warns, both where the shader is set and once per render.
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
    "ManimMaterial",
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

# Image slots the renderer has a sampler for. ``set_material`` forwards these
# onto the geometry (``Mob._accept_material_textures``), which is where Algan's
# texture pipeline lives: a Surface or a TriangleMesh carries the UVs, and the
# maps are then sampled bilinearly per fragment inside the trace kernel. The
# value each one drives is named beside it.
#
# Three.js reads ``roughnessMap`` from an image's GREEN channel and
# ``metalnessMap`` from its BLUE one, so one packed occlusion/roughness/
# metalness image drives both; a single-channel image is used as-is. Algan's
# "reflectivity" is the same quantity Three.js calls metalness (see
# Model3D._apply_pbr_material, which already reads a glTF
# metallic-roughness map that way).
_MAP_SLOT_PROPERTIES = {
    "roughness_map": ("roughness", 1),
    "metalness_map": ("reflectivity", 2),
}
_FORWARDED_TEXTURE_SLOTS = frozenset({"map", "normal_map"} | set(_MAP_SLOT_PROPERTIES))

# Image slots with no channel anywhere in the renderer. Accepted so a Three.js
# material transcribes without edits, then dropped -- there is nothing to
# forward them to. (The kernel does have an unused transmission channel, but no
# geometry exposes a way to author one, so ``transmission_map`` stays here.)
_UNSUPPORTED_TEXTURE_SLOTS = frozenset(
    {
        "alpha_map",
        "ao_map",
        "env_map",
        "light_map",
        "bump_map",
        "displacement_map",
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
    }
)

# Scalars that ride in the same ``**texture_kwargs`` bag because Three.js
# groups them with the maps they modify. ``normal_scale`` is honoured (it
# scales a forwarded normal map's tangential components); the displacement
# pair is not -- no map moves a vertex in Algan.
_UNSUPPORTED_SCALAR_SLOTS = frozenset({"displacement_scale", "displacement_bias"})

_TEXTURE_SLOTS = (
    _FORWARDED_TEXTURE_SLOTS
    | _UNSUPPORTED_TEXTURE_SLOTS
    | _UNSUPPORTED_SCALAR_SLOTS
    | {"normal_scale"}
)


#: Channel each per-texel material property occupies in the packed material
#: texture the trace kernel samples, and the bit that marks it texture-driven
#: (``material_texture_flags``; an unset bit keeps the per-vertex value, see
#: ``_flat_triangle_material`` in ``wavefront_kernels_taichi``). Channel 3 is
#: transmission, which the kernel reads but no geometry authors yet.
_MATERIAL_TEXTURE_CHANNELS = {"reflectivity": 0, "roughness": 1, "refractive_index": 2}


def _as_texture_stack(tex, channels):
    """Normalize a user-supplied texture to ``[T, W, H, channels]``.

    Accepts ``[W, H]`` (single-channel maps only), ``[W, H, channels]`` or
    ``[T, W, H, channels]``; ``W`` is the ``u`` axis, ``H`` the ``v`` axis of
    the surface's intrinsic coordinates.
    """
    import torch

    tex = torch.as_tensor(tex).float()
    if tex.dim() == 2:
        if channels != 1:
            raise ValueError(
                f"a 2-D texture is only valid for single-channel "
                f"properties, expected {channels} channels"
            )
        tex = tex.unsqueeze(-1)
    if tex.dim() == 3:
        tex = tex.unsqueeze(0)
    if tex.dim() != 4 or tex.shape[-1] != channels:
        raise ValueError(
            f"texture must have shape [W, H, {channels}] or "
            f"[T, W, H, {channels}], got {tuple(tex.shape)}"
        )
    return tex


def _pack_material_texture(properties, device):
    """Combine per-property maps into one ``[T, W, H, 5]`` material texture at
    the finest common resolution, plus the bitmask of which channels are
    texture-driven.

    ``properties`` maps the names of :data:`_MATERIAL_TEXTURE_CHANNELS` to
    ``[W, H, 1]`` (or ``[T, W, H, 1]``) maps. Returns ``(texture, flags)``;
    channels without a map keep their per-vertex value in-kernel.
    """
    import torch
    import torch.nn.functional as F

    texs = {k: _as_texture_stack(v, 1).to(device) for k, v in properties.items()}
    T = max(t.shape[0] for t in texs.values())
    W = max(t.shape[1] for t in texs.values())
    H = max(t.shape[2] for t in texs.values())
    combined = torch.zeros((T, W, H, 5), device=device)
    flags = 0
    for name, t in texs.items():
        if t.shape[1:3] != (W, H):
            t = F.interpolate(
                t.permute(0, 3, 1, 2),
                size=(W, H),
                mode="bilinear",
                align_corners=True,
            ).permute(0, 2, 3, 1)
        slot = _MATERIAL_TEXTURE_CHANNELS[name]
        combined[..., slot] = t.expand(T, W, H, 1)[..., 0]
        flags |= 1 << slot
    return combined, flags


def _load_material_image(value, slot):
    """A Material image slot as a float ``[H, W, C]`` image, rows top-down.

    Takes what every other image entry point in Algan takes -- a path (resolved
    against the working directory and then the main script's directory), a
    numpy array, or a tensor -- so ``MeshStandardMaterial(map="brick.png")``
    works. This is deliberately the *image* convention, not the ``[W, H, C]``
    ``(u, v)`` layout :class:`~algan.mobs.surfaces.surface.Surface`'s
    ``color_texture`` and friends take: a material slot holds a picture, and
    ``[H, W]`` and ``[W, H]`` are indistinguishable once handed over.
    """
    from algan.utils.file_utils import get_image

    image = get_image(value)
    if image.dim() != 3:
        raise ValueError(
            f"{slot} must be an image [H, W, C] (or a path to one), got shape "
            f"{tuple(image.shape)}"
        )
    return image


def _normalize_forwarded_maps(textures):
    """The forwardable image slots of ``textures``, in the engine's texture
    layout, ready to hand to a geometry's ``_accept_material_textures``.

    Returns ``{slot_name: tensor}`` holding ``map`` as ``[W, H, 5]``
    (RGB + glow + alpha), ``normal_map`` as ``[W, H, 3]`` with components in
    ``[-1, 1]``, and each entry of :data:`_MAP_SLOT_PROPERTIES` as
    ``[W, H, 1]``. Slots that are absent stay absent, and unsupported ones are
    dropped here rather than reported -- :meth:`Material.emit_warnings` is what
    speaks about them.
    """
    from algan.mobs.three_d_models.mesh import image_to_normal_map, image_to_texture_map

    maps = {}
    if "map" in textures:
        maps["map"] = image_to_texture_map(_load_material_image(textures["map"], "map"))
    if "normal_map" in textures:
        normal = image_to_normal_map(
            _load_material_image(textures["normal_map"], "normal_map")
        )
        # Three.js's normalScale scales the tangential (x, y) components; the
        # z component is what is left of a unit normal, so the kernel's own
        # normalization finishes the job.
        scale = textures.get("normal_scale")
        if scale is not None and float(scale) != 1.0:
            normal = normal.clone()
            normal[..., :2] *= float(scale)
        maps["normal_map"] = normal
    for slot, (_, channel) in _MAP_SLOT_PROPERTIES.items():
        if slot not in textures:
            continue
        image = _load_material_image(textures[slot], slot)
        if image.shape[-1] > 1:
            image = image[..., channel : channel + 1]
        # Same spatial transform image_to_texture_map applies, so every map a
        # material forwards lines up with the same (v-flipped) UVs.
        maps[slot] = image.transpose(-3, -2).flip(-2).contiguous()
    return maps


# -- lighting a vertex bake cannot answer -----------------------------------
# A material is shaded in the render kernel only when its shader has an
# in-kernel port (raytracing.settings._core_shader_ids), which every built-in
# material class has. Everything else -- a custom per-vertex shader handed to
# ``set_shader`` as a plain function -- is baked into vertex colors before the
# frame renders, and that bake sees only plain point lights
# (RayTracedTrianglePrimitive._shade_vertex_colors skips every light carrying
# ``_render_aux``). The same shaders pack the unlit in-kernel material id,
# which the sheet resolve refuses to build shadow events for, so they receive
# no shadows either. Neither is recoverable at render time, so the honest
# thing is to say so at the point the combination is authored.
#
# (MeshToonMaterial, MeshNormalMaterial, MeshMatcapMaterial and
# MeshDepthMaterial used to be in this bake-only group; they have in-kernel
# stages now and light like every other core material.)
#
# Deliberately a warning rather than raytracing.settings.report_unsupported_features:
# that policy defaults to raising, and custom per-vertex shaders are shipped,
# documented and render perfectly well -- what they drop is part of the
# lighting rig, not the render.
_PER_FRAGMENT_ADVICE = (
    "Shade per fragment instead -- with any of the Three.js-style material "
    "classes (MeshLambertMaterial, MeshPhongMaterial, MeshStandardMaterial, "
    "MeshPhysicalMaterial, ...), or with set_fragment_shader()."
)


def _shades_per_fragment(shader):
    """Whether ``shader`` is evaluated in the render kernel rather than baked
    into vertex colors.

    ``None`` counts (no shader, so nothing is baked and nothing is lost), as do
    the ``set_fragment_shader`` pipelines, which always shade in-kernel.
    """
    if shader is None:
        return True
    if getattr(shader, "_frag_pipeline_id", None) is not None:
        return True
    from algan.rendering.raytracing.settings import _shader_is_core

    return _shader_is_core(shader)


def _lighting_beyond_vertex_bake(lights=(), *, shadows=None, environment_map=None):
    """The parts of a lighting rig that only per-fragment shading delivers.

    Each entry is a phrase naming what is asked for and what becomes of it, for
    a mob whose shading is baked at vertices. ``shadows`` defaults to the live
    ``SETTINGS.raytracing.shadows``.
    """
    from algan.rendering.lights import light_is_extended
    from algan.rendering.raytracing import settings as rt_settings

    if shadows is None:
        shadows = rt_settings.shadows

    # Phrased so each entry reads correctly after both "a custom per-vertex
    # shader's shading is baked ..., so" and
    # "N Mob(s) ... bake into vertex colors, so".
    features = []
    extended = sorted({type(_).__name__ for _ in lights if light_is_extended(_)})
    if extended:
        features.append(
            f"lights beyond a plain PointLight are skipped ({', '.join(extended)}) "
            "-- a vertex bake sees only point lights with no decay, distance or "
            "shadow_radius"
        )
    if shadows:
        features.append(
            "no shadow is received, whatever SETTINGS.raytracing.shadows is set to"
        )
    if environment_map is not None:
        features.append("the environment map's diffuse lighting is not applied")
    return tuple(features)


def _to_rgb(value):
    """Parse a Three.js-style color into a 3-channel rgb tensor ``[..., 3]``."""
    return _to_color5(value)[..., :3]


def _attenuation_sigma(attenuation_color, attenuation_distance):
    """The Beer-Lambert absorption coefficient ``sigma_rgb`` of a transmissive
    medium, from glTF ``KHR_materials_volume`` fields::

        sigma = -ln(clamp(linear(attenuation_color), 1e-6, 1)) / attenuation_distance

    so a ray crossing path length ``d`` leaves ``exp(-sigma * d)`` of its
    throughput: at ``d == attenuation_distance`` that is exactly
    ``attenuation_color``, and white attenuates nothing at any distance.
    Packed as a coefficient rather than as the two authored fields because it
    makes "no absorption" the all-zeros value, which is what a zero-padded
    custom-pipeline material block must mean (see ``shading_taichi.MAT_W``).
    No attenuation (white color, or an infinite / non-positive distance)
    therefore packs as zeros.

    The log is taken in the working color space: authored color is
    display-referred, and under the linear working space it is decoded first --
    the same decode ``scene_builder._decode_merged_colors`` gives every other
    color, gated on the same setting -- so three.js's "decode at Color, take
    the log of linear" behaviour is reproduced.
    """
    import torch

    if (
        attenuation_distance is None
        or math.isinf(attenuation_distance)
        or attenuation_distance <= 0
    ):
        return torch.zeros(3)
    c = _to_rgb(attenuation_color).detach().float()
    from algan.rendering.raytracing import settings as rt_settings
    from algan.utils.color_space import srgb_to_linear

    if rt_settings.linear_color_space:
        c = srgb_to_linear(c)
    return -torch.log(c.clamp(1e-6, 1.0)) / float(attenuation_distance)


def _to_color5(value):
    """Parse a color into a 5-channel :class:`Color` ``[R, G, B, glow, opacity]``.

    Accepts a hex int (``0xff0000``), a hex string (``"#ff0000"``), an RGB tuple
    in ``[0, 1]``, or an existing :class:`Color` / tensor.
    """
    if isinstance(value, Color):
        return value
    if isinstance(value, bool):  # guard: bool is a subclass of int
        raise TypeError(f"invalid color value: {value!r}")
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
    #: Whether the material's ``color`` should drive the mob's base color.
    #: Even then, the default ``color=None`` means "keep the mob's existing
    #: color" -- unlike Three.js, where an unset material color is white --
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
        # Stash the image slots so set_material can forward the ones the
        # renderer samples onto the geometry, and warn about the rest.
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
    def emit_warnings(self, forwarded=None):
        """Warn (once per call) about properties Algan's renderer cannot honour.

        Parameters
        ----------
        forwarded
            What ``set_material`` managed to hand to the geometry, as
            ``{slot_name: is_animatable}`` (the union over the Mobs it was
            applied to). A slot that reached a geometry is sampled, so it is
            reported as *static* rather than ignored -- or not reported at all
            when the geometry it landed on made it animatable. ``None`` means
            nothing was forwarded, which is what a bare
            ``Material(...).emit_warnings()`` describes.
        """
        forwarded = {} if forwarded is None else forwarded
        msgs = []
        static = sorted(k for k, animatable in forwarded.items() if not animatable)
        if static:
            msgs.append(
                f"texture maps {static} are sampled per fragment, but they are "
                "static: unlike the material's scalar properties (mob.roughness "
                "and the rest) they are not animatable attributes, so the image "
                "the Mob spawns with is the image it keeps"
            )
        dropped = sorted(
            (set(self._textures) & _FORWARDED_TEXTURE_SLOTS) - set(forwarded)
        )
        if dropped:
            msgs.append(
                f"texture maps {dropped} are ignored: they are sampled against "
                "per-vertex UVs, which only a Surface (Sphere, Cylinder, "
                "Torus, ImageMob, ...) or a TriangleMesh built with `uvs` "
                "carries. Not every part of this Mob that renders is one, so "
                "the maps are dropped rather than applied to some of it"
            )
        unsupported = sorted(set(self._textures) & _UNSUPPORTED_TEXTURE_SLOTS)
        if unsupported:
            msgs.append(
                f"image properties {unsupported} have no channel in Algan's "
                "renderer and are ignored"
            )
        scalars = sorted(set(self._textures) & _UNSUPPORTED_SCALAR_SLOTS)
        if scalars:
            msgs.append(f"{scalars} are not supported and are ignored")
        if "normal_scale" in self._textures and "normal_map" not in forwarded:
            msgs.append(
                "normal_scale scales a normal map's tangential components, and "
                "no normal_map reached the geometry, so it does nothing"
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
    """Unlit material: renders the flat base color, ignoring lights."""

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


class ManimMaterial(Material):
    """Manim's default 3-D shading: one achromatic offset per light, nothing else.

    Per light it adds an achromatic ``0.5 * (n . to_light) ** 3`` offset
    (halved when back-facing) to the base color -- no ambient, no specular,
    no falloff.

    Reproduces Manim's ``get_shaded_rgb`` exactly under the rig
    :meth:`~.Scene.use_manim_defaults` installs (one white intensity-1 point
    light); see :func:`~.material_shaders.manim_shader` for the precise
    conditions. It has no properties of its own beyond the shared
    :class:`Material` ones, but
    :meth:`~.Material.get_shader_param_values` still carries
    ``flat_shading``: the packed material block is written name by name, so
    an empty dict would leave that slot at its default and
    ``ManimMaterial(flat_shading=True)`` would silently do nothing.

    :meth:`~.Scene.use_manim_defaults` installs this material as the
    default for 3-D Mobs alongside the rig it sets up.
    """

    shader = staticmethod(ms.manim_shader)

    def get_shader_param_values(self):
        return {"flat_shading": self._flat()}


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
            # Beer-Lambert absorption coefficient (see _attenuation_sigma);
            # zeros when the material does not attenuate.
            "attenuation_sigma": _attenuation_sigma(
                self.attenuation_color, self.attenuation_distance
            ),
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
    """Encodes the surface normal as RGB. Does not use a base color."""

    shader = staticmethod(ms.normal_shader)
    applies_color = False

    def get_shader_param_values(self):
        return {"flat_shading": self._flat()}


class MeshMatcapMaterial(Material):
    """Material-capture shading. The matcap image is not sampled; a default
    view-facing approximation is used (tinted by the base color).
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
