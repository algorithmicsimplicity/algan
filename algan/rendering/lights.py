"""Light sources for Algan's ray-traced renderer.

Mirrors the Three.js light catalogue: :class:`PointLight`,
:class:`DirectionalLight`, :class:`AmbientLight`, :class:`HemisphereLight`,
:class:`SpotLight` and :class:`RectAreaLight`, each with an ``intensity``
multiplier and (where physical) ``decay``/``distance`` falloff parameters.

Lights are :class:`~.Renderable` mobs: their ``location``, ``color`` and
``opacity`` are animatable like any other mob. The extra parameters
(``intensity``, ``decay``, cone angles, ...) are plain per-light constants.

Soft (penumbra) shadows: point / spot lights take a ``shadow_radius`` (the
world-space radius of the emitting disk) and directional lights a
``shadow_angle`` (angular size, degrees); when non-zero and ray-traced shadows
are enabled (:func:`~algan.rendering.raytracing.settings.set_ray_traced_shadows`)
the deterministic tracer fires a fixed fan of shadow rays across the emitter
instead of a single ray, producing smooth penumbras. :class:`RectAreaLight` is
sampled at a fixed grid of emitter points (``samples``), so both its lighting
and its shadows are naturally soft.

Only the default plain :class:`PointLight` is supported by every render path;
the extended light types are rendered by the deterministic (single-sample)
ray tracer with per-fragment shading, which Algan enables automatically when
any extended light is present in the scene.
"""
import math

import torch
import torch.nn.functional as F

from algan.animatable_base.mob import Mob
from algan.constants.spatial import ORIGIN, UP
from algan.errors import AlganConfigurationError

__all__ = [
    "Light",
    "PointLight",
    "DirectionalLight",
    "AmbientLight",
    "HemisphereLight",
    "SpotLight",
    "RectAreaLight",
]

# Light type ids (column 3 of the packed light row; see
# scene_builder._pack_lights for the full row layout).
LIGHT_POINT = 0.0
LIGHT_DIRECTIONAL = 1.0
LIGHT_AMBIENT = 2.0
LIGHT_HEMISPHERE = 3.0
LIGHT_SPOT = 4.0
LIGHT_AREA_SAMPLE = 5.0
LIGHT_ENV_SH = 6.0

# Number of aux columns following the RGB color in a packed light row
# (packed row width 16 = 3 color + 13 aux).
LIGHT_AUX_COLS = 13



def _finite_number(
    name,
    value,
    *,
    minimum=None,
    maximum=None,
    minimum_inclusive=True,
    maximum_inclusive=True,
):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AlganConfigurationError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise AlganConfigurationError(f"{name} must be a finite number")
    if minimum is not None:
        invalid = result < minimum if minimum_inclusive else result <= minimum
        if invalid:
            relation = "at least" if minimum_inclusive else "greater than"
            raise AlganConfigurationError(f"{name} must be {relation} {minimum}")
    if maximum is not None:
        invalid = result > maximum if maximum_inclusive else result >= maximum
        if invalid:
            relation = "at most" if maximum_inclusive else "less than"
            raise AlganConfigurationError(f"{name} must be {relation} {maximum}")
    return result


def _positive_sample_count(value):
    if isinstance(value, bool):
        raise AlganConfigurationError("samples must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AlganConfigurationError("samples must be a positive integer") from exc
    try:
        exact = float(value) == result
    except (TypeError, ValueError, OverflowError):
        exact = False
    if result < 1 or not exact:
        raise AlganConfigurationError("samples must be a positive integer")
    return result


def _as_direction_target(target):
    t = target
    if not torch.is_tensor(t):
        t = torch.tensor(t, dtype=torch.float32)
    return t.float().reshape(-1)[:3]


class Light(Mob):
    """Base class of all light sources.

    Parameters
    ----------
    intensity
        Scalar multiplier applied to the light's color.
    """

    light_type = LIGHT_POINT

    def __init__(self, *args, intensity=1.0, **kwargs):
        self.intensity = _finite_number("intensity", intensity, minimum=0.0)
        kwargs["add_to_scene"] = False
        super().__init__(*args, **kwargs)

    def set_intensity(self, intensity):
        """Set the non-negative light intensity and return this light."""
        self.intensity = _finite_number("intensity", intensity, minimum=0.0)
        return self

    def spawn(self, animate: bool = True):
        """Spawn this light and register it with its owning scene exactly once."""
        result = super().spawn(animate=animate)
        if not hasattr(self.scene, "light_sources"):
            self.scene.light_sources = []
        if not any(light is self for light in self.scene.light_sources):
            self.scene.light_sources.append(self)
        return result

    def is_extended(self):
        """Whether this light needs the extended (16-column) packed row.
        Plain point lights return False and keep the compact legacy packing
        (which keeps the no-new-features render byte-identical)."""
        return True

    def num_samples(self):
        """Number of emitter sample points (rows) this light packs to."""
        return 1

    def get_sample_positions(self, location):
        """World positions of the emitter samples, ``[T, K, 3]`` for per-frame
        light locations ``location [T, 3]``."""
        return location.unsqueeze(-2)

    def _blank_aux(self, location):
        aux = torch.zeros((location.shape[0], self.num_samples(),
                           LIGHT_AUX_COLS), dtype=torch.float32)
        aux[..., 0] = self.light_type
        # Power fraction: the share of a whole light each packed row carries
        # (1/K for one of an area light's K emitter samples). Consumed by the
        # legacy lerp-based default shader so a many-sample light displaces at
        # most one whole light's worth of base colour.
        aux[..., 12] = 1.0 / self.num_samples()
        return aux

    def build_aux(self, location):
        """Aux columns 3..15 of the packed light row, ``[T, K, 13]``:

        ==  =========================================================
        0   light type id
        1   decay exponent (0 = no distance falloff)
        2   distance (range; 0 = infinite)
        3-5 direction (unit; emission dir for directional/spot, up for
            hemisphere, surface normal for area samples)
        6   cos(outer cone angle) (spot)
        7   cos(inner cone angle) (spot)
        8   shadow softness (world radius; directional: tan(angle))
        9-11 ground color RGB (hemisphere) / SH linear coeffs (env)
        12  power fraction of this row (1/K for area samples, else 1)
        ==  =========================================================
        """
        return self._blank_aux(location)


class PointLight(Light):
    """Omnidirectional light emitting from a single point.

    Parameters
    ----------
    intensity
        Color multiplier.
    decay
        Distance falloff exponent (0 = none -- the legacy Algan default;
        2 = physically correct inverse-square).
    distance
        Maximum range of the light (0 = unlimited). When set, the light
        fades out smoothly toward this distance.
    shadow_radius
        World-space radius of the emitting disk used for soft shadows
        (0 = sharp/hard shadows).
    """

    light_type = LIGHT_POINT

    def __init__(self, *args, intensity=1.0, decay=0.0, distance=0.0,
                 shadow_radius=0.0, **kwargs):
        self.decay = _finite_number("decay", decay, minimum=0.0)
        self.distance = _finite_number("distance", distance, minimum=0.0)
        self.shadow_radius = _finite_number(
            "shadow_radius", shadow_radius, minimum=0.0
        )
        super().__init__(*args, intensity=intensity, **kwargs)

    def is_extended(self):
        return (self.decay != 0.0 or self.distance != 0.0
                or self.shadow_radius != 0.0)

    def build_aux(self, location):
        aux = self._blank_aux(location)
        aux[..., 1] = self.decay
        aux[..., 2] = self.distance
        aux[..., 8] = self.shadow_radius
        return aux


class _TargetedLight(Light):
    """Shared behaviour for lights that aim at a target point: the per-frame
    emission direction is ``normalize(target - location)``, so animating the
    light's location (or re-targeting) swings the beam."""

    def __init__(self, *args, target=ORIGIN, **kwargs):
        self.target = _as_direction_target(target)
        super().__init__(*args, **kwargs)

    def set_target(self, target):
        self.target = _as_direction_target(target)
        return self

    def _directions(self, location):
        """Unit emission direction per frame, ``[T, 3]``."""
        return F.normalize(self.target.to(location.device) - location,
                           p=2, dim=-1)


class DirectionalLight(_TargetedLight):
    """Sun-like light: parallel rays along ``normalize(target - location)``,
    no distance falloff.

    Parameters
    ----------
    target
        World point the light shines toward (default: the origin).
    shadow_angle
        Angular size of the emitter in degrees (like the sun's ~0.5); when
        non-zero, shadows get soft edges.
    """

    light_type = LIGHT_DIRECTIONAL

    def __init__(self, *args, target=ORIGIN, shadow_angle=0.0, **kwargs):
        self.shadow_angle = _finite_number(
            "shadow_angle", shadow_angle, minimum=0.0, maximum=180.0,
            maximum_inclusive=False,
        )
        super().__init__(*args, target=target, **kwargs)

    def build_aux(self, location):
        aux = self._blank_aux(location)
        aux[..., 3:6] = self._directions(location).unsqueeze(-2)
        aux[..., 8] = math.tan(math.radians(self.shadow_angle) * 0.5)
        return aux


class AmbientLight(Light):
    """Uniform, direction-less illumination applied to every surface."""

    light_type = LIGHT_AMBIENT

    def __init__(self, *args, **kwargs):
        if not args and "location" not in kwargs:
            kwargs["location"] = ORIGIN
        super().__init__(*args, **kwargs)


class HemisphereLight(Light):
    """Sky/ground gradient ambient: surfaces facing ``up`` receive the light's
    ``color`` (sky), surfaces facing away receive ``ground_color``.

    Parameters
    ----------
    ground_color
        Color of the illumination coming from below.
    up
        The sky direction (default ``UP``).
    """

    light_type = LIGHT_HEMISPHERE

    def __init__(self, *args, ground_color=None, up=UP, **kwargs):
        if not args and "location" not in kwargs:
            kwargs["location"] = ORIGIN
        self.ground_color = ground_color
        self.up = _as_direction_target(up)
        super().__init__(*args, **kwargs)

    def build_aux(self, location):
        aux = self._blank_aux(location)
        aux[..., 3:6] = F.normalize(self.up, p=2, dim=-1)
        gc = self.ground_color
        if gc is None:
            gc = torch.zeros(3)
        if not torch.is_tensor(gc):
            gc = torch.tensor(gc, dtype=torch.float32)
        aux[..., 9:12] = gc.float().reshape(-1)[:3] * self.intensity
        return aux


class SpotLight(_TargetedLight):
    """Cone of light aimed at a target.

    Parameters
    ----------
    target
        World point the cone is aimed at.
    angle
        Half-angle of the cone in degrees (default 30).
    penumbra
        Portion ``[0, 1]`` of the cone over which the light fades to zero
        at the edge (0 = hard edge).
    decay / distance / shadow_radius
        As on :class:`PointLight`.
    """

    light_type = LIGHT_SPOT

    def __init__(self, *args, target=ORIGIN, angle=30.0, penumbra=0.0,
                 decay=0.0, distance=0.0, shadow_radius=0.0, **kwargs):
        self.angle = _finite_number(
            "angle", angle, minimum=0.0, maximum=90.0,
            minimum_inclusive=False,
        )
        self.penumbra = _finite_number(
            "penumbra", penumbra, minimum=0.0, maximum=1.0
        )
        self.decay = _finite_number("decay", decay, minimum=0.0)
        self.distance = _finite_number("distance", distance, minimum=0.0)
        self.shadow_radius = _finite_number(
            "shadow_radius", shadow_radius, minimum=0.0
        )
        super().__init__(*args, target=target, **kwargs)

    def build_aux(self, location):
        aux = self._blank_aux(location)
        aux[..., 1] = self.decay
        aux[..., 2] = self.distance
        aux[..., 3:6] = self._directions(location).unsqueeze(-2)
        outer = math.radians(self.angle)
        inner = outer * (1.0 - self.penumbra)
        aux[..., 6] = math.cos(outer)
        # Keep a minimal inner/outer separation so the smoothstep in the
        # kernel never divides by zero.
        aux[..., 7] = math.cos(max(inner, 1e-4)) if self.penumbra > 0 \
            else math.cos(outer) + 1e-4
        aux[..., 8] = self.shadow_radius
        return aux


class RectAreaLight(_TargetedLight):
    """Rectangular area light, sampled deterministically.

    The rectangle is centered on the light's location, faces
    ``normalize(target - location)``, and is expanded at render time into a
    fixed grid of ``samples`` point emitters (each carrying ``1/samples`` of
    the power, with one-sided cosine emission). Both the lighting and -- with
    ray-traced shadows enabled -- the penumbras are therefore smooth, with a
    smoothness set by ``samples``.

    Parameters
    ----------
    width / height
        Size of the rectangle in world units.
    target
        World point the rectangle faces.
    samples
        Number of emitter samples (rounded up to a k x k grid). More samples
        = smoother penumbras, linearly more shadow-ray cost.
    decay / distance
        As on :class:`PointLight` (set ``decay=2`` for physical falloff).
    """

    light_type = LIGHT_AREA_SAMPLE

    def __init__(self, *args, width=2.0, height=2.0, target=ORIGIN,
                 samples=4, decay=0.0, distance=0.0, **kwargs):
        self.width = _finite_number(
            "width", width, minimum=0.0, minimum_inclusive=False
        )
        self.height = _finite_number(
            "height", height, minimum=0.0, minimum_inclusive=False
        )
        self.samples = _positive_sample_count(samples)
        self.decay = _finite_number("decay", decay, minimum=0.0)
        self.distance = _finite_number("distance", distance, minimum=0.0)
        super().__init__(*args, target=target, **kwargs)

    def num_samples(self):
        k = int(math.ceil(math.sqrt(self.samples)))
        return k * k

    def _rect_axes(self, location):
        """Per-frame (right, up) unit axes of the rectangle, ``[T, 3]`` each."""
        n = self._directions(location)
        ref = UP.reshape(-1)[:3].to(n.device).expand_as(n)
        # Fall back to a different reference axis where the normal is
        # (nearly) parallel to UP.
        parallel = (F.cosine_similarity(n, ref, dim=-1).abs() > 0.99)
        alt = torch.tensor((1.0, 0.0, 0.0), device=n.device).expand_as(n)
        ref = torch.where(parallel.unsqueeze(-1), alt, ref)
        right = F.normalize(torch.linalg.cross(ref, n, dim=-1), p=2, dim=-1)
        up = torch.linalg.cross(n, right, dim=-1)
        return right, up

    def get_sample_positions(self, location):
        k = int(math.ceil(math.sqrt(self.samples)))
        right, up = self._rect_axes(location)
        # Cell-centered k x k grid over the rectangle.
        u = (torch.arange(k, dtype=torch.float32) + 0.5) / k - 0.5
        offs_u, offs_v = torch.meshgrid(u, u, indexing="ij")
        offs = torch.stack((offs_u.flatten(), offs_v.flatten()), -1)  # [K, 2]
        offs = offs.to(location.device)
        return (location.unsqueeze(-2)
                + offs[..., :1] * self.width * right.unsqueeze(-2)
                + offs[..., 1:] * self.height * up.unsqueeze(-2))

    def build_aux(self, location):
        aux = self._blank_aux(location)
        aux[..., 1] = self.decay
        aux[..., 2] = self.distance
        aux[..., 3:6] = self._directions(location).unsqueeze(-2)
        return aux


def light_is_extended(light):
    """True when ``light`` needs the extended packed-light row (any light
    beyond a plain, falloff-free point light)."""
    fn = getattr(light, "is_extended", None)
    return bool(fn()) if fn is not None else False
