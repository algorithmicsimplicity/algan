"""Light sources for Algan's ray-traced renderer.

Mirrors the Three.js light catalogue: :class:`PointLight`,
:class:`DirectionalLight`, :class:`AmbientLight`, :class:`HemisphereLight`,
:class:`SpotLight` and :class:`RectAreaLight`, each with an ``intensity``
multiplier and (where physical) ``decay``/``distance`` falloff parameters.

Lights are renderable Mobs: their ``location``, ``color``, ``opacity``
and ``intensity`` are animatable like any other mob attribute. The remaining
parameters (``decay``, ``distance``, cone angles, emitter sizes) are plain
per-light constants.

Soft (penumbra) shadows: point / spot lights take a ``shadow_radius`` (the
world-space radius of the emitting disk) and directional lights a
``shadow_angle`` (angular size, degrees); when non-zero and ray-traced shadows
are enabled (:func:`~algan.rendering.raytracing.settings.set_shadows`)
the deterministic tracer fires a fixed fan of shadow rays across the emitter
instead of a single ray, producing smooth penumbras. :class:`RectAreaLight` is
sampled at a fixed grid of emitter points (``samples``), and each row's fan
integrates visibility over its own cell of the rectangle rather than testing
only the cell's centre, so its penumbra is continuous rather than a stack of
hard shadows -- gated by ``area_light_soft_shadows``, at the ray cost noted on
:class:`RectAreaLight`.

Only the default plain :class:`PointLight` is supported by every render path;
the extended light types are rendered by the deterministic (single-sample)
ray tracer with per-fragment shading, which Algan enables automatically when
any extended light is present in the scene.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from algan.animatable_base.mob import Mob
from algan.constants.spatial import ORIGIN, UP
from algan.errors import AlganConfigurationError
from algan.rendering.raytracing import settings as rt_settings
from algan.utils.color_space import srgb_to_linear
from algan.utils.tensor_utils import cast_to_tensor

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


def _validated_intensity(value):
    """Validate an ``intensity`` write, tolerating both scalars and tensors.

    Tensors are expected, not a defensive nicety: after any state
    materialization the attribute holds a ``[T, 1, 1]`` row, and
    ``Animatable.__deepcopy__`` copies every animatable attribute through its
    setter -- so ``light.clone()`` feeds exactly such a tensor through here.
    """
    # The finite/non-negative check is load-bearing, not cosmetic: a light
    # outside its lifespan is made inert by its OPACITY row being zeroed, and
    # intensity only ever reaches the render multiplied by that opacity. A NaN
    # or inf intensity turns 0 * inf into NaN and resurrects emission on frames
    # where the light does not exist. (The intensity timeline itself is not
    # endpoint-masked -- record_end_points is set only for opacity.)
    if torch.is_tensor(value):
        if not bool(torch.isfinite(value).all()):
            raise AlganConfigurationError("intensity must be a finite number")
        if bool((value < 0).any()):
            raise AlganConfigurationError("intensity must be at least 0.0")
        return value
    return _finite_number("intensity", value, minimum=0.0)


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
        Dimensionless multiplier applied to the light's color: at ``2`` the
        light contributes exactly twice the radiance it does at ``1``. Pixel
        values do not double with it, because exposure and the sRGB transfer
        function sit between radiance and the byte written to the frame. Must
        be a finite number of at least ``0.0``. Defaults to ``1.0``. The
        constructor argument is the light's initial value and is not animated;
        animate with assignment after spawn -- see `Animation` below.

    Raises
    ------
    :class:`.AlganConfigurationError`
        If ``intensity`` is not a finite number of at least ``0.0``, on
        construction or on any later write.

    Attributes
    ----------
    intensity
        The light's brightness multiplier: a dimensionless finite number of at
        least ``0.0``, applied to the light's color every frame. Defaults to
        ``1.0``.

    Animation
    ---------
    Writing ``light.intensity = 3`` after spawn is *recorded*: the value
    interpolates from its current value to the target over the current
    context's runtime (1 second by default), exactly like writing ``color``
    -- wrap it in ``with Off():`` to apply instantly instead. Writes made
    before spawn are instant setup, as with ``location``. Like any Mob
    attribute write the change propagates to descendants (lights normally
    have none).
    """

    light_type = LIGHT_POINT

    #: Aux-column range ``(start, stop)`` carrying RADIANCE (not geometry):
    #: those columns must scale with the light's per-frame opacity *and*
    #: intensity at materialization, so a light outside its lifespan is a
    #: genuinely inert all-zero row (its RGB columns already scale with
    #: opacity and intensity). ``None`` when every emitted quantity lives in
    #: the RGB columns.
    _AUX_RADIANCE_COLS = None

    def __init__(self, *args, intensity=1.0, **kwargs):
        # Registered before super().__init__() so Animatable's accessor generation
        # sees it and installs set_intensity / get_intensity like every other
        # animatable attribute.
        self.register_attrs_as_animatable(["intensity"], Light)
        # Into a local, not self.intensity: once the property is attached this
        # assignment would route into set_animated_attribute, which reads
        # state Animatable.__init__ has not built yet.
        intensity = _finite_number("intensity", intensity, minimum=0.0)
        kwargs["add_to_scene"] = False
        super().__init__(*args, **kwargs)
        # _init_default_attr writes the timeline row directly, bypassing the property
        # setter (which cannot run before Animatable.__init__ has built its state), so
        # the constructor validates the value itself, above.
        self._init_default_attr("intensity", cast_to_tensor(intensity))

    def set_animated_attribute(self, attr, value, recursive=True):
        """Animate one animatable attribute to a new value, by name.

        As :meth:`~.Mob.set_animated_attribute`, and additionally the one place
        an ``intensity`` write is checked. Every route to that attribute --
        assignment, ``set_intensity``, :meth:`~.Mob.set` and
        :meth:`~.Mob.set_non_recursive` -- passes through here, so a light
        cannot reach the renderer with a negative or non-finite brightness.

        Raises
        ------
        :class:`.AlganConfigurationError`
            If ``attr`` is ``"intensity"`` and ``value`` is not a finite number
            (or tensor of numbers) of at least ``0.0``.
        """
        if attr == "intensity":
            value = _validated_intensity(value)
        return super().set_animated_attribute(attr, value, recursive=recursive)

    def spawn(self, animate: bool = True):
        """Spawn this light and register it with its owning scene exactly once."""
        result = super().spawn(animate=animate)
        if not hasattr(self.scene, "light_sources"):
            self.scene.light_sources = []
        if not any(light is self for light in self.scene.light_sources):
            self.scene.light_sources.append(self)
        return result

    def _is_extended(self):
        """Whether this light needs the extended (16-column) packed row.
        Plain point lights return False and keep the compact legacy packing
        (which keeps the no-new-features render byte-identical).
        """
        return True

    def _num_samples(self):
        """Number of emitter sample points (rows) this light packs to."""
        return 1

    def _get_sample_positions(self, location):
        """World positions of the emitter samples, ``[T, K, 3]`` for per-frame
        light locations ``location [T, 3]``.
        """
        return location.unsqueeze(-2)

    def _blank_aux(self, location):
        aux = torch.zeros(
            (location.shape[0], self._num_samples(), LIGHT_AUX_COLS),
            dtype=torch.float32,
        )
        aux[..., 0] = self.light_type
        # Power fraction: the share of a whole light each packed row carries
        # (1/K for one of an area light's K emitter samples). No shading stage
        # reads it now -- it told the default shader's base fade that K samples
        # are one light, which the fade works out for itself since it became
        # radiance-weighted (see shading_taichi._light_eval). Still packed: it
        # is part of the row layout, and it is the only place the split is
        # recorded.
        aux[..., 12] = 1.0 / self._num_samples()
        return aux

    def _build_aux(self, location):
        """Internal: pack this light's per-frame parameters for the renderer.

        Subclasses override this to fill in the columns they use. The layout is:

        ====  ==========================================================
        0     light type id
        1     decay exponent (0 = no distance falloff)
        2     distance (range; 0 = infinite)
        3-5   direction (unit; emission dir for directional/spot, up for
              hemisphere, surface normal for area samples)
        6     cos(outer cone angle) (spot)
        7     cos(inner cone angle) (spot)
        8     shadow softness (world radius; directional: tan(angle); area:
              equal-area radius of one emitter cell, see RectAreaLight)
        9-11  ground color RGB (hemisphere) / SH linear coeffs (env) / right
              axis of the emitter rectangle (area);  the
              ground row is decoded to linear light here, then scaled by
              per-frame opacity x intensity downstream at materialization
        12    power fraction of this row (1/K for area samples, else 1)
        ====  ==========================================================

        Parameters
        ----------
        location
            Per-frame light locations, shape ``[T, 3]``.

        Returns
        -------
        torch.Tensor
            Aux columns 3..15 of the packed light row, shape ``[T, K, 13]``.
        """
        return self._blank_aux(location)


class PointLight(Light):
    """Omnidirectional light emitting from a single point.

    Parameters
    ----------
    intensity
        As on :class:`~.Light`: a dimensionless multiplier on the light's
        color, animatable like any Mob attribute. Defaults to ``1.0``.
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

    def __init__(
        self, *args, intensity=1.0, decay=0.0, distance=0.0, shadow_radius=0.0, **kwargs
    ):
        self.decay = _finite_number("decay", decay, minimum=0.0)
        self.distance = _finite_number("distance", distance, minimum=0.0)
        self.shadow_radius = _finite_number("shadow_radius", shadow_radius, minimum=0.0)
        super().__init__(*args, intensity=intensity, **kwargs)

    def _is_extended(self):
        """Whether this light needs the extended packed row.

        Returns
        -------
        bool
            True only when falloff, range or soft shadows are in use; a plain point
            light keeps the compact packing, which keeps renders that use no new
            features byte-identical.
        """
        return self.decay != 0.0 or self.distance != 0.0 or self.shadow_radius != 0.0

    def _build_aux(self, location):
        """Internal: pack this light's falloff, range and shadow radius.

        Parameters
        ----------
        location
            Per-frame light locations, shape ``[T, 3]``.

        Returns
        -------
        torch.Tensor
            Aux columns for the packed light row, shape ``[T, K, 13]``.
        """
        aux = self._blank_aux(location)
        aux[..., 1] = self.decay
        aux[..., 2] = self.distance
        aux[..., 8] = self.shadow_radius
        return aux


class _TargetedLight(Light):
    """Shared behaviour for lights that aim at a target point: the per-frame
    emission direction is ``normalize(target - location)``, so animating the
    light's location (or re-targeting) swings the beam.
    """

    def __init__(self, *args, target=ORIGIN, **kwargs):
        self.target = _as_direction_target(target)
        super().__init__(*args, **kwargs)

    def set_target(self, target):
        """Aim the light at a point, swinging its beam.

        The emission direction is recomputed each frame as
        ``normalize(target - location)``, so a light aimed at a point keeps pointing
        there as it moves.

        Animation
        ---------
        Not animated: the new target applies from this point in the timeline onwards.
        Animate the light's :attr:`~.Mob.location` to swing the beam smoothly.

        Parameters
        ----------
        target
            Point to aim at, shape ``(*, 3)``, or a Mob whose position to aim at.

        Returns
        -------
        :class:`~.Light`
            This light, so calls can be chained.
        """
        self.target = _as_direction_target(target)
        return self

    def _directions(self, location):
        """Unit emission direction per frame, ``[T, 3]``."""
        return F.normalize(self.target.to(location.device) - location, p=2, dim=-1)


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
            "shadow_angle",
            shadow_angle,
            minimum=0.0,
            maximum=180.0,
            maximum_inclusive=False,
        )
        super().__init__(*args, target=target, **kwargs)

    def _build_aux(self, location):
        """Internal: pack this light's emission direction and shadow softness.

        Parameters
        ----------
        location
            Per-frame light locations, shape ``[T, 3]``.

        Returns
        -------
        torch.Tensor
            Aux columns for the packed light row, shape ``[T, K, 13]``.
        """
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

    # Ground color (aux columns 9:12) is emitted radiance: without opacity
    # scaling, a not-yet-spawned or despawned hemisphere light would keep
    # lighting downward-facing surfaces from its aux row alone.
    _AUX_RADIANCE_COLS = (9, 12)

    def __init__(self, *args, ground_color=None, up=UP, **kwargs):
        if not args and "location" not in kwargs:
            kwargs["location"] = ORIGIN
        self.ground_color = ground_color
        # ``up`` is the constructor's name for it; the attribute is spelled
        # differently because ``Mob.up`` is the mob's own up direction.
        self.sky_direction = _as_direction_target(up)
        super().__init__(*args, **kwargs)

    def _build_aux(self, location):
        """Internal: pack this light's up direction and ground color.

        Parameters
        ----------
        location
            Per-frame light locations, shape ``[T, 3]``.

        Returns
        -------
        torch.Tensor
            Aux columns for the packed light row, shape ``[T, K, 13]``.
        """
        aux = self._blank_aux(location)
        aux[..., 3:6] = F.normalize(self.sky_direction, p=2, dim=-1)
        gc = self.ground_color
        if gc is None:
            gc = torch.zeros(3)
        if not torch.is_tensor(gc):
            gc = torch.tensor(gc, dtype=torch.float32)
        # Decoded to linear light here, and left at that: cols 9:12 carry the
        # decoded ground color -- an authored color like any other, and the
        # one radiance-bearing aux column, so it has to make the same trip
        # into linear light the RGB columns do (srgb_to_linear(c * i) is not
        # srgb_to_linear(c) * i). The per-frame opacity and intensity scaling
        # happens downstream at materialization
        # (RenderLoopMixin._materialize_render_state), never here.
        ground = gc.float().reshape(-1)[:3]
        if rt_settings.linear_color_space:
            ground = srgb_to_linear(ground)
        aux[..., 9:12] = ground
        return aux


class SpotLight(_TargetedLight):
    """Cone of light aimed at a target.

    Parameters
    ----------
    target
        World point the cone is aimed at.
    cone_angle
        Half-angle of the cone, in degrees unless ``degrees`` is False
        (default 30).
    degrees
        Whether ``cone_angle`` is in degrees. Defaults to True; pass False to
        give it in radians.
    penumbra
        Portion ``[0, 1]`` of the cone over which the light fades to zero
        at the edge (0 = hard edge).
    decay / distance / shadow_radius
        As on :class:`PointLight`.
    """

    light_type = LIGHT_SPOT

    def __init__(
        self,
        *args,
        target=ORIGIN,
        cone_angle=30.0,
        penumbra=0.0,
        decay=0.0,
        distance=0.0,
        shadow_radius=0.0,
        degrees: bool = True,
        **kwargs,
    ):
        if not degrees:
            cone_angle = math.degrees(
                _finite_number("cone_angle", cone_angle, minimum=0.0)
            )
        self.cone_angle = _finite_number(
            "cone_angle",
            cone_angle,
            minimum=0.0,
            maximum=90.0,
            minimum_inclusive=False,
        )
        self.penumbra = _finite_number("penumbra", penumbra, minimum=0.0, maximum=1.0)
        self.decay = _finite_number("decay", decay, minimum=0.0)
        self.distance = _finite_number("distance", distance, minimum=0.0)
        self.shadow_radius = _finite_number("shadow_radius", shadow_radius, minimum=0.0)
        super().__init__(*args, target=target, **kwargs)

    def _build_aux(self, location):
        """Internal: pack this light's cone angles, falloff and shadow radius.

        Parameters
        ----------
        location
            Per-frame light locations, shape ``[T, 3]``.

        Returns
        -------
        torch.Tensor
            Aux columns for the packed light row, shape ``[T, K, 13]``.
        """
        aux = self._blank_aux(location)
        aux[..., 1] = self.decay
        aux[..., 2] = self.distance
        aux[..., 3:6] = self._directions(location).unsqueeze(-2)
        outer = math.radians(self.cone_angle)
        inner = outer * (1.0 - self.penumbra)
        aux[..., 6] = math.cos(outer)
        # Keep a minimal inner/outer separation so the smoothstep in the
        # kernel never divides by zero.
        aux[..., 7] = (
            math.cos(max(inner, 1e-4)) if self.penumbra > 0 else math.cos(outer) + 1e-4
        )
        aux[..., 8] = self.shadow_radius
        return aux


class RectAreaLight(_TargetedLight):
    """Rectangular area light, sampled deterministically.

    The rectangle is centered on the light's location, faces
    ``normalize(target - location)``, and is expanded at render time into a
    fixed grid of ``samples`` point emitters (each carrying ``1/samples`` of
    the power, with one-sided cosine emission).

    With ray-traced shadows enabled, each emitter row stands for one cell of
    the grid and its shadow fan integrates visibility over that whole cell --
    placing its samples inside the cell, in the light's own plane -- instead
    of testing only the cell's centre point. The penumbra is therefore
    continuous rather than a stack of hard shadows. This costs
    ``SOFT_SHADOW_SAMPLES`` shadow rays per row instead of one, i.e.
    :meth:`_num_samples` ``* SOFT_SHADOW_SAMPLES`` (default 8) per shaded
    fragment while an area light is in the scene; ``samples`` stays the dial
    for both quality and cost. The integration can be turned off, restoring
    one hard ray per row, with
    ``SETTINGS.raytracing.experimental.area_light_soft_shadows``.

    Under the path tracer (``samples_per_pixel > 1``) the rectangle is real
    emissive geometry instead: two triangles the renderer samples over, still
    invisible to the camera and still not an occluder, but hittable by
    bounced rays -- so a mirror shows the light's reflection, and ``samples``
    no longer affects the cost. ``decay`` and ``distance`` mean the same thing
    in both renderers.

    Parameters
    ----------
    width / height
        Size of the rectangle in world units. Defaults to 2 for each.
    target
        World point the rectangle faces. Defaults to the origin.
    samples
        Number of emitter samples, rounded up to a square k x k grid: a k x k
        grid of at least this many cells is laid out over the rectangle.
        Defaults to 4. More samples = finer cells and smoother lighting and
        shadows, linearly more shadow-ray cost.
    decay / distance
        As on :class:`PointLight` (set ``decay=2`` for physical falloff).
        Defaults to no falloff and unlimited range.
    """

    light_type = LIGHT_AREA_SAMPLE

    def __init__(
        self,
        *args,
        width=2.0,
        height=2.0,
        target=ORIGIN,
        samples=4,
        decay=0.0,
        distance=0.0,
        **kwargs,
    ):
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

    def _grid_side(self):
        """Internal: side ``k`` of the square emitter grid."""
        return int(math.ceil(math.sqrt(self.samples)))

    def _num_samples(self):
        """Number of emitter samples this area light packs to.

        Returns
        -------
        int
            ``samples`` rounded up to the next square number, since the emitters are
            laid out on a square grid.
        """
        k = self._grid_side()
        return k * k

    def _rect_axes(self, location):
        """Per-frame (right, up) unit axes of the rectangle, ``[T, 3]`` each."""
        n = self._directions(location)
        ref = UP.reshape(-1)[:3].to(n.device).expand_as(n)
        # Fall back to a different reference axis where the normal is
        # (nearly) parallel to UP.
        parallel = F.cosine_similarity(n, ref, dim=-1).abs() > 0.99
        alt = torch.tensor((1.0, 0.0, 0.0), device=n.device).expand_as(n)
        ref = torch.where(parallel.unsqueeze(-1), alt, ref)
        right = F.normalize(torch.linalg.cross(ref, n, dim=-1), p=2, dim=-1)
        up = torch.linalg.cross(n, right, dim=-1)
        return right, up

    def _get_sample_positions(self, location):
        """Get the world positions of this area light's emitter samples.

        The samples sit at the centres of a square grid covering the rectangle, which
        is what gives the light smooth penumbras.

        Parameters
        ----------
        location
            Per-frame light locations, shape ``[T, 3]``.

        Returns
        -------
        torch.Tensor
            Sample positions, shape ``[T, K, 3]`` where ``K`` is
            :meth:`~.RectAreaLight._num_samples`.
        """
        k = self._grid_side()
        right, up = self._rect_axes(location)
        # Cell-centered k x k grid over the rectangle.
        u = (torch.arange(k, dtype=torch.float32) + 0.5) / k - 0.5
        offs_u, offs_v = torch.meshgrid(u, u, indexing="ij")
        offs = torch.stack((offs_u.flatten(), offs_v.flatten()), -1)  # [K, 2]
        offs = offs.to(location.device)
        return (
            location.unsqueeze(-2)
            + offs[..., :1] * self.width * right.unsqueeze(-2)
            + offs[..., 1:] * self.height * up.unsqueeze(-2)
        )

    def _build_aux(self, location):
        """Internal: pack this light's falloff, range and surface normal, and
        -- when ``area_light_soft_shadows`` is on -- each emitter row's cell
        geometry for the soft-shadow fans: aux 6/7 the cell's half-extents
        along the rectangle's ``right``/``up``, aux 8 the cell's equal-area
        radius (the fans' gate and their isotropic fallback), aux 9-11 the
        rectangle's ``right`` unit axis (``up`` is recovered in-kernel as
        ``cross(normal, right)``).

        Parameters
        ----------
        location
            Per-frame light locations, shape ``[T, 3]``.

        Returns
        -------
        torch.Tensor
            Aux columns for the packed light row, shape ``[T, K, 13]``.
        """
        aux = self._blank_aux(location)
        aux[..., 1] = self.decay
        aux[..., 2] = self.distance
        aux[..., 3:6] = self._directions(location).unsqueeze(-2)
        if rt_settings.area_light_soft_shadows:
            k = self._grid_side()
            right, _ = self._rect_axes(location)
            hu = self.width / (2.0 * k)
            hv = self.height / (2.0 * k)
            aux[..., 6] = hu
            aux[..., 7] = hv
            # Column 8 is the shadow-radius column every extended row carries;
            # for an area row it holds the EQUAL-AREA disk radius of the cell,
            # sqrt(4*hu*hv/pi). It serves twice: it is the fans' ``radius > 0``
            # gate that turns the multi-sample fan on, and the isotropic
            # fallback for any reader that knows a shadow radius but not the
            # rectangle. Equal-area is the honest scalar stand-in for the cell.
            aux[..., 8] = math.sqrt(4.0 * hu * hv / math.pi)
            aux[..., 9:12] = right.unsqueeze(-2)
        return aux


def light_is_extended(light):
    """True when ``light`` needs the extended packed-light row (any light
    beyond a plain, falloff-free point light).
    """
    fn = getattr(light, "_is_extended", None)
    return bool(fn()) if fn is not None else False
