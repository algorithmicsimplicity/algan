"""Compatibility Mobs backed by Algan's vendored Manim Community geometry.

The compatibility layer is intentionally narrow at the rendering boundary:
Manim constructs and manipulates its normal Mobject graph, then :class:`ManimMob`
converts the resulting cubic Bezier circuits into Algan render primitives.  This
lets Algan expose Manim's large collection of composite Mobjects without
copying the geometry algorithms, while all animation, materials and rendering
remain native to Algan.
"""

from __future__ import annotations

import contextlib
import inspect
import re
from collections.abc import Mapping
from functools import wraps
from typing import Any

import manim as _manim
import numpy as np
import torch

from algan.animatable_base.mob import Mob
from algan.animation_timeline.animation_contexts import Off
from algan.constants.color import Color, to_color
from algan.constants.spatial import OUTWARD
from algan.mobs.bezier_circuit import BezierCircuitCubic
from algan.mobs.group import Group
from algan.mobs.image_mob import ImageMob
from algan.mobs.manim_mob import ManimMob
from algan.settings import SETTINGS

# Every other manim entry point in Algan goes through a ``LazyModule`` whose
# ``extras`` pull in the svg cache, which is what redirects manim's Tex/text
# scratch directories out of the CWD and repairs its single-level
# ``tex_dir.mkdir()``. This module imports manim eagerly, so without this the
# compatibility wrappers -- ``MathTex``, ``Title``, anything reaching LaTeX
# without a ``Tex`` being built first -- construct their manim source object
# against manim's unpatched default ``media/Tex`` and die on a clean directory.
# Importing the cache writes nothing to disk; the directories are made on first
# use.
from algan.utils import manim_svg_cache as _manim_svg_cache  # noqa: F401
from algan.utils.tensor_utils import cast_to_tensor

# Public compatibility classes are registered by Manim class name so methods
# such as Axes.plot can convert their returned Mobjects back to the most useful
# Algan wrapper type.
_MANIM_WRAPPER_REGISTRY: dict[str, type[ManimCompatMob]] = {}


def _tensor_to_manim(value: torch.Tensor):
    value = value.detach().cpu()
    if value.ndim == 0:
        return value.item()
    # Algan carries a leading batch dimension that Manim's geometry does not:
    # Manim points are ``(N, 3)`` and single vectors are ``(3,)``.  Drop the
    # batch axes so that, for example, a ``(1, 1, 3)`` location arrives as the
    # one point Manim expects.  Without this, Manim's in-place point arithmetic
    # (``mob.points += vector``) raises a non-broadcastable operand error.
    while value.ndim > 1 and value.shape[0] == 1:
        value = value[0]
    return value.numpy()


def _color_to_manim(value: Color):
    rgb = value.rgb.detach().cpu().reshape(-1, 3)[0].clamp(0, 1)
    return "#" + "".join(f"{round(float(x) * 255):02X}" for x in rgb)


def _algan_bezier_to_manim(mob: BezierCircuitCubic):
    """Create a Manim VMobject snapshot from an Algan cubic circuit."""
    points = mob.control_points.location[0].detach().cpu().numpy()
    if type(mob).__name__ == "Line" and hasattr(mob, "get_start"):
        start = mob.get_start().detach().reshape(-1, 3)[0].cpu().numpy()
        end = mob.get_end().detach().reshape(-1, 3)[0].cpu().numpy()
        result = _manim.Line(start, end)
    else:
        result = _manim.VMobject()
        result.set_points(points)

    color = mob.color[0].reshape(-1, mob.color.shape[-1])[0]
    fill_color = "#" + "".join(
        f"{round(float(x) * 255):02X}" for x in color[:3].detach().cpu().clamp(0, 1)
    )
    fill_opacity = float((color[-1] * mob.opacity[0].reshape(-1)[0]).detach().cpu())
    result.set_fill(fill_color, opacity=fill_opacity if mob.filled else 0.0)

    border = mob.stroke_color[0].reshape(-1, mob.stroke_color.shape[-1])[0]
    stroke_color = "#" + "".join(
        f"{round(float(x) * 255):02X}" for x in border[:3].detach().cpu().clamp(0, 1)
    )
    stroke_opacity = float(border[-1].detach().cpu())
    # Algan units out, Manim units in: twice, the export side of the
    # conversion ``algan.manim`` owns.
    stroke_width = (
        float(mob.stroke_width[0].reshape(-1)[0].detach().cpu())
        * SETTINGS.style.manim_stroke_width_ratio
    )
    result.set_stroke(stroke_color, width=stroke_width, opacity=stroke_opacity)
    return result


def _row_backed_animatable_state(mob: Mob) -> dict[str, torch.Tensor]:
    """Snapshot this Mob's concrete timeline-backed animatable attributes.

    ``animatable_attrs`` also contains derived properties such as
    ``scale_coefficient`` whose value is computed from another timeline-backed
    attribute.  Those properties must not be copied independently when merging
    a Manim edit back into Algan, otherwise one semantic change can be applied
    twice.  Restricting the snapshot to attributes with rows owned by ``mob``
    gives us the actual state that ``become`` can overwrite.
    """
    timeline_manager = mob.scene.timeline_manager
    state = {}
    for attr in dict.fromkeys(mob.animatable_attrs):
        timeline = timeline_manager.attr_to_timeline.get(attr)
        if timeline is None or mob.id not in timeline.mob_id_to_inds:
            continue
        state[attr] = mob.get_animated_attribute(
            attr, include_descendants=False, copy=True
        )
    return state


def _same_tensor_value(left: torch.Tensor, right: torch.Tensor) -> bool:
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    return bool(torch.equal(left, right))


def _preserve_algan_state_unchanged_by_manim(
    current: Mob,
    before: Mob,
    after: Mob,
):
    """Three-way merge native Algan state into a converted Manim result.

    ``before`` and ``after`` are conversions of the backing Manim object before
    and after one compatibility operation.  Whenever Manim left an animatable
    attribute unchanged, copy the exact current Algan value onto ``after``.
    This preserves recursive parent edits to *all* timeline-backed attributes,
    including state that Manim has no representation for (for example glow),
    while still allowing a delegated Manim operation to intentionally change
    attributes it owns.

    The merge is structural and only pairs nodes that existed before the Manim
    operation.  Newly-added Manim submobjects retain the state produced by Manim,
    which is the same behavior as adding a new Algan child after a previous
    recursive parent edit: that historical edit is not retroactively replayed on
    the new child.
    """
    current_state = _row_backed_animatable_state(current)
    before_state = _row_backed_animatable_state(before)
    after_state = _row_backed_animatable_state(after)

    for attr, current_value in current_state.items():
        before_value = before_state.get(attr)
        after_value = after_state.get(attr)
        if before_value is None or after_value is None:
            continue
        if not _same_tensor_value(before_value, after_value):
            continue
        if current_value.shape != after_value.shape:
            # A structural Manim operation can change a point batch.  Do not
            # force old per-point state onto a differently-sized new geometry.
            continue
        after._setattr_without_record(attr, current_value)

    for current_child, before_child, after_child in zip(
        current.children, before.children, after.children
    ):
        _preserve_algan_state_unchanged_by_manim(
            current_child, before_child, after_child
        )


def _uniform_color_and_opacity(color, opacity):
    """Return one Manim-compatible RGB string and effective opacity."""
    color = color.reshape(-1, color.shape[-1])[0]
    opacity = opacity.reshape(-1)[0]
    rgb = "#" + "".join(
        f"{round(float(x) * 255):02X}" for x in color[:3].detach().cpu().clamp(0, 1)
    )
    alpha = float((color[-1] * opacity).detach().cpu())
    return rgb, alpha


def _sync_image_geometry_to_manim(algan_mob: ImageMob, manim_mob):
    """Push an ImageMob's current affine pose into its Manim ImageMobject."""
    location = algan_mob.location.reshape(-1, 3)[0].detach().cpu().numpy()
    basis = algan_mob.basis.reshape(-1, 3, 3)[0].detach().cpu().numpy()
    right, up = basis[0], basis[1]
    # Manim ImageMobject point order is top-left, top-right, bottom-left,
    # bottom-right.  Algan's ImageMob basis rows are its half-width/half-height
    # axes, so this preserves translation, rotation and non-uniform scale.
    manim_mob.points = np.stack(
        (
            location - right + up,
            location + right + up,
            location - right - up,
            location + right - up,
        )
    )
    if hasattr(manim_mob, "set_opacity"):
        opacity = float(algan_mob.opacity.reshape(-1)[0].detach().cpu())
        manim_mob.set_opacity(opacity)


def _sync_manim_node_from_algan(algan_mob: Mob, manim_mob):
    """Mutate one retained Manim node to match its current Algan counterpart.

    The object itself is deliberately retained: replacing it with a generic
    VMobject would discard semantic state used by APIs such as ``Axes.plot``.
    We therefore update only geometry/style fields in place and recurse through
    the already-corresponding subobject graph.
    """
    if isinstance(manim_mob, _manim.ImageMobject) and isinstance(algan_mob, ImageMob):
        _sync_image_geometry_to_manim(algan_mob, manim_mob)
        return

    if isinstance(algan_mob, ManimMob):
        if len(manim_mob.points) > 0:
            points = (
                algan_mob.control_points.location.reshape(-1, 3).detach().cpu().numpy()
            )
            # Parent transforms cannot change a Manim path's point count.  If
            # an Algan-only structural morph has done so, keeping the semantic
            # Manim object's original topology is safer than assigning an
            # incompatible point layout to a class such as NumberLine/Axes.
            if len(points) == len(manim_mob.points):
                manim_mob.points = points.copy()

        # Style is synchronized only onto nodes that draw something. A Manim
        # node with no points of its own has no appearance -- its style is a
        # template that ``init_colors`` and ``match_style`` broadcast over the
        # family whenever Manim rebuilds it. The matching Algan node is a bare
        # container whose color and opacity rows are placeholders, so writing
        # them here hands Manim a template that erases the real geometry:
        # ``DecimalNumber.set_value`` rebuilds its glyphs and then calls
        # ``init_colors()``, which is what made every ``set_value`` render an
        # invisible number.
        styles_own_geometry = len(manim_mob.points) > 0
        if styles_own_geometry and hasattr(manim_mob, "set_fill"):
            fill_color, fill_opacity = _uniform_color_and_opacity(
                algan_mob.color, algan_mob.opacity
            )
            manim_mob.set_fill(
                fill_color,
                opacity=fill_opacity if algan_mob.filled else 0.0,
                family=False,
            )

        if styles_own_geometry and hasattr(manim_mob, "set_stroke"):
            border_opacity_source = algan_mob.border_grid.opacity
            stroke_color, stroke_opacity = _uniform_color_and_opacity(
                algan_mob.stroke_color, border_opacity_source
            )
            stroke_width = (
                float(algan_mob.stroke_width.reshape(-1)[0].detach().cpu())
                * SETTINGS.style.manim_stroke_width_ratio
            )
            manim_mob.set_stroke(
                stroke_color,
                width=stroke_width,
                opacity=stroke_opacity,
                family=False,
            )

        for algan_child, manim_child in zip(
            algan_mob.submobjects, manim_mob.submobjects
        ):
            _sync_manim_node_from_algan(algan_child, manim_child)


def _algan_mob_to_manim(mob: Mob):
    if isinstance(mob, ManimCompatMob):
        return mob._sync_manim_from_algan()
    source = getattr(mob, "manim_mobject", None)
    if source is not None:
        return source
    if isinstance(mob, BezierCircuitCubic):
        return _algan_bezier_to_manim(mob)
    if isinstance(mob, Group):
        converted = [_algan_mob_to_manim(child) for child in mob]
        if all(isinstance(child, _manim.VMobject) for child in converted):
            return _manim.VGroup(*converted)
        return _manim.Group(*converted)

    # A number of Manim constructors accept an arbitrary Mobject only to query
    # its bounds (SurroundingRectangle, Brace, Line endpoints, and so on).
    # Preserve that functionality for non-vector Algan Mobs with a rectangular
    # VMobject snapshot of their current world-space bounding box.
    bbox = mob.get_bounding_box()[0].detach().cpu().numpy()
    z = float(np.mean(bbox[:, 2]))
    left, right = float(np.min(bbox[:, 0])), float(np.max(bbox[:, 0]))
    bottom, top = float(np.min(bbox[:, 1])), float(np.max(bbox[:, 1]))
    return _manim.Polygon(
        np.array([left, bottom, z]),
        np.array([right, bottom, z]),
        np.array([right, top, z]),
        np.array([left, top, z]),
    )


def to_manim(value: Any):
    """Recursively convert Algan values used as Manim constructor arguments."""
    # Constructor hooks such as ``label_constructor`` and Graph's
    # ``vertex_type`` are classes rather than instances.  Translate Algan Mob
    # classes to the corresponding vendored Manim implementation when one is
    # available.  Plain Text falls back to Tex because Pango is optional in
    # Algan's vendored dependency set.
    if inspect.isclass(value) and issubclass(value, Mob):
        manim_class = getattr(value, "_manim_class", None)
        if manim_class is not None:
            return manim_class
        manim_class = getattr(_manim, value.__name__, None)
        if manim_class is not None:
            return manim_class
        if value.__name__ in {"Text", "MarkupText"}:
            return _manim.Tex
    if isinstance(value, Color):
        return _color_to_manim(value)
    if isinstance(value, Mob):
        return _algan_mob_to_manim(value)
    if isinstance(value, torch.Tensor):
        return _tensor_to_manim(value)
    if isinstance(value, Mapping):
        return type(value)((key, to_manim(item)) for key, item in value.items())
    if isinstance(value, tuple):
        return tuple(to_manim(item) for item in value)
    if isinstance(value, list):
        return [to_manim(item) for item in value]
    if isinstance(value, set):
        return {to_manim(item) for item in value}
    return value


def _scale_factor_to_manim(scale_factor):
    """Convert a scale factor for a delegated Manim ``scale``.

    This is deliberately *not* ``to_manim``, and not only because a bare torch
    tensor passed through it poisons Manim's point arithmetic: NumPy defers to
    the tensor's own multiplication, so ``mob.points`` comes back a torch
    tensor and ``torch.from_numpy`` rejects it when the Mob converts back
    (``TypeError: expected np.ndarray (got Tensor)``). A factor is a
    multiplier, not a coordinate: ``to_manim`` is the converter for values that
    *mean* something in Algan's coordinate system (points, directions, edges),
    and coupling a per-axis multiplier to that conversion would be wrong by
    category -- Algan's ``OUTWARD`` is ``-z`` where Manim's ``OUT`` is ``+z``
    (see
    "The z mirror" in CLAUDE.md), so any coordinate-style treatment of the
    forward component would negate the one stretch the user asked for.

    The result is also flattened to a bare ``(3,)`` array (or a scalar): Algan
    tensors carry leading batch dimensions Manim's geometry does not, and a
    ``(1, 1, 3)`` factor would broadcast ``points`` from ``(N, 3)`` into
    ``(1, N, 3)``, reshaping the very geometry the stretch was meant to resize.
    """
    if isinstance(scale_factor, torch.Tensor):
        scale_factor = scale_factor.detach().cpu()
        if scale_factor.ndim == 0:
            return scale_factor.item()
        return scale_factor.reshape(-1).numpy()
    return scale_factor


def _wrapper_type_for(source) -> type[ManimCompatMob]:
    for cls in type(source).__mro__:
        wrapper = _MANIM_WRAPPER_REGISTRY.get(cls.__name__)
        if wrapper is not None:
            return wrapper
    return ManimCompatMob


def _manim_mobjects_in(value: Any):
    """Yield every Manim Mobject reachable in a delegated call's return value."""
    if isinstance(value, _manim.Mobject):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _manim_mobjects_in(item)
    elif isinstance(value, (tuple, list, set)):
        for item in value:
            yield from _manim_mobjects_in(item)


def from_manim(value: Any, *, scene=None, add_to_scene: bool = True):
    """Recursively convert values returned by delegated Manim APIs.

    Converted Mobs register themselves with ``scene`` by default, exactly as
    directly constructed Mobs do, because Algan builds render primitives from
    the Scene's actor list: a Mob that is not an actor never draws, no matter
    how it is spawned or styled.  Pass ``add_to_scene=False`` only for
    conversions that duplicate geometry some other Mob already renders.
    """
    if isinstance(value, _manim.ImageMobject):
        return ImageMob(value, scene=scene, add_to_scene=add_to_scene)
    if isinstance(value, _manim.Mobject):
        return _wrapper_type_for(value)._from_manim(
            value, scene=scene, add_to_scene=add_to_scene
        )
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value).to(torch.get_default_device())
    if isinstance(value, Mapping):
        return type(value)(
            (
                key,
                from_manim(item, scene=scene, add_to_scene=add_to_scene),
            )
            for key, item in value.items()
        )
    if isinstance(value, tuple):
        return tuple(
            from_manim(item, scene=scene, add_to_scene=add_to_scene) for item in value
        )
    if isinstance(value, list):
        return [
            from_manim(item, scene=scene, add_to_scene=add_to_scene) for item in value
        ]
    return value


class ManimCompatMob(ManimMob):
    """Base class for Mobs whose construction/query API is supplied by Manim.

    Subclasses define ``_manim_class``.  Constructor arguments use Manim's API;
    ``add_to_scene``, ``glow`` and ``glow_radius`` remain Algan-only options.
    Methods not implemented by Algan are delegated to the backing Manim object.
    Returned Mobjects are converted to Algan Mobs -- newly built ones are
    registered with the owning Scene, so ``axes.plot(...).spawn()`` renders --
    while mutations of the backing object are immediately resynchronised into
    this Mob's geometry.
    """

    _manim_class = _manim.VMobject
    _ALGAN_ONLY_KWARGS = {"add_to_scene", "glow", "glow_radius", "batch", "scene"}
    #: True for wrappers whose Manim source typesets through LaTeX on
    #: construction (``MathTex``, ``Title``, the ``Matrix`` family, ...), so a
    #: missing TeX distribution is reported up front, in Algan's words, rather
    #: than as a ``FileNotFoundError: 'latex'`` from inside Manim after it has
    #: written a scratch file. Set by :func:`_make_manim_wrapper`.
    _needs_latex = False

    def __init__(self, *args, **kwargs):
        if self._needs_latex:
            from algan.mobs.text import _require_latex_toolchain

            _require_latex_toolchain()
        algan_kwargs = {
            key: kwargs.pop(key)
            for key in tuple(kwargs)
            if key in self._ALGAN_ONLY_KWARGS
        }
        batch = bool(algan_kwargs.pop("batch", False))
        # Color keywords are normalized before conversion because Manim's
        # parser is narrower than Algan's: it reads a tuple of floats as a
        # *list of colors* and rejects each element. Everything Algan accepts
        # -- a Color, a hex string, a hex int, an RGB sequence -- becomes one
        # Color here, and to_manim renders that as a value Manim does take.
        kwargs = {
            key: (to_color(value) if "color" in key else value)
            for key, value in kwargs.items()
        }
        manim_kwargs = {key: to_manim(value) for key, value in kwargs.items()}
        # Several Manim graph/vector-field classes append a default step to
        # ranges in-place even though their public API accepts generic
        # sequences. Supply a mutable list for tuple ranges.
        for key, value in tuple(manim_kwargs.items()):
            if key.endswith("_range") and isinstance(value, tuple):
                manim_kwargs[key] = list(value)
        source = self._manim_class(
            *(to_manim(arg) for arg in args),
            **manim_kwargs,
        )
        self._initialize_from_manim(source, batch=batch, **algan_kwargs)

    def _initialize_from_manim(self, source, *, batch=False, **kwargs):
        self.manim_mobject = source
        self._exposed_manim_baseline = None
        super().__init__(source, batch=batch, **kwargs)

    @classmethod
    def _from_manim(cls, source, *, scene=None, add_to_scene=True):
        obj = cls.__new__(cls)
        obj._initialize_from_manim(source, scene=scene, add_to_scene=add_to_scene)
        return obj

    def get_manim_mobject(self):
        """Return the backing Manim object used for compatibility operations."""
        self._sync_manim_from_algan()
        # If callers directly mutate the exposed Manim object and then call
        # ``sync_from_manim()``, this gives the three-way merge a before-state
        # without retaining a permanent duplicate of every compatibility Mob.
        self._exposed_manim_baseline = self.manim_mobject.copy()
        return self.manim_mobject

    def _sync_manim_from_algan(self):
        """Lazily push current Algan geometry/style into the retained Manim graph.

        Recursive parent transforms write descendant timeline rows directly and
        therefore bypass this class's method overrides.  Before any Manim API is
        consulted, synchronize from those rows in place so semantic objects such
        as Axes keep their class-specific state while observing the current Algan
        pose and style.
        """
        _sync_manim_node_from_algan(self, self.manim_mobject)
        return self.manim_mobject

    def _prepare_manim_edit(self):
        """Return independent before/edit copies from a synchronized backing Mob."""
        self._sync_manim_from_algan()
        before = self.manim_mobject.copy()
        return before, before.copy()

    def _animate_to_manim(self, source, *, before_source=None):
        """Record an Algan morph to an edited copy of the backing Mobject.

        Native Algan state that the Manim edit did not touch is merged into the
        target before ``become``.  This is what makes parent changes to opacity,
        glow, colors and any other timeline-backed attribute survive a later
        delegated geometry operation.
        """
        if before_source is None:
            self._sync_manim_from_algan()
            before_source = self.manim_mobject.copy()
        self.manim_mobject = source
        # Purely intermediate: ``become`` reads the target's state into this
        # Mob's existing rows and nothing of the target survives the call, so
        # none of it may be registered as an actor.
        target = ManimMob(source, scene=self.scene, add_to_scene=False)
        before = ManimMob(before_source, scene=self.scene, add_to_scene=False)
        _preserve_algan_state_unchanged_by_manim(self, before, target)
        return self.become(target, detach_history=False)

    # These names also exist on Algan's Mob.  Override them so compatibility
    # objects retain Manim's keyword arguments and backing geometry.  Where the
    # two libraries disagree about what an argument *means*, Algan's meaning
    # wins: these are Algan Mobs, animated on Algan's timeline, and a name that
    # silently changes units between one Mob and the next is a trap.  See
    # ``rotate``, the only one of them where the two readings differ.
    def move_to(
        self,
        point_or_mobject,
        aligned_edge=_manim.ORIGIN,
        coor_mask=np.array([1, 1, 1]),
    ):
        before, source = self._prepare_manim_edit()
        source.move_to(
            to_manim(point_or_mobject),
            aligned_edge=to_manim(aligned_edge),
            coor_mask=to_manim(coor_mask),
        )
        return self._animate_to_manim(source, before_source=before)

    def move(self, displacement, arc_angle=None, recursive=True, **kwargs):
        """Move by a displacement, applying it as a Manim ``shift``.

        Algan's generic implementation moves to ``self.location + displacement``,
        but a compatibility Mob's location is the center of the backing
        Mobject's *own* points, which is not the composite's center whenever it
        also has submobjects (an :class:`Arrow`'s tip, for example).  Shifting
        the backing geometry instead keeps the travelled displacement exact for
        every Mob, and is what the relative-placement helpers
        (:meth:`~.Mob.move_to_screen_edge`, :meth:`~.Mob.move_next_to`, ...) are built
        on.
        """
        displacement = cast_to_tensor(displacement)
        if arc_angle is not None or not recursive or kwargs:
            # Curved paths and non-recursive moves have no Manim equivalent.
            # Let Algan record the motion, then derive the backing geometry from
            # the resulting rows rather than trying to mirror the operation.
            target = self.location + displacement
            if arc_angle is None:
                self.set_location(target, recursive=recursive, **kwargs)
            else:
                # ``_move_along_arc``, not ``move_to``: this class overrides
                # ``move_to`` with Manim's signature.
                self._move_along_arc(target, arc_angle, recursive=recursive, **kwargs)
            self._sync_manim_from_algan()
            return self
        before, source = self._prepare_manim_edit()
        source.shift(to_manim(displacement))
        return self._animate_to_manim(source, before_source=before)

    def scale(self, scale_factor, **kwargs):
        before, source = self._prepare_manim_edit()
        source.scale(
            _scale_factor_to_manim(scale_factor),
            **{key: to_manim(value) for key, value in kwargs.items()},
        )
        return self._animate_to_manim(source, before_source=before)

    def rotate(
        self,
        angle: float | torch.Tensor,
        axis: torch.Tensor = OUTWARD,
        about: torch.Tensor | None = None,
        *,
        degrees: bool = True,
    ) -> Mob:
        """Rotate the Mob, using Algan's rotation rather than Manim's.

        ``rotate`` is one of the names this class shares with :class:`~.Mob`,
        and the two libraries mean different things by it: Algan measures
        ``angle`` in degrees where Manim's angle is in radians, and an
        explicit ``axis`` turns the opposite way in each (their default z axis
        are opposite vectors, which is exactly what makes the *default*
        rotation agree).  A compatibility Mob is animated as an Algan Mob, so
        this follows :meth:`~.MobOrientationMixin.rotate` exactly -- degrees,
        Algan's direction constants, and a real rotation of the Mob's basis
        that sweeps from 0 rather than a linear morph between the two poses
        (which cannot express a turn of 180 degrees, let alone a full one).
        The backing Manim object picks the new pose up through the usual lazy
        synchronization.

        Only the pivot is taken from Manim: it rotates about the composite's
        center, whereas Algan's generic implementation uses ``location``, the
        center of the backing Mobject's *own* points.  The two differ for every
        Mob that also has submobjects -- an :class:`Arrow` would otherwise turn
        about its shaft rather than in place -- so ``about`` defaults to
        :meth:`~.MobLayoutMixin.get_center`, which agrees with Manim's
        ``get_center`` for these objects.
        """
        if about is None:
            about = self.get_center()
        return Mob.rotate(self, angle, axis, about, degrees=degrees)

    def set(self, **kwargs):
        # Algan's internal morphing path calls ``set`` with animatable state
        # attributes. Preserve that path; delegate user-facing Manim property
        # updates such as ``set(width=...)`` to the backing object.
        if set(kwargs).issubset(set(self.animatable_attrs)):
            return Mob.set(self, **kwargs)
        before, source = self._prepare_manim_edit()
        source.set(**{key: to_manim(value) for key, value in kwargs.items()})
        return self._animate_to_manim(source, before_source=before)

    def copy(self):
        # Matches :meth:`~.Animatable.clone`, which registers the copy: a copy
        # you cannot render is of no use to the caller.
        source = self.get_manim_mobject()
        return _wrapper_type_for(source)._from_manim(
            source.copy(), scene=self.scene, add_to_scene=True
        )

    def sync_from_manim(self, *, before_source=None):
        """Refresh converted geometry after directly editing the backing object."""
        if before_source is None:
            before_source = self._exposed_manim_baseline
        # Unlike ``_animate_to_manim``'s morph target, part of this conversion is
        # grafted into this Mob's hierarchy below and has to render, so it is
        # built as a registered subtree.  Only the target's own root is
        # discarded, and an unspawned actor never reaches the renderer.
        replaying = self.scene.timeline_manager.is_replaying()
        target = ManimMob(
            self.manim_mobject, scene=self.scene, add_to_scene=not replaying
        )
        if before_source is not None:
            before = ManimMob(before_source, scene=self.scene, add_to_scene=False)
            _preserve_algan_state_unchanged_by_manim(self, before, target)
        if self.is_spawned() and not replaying:
            # Spawning stamps a lifespan and re-lays the endpoint map, which
            # during a render would rewrite the very bounds the batch in flight
            # materialized from. The morph below is all this frame needs: the
            # target only has to supply geometry, and an unspawned Mob is a
            # perfectly good source for that.
            target = target.spawn(animate=False)
        with Off(animation_manager=self.animation_manager):
            self.become(target, detach_history=False)

        if self.scene.timeline_manager.is_replaying():
            # Called from an updater the render is re-executing (a counting
            # DecimalNumber does this on every frame). The morph above has
            # already put the new geometry on the rows the batch is drawing,
            # which is the whole of what this frame can show: the batch's
            # primitives were built from the hierarchy as authored, and
            # restructuring it now would desynchronize the render from them
            # and leave the Scene different afterwards from how it was written.
            self._exposed_manim_baseline = None
            return self

        # ``become`` morphs existing child slots, but a delegated Manim method
        # may add or remove submobjects (``add_tip``, ``add_coordinates``, ...).
        # Replace only the non-component portion of the hierarchy so this Mob's
        # render components keep their timeline identity while the composite
        # structure exactly follows the backing Manim object.
        target_non_components = target.get_non_component_children()
        self.children[:] = list(self.components) + target_non_components
        self.submobjects = target_non_components
        self._exposed_manim_baseline = None
        self._note_hierarchy_change()
        return self

    def _is_backing_geometry(self, value):
        """Whether ``value`` is (or contains) part of this Mob's own hierarchy."""
        family = {id(mob) for mob in self.manim_mobject.get_family()}
        return any(id(mob) in family for mob in _manim_mobjects_in(value))

    def _convert_delegated_result(self, result):
        """Convert a delegated Manim method's return value into Algan values.

        A delegated method either builds something new (``Axes.plot``,
        ``Brace.get_text``, ``Axes.get_axis_labels``) or hands back a piece of
        this Mob's own backing hierarchy (``Axes.get_x_axis``,
        ``Tex.get_part_by_tex``).  New geometry is registered with the owning
        Scene so that spawning it is enough to make it render; own geometry is
        not, because this Mob's converted children already draw it and a second
        registration would draw it twice at the same place.
        """
        return from_manim(
            result,
            scene=self.scene,
            add_to_scene=not self._is_backing_geometry(result),
        )

    def __getattr__(self, name):
        # Normal Mob/Animatable attribute lookup gets first chance.  This method
        # is reached only for Manim-specific APIs.
        if (
            name.startswith("_")
            or "manim_mobject" not in self.__dict__
            or "control_points" not in self.__dict__
        ):
            raise AttributeError(name)
        self._sync_manim_from_algan()
        attribute = getattr(self.manim_mobject, name)
        if not callable(attribute):
            # Reading an attribute is a query, not construction: it must stay
            # free of side effects on the Scene, and repeated reads must not
            # accumulate actors.  Use the method that built the geometry (or
            # ``Scene.add_actor``) to get a renderable Mob.
            return from_manim(attribute, scene=self.scene, add_to_scene=False)

        @wraps(attribute)
        def delegated(*args, **kwargs):
            # A delegated method can be saved and called later.  Parent edits may
            # happen between attribute lookup and invocation, so synchronize and
            # re-bind the Manim method at call time rather than relying on the
            # bound method captured by ``__getattr__``.
            self._sync_manim_from_algan()
            before_source = self.manim_mobject.copy()
            current_attribute = getattr(self.manim_mobject, name)
            result = current_attribute(
                *(to_manim(arg) for arg in args),
                **{key: to_manim(value) for key, value in kwargs.items()},
            )
            if result is self.manim_mobject:
                self.sync_from_manim(before_source=before_source)
                return self
            if result is None:
                self.sync_from_manim(before_source=before_source)
                return None
            return self._convert_delegated_result(result)

        return delegated

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(dir(self.manim_mobject)))


# Generated wrappers inherit their backing class's docstring, which is the right
# default: Manim's prose describes arguments that really do work here. Its
# *examples* are the wrong answer everywhere: a ``.. manim::`` block is Manim
# scene code written against Manim's ``Scene.construct``, and it teaches a script
# that will not run under Algan. Algan does not register that directive either
# (see ``docs/source/conf.py``), so leaving one in an inherited docstring is an
# "Unknown directive type" error in the docs build. ``_strip_manim_examples``
# takes them out; the LaTeX-bearing classes get an Algan-authored docstring
# instead, below. See DOCSTRINGS.md.
#
# Manim's own docstrings are not uniform about the space before ``::``
# (``Torus`` writes ``.. manim :: ExampleTorus``), so match either spelling.
_MANIM_DIRECTIVE_RE = re.compile(r"^\s*\.\.\s+manim\s*::")


def _numpydoc_section_starts(lines: list[str]) -> list[int]:
    """Index every NumPy-style section header (``Examples`` + ``--------``)."""
    starts = []
    for index in range(len(lines) - 1):
        title, underline = lines[index].strip(), lines[index + 1].strip()
        if (
            title
            and underline
            and set(underline) == {"-"}
            and len(underline) >= len(title)
        ):
            starts.append(index)
    return starts


def _strip_manim_examples(doc: str | None) -> str | None:
    """Remove ``.. manim::`` example blocks from an inherited Manim docstring.

    An ``Examples`` section that holds one is dropped whole: its prose ("the
    first example shows...") only describes the renders being removed. A
    ``.. manim::`` block anywhere else is dropped on its own, along with the
    indented body that belongs to it.
    """
    if not doc or not any(_MANIM_DIRECTIVE_RE.match(line) for line in doc.splitlines()):
        return doc

    lines = doc.splitlines()
    starts = _numpydoc_section_starts(lines)
    bounds = [
        (start, starts[position + 1] if position + 1 < len(starts) else len(lines))
        for position, start in enumerate(starts)
    ]
    drop = set()
    for start, end in bounds:
        if lines[start].strip() == "Examples" and any(
            _MANIM_DIRECTIVE_RE.match(line) for line in lines[start:end]
        ):
            drop.update(range(start, end))

    kept = [line for index, line in enumerate(lines) if index not in drop]

    # Any ``.. manim::`` left over sat outside an Examples section. Drop the
    # directive line plus everything indented under it (options and body).
    result: list[str] = []
    index = 0
    while index < len(kept):
        line = kept[index]
        if not _MANIM_DIRECTIVE_RE.match(line):
            result.append(line)
            index += 1
            continue
        indent = len(line) - len(line.lstrip())
        index += 1
        while index < len(kept):
            body = kept[index]
            if body.strip() and (len(body) - len(body.lstrip())) <= indent:
                break
            index += 1

    return "\n".join(result).rstrip() + "\n"


_MATHTEX_DOC = """A LaTeX string typeset in math mode, wrapping Manim's ``MathTex``.

Manim compiles the formula and builds its glyph outlines; Algan converts those
to cubic bezier circuits and animates them on its own timeline. Manim's own
arguments work as written -- ``tex_to_color_map`` and ``substrings_to_isolate``
included -- and Manim methods Algan does not implement are delegated to the
backing object.

Reach for Algan's :class:`~algan.mobs.text.Tex` when you are not porting a
script. The two produce the same outlines (``MathTex("x^2")`` matches
``Tex("x^2", font_size=48)``, because Algan's ``Tex`` also builds at Manim's 48
and then scales), but this one is a single Mob with no per-glyph views:
``formula[0]`` raises :class:`TypeError`, and character indexing,
:meth:`~algan.mobs.text.Tex.get_segment` and :meth:`~algan.mobs.text.Tex.write`
all live on ``Tex`` instead.

Animation
---------
Constructing one records nothing: LaTeX runs immediately and the Mob joins the
active Scene unspawned. Call
:meth:`~algan.animatable_base.animatable.Animatable.spawn` to make it appear.
Everything after that -- ``formula.color = BLUE``, a move, a delegated Manim
edit -- is recorded on the timeline like any other Mob's, over the current
context's runtime.

Parameters
----------
*tex_strings
    One or more LaTeX sources, joined by ``arg_separator`` and compiled as a
    single document.
arg_separator
    Inserted between consecutive ``tex_strings`` in the compiled source.
    Defaults to ``" "``, one space.
substrings_to_isolate
    Substrings to split the source on before compiling, so each ends up as its
    own piece of the result. Defaults to ``None``, meaning no extra splitting
    beyond what the separate ``tex_strings`` already give.
tex_to_color_map
    Maps a substring of the source to the color its glyphs take; the substring
    is isolated for you. Accepts Manim colors, hex strings, or Algan
    :class:`~algan.constants.color.Color` values -- an Algan color is converted
    to hex on the way through Manim, so its glow and opacity are dropped and
    have to be set on the Mob afterwards. Defaults to ``None``, one color
    throughout.
tex_environment
    Name of the LaTeX environment to typeset in, such as ``"gather*"``.
    Defaults to ``"align*"``.
**kwargs
    Manim's remaining ``MathTex`` arguments -- notably ``font_size`` (defaults
    to ``48``), ``color``, ``tex_template``, ``stroke_width`` and
    ``should_center`` -- plus the Algan-only ``scene``, ``add_to_scene``,
    ``glow`` and ``glow_radius``.

Raises
------
:class:`ValueError`
    If LaTeX fails to compile the source. The most common cause is
    ``tex_to_color_map`` or ``substrings_to_isolate``: they split the source on
    the literal substring, so a key that also occurs inside a control sequence
    or a brace group cuts it in half. Coloring ``"n"`` in a formula containing
    ``\\infty`` is enough to do it. Pick a key that stands alone, or pass the
    pieces as separate ``tex_strings``.

See Also
--------
:class:`~algan.mobs.text.Tex` : Algan's own LaTeX Mob, with per-glyph views,
    segments and the hand-writing animation.

Examples
--------
A formula, and one with two symbols picked out by ``tex_to_color_map``:

.. algan:: Example1MathTex
    :save_last_frame:

    from algan import *

    MathTex(r"\\sum_{n=1}^{\\infty} \\frac{1}{n^2} = \\frac{\\pi^2}{6}",
            font_size=36).move(UP * 0.5).spawn()
    MathTex(r"a^2 + b^2 = c^2", font_size=36,
            tex_to_color_map={"a": BLUE, "b": YELLOW}).move(DOWN * 0.5).spawn()

    Scene.save_video()
"""

_TITLE_DOC = """An underlined heading, wrapping Manim's ``Title``.

The text is typeset by LaTeX in text mode (so ``Title("Chapter 1")`` reads as
written, no maths escaping needed) with a horizontal rule beneath it, and the
whole thing is moved to the top of the frame.

**The frame it moves to the top of is Manim's, not Algan's, and it lands flush
against the edge.** Manim's frame is 8 world units tall and its ``to_edge``
leaves a 0.5 gap, so the title's top comes to rest at ``y = 3.5`` -- which is
exactly where Algan's default camera puts its top border. Nothing is cut off,
but the text touches the frame edge with no margin at all. Call
:meth:`~algan.animatable_base.mob_movement.MobMovementMixin.move_to_screen_edge` to
inset it by the usual buffer, or ``.move(DOWN * 1)`` to place it by hand.

The default rule is sized from Manim's frame too, at ``frame_width - 2``, which
comes out just narrower than Algan's visible width. Pass
``match_underline_width_to_text=True`` for a rule the width of the words.

Animation
---------
Constructing one records nothing: LaTeX runs immediately and the Mob joins the
active Scene unspawned. Call
:meth:`~algan.animatable_base.animatable.Animatable.spawn` to make it appear;
later changes are recorded on the timeline like any other Mob's.

Parameters
----------
*text_parts
    One or more strings, joined and typeset as a single line of text.
include_underline
    Whether to draw the rule beneath the text. Defaults to ``True``.
match_underline_width_to_text
    Whether the rule spans only the width of the text. Defaults to ``False``,
    which spans Manim's frame width less 2 world units instead.
underline_buff
    Gap between the text and the rule, in world units. Defaults to ``0.25``
    (Manim's ``MED_SMALL_BUFF``).
**kwargs
    Manim's remaining ``Tex`` arguments -- notably ``font_size`` (defaults to
    ``48``), ``color`` and ``tex_template`` -- plus the Algan-only ``scene``,
    ``add_to_scene``, ``glow`` and ``glow_radius``.

Attributes
----------
underline
    The rule Mob, present only when ``include_underline`` is true. Animate it
    like any other Mob.

Examples
--------
A title placed where Algan's camera can see all of it, over a shape:

.. algan:: Example1Title
    :save_last_frame:

    from algan import *

    Title("A Title", match_underline_width_to_text=True).move(DOWN * 1).spawn()
    Circle(radius=0.8, color=BLUE).spawn()

    Scene.save_video()
"""

_WRAPPER_DOCSTRINGS: dict[str, str] = {
    "MathTex": _MATHTEX_DOC,
    "Title": _TITLE_DOC,
}


# Composite Manim classes that always typeset their parts with LaTeX even
# though they are not ``Tex`` subclasses themselves: the matrix family builds a
# ``MathTex`` per entry, ``Variable`` and ``BraceLabel`` label with one.
_LATEX_COMPOSITES = frozenset(
    {"Matrix", "IntegerMatrix", "DecimalMatrix", "Variable", "BraceLabel"}
)


def _make_manim_wrapper(name: str):
    manim_class = getattr(_manim, name)
    needs_latex = name in _LATEX_COMPOSITES or (
        isinstance(manim_class, type)
        and issubclass(manim_class, _manim.SingleStringMathTex)
    )
    wrapper = type(
        name,
        (ManimCompatMob,),
        {
            "_manim_class": manim_class,
            "_needs_latex": needs_latex,
            "__module__": __name__,
            "__doc__": _WRAPPER_DOCSTRINGS.get(
                name, _strip_manim_examples(manim_class.__doc__)
            ),
        },
    )
    with contextlib.suppress(TypeError, ValueError):
        wrapper.__signature__ = inspect.signature(manim_class)
    _MANIM_WRAPPER_REGISTRY[name] = wrapper
    globals()[name] = wrapper
    return wrapper


# Classes that are cubic-Bezier/image/composite Mobjects in the vendored Manim
# implementation.
#
# Names with a native Algan equivalent are wrapped too, and deliberately so:
# this module is reached as ``algan.manim``, a namespace where every name means
# "Manim's version, by Manim's conventions". ``Sphere`` is Algan's and
# ``algan.manim.Sphere`` is Manim's; omitting the overlapping ones would leave
# holes in that namespace whose only explanation is which classes Algan happened
# to implement natively. Manim ``Surface`` subclasses (``Sphere``, ``Torus``,
# ``Cone``) convert as well as the flat ones -- ManimMob turns their quad grids
# into curved patches -- so there is nothing to exclude on capability grounds.
_WRAPPED_MANIM_CLASS_NAMES = (
    "Angle",
    "AnnotationDot",
    "AnnularSector",
    "Annulus",
    "Arc",
    "ArcBetweenPoints",
    "ArcBrace",
    "ArcPolygon",
    "ArcPolygonFromArcs",
    "Arrow",
    "Arrow3D",
    "ArrowCircleFilledTip",
    "ArrowCircleTip",
    "ArrowSquareFilledTip",
    "ArrowSquareTip",
    "ArrowTip",
    "ArrowTriangleFilledTip",
    "ArrowTriangleTip",
    "ArrowVectorField",
    "Axes",
    "BackgroundRectangle",
    "BarChart",
    "Brace",
    "BraceBetweenPoints",
    "BraceLabel",
    "BraceText",
    "BulletedList",
    "Circle",
    "Code",
    "ComplexPlane",
    "ComplexValueTracker",
    "Cone",
    "ConvexHull",
    "ConvexHull3D",
    "Cross",
    "Cube",
    "CubicBezier",
    "CurvedArrow",
    "CurvedDoubleArrow",
    "CurvesAsSubmobjects",
    "Cutout",
    "Cylinder",
    "DashedLine",
    "DashedVMobject",
    "DecimalMatrix",
    "DecimalNumber",
    "DecimalTable",
    "DiGraph",
    "Difference",
    "Dodecahedron",
    "Dot",
    "Dot3D",
    "DoubleArrow",
    "Elbow",
    "Ellipse",
    "Exclusion",
    "FullScreenRectangle",
    "FunctionGraph",
    "Graph",
    "Group",
    "Icosahedron",
    "ImplicitFunction",
    "Integer",
    "IntegerMatrix",
    "IntegerTable",
    "Intersection",
    "Label",
    "LabeledArrow",
    "LabeledDot",
    "LabeledLine",
    "LabeledPolygram",
    "Line",
    "Line3D",
    "ManimBanner",
    "MarkupText",
    "MathTable",
    "MathTex",
    "Matrix",
    "MobjectMatrix",
    "MobjectTable",
    "NumberLine",
    "NumberPlane",
    "Octahedron",
    "Paragraph",
    "ParametricFunction",
    "Point",
    "PolarPlane",
    "Polygon",
    "Polygram",
    "Polyhedron",
    "Prism",
    "Rectangle",
    "RegularPolygon",
    "RegularPolygram",
    "RightAngle",
    "RoundedRectangle",
    "SVGMobject",
    "SampleSpace",
    "ScreenRectangle",
    "Sector",
    "SingleStringMathTex",
    "Sphere",
    "Square",
    "Star",
    "StealthTip",
    "StreamLines",
    "Surface",
    "SurroundingRectangle",
    "Table",
    "TangentLine",
    "Tetrahedron",
    "Tex",
    "Text",
    "ThreeDAxes",
    "ThreeDVMobject",
    "TipableVMobject",
    "Title",
    "Torus",
    "Triangle",
    "Underline",
    "Union",
    "UnitInterval",
    "VDict",
    "VGroup",
    "VMobject",
    "VMobjectFromSVGPath",
    "ValueTracker",
    "Variable",
    "Vector",
    "VectorField",
    "VectorizedPoint",
)

for _name in _WRAPPED_MANIM_CLASS_NAMES:
    if hasattr(_manim, _name):
        _make_manim_wrapper(_name)

ArcBetweenPoints = _MANIM_WRAPPER_REGISTRY["ArcBetweenPoints"]
SingleStringMathTex = _MANIM_WRAPPER_REGISTRY["SingleStringMathTex"]


# Algan's Mob is the renderer-independent equivalent of Manim's Mobject.
Mobject = Mob


# Private base used by Manim's boolean-operation Mobjects.  It is source-defined
# (and therefore part of the parity inventory) even though it is not re-exported
# by Manim's top-level package.
try:
    from manim.mobject.geometry.boolean_ops import _BooleanOps as _ManimBooleanOps
except ImportError:  # pragma: no cover - only for unusually old vendored copies
    _ManimBooleanOps = _manim.VMobject


class _BooleanOps(ManimCompatMob):
    _manim_class = _ManimBooleanOps


from manim.mobject.geometry import arc as _manim_arc
from manim.mobject.svg.brace import BraceText as _ManimBraceText


class BraceText(ManimCompatMob):
    """Brace with a plain-text label.

    Not re-exported by Manim's top-level package, so it is wrapped by name
    from its own module rather than picked up by
    :data:`_WRAPPED_MANIM_CLASS_NAMES`.
    """

    _manim_class = _ManimBraceText


BraceText.__signature__ = inspect.signature(_ManimBraceText)
_MANIM_WRAPPER_REGISTRY["BraceText"] = BraceText


# ``LabeledDot`` and ``TangentialArc`` were both reimplemented here, because
# the Manim copy vendored at the time had neither ``LabeledDot``'s ``buff``
# parameter nor ``TangentialArc`` at all. The vendored subset is Manim 0.21.0
# and carries both, so these are ordinary wrappers again -- ``LabeledDot``
# through :data:`_WRAPPED_MANIM_CLASS_NAMES` above, ``TangentialArc`` here
# because it is not one of Manim's top-level exports.
class TangentialArc(ManimCompatMob):
    """An arc tangent to two intersecting lines.

    Not re-exported by Manim's top-level package, so it is wrapped by hand
    rather than by name; the geometry is entirely
    :class:`manim.mobject.geometry.arc.TangentialArc`'s.
    """

    _manim_class = _manim_arc.TangentialArc


TangentialArc.__signature__ = inspect.signature(_manim_arc.TangentialArc)
# Deliberately *not* in _MANIM_WRAPPER_REGISTRY: that registry is what
# manim_adapters adapts, and adding a name to it adds a root spelling to
# `from algan import *`. TangentialArc has always been `mn.`-only.


class ValueTracker(Mob):
    """A non-rendering Mob storing one animatable scalar."""

    def __init__(self, value=0.0, **kwargs):
        kwargs.setdefault("add_to_scene", False)
        super().__init__(**kwargs)
        self.register_attrs_as_animatable("value", ValueTracker)
        tensor = torch.as_tensor(value, dtype=torch.get_default_dtype()).reshape(1, 1)
        self._init_default_attr("value", tensor)

    def get_value(self):
        return float(self.value.reshape(-1)[0])

    def set_value(self, value):
        self.value = torch.as_tensor(value, dtype=torch.get_default_dtype()).reshape(
            1, 1
        )
        return self

    def increment_value(self, d_value):
        return self.set_value(self.get_value() + d_value)

    def __iadd__(self, value):
        return self.increment_value(value)

    def __isub__(self, value):
        return self.increment_value(-value)

    def __float__(self):
        return self.get_value()


class ComplexValueTracker(Mob):
    """Complex-valued counterpart of :class:`ValueTracker`."""

    def __init__(self, value=0j, **kwargs):
        kwargs.setdefault("add_to_scene", False)
        super().__init__(**kwargs)
        self.register_attrs_as_animatable("complex_value", ComplexValueTracker)
        value = complex(value)
        tensor = torch.tensor(
            (value.real, value.imag), dtype=torch.get_default_dtype()
        ).reshape(1, 2)
        self._init_default_attr("complex_value", tensor)

    def get_value(self):
        value = self.complex_value.reshape(-1)
        return complex(float(value[0]), float(value[1]))

    def set_value(self, value):
        value = complex(value)
        self.complex_value = torch.tensor(
            (value.real, value.imag), dtype=torch.get_default_dtype()
        ).reshape(1, 2)
        return self

    def increment_value(self, d_value):
        return self.set_value(self.get_value() + d_value)

    def __complex__(self):
        return self.get_value()


_MANIM_WRAPPER_REGISTRY["ValueTracker"] = ValueTracker
_MANIM_WRAPPER_REGISTRY["ComplexValueTracker"] = ComplexValueTracker


# Manim's marker class for the pieces of a MathTex. It has no constructor of
# its own, so the vendored SingleStringMathTex behaviour is the closest
# meaningful compatibility type.
if "SingleStringMathTex" in globals():

    class MathTexPart(SingleStringMathTex):
        pass


# Manim's OpenGL-specific class names describe an alternate renderer backend.
# Algan has one ray-traced Mob representation, so those names intentionally map
# to the equivalent renderer-independent compatibility/native class.
_OPENGL_EQUIVALENTS = {
    "OpenGLAnnularSector": "AnnularSector",
    "OpenGLAnnulus": "Annulus",
    "OpenGLArc": "Arc",
    "OpenGLArcBetweenPoints": "ArcBetweenPoints",
    "OpenGLArrow": "Arrow",
    "OpenGLArrowTip": "ArrowTriangleTip",
    "OpenGLCircle": "Circle",
    "OpenGLCubicBezier": "CubicBezier",
    "OpenGLCurvedArrow": "CurvedArrow",
    "OpenGLCurvedDoubleArrow": "CurvedDoubleArrow",
    "OpenGLCurvesAsSubmobjects": "CurvesAsSubmobjects",
    "OpenGLDashedLine": "DashedLine",
    "OpenGLDashedVMobject": "DashedVMobject",
    "OpenGLDot": "Dot",
    "OpenGLDoubleArrow": "DoubleArrow",
    "OpenGLElbow": "Elbow",
    "OpenGLEllipse": "Ellipse",
    "OpenGLGroup": "Group",
    "OpenGLImageMobject": "ImageMobject",
    "OpenGLLine": "Line",
    "OpenGLMobject": "Mob",
    "OpenGLPoint": "Point",
    "OpenGLPolygon": "Polygon",
    "OpenGLRectangle": "Rectangle",
    "OpenGLRegularPolygon": "RegularPolygon",
    "OpenGLRoundedRectangle": "RoundedRectangle",
    "OpenGLSector": "Sector",
    "OpenGLSquare": "Square",
    "OpenGLTangentLine": "TangentLine",
    "OpenGLTipableVMobject": "TipableVMobject",
    "OpenGLTriangle": "Triangle",
    "OpenGLVGroup": "VGroup",
    "OpenGLVMobject": "VMobject",
    "OpenGLVector": "Vector",
    "OpenGLVectorizedPoint": "VectorizedPoint",
}


def install_opengl_aliases(namespace: Mapping[str, Any] | dict[str, Any]):
    """Install renderer-equivalent OpenGL names after native Mobs are imported."""
    installed = []
    for alias, target in _OPENGL_EQUIVALENTS.items():
        if target in namespace:
            value = namespace[target]
            namespace[alias] = value
            globals()[alias] = value
            installed.append(alias)
    return tuple(installed)


#: The registry, not :data:`_WRAPPED_MANIM_CLASS_NAMES`. The two differ by
#: exactly the Pango classes -- ``Text``, ``MarkupText`` and ``Paragraph``,
#: which the vendored Manim subset exports only when the optional
#: ``manimpango`` is installed, so no wrapper was built for them here. Naming
#: them in ``__all__`` regardless would break ``import algan.manim`` outright,
#: since that module resolves every name in this one's ``__all__``.
__all__ = [
    "ManimCompatMob",
    "to_manim",
    "from_manim",
    "install_opengl_aliases",
    "MathTexPart",
    "Mobject",
    "BraceText",
    "TangentialArc",
    "_BooleanOps",
    *(name for name in _WRAPPED_MANIM_CLASS_NAMES if name in _MANIM_WRAPPER_REGISTRY),
]
