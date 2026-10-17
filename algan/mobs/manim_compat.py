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
import functools
import inspect
import re
import warnings
from collections.abc import Mapping
from functools import wraps
from typing import Any

import manim as _manim
import numpy as np
import torch

from algan.animatable_base.animatable import _reports_refused_keywords
from algan.animatable_base.mob import _MANIM_CONSTRUCTOR_HINTS, Mob
from algan.animation_timeline.animation_contexts import Off
from algan.constants.color import Color, to_color
from algan.constants.spatial import OUTWARD
from algan.errors import UnsupportedFeatureWarning, _user_stacklevel
from algan.mobs.bezier_circuit import BezierCircuitCubic
from algan.mobs.group import Group as _AlganGroup
from algan.mobs.image_mob import ImageMob
from algan.mobs.manim_mob import ManimMob, _conversion_signature
from algan.mobs.stroke_style import _stroke_style
from algan.mobs.typst import MathTypst as _NativeMathTypst
from algan.mobs.typst import Typst as _NativeTypst
from algan.mobs.typst import _TypstSelection
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
        # The ends alone would draw an arc (``path_arc``) as its chord.
        result.set_points(points)
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
    cap, join, limit = _stroke_style(mob.cap_style, mob.joint_type, mob.miter_limit)
    result.cap_style = _manim.CapStyleType[cap.upper()]
    result.joint_type = _manim.LineJointType[join.upper()]
    result.miter_limit = limit
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
    grid = algan_mob.grid.location.reshape(
        -1, algan_mob.grid_width, algan_mob.grid_height, 3
    )[0]
    # Manim ImageMobject point order is top-left, top-right, bottom-left,
    # bottom-right. Read the rendered corners: a Mob's basis carries scale,
    # not the image's aspect ratio or its half-extents.
    manim_mob.points = (
        torch.stack((grid[0, -1], grid[-1, -1], grid[0, 0], grid[-1, 0]))
        .detach()
        .cpu()
        .numpy()
        .copy()
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
        cap, join, limit = _stroke_style(
            algan_mob.cap_style, algan_mob.joint_type, algan_mob.miter_limit
        )
        manim_mob.cap_style = _manim.CapStyleType[cap.upper()]
        manim_mob.joint_type = _manim.LineJointType[join.upper()]
        manim_mob.miter_limit = limit
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

        for index, (algan_child, manim_child) in enumerate(
            zip(algan_mob._drawn_parts(), manim_mob.submobjects)
        ):
            if (
                getattr(algan_child, "manim_mobject", None) is not manim_child
                and not isinstance(manim_child, _manim.ImageMobject)
                and not _stands_for(algan_child, manim_child)
            ):
                # ``become`` put something else in this part's place. The
                # Manim side follows, or the next Manim edit of the whole --
                # ``scale``, ``set_color_by_tex`` -- would bring back the part
                # it replaced.
                manim_mob.submobjects[index] = _algan_mob_to_manim(algan_child)
                continue
            _sync_manim_node_from_algan(algan_child, manim_child)


def _native_text_glyphs(mob: Mob):
    """The glyph circuits of a native :class:`~algan.mobs.text.Tex`, or None.

    None for any other Mob, and for a triangulated Tex, whose glyphs are not
    cubic circuits.
    """
    from algan.mobs.text import Tex

    if not isinstance(mob, Tex):
        return None
    batch = mob._character_batch
    if batch is None:
        return []
    if not isinstance(batch, BezierCircuitCubic):
        return None
    return list(mob.character_mobs)


def _algan_mob_to_manim(mob: Mob):
    if isinstance(mob, (ManimCompatMob, _TypstSelection)):
        return mob._sync_manim_from_algan()
    if isinstance(mob, ManimMob):
        # A plain conversion keeps the Mobject it was built from, as it was
        # built: handed over as it is, a ManimMob moved since would be framed,
        # braced and lined up where it started. Manim gets a copy where it is
        # now, and the caller's own Mobject is left alone.
        source = mob.manim_mobject.copy()
        _sync_manim_node_from_algan(mob, source)
        return source
    if isinstance(mob, BezierCircuitCubic):
        return _algan_bezier_to_manim(mob)
    # Algan's Group, imported under an alias: ``Group`` in this module is the
    # wrapper of Manim's, and testing for that one sent every native Group
    # down to the bounding-box fallback below, drawn as a rectangle.
    if isinstance(mob, _AlganGroup):
        converted = [_algan_mob_to_manim(child) for child in mob]
        if all(isinstance(child, _manim.VMobject) for child in converted):
            return _manim.VGroup(*converted)
        return _manim.Group(*converted)
    glyphs = _native_text_glyphs(mob)
    if glyphs is not None:
        # Manim draws a label as it is handed one (``get_axis_labels(x_label=
        # Tex(...))``, ``LabeledDot(label=...)``, a MobTable entry), so the
        # glyphs go over as outlines: the bounding box below would be drawn
        # as a rectangle where the text should be.
        if mob.image_mobs:
            warnings.warn(
                f"{type(mob).__name__}'s image characters (emoji) have no "
                "outline to hand to Manim, so the Manim-backed Mob it was "
                "passed to leaves them out.",
                UnsupportedFeatureWarning,
                stacklevel=_user_stacklevel(),
            )
        return _manim.VGroup(*(_algan_bezier_to_manim(glyph) for glyph in glyphs))

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


def _scaled_natively_during_replay(scale):
    """Route a compatibility Mob's ``scale`` to Algan's own while replaying.

    A render re-runs a recorded step over a batch of frames at once, so a
    factor computed there holds one value per frame (Indicate's pulse), and
    Manim's ``scale`` takes a single number. Algan's own scale takes the batch,
    and the backing Mobject catches up lazily, as it does after a native
    rotation. A call with Manim's keywords (``about_point``, ``about_edge``)
    still goes to Manim.
    """

    @wraps(scale)
    def wrapper(self, scale_factor, **kwargs):
        if not kwargs and self.scene.timeline_manager.is_replaying():
            return Mob.scale(self, scale_factor)
        return scale(self, scale_factor, **kwargs)

    return wrapper


# A render replays an updater over a batch of frames at once, so the rows a
# compatibility Mob reads there hold one pose per frame, while its backing
# Mobject holds one pose -- the one it was authored in, since it cannot be
# synchronized from a batch. A Manim method that moves the Mob would fail on a
# point per frame, or rebuild the Mob mid-render, and one that measures it
# would answer for the authored pose on every frame. The translations and
# bounding-box measurements below are plain arithmetic on the rows, so while
# replaying they are answered there instead, for every frame of the batch. The
# Algan and Manim bounding boxes of a compatibility Mob agree exactly.


def _critical_point(mob: Mob, direction) -> torch.Tensor:
    """Manim's ``get_critical_point`` read from the Mob's rows, shape ``(*, 1, 3)``.

    Per axis: the far side of the bounding box where ``direction`` is positive,
    the near side where it is negative, the middle where it is zero.
    """
    corners = mob.get_bounding_box()
    low = corners.amin(-2, keepdim=True)
    high = corners.amax(-2, keepdim=True)
    direction = cast_to_tensor(direction).to(corners)
    return torch.where(
        direction > 0, high, torch.where(direction < 0, low, (low + high) * 0.5)
    )


def _point_of(target, direction) -> torch.Tensor:
    """A Manim ``mobject_or_point`` argument as a point: a Mob's critical point."""
    if isinstance(target, Mob):
        return _critical_point(target, direction)
    if isinstance(target, _manim.Mobject):
        return cast_to_tensor(target.get_critical_point(to_manim(direction)))
    return cast_to_tensor(target)


def _manim_arguments(mob: ManimCompatMob, name: str, args, kwargs) -> dict:
    """``args`` and ``kwargs`` bound to the backing Mobject's ``name``, defaults filled."""
    method = getattr(type(mob.manim_mobject), name)
    bound = inspect.signature(method).bind(mob.manim_mobject, *args, **kwargs)
    bound.apply_defaults()
    return bound.arguments


def _shift_now(mob, *vectors):
    return mob._translate_now(sum((cast_to_tensor(v) for v in vectors), 0.0))


def _next_to_now(mob, *args, **kwargs):
    given = _manim_arguments(mob, "next_to", args, kwargs)
    if (
        given["submobject_to_align"] is not None
        or given["index_of_submobject_to_align"] is not None
    ):
        return NotImplemented
    direction = cast_to_tensor(given["direction"])
    edge = cast_to_tensor(given["aligned_edge"])
    target = _point_of(given["mobject_or_point"], edge + direction)
    offset = target - _critical_point(mob, edge - direction)
    offset = (offset + given["buff"] * direction) * cast_to_tensor(given["coor_mask"])
    return mob._translate_now(offset)


def _align_to_now(mob, *args, **kwargs):
    given = _manim_arguments(mob, "align_to", args, kwargs)
    direction = cast_to_tensor(given["direction"])
    point = _point_of(given["mobject_or_point"], direction)
    offset = point - _critical_point(mob, direction)
    return mob._translate_now(offset * (direction != 0))


def _coordinate_now(dim):
    def coordinate(mob, direction=_manim.ORIGIN):
        return _critical_point(mob, direction)[..., dim : dim + 1]

    return coordinate


def _length_over_dim_now(mob, dim):
    return (
        _critical_point(mob, _manim.UR + _manim.OUT)
        - _critical_point(mob, _manim.DL + _manim.IN)
    )[..., dim : dim + 1]


def _edge_now(direction):
    return lambda mob: _critical_point(mob, direction)


#: Manim methods answered from the rows while a render replays an updater.
_ANSWERED_FROM_ROWS_DURING_REPLAY = {
    "shift": _shift_now,
    "next_to": _next_to_now,
    "align_to": _align_to_now,
    "get_critical_point": _critical_point,
    "get_edge_center": _critical_point,
    "get_corner": _critical_point,
    "get_top": _edge_now(_manim.UP),
    "get_bottom": _edge_now(_manim.DOWN),
    "get_right": _edge_now(_manim.RIGHT),
    "get_left": _edge_now(_manim.LEFT),
    "get_zenith": _edge_now(_manim.OUT),
    "get_nadir": _edge_now(_manim.IN),
    "get_x": _coordinate_now(0),
    "get_y": _coordinate_now(1),
    "get_z": _coordinate_now(2),
    "length_over_dim": _length_over_dim_now,
}


# Manim queries that look a value up on the backing Mobject's geometry. An
# Algan number carries a leading frame axis -- an updater's ``t`` is ``(frames,
# 1, 1)`` -- which Manim reads as an array of inputs and answers in a shape of
# its own: ``c2p`` returns a ``(3, frames)`` column, ``n2p`` a ``(frames, 3)``
# block that reads as one Mob's rows, and ``point_from_proportion`` raises. So
# a call whose numbers carry frames, or one made while a replay writes several,
# is asked once per frame -- with that frame's numbers and that frame's pose,
# the way a Manim script asks -- and the answers come back with the frame axis
# leading: a point as ``(frames, 1, 3)``, a number as ``(frames, 1, 1)``, and
# coordinates as one ``(frames, 1, 1)`` per coordinate, so that
# ``x, y = axes.p2c(point)`` still unpacks.
_POINT_QUERIES = frozenset(
    {
        "c2p",
        "coords_to_point",
        "n2p",
        "number_to_point",
        "i2gp",
        "input_to_graph_point",
        "pr2pt",
        "polar_to_point",
        "point_from_proportion",
        "get_midpoint",
        "get_start",
        "get_end",
        "get_vector",
        "get_unit_vector",
    }
)
_NUMBER_QUERIES = frozenset(
    {
        "p2n",
        "point_to_number",
        "proportion_from_point",
        "slope_of_tangent",
        "angle_of_tangent",
        "get_angle",
        "get_slope",
        "get_length",
    }
)
_COORDINATE_QUERIES = frozenset(
    {"p2c", "point_to_coords", "pt2pr", "point_to_polar", "input_to_graph_coords"}
)
_QUERIES_ASKED_PER_FRAME = _POINT_QUERIES | _NUMBER_QUERIES | _COORDINATE_QUERIES


def _has_frame_axis(value) -> bool:
    return isinstance(value, torch.Tensor) and value.dim() >= 3


def _at_frame(value, index: int):
    """``value`` as Manim takes it for one frame: a float, a point, or as is."""
    if not _has_frame_axis(value):
        return to_manim(value)
    row = value[index if len(value) > 1 else 0]
    return row.item() if row.numel() == 1 else _tensor_to_manim(row)


def _wrapper_type_for(source) -> type[ManimCompatMob]:
    for cls in type(source).__mro__:
        wrapper = _MANIM_WRAPPER_REGISTRY.get(cls.__name__)
        if wrapper is not None:
            return wrapper
    return ManimCompatMob


def _compat_part(source, **kwargs):
    """Convert one submobject of a compatibility Mob as a compatibility Mob.

    ``ManimMob``'s ``_part_factory``: a part built this way is both the Mob on
    screen and one that answers Manim's methods, so ``axes.x_axis.n2p(2)`` and
    ``axes.x_axis.set_color(RED)`` both work on the axis that is drawn.
    """
    wrapper = _wrapper_type_for(source)
    part = wrapper.__new__(wrapper)
    part._initialize_from_manim(source, **kwargs)
    return part


def _parts_mirror(mob, node) -> bool:
    """Whether converting ``node``'s submobjects afresh rebuilds ``mob``'s parts.

    True when every part is a conversion of a node of the same signature as the
    one it now stands for (see ``_conversion_signature``), all the way down:
    then the parts can be morphed into the new conversion where they stand, and
    every Mob handed out for one of them stays the one on screen.
    """
    if getattr(mob, "_parts_packed", False):
        return False
    parts = mob._drawn_parts()
    return len(parts) == len(node.submobjects) and all(
        _stands_for(part, child) for part, child in zip(parts, node.submobjects)
    )


def _stands_for(part, node) -> bool:
    """Whether ``part`` is a conversion of a node like ``node``, part for part."""
    return (
        not isinstance(node, _manim.ImageMobject)
        and isinstance(part, ManimMob)
        and getattr(part, "_converted_from", None) == _conversion_signature(node)
        and _parts_mirror(part, node)
    )


def _rebind_parts(mob, previous, current):
    """Point the parts drawing ``previous``'s nodes at ``current``'s.

    A compatibility edit runs on a copy of the backing hierarchy and keeps the
    edited copy, which leaves every part holding a node of the old one. Bound
    to the new nodes, a part handed to Manim is a part Manim recognizes
    (``formula.index_of_part(formula[1])``). A part ``become`` replaced holds
    a node of its own and keeps it.
    """
    for part, old, new in zip(
        mob._drawn_parts(), previous.submobjects, current.submobjects
    ):
        if (
            isinstance(part, ManimMob)
            and part.manim_mobject is old
            and type(old) is type(new)
        ):
            part.manim_mobject = new
            _rebind_parts(part, old, new)


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


@functools.cache
def _accepts_keyword(manim_class, name):
    """Whether a Manim constructor takes ``name``, directly or via ``**kwargs``."""
    try:
        parameters = inspect.signature(manim_class.__init__).parameters
    except (TypeError, ValueError):
        return True
    return name in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


class ManimCompatMob(ManimMob):
    """Base class for Mobs whose construction/query API is supplied by Manim.

    Subclasses define ``_manim_class``.  Constructor arguments use Manim's API;
    ``add_to_scene``, ``glow`` and ``glow_radius`` remain Algan-only options.
    Methods not implemented by Algan are delegated to the backing Manim object.
    Returned Mobjects are converted to Algan Mobs -- newly built ones are
    registered with the owning Scene, so ``axes.plot(...).spawn()`` renders --
    while mutations of the backing object are immediately resynchronised into
    this Mob's geometry.

    The parts are compatibility Mobs too, and a piece of this Mob's own
    hierarchy -- ``axes.x_axis``, ``axes.get_x_axis()``, ``formula[0]``,
    ``formula.get_part_by_tex("x")`` -- comes back as the part that draws it,
    so editing it edits what is on screen. A part stays the one on screen
    through edits of the whole that keep its shape (moves, recolors,
    ``set_color_by_tex``); an edit that adds or removes parts, or changes a
    path's point count (``add_coordinates``, a ``set_value`` with new digits),
    rebuilds them, and they are looked up again afterwards.
    """

    _manim_class = _manim.VMobject
    _ALGAN_ONLY_KWARGS = {
        "add_to_scene",
        "glow",
        "glow_radius",
        "batch",
        "scene",
        "miter_limit",
        "opacity",
        "unlit",
        "name",
    }
    #: True for wrappers whose Manim source typesets through LaTeX on
    #: construction (``MathTex``, ``Title``, the ``Matrix`` family, ...), so a
    #: missing TeX distribution is reported up front, in Algan's words, rather
    #: than as a ``FileNotFoundError: 'latex'`` from inside Manim after it has
    #: written a scratch file. Set by :func:`_make_manim_wrapper`.
    _needs_latex = False
    #: The constructor takes Manim's keywords as Manim spells them, so of the
    #: hints for Manim spellings only the ones about what no Mob constructor
    #: takes in either library -- timing, which Algan sets with a context --
    #: still hold.
    _constructor_keyword_hints = {
        name: _MANIM_CONSTRUCTOR_HINTS[name]
        for name in ("run_time", "duration", "rate_func", "lag_ratio", "scale_factor")
    }

    @_reports_refused_keywords
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
        # Most Manim classes forward fill and stroke opacity to VMobject, but a
        # few spell out a closed signature without them (SampleSpace has no
        # stroke_opacity). Those are applied through Algan's own properties
        # after construction, so every root class accepts both.
        component_opacity = {
            key: kwargs.pop(key)
            for key in ("fill_opacity", "stroke_opacity")
            if key in kwargs and not _accepts_keyword(self._manim_class, key)
        }
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
        for key, value in component_opacity.items():
            setattr(self, key, value)

    def _initialize_from_manim(self, source, *, batch=False, **kwargs):
        self.manim_mobject = source
        self._exposed_manim_baseline = None
        kwargs.setdefault("_part_factory", _compat_part)
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
        previous, self.manim_mobject = self.manim_mobject, source
        _rebind_parts(self, previous, source)
        # Purely intermediate: ``become`` reads the target's state into this
        # Mob's existing rows and nothing of the target survives the call, so
        # none of it may be registered as an actor.
        target = ManimMob(source, scene=self.scene, add_to_scene=False)
        before = ManimMob(before_source, scene=self.scene, add_to_scene=False)
        _preserve_algan_state_unchanged_by_manim(self, before, target)
        return self.become(target, detach_history=False)

    def _ask_each_frame(self, name, args, kwargs):
        """Ask the backing Mobject a geometry query once per frame.

        See ``_QUERIES_ASKED_PER_FRAME``. ``NotImplemented`` when the call has
        no frames to answer for, or an answer is not made of numbers.
        """
        manager = self.scene.timeline_manager
        frames = manager.replay_frames()
        values = (*args, *kwargs.values())
        counts = {len(value) for value in values if _has_frame_axis(value)}
        if not frames and not counts:
            return NotImplemented
        count = len(frames) if frames else max(counts)
        if counts - {1, count}:
            return NotImplemented
        # Synchronizing a labelled Axes takes longer than the query itself, and
        # most Mobs a query is asked of hold still for the whole batch.
        moving = bool(frames) and not all(
            bool((rows == rows[:1]).all())
            for rows in (part.location for part in self.get_descendants())
        )
        raw, answers = [], []
        for index in range(count):
            focus = (
                manager.replaying_one_frame(frames[index])
                if frames
                else contextlib.nullcontext()
            )
            with focus:
                if index == 0 or moving:
                    self._sync_manim_from_algan()
                answer = getattr(self.manim_mobject, name)(
                    *(_at_frame(value, index) for value in args),
                    **{key: _at_frame(value, index) for key, value in kwargs.items()},
                )
            try:
                answers.append(np.asarray(answer, dtype=float))
            except (TypeError, ValueError):
                return NotImplemented
            raw.append(answer)
        if not frames and count == 1:
            # Authoring, with one frame's numbers: Manim's own answer.
            return self._convert_delegated_result(raw[0])
        answers = torch.from_numpy(np.stack(answers)).to(torch.get_default_dtype())
        if name in _COORDINATE_QUERIES:
            return answers.reshape(count, -1).T.reshape(-1, count, 1, 1)
        return answers.reshape(count, 1, -1)

    def _translate_now(self, displacement):
        """Translate this Mob and its descendants by Algan's own arithmetic.

        Used while a render replays an updater, where ``displacement`` can hold
        one row per frame of the batch. The backing Mobject catches up lazily,
        as it does after a native rotation.
        """
        return self.set_location(self.location + displacement)

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
        if self.scene.timeline_manager.is_replaying():
            # Inside an updater: see _ANSWERED_FROM_ROWS_DURING_REPLAY.
            offset = _point_of(point_or_mobject, aligned_edge) - _critical_point(
                self, aligned_edge
            )
            return self._translate_now(offset * cast_to_tensor(coor_mask))
        before, source = self._prepare_manim_edit()
        source.move_to(
            to_manim(point_or_mobject),
            aligned_edge=to_manim(aligned_edge),
            coor_mask=to_manim(coor_mask),
        )
        return self._animate_to_manim(source, before_source=before)

    def move(
        self,
        displacement: torch.Tensor,
        arc_angle: float | torch.Tensor | None = None,
        recursive: bool = True,
        **kwargs,
    ) -> Mob:
        """Translate the Mob by a displacement in world space.

        Animation
        ---------
        Recorded over the current context's runtime (1 second by default).
        Call before spawning, or inside ``Off()``, to place the Mob instantly.

        Parameters
        ----------
        displacement
            Displacement in world units, shape ``(*, 3)``.
        arc_angle
            Signed arc sweep in degrees. Defaults to ``None`` for a straight
            path; otherwise the displacement joins the arc's endpoints.
        recursive
            Whether descendants move with this Mob. Defaults to True.
        **kwargs
            Additional movement options; none by default. Curved moves accept
            ``arc_normal`` to choose the arc's plane.

        Returns
        -------
        :class:`~.Mob`
            This Mob, so calls can be chained.

        Examples
        --------
        .. algan:: CompatMoveExample

            from algan import *

            axes = Axes().spawn()
            axes.move(RIGHT * 2)
            Scene.save_video()
        """
        displacement = cast_to_tensor(displacement)
        if (
            arc_angle is None
            and recursive
            and not kwargs
            and len(self.manim_mobject.points)
            and self.manim_mobject.submobjects
            # An updater's displacement can hold one row per frame of a batch.
            and not self.scene.timeline_manager.is_replaying()
        ):
            # Point-bearing composites (notably Arrow's shaft and tip) are
            # re-anchored by the Manim conversion. Keep that arithmetic at
            # their joins: a native translation can move a UHD seam pixel
            # outside the render parity tolerance even at the same pose.
            before, source = self._prepare_manim_edit()
            source.shift(to_manim(displacement))
            return self._animate_to_manim(source, before_source=before)
        target = self.location + displacement
        # Write a displacement to the existing hierarchy; rebuilding it through
        # Manim shift -> become needlessly morphs every unchanged attribute.
        # Do not call the compatibility move_to: its target is the composite's
        # center, which can differ from this Mob's location (e.g. Arrow's shaft).
        # Delegated queries synchronize the backing Manim geometry lazily, as
        # they already do for native rotations and parent-driven translations.
        if arc_angle is None:
            return self.set_location(target, recursive=recursive, **kwargs)
        return self._move_along_arc(target, arc_angle, recursive=recursive, **kwargs)

    @_scaled_natively_during_replay
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
        # An edit that only restyles or moves the parts (``set_color_by_tex``,
        # ``put_start_and_end_on``) leaves every part a conversion of a node
        # like the one it was built from. The morph below then brings each one
        # to its new state in place, and a part handed out earlier -- held as
        # ``x = formula[0]`` -- is still the one on screen afterwards. Any
        # other edit grafts part of this conversion into the hierarchy below,
        # where it has to render, so it is then built as a registered subtree.
        # Only the target's own root is discarded, and an unspawned actor never
        # reaches the renderer.
        replaying = self.scene.timeline_manager.is_replaying()
        keep_parts = replaying or _parts_mirror(self, self.manim_mobject)
        target = ManimMob(
            self.manim_mobject,
            scene=self.scene,
            add_to_scene=not keep_parts,
            _part_factory=None if keep_parts else _compat_part,
        )
        if before_source is not None:
            before = ManimMob(before_source, scene=self.scene, add_to_scene=False)
            _preserve_algan_state_unchanged_by_manim(self, before, target)
        # The target only has to supply geometry, and an unspawned Mob is a
        # perfectly good source for that. Only the part grafted in below is
        # spawned: the target's own root would draw a second copy of this Mob.
        with Off(animation_manager=self.animation_manager):
            self.become(target, detach_history=False)

        if keep_parts:
            # Inside a replay that is all a frame can show, whatever the edit
            # did: an updater the render is re-executing (a counting
            # DecimalNumber does this on every frame) cannot restructure the
            # hierarchy, since the batch's primitives were built from it as
            # authored, and doing so would leave the Scene different afterwards
            # from how it was written.
            self._exposed_manim_baseline = None
            return self

        # ``become`` morphs existing child slots, but a delegated Manim method
        # may add or remove submobjects (``add_tip``, ``add_coordinates``, ...).
        # Replace only the non-component portion of the hierarchy so this Mob's
        # render components keep their timeline identity while the composite
        # structure exactly follows the backing Manim object.
        target_non_components = target.get_non_component_children()
        grafted = {id(child) for child in target_non_components}
        retired = [
            child
            for child in self.get_non_component_children()
            if id(child) not in grafted
        ]
        self.children[:] = list(self.components) + target_non_components
        self.submobjects = target.submobjects
        self._parts_group = target._parts_group
        self._exposed_manim_baseline = None
        self._note_hierarchy_change()
        if self.is_spawned():
            # The grafted children take over from the ones they replace at
            # this moment. A replaced child left spawned stays on screen,
            # frozen where the edit put it, under the Mob that moves on.
            with Off(animation_manager=self.animation_manager):
                for child in target_non_components:
                    child.spawn(animate=False)
                for child in retired:
                    child.despawn(animate=False)
        if isinstance(self, _TypstSelection):
            # Delegated edits graft a freshly converted hierarchy. Selections
            # must follow those live paths rather than the retired source rows.
            self._setup_typst_parts(
                self.manim_mobject, batch=False, parts=target.submobjects
            )
        return self

    def _parts_by_node(self):
        """Map each node of the backing Manim hierarchy to the Mob drawing it.

        The converted hierarchy mirrors the Manim one part for part, which is
        how ``_sync_manim_node_from_algan`` pairs them too, so a node's place
        in one names its Mob in the other. A Manim node can hang in two places;
        the first in drawing order is the one that counts. A packed conversion
        maps only itself: its parts are scaffolding nothing draws.
        """
        parts = {}
        stack = [(self, self.manim_mobject)]
        while stack:
            mob, node = stack.pop()
            if id(node) in parts:
                continue
            parts[id(node)] = mob
            if isinstance(mob, ManimMob) and not mob._parts_packed:
                pairs = zip(mob._drawn_parts(), node.submobjects)
                stack.extend(reversed(list(pairs)))
        return parts

    def _live_view(self, value, parts=None, *, add_to_scene=True):
        """``value`` in Algan's terms, with this Mob's parts as the ones drawn.

        A Manim method either builds something new (``Axes.plot``,
        ``Brace.get_text``) or hands back a piece of this Mob's own hierarchy
        (``Axes.get_x_axis``, ``MathTex.get_part_by_tex``, ``formula[0]``). A
        piece of this Mob comes back as the part that draws it, so editing it
        edits what is on screen -- a converted copy would draw nothing, or draw
        the piece twice -- and a new Manim group of such pieces as a Group that
        recurses over them without adopting them. New geometry is converted
        and, unless ``add_to_scene`` is False, registered with the Scene, so
        spawning it is enough to make it render.
        """
        if parts is None:
            parts = self._parts_by_node()
        if isinstance(value, _manim.Mobject):
            part = parts.get(id(value))
            if part is not None:
                return part
            own = [mob for mob in value.get_family() if id(mob) in parts]
            if own and not len(value.points):
                return _AlganGroup(
                    [
                        self._live_view(sub, parts, add_to_scene=add_to_scene)
                        for sub in value.submobjects
                    ],
                    scene=self.scene,
                    add_to_scene=False,
                    link_children=False,
                )
            family = {id(mob) for mob in self.manim_mobject.get_family()}
            if own or id(value) in family:
                # Geometry this Mob already draws, in a form no part of it
                # matches (a packed part, a new path holding old ones).
                return from_manim(value, scene=self.scene, add_to_scene=False)
            return from_manim(value, scene=self.scene, add_to_scene=add_to_scene)
        if isinstance(value, Mapping):
            return type(value)(
                (key, self._live_view(item, parts, add_to_scene=add_to_scene))
                for key, item in value.items()
            )
        if isinstance(value, (tuple, list)):
            items = [
                self._live_view(item, parts, add_to_scene=add_to_scene)
                for item in value
            ]
            return tuple(items) if isinstance(value, tuple) else items
        return from_manim(value, scene=self.scene, add_to_scene=add_to_scene)

    def _convert_delegated_result(self, result):
        """Convert a delegated Manim method's return value into Algan values."""
        return self._live_view(result)

    def _refuse_packed_parts(self):
        if getattr(self, "_parts_packed", False):
            raise TypeError(
                f"{type(self).__name__} was built with batch=True, which packs "
                "its parts into one Mob, so they cannot be picked out. Build it "
                "without batch to index its parts."
            )

    def __getitem__(self, item):
        """Get the part Manim's ``mobject[item]`` names, as the Mob drawing it.

        Indexing follows Manim exactly -- ``formula[0]`` is the first
        substring, ``graph[1]`` vertex 1, and a Mobject with points of its own
        counts itself first -- and the part that comes back is the one on
        screen: color it or move it and this Mob changes. A slice is a Group
        that recurses over the parts it selects without adopting them; it is
        not added to the Scene and never needs spawning.

        Animation
        ---------
        Not animated: indexing only selects. Animate the result to change what
        it covers.

        Parameters
        ----------
        item
            Whatever the backing Manim class indexes by: a position or a slice,
            a vertex of a ``Graph``, a key of a ``VDict``.

        Returns
        -------
        :class:`~.Mob`
            The selected part, or a Group view of the selected parts.

        Raises
        ------
        TypeError
            If this Mob was built with ``batch=True``.
        """
        self._refuse_packed_parts()
        return self._live_view(self.manim_mobject[item])

    def __iter__(self):
        """Iterate over the parts Manim's ``for part in mobject`` visits."""
        self._refuse_packed_parts()
        parts = self._parts_by_node()
        return iter([self._live_view(part, parts) for part in self.manim_mobject])

    def __len__(self):
        """Count the parts Manim's ``len(mobject)`` counts."""
        if "manim_mobject" not in self.__dict__:
            return super().__len__()
        return len(self.manim_mobject)

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
            # accumulate actors.  Parts of this Mob (``axes.x_axis``) come back
            # as the parts drawn; anything else is a copy no render sees -- use
            # the method that built it (or ``Scene.add_actor``) to draw it.
            return self._live_view(attribute, add_to_scene=False)

        @wraps(attribute)
        def delegated(*args, **kwargs):
            answer = _ANSWERED_FROM_ROWS_DURING_REPLAY.get(name)
            if answer is not None and self.scene.timeline_manager.is_replaying():
                result = answer(self, *args, **kwargs)
                if result is not NotImplemented:
                    return result
            if name in _QUERIES_ASKED_PER_FRAME:
                result = self._ask_each_frame(name, args, kwargs)
                if result is not NotImplemented:
                    return result
            # A delegated method can be saved and called later.  Parent edits may
            # happen between attribute lookup and invocation, so synchronize and
            # re-bind the Manim method at call time rather than relying on the
            # bound method captured by ``__getattr__``.
            self._sync_manim_from_algan()
            before_source = self.manim_mobject.copy()
            before_topology = tuple(
                (id(mob), tuple(id(child) for child in mob.submobjects))
                for mob in self.manim_mobject.get_family()
            )
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
            after_topology = tuple(
                (id(mob), tuple(id(child) for child in mob.submobjects))
                for mob in self.manim_mobject.get_family()
            )
            if after_topology != before_topology:
                self.sync_from_manim(before_source=before_source)
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
    "Typst": _NativeTypst.__doc__.replace(
        "Algan stroke units", "Manim stroke units (twice Algan's default unit)"
    ),
    "MathTypst": _NativeMathTypst.__doc__,
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
        (
            (_TypstSelection, ManimCompatMob)
            if name in {"Typst", "MathTypst"}
            else (ManimCompatMob,)
        ),
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
    if name in {"Typst", "MathTypst"}:
        wrapper.select = _TypstSelection.select
        wrapper.get_baseline_frame = _TypstSelection.get_baseline_frame
        wrapper.baseline_frames = _TypstSelection.baseline_frames
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
    "MathTypst",
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
    "Typst",
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
        """The tracked value, as a float.

        A render replays an updater that reads it once per frame, so the
        updater gets each frame's value, as it does in Manim, whatever it does
        with it -- ``math.sin``, an f-string, an ``if``.
        """
        timeline = self.scene.timeline_manager
        timeline.replay_running_updater_one_frame_at_a_time()
        values = self.value.reshape(-1)
        if timeline.is_replaying() and bool((values != values[0]).any()):
            # Frames with different values at once: an updater that first read
            # the tracker during this render. It is replayed one frame at a
            # time from its next batch on; for this one, a value per frame, in
            # the shape of its ``t``.
            return self.value.reshape(-1, 1, 1)
        return float(values[0])

    def set_value(self, value):
        value = torch.as_tensor(value, dtype=torch.get_default_dtype())
        # One value per frame when an updater sets it from its ``t``.
        self.value = (
            value.reshape(-1, 1, 1) if value.numel() > 1 else value.reshape(1, 1)
        )
        return self

    def increment_value(self, d_value):
        return self.set_value(self.get_value() + d_value)

    def __iadd__(self, value):
        return self.increment_value(value)

    def __isub__(self, value):
        return self.increment_value(-value)

    def __float__(self):
        value = self.get_value()
        return float(value.reshape(-1)[0]) if torch.is_tensor(value) else value


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
        """The tracked value, as a complex number.

        As :meth:`ValueTracker.get_value`, an updater that reads it gets each
        frame's value.
        """
        timeline = self.scene.timeline_manager
        timeline.replay_running_updater_one_frame_at_a_time()
        values = self.complex_value.reshape(-1, 2)
        if timeline.is_replaying() and bool((values != values[:1]).any()):
            # As in ValueTracker.get_value.
            return torch.complex(values[:, 0], values[:, 1]).reshape(-1, 1, 1)
        return complex(float(values[0, 0]), float(values[0, 1]))

    def set_value(self, value):
        if isinstance(value, torch.Tensor) and value.numel() > 1:
            # One value per frame when an updater sets it from its ``t``.
            value = value.to(torch.complex64).reshape(-1, 1)
            self.complex_value = torch.stack((value.real, value.imag), -1).to(
                torch.get_default_dtype()
            )
            return self
        value = complex(value)
        self.complex_value = torch.tensor(
            (value.real, value.imag), dtype=torch.get_default_dtype()
        ).reshape(1, 2)
        return self

    def increment_value(self, d_value):
        return self.set_value(self.get_value() + d_value)

    def __complex__(self):
        value = self.get_value()
        return complex(value.reshape(-1)[0]) if torch.is_tensor(value) else value


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
