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
from collections.abc import Mapping
from functools import wraps
from typing import Any

import manim as _manim
import numpy as np
import torch

from algan.animatable_base.mob import Mob
from algan.animation_timeline.animation_contexts import Off
from algan.animation_timeline.timeline import bump_hierarchy_version
from algan.constants.color import Color
from algan.mobs.bezier_circuit import BezierCircuitCubic
from algan.mobs.group import Group
from algan.mobs.image_mob import ImageMob
from algan.mobs.manim_mob import ManimMob
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

    border = mob.border_color[0].reshape(-1, mob.border_color.shape[-1])[0]
    stroke_color = "#" + "".join(
        f"{round(float(x) * 255):02X}" for x in border[:3].detach().cpu().clamp(0, 1)
    )
    stroke_opacity = float(border[-1].detach().cpu())
    stroke_width = float(mob.border_width[0].reshape(-1)[0].detach().cpu()) * 2
    result.set_stroke(stroke_color, width=stroke_width, opacity=stroke_opacity)
    return result


def _algan_mob_to_manim(mob: Mob):
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
            from_manim(item, scene=scene, add_to_scene=add_to_scene)
            for item in value
        )
    if isinstance(value, list):
        return [
            from_manim(item, scene=scene, add_to_scene=add_to_scene)
            for item in value
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
    _ALGAN_ONLY_KWARGS = {
        "add_to_scene", "glow", "glow_radius", "batch", "scene"
    }

    def __init__(self, *args, **kwargs):
        algan_kwargs = {
            key: kwargs.pop(key)
            for key in tuple(kwargs)
            if key in self._ALGAN_ONLY_KWARGS
        }
        batch = bool(algan_kwargs.pop("batch", False))
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
        super().__init__(source, batch=batch, **kwargs)

    @classmethod
    def _from_manim(cls, source, *, scene=None, add_to_scene=True):
        obj = cls.__new__(cls)
        obj._initialize_from_manim(
            source, scene=scene, add_to_scene=add_to_scene
        )
        return obj

    def get_manim_mobject(self):
        """Return the backing Manim object used for compatibility operations."""
        return self.manim_mobject

    def _animate_to_manim(self, source):
        """Record an Algan morph to an edited copy of the backing Mobject."""
        self.manim_mobject = source
        # Purely intermediate: ``become`` reads the target's state into this
        # Mob's existing rows and nothing of the target survives the call, so
        # none of it may be registered as an actor.
        target = ManimMob(source, scene=self.scene, add_to_scene=False)
        return self.become(target, detach_history=False)

    # These names also exist on Algan's Mob.  Override them so compatibility
    # objects retain Manim's units, keyword arguments, and backing geometry.
    def move_to(self, point_or_mobject, aligned_edge=_manim.ORIGIN, coor_mask=np.array([1, 1, 1])):
        source = self.manim_mobject.copy()
        source.move_to(
            to_manim(point_or_mobject),
            aligned_edge=to_manim(aligned_edge),
            coor_mask=to_manim(coor_mask),
        )
        return self._animate_to_manim(source)

    def move(self, displacement, path_arc_angle=None, recursive=True, **kwargs):
        """Move by a displacement, applying it as a Manim ``shift``.

        Algan's generic implementation moves to ``self.location + displacement``,
        but a compatibility Mob's location is the center of the backing
        Mobject's *own* points, which is not the composite's center whenever it
        also has submobjects (an :class:`Arrow`'s tip, for example).  Shifting
        the backing geometry instead keeps the travelled displacement exact for
        every Mob, and is what the relative-placement helpers
        (:meth:`~.Mob.move_to_edge`, :meth:`~.Mob.move_next_to`, ...) are built
        on.
        """
        displacement = cast_to_tensor(displacement)
        if path_arc_angle is not None or not recursive or kwargs:
            # Curved paths and non-recursive moves have no Manim equivalent.
            # Let Algan record the motion, then bring the backing Mobject to
            # the same final position so delegated queries stay accurate.
            target = self.location + displacement
            if path_arc_angle is None:
                self.set_location(target, recursive=recursive, **kwargs)
            else:
                self.move_to_point_along_arc(
                    target, path_arc_angle, recursive=recursive, **kwargs
                )
            self.manim_mobject = self.manim_mobject.copy().shift(
                to_manim(displacement)
            )
            return self
        source = self.manim_mobject.copy()
        source.shift(to_manim(displacement))
        return self._animate_to_manim(source)

    def scale(self, scale_factor, **kwargs):
        source = self.manim_mobject.copy()
        source.scale(
            scale_factor,
            **{key: to_manim(value) for key, value in kwargs.items()},
        )
        return self._animate_to_manim(source)

    def rotate(self, angle, axis=_manim.OUT, about_point=None, **kwargs):
        source = self.manim_mobject.copy()
        source.rotate(
            angle,
            axis=to_manim(axis),
            about_point=None if about_point is None else to_manim(about_point),
            **{key: to_manim(value) for key, value in kwargs.items()},
        )
        return self._animate_to_manim(source)

    def set(self, **kwargs):
        # Algan's internal morphing path calls ``set`` with animatable state
        # attributes. Preserve that path; delegate user-facing Manim property
        # updates such as ``set(width=...)`` to the backing object.
        if set(kwargs).issubset(set(self.animatable_attrs)):
            return Mob.set(self, **kwargs)
        source = self.manim_mobject.copy()
        source.set(
            **{key: to_manim(value) for key, value in kwargs.items()}
        )
        return self._animate_to_manim(source)

    def copy(self):
        # Matches :meth:`~.Animatable.clone`, which registers the copy: a copy
        # you cannot render is of no use to the caller.
        return _wrapper_type_for(self.manim_mobject)._from_manim(
            self.manim_mobject.copy(), scene=self.scene, add_to_scene=True
        )

    def sync_from_manim(self):
        """Refresh converted geometry after directly editing the backing object."""
        # Unlike ``_animate_to_manim``'s morph target, part of this conversion is
        # grafted into this Mob's hierarchy below and has to render, so it is
        # built as a registered subtree.  Only the target's own root is
        # discarded, and an unspawned actor never reaches the renderer.
        target = ManimMob(self.manim_mobject, scene=self.scene, add_to_scene=True)
        if self.is_spawned():
            target = target.spawn(animate=False)
        with Off(animation_manager=self.animation_manager):
            self.become(target, detach_history=False)

        # ``become`` morphs existing child slots, but a delegated Manim method
        # may add or remove submobjects (``add_tip``, ``add_coordinates``, ...).
        # Replace only the non-component portion of the hierarchy so this Mob's
        # render components keep their timeline identity while the composite
        # structure exactly follows the backing Manim object.
        target_non_components = target.get_non_component_children()
        self.children = list(self.components) + target_non_components
        self.submobjects = target_non_components
        bump_hierarchy_version()
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
        if name.startswith("_") or "manim_mobject" not in self.__dict__:
            raise AttributeError(name)
        attribute = getattr(self.manim_mobject, name)
        if not callable(attribute):
            # Reading an attribute is a query, not construction: it must stay
            # free of side effects on the Scene, and repeated reads must not
            # accumulate actors.  Use the method that built the geometry (or
            # ``Scene.add_actor``) to get a renderable Mob.
            return from_manim(attribute, scene=self.scene, add_to_scene=False)

        @wraps(attribute)
        def delegated(*args, **kwargs):
            result = attribute(
                *(to_manim(arg) for arg in args),
                **{key: to_manim(value) for key, value in kwargs.items()},
            )
            if result is self.manim_mobject:
                self.sync_from_manim()
                return self
            if result is None:
                self.sync_from_manim()
                return None
            return self._convert_delegated_result(result)

        return delegated

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(dir(self.manim_mobject)))


def _make_manim_wrapper(name: str):
    manim_class = getattr(_manim, name)
    wrapper = type(
        name,
        (ManimCompatMob,),
        {
            "_manim_class": manim_class,
            "__module__": __name__,
            "__doc__": manim_class.__doc__,
        },
    )
    with contextlib.suppress(TypeError, ValueError):
        wrapper.__signature__ = inspect.signature(manim_class)
    _MANIM_WRAPPER_REGISTRY[name] = wrapper
    globals()[name] = wrapper
    return wrapper


# Classes that are cubic-Bezier/image/composite Mobjects in the vendored Manim
# implementation. Existing native Algan classes are intentionally omitted.
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
    "ComplexPlane",
    "ComplexValueTracker",
    "ConvexHull",
    "Cross",
    "CubicBezier",
    "CurvedArrow",
    "CurvedDoubleArrow",
    "CurvesAsSubmobjects",
    "Cutout",
    "DashedLine",
    "DashedVMobject",
    "DecimalMatrix",
    "DecimalNumber",
    "DecimalTable",
    "DiGraph",
    "Difference",
    "DoubleArrow",
    "Elbow",
    "Ellipse",
    "Exclusion",
    "FullScreenRectangle",
    "FunctionGraph",
    "Graph",
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
    "ManimBanner",
    "MathTable",
    "MathTex",
    "Matrix",
    "MobjectMatrix",
    "MobjectTable",
    "NumberLine",
    "NumberPlane",
    "ParametricFunction",
    "PolarPlane",
    "Polygram",
    "RegularPolygram",
    "RightAngle",
    "RoundedRectangle",
    "SVGMobject",
    "SampleSpace",
    "ScreenRectangle",
    "Sector",
    "SingleStringMathTex",
    "Star",
    "StealthTip",
    "StreamLines",
    "Table",
    "TangentLine",
    "ThreeDAxes",
    "ThreeDVMobject",
    "TipableVMobject",
    "Title",
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


from manim.mobject.svg.brace import BraceText as _ManimBraceText


class BraceText(ManimCompatMob):
    """Brace with a plain-text label, matching Manim 0.20.1."""

    _manim_class = _ManimBraceText


BraceText.__signature__ = inspect.signature(_ManimBraceText)
_MANIM_WRAPPER_REGISTRY["BraceText"] = BraceText


_ManimLabeledDot = _manim.LabeledDot


class LabeledDot(ManimCompatMob):
    """A dot containing a centered label, matching Manim 0.20.1.

    Manim 0.20 added the ``buff`` parameter after the version vendored by
    Algan.  Supplying the computed radius to the vendored implementation
    preserves the new sizing rule without duplicating its remaining geometry.
    """

    _manim_class = _ManimLabeledDot

    def __init__(self, label, radius=None, buff=_manim.SMALL_BUFF, **kwargs):
        converted_label = to_manim(label)
        if radius is None:
            if isinstance(converted_label, str):
                converted_label = _manim.MathTex(converted_label, color=_manim.BLACK)
            radius = float(buff) + float(
                np.linalg.norm([converted_label.width, converted_label.height]) / 2
            )
        super().__init__(converted_label, radius=radius, **kwargs)


LabeledDot.__signature__ = inspect.Signature(
    parameters=[
        inspect.Parameter("label", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        inspect.Parameter(
            "radius", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None
        ),
        inspect.Parameter(
            "buff",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default=_manim.SMALL_BUFF,
        ),
        inspect.Parameter("kwargs", inspect.Parameter.VAR_KEYWORD),
    ]
)
_MANIM_WRAPPER_REGISTRY["LabeledDot"] = LabeledDot


class TangentialArc(ArcBetweenPoints):
    """An arc tangent to two intersecting lines (Manim 0.20 API)."""

    def __init__(self, line1, line2, radius, corner=(1, 1), **kwargs):
        def point(value):
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            return np.asarray(value, dtype=float).reshape(-1, 3)[0]

        p1, p2 = point(line1.get_start()), point(line1.get_end())
        p3, p4 = point(line2.get_start()), point(line2.get_end())
        d1, d2 = p2 - p1, p4 - p3
        cross = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(cross) < 1e-12:
            raise ValueError("TangentialArc requires intersecting, non-parallel lines")
        delta = p3 - p1
        t = (delta[0] * d2[1] - delta[1] * d2[0]) / cross
        intersection = p1 + t * d1
        d1 = d1 / np.linalg.norm(d1)
        d2 = d2 / np.linalg.norm(d2)
        s1, s2 = corner
        u1, u2 = s1 * d1, s2 * d2
        angle = np.arccos(np.clip(np.dot(u1, u2), -1.0, 1.0))
        distance = radius / np.tan(angle / 2)
        tangent1 = intersection + distance * u1
        tangent2 = intersection + distance * u2
        cross_u = u1[0] * u2[1] - u1[1] * u2[0]
        start, end = (tangent1, tangent2) if cross_u < 0 else (tangent2, tangent1)
        self.line1 = line1
        self.line2 = line2
        super().__init__(start=start, end=end, radius=radius, **kwargs)


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
        self.value = torch.as_tensor(value, dtype=torch.get_default_dtype()).reshape(1, 1)
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
        tensor = torch.tensor((value.real, value.imag), dtype=torch.get_default_dtype()).reshape(1, 2)
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


# Current Manim 0.20.1 introduced this marker class for MathTex pieces.  It has
# no constructor of its own, so the vendored SingleStringMathTex behavior is the
# closest meaningful compatibility type.
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
    *_WRAPPED_MANIM_CLASS_NAMES,
]
