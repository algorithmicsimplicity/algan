"""Native Algan classes for Manim geometry Algan has no implementation of.

:mod:`algan.manim` holds Manim's Mobjects under Manim's conventions. Most of
them have no native Algan counterpart at all -- there is no Algan ``Axes``,
``Brace`` or ``Arc`` -- and leaving them reachable only as ``mn.Axes`` would
take the ordinary building blocks of an explanatory maths video out of
``from algan import *``.

The classes here close that gap. Each is a thin subclass of the corresponding
compatibility wrapper that converts Algan's conventions to Manim's on the way
in and otherwise delegates, so the same geometry is reachable both ways and
each namespace keeps its own units::

    Arc(angle=90)  # native: degrees, like rotate() and orbit()
    mn.Arc(angle=PI / 2)  # Manim's: radians

Conversions are declared per class in :data:`_ADAPTED`, not written out by
hand, so the table is the specification and a class cannot quietly drift from
it. Only two conventions differ; the second is not implemented yet:

``angle_params``
    Angles. Algan states angles in degrees everywhere (``rotate``, ``orbit``,
    ``SpotLight``), Manim in radians.

``stroke_width``
    Manim's stroke width is twice Algan's, so an adapter halves nothing and
    **doubles** on the way in: ``Arrow(stroke_width=4)`` and
    ``mn.Arrow(stroke_width=8)`` draw the same outline. Unlike the angle
    conversions this is not declared per class, because a Manim class accepts
    ``stroke_width`` whether or not its own signature names it -- ``Star`` and
    ``DashedLine`` take it through ``**kwargs`` to ``VMobject`` -- and a
    signature-driven table would give those two Manim's unit while the six
    that declare it got Algan's.

An adapter converts the conventions it declares and nothing else. Manim
behaviour outside those parameters still shows through -- ``Title`` positions
against Manim's 8-unit frame rather than Algan's, as its own docstring warns.
That is a known limit of delegating rather than reimplementing.
"""

from __future__ import annotations

from algan.constants.math import DEGREES_TO_RADIANS
from algan.mobs.manim_compat import _MANIM_WRAPPER_REGISTRY

#: Manim class name -> parameters whose value arrives in degrees and must be
#: handed to Manim in radians. A name mapped to ``()`` needs no conversion and
#: is adapted purely to give it a root-namespace spelling.
#:
#: Checked against the vendored Manim signatures: these are every curated class
#: carrying an angle parameter. ``Table`` and ``Matrix`` are *not* among them --
#: they match an "angle" substring search only through
#: ``background_rectangle``.
_ANGLE_PARAMS: dict[str, tuple[str, ...]] = {
    "Angle": ("other_angle",),
    "AnnularSector": ("angle", "start_angle"),
    "Arc": ("angle", "start_angle"),
    "ArcBetweenPoints": ("angle",),
    "ArcPolygon": ("angle",),
    "Elbow": ("angle",),
    "NumberLine": ("rotation",),
    "Sector": ("angle", "start_angle"),
}

#: The curated subset: compatibility classes that earn a native spelling.
#:
#: A class qualifies when it (a) has no native Algan equivalent, (b) is
#: something an author reaches for directly, and (c) is not a Manim
#: architecture or renderer construct. (a) is enforced below rather than
#: trusted -- a name that gains a native implementation must leave this list,
#: or the root namespace would carry two spellings of it.
_ADAPTED: tuple[str, ...] = (
    # geometry
    "Angle",
    "AnnularSector",
    "Annulus",
    "Arc",
    "ArcBetweenPoints",
    "ArcPolygon",
    "ConvexHull",
    "Cross",
    "CubicBezier",
    "DashedLine",
    "Elbow",
    "Ellipse",
    "Polygram",
    "RegularPolygram",
    "RightAngle",
    "RoundedRectangle",
    "Sector",
    "Star",
    "TangentLine",
    # boolean operations
    "Cutout",
    "Difference",
    "Exclusion",
    "Intersection",
    "Union",
    # arrows
    "Arrow",
    "CurvedArrow",
    "CurvedDoubleArrow",
    "DoubleArrow",
    "Vector",
    # plots and coordinate systems
    "Axes",
    "BarChart",
    "ComplexPlane",
    "FunctionGraph",
    "ImplicitFunction",
    "NumberLine",
    "NumberPlane",
    "ParametricFunction",
    "PolarPlane",
    "ThreeDAxes",
    # graphs
    "DiGraph",
    "Graph",
    # tables and matrices
    "DecimalMatrix",
    "DecimalTable",
    "IntegerMatrix",
    "IntegerTable",
    "MathTable",
    "Matrix",
    "MobjectMatrix",
    "MobjectTable",
    "Table",
    # braces and labels
    "Brace",
    "BraceBetweenPoints",
    "BraceLabel",
    "BraceText",
    "Label",
    "LabeledArrow",
    "LabeledDot",
    "LabeledLine",
    "LabeledPolygram",
    # text
    "BulletedList",
    "MathTex",
    "SVGMobject",
    "Title",
    "Underline",
    # values
    "Integer",
    "Variable",
)


def _converted_kwargs(signature, angle_params, args, kwargs):
    """Map ``args``/``kwargs`` onto parameter names and convert the angles.

    Only parameters the caller actually supplied are converted. Manim's own
    defaults are already radians and already right -- ``Arc``'s ``angle`` of
    ``TAU/4`` is a quarter turn in either convention -- so applying defaults
    before converting would read that ``1.57`` as degrees and build a 1.57
    degree arc.
    """
    if signature is None:
        # No usable signature: only keywords can be identified, which is what
        # every angle-carrying class in the table is called with anyway.
        supplied = kwargs
    else:
        bound = signature.bind_partial(*args, **kwargs)
        supplied = bound.arguments

    converted = dict(kwargs)
    positional = list(args)
    for name in angle_params:
        if name not in supplied:
            continue
        value = supplied[name]
        if value is None:
            continue
        if name in converted:
            converted[name] = value * DEGREES_TO_RADIANS
        else:
            # Supplied positionally: rebuild that slot in place.
            index = list(signature.parameters).index(name) - 1  # drop ``self``
            positional[index] = value * DEGREES_TO_RADIANS
    return tuple(positional), converted


def _to_manim_stroke_width(kwargs):
    """Double an Algan-unit ``stroke_width`` into Manim's, in place.

    Applied to every adapter rather than to a declared list: a Manim class
    accepts ``stroke_width`` whether or not its signature names it, so a
    per-class table would leave ``Star`` and ``DashedLine`` -- which take it
    through ``**kwargs`` to ``VMobject`` -- on Manim's unit while the classes
    that declare it moved to Algan's.
    """
    width = kwargs.get("stroke_width")
    if width is not None:
        kwargs["stroke_width"] = width * 2
    return kwargs


def _make_adapter(name: str, angle_params: tuple[str, ...]):
    wrapper = _MANIM_WRAPPER_REGISTRY[name]
    signature = getattr(wrapper, "__signature__", None)

    def __init__(self, *args, **kwargs):
        if angle_params:
            args, kwargs = _converted_kwargs(signature, angle_params, args, kwargs)
        super(adapter, self).__init__(*args, **_to_manim_stroke_width(kwargs))

    converted = [
        "``stroke_width`` is in Algan's unit, half Manim's for the same visual weight"
    ]
    if angle_params:
        listed = ", ".join(f"``{p}``" for p in angle_params)
        converted.insert(0, f"{listed} are given in **degrees**")
    doc = (
        f"{wrapper.__doc__ or name}\n\n"
        f"    Algan's spelling: {'; '.join(converted)}.\n"
        f"    ``algan.manim.{name}`` is the same class under Manim's\n"
        f"    conventions -- radians, and Manim's stroke unit.\n"
    )
    adapter = type(
        name,
        (wrapper,),
        {"__init__": __init__, "__module__": __name__, "__doc__": doc},
    )
    if signature is not None:
        adapter.__signature__ = signature
    return adapter


def _build():
    missing = [n for n in _ADAPTED if n not in _MANIM_WRAPPER_REGISTRY]
    if missing:
        raise RuntimeError(
            f"manim_adapters names not present in the compatibility layer: {missing}"
        )
    stray = [n for n in _ANGLE_PARAMS if n not in _ADAPTED]
    if stray:
        raise RuntimeError(
            f"_ANGLE_PARAMS declares conversions for non-adapted classes: {stray}"
        )
    for name in _ADAPTED:
        globals()[name] = _make_adapter(name, _ANGLE_PARAMS.get(name, ()))


_build()

__all__ = list(_ADAPTED)

del _build
