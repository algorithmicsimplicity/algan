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
    Manim's stroke width is twice Algan's. **Not converted here** -- the
    native attribute is still spelled ``border_width`` and its unit is settled
    in the border/stroke rename, which is where that conversion belongs. No
    adapter declares it yet.

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


def _make_adapter(name: str, angle_params: tuple[str, ...]):
    wrapper = _MANIM_WRAPPER_REGISTRY[name]
    signature = getattr(wrapper, "__signature__", None)

    if not angle_params:
        # Nothing to convert: the native spelling is the wrapper itself under a
        # subclass, so ``type(mob)`` still reports the class the user named.
        return type(
            name, (wrapper,), {"__module__": __name__, "__doc__": wrapper.__doc__}
        )

    def __init__(self, *args, **kwargs):
        args, kwargs = _converted_kwargs(signature, angle_params, args, kwargs)
        super(adapter, self).__init__(*args, **kwargs)

    listed = ", ".join(f"``{p}``" for p in angle_params)
    doc = (
        f"{wrapper.__doc__ or name}\n\n"
        f"    Algan's spelling: {listed} are given in **degrees**, matching\n"
        f"    :meth:`~algan.animatable_base.mob_orientation.MobOrientationMixin.rotate`\n"
        f"    and the rest of Algan. ``algan.manim.{name}`` is the same class\n"
        f"    taking radians, as Manim does.\n"
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
