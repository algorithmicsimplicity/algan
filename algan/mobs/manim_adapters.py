r"""Native Algan spellings for Manim's Mobjects, in Algan's conventions.

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

The adapted set is everything, by default
-----------------------------------------
The rule is inverted from a curated allow-list: **every** class in the
compatibility registry gets a root spelling unless it is named in
:data:`_NATIVE` (Algan implements it itself, so the native class keeps the
root name) or :data:`_NOT_ADAPTED` (a Manim base class or architecture
construct that is not something an author builds). :data:`_ADAPTED` is what
is left, computed rather than listed, so a class added to the compatibility
layer is adapted automatically and a class that should not be has to say why.

Two conventions differ, and both are converted here and nowhere else:

``stroke_width``
    Manim's stroke width is twice Algan's, so an adapter halves nothing and
    **doubles** on the way in: ``Arrow(stroke_width=4)`` and
    ``mn.Arrow(stroke_width=8)`` draw the same outline. This is applied to
    every adapter unconditionally, not declared per class, because a Manim
    class accepts ``stroke_width`` whether or not its own signature names it
    -- ``Star`` and ``DashedLine`` take it through ``**kwargs`` to
    ``VMobject`` -- and a signature-driven table would give those two Manim's
    unit while the ones that declare it got Algan's.

``angles``
    Algan states angles in degrees everywhere (``rotate``, ``orbit``,
    ``SpotLight``), Manim in radians. Which parameters those are is *derived*,
    not hand-listed: :func:`_angle_params_for` walks each Manim class's whole
    MRO -- ``Sector`` takes ``angle`` through ``**kwargs`` from
    ``AnnularSector`` and ``ArrowTriangleFilledTip`` takes ``start_angle``
    from ``ArrowTriangleTip``, neither of which their own signature shows --
    collects every parameter whose name looks like an angle or whose default
    is a multiple of :math:`\pi/4`, and classifies each against
    :data:`_ANGLE_PARAM_NAMES` and :data:`_NOT_ANGLE_PARAM_NAMES`. A
    parameter in neither raises at import, so a Manim upgrade that adds an
    angle cannot slip past silently into a wrong conversion.

An adapter converts the conventions it declares and nothing else. Manim
behaviour outside those parameters still shows through -- ``Title`` positions
against Manim's 8-unit frame rather than Algan's, as its own docstring warns.
That is a known limit of delegating rather than reimplementing.

The z axis is not converted here. Manim's ``OUT`` is ``+z`` where Algan's
``OUTWARD`` is ``-z``, but that is a property of a whole scene's basis rather
than of a constructor argument -- mirroring one object without its camera and
lights renders it back-to-front -- so it stays with
:attr:`Scene.manim_coordinates <algan.scene.Scene.manim_coordinates>`.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Mapping, Sequence

from algan.constants.math import DEGREES_TO_RADIANS
from algan.mobs.manim_compat import _MANIM_WRAPPER_REGISTRY

#: Compatibility classes Algan implements natively. The native class keeps the
#: root name and Manim's stays reachable as ``algan.manim.<name>``; adapting
#: one as well would put two spellings of the same thing in the root
#: namespace, which is what this boundary exists to prevent.
#:
#: ``tests/unit_tests/test_public_api_surface.py`` checks this against what the
#: root namespace actually resolves each name to, so a class that gains or
#: loses a native implementation fails there rather than silently shadowing.
_NATIVE: frozenset[str] = frozenset(
    {
        "Arrow3D",
        "Circle",
        "Code",
        "Cone",
        "ConvexHull3D",
        "Cube",
        "Cylinder",
        "DecimalNumber",
        "Dodecahedron",
        "Dot",
        "Dot3D",
        "Group",
        "Icosahedron",
        "Line",
        "Line3D",
        "MarkupText",
        "Octahedron",
        "Paragraph",
        "Point",
        "Polygon",
        "Polyhedron",
        "Prism",
        "Rectangle",
        "RegularPolygon",
        "Sphere",
        "Square",
        "Surface",
        "SurroundingRectangle",
        "Tetrahedron",
        "Tex",
        "Text",
        "Torus",
        "Triangle",
    }
)

#: Compatibility classes that stay ``mn.``-only, and why. These are Manim's
#: own scaffolding rather than geometry an author reaches for: abstract bases
#: whose only use at the root would be ``isinstance``, container types Algan
#: spells differently, and value holders that are not Mobjects at all.
_NOT_ADAPTED: dict[str, str] = {
    "ArrowTip": "abstract base; the concrete tips are adapted",
    "ComplexValueTracker": "Manim's animation model, not a Mob",
    "ManimBanner": "Manim's own branding, not geometry",
    "SingleStringMathTex": "internal to MathTex's parsing",
    "ThreeDVMobject": "abstract base",
    "TipableVMobject": "abstract base",
    "VDict": "Algan groups with Group",
    "VGroup": "Algan groups with Group",
    "VMobject": "abstract base; Algan's is Mob",
    "VMobjectFromSVGPath": "internal to SVGMobject's parsing",
    "ValueTracker": "Manim's animation model, not a Mob",
    "VectorizedPoint": "abstract base; Algan's is Point",
}

#: Parameter names whose value is an angle in radians on Manim's side and in
#: degrees on Algan's. Names, not (class, name) pairs: Manim is consistent
#: about these across its geometry, and one flat set is what lets a newly
#: adapted class be covered without a new entry.
_ANGLE_PARAM_NAMES: frozenset[str] = frozenset(
    {
        "angle",
        "azimuth_offset",
        "path_arc",
        "rotation",
        "start_angle",
    }
)

#: Parameter names that look like angles to :func:`_angle_params_for`'s
#: detector but are not, with what they actually are:
#:
#: ``arc``/``arcs``  Arc *Mobjects* (``ArcBrace``, ``ArcPolygonFromArcs``).
#: ``arc_center``    A point.
#: ``arc_config``    Per-arc keyword dicts -- handled by
#:                   :data:`_NESTED_ANGLE_PARAMS`, which converts the angles
#:                   *inside* it.
#: ``azimuth_*``     ``PolarPlane``'s label and tick styling: a count, two
#:                   strings, a flag, a buffer and a font size. Only
#:                   ``azimuth_offset`` among them is an angle.
#: ``other_angle``   A **bool** on ``Angle``, selecting the explementary
#:                   angle. Converting it (as this module once did) multiplied
#:                   ``True`` by ``pi/180``.
#: ``*background_rectangle*``  Matches only on the substring "angle" in
#:                   "rectangle".
_NOT_ANGLE_PARAM_NAMES: frozenset[str] = frozenset(
    {
        "add_background_rectangles_to_entries",
        "arc",
        "arc_center",
        "arc_config",
        "arcs",
        "azimuth_compact_fraction",
        "azimuth_direction",
        "azimuth_label_buff",
        "azimuth_label_font_size",
        "azimuth_step",
        "azimuth_units",
        "background_rectangle_color",
        "include_background_rectangle",
        "other_angle",
    }
)

#: Parameters that carry angles one level down, as a mapping of Manim keyword
#: arguments (or a sequence of them). ``ArcPolygon(arc_config={"angle": 90})``
#: reaches ``Arc`` through here, so the degrees have to be converted inside
#: the dict; the detector cannot see into it and the top-level waiver above
#: would otherwise leave it in Manim's unit while the sibling ``angle``
#: parameter took Algan's.
_NESTED_ANGLE_PARAMS: dict[str, tuple[str, ...]] = {
    "arc_config": tuple(sorted(_ANGLE_PARAM_NAMES)),
}

#: Names that make a parameter a candidate angle. Deliberately over-broad --
#: every match is classified explicitly, so a false positive costs a waiver
#: entry while a miss would cost a silent wrong conversion.
_ANGLE_NAME_HINTS: tuple[str, ...] = (
    "angle",
    "arc",
    "azimuth",
    "degree",
    "phi",
    "radian",
    "rotat",
    "theta",
    "tilt",
    "turn",
)


def _looks_like_radians(default) -> bool:
    """Whether ``default`` is a non-zero exact multiple of ``pi/4``.

    The second detector, and the one that catches an angle whose name gives
    nothing away. Manim writes its angle defaults as ``PI``, ``TAU / 4``,
    ``PI / 2`` and the like, which land on this grid; a length, a buffer or a
    font size does not.
    """
    if isinstance(default, bool) or not isinstance(default, (int, float)):
        return False
    if default == 0:
        return False
    quarters = default / (math.pi / 4)
    return abs(quarters - round(quarters)) < 1e-9


def _angle_params_for(name: str) -> tuple[str, ...]:
    """Derive the angle parameters of the Manim class behind wrapper ``name``.

    Walks the whole MRO rather than the class's own signature: Manim's
    geometry inherits most of its angle parameters through ``**kwargs``, so
    ``Sector``, ``Ellipse`` and every ``*FilledTip`` accept an angle their own
    signature never names. Anything the detector flags and neither
    :data:`_ANGLE_PARAM_NAMES` nor :data:`_NOT_ANGLE_PARAM_NAMES` classifies
    raises, which is what keeps the two sets honest across a Manim upgrade.
    """
    manim_class = _MANIM_WRAPPER_REGISTRY[name]._manim_class
    angles: dict[str, None] = {}
    unclassified = []
    for base in manim_class.__mro__:
        try:
            signature = inspect.signature(base.__init__)
        except (TypeError, ValueError):
            continue
        for param in signature.parameters.values():
            if param.kind in (param.VAR_KEYWORD, param.VAR_POSITIONAL):
                continue
            if param.name == "self":
                continue
            flagged = any(
                hint in param.name for hint in _ANGLE_NAME_HINTS
            ) or _looks_like_radians(param.default)
            if not flagged:
                continue
            if param.name in _ANGLE_PARAM_NAMES:
                angles[param.name] = None
            elif param.name not in _NOT_ANGLE_PARAM_NAMES:
                unclassified.append(f"{base.__name__}.{param.name}")
    if unclassified:
        raise RuntimeError(
            f"{name}: parameters look like angles but are classified by neither "
            f"_ANGLE_PARAM_NAMES nor _NOT_ANGLE_PARAM_NAMES: {sorted(set(unclassified))}. "
            "Add each to whichever set it belongs to in algan/mobs/manim_adapters.py."
        )
    return tuple(sorted(angles))


def _supplied_arguments(signature, args, kwargs) -> dict:
    """Map ``args``/``kwargs`` onto parameter names.

    ``bind_partial`` files anything the signature does not name under the
    ``**kwargs`` parameter rather than at the top level, so a plain
    ``bound.arguments`` lookup misses exactly the inherited angles that
    :func:`_angle_params_for` exists to find -- ``Sector(angle=90)`` would
    arrive as ``{"kwargs": {"angle": 90}}`` and convert nothing.
    """
    if signature is None:
        # No usable signature: only keywords can be identified, which is what
        # every angle-carrying class is called with anyway.
        return dict(kwargs)
    try:
        bound = signature.bind_partial(*args, **kwargs)
    except TypeError:
        # Let Manim raise the real signature error rather than masking it here.
        return dict(kwargs)
    supplied = {}
    for name, value in bound.arguments.items():
        kind = signature.parameters[name].kind
        if kind is inspect.Parameter.VAR_KEYWORD:
            supplied.update(value)
        elif kind is not inspect.Parameter.VAR_POSITIONAL:
            supplied[name] = value
    return supplied


def _to_radians(value):
    return value * DEGREES_TO_RADIANS


def _nested_to_radians(value, keys: tuple[str, ...]):
    """Convert the angle-valued entries of a keyword mapping, or of a sequence of them."""
    if isinstance(value, Mapping):
        return {
            k: (_to_radians(v) if k in keys and v is not None else v)
            for k, v in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_nested_to_radians(item, keys) for item in value]
    return value


def _converted_kwargs(signature, angle_params, args, kwargs):
    """Map ``args``/``kwargs`` onto parameter names and convert the angles.

    Only parameters the caller actually supplied are converted. Manim's own
    defaults are already radians and already right -- ``Arc``'s ``angle`` of
    ``TAU/4`` is a quarter turn in either convention -- so applying defaults
    before converting would read that ``1.57`` as degrees and build a 1.57
    degree arc.
    """
    supplied = _supplied_arguments(signature, args, kwargs)
    converted = dict(kwargs)
    positional = list(args)
    names = list(signature.parameters) if signature is not None else []

    def store(name, value):
        if name in converted:
            converted[name] = value
        else:
            # Supplied positionally: rebuild that slot in place. The signature
            # is the Manim class's, which does not carry ``self``, so the
            # parameter's index is the positional index directly.
            positional[names.index(name)] = value

    for name in angle_params:
        value = supplied.get(name)
        if value is None:
            continue
        store(name, _to_radians(value))

    for name, keys in _NESTED_ANGLE_PARAMS.items():
        value = supplied.get(name)
        if value is None:
            continue
        store(name, _nested_to_radians(value, keys))

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
        # Run unconditionally rather than only for a class with declared angle
        # parameters: a nested ``arc_config`` can reach a class through
        # ``**kwargs`` too, and skipping the pass is how one gets missed.
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


def _adapted_names() -> tuple[str, ...]:
    stray = sorted((set(_NATIVE) | set(_NOT_ADAPTED)) - set(_MANIM_WRAPPER_REGISTRY))
    if stray:
        raise RuntimeError(
            "manim_adapters exclusion lists name classes the compatibility layer "
            f"does not wrap: {stray}"
        )
    overlap = sorted(set(_NATIVE) & set(_NOT_ADAPTED))
    if overlap:
        raise RuntimeError(
            f"classes listed as both native and deliberately un-adapted: {overlap}"
        )
    return tuple(
        sorted(set(_MANIM_WRAPPER_REGISTRY) - set(_NATIVE) - set(_NOT_ADAPTED))
    )


#: The adapted set: every compatibility class that is neither native nor
#: deliberately excluded. Computed, not listed -- see the module docstring.
_ADAPTED: tuple[str, ...] = _adapted_names()

#: Manim class name -> the parameters of it that arrive in degrees. Derived
#: from each class's MRO; see :func:`_angle_params_for`.
_ANGLE_PARAMS: dict[str, tuple[str, ...]] = {
    name: _angle_params_for(name) for name in _ADAPTED
}


def _build():
    for name in _ADAPTED:
        globals()[name] = _make_adapter(name, _ANGLE_PARAMS[name])


_build()

__all__ = list(_ADAPTED)

del _build
