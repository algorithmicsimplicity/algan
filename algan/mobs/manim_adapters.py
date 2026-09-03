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

There is no third conversion: Manim's ``OUT`` and Algan's ``OUTWARD`` are both
``+z``, so an imported point keeps the coordinates it was written with.

What the adapter *shows*
------------------------
Delegating gave the root spellings Manim's ``__signature__`` and Manim's
docstring, which then said the wrong thing in the one place a user looks:
``help(Arc)`` showed ``angle: float = 1.5707963267948966`` for an argument this
module reads as degrees, and ``Brace(mobject: 'Mobject')`` named a type and a
keyword Algan does not have. So each adapter carries its own:

- :func:`_root_signature` renames the superseded keywords
  (:data:`~algan.utils.api_renames._ROOT_KEYWORD_RENAMES`), restates every angle
  default in degrees, and drops annotations naming Manim types.
- :func:`_root_docstring` replaces Manim's prose rather than appending to it,
  so no ``.. manim::`` block, ``class X(Scene)`` or ``self.play(...)`` reaches
  Algan's reference pages. The bodies are **generated** from Manim's summary
  line plus the converted signature; the two classes with hand-written Algan
  docstrings (``MathTex``, ``Title``, in ``_WRAPPER_DOCSTRINGS``) keep them.
- The five classes whose Manim names spell "Mobject" take Algan's spelling at
  the root (:data:`_ROOT_CLASS_NAMES`); the Manim names stay in
  :mod:`algan.manim`.
- A supplied angle that looks like radians warns
  (:func:`~algan.utils.api_renames._warn_if_angle_looks_like_radians`), because
  ``Arc(angle=PI / 2)`` is a legal 1.57 degree sliver and nothing else would
  say so.
"""

from __future__ import annotations

import inspect
import math
import re
from collections.abc import Mapping, Sequence

from algan.constants.math import DEGREES_TO_RADIANS, RADIANS_TO_DEGREES
from algan.mobs.manim_compat import _MANIM_WRAPPER_REGISTRY, _WRAPPER_DOCSTRINGS
from algan.settings import SETTINGS
from algan.utils.api_renames import (
    _ROOT_KEYWORD_RENAMES,
    _reject_renamed_keywords,
    _warn_if_angle_looks_like_radians,
)

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

    Each supplied angle is also checked for the opposite mistake: a value
    written in radians is a legal, silent, wrong picture, so it warns.
    """
    supplied = _supplied_arguments(signature, args, kwargs)
    for name in angle_params:
        if name in supplied:
            _warn_if_angle_looks_like_radians(name, supplied[name])
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
    """Scale an Algan-unit ``stroke_width`` into Manim's, in place.

    The factor is ``SETTINGS.style.manim_stroke_width_ratio`` -- 2 under Algan's
    own convention, the exact 2.0202 once ``use_manim_defaults`` has run -- and
    it is the same one the import and export conversions use, so a round trip
    returns the width it started with whichever is in force.

    Applied to every adapter rather than to a declared list: a Manim class
    accepts ``stroke_width`` whether or not its signature names it, so a
    per-class table would leave ``Star`` and ``DashedLine`` -- which take it
    through ``**kwargs`` to ``VMobject`` -- on Manim's unit while the classes
    that declare it moved to Algan's.
    """
    width = kwargs.get("stroke_width")
    if width is not None:
        kwargs["stroke_width"] = width * SETTINGS.style.manim_stroke_width_ratio
    return kwargs


#: Manim class name -> the root spelling of it. Manim names its base class
#: ``Mobject`` and spells five concrete classes after it; Algan's is ``Mob``,
#: and a root namespace that says ``Mobject`` in five places and ``Mob``
#: everywhere else teaches a word Algan does not use. The Manim names stay
#: reachable as ``algan.manim.<name>``, where they are correct.
_ROOT_CLASS_NAMES: dict[str, str] = {
    "CurvesAsSubmobjects": "CurvesAsChildren",
    "DashedVMobject": "DashedMob",
    "MobjectMatrix": "MobMatrix",
    "MobjectTable": "MobTable",
    "SVGMobject": "SVGMob",
}

#: Annotation strings worth showing. Manim annotates with its own aliases
#: (``Point3DLike``, ``Vector3D``, ``ManimColor``, ``Mobject``) which name types
#: Algan does not have, and Sphinx renders the annotation verbatim -- so
#: anything but a plain builtin is dropped rather than translated into a
#: half-truth. ``float`` and ``int`` are the ones that carry real information.
_KEPT_ANNOTATION_TOKENS: frozenset[str] = frozenset(
    {"bool", "complex", "float", "int", "None", "str"}
)

#: Algan-only constructor arguments every compatibility Mob accepts, named in
#: the generated ``**kwargs`` entry so the passthrough is not a dead end.
_ALGAN_ONLY_KWARGS_DOC = "``scene``, ``add_to_scene``, ``glow`` and ``glow_radius``"

_ROLE_MARKUP = re.compile(r":[a-zA-Z:]+:`([^`]*)`")


def _plain_text(text: str) -> str:
    """Strip Sphinx roles out of borrowed prose, keeping the words.

    Manim's summary lines cross-reference Manim's own reference pages
    (``:class:`~.VMobject```), which do not exist in Algan's. Rendering the
    name as a literal keeps the sentence readable and cannot dangle.
    """

    def replace(match):
        target = match.group(1)
        if "<" in target:
            target = target.split("<", 1)[0]
        return f"``{target.strip().lstrip('~.')}``"

    return _ROLE_MARKUP.sub(replace, text)


def _manim_summary(manim_class) -> str | None:
    """The first paragraph of ``manim_class``'s docstring, as one plain line."""
    doc = manim_class.__doc__
    if not doc:
        return None
    collected: list[str] = []
    for line in doc.expandtabs().splitlines():
        stripped = line.strip()
        if not stripped:
            if collected:
                break
            continue
        collected.append(stripped)
    if not collected:
        return None
    return _plain_text(" ".join(collected))


class _Literal:
    """A default shown as source text rather than as its own ``repr``.

    ``__signature__`` here is display-only -- an adapter's ``__init__`` takes
    ``*args, **kwargs`` and binds against the *wrapper's* signature -- so a
    default can be swapped for something that reads like the code a user would
    write. Manim's own reprs do not: a direction arrives as
    ``array([ 0., -1., 0.])`` and a colour as ``ManimColor('#000000')``, both of
    which name Manim's world rather than Algan's.
    """

    __slots__ = ("_text",)

    def __init__(self, text: str):
        self._text = text

    def __repr__(self) -> str:
        return self._text


#: The 3-vectors Manim writes as its own module constants, spelled the way an
#: Algan script would spell them. Keyed by tuple so a numpy default matches.
_DIRECTION_LITERALS: dict[tuple[float, float, float], str] = {
    (0.0, 0.0, 0.0): "ORIGIN",
    (0.0, 1.0, 0.0): "UP",
    (0.0, -1.0, 0.0): "DOWN",
    (-1.0, 0.0, 0.0): "LEFT",
    (1.0, 0.0, 0.0): "RIGHT",
    (0.0, 0.0, 1.0): "OUT",
    (0.0, 0.0, -1.0): "IN",
    (1.0, 1.0, 0.0): "UP + RIGHT",
    (-1.0, 1.0, 0.0): "UP + LEFT",
    (1.0, -1.0, 0.0): "DOWN + RIGHT",
    (-1.0, -1.0, 0.0): "DOWN + LEFT",
}


def _display_default(default):
    """``default`` as an Algan script would write it, for the shown signature."""
    if inspect.isclass(default):
        return _Literal(default.__name__)
    to_hex = getattr(default, "to_hex", None)
    if callable(to_hex) and type(default).__name__ == "ManimColor":
        return _Literal(repr(to_hex()))
    shape = getattr(default, "shape", None)
    if shape == (3,):
        key = tuple(round(float(component), 9) for component in default)
        named = _DIRECTION_LITERALS.get(key)
        return _Literal(named if named is not None else repr(key))
    if isinstance(default, (list, tuple)) and default:
        rendered = [_display_default(item) for item in default]
        if any(isinstance(item, _Literal) for item in rendered):
            body = ", ".join(repr(item) for item in rendered)
            return _Literal(f"[{body}]" if isinstance(default, list) else f"({body})")
        return default
    # Anything whose repr carries its address: a bare repr would make the
    # rendered reference differ between two builds of identical source, and a
    # lambda's qualified name is Manim's class name rather than Algan's.
    if " at 0x" in repr(default):
        if inspect.isfunction(default) and default.__name__ != "<lambda>":
            return _Literal(default.__qualname__)
        if inspect.isroutine(default):
            return _Literal("<default>")
        return _Literal(f"<{type(default).__name__}>")
    return default


def _degrees_default(default):
    """Manim's radian default, restated in the degrees the adapter reads."""
    if isinstance(default, bool) or not isinstance(default, (int, float)):
        return default
    degrees = round(float(default) * RADIANS_TO_DEGREES, 6)
    return int(degrees) if float(degrees).is_integer() else degrees


def _root_annotation(annotation):
    if annotation is inspect.Parameter.empty:
        return annotation
    text = (
        annotation
        if isinstance(annotation, str)
        else getattr(annotation, "__name__", None)
    )
    if not isinstance(text, str):
        return inspect.Parameter.empty
    tokens = {token for token in text.replace("|", " ").split() if token}
    if tokens and tokens <= _KEPT_ANNOTATION_TOKENS:
        return text
    return inspect.Parameter.empty


def _root_signature(signature, angle_params, renames):
    """Manim's signature restated in Algan's names, units and types."""
    if signature is None:
        return None
    parameters = []
    for parameter in signature.parameters.values():
        default = parameter.default
        if parameter.name in angle_params:
            default = _degrees_default(default)
        elif parameter.name == "stroke_width" and isinstance(default, (int, float)):
            # ``_to_manim_stroke_width`` doubles what the caller passes, so
            # Manim's own default is twice the number an Algan author would
            # write for the same line.
            default = default / 2
        elif default is not inspect.Parameter.empty:
            default = _display_default(default)
        parameters.append(
            parameter.replace(
                name=renames.get(parameter.name, parameter.name),
                default=default,
                annotation=_root_annotation(parameter.annotation),
            )
        )
    return signature.replace(
        parameters=parameters, return_annotation=inspect.Signature.empty
    )


def _parameter_entries(signature, angle_params, name) -> list[str]:
    """A generated ``Parameters`` block: every argument, its unit and default."""
    if signature is None:
        return []
    entries = []
    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            entries.append(
                f"**{parameter.name}\n"
                f"    Manim's remaining ``{name}`` arguments, converted as above, "
                f"plus the Algan-only {_ALGAN_ONLY_KWARGS_DOC}."
            )
            continue
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            entries.append(
                f"*{parameter.name}\n    Passed positionally to Manim's ``{name}``."
            )
            continue
        sentences = []
        if parameter.name in angle_params:
            sentences.append("In **degrees**")
        if parameter.name == "stroke_width":
            sentences.append("In Algan's stroke unit, half Manim's")
        if parameter.default is inspect.Parameter.empty:
            sentences.append("Required")
        else:
            sentences.append(f"Defaults to ``{parameter.default!r}``")
        entries.append(f"{parameter.name}\n    " + ". ".join(sentences) + ".")
    return entries


def _root_docstring(name, root_name, wrapper, root_sig, angle_params) -> str:
    """Algan-facing prose for an adapter, replacing Manim's inherited docstring.

    Hand-written where one exists (``_WRAPPER_DOCSTRINGS``); otherwise
    generated, and the ``Notes`` section says so, because a generated entry can
    state an argument's unit and default but not its meaning.
    """
    converted = [
        "``stroke_width`` is in Algan's unit, half Manim's for the same visual weight"
    ]
    if angle_params:
        listed = ", ".join(f"``{param}``" for param in angle_params)
        converted.insert(0, f"{listed} in **degrees** rather than radians")
    boundary = (
        f"Manim's ``{name}``, under Algan's conventions: "
        f"{'; '.join(converted)}. ``algan.manim.{name}`` is the same class "
        f"under Manim's own conventions."
    )
    animation = (
        "Animation\n"
        "---------\n"
        "Constructing one records nothing: the Mob joins the active Scene "
        "unspawned, and\n"
        ":meth:`~algan.animatable_base.animatable.Animatable.spawn` is what "
        "makes it appear.\n"
        "Everything after that -- a move, a colour change, a delegated Manim "
        "edit -- is\n"
        "recorded on the Scene's timeline over the current context's runtime."
    )

    if name in _WRAPPER_DOCSTRINGS:
        return f"{wrapper.__doc__}\n\nNotes\n-----\n{boundary}\n"

    summary = _manim_summary(wrapper._manim_class) or f"Manim's ``{name}``."
    parameters = _parameter_entries(root_sig, angle_params, name)
    sections = [summary, boundary, animation]
    if parameters:
        sections.append("Parameters\n----------\n" + "\n".join(parameters))
    sections.append(
        "Notes\n"
        "-----\n"
        f"This description is generated from Manim's own summary line and the\n"
        f"converted signature of ``{root_name}``: each entry states its "
        f"argument's unit\n"
        "and default rather than its meaning. Manim's documentation for "
        f"``{name}``\ndescribes what each one does."
    )
    return "\n\n".join(sections) + "\n"


def _make_adapter(name: str, angle_params: tuple[str, ...]):
    wrapper = _MANIM_WRAPPER_REGISTRY[name]
    signature = getattr(wrapper, "__signature__", None)
    root_name = _ROOT_CLASS_NAMES.get(name, name)
    # Only the spellings this class actually declares are translated on the way
    # in; the rejection below is unconditional, because a Manim keyword the
    # signature does not name still reaches the backing class through
    # ``**kwargs`` and would otherwise be accepted in silence.
    inbound = {
        new: old
        for old, new in _ROOT_KEYWORD_RENAMES.items()
        if signature is not None and old in signature.parameters
    }
    root_sig = _root_signature(signature, angle_params, _ROOT_KEYWORD_RENAMES)

    def __init__(self, *args, **kwargs):
        _reject_renamed_keywords(root_name, kwargs, manim_alternative=name)
        for new, old in inbound.items():
            if new in kwargs:
                kwargs[old] = kwargs.pop(new)
        # Run unconditionally rather than only for a class with declared angle
        # parameters: a nested ``arc_config`` can reach a class through
        # ``**kwargs`` too, and skipping the pass is how one gets missed.
        args, kwargs = _converted_kwargs(signature, angle_params, args, kwargs)
        super(adapter, self).__init__(*args, **_to_manim_stroke_width(kwargs))

    adapter = type(
        root_name,
        (wrapper,),
        {
            "__init__": __init__,
            "__module__": __name__,
            "__doc__": _root_docstring(
                name, root_name, wrapper, root_sig, angle_params
            ),
        },
    )
    if root_sig is not None:
        adapter.__signature__ = root_sig
    return adapter


#: Compatibility classes that exist only when Pango does.
#:
#: The vendored Manim subset exports ``Text``, ``MarkupText`` and ``Paragraph``
#: only if ``manimpango`` imports, and that is an optional extra
#: (``pip install "algan[pango]"``) because it publishes no Linux wheel. All
#: three are :data:`_NATIVE` -- Algan has its own, and ``algan.Text`` falls back
#: to LaTeX's text mode -- so their absence costs the ``mn.`` spelling and
#: nothing else. See ``algan/external_libraries/manim/VENDORING.md``.
_PANGO_ONLY: frozenset[str] = frozenset({"MarkupText", "Paragraph", "Text"})


def _adapted_names() -> tuple[str, ...]:
    stray = sorted(
        (set(_NATIVE) | set(_NOT_ADAPTED)) - set(_MANIM_WRAPPER_REGISTRY) - _PANGO_ONLY
    )
    if stray:
        raise RuntimeError(
            "manim_adapters exclusion lists name classes the compatibility layer "
            f"does not wrap: {stray}. Algan's vendored Manim subset is in "
            "algan/external_libraries/manim; if these names moved or were "
            "removed upstream, the exclusion lists here need the same edit."
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


#: Manim class name -> the name the root namespace answers with. Identity for
#: all but :data:`_ROOT_CLASS_NAMES`; tests walk it rather than ``_ADAPTED``,
#: since that is what ``from algan import *`` actually publishes.
_ROOT_NAME_FOR: dict[str, str] = {
    name: _ROOT_CLASS_NAMES.get(name, name) for name in _ADAPTED
}


def _build():
    for name in _ADAPTED:
        globals()[_ROOT_NAME_FOR[name]] = _make_adapter(name, _ANGLE_PARAMS[name])


_build()

__all__ = sorted(_ROOT_NAME_FOR.values())

del _build
