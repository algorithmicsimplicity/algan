"""The root spellings of Manim's Mobjects convert Manim's conventions to Algan's.

``algan/mobs/manim_adapters.py`` builds a native spelling for every class in the
compatibility registry except the ones Algan implements itself and the ones
deliberately left ``mn.``-only. These tests hold the two halves of that to
account: that the exclusion lists match what the root namespace actually
resolves each name to, and that a parameter the module says is in degrees really
does reach Manim in radians.

The parity checks are written as "Algan's spelling with 90 builds what Manim's
spelling with pi/2 builds", which is the whole contract in one line and fails
loudly whichever way the conversion breaks -- a missing conversion and a doubled
one both move the geometry.
"""

from __future__ import annotations

import inspect
import math
import warnings

import numpy as np
import pytest

import algan
import algan.manim as mn
from algan.errors import AlganConfigurationError, ApproximationWarning
from algan.mobs.manim_adapters import (
    _ADAPTED,
    _ANGLE_PARAMS,
    _NATIVE,
    _NOT_ADAPTED,
    _ROOT_CLASS_NAMES,
    _ROOT_NAME_FOR,
    _angle_params_for,
)
from algan.mobs.manim_compat import _MANIM_WRAPPER_REGISTRY

_ADAPTER_MODULE = "algan.mobs.manim_adapters"


def _points(mob):
    """The Manim geometry an Algan compatibility Mob was built from."""
    return mob.manim_mobject.points


def _same_geometry(algan_side, manim_side):
    left, right = _points(algan_side), _points(manim_side)
    assert left.shape == right.shape, (
        f"different point counts: {left.shape} vs {right.shape}"
    )
    return np.allclose(left, right, atol=1e-6)


def test_every_registry_class_is_adapted_native_or_excluded():
    """The three lists partition the compatibility registry, with nothing left over."""
    accounted = set(_ADAPTED) | set(_NATIVE) | set(_NOT_ADAPTED)
    assert accounted == set(_MANIM_WRAPPER_REGISTRY), (
        "every wrapped Manim class must be adapted, native, or explicitly "
        "excluded: "
        f"unaccounted {sorted(set(_MANIM_WRAPPER_REGISTRY) - accounted)}, "
        f"unknown {sorted(accounted - set(_MANIM_WRAPPER_REGISTRY))}"
    )


def test_native_list_matches_what_the_root_namespace_resolves():
    """``_NATIVE`` is the set of names the root namespace answers with a native class.

    The list is what keeps an adapter from shadowing a native implementation
    (or being shadowed by one, which is what import order would do silently).
    Checking it against the live namespace is what makes it a specification
    rather than a comment.
    """
    misfiled = {}
    for name in _NATIVE:
        exported = getattr(algan, name, None)
        if exported is None:
            misfiled[name] = "not exported at the root at all"
        elif exported.__module__ == _ADAPTER_MODULE:
            misfiled[name] = "resolves to an adapter, so it is not native"
    assert not misfiled, (
        f"_NATIVE disagrees with the root namespace: {misfiled}. A class that "
        "gained a native implementation belongs in _NATIVE; one that lost its "
        "native implementation belongs out of it."
    )

    for name in _ADAPTED:
        root_name = _ROOT_NAME_FOR[name]
        exported = getattr(algan, root_name, None)
        assert exported is not None, (
            f"{name} is adapted but {root_name} is not exported at the root"
        )
        assert exported.__module__ == _ADAPTER_MODULE, (
            f"{name} is in the adapted set but the root name {root_name} "
            f"resolves to {exported.__module__}, which shadows the adapter. "
            "Move it to _NATIVE."
        )


def test_deliberately_unadapted_classes_stay_manim_only():
    """A base class or value tracker has no root spelling, and says why in the table."""
    for name, reason in _NOT_ADAPTED.items():
        assert reason, f"{name} is excluded without a reason"
        exported = getattr(algan, name, None)
        assert exported is None or exported.__module__ != _ADAPTER_MODULE, (
            f"{name} is listed as un-adapted but an adapter for it is exported"
        )
        assert hasattr(mn, name), (
            f"{name} must still be reachable as algan.manim.{name}"
        )


def test_every_adapted_class_has_a_derived_angle_table():
    """``_ANGLE_PARAMS`` covers the adapted set; deriving it again is stable."""
    assert set(_ANGLE_PARAMS) == set(_ADAPTED)
    for name in _ADAPTED:
        assert _angle_params_for(name) == _ANGLE_PARAMS[name]


@pytest.mark.parametrize(
    ("name", "algan_kwargs", "manim_kwargs"),
    [
        # Declared on the class's own signature.
        ("Arc", {"angle": 90}, {"angle": math.pi / 2}),
        ("Arc", {"start_angle": 45}, {"start_angle": math.pi / 4}),
        ("AnnularSector", {"angle": 90}, {"angle": math.pi / 2}),
        (
            "ArcBetweenPoints",
            {"start": (-1.0, 0.0, 0.0), "end": (1.0, 0.0, 0.0), "angle": 60},
            {"start": (-1.0, 0.0, 0.0), "end": (1.0, 0.0, 0.0), "angle": math.pi / 3},
        ),
        ("Elbow", {"angle": 90}, {"angle": math.pi / 2}),
        ("NumberLine", {"rotation": 90}, {"rotation": math.pi / 2}),
        ("PolarPlane", {"azimuth_offset": 90}, {"azimuth_offset": math.pi / 2}),
        # Inherited through **kwargs, which the class's own signature does not
        # name -- the case an MRO walk finds and a signature scan does not.
        ("Sector", {"angle": 90}, {"angle": math.pi / 2}),
        ("Star", {"start_angle": 90}, {"start_angle": math.pi / 2}),
        ("StealthTip", {"start_angle": 90}, {"start_angle": math.pi / 2}),
        ("ArrowTriangleFilledTip", {"start_angle": 90}, {"start_angle": math.pi / 2}),
        # ``path_arc`` comes from ``Line``. Checked on ``DashedLine`` rather than
        # ``Arrow`` because vendored Manim's ``Arrow`` is not reproducible: two
        # ``manim.Arrow`` instances built with identical arguments in one
        # process disagree from the second onwards, with no Algan code involved.
        ("DashedLine", {"path_arc": 90}, {"path_arc": math.pi / 2}),
    ],
)
def test_declared_angles_arrive_in_radians(name, algan_kwargs, manim_kwargs):
    """Algan's spelling in degrees builds what Manim's spelling in radians builds."""
    algan_side = getattr(algan, name)(**algan_kwargs)
    manim_side = getattr(mn, name)(**manim_kwargs)
    assert _same_geometry(algan_side, manim_side), (
        f"{name}({algan_kwargs}) should equal mn.{name}({manim_kwargs})"
    )


def test_positional_angles_are_converted_in_place():
    """A positional angle converts too, at the slot the Manim signature gives it.

    ``Arc``'s signature is ``(radius, start_angle, angle, ...)`` and carries no
    ``self``, so the parameter's index in it *is* the positional index. Getting
    that offset wrong converts the neighbouring argument instead, which is why
    this asserts the radius survived rather than only that the angle moved.
    """
    positional = algan.Arc(2.0, 0, 90)
    keyword = algan.Arc(radius=2.0, start_angle=0, angle=90)
    manim_side = mn.Arc(2.0, 0, math.pi / 2)
    assert _same_geometry(positional, keyword)
    assert _same_geometry(positional, manim_side)


def test_nested_arc_config_angles_are_converted():
    """``ArcPolygon``'s per-arc keyword dicts carry angles a level down."""
    corners = [
        np.array([-1.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
    ]
    algan_side = algan.ArcPolygon(
        *corners, arc_config=[{"angle": 90}, {"angle": 30}, {"angle": 30}]
    )
    manim_side = mn.ArcPolygon(
        *corners,
        arc_config=[
            {"angle": math.pi / 2},
            {"angle": math.pi / 6},
            {"angle": math.pi / 6},
        ],
    )
    assert _same_geometry(algan_side, manim_side)


def test_other_angle_is_a_flag_not_an_angle():
    """``Angle(other_angle=...)`` is a bool selecting the explementary angle.

    It matches the detector on its name alone; converting it multiplied ``True``
    by ``pi/180``, which stayed truthy and so went unnoticed. The waiver is what
    keeps it a flag.
    """
    line1 = mn.Line(np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    line2 = mn.Line(np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    algan_side = algan.Angle(line1.manim_mobject, line2.manim_mobject, other_angle=True)
    manim_side = mn.Angle(line1.manim_mobject, line2.manim_mobject, other_angle=True)
    assert _same_geometry(algan_side, manim_side)


def test_stroke_width_is_algan_units_at_the_root():
    """Half of Manim's, on a class whose own signature never names it."""
    assert float(algan.Star(stroke_width=3).stroke_width.reshape(-1)[0]) == 3.0
    assert float(mn.Star(stroke_width=6).stroke_width.reshape(-1)[0]) == 3.0


# --------------------------------------------------------------------------
# What the adapter *shows*: signature, docstring and the radians warning.
# Delegating used to hand the root spellings Manim's own, which then said the
# wrong thing in the one place a user looks -- ``help()`` and the reference.
# --------------------------------------------------------------------------


def test_displayed_angle_defaults_are_in_degrees():
    """``help(Arc)`` must not answer a degrees argument with 1.5707963267948966."""
    parameters = inspect.signature(algan.Arc).parameters
    assert parameters["angle"].default == 90
    assert parameters["start_angle"].default == 0
    # Manim's own spelling keeps Manim's default, which is the whole point of
    # having two namespaces.
    assert mn.Arc.__signature__.parameters["angle"].default == pytest.approx(
        math.pi / 2
    )


def test_no_manim_type_leaks_into_a_displayed_signature():
    """A displayed annotation names a builtin or nothing -- never a Manim alias."""
    leaked = {}
    for name in _ADAPTED:
        adapter = getattr(algan, _ROOT_NAME_FOR[name])
        text = str(inspect.signature(adapter))
        hits = [
            token
            for token in ("Mobject", "VMobject", "Point3D", "Point2D", "Vector3D")
            if token in text
        ]
        if hits:
            leaked[name] = hits
    assert not leaked, f"Manim types reach the displayed signature: {leaked}"


def test_no_adapter_docstring_teaches_a_manim_scene():
    """Manim's prose is replaced, not appended to.

    Manim's docstrings carry ``.. manim::`` blocks -- ``class X(Scene)`` bodies
    calling ``self.play`` -- which Algan's docs build would execute and render
    into Algan's own reference pages, teaching a script that does not run here.
    """
    offenders = {}
    for name in _ADAPTED:
        doc = getattr(algan, _ROOT_NAME_FOR[name]).__doc__ or ""
        hits = [
            token for token in ("self.play", ".. manim::", "(Scene)") if token in doc
        ]
        if hits:
            offenders[name] = hits
    assert not offenders, f"Manim scene code reaches Algan's docstrings: {offenders}"


def test_every_adapter_documents_its_conversions():
    for name in _ADAPTED:
        doc = getattr(algan, _ROOT_NAME_FOR[name]).__doc__ or ""
        assert f"algan.manim.{name}" in doc, f"{name} does not point at its Manim twin"
        assert "stroke_width" in doc


@pytest.mark.parametrize("value", [math.pi / 2, 0.5, -1.25, 3.14159])
def test_a_radian_looking_angle_warns(value):
    with pytest.warns(ApproximationWarning, match="looks like radians"):
        algan.Arc(angle=value)


@pytest.mark.parametrize("value", [90, 5, 45.0, -180, 400.5])
def test_a_degree_angle_does_not_warn(value):
    """Whole numbers never warn, and neither does anything past a full turn.

    ``Arc(angle=5)`` is a legitimate five degree arc, and warning on it would
    train the reader to ignore the message that catches ``PI / 2``.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", ApproximationWarning)
        algan.Arc(angle=value)


def test_the_radians_warning_names_the_degree_spelling():
    with pytest.warns(ApproximationWarning) as caught:
        algan.Arc(angle=math.pi / 2)
    assert (
        str(caught[0].message)
        == "`angle=1.5708` looks like radians; Algan takes degrees "
        "(did you mean `angle=90`?)"
    )


def test_manim_keyword_spellings_are_refused_at_the_root():
    """``mobject=`` and ``element_to_mobject=`` name arguments Algan does not have."""
    square = algan.Square()
    with pytest.raises(AlganConfigurationError, match="Algan spells it `mob`"):
        algan.Brace(mobject=square)
    with pytest.raises(AlganConfigurationError, match="element_to_mob"):
        algan.Table([["a"]], element_to_mobject=None)


def test_manim_keyword_spellings_still_work_under_algan_manim():
    """The compatibility namespace is Manim's conventions, spellings included."""
    square = mn.Square()
    assert mn.Brace(mobject=square) is not None
    assert algan.Brace(mob=square) is not None


@pytest.mark.parametrize(("manim_name", "root_name"), sorted(_ROOT_CLASS_NAMES.items()))
def test_the_mobject_spelled_classes_are_renamed_at_the_root(manim_name, root_name):
    """Algan's word is ``Mob``; ``algan.manim`` keeps Manim's."""
    assert getattr(algan, manim_name, None) is None
    assert getattr(algan, root_name).__module__ == _ADAPTER_MODULE
    assert hasattr(mn, manim_name)
    assert root_name in algan.__all__
