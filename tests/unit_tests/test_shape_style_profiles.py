"""The opt-in Manim shape-style profile (``SETTINGS.style.shape_style_profile``).

Enabling the profile must adopt Manim Community's constructor defaults for
Algan's built-in shapes -- reading them out of the installed ``manim`` at
enable time, never hardcoding them -- and disabling it must restore Algan's
own defaults exactly. An explicit keyword passed to a shape always wins over
the profile.
"""

from __future__ import annotations

import pytest

from algan import (
    LEFT,
    RIGHT,
    SETTINGS,
    UP,
    AlganConfigurationError,
    Circle,
    Cone,
    Cube,
    Cylinder,
    Dot,
    Line,
    Point,
    Polygon,
    Prism,
    Rectangle,
    RegularPolygon,
    Sphere,
    Square,
    Torus,
    Triangle,
)
from algan.constants.color import BLUE
from algan.settings.shape_style_profiles import (
    _MANIM_SHAPE_STYLE_CLASSES,
    _ensure_manim_shape_styles,
    _resolve_shape_style,
)


@pytest.fixture(autouse=True)
def restore_settings():
    snapshot = SETTINGS.snapshot()
    yield
    SETTINGS.restore(snapshot)


def hex_of(color):
    row = color.reshape(-1, color.shape[-1])[0]
    return "#{:02X}{:02X}{:02X}".format(*[int(round(float(c) * 255)) for c in row[:3]])


def fill_opacity_of(mob):
    return round(float(mob.color.reshape(-1, mob.color.shape[-1])[0][4]), 4)


def grid_hexes(surface):
    return {
        hex_of(row)
        for row in surface.grid.color.reshape(-1, surface.grid.color.shape[-1])
    }


def face_style(solid):
    face = solid.faces[0][0]
    return (
        hex_of(face.color),
        round(float(face.opacity.reshape(-1)[0]), 4),
    )


# --------------------------------------------------------------------------
# Profile off (the default): nothing moves.
# --------------------------------------------------------------------------
def test_default_profile_is_algan():
    assert SETTINGS.style.shape_style_profile == "algan"


def test_profile_off_leaves_the_algan_defaults():
    square = Square()
    assert (hex_of(square.color), bool(square.filled)) == ("#FC6255", True)
    assert float(square.border_width) == 5.0
    circle = Circle()
    assert hex_of(circle.color) == "#58C4DD"
    line = Line(LEFT, RIGHT)
    assert float(line.border_width) == 5.0


def test_switching_back_off_restores_the_algan_defaults():
    SETTINGS.style.set(shape_style_profile="manim")
    SETTINGS.style.set(shape_style_profile="algan")
    square = Square()
    assert hex_of(square.color) == "#FC6255"
    assert bool(square.filled)
    assert float(square.border_width) == 5.0
    assert hex_of(Sphere().color) == "#83C167"


# --------------------------------------------------------------------------
# Profile on: shapes adopt Manim's constructor defaults.
# --------------------------------------------------------------------------
# Profile on: shapes adopt Manim's constructor defaults.
# --------------------------------------------------------------------------
def _build(shape_name):
    if shape_name == "Polygon":
        return Polygon(LEFT, RIGHT, UP)
    if shape_name == "Line":
        return Line(LEFT, RIGHT)
    builders = {
        "Square": Square,
        "Rectangle": Rectangle,
        "Circle": Circle,
        "Triangle": Triangle,
        "RegularPolygon": RegularPolygon,
        "Dot": Dot,
    }
    return builders[shape_name]()


@pytest.mark.parametrize(
    ("shape_name", "expected"),
    [
        # fill hex @ opacity, border hex, filled, border_width -- Algan units,
        # i.e. border_width is Manim's stroke_width / 2.
        ("Square", ("#FFFFFF", 0.0, "#FFFFFF", False, 2.0)),
        ("Rectangle", ("#FFFFFF", 0.0, "#FFFFFF", False, 2.0)),
        ("Circle", ("#FC6255", 0.0, "#FC6255", False, 2.0)),
        ("Triangle", ("#58C4DD", 0.0, "#58C4DD", False, 2.0)),
        ("RegularPolygon", ("#58C4DD", 0.0, "#58C4DD", False, 2.0)),
        ("Polygon", ("#58C4DD", 0.0, "#58C4DD", False, 2.0)),
        ("Line", ("#9A72AC", 1.0, "#FFFFFF", False, 2.0)),
        ("Dot", ("#FFFFFF", 1.0, "#FFFFFF", True, 0.0)),
    ],
)
def test_profile_on_adopts_manim_circuit_defaults(shape_name, expected):
    SETTINGS.style.set(shape_style_profile="manim")
    mob = _build(shape_name)
    fill_hex, fill_opacity, border_hex, filled, border_width = expected
    assert hex_of(mob.color) == fill_hex
    assert fill_opacity_of(mob) == fill_opacity
    assert hex_of(mob.border_color) == border_hex
    assert bool(mob.filled) is filled
    assert float(mob.border_width) == border_width


def test_profile_on_gives_the_curved_solids_manim_s_checkerboard():
    SETTINGS.style.set(shape_style_profile="manim")
    for surface in (Sphere(), Cylinder(), Torus()):
        assert grid_hexes(surface) == {"#29ABCA", "#236B8E"}
    # Manim's Cone carries one fill colour and no checkerboard.
    assert grid_hexes(Cone()) == {"#29ABCA"}


def test_profile_on_gives_the_flat_solids_manim_s_face_fill():
    SETTINGS.style.set(shape_style_profile="manim")
    assert face_style(Cube(size=1)) == ("#58C4DD", 0.75)
    assert face_style(Prism(width=1, height=1, depth=1)) == ("#58C4DD", 0.75)


def test_enabling_resolves_the_snapshots_eagerly():
    # Enabling pays the manim import itself; afterwards no Mob construction
    # needs to touch manim again.
    SETTINGS.style.set(shape_style_profile="manim")
    styles = _ensure_manim_shape_styles()
    assert set(styles) == set(_MANIM_SHAPE_STYLE_CLASSES)


# --------------------------------------------------------------------------
# Explicit keywords win over the profile.
# --------------------------------------------------------------------------
def test_explicit_kwargs_win_over_the_profile():
    SETTINGS.style.set(shape_style_profile="manim")
    square = Square(color=BLUE, border_width=7, filled=True)
    assert hex_of(square.color) == "#58C4DD"
    assert float(square.border_width) == 7.0
    assert bool(square.filled)

    circle = Circle(fill_opacity=0.5)
    assert abs(fill_opacity_of(circle) - 0.5) < 1e-6

    sphere = Sphere(color=BLUE)
    assert hex_of(sphere.color) == "#58C4DD"
    assert grid_hexes(sphere) == {"#58C4DD"}


def test_explicit_stroke_kwargs_win_over_the_profile():
    SETTINGS.style.set(shape_style_profile="manim")
    circle = Circle(stroke_color=BLUE, stroke_width=8)
    assert hex_of(circle.border_color) == "#58C4DD"
    assert float(circle.border_width) == 4.0


def test_unmapped_shapes_are_unchanged_under_the_profile():
    # Point is not a mapped shape, so the profile must leave it exactly as
    # Algan built it before.
    point_off = Point()
    SETTINGS.style.set(shape_style_profile="manim")
    point_on = Point()
    assert hex_of(point_on.color) == hex_of(point_off.color)
    assert float(point_on.border_width) == float(point_off.border_width)
    assert bool(point_on.filled) is bool(point_off.filled)


# --------------------------------------------------------------------------
# The setting itself.
# --------------------------------------------------------------------------
def test_an_unknown_profile_value_is_rejected():
    with pytest.raises(AlganConfigurationError):
        SETTINGS.style.set(shape_style_profile="flavourless")


def test_a_missing_manim_class_is_skipped_rather_than_raised():
    assert _resolve_shape_style("NoSuchShape", "DefinitelyNotAManimClass") is None
    # And a shape whose resolution failed simply keeps Algan's defaults.
    SETTINGS.style.set(shape_style_profile="manim")
    from algan.settings.shape_style_profiles import _manim_shape_style_for

    class NotARealManimShape(Square):
        pass

    assert _manim_shape_style_for(NotARealManimShape) is None


def test_snapshot_matches_the_installed_manim():
    """Drift guard: the cached values come from the installed manim itself."""
    manim = pytest.importorskip("manim")
    SETTINGS.style.set(shape_style_profile="manim")
    styles = _ensure_manim_shape_styles()

    manim_square = manim.Square()
    style = styles["Square"]
    assert hex_of(style["color"]) == manim_square.fill_color.to_hex().upper()
    assert hex_of(style["border_color"]) == manim_square.stroke_color.to_hex().upper()
    assert style["border_width"] == float(manim_square.stroke_width) / 2
    assert style["filled"] == (float(manim_square.fill_opacity) > 1e-5)

    manim_sphere = manim.Sphere()
    style = styles["Sphere"]
    assert hex_of(style["color"]) == manim_sphere.fill_color.to_hex().upper()
    assert hex_of(style["checker_color"]) == (
        manim_sphere.checkerboard_colors[1].to_hex().upper()
    )
