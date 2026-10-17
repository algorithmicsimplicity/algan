"""A warning about a call points at that call, not at the Algan code it ran.

Python prints a warning against the file and line its ``stacklevel`` reaches,
shows that line's source, and by default shows a warning once per line. A fixed
``stacklevel`` reaches the script's line only from the call path it was counted
on, if any: from a subclass, an adapter or a wrapper it stopped inside Algan, on
a line the script does not have, and the same warning from two lines of the
script was shown once.
"""

import linecache
import math
import os
import shutil

import pytest
from test_audit_model_regressions import _write_glb

import algan.manim as mn
from algan import (
    LEFT,
    RIGHT,
    Arc,
    Arrow,
    Circle,
    Line,
    MarkupText,
    MeshStandardMaterial,
    Model3D,
    Paragraph,
    Sphere,
    Square,
    Wiggle,
    errors,
)
from algan.errors import (
    ApproximationWarning,
    DespawnedMobWarning,
    UnsupportedFeatureWarning,
)
from algan.mobs import text as text_module


def _line_warned(caught, text):
    """The source line of the one warning in ``caught`` whose message has ``text``."""
    (warning,) = [w for w in caught if text in str(w.message)]
    assert warning.filename == __file__
    return linecache.getline(__file__, warning.lineno).strip()


def test_spawning_a_despawned_mob(fresh_scene):
    square = Square().spawn()
    square.despawn()
    with pytest.warns(DespawnedMobWarning) as caught:
        square.spawn()
    assert _line_warned(caught, "did nothing") == "square.spawn()"


@pytest.mark.parametrize(
    "shape",
    [Square, Circle, Line, pytest.param(mn.Square, id="algan.manim.Square")],
)
def test_a_background_stroke(fresh_scene, shape):
    with pytest.warns(UnsupportedFeatureWarning) as caught:
        shape(background_stroke_width=4)
    assert (
        _line_warned(caught, "Background strokes") == "shape(background_stroke_width=4)"
    )


def test_a_material_property_the_renderer_ignores(fresh_scene):
    square = Square()
    with pytest.warns(UserWarning, match="wireframe") as caught:
        square.set_material(MeshStandardMaterial(wireframe=True))
    assert (
        _line_warned(caught, "wireframe")
        == "square.set_material(MeshStandardMaterial(wireframe=True))"
    )


def test_an_angle_that_looks_like_radians(fresh_scene):
    # Through an adapted Manim class, and through a keyword-checking wrapper.
    with pytest.warns(ApproximationWarning) as caught:
        Arc(angle=math.pi / 2)
    assert _line_warned(caught, "`angle=") == "Arc(angle=math.pi / 2)"

    with pytest.warns(ApproximationWarning) as caught:
        Arrow(LEFT, RIGHT, path_arc=math.pi / 2)
    assert (
        _line_warned(caught, "`path_arc=") == "Arrow(LEFT, RIGHT, path_arc=math.pi / 2)"
    )

    square = Square().spawn()
    with pytest.warns(ApproximationWarning) as caught:
        Wiggle(square, rotation_angle=0.02 * math.pi)
    assert (
        _line_warned(caught, "`rotation_angle=")
        == "Wiggle(square, rotation_angle=0.02 * math.pi)"
    )


def test_a_geometry_tolerance_no_grid_meets(fresh_scene):
    Sphere.clear_geometry_resolution_cache()
    with pytest.warns(RuntimeWarning) as caught:
        Sphere(geometry_tolerance=1e-7, max_grid_resolution=8)
    assert _line_warned(caught, "max_grid_resolution") == (
        "Sphere(geometry_tolerance=1e-7, max_grid_resolution=8)"
    )

    # The second Sphere takes its grid from the cache, which warns again.
    with pytest.warns(RuntimeWarning) as caught:
        Sphere(max_grid_resolution=8, geometry_tolerance=1e-7)
    assert _line_warned(caught, "max_grid_resolution") == (
        "Sphere(max_grid_resolution=8, geometry_tolerance=1e-7)"
    )


def test_a_gltf_animation_that_cannot_be_read(fresh_scene, tmp_path):
    import pygltflib

    path = tmp_path / "broken.glb"
    _write_glb(path)
    model = pygltflib.GLTF2().load(str(path))
    model.animations[0].samplers[0].output = 99  # no such accessor
    model.save_binary(str(path))

    with pytest.warns(UnsupportedFeatureWarning) as caught:
        Model3D(str(path))
    assert _line_warned(caught, "Could not load glTF animation") == (
        "Model3D(str(path))"
    )


@pytest.mark.skipif(
    shutil.which("latex") is None or shutil.which("dvisvgm") is None,
    reason="needs a TeX distribution with dvisvgm",
)
def test_text_typeset_through_latex_without_pango(fresh_scene, monkeypatch):
    # Said once per process, by the first Text: here, Text built by a subclass
    # and by a Paragraph of lines.
    monkeypatch.setattr(text_module, "_pango_available", lambda: False)
    monkeypatch.setattr(text_module, "_LATEX_TEXT_FALLBACK_WARNED", False)
    with pytest.warns(UnsupportedFeatureWarning) as caught:
        MarkupText("<b>bold</b>", add_to_scene=False)
    assert _line_warned(caught, "through LaTeX") == (
        'MarkupText("<b>bold</b>", add_to_scene=False)'
    )

    monkeypatch.setattr(text_module, "_LATEX_TEXT_FALLBACK_WARNED", False)
    with pytest.warns(UnsupportedFeatureWarning) as caught:
        Paragraph("one", "two", add_to_scene=False)
    assert _line_warned(caught, "through LaTeX") == (
        'Paragraph("one", "two", add_to_scene=False)'
    )


def test_a_script_named_like_the_package_beside_it_is_the_users():
    """``algan_scenes/`` beside an editable checkout's ``algan/`` shares its
    name as a prefix, and its frames were skipped as Algan's own, so a warning
    pointed past the script's line.
    """
    beside = errors._PACKAGE_DIR + "_scenes" + os.sep + "scene.py"
    script = {}
    exec(
        compile("def line(): return locate(), level()\n", beside, "exec"),
        {"locate": errors._user_location, "level": errors._user_stacklevel},
        script,
    )
    location, level = script["line"]()
    assert location[:2] == (beside, 1)
    assert level == 1
    inside = os.path.join(errors._PACKAGE_DIR, "scene.py")
    assert errors._in_algan(inside)
    assert not errors._in_algan(beside)
