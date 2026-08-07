"""Audits that the full-render suite still covers the public rendering API.

The render suite is deliberately small: five dense scenes instead of one scene
per concept.  That only stays trustworthy if adding a new public renderable
class forces someone to either put it in a scene or say out loud why it does not
need one.  This module derives the required set from ``algan.__all__`` at import
time and fails on anything neither covered nor explicitly exempted, so the
coverage claim cannot silently rot.

It also enforces the scene-file conventions the harness depends on.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import re
from pathlib import Path

import pytest

import algan
from algan.animatable_base.mob import Mob
from algan.rendering.lights import Light
from algan.rendering.shaders.materials import Material

SCENES_DIR = Path(__file__).resolve().parents[1] / "full_renders" / "scenes"

# Modules holding Algan's own renderable classes. The Manim-compatibility and
# OpenGL-alias modules are excluded: that surface is enormous, is audited
# separately by the manim parity tests, and is exercised as a family by
# ``manim_compat_and_plots``.
NATIVE_MOB_MODULES = (
    "algan.mobs.shapes_2d",
    "algan.mobs.shapes_3d",
    "algan.mobs.text",
    "algan.mobs.image_mob",
    "algan.mobs.surfaces.surface",
    "algan.mobs.numeric_display",
    "algan.mobs.bezier_circuit",
    "algan.mobs.group",
    "algan.mobs.point_cloud",
    "algan.mobs.triangulated_bezier_circuit",
    "algan.mobs.three_d_models.model_mob",
)

# Public names that are deliberately not in any scene, each with the reason.
# Adding to this dict is a decision; leaving a name out of it is a test failure.
EXEMPT = {
    # Abstract bases and internal geometry holders: never instantiated by users.
    "Mob": "abstract base",
    "Material": "abstract base",
    "Light": "abstract base",
    "Point": "a degenerate circuit used as a location holder",
    "Polyhedron": "abstract base for the Platonic solids, which are covered",
    "TriangleVertices": "internal vertex holder for TriangleTriangulated",
    "TriangulatedBezierCircuit": "base class; covered through Text/TexTriangulated",
    "BezierCurveCubic": "single-segment view of BezierCircuitCubic, which is covered",
    # The point-cloud family builds points but defines no
    # ``get_render_primitives``, so nothing in it can reach the renderer today.
    # tests/unit_tests/test_point_cloud_rendering.py pins that gap; when it is
    # closed, that test starts XPASSing and these exemptions must go.
    "PMobject": "no render primitives yet -- see test_point_cloud_rendering.py",
    "Mobject1D": "no render primitives yet -- see test_point_cloud_rendering.py",
    "Mobject2D": "no render primitives yet -- see test_point_cloud_rendering.py",
    "PGroup": "no render primitives yet -- see test_point_cloud_rendering.py",
    "DotCloud": "no render primitives yet -- see test_point_cloud_rendering.py",
    "PointCloudDot": "no render primitives yet -- see test_point_cloud_rendering.py",
    "TrueDot": "no render primitives yet -- see test_point_cloud_rendering.py",
    # Manim's OpenGL renderer aliases. Algan has one renderer; these exist so
    # that Manim code that names them keeps importing.
    "OpenGLPMobject": "OpenGL-renderer alias",
    "OpenGLPGroup": "OpenGL-renderer alias",
    "OpenGLPMPoint": "OpenGL-renderer alias",
    # Pre-Three.js material API, kept working for existing scripts and pinned
    # by tests/unit_tests/test_materials.py.
    "PBRMaterial": "legacy material API, unit-tested",
    "AdvancedPBRMaterial": "legacy material API, unit-tested",
    "DiffuseMaterial": "legacy material API, unit-tested",
    "SpecularMaterial": "legacy material API, unit-tested",
    "UnlitMaterial": "legacy material API, unit-tested",
}


def _scene_paths():
    return sorted(SCENES_DIR.glob("*.py"))


def _scene_source():
    return "\n".join(path.read_text(encoding="utf-8") for path in _scene_paths())


def _public_subclasses(base, modules=None):
    exported = set(algan.__all__)
    found = set()
    if modules is None:
        for name in exported:
            value = getattr(algan, name)
            if inspect.isclass(value) and issubclass(value, base):
                found.add(name)
        return found
    for module_name in modules:
        module = importlib.import_module(module_name)
        for name, value in vars(module).items():
            if (
                name in exported
                and inspect.isclass(value)
                and issubclass(value, base)
                and value.__module__ == module_name
            ):
                found.add(name)
    return found


def _missing(names, source):
    return sorted(
        name
        for name in names
        if name not in EXEMPT and not re.search(rf"\b{re.escape(name)}\b", source)
    )


SOURCE = _scene_source()


def test_scene_directory_is_not_empty():
    assert _scene_paths(), f"no full-render scenes in {SCENES_DIR}"


def test_every_native_renderable_class_appears_in_a_scene():
    missing = _missing(_public_subclasses(Mob, NATIVE_MOB_MODULES), SOURCE)
    assert not missing, (
        "these public Mob classes are not in any full-render scene: "
        f"{missing}. Put them in a scene, or add them to EXEMPT with a reason."
    )


def test_every_material_class_appears_in_a_scene():
    missing = _missing(_public_subclasses(Material), SOURCE)
    assert not missing, f"materials with no render coverage: {missing}"


def test_every_light_class_appears_in_a_scene():
    missing = _missing(_public_subclasses(Light), SOURCE)
    assert not missing, f"light types with no render coverage: {missing}"


@pytest.mark.parametrize(
    ("feature", "names"),
    [
        ("animation contexts", ("Seq", "Sync", "Lag", "Off")),
        (
            "indication animations",
            (
                "Indicate",
                "Wiggle",
                "Circumscribe",
                "Flash",
                "FocusOn",
                "Blink",
                "ApplyWave",
                "ShowPassingFlash",
                "ShowPassingFlashWithThinningStrokeWidth",
            ),
        ),
        (
            "manim animations",
            ("ApplyMatrix", "ApplyComplexFunction", "Homotopy", "MoveAlongPath"),
        ),
        (
            "timeline features",
            ("become(", "add_updater(", "remove_updater(", "wave_color("),
        ),
        (
            "camera and layout",
            (
                "get_camera().rotate(",
                "orbit(",
                "fit_to_screen_rectangle(",
                "move_center_to_screen_position(",
                "move_out_of_screen(",
            ),
        ),
        (
            "material presets",
            ("GLASS", "MIRROR", "COPPER"),
        ),
        (
            "shading and media",
            (
                "set_material(",
                "set_fragment_shader(",
                "ImageMob(",
                "ThreeDModelMob(",
                "glow",
                "opacity",
            ),
        ),
        ("rate functions", ("rate_funcs.linear", "rate_funcs.ease_out_expo")),
        ("plots and tables", ("Axes(", ".plot(", "BarChart(", "Brace(")),
    ],
)
def test_authoring_features_are_covered_by_a_scene(feature, names):
    missing = [name for name in names if name not in SOURCE]
    assert not missing, f"{feature} coverage is missing {missing}"


@pytest.mark.parametrize(
    "scene_path", _scene_paths(), ids=lambda path: path.stem
)
def test_scene_file_follows_the_harness_conventions(scene_path):
    """Scenes author a Scene; they never render one, and never import the world.

    ``torch`` is allowed because building raw primitives needs tensors, which is
    exactly what a user would do.
    """
    source = scene_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(scene_path))

    assert ast.get_docstring(tree), f"{scene_path.name} needs a module docstring"

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.add(f"from {node.module} import " + ",".join(
                alias.name for alias in node.names
            ))
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    assert imported <= {"from algan import *", "torch"}, (
        f"{scene_path.name} imports more than the public API: {sorted(imported)}"
    )
    assert "from algan import *" in imported, (
        f"{scene_path.name} must author against the public star import"
    )

    called = {
        node.func.id for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    } | {
        node.func.attr for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    forbidden = {"render_all_funcs", "save_video", "save_frame"} & called
    assert not forbidden, (
        f"{scene_path.name} must define a scene, not render one: {sorted(forbidden)}"
    )


def test_every_exemption_names_something_that_still_exists():
    """A stale exemption would silently excuse a class that was renamed."""
    unknown = sorted(name for name in EXEMPT if not hasattr(algan, name))
    assert not unknown, f"EXEMPT names that are no longer exported: {unknown}"
