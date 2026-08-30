"""Audits that the full-render suite still covers the public rendering API.

The render suite is deliberately small: six dense scenes instead of one scene
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
    "PMobject": "point-cloud base; covered through its concrete subclasses",
    "Mobject1D": "point-cloud base; covered through PointCloudDot",
    "Mobject2D": "point-cloud base; covered through its concrete subclasses",
    # Manim's OpenGL renderer aliases. Algan has one renderer; these exist so
    # that Manim code that names them keeps importing.
    "OpenGLPMobject": "OpenGL-renderer alias",
    "OpenGLPGroup": "OpenGL-renderer alias",
    "OpenGLPMPoint": "OpenGL-renderer alias",
    # Pre-Three.js material API, kept working for existing scripts and pinned
    # by tests/unit_tests/test_materials.py.
    "PBRMaterial": "legacy material API, unit-tested",
    "AdvancedPBRMaterial": "legacy material API, unit-tested",
    # Algan's default: installed at import as SETTINGS.style.default_material,
    # so every 3-D Mob with no material of its own renders through it. Pinned
    # by tests/unit_tests/test_default_material.py.
    "DiffuseMaterial": "the default 3-D material, installed at import; unit-tested",
    "SpecularMaterial": "legacy material API, unit-tested",
    "UnlitMaterial": "legacy material API, unit-tested",
    # Reached through Scene.use_manim_defaults(), which repoints the default
    # material rather than asking scenes to author it; the full-render scenes
    # all pin explicit materials. Pinned by
    # tests/unit_tests/test_manim_shader.py.
    "ManimMaterial": "installed by use_manim_defaults, unit-tested",
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
                "fit_to_screen(",
                "move_center_to_screen_position(",
                "move_off_screen(",
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
                "Model3D(",
                "glow",
                "opacity",
            ),
        ),
        ("rate functions", ("easings.linear", "easings.ease_out_expo")),
        ("plots and tables", ("Axes(", ".plot(", "BarChart(", "Brace(")),
    ],
)
def test_authoring_features_are_covered_by_a_scene(feature, names):
    missing = [name for name in names if name not in SOURCE]
    assert not missing, f"{feature} coverage is missing {missing}"


@pytest.mark.parametrize("scene_path", _scene_paths(), ids=lambda path: path.stem)
def test_scene_file_follows_the_harness_conventions(scene_path):
    """Scenes author a Scene; they never render one, and never import the world.

    ``torch`` is allowed because building raw primitives needs tensors, which is
    exactly what a user would do, and so is ``algan.manim``: since the API
    overhaul's Phase 1 that is the public spelling of the compatibility layer,
    and a scene covering compat geometry has to reach it the way a user would.
    """
    source = scene_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(scene_path))

    assert ast.get_docstring(tree), f"{scene_path.name} needs a module docstring"

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.add(
                f"from {node.module} import "
                + ",".join(alias.name for alias in node.names)
            )
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    assert imported <= {"from algan import *", "torch", "algan.manim"}, (
        f"{scene_path.name} imports more than the public API: {sorted(imported)}"
    )
    assert "from algan import *" in imported, (
        f"{scene_path.name} must author against the public star import"
    )

    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    } | {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    forbidden = {"render_all_funcs", "save_video", "save_frame"} & called
    assert not forbidden, (
        f"{scene_path.name} must define a scene, not render one: {sorted(forbidden)}"
    )


#: Text classes whose glyph layout comes from a font the host supplies, and
#: which therefore have to name one of the vendored families explicitly. ``Tex``
#: and ``MathTex`` are deliberately absent: they go through LaTeX and dvisvgm to
#: outlines and do not consult fontconfig at all.
_FONT_BEARING_CLASSES = {"Text", "MarkupText", "Paragraph"}


@pytest.mark.parametrize("scene_path", _scene_paths(), ids=lambda path: path.stem)
def test_scene_text_pins_a_vendored_font(scene_path):
    """Every Text-like call names a font, so renders do not depend on the host.

    ``Text`` defaults to ``font=""``, which Pango resolves through fontconfig,
    so the glyph advances change with whatever the machine has installed. That
    is not hypothetical: before the fonts were vendored, the CPU and CUDA
    baselines' Text differed by up to 230 channel values -- structurally, not by
    a sub-pixel shift -- while their geometry agreed to a mean of 0.36.

    One unpinned call is enough to reintroduce the drift for a whole scene, and
    it would surface as a baseline failure that looks like a renderer
    regression, so this is checked rather than left to review.
    """
    tree = ast.parse(scene_path.read_text(encoding="utf-8"), filename=str(scene_path))

    unpinned = [
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FONT_BEARING_CLASSES
        and not any(keyword.arg == "font" for keyword in node.keywords)
    ]
    assert not unpinned, (
        f"{scene_path.name} has {len(unpinned)} {sorted(set(unpinned))} call(s) with no "
        "font=; pass font=FONT so the render does not depend on the host's fonts"
    )


def test_every_exemption_names_something_that_still_exists():
    """A stale exemption would silently excuse a class that was renamed.

    Both namespaces count. An exemption says "this class does not need render
    coverage", which stays true wherever the class is reachable from -- the
    point-cloud and OpenGL-alias bases live in ``algan.manim`` rather than the
    root namespace, and are no less real for it.
    """
    import algan.manim as manim_namespace

    unknown = sorted(
        name
        for name in EXEMPT
        if not hasattr(algan, name) and not hasattr(manim_namespace, name)
    )
    assert not unknown, (
        f"EXEMPT names that exist in neither algan nor algan.manim: {unknown}"
    )
