import ast
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parents[1]
SCENES_DIR = TESTS_DIR / "full_renders" / "scenes"
EXPECTED_SCENES = {
    "geometry_and_camera.py",
    "materials_and_lighting.py",
    "media_and_shaders.py",
    "timeline_and_text.py",
}
REQUIRED_SCENE_FEATURES = {
    "2D shapes": ("Circle(", "Square(", "RegularPolygon(", "Star("),
    "3D shapes": ("Sphere(", "Cylinder(", "Cone(", "Torus(", "Cube("),
    "animation contexts": ("Seq(", "Sync(", "Lag(", "Off("),
    "text": ("Text(", "Tex(", "NumericDisplay("),
    "materials": (
        "MeshBasicMaterial(",
        "MeshLambertMaterial(",
        "MeshPhongMaterial(",
        "MeshStandardMaterial(",
        "MeshPhysicalMaterial(",
        "MeshToonMaterial(",
        "MeshNormalMaterial(",
        "MeshMatcapMaterial(",
        "MeshDepthMaterial(",
    ),
    "media and shaders": ("ImageMob(", "ThreeDModelMob(", "set_fragment_shader("),
    "timeline features": (
        "become(",
        "add_updater(",
        "remove_updater(",
        "Indicate(",
        "Circumscribe(",
        "ApplyWave(",
    ),
    "camera and lighting": (
        "get_camera().rotate(",
        "AmbientLight(",
        "DirectionalLight(",
    ),
}


def _scene_paths():
    return sorted(SCENES_DIR.glob("*.py"))


def test_full_render_scene_set_is_deliberate_and_stable():
    assert {path.name for path in _scene_paths()} == EXPECTED_SCENES


def test_scene_files_use_only_the_public_star_import_and_do_not_render_themselves():
    forbidden_calls = {"render_all_funcs", "save_video"}
    for path in _scene_paths():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports = [
            node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        assert len(imports) == 1, f"{path.name} must have exactly one import"
        import_node = imports[0]
        assert isinstance(import_node, ast.ImportFrom)
        assert import_node.module == "algan"
        assert [alias.name for alias in import_node.names] == ["*"]

        calls = [
            node.func
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
        ]
        call_names = {
            func.id
            for func in calls
            if isinstance(func, ast.Name)
        } | {
            func.attr
            for func in calls
            if isinstance(func, ast.Attribute)
        }
        assert forbidden_calls.isdisjoint(call_names), (
            f"{path.name} must define a scene, not invoke the render harness"
        )
        assert "save_frame" in call_names, (
            f"{path.name} needs an authored visual-checkpoint frame"
        )


def test_full_render_scenes_cover_the_audited_authoring_surface():
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in _scene_paths()
    )
    for feature, tokens in REQUIRED_SCENE_FEATURES.items():
        missing = [token for token in tokens if token not in source]
        assert not missing, f"{feature} coverage is missing {missing}"
