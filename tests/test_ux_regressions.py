from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

import algan
from algan.animation_timeline.animation_contexts import AnimationManager, Sync
from algan.errors import (
    AlganConfigurationError,
    HierarchyError,
    UnsupportedFeatureError,
)
from algan.mobs.group import Group
from algan.mobs.shapes_2d import Square
from algan.rendering.lights import PointLight, RectAreaLight, SpotLight
from algan.rendering.camera import Camera
from algan.rendering.raytracing import settings as rt_settings
from algan.rendering.raytracing.tracer import (
    RenderPlan,
    _validate_render_capabilities,
)
from algan.scene_manager import SceneManager
from algan.settings.render_settings import PREVIEW, RenderSettings
from algan.sound.audio_effect import AudioManager
from algan.utils import algan_utils


@pytest.fixture(autouse=True)
def reset_global_authoring_state():
    SceneManager.reset()
    AudioManager.reset()
    previous_policy = rt_settings.UNSUPPORTED_FEATURE_POLICY
    previous_textured = rt_settings.WF_TEXTURED
    previous_sorted = rt_settings.WAVEFRONT_SORT_MATERIALS
    yield
    rt_settings.UNSUPPORTED_FEATURE_POLICY = previous_policy
    rt_settings.WF_TEXTURED = previous_textured
    rt_settings.WAVEFRONT_SORT_MATERIALS = previous_sorted
    SceneManager.reset()
    AudioManager.reset()


def test_import_does_not_change_pytorch_autograd_state():
    code = """
import torch
before = (torch.is_grad_enabled(), torch.is_inference_mode_enabled())
import algan
assert (torch.is_grad_enabled(), torch.is_inference_mode_enabled()) == before
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


def test_context_is_restored_after_user_exception():
    root = AnimationManager.instance().context
    children_before = list(root.child_contexts)
    failed = Sync()
    with pytest.raises(RuntimeError):
        with failed:
            raise RuntimeError("boom")
    assert AnimationManager.instance().context is root
    assert root.child_contexts == children_before
    assert failed not in root.child_contexts


def test_same_run_time_tolerates_zero_duration_children():
    with Sync(same_run_time=True):
        with algan.Off():
            pass


def test_save_frame_restores_all_derived_render_state(monkeypatch, tmp_path):
    scene = SceneManager.instance()
    scene.set_render_settings(PREVIEW)
    before = {
        "render_settings": scene.render_settings,
        "size": scene.size.clone() if torch.is_tensor(scene.size) else scene.size,
        "frames_per_second": scene.frames_per_second,
        "frame_size": tuple(scene.frame_size),
        "num_pixels": scene.num_pixels,
        "width": scene.num_pixels_screen_width,
        "height": scene.num_pixels_screen_height,
    }
    temporary = RenderSettings((17, 13), 2, anti_alias_level=1)

    def fake_frames(*_args, **_kwargs):
        yield torch.zeros(
            (1, scene.num_pixels_screen_height, scene.num_pixels_screen_width, 4),
            dtype=torch.uint8,
        )

    saved = []
    monkeypatch.setattr(scene, "get_frames", fake_frames)
    import torchvision.utils

    monkeypatch.setattr(
        torchvision.utils,
        "save_image",
        lambda frame, path: saved.append((tuple(frame.shape), path)),
    )

    scene.save_frame(tmp_path / "still", render_settings=temporary)

    assert scene.render_settings is before["render_settings"]
    if torch.is_tensor(scene.size):
        assert torch.equal(scene.size, before["size"])
    else:
        assert scene.size == before["size"]
    assert scene.frames_per_second == before["frames_per_second"]
    assert tuple(scene.frame_size) == before["frame_size"]
    assert scene.num_pixels == before["num_pixels"]
    assert scene.num_pixels_screen_width == before["width"]
    assert scene.num_pixels_screen_height == before["height"]
    assert saved == [((4, 13, 17), str(tmp_path / "still.png"))]


def test_overwrite_false_checks_final_suffixed_path_and_preserves_scene(tmp_path):
    scene = SceneManager.instance()
    Square(add_to_scene=True)
    actor_count = len(scene.actors)
    destination = tmp_path / "scene.mp4"
    destination.write_bytes(b"existing")

    result = algan_utils.render_to_file(
        "scene",
        output_path=tmp_path,
        output_dir="",
        render_settings=RenderSettings((8, 8), 1, anti_alias_level=1),
        overwrite=False,
    )

    assert result.status == "skipped"
    assert result.output_path == destination
    assert len(scene.actors) == actor_count
    assert SceneManager.instance() is scene


def test_conflicting_output_extensions_fail_before_render(tmp_path):
    scene = SceneManager.instance()
    before_settings = scene.render_settings
    before_background = scene.background_frame
    with pytest.raises(AlganConfigurationError, match="Conflicting"):
        algan_utils.render_to_file(
            "scene.mp4",
            output_path=tmp_path,
            output_dir="",
            file_extension="mov",
            render_settings=RenderSettings((8, 8), 1, anti_alias_level=1),
        )
    assert SceneManager.instance() is scene
    assert scene.render_settings is before_settings
    assert scene.background_frame is before_background


def test_render_setup_failure_resets_scene_and_audio(monkeypatch, tmp_path):
    scene = SceneManager.instance()
    AudioManager.instance().video_transcript = "stale"
    monkeypatch.setattr(scene, "render_audio_to_file", lambda *_a, **_k: None)
    monkeypatch.setattr(
        algan_utils,
        "get_file_writer",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("writer failed")),
    )

    with pytest.raises(RuntimeError, match="writer failed"):
        algan_utils.render_to_file(
            "failure.mp4",
            output_path=tmp_path,
            output_dir="",
            render_settings=RenderSettings((8, 8), 1, anti_alias_level=1),
            animate_fade_out=False,
        )

    replacement_scene = SceneManager.instance()
    assert replacement_scene is not scene
    assert AudioManager.instance().video_transcript == ""
    assert not replacement_scene.camera.location.is_inference()


def test_group_uses_one_member_store_and_repairs_parent_links():
    first = Square(add_to_scene=False)
    second = Square(add_to_scene=False)
    replacement = Square(add_to_scene=False)
    group = Group(first, second, add_to_scene=False)

    assert group.mobs is group.children
    group[0] = replacement

    assert group.children == [replacement, second]
    assert group not in first.parents
    assert group in replacement.parents


def test_group_slicing_is_pure_and_empty_slices_remain_groups():
    scene = SceneManager.instance()
    first = Square(add_to_scene=False)
    second = Square(add_to_scene=False)
    group = Group(first, second, add_to_scene=False)
    actor_count = len(scene.actors)
    parents_before = (list(first.parents), list(second.parents))

    view = group[:1]
    empty = group[:0]

    assert isinstance(view, Group)
    assert isinstance(empty, Group)
    assert len(empty) == 0
    assert len(scene.actors) == actor_count
    assert first.parents == parents_before[0]
    assert second.parents == parents_before[1]


def test_group_layouts_are_safe_for_empty_and_ragged_groups():
    empty = Group(add_to_scene=False)
    assert empty.arrange_in_line() is empty
    assert empty.arrange_between_points(torch.zeros(3), torch.ones(3)) is empty
    assert empty.arrange_in_grid() is empty

    ragged = Group(
        *(Square(add_to_scene=False) for _ in range(5)),
        add_to_scene=False,
    )
    assert ragged.arrange_in_grid(num_rows=2, tight_axis=0) is ragged
    with pytest.raises(AlganConfigurationError):
        ragged.arrange_in_grid(num_rows=0)


def test_hierarchy_rejects_cycles_and_duplicates():
    child = Square(add_to_scene=False)
    group = Group(child, add_to_scene=False)
    with pytest.raises(HierarchyError):
        group.add_children(group)
    with pytest.raises(HierarchyError):
        group.replace_children([child, child])


def test_render_settings_are_immutable_validated_and_typo_safe():
    with pytest.raises(AlganConfigurationError, match="Did you mean 'resolution'"):
        PREVIEW.set(resoluton=(1, 1))
    with pytest.raises(AlganConfigurationError, match="positive"):
        PREVIEW.set(resolution=(0, 1))
    with pytest.raises(Exception):
        PREVIEW.frames_per_second = 99

    changed = PREVIEW.replace(frames_per_second=12)
    assert changed.frames_per_second == 12
    assert PREVIEW.frames_per_second != 12


def test_spawned_light_registers_once_and_add_light_is_chainable():
    scene = SceneManager.instance()
    initial = len(scene.light_sources)
    light = PointLight()

    assert light.spawn(animate=False) is light
    light.spawn(animate=False)
    scene.add_light_source(light)

    assert len(scene.light_sources) == initial + 1
    assert sum(item is light for item in scene.light_sources) == 1
    assert scene.remove_light(light) is light
    assert all(item is not light for item in scene.light_sources)


def test_light_parameters_are_validated_instead_of_silently_clamped():
    with pytest.raises(AlganConfigurationError, match="intensity"):
        PointLight(intensity=-1)
    with pytest.raises(AlganConfigurationError, match="penumbra"):
        SpotLight(penumbra=2)
    with pytest.raises(AlganConfigurationError, match="samples"):
        RectAreaLight(samples=0)


def test_monte_carlo_unsupported_features_fail_preflight():
    rt_settings.set_unsupported_feature_policy("error")
    merged = {"has_refractive": True, "has_user_pipeline": True}
    extended_light = SimpleNamespace(_render_aux=object())

    with pytest.raises(UnsupportedFeatureError) as exc_info:
        _validate_render_capabilities(
            4,
            torch.zeros((1, 1, 3)),
            merged,
            [extended_light],
        )
    message = str(exc_info.value)
    assert "environment maps" in message
    assert "refractive materials" in message
    assert "custom fragment-shader pipelines" in message
    assert "extended lights" in message


def test_render_plan_describes_supported_deterministic_route():
    plan = _validate_render_capabilities(
        1,
        None,
        {"has_refractive": True, "has_user_pipeline": False},
        [],
    )
    assert isinstance(plan, RenderPlan)
    assert plan.backend == "deterministic_wavefront"
    assert plan.samples_per_pixel == 1
    assert plan.requested_features == ("refractive materials",)
    assert plan.is_supported
    assert plan.as_dict()["unsupported_features"] == []


def test_known_broken_renderer_switches_are_hard_disabled():
    with pytest.raises(UnsupportedFeatureError):
        rt_settings.set_textured_wavefront(True)
    with pytest.raises(UnsupportedFeatureError):
        rt_settings.set_textured_features(1)
    with pytest.raises(UnsupportedFeatureError):
        rt_settings.set_material_sorting(True)


def test_scene_decorator_prevents_helpers_from_being_discovered(monkeypatch):
    import types

    module_name = "_algan_scene_registry_test"
    module = types.ModuleType(module_name)
    calls = []

    def helper():
        calls.append("helper")

    @algan_utils.scene(name="main")
    def entry_point():
        calls.append("scene")

    helper.__module__ = module_name
    entry_point.__module__ = module_name
    module.helper = helper
    module.entry_point = entry_point
    monkeypatch.setitem(sys.modules, module_name, module)

    results = algan_utils.render_all_funcs(module_name, smoke_test=True)

    assert calls == ["scene"]
    assert results == []


def test_root_star_exports_exclude_dependency_modules_and_typing_helpers():
    namespace = {}
    exec("from algan import *", namespace)
    for leaked in ("os", "sys", "torch", "np", "F", "Any", "Callable"):
        assert leaked not in namespace
    for expected in ("Square", "Scene", "render", "render_to_file", "scene", "RED"):
        assert expected in namespace


def test_camera_validates_projection_and_clip_parameters():
    with pytest.raises(AlganConfigurationError, match="fov"):
        Camera(fov=180, add_to_scene=False)
    with pytest.raises(AlganConfigurationError, match="near"):
        Camera(near=2, far=1, add_to_scene=False)

    root_end = AnimationManager.instance().context.timespan.original_end
    orthographic_camera = Camera(orthographic=True, add_to_scene=False)
    assert orthographic_camera.orthographic is True
    assert AnimationManager.instance().context.timespan.original_end == root_end

    camera = Camera(add_to_scene=False)
    with pytest.raises(AlganConfigurationError, match="positive"):
        camera.set_distance_to_screen(0)
    assert camera.set_near_orthographic(1000) is camera
    assert camera.orthographic is True
    assert camera.set_fov(45) is camera
    assert camera.orthographic is False
    scene = SceneManager.instance()
    scene.set_render_settings(RenderSettings((20, 10), 1, anti_alias_level=1))
    assert camera.pixel_height == pytest.approx(0.2)
    scene.set_render_settings(RenderSettings((40, 20), 1, anti_alias_level=1))
    assert camera.pixel_height == pytest.approx(0.1)


def test_static_off_scene_gets_one_frame_before_final_despawn(monkeypatch, tmp_path):
    scene = SceneManager.instance()
    with algan.Off():
        Square(add_to_scene=True).spawn(animate=False)

    observed = {}

    class Writer:
        def close(self):
            return None

    monkeypatch.setattr(scene, "render_audio_to_file", lambda *_a, **_k: None)
    monkeypatch.setattr(algan_utils, "get_file_writer", lambda *_a, **_k: Writer())

    def fake_render_to_video(*_args, **render_kwargs):
        observed["end"] = AnimationManager.instance().context.timespan.original_end
        observed["background_override"] = render_kwargs.get("background_color")

    monkeypatch.setattr(scene, "render_to_video", fake_render_to_video)
    result = algan_utils.render_to_file(
        "static.mp4",
        output_path=tmp_path,
        output_dir="",
        render_settings=RenderSettings((8, 8), 4, anti_alias_level=1),
        animate_fade_out=False,
    )

    assert observed["end"] >= 0.25
    assert observed["background_override"] is None
    assert result.status == "rendered"
