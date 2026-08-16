from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import algan
from algan import render_loop
from algan.animation_timeline.animation_contexts import Sync
from algan.errors import (
    AlganConfigurationError,
    HierarchyError,
    UnsupportedFeatureError,
)
from algan.mobs.group import Group
from algan.mobs.shapes_2d import Square
from algan.rendering.camera import Camera
from algan.rendering.lights import PointLight, RectAreaLight, SpotLight
from algan.rendering.raytracing import settings as rt_settings
from algan.rendering.raytracing.tracer import (
    RenderPlan,
    _validate_render_capabilities,
)
from algan.rendering.taichi_runtime import _loaded_from_offline_cache
from algan.scene_manager import SceneManager
from algan.settings.video_settings import PREVIEW, VideoSettings
from algan.utils import algan_utils


@pytest.fixture(autouse=True)
def reset_global_authoring_state():
    SceneManager.reset()
    previous_policy = rt_settings.UNSUPPORTED_FEATURE_POLICY
    previous_textured = rt_settings.WF_TEXTURED
    previous_sorted = rt_settings.WAVEFRONT_SORT_MATERIALS
    yield
    rt_settings.UNSUPPORTED_FEATURE_POLICY = previous_policy
    rt_settings.WF_TEXTURED = previous_textured
    rt_settings.WAVEFRONT_SORT_MATERIALS = previous_sorted
    SceneManager.reset()


def test_context_is_restored_after_user_exception():
    scene = SceneManager.instance().current_scene
    root = scene.animation_manager.context
    children_before = list(root.child_contexts)
    failed = Sync()
    with pytest.raises(RuntimeError), failed:
        raise RuntimeError("boom")
    assert scene.animation_manager.context is root
    assert root.child_contexts == children_before
    assert failed not in root.child_contexts


def test_kernel_compile_notice_ignores_offline_cache_hits():
    assert _loaded_from_offline_cache(b"Create kernel 'wavefront_shade' from cache")
    assert not _loaded_from_offline_cache(b"Cache kernel 'wavefront_shade'")


def test_same_run_time_tolerates_zero_duration_children():
    with Sync(same_run_time=True), algan.Off():
        pass


def test_save_frame_restores_all_derived_render_state(monkeypatch, tmp_path):
    scene = SceneManager.instance().current_scene
    scene.set_video_settings(PREVIEW)
    before = {
        "video_settings": scene.video_settings,
        "size": scene.size.clone() if torch.is_tensor(scene.size) else scene.size,
        "frames_per_second": scene.frames_per_second,
        "frame_size": tuple(scene.frame_size),
        "num_pixels": scene.num_pixels,
        "width": scene.num_pixels_screen_width,
        "height": scene.num_pixels_screen_height,
    }
    temporary = VideoSettings((17, 13), 2, anti_alias_level=1)

    def fake_frames(*_args, **_kwargs):
        yield torch.zeros(
            (1, scene.num_pixels_screen_height, scene.num_pixels_screen_width, 4),
            dtype=torch.uint8,
        )

    monkeypatch.setattr(scene, "get_frames", fake_frames)

    scene.save_frame(tmp_path / "still", temporary)

    assert scene.video_settings == before["video_settings"]
    if torch.is_tensor(scene.size):
        assert torch.equal(scene.size, before["size"])
    else:
        assert scene.size == before["size"]
    assert scene.frames_per_second == before["frames_per_second"]
    assert tuple(scene.frame_size) == before["frame_size"]
    assert scene.num_pixels == before["num_pixels"]
    assert scene.num_pixels_screen_width == before["width"]
    assert scene.num_pixels_screen_height == before["height"]
    from PIL import Image

    with Image.open(tmp_path / "still.png") as still:
        assert still.size == (17, 13)
        assert still.mode == "RGBA"


def _stub_out_frame_writing(monkeypatch, scene, on_render=None):
    """Make save_frame go through its full body without a real render."""

    def fake_frames(*_args, **_kwargs):
        if on_render is not None:
            on_render()
        yield torch.zeros(
            (1, scene.num_pixels_screen_height, scene.num_pixels_screen_width, 4),
            dtype=torch.uint8,
        )

    monkeypatch.setattr(scene, "get_frames", fake_frames)


def test_save_frame_resolves_negative_at_from_current_context_time(
    monkeypatch, tmp_path
):
    scene = SceneManager.instance().current_scene
    scene.set_video_settings(VideoSettings((17, 13), 10, anti_alias_level=1))
    scene.animation_manager.context.timespan.current_time = 3.0
    requested_windows = []

    def fake_frames(start_ind, end_ind, **_kwargs):
        requested_windows.append((start_ind, end_ind))
        yield torch.zeros((1, 13, 17, 4), dtype=torch.uint8)

    monkeypatch.setattr(scene, "get_frames", fake_frames)

    scene.save_frame(tmp_path / "earlier", at=-0.5)

    assert requested_windows == [(25, 26)]


def test_save_frame_rejects_negative_at_before_scene_start(monkeypatch, tmp_path):
    scene = SceneManager.instance().current_scene
    scene.animation_manager.context.timespan.current_time = 0.25
    _stub_out_frame_writing(monkeypatch, scene)

    with pytest.raises(AlganConfigurationError, match="non-negative"):
        scene.save_frame(tmp_path / "before_start", at=-0.5)


def test_text_creates_manim_directories_inside_algan_cache(monkeypatch, tmp_path):
    from algan.mobs import text as text_module

    class FakeManimConfig:
        tex_dir = "{media_dir}/Tex"
        text_dir = "{media_dir}/texts"

        def get_dir(self, name):
            return Path(getattr(self, name))

    config = FakeManimConfig()
    monkeypatch.setattr(text_module, "mn", SimpleNamespace(config=config))
    algan_cache = tmp_path / "algan_cache"

    with algan.SETTINGS.paths.override(cache_directory=algan_cache):
        text_module.make_manim_dir()

    assert Path(config.tex_dir) == algan_cache / "manim" / "Tex"
    assert Path(config.text_dir) == algan_cache / "manim" / "texts"
    assert Path(config.tex_dir).is_dir()
    assert Path(config.text_dir).is_dir()


@pytest.mark.slow
def test_importing_algan_redirects_manim_tex_dirs_without_touching_disk(tmp_path):
    # ``make_manim_dir`` used to be reached only from ``Tex.__init__``, via the
    # ``LazyModule`` extras that pull in the svg cache.  ``manim_compat``
    # imports manim eagerly and bypassed them, so every Manim-backed mob that
    # reaches LaTeX without a ``Tex`` being built first -- ``MathTex``,
    # ``Title``, ``ManimMob(manim.MathTex(...))`` -- ran against manim's
    # default ``media/Tex`` and died with ``FileNotFoundError`` on a clean
    # directory.  Docs builds masked it by exec'ing every example in one
    # process, where an earlier ``Tex`` had already installed the redirect.
    #
    # Checked in a subprocess with a pristine cwd because it is a property of
    # import order, and the in-process suite has long since built a ``Tex``.
    import os
    import subprocess

    environ = dict(os.environ)
    environ["ALGAN_HOME"] = os.fspath(tmp_path / "algan_home")

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import pathlib, algan, manim;"
            "print('TEX_DIR', manim.config.tex_dir);"
            "print('PATCHED', getattr("
            "manim.utils.tex_file_writing.generate_tex_file,"
            " '_algan_ensures_tex_dir', False));"
            "print('MEDIA', pathlib.Path('media').exists())",
        ],
        capture_output=True,
        check=False,
        cwd=os.fspath(tmp_path),
        env=environ,
        text=True,
        timeout=300,
    )

    assert completed.returncode == 0, completed.stderr
    assert "TEX_DIR " + os.fspath(tmp_path / "algan_home") in completed.stdout
    # The single-level ``tex_dir.mkdir()`` in manim's ``generate_tex_file`` is
    # what actually raised; the wrapper has to be in place before first use.
    assert "PATCHED True" in completed.stdout
    # Importing Algan must still not write anything: the redirect no longer
    # creates the directories, they are made on first use.
    assert "MEDIA False" in completed.stdout
    assert not (tmp_path / "media").exists()


def test_save_frame_does_not_freeze_replay_windows_of_an_open_context(
    monkeypatch, tmp_path
):
    # A render resolves every edit's replay window by snapshotting its
    # context-rescaled end time into a plain float. Called from inside a
    # context that has not exited yet, those ends are pre-rescale, and nothing
    # invalidates them once the block is rescaled -- so a later render replayed
    # the animations against stale, too-early ends and cut them short.
    scene = SceneManager.instance().current_scene
    scene.set_video_settings(PREVIEW)
    timeline = scene.timeline_manager

    def prepare_transient_queries():
        timeline._resolve_replay_windows()
        for attr_timeline in timeline.attr_to_timeline.values():
            attr_timeline.prepare_for_queries()
            attr_timeline._prepared_queries(torch.tensor([0.0]))

    _stub_out_frame_writing(monkeypatch, scene, on_render=prepare_transient_queries)

    square = Square().spawn()
    with algan.Seq(run_time=8):
        square.move(algan.RIGHT)
        square.move(algan.UP)
        # Last statement in the block: nothing after it records an edit, so
        # nothing would invalidate a resolution left behind here.
        scene.save_frame(tmp_path / "still")
        for attr_timeline in timeline.attr_to_timeline.values():
            assert not attr_timeline._is_ready_for_queries
            assert not attr_timeline._query_cache
            assert not attr_timeline._edits_sorted

    timeline._resolve_replay_windows()
    edits = [edit for t in timeline.attr_to_timeline.values() for edit in t.edits]
    assert edits
    # The block really was rescaled on exit: its two moves span 8 seconds from
    # t=1 (the spawn ahead of it) rather than the 2 they were recorded at.
    assert max(float(edit.time.end) for edit in edits) == pytest.approx(9.0)
    for edit in edits:
        assert edit.replay_end == pytest.approx(float(edit.time.end))
    # Query dictionaries cache replay_end as plain timestamps too.  The ones
    # prepared by the mid-context render must have been discarded rather than
    # carrying the pre-rescale two-second block into this final resolution.
    for attr_timeline in timeline.attr_to_timeline.values():
        attr_timeline.prepare_for_queries()
        edit_timestamps = [
            edit["timestamp"] for edit in attr_timeline._edits_sorted[:-1]
        ]
        assert edit_timestamps == pytest.approx(
            [
                edit.replay_end if edit.replay_end is not None else float(edit.time.end)
                for edit in attr_timeline.edits
            ]
        )


def test_save_frame_leaves_a_finished_scene_s_replay_windows_alone(
    monkeypatch, tmp_path
):
    scene = SceneManager.instance().current_scene
    scene.set_video_settings(PREVIEW)
    timeline = scene.timeline_manager
    _stub_out_frame_writing(
        monkeypatch, scene, on_render=timeline._resolve_replay_windows
    )

    square = Square().spawn()
    with algan.Seq(run_time=8):
        square.move(algan.RIGHT)

    # Timings are final here, so resolving is legitimate and must survive.
    timeline._resolve_replay_windows()
    before = [
        (edit, edit.replay_end)
        for t in timeline.attr_to_timeline.values()
        for edit in t.edits
    ]
    known_lifespans = dict(timeline.mob_id_to_lifespan)

    scene.save_frame(tmp_path / "still")

    assert timeline._replay_windows_resolved
    assert [(edit, edit.replay_end) for edit, _ in before] == before
    assert timeline.mob_id_to_lifespan == known_lifespans


def _stub_out_video_writing(monkeypatch, scene, on_render=None):
    """Drive render_to_video without ffmpeg. Returns the recorded windows."""
    windows = []

    def fake_frames(start_ind, end_ind, **_kwargs):
        windows.append((start_ind, end_ind))
        if on_render is not None:
            on_render()
        return iter(())

    monkeypatch.setattr(scene, "get_frames", fake_frames)
    monkeypatch.setattr(
        render_loop,
        "write_frames_from_queue",
        lambda queue, _writer: [None for _ in iter(queue.get, None)],
    )
    return windows


def _render_to_video(scene, tmp_path, name="clip"):
    source = tmp_path / f"{name}_temp.mp4"
    source.write_bytes(b"")
    scene.render_to_video(
        SimpleNamespace(close=lambda: None),
        str(source),
        str(tmp_path / f"{name}.mp4"),
        despawn_camera_and_lights=False,
        preserve_authoring_state=True,
    )


def test_render_window_covers_the_whole_open_context_chain(monkeypatch, tmp_path):
    # Mid-block the active context is the innermost open one, and its window
    # covers only its own block. An enclosing Sync can already hold animations
    # running well past it, which a preview must not cut off.
    scene = SceneManager.instance().current_scene
    scene.set_video_settings(PREVIEW)
    windows = _stub_out_video_writing(monkeypatch, scene)

    square = Square().spawn()
    with algan.Sync():
        with algan.Seq(run_time=5):
            square.move(algan.RIGHT)
            square.move(algan.UP)
        with algan.Seq():
            square.move(algan.DOWN)
            assert scene.animation_manager.context.timespan.original_end == (
                pytest.approx(2.0)
            )
            _render_to_video(scene, tmp_path)

    assert windows == [(0, round(6.0 * scene.frames_per_second))]


def test_save_video_reset_false_rolls_back_derived_state_mid_block(
    monkeypatch, tmp_path
):
    # Same defect as save_frame, reached through the video path: a render from
    # inside an unfinished block must leave nothing behind that a later render
    # would replay against.
    scene = SceneManager.instance().current_scene
    scene.set_video_settings(PREVIEW)
    timeline = scene.timeline_manager
    _stub_out_video_writing(
        monkeypatch, scene, on_render=timeline._resolve_replay_windows
    )
    scene_times_before = [list(pair) for pair in scene.scene_times]

    square = Square().spawn()
    with algan.Seq(run_time=8):
        square.move(algan.RIGHT)
        square.move(algan.UP)
        # Last statement in the block, so nothing afterwards invalidates a
        # resolution left behind here.
        _render_to_video(scene, tmp_path)

    assert scene.scene_times == scene_times_before

    timeline._resolve_replay_windows()
    edits = [edit for t in timeline.attr_to_timeline.values() for edit in t.edits]
    assert edits
    assert max(float(edit.time.end) for edit in edits) == pytest.approx(9.0)
    for edit in edits:
        assert edit.replay_end == pytest.approx(float(edit.time.end))


def test_overwrite_false_checks_final_suffixed_path_and_preserves_scene(tmp_path):
    scene = SceneManager.instance().current_scene
    Square(add_to_scene=True)
    actor_count = len(scene.actors)
    destination = tmp_path / "scene.mp4"
    destination.write_bytes(b"existing")

    result = algan.Scene.save_video(
        tmp_path / "scene",
        video_settings=VideoSettings((8, 8), 1, anti_alias_level=1),
        overwrite=False,
    )

    assert result.status == "skipped"
    assert result.output_path == destination
    assert len(scene.actors) == actor_count
    assert SceneManager.instance().current_scene is scene


def test_transparent_mp4_fails_before_render_and_preserves_scene(tmp_path):
    scene = SceneManager.instance().current_scene
    before_settings = scene.video_settings
    before_background = scene.background_frame
    with pytest.raises(AlganConfigurationError, match="MP4"):
        algan.Scene.save_video(
            tmp_path / "scene.mp4",
            video_settings=VideoSettings((8, 8), 1, anti_alias_level=1),
            background_color=algan.TRANSPARENT,
        )
    assert SceneManager.instance().current_scene is scene
    assert scene.video_settings == before_settings
    assert scene.background_frame is before_background


def test_render_setup_failure_resets_scene_and_audio(monkeypatch, tmp_path):
    scene = SceneManager.instance().current_scene
    old_managers = (
        scene.timeline_manager,
        scene.animation_manager,
        scene.audio_manager,
    )
    scene.audio_manager.video_transcript = "stale"
    monkeypatch.setattr(scene, "render_audio_to_file", lambda *_a, **_k: None)
    monkeypatch.setattr(
        algan_utils,
        "get_file_writer",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("writer failed")),
    )

    with pytest.raises(RuntimeError, match="writer failed"):
        algan.Scene.save_video(
            tmp_path / "failure.mp4",
            video_settings=VideoSettings((8, 8), 1, anti_alias_level=1),
            animate_fade_out=False,
            reset=True,
        )

    replacement_scene = SceneManager.instance().current_scene
    assert replacement_scene is scene
    assert replacement_scene.timeline_manager is not old_managers[0]
    assert replacement_scene.animation_manager is not old_managers[1]
    assert replacement_scene.audio_manager is not old_managers[2]
    assert replacement_scene.audio_manager.video_transcript == ""
    assert replacement_scene.camera.location.is_inference()


def test_default_render_keeps_the_scene_authorable(monkeypatch, tmp_path):
    """save_video defaults to reset=False: mobs stay valid and spawned."""
    scene = SceneManager.instance().current_scene
    managers = (
        scene.timeline_manager,
        scene.animation_manager,
        scene.audio_manager,
    )
    square = Square(add_to_scene=True).spawn(animate=False)

    class Writer:
        def close(self):
            return None

    monkeypatch.setattr(scene, "render_audio_to_file", lambda *_a, **_k: None)
    monkeypatch.setattr(algan_utils, "get_file_writer", lambda *_a, **_k: Writer())
    monkeypatch.setattr(scene, "render_to_video", lambda *_a, **_k: None)

    result = algan.Scene.save_video(
        tmp_path / "keep.mp4",
        video_settings=VideoSettings((8, 8), 1, anti_alias_level=1),
        animate_fade_out=False,
    )

    assert result.status == "rendered"
    # Same managers, so every mob reference from before the render still works.
    assert scene.timeline_manager is managers[0]
    assert scene.animation_manager is managers[1]
    assert scene.audio_manager is managers[2]
    assert square.is_spawned()
    assert not square.is_despawned()
    assert scene.camera.is_spawned()
    assert not scene.camera.is_despawned()


def test_reset_true_discards_the_authored_scene(monkeypatch, tmp_path):
    scene = SceneManager.instance().current_scene
    managers = (
        scene.timeline_manager,
        scene.animation_manager,
        scene.audio_manager,
    )
    Square(add_to_scene=True).spawn(animate=False)

    class Writer:
        def close(self):
            return None

    monkeypatch.setattr(scene, "render_audio_to_file", lambda *_a, **_k: None)
    monkeypatch.setattr(algan_utils, "get_file_writer", lambda *_a, **_k: Writer())
    monkeypatch.setattr(scene, "render_to_video", lambda *_a, **_k: None)

    algan.Scene.save_video(
        tmp_path / "discard.mp4",
        video_settings=VideoSettings((8, 8), 1, anti_alias_level=1),
        animate_fade_out=False,
        reset=True,
    )

    assert scene.timeline_manager is not managers[0]
    assert scene.animation_manager is not managers[1]
    assert scene.audio_manager is not managers[2]


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
    scene = SceneManager.instance().current_scene
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


def test_video_settings_are_immutable_validated_and_typo_safe():
    with pytest.raises(AlganConfigurationError, match="Did you mean 'resolution'"):
        PREVIEW.set(resoluton=(1, 1))
    with pytest.raises(AlganConfigurationError, match="positive"):
        PREVIEW.set(resolution=(0, 1))
    with pytest.raises(AlganConfigurationError, match="immutable"):
        PREVIEW.frames_per_second = 99

    changed = PREVIEW.set(frames_per_second=12)
    assert changed.frames_per_second == 12
    assert PREVIEW.frames_per_second != 12


def test_spawned_light_registers_once_and_add_light_is_chainable():
    scene = SceneManager.instance().current_scene
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

    @algan_utils.scene_function(name="main")
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
    for expected in ("Square", "Scene", "SETTINGS", "HD", "RED", "Sync", "rate_funcs"):
        assert expected in namespace


def test_root_star_exports_exclude_internal_helpers():
    """Generic tensor/plumbing helpers must not shadow user or stdlib names."""
    namespace = {}
    exec("from algan import *", namespace)
    for leaked in (
        "mean",
        "interpolate",
        "offset",
        "shuffle",
        "broadcast",
        "traverse",
        "squish",
        "implements",
        "pack_tensor",
        "cast_to_tensor",
        "get_image",
        "midpoint",
        "wiggle",
        "scene",
        "KERNEL_REGISTRY",
        "RENDERER_REGISTRY",
        "MobLayoutMixin",
        "profile_func",
        "concatenate_videos",
    ):
        assert leaked not in namespace, f"{leaked} leaked into the star namespace"
    # Still importable from their real home.
    from algan.utils.algan_utils import scene_function  # noqa: F401
    from algan.utils.tensor_utils import mean  # noqa: F401


def test_camera_validates_projection_and_clip_parameters():
    with pytest.raises(AlganConfigurationError, match="fov"):
        Camera(fov=180, add_to_scene=False)
    with pytest.raises(AlganConfigurationError, match="near"):
        Camera(near=2, far=1, add_to_scene=False)

    scene = SceneManager.instance().current_scene
    root_end = scene.animation_manager.context.timespan.original_end
    orthographic_camera = Camera(orthographic=True, add_to_scene=False)
    assert orthographic_camera.orthographic is True
    assert scene.animation_manager.context.timespan.original_end == root_end

    camera = Camera(add_to_scene=False)
    with pytest.raises(AlganConfigurationError, match="positive"):
        camera.set_distance_to_screen(0)
    assert camera.set_near_orthographic(1000) is camera
    assert camera.orthographic is True
    assert camera.set_fov(45) is camera
    assert camera.orthographic is False
    scene.set_video_settings(VideoSettings((20, 10), 1, anti_alias_level=1))
    assert camera.pixel_height == pytest.approx(0.2)
    scene.set_video_settings(VideoSettings((40, 20), 1, anti_alias_level=1))
    assert camera.pixel_height == pytest.approx(0.1)


def test_camera_clip_properties_survive_generic_material_parameter_names():
    algan.Cone(add_to_scene=False).set_material(algan.MeshDepthMaterial(near=2, far=12))

    camera = Camera(near=0.5, far=20, add_to_scene=False)

    assert camera.near == pytest.approx(0.5)
    assert camera.far == pytest.approx(20)


def test_static_off_scene_gets_one_frame_before_final_despawn(monkeypatch, tmp_path):
    scene = SceneManager.instance().current_scene
    with algan.Off():
        Square(add_to_scene=True).spawn(animate=False)

    observed = {}

    class Writer:
        def close(self):
            return None

    monkeypatch.setattr(scene, "render_audio_to_file", lambda *_a, **_k: None)
    monkeypatch.setattr(algan_utils, "get_file_writer", lambda *_a, **_k: Writer())

    def fake_render_to_video(*_args, **render_kwargs):
        observed["end"] = scene.animation_manager.context.timespan.original_end
        observed["background_override"] = render_kwargs.get("background_color")

    monkeypatch.setattr(scene, "render_to_video", fake_render_to_video)
    result = algan.Scene.save_video(
        tmp_path / "static.mp4",
        video_settings=VideoSettings((8, 8), 4, anti_alias_level=1),
        animate_fade_out=False,
    )

    assert observed["end"] >= 0.25
    assert observed["background_override"] is None
    assert result.status == "rendered"


def test_draw_border_then_fill_accepts_any_iterable_of_mobs():
    """Border-textured Mobs still animate when supplied through any iterable."""
    from algan.animations.manim_animations import draw_border_then_fill

    squares = [Square(add_to_scene=True).spawn(animate=False) for _ in range(3)]
    assert squares[0].border_color.shape[-1] == 5

    animated = draw_border_then_fill(squares)

    assert animated == squares
    # A generator is an iterable too, and must not be consumed twice.
    assert draw_border_then_fill(mob for mob in squares) == squares


def test_draw_border_then_fill_restores_the_original_style():
    """The temporary outline must not become the Mob's permanent style."""
    from algan.animations.manim_animations import draw_border_then_fill

    square = Square(
        color=algan.BLUE,
        border_color=algan.RED,
        border_width=0.25,
        add_to_scene=True,
    ).spawn(False)
    original_colors = [
        descendant.color.clone() for descendant in square.get_descendants()
    ]
    original_border_width = square.border_width.clone()

    draw_border_then_fill(square for _ in range(1))

    assert torch.allclose(square.border_width, original_border_width)
    assert all(
        torch.allclose(descendant.color, original)
        for descendant, original in zip(square.get_descendants(), original_colors)
    )


def test_draw_border_then_fill_can_reverse_iteration_order(monkeypatch):
    from algan.animations.manim_animations import draw_border_then_fill

    squares = [Square(add_to_scene=True).spawn(False) for _ in range(3)]
    drawn = []

    for square in squares:
        monkeypatch.setattr(
            square,
            "draw",
            lambda _t=1.0, square=square: drawn.append(square) or square,
        )

    assert draw_border_then_fill(squares, reverse=True) == list(reversed(squares))
    assert drawn == list(reversed(squares))


def test_text_write_materializes_manim_outline_and_fill_styles():
    """Colored Pango text traces white, then restores its stroke-free style."""
    text = algan.Text("A", color=algan.YELLOW, add_to_scene=True).spawn(False)
    glyph = text.character_mobs[0]

    text.write(run_time=2)
    text.scene.timeline_manager.set_state_to_times(
        torch.tensor([0.5, 1.5, 1.999], dtype=torch.get_default_dtype())
    )

    assert torch.allclose(
        glyph.texture_points.color[:, 0, -1],
        torch.tensor([0.0, 0.5, 0.999]),
        atol=1e-4,
    )
    assert torch.allclose(glyph.border_color[0, 0, :3], algan.WHITE[:3])
    assert torch.allclose(
        glyph.border_color[1, 0, :3],
        torch.tensor([1.0, 1.0, 0.5]),
        atol=1e-4,
    )
    assert torch.allclose(
        glyph.border_width[:, 0, 0],
        torch.tensor([1.0, 0.5, 0.001]),
        atol=1e-4,
    )


def test_draw_border_then_fill_tolerates_an_empty_iterable():
    from algan.animations.manim_animations import draw_border_then_fill

    assert draw_border_then_fill([]) == []


def test_text_write_is_the_glyph_wise_shorthand(monkeypatch):
    import algan.animations.manim_animations as manim_animations

    text = algan.Text("hi", add_to_scene=True)
    seen = {}

    def fake(
        mobs,
        border_width=1,
        run_time=None,
        lag_ratio=None,
        border_color=None,
        **kwargs,
    ):
        seen["mobs"] = list(mobs)
        seen["border_width"] = border_width
        seen["border_color"] = border_color
        return seen["mobs"]

    monkeypatch.setattr(manim_animations, "draw_border_then_fill", fake)

    assert text.write() is text
    assert len(seen["mobs"]) == len(text.character_mobs)
    assert torch.allclose(seen["border_color"], algan.WHITE)


# --- UX audit fixes -------------------------------------------------------


def _stub_render(monkeypatch, scene):
    """Render nothing, so a save_video call exercises only the authoring path."""

    class Writer:
        def close(self):
            return None

    monkeypatch.setattr(scene, "render_audio_to_file", lambda *_a, **_k: None)
    monkeypatch.setattr(algan_utils, "get_file_writer", lambda *_a, **_k: Writer())
    monkeypatch.setattr(scene, "render_to_video", lambda *_a, **_k: None)


def test_context_kwargs_on_a_method_point_at_the_context():
    """``mob.move(RIGHT, run_time=2)`` is the Manim reflex; say what to write."""
    square = Square().spawn()
    with pytest.raises(TypeError, match=r"with Seq\(run_time=2\)"):
        square.move(algan.RIGHT, run_time=2)
    with pytest.raises(TypeError, match=r"with Seq\(run_time=2\)"):
        square.set(color=algan.BLUE, run_time=2)
    # lag_ratio must suggest Lag, which takes it positionally.
    with pytest.raises(TypeError, match=r"with Lag\(0\.3\)"):
        square.set(lag_ratio=0.3)


def test_a_genuine_keyword_typo_still_raises():
    """The context-kwarg catch must not swallow real mistakes."""
    square = Square().spawn()
    with pytest.raises(TypeError, match="path_arc"):
        square.move(algan.RIGHT, path_arc=90)


def test_property_typo_suggests_the_real_name_and_lists_settable_ones():
    from algan.mobs.shapes_2d import Circle

    with pytest.raises(AttributeError, match=r"Did you mean 'color'\?"):
        Square().spawn().set(colour=algan.RED)
    # border_color is accepted by set(), so it must be advertised by the error.
    circle = Circle().spawn()
    with pytest.raises(AttributeError, match="border_color"):
        circle.set(bordercolour=algan.RED)
    circle.set(border_color=algan.PINK)


def test_unknown_setting_lists_the_valid_names():
    with pytest.raises(AlganConfigurationError, match="frames_per_second"):
        VideoSettings((8, 8), 4).set(fps=60)


def test_vector_arguments_reject_scalars():
    """A scalar broadcasts to the (1, 1, 1) diagonal instead of raising."""
    square = Square().spawn()
    for call in (
        lambda: square.move(1),
        lambda: square.move_to(1),
        lambda: square.rotate(90, 1),
        lambda: square.rotate(algan.OUT, 90),
        lambda: square.move_to_edge(1),
    ):
        with pytest.raises(AlganConfigurationError):
            call()
    # ... while real vectors keep working.
    square.move(algan.RIGHT)
    square.move([0, 1, 0])
    square.rotate(90, algan.OUT)


def test_empty_output_path_is_rejected():
    with pytest.raises(AlganConfigurationError, match="empty string"):
        algan_utils._resolve_output_destination("", ".mp4")


def test_never_spawned_mob_warns(monkeypatch, tmp_path):
    from algan.errors import NeverSpawnedMobWarning
    from algan.mobs.shapes_2d import Circle

    scene = SceneManager.instance().current_scene
    Square().spawn()
    Circle()  # forgotten
    _stub_render(monkeypatch, scene)
    with pytest.warns(NeverSpawnedMobWarning, match="Circle"):
        algan.Scene.save_video(
            tmp_path / "forgot.mp4",
            video_settings=VideoSettings((8, 8), 4, anti_alias_level=1),
        )


def test_add_to_scene_false_is_the_only_way_to_mark_reference_geometry(
    monkeypatch, tmp_path, recwarn
):
    """Reference geometry is excluded by construction, not by a special case.

    ``add_to_scene=False`` means "never intended to be shown", so such a Mob
    never enters ``scene.actors`` and the warning cannot see it. A ``become``
    target built without the flag is indistinguishable from a forgotten spawn,
    and is meant to be reported.
    """
    from algan.errors import NeverSpawnedMobWarning
    from algan.mobs.shapes_2d import Circle

    scene = SceneManager.instance().current_scene
    square = Square().spawn()
    square.become(Circle(add_to_scene=False))
    _stub_render(monkeypatch, scene)
    algan.Scene.save_video(
        tmp_path / "become.mp4",
        video_settings=VideoSettings((8, 8), 4, anti_alias_level=1),
    )
    assert not [w for w in recwarn if issubclass(w.category, NeverSpawnedMobWarning)]


def test_unflagged_become_target_is_reported(monkeypatch, tmp_path):
    """Without the flag it is just an unspawned Mob, and says so."""
    from algan.errors import NeverSpawnedMobWarning
    from algan.mobs.shapes_2d import Circle

    scene = SceneManager.instance().current_scene
    Square().spawn().become(Circle())
    _stub_render(monkeypatch, scene)
    with pytest.warns(NeverSpawnedMobWarning, match="add_to_scene=False"):
        algan.Scene.save_video(
            tmp_path / "become_unflagged.mp4",
            video_settings=VideoSettings((8, 8), 4, anti_alias_level=1),
        )


def test_angle_unit_constants_are_exported_with_algan_convention():
    """Algan's DEGREES is 1, the reciprocal of Manim's -- guard the value."""
    assert {"DEGREES", "RADIANS"} <= set(algan.__all__)
    assert algan.DEGREES == 1.0
    assert pytest.approx(180.0) == algan.PI * algan.RADIANS


def test_internal_helpers_are_importable_but_not_star_exported():
    """Trimmed from `from algan import *`, still public at their real path."""
    from algan.geometry.geometry import project_onto_basis  # noqa: F401
    from algan.utils.animation_utils import animate_lagged_by_location  # noqa: F401
    from algan.utils.mob_utils import batch_mobs  # noqa: F401

    for name in (
        "project_onto_basis",
        "animate_lagged_by_location",
        "batch_mobs",
        "get_orthonormal_vector",
        "get_rotation_between_bases",
    ):
        assert name not in algan.__all__


@pytest.mark.parametrize(
    ("feature", "expected"),
    [
        ("environment_map", "environment maps"),
        ("refraction", "refractive materials"),
        ("fragment_pipeline", "custom fragment-shader pipelines"),
        ("extended_light", "extended lights"),
    ],
)
def test_an_authored_scene_reaches_the_monte_carlo_capability_check(
    feature, expected, tmp_path
):
    """The preflight is only worth having if real authoring actually trips it.

    ``_validate_render_capabilities`` is unit-tested above against hand-built
    ``merged`` dicts, which proves the message but not the wiring: that
    ``set_material(MeshPhysicalMaterial(transmission=...))` really sets
    ``has_refractive``, that ``set_environment_map`` reaches the check at all,
    and so on. Authoring each feature the way a user would and asserting the
    render refuses is what closes that gap -- and it needs no GPU, because the
    check runs on host metadata before any arena reservation or kernel
    compilation.
    """
    from algan.constants.color import BLUE, WHITE
    from algan.constants.spatial import OUT, UP
    from algan.mobs.shapes_3d import Sphere
    from algan.rendering.shaders.fragment_shaders import STAGE_STANDARD, cosine_color
    from algan.rendering.shaders.materials import MeshPhysicalMaterial
    from algan.settings.video_settings import SMOKE_TEST

    rt_settings.set_unsupported_feature_policy("error")
    scene = SceneManager.instance().current_scene
    scene.set_video_settings(SMOKE_TEST)

    with algan.SETTINGS.raytracing.override(samples_per_pixel=4):
        if feature == "environment_map":
            scene.set_environment_map(torch.rand((4, 8, 3)))
            Sphere(radius=0.6, color=BLUE).spawn()
        elif feature == "refraction":
            sphere = Sphere(radius=0.6, color=BLUE)
            sphere.set_material(
                MeshPhysicalMaterial(transmission=1.0, ior=1.5, thickness=0.5)
            )
            sphere.spawn()
        elif feature == "fragment_pipeline":
            sphere = Sphere(radius=0.6, color=BLUE)
            sphere.set_fragment_shader([cosine_color, STAGE_STANDARD])
            sphere.spawn()
        elif feature == "extended_light":
            with algan.Off():
                scene.clear_light_sources()
                RectAreaLight(
                    location=UP * 3 + OUT * 3, color=WHITE, intensity=3
                ).spawn()
            Sphere(radius=0.6, color=BLUE).spawn()

        scene.wait(0.2)

        with pytest.raises(UnsupportedFeatureError, match=expected):
            scene.save_video(tmp_path / f"spp_{feature}", overwrite=True)
