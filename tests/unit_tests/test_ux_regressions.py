from __future__ import annotations

import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import algan
from algan import render_loop
from algan.animation_timeline.animation_contexts import Sync
from algan.constants.math import PI
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
from algan.settings.video_settings import PREVIEW, SMOKE_TEST, VideoSettings
from algan.utils import algan_utils

# Marked per test rather than for the module, unlike the other fast-suite
# files. Those are each about one mechanism, so a new test in them is the same
# kind of test. This one is a catch-all for whatever last bit the authoring
# surface -- ``save_video`` / ``save_frame`` and what they leave the Scene in,
# the animation contexts, ``Group``, the star exports, the errors users hit --
# and the kinds of test that land here differ enough that a module-level mark
# would enrol new ones by accident. That is the thing this suite exists to
# stop, so each test says for itself whether a change elsewhere can break it.


@pytest.fixture(autouse=True)
def reset_global_authoring_state():
    SceneManager.reset()
    previous_policy = rt_settings.unsupported_feature_policy
    yield
    rt_settings.unsupported_feature_policy = previous_policy
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


def test_match_durations_tolerates_zero_duration_children():
    with Sync(match_durations=True), algan.Off():
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
    temporary = VideoSettings((17, 13), 2, supersampling=1)

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
    scene.set_video_settings(VideoSettings((17, 13), 10, supersampling=1))
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


def test_save_frame_logs_completion_message(monkeypatch, tmp_path, caplog):
    import logging
    import re

    algan_logger = logging.getLogger("algan")
    algan_logger.addHandler(caplog.handler)
    try:
        scene = SceneManager.instance().current_scene
        scene.set_video_settings(VideoSettings((17, 13), 10, supersampling=1))
        _stub_out_frame_writing(monkeypatch, scene)

        scene.save_frame(tmp_path / "single_still", at=0.0)

        pattern = re.compile(r"^Finished rendering single_still\.png in \d+\.\d+ s$")
        assert any(
            pattern.match(record.message)
            for record in caplog.records
            if record.levelno == logging.INFO
        )

        caplog.clear()
        scene.save_frame(tmp_path / "single_still", at=0.0, overwrite=False)

        assert not any(
            "Finished rendering single_still.png" in record.message
            for record in caplog.records
        )

        caplog.clear()
        scene.save_frame(tmp_path / "multi_still", at=[0.0, 1.0])

        assert any(
            re.match(
                r"^Finished rendering multi_still_0\.0\.png in \d+\.\d+ s$",
                record.message,
            )
            for record in caplog.records
            if record.levelno == logging.INFO
        )
        assert any(
            re.match(
                r"^Finished rendering multi_still_1\.0\.png in \d+\.\d+ s$",
                record.message,
            )
            for record in caplog.records
            if record.levelno == logging.INFO
        )
    finally:
        algan_logger.removeHandler(caplog.handler)


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
    with algan.Seq(duration=8):
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
    with algan.Seq(duration=8):
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
        with algan.Seq(duration=5):
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
    with algan.Seq(duration=8):
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
        video_settings=VideoSettings((8, 8), 1, supersampling=1),
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
            video_settings=VideoSettings((8, 8), 1, supersampling=1),
            background=algan.TRANSPARENT,
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
    monkeypatch.setattr(scene, "save_audio", lambda *_a, **_k: None)
    monkeypatch.setattr(
        algan_utils,
        "get_file_writer",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("writer failed")),
    )

    with pytest.raises(RuntimeError, match="writer failed"):
        algan.Scene.save_video(
            tmp_path / "failure.mp4",
            video_settings=VideoSettings((8, 8), 1, supersampling=1),
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

    monkeypatch.setattr(scene, "save_audio", lambda *_a, **_k: None)
    monkeypatch.setattr(algan_utils, "get_file_writer", lambda *_a, **_k: Writer())
    monkeypatch.setattr(scene, "render_to_video", lambda *_a, **_k: None)

    result = algan.Scene.save_video(
        tmp_path / "keep.mp4",
        video_settings=VideoSettings((8, 8), 1, supersampling=1),
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

    monkeypatch.setattr(scene, "save_audio", lambda *_a, **_k: None)
    monkeypatch.setattr(algan_utils, "get_file_writer", lambda *_a, **_k: Writer())
    monkeypatch.setattr(scene, "render_to_video", lambda *_a, **_k: None)

    algan.Scene.save_video(
        tmp_path / "discard.mp4",
        video_settings=VideoSettings((8, 8), 1, supersampling=1),
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

    assert not hasattr(group, "mobs")  # `children` is the one spelling
    group[0] = replacement

    assert group.children == [replacement, second]
    assert group not in first.parents
    assert group in replacement.parents


@pytest.mark.fast
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


@pytest.mark.fast
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


@pytest.mark.fast
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
    scene.add_light(light)

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


def test_path_tracer_unsupported_features_fail_preflight():
    """What the path tracer still refuses: custom scatter overrides
    (user-defined ray continuation has no sampling density for stochastic
    transport to weight). Environment maps graduated to full support --
    integrated for real through CDF next-event estimation and escaping rays
    -- so they must NOT appear in the refusal.
    """
    rt_settings.set_unsupported_feature_policy("error")
    merged = {"has_user_pipeline": True, "has_custom_scatter": True}

    with pytest.raises(UnsupportedFeatureError) as exc_info:
        _validate_render_capabilities(
            4,
            torch.zeros((1, 1, 3)),
            merged,
            [],
        )
    message = str(exc_info.value)
    assert "custom scatter overrides" in message
    assert "environment maps" not in message


def test_path_tracer_supports_the_features_the_monte_carlo_kernel_refused():
    """Refraction, custom fragment pipelines, extended lights and
    environment maps were Monte Carlo rejections; the path tracer honours
    all four, so they are *requested* without being *unsupported*.
    """
    rt_settings.set_unsupported_feature_policy("error")
    extended_light = SimpleNamespace(_render_aux=object())

    plan = _validate_render_capabilities(
        4,
        torch.zeros((1, 1, 3)),
        {
            "has_refractive": True,
            "has_user_pipeline": True,
            "has_custom_scatter": False,
        },
        [extended_light],
    )
    assert plan.backend == "path_tracer"
    assert plan.is_supported
    assert "refractive materials" in plan.requested_features
    assert "custom fragment-shader pipelines" in plan.requested_features
    assert "extended lights" in plan.requested_features
    assert "environment maps" in plan.requested_features


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


def test_scene_decorator_prevents_helpers_from_being_discovered(monkeypatch):
    import types

    module_name = "_algan_scene_registry_test"
    module = types.ModuleType(module_name)
    calls = []

    def helper():
        calls.append("helper")

    @algan_utils.algan_scene(name="main")
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
    for expected in ("Square", "Scene", "SETTINGS", "HD", "RED", "Sync", "easings"):
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
    from algan.utils.algan_utils import algan_scene  # noqa: F401
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
    scene.set_video_settings(VideoSettings((20, 10), 1, supersampling=1))
    assert camera.pixel_height == pytest.approx(0.2)
    scene.set_video_settings(VideoSettings((40, 20), 1, supersampling=1))
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

    monkeypatch.setattr(scene, "save_audio", lambda *_a, **_k: None)
    monkeypatch.setattr(algan_utils, "get_file_writer", lambda *_a, **_k: Writer())

    def fake_render_to_video(*_args, **render_kwargs):
        observed["end"] = scene.animation_manager.context.timespan.original_end
        observed["background_override"] = render_kwargs.get("background")

    monkeypatch.setattr(scene, "render_to_video", fake_render_to_video)
    result = algan.Scene.save_video(
        tmp_path / "static.mp4",
        video_settings=VideoSettings((8, 8), 4, supersampling=1),
        animate_fade_out=False,
    )

    assert observed["end"] >= 0.25
    assert observed["background_override"] is None
    assert result.status == "rendered"


@pytest.mark.fast
def test_draw_border_then_fill_accepts_any_iterable_of_mobs():
    """Border-textured Mobs still animate when supplied through any iterable."""
    from algan.animations.manim_animations import DrawBorderThenFill

    squares = [Square(add_to_scene=True).spawn(animate=False) for _ in range(3)]
    assert squares[0].stroke_color.shape[-1] == 5

    animated = DrawBorderThenFill(squares)

    assert animated == squares
    # A generator is an iterable too, and must not be consumed twice.
    assert DrawBorderThenFill(mob for mob in squares) == squares


@pytest.mark.fast
def test_draw_border_then_fill_restores_the_original_style():
    """The temporary outline must not become the Mob's permanent style."""
    from algan.animations.manim_animations import DrawBorderThenFill

    square = Square(
        color=algan.BLUE,
        stroke_color=algan.RED,
        stroke_width=0.25,
        add_to_scene=True,
    ).spawn(False)
    original_colors = [
        descendant.color.clone() for descendant in square.get_descendants()
    ]
    original_stroke_width = square.stroke_width.clone()

    DrawBorderThenFill(square for _ in range(1))

    assert torch.allclose(square.stroke_width, original_stroke_width)
    assert all(
        torch.allclose(descendant.color, original)
        for descendant, original in zip(square.get_descendants(), original_colors)
    )


def test_draw_border_then_fill_can_reverse_iteration_order(monkeypatch):
    from algan.animations.manim_animations import DrawBorderThenFill

    squares = [Square(add_to_scene=True).spawn(False) for _ in range(3)]
    drawn = []

    for square in squares:
        monkeypatch.setattr(
            square,
            "draw",
            lambda _t=1.0, square=square: drawn.append(square) or square,
        )

    assert DrawBorderThenFill(squares, reverse=True) == list(reversed(squares))
    assert drawn == list(reversed(squares))


@pytest.mark.fast
def test_text_write_materializes_manim_outline_and_fill_styles():
    """Colored Pango text traces white, then restores its stroke-free style."""
    text = algan.Text("A", color=algan.YELLOW, add_to_scene=True).spawn(False)
    glyph = text.character_mobs[0]

    text.write(duration=2)
    text.scene.timeline_manager.set_state_to_times(
        torch.tensor([0.5, 1.5, 1.999], dtype=torch.get_default_dtype())
    )

    assert torch.allclose(
        glyph.texture_points.color[:, 0, -1],
        torch.tensor([0.0, 0.5, 0.999]),
        atol=1e-4,
    )
    assert torch.allclose(glyph.stroke_color[0, 0, :3], algan.WHITE[:3])
    assert torch.allclose(
        glyph.stroke_color[1, 0, :3],
        torch.tensor([1.0, 1.0, 0.5]),
        atol=1e-4,
    )
    assert torch.allclose(
        glyph.stroke_width[:, 0, 0],
        torch.tensor([1.0, 0.5, 0.001]),
        atol=1e-4,
    )


def test_draw_border_then_fill_tolerates_an_empty_iterable():
    from algan.animations.manim_animations import DrawBorderThenFill

    assert DrawBorderThenFill([]) == []


def test_text_write_is_the_glyph_wise_shorthand(monkeypatch):
    import algan.animations.manim_animations as manim_animations

    text = algan.Text("hi", add_to_scene=True)
    seen = {}

    def fake(
        mobs,
        stroke_width=1,
        duration=None,
        lag_ratio=None,
        stroke_color=None,
        **kwargs,
    ):
        seen["mobs"] = list(mobs)
        seen["stroke_width"] = stroke_width
        seen["stroke_color"] = stroke_color
        return seen["mobs"]

    monkeypatch.setattr(manim_animations, "DrawBorderThenFill", fake)

    assert text.write() is text
    assert len(seen["mobs"]) == len(text.character_mobs)
    assert torch.allclose(seen["stroke_color"], algan.WHITE)


# --- UX audit fixes -------------------------------------------------------


def _stub_render(monkeypatch, scene):
    """Render nothing, so a save_video call exercises only the authoring path."""

    class Writer:
        def close(self):
            return None

    monkeypatch.setattr(scene, "save_audio", lambda *_a, **_k: None)
    monkeypatch.setattr(algan_utils, "get_file_writer", lambda *_a, **_k: Writer())
    monkeypatch.setattr(scene, "render_to_video", lambda *_a, **_k: None)


def test_context_kwargs_on_a_method_point_at_the_context():
    """``mob.move(RIGHT, duration=2)`` is the Manim reflex; say what to write."""
    square = Square().spawn()
    with pytest.raises(TypeError, match=r"with Seq\(duration=2\)"):
        square.move(algan.RIGHT, duration=2)
    with pytest.raises(TypeError, match=r"with Seq\(duration=2\)"):
        square.set(color=algan.BLUE, duration=2)
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
    # stroke_color is accepted by set(), so it must be advertised by the error.
    circle = Circle().spawn()
    with pytest.raises(AttributeError, match="stroke_color"):
        circle.set(bordercolour=algan.RED)
    circle.set(stroke_color=algan.PINK)


@pytest.mark.fast
def test_by_name_attribute_api_handles_derived_properties():
    """``scale_coefficient`` is the row norms of ``basis``, not a buffer.

    The by-name API addresses timeline rows, so a derived property has nothing
    for it to read. Both halves used to get that wrong in opposite directions:
    ``map_animated_attribute`` raised a bare ``AttributeError`` with no message
    at all, and ``set_animated_attribute`` allocated a ``scale_coefficient``
    buffer nothing reads and animated nothing.
    """
    from algan.mobs.shapes_2d import Circle

    scene = SceneManager.instance().current_scene
    row = Group([Square().move(algan.LEFT * 2), Circle()])
    row.spawn()

    with pytest.raises(AttributeError, match="derived property"):
        row.map_animated_attribute("scale_coefficient", lambda s: s * 0.25)
    # The message has to say what to do instead, not just what failed.
    with pytest.raises(AttributeError, match=r"Mob\.scale"):
        row.map_animated_attribute("scale_coefficient", lambda s: s * 0.25)

    by_name = Group([Square().move(algan.LEFT * 2), Circle()])
    by_name.spawn()
    by_name.set_animated_attribute("scale_coefficient", torch.tensor([0.5, 0.5, 0.5]))
    assigned = Group([Square().move(algan.LEFT * 2), Circle()])
    assigned.spawn()
    assigned.scale_coefficient = torch.tensor([0.5, 0.5, 0.5])

    assert "scale_coefficient" not in scene.timeline_manager.attr_to_timeline
    for a, b in ((by_name, assigned), (by_name.children[0], assigned.children[0])):
        assert torch.allclose(a.scale_coefficient, b.scale_coefficient)
        assert torch.allclose(a.location, b.location)
    assert float(by_name.scale_coefficient.flatten()[0]) == pytest.approx(0.5)

    # An attribute that really is timeline-backed still goes the normal way.
    row.map_animated_attribute("opacity", lambda o: o * 0.5)
    assert float(row.opacity.flatten()[0]) == pytest.approx(0.5, abs=1e-4)


@pytest.mark.fast
def test_by_name_basis_write_carries_the_subtree():
    """``basis`` has rows, and writing them alone is still half the operation.

    A shape's geometry is its control points' *locations*, so a rotation has to
    move them too -- which the property setter does and a flat per-row write
    does not. Measured before the fix: a 45 degree rotation through
    ``set_animated_attribute`` left a Square's corners byte-identical to an
    untouched one, while ``mob.basis =`` moved them.
    """
    from algan.animation_timeline.animation_contexts import Off

    half = torch.tensor(45.0 * 3.141592653589793 / 180.0)
    cos, sin = torch.cos(half), torch.sin(half)
    rotation = torch.tensor(
        [[cos, sin, 0.0], [-sin, cos, 0.0], [0.0, 0.0, 1.0]]
    ).reshape(1, 1, 9)

    def corners(mob):
        return mob.get_render_primitives().corners.reshape(-1, 3)

    untouched = Square().spawn()
    by_name = Square().spawn()
    assigned = Square().spawn()
    with Off():
        by_name.set_animated_attribute("basis", rotation)
        assigned.basis = rotation

    assert torch.allclose(corners(by_name), corners(assigned), atol=1e-5)
    assert not torch.allclose(corners(by_name), corners(untouched), atol=1e-5)

    # Mapping it row-wise cannot be made to mean anything, so it is refused
    # rather than half-applied to the texture frame alone.
    with pytest.raises(AttributeError, match="hierarchical transform"):
        untouched.map_animated_attribute("basis", lambda b: b * 0.25)
    with pytest.raises(AttributeError, match=r"Mob\.rotate"):
        untouched.map_animated_attribute("basis", lambda b: b * 0.25)


def test_unknown_setting_lists_the_valid_names():
    with pytest.raises(AlganConfigurationError, match="frames_per_second"):
        VideoSettings((8, 8), 4).set(frame_rate=60)


@pytest.mark.fast
def test_vector_arguments_reject_scalars():
    """A scalar broadcasts to the (1, 1, 1) diagonal instead of raising."""
    square = Square().spawn()
    for call in (
        lambda: square.move(1),
        lambda: square.move_to(1),
        lambda: square.rotate(90, 1),
        lambda: square.rotate(algan.OUT, 90),
        lambda: square.move_to_screen_edge(1),
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
            video_settings=VideoSettings((8, 8), 4, supersampling=1),
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
        video_settings=VideoSettings((8, 8), 4, supersampling=1),
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
            video_settings=VideoSettings((8, 8), 4, supersampling=1),
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
    from algan.utils.mob_utils import (  # noqa: F401
        batch_mobs,
        pack_animatable_rows,
        pack_member_rows,
    )

    for name in (
        "project_onto_basis",
        "animate_lagged_by_location",
        "batch_mobs",
        "pack_animatable_rows",
        "pack_member_rows",
        "get_orthonormal_vector",
        "get_rotation_between_bases",
    ):
        assert name not in algan.__all__


@pytest.mark.parametrize(
    ("feature", "expected"),
    [
        ("custom_scatter", "custom scatter overrides"),
    ],
)
def test_an_authored_scene_reaches_the_path_tracer_capability_check(
    feature, expected, tmp_path
):
    """The preflight is only worth having if real authoring actually trips it.

    ``_validate_render_capabilities`` is unit-tested above against hand-built
    ``merged`` dicts, which proves the message but not the wiring: that a
    fragment pipeline carrying a custom scatter registers as one. Authoring
    the still-unsupported feature the way a user would and asserting the
    render refuses is what closes that gap -- and it needs no GPU, because
    the check runs on host metadata before any arena reservation or kernel
    compilation.
    """
    import taichi as ti

    from algan.constants.color import BLUE
    from algan.mobs.shapes_3d import Sphere
    from algan.rendering.shaders.fragment_shaders import (
        STAGE_STANDARD,
        FragmentStage,
    )
    from algan.settings.video_settings import SMOKE_TEST

    rt_settings.set_unsupported_feature_policy("error")
    scene = SceneManager.instance().current_scene
    scene.set_video_settings(SMOKE_TEST)

    with algan.SETTINGS.raytracing.override(samples_per_pixel=4):
        if feature == "custom_scatter":

            @ti.func
            def _test_scatter_noop():
                pass

            sphere = Sphere(radius=0.6, color=BLUE)
            sphere.set_fragment_shader(
                [
                    FragmentStage(
                        STAGE_STANDARD.ti_func,
                        STAGE_STANDARD.param_specs,
                        scatter=_test_scatter_noop,
                    )
                ]
            )
            sphere.spawn()

        scene.wait(0.2)

        with pytest.raises(UnsupportedFeatureError, match=expected):
            scene.save_video(tmp_path / f"spp_{feature}", overwrite=True)


@pytest.mark.parametrize(
    "feature",
    ["refraction", "fragment_pipeline", "extended_light", "environment_map"],
)
def test_lifted_path_tracer_features_render(feature, tmp_path):
    """Features the path tracer gained (refraction, fragment pipelines,
    extended lights and environment maps were Monte Carlo rejections before
    it) pass preflight and render one small frame end-to-end.
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

    # denoise off: this exercises the estimator's feature coverage, and CI
    # must not depend on the denoiser weights being downloadable.
    with algan.SETTINGS.raytracing.override(samples_per_pixel=4, denoise=False):
        if feature == "refraction":
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
                scene.clear_lights()
                RectAreaLight(
                    location=UP * 3 + OUT * 3, color=WHITE, intensity=3
                ).spawn()
            Sphere(radius=0.6, color=BLUE).spawn()
        elif feature == "environment_map":
            scene.set_environment_map(torch.rand((4, 8, 3)))
            Sphere(radius=0.6, color=BLUE).spawn()

        result = scene.save_frame(tmp_path / f"pt_{feature}.png", overwrite=True)
        assert result.render_plan.backend == "path_tracer"
        assert not result.render_plan.unsupported_features


def test_arrow3d_endpoints_follow_the_arrow():
    """``get_start`` / ``get_end`` report where the arrow IS, not where it was.

    They read two ``opacity=0`` marker Mobs, and those used to be built loose:
    registered into whichever Scene happened to be active rather than the
    arrow's, and attached to nothing -- so a moved or rotated arrow still
    reported its construction endpoints, and ``get_vector`` still pointed the
    way it was first built.
    """
    with algan.Scene() as scene, algan.Off():
        arrow = algan.Arrow3D(
            start=algan.ORIGIN, end=algan.RIGHT * 1.1, shaft_radius=0.05
        )
        arrow.spawn()

        assert arrow.start_point.scene is scene
        assert arrow.end_point.scene is scene

        arrow.move(algan.UP * 2 + algan.RIGHT * 0.5)
        assert torch.allclose(
            arrow.get_start().reshape(-1),
            torch.tensor((0.5, 2.0, 0.0)),
            atol=1e-5,
        )
        assert torch.allclose(
            arrow.get_end().reshape(-1),
            torch.tensor((1.6, 2.0, 0.0)),
            atol=1e-5,
        )

        arrow.rotate(90, algan.OUT, about=arrow.get_start())
        assert torch.allclose(
            arrow.get_vector().reshape(-1),
            torch.tensor((0.0, 1.1, 0.0)),
            atol=1e-4,
        )


@pytest.mark.fast
def test_an_animation_context_object_refuses_a_second_with_block():
    """One context object, one ``with`` block -- and it says so.

    Entering the same object twice used to be accepted, and it corrupted the
    process: ``__enter__`` keeps its ``ContextVar`` reset token on the instance,
    so the second entry overwrote the first's and nothing ever undid it. From
    then on every ``Sync`` and ``Lag`` in the process -- in any Scene -- resolved
    against a dead Scene's AnimationManager and silently played its animations
    in sequence instead of together. Nesting the object in itself also made
    ``prev_context`` point at itself, so the stack could never unwind.
    """
    from algan.animation_timeline import animation_contexts
    from algan.errors import ContextReuseError

    with algan.Scene():
        square = Square().spawn()

        nested = Sync()
        # The nesting is the bug under test; ruff's "combine these" advice
        # would remove exactly what is being reproduced.
        with pytest.raises(ContextReuseError, match="already being used"):  # noqa: PT012, SIM117
            with nested:
                with nested:
                    square.move(algan.RIGHT)

        sequential = Sync()
        with sequential:
            square.move(algan.RIGHT)
        with pytest.raises(ContextReuseError, match="already been used"):  # noqa: SIM117
            with sequential:
                square.move(algan.UP)

        # Distinct objects nest as they always did, and nothing leaked.
        with Sync():  # noqa: SIM117 -- nesting distinct contexts is the point
            with Sync():
                square.move(algan.UP)

    assert animation_contexts._ANIMATION_MANAGER_OVERRIDE.get(None) is None


@pytest.mark.fast
def test_sync_still_overlaps_after_a_rejected_context_reuse():
    """The rejection leaves timing intact -- the leak's symptom is gone.

    This is the observable half of the bug above: the corruption showed up as
    ``Sync`` behaving like ``Seq``, which nothing in the failing script pointed
    at.
    """
    from algan.errors import ContextReuseError

    def sync_end_time():
        with algan.Scene() as scene:
            square = Square().spawn(animate=False)
            with Sync() as context:
                square.move(algan.RIGHT * 0.1)
                square.move(algan.RIGHT * 0.1)
                square.move(algan.RIGHT * 0.1)
            return float(context.timespan.end) - float(context.timespan.start)

    before = sync_end_time()

    with algan.Scene():
        square = Square().spawn()
        reused = Sync()
        with pytest.raises(ContextReuseError):  # noqa: PT012, SIM117
            with reused:
                with reused:
                    square.move(algan.RIGHT)

    assert sync_end_time() == pytest.approx(before)


@pytest.mark.fast
def test_the_scenes_own_video_settings_reach_both_render_calls(tmp_path):
    """``save_video`` used to ignore them while ``save_frame`` honoured them.

    ``Scene(video_settings=...)`` and ``set_video_settings`` describe the
    Scene, so a render that names no settings of its own should use them. The
    video path resolved ``None`` straight to ``SETTINGS.video`` instead, so the
    same Scene wrote a 32x32 still and an 864x486 video, and
    ``set_video_settings`` -- whose whole job is to set the frame rate and
    resolution -- changed nothing about the video.
    """
    import cv2

    tiny = VideoSettings((32, 32), 2, supersampling=1)

    with algan.Scene(video_settings=tiny):
        Square(color=algan.BLUE).spawn()
        still = algan.Scene.save_frame(str(tmp_path / "still.png"))
        clip = algan.Scene.save_video(str(tmp_path / "clip.mp4"))

    image = cv2.imread(str(still.output_path), cv2.IMREAD_UNCHANGED)
    assert (image.shape[1], image.shape[0]) == (32, 32)

    capture = cv2.VideoCapture(str(clip.output_path))
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture.get(cv2.CAP_PROP_FPS)
    finally:
        capture.release()
    assert (width, height) == (32, 32)
    assert fps == pytest.approx(2)


@pytest.mark.fast
def test_a_later_settings_change_still_reaches_a_scene_that_chose_nothing(
    tmp_path,
):
    """The half of the old behaviour that was right, kept.

    A Scene that was never given settings holds a snapshot of
    ``SETTINGS.video`` from when it was built -- which is before the first line
    of most scripts, since the first Mob builds the default Scene. Preferring
    that snapshot would quietly break ``SETTINGS.video.set(...)``.
    """
    import cv2

    restore = algan.SETTINGS.video.as_preset()
    try:
        with algan.Scene():
            Square(color=algan.BLUE).spawn()
            algan.SETTINGS.video.set(VideoSettings((64, 48), 3, supersampling=1))
            clip = algan.Scene.save_video(str(tmp_path / "clip.mp4"))
    finally:
        algan.SETTINGS.video.set(restore)

    capture = cv2.VideoCapture(str(clip.output_path))
    try:
        size = (
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        capture.release()
    assert size == (64, 48)


@pytest.mark.fast
@pytest.mark.parametrize(
    "spelling",
    ["#FF0000", "red", 0xFF0000, (1.0, 0.0, 0.0), [1.0, 0.0, 0.0]],
    ids=["hex-string", "css-name", "hex-int", "rgb-tuple", "rgb-list"],
)
@pytest.mark.parametrize(
    "build",
    [
        lambda color: algan.Square(color=color),
        lambda color: algan.Sphere(color=color),
        lambda color: algan.Line(algan.LEFT, algan.RIGHT, color=color),
        lambda color: algan.Text("x", color=color),
        lambda color: algan.Arc(color=color),
    ],
    ids=["Square", "Sphere", "Line", "Text", "Arc"],
)
def test_every_colour_spelling_reaches_every_mob(build, spelling):
    """One parser for colour, wherever a colour is written.

    ``Color("#ff0000")`` always worked and ``Square(color="#ff0000")`` did not:
    it raised ``AttributeError: 'str' object has no attribute 'reshape'`` from
    inside the timeline. Materials had accepted all of these spellings from the
    start -- the shipped presets are written ``MeshStandardMaterial(color=
    0x8B5A2B)`` -- so the same literal was a colour in one place and an
    AttributeError in the other.
    """
    with algan.Scene():
        mob = build(spelling)
        red = mob.color.reshape(-1, mob.color.shape[-1])[0]

    assert float(red[0]) == pytest.approx(1.0)
    assert float(red[1]) == pytest.approx(0.0)
    assert float(red[2]) == pytest.approx(0.0)


@pytest.mark.fast
def test_an_unparseable_colour_says_so():
    from algan.errors import InvalidColorError

    with algan.Scene():
        with pytest.raises(InvalidColorError, match="Invalid color string"):
            algan.Square(color="octarine")
        with pytest.raises(InvalidColorError, match="Invalid color value"):
            algan.Square(color=True)


@pytest.mark.fast
def test_spawning_a_despawned_mob_warns_instead_of_doing_nothing():
    """It cannot work, and the silence left a blank video and no clue why.

    ``spawn`` returned early because ``is_spawned()`` stays True after a
    despawn, so the call looked like it had worked. The advice is the one
    ``despawn``'s docstring already gave, said where it is needed.
    """
    from algan.errors import DespawnedMobWarning

    with algan.Scene():
        square = Square().spawn()
        square.despawn()
        with pytest.warns(DespawnedMobWarning, match="cannot be brought back"):
            square.spawn()

        # Spawning an already-spawned (not despawned) Mob is still a quiet
        # no-op -- that one is documented and harmless.
        other = Square().spawn()
        with warnings.catch_warnings():
            warnings.simplefilter("error", DespawnedMobWarning)
            other.spawn()


@pytest.mark.fast
def test_set_material_rejects_a_non_material():
    """``set_material(GOLD)`` is a natural mistake: CHROME and COPPER are
    materials while GOLD is a colour. It used to answer with an AttributeError
    about ``shader``, which names nothing the caller wrote.
    """
    with algan.Scene():
        for value in (algan.GOLD, None, "gold"):
            with pytest.raises(AlganConfigurationError, match="expects a Material"):
                algan.Sphere().set_material(value)

        assert algan.Sphere().set_material(algan.GLASS) is not None


def test_a_surface_with_no_extent_renders_nothing_rather_than_failing(tmp_path):
    """A zero-triangle tessellation used to fail the whole render.

    ``Sphere(radius=0)`` -- or a radius a calculation drove to 1e-9 -- built an
    empty primitive that reached ``broadcast_all`` against one row of colour
    and raised a tensor-shape error naming neither the Mob nor the radius.
    There is nothing to draw, so the actor now contributes no geometry.
    """
    import cv2

    def brightest(build):
        with algan.Scene(video_settings=SMOKE_TEST):
            build().spawn()
            result = algan.Scene.save_frame(str(tmp_path / "degenerate.png"))
        return cv2.imread(str(result.output_path), cv2.IMREAD_UNCHANGED).max()

    assert brightest(lambda: algan.Sphere(radius=0, color=algan.BLUE)) == 0
    assert brightest(lambda: algan.Sphere(radius=1e-9, color=algan.BLUE)) == 0
    assert brightest(lambda: algan.Cylinder(radius=0, color=algan.BLUE)) == 0
    # The same surface with extent still draws, so this is not a blanket skip.
    assert brightest(lambda: algan.Sphere(radius=1, color=algan.BLUE)) > 0


@pytest.mark.fast
def test_seq_and_sync_explain_that_their_lag_ratio_is_fixed():
    """Both pass their own ``lag_ratio`` positionally to ``Lag``, so a caller's
    keyword collided with it and Python blamed ``Lag.__init__`` -- a class the
    caller never mentioned.
    """
    from algan.animation_timeline.animation_contexts import Seq

    with pytest.raises(TypeError, match=r"Seq is Lag with ratio=1"):
        Seq(lag_ratio=0.5)
    with pytest.raises(TypeError, match=r"Sync is Lag with ratio=0"):
        Sync(lag_ratio=0.5)
    # ``ratio`` is Lag's own spelling, and is caught the same way.
    with pytest.raises(TypeError, match=r"Seq is Lag with ratio=1"):
        Seq(ratio=0.5)
    assert algan.Lag(0.5) is not None


@pytest.mark.fast
def test_add_parent_rejects_cycles_like_group_does():
    with algan.Scene():
        square = Square()
        with pytest.raises(HierarchyError, match="its own parent"):
            square.add_parent(square)

        first, second, third = Square(), algan.Circle(), algan.Triangle()
        first.add_parent(second)
        with pytest.raises(HierarchyError, match="create a cycle"):
            second.add_parent(first)

        second.add_parent(third)
        with pytest.raises(HierarchyError, match="create a cycle"):
            third.add_parent(first)

        # A chain that is not a cycle is still fine.
        assert Square().add_parent(second) is not None


@pytest.mark.fast
def test_manim_method_names_point_at_the_algan_one():
    """Algan carries no aliases for its own API, so these names will never
    exist -- but ``AttributeError: 'Square' object has no attribute 'shift'``
    does not say that ``move`` is right there.
    """
    with algan.Scene():
        square = Square().spawn()
        with pytest.raises(AttributeError, match=r"in Algan use move\(\.\.\.\)"):
            square.shift
        with pytest.raises(AttributeError, match="Algan records animations"):
            square.animate
        # An ordinary typo keeps the ordinary message, and the default-valued
        # getattr callers all over the engine keep working.
        with pytest.raises(AttributeError, match="has no attribute 'wibble'"):
            square.wibble
        assert getattr(square, "wibble", "default") == "default"


@pytest.mark.fast
def test_an_updater_that_cannot_take_the_elapsed_time_says_so():
    """Manim's updaters take the mobject alone, so this is the first thing a
    reader from there writes. It failed with a bare arity TypeError from the
    immediate zero-time application, naming nothing about updaters.
    """
    with algan.Scene():
        square = Square().spawn()
        with pytest.raises(TypeError, match="needs 2 positional parameters"):
            square.add_updater(lambda mob: None)
        with pytest.raises(TypeError, match="expects a callable"):
            square.add_updater(42)
        assert square.add_updater(lambda mob, t: None) is not None


@pytest.mark.fast
def test_a_non_callable_post_process_is_rejected_before_the_render(tmp_path):
    with algan.Scene(video_settings=SMOKE_TEST):
        Square(color=algan.BLUE).spawn()
        with pytest.raises(AlganConfigurationError, match="post_processes"):
            algan.Scene.save_frame(str(tmp_path / "still.png"), post_processes=(42,))


@pytest.mark.fast
def test_a_background_string_is_read_as_a_colour_first():
    """Read as a path unconditionally, a colour name was reported as a missing
    file -- and a mistyped path said the same thing as a word that was never a
    colour.
    """
    with algan.Scene(video_settings=SMOKE_TEST) as scene:
        scene.set_background("navy")
        with pytest.raises(AlganConfigurationError, match="neither a color"):
            scene.set_background("not a color", overwrite=True)
        with pytest.raises(AlganConfigurationError, match="neither a color"):
            scene.set_background("missing_background.png", overwrite=True)


@pytest.mark.fast
def test_empty_text_spawns_instead_of_failing_on_torch_cat():
    """``Text("")`` has no glyphs, so its entrance wave has nothing to stagger.
    It used to die in ``torch.cat`` on an empty list.
    """
    with algan.Scene(video_settings=SMOKE_TEST):
        for text in ("", "   ", "\n"):
            assert algan.Text(text).spawn() is not None
        assert algan.Text("hi").spawn() is not None


def test_an_unusable_codec_is_named(tmp_path):
    """It used to cost a whole render and then surface as a missing temp file."""
    from algan.utils.video_encoding import _listed_encoders

    if _listed_encoders("ffmpeg") is None:
        pytest.skip("this FFmpeg cannot be asked for its encoder list")

    with algan.Scene(video_settings=SMOKE_TEST):
        Square(color=algan.BLUE).spawn()
        with pytest.raises(AlganConfigurationError, match="cannot encode with codec"):
            algan.Scene.save_video(
                str(tmp_path / "clip.mp4"), SMOKE_TEST, codec="notacodec"
            )


@pytest.mark.fast
def test_image_pixels_may_arrive_as_a_numpy_array():
    """It is what every imaging library returns and what Manim's ImageMobject
    takes, and it used to reach ``torch.zeros_like`` and fail there.
    """
    import numpy as np

    with algan.Scene(video_settings=SMOKE_TEST):
        assert algan.ImageMob(np.zeros((8, 8, 4), np.uint8)) is not None
        assert algan.ImageMob(np.zeros((8, 8, 3), np.float32)) is not None

        with pytest.raises(AlganConfigurationError, match="needs a height"):
            algan.ImageMob(torch.zeros(8, 8))
        # The channel-count complaint reports the shape that was passed, not
        # the one padding produced.
        with pytest.raises(ValueError, match=r"got \(8, 8, 2\)"):
            algan.ImageMob(torch.zeros(8, 8, 2))


@pytest.mark.fast
def test_the_mob_positioning_surface_answers_to_its_public_names():
    """Phase 3 of the public API overhaul renamed most of ``Mob``'s
    positioning surface and privatized the rest. Every name below is called
    here so a later rename fails at a call site and not only in the export
    snapshot.
    """
    with algan.Scene():
        square = Square().spawn()
        other = Square().spawn()

        # 3b -- one edge-point query, and the center-projected boundary beside
        # it under a name that says they are a pair.
        assert square.get_edge_point(algan.RIGHT) is not None
        assert square.get_edge_point(algan.RIGHT, recursive=False) is not None
        assert square.get_boundary_point(algan.RIGHT) is not None

        # 3c -- the bounding-box trio all describe the same box.
        torch.testing.assert_close(
            square.get_bounding_box_max() - square.get_bounding_box_min(),
            square.get_bounding_box_size(),
        )
        assert len(square.sample_points_in_direction(algan.UP, count=4)) == 4
        assert square.move_to_screen_edge(algan.RIGHT) is square
        assert square.move_to_screen_corner((algan.UP, algan.RIGHT)) is square
        assert square.move_between(algan.ORIGIN, algan.RIGHT) is square
        assert square.move_to_point_with_displacement(algan.ORIGIN, algan.UP) is square
        assert square.move_to(algan.RIGHT, arc_angle=90) is square
        assert square.rotate(90, algan.OUT, about=algan.ORIGIN) is square
        assert square.orbit(90, algan.OUT, about=algan.ORIGIN) is square

        # A radian spelling of the same turn is available on both.
        assert square.rotate(PI / 2, algan.OUT, degrees=False) is square
        assert (
            square.orbit(PI / 2, algan.OUT, about=algan.ORIGIN, degrees=False) is square
        )

        # 3d -- the direction getters and their property spellings agree, and
        # the basis getters have no property spelling.
        for name in ("right", "up", "forward"):
            torch.testing.assert_close(
                getattr(square, name), getattr(square, f"get_{name}_direction")()
            )
            assert getattr(square, f"get_{name}_basis")() is not None

        # 3e -- coordinates are properties, and assigning one is recorded like
        # any other Mob attribute.
        with algan.Off():
            square.x = 3.0
        torch.testing.assert_close(square.x, torch.full_like(square.x, 3.0))
        with algan.Off():
            square.xy = (1.0, 2.0)
        torch.testing.assert_close(square.y, torch.full_like(square.y, 2.0))
        assert square.get_coord([0, 2]) is not None
        assert square.set_coord(2, 1.0) is square

        # 3f -- one alignment method with three anchors.
        for anchor in ("center", "edge", "boundary", "BOUNDARY"):
            assert square.align_with(other, algan.UP, anchor=anchor) is square
        with pytest.raises(AlganConfigurationError, match="anchor must be"):
            square.align_with(other, algan.UP, anchor="middle")

        # 3g -- look's axis is named, not an index.
        assert square.look(algan.UP, with_axis="up") is square
        assert square.look_at(algan.ORIGIN, with_axis="Forward") is square
        with pytest.raises(AlganConfigurationError, match="with_axis must be"):
            square.look(algan.UP, with_axis=2)

        # 3a/3b -- the privatized and deleted names are gone.
        for gone in (
            "get_boundary_edge_point",
            "get_boundary_in_direction",
            "set_x_coord",
            "get_x_coord",
            "set_individual_coords",
            "set_x_y_coord",
            "move_to_edge",
            "move_out_of_screen",
            "move_inline_with_center",
            "get_upwards_direction",
            "morph_soup_parts",
            "resolved_shadow_flags",
            "check_properties_are_valid",
        ):
            assert not hasattr(square, gone), gone


@pytest.mark.fast
def test_move_next_to_align_edge_only_moves_along_the_alignment_axis():
    """``align_edge`` refines a placement; it used to undo it.

    It was implemented on ``move_inline_with_boundary``, which moved the whole
    boundary-to-boundary displacement rather than its component along the
    alignment axis -- so ``move_next_to(chart, RIGHT, align_edge=DOWN)`` lined
    the bottoms up *and* slid the caption back on top of the chart in x.
    """
    with algan.Scene():
        chart = Square().scale(1.5).spawn()
        caption = Square().scale(0.3).spawn()

        with algan.Off():
            caption.move_next_to(chart, algan.RIGHT)
            beside = caption.get_center().clone()
            caption.move_next_to(chart, algan.RIGHT, align_edge=algan.DOWN)

        # x is untouched by the secondary alignment ...
        torch.testing.assert_close(caption.get_center()[..., 0], beside[..., 0])
        # ... and the bottoms now agree.
        torch.testing.assert_close(
            caption.get_boundary_point(algan.DOWN)[..., 1],
            chart.get_boundary_point(algan.DOWN)[..., 1],
        )


@pytest.mark.fast
def test_the_scene_camera_light_and_group_surface_answers_to_its_public_names():
    """Phase 4 of the public API overhaul renamed the Scene, Camera, Light and
    Group surfaces and collapsed the two lifecycle families. Every name below
    is called here so a later rename fails at a call site.
    """
    with algan.Scene() as scene:
        # Scene -- lights. `add_light` used to exist as an alias of
        # `add_light_source`; now it is the only spelling.
        light = PointLight(location=algan.UP * 3, add_to_scene=False)
        assert scene.add_light(light) is light
        assert scene.remove_light(light) is light
        assert scene.clear_lights() is scene
        for gone in ("add_light_source", "remove_light_source", "clear_light_sources"):
            assert not hasattr(scene, gone), gone

        # Scene -- pixel conversions round-trip.
        assert scene.pixels_to_length(scene.length_to_pixels(2.0)) == pytest.approx(2.0)

        # Scene -- one despawn method with flags, and one reset with a flag.
        Square().spawn()
        assert scene.despawn_mobs(retain_history=True, duration=0.5) is scene
        assert scene.reset(rebuild_timeline=False) is scene
        assert scene.reset() is scene
        for gone in (
            "clear_scene",
            "despawn_scene",
            "reset_scene",
            "render_audio_to_file",
        ):
            assert not hasattr(scene, gone), gone
        # `clear` was an alias of `clear_scene`; both are gone.
        assert not hasattr(scene, "clear")

        # Camera.
        camera = scene.get_camera()
        assert camera.screen_half_height > 0
        assert camera.center_on(Square().spawn()) is camera
        assert camera.set_euler_angles(5, 0, 0) is camera
        assert camera.set_euler_angles(PI / 36, 0, 0, degrees=False) is camera
        assert camera.set_near_orthographic() is camera
        for gone in (
            "move_to_make_mob_center_of_view",
            "set_to_orthographic",
            "screen_scale_factor",
            "retroactive_center",
            "get_render_screen_basis",
        ):
            assert not hasattr(camera, gone), gone

        # Lights -- the packing hooks are private, and SpotLight's cone is named.
        spot = SpotLight(cone_angle=20.0, add_to_scene=False)
        assert spot.cone_angle == 20.0
        assert SpotLight(
            cone_angle=PI / 9, degrees=False, add_to_scene=False
        ).cone_angle == (pytest.approx(20.0))
        for gone in ("build_aux", "is_extended", "num_samples", "get_sample_positions"):
            assert not hasattr(spot, gone), gone

        # Group -- the two arrange parameters that were ambiguous.
        group = Group(*[Square(add_to_scene=False) for _ in range(6)])
        assert group.arrange_in_grid(2, row_buffer=0.4, column_buffer=0.2) is group
        assert (
            group.arrange_in_line(algan.RIGHT, equal_widths=True, align_to=algan.DOWN)
            is group
        )
