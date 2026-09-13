"""Sparse screenshot jobs and the Project author-then-render boundary."""

from __future__ import annotations

import pytest
import torch
from PIL import Image

from algan import RIGHT, Project, Scene, SceneManager, Seq, Square
from algan.errors import AlganConfigurationError
from algan.settings import SETTINGS
from algan.settings.video_settings import VideoSettings

STILLS = VideoSettings((8, 6), 10, supersampling=1)


@pytest.fixture(autouse=True)
def authoring_state():
    SceneManager.reset()
    raytracing = SETTINGS.raytracing.to_dict()
    computing = SETTINGS.computing.to_dict()
    yield
    SETTINGS.raytracing._restore(raytracing)
    SETTINGS.computing.set(**computing)
    SceneManager.reset()


def _project(tmp_path, *functions):
    return Project(
        functions,
        video_settings=STILLS,
        file_path=tmp_path / "video.mp4",
        video_directory=tmp_path / "videos",
        screenshot_directory=tmp_path / "shots",
        transcript_directory=tmp_path / "transcripts",
    )


def _frame_spy(monkeypatch, *, before_render=None, chunk_size=2):
    calls = []
    closed = []

    def frames(scene, start, end, **options):
        if before_render:
            before_render(scene)
        indices = tuple(options.get("frame_indices", range(start, end)))
        if "frame_indices" in options:
            indices = indices[start:end]
        calls.append((indices, scene.video_settings, options))
        try:
            for offset in range(0, len(indices), chunk_size):
                yield torch.stack(
                    [
                        torch.full(
                            (
                                scene.num_pixels_screen_height,
                                scene.num_pixels_screen_width,
                                4,
                            ),
                            index % 256,
                            dtype=torch.uint8,
                        )
                        for index in indices[offset : offset + chunk_size]
                    ]
                )
        finally:
            closed.append(indices)

    monkeypatch.setattr(Scene, "get_frames", frames)
    return calls, closed


def _pixel(result):
    with Image.open(result.output_path) as image:
        return image.getpixel((0, 0))[0]


def test_multiple_timestamps_use_one_sparse_job_and_keep_names_and_order(
    monkeypatch, tmp_path
):
    calls, closed = _frame_spy(monkeypatch)
    with Scene(video_settings=STILLS) as scene:
        results = scene.save_frame(tmp_path / "shot", at=[9.0, 0.1, 0.11, 9.0])
    assert [call[0] for call in calls] == [(1, 90)]
    assert closed == [(1, 90)]
    assert [_pixel(result) for result in results] == [90, 1, 1, 90]
    assert [r.output_path.name for r in results] == [
        "shot_9.0.png",
        "shot_0.1.png",
        "shot_0.11.png",
        "shot_9.0.png",
    ]
    assert all(r.status == "rendered" for r in results)
    assert len({r.walltime_seconds for r in results}) == 1


def test_existing_and_duplicate_destinations_are_skipped_before_render(
    monkeypatch, tmp_path
):
    calls, _ = _frame_spy(monkeypatch)
    existing = tmp_path / "shot_0.1.png"
    existing.write_bytes(b"leave this file alone")
    with Scene(video_settings=STILLS) as scene:
        results = scene.save_frame(
            tmp_path / "shot", at=[0.1, 0.2, 0.2, 0.3], overwrite=False
        )
    assert [call[0] for call in calls] == [(2, 3)]
    assert [r.status for r in results] == ["skipped", "rendered", "skipped", "rendered"]
    assert existing.read_bytes() == b"leave this file alone"


def test_empty_timestamps_do_not_start_a_render(monkeypatch, tmp_path):
    calls, _ = _frame_spy(monkeypatch)
    with Scene(video_settings=STILLS) as scene:
        assert scene.save_frame(tmp_path / "empty", at=[]) == []
    assert not calls


@pytest.mark.parametrize("at", [[0.2, float("nan")], [0.2, float("inf")], [-100]])
def test_invalid_timestamps_fail_before_any_output(monkeypatch, tmp_path, at):
    calls, _ = _frame_spy(monkeypatch)
    with (
        Scene(video_settings=STILLS) as scene,
        pytest.raises(AlganConfigurationError, match="finite and non-negative"),
    ):
        scene.save_frame(tmp_path / "bad", at=at)
    assert not calls
    assert not list(tmp_path.glob("*.png"))


@pytest.mark.fast
def test_project_authors_all_checkpoints_before_one_batched_render(
    monkeypatch, tmp_path
):
    authored = []
    placeholders = []

    def intro():
        Scene.wait(1)
        placeholders.append(Scene.save_frame("first"))
        assert not placeholders[-1].output_path.exists()
        Scene.wait(2)
        placeholders.extend(Scene.save_frame("middle", at=[-0.5, 0.2]))
        Scene.wait(1)
        placeholders.append(Scene.save_frame("last"))
        authored.append("tail")

    def before_render(scene):
        assert authored == ["tail"]
        assert scene.animation_manager.context.timespan.current_time == pytest.approx(4)

    calls, _ = _frame_spy(monkeypatch, before_render=before_render)
    results = _project(tmp_path, intro).render_screenshots()
    assert [call[0] for call in calls] == [(2, 12, 25, 42)]
    assert [_pixel(result) for result in results] == [12, 25, 2, 42]
    assert [r.output_path.name for r in results] == [
        "s0_f0_first.png",
        "s0_f1_middle_-0.5.png",
        "s0_f1_middle_0.2.png",
        "s0_f2_last.png",
    ]
    assert all(r.status == "deferred" and r.walltime_seconds == 0 for r in placeholders)
    assert all(r.status == "rendered" for r in results)


@pytest.mark.fast
def test_relative_deferred_time_follows_context_rescaling(monkeypatch, tmp_path):
    def intro():
        with Seq(runtime=4):
            Scene.wait(1)
            Scene.save_frame("rescaled_cursor")
            Scene.save_frame("rescaled_offset", at=-0.5)
            Scene.save_frame("absolute", at=0.5)
            Scene.wait(1)

    calls, _ = _frame_spy(monkeypatch)
    results = _project(tmp_path, intro).render_screenshots()
    assert [call[0] for call in calls] == [(5, 15, 22)]
    assert [_pixel(result) for result in results] == [22, 15, 5]


def test_project_selection_and_early_stop_still_render_after_unwind(
    monkeypatch, tmp_path
):
    authored = []

    def intro():
        Scene.save_frame("ignored", at=0.1)
        Scene.wait(1)
        Scene.save_frame("chosen", at=[0.2, 0.8])
        authored.append("unwanted tail")

    calls, _ = _frame_spy(monkeypatch)
    project = _project(tmp_path, intro)
    results = project.render_screenshots(frames=[1], stop_early=True)
    assert not authored
    assert [call[0] for call in calls] == [(2, 8)]
    assert [r.output_path.name for r in results] == [
        "s0_f1_chosen_0.2.png",
        "s0_f1_chosen_0.8.png",
    ]
    assert not list(project.transcript_directory.glob("*.txt"))


def test_failed_authoring_does_not_render_or_leak_requests(monkeypatch, tmp_path):
    fail = [True]

    def intro():
        Scene.save_frame("checkpoint", at=[0.1, 0.8])
        if fail[0]:
            raise RuntimeError("unfinished scene")

    calls, _ = _frame_spy(monkeypatch)
    project = _project(tmp_path, intro)
    with pytest.raises(RuntimeError, match="unfinished scene"):
        project.render_screenshots()
    assert not calls
    assert not list(project.screenshot_directory.glob("*.png"))
    fail[0] = False
    assert len(project.render_screenshots()) == 2
    assert [call[0] for call in calls] == [(1, 8)]


def test_per_call_settings_are_snapshotted_and_separate_incompatible_jobs(
    monkeypatch, tmp_path
):
    observed = []

    def postprocess(frames):
        return frames

    def intro():
        Scene.wait(2)
        Scene.save_frame("normal", at=[0.1, 0.2])
        with SETTINGS.raytracing.override(shadows=False):
            Scene.save_frame(
                "different",
                VideoSettings((12, 9), 20, supersampling=1),
                at=[0.3, 0.4],
                background="RED",
                post_processes=(postprocess,),
            )
        Scene.save_frame("normal_again", at=[0.5, 0.6])

    previous_shadows = SETTINGS.raytracing.shadows
    calls, _ = _frame_spy(
        monkeypatch,
        before_render=lambda scene: observed.append(SETTINGS.raytracing.shadows),
    )
    results = _project(tmp_path, intro).render_screenshots()
    assert [call[0] for call in calls] == [(1, 2), (6, 8), (5, 6)]
    assert [call[1].resolution for call in calls] == [(8, 6), (12, 9), (8, 6)]
    assert calls[1][2]["post_processes"] == (postprocess,)
    assert observed == [previous_shadows, False, previous_shadows]
    assert SETTINGS.raytracing.shadows is previous_shadows
    with Image.open(results[2].output_path) as image:
        assert image.size == (12, 9)


def test_project_overwrite_false_and_default_postprocess_options(monkeypatch, tmp_path):
    def intro():
        Scene.save_frame("same", at=[0.1, 0.2])

    project = _project(tmp_path, intro)
    calls, _ = _frame_spy(monkeypatch)
    first = project.render_screenshots(post_processes=())
    second = project.render_screenshots(overwrite=False, post_processes=())
    assert [r.status for r in first] == ["rendered", "rendered"]
    assert [r.status for r in second] == ["skipped", "skipped"]
    assert len(calls) == 1
    assert calls[0][2]["post_processes"] == ()


def test_project_keeps_scenes_in_separate_jobs(monkeypatch, tmp_path):
    def intro():
        Scene.save_frame("same", at=[0.1, 0.2])

    def outro():
        Scene.save_frame("same", at=[0.3, 0.4])

    calls, _ = _frame_spy(monkeypatch)
    results = _project(tmp_path, intro, outro).render_screenshots()
    assert [call[0] for call in calls] == [(1, 2), (3, 4)]
    assert [r.output_path.name for r in results] == [
        "s0_f0_same_0.1.png",
        "s0_f0_same_0.2.png",
        "s1_f0_same_0.3.png",
        "s1_f0_same_0.4.png",
    ]


@pytest.mark.parametrize("count", [0, 1, 3])
def test_bad_render_output_restores_settings_and_closes_generator(
    monkeypatch, tmp_path, count
):
    closed = []

    def frames(scene, *_args, **_kwargs):
        try:
            yield torch.zeros((count, 6, 8, 4), dtype=torch.uint8)
        finally:
            closed.append(True)

    monkeypatch.setattr(Scene, "get_frames", frames)
    original = VideoSettings((16, 12), 30, supersampling=1)
    with Scene(video_settings=original) as scene:
        background = scene.background_frame
        with pytest.raises(RuntimeError, match="frames were produced"):
            scene.save_frame(tmp_path / "bad", STILLS, at=[0.1, 0.2], background="RED")
        assert scene.video_settings == original
        assert scene.frames_per_second == 30
        assert scene.background_frame is background
    assert closed == [True]


def test_image_write_failure_closes_renderer(monkeypatch, tmp_path):
    calls, closed = _frame_spy(monkeypatch, chunk_size=1)

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(Image.Image, "save", fail)
    with (
        Scene(video_settings=STILLS) as scene,
        pytest.raises(OSError, match="disk full"),
    ):
        scene.save_frame(tmp_path / "bad", at=[0.1, 0.2])
    assert len(calls) == 1
    assert closed == [(1, 2)]


@pytest.mark.parametrize("indices", [[0, 0], [2, 1], [-1, 2], [0.5, 2], [True, 2]])
def test_sparse_index_validation(indices):
    with (
        Scene(video_settings=STILLS) as scene,
        pytest.raises(AlganConfigurationError, match="strictly increasing"),
    ):
        list(scene.get_frames(0, len(indices), frame_indices=indices))


@pytest.mark.parametrize("prefetch", [False, True])
@pytest.mark.parametrize("geometry", [False, True])
def test_sparse_pixels_match_individual_renders_with_bounded_materialization(
    monkeypatch, tmp_path, geometry, prefetch
):
    """Exercise the real CPU renderer, not the screenshot writer's stub."""
    times = [0.1, 0.5, 1.1, 1.5]
    settings = VideoSettings((24, 20), 10, supersampling=1)

    def background(x, y, time):
        value = 0.1 + time * 0.2 + x * 0.03 + y * 0.02
        return value.expand(-1, -1, -1, 3)

    monkeypatch.setenv("ALGAN_PREFETCH_BATCHES", "1" if prefetch else "0")
    with (
        SETTINGS.computing.override(max_animation_batch_size=2),
        Scene(video_settings=settings) as scene,
    ):
        scene.set_background(background)
        if geometry:
            square = Square().spawn()
            square.move(RIGHT)
            Square().spawn()
            Scene.wait(1)
        else:
            Scene.wait(2)
        singles = [
            scene.save_frame(tmp_path / f"one_{i}", at=t, post_processes=())
            for i, t in enumerate(times)
        ]
        observed = []
        original = scene.timeline_manager.set_state_to_times

        def record(times, *args, **kwargs):
            observed.append(times.detach().cpu().clone())
            return original(times, *args, **kwargs)

        monkeypatch.setattr(scene.timeline_manager, "set_state_to_times", record)
        batches = scene.save_frame(tmp_path / "batch", at=times, post_processes=())
    assert observed
    assert all(len(window) <= 2 for window in observed)
    assert all(
        any(abs(float(t) - wanted) < 1e-5 for wanted in times)
        for window in observed
        for t in window.flatten()
    )
    for single, batch in zip(singles, batches):
        with Image.open(single.output_path) as a, Image.open(batch.output_path) as b:
            import numpy as np

            delta = np.abs(np.asarray(a).astype(int) - np.asarray(b).astype(int))
        assert delta.max() <= 2


def test_inference_background_captures_do_not_require_version_counter():
    from algan._still_frames import _background_key

    with torch.inference_mode():
        background = torch.zeros((2, 2, 4))
    assert _background_key(background) != _background_key(background)


def test_deferred_result_does_not_preview_existing_file(tmp_path):
    from algan.utils.algan_utils import RenderResult

    path = tmp_path / "old.png"
    Image.new("RGB", (2, 2)).save(path)
    result = RenderResult("deferred", path)
    assert "deferred" in result._repr_html_()
    assert "<img" not in result._repr_html_()
