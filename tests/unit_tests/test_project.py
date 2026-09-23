from __future__ import annotations

from pathlib import Path

import pytest

from algan import Project, Scene, SceneManager, Speech
from algan.errors import AlganConfigurationError
from algan.settings import SETTINGS
from algan.utils import algan_utils
from algan.utils.algan_utils import RenderResult, algan_scene


class _SilentClip:
    # ``duration`` is moviepy's attribute name, which is what ``Audio`` reads.
    duration = 0


def test_validate_resolves_speech_and_suppresses_exports(monkeypatch, tmp_path):
    class Narration:
        duration = 3

    def scene():
        with Speech("A triple-nested function.", wait_at_end=0):
            Scene.wait(1)
        Scene.save_frame("midpoint")
        Scene.save_video("ignored.mp4")
        Scene.wait(2)

    monkeypatch.setattr(
        algan_utils,
        "_render_scene_to_file",
        lambda *a, **k: pytest.fail("validation exported video"),
    )
    monkeypatch.setattr(
        Scene,
        "_render_still",
        lambda *a, **k: pytest.fail("validation rendered a frame"),
    )
    project = Project(
        [scene],
        speech_source=lambda _: Narration(),
        transcript_line_length=10,
        **_project_paths(tmp_path),
    )
    report = project.validate()
    assert report.duration_seconds == pytest.approx(5)
    assert report.scenes[0].checkpoints == 1
    assert report.scenes[0].transcript.strip() == "A triple-nested function."
    assert project.global_transcript_path.exists()


def test_validate_propagates_authoring_error_with_scene_name(tmp_path):
    def broken():
        raise ValueError("missing asset")

    with pytest.raises(ValueError, match="missing asset") as error:
        Project([broken], **_project_paths(tmp_path)).validate()
    if hasattr(error.value, "__notes__"):
        assert "0_broken" in error.value.__notes__[0]


def test_validate_script_compares_exact_words_in_project_order(tmp_path):
    def first():
        with Speech("One word.", wait_at_end=0):
            pass

    def second():
        with Speech("Then another.", wait_at_end=0):
            pass

    project = Project(
        [first, second],
        speech_source=lambda _: _SilentClip(),
        **_project_paths(tmp_path),
    )
    project.validate(["second", "first"], script="One   word.\n\nThen another.")
    expected = tmp_path / "script.txt"
    expected.write_text("One word. Then another.", encoding="utf-8")
    project.validate(script=expected)
    project.validate("second", script="Then another.")
    for script, word in [
        ("One wrong. Then another.", 2),
        ("One word.", 3),
        ("One word. Then another. Extra.", 5),
        ("one word. Then another.", 1),
    ]:
        with pytest.raises(AlganConfigurationError, match=f"word {word}:"):
            project.validate(script=script)
    assert project.run_cli(["--validate", "--script", str(expected)])


def test_script_mismatch_names_the_scene_and_shows_context(tmp_path):
    def opening():
        with Speech("A short opening line.", wait_at_end=0):
            pass

    def middle():
        with Speech("The model tried hacking into the servers today.", wait_at_end=0):
            pass

    project = Project(
        [opening, middle],
        speech_source=lambda _: _SilentClip(),
        **_project_paths(tmp_path),
    )
    script = "A short opening line. The model tried hacked into the servers today."
    with pytest.raises(AlganConfigurationError) as error:
        project.validate(script=script)
    message = str(error.value)
    assert "word 8: expected 'hacked', got 'hacking'" in message
    assert "middle" in message
    assert "word 4 of that scene" in message
    assert "script:    ... short opening line. The model tried [hacked] into" in message
    assert (
        "narration: ... short opening line. The model tried [hacking] into" in message
    )
    with pytest.raises(AlganConfigurationError, match="after the end of scene"):
        project.validate(script=script.replace("hacked", "hacking") + " Extra.")


def test_project_post_process_defaults_reach_cli_video_and_stills(
    monkeypatch, tmp_path
):
    from PIL import Image

    calls = []

    def effect(frames, **kwargs):
        return frames

    def scene():
        Scene.wait(1)
        Scene.save_frame("checkpoint")

    def fake_video(active_scene, file_path=None, video_settings=None, **kwargs):
        calls.append(("video", kwargs["post_processes"]))
        return RenderResult("rendered", Path(file_path), 1)

    def fake_still(batch):
        calls.append(("still", batch.post_processes))
        results = []
        for target in batch.targets:
            target.path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (12, 20), "red").save(target.path)
            results.append(RenderResult("rendered", target.path, 1))
        return results

    monkeypatch.setattr(algan_utils, "_render_scene_to_file", fake_video)
    monkeypatch.setattr("algan._still_frames._StillBatch.render", fake_still)
    project = Project([scene], post_processes=[effect], **_project_paths(tmp_path))
    project.run_cli(["--render-video"])
    project.run_cli(["--render-screenshots", "--contact-sheet"])
    assert calls == [("video", (effect,)), ("still", (effect,))]
    assert (
        project.last_contact_sheet_path
        == tmp_path / "screenshots" / "contact_sheet.png"
    )
    with Image.open(project.last_contact_sheet_path) as sheet:
        assert sheet.mode == "RGB"
        assert sheet.size[0] == 344
        assert sheet.getpixel((172, 100)) == (255, 0, 0)
        assert sheet.crop((12, 258, 320, sheet.height - 12)).getextrema()[0][1] > 200
    project.render_video(post_processes=[])
    project.render_screenshots(post_processes=[])
    assert calls[-2:] == [("video", []), ("still", ())]


def test_contact_sheet_handles_mixed_aspects_and_does_not_overwrite_stills(tmp_path):
    from PIL import Image

    def scene():
        pass

    project = Project([scene], **_project_paths(tmp_path))
    results = []
    for index, size in enumerate(((20, 40), (40, 20), (30, 30))):
        path = tmp_path / f"s{index}_f0_label.png"
        Image.new("RGB", size, "blue").save(path)
        results.append(RenderResult("rendered", path, 0))
    sheet = project._write_contact_sheet(results, True, 2, overwrite=True)
    with Image.open(sheet) as image:
        assert image.width == 676
        assert image.height > 500
    before = sheet.read_bytes()
    project._write_contact_sheet(results[:1], True, 1, overwrite=False)
    assert sheet.read_bytes() == before
    with pytest.raises(AlganConfigurationError, match="overwrite"):
        project._write_contact_sheet(results, results[0].output_path, 2, overwrite=True)


def test_project_profile_uses_managed_speech_and_suppresses_manual_saves(
    monkeypatch, tmp_path
):
    from algan.utils import profiling_utils

    seen = []

    def scene():
        with Speech("exact words", wait_at_end=0):
            pass
        assert Scene.save_frame("ignored") == []
        assert Scene.save_video("ignored.mp4").status == "skipped"

    def fake_profile(author, settings, **options):
        SceneManager.reset()
        author()
        active = Scene.current()
        seen.append(
            (
                active.audio_manager.video_transcript,
                active._project_run.allow_video_render,
                options,
            )
        )
        return [{"total": 5}]

    monkeypatch.setattr(profiling_utils, "profile_scene", fake_profile)
    project = Project(
        [scene], speech_source=lambda _: _SilentClip(), **_project_paths(tmp_path)
    )
    assert project.profile(0) == {"0_scene": [{"total": 5}]}
    assert seen[0][0].strip() == "exact words"
    assert seen[0][1] is True
    assert seen[0][2]["runs"] == 2
    assert seen[0][2]["output_directory"] == project.video_directory / "profiling"


def test_estimate_uses_weighted_warm_rate_and_separate_startup(monkeypatch, tmp_path):
    def intro():
        Scene.wait(2)

    def dense():
        Scene.wait(8)

    def tail():
        Scene.wait(10)

    project = Project([intro, dense, tail], **_project_paths(tmp_path))

    def profile(scenes, **kwargs):
        assert scenes == [0, 1]
        return {
            "0_intro": [
                {"total": 6, "scene_seconds": 2},
                {"total": 2, "scene_seconds": 2},
            ],
            "1_dense": [
                {"total": 20, "scene_seconds": 8},
                {"total": 16, "scene_seconds": 8},
            ],
        }

    monkeypatch.setattr(project, "profile", profile)
    estimate = project.estimate_render_time([0, 1])
    assert estimate.warm_seconds == 36
    assert estimate.cold_overhead_seconds == 8
    assert estimate.estimated_seconds == 44
    assert estimate.range_seconds == (28, 48)


@pytest.mark.parametrize("runs", [0, 1, True, 2.5])
def test_estimate_rejects_insufficient_passes(tmp_path, runs):
    with pytest.raises(AlganConfigurationError, match="at least two"):
        Project([lambda: None], **_project_paths(tmp_path)).estimate_render_time(
            0, runs=runs
        )


def test_validate_cli_prints_duration_without_rendering(tmp_path, capsys):
    def intro():
        Scene.wait(2)

    project = Project([intro], **_project_paths(tmp_path))
    assert project.run_cli(["--validate", "intro"])
    assert "2.00s total" in capsys.readouterr().out


@pytest.fixture(autouse=True)
def reset_scene_manager():
    SceneManager.reset()
    yield
    SceneManager.reset()


def _project_paths(tmp_path):
    return {
        "file_path": tmp_path / "combined.mp4",
        "video_directory": tmp_path / "videos",
        "screenshot_directory": tmp_path / "screenshots",
        "transcript_directory": tmp_path / "transcripts",
    }


def test_project_builds_stable_scene_and_frame_ids(tmp_path):
    def introduction(optional="accepted"):
        return optional

    @algan_scene(name="custom_name")
    def implementation():
        pass

    project = Project([introduction, implementation], **_project_paths(tmp_path))

    assert project.scene_names == ("0_introduction", "1_custom_name")
    assert project.frame_id("introduction", 0) == 0
    assert project.frame_id("1_custom_name", 0) == 1
    assert project.frame_id(0, 1) == 2
    assert project.frame_id(1, 1) == 3


def test_project_constructor_validation(tmp_path):
    paths = _project_paths(tmp_path)

    with pytest.raises(AlganConfigurationError, match="cannot be empty"):
        Project([], **paths)
    with pytest.raises(AlganConfigurationError, match="not callable"):
        Project([None], **paths)

    def requires_argument(value):
        return value

    with pytest.raises(AlganConfigurationError, match="requires arguments: value"):
        Project([requires_argument], **paths)

    @algan_scene(name="duplicate")
    def first():
        pass

    @algan_scene(name="duplicate")
    def second():
        pass

    with pytest.raises(AlganConfigurationError, match="must be unique"):
        Project([first, second], **paths)

    def scene():
        pass

    with pytest.raises(AlganConfigurationError, match="file_path collides"):
        Project(
            [scene],
            file_path=tmp_path / "videos" / "0_scene.mp4",
            video_directory=tmp_path / "videos",
            screenshot_directory=tmp_path / "screenshots",
            transcript_directory=tmp_path / "transcripts",
        )

    with pytest.raises(AlganConfigurationError, match="positive integer"):
        Project([scene], transcript_line_length=0, **paths)


def test_project_directories_follow_scene_path_resolution_rules(tmp_path):
    def scene():
        pass

    with SETTINGS.paths.override(output_root=tmp_path, output_directory="outputs"):
        project = Project([scene], file_path="combined")

    assert project.file_path == tmp_path / "outputs" / "combined.mp4"
    assert project.video_directory == tmp_path / "outputs" / "videos"
    assert project.screenshot_directory == tmp_path / "outputs" / "screenshots"
    assert project.transcript_directory == tmp_path / "outputs" / "transcripts"


def test_video_render_skips_scene_save_calls_and_renders_one_managed_video(
    monkeypatch, tmp_path
):
    frame_returns = []
    manual_video_returns = []
    rendered_videos = []

    def fake_render_scene(scene, file_path=None, **_kwargs):
        destination = Path(file_path)
        rendered_videos.append((scene, destination))
        return RenderResult("rendered", destination)

    monkeypatch.setattr(algan_utils, "_render_scene_to_file", fake_render_scene)
    monkeypatch.setattr(
        Scene,
        "_render_still",
        lambda *_args, **_kwargs: pytest.fail("video mode rendered a screenshot"),
    )

    def scene():
        frame_returns.append(Scene.save_frame("ignored"))
        manual_video_returns.append(Scene.save_video(tmp_path / "manual.mp4"))

    project = Project([scene], **_project_paths(tmp_path))
    results = project.render_video(0)

    assert frame_returns == [[]]
    assert len(manual_video_returns) == 1
    assert manual_video_returns[0].status == "skipped"
    assert [path for _, path in rendered_videos] == [
        tmp_path / "videos" / "0_scene.mp4"
    ]
    assert results == [RenderResult("rendered", tmp_path / "videos" / "0_scene.mp4")]


def test_project_writes_wrapped_scene_and_global_transcripts(tmp_path):
    def first():
        with Speech("alpha beta gamma delta epsilon"):
            pass
        with Speech("a second paragraph"):
            pass

    def second():
        with Speech("the later scene transcript"):
            pass

    def silent():
        pass

    project = Project(
        [first, second, silent],
        transcript_line_length=12,
        speech_source=lambda _script: _SilentClip(),
        **_project_paths(tmp_path),
    )

    project.render_screenshots("second")
    second_path = tmp_path / "transcripts" / "1_second.txt"
    assert second_path.exists()
    assert project.global_transcript_path.read_text(encoding="utf-8") == (
        second_path.read_text(encoding="utf-8")
    )

    project.render_screenshots("0_first")
    first_path = tmp_path / "transcripts" / "0_first.txt"
    first_text = first_path.read_text(encoding="utf-8")
    assert "\n\n" in first_text
    assert all(len(line) <= 12 for line in first_text.splitlines())
    assert project.global_transcript_path.read_text(encoding="utf-8") == (
        first_text.strip()
        + "\n\n"
        + second_path.read_text(encoding="utf-8").strip()
        + "\n"
    )

    project.render_screenshots("silent")
    assert not (tmp_path / "transcripts" / "2_silent.txt").exists()


def test_project_concatenates_to_its_resolved_file_path(monkeypatch, tmp_path):
    calls = []

    def scene():
        pass

    def fake_concatenate(directory, **kwargs):
        calls.append((directory, kwargs))
        return Path(kwargs["output_file"])

    monkeypatch.setattr(algan_utils, "concatenate_videos", fake_concatenate)
    project = Project([scene], **_project_paths(tmp_path))

    result = project.concatenate_videos(threads=3, reencode=True)

    assert result == (tmp_path / "combined.mp4").resolve()
    assert calls == [
        (
            str(tmp_path / "videos"),
            {
                "threads": 3,
                "reencode": True,
                "output_file": str((tmp_path / "combined.mp4").resolve()),
                "input_files": (str((tmp_path / "videos" / "0_scene.mp4").resolve()),),
            },
        )
    ]


def test_project_run_cli_dispatches_project_actions(monkeypatch, tmp_path):
    def first():
        pass

    def second():
        pass

    project = Project([first, second], **_project_paths(tmp_path))
    calls = []
    monkeypatch.setattr(
        project,
        "render_screenshots",
        lambda scenes=None: calls.append(("screenshots", scenes)),
    )
    monkeypatch.setattr(
        project,
        "render_video",
        lambda scenes=None: calls.append(("video", scenes)),
    )
    monkeypatch.setattr(
        project,
        "concatenate_videos",
        lambda: calls.append(("concatenate", None)),
    )

    assert project.run_cli(["--render-screenshots", "0", "second"]) is True
    assert project.run_cli(["--render-video"]) is True
    assert project.run_cli(["--concatenate-videos"]) is True
    assert calls == [
        ("screenshots", (0, "second")),
        ("video", None),
        ("concatenate", None),
    ]


def test_project_run_cli_uses_process_args_and_ignores_unrecognized_args(
    monkeypatch, tmp_path
):
    def scene():
        pass

    project = Project([scene], **_project_paths(tmp_path))
    rendered = []
    monkeypatch.setattr(
        project,
        "render_video",
        lambda scenes=None: rendered.append(scenes),
    )
    monkeypatch.setattr(
        "sys.argv",
        ["project_script.py", "--external-option", "--render-video", "scene"],
    )

    assert project.run_cli() is True
    assert rendered == [("scene",)]
    assert project.run_cli(["--external-option", "value"]) is False


def test_project_rejects_invalid_scene_selectors_and_has_no_public_render(tmp_path):
    def first():
        pass

    def second():
        pass

    project = Project([first, second], **_project_paths(tmp_path))

    with pytest.raises(AlganConfigurationError, match="Unknown Project scene ID"):
        project.render_screenshots(9)
    with pytest.raises(AlganConfigurationError, match="does not match"):
        project.render_video("1_first")
    assert not hasattr(project, "render")
