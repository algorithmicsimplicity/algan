from __future__ import annotations

from pathlib import Path

import pytest

from algan import Project, Scene, SceneManager, Speech
from algan.errors import AlganConfigurationError
from algan.settings import SETTINGS
from algan.utils import algan_utils
from algan.utils.algan_utils import RenderResult, scene_function


class _SilentClip:
    duration = 0


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

    @scene_function(name="custom_name")
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

    @scene_function(name="duplicate")
    def first():
        pass

    @scene_function(name="duplicate")
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


def test_screenshot_render_uses_stable_global_frame_ids_for_subsets(
    monkeypatch, tmp_path
):
    rendered = []

    def fake_render_still(_scene, destination, timestamp):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"frame")
        rendered.append((destination, timestamp))
        return destination

    monkeypatch.setattr(Scene, "_render_still", fake_render_still)

    def first():
        Scene.save_frame("first")

    def second():
        Scene.save_frame("second_a")
        Scene.save_frame("second_b")

    project = Project([first, second], **_project_paths(tmp_path))

    subset_results = project.render_screenshots("second")
    assert [result.output_path.name for result in subset_results] == [
        "1_second_a.png",
        "3_second_b.png",
    ]

    rendered.clear()
    all_results = project.render_screenshots()
    assert [result.output_path.name for result in all_results] == [
        "0_first.png",
        "1_second_a.png",
        "3_second_b.png",
    ]
    assert project.render_screenshots([]) == []


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
