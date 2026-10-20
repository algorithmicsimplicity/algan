"""How a Project picks its scenes and frames, which settings it renders them at,
and what its command line forwards.

A Project names its scene functions by ID, name or ``<id>_<name>`` stem. It
picks save-frame calls with ``frames`` patterns -- a frame's index within its
scene, a glob, or a substring, matched with and without the
``s<scene>_f<index>_`` prefix and ignoring case. ``--video-settings`` names a
preset in any case, and each selected scene is authored in a Scene of its own
at the effective video settings. These tests pin those rules as the
docstrings and ``run_cli``'s help state them, the constructor's refusals, the
warnings a partial or unmatched selection prints, and that ``~`` in an output
directory is the home directory.

Rendering is stubbed at ``_StillBatch.render`` (stills) and
``_render_scene_to_file`` (videos), so nothing here pays for a frame.
``test_project.py`` and ``test_batched_stills.py`` cover what is not repeated
here: validation against a script, transcripts, profiling estimates, contact
sheets and the author-then-render boundary of a screenshot pass.

A feature test, so it sits outside the fast suite: only a change to
``algan/project.py`` breaks it.
"""

from __future__ import annotations

import logging
import re
from functools import partial
from pathlib import Path

import pytest
from PIL import Image

from algan import Project, Scene, SceneManager, Speech
from algan.errors import AlganConfigurationError
from algan.project import (
    ProjectValidation,
    RenderEstimate,
    _frame_pattern_matches,
    _video_settings_for_name,
)
from algan.settings import SETTINGS
from algan.settings.video_settings import (
    _PRESETS_BY_NAME,
    HD,
    LD,
    MD,
    PREVIEW,
    SMOKE_TEST,
)
from algan.utils import algan_utils
from algan.utils.algan_utils import RenderResult, algan_scene


class _SilentClip:
    # ``duration`` is moviepy's attribute name, which is what ``Audio`` reads.
    duration = 0


@pytest.fixture(autouse=True)
def _fresh_scenes():
    SceneManager.reset()
    yield
    SceneManager.reset()


@pytest.fixture
def algan_logs(caplog):
    """``caplog`` wired to Algan's logger, which does not propagate to root."""
    algan_logger = logging.getLogger("algan")
    previous_level = algan_logger.level
    caplog.set_level(logging.INFO)
    algan_logger.addHandler(caplog.handler)
    algan_logger.setLevel(logging.INFO)
    try:
        yield caplog
    finally:
        algan_logger.removeHandler(caplog.handler)
        algan_logger.setLevel(previous_level)


def _warnings(logs):
    return [r.getMessage() for r in logs.records if r.levelno >= logging.WARNING]


@pytest.fixture
def stills(monkeypatch):
    """Stub the still renderer; record each written path with its settings."""
    rendered = []

    def render(batch):
        results = []
        for target in batch.targets:
            target.path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (2, 2), "white").save(target.path)
            rendered.append((target.path, batch.video_settings))
            results.append(RenderResult("rendered", target.path, 0))
        return results

    monkeypatch.setattr("algan._still_frames._StillBatch.render", render)
    return rendered


@pytest.fixture
def videos(monkeypatch):
    """Stub the video renderer; record each requested path with its settings."""
    rendered = []

    def render(scene, file_path=None, video_settings=None, **options):
        rendered.append((Path(file_path), video_settings))
        return RenderResult("rendered", Path(file_path))

    monkeypatch.setattr(algan_utils, "_render_scene_to_file", render)
    return rendered


def _paths(tmp_path):
    return {
        "file_path": tmp_path / "combined.mp4",
        "video_directory": tmp_path / "videos",
        "screenshot_directory": tmp_path / "screenshots",
        "transcript_directory": tmp_path / "transcripts",
    }


def _names(rendered):
    return [path.name for path, _ in rendered]


def walkthrough():
    Scene.wait(1)
    Scene.save_frame("initial_output")
    Scene.save_frame("s05_perturbed_output_distance")
    Scene.save_frame("final")


def epilogue():
    Scene.save_frame("final")


# -- frame patterns --------------------------------------------------------------

# The second frame of scene 1, saved as "s05_perturbed_output_distance".
LOCAL = "s05_perturbed_output_distance"
FULL = "s1_f4_s05_perturbed_output_distance"


@pytest.mark.parametrize(
    ("pattern", "selected"),
    [
        ("4", True),  # digits: the frame's index within its scene
        ("04", True),
        ("5", False),  # a "5" in the name is not an index of 5
        ("1", False),  # nor is the scene's own number
        ("s05_*", True),  # a glob, against the name as saved
        ("S05_*", True),  # ignoring case
        ("s1_f4_*", True),  # or against the prefixed name
        ("*_DISTANCE", True),
        ("s?5_p*", True),
        ("[rs]05_*", True),
        ("perturbed*", False),  # a glob matches the whole name
        ("*.png", False),  # the save-frame name, not the file on disk
        ("Perturbed", True),  # a substring, anywhere, ignoring case
        ("f4_s05", True),  # including across the prefix
        ("output_distance_2", False),
    ],
)
def test_a_frame_pattern_is_an_index_a_glob_or_a_substring(pattern, selected):
    assert _frame_pattern_matches(pattern, 4, LOCAL, FULL) is selected


def test_cli_frames_render_only_the_matching_stills(tmp_path, stills):
    """``--frames`` takes several patterns, and comma-separated lists in each."""
    project = Project([walkthrough, epilogue], **_paths(tmp_path))
    assert project.run_cli(
        ["--render-screenshots", "walkthrough", "--frames", "PERTURBED,2"]
    )
    assert _names(stills) == [
        "s0_f1_s05_perturbed_output_distance.png",
        "s0_f2_final.png",
    ]

    stills.clear()
    assert project.run_cli(["--render-screenshots", "--frames", "s?_f0_*", "final"])
    assert _names(stills) == [
        "s0_f0_initial_output.png",
        "s0_f2_final.png",
        "s1_f0_final.png",
    ]
    on_disk = sorted(p.name for p in project.screenshot_directory.iterdir())
    assert on_disk == [
        "s0_f0_initial_output.png",
        "s0_f1_s05_perturbed_output_distance.png",
        "s0_f2_final.png",
        "s1_f0_final.png",
    ]


def test_frame_selections_are_split_stripped_and_deduplicated():
    normalized = Project._normalized_frame_patterns
    assert normalized(None) == ()
    assert normalized(3) == ("3",)
    assert normalized("intro") == ("intro",)
    assert normalized(["a, b", "b", 7, " a ", "c,,"]) == ("a", "b", "7", "c")


@pytest.mark.parametrize(
    ("frames", "message"),
    [
        (True, "Invalid Project frame selection: True"),
        (2.5, "Invalid Project frame selection: 2.5"),
        ([True], "Invalid Project frame selector: True"),
        (["intro", None], "Invalid Project frame selector: None"),
        ([], "Project frame selection cannot be empty"),
        (" , ", "Project frame selection cannot be empty"),
    ],
)
def test_a_bad_frame_selection_is_refused_before_authoring(
    tmp_path, stills, frames, message
):
    authored = []
    project = Project([lambda: authored.append(True)], **_paths(tmp_path))
    with pytest.raises(AlganConfigurationError, match=re.escape(message)):
        project.render_screenshots(frames=frames)
    assert not authored
    assert not stills


def test_stop_early_abandons_the_scene_and_says_it_is_partial(
    tmp_path, stills, algan_logs
):
    authored = []

    def scene():
        Scene.save_frame("first")
        authored.append("after first")
        Scene.save_frame("second")
        authored.append("after second")
        Scene.save_frame("third")

    project = Project([scene], **_paths(tmp_path))
    assert project.run_cli(
        ["--render-screenshots", "--frames", "second", "--stop-early"]
    )
    assert authored == ["after first"]
    assert _names(stills) == ["s0_f1_second.png"]
    [warning] = _warnings(algan_logs)
    assert warning.startswith(
        "PARTIAL RENDER: scene 0_scene was abandoned after frame s0_f1_second "
        "because every frame pattern had matched."
    )
    assert any(
        r.getMessage() == "Stopped early in project scene 0_scene in screenshots mode"
        for r in algan_logs.records
    )


def test_patterns_that_match_nothing_are_named_in_one_warning(
    tmp_path, stills, algan_logs
):
    """Matched in any selected scene counts; the rest are listed, once."""
    project = Project([walkthrough, epilogue], **_paths(tmp_path))
    project.render_screenshots(frames=["final", "nope", "*.png", "initial"])
    assert _names(stills) == [
        "s0_f0_initial_output.png",
        "s0_f2_final.png",
        "s1_f0_final.png",
    ]
    assert _warnings(algan_logs) == [
        "No frame matched 'nope', '*.png' in any selected scene, so nothing was "
        "rendered for them. Frame patterns are matched against the save-frame "
        "name, not the file on disk."
    ]

    algan_logs.clear()
    project.render_screenshots("epilogue", frames="initial")
    assert _warnings(algan_logs) == [
        "No frame matched 'initial' in any selected scene, so nothing was "
        "rendered for it. Frame patterns are matched against the save-frame "
        "name, not the file on disk."
    ]


def test_screenshot_only_options_are_refused_before_authoring(tmp_path, stills, videos):
    authored = []
    project = Project([lambda: authored.append(True)], **_paths(tmp_path))
    with pytest.raises(AlganConfigurationError, match="only apply to screenshot"):
        project.render_video(frames="intro")
    with pytest.raises(AlganConfigurationError, match="only apply to screenshot"):
        project.render_video(stop_early=True)
    with pytest.raises(AlganConfigurationError, match="needs a frames selection"):
        project.render_screenshots(stop_early=True)
    with pytest.raises(
        AlganConfigurationError, match="Unsupported screenshot options: codec, fps"
    ):
        project.render_screenshots(codec="libx264", fps=3)
    assert not authored
    assert not stills
    assert not videos


# -- scenes and their settings ---------------------------------------------------


def test_selected_scenes_render_once_each_in_project_order(tmp_path, videos):
    def first():
        Scene.wait(1)

    def second():
        Scene.wait(1)

    def third():
        Scene.wait(1)

    project = Project([first, second, third], **_paths(tmp_path))
    results = project.render_video(["third", 0, "0_first", "first", 2])
    assert _names(videos) == ["0_first.mp4", "2_third.mp4"]
    assert [result.output_path.name for result in results] == [
        "0_first.mp4",
        "2_third.mp4",
    ]


def test_each_scene_is_authored_and_rendered_at_the_effective_settings(
    tmp_path, videos
):
    """A call's settings, else the Project's, else ``SETTINGS.video`` at the time."""
    seen = []

    def first():
        seen.append(Scene.current().video_settings.resolution)
        Scene.wait(1)

    def second():
        seen.append(Scene.current().video_settings.resolution)
        Scene.wait(1)

    project = Project(
        [first, second],
        video_settings=SMOKE_TEST.set(resolution=(16, 8)),
        **_paths(tmp_path),
    )
    project.render_video()
    project.render_video(video_settings=SMOKE_TEST.set(resolution=(8, 8)))
    unset = Project([first, second], **_paths(tmp_path))
    SETTINGS.video.set(SMOKE_TEST.set(resolution=(12, 6)))
    unset.render_video()

    expected = [(16, 8)] * 2 + [(8, 8)] * 2 + [(12, 6)] * 2
    assert seen == expected
    assert [settings.resolution for _, settings in videos] == expected


def test_screenshots_render_at_the_projects_settings(tmp_path, stills):
    project = Project(
        [walkthrough],
        video_settings=SMOKE_TEST.set(resolution=(10, 6)),
        **_paths(tmp_path),
    )
    project.render_screenshots(frames="final")
    project.render_screenshots(frames="final", video_settings=HD)
    assert [settings.resolution for _, settings in stills] == [(10, 6), HD.resolution]


# -- presets and the command line ------------------------------------------------


def test_video_settings_presets_are_named_in_any_case():
    for name, preset in _PRESETS_BY_NAME.items():
        assert _video_settings_for_name(name.lower()) is preset
        assert _video_settings_for_name(f"  {name.title()} ") is preset
    with pytest.raises(
        AlganConfigurationError, match="Unknown video settings preset: '4k'"
    ) as error:
        _video_settings_for_name("4k")
    assert all(name in str(error.value) for name in _PRESETS_BY_NAME)


def _record_actions(monkeypatch, project):
    """Replace every action ``run_cli`` dispatches with a recorder."""
    calls = []

    def recorder(action, result=None):
        def record(scenes=None, **options):
            calls.append((action, scenes, options))
            return result

        return record

    monkeypatch.setattr(project, "render_screenshots", recorder("screenshots"))
    monkeypatch.setattr(project, "render_video", recorder("video"))
    monkeypatch.setattr(
        project, "validate", recorder("validate", ProjectValidation(()))
    )
    monkeypatch.setattr(project, "profile", recorder("profile", {}))
    estimate = RenderEstimate(20.0, 36.0, 8.0, (28.0, 48.0), {})
    monkeypatch.setattr(project, "estimate_render_time", recorder("estimate", estimate))
    monkeypatch.setattr(project, "concatenate_videos", recorder("concatenate"))
    return calls


def test_cli_video_settings_reach_every_action(monkeypatch, tmp_path):
    project = Project([walkthrough, epilogue], **_paths(tmp_path))
    calls = _record_actions(monkeypatch, project)
    for argv in (
        ["--render-screenshots", "--video-settings", "hd"],
        ["--render-video", "01", "--video-settings", "Smoke_Test"],
        ["--validate", "--video-settings", "LD"],
        ["--profile", "walkthrough", "--video-settings", "md"],
        ["--estimate-render-time", "0", "--video-settings", "preview"],
    ):
        assert project.run_cli(argv) is True
    assert calls == [
        ("screenshots", None, {"video_settings": HD}),
        ("video", (1,), {"video_settings": SMOKE_TEST}),
        ("validate", None, {"video_settings": LD}),
        ("profile", ("walkthrough",), {"runs": 2, "video_settings": MD}),
        ("estimate", (0,), {"runs": 2, "video_settings": PREVIEW}),
    ]


def test_cli_forwards_only_the_screenshot_flags_given(monkeypatch, tmp_path):
    project = Project([walkthrough, epilogue], **_paths(tmp_path))
    calls = _record_actions(monkeypatch, project)
    project.run_cli(
        [
            "--render-screenshots",
            "walkthrough",
            "1",
            "--frames",
            "a,b",
            "c",
            "--stop-early",
            "--contact-sheet",
            "sheet.png",
        ]
    )
    project.run_cli(["--render-screenshots", "--contact-sheet"])
    project.run_cli(["--render-screenshots"])
    assert calls == [
        (
            "screenshots",
            ("walkthrough", 1),
            {"frames": ("a,b", "c"), "stop_early": True, "contact_sheet": "sheet.png"},
        ),
        ("screenshots", None, {"contact_sheet": True}),
        ("screenshots", None, {}),
    ]


def test_cli_profile_runs_and_the_estimate_it_prints(monkeypatch, tmp_path, capsys):
    project = Project([walkthrough, epilogue], **_paths(tmp_path))
    calls = _record_actions(monkeypatch, project)
    assert project.run_cli(["--profile"]) is True
    assert project.run_cli(["--profile", "0", "epilogue", "--profile-runs", "3"])
    assert project.run_cli(
        ["--estimate-render-time", "walkthrough", "--profile-runs", "4"]
    )
    assert calls == [
        ("profile", None, {"runs": 2}),
        ("profile", (0, "epilogue"), {"runs": 3}),
        ("estimate", ("walkthrough",), {"runs": 4}),
    ]
    assert (
        "Estimated export: 44.0s; sampled range 28.0-48.0s. "
        "Observed startup overhead: 8.0s." in capsys.readouterr().out
    )


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--render-video", "--frames", "a"], "--frames and --stop-early only apply"),
        (["--render-video", "--stop-early"], "--frames and --stop-early only apply"),
        (["--validate", "--frames", "a"], "--frames and --stop-early only apply"),
        (["--profile", "--stop-early"], "--frames and --stop-early only apply"),
        (
            ["--estimate-render-time", "0", "--frames", "a"],
            "--frames and --stop-early only apply",
        ),
        (["--render-video", "--contact-sheet"], "--contact-sheet only applies"),
        (["--render-screenshots", "--script", "s.txt"], "--script only applies"),
        (["--render-video", "--video-settings", "4k"], "Unknown video settings"),
    ],
)
def test_cli_refuses_a_flag_its_action_does_not_take(
    monkeypatch, tmp_path, argv, message
):
    project = Project([walkthrough], **_paths(tmp_path))
    calls = _record_actions(monkeypatch, project)
    with pytest.raises(AlganConfigurationError, match=re.escape(message)):
        project.run_cli(argv)
    assert not calls


def test_cli_actions_exclude_each_other_and_help_exits_cleanly(
    monkeypatch, tmp_path, capsys
):
    project = Project([walkthrough], **_paths(tmp_path))
    calls = _record_actions(monkeypatch, project)
    with pytest.raises(SystemExit) as refused:
        project.run_cli(["--render-video", "--validate"])
    assert refused.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err

    with pytest.raises(SystemExit) as helped:
        project.run_cli(["--help"])
    assert helped.value.code == 0
    help_text = capsys.readouterr().out
    assert "--render-screenshots" in help_text
    assert all(name in help_text for name in _PRESETS_BY_NAME)

    assert project.run_cli([]) is False
    assert project.run_cli(["--video-settings", "HD"]) is False
    assert not calls


# -- construction ----------------------------------------------------------------


def test_scene_functions_may_be_any_iterable_with_optional_parameters(tmp_path):
    def intro(*args, title="Intro", **kwargs):
        pass

    def wide(width):
        pass

    project = Project(
        iter([intro, algan_scene(partial(wide, 3), name="wide")]),
        **_paths(tmp_path),
    )
    assert project.scene_names == ("0_intro", "1_wide")


def test_the_constructor_refuses_what_it_could_not_run(tmp_path):
    paths = _paths(tmp_path)

    def intro():
        pass

    with pytest.raises(AlganConfigurationError, match="nonempty iterable of callables"):
        Project(intro, **paths)  # the function itself, not a list of them

    def uninspectable():
        pass

    uninspectable.__signature__ = 42
    with pytest.raises(AlganConfigurationError, match="Could not inspect .* entry 1"):
        Project([intro, uninspectable], **paths)

    def keyword_only(*, take):
        pass

    with pytest.raises(
        AlganConfigurationError, match="entry 0 requires arguments: take"
    ):
        Project([keyword_only], **paths)

    def wide(width):
        pass

    with pytest.raises(AlganConfigurationError, match="does not have a valid name"):
        Project([partial(wide, 3)], **paths)

    for length in (True, 2.5, -1):
        with pytest.raises(AlganConfigurationError, match="positive integer"):
            Project([intro], transcript_line_length=length, **paths)

    def effect(frames, **kwargs):
        return frames

    with pytest.raises(AlganConfigurationError, match="not one pass"):
        Project([intro], post_processes=effect, **paths)
    with pytest.raises(AlganConfigurationError, match=r"post_processes\[1\] is not"):
        Project([intro], post_processes=[effect, 3], **paths)


def test_tilde_in_an_output_directory_is_the_home_directory(monkeypatch, tmp_path):
    """REGRESSION. ``~`` in a Project directory was a directory named ``~``.

    ``Scene.save_video`` and ``save_frame`` expand ``~`` in their own paths, but
    ``Project`` did not expand its directories: ``transcript_directory="~/words"``
    wrote the transcripts into a literal ``./~/words`` under the working
    directory, ``concatenate_videos`` looked for the scene videos under
    ``./~/vids`` while the videos themselves were written to the home
    directory, and the directories the Project reported were the unexpanded
    strings.
    """
    home = tmp_path / "home"
    work = tmp_path / "work"
    home.mkdir()
    work.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.chdir(work)

    def intro():
        with Speech("Hello there.", wait_at_end=0):
            Scene.wait(1)

    project = Project(
        [intro],
        file_path=tmp_path / "combined.mp4",
        video_directory="~/vids",
        screenshot_directory="~/shots",
        transcript_directory="~/words",
        speech_source=lambda _: _SilentClip(),
    )
    assert project.video_directory == home / "vids"
    assert project.screenshot_directory == home / "shots"
    assert project.transcript_directory == home / "words"

    project.validate()
    transcript = home / "words" / "0_intro.txt"
    assert transcript.read_text(encoding="utf-8") == "Hello there.\n"
    assert (home / "words" / "transcript.txt").exists()
    assert not (work / "~").exists()

    joined = []
    monkeypatch.setattr(
        algan_utils,
        "concatenate_videos",
        lambda directory, **options: joined.append((directory, options)),
    )
    project.concatenate_videos()
    [(directory, options)] = joined
    assert Path(directory) == home / "vids"
    assert [Path(p) for p in options["input_files"]] == [
        (home / "vids" / "0_intro.mp4").resolve()
    ]
