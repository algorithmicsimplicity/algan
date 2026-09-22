"""Subtitle exports exercise the authored timeline without rendering or TTS."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from algan import SETTINGS, SMOKE_TEST, Off, Project, Scene, Seq, Speech
from algan.errors import AlganConfigurationError


def _cues(path):
    blocks = path.read_text(encoding="utf-8").strip().split("\n\n")
    result = []
    for block in blocks:
        if not block or block == "WEBVTT":
            continue
        _, timing, *lines = block.splitlines()
        start, end = timing.split(" --> ")

        def seconds(timestamp):
            hours, minutes, seconds = timestamp.replace(",", ".").split(":")
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds)

        result.append((seconds(start), seconds(end), "\n".join(lines)))
    return result


def _source(scene, *, duration, timestamps=None):
    calls = []

    def generate(text):
        calls.append(text)
        return SimpleNamespace(duration=duration, algan_word_timestamps=timestamps)

    scene.audio_manager.set_speech_source(generate)
    return calls


@pytest.mark.parametrize("suffix", ["srt", "vtt"])
def test_manual_captions_serialize_as_literal_unicode(tmp_path, fresh_scene, suffix):
    Scene.add_subcaption("  x < y & z --> α\r\n\r\nSecond line  ", duration=2)
    Scene.wait(3)
    path = Scene.save_subtitles(tmp_path / f"caption.{suffix}")
    decimal = "," if suffix == "srt" else "."
    header = "" if suffix == "srt" else "WEBVTT\n\n"
    assert path.read_bytes() == (
        header
        + f"1\n00:00:00{decimal}000 --> 00:00:02{decimal}000\n"
        + "x &lt; y &amp; z --&gt; α\nSecond line\n\n"
    ).encode("utf-8")
    assert path.is_absolute()


@pytest.mark.fast
def test_caption_cursors_follow_context_rescaling_and_scene_ownership(
    tmp_path, fresh_scene
):
    scene = Scene.current()
    Scene.wait(1)
    with Seq(runtime=8):
        Scene.wait(1)
        assert Scene.add_subcaption("nested", duration=1, offset=0.25) is scene
        Scene.wait(3)
    assert scene._recorded_end_time_for_render() == 9
    with Scene() as other:
        other.add_subcaption("other", duration=2)
        other.wait(2)
        # Instance dispatch must not attach the caption to the active Scene.
        scene.add_subcaption("root", duration=1, offset=-1)
        assert _cues(scene.save_subtitles(tmp_path / "first.srt")) == [
            (3.25, 4.25, "nested"),
            (8, 9, "root"),
        ]
        assert _cues(other.save_subtitles(tmp_path / "other.srt")) == [(0, 2, "other")]
        other.reset()
        assert other.audio_manager._subcaptions == []
    assert _cues(scene.save_subtitles(tmp_path / "again.srt")) == [
        (3.25, 4.25, "nested"),
        (8, 9, "root"),
    ]
    assert scene._recorded_end_time_for_render() == 9


def test_captions_clip_sort_and_overlap_without_extending_scene(tmp_path, fresh_scene):
    Scene.add_subcaption("late", duration=20, offset=2)
    Scene.add_subcaption("before", duration=2, offset=-1)
    Scene.add_subcaption("same start", duration=0.5)
    Scene.add_subcaption("outside", duration=1, offset=5)
    Scene.add_subcaption("expired", duration=1, offset=-3)
    assert Scene.current()._recorded_end_time_for_render() == 0
    Scene.wait(3)
    assert _cues(Scene.save_subtitles(tmp_path / "clipped.srt")) == [
        (0, 1, "before"),
        (0, 0.5, "same start"),
        (2, 3, "late"),
    ]


def test_timestamp_rounding_handles_rollover_and_short_cues(tmp_path, fresh_scene):
    Scene.add_subcaption("tiny", duration=0.0001)
    Scene.wait(3599.9996)
    Scene.add_subcaption("hour", duration=0.9998)
    Scene.wait(1)
    data = Scene.save_subtitles(tmp_path / "rounding.srt").read_text()
    assert "00:00:00,000 --> 00:00:00,001" in data
    assert "01:00:00,000 --> 01:00:00,999" in data


@pytest.mark.parametrize(
    "kwargs",
    [
        {"content": ""},
        {"content": " \n\t "},
        {"content": None},
        {"content": "a\0b"},
        {"duration": 0},
        {"duration": -1},
        {"duration": float("nan")},
        {"duration": True},
        {"offset": float("inf")},
        {"offset": "not a time"},
    ],
)
def test_invalid_caption_does_not_record_anything(fresh_scene, kwargs):
    with pytest.raises(AlganConfigurationError):
        Scene.add_subcaption(**({"content": "words"} | kwargs))
    assert Scene.current().audio_manager._subcaptions == []
    assert Scene.current()._recorded_end_time_for_render() == 0


def test_speech_uses_alignment_without_stretching_or_regenerating_audio(
    tmp_path, fresh_scene
):
    scene = Scene.current()
    calls = _source(
        scene,
        duration=2,
        timestamps=[("HELLO", 0.1, 0.4), ("WORLD", 1.1, 1.5)],
    )
    with Seq(runtime=12):
        Scene.wait(2)
        with Speech("🙂 Hello, world!", wait_at_end=2):
            Scene.wait(1)
    assert _cues(Scene.save_subtitles(tmp_path / "speech.srt")) == [
        (4.1, 5.5, "🙂 Hello, world!")
    ]
    assert _cues(Scene.save_subtitles(tmp_path / "short.vtt", max_duration=0.75)) == [
        (4.1, 4.4, "🙂 Hello,"),
        (5.1, 5.5, "world!"),
    ]
    assert calls == ["🙂 Hello, world!"]
    assert scene._recorded_end_time_for_render() == 12


def test_speech_estimates_wrap_and_preserve_authored_breaks(tmp_path, fresh_scene):
    _source(Scene.current(), duration=7)
    with Speech("aa bb cc\ndd\n\nee ff superlongword", wait_at_end=2):
        Scene.wait(1)
    cues = _cues(
        Scene.save_subtitles(
            tmp_path / "wrapped.srt", max_chars_per_line=5, max_lines=2
        )
    )
    assert [text for _, _, text in cues] == ["aa bb\ncc", "dd", "ee ff\nsuperlongword"]
    assert cues[0][0] == 0
    assert cues[-1][1] == 7  # exclude the two-second hold
    one_line = _cues(
        Scene.save_subtitles(
            tmp_path / "one_line.srt", max_chars_per_line=5, max_lines=1
        )
    )
    assert [text for _, _, text in one_line] == [
        "aa bb",
        "cc",
        "dd",
        "ee ff",
        "superlongword",
    ]


def test_clipping_omits_whole_words_and_their_attached_punctuation(
    tmp_path, fresh_scene
):
    # Custom audio origins may precede a view's time range. Zero-length aligned
    # words are also inaudible; neither should return via a substring operation.
    from algan.sound.transcript import _SpeechBlock

    entry = _SpeechBlock.from_clip(
        "🙂 early — kept absent last!",
        SimpleNamespace(
            duration=3,
            algan_word_timestamps=[
                ("early", 0, 1),
                ("kept", 1, 2),
                ("absent", 2, 2),
                ("last", 2, 3),
            ],
        ),
    )
    entry.start_time_func = lambda: -1
    Scene.current().audio_manager._speech_blocks.append(entry)
    Scene.wait(1.5)
    assert _cues(Scene.save_subtitles(tmp_path / "trim.srt")) == [
        (0, 1.5, "kept last!")
    ]


def test_paragraph_break_before_punctuation_starts_a_new_cue(tmp_path, fresh_scene):
    _source(Scene.current(), duration=2)
    with Speech("One.\n\n— Two.", wait_at_end=0):
        Scene.wait(1)
    assert _cues(Scene.save_subtitles(tmp_path / "paragraphs.srt")) == [
        (0, 1, "One."),
        (1, 2, "— Two."),
    ]


def test_unavailable_speech_is_omitted_and_manual_only_is_supported(
    tmp_path, fresh_scene
):
    _source(Scene.current(), duration=1)
    with Off(), Speech("suppressed"):
        pass
    Speech("never entered")
    with Speech("audible", wait_at_end=0):
        Scene.wait(1)
    Scene.add_subcaption("translation", duration=1, offset=-1)
    assert [cue[2] for cue in _cues(Scene.save_subtitles(tmp_path / "all.srt"))] == [
        "audible",
        "translation",
    ]
    assert _cues(
        Scene.save_subtitles(tmp_path / "manual.vtt", include_speech=False)
    ) == [(0, 1, "translation")]


def test_empty_outputs_have_the_appropriate_header(tmp_path, fresh_scene):
    assert Scene.save_subtitles(tmp_path / "empty.srt").read_bytes() == b""
    assert Scene.save_subtitles(tmp_path / "empty.vtt").read_bytes() == b"WEBVTT\n\n"


def test_subtitle_path_rules_and_overwrite(tmp_path, fresh_scene, monkeypatch):
    monkeypatch.setattr(SETTINGS.paths, "output_root", tmp_path)
    monkeypatch.setattr(SETTINGS.paths, "output_directory", "exports")
    monkeypatch.setattr(SETTINGS.paths, "output_filename", "lesson.mp4")
    Scene.add_subcaption("first")
    Scene.wait(1)
    default = Scene.save_subtitles()
    assert default == tmp_path / "exports" / "lesson.srt"
    assert Scene.save_subtitles(subtitle_format="VTT") == (
        tmp_path / "exports" / "lesson.vtt"
    )
    assert Scene.save_subtitles("explicit.VTT").suffix == ".VTT"
    assert Scene.save_subtitles("bare") == tmp_path / "exports" / "bare.srt"
    assert Scene.save_subtitles(str(tmp_path / "directory") + "/") == (
        tmp_path / "directory" / "lesson.srt"
    )
    before = default.read_bytes()
    Scene.add_subcaption("later")
    Scene.wait(1)
    assert Scene.save_subtitles(overwrite=False) == default
    assert default.read_bytes() == before
    Scene.save_subtitles()
    assert "later" in default.read_text()


@pytest.mark.parametrize(
    "options",
    [
        {"subtitle_format": "ass"},
        {"subtitle_format": ""},
        {"subtitle_format": "vtt"},  # conflicts with .srt
        {"max_lines": 0},
        {"max_lines": True},
        {"max_chars_per_line": 2.5},
        {"max_duration": 0},
        {"max_duration": float("nan")},
        {"include_speech": "yes"},
    ],
)
def test_invalid_export_does_not_overwrite_file(tmp_path, fresh_scene, options):
    path = tmp_path / "existing.srt"
    path.write_text("keep")
    with pytest.raises(AlganConfigurationError):
        Scene.save_subtitles(path, **options)
    assert path.read_text() == "keep"


def _project(functions, tmp_path, **kwargs):
    return Project(
        functions,
        file_path=tmp_path / "lesson.mp4",
        video_directory=tmp_path / "videos",
        screenshot_directory=tmp_path / "frames",
        transcript_directory=tmp_path / "transcripts",
        **kwargs,
    )


def test_project_offsets_include_silent_scenes_and_speech_holds(
    tmp_path, fresh_scene, monkeypatch
):
    from algan.utils import algan_utils

    calls = []

    def generate(text):
        calls.append(text)
        return SimpleNamespace(duration=2)

    def intro():
        with Speech("Welcome!", wait_at_end=1):
            Scene.wait(1)
        Scene.save_video(tmp_path / "ignored.mp4")
        Scene.save_frame(tmp_path / "ignored.png")
        Scene.save_subtitles(tmp_path / "ignored.srt")
        with Scene() as nested:
            nested.save_subtitles(tmp_path / "nested.srt")

    def silent():
        Scene.wait(2)

    def ending():
        Scene.add_subcaption("The end", duration=1)
        Scene.wait(1)

    monkeypatch.setattr(
        algan_utils,
        "_render_scene_to_file",
        lambda *a, **k: pytest.fail("subtitle export rendered a video"),
    )
    monkeypatch.setattr(
        Scene,
        "_render_still",
        lambda *a, **k: pytest.fail("subtitle export rendered a still"),
    )
    original = Scene.current()
    original.add_subcaption("outside")
    original.wait(1)
    project = _project([intro, silent, ending], tmp_path, speech_source=generate)
    path = project.save_subtitles()
    assert path == tmp_path / "lesson.srt"
    assert _cues(path) == [(0, 2, "Welcome!"), (5, 6, "The end")]
    assert calls == ["Welcome!"]
    assert Scene.current() is original
    assert original._recorded_end_time_for_render() == 1
    assert len(original.audio_manager._subcaptions) == 1
    assert project.global_transcript_path.read_text().strip() == "Welcome!"
    assert not (tmp_path / "ignored.srt").exists()
    assert not (tmp_path / "nested.srt").exists()
    assert not (tmp_path / "ignored.mp4").exists()
    # Reversed, repeated and generator selectors still use project order.
    selected = project.save_subtitles(
        tmp_path / "selected.vtt", scenes=(i for i in ["ending", 0, 0])
    )
    assert _cues(selected) == [(0, 2, "Welcome!"), (3, 4, "The end")]


def test_project_offsets_match_video_frame_rounding(tmp_path, fresh_scene):
    def short():
        # Project video export passes its effective settings explicitly, even
        # if a scene function selects a different authoring frame rate.
        Scene.set_video_settings(SMOKE_TEST.set(frames_per_second=20))
        Scene.wait(0.26)

    def zero():
        from algan import Circle

        with Off():
            Circle().spawn()  # save_video keeps one frame of visible geometry

    def caption():
        Scene.add_subcaption("after", duration=1)
        Scene.wait(1)

    project = _project([short, zero, caption], tmp_path)
    result = project.save_subtitles(video_settings=SMOKE_TEST.set(frames_per_second=10))
    assert _cues(result) == [(0.4, 1.4, "after")]


@pytest.mark.parametrize("fade_out", [True, False, None])
def test_project_offsets_include_requested_final_fades(
    tmp_path, fresh_scene, monkeypatch, fade_out
):
    from algan import Circle

    monkeypatch.setattr(SETTINGS.style, "fade_out_on_scene_end", True)

    def first():
        with Off():
            Circle().spawn()
        Scene.wait(1)

    def second():
        Scene.add_subcaption("next")
        Scene.wait(1)

    project = _project([first, second], tmp_path)
    path = project.save_subtitles(
        animate_fade_out=fade_out,
        video_settings=SMOKE_TEST.set(frames_per_second=10),
    )
    start = 1.0 if fade_out is False else 1.5
    assert _cues(path) == [(start, start + 1, "next")]


def test_project_overwrite_skip_validation_and_error_cleanup(tmp_path, fresh_scene):
    def broken():
        raise ValueError("authoring failed")

    project = _project([broken], tmp_path)
    path = tmp_path / "lesson.srt"
    path.write_text("keep")
    assert project.save_subtitles(overwrite=False) == path
    with pytest.raises(AlganConfigurationError):
        project.save_subtitles(scenes="missing", overwrite=False)
    original = Scene.current()
    with pytest.raises(ValueError, match="authoring failed"):
        project.save_subtitles()
    assert path.read_text() == "keep"
    assert Scene.current() is original
    Scene.add_subcaption("still usable")
    Scene.wait(1)
    assert _cues(Scene.save_subtitles(tmp_path / "after_error.srt")) == [
        (0, 1, "still usable")
    ]
