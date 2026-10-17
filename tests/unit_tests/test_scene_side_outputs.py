"""A Scene's outputs besides its video: the audio track and a displayed frame.

``Scene.save_audio`` mixes the recorded audio down on its own, and
``Scene.show_frame`` renders one frame and shows it with matplotlib. Both are
a script asking for output, which is what decides whether a plain
``python scene.py`` run is told at exit that it "finished without rendering
anything" -- so that report is tested here end to end too, in a subprocess,
since it only arms itself for a real script run.

A feature test, so it sits outside the fast suite: it only breaks when these
outputs themselves change.
"""

from __future__ import annotations

import math
import os
import struct
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import torch

from algan import RED, RIGHT, SMOKE_TEST, Off, Scene, Square
from algan.animation_timeline.animation_contexts import Audio
from algan.scene import renders_requested

moviepy = pytest.importorskip("moviepy")

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTHING_RENDERED = "finished without rendering anything"


def _tone(seconds: float, rate: int = 44100):
    samples = int(rate * seconds)
    wave_ = np.sin(np.linspace(0, 2 * math.pi * 440 * seconds, samples)) * 0.5
    return moviepy.AudioArrayClip(wave_.reshape(-1, 1).repeat(2, 1), fps=rate)


def _read_wav(path):
    """A 32-bit PCM file as ``(frame rate, [samples, channels] in [-1, 1])``.

    Read by hand: FFmpeg writes 32-bit PCM as ``WAVE_FORMAT_EXTENSIBLE``, which
    the standard library's ``wave`` reads only from Python 3.12.
    """
    data = Path(path).read_bytes()
    assert data[:4] == b"RIFF"
    assert data[8:12] == b"WAVE"
    chunks, offset = {}, 12
    while offset + 8 <= len(data):
        name = data[offset : offset + 4]
        (size,) = struct.unpack_from("<I", data, offset + 4)
        chunks.setdefault(name, data[offset + 8 : offset + 8 + size])
        offset += 8 + size + size % 2
    fmt = chunks[b"fmt "]
    tag, channels, rate = struct.unpack_from("<HHI", fmt)
    bits = struct.unpack_from("<H", fmt, 14)[0]
    if tag == 0xFFFE:
        # The extensible header names the real format in its sub-format GUID.
        tag = struct.unpack_from("<H", fmt, 24)[0]
    assert (tag, bits) == (1, 32)
    samples = np.frombuffer(chunks[b"data"], dtype="<i4").reshape(-1, channels)
    return rate, samples.astype(np.float64) / 2**31


def _scene_with_a_clip_at_one_second():
    """One second of silence, half a second of tone, half a second of silence."""
    square = Square()
    with Off():
        square.spawn()
    square.wait(1)
    with Audio(_tone(0.5)):
        square.move(RIGHT)
    square.wait(0.5)


@pytest.fixture
def scene():
    with Scene() as active:
        yield active


def test_a_scene_without_audio_writes_no_file(scene, tmp_path):
    Square().spawn()
    target = tmp_path / "silent.wav"

    assert scene.save_audio(target) is None
    assert not target.exists()


def test_a_clip_lands_at_the_time_it_was_recorded(scene, tmp_path):
    _scene_with_a_clip_at_one_second()
    assert [effect.start_time_func() for effect in scene.effects] == [1.0]

    target = tmp_path / "mix.wav"
    assert scene.save_audio(target) == target
    rate, samples = _read_wav(target)

    assert rate == 44100
    # The mix spans the whole Scene, not just the clip.
    assert len(samples) == pytest.approx(2.0 * rate, abs=1)
    loudness = np.abs(samples).max(-1)
    assert loudness[: int(0.98 * rate)].max() == 0
    assert loudness[int(1.02 * rate) : int(1.48 * rate)].max() > 0.4
    assert loudness[int(1.52 * rate) :].max() == 0


def test_the_mix_is_written_at_the_sample_rate_asked_for(scene, tmp_path):
    _scene_with_a_clip_at_one_second()
    target = tmp_path / "low.wav"
    scene.save_audio(target, sample_rate=22050)

    rate, samples = _read_wav(target)
    assert rate == 22050
    assert len(samples) == pytest.approx(2.0 * 22050, abs=1)


def test_save_audio_on_the_class_mixes_the_active_scene(scene, tmp_path):
    """REGRESSION. ``save_audio`` and ``show_frame`` were plain methods, so
    the class-level spelling every other output call takes --
    ``Scene.save_video()`` -- raised ``TypeError: missing 1 required
    positional argument``.
    """
    _scene_with_a_clip_at_one_second()
    target = tmp_path / "mix.wav"

    assert Scene.save_audio(target) == target
    assert _read_wav(target)[0] == 44100


def test_saving_the_audio_counts_as_asking_for_output(scene, tmp_path):
    """REGRESSION. ``save_audio`` was not counted as a request for output,
    unlike ``save_subtitles``, so a script ending in ``Scene.save_audio(...)``
    -- the documented way to get the audio on its own -- was told at exit
    that it never rendered anything and should add ``save_video``.
    """
    Square().spawn()
    before = renders_requested()
    scene.save_audio(tmp_path / "silent.wav")
    assert renders_requested() > before


def test_show_frame_displays_the_frame_it_rendered(scene, monkeypatch):
    """REGRESSION. The frame went to ``plot_tensor`` with its default
    ``norm=True``, which stretches an image to its own minimum and maximum:
    every color shown was shifted, and a frame of one flat color became 0/0.
    It also did not count as a request for output, so a script previewing a
    frame was told at exit that it had rendered nothing.
    """
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib import pyplot
    from matplotlib.axes import Axes

    shown = []
    imshow = Axes.imshow

    def record(axes, image, *args, **kwargs):
        shown.append(np.asarray(image, dtype=np.float64))
        return imshow(axes, image, *args, **kwargs)

    monkeypatch.setattr(Axes, "imshow", record)
    monkeypatch.setattr(pyplot, "pause", lambda *args, **kwargs: None)
    monkeypatch.setattr(pyplot, "show", lambda *args, **kwargs: None)
    scene.set_video_settings(SMOKE_TEST)
    Square(color=RED).scale(2).spawn()

    before = renders_requested()
    frames = scene.show_frame()
    try:
        assert renders_requested() > before
        assert len(frames) == len(shown) == 1
        frame = frames[0]
        assert frame.shape == (3, *reversed(SMOKE_TEST.resolution))
        assert float(frame.min()) >= 0
        # Red on a dark background: both are in the frame, and neither is
        # pure white, so a stretch to the frame's own range would show.
        assert float(frame.max()) < 1
        np.testing.assert_allclose(shown[0], frame.permute(1, 2, 0).numpy())

        # Showing a frame leaves the Scene as authored, and the class-level
        # spelling shows the active Scene.
        assert torch.equal(Scene.show_frame()[0], frame)
    finally:
        pyplot.close("all")


def _write_silence(path, seconds):
    from scipy.io import wavfile

    rate = 8000
    wavfile.write(path, rate, np.zeros(int(rate * seconds), dtype=np.int16))
    return path


def test_an_audio_file_beside_the_running_script_is_found(scene, tmp_path, monkeypatch):
    """REGRESSION. Every image, model and environment map path is looked for
    beside the script when the working directory does not have it, which is
    what lets a script load the file next to it from wherever it is launched.
    ``Audio`` opened the path as given, so ``Audio("whoosh.wav")`` failed
    unless Python was started in the script's directory.
    """
    from algan.settings import path_settings

    project = tmp_path / "project"
    project.mkdir()
    script = project / "scene.py"
    script.write_text("")
    _write_silence(project / "whoosh.wav", 0.5)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    monkeypatch.setattr(path_settings, "_main_script_path", lambda: str(script))

    context = Audio("whoosh.wav")
    with context:
        Square().spawn()
    assert context.runtime == pytest.approx(0.5, abs=0.01)


def test_an_audio_path_may_be_a_pathlib_path(scene, tmp_path):
    """REGRESSION. Only a ``str`` was opened: a :class:`~pathlib.Path` was
    taken for a clip already loaded, and failed with "'PosixPath' object has
    no attribute 'duration'".
    """
    context = Audio(_write_silence(tmp_path / "whoosh.wav", 0.5))
    with context:
        Square().spawn()
    assert context.runtime == pytest.approx(0.5, abs=0.01)


def _run_script(tmp_path, source: str) -> subprocess.CompletedProcess:
    script = tmp_path / "scene_script.py"
    script.write_text(textwrap.dedent(source))
    # The report arms itself only for a plain script run: not under pytest,
    # and not handed to a render daemon, where atexit handlers do not run.
    # pytest-cov's subprocess hook, armed by its COV_CORE_ variables, imports
    # pytest into the child, so it goes too. So does ALGAN_DAEMON_CHILD:
    # importing algan.daemon sets it, the daemon tests import it at
    # collection, and a child that inherits it runs as a daemon's script.
    env = {
        name: value
        for name, value in os.environ.items()
        if name not in ("PYTEST_CURRENT_TEST", "ALGAN_DAEMON_CHILD")
        and not name.startswith("COV_CORE_")
    }
    env["ALGAN_USE_DAEMON"] = "0"
    env["ALGAN_AUTO_DAEMON"] = "0"
    # Import the checkout under test, not whichever one is installed.
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT), *filter(None, [env.get("PYTHONPATH")])]
    )
    return subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )


def test_a_script_that_renders_nothing_is_told_so_at_exit(tmp_path):
    result = _run_script(
        tmp_path,
        """
        from algan import *

        Square().spawn()
        """,
    )
    assert result.returncode == 0, result.stderr
    assert NOTHING_RENDERED in result.stderr
    assert 'Scene.save_video("scene_script")' in result.stderr


def test_a_script_that_only_saves_its_audio_is_not_told_it_rendered_nothing(
    tmp_path,
):
    """REGRESSION. See ``test_saving_the_audio_counts_as_asking_for_output``."""
    result = _run_script(
        tmp_path,
        """
        import numpy as np
        from moviepy import AudioArrayClip
        from algan import *
        from algan.animation_timeline.animation_contexts import Audio

        tone = AudioArrayClip(np.zeros((4410, 2)) + 0.25, fps=44100)
        square = Square().spawn()
        with Audio(tone):
            square.move(RIGHT)
        Scene.save_audio("narration.wav")
        """,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "narration.wav").exists()
    assert NOTHING_RENDERED not in result.stderr


def test_a_script_that_crashes_is_not_also_told_it_rendered_nothing(tmp_path):
    result = _run_script(
        tmp_path,
        """
        from algan import *

        Square().spawn()
        raise RuntimeError("the script's own mistake")
        """,
    )
    assert result.returncode != 0
    assert "the script's own mistake" in result.stderr
    assert NOTHING_RENDERED not in result.stderr
