"""Tests for automatic video-encoder selection (``select_video_encoder``)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from algan.utils import video_encoding
from algan.utils.video_encoding import select_video_encoder

SOFTWARE_PAIR = ("libx264", ["-crf", "17", "-preset", "slower"])
NVENC_PAIR = (
    "h264_nvenc",
    [
        "-preset",
        "p4",
        "-tune",
        "hq",
        "-rc",
        "vbr",
        "-cq",
        "19",
        "-b:v",
        "0",
        "-pix_fmt",
        "yuv420p",
    ],
)


@pytest.fixture(autouse=True)
def _fresh_probe_and_no_env(monkeypatch):
    """Each test starts with an unrun probe and no ALGAN_VIDEO_ENCODER set."""
    monkeypatch.setattr(video_encoding, "_probe_cache", None)
    monkeypatch.delenv("ALGAN_VIDEO_ENCODER", raising=False)


def _patch_probe(monkeypatch, usable, calls=None):
    """Stub the uncached probe: usable, via a binary worth naming."""
    binary = "/probed/nvenc-ffmpeg" if usable else None

    def probe():
        if calls is not None:
            calls.append(1)
        return usable, binary

    monkeypatch.setattr(video_encoding, "_probe_once", probe)


def _patch_candidates(monkeypatch, binaries, verdicts, calls=None):
    """Stub the candidate list and per-binary usability, keeping the real
    first-usable-wins walk of ``_probe_once`` under test.
    """

    def usable(binary):
        if calls is not None:
            calls.append(binary)
        return verdicts[binary]

    monkeypatch.setattr(video_encoding, "_candidate_binaries", lambda: list(binaries))
    monkeypatch.setattr(video_encoding, "_binary_usable", usable)


def _patch_subprocess_probe(monkeypatch, capabilities):
    """Drive the real ``_binary_usable`` through a fake subprocess.run.

    ``capabilities`` maps binary -> dict with optional keys "lists_nvenc"
    (default False) and "encodes" (default False); binaries absent from the
    mapping fail to launch at all.
    """

    def run(cmd, **kwargs):
        binary = cmd[0]
        caps = capabilities.get(binary)
        if caps is None:
            raise OSError(f"cannot run {binary}")
        if "-encoders" in cmd:
            listing = (
                " V..... h264_nvenc           NVIDIA NVENC H.264 encoder\n"
                if caps.get("lists_nvenc")
                else " V..... libx264              libx264 H.264\n"
            )
            return SimpleNamespace(returncode=0, stdout=listing, stderr="")
        return SimpleNamespace(
            returncode=0 if caps.get("encodes") else 1,
            stdout=b"",
            stderr=b"InitializeEncoder failed",
        )

    monkeypatch.setattr(video_encoding.subprocess, "run", run)


def test_auto_with_unusable_probe_returns_todays_software_pair(
    monkeypatch,
):
    _patch_probe(monkeypatch, usable=False)
    assert select_video_encoder(None, None, False) == SOFTWARE_PAIR


def test_software_mode_never_probes(monkeypatch):
    calls = []
    _patch_probe(monkeypatch, usable=True, calls=calls)
    monkeypatch.setenv("ALGAN_VIDEO_ENCODER", "software")
    assert select_video_encoder(None, None, False) == SOFTWARE_PAIR
    assert calls == []


def test_nvenc_mode_forces_h264_nvenc_without_probing(monkeypatch):
    calls = []
    _patch_probe(monkeypatch, usable=False, calls=calls)
    monkeypatch.setenv("ALGAN_VIDEO_ENCODER", "nvenc")
    assert select_video_encoder(None, None, False) == NVENC_PAIR
    # Forced mode trusts the caller; the unusable probe above proves it
    # never consulted the probe.
    assert calls == []


def test_explicit_codec_passes_through_untouched(monkeypatch):
    _patch_probe(monkeypatch, usable=True)
    monkeypatch.setenv("ALGAN_VIDEO_ENCODER", "nvenc")
    assert select_video_encoder("libx264rgb", ["-crf", "0"], False) == (
        "libx264rgb",
        ["-crf", "0"],
    )
    assert select_video_encoder("png", [], True) == ("png", [])


def test_probe_result_is_cached_across_calls(monkeypatch):
    calls = []
    _patch_probe(monkeypatch, usable=True, calls=calls)
    assert select_video_encoder(None, None, False) == NVENC_PAIR
    assert select_video_encoder(None, None, False) == NVENC_PAIR
    assert select_video_encoder(None, ["-movflags", "+faststart"], False) == (
        "h264_nvenc",
        ["-movflags", "+faststart"],
    )
    assert len(calls) == 1


def test_caller_x264_params_keep_the_software_encoder(monkeypatch):
    """-preset/-crf are x264 rate control: honoured on libx264, never mixed."""
    _patch_probe(monkeypatch, usable=True)
    params = ["-crf", "18"]
    assert select_video_encoder(None, params, False) == ("libx264", params)
    monkeypatch.setenv("ALGAN_VIDEO_ENCODER", "nvenc")
    assert select_video_encoder(None, ["-preset", "slow"], False) == (
        "libx264",
        ["-preset", "slow"],
    )


def test_transparent_output_is_untouched(monkeypatch):
    _patch_probe(monkeypatch, usable=True)
    assert select_video_encoder(None, None, True) == (None, None)


def test_unknown_value_warns_and_behaves_as_auto(monkeypatch):
    calls = []
    _patch_probe(monkeypatch, usable=False, calls=calls)
    monkeypatch.setenv("ALGAN_VIDEO_ENCODER", "nveng")
    with pytest.warns(Warning, match="ALGAN_VIDEO_ENCODER='nveng' is not one of"):
        assert select_video_encoder(None, None, False) == SOFTWARE_PAIR


def test_moviepy_binary_without_nvenc_system_with_nvenc_picks_system(monkeypatch):
    calls = []
    _patch_candidates(
        monkeypatch,
        ["/fake/moviepy-ffmpeg", "/usr/bin/ffmpeg"],
        {"/fake/moviepy-ffmpeg": False, "/usr/bin/ffmpeg": True},
        calls,
    )
    assert select_video_encoder(None, None, False) == NVENC_PAIR
    # Moviepy's own binary was tried first and rejected...
    assert calls == ["/fake/moviepy-ffmpeg", "/usr/bin/ffmpeg"]
    # ...so the writer must run the system binary that won, not moviepy's.
    assert video_encoding.resolve_encode_binary("h264_nvenc") == "/usr/bin/ffmpeg"


def test_no_candidate_usable_software_stays_on_moviepys_binary(monkeypatch):
    calls = []
    _patch_candidates(
        monkeypatch,
        ["/fake/moviepy-ffmpeg", "/usr/bin/ffmpeg"],
        {"/fake/moviepy-ffmpeg": False, "/usr/bin/ffmpeg": False},
        calls,
    )
    assert select_video_encoder(None, None, False) == SOFTWARE_PAIR
    # Software encoding never moves the binary off moviepy's configuration;
    # even an NVENC question comes back empty rather than guessing.
    assert video_encoding.resolve_encode_binary("libx264") is None
    assert video_encoding.resolve_encode_binary("h264_nvenc") is None
    # One probe served all three queries.
    assert len(calls) == 2


def test_first_usable_candidate_wins(monkeypatch):
    _patch_candidates(
        monkeypatch,
        ["/env/ffmpeg", "/fake/moviepy-ffmpeg", "/usr/bin/ffmpeg"],
        dict.fromkeys(("/env/ffmpeg", "/fake/moviepy-ffmpeg", "/usr/bin/ffmpeg"), True),
    )
    assert select_video_encoder(None, None, False) == NVENC_PAIR
    assert video_encoding.resolve_encode_binary("h264_nvenc") == "/env/ffmpeg"


def test_candidate_order_is_env_then_moviepy_then_path(monkeypatch):
    monkeypatch.setattr(video_encoding, "_moviepy_ffmpeg_binary", lambda: "/env/ffmpeg")
    monkeypatch.setattr(video_encoding.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    # With the variable set, moviepy resolves to it too: probed once.
    monkeypatch.setenv("FFMPEG_BINARY", "/env/ffmpeg")
    assert video_encoding._candidate_binaries() == [
        "/env/ffmpeg",
        "/usr/bin/ffmpeg",
    ]
    # Without it, moviepy's configured binary comes before the PATH.
    monkeypatch.delenv("FFMPEG_BINARY", raising=False)
    monkeypatch.setattr(
        video_encoding, "_moviepy_ffmpeg_binary", lambda: "/moviepy/ffmpeg"
    )
    assert video_encoding._candidate_binaries() == [
        "/moviepy/ffmpeg",
        "/usr/bin/ffmpeg",
    ]


def test_unusable_candidates_are_skipped_by_the_real_check(monkeypatch):
    """The launch failure and both usability failures route through DEBUG
    logging and fall through to the working candidate.
    """
    monkeypatch.setattr(
        video_encoding,
        "_candidate_binaries",
        lambda: [
            "/bin/missing-ffmpeg",
            "/fake/no-nvenc",
            "/fake/broken-nvenc",
            "/usr/bin/ffmpeg",
        ],
    )
    _patch_subprocess_probe(
        monkeypatch,
        {
            "/fake/no-nvenc": {"lists_nvenc": False},
            "/fake/broken-nvenc": {"lists_nvenc": True, "encodes": False},
            "/usr/bin/ffmpeg": {"lists_nvenc": True, "encodes": True},
        },
    )
    assert video_encoding._probe_cached() == (True, "/usr/bin/ffmpeg")


def test_forced_nvenc_mode_runs_a_usable_binary_when_one_exists(monkeypatch):
    _patch_probe(monkeypatch, usable=True)
    monkeypatch.setenv("ALGAN_VIDEO_ENCODER", "nvenc")
    assert select_video_encoder(None, None, False) == NVENC_PAIR
    assert video_encoding.resolve_encode_binary("h264_nvenc") == "/probed/nvenc-ffmpeg"


def test_forced_nvenc_with_no_usable_binary_keeps_moviepys_binary(monkeypatch):
    """The codec is still forced -- no silent fallback -- but without a usable
    binary the writer stays on moviepy's own, failing loudly as before.
    """
    _patch_probe(monkeypatch, usable=False)
    monkeypatch.setenv("ALGAN_VIDEO_ENCODER", "nvenc")
    assert select_video_encoder(None, None, False) == NVENC_PAIR
    assert video_encoding.resolve_encode_binary("h264_nvenc") is None


def test_override_moviepy_ffmpeg_binary_is_restored_afterwards():
    moviepy_config = pytest.importorskip("moviepy.config")
    ffmpeg_writer = pytest.importorskip("moviepy.video.io.ffmpeg_writer")

    original_config = moviepy_config.FFMPEG_BINARY
    original_writer = ffmpeg_writer.FFMPEG_BINARY
    with video_encoding.override_moviepy_ffmpeg_binary("/chosen/ffmpeg"):
        assert moviepy_config.FFMPEG_BINARY == "/chosen/ffmpeg"
        assert ffmpeg_writer.FFMPEG_BINARY == "/chosen/ffmpeg"
    assert original_config == moviepy_config.FFMPEG_BINARY
    assert original_writer == ffmpeg_writer.FFMPEG_BINARY


def test_override_without_a_binary_touches_nothing():
    moviepy_config = pytest.importorskip("moviepy.config")
    ffmpeg_writer = pytest.importorskip("moviepy.video.io.ffmpeg_writer")

    before = (moviepy_config.FFMPEG_BINARY, ffmpeg_writer.FFMPEG_BINARY)
    with video_encoding.override_moviepy_ffmpeg_binary(None):
        pass
    assert before == (moviepy_config.FFMPEG_BINARY, ffmpeg_writer.FFMPEG_BINARY)


def test_small_or_odd_outputs_stay_on_software_even_when_nvenc_is_usable(
    monkeypatch,
):
    """NVENC refuses frames under its minimum size and odd-sided 4:2:0 frames,
    and a refused encoder leaves an empty file with no error in Python -- the
    SMOKE_TEST-sized renders of the unit suite found this. Those outputs stay
    on x264 whatever the probe or the mode says.
    """
    _patch_probe(monkeypatch, usable=True)
    assert select_video_encoder(None, None, False, (64, 36)) == SOFTWARE_PAIR
    assert select_video_encoder(None, None, False, (704, 395)) == SOFTWARE_PAIR
    assert select_video_encoder(None, None, False, (704, 396)) == NVENC_PAIR
    monkeypatch.setenv("ALGAN_VIDEO_ENCODER", "nvenc")
    assert select_video_encoder(None, None, False, (64, 36)) == SOFTWARE_PAIR
    assert select_video_encoder(None, None, False, None) == NVENC_PAIR


def test_configured_ffmpeg_binary_outranks_every_other_candidate():
    """``SETTINGS.paths.ffmpeg_binary`` pins the binary for every codec.

    The usual reason to name one is that the build moviepy found lacks a codec
    the named one has, so an explicit setting has to beat the probe rather than
    join it -- and it applies to software codecs too, where the probe would
    otherwise return ``None`` and leave the writer on moviepy's binary.
    """
    from algan.settings import SETTINGS

    previous = SETTINGS.paths.ffmpeg_binary
    try:
        SETTINGS.paths.set(ffmpeg_binary="/opt/custom/ffmpeg")
        assert video_encoding.resolve_encode_binary(None) == "/opt/custom/ffmpeg"
        assert video_encoding.resolve_encode_binary("libx264") == "/opt/custom/ffmpeg"
        assert video_encoding._candidate_binaries()[0] == "/opt/custom/ffmpeg"
    finally:
        SETTINGS.paths.set(ffmpeg_binary=previous)


def test_unset_ffmpeg_binary_leaves_the_existing_resolution_alone():
    """The default must stay exactly what it was before the setting existed."""
    from algan.settings import SETTINGS

    previous = SETTINGS.paths.ffmpeg_binary
    try:
        SETTINGS.paths.set(ffmpeg_binary=None)
        assert video_encoding.resolve_encode_binary("libx264") is None
        assert "/opt/custom/ffmpeg" not in video_encoding._candidate_binaries()
    finally:
        SETTINGS.paths.set(ffmpeg_binary=previous)
