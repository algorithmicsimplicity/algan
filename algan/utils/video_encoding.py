"""Automatic choice of the FFmpeg video encoder behind ``save_video``.

Encoding a large render with ``libx264 -preset slower`` is pure CPU work and
can be the slowest stage of a render on a small machine, stalling the bounded
frame queue. When some FFmpeg binary on the machine carries NVIDIA's
``h264_nvenc`` encoder and the driver can drive it, Algan moves encoding onto
the GPU -- and points the writer at whichever binary that was, because moviepy
is often configured with a stripped-down static build that has no NVENC
encoders even though the system's ffmpeg has them.

:func:`select_video_encoder` makes that choice once per video, from the
caller's explicit arguments first and the ``ALGAN_VIDEO_ENCODER`` environment
variable second (``auto`` by default: hardware when usable, software
otherwise). :func:`resolve_encode_binary` says which binary must run it.
Both answers come from one probe, run at most once per process.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import warnings
from contextlib import contextmanager

from algan.environment import env_str
from algan.errors import AlganConfigurationError, AlganWarning
from algan.logging.logger import get_logger

logger = get_logger()

#: The knob behind the automatic choice, declared in ``algan/environment.py``
#: as read live so flipping it between two renders in one process works.
_ENV_ENCODER = "ALGAN_VIDEO_ENCODER"

#: The software encoding Algan has always used by default.
_SOFTWARE_CODEC = "libx264"
_SOFTWARE_FFMPEG_PARAMS = ["-crf", "17", "-preset", "slower"]

#: Hardware encoding through NVIDIA's NVENC engine. Constant-quality VBR at
#: CQ 19 lands visually close to the software pair's CRF 17 at a fraction of
#: the CPU cost. moviepy's writer inserts its own ``-preset medium`` before
#: these, which these later options override (FFmpeg takes the last spelling),
#: and appends a trailing ``-pix_fmt yuv420p``-class option for even-sized
#: frames after them; FFmpeg then warns and auto-selects ``yuv420p``, so no
#: pixel-format or size flag here contradicts what the writer builds.
_NVENC_CODEC = "h264_nvenc"
_NVENC_FFMPEG_PARAMS = [
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
]

#: Rate-control flags that only mean something to x264. Found in caller
#: parameters they are honoured by staying on the software encoder.
_X264_ONLY_FLAGS = ("-preset", "-crf")

#: The smallest frame NVENC is asked to encode. The hardware's documented H.264
#: floor is 145x49 on current parts and a 64x64 canary fails outright; a
#: unit-test-sized render (the SMOKE_TEST preset) under this floor produced an
#: empty file with no Python-visible error, so anything smaller -- and any
#: odd-sized frame, which 4:2:0 hardware encoding rejects -- stays on x264.
_NVENC_MIN_RESOLUTION = (256, 128)


def _nvenc_accepts(resolution) -> bool:
    """Whether a frame of ``resolution`` (``(width, height)``) can go to NVENC."""
    if resolution is None:
        return True
    width, height = int(resolution[0]), int(resolution[1])
    if width < _NVENC_MIN_RESOLUTION[0] or height < _NVENC_MIN_RESOLUTION[1]:
        logger.debug(
            "NVENC skipped: %dx%d is below the %dx%d floor; encoding in software",
            width,
            height,
            *_NVENC_MIN_RESOLUTION,
        )
        return False
    if width % 2 or height % 2:
        logger.debug(
            "NVENC skipped: %dx%d has an odd side; encoding in software",
            width,
            height,
        )
        return False
    return True


#: NVENC refuses to open an encoder below its minimum frame dimension (a
#: 64x64 canary fails outright on current drivers), so the test encode uses a
#: size any real render clears but that still costs nothing.
_PROBE_SIZE = "256x256"

#: Bound both probe invocations; a hung ffmpeg must not hang a render.
_PROBE_TIMEOUT_SECONDS = 10

#: Result of the usability probe: ``None`` until it has run, then a
#: ``(usable, binary)`` pair where ``binary`` is the FFmpeg executable the
#: probe found usable (``None`` when no candidate qualified).
_probe_cache = None


def select_video_encoder(
    codec: str | None,
    ffmpeg_params: list[str] | None,
    transparent: bool,
    resolution: tuple[int, int] | None = None,
) -> tuple[str | None, list[str] | None]:
    """Pick the codec and FFmpeg parameters a ``save_video`` render encodes with.

    An explicit ``codec`` wins verbatim. Transparent output is untouched.
    Otherwise the ``ALGAN_VIDEO_ENCODER`` environment variable decides:
    ``auto`` (the default) picks ``h264_nvenc`` when some FFmpeg binary on
    the machine can drive it and software encoding otherwise, ``nvenc``
    forces hardware encoding, and ``software`` reproduces the historical
    default exactly. When hardware encoding is chosen,
    :func:`resolve_encode_binary` names the binary the writer must run --
    which is not necessarily the one moviepy is configured with.

    Returns
    -------
    tuple
        ``(codec, ffmpeg_params)`` for the writer.

    Parameters
    ----------
    codec
        The caller's explicit codec, or ``None`` to have one chosen.
    ffmpeg_params
        The caller's own FFmpeg parameters, or ``None`` for defaults.
        Parameters naming an x264-only rate-control flag (``-preset`` /
        ``-crf``) keep the software encoder even in ``nvenc`` mode, because
        those flags would fail or change meaning under NVENC.
    transparent
        Whether the output carries alpha; such output is returned untouched.
    resolution
        The output's ``(width, height)`` in pixels, when known. NVENC refuses
        frames below its minimum dimensions and odd-sized 4:2:0 frames, and a
        refused encoder leaves an empty file behind, so an output under
        ``_NVENC_MIN_RESOLUTION`` or with an odd side is encoded in software
        whatever the mode says. ``None`` skips the check.
    """
    # An explicit codec is the caller's decision, whatever it says, and
    # transparent output (.mov/.png) keeps today's behaviour untouched.
    if codec is not None or transparent:
        return codec, ffmpeg_params

    # Parameters without a codec are the caller's own quality control. The
    # two flags above are x264-specific -- h264_nvenc either rejects them or
    # silently reads them as something else -- so honouring their intent
    # means staying on libx264 rather than mixing rate-control dialects in
    # one command line.
    if ffmpeg_params is not None and any(
        flag in _X264_ONLY_FLAGS for flag in ffmpeg_params
    ):
        return _SOFTWARE_CODEC, list(ffmpeg_params)

    if _use_nvenc() and _nvenc_accepts(resolution):
        if ffmpeg_params is None:
            return _NVENC_CODEC, list(_NVENC_FFMPEG_PARAMS)
        # Neutral caller parameters (container/mux flags, say) ride along;
        # they say nothing about rate control.
        return _NVENC_CODEC, list(ffmpeg_params)

    if ffmpeg_params is None:
        return _SOFTWARE_CODEC, list(_SOFTWARE_FFMPEG_PARAMS)
    return _SOFTWARE_CODEC, list(ffmpeg_params)


def _resolve_encoder_mode() -> str:
    """The validated value of ``ALGAN_VIDEO_ENCODER``, defaulting to auto."""
    raw = env_str(_ENV_ENCODER, "auto")
    mode = raw.strip().lower()
    if mode in ("auto", "nvenc", "software"):
        return mode
    warnings.warn(
        f"{_ENV_ENCODER}={raw!r} is not one of 'auto', 'nvenc', 'software'; "
        "using 'auto'.",
        AlganWarning,
        stacklevel=3,
    )
    return "auto"


def _use_nvenc() -> bool:
    """Whether this render should encode with NVENC."""
    mode = _resolve_encoder_mode()
    if mode == "software":
        return False
    if mode == "nvenc":
        # Forced: the caller asked for it whether or not the probe agrees.
        return True
    # Auto: hardware exactly when some candidate binary can drive it. The
    # same cached probe later names that binary for the writer.
    return _probe_cached()[0]


def _moviepy_ffmpeg_binary() -> str:
    # Read live off the module so a user who repoints moviepy before
    # rendering (the FFMPEG_BINARY environment variable, or the attribute)
    # is probed against the binary that will actually run. Deferred import:
    # pulling moviepy in costs ~0.3 s and is only needed at the first
    # non-transparent save_video without an explicit codec.
    import moviepy.config

    return moviepy.config.FFMPEG_BINARY


def _env_ffmpeg_binary() -> str | None:
    # moviepy's own variable: when the user set it, it names exactly the
    # binary moviepy will run, so it is the most specific candidate. The
    # "auto" sentinel is skipped (it is not a path).
    raw = os.environ.get("FFMPEG_BINARY")
    if not raw or raw.strip().lower() == "auto":
        return None
    return raw


def _system_ffmpeg_binary() -> str | None:
    return shutil.which("ffmpeg")


def _candidate_binaries() -> list[str]:
    """The FFmpeg binaries worth probing, most specific first.

    In order: the ``FFMPEG_BINARY`` environment variable if set (it points
    moviepy at that binary too), moviepy's configured binary -- often
    imageio-ffmpeg's static build, which carries no NVENC encoders even on
    machines with a working NVENC driver -- and finally whatever ``ffmpeg``
    is on the PATH. Duplicates are probed once.
    """
    candidates = [_env_ffmpeg_binary(), None, _system_ffmpeg_binary()]
    try:
        candidates[1] = _moviepy_ffmpeg_binary()
    except Exception:
        logger.debug(
            "NVENC probe: could not resolve moviepy's ffmpeg binary",
            exc_info=True,
        )
    return [c for c in dict.fromkeys(candidates) if c is not None]


def _binary_usable(binary: str) -> bool:
    """Whether this FFmpeg binary can encode with ``h264_nvenc``.

    Two conditions, because each fails independently in the field: the binary
    must have been built with the encoder, and the encoder must open (a build
    can ship it yet fail without a driver or on a headless GPU). Failures log
    at DEBUG -- falling back to another binary or to software encoding is the
    designed outcome, not a fault.
    """
    try:
        listed = subprocess.run(
            [binary, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        logger.debug("NVENC probe: could not run %s -encoders", binary, exc_info=True)
        return False
    if "h264_nvenc" not in listed.stdout:
        logger.debug(
            "NVENC probe: %s does not list h264_nvenc in -encoders",
            binary,
        )
        return False

    try:
        test_encode = subprocess.run(
            [
                binary,
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                f"color=black:s={_PROBE_SIZE}:d=0.1",
                "-c:v",
                _NVENC_CODEC,
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            timeout=_PROBE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        logger.debug("NVENC probe: test encode could not run", exc_info=True)
        return False
    if test_encode.returncode != 0:
        stderr = (test_encode.stderr or b"").decode(errors="replace").strip()
        logger.debug(
            "NVENC probe: h264_nvenc failed a test encode via %s (%s)",
            binary,
            stderr,
        )
        return False
    logger.debug("NVENC probe: h264_nvenc is usable via %s", binary)
    return True


def _probe_once() -> tuple[bool, str | None]:
    """Probe candidates in order; the first usable binary wins."""
    candidates = _candidate_binaries()
    if not candidates:
        logger.debug("NVENC probe: no ffmpeg binary candidate found")
        return False, None
    for binary in candidates:
        if _binary_usable(binary):
            return True, binary
    logger.debug(
        "NVENC probe: none of %s can drive h264_nvenc; encoding on the CPU instead",
        ", ".join(candidates),
    )
    return False, None


def _probe_cached() -> tuple[bool, str | None]:
    """Cached :func:`_probe_once`; the answer cannot change within a process."""
    global _probe_cache
    if _probe_cache is None:
        _probe_cache = _probe_once()
    return _probe_cache


def resolve_encode_binary(codec: str | None) -> str | None:
    """The FFmpeg binary the video writer must run for this codec.

    ``None`` means the writer stays on whatever binary moviepy is
    configured with -- the answer for every software codec. Only an NVENC
    codec can move the binary, and only onto one a probe found usable:
    candidates are tried in :func:`_candidate_binaries` order (the
    ``FFMPEG_BINARY`` environment variable, moviepy's configured binary,
    then ``ffmpeg`` on the PATH), and when none qualifies the writer keeps
    moviepy's own binary -- for ``auto`` that pairs with the software
    fallback, and for a forced ``nvenc`` it fails loudly at encode time
    exactly as it did before binaries were selectable, which is the honest
    outcome of forcing what the machine cannot serve.
    """
    if codec != _NVENC_CODEC:
        return None
    if _resolve_encoder_mode() == "software":
        # 'software' pins today's behaviour; today the binary was moviepy's.
        return None
    # In auto mode this is the same cached probe the codec decision consulted;
    # in forced mode it runs once here to serve the demand the caller made.
    return _probe_cached()[1]


def _listed_encoders(binary: str) -> set[str] | None:
    """Encoder names ``binary`` reports, or ``None`` if it could not be asked."""
    try:
        listed = subprocess.run(
            [binary, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if listed.returncode != 0:
        return None
    names = set()
    for line in listed.stdout.splitlines():
        parts = line.split()
        # Rows look like " V....D h264   H.264 ...": flags, name, description.
        if len(parts) >= 2 and len(parts[0]) == 6 and parts[0][0] in "VAS":
            names.add(parts[1])
    return names or None


def check_codec_is_available(codec: str | None) -> None:
    """Fail early, and by name, on a codec FFmpeg cannot encode with.

    An unusable codec used to surface only after the whole render: FFmpeg
    exited, the temporary file was never written, and the move to the final
    path raised ``FileNotFoundError`` naming two paths and no codec. Asking
    FFmpeg for its encoder list costs one subprocess and happens only when the
    caller named a codec.

    Silent when the list cannot be obtained -- an unaskable FFmpeg is not
    evidence against the codec, and the encode is still free to try.
    """
    if codec is None:
        return
    try:
        default_binary = _moviepy_ffmpeg_binary()
    except Exception:  # noqa: BLE001 -- an unimportable moviepy is not our error
        default_binary = "ffmpeg"
    binary = resolve_encode_binary(codec) or default_binary or "ffmpeg"
    available = _listed_encoders(binary)
    if available is None or codec in available:
        return
    close = sorted(name for name in available if codec.lower() in name.lower())
    suggestion = f" Did you mean: {', '.join(close[:5])}?" if close else ""
    raise AlganConfigurationError(
        f"{binary} cannot encode with codec {codec!r}.{suggestion} "
        f"Leave codec unset to let Algan choose (libx264, or h264_nvenc where "
        f"the hardware encoder is usable), or pass one this FFmpeg lists under "
        f"`{binary} -encoders`."
    )


@contextmanager
def override_moviepy_ffmpeg_binary(binary: str | None):
    """Point moviepy's FFmpeg writer at ``binary`` for the duration of the block.

    Route taken: moviepy 2.x offers no per-writer binary argument -- its
    ``FFMPEG_VideoWriter`` builds its command line from the module global
    ``moviepy.video.io.ffmpeg_writer.FFMPEG_BINARY``, which that module bound
    from ``moviepy.config`` at import time -- so both attributes are set here
    and restored afterwards rather than only ``moviepy.config.FFMPEG_BINARY``
    (which an already-imported writer module would never re-read). The writer
    spawns its ffmpeg process at construction, so wrapping the construction is
    enough: frame writes and close talk to the already-running process, and
    unrelated moviepy use sees its own configuration again the moment the
    block exits. ``None`` leaves everything untouched.
    """
    if binary is None:
        yield
        return
    import moviepy.config
    import moviepy.video.io.ffmpeg_writer as ffmpeg_writer_module

    to_restore = (
        (moviepy.config, moviepy.config.FFMPEG_BINARY),
        (ffmpeg_writer_module, ffmpeg_writer_module.FFMPEG_BINARY),
    )
    try:
        moviepy.config.FFMPEG_BINARY = binary
        ffmpeg_writer_module.FFMPEG_BINARY = binary
        yield
    finally:
        for module, previous in to_restore:
            module.FFMPEG_BINARY = previous
