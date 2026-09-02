"""Top-level authoring helpers and render results.

:class:`RenderResult` is the value returned by
:meth:`~algan.scene.Scene.save_video` and
:meth:`~algan.scene.Scene.save_frame`: its ``status``, ``output_path``,
``walltime_seconds`` and ``render_plan`` describe what was written and how it was
rendered.

The rest is scene-level tooling: ``algan_scene`` for declaring a renderable
scene, ``combine_scenes`` and ``concatenate_videos`` for assembling a long video
out of separately authored parts (see
:doc:`/advanced_user_tutorials/multi_scene_projects`), and ``profile_func`` for
timing a render.
"""

from __future__ import annotations

import base64
import cProfile
import inspect
import multiprocessing
import os.path
import pstats
import re
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from algan.animation_timeline.animation_contexts import Off
from algan.errors import (
    AlganConfigurationError,
    LegacySceneDiscoveryWarning,
    NeverSpawnedMobWarning,
    _user_stacklevel,
)
from algan.logging.logger import get_logger
from algan.rendering.camera import Camera
from algan.settings import SETTINGS
from algan.utils.video_encoding import (
    check_codec_is_available,
    override_moviepy_ffmpeg_binary,
    resolve_encode_binary,
    select_video_encoder,
)

logger = get_logger()


def algan_scene(function=None, *, name=None):
    """Mark a zero-argument function as an Algan scene entry point.

    ``render_all_funcs`` prefers explicitly decorated functions and falls back
    to its historical zero-argument scan only when a module has no decorated
    scenes.

    Named ``algan_scene`` rather than ``scene`` so that the decorator does
    not collide with the conventional variable name for a Scene instance.
    """

    def decorate(func):
        if not callable(func):
            raise TypeError("@algan_scene can only decorate callables")
        func.name = name or func.__name__
        func.__algan_scene__ = True
        func.__algan_scene_name__ = func.name
        return func

    if function is None:
        return decorate
    return decorate(function)


from algan.utils.memory_utils import release_torch_memory


def get_file_writer(
    temp_file_path,
    video_settings_resolution,
    codec,
    fps,
    with_mask,
    ffmpeg_params,
    audiofile,
    audio_codec,
):
    from moviepy.video.io.ffmpeg_writer import (
        FFMPEG_VideoWriter,  # deferred: ~0.3 s of import algan
    )

    try:
        file_writer = FFMPEG_VideoWriter(
            filename=temp_file_path,
            size=video_settings_resolution,
            codec=codec,
            fps=fps,
            with_mask=with_mask,
            ffmpeg_params=ffmpeg_params,
            audiofile=audiofile,
            audio_codec=audio_codec,
        )
    except TypeError:
        # Older moviepy releases spell the mask parameter "withmask".
        file_writer = FFMPEG_VideoWriter(
            filename=temp_file_path,
            size=video_settings_resolution,
            codec=codec,
            fps=fps,
            withmask=with_mask,
            ffmpeg_params=ffmpeg_params,
            audiofile=audiofile,
            audio_codec=audio_codec,
        )
    return file_writer


# ‘mpeg4’ > ‘libx264’
# @compiled
@dataclass(frozen=True)
class RenderResult:
    """Outcome metadata returned by :meth:`algan.scene.Scene.save_video`.

    Attributes
    ----------
    status
        ``"rendered"`` when a file was written, ``"skipped"`` when one already
        existed and ``overwrite=False``.
    output_path
        Where the file is, or would have been. Always **absolute**, whatever
        was passed as ``file_path``.
    walltime_seconds
        Seconds the render took, ``0.0`` for a skipped one.
    render_plan
        The last batch's :class:`~algan.rendering.raytracing.RenderPlan`: which
        renderer ran, what it could not honor, and how often each ceiling bound.
    """

    status: Literal["rendered", "skipped"]
    output_path: Path
    walltime_seconds: float = 0.0
    render_plan: object | None = None

    @property
    def rendered(self) -> bool:
        return self.status == "rendered"

    def _repr_html_(self) -> str | None:
        """HTML representation for rich rendering in Jupyter / Colab notebooks."""
        if not self.output_path or not self.output_path.exists():
            return f"<p>RenderResult: <code>{self.status}</code> (file: {self.output_path})</p>"

        ext = self.output_path.suffix.lower()
        if ext in (".mp4", ".webm", ".mov"):
            mime = (
                "video/mp4"
                if ext == ".mp4"
                else ("video/webm" if ext == ".webm" else "video/quicktime")
            )
            try:
                data = self.output_path.read_bytes()
                if len(data) <= 50 * 1024 * 1024:
                    b64 = base64.b64encode(data).decode("ascii")
                    return (
                        f'<div style="max-width: 720px; margin: 8px 0;">'
                        f'<video controls autoplay loop playsinline style="max-width: 100%; border-radius: 6px; box-shadow: 0 2px 8px rgba(0,0,0,0.15);">'
                        f'<source src="data:{mime};base64,{b64}" type="{mime}">'
                        f"Your browser does not support the video tag."
                        f"</video>"
                        f'<div style="font-size: 0.85em; color: #666; margin-top: 4px;">Rendered to <code>{self.output_path.name}</code> ({self.walltime_seconds:.2f}s)</div>'
                        f"</div>"
                    )
            except Exception:
                pass
            return f"<p>RenderResult: <code>{self.status}</code> &mdash; <code>{self.output_path.name}</code> ({self.walltime_seconds:.2f}s)</p>"

        if ext in (".png", ".jpg", ".jpeg", ".webp"):
            mime = (
                "image/png"
                if ext == ".png"
                else ("image/jpeg" if ext in (".jpg", ".jpeg") else "image/webp")
            )
            try:
                data = self.output_path.read_bytes()
                if len(data) <= 20 * 1024 * 1024:
                    b64 = base64.b64encode(data).decode("ascii")
                    return (
                        f'<div style="max-width: 720px; margin: 8px 0;">'
                        f'<img src="data:{mime};base64,{b64}" style="max-width: 100%; border-radius: 6px; box-shadow: 0 2px 8px rgba(0,0,0,0.15);" alt="{self.output_path.name}" />'
                        f'<div style="font-size: 0.85em; color: #666; margin-top: 4px;">Frame saved to <code>{self.output_path.name}</code></div>'
                        f"</div>"
                    )
            except Exception:
                pass
            return f"<p>RenderResult: <code>{self.status}</code> &mdash; <code>{self.output_path.name}</code></p>"

        return None

    def _repr_png_(self) -> bytes | None:
        """PNG representation for IPython displays."""
        if (
            self.output_path
            and self.output_path.suffix.lower() == ".png"
            and self.output_path.exists()
        ):
            try:
                return self.output_path.read_bytes()
            except Exception:
                return None
        return None


#: Container formats Algan will write a video into. FFmpeg infers the muxer
#: from the extension, so an unknown one is not a slow encode -- it is a failed
#: one, and until this check existed it failed *after* the render, as a
#: ``FileNotFoundError`` on the temporary file's rename.
_SUPPORTED_VIDEO_CONTAINERS = (".mp4", ".mov", ".webm", ".mkv", ".avi", ".gif")

#: Still-image formats :meth:`~algan.scene.Scene.save_frame` will write.
_SUPPORTED_IMAGE_FORMATS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")


def _check_container_is_supported(destination, *, still: bool = False) -> None:
    """Fail before the render on an output extension Algan cannot write.

    Same shape as the transparent-MP4 error: say what is wrong, then list what
    to write instead. Raised up front because the alternative is paying for a
    whole render and then discovering the muxer never existed.
    """
    supported = _SUPPORTED_IMAGE_FORMATS if still else _SUPPORTED_VIDEO_CONTAINERS
    suffix = Path(destination).suffix
    if suffix.lower() in supported:
        return
    kind = "still-image format" if still else "video container"
    raise AlganConfigurationError(
        f"{suffix or 'that name'!r} is not a {kind} Algan can write. Use one "
        f"of {', '.join(supported)}, or leave the extension off and Algan "
        f"picks the right one."
    )


def _resolve_output_destination(file_path, default_extension: str) -> Path:
    """Turn a user's ``file_path`` into the absolute file to write.

    The rules, in order: an empty string is refused; a target that names a
    directory gets ``SETTINGS.paths.output_filename`` placed inside it; a
    missing extension becomes ``default_extension``; a bare filename lands in
    ``output_root / output_directory``; everything else is used as supplied.
    The result is always absolute, because it is also what
    :class:`RenderResult` reports back and what the finish log prints -- a
    relative path there is only as good as the reader's guess at the working
    directory.
    """
    if file_path is None:
        file_path = SETTINGS.paths.output_filename

    raw_path = os.fspath(file_path)
    if not str(raw_path).strip():
        raise AlganConfigurationError(
            "file_path must be a file name or path, such as 'my_video' or "
            "'renders/final.mp4'; got an empty string. Pass None to use the "
            "default output name."
        )
    requested = Path(raw_path).expanduser()

    # The same rule ``algan render -o`` applies (see cli._output_settings): a
    # trailing separator or an existing directory names somewhere to write
    # into, not the file itself. Without it ``save_video("renders/")`` dropped
    # the directory and wrote ``renders.mp4`` beside it.
    names_directory = str(raw_path).endswith(("/", os.sep)) or requested.is_dir()
    if names_directory:
        requested = requested / SETTINGS.paths.output_filename

    if requested.suffix == "":
        requested = requested.with_suffix(default_extension)

    # A bare filename uses Algan's standard output directory. Paths with an
    # explicit parent (including ``./``), and directories named as such, are
    # honoured exactly as supplied.
    if (
        not names_directory
        and not requested.is_absolute()
        and os.path.dirname(raw_path) == ""
    ):
        requested = (
            Path(SETTINGS.paths.output_root)
            / SETTINGS.paths.output_directory
            / requested
        )

    requested = Path(os.path.abspath(requested))
    requested.parent.mkdir(parents=True, exist_ok=True)
    return requested


def _render_scene_to_file(
    scene,
    file_path=None,
    video_settings=None,
    overwrite=True,
    reset=False,
    codec=None,
    audio_codec=None,
    background=None,
    ffmpeg_params=None,
    animate_fade_out=None,
    **kwargs,
):
    """Render ``scene`` to a video file.

    This is the implementation behind :meth:`algan.scene.Scene.save_video`,
    which carries the user-facing signature and documentation.
    """
    # The Scene's own settings outrank SETTINGS.video when it was given them:
    # ``Scene(video_settings=...)`` and ``set_video_settings`` used to have no
    # effect here at all, while ``save_frame`` on the same Scene honoured them.
    video_settings = scene._resolve_video_settings(video_settings)

    previous_settings = scene.video_settings
    previous_explicit = getattr(scene, "_video_settings_explicit", False)
    previous_background = (
        scene.background_frame,
        getattr(scene, "background", None),
        scene.background_is_set,
    )
    destination = None
    temp_file_path = None
    audio_file_path = None
    file_writer = None
    render_started = False
    scene_finalized = False

    try:
        scene.set_video_settings(video_settings)
        explicit_background = background is not None
        if background is None:
            background = SETTINGS.style.background
        scene.set_background(
            background,
            overwrite=explicit_background,
        )
        # The scene now owns the normalized color/image tensor. Passing the
        # original argument into get_frames would override authored backgrounds
        # and would reintroduce string image paths after they were decoded.
        frame_background_override = None

        default_extension = ".mov" if scene.background_is_transparent() else ".mp4"
        destination = _resolve_output_destination(file_path, default_extension)
        # Up front, beside the codec check below and for the same reason: a
        # container FFmpeg has no muxer for must not cost a whole render first.
        _check_container_is_supported(destination)

        if destination.exists() and not overwrite:
            return RenderResult("skipped", destination)

        suffix = destination.suffix.lower()
        if suffix == ".mp4" and scene.background_is_transparent():
            raise AlganConfigurationError(
                "MP4 does not support Algan's transparent output. Use .mov or "
                ".webm, or choose an opaque background."
            )

        if scene.camera is None:
            scene.camera = Camera(False, scene=scene)
        release_torch_memory(force_gc=False)

        if animate_fade_out is None:
            animate_fade_out = SETTINGS.style.fade_out_on_scene_end
        render_started = True
        # A scene authored entirely inside Off() otherwise has zero runtime,
        # which produces an empty video. Keep one frame alive. This is needed
        # in every mode, because it decides how many frames get rendered.
        timeline_context = scene.animation_manager.context
        if (
            timeline_context is not None
            and timeline_context.timespan.original_end <= 0
            and any(actor.is_spawned() for actor in scene.actors)
        ):
            timeline_context.wait(1.0 / video_settings.frames_per_second)

        # Must run before despawn_mobs below, which drops never-spawned actors
        # from scene.actors and would leave nothing to report. Skipped when
        # nothing spawned at all, because EmptySceneWarning already covers that.
        if any(actor.is_spawned() for actor in scene.actors):
            never_spawned = scene._never_spawned_root_mobs()
            if never_spawned:
                # Named Mobs are listed individually -- a name is given
                # precisely so its owner can be picked out of a crowd -- while
                # unnamed ones collapse to their class as before.
                kinds = ", ".join(sorted({m._describe() for m in never_spawned}))
                warnings.warn(
                    f"{len(never_spawned)} Mob(s) were never spawn()ed and will "
                    f"not appear in the video ({kinds}). Call .spawn() on them, "
                    "or build them with add_to_scene=False if they exist only "
                    "as data.",
                    NeverSpawnedMobWarning,
                    stacklevel=_user_stacklevel(),
                )

        # A fade-out is part of the requested output, so it is recorded on the
        # timeline whether or not the scene is being reset afterwards.
        if animate_fade_out:
            scene_finalized = True
            scene.despawn_mobs(retain_history=True, runtime=0.5)
        elif reset:
            # The scene is about to be discarded: keep the historical
            # instantaneous despawn so reset=True behaves exactly as before.
            scene_finalized = True
            with Off(animation_manager=scene.animation_manager):
                scene.despawn_mobs(retain_history=True, runtime=0.5, animate=False)
        else:
            # Leave the authored scene re-renderable: no despawns, no actor
            # filtering. Mobs stay spawned and the timeline stays as written,
            # so authoring can continue after save_video returns.
            scene_finalized = False

        transparent = scene.background_is_transparent()
        # Selection sees the raw arguments, before defaults are filled in:
        # an explicit codec must win verbatim, caller-supplied parameters
        # must not be confused with Algan's own, and its software fallback
        # reproduces exactly the pair the fills below used to produce.
        codec, ffmpeg_params = select_video_encoder(
            codec, ffmpeg_params, transparent, tuple(video_settings.resolution)
        )
        # A hardware pick can land on a binary other than moviepy's own
        # (moviepy is often configured with a static build that has no NVENC
        # encoders); None keeps moviepy's configuration untouched.
        # Before the render, not after it: an unusable codec used to cost a
        # whole render and then surface as a missing temporary file.
        check_codec_is_available(codec)
        encode_binary = resolve_encode_binary(codec)
        if codec is None:
            codec = "png" if transparent else "libx264"
        if audio_codec is None:
            audio_codec = "mp3"
        if ffmpeg_params is None:
            ffmpeg_params = [] if transparent else ["-crf", "17", "-preset", "slower"]
        logger.info(f"Encoding video with {codec}")

        temp_file_path = destination.with_name(
            f"{destination.stem}_temp{destination.suffix}"
        )
        audio_file_path = destination.with_name(f"{destination.stem}_temp.wav")
        script_file_path = destination.with_name(f"{destination.stem}_script.txt")

        logger.info(f"Began rendering {destination}")
        start_time = time.perf_counter()
        audiofile = scene.save_audio(
            str(audio_file_path),
            video_settings.audio_sample_rate,
            nbytes=4,
            codec="pcm_s32le",
        )
        if audiofile is not None:
            if not getattr(scene, "_suppress_automatic_transcript", False):
                script_file_path.write_text(
                    scene.audio_manager.video_transcript,
                    encoding="utf-8",
                )
            logger.info("Audio rendered, now rendering video")

        # The writer spawns its ffmpeg process at construction, so the
        # override only needs to span this call; it is restored right after.
        with override_moviepy_ffmpeg_binary(encode_binary):
            file_writer = get_file_writer(
                str(temp_file_path),
                video_settings.resolution,
                codec,
                video_settings.frames_per_second,
                scene.background_is_transparent(),
                ffmpeg_params,
                audiofile,
                audio_codec,
            )
        scene._render_to_video(
            file_writer,
            str(temp_file_path),
            str(destination),
            background=frame_background_override,
            despawn_camera_and_lights=scene_finalized,
            # reset=False leaves the Scene re-renderable, so the render must
            # not consume the derived state a later render depends on. With
            # reset=True the timeline is rebuilt anyway.
            preserve_authoring_state=not reset,
            **kwargs,
        )
        walltime = time.perf_counter() - start_time
        plan = getattr(scene, "last_render_plan", None)
        # The absolute path, not the bare name: "Finished rendering out.mp4"
        # left the reader to guess which directory it landed in.
        logger.info("Finished rendering %s in %.1f s", destination, walltime)
        return RenderResult("rendered", destination, walltime, plan)
    finally:
        if file_writer is not None:
            try:
                file_writer.close()
            except Exception:
                logger.debug("Video writer cleanup failed", exc_info=True)
        for temporary in (temp_file_path, audio_file_path):
            if temporary is None:
                continue
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                logger.debug(
                    "Could not remove temporary file %s",
                    temporary,
                    exc_info=True,
                )

        scene.set_video_settings(previous_settings, _explicit=previous_explicit)
        if render_started and reset:
            # Cleanup is unconditional, including failures during audio setup,
            # writer construction, rendering, or final file replacement.
            scene.reset()
        else:
            # Preflight failures, skipped renders and ordinary reset=False
            # renders all leave the authored scene and audio timeline intact.
            (
                scene.background_frame,
                scene.background,
                scene.background_is_set,
            ) = previous_background


# @compiled
def render_all_funcs(
    module_name,
    video_settings=None,
    profile=False,
    overwrite=True,
    start_index=0,
    max_rendered=-1,
    output_directory=None,
    output_root=None,
    file_extension="mp4",
    smoke_test=False,
    prefix=None,
    funcs=None,
    speech_source=None,
    **kwargs,
):
    def run(output_directory=None, video_settings=None, output_root=None, prefix=None):
        if funcs is None:
            module = (
                sys.modules[module_name]
                if isinstance(module_name, str)
                else module_name
            )
            if prefix is None:
                prefix = module.__name__
            defined_functions = [
                (function_name, function)
                for function_name, function in inspect.getmembers(
                    module, inspect.isfunction
                )
                if function.__module__[: len(prefix)] == prefix
            ]
            registered = [
                (getattr(function, "__algan_scene_name__", function_name), function)
                for function_name, function in defined_functions
                if getattr(function, "__algan_scene__", False)
            ]
            if registered:
                scene_funcs = registered
            else:
                scene_funcs = [
                    (function_name, function)
                    for function_name, function in defined_functions
                    if len(inspect.signature(function).parameters) == 0
                ]
                if scene_funcs:
                    warnings.warn(
                        "render_all_funcs is using legacy implicit zero-argument "
                        "function discovery. Decorate scene entry points with @algan_scene "
                        "to prevent helper functions from rendering accidentally.",
                        LegacySceneDiscoveryWarning,
                        stacklevel=2,
                    )
            scene_funcs.sort(key=lambda item: item[1].__code__.co_firstlineno)
        else:
            scene_funcs = funcs

        if video_settings is None:
            video_settings = SETTINGS.video

        if output_root is None:
            output_root = SETTINGS.paths.output_root
        if output_directory is None:
            output_directory = SETTINGS.paths.output_directory
        output_directory = os.path.join(output_directory, module_name)
        start = start_index + len(scene_funcs) if start_index < 0 else start_index
        end = len(scene_funcs) if max_rendered < 0 else start + max_rendered

        results = []
        from algan.scene import Scene

        for index, (scene_name, function) in list(enumerate(scene_funcs))[start:end]:
            with Scene(video_settings=video_settings) as active_scene:
                if "background" in kwargs:
                    active_scene.set_background(kwargs["background"], overwrite=False)
                active_scene.audio_manager.set_speech_source(speech_source)
                function()
                if not smoke_test:
                    results.append(
                        active_scene.save_video(
                            Path(output_root)
                            / output_directory
                            / f"{index}_{scene_name}.{file_extension}",
                            video_settings=video_settings,
                            overwrite=overwrite,
                            **kwargs,
                        )
                    )
        return results

    if profile:
        pr = cProfile.Profile()
        start = time.time()
        pr.enable()
        out = run(output_directory, video_settings, output_root, prefix)
        pr.disable()
        end = time.time()

        with open("profiler_dump.txt", "w") as f:
            ps = pstats.Stats(pr, stream=f).sort_stats(pstats.SortKey.CUMULATIVE)
            ps.print_stats()
        ps = pstats.Stats(pr).sort_stats(pstats.SortKey.CUMULATIVE)
        ps.print_stats()
        logger.info(f"took {end - start} seconds.")
        return out
    else:
        return run(output_directory, video_settings, output_root, prefix)


def profile_func(func):
    pr = cProfile.Profile()
    start = time.time()
    pr.enable()
    out = func()
    pr.disable()
    end = time.time()

    with open("profiler_dump.txt", "w") as f:
        ps = pstats.Stats(pr, stream=f).sort_stats(pstats.SortKey.CUMULATIVE)
        ps.print_stats()
    ps = pstats.Stats(pr).sort_stats(pstats.SortKey.CUMULATIVE)
    ps.print_stats()
    logger.info(f"took {end - start} seconds.")
    return out


def concatenate_videos(
    directory: str,
    threads: int = None,
    reencode: bool = False,
    output_file="output.mp4",
    input_files=None,
):
    """
    Concatenate all .mp4 files in a directory into output.mp4.

    Files are sorted by their numeric prefix (e.g., 1_intro.mp4, 2_scene.mp4).
    Uses ffmpeg with multithreading support.

    Args:
        directory: Path to directory containing .mp4 files
        threads: Number of threads for ffmpeg (default: CPU count)
        reencode: If True, re-encode videos with multithreading.
                 If False, use stream copy (faster, no re-encoding)
        input_files: Optional explicit iterable of videos to concatenate. Paths
                     default to ``directory`` when relative. When omitted, all
                     MP4 files directly inside ``directory`` are used.

    Returns:
        Path to output file if successful, None otherwise
    """
    # Default to CPU count
    if threads is None:
        threads = multiprocessing.cpu_count()

    dir_path = Path(directory).resolve()

    output_path = Path(output_file)
    if not output_path.is_absolute():
        output_path = dir_path / output_path
    output_path = output_path.resolve()

    # Find all .mp4 files, excluding the combined output when it lives in the
    # same directory as its inputs. Project passes its exact scene manifest so
    # unrelated MP4 files in the output directory cannot join the final video.
    if input_files is None:
        candidates = dir_path.glob("*.mp4")
    else:
        candidates = (
            path if (path := Path(file)).is_absolute() else dir_path / path
            for file in input_files
        )
    mp4_files = [
        path
        for path in candidates
        if path.suffix.lower() == ".mp4"
        and path.exists()
        and path.resolve() != output_path
    ]

    if not mp4_files:
        logger.warning(f"No .mp4 files found in {directory}")
        return None

    # Sort by numeric prefix
    def get_prefix_number(file_path):
        match = re.match(r"(\d+)_", file_path.name)
        if match:
            return int(match.group(1))
        # Files without numeric prefix go to end, maintain alphabetical order
        return float("inf")  # , file_path.name)

    sorted_files = sorted(mp4_files, key=get_prefix_number)

    # Create concat list file for ffmpeg
    concat_file = dir_path / "ffmpeg_concat_list.txt"
    try:
        with open(concat_file, "w", encoding="utf-8") as f:
            for video_file in sorted_files:
                # Use absolute path with proper escaping for ffmpeg
                # Replace backslashes with forward slashes for ffmpeg on Windows
                abs_path = str(video_file.resolve()).replace("\\", "/")
                f.write(f"file '{abs_path}'\n")

        # Build ffmpeg command
        cmd = ["ffmpeg", "-f", "concat", "-safe", "0", "-i", str(concat_file)]

        if reencode:
            # Re-encode with multithreading
            cmd.extend(
                [
                    "-c:v",
                    "libx264",
                    "-preset",
                    "medium",
                    "-crf",
                    "23",
                    "-threads",
                    str(threads),
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                ]
            )
            logger.info(f"Re-encoding with {threads} threads...")
        else:
            # Stream copy (fast, no re-encoding, but limited multithreading benefit)
            cmd.extend(["-c", "copy"])
            logger.info("Using stream copy (no re-encoding)...")

        cmd.extend(["-y", str(output_path)])  # -y to overwrite without asking

        logger.info(f"\nConcatenating {len(sorted_files)} videos:")
        for i, f in enumerate(sorted_files, 1):
            logger.info(f"  {i}. {f.name}")
        logger.info(f"\nOutput: {output_path}\n")

        # Run ffmpeg
        result = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
        )

        if result.returncode == 0:
            logger.info(f"✓ Successfully created {output_path.name}")
            logger.info(f"  Size: {output_path.stat().st_size / (1024 * 1024):.2f} MB")
            return output_path
        else:
            logger.error("✗ Error running ffmpeg:")
            logger.error(result.stderr)
            return None

    finally:
        # Clean up concat list file
        if concat_file.exists():
            concat_file.unlink()


def combine_scenes(directory):
    ext = None
    output_text_file = os.path.join(directory, "transcript.txt")
    transcript = ""
    video_files = []

    def starts_with_int(s):
        try:
            int(s.split("_")[0])
            return True
        except ValueError:
            return False

    for f in sorted(
        [_ for _ in os.listdir(directory) if starts_with_int(_)],
        key=lambda x: int(x.split("_")[0]),
    ):
        if f.endswith(".mp4"):
            video_files.append(os.path.join(directory, f))
            if ext is None:
                ext = f.split(".")[-1]
        elif f.endswith(".txt"):
            with open(os.path.join(directory, f)) as f:
                transcript += f.read()

    with open(output_text_file, "w") as f:
        f.write(transcript)

    # concatenate_videos(directory, output_file=f"video.{ext}")
