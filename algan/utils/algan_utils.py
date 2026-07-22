import cProfile
from dataclasses import dataclass
from typing import Literal
import os.path
import time

from pathlib import Path

import multiprocessing
import re
import inspect
import pstats
import sys
import subprocess
import warnings

from algan.settings.defaults import *
from algan.errors import AlganConfigurationError, LegacySceneDiscoveryWarning
from algan.settings.style_defaults import STYLE_DEFAULTS
from algan.animation.animation_contexts import AnimationManager, Off
from algan.rendering.camera import Camera
from algan.scene_manager import SceneManager
from algan.logging.logger import get_logger
from algan.sound.audio_effect import AudioManager

logger = get_logger()


def scene(function=None, *, name=None):
    """Mark a zero-argument function as an Algan scene entry point.

    ``render_all_funcs`` prefers explicitly decorated functions and falls back
    to its historical zero-argument scan only when a module has no decorated
    scenes.
    """
    def decorate(func):
        if not callable(func):
            raise TypeError("@scene can only decorate callables")
        setattr(func, "__algan_scene__", True)
        setattr(func, "__algan_scene_name__", name or func.__name__)
        return func

    if function is None:
        return decorate
    return decorate(function)
from algan.utils.memory_utils import empty_cache


def get_file_writer(temp_file_path, render_settings_resolution, codec, fps, with_mask, ffmpeg_params, audiofile, audio_codec):
    from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter  # deferred: ~0.3 s of import algan

    try:
        file_writer = FFMPEG_VideoWriter(
            filename=temp_file_path,
            size=render_settings_resolution,
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
            size=render_settings_resolution,
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
    """Outcome metadata returned by :func:`render_to_file`."""

    status: Literal["rendered", "skipped"]
    output_path: Path
    duration_seconds: float = 0.0
    render_plan: object | None = None

    @property
    def rendered(self) -> bool:
        return self.status == "rendered"


def _resolve_output_destination(
    file_name,
    output_directory: Path,
    file_extension,
    default_extension: str,
) -> Path:
    requested = Path(str(file_name))
    supplied_suffix = requested.suffix.lower()
    override_suffix = None
    if file_extension is not None:
        cleaned = str(file_extension).strip().lstrip(".").lower()
        if not cleaned:
            raise AlganConfigurationError("file_extension cannot be empty")
        override_suffix = f".{cleaned}"
        if supplied_suffix and supplied_suffix != override_suffix:
            raise AlganConfigurationError(
                f"Conflicting output extensions: '{supplied_suffix}' in file_name "
                f"and '{override_suffix}' from file_extension"
            )

    suffix = override_suffix or supplied_suffix or default_extension
    if not suffix.startswith("."):
        suffix = f".{suffix}"
    requested = requested.with_suffix(suffix)
    destination = requested if requested.is_absolute() else output_directory / requested
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def render_to_file(
    file_name=None,
    output_dir=None,
    output_path=None,
    render_settings=None,
    overwrite=True,
    codec=None,
    audio_codec=None,
    audio_fps=44100,
    file_extension=None,
    background_color=None,
    ffmpeg_params=None,
    animate_fade_out=None,
    **kwargs,
):
    """Render the active scene to a video file.

    Parameters
    ----------
    file_name
        Name of the output file (without extension). If None will use `DIRECTORY_DEFAULTS.output_filename`.
    output_dir
        Directory where to save the video. If None will use the directory of the running script.
    render_settings
        The :class:`.RenderSettings` object to use to specify video properties. If None will use `RENDERING_DEFAULTS.render_settings`.
    overwrite
        Whether the existing file at the output destination should be overwritten if one exists.
    codec
        The codec to use to encode the video frames.
    background_color
        A color/image or procedural callable ``(x, y, time) -> color``.
        Python callables receive broadcastable Torch tensors. A Taichi
        ``@ti.func`` receives scalar normalized coordinates and time and must
        return a color vector; it is evaluated for the whole render batch by
        one Taichi kernel writing directly into the output buffer.

    Returns
    -------
    RenderResult
        Structured metadata indicating whether the file was rendered or
        skipped because it already existed.
    """
    if file_name is None:
        file_name = DIRECTORY_DEFAULTS.output_filename
    if output_dir is None:
        output_dir = DIRECTORY_DEFAULTS.output_directory
    if output_path is None:
        output_path = DIRECTORY_DEFAULTS.output_path
        if output_path is None:
            output_path = DIRECTORY_DEFAULTS.base_directory
    output_directory = Path(output_path) / output_dir
    output_directory.mkdir(parents=True, exist_ok=True)
    if render_settings is None:
        render_settings = RENDERING_DEFAULTS.settings

    scene = SceneManager.instance()
    previous_settings = scene.render_settings
    previous_background = (
        scene.background_frame,
        getattr(scene, "background_color", None),
        scene.background_is_set,
    )
    destination = None
    temp_file_path = None
    audio_file_path = None
    file_writer = None
    destructive_render_started = False

    try:
        scene.set_render_settings(render_settings)
        explicit_background = background_color is not None
        if background_color is None:
            background_color = STYLE_DEFAULTS.background_color
        scene.set_background_color(
            background_color,
            overwrite=explicit_background,
        )
        # The scene now owns the normalized color/image tensor. Passing the
        # original argument into get_frames would override authored backgrounds
        # and would reintroduce string image paths after they were decoded.
        frame_background_override = None

        default_extension = (
            ".mov" if scene.background_is_transparent() else ".mp4"
        )
        destination = _resolve_output_destination(
            file_name,
            output_directory,
            file_extension,
            default_extension,
        )

        if destination.exists() and not overwrite:
            return RenderResult("skipped", destination)

        suffix = destination.suffix.lower()
        if suffix == ".mp4" and scene.background_is_transparent():
            raise AlganConfigurationError(
                "MP4 does not support Algan's transparent output. Use .mov or "
                ".webm, or choose an opaque background."
            )

        if scene.camera is None:
            scene.camera = Camera(False)
        empty_cache()

        if animate_fade_out is None:
            animate_fade_out = STYLE_DEFAULTS.fade_out_on_scene_end
        # From this point onward the active timeline/scene may be intentionally
        # modified for finalization. Preserve the historical contract by
        # resetting managers on both success and failure.
        destructive_render_started = True
        if animate_fade_out:
            scene.clear_scene()
        else:
            # A scene authored entirely inside Off() otherwise has zero
            # duration; despawning it at t=0 produces an empty video. Keep one
            # frame alive before the instantaneous final despawn.
            timeline_context = AnimationManager.instance().context
            if (
                timeline_context is not None
                and timeline_context.timespan.original_end <= 0
                and any(actor.is_spawned() for actor in scene.actors[-1])
            ):
                timeline_context.wait(1.0 / render_settings.frames_per_second)
            with Off():
                scene.clear_scene(animate=False)

        if codec is None:
            codec = "png" if scene.background_is_transparent() else "libx264"
        if audio_codec is None:
            audio_codec = "mp3"
        if ffmpeg_params is None:
            ffmpeg_params = (
                ["-crf", "17", "-preset", "slower"]
                if not scene.background_is_transparent()
                else []
            )

        temp_file_path = destination.with_name(
            f"{destination.stem}_temp{destination.suffix}"
        )
        audio_file_path = destination.with_name(
            f"{destination.stem}_temp.wav"
        )
        script_file_path = destination.with_name(
            f"{destination.stem}_script.txt"
        )

        logger.info(f"Began rendering {destination.name}")
        start_time = time.perf_counter()
        audiofile = scene.render_audio_to_file(
            str(audio_file_path), audio_fps, nbytes=4, codec="pcm_s32le"
        )
        if audiofile is not None:
            script_file_path.write_text(
                AudioManager.instance().video_transcript,
                encoding="utf-8",
            )
            logger.info("Audio rendered, now rendering video")

        file_writer = get_file_writer(
            str(temp_file_path),
            render_settings.resolution,
            codec,
            render_settings.frames_per_second,
            scene.background_is_transparent(),
            ffmpeg_params,
            audiofile,
            audio_codec,
        )
        scene.render_to_video(
            file_writer,
            str(temp_file_path),
            str(destination),
            background_color=frame_background_override,
            **kwargs,
        )
        duration = time.perf_counter() - start_time
        plan = getattr(scene, "last_render_plan", None)
        logger.info(f"Finished rendering {destination.name}")
        return RenderResult("rendered", destination, duration, plan)
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

        if destructive_render_started:
            # Cleanup is unconditional, including failures during audio setup,
            # writer construction, rendering, or final file replacement.
            SceneManager.reset()
            AudioManager.reset()
        else:
            # Preflight failures and skipped renders are observational: leave
            # the authored scene and audio timeline intact.
            scene.set_render_settings(previous_settings)
            (
                scene.background_frame,
                scene.background_color,
                scene.background_is_set,
            ) = previous_background


# Concise stable authoring alias. Keep ``render_to_file`` as the descriptive
# compatibility name while new examples can use ``render(...)``.
render = render_to_file


# @compiled
def render_all_funcs(
    module_name,
    render_settings=None,
    profile=False,
    overwrite=True,
    start_index=0,
    max_rendered=-1,
    output_dir=None,
    output_path=None,
    file_extension="mp4",
    smoke_test=False,
        prefix=None,
        funcs=None,
    **kwargs,
):
    def run(output_dir=None, render_settings=None, output_path=None, prefix=None):
        if funcs is None:
            module = sys.modules[module_name] if isinstance(module_name, str) else module_name
            if prefix is None:
                prefix = module.__name__
            defined_functions = [
                (function_name, function)
                for function_name, function in inspect.getmembers(module, inspect.isfunction)
                if function.__module__[:len(prefix)] == prefix
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
                        "function discovery. Decorate scene entry points with @scene "
                        "to prevent helper functions from rendering accidentally.",
                        LegacySceneDiscoveryWarning,
                        stacklevel=2,
                    )
            scene_funcs.sort(key=lambda item: item[1].__code__.co_firstlineno)
        else:
            scene_funcs = funcs

        if render_settings is None:
            render_settings = RENDERING_DEFAULTS.settings

        if output_path is None:
            output_path = DIRECTORY_DEFAULTS.output_path
            if output_path is None:
                output_path = DIRECTORY_DEFAULTS.base_directory
        if output_dir is None:
            output_dir = DIRECTORY_DEFAULTS.output_directory
        output_dir = os.path.join(output_dir, module_name)
        if start_index < 0:
            start = start_index + len(scene_funcs)
        else:
            start = start_index
        if max_rendered < 0:
            end = len(scene_funcs)
        else:
            end = start + max_rendered

        results = []
        for index, (scene_name, function) in list(enumerate(scene_funcs))[start:end]:
            active_scene = SceneManager.reset()
            active_scene.set_render_settings(render_settings)
            if "background_color" in kwargs:
                active_scene.set_background_color(kwargs["background_color"])
            function()
            if not smoke_test:
                results.append(
                    render_to_file(
                        f"{index}_{scene_name}.{file_extension}",
                        output_dir,
                        output_path,
                        render_settings,
                        overwrite,
                        **kwargs,
                    )
                )
        return results


    if profile:
        pr = cProfile.Profile()
        start = time.time()
        pr.enable()
        out = run(output_dir, render_settings, output_path, prefix)
        pr.disable()
        end = time.time()

        with open('profiler_dump.txt', 'w') as f:
            ps = pstats.Stats(pr, stream=f).sort_stats(pstats.SortKey.CUMULATIVE)
            ps.print_stats()
        ps = pstats.Stats(pr).sort_stats(pstats.SortKey.CUMULATIVE)
        ps.print_stats()
        logger.info(f'took {end-start} seconds.')
        return out
    else:
        return run(output_dir, render_settings, output_path, prefix)


def profile_func(func):
    pr = cProfile.Profile()
    start = time.time()
    pr.enable()
    out = func()
    pr.disable()
    end = time.time()

    with open('profiler_dump.txt', 'w') as f:
        ps = pstats.Stats(pr, stream=f).sort_stats(pstats.SortKey.CUMULATIVE)
        ps.print_stats()
    ps = pstats.Stats(pr).sort_stats(pstats.SortKey.CUMULATIVE)
    ps.print_stats()
    logger.info(f'took {end - start} seconds.')
    return out


def concatenate_videos(directory: str, threads: int = None, reencode: bool = False,
                       output_file='output.mp4'):
    """
    Concatenate all .mp4 files in a directory into output.mp4.

    Files are sorted by their numeric prefix (e.g., 1_intro.mp4, 2_scene.mp4).
    Uses ffmpeg with multithreading support.

    Args:
        directory: Path to directory containing .mp4 files
        threads: Number of threads for ffmpeg (default: CPU count)
        reencode: If True, re-encode videos with multithreading.
                 If False, use stream copy (faster, no re-encoding)

    Returns:
        Path to output file if successful, None otherwise
    """
    # Default to CPU count
    if threads is None:
        threads = multiprocessing.cpu_count()

    dir_path = Path(directory).resolve()

    # Find all .mp4 files (excluding output.mp4 if it exists)
    mp4_files = [f for f in dir_path.glob("*.mp4") if f.name != output_file]

    if not mp4_files:
        logger.warning(f"No .mp4 files found in {directory}")
        return None

    # Sort by numeric prefix
    def get_prefix_number(file_path):
        match = re.match(r'(\d+)_', file_path.name)
        if match:
            return int(match.group(1))
        # Files without numeric prefix go to end, maintain alphabetical order
        return (float('inf'))#, file_path.name)

    sorted_files = sorted(mp4_files, key=get_prefix_number)

    # Create concat list file for ffmpeg
    concat_file = dir_path / "ffmpeg_concat_list.txt"
    try:
        with open(concat_file, 'w', encoding='utf-8') as f:
            for video_file in sorted_files:
                # Use absolute path with proper escaping for ffmpeg
                # Replace backslashes with forward slashes for ffmpeg on Windows
                abs_path = str(video_file.resolve()).replace('\\', '/')
                f.write(f"file '{abs_path}'\n")

        # Build ffmpeg command
        output_path = dir_path / output_file

        cmd = [
            'ffmpeg',
            '-f', 'concat',
            '-safe', '0',
            '-i', str(concat_file)
        ]

        if reencode:
            # Re-encode with multithreading
            cmd.extend([
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '23',
                '-threads', str(threads),
                '-c:a', 'aac',
                '-b:a', '192k'
            ])
            logger.info(f"Re-encoding with {threads} threads...")
        else:
            # Stream copy (fast, no re-encoding, but limited multithreading benefit)
            cmd.extend(['-c', 'copy'])
            logger.info("Using stream copy (no re-encoding)...")

        cmd.extend(['-y', str(output_path)])  # -y to overwrite without asking

        logger.info(f"\nConcatenating {len(sorted_files)} videos:")
        for i, f in enumerate(sorted_files, 1):
            logger.info(f"  {i}. {f.name}")
        logger.info(f"\nOutput: {output_path}\n")

        # Run ffmpeg
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )

        if result.returncode == 0:
            logger.info(f"✓ Successfully created {output_path.name}")
            logger.info(f"  Size: {output_path.stat().st_size / (1024*1024):.2f} MB")
            return output_path
        else:
            logger.error("✗ Error running ffmpeg:")
            logger.error(result.stderr)
            return None

    finally:
        # Clean up concat list file
        if concat_file.exists():
            concat_file.unlink()


def combine_scenes(dir):
    ext = None
    output_text_file = os.path.join(dir, "transcript.txt")
    transcript = ""
    video_files = []
    def starts_with_int(s):
        try:
            int(s.split('_')[0])
            return True
        except ValueError:
            return False
    for f in sorted([_ for _ in os.listdir(dir) if starts_with_int(_)], key=lambda x: int(x.split('_')[0])):
        if f.endswith('.mp4'):
            video_files.append(os.path.join(dir, f))
            if ext is None:
                ext = f.split('.')[-1]
        elif f.endswith('.txt'):
            with open(os.path.join(dir, f), 'r') as f:
                transcript += f.read()

    with open(output_text_file, 'w') as f:
        f.write(transcript)

    concatenate_videos(dir, output_file=f"video.{ext}")

