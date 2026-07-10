import cProfile
import os.path
import time

from pathlib import Path

import multiprocessing
import re
import inspect
import pstats
import sys
import subprocess

import torch
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter

from algan.settings.defaults import *
from algan.settings.style_defaults import STYLE_DEFAULTS
from algan.animation.animation_contexts import Off
from algan.rendering.camera import Camera
from algan.scene_manager import SceneManager
from algan.sound.audio_effect import AudioManager
from algan.utils.memory_utils import empty_cache


def get_file_writer(temp_file_path, render_settings_resolution, codec, fps, with_mask, ffmpeg_params, audiofile, audio_codec):
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
    """Runs all of the animations specified in the active :class:`~.Scene`, then renders the animations to video
    as captured by the active :class:`~.Camera`, and saves the video to a file.

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

    """
    with torch.inference_mode():
        if file_name is None:
            file_name = DIRECTORY_DEFAULTS.output_filename
        if output_dir is None:
            output_dir = DIRECTORY_DEFAULTS.output_directory
        if output_path is None:
            output_path = DIRECTORY_DEFAULTS.output_path
            if output_path is None:
                output_path = DIRECTORY_DEFAULTS.base_directory
        output_dir = os.path.join(output_path, output_dir)
        if render_settings is None:
            render_settings = RENDERING_DEFAULTS.settings

        file_name, file_ext = os.path.splitext(file_name)

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        temp_file_path = os.path.join(output_dir, f"{file_name}_temp")
        file_path = os.path.join(output_dir, f"{file_name}")
        audio_file_path = os.path.join(output_dir, f"{file_name}_temp.wav")
        script_file_path = os.path.join(output_dir, f"{file_name}_script.txt")

        if os.path.exists(file_path) and not overwrite:
            return

        scene = SceneManager.instance()
        scene.set_render_settings(render_settings)
        if scene.camera is None:
            scene.camera = Camera(False)
        empty_cache()
        if background_color is None:
            background_color = STYLE_DEFAULTS.background_color
        scene.set_background_color(background_color)

        if animate_fade_out is None:
            animate_fade_out = STYLE_DEFAULTS.fade_out_on_scene_end
        if animate_fade_out:
            scene.clear_scene()
        else:
            with Off():
                scene.clear_scene(animate=False)

        if file_ext == "":
            file_ext = ".mov" if scene.background_is_transparent() else ".mp4"
        if file_extension is not None:
            file_ext = f".{file_extension}"
        temp_file_path = f"{temp_file_path}{file_ext}"
        file_path = f"{file_path}{file_ext}"

        if file_ext in [".mp4"] and scene.background_is_transparent():
            raise ValueError(
                f"You are trying to render a scene with a transparent background to a file format which"
                f"does not support alpha channels ({file_ext}). Please use a file format that supports"
                f"alpha channels such as .mov or .webm, or change the background color to be opaque."
            )

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

        print(f"Began rendering {file_name}{file_ext}")
        audiofile = scene.render_audio_to_file(audio_file_path, audio_fps,
                                               nbytes=4, codec='pcm_s32le', )
        if audiofile is not None:
            with open(script_file_path, "w") as f:
                f.write(AudioManager._video_transcript)
            print("Audio rendered, now rendering video")

        file_writer = get_file_writer(temp_file_path,
                    render_settings.resolution,
                    codec,
                    render_settings.frames_per_second,
                    scene.background_is_transparent(),
                    ffmpeg_params,
                    audiofile,
                    audio_codec)

        try:
            scene.render_to_video(file_writer, temp_file_path, file_path, **kwargs)
            print(f"Finished rendering {file_name}{file_ext}")
        finally:
            # file_writer.release()
            file_writer.close()
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            if os.path.exists(audio_file_path):
                os.remove(audio_file_path)

        SceneManager.reset()
        AudioManager.reset()
        # AnimationManager.reset()
        # scene = SceneManager.instance()
        # scene.set_render_settings(render_settings)


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
    **kwargs,
):
    def run(output_dir=None, render_settings=None, output_path=None):
        with torch.inference_mode():
            module = sys.modules[module_name]
            scene_funcs = [
                a
                for a in inspect.getmembers(module)
                if inspect.isfunction(a[1])
                and a[1].__globals__["__file__"] == inspect.getfile(module)
                and len(inspect.signature(a[1]).parameters) == 0
            ]
            scene_funcs = list(
                sorted(scene_funcs, key=lambda x: x[1].__code__.co_firstlineno)
            )

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
                s = start_index + len(scene_funcs)
            else:
                s = start_index
            if max_rendered < 0:
                e = len(scene_funcs)
            else:
                e = s + max_rendered
            for i, (func_name, f) in list(enumerate(scene_funcs))[s:e]:
                scene = SceneManager.reset()
                scene.set_render_settings(render_settings)
                if 'background_color' in kwargs:
                    scene.set_background_color(kwargs['background_color'])
                f()
                if not smoke_test:
                    render_to_file(
                        f"{i}_{func_name}.{file_extension}",
                        output_dir,
                        output_path,
                        render_settings,
                        overwrite,
                        **kwargs,
                    )

            #combine_scenes(output_dir)
            return

    if profile:
        pr = cProfile.Profile()
        start = time.time()
        pr.enable()
        out = run(output_dir, render_settings, output_path)
        pr.disable()
        end = time.time()

        with open('profiler_dump.txt', 'w') as f:
            ps = pstats.Stats(pr, stream=f).sort_stats(pstats.SortKey.CUMULATIVE)
            ps.print_stats()
        ps = pstats.Stats(pr).sort_stats(pstats.SortKey.CUMULATIVE)
        ps.print_stats()
        print(f'took {end-start} seconds.')
        return out
    else:
        return run(output_dir, render_settings, output_path)


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
    print(f'took {end - start} seconds.')
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
        print(f"No .mp4 files found in {directory}")
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
            print(f"Re-encoding with {threads} threads...")
        else:
            # Stream copy (fast, no re-encoding, but limited multithreading benefit)
            cmd.extend(['-c', 'copy'])
            print("Using stream copy (no re-encoding)...")

        cmd.extend(['-y', str(output_path)])  # -y to overwrite without asking

        print(f"\nConcatenating {len(sorted_files)} videos:")
        for i, f in enumerate(sorted_files, 1):
            print(f"  {i}. {f.name}")
        print(f"\nOutput: {output_path}\n")

        # Run ffmpeg
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )

        if result.returncode == 0:
            print(f"✓ Successfully created {output_path.name}")
            print(f"  Size: {output_path.stat().st_size / (1024*1024):.2f} MB")
            return output_path
        else:
            print(f"✗ Error running ffmpeg:")
            print(result.stderr)
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
        except:
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

