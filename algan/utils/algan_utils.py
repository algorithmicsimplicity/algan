import cProfile
import os.path

from pathlib import Path

import inspect
import pstats
import sys

import torch
import cv2
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter

import algan
from algan.settings.defaults import *
from algan.settings.style_defaults import STYLE_DEFAULTS
from algan import compiled
from algan.animation.animation_contexts import AnimationManager, Off
from algan.rendering.camera import Camera
from algan import SceneManager
from algan.utils.memory_utils import empty_cache

#‘mpeg4’ > ‘libx264’
#@compiled
def render_to_file(file_name=None, output_dir=None, output_path=None, render_settings=None, overwrite=True, codec=None,
                   file_extension=None, background_color=None, ffmpeg_params=None, animate_fade_out=None, **kwargs):
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

        if os.path.exists(file_path) and not overwrite:
            return

        scene = SceneManager.instance()
        scene.set_render_settings(render_settings)
        if scene.camera is None:
            scene.camera = Camera(False)
        empty_cache()
        if background_color is None:
            background_color = STYLE_DEFAULTS.background_color
        scene.background_frame = scene.background_color = background_color

        if file_ext == '':
            file_ext = '.mov' if scene.background_is_transparent() else '.mp4'
        if file_extension is not None:
            file_ext = f'.{file_extension}'
        temp_file_path = f'{temp_file_path}{file_ext}'
        file_path = f'{file_path}{file_ext}'

        if file_ext in ['.mp4'] and scene.background_is_transparent():
            raise ValueError(f'You are trying to render a scene with a transparent background to a file format which'
                             f'does not support alpha channels ({file_ext}). Please use a file format that supports'
                             f'alpha channels such as .mov or .webm, or change the background color to be opaque.')

        if codec is None:
            codec = 'png' if scene.background_is_transparent() else 'libx264'
        if ffmpeg_params is None:
            ffmpeg_params = [
                '-crf', '15',
                '-preset', 'veryslow'
            ] if not scene.background_is_transparent() else []
        try:
            file_writer = FFMPEG_VideoWriter(temp_file_path, size=render_settings.resolution, codec=codec,
                                             fps=render_settings.frames_per_second, with_mask=scene.background_is_transparent(),
                                             ffmpeg_params=ffmpeg_params)
        except TypeError:
            file_writer = FFMPEG_VideoWriter(temp_file_path, size=render_settings.resolution, codec=codec,
                                             fps=render_settings.frames_per_second,
                                             withmask=scene.background_is_transparent(),
                                             ffmpeg_params=ffmpeg_params)

        try:
            if animate_fade_out is None:
                animate_fade_out = STYLE_DEFAULTS.fade_out_on_scene_end
            if animate_fade_out:
                scene.clear_scene()
            else:
                with Off():
                    scene.clear_scene(animate=False)
            print(f'Began rendering {file_name}{file_ext}')
            scene.render_to_video(file_writer, temp_file_path, file_path, audio_file_path, **kwargs)
            print(f'Finished rendering {file_name}{file_ext}')
        finally:
            #file_writer.release()
            file_writer.close()
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            if os.path.exists(audio_file_path):
                os.remove(audio_file_path)


        SceneManager.reset()
        #AnimationManager.reset()
        #scene = SceneManager.instance()
        #scene.set_render_settings(render_settings)


#@compiled
def render_all_funcs(module_name, render_settings=None, profile=True, overwrite=True, start_index=0,
                     max_rendered=-1, output_dir=None, output_path=None, file_extension='mp4', **kwargs):
    def run(output_dir=None, render_settings=None, output_path=None):
        with torch.inference_mode():
            module = sys.modules[module_name]
            scene_funcs = [a for a in inspect.getmembers(module) if inspect.isfunction(a[1]) and
                           a[1].__globals__['__file__'] == inspect.getfile(module) and
                           len(inspect.signature(a[1]).parameters) == 0]
            scene_funcs = list(sorted(scene_funcs, key=lambda x: x[1].__code__.co_firstlineno))

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
                e = s+max_rendered
            for i, (func_name, f) in list(enumerate(scene_funcs))[s:e]:
                scene = SceneManager.reset()
                scene.set_render_settings(render_settings)
                f()
                render_to_file(f'{i}_{func_name}.{file_extension}', output_dir, output_path, render_settings, overwrite, **kwargs)
            return

    if profile:
        pr = cProfile.Profile()
        pr.enable()
        out = run(output_dir, render_settings, output_path)
        pr.disable()

        ps = pstats.Stats(pr).sort_stats(pstats.SortKey.CUMULATIVE)
        ps.print_stats()
        return out
    else:
        return run(output_dir, render_settings, output_path)
