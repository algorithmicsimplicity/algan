import cProfile
import os.path

from pathlib import Path

import inspect
import pstats
import sys

import torch
import cv2

import algan
from algan import compiled
from algan.animation.animation_contexts import AnimationManager, Off
from algan.rendering.camera import Camera
from algan import SceneManager


@compiled
def render_to_file(file_name=None, output_dir=None, output_path=None, render_settings=None, overwrite=True, codec='h264', file_extension=None, **kwargs):
    """Runs all of the animations specified in the active :class:`~.Scene`, then renders the animations to video
    as captured by the active :class:`~.Camera`, and saves the video to a file.

    Parameters
    ----------
    file_name
        Name of the output file (without extension). If None will use `DEFAULT_OUTPUT_FILENAME`.
    output_dir
        Directory where to save the video. If None will use the directory of the running script.
    render_settings
        The :class:`.RenderSettings` object to use to specify video properties. If None will use `DEFAULT_RENDER_SETTINGS`.
    overwrite
        Whether the existing file at the output destination should be overwritten if one exists.
    codec
        The codec to use to encode the video frames.

    """
    with torch.inference_mode():
        if file_name is None:
            file_name = algan.defaults.directory_defaults.DEFAULT_OUTPUT_FILENAME
        if output_dir is None:
            output_dir = algan.defaults.directory_defaults.DEFAULT_OUTPUT_DIRECTORY
        if output_path is None:
            output_path = algan.defaults.directory_defaults.DEFAULT_OUTPUT_PATH
            if output_path is None:
                output_path = algan.defaults.directory_defaults.DEFAULT_DIRECTORY
        output_dir = os.path.join(output_path, output_dir)
        if render_settings is None:
            render_settings = algan.defaults.render_defaults.DEFAULT_RENDER_SETTINGS

        file_name, file_ext = os.path.splitext(file_name)
        if file_ext == '':
            file_ext = '.mp4'
        if file_extension is not None:
            file_ext = f'.{file_extension}'

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
        torch.cuda.empty_cache()

        temp_file_path = f'{temp_file_path}{file_ext}'
        file_path = f'{file_path}{file_ext}'
        file_writer = cv2.VideoWriter(temp_file_path, cv2.VideoWriter_fourcc(*codec),
                                      render_settings.frames_per_second, render_settings.resolution)

        try:
            if render_settings.save_image:
                with Off():
                    scene.clear_scene(animate=False)
            else:
                scene.clear_scene()
            print(f'Rendering {file_name}')
            scene.render_to_video(file_writer, temp_file_path, file_path, audio_file_path, **kwargs)
        finally:
            file_writer.release()
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            if os.path.exists(audio_file_path):
                os.remove(audio_file_path)


        SceneManager.reset()
        #AnimationManager.reset()
        #scene = SceneManager.instance()
        #scene.set_render_settings(render_settings)


@compiled
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
                render_settings = algan.defaults.render_defaults.DEFAULT_RENDER_SETTINGS

            if output_path is None:
                output_path = algan.defaults.directory_defaults.DEFAULT_OUTPUT_PATH
                if output_path is None:
                    output_path = algan.defaults.directory_defaults.DEFAULT_DIRECTORY
            if output_dir is None:
                output_dir = algan.defaults.directory_defaults.DEFAULT_OUTPUT_DIRECTORY
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
