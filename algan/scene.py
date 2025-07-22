from __future__ import annotations

import collections
import gc
import math
import os
import time
import warnings
import wave

import numpy as np
import torch
import torch.nn.functional as F

import algan
from algan import compiled

# from algan.rendering.lights import PointLight
from algan.animation.animation_contexts import AnimationManager, Off, Sync

# from algan.rendering.camera import Camera
from algan.constants.color import *
from algan.constants.spatial import *
from algan.logging.logger import LoggerManager
from algan.rendering.post_processing import bloom_filter
from algan.rendering.primitives.primitive import OutOfRenderMemory
from algan.settings.defaults import *
from algan.settings.style_defaults import STYLE_DEFAULTS
from algan.utils.memory_utils import ManualMemory, empty_cache, get_num_available_bytes
from algan.utils.tensor_utils import unsquish


class EmptySceneWarning(Warning):
    pass


class Scene:
    def __init__(
        self,
        background_frame=STYLE_DEFAULTS.frame,
        output_path="output",
        memory=None,
        render_settings=RENDERING_DEFAULTS.settings,
        scene_initializer=lambda x: x,
    ):
        self.set_render_settings(render_settings)
        self.current_time = 0
        self.min_time = 0
        self.max_time = 0
        if callable(background_frame):
            background_frame = background_frame(
                torch.stack(
                    (
                        torch.arange(self.num_pixels_screen_height)
                        .view(-1, 1)
                        .expand([-1, self.num_pixels_screen_width]),
                        torch.arange(self.num_pixels_screen_width)
                        .view(1, -1)
                        .expand([self.num_pixels_screen_height, -1]),
                    ),
                    -1,
                )
            )
        else:
            background_frame = background_frame
        self.background_frame = background_frame
        self.actors = [[]]
        self.effects = []
        self.scene_times = [(self.current_time, self.current_time)]
        self.background_depths = torch.full_like(
            self.background_frame[..., :1],
            dtype=torch.get_default_dtype(),
            fill_value=1e12,
        )
        self.animation_off = False
        self.output_path = output_path
        self.priority = 0
        self.id_count = 0
        # self.camera = None
        self.scene_initializer = scene_initializer
        self.reset_scene()
        self.allow_new_actors = True
        self.animate_scene_clear = False

        self.memory = memory

    @staticmethod
    def wait(time):
        return AnimationManager.wait(time)

    @staticmethod
    def instance():
        return algan.SceneManager.instance()

    @staticmethod
    def get_camera():
        return algan.SceneManager.instance().camera

    @staticmethod
    def get_light_sources():
        return algan.SceneManager.instance().light_sources

    @staticmethod
    def add_light_source(light_source):
        algan.SceneManager.instance().light_sources.append(light_source)

    def length_to_num_pixels(self, length):
        return length * 0.5 * self.num_pixels_screen_height

    def num_pixels_to_length(self, length):
        return length / (0.5 * self.num_pixels_screen_height)

    def set_current_time(self, t):
        self.current_time = t
        self.update_max_time(self.current_time)
        return self

    def increment_current_time(self, t):
        self.set_current_time(self.current_time + t)
        return self

    def update_max_time(self, t):
        self.context_max_time = max(self.context_max_time, t)
        self.max_time = max(self.max_time, t)
        return self

    def set_time_to_latest(self):
        self.current_time = self.max_time
        return self

    def get_actors(self):
        return [_ for _ in self.actors[-1] if not _.destroyed]

    def add_actor(self, actor):
        if self.allow_new_actors:
            self.actors[-1].append(actor)
        return self

    def add_effect(self, effect):
        self.effects.append(effect)
        return self

    def get_num_batches(self, start, end, batch_size):
        num_frames = int(end - start)
        num_batches = (max(num_frames - 1, 0) // batch_size) + 1
        return num_batches

    def initialize_frames(self):
        self.num_frames = int((self.max_time - self.min_time) * self.frames_per_second)
        return

    @staticmethod
    def clear():
        algan.SceneManager.instance().clear_scene()

    def clear_scene(self, **kwargs):
        with Sync():
            for actor in sorted(
                self.actors[-1], key=lambda x: x.anchor_priority, reverse=True
            ):
                if actor.data.spawn_time() >= 0:
                    actor.despawn(**kwargs)

    def get_audio(self, actors, start, end):
        active_actors = []
        time_inds = torch.arange(start, end)
        for _actor_id, actor in enumerate(
            sorted(actors, key=lambda x: x.anchor_priority, reverse=True)
        ):
            if (
                (not hasattr(actor, "spawn_ind"))
                or end <= actor.spawn_ind
                or actor.despawn_ind <= start
                or not hasattr(actor, "render_audio")
            ):
                continue
            active_actors.append(actor)
            actor.set_state_to_time_t(time_inds)

        if len(active_actors) == 0:
            nt = int(
                (end - start)
                * self.render_settings.audio_frames_per_second
                / self.frames_per_second
            )
            return torch.zeros((nt,)).cpu().numpy()
        return sum(a.render_audio() for a in active_actors)

    @compiled
    def render_primitive_batch(
        self,
        primitive_batch,
        start_ind,
        end_ind,
        save_image=False,
        post_processes=[],
        transparent_background=False,
        background_color=None,
    ):
        time_inds = torch.arange(start_ind, end_ind)
        camera = self.camera
        camera.screen.set_state_to_time_t(time_inds)
        camera.set_state_to_time_t(time_inds)
        camera.screen_width = (
            self.num_pixels_screen_width * self.render_settings.anti_alias_level
        )
        camera.screen_height = (
            self.num_pixels_screen_height * self.render_settings.anti_alias_level
        )
        for l in self.light_sources:
            l.set_state_to_time_t(time_inds)
            l.origin = l.location.unsqueeze(-2).to(COMPUTING_DEFAULTS.render_device)
            l.light_color = (
                (l.color[..., :-1] * l.color[..., -1:] * l.opacity)
                .unsqueeze(-2)
                .to(COMPUTING_DEFAULTS.render_device)
            )

        gc.collect()
        empty_cache()
        self.memory = ManualMemory(
            COMPUTING_DEFAULTS.portion_of_memory_used_for_rendering
        )
        logger = LoggerManager.instance().set_class("batching")
        for primitive in primitive_batch:
            logger.log_message(
                f"Pre-projecting primitive {primitive} with corners.shape: {primitive.corners.shape},"
                f"camera.location.shape: {camera.location.shape}, camera.ray_origin.shape: {camera.ray_origin.shape},"
                f"light_source.location.shape: {self.light_sources[0].location.shape}, "
                f"light_source.origin: {self.light_sources[0].origin.shape}"
            )
            primitive.memory = self.memory
            primitive.project_to_screen(camera, self.light_sources)

        current_ind = start_ind
        start_pointer = self.memory.current_pointer
        while True:
            duration = end_ind - current_ind
            while True:
                mem_used = sum(
                    [
                        _.get_memory_used(
                            current_ind - start_ind, current_ind + duration - start_ind
                        )
                        for _ in primitive_batch
                    ]
                )
                if mem_used <= self.memory.get_num_bytes_remaining():
                    break
                duration = duration // 2
                if duration <= 1:
                    duration = 1
                    break
            new_ind = current_ind + duration

            # time_inds = torch.arange(current_ind, new_ind)
            # camera.set_state_to_time_t(time_inds)
            # camera.screen.set_state_to_time_t(time_inds)
            # for l in self.light_sources:
            #    l.set_state_to_time_t(time_inds)

            primitive_batch[0].render(
                primitive_batch,
                self,
                save_image,
                self.num_pixels_screen_width,
                self.num_pixels_screen_height,
                current_ind - start_ind,
                new_ind - start_ind,
                self.background_frame if background_color is None else background_color,
                transparent_background,
                camera.ray_origin,
                camera.screen_point,
                camera.screen_basis,
                anti_alias_level=self.render_settings.anti_alias_level,
                light_sources=self.light_sources,
                memory=self.memory,
                post_processes=post_processes,
            )

            self.memory.current_pointer = start_pointer
            self.memory.max_pointer = start_pointer
            current_ind = new_ind
            if current_ind >= end_ind:
                break

        self.memory.data = None
        self.memory = None
        for actor in [self.camera, self.camera.screen, *self.light_sources]:
            actor.reset_state()

    def get_frame(self, i):
        actors = self.actors[-1]
        for _actor_id, actor in enumerate(
            sorted(actors, key=lambda x: x.anchor_priority, reverse=True)
        ):
            actor.set_state_full()
        self.camera.set_state_full()
        return next(
            self.get_frames_from_fragments(self.get_fragments(actors, i, i + 1))
        )

    def reset_scene(self):
        self.actors = [[]]
        self.scene_initializer(self)

    def set_render_settings(self, render_settings):
        self.render_settings = render_settings
        self.num_pixels_screen_width, self.num_pixels_screen_height = (
            render_settings.resolution
        )
        self.frame_size = torch.tensor(
            (self.num_pixels_screen_height, self.num_pixels_screen_width)
        )
        self.frames_per_second = render_settings.frames_per_second
        self.num_pixels = self.frame_size.prod()
        self.size = self.num_pixels_screen_width, self.num_pixels_screen_height

    def get_batch_of_primitives(
        self, start_time_ind, max_end_time_ind, actors, max_mem_used
    ):
        max_end_time = max_end_time_ind / self.frames_per_second
        start_time = start_time_ind / self.frames_per_second
        primitive_actors = [
            _
            for _ in actors
            if (_.data.spawn_time() <= max_end_time)
            and (_.data.despawn_time() >= start_time)
            and hasattr(_, "get_render_primitives")
        ]

        # Binary search to find a batch size that will fit in memory.
        duration = max_end_time_ind - start_time_ind
        duration = min(duration, COMPUTING_DEFAULTS.max_animate_batch_size)
        while True:
            selected_actors = [
                _
                for _ in primitive_actors
                if (
                    _.data.spawn_time()
                    <= (start_time_ind + duration) / self.frames_per_second
                )
            ]
            mem_used = sum(
                [_.get_memory_used_per_timestep() * duration for _ in selected_actors]
            )
            if mem_used <= max_mem_used:
                break
            duration = duration // 2
            if duration <= 1:
                duration = 1
                break
        logger = LoggerManager.instance().set_class("batching")
        logger.log_message(
            f"Fetching batch of primitives from {start_time_ind}:{start_time_ind + duration}."
        )
        actors = [
            _
            for _ in actors
            if (
                _.data.spawn_time()
                <= (start_time_ind + duration) / self.frames_per_second
            )
            and (_.data.despawn_time() >= start_time_ind / self.frames_per_second)
        ]
        time_inds = torch.arange(start_time_ind, start_time_ind + duration)

        grouped_primitives = collections.defaultdict(lambda: [None, []])
        for actor in sorted(actors, key=lambda x: x.anchor_priority, reverse=True):
            if hasattr(actor, "already_set_state") and actor.already_set_state:
                continue
            actor.set_state_full(start_time_ind, start_time_ind + duration)
            if hasattr(actor, "get_render_primitives"):
                actor.set_state_to_time_t(time_inds)
                for component in actor.components:
                    component.set_state_full(start_time_ind, start_time_ind + duration)
                    component.set_state_to_time_t(time_inds)
                primitive = actor.get_render_primitives()
                if primitive is not None:
                    grouped_primitives[primitive.get_batch_identifier()][0] = (
                        primitive.__class__
                    )
                    grouped_primitives[primitive.get_batch_identifier()][1].append(
                        primitive
                    )
            if not (
                actor == self.camera
                or actor == self.camera.screen
                or actor in self.light_sources
            ):
                actor.reset_state()
                for component in actor.components:
                    component.reset_state()

        primitive_collections = []
        for _, (primitive_class, primitives) in grouped_primitives.items():
            primitive_collections.append(
                primitive_class(triangle_collection=primitives)
            )
            primitive_collections[-1].memory = self.memory
            primitive_collections[-1].scene = self

        return primitive_collections, start_time_ind + duration

    def background_is_transparent(self):
        return (self.background_frame[..., -1].min() < 1).item()

    def get_pixel_format(self):
        return "rgba" if self.background_is_transparent() else "rgb"

    def render_to_video(
        self,
        file_writer,
        file_path,
        file_path_out,
        audio_file_path,
        batch_size_actors=None,
        batch_size_frames=None,
        post_processes=[bloom_filter],
        background_color=None,
    ):
        self.scene_times.append(
            (
                self.scene_times[-1][1],
                (
                    round(
                        AnimationManager.instance().context.end_time
                        * self.frames_per_second
                    )
                ),
            )
        )
        self.initialize_frames()
        self.original_background_frame = self.background_frame
        if background_color is not None:
            self.background_frame = background_color

        transparent_background = self.background_is_transparent()

        # self.camera.wait(1/self.frames_per_second + 1e-4)
        self.camera.despawn(animate=False)
        for l in self.light_sources:
            l.is_primitive = True
            l.despawn(animate=False)
        self.actors = [
            [self.camera, self.camera.screen, *self.light_sources, *self.actors[-1]]
        ]
        save_image = False

        self.has_any_active_actors = False
        with (
            Off(
                record_attr_modifications=False,
                record_funcs=False,
                priority_level=math.inf,
            ),
            wave.open(audio_file_path, "wb") as wav_file,
        ):
            wav_file.setnchannels(1)
            wav_file.setsampwidth(1)
            wav_file.setframerate(self.render_settings.audio_frames_per_second)
            for _scene_num, (actors, (scene_start, scene_end)) in enumerate(
                zip(self.actors, self.scene_times[-len(self.actors) :])
            ):
                if scene_end < scene_start:
                    continue
                if scene_end == scene_start and not save_image:
                    scene_end += 1
                    save_image = True
                    file_path = f"{file_path}.png"
                    file_path_out = f"{file_path_out}.png"

                self.file_path = file_path
                self.file_writer = file_writer

                current_time_ind = scene_start

                max_animate_mem = int(
                    COMPUTING_DEFAULTS.portion_of_memory_used_for_animating
                    * get_num_available_bytes(COMPUTING_DEFAULTS.render_device)
                )

                while True:
                    primitives, new_time_ind = self.get_batch_of_primitives(
                        current_time_ind, scene_end, actors, max_animate_mem
                    )
                    if new_time_ind <= current_time_ind:
                        raise OutOfRenderMemory(
                            "Insufficient memory to render this scene,"
                            "please reduce the number of Mobs used."
                        )
                    if len(primitives) > 0:
                        self.has_any_active_actors = True

                        s = time.time()
                        print(
                            f"Rendering {(new_time_ind - current_time_ind) / self.frames_per_second} seconds of video."
                        )
                        self.render_primitive_batch(
                            primitives,
                            current_time_ind,
                            new_time_ind,
                            save_image,
                            post_processes,
                            transparent_background,
                            background_color,
                        )
                        audio = self.get_audio(actors, current_time_ind, new_time_ind)
                        wav_file.writeframes(
                            bytes(((audio + 1) * 255 / 2).astype(np.uint8))
                        )
                        e = time.time()
                        print(
                            f"{current_time_ind}:{new_time_ind}, took {e - s} seconds"
                        )

                    current_time_ind = new_time_ind
                    if new_time_ind >= scene_end:
                        break

        self.background_frame = self.original_background_frame

        file_writer.close()
        if True:  # len(self.effects) == 0:
            if os.path.exists(file_path_out):
                os.remove(file_path_out)
            os.rename(file_path, file_path_out)
            if not self.has_any_active_actors:
                warnings.warn(
                    "You rendered an empty scene! Did you forget to spawn() your Mobs?",
                    EmptySceneWarning,
                    stacklevel=2,
                )
            return save_image
        # TODO fix this so we can write audio to the fiie as well.
        videoclip = VideoFileClip(file_path)
        try:
            # audioclip = AudioFileClip("audioname.mp3")

            videoclip = videoclip.set_audio(
                CompositeAudioClip(
                    [
                        effect.audio.subclip(0, videoclip.duration)
                        for effect in sorted(self.effects, key=lambda e: e.spawn_time())
                    ]
                )
            )

            if os.path.exists(file_path_out):
                os.remove(file_path_out)

            videoclip.write_videofile(file_path_out, codec="mpeg4")
        finally:
            videoclip.close()
            os.remove(file_path)

    @torch.compiler.disable(recursive=True)
    def get_frames_from_fragments(self, fragments, window, frame, anti_alias_level=1):
        device = fragments[0].device if fragments is not None else frame.device
        bgf = self.background_frame
        if bgf.shape[-1] == 3:
            bgf = torch.cat((bgf, torch.zeros_like(bgf[..., :1])), -1)
        bgf = (bgf * 255).to(device, torch.uint8)
        window_height = window[3] - window[1]
        window_width = window[2] - window[0]
        window_size = window_width * window_height

        if fragments is None:
            frame[:] = bgf[..., : frame.shape[-1]]
            frame_out = unsquish(frame, 0, -window_height)
            yield frame_out
            return
            frame_out = (
                F.avg_pool2d(frame_out.float().permute(2, 0, 1), anti_alias_level)
                .permute(1, 2, 0)
                .to(torch.uint8)
            )
            frame_out = bloom_filter(frame_out)
            yield frame_out.cpu().flip((-3, -1)).numpy()
            return

        frames, inds, num_pixels_in_frame = fragments
        frames = frames[..., : frame.shape[-1]]
        if inds is None:
            frame[:] = bgf
            frame = unsquish(frame, 0, -window_height)  # .cpu().flip((-3, -1)).numpy()
            for _ in range(len(frames)):
                yield frame
            return

        frame_ind_delimits = num_pixels_in_frame.cumsum(0)
        inds = inds % window_size
        inds = inds.unsqueeze(-1).expand([-1, frames.shape[-1]])
        frames = (frames * 255).to(torch.uint8)

        for i in range(len(frame_ind_delimits)):
            frame[:] = bgf[..., : frame.shape[-1]]
            ind_begin = frame_ind_delimits[i - 1] if i > 0 else 0
            ind_end = frame_ind_delimits[i]
            frame.scatter_(0, inds[ind_begin:ind_end], frames[ind_begin:ind_end])

            frame_out = unsquish(frame, 0, -window_height)
            yield frame_out

    def get_current_frame(self):
        return self.background_frame

    def get_new_id(self):
        self.id_count += 1
        return self.id_count - 1

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self
