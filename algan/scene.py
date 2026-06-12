import collections
import gc
import math
import multiprocessing
import os
import threading
import time
import wave
import warnings
from queue import Queue

import torch
import torch.nn.functional as F
import torchvision.utils
from moviepy import CompositeAudioClip

import algan
from algan import csync
from algan import not_compiled
from algan.rendering.primitives.bezier_circuit_primitive import BezierCircuitPrimitive
from algan.settings.defaults import *
from algan.settings.style_defaults import STYLE_DEFAULTS
from algan import compiled

# from algan.rendering.camera import Camera
from algan.constants.color import *
from algan.constants.spatial import *

# from algan.rendering.lights import PointLight
from algan.animation.animation_contexts import Seq, Sync, AnimationManager, Off
import numpy as np

from algan.rendering.post_processing.bloom import bloom_filter
from algan.rendering.primitives.primitive import OutOfRenderMemory
from algan.utils.memory_utils import get_num_available_bytes, ManualMemory, empty_cache
from algan.utils.tensor_utils import unsquish, wait_for_cuda
from algan.utils.file_utils import get_image


class EmptySceneWarning(Warning):
    pass


def write_frames_from_queue(queue, file_writer):
    #with get_file_writer() as file_writer:
    while True:
        frame = queue.get()
        if frame is None:  # Sentinel value to signal the end
            break
        file_writer.write_frame(frame.numpy())


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
        self.background_is_set = False
        if hasattr(background_frame, "__call__"):
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
        self.scene_times = [[self.current_time, self.current_time]]
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
    def wait(time=1):
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
        num_frames = int((end - start))
        num_batches = (max(num_frames - 1, 0) // batch_size) + 1
        return num_batches

    def initialize_frames(self):
        self.num_frames = int((self.max_time - self.min_time) * self.frames_per_second)
        return

    @staticmethod
    def clear():
        algan.SceneManager.instance().clear_scene()

    def despawn_scene(self, **kwargs):
        with Sync():
            for actor in list(
                sorted(self.actors[-1], key=lambda x: x.anchor_priority, reverse=True)
            ):
                if actor.data.spawn_time() >= 0:
                    actor.despawn(**kwargs)

    def clear_scene(self, **kwargs):
        with Seq(run_time=0.5):
            self.despawn_scene(**kwargs)
        self.actors[-1] = [_ for _ in self.actors[-1] if (_.data.spawn_time() >= 0 and _.data.despawn_time() >= 0)]

    def render_audio_to_file(self, file_path, frames_per_second=44100, codec='pcm_s32le', nbytes=4):
        if len(self.effects) == 0:
            return None

        clips_to_compose = []
        start_time = self.scene_times[-1][0] / self.render_settings.frames_per_second
        for audio_effect in self.effects:
            timed_clip = audio_effect.audio_clip.with_start(
                audio_effect.start_time_func() - start_time
            )
            clips_to_compose.append(timed_clip)

        audio_clip = CompositeAudioClip(clips_to_compose)
        audio_clip.duration = AnimationManager.instance().context.end_time
        audio_clip.write_audiofile(file_path, fps=frames_per_second, codec=codec, nbytes=nbytes)
        audio_clip.close()
        return file_path

    #@compiled
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
        wait_for_cuda()
        with (torch.no_grad()):
            time_inds = torch.arange(start_ind, end_ind)
            camera = self.camera
            camera.screen.reset_state()
            camera.reset_state()
            camera.set_state_full(time_inds[0], time_inds[-1]+1)
            camera.set_state_to_time_t(time_inds)
            camera.screen.set_state_full(time_inds[0], time_inds[-1] + 1)
            camera.screen.set_state_to_time_t(time_inds)
            camera.ray_origin = camera.location.unsqueeze(-2).to(COMPUTING_DEFAULTS.render_device)
            camera.screen_point = camera.screen.location.unsqueeze(-2).to(COMPUTING_DEFAULTS.render_device)
            camera.screen_basis = camera.get_render_screen_basis().to(COMPUTING_DEFAULTS.render_device)
            camera.screen_width = (
                self.num_pixels_screen_width * self.render_settings.anti_alias_level
            )
            camera.screen_height = (
                self.num_pixels_screen_height * self.render_settings.anti_alias_level
            )
            for l in self.light_sources:
                l.reset_state()
                l.set_state_full(time_inds[0], time_inds[-1]+1)
                l.set_state_to_time_t(time_inds)
                l.origin = l.location.unsqueeze(-2).to(COMPUTING_DEFAULTS.render_device)
                l.light_color = (
                    (l.color[..., :-1] * l.color[..., -1:] * l.opacity)
                    .unsqueeze(-2)
                    .to(COMPUTING_DEFAULTS.render_device)
                )

            #torch.compiler.cudagraph_mark_step_begin()
            self.memory.scene = self
            original_pointers = self.memory.get_pointers()
            for primitive in primitive_batch:
                primitive.memory = self.memory
                primitive.project_to_screen(camera, self.light_sources)

            render_pointers = self.memory.get_pointers()
            current_ind = start_ind
            num_bytes_for_post_processing_per_frame = self.num_pixels_screen_width * self.num_pixels_screen_height * 5 * 4 * 4
            while True:
                mem_per_time_step = max(max([_.get_memory_used(0, 1) - _.get_memory_used_for_blending(0, 1)
                     for _ in primitive_batch]) + max([_.get_memory_used_for_blending(0, 1) for _ in primitive_batch]),
                                        num_bytes_for_post_processing_per_frame)
                duration = int(self.memory.get_num_bytes_remaining() // mem_per_time_step)
                duration = min(duration, end_ind - current_ind)
                duration = max(duration, 1)
                #duration = end_ind - current_ind
                while False:
                    mem_used = max(
                        [
                            _.get_memory_used(
                                current_ind - start_ind, current_ind + duration - start_ind
                            ) - _.get_memory_used_for_blending(current_ind - start_ind, current_ind + duration - start_ind)
                            for _ in primitive_batch
                        ]
                    ) + sum([_.get_memory_used_for_blending(
                                current_ind - start_ind, current_ind + duration - start_ind
                            )
                            for _ in primitive_batch]) + (self.num_pixels_screen_width * self.num_pixels_screen_height * 5)
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
                print(f'rendering batch with duration {duration}')

                bgf = self.background_frame if background_color is None else background_color
                if hasattr(bgf, '__call__'):
                    device = camera.ray_origin.device
                    aa = self.render_settings.anti_alias_level
                    x = torch.arange(self.num_pixels_screen_width * aa, device=device).view(1,-1,1)
                    y = torch.arange(self.num_pixels_screen_height * aa, device=device).view(-1,1,1)
                    bgf = bgf(x / (self.num_pixels_screen_width * aa), y / (self.num_pixels_screen_height * aa),
                              torch.arange(current_ind, new_ind, device=device).view(-1,1,1,1) / self.frames_per_second)
                if bgf.dim() > 1:
                    if bgf.shape[0] == 1:
                        bgf = bgf.expand(new_ind - current_ind, *[-1 for _ in range(bgf.dim()-1)]).contiguous()
                    bgf = bgf.view(-1,bgf.shape[-1])
                    bgf = torch.cat((bgf[:1], bgf))
                    bgf = ((bgf + (0.5/255)) * 255).to(torch.uint8).clamp_max_(255)
                yield primitive_batch[0].render(
                    primitive_batch,
                    self,
                    save_image,
                    self.num_pixels_screen_width,
                    self.num_pixels_screen_height,
                    current_ind - start_ind,
                    new_ind - start_ind,
                    bgf,
                    transparent_background,
                    camera.ray_origin,
                    camera.screen_point,
                    camera.screen_basis,
                    anti_alias_level=self.render_settings.anti_alias_level,
                    light_sources=self.light_sources,
                    memory=self.memory,
                    post_processes=post_processes,
                )

                self.memory.set_pointers(render_pointers)
                current_ind = new_ind
                if current_ind >= end_ind:
                    break

            self.memory.set_pointers(original_pointers)
            self.memory.max_pointer = self.memory.current_pointer = (len(self.memory) - self.memory.current_reverse_pointer)
            for actor in [self.camera, self.camera.screen, *self.light_sources]:
                actor.reset_state()
            wait_for_cuda()

    def get_frame(self, i):
        actors = self.actors[-1]
        for actor_id, actor in enumerate(
            list(sorted(actors, key=lambda x: x.anchor_priority, reverse=True))
        ):
            actor.set_state_full()
        self.camera.set_state_full()
        return next(
            self.get_frames_from_fragments(self.get_fragments(actors, i, i + 1))
        )

    def reset_scene(self):
        self.actors = [[]]
        self.effects = []
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

    @csync
    def get_batch_of_primitives(
        self, start_time_ind, max_end_time_ind, actors, max_mem_used
    ):
        max_end_time = max_end_time_ind / self.frames_per_second
        start_time = start_time_ind / self.frames_per_second
        primitive_actors = [
            _
            for _ in actors
            if (_.data.spawn_time() <= max_end_time)
            and ((_.data.despawn_time() >= start_time) or _.data.despawn_time() < 0)
            and hasattr(_, "get_render_primitives")
        ]

        # Binary search to find a batch size that will fit in memory.
        @csync
        def get_duration():
            #return 90
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
                    [
                        _.get_memory_used_per_timestep() * duration
                        for _ in selected_actors
                    ]
                )
                if mem_used <= max_mem_used:
                    break
                duration = duration // 2
                if duration <= 1:
                    duration = 1
                    break
            wait_for_cuda()
            return duration

        duration = get_duration()
        actors = [
            _
            for _ in actors
            if (
                _.data.spawn_time()
                <= (start_time_ind + duration) / self.frames_per_second
            )
            and ((_.data.despawn_time() >= start_time_ind / self.frames_per_second) or (_.data.despawn_time() < 0))
        ]
        time_inds = torch.arange(start_time_ind, start_time_ind + duration)

        grouped_primitives = collections.defaultdict(lambda: [None, []])
        for actor in sorted(actors, key=lambda x: x.anchor_priority, reverse=True):
            if hasattr(actor, "already_set_state") and actor.already_set_state:
                continue
            if (not actor.is_primitive) and len(list(actor.data.history.function_applications.items())) == 0:
                actor.reset_state()
                continue
            actor.set_state_full(start_time_ind, start_time_ind + duration)
            if hasattr(actor, "get_render_primitives"):
                actor.set_state_to_time_t(time_inds)
                for component in actor.components:
                    component.set_state_full(start_time_ind, start_time_ind + duration)
                    component.set_state_to_time_t(time_inds)
                wait_for_cuda()
                primitive = actor.get_render_primitives()
                if primitive is not None:
                    if not isinstance(primitive, list):
                        primitive = [primitive]
                    for p in primitive:
                        grouped_primitives[p.get_batch_identifier()][0] = p.__class__
                        grouped_primitives[p.get_batch_identifier()][1].append(p)
            if not (
                actor == self.camera
                or actor == self.camera.screen
                or actor in self.light_sources
            ):
                actor.reset_state()
                for component in actor.components:
                    component.reset_state()

        primitive_collections = []
        max_bezier_batch_size = 50000
        gc.collect()
        torch.cuda.empty_cache()
        for _, (primitive_class, primitives) in grouped_primitives.items():
            if primitive_class is BezierCircuitPrimitive:
                counts = torch.tensor([_.corners.shape[1] for _ in primitives]).cumsum(
                    0
                )
                num_sub_batches = (counts[-1] // max_bezier_batch_size) + 1
                current_ind = 0
                for i in range(num_sub_batches):
                    inds = (counts > max_bezier_batch_size).nonzero()
                    if len(inds) == 0:
                        next_ind = len(primitives)
                    else:
                        next_ind = max(inds[0], current_ind + 1)
                    primitive_collections.append(
                        primitive_class(
                            triangle_collection=primitives[current_ind:next_ind]
                        )
                    )
                    current_ind = next_ind
                    primitive_collections[-1].memory = self.memory
                    primitive_collections[-1].scene = self
                    if current_ind >= len(primitives):
                        break
                    counts -= counts[current_ind - 1]
            else:
                primitive_collections.append(
                    primitive_class(triangle_collection=primitives)
                )
                primitive_collections[-1].memory = self.memory
                primitive_collections[-1].scene = self

        wait_for_cuda()
        return primitive_collections, start_time_ind + duration

    def background_is_transparent(self):
        if hasattr(self.background_frame, '__call__'):
            return False
        return (self.background_frame[..., -1].min() < (1-(0.5/255))).item()

    def get_pixel_format(self):
        return "rgba" if self.background_is_transparent() else "rgb"

    def show_frame(self, time_stamp=None):
        from algan.utils.plotting_utils import plot_tensor
        if time_stamp is None:
            time_stamp = AnimationManager.instance().context.current_time + 1.5/self.render_settings.frames_per_second
        time_ind = round(time_stamp * self.render_settings.frames_per_second)
        frames = []
        for frame in self.get_frames(time_ind-1, time_ind):
            frame = frame.float() / 255
            frames.append(frame.squeeze(0).permute(-1,0,1))
        for frame in frames:
            plot_tensor(frame)

        return frames

    def save_frame(self, filename, time_stamp=None):
        if not COMPUTING_DEFAULTS.allow_save_frame:
            return
        from algan.utils.plotting_utils import plot_tensor
        if time_stamp is None:
            time_stamp = AnimationManager.instance().context.current_time + 1.5/self.render_settings.frames_per_second
        time_ind = round(time_stamp * self.render_settings.frames_per_second)
        frames = []
        for frame in self.get_frames(time_ind-1, time_ind):
            frame = frame.float() / 255
            frames.append(frame.squeeze(0).permute(-1,0,1))
        torchvision.utils.save_image(frames[-1], filename)
        return frames

    def get_frames(self, start_time_ind, end_time_ind, background_color=None, post_processes=[bloom_filter], manual_memory=True):
        if end_time_ind <= start_time_ind:
            yield []
            return

        self.original_background_frame = self.background_frame
        if background_color is not None:
            self.background_frame = background_color

        transparent_background = self.background_is_transparent()

        # self.camera.wait(1/self.frames_per_second + 1e-4)
        for l in self.light_sources:
            l.is_primitive = True
        actors = [self.camera, self.camera.screen, *self.light_sources, *self.actors[-1]]
        save_image = False

        self.has_any_active_actors = False
        gc.collect()
        if COMPUTING_DEFAULTS.render_device == torch.device('cuda'):
            torch.cuda.empty_cache()
        self.memory = ManualMemory(
            COMPUTING_DEFAULTS.portion_of_memory_used_for_rendering, managed=manual_memory,
        )

        with Off(
                record_attr_modifications=False,
                record_funcs=False,
                priority_level=math.inf,
        ):
            current_time_ind = start_time_ind

            max_animate_mem = int(
                    COMPUTING_DEFAULTS.portion_of_memory_used_for_animating
                    * get_num_available_bytes(COMPUTING_DEFAULTS.render_device)
                )

            while True:
                primitives, new_time_ind = self.get_batch_of_primitives(
                    current_time_ind, end_time_ind, actors, max_animate_mem
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
                    yield from self.render_primitive_batch(
                        primitives,
                        current_time_ind,
                        new_time_ind,
                        save_image,
                        post_processes,
                        transparent_background,
                        background_color,
                    )
                    del primitives
                    e = time.time()
                    print(
                        f"{current_time_ind}:{new_time_ind}, took {e - s} seconds"
                    )

                current_time_ind = new_time_ind
                if new_time_ind >= end_time_ind:
                    break

        self.background_frame = self.original_background_frame

        self.memory.data = None
        self.memory = None

    def set_background_color(self, background_color, overwrite=False):
        if self.background_is_set and not overwrite:
            return self
        if isinstance(background_color, str):
            background_color = F.interpolate(get_image(background_color).transpose(0,-1).unsqueeze(0), tuple(self.frame_size),
                                             mode='bilinear', antialias='bilinear').squeeze(0).permute(1,2,0).unsqueeze(0)
        self.background_frame = self.background_color = background_color
        self.background_is_set = True
        return self

    def get_background_color(self):
        return self.background_color

    def render_to_video(
        self,
        file_writer,
        file_path,
        file_path_out,
        post_processes=[bloom_filter],
        background_color=None,
    ):
        self.scene_times.append(
            [
                self.scene_times[-1][0],
                (
                    round(
                        AnimationManager.instance().context.end_time
                        * self.frames_per_second
                    )
                ),
            ]
        )
        self.initialize_frames()

        self.camera.despawn(animate=False)
        for l in self.light_sources:
            l.despawn(animate=False)

        self.file_path = file_path
        self.file_writer = file_writer

        '''
                frame_queue = multiprocessing.Queue(maxsize=40)
                writer_process = multiprocessing.Process(
                    target=write_frames_from_queue,
                    args=(frame_queue, file_writer)
                )
                '''
        frame_queue = Queue(maxsize=8)
        writer_process = threading.Thread(target=write_frames_from_queue, args=(frame_queue, file_writer))
        writer_process.daemon = True
        writer_process.start()

        self.frame_queue = frame_queue
        # Wait for the writer process to complete
        for frame_batch in self.get_frames(*self.scene_times[-1], background_color=background_color,
                                           post_processes=post_processes, manual_memory=True):
            for frame in frame_batch:
                frame_queue.put(frame)

        self.frame_queue.put(None)
        writer_process.join()
        file_writer.close()

        #file_writer.close()
        if os.path.exists(file_path_out):
            os.remove(file_path_out)
        os.rename(file_path, file_path_out)
        if not self.has_any_active_actors:
            warnings.warn(
                "You rendered an empty scene! Did you forget to spawn() your Mobs?",
                EmptySceneWarning,
            )

    @not_compiled  #@torch.compiler.disable(recursive=True)
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
            for i in range(len(frames)):
                yield frame
            return

        frame_ind_delimits = num_pixels_in_frame.cumsum(0)
        inds = inds % window_size
        inds = inds.unsqueeze(-1).expand([-1, frames.shape[-1]])
        frames *= 255
        frames = self.memory.cast(frames, torch.uint8)

        for i in range(len(frame_ind_delimits)):
            frame[:] = bgf[..., :frame.shape[-1]]
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
