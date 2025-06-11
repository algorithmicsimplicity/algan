import collections
import math
import os
import time
import wave
import warnings

import torch
import torch.nn.functional as F

from algan.animation.animation_contexts import Sync, AnimationManager, Off
from algan.defaults.batch_defaults import DEFAULT_BATCH_SIZE_FRAMES, DEFAULT_BATCH_SIZE_ACTORS, \
    DEFAULT_PORTION_MEMORY_USED_FOR_RENDERING
from algan.defaults.device_defaults import DEFAULT_RENDER_DEVICE
from algan.defaults.render_defaults import DEFAULT_RENDER_SETTINGS
from algan.defaults.style_defaults import DEFAULT_FRAME
import numpy as np

from algan.rendering.post_processing import bloom_filter
from algan.utils.memory_utils import ManualMemory
from algan.utils.tensor_utils import unsquish


class EmptySceneWarning(Warning):
    pass


class Scene:
    def __init__(self, background_frame=DEFAULT_FRAME, output_path='output', memory=None, render_settings=DEFAULT_RENDER_SETTINGS):
        self.set_render_settings(render_settings)
        self.current_time = 0
        self.min_time = 0
        self.max_time = 0
        if hasattr(background_frame, '__call__'):
            background_frame = background_frame(torch.stack((torch.arange(self.num_pixels_screen_height).view(-1,1).expand(-1,self.num_pixels_screen_width),
                                                torch.arange(self.num_pixels_screen_width).view(1,-1).expand(self.num_pixels_screen_height, -1)), -1))
        else:
            background_frame = background_frame
        self.background_frame = background_frame
        self.actors = [[]]
        self.effects = []
        self.scene_times = [(self.current_time, self.current_time)]
        self.background_depths = torch.full_like(self.background_frame[...,:1], dtype=torch.get_default_dtype(), fill_value=1e12)
        self.animation_off = False
        self.output_path = output_path
        self.priority = 0
        self.id_count = 0
        self.camera = None
        self.allow_new_actors = True

        self.memory = memory

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

    def clear_scene(self, **kwargs):
        with Sync():
            for actor in list(sorted(self.actors[-1], key=lambda x: x.anchor_priority, reverse=True)):
                actor.despawn(**kwargs)
        AnimationManager.wait()

    def get_audio(self, actors, start, end):
        active_actors = []
        time_inds = torch.arange(start, end)
        for actor_id, actor in enumerate(list(sorted(actors, key=lambda x: x.anchor_priority, reverse=True))):
            if end <= actor.spawn_ind or actor.despawn_ind <= start or not hasattr(actor, 'render_audio'):
                continue
            active_actors.append(actor)
            actor.set_state_to_time_t(time_inds)

        if len(active_actors) == 0:
            nt = int((end-start) * self.render_settings.audio_frames_per_second / self.frames_per_second)
            return torch.zeros((nt,)).cpu().numpy()
        return sum((a.render_audio() for a in active_actors))

    def get_fragments(self, actors, start, end, save_image=False):
        camera = self.camera
        nt = end-start
        active_actors = []
        time_inds = torch.arange(start, end)
        for actor_id, actor in enumerate(list(sorted(actors, key=lambda x: x.anchor_priority, reverse=True))):
            if end <= actor.spawn_ind or actor.despawn_ind <= start or not actor.is_primitive:
                continue
            active_actors.append(actor)
            actor.set_state_to_time_t(time_inds)

        active_actors = [a for a in active_actors if hasattr(a, 'get_render_primitives')]
        if len(active_actors) == 0:
            return torch.empty((nt,)), None, None
        self.has_any_active_actors = True
        camera.set_state_to_time_t(time_inds)
        camera.screen.set_state_to_time_t(time_inds)

        grouped_primitives = collections.defaultdict(lambda: [None, []])

        for primitive in [actor.get_render_primitives() for actor in active_actors]:
            if primitive is None:
                continue
            grouped_primitives[primitive.get_batch_identifier()][0] = primitive.__class__
            grouped_primitives[primitive.get_batch_identifier()][1].append(primitive)

        # Return the values (the lists of grouped items) from the dictionary.
        primitive_collections = []
        for _, (primitive_class, primitives) in grouped_primitives.items():
            primitive_collections.append(primitive_class(triangle_collection=primitives))
            primitive_collections[-1].memory = self.memory
            primitive_collections[-1].scene = self
        self.memory.reset()
        return primitive_collections[0].render(primitive_collections, self, save_image, self.num_pixels_screen_width,
                                               self.num_pixels_screen_height, self.background_frame, camera.location.to(DEFAULT_RENDER_DEVICE, non_blocking=True),
                                               camera.screen.location.to(DEFAULT_RENDER_DEVICE, non_blocking=True),
                                               camera.screen.basis.to(DEFAULT_RENDER_DEVICE, non_blocking=True),
                                               anti_alias_level=self.render_settings.anti_alias_level,
                                               light_origin=camera.light_source_location.to(DEFAULT_RENDER_DEVICE),
                                               light_color=camera.light_color.to(DEFAULT_RENDER_DEVICE), memory=self.memory)

    def get_frame(self, i):
        actors = self.actors[-1]
        for actor_id, actor in enumerate(list(sorted(actors, key=lambda x: x.anchor_priority, reverse=True))):
            actor.set_state_full()
        self.camera.set_state_full()
        return next(self.get_frames_from_fragments(self.get_fragments(actors, i, i+1)))

    def reset_scene(self):
        self.actors = [[]]

    def set_render_settings(self, render_settings):
        self.render_settings = render_settings
        self.num_pixels_screen_width, self.num_pixels_screen_height = render_settings.resolution
        self.frame_size = torch.tensor((self.num_pixels_screen_height, self.num_pixels_screen_width))
        self.frames_per_second = render_settings.frames_per_second
        self.num_pixels = self.frame_size.prod()
        self.size = self.num_pixels_screen_width, self.num_pixels_screen_height

    def render_to_video(self, file_writer, file_path, file_path_out, audio_file_path,
                        batch_size_actors=DEFAULT_BATCH_SIZE_ACTORS, batch_size_frames=DEFAULT_BATCH_SIZE_FRAMES):
        self.scene_times.append((self.scene_times[-1][1], (math.ceil(AnimationManager.instance().context.end_time * self.frames_per_second)+1)))
        self.initialize_frames()

        self.camera.despawn(animate=False)
        self.actors = [[self.camera, self.camera.screen, *self.actors[-1]]]
        save_image = False

        self.has_any_active_actors = False
        with Off(record_attr_modifications=False, record_funcs=False, priority_level=math.inf), wave.open(audio_file_path, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(1)
            wav_file.setframerate(self.render_settings.audio_frames_per_second)
            for scene_num, (actors, (scene_start, scene_end)) in enumerate(zip(self.actors, self.scene_times[-len(self.actors):])):
                if scene_end < scene_start:
                    continue
                if scene_end == scene_start and not save_image:
                    scene_end += 1
                    save_image = True
                    file_path = f'{file_path}.png'
                    file_path_out = f'{file_path_out}.png'
                if not save_image:
                    file_path = f'{file_path}.mp4'
                    file_path_out = f'{file_path_out}.mp4'

                self.file_path = file_path
                self.file_writer = file_writer

                num_actor_batches = self.get_num_batches(scene_start, scene_end, batch_size_actors)
                all_actors = actors
                for i in range(num_actor_batches):
                    actor_s = i * batch_size_actors
                    actor_e = min((i+1) * batch_size_actors, scene_end)
                    if actor_s >= scene_end:
                        continue
                    for actor_id, actor in enumerate(list(sorted(actors, key=lambda x: x.anchor_priority, reverse=True))):
                        actor.set_state_full(actor_s, actor_e)

                    if self.camera.data.time_inds_materialized is None:
                        break
                    num_batches = self.get_num_batches(actor_s, actor_e, batch_size_frames)
                    s = time.time()
                    print(f'Rendering {(actor_e - actor_s) / self.frames_per_second} seconds of video.')
                    for i in range(num_batches):
                        start = actor_s + i * batch_size_frames
                        if start >= scene_end:
                            continue
                        end = min(actor_s + (i + 1) * batch_size_frames, actor_e)

                        def run():
                            self.get_fragments(actors, start, end, save_image)
                            audio = self.get_audio(actors, start, end)
                            wav_file.writeframes(bytes(((audio+1)*255/2).astype(np.uint8)))
                            torch.cuda.empty_cache()
                        run()
                        e = time.time()
                        print(f'{i}: {start}:{end}, took {e-s} seconds')
                        s = e
                    actors = [a for a in actors if a.despawn_ind >= actor_e]
                    for _ in all_actors:
                        _.reset_state()

        file_writer.release()
        if True:#len(self.effects) == 0:
            if os.path.exists(file_path_out):
                os.remove(file_path_out)
            os.rename(file_path, file_path_out)
            if not self.has_any_active_actors:
                 warnings.warn("You rendered an empty scene! Did you forget to spawn() your Mobs?", EmptySceneWarning)
            return save_image
        #TODO fix this so we can write audio to the fiie as well.
        videoclip = VideoFileClip(file_path)
        try:
            #audioclip = AudioFileClip("audioname.mp3")

            videoclip = videoclip.set_audio(CompositeAudioClip([effect.audio.subclip(0, videoclip.duration) for effect in sorted(self.effects, key=lambda e: e.spawn_time())]))

            if os.path.exists(file_path_out):
                os.remove(file_path_out)

            videoclip.write_videofile(file_path_out, codec='mpeg4')
        finally:
            videoclip.close()
            os.remove(file_path)

    def get_frames_from_fragments(self, fragments, window, frame, anti_alias_level=1):
        device = fragments[0].device if fragments is not None else frame.device
        bgf = self.background_frame
        if bgf.shape[-1] == 3:
            bgf = torch.cat((bgf, torch.zeros_like(bgf[...,:1])), -1)
        bgf = (bgf * 255).to(device, torch.uint8, non_blocking=True)
        window_height = (window[-1] - window[1])
        window_width = (window[-2] - window[0])
        window_size = window_width * window_height

        if fragments is None:
            frame[:] = bgf[...,:frame.shape[-1]]
            frame_out = unsquish(frame, 0, -window_height)
            yield frame_out
            return
            frame_out = F.avg_pool2d(frame_out.float().permute(2,0,1), anti_alias_level).permute(1,2,0).to(torch.uint8)
            frame_out = bloom_filter(frame_out)
            yield frame_out.cpu().flip((-3, -1)).numpy()
            return


        frames, inds, num_pixels_in_frame = fragments
        if inds is None:
            frame[:] = bgf
            frame = unsquish(frame, 0, -window_height).cpu().flip((-3, -1)).numpy()
            for i in range(len(frames)):
                yield frame
            return

        frame_ind_delimits = num_pixels_in_frame.cumsum(0)
        inds = inds % window_size
        inds = inds.unsqueeze(-1).expand(-1,frames.shape[-1])
        frames = (frames * 255).to(torch.uint8)

        for i in range(len(frame_ind_delimits)):
            frame[:] = bgf[...,:frame.shape[-1]]
            ind_begin = frame_ind_delimits[i-1] if i > 0 else 0
            ind_end = frame_ind_delimits[i]
            frame.scatter_(0, inds[ind_begin:ind_end], frames[ind_begin:ind_end])

            frame_out = unsquish(frame, 0, -window_height)
            yield frame_out

    def get_current_frame(self):
        return self.background_frame

    def get_new_id(self):
        self.id_count += 1
        return self.id_count-1

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self


class SceneTracker:
    _instance = None
    _memory = None

    def __init__(self):
        raise RuntimeError('Call instance() instead.')

    @classmethod
    def reset(cls):
        cls._instance = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            if cls._memory is None:
                #TODO make this work for CPU
                cls._memory = ManualMemory(((int((torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0))*DEFAULT_PORTION_MEMORY_USED_FOR_RENDERING))))
            cls._instance = Scene(memory=cls._memory)
        return cls._instance
