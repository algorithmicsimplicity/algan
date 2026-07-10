import collections
import math
import os
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import torch
import torch.nn.functional as F
import torchvision.utils
from moviepy import CompositeAudioClip

import algan
from algan.settings.defaults import *
from algan.settings.style_defaults import STYLE_DEFAULTS

from algan.constants.color import *
from algan.constants.spatial import *

from algan.animation.animation_contexts import Seq, Sync, AnimationManager, Off

from algan.rendering.post_processing.bloom import bloom_filter
from algan.rendering.primitives.primitive import OutOfRenderMemory
from algan.rendering.primitives.bezier_circuit_primitive import BezierCircuitPrimitive
from algan.utils.memory_utils import get_num_available_bytes, ManualMemory, empty_cache
from algan.utils.file_utils import get_image
from algan.rendering.taichi_runtime import sync_devices as _sync_devices
from algan.animation.timeline import TimelineManager


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

    @staticmethod
    def set_environment_map(source, intensity=1.0, ambient=True):
        """Set an equirectangular environment map for the scene.

        The map is used as a skybox (rays that leave the scene show the map,
        including in reflections and refractions) and -- when ``ambient`` is
        True -- as diffuse image-based lighting: every lit surface receives
        the map's irradiance (an order-1 spherical-harmonics approximation)
        in addition to the scene's explicit lights.

        Supported by the deterministic (single-sample) ray tracer.

        Parameters
        ----------
        source
            Path to an image file, or a ``[H, W, >=3]`` tensor/array holding
            an equirectangular (longitude x latitude, sky at the top row)
            RGB image. Values may be 0-255 or 0-1. Pass ``None`` to remove
            the environment map.
        intensity
            Brightness multiplier applied to the map.
        ambient
            Whether the map also lights surfaces (image-based lighting), or
            is only visible as a background/in reflections.
        """
        scene = algan.SceneManager.instance()
        if source is None:
            scene.environment_map = None
            return
        env = source
        if isinstance(env, str):
            import cv2

            img = cv2.imread(env, cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(
                    f"Could not read environment map image: {env}")
            env = torch.from_numpy(img[..., ::-1].copy())  # BGR -> RGB
        if not torch.is_tensor(env):
            env = torch.tensor(env)
        env = env.float()
        if env.dim() != 3 or env.shape[-1] < 3:
            raise ValueError(
                "Environment map must have shape [height, width, >=3], got "
                f"{tuple(env.shape)}")
        if env.max() > 1.5:
            env = env / 255.0
        scene.environment_map = env[..., :3].contiguous()
        scene.environment_intensity = float(intensity)
        scene.environment_ambient = bool(ambient)

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

    def add_actor(self, actor):
        if self.allow_new_actors:
            self.actors[-1].append(actor)
        return self

    def add_effect(self, effect):
        self.effects.append(effect)
        return self

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
                if actor.is_spawned():
                    actor.despawn(**kwargs)

    def clear_scene(self, **kwargs):
        with Seq(run_time=0.5):
            self.despawn_scene(**kwargs)
        self.actors[-1] = [
            _ for _ in self.actors[-1] if (_.is_spawned() and _.is_despawned())
        ]

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
        audio_clip.duration = AnimationManager.instance().context.timespan.original_end
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
        post_processes=(),
        transparent_background=False,
        background_color=None,
        render_state=None,
    ):
        with torch.no_grad():
            camera = self.camera
            if render_state is None:
                render_state = self._materialize_render_state(start_ind, end_ind)
            camera.ray_origin = render_state["ray_origin"]
            camera.screen_point = render_state["screen_point"]
            camera.screen_basis = render_state["screen_basis"]
            camera.screen_width = (
                self.num_pixels_screen_width * self.render_settings.anti_alias_level
            )
            camera.screen_height = (
                self.num_pixels_screen_height * self.render_settings.anti_alias_level
            )
            for l, (origin, light_color, aux) in zip(
                self.light_sources, render_state["lights"]
            ):
                l.origin = origin
                l.light_color = light_color
                l._render_aux = aux

            #torch.compiler.cudagraph_mark_step_begin()
            self.memory.scene = self
            original_pointers = self.memory.get_pointers()
            for primitive in primitive_batch:
                primitive.memory = self.memory
                primitive.project_to_screen(camera, self.light_sources)

            # Reclaim animation-phase residuals before render batching.
            empty_cache(force_gc=False)

            render_pointers = self.memory.get_pointers()
            current_ind = start_ind
            num_bytes_for_post_processing_per_frame = self.num_pixels_screen_width * self.num_pixels_screen_height * 5 * 4 * 4
            while True:
                mem_per_time_step = max(max([_.get_memory_used(0, 1) - _.get_memory_used_for_blending(0, 1)
                     for _ in primitive_batch]) + max([_.get_memory_used_for_blending(0, 1) for _ in primitive_batch]),
                                        num_bytes_for_post_processing_per_frame)
                duration = int(self.memory.get_num_bytes_remaining() // mem_per_time_step) * 1
                duration = min(duration, end_ind - current_ind)
                duration = max(duration, 1)
                new_ind = current_ind + duration

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
            # Camera/screen/light state is no longer reset here: batch prep
            # (get_batch_of_primitives) resets and re-materializes it at the
            # start of each batch, and may already be running on a worker
            # thread for the next batch while this render executes.

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

    def _is_batchable_surface(self, actor):
        """True if this actor's geometry build can be stacked with same-shaped
        peers into one tensor pass (see surface.get_render_primitives_batched).
        Requires the stock Surface build (no subclass override), the plain
        vertex-color path, and computed normals. Set ALGAN_BATCH_SURFACE_PREP=0
        to disable batching (A/B against the per-surface path)."""
        if os.environ.get("ALGAN_BATCH_SURFACE_PREP", "1") == "0":
            return False
        from algan.mobs.surfaces.surface import Surface

        if not isinstance(actor, Surface):
            return False
        if type(actor).get_render_primitives is not Surface.get_render_primitives:
            return False
        if actor.color_texture is not None or actor.ignore_normals:
            return False
        if (getattr(actor, "material_texture", None) is not None
                or getattr(actor, "normal_texture", None) is not None):
            return False
        if (
            actor is self.camera
            or actor is self.camera.screen
            or actor in self.light_sources
        ):
            return False
        return True

    def _build_deferred_surfaces(self, deferred):
        """Build geometry for all deferred surfaces, one stacked tensor pass
        per (grid shape, materialized location shape) group (see
        surface.get_render_primitives_batched)."""
        from algan.mobs.surfaces.surface import get_render_primitives_batched

        groups = collections.defaultdict(list)
        for entry in deferred:
            actor = entry["actor"]
            key = (
                actor.grid_width,
                actor.grid_height,
                tuple(actor.grid.location.shape),
            )
            groups[key].append(entry)

        for entries in groups.values():
            prims = get_render_primitives_batched([e["actor"] for e in entries])
            for entry, p in zip(entries, prims):
                if isinstance(p, list):
                    entry["prims"] = p
                else:
                    entry["prims"] = [p] if p is not None else []

    def get_batch_of_primitives(
        self, start_time_ind, max_end_time_ind, actors, max_mem_used
    ):
        max_end_time = max_end_time_ind / self.frames_per_second
        start_time = start_time_ind / self.frames_per_second
        primitive_actors = [
            actor
            for actor in actors
            if (actor.lifespan.start() <= max_end_time)
            and ((actor.lifespan.end() >= start_time) or actor.lifespan.end() < 0)
            and hasattr(actor, "get_render_primitives")
        ]

        # Precompute memory per timestep once to avoid redundant calls inside binary search loop
        actor_mem = {actor: actor.get_memory_used_per_timestep() for actor in primitive_actors}

        # Binary search to find a batch size that will fit in memory.
        def get_duration():
            #return 90
            duration = max_end_time_ind - start_time_ind
            duration = min(duration, COMPUTING_DEFAULTS.max_animate_batch_size)
            while True:
                selected_actors = [
                    actor
                    for actor in primitive_actors
                    if (
                        actor.lifespan.start()
                        <= (start_time_ind + duration) / self.frames_per_second
                    )
                ]
                mem_used = sum(
                    [
                        actor_mem[actor] * duration
                        for actor in selected_actors
                    ]
                )
                if mem_used <= max_mem_used:
                    break
                duration = duration // 2
                if duration <= 1:
                    duration = 1
                    break
            return duration

        duration = get_duration()
        actors = [
            actor
            for actor in actors
            if (
                actor.lifespan.start()
                <= (start_time_ind + duration) / self.frames_per_second
            )
            and ((actor.lifespan.end() >= start_time_ind / self.frames_per_second)
                 or (actor.lifespan.end() < 0))
        ]
        time_inds = torch.arange(start_time_ind, start_time_ind + duration)

        timeline = TimelineManager.instance()
        timeline.set_state_to_times(time_inds / self.frames_per_second)

        grouped_primitives = collections.defaultdict(lambda: [None, []])
        # Surfaces sharing a grid shape are not built one-by-one: their state
        # is materialized per-actor below (in anchor-priority order, exactly as
        # before), but the geometry build is deferred so all of them can run as
        # one stacked tensor pass (_build_deferred_surfaces). ordered_items
        # records primitives / deferred entries in actor order so the final
        # grouping (and thus the merged collection layout) is unchanged.
        ordered_items = []
        deferred_surfaces = []
        for actor in sorted(actors, key=lambda x: x.anchor_priority, reverse=True):
            if not hasattr(actor, "get_render_primitives"):
                continue
            if self._is_batchable_surface(actor):
                entry = {"actor": actor, "prims": None}
                deferred_surfaces.append(entry)
                ordered_items.append(entry)
                continue
            primitive = actor.get_render_primitives()
            if primitive is not None:
                if not isinstance(primitive, list):
                    primitive = [primitive]
                ordered_items.append(primitive)

        if deferred_surfaces:
            self._build_deferred_surfaces(deferred_surfaces)

        for item in ordered_items:
            primitives = item["prims"] if isinstance(item, dict) else item
            if not primitives:
                continue
            for p in primitives:
                grouped_primitives[p.get_batch_identifier()][0] = p.__class__
                grouped_primitives[p.get_batch_identifier()][1].append(p)

        primitive_collections = []
        max_bezier_batch_size = 50000
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
                textured = []
                colored = []
                for p in primitives:
                    if getattr(p, "uvs", None) is not None or getattr(p, "texture_map", None) is not None:
                        textured.append(p)
                    else:
                        colored.append(p)
                if colored:
                    primitive_collections.append(
                        primitive_class(triangle_collection=colored)
                    )
                    primitive_collections[-1].memory = self.memory
                    primitive_collections[-1].scene = self
                # Textured primitives are batched one per collection: a
                # collection carries a single texture map set (color/material/
                # normal), so merging two differently-textured primitives
                # would drop all but the first primitive's maps. Their
                # geometry is still merged into one kernel launch downstream
                # (see _merge_scene).
                for p in textured:
                    primitive_collections.append(
                        primitive_class(triangle_collection=[p])
                    )
                    primitive_collections[-1].memory = self.memory
                    primitive_collections[-1].scene = self
        render_state = self._materialize_render_state(
            start_time_ind, start_time_ind + duration
        )
        return primitive_collections, start_time_ind + duration, render_state

    def _materialize_render_state(self, start_ind, end_ind):
        """Materialize camera/screen/light state over ``[start_ind, end_ind)``
        and extract the plain tensors the renderer consumes (this used to be
        the first thing render_primitive_batch did). Returning a snapshot
        instead of writing camera attributes means the render thread never
        reads animated state -- by the time a batch renders, prep for the
        *next* batch may be mutating that state on a worker thread.
        """
        camera = self.camera
        device = COMPUTING_DEFAULTS.render_device
        lights = []
        for l in self.light_sources:
            loc = l.location
            col = l.color[..., :-1] * l.color[..., -1:] * l.opacity
            intensity = float(getattr(l, "intensity", 1.0))
            if intensity != 1.0:
                col = col * intensity
            is_ext = getattr(l, "is_extended", None)
            if is_ext is not None and is_ext():
                # Extended light (see algan.rendering.lights): snapshot its
                # emitter sample positions and packed aux parameter columns.
                # Area lights expand into K samples, each carrying 1/K of the
                # light's power.
                loc_f = loc.reshape(loc.shape[0], -1)[:, :3]   # [T, 3]
                col_f = col.reshape(col.shape[0], -1)          # [T, C]
                pos_rows = l.get_sample_positions(loc_f)       # [T, K, 3]
                k = pos_rows.shape[-2]
                col_rows = ((col_f / k if k > 1 else col_f)
                            .unsqueeze(-2).expand(-1, k, -1))
                aux = l.build_aux(loc_f)                       # [T, K, 13]
                lights.append((pos_rows.to(device), col_rows.to(device),
                               aux.to(device)))
            else:
                lights.append((
                    loc.unsqueeze(-2).to(device),
                    col.unsqueeze(-2).to(device),
                    None,
                ))
        return dict(
            ray_origin=camera.location.unsqueeze(-2).to(device),
            screen_point=camera.screen.location.unsqueeze(-2).to(device),
            screen_basis=camera.get_render_screen_basis().to(device),
            lights=lights,
        )

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

        if time_stamp is None:
            time_stamp = AnimationManager.instance().context.timespan.current_time + 1.5/self.render_settings.frames_per_second
        time_ind = round(time_stamp * self.render_settings.frames_per_second)
        frames = []
        for frame in self.get_frames(time_ind-1, time_ind):
            frame = frame.float() / 255
            frames.append(frame.squeeze(0).permute(-1,0,1))
        torchvision.utils.save_image(frames[-1], filename)
        return frames

    def save_frames(self, filename, time_stamps=None):
        if not hasattr(time_stamps, '__len__'):
            time_stamps = [time_stamps]
        return [self.save_frame(f'{".".join(filename.split(".")[:-1])}_{t}.{filename.split(".")[-1]}',
                                t) for t in time_stamps]

    def get_frames(self, start_time_ind, end_time_ind, background_color=None, post_processes=(bloom_filter,), manual_memory=True):
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

            # Prefetch pipeline: while batch b renders on this thread, batch
            # b+1 is prepped (CPU geometry generation + host-to-GPU upload) on
            # a single worker thread. Batch prep touches only animated mob
            # state (owned by prep; preps run strictly sequentially on the one
            # worker) while rendering consumes only the primitive tensors and
            # the render-state snapshot handed over by prep, so the two phases
            # share no mutable state. All Taichi work stays on this thread.
            # Set ALGAN_PREFETCH_BATCHES=0 to fall back to serial (also
            # reduces peak memory by one batch's tensors).
            prefetch_enabled = (
                os.environ.get("ALGAN_PREFETCH_BATCHES", "1") != "0"
            )
            # inference_mode is thread-local; mirror the caller's mode in the
            # worker so prep-created tensors can be mutated in-place later by
            # the render thread (inference tensors may only be modified while
            # inference mode is on).
            inference_mode_enabled = torch.is_inference_mode_enabled()

            def fetch_batch(time_ind):
                with torch.inference_mode(inference_mode_enabled):
                    return self.get_batch_of_primitives(
                        time_ind, end_time_ind, actors, max_animate_mem
                    )

            executor = (
                ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="algan-batch-prep"
                )
                if prefetch_enabled
                else None
            )
            pending = None
            try:
                while True:
                    _sync_devices()
                    s = time.time()
                    print(
                        f"Fetching batch {current_time_ind}:{end_time_ind}."
                    )
                    if pending is not None:
                        primitives, new_time_ind, render_state = pending.result()
                        pending = None
                    else:
                        primitives, new_time_ind, render_state = fetch_batch(
                            current_time_ind
                        )
                    _sync_devices()
                    e = time.time()
                    print(
                        f"Batch fetch took {e - s} seconds"
                    )
                    if new_time_ind <= current_time_ind:
                        raise OutOfRenderMemory(
                            "Insufficient memory to render this scene,"
                            "please reduce the number of Mobs used."
                        )
                    if executor is not None and new_time_ind < end_time_ind:
                        pending = executor.submit(fetch_batch, new_time_ind)
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
                            render_state=render_state,
                        )
                        del primitives
                        # Free previous batch data before allocating next batch.
                        empty_cache(force_gc=False)
                        _sync_devices()
                        e = time.time()
                        print(
                            f"{current_time_ind}:{new_time_ind}, took {e - s} seconds"
                        )

                    current_time_ind = new_time_ind
                    if new_time_ind >= end_time_ind:
                        break
                TimelineManager.instance().clear_buffers()
            finally:
                # Always drain the worker before leaving (normal completion,
                # error, or abandoned generator): a prep still running while
                # the caller resets or reuses the scene would race it.
                if pending is not None:
                    try:
                        pending.result()
                    except Exception:
                        pass
                if executor is not None:
                    executor.shutdown(wait=True)

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
        post_processes=(bloom_filter,),
        background_color=None,
    ):
        self.scene_times.append(
            [
                self.scene_times[-1][0],
                (
                    round(
                        AnimationManager.instance().context.timespan.original_end
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
        if (not hasattr(self, 'has_any_active_actors')) or (not self.has_any_active_actors):
            warnings.warn(
                "You rendered an empty scene! Did you forget to spawn() your Mobs?",
                EmptySceneWarning,
            )

    def get_new_id(self):
        self.id_count += 1
        return self.id_count - 1

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self
