import collections
import gc
import math
import multiprocessing
import os
import threading
import time
import wave
import warnings
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import torch
import torch.nn.functional as F
import torchvision.utils
from moviepy import CompositeAudioClip

import algan
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
from algan.utils.tensor_utils import unsquish
from algan.utils.file_utils import get_image
from algan import _sync_devices


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
            for l, (origin, light_color) in zip(
                self.light_sources, render_state["lights"]
            ):
                l.origin = origin
                l.light_color = light_color

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

                empty_cache

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

    def _time_compress_mode(self):
        try:
            from algan.rendering.raytracing import primitives as rtp
            return getattr(rtp, "TIME_COMPRESS", 0)
        except Exception:
            return 0

    def _try_timeline_direct(self, actor, time_inds):
        """Mode-3 timeline-direct geometry. If the actor moves along a single
        eased straight line over this batch (probed cheaply from its dense
        ``location``), build its render geometry at only the two segment
        endpoints and linearly expand to every frame with the per-frame eased
        fraction ``z`` -- geometry is affine in location, so this is exact --
        instead of running the (expensive) per-frame geometry build over all
        frames. Returns a primitive list, or ``None`` to tell the caller to use
        the dense path (not linear, or a nonlinear attribute -- e.g. normals
        under non-uniform scale -- failed the midpoint check)."""
        if self._time_compress_mode() != 3:
            return None
        T = len(time_inds)
        if T <= 2:
            return None
        from algan.rendering.raytracing.time_compression import (
            extract_global_linear_z)

        actor.set_state_to_time_t(time_inds)
        for component in actor.components:
            component.set_state_to_time_t(time_inds)
        loc = getattr(actor, "location", None)
        if not torch.is_tensor(loc):
            return None
        loc3 = loc
        while loc3.dim() < 3:
            loc3 = loc3.unsqueeze(0)
        if loc3.shape[0] != T:
            return None
        z = extract_global_linear_z(loc3)
        if z is None:
            return None

        mid = T // 2
        knot_inds = time_inds[torch.tensor([0, mid, T - 1])]
        actor.set_state_to_time_t(knot_inds)
        for component in actor.components:
            component.set_state_to_time_t(knot_inds)
        prims = actor.get_render_primitives()
        if prims is None:
            return None
        if not isinstance(prims, list):
            prims = [prims]
        z_mid = float(z[mid])
        for p in prims:
            if not self._expand_primitive_linear(p, z, z_mid):
                return None
        return prims

    def _expand_primitive_linear(self, p, z, z_mid, tol=1e-3):
        """Expand a primitive whose per-frame geometry was built at exactly the
        three probe frames [start, mid, end] to all ``len(z)`` frames, by
        linearly blending the endpoint values with ``z``. Each varying tensor
        attribute (3 frames on its time axis) is verified at the midpoint against
        the linear blend; a mismatch (a nonlinear attribute) aborts to the dense
        path. Static attributes (1 frame) are left untouched (the kernel already
        broadcasts them)."""
        from algan.rendering.raytracing.time_compression import expand_linear

        def expand_attr(t):
            n0 = t.shape[0]
            if n0 == 1:
                return t, True            # static: leave as-is
            if n0 != 3:
                return t, False           # unexpected layout -> dense fallback
            lo, md, hi = t[0], t[1], t[2]
            recon = lo + z_mid * (hi - lo)
            if float((md - recon).abs().amax()) > tol:
                return t, False           # nonlinear -> dense fallback
            return expand_linear(lo, hi, z.to(t.device)), True

        for name in ("corners", "normals", "colors", "uvs"):
            t = getattr(p, name, None)
            if not torch.is_tensor(t) or t.dim() < 1:
                continue
            new_t, ok = expand_attr(t)
            if not ok:
                return False
            setattr(p, name, new_t)

        sp = getattr(p, "shader_param_values", None)
        if isinstance(sp, (list, tuple)):
            new_sp = []
            for t in sp:
                if torch.is_tensor(t) and t.dim() >= 1:
                    new_t, ok = expand_attr(t)
                    if not ok:
                        return False
                    new_sp.append(new_t)
                else:
                    new_sp.append(t)
            p.shader_param_values = list(new_sp)
        return True

    def _ensure_actor_state(self, actor, start_time_ind, end_time_ind):
        """Materialize the actor's animated state for this batch. In
        full-range mode (see get_frames) the state is materialized once over
        the whole render window and reused by every later batch -- attribute
        reads slice it through set_state_to_time_t -- which is valid because
        animation history cannot change during rendering. Per-frame values are
        identical either way (materialization is per-frame math), so output is
        unchanged. Camera/screen/lights keep the per-batch protocol: their
        state is reset every batch as part of the render-state snapshot
        handover."""
        full = getattr(self, "_full_state_range", None)
        if full is None or (
            actor is self.camera
            or actor is self.camera.screen
            or actor in self.light_sources
        ):
            actor.set_state_full(start_time_ind, end_time_ind)
            return
        d = actor.data
        tm = d.time_inds_materialized
        if (
            getattr(actor, "_full_state_applied", False)
            and d.set_pre_function_application
            and tm is not None
            and int(tm[0]) <= start_time_ind
            and int(tm[-1]) >= end_time_ind - 1
        ):
            # Reused from an earlier batch. Mark as set so a duplicate
            # occurrence later in this batch's actor loop is skipped, exactly
            # as set_state_full would have caused.
            actor.already_set_state = True
            return
        actor.set_state_full(*full)
        actor._full_state_applied = True

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
        if (
            actor is self.camera
            or actor is self.camera.screen
            or actor in self.light_sources
        ):
            return False
        return True

    def _prepare_deferred_surface(self, actor, time_inds):
        """Materialize the actor's state for the deferred geometry build --
        at the three mode-3 knot frames when its motion is linear over this
        batch (mirroring _try_timeline_direct), else densely at every frame --
        and return the deferred entry consumed by _build_deferred_surfaces."""
        entry = {"actor": actor, "z": None, "z_mid": None, "prims": None}
        T = len(time_inds)
        if self._time_compress_mode() == 3 and T > 2:
            from algan.rendering.raytracing.time_compression import (
                extract_global_linear_z)

            actor.set_state_to_time_t(time_inds)
            for component in actor.components:
                component.set_state_to_time_t(time_inds)
            loc = getattr(actor, "location", None)
            if torch.is_tensor(loc):
                loc3 = loc
                while loc3.dim() < 3:
                    loc3 = loc3.unsqueeze(0)
                if loc3.shape[0] == T:
                    z = extract_global_linear_z(loc3)
                    if z is not None:
                        mid = T // 2
                        knot_inds = time_inds[torch.tensor([0, mid, T - 1])]
                        actor.set_state_to_time_t(knot_inds)
                        for component in actor.components:
                            component.set_state_to_time_t(knot_inds)
                        entry["z"] = z
                        entry["z_mid"] = float(z[mid])
                        return entry
        actor.set_state_to_time_t(time_inds)
        for component in actor.components:
            component.set_state_to_time_t(time_inds)
        return entry

    def _build_deferred_surfaces(self, deferred, time_inds):
        """Build geometry for all deferred surfaces, one stacked tensor pass
        per (grid shape, materialized location shape) group, then apply the
        mode-3 linear expansion per surface (dense per-surface rebuild on a
        failed midpoint check) and release the actors' animated state."""
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
                if entry["z"] is not None:
                    if not self._expand_primitive_linear(p, entry["z"], entry["z_mid"]):
                        # A nonlinear attribute failed the midpoint check:
                        # rebuild this surface densely, as the per-surface
                        # path would have.
                        actor = entry["actor"]
                        actor.set_state_to_time_t(time_inds)
                        for component in actor.components:
                            component.set_state_to_time_t(time_inds)
                        p = actor.get_render_primitives()
                if isinstance(p, list):
                    entry["prims"] = p
                else:
                    entry["prims"] = [p] if p is not None else []

        for entry in deferred:
            actor = entry["actor"]
            if getattr(self, "_full_state_range", None) is not None:
                actor.already_set_state = False
                for component in actor.components:
                    component.already_set_state = False
            else:
                actor.reset_state()
                for component in actor.components:
                    component.reset_state()

    def get_batch_of_primitives(
        self, start_time_ind, max_end_time_ind, actors, max_mem_used
    ):
        # Camera, screen and light animation state is owned by batch prep: it
        # is reset here (this used to happen at the end of
        # render_primitive_batch), re-materialized by the actor loop below for
        # this batch's range, and the plain tensors the renderer consumes are
        # snapshot just before returning (_materialize_render_state). The
        # render path then never touches animated state, which lets the next
        # batch be prepped on a worker thread while the current one renders.
        for _ in (self.camera, self.camera.screen, *self.light_sources):
            _.reset_state()
        max_end_time = max_end_time_ind / self.frames_per_second
        start_time = start_time_ind / self.frames_per_second
        primitive_actors = [
            _
            for _ in actors
            if (_.data.spawn_time() <= max_end_time)
            and ((_.data.despawn_time() >= start_time) or _.data.despawn_time() < 0)
            and hasattr(_, "get_render_primitives")
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
                    _
                    for _ in primitive_actors
                    if (
                        _.data.spawn_time()
                        <= (start_time_ind + duration) / self.frames_per_second
                    )
                ]
                mem_used = sum(
                    [
                        actor_mem[_] * duration
                        for _ in selected_actors
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
        # Surfaces sharing a grid shape are not built one-by-one: their state
        # is materialized per-actor below (in anchor-priority order, exactly as
        # before), but the geometry build is deferred so all of them can run as
        # one stacked tensor pass (_build_deferred_surfaces). ordered_items
        # records primitives / deferred entries in actor order so the final
        # grouping (and thus the merged collection layout) is unchanged.
        ordered_items = []
        deferred_surfaces = []
        for actor in sorted(actors, key=lambda x: x.anchor_priority, reverse=True):
            if hasattr(actor, "already_set_state") and actor.already_set_state:
                continue
            if (not actor.is_primitive) and not actor.data.history.function_applications:
                actor.reset_state()
                continue
            self._ensure_actor_state(actor, start_time_ind, start_time_ind + duration)
            if hasattr(actor, "get_render_primitives"):
                for component in actor.components:
                    self._ensure_actor_state(component, start_time_ind, start_time_ind + duration)
                if self._is_batchable_surface(actor):
                    entry = self._prepare_deferred_surface(actor, time_inds)
                    deferred_surfaces.append(entry)
                    ordered_items.append(entry)
                    # State must stay materialized for the batched build; the
                    # reset happens in _build_deferred_surfaces.
                    continue
                # Mode-3: build geometry at just the segment endpoints and expand
                # (skips the dense per-frame build); None falls back to dense.
                primitive = self._try_timeline_direct(actor, time_inds)
                if primitive is None:
                    actor.set_state_to_time_t(time_inds)
                    for component in actor.components:
                        component.set_state_to_time_t(time_inds)
                    primitive = actor.get_render_primitives()
                if primitive is not None:
                    if not isinstance(primitive, list):
                        primitive = [primitive]
                    ordered_items.append(primitive)
            if not (
                actor == self.camera
                or actor == self.camera.screen
                or actor in self.light_sources
            ):
                if getattr(self, "_full_state_range", None) is not None:
                    # Keep the materialized state for later batches; only the
                    # per-batch dedup flag set_state_full raised is cleared.
                    actor.already_set_state = False
                    for component in actor.components:
                        component.already_set_state = False
                else:
                    actor.reset_state()
                    for component in actor.components:
                        component.reset_state()

        if deferred_surfaces:
            self._build_deferred_surfaces(deferred_surfaces, time_inds)

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
                if textured:
                    primitive_collections.append(
                        primitive_class(triangle_collection=textured)
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

        The camera uses the base ``Animatable.set_state_to_time_t``: the
        Camera override additionally writes ray_origin/screen_point/
        screen_basis onto the camera object, which would race with the render
        thread reading those attributes for the batch currently on screen.
        """
        from algan.animation.animatable import Animatable

        time_inds = torch.arange(start_ind, end_ind)
        camera = self.camera
        camera.screen.reset_state()
        camera.reset_state()
        camera.set_state_full(time_inds[0], time_inds[-1] + 1)
        Animatable.set_state_to_time_t(camera, time_inds)
        camera.screen.set_state_full(time_inds[0], time_inds[-1] + 1)
        camera.screen.set_state_to_time_t(time_inds)
        device = COMPUTING_DEFAULTS.render_device
        lights = []
        for l in self.light_sources:
            l.reset_state()
            l.set_state_full(time_inds[0], time_inds[-1] + 1)
            l.set_state_to_time_t(time_inds)
            lights.append((
                l.location.unsqueeze(-2).to(device),
                (l.color[..., :-1] * l.color[..., -1:] * l.opacity)
                .unsqueeze(-2)
                .to(device),
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

        # Full-range state reuse: animation history is frozen while rendering
        # (the Off context below records nothing), so each actor's animated
        # state can be materialized ONCE over the whole render window and
        # sliced per batch by set_state_to_time_t, instead of re-walking its
        # attribute/function history for every batch (set_state_full was ~25%
        # of batch prep). OPT-IN (ALGAN_FULL_RANGE_STATE=1): output is not
        # byte-identical to the per-batch protocol -- rate funcs evaluate
        # transcendentals (sigmoid) whose CPU vector lanes round 1 ULP
        # differently by tensor size, so the eased fraction z at a few frames
        # shifts by 1 ULP, which can flip the odd silhouette-edge pixel
        # (measured 0.00006% of values on neural_net). Also guarded by an
        # estimate of the total CPU-side state (per-frame attr bytes x
        # frames); ALGAN_FULL_RANGE_STATE_MB overrides the cap.
        self._full_state_range = None
        if os.environ.get("ALGAN_FULL_RANGE_STATE", "0") == "1":
            total_frames = int(end_time_ind) - int(start_time_ind)
            state_bytes_per_frame = 0
            for a in self.actors[-1]:
                for m in (a, *getattr(a, "components", [])):
                    data = getattr(m, "data", None)
                    if data is None:
                        continue
                    for v in data.data_dict_active.values():
                        if torch.is_tensor(v):
                            state_bytes_per_frame += (
                                v.numel() // max(v.shape[0], 1)
                            ) * v.element_size()
            cap = float(
                os.environ.get("ALGAN_FULL_RANGE_STATE_MB", "2048")
            ) * 2**20
            if state_bytes_per_frame * total_frames <= cap:
                self._full_state_range = (int(start_time_ind), int(end_time_ind))

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
                # Full-range mode keeps actor state materialized across
                # batches; release it now so the scene is left exactly as the
                # per-batch protocol leaves it (all actors reset).
                if getattr(self, "_full_state_range", None) is not None:
                    self._full_state_range = None
                    for a in self.actors[-1]:
                        a.reset_state()
                        for component in getattr(a, "components", []):
                            component.reset_state()

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
        if (not hasattr(self, 'has_any_active_actors')) or (not self.has_any_active_actors):
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
