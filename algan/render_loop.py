"""The frame-batching render loop, split out of :mod:`algan.scene`.

:class:`RenderLoopMixin` is mixed into :class:`~algan.scene.Scene` and is not
useful standalone (``self`` is always the Scene). It owns everything between
"the timeline has recorded animations" and "frames are streamed to the video
writer": batch sizing by memory budget, timeline state materialization and
primitive batching (:meth:`~RenderLoopMixin.get_batch_of_primitives`),
the prefetch pipeline (:meth:`~RenderLoopMixin.get_frames`), per-batch
rendering (:meth:`~RenderLoopMixin.render_primitive_batch`), and video file
output (:meth:`~RenderLoopMixin.render_to_video`).
"""

import collections
import math
import os
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import torch

from algan.animation.animation_contexts import AnimationManager, Off
from algan.animation.timeline import TimelineManager
from algan.logging.logger import get_logger
from algan.rendering.post_processing.bloom import bloom_filter
from algan.rendering.primitives.bezier_circuit_primitive import BezierCircuitPrimitive
from algan.rendering.primitives.primitive import OutOfRenderMemory
from algan.rendering.taichi_runtime import sync_devices as _sync_devices
from algan.settings.defaults import COMPUTING_DEFAULTS
from algan.utils.memory_utils import (
    ManualMemory,
    empty_cache,
    get_num_available_bytes,
)

logger = get_logger("scene")

#: Sentinel "class" marking an entry of grouped_primitives that already holds
#: finished collections (merged bezier groups) rather than per-actor
#: primitives awaiting concatenation.
_PREBUILT_COLLECTION = object()


class EmptySceneWarning(Warning):
    pass


def write_frames_from_queue(queue, file_writer):
    while True:
        frame = queue.get()
        if frame is None:  # Sentinel value to signal the end
            break
        file_writer.write_frame(frame.numpy())


def _max_render_duration(bytes_remaining, requested_frames, bytes_per_frame,
                         fixed_bytes_for_frames):
    """Largest frame count fitting ``fixed(n) + n * per_frame`` bytes.

    ``fixed_bytes_for_frames`` models bounded wavefront tile state. It grows
    only until a tile is full, unlike the output/post-process buffers that grow
    for every frame. Returning one on an undersized arena preserves the
    renderer's existing single-frame OOM diagnostic/retry path.
    """
    requested_frames = max(1, int(requested_frames))
    bytes_per_frame = max(1, int(bytes_per_frame))
    lo, hi, best = 1, requested_frames, 1
    while lo <= hi:
        mid = (lo + hi) // 2
        needed = bytes_per_frame * mid + int(fixed_bytes_for_frames(mid))
        if needed <= bytes_remaining:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


class RenderLoopMixin:
    """Frame batching, batch preparation, and the render/video-output loop
    (mixed into :class:`~algan.scene.Scene`)."""

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

            self.memory.scene = self
            original_pointers = self.memory.get_pointers()
            for primitive in primitive_batch:
                primitive.memory = self.memory
                if getattr(primitive, "_rt_projected", False):
                    # Already projected on the batch-prep worker against this
                    # batch's snapshot (see _prewarm_render_batch); its source
                    # geometry has been released, so re-projection is both
                    # redundant and impossible.
                    primitive._rt_projected = False
                    continue
                primitive.project_to_screen(camera, self.light_sources)

            # Reclaim animation-phase residuals before render batching.
            empty_cache(force_gc=False)

            render_pointers = self.memory.get_pointers()
            current_ind = start_ind
            num_bytes_for_post_processing_per_frame = self.num_pixels_screen_width * self.num_pixels_screen_height * 5 * 4 * 4 * 4
            while True:
                mem_per_time_step = max(
                    max([_.get_memory_used(0, 1)
                         - _.get_memory_used_for_blending(0, 1)
                         for _ in primitive_batch])
                    + max([_.get_memory_used_for_blending(0, 1)
                           for _ in primitive_batch]),
                    num_bytes_for_post_processing_per_frame)

                def fixed_bytes(num_frames):
                    return max(_.get_fixed_memory_used(num_frames)
                               for _ in primitive_batch)

                duration = _max_render_duration(
                    self.memory.get_num_bytes_remaining(),
                    end_ind - current_ind,
                    mem_per_time_step,
                    fixed_bytes)
                new_ind = current_ind + duration

                logger.debug(f'rendering batch with duration {duration}')

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

    def _is_batchable_bezier(self, actor):
        """True if this bezier circuit's primitive build can be merged with
        same-shaped peers into one vectorized pass (see
        bezier_circuit.build_render_primitives_batched). Requires the stock
        BezierCircuitCubic build methods, a non-empty circuit, un-batched
        control points, and singleton rows for the per-circuit attributes.
        Set ALGAN_BATCH_BEZIER_PREP=0 to disable (A/B against the per-actor
        path)."""
        if os.environ.get("ALGAN_BATCH_BEZIER_PREP", "1") == "0":
            return False
        from algan.mobs.bezier_circuit import BezierCircuitCubic

        if not isinstance(actor, BezierCircuitCubic):
            return False
        t = type(actor)
        if t.get_render_primitives is not BezierCircuitCubic.get_render_primitives:
            return False
        if t._get_render_primitives is not BezierCircuitCubic._get_render_primitives:
            return False
        if actor.empty:
            return False
        if actor.control_points.parent_batch_sizes is not None:
            return False
        timeline = TimelineManager.instance()
        try:
            for attr in ("opacity", "basis", "glow", "border_width",
                         "border_color", "glow_radius", "location"):
                if timeline.attr_to_timeline[attr].mob_id_to_inds[
                        actor.id].numel() != 1:
                    return False
            loc_inds = timeline.attr_to_timeline["location"].mob_id_to_inds
            if loc_inds[actor.control_points.id].numel() % 4 != 0:
                return False
            timeline.attr_to_timeline["color"].mob_id_to_inds[
                actor.texture_points.id]
        except (KeyError, AttributeError):
            return False
        return True

    def _bezier_group_key(self, actor):
        from algan.rendering.primitives.bezier_circuit_primitive import (
            BezierCircuitPrimitive,
        )

        timeline = TimelineManager.instance()
        tex_rows = timeline.attr_to_timeline["color"].mob_id_to_inds[
            actor.texture_points.id].numel()
        return (
            BezierCircuitPrimitive.batch_identifier_for(
                actor.num_texture_points, actor.filled),
            tex_rows,
            actor.render_primitive,
        )

    def _build_deferred_beziers(self, deferred):
        """Build one merged bezier primitive per group of deferred circuits
        in a single vectorized pass (see
        bezier_circuit.build_render_primitives_batched). The merged primitive
        is attached to the group's first entry (matching the position the
        group's collection had in the per-actor path); later entries stay
        empty."""
        from algan.mobs.bezier_circuit import build_render_primitives_batched

        groups = {}
        for entry in deferred:
            groups.setdefault(self._bezier_group_key(entry["actor"]),
                              []).append(entry)
        for entries in groups.values():
            mega = build_render_primitives_batched(
                [e["actor"] for e in entries], self)
            entries[0]["prebuilt"] = [mega]

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
            if (actor.lifespan.start() >= 0)
            and (actor.lifespan.start() <= max_end_time)
            and ((actor.lifespan.end() >= start_time) or actor.lifespan.end() < 0)
            and hasattr(actor, "get_render_primitives")
        ]

        # Precompute memory per timestep once to avoid redundant calls inside binary search loop
        actor_mem = {actor: actor.get_memory_used_per_timestep() for actor in primitive_actors}

        # Binary search to find a batch size that will fit in memory.
        def get_duration():
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
            if (actor.lifespan.start() >= 0)
            and (
                actor.lifespan.start()
                <= (start_time_ind + duration) / self.frames_per_second
            )
            and ((actor.lifespan.end() >= start_time_ind / self.frames_per_second)
                 or (actor.lifespan.end() < 0))
        ]
        time_inds = torch.arange(start_time_ind, start_time_ind + duration)

        timeline = TimelineManager.instance()
        # Restrict base-state queries to actors that can contribute to this
        # frame window. Animation replay retains global row ids, and the
        # timeline conservatively falls back to all rows for user callbacks or
        # updaters whose dependencies cannot be discovered safely.
        timeline.set_state_to_times(
            time_inds / self.frames_per_second, active_mobs=actors)

        grouped_primitives = collections.defaultdict(lambda: [None, []])
        # Surfaces sharing a grid shape are not built one-by-one: their state
        # is materialized per-actor below (in anchor-priority order, exactly as
        # before), but the geometry build is deferred so all of them can run as
        # one stacked tensor pass (_build_deferred_surfaces). ordered_items
        # records primitives / deferred entries in actor order so the final
        # grouping (and thus the merged collection layout) is unchanged.
        ordered_items = []
        deferred_surfaces = []
        deferred_beziers = []
        for actor in sorted(actors, key=lambda x: x.anchor_priority, reverse=True):
            if not hasattr(actor, "get_render_primitives"):
                continue
            if self._is_batchable_surface(actor):
                entry = {"actor": actor, "prims": None}
                deferred_surfaces.append(entry)
                ordered_items.append(entry)
                continue
            if self._is_batchable_bezier(actor):
                entry = {"actor": actor, "prims": None, "prebuilt": None}
                deferred_beziers.append(entry)
                ordered_items.append(entry)
                continue
            primitive = actor.get_render_primitives()
            if primitive is not None:
                if not isinstance(primitive, list):
                    primitive = [primitive]
                ordered_items.append(primitive)

        if deferred_surfaces:
            self._build_deferred_surfaces(deferred_surfaces)

        if deferred_beziers:
            # A non-batchable primitive sharing a group's batch identifier
            # would have been concatenated into the same collection,
            # interleaved by actor order; fall back to the per-actor build
            # for such (rare) groups so the collection layout is unchanged.
            raw_identifiers = set()
            for item in ordered_items:
                if isinstance(item, dict):
                    continue
                for p in item:
                    raw_identifiers.add(p.get_batch_identifier())
            clean = []
            for entry in deferred_beziers:
                if self._bezier_group_key(entry["actor"])[0] in raw_identifiers:
                    primitive = entry["actor"].get_render_primitives()
                    if primitive is not None:
                        entry["prims"] = (primitive if isinstance(primitive, list)
                                          else [primitive])
                else:
                    clean.append(entry)
            if clean:
                self._build_deferred_beziers(clean)

        for item in ordered_items:
            if isinstance(item, dict) and item.get("prebuilt"):
                # Pre-merged bezier collection: registered under its batch
                # identifier at the position of the group's first actor, so
                # the final collection order matches the per-actor path.
                for collection in item["prebuilt"]:
                    key = collection.get_batch_identifier()
                    grouped_primitives[key][0] = _PREBUILT_COLLECTION
                    grouped_primitives[key][1].append(collection)
                continue
            primitives = item["prims"] if isinstance(item, dict) else item
            if not primitives:
                continue
            for p in primitives:
                grouped_primitives[p.get_batch_identifier()][0] = p.__class__
                grouped_primitives[p.get_batch_identifier()][1].append(p)

        primitive_collections = []
        max_bezier_batch_size = 50000
        for _, (primitive_class, primitives) in grouped_primitives.items():
            if primitive_class is _PREBUILT_COLLECTION:
                for collection in primitives:
                    collection.memory = self.memory
                    collection.scene = self
                    primitive_collections.append(collection)
                continue
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

    def _prewarm_render_batch(self, primitives, render_state):
        """Run a batch's ``project_to_screen`` + merged-scene/STBVH build
        ahead of the render (called from ``fetch_batch``, i.e. on the
        batch-prep worker when prefetch is on).

        Both steps are torch-only, but they normally run on the render thread
        against shared mutable state, so this uses isolated stand-ins:

        * a shim camera / shim lights carrying the batch's *snapshot* tensors
          (the exact objects ``render_primitive_batch`` would assign to the
          live camera/lights) -- the live objects still hold the in-flight
          batch's values and must not be touched from this thread;
        * a scratch unmanaged memory for shading temporaries -- the shared
          render pool's bump pointer is owned by the render thread (a
          concurrent ``temp()`` save/restore would free live render tensors).

        Identical inputs, identical math, so the packed arrays and merged
        scene are byte-identical to main-thread projection. Each successfully
        projected primitive is marked ``_rt_projected`` and skipped by
        ``render_primitive_batch``; on any failure the un-projected remainder
        (and the merge) simply run on the render thread as before.
        """
        try:
            from algan.rendering.raytracing.primitives import (
                RayTracedBezierCircuitPrimitive, RayTracedTrianglePrimitive)
            from algan.rendering.raytracing.scene_builder import (
                prewarm_merge_cache)
        except Exception:
            return
        if not primitives or not isinstance(
                primitives[0], (RayTracedTrianglePrimitive,
                                RayTracedBezierCircuitPrimitive)):
            return
        aa = self.render_settings.anti_alias_level

        class _ShimCamera:
            pass

        camera = _ShimCamera()
        camera.ray_origin = render_state["ray_origin"]
        camera.screen_point = render_state["screen_point"]
        camera.screen_basis = render_state["screen_basis"]
        camera.screen_width = self.num_pixels_screen_width * aa
        camera.screen_height = self.num_pixels_screen_height * aa

        class _ShimLight:
            pass

        lights = []
        for origin, light_color, aux in render_state["lights"]:
            light = _ShimLight()
            light.origin = origin
            light.light_color = light_color
            light._render_aux = aux
            lights.append(light)

        scratch = ManualMemory(0, managed=False)
        for primitive in primitives:
            original_memory = primitive.memory
            try:
                primitive.memory = scratch
                primitive.project_to_screen(camera, lights)
                primitive._rt_projected = True
            finally:
                primitive.memory = original_memory
        prewarm_merge_cache(primitives)

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

    def get_frames(self, start_time_ind, end_time_ind, background_color=None, post_processes=(bloom_filter,), manual_memory=True):
        if end_time_ind <= start_time_ind:
            yield []
            return

        self.original_background_frame = self.background_frame
        if background_color is not None:
            self.background_frame = background_color

        transparent_background = self.background_is_transparent()

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
                    batch = self.get_batch_of_primitives(
                        time_ind, end_time_ind, actors, max_animate_mem
                    )
                    # Pre-run the ray tracer's vertex shade + packing
                    # (project_to_screen) and merged-scene / STBVH build here
                    # (all torch-only) so they ride the prefetch: batch b+1's
                    # prep runs on the worker while batch b renders, turning
                    # seconds of otherwise-serial render-thread CPU work into
                    # hidden time. ALGAN_PREFETCH_MERGE=0 falls back to
                    # projecting + merging on the render thread.
                    if (batch[0]
                            and os.environ.get("ALGAN_PREFETCH_MERGE", "1")
                            != "0"):
                        try:
                            self._prewarm_render_batch(batch[0], batch[2])
                        except Exception as e:
                            logger.warning(
                                f"render-batch prewarm failed (deferring to "
                                f"the render thread): {e}")
                    return batch

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
                    logger.info(
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
                    logger.info(
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
                        logger.info(
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
                        logger.info(
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

        if os.path.exists(file_path_out):
            os.remove(file_path_out)
        os.rename(file_path, file_path_out)
        if (not hasattr(self, 'has_any_active_actors')) or (not self.has_any_active_actors):
            warnings.warn(
                "You rendered an empty scene! Did you forget to spawn() your Mobs?",
                EmptySceneWarning,
            )
