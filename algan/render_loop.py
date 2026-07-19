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
import logging
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
    InsufficientMemoryException,
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


def _max_duration_that_fits(requested_frames, fits):
    """Largest positive duration for which the monotone ``fits`` predicate is
    true.

    Returning one when even a single frame does not fit preserves the existing
    downstream single-frame OOM diagnostic.  Unlike the old repeated-halving
    loop, this does not discard as much as half of an otherwise usable
    animation-device budget.
    """
    requested_frames = max(1, int(requested_frames))
    lo, hi, best = 1, requested_frames, 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if fits(mid):
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def _primitive_source_device(primitive, fallback=None):
    """Device holding a not-yet-projected primitive's source geometry."""
    for name in (
        "corners", "colors", "normals", "mob_center", "next_segment_inds",
    ):
        value = getattr(primitive, name, None)
        if torch.is_tensor(value):
            return value.device
    if fallback is not None:
        return torch.device(fallback)
    return COMPUTING_DEFAULTS.animation_device


def _slice_render_state(render_state, start, end, total_frames):
    """Return a frame-window view of an immutable render-state snapshot."""
    start = int(start)
    end = int(end)
    total_frames = int(total_frames)

    def sliced(value):
        if (torch.is_tensor(value) and value.ndim > 0
                and int(value.shape[0]) == total_frames):
            return value[start:end]
        return value

    return dict(
        ray_origin=sliced(render_state["ray_origin"]),
        screen_point=sliced(render_state["screen_point"]),
        screen_basis=sliced(render_state["screen_basis"]),
        lights=[
            (sliced(origin), sliced(color), sliced(aux))
            for origin, color, aux in render_state["lights"]
        ],
    )


def _arena_allocation_end(pointer, numel, dtype):
    """Pointer after one :class:`ManualMemory` forward allocation."""
    alignment = int(dtype.itemsize)
    pointer = int(pointer) + (-int(pointer)) % alignment
    return pointer + int(numel) * alignment


def _raytrace_persistent_input_end(
    initial_pointer,
    num_frames,
    light_sources,
    merged_scene,
    environment_map,
    environment_ambient,
):
    """Arena pointer after the tracer's camera and light input copies.

    This mirrors the allocations before ``render_chunk`` in
    ``render_batch_raytraced``.  Camera and packed-light inputs cover the whole
    prepared primitive batch, so they are paid once and do not shrink with an
    individual render chunk.
    """
    from algan.rendering.raytracing import settings as rt_settings
    from algan.rendering.raytracing.settings import _scene_has_user_pipeline

    pointer = int(initial_pointer)
    num_frames = int(num_frames)

    # cam_origin, screen_point, pixel_basis_x and pixel_basis_y are [T, 3];
    # pixel_world_scale is [T].  _flat_frames casts every input to float32.
    pointer = _arena_allocation_end(
        pointer, num_frames * (3 + 3 + 3 + 3 + 1), torch.float32)

    samples = max(1, int(rt_settings.SAMPLES_PER_PIXEL))
    if samples > 1:
        return pointer

    lights_extended = any(
        getattr(light, "_render_aux", None) is not None
        for light in (light_sources or ())
    )
    det_frag = (
        bool(rt_settings.FRAGMENT_SHADING)
        or bool(rt_settings.SHADOWS)
        or _scene_has_user_pipeline(merged_scene)
        or lights_extended
        or environment_map is not None
    )
    if not det_frag:
        # Two [1, 1, 3] float32 placeholders.
        return _arena_allocation_end(pointer, 6, torch.float32)

    if lights_extended:
        num_light_rows = sum(
            int(light.origin.shape[-2]) for light in (light_sources or ())
        )
        color_width = 16
    else:
        num_light_rows = len(light_sources or ())
        color_width = 3

    append_environment = (
        environment_map is not None and bool(environment_ambient)
    )
    if append_environment:
        # The SH environment row widens compact point-light colors to 16.
        color_width = 16
        if num_light_rows:
            num_light_rows += 1
            light_frames = num_frames
        else:
            # _pack_lights returns one placeholder frame when there are no
            # explicit lights; _append_env_sh_light preserves that shape.
            num_light_rows = 1
            light_frames = 1
    elif num_light_rows:
        light_frames = num_frames
    else:
        # _pack_lights' empty-scene placeholders are copied even when fragment
        # shading is enabled.
        return _arena_allocation_end(pointer, 6, torch.float32)

    pointer = _arena_allocation_end(
        pointer, light_frames * num_light_rows * 3, torch.float32)
    return _arena_allocation_end(
        pointer, light_frames * num_light_rows * color_width, torch.float32)


def _raytrace_frame_buffers_end(
    initial_pointer, num_frames, width, height, channels, frame_dtype, samples
):
    """Arena pointer after the output and optional Monte Carlo accumulator."""
    pixels = int(width) * int(height)
    pointer = _arena_allocation_end(
        initial_pointer, int(num_frames) * pixels * int(channels), frame_dtype)
    if int(samples) > 1:
        pointer = _arena_allocation_end(
            pointer, int(num_frames) * pixels * 5, torch.float32)
    return pointer


def _postprocess_memory_used(
    *, frame_shape, frame_dtype, anti_alias_level, post_processes, apply_fxaa,
    initial_pointer, device,
):
    """Exact additional arena peak of the built-in post-processing pipeline."""
    from algan.rendering.post_processing import post_process as post_process_module

    return int(post_process_module.get_post_process_memory_required(
        frame_shape=frame_shape,
        frame_dtype=frame_dtype,
        anti_alias_level=anti_alias_level,
        post_processes=post_processes,
        apply_fxaa=apply_fxaa,
        initial_pointer=initial_pointer,
        device=device,
    ))


def _prepare_background_for_chunk(
    background,
    *,
    screen_width,
    screen_height,
    anti_alias_level,
    current_ind,
    new_ind,
    frames_per_second,
    device,
):
    """Prepare one chunk's background for rendering.

    Callable backgrounds are represented by a lightweight deferred value.
    After the arena-backed output exists, the ray tracer streams Python
    callables one frame at a time or evaluates a Taichi ``@ti.func`` across
    the complete batch in one kernel launch. Neither path retains a second
    full-batch image allocation beside the render arena. Image tensors retain
    the eager path.
    """
    aa = int(anti_alias_level)
    width = int(screen_width) * aa
    height = int(screen_height) * aa
    if callable(background):
        from algan.rendering.raytracing.scene_builder import _DeferredBackground

        return _DeferredBackground(
            callback=background,
            width=width,
            height=height,
            anti_alias_level=aa,
            first_frame=current_ind,
            frames_per_second=frames_per_second,
            device=torch.device(device),
        )

    if torch.is_tensor(background):
        background = background.to(device)
    else:
        background = torch.as_tensor(background, device=device)

    if background.dim() > 1:
        if background.shape[0] == 1:
            background = background.expand(
                new_ind - current_ind,
                *[-1 for _ in range(background.dim() - 1)],
            ).contiguous()
        background = background.view(-1, background.shape[-1])
        background = torch.cat((background[:1], background))
        # 0.5 + 255 * bg: scale [0,1] floats to bytes with round-to-nearest
        # (clamp before the cast -- float -> uint8 wraps instead of saturating).
        background = torch.add(0.5, background, alpha=255).clamp_(0, 255)
        background = background.to(torch.uint8)
    return background


class RenderLoopMixin:
    """Frame batching, batch preparation, and the render/video-output loop
    (mixed into :class:`~algan.scene.Scene`)."""

    def _prepare_merged_host_scene(self, primitive_batch):
        """Return the cached source-device scene used for upload/preflight."""
        first = primitive_batch[0]
        cached = getattr(first, "_rt_prepared_host_scene", None)
        if cached is not None:
            return cached

        from algan.rendering.raytracing import settings as rt_settings
        from algan.rendering.raytracing.scene_builder import _merge_scene

        merged_host = _merge_scene(primitive_batch)
        env_map = getattr(self, "environment_map", None)
        if env_map is not None and int(rt_settings.SAMPLES_PER_PIXEL) > 1:
            env_map = None
        first._rt_env_meta = None
        if env_map is not None:
            # Environment resampling/packing is source-device scene prep too;
            # cache it so arena preflight and the subsequent upload see the
            # same widened texture storage without doing the work twice.
            from algan.rendering.raytracing.tracer import _append_env_texture

            merged_host = dict(merged_host)
            texture_device = merged_host["textures"].device
            merged_host["textures"], env_meta = _append_env_texture(
                merged_host["textures"],
                env_map,
                float(getattr(self, "environment_intensity", 1.0)),
                texture_device,
            )
            first._rt_env_meta = env_meta

        cached = (merged_host, env_map)
        first._rt_prepared_host_scene = cached
        return cached

    def _gpu_merge_headroom_bytes(self):
        """Device bytes available outside the render arena for the GPU merge's
        transient out-of-place build scratch (see settings.MERGE_ON_GPU).

        The arena block was reserved from a fraction of the pool at
        ``get_frames`` start, so the pool's *current* free bytes are exactly the
        headroom the merge draws from. A margin is left for Taichi's own
        allocation growth during the render that follows.
        """
        device = self.memory.data.device
        if device.type != "cuda":
            return float("inf")
        return int(get_num_available_bytes(device) * 0.9)

    def _fetched_window_has_stable_actor_set(
        self, actors, start_ind, end_ind
    ):
        """Whether no renderable actor spawns inside a fetched frame window.

        Prefix slicing is exactly equivalent to fetching that prefix only when
        the actor set is unchanged.  Actors that despawn inside the window are
        safe: their already-materialized opacity becomes zero.
        """
        start_time = start_ind / self.frames_per_second
        end_time = end_ind / self.frames_per_second
        for actor in actors:
            if not hasattr(actor, "get_render_primitives"):
                continue
            try:
                spawn_time = float(actor.lifespan.start())
            except (AttributeError, TypeError, ValueError):
                return False
            if start_time < spawn_time <= end_time:
                return False
        return True

    def _can_slice_fetched_batch(self, primitive_batch, total_frames):
        """True when arena probes can use views of one pristine fetch.

        Projection must upload source tensors to a different device.  That
        keeps the candidate's shader/projection mutations isolated from the
        original CPU batch, which is retained for later probes.
        """
        if os.environ.get("ALGAN_REUSE_FETCHED_BATCH", "1") == "0":
            return False
        if total_frames <= 1 or not primitive_batch:
            return False

        from algan.rendering.raytracing import settings as rt_settings

        if not rt_settings.project_on_gpu_active():
            return False
        render_device = torch.device(COMPUTING_DEFAULTS.render_device)
        for primitive in primitive_batch:
            if getattr(primitive, "_rt_projected", False):
                return False
            if not callable(getattr(primitive, "slice_time_window", None)):
                return False
            if not getattr(primitive, "frame_dependent_source_attrs", ()):
                # The base method is intentionally inert until a primitive
                # declares its time-bearing source tensors.
                return False
            if _primitive_source_device(primitive) == render_device:
                return False
        return True

    def _slice_fetched_batch(
        self, primitive_batch, render_state, duration, total_frames
    ):
        primitives = [
            primitive.slice_time_window(0, duration, total_frames)
            for primitive in primitive_batch
        ]
        return primitives, _slice_render_state(
            render_state, 0, duration, total_frames
        )

    def _release_preflight_candidate(self, primitive_batch):
        """Drop projected/merged state belonging to a rejected arena probe."""
        for primitive in primitive_batch:
            for name in tuple(vars(primitive)):
                if name.startswith("_rt_"):
                    setattr(primitive, name, None)
        self.memory.reset()
        empty_cache(force_gc=False)

    def _select_largest_fitting_fetched_prefix(
        self,
        primitive_batch,
        render_state,
        total_frames,
        post_processes,
        transparent_background,
    ):
        """Preflight slices of one fetched source batch and keep the largest.

        The old outer retry loop rematerialized the timeline and rebuilt every
        source primitive for each binary-search candidate.  Here the pristine
        animation-device batch is fetched once.  Each probe is a shallow
        frame-window view uploaded independently for projection, so rejected
        probes cannot mutate it.

        Returns ``(primitives, duration, render_state)`` or ``None`` when this
        batch cannot safely use the reuse path.
        """
        total_frames = int(total_frames)
        if not self._can_slice_fetched_batch(primitive_batch, total_frames):
            return None

        from algan.rendering.raytracing import settings as rt_settings
        from algan.rendering.raytracing.scene_builder import (
            gpu_project_input_bytes,
        )

        # Projection scratch lives outside the arena.  Find its exact source-
        # byte upper bound using cheap views before launching any GPU work.
        headroom = self._gpu_merge_headroom_bytes()

        def project_fits(duration):
            candidate, _ = self._slice_fetched_batch(
                primitive_batch, render_state, duration, total_frames
            )
            estimated_peak = int(
                rt_settings.PROJECT_GPU_PEAK_FACTOR
                * gpu_project_input_bytes(candidate)
            )
            return estimated_peak <= headroom

        upper = _max_duration_that_fits(total_frames, project_fits)

        def probe(duration):
            candidate, candidate_state = self._slice_fetched_batch(
                primitive_batch, render_state, duration, total_frames
            )
            fits = self._prepared_batch_fits_render_arena(
                candidate,
                candidate_state,
                post_processes,
                transparent_background,
            )
            if fits:
                return candidate, candidate_state
            self._release_preflight_candidate(candidate)
            return None

        # Test the largest duration allowed by projection first.  It is often
        # already the final answer, turning the previous retry cascade into one
        # source fetch and one exact preflight.
        result = probe(upper)
        if result is not None:
            logger.info(
                "Arena planner selected %s/%s fetched frames on its first "
                "exact preflight.", upper, total_frames
            )
            return result[0], upper, result[1]

        if upper <= 1:
            raise OutOfRenderMemory(
                "The prepared scene plus one rendered frame does not fit in "
                "the allocated render memory. Please lower the resolution, "
                "anti-alias level, or scene complexity."
            )

        low = 1
        high = upper - 1
        best = 0
        while low <= high:
            duration = (low + high + 1) // 2
            result = probe(duration)
            if result is None:
                high = duration - 1
                continue

            best = duration
            if duration == high:
                logger.info(
                    "Arena planner selected %s/%s fetched frames without "
                    "rematerializing the batch.", duration, total_frames
                )
                return result[0], duration, result[1]

            # A larger prefix may fit.  Release this prepared candidate while
            # retaining only its duration; keeping two merged scenes resident
            # would invalidate the next headroom measurement.
            self._release_preflight_candidate(result[0])
            low = duration + 1

        if best <= 0:
            raise OutOfRenderMemory(
                "The prepared scene plus one rendered frame does not fit in "
                "the allocated render memory. Please lower the resolution, "
                "anti-alias level, or scene complexity."
            )

        # The final binary-search step can be a failure immediately above the
        # best fitting duration, so recreate that winning prefix once.
        result = probe(best)
        if result is None:
            raise OutOfRenderMemory(
                "Render-arena fit was not monotone while selecting a batch."
            )
        logger.info(
            "Arena planner selected %s/%s fetched frames without "
            "rematerializing the batch.", best, total_frames
        )
        return result[0], best, result[1]

    def _prepared_batch_fits_render_arena(
        self,
        primitive_batch,
        render_state,
        post_processes,
        transparent_background,
    ):
        """Whether the prepared scene and at least one frame fit exactly.

        The scene upload grows the arena's reverse pointer; camera/lights,
        output, wavefront state and post-processing grow the forward pointer.
        Preflighting both sides lets the outer batching loop binary-search a
        maximum fitting prepared duration without rendering speculative frames.
        """
        self._last_arena_preflight = None
        if not getattr(self.memory, "managed", False):
            return True

        from algan.rendering.raytracing import settings as rt_settings
        from algan.rendering.raytracing.scene_builder import (
            get_merged_scene_arena_nbytes,
            gpu_merge_input_bytes,
            gpu_project_input_bytes,
        )
        from algan.rendering.raytracing.settings import (
            is_post_process_tonemap_enabled,
        )
        from algan.rendering.raytracing.tracer import (
            get_wavefront_memory_required,
        )

        # Prefetch defers projection to this render thread when it runs on the
        # device (project-on-gpu); otherwise it merely finishes any CPU
        # projection the worker didn't complete. Its transient device scratch
        # (source geometry + shading workspace + packed _rt_* output) lives in
        # the pool's non-arena headroom, so -- like the merge below -- estimate
        # its peak from the source-geometry bytes and shrink the window before
        # attempting it, with the OOM handler as the exact fallback.
        if rt_settings.project_on_gpu_active():
            estimated_project_peak = int(
                rt_settings.PROJECT_GPU_PEAK_FACTOR
                * gpu_project_input_bytes(primitive_batch))
            if estimated_project_peak > self._gpu_merge_headroom_bytes():
                logger.debug(
                    "GPU projection peak estimate %.1f MB exceeds pool "
                    "headroom %.1f MB; shrinking frame window.",
                    estimated_project_peak / 1e6,
                    self._gpu_merge_headroom_bytes() / 1e6)
                return False
        try:
            self._prewarm_render_batch(primitive_batch, render_state)
        except (InsufficientMemoryException, torch.OutOfMemoryError):
            # Device projection overran the pool headroom. Drop partial state
            # and report not-fitting so the caller shrinks the frame window.
            primitive_batch[0]._rt_merged_scene = None
            primitive_batch[0]._rt_prepared_host_scene = None
            empty_cache(force_gc=False)
            return False
        if not all(
            getattr(primitive, "_rt_projected", False)
            for primitive in primitive_batch
        ):
            # Non-ray-traced/custom primitives retain their existing render
            # path; there is no ray-scene layout to preflight here.
            return True

        # The GPU merge's transient out-of-place scratch (inputs relocated to
        # the device + cat/sort/BVH-pyramid workspace + merged output) lives in
        # the render pool's non-arena headroom, separate from the arena bytes
        # sized below. Estimate its peak from the packed inputs and reject a
        # batch that would clearly overflow that headroom before paying for the
        # merge; a low estimate is still caught by the OOM handling just below,
        # which routes to the outer window-shrink retry.
        gpu_merge = rt_settings.merge_on_gpu_active()
        if gpu_merge:
            estimated_merge_peak = int(
                rt_settings.MERGE_GPU_PEAK_FACTOR
                * gpu_merge_input_bytes(primitive_batch))
            if estimated_merge_peak > self._gpu_merge_headroom_bytes():
                logger.debug(
                    "GPU merge peak estimate %.1f MB exceeds pool headroom "
                    "%.1f MB; shrinking frame window.",
                    estimated_merge_peak / 1e6,
                    self._gpu_merge_headroom_bytes() / 1e6)
                return False
        try:
            merged_host, env_map = self._prepare_merged_host_scene(
                primitive_batch)
        except (InsufficientMemoryException, torch.OutOfMemoryError):
            # The device build overran the pool headroom. Drop any partial
            # merge state and report the batch as not fitting so the caller
            # shrinks the frame window and retries.
            primitive_batch[0]._rt_merged_scene = None
            primitive_batch[0]._rt_prepared_host_scene = None
            empty_cache(force_gc=False)
            return False
        scene_bytes = get_merged_scene_arena_nbytes(
            merged_host, self.memory, persist=True)
        if gpu_merge and logger.isEnabledFor(logging.DEBUG):
            measured = int(merged_host.get("_gpu_merge_peak_bytes", -1))
            measured_str = (f"{measured / 1e6:.1f}" if measured >= 0
                            else "n/a")
            logger.debug(
                "GPU merge: est peak %.1f MB (measured %s MB), headroom "
                "%.1f MB, arena scene %.1f MB.",
                estimated_merge_peak / 1e6,
                measured_str,
                self._gpu_merge_headroom_bytes() / 1e6,
                scene_bytes / 1e6)
        bytes_remaining = self.memory.get_num_bytes_remaining()
        if scene_bytes > bytes_remaining:
            return False

        class _LightSnapshot:
            pass

        lights = []
        for origin, light_color, aux in render_state["lights"]:
            light = _LightSnapshot()
            light.origin = origin
            light.light_color = light_color
            light._render_aux = aux
            lights.append(light)

        aa = int(self.render_settings.anti_alias_level)
        render_height = self.num_pixels_screen_height * aa
        render_width = self.num_pixels_screen_width * aa
        render_channels = 5 if transparent_background else 4
        frame_dtype = (
            torch.float32
            if is_post_process_tonemap_enabled()
            else torch.uint8
        )
        samples = max(1, int(rt_settings.SAMPLES_PER_PIXEL))
        initial_pointer = self.memory.current_pointer
        persistent_input_end = _raytrace_persistent_input_end(
            initial_pointer,
            merged_host["num_frames"],
            lights,
            merged_host,
            env_map,
            getattr(self, "environment_ambient", True),
        )
        frame_buffers_end = _raytrace_frame_buffers_end(
            persistent_input_end,
            1,
            render_width,
            render_height,
            render_channels,
            frame_dtype,
            samples,
        )
        postprocess_bytes = _postprocess_memory_used(
            frame_shape=(1, render_height, render_width, render_channels),
            frame_dtype=frame_dtype,
            anti_alias_level=aa,
            post_processes=post_processes,
            apply_fxaa=self.render_settings.fxaa,
            initial_pointer=frame_buffers_end,
            device=self.memory.data.device,
        )
        wavefront_bytes = get_wavefront_memory_required(
            merged_host,
            1,
            render_width,
            render_height,
            light_sources=lights,
            environment_map=env_map,
            near_clip=float(getattr(self.camera, "near", 0.0) or 0.0),
            far_clip=float(getattr(self.camera, "far", 0.0) or 0.0),
        )
        if wavefront_bytes:
            wavefront_bytes += (-frame_buffers_end) % 4
        forward_bytes = (
            frame_buffers_end
            - initial_pointer
            + max(wavefront_bytes, postprocess_bytes)
        )
        need_bytes = scene_bytes + forward_bytes
        self._last_arena_preflight = (need_bytes, bytes_remaining)
        margin = int(getattr(self, "_arena_unmodeled_bytes", 0))
        return need_bytes <= bytes_remaining - margin

    def _note_render_arena_underestimate(self):
        """Grow the preflight safety margin after a batch that passed the
        arena preflight still failed to render: some allocation is not being
        modeled, so future preflights must leave real slack. The failed
        batch's own (need, remaining) pair makes the margin large enough to
        reject at least that exact configuration; repeated failures grow it
        geometrically."""
        prev = int(getattr(self, "_arena_unmodeled_bytes", 0))
        last = getattr(self, "_last_arena_preflight", None)
        observed = 0
        if last is not None:
            need, remaining = last
            observed = max(0, int(remaining) - int(need)) + 1
        self._arena_unmodeled_bytes = max(prev * 2, observed, 8 << 20)
        logger.warning(
            "Arena preflight under-modeled the render; raising its safety "
            "margin to %.1f MB for the rest of this job.",
            self._arena_unmodeled_bytes / 1e6)

    def _reset_render_arena_after_failure(self):
        """Release every allocation owned by a failed render attempt.

        OOM exceptions retain their traceback until the ``except`` block has
        exited. Those frames can own arena views and ordinary CUDA tensors, so
        retry cleanup must run afterwards and must collect reference cycles.
        Resetting both arena pointers releases its forward and persistent
        allocations without reallocating the backing tensor.
        """
        self.memory.reset()
        empty_cache(force_gc=True)

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
                if getattr(primitive, "_rt_projected", False):
                    # Already projected on the batch-prep worker against this
                    # batch's snapshot (see _prewarm_render_batch); its source
                    # geometry has been released, so re-projection is both
                    # redundant and impossible.
                    primitive._rt_projected = False
                    continue
                # Projection belongs to batch preparation, not to the render
                # arena.  Keep every temporary beside the primitive's source
                # tensors (normally the CPU animation device); the scene upload
                # step later moves the packed result into managed render memory.
                original_memory = primitive.memory
                source_device = _primitive_source_device(
                    primitive, fallback=camera.ray_origin.device)
                scratch = ManualMemory(
                    0, device=source_device, managed=False)
                try:
                    primitive.memory = scratch
                    primitive.project_to_screen(camera, self.light_sources)
                finally:
                    primitive.memory = original_memory

            # Reclaim animation-phase residuals before render batching.
            empty_cache(force_gc=False)

            # Projection ran on the source (CPU) device; the merge + STBVH
            # build ran on the CPU or (default) the render device. Upload each
            # unique finished storage directly into the persistent end of
            # ManualMemory before sizing frame chunks, so the exact scene
            # footprint is subtracted from the arena automatically. On the CPU
            # merge no CUDA-side cat/BVH scratch ever sat beside the pool; on
            # the GPU merge that scratch has already been freed (its transient
            # peak was bounded against the pool headroom in the preflight).
            from algan.rendering.raytracing import settings as rt_settings
            from algan.rendering.raytracing.scene_builder import (
                copy_merged_scene_to_arena,
            )

            merged_host, env_map = self._prepare_merged_host_scene(
                primitive_batch)
            device_scene = copy_merged_scene_to_arena(
                merged_host, self.memory, persist=True)
            primitive_batch[0]._rt_device_scene = device_scene
            # The uploaded scene now owns every render-facing tensor.  Drop the
            # extra environment-widened host dict (the base merge remains in its
            # normal primitive cache until the batch is released).
            primitive_batch[0]._rt_prepared_host_scene = None
            if rt_settings.merge_on_gpu_active():
                # The base merge is a second copy of every scene tensor on the
                # render device. The arena copy now owns them, so release the
                # base merge and read the remaining scalar scene metadata below
                # from the arena scene -- keeping steady-state device memory at
                # one scene (parity with the CPU merge, whose base copy is cheap
                # host memory kept until the batch is released).
                primitive_batch[0]._rt_merged_scene = None
                merged_host = device_scene
                empty_cache(force_gc=False)

            render_pointers = self.memory.get_pointers()
            from algan.rendering.raytracing.settings import (
                is_post_process_tonemap_enabled,
            )
            from algan.rendering.raytracing.tracer import (
                get_wavefront_memory_required,
            )

            aa = int(self.render_settings.anti_alias_level)
            render_height = self.num_pixels_screen_height * aa
            render_width = self.num_pixels_screen_width * aa
            render_channels = 5 if transparent_background else 4
            frame_dtype = (
                torch.float32
                if is_post_process_tonemap_enabled()
                else torch.uint8
            )
            samples = max(1, int(rt_settings.SAMPLES_PER_PIXEL))
            persistent_input_end = _raytrace_persistent_input_end(
                render_pointers[0],
                merged_host["num_frames"],
                self.light_sources,
                merged_host,
                env_map,
                getattr(self, "environment_ambient", True),
            )

            def chunk_memory_required(num_frames):
                """Exact forward-arena peak for one candidate render chunk."""
                frame_buffers_end = _raytrace_frame_buffers_end(
                    persistent_input_end,
                    num_frames,
                    render_width,
                    render_height,
                    render_channels,
                    frame_dtype,
                    samples,
                )
                postprocess_bytes = _postprocess_memory_used(
                    frame_shape=(
                        num_frames,
                        render_height,
                        render_width,
                        render_channels,
                    ),
                    frame_dtype=frame_dtype,
                    anti_alias_level=aa,
                    post_processes=post_processes,
                    apply_fxaa=self.render_settings.fxaa,
                    initial_pointer=frame_buffers_end,
                    device=self.memory.data.device,
                )
                wavefront_bytes = get_wavefront_memory_required(
                    merged_host,
                    num_frames,
                    render_width,
                    render_height,
                    light_sources=self.light_sources,
                    environment_map=env_map,
                    near_clip=float(getattr(camera, "near", 0.0) or 0.0),
                    far_clip=float(getattr(camera, "far", 0.0) or 0.0),
                )
                if wavefront_bytes:
                    # Every wavefront state allocation is float32/int32.  The
                    # output can end unaligned for a transparent uint8 frame.
                    wavefront_bytes += (-frame_buffers_end) % 4
                temporary_bytes = max(wavefront_bytes, postprocess_bytes)
                return (
                    frame_buffers_end
                    - render_pointers[0]
                    + temporary_bytes
                )

            bytes_remaining = self.memory.get_num_bytes_remaining()

            def chunk_fits(num_frames):
                return chunk_memory_required(num_frames) <= bytes_remaining

            current_ind = start_ind
            while True:
                if getattr(self.memory, "managed", False):
                    duration = _max_duration_that_fits(
                        end_ind - current_ind,
                        chunk_fits,
                    )
                else:
                    # Unmanaged mode deliberately uses PyTorch's ordinary
                    # allocator.  There is no finite arena to size against,
                    # and arbitrary custom post-process callables remain valid
                    # because their allocations do not need an arena planner.
                    duration = end_ind - current_ind
                new_ind = current_ind + duration

                logger.debug(f'rendering batch with duration {duration}')

                background_source = (
                    self.background_frame
                    if background_color is None else background_color
                )
                bgf = _prepare_background_for_chunk(
                    background_source,
                    screen_width=self.num_pixels_screen_width,
                    screen_height=self.num_pixels_screen_height,
                    anti_alias_level=self.render_settings.anti_alias_level,
                    current_ind=current_ind,
                    new_ind=new_ind,
                    frames_per_second=(
                        self.frames_per_second
                        if callable(background_source) else 1
                    ),
                    device=COMPUTING_DEFAULTS.render_device
                )
                empty_cache()
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

        # Binary search for the largest batch that fits the animation-device
        # budget.  The selected actor set grows monotonically with duration, so
        # the memory predicate is monotone too.
        def get_duration():
            requested_duration = min(
                max_end_time_ind - start_time_ind,
                COMPUTING_DEFAULTS.max_animate_batch_size,
            )

            def fits(duration):
                selected_actors = [
                    actor
                    for actor in primitive_actors
                    if (
                        actor.lifespan.start()
                        <= (start_time_ind + duration) / self.frames_per_second
                    )
                ]
                mem_used = sum(
                    actor_mem[actor] * duration
                    for actor in selected_actors
                )
                return mem_used <= max_mem_used

            return _max_duration_that_fits(requested_duration, fits)

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
        """Run a batch's ``project_to_screen`` (+ the CPU merge when merge and
        project both stay on the CPU) ahead of the render.

        Called on the prefetch worker (from ``fetch_batch``) when projection is
        on the CPU, so the work stays hidden behind the previous batch's
        render; when projection is on the render device (default; see
        settings.PROJECT_ON_GPU) it is deferred to the render thread instead
        (called from the arena preflight), so its transient device peak is
        measured/bounded without a concurrent render polluting the pool.

        Runs against isolated stand-ins so it never touches live render state:

        * a shim camera / shim lights carrying the batch's *snapshot* tensors
          (moved to the render device when projecting there);
        * a scratch unmanaged memory for shading temporaries -- the shared
          render pool's bump pointer is owned by the render thread.

        Identical inputs, identical math, so the packed arrays are byte-exact
        (within device float tolerance) to main-thread preparation. Each
        successfully projected primitive is marked ``_rt_projected`` and skipped
        by ``render_primitive_batch``; on any failure the un-projected remainder
        (and the merge) simply runs on the render thread as before.
        """
        try:
            from algan.rendering.raytracing.primitives import (
                RayTracedBezierCircuitPrimitive, RayTracedTrianglePrimitive)
            from algan.rendering.raytracing.scene_builder import (
                prewarm_merge_cache, upload_primitive_source)
        except Exception:
            return
        if not primitives or not isinstance(
                primitives[0], (RayTracedTrianglePrimitive,
                                RayTracedBezierCircuitPrimitive)):
            return
        from algan.rendering.raytracing import settings as rt_settings

        aa = self.render_settings.anti_alias_level
        # Projection runs on the render device by default (see
        # settings.PROJECT_ON_GPU); the primitive source geometry and the
        # camera/light snapshot are moved there so the packed _rt_* outputs are
        # built on it (ready for the GPU merge, no upload). Off keeps
        # projection on the snapshot's source (CPU) device.
        gpu_project = rt_settings.project_on_gpu_active()
        project_device = (COMPUTING_DEFAULTS.render_device
                          if gpu_project else None)

        def _to_device(value):
            if gpu_project and torch.is_tensor(value):
                return value.to(project_device)
            return value

        class _ShimCamera:
            pass

        camera = _ShimCamera()
        camera.ray_origin = _to_device(render_state["ray_origin"])
        camera.screen_point = _to_device(render_state["screen_point"])
        camera.screen_basis = _to_device(render_state["screen_basis"])
        camera.screen_width = self.num_pixels_screen_width * aa
        camera.screen_height = self.num_pixels_screen_height * aa

        class _ShimLight:
            pass

        lights = []
        for origin, light_color, aux in render_state["lights"]:
            light = _ShimLight()
            light.origin = _to_device(origin)
            light.light_color = _to_device(light_color)
            light._render_aux = _to_device(aux)
            lights.append(light)

        scratch_by_device = {}
        for primitive in primitives:
            if getattr(primitive, "_rt_projected", False):
                continue
            original_memory = primitive.memory
            try:
                if gpu_project:
                    upload_primitive_source(primitive, project_device)
                    source_device = project_device
                else:
                    source_device = _primitive_source_device(
                        primitive, fallback=render_state["ray_origin"].device)
                scratch = scratch_by_device.get(source_device)
                if scratch is None:
                    scratch = ManualMemory(
                        0, device=source_device, managed=False)
                    scratch_by_device[source_device] = scratch
                primitive.memory = scratch
                primitive.project_to_screen(camera, lights)
                primitive._rt_projected = True
            finally:
                primitive.memory = original_memory
        # The merge + STBVH build ride the prefetch worker only when they run
        # on the CPU. When they run on the render device (the default; see
        # settings.MERGE_ON_GPU) they are deferred to the render thread so
        # their transient device peak is measured/bounded without a concurrent
        # render polluting the pool.
        if not rt_settings.merge_on_gpu_active():
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
        # Batch preparation is CPU/source-device work.  Keeping this snapshot
        # beside the materialized animation tensors prevents the prefetch worker
        # from allocating the next batch on the render device while the current
        # batch is still resident there.
        camera_location = camera.location
        device = camera_location.device
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
            ray_origin=camera_location.unsqueeze(-2).to(device),
            screen_point=camera.screen.location.unsqueeze(-2).to(device),
            screen_basis=camera.get_render_screen_basis().to(device),
            lights=lights,
        )

    def get_frames(self, start_time_ind, end_time_ind, background_color=None,
                   post_processes=(bloom_filter,), manual_memory=True):
        """Yield frames and always release per-render state on exit.

        The wrapper is deliberately outside the implementation generator so
        its ``finally`` also runs for OOMs, worker failures, and callers that
        close the generator before consuming every frame.
        """
        original_background = self.background_frame
        original_memory = self.memory
        try:
            # Rendering is inference-only, but the scope is local to Algan so
            # importing the library does not alter PyTorch autograd globally.
            with torch.inference_mode():
                yield from self._get_frames_impl(
                    start_time_ind,
                    end_time_ind,
                    background_color=background_color,
                    post_processes=post_processes,
                    manual_memory=manual_memory,
                )
        finally:
            self.background_frame = original_background
            render_memory = self.memory
            if render_memory is not None and render_memory is not original_memory:
                render_memory.data = None
            self.memory = original_memory

    def _get_frames_impl(self, start_time_ind, end_time_ind,
                         background_color=None,
                         post_processes=(bloom_filter,), manual_memory=True):
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
        # Safety margin learned from render failures this job: when a batch
        # that passed the arena preflight still fails to render, the preflight
        # under-modeled some allocation, so subsequent preflights must leave at
        # least this much slack (see _note_render_arena_underestimate).
        self._arena_unmodeled_bytes = 0
        self._last_arena_preflight = None

        # Adaptive gen-fused forecast (settings.WF_GEN_FUSED == "auto") is fed
        # per-batch render timings below; a new job restarts its batch count.
        from algan.rendering.raytracing import settings as _rt_settings

        _rt_settings._begin_render_job()

        with Off(
                record_attr_modifications=False,
                record_funcs=False,
                priority_level=math.inf,
        ):
            current_time_ind = start_time_ind

            max_animate_mem = int(
                    COMPUTING_DEFAULTS.portion_of_memory_used_for_animating
                    * get_num_available_bytes(COMPUTING_DEFAULTS.animation_device)
                )

            # Prefetch pipeline: while batch b renders on this thread, batch
            # b+1 is prepped (animation/source-device geometry generation) on
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

            def fetch_batch(time_ind, batch_end_ind=None):
                if batch_end_ind is None:
                    batch_end_ind = end_time_ind
                with torch.inference_mode(inference_mode_enabled):
                    batch = self.get_batch_of_primitives(
                        time_ind, batch_end_ind, actors, max_animate_mem
                    )
                    # Pre-run the ray tracer's vertex shade + packing
                    # (project_to_screen) and merged-scene / STBVH build here
                    # (all torch-only) so they ride the prefetch: batch b+1's
                    # prep runs on the worker while batch b renders, turning
                    # seconds of otherwise-serial render-thread CPU work into
                    # hidden time. ALGAN_PREFETCH_MERGE=0 falls back to
                    # projecting + merging on the render thread. When projection
                    # runs on the render device (settings.PROJECT_ON_GPU) it is
                    # deferred to the render thread entirely -- GPU work on this
                    # worker would contend with the in-flight render and pollute
                    # the transient-peak stats -- so only the CPU-projection
                    # path prewarms here.
                    from algan.rendering.raytracing import (
                        settings as rt_settings)

                    if (batch[0]
                            and os.environ.get("ALGAN_PREFETCH_MERGE", "1")
                            != "0"
                            and not rt_settings.project_on_gpu_active()):
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
            retry_end_ind = None
            retry_lower_duration = 0
            retry_upper_duration = None
            try:
                while True:
                    _sync_devices()
                    s = time.time()
                    fetch_end_ind = (
                        retry_end_ind
                        if retry_end_ind is not None else end_time_ind
                    )
                    logger.info(
                        f"Fetching batch {current_time_ind}:{fetch_end_ind}."
                    )
                    if retry_end_ind is not None:
                        primitives, new_time_ind, render_state = fetch_batch(
                            current_time_ind, retry_end_ind)
                        retry_end_ind = None
                    elif pending is not None:
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

                    duration = new_time_ind - current_time_ind
                    planned_prefix = None
                    if (
                        retry_upper_duration is None
                        and primitives
                        and self._fetched_window_has_stable_actor_set(
                            actors, current_time_ind, new_time_ind
                        )
                    ):
                        planned_prefix = (
                            self._select_largest_fitting_fetched_prefix(
                                primitives,
                                render_state,
                                duration,
                                post_processes,
                                transparent_background,
                            )
                        )

                    if planned_prefix is not None:
                        fetched_primitives = primitives
                        primitives, duration, render_state = planned_prefix
                        new_time_ind = current_time_ind + duration
                        del fetched_primitives
                        batch_fits = True
                    else:
                        batch_fits = (
                            not primitives
                            or self._prepared_batch_fits_render_arena(
                                primitives,
                                render_state,
                                post_processes,
                                transparent_background,
                            )
                        )
                    if not batch_fits:
                        retry_upper_duration = min(
                            duration,
                            retry_upper_duration
                            if retry_upper_duration is not None
                            else duration,
                        )
                        if duration <= 1 and retry_lower_duration == 0:
                            raise OutOfRenderMemory(
                                "The prepared scene plus one rendered frame "
                                "does not fit in the allocated render memory. "
                                "Please lower the resolution, anti-alias "
                                "level, or scene complexity."
                            )
                        logger.warning(
                            "Prepared batch does not fit the render arena; "
                            "binary-searching the largest fitting duration."
                        )
                        if primitives:
                            primitives[0]._rt_device_scene = None
                            primitives[0]._rt_prepared_host_scene = None
                            primitives[0]._rt_merged_scene = None
                        del primitives
                        self.memory.reset()
                        empty_cache(force_gc=False)
                        target_duration = max(
                            1,
                            (
                                retry_lower_duration
                                + retry_upper_duration
                            ) // 2,
                        )
                        retry_end_ind = current_time_ind + target_duration
                        continue

                    if retry_upper_duration is not None:
                        retry_lower_duration = max(
                            retry_lower_duration, duration)
                        if False:#retry_upper_duration - retry_lower_duration > 1:
                            # This candidate fits, but a failed upper bound
                            # leaves room to probe a larger prepared batch
                            # without emitting speculative frames.
                            if primitives:
                                primitives[0]._rt_device_scene = None
                                primitives[0]._rt_prepared_host_scene = None
                                primitives[0]._rt_merged_scene = None
                            del primitives
                            self.memory.reset()
                            empty_cache(force_gc=False)
                            target_duration = (
                                retry_lower_duration
                                + retry_upper_duration
                            ) // 2
                            retry_end_ind = (
                                current_time_ind + target_duration)
                            continue

                    # Only prefetch the successor once the current duration is
                    # final. A speculative successor would start at the wrong
                    # boundary while the binary preflight search is active.
                    if executor is not None and new_time_ind < end_time_ind:
                        pending = executor.submit(fetch_batch, new_time_ind)
                    if len(primitives) > 0:
                        self.has_any_active_actors = True

                        s = time.time()
                        logger.info(
                            f"Rendering {(new_time_ind - current_time_ind) / self.frames_per_second} seconds of video."
                        )
                        produced_output = False
                        retry_after_render_failure = False
                        try:
                            for frame_batch in self.render_primitive_batch(
                                primitives,
                                current_time_ind,
                                new_time_ind,
                                save_image,
                                post_processes,
                                transparent_background,
                                background_color,
                                render_state=render_state,
                            ):
                                produced_output = True
                                yield frame_batch
                        except (InsufficientMemoryException,
                                OutOfRenderMemory,
                                torch.OutOfMemoryError) as render_exc:
                            if produced_output or duration <= 1:
                                raise
                            self._note_render_arena_underestimate()
                            logger.warning(
                                "Render failed despite arena preflight "
                                f"({type(render_exc).__name__}: {render_exc}); "
                                f"retrying {current_time_ind}:{new_time_ind} "
                                "at half duration.")
                            # A prefetched successor starts at the old end and
                            # is invalid after this split. Drain and discard it
                            # before rematerializing the smaller current batch.
                            if pending is not None:
                                try:
                                    pending.result()
                                except Exception:
                                    pass
                                pending = None
                            if primitives:
                                primitives[0]._rt_device_scene = None
                                primitives[0]._rt_prepared_host_scene = None
                                primitives[0]._rt_merged_scene = None
                            del primitives
                            # The arena preflight approved this duration and
                            # was wrong, so it cannot arbitrate durations just
                            # below it: binary-probing upward would converge to
                            # duration-1, render-fail again, and repeat -- an
                            # O(frames) cascade of ~seconds-long refetches.
                            # Back off geometrically instead: cap the bounds at
                            # half so the halved candidate renders immediately,
                            # and let the raised preflight margin (see above)
                            # arbitrate everything afterwards.
                            retry_lower_duration = 0
                            retry_upper_duration = max(1, duration // 2)
                            retry_end_ind = (
                                current_time_ind + max(1, duration // 2))
                            retry_after_render_failure = True
                        if retry_after_render_failure:
                            # This deliberately runs after the exception handler:
                            # only then has Python released the exception state
                            # and traceback frames that may own CUDA tensors.
                            self._reset_render_arena_after_failure()
                            continue
                        del primitives
                        # Free previous batch data before allocating next batch.
                        empty_cache(force_gc=False)
                        _sync_devices()
                        e = time.time()
                        logger.info(
                            f"{current_time_ind}:{new_time_ind}, took {e - s} seconds"
                        )
                        if _rt_settings._note_batch_rendered(
                                new_time_ind - current_time_ind, e - s,
                                end_time_ind - new_time_ind):
                            logger.info(
                                "Adaptive gen-fused: forecasted remaining "
                                "render time justifies compiling the fused "
                                "generation kernels; fusing from the next "
                                "batch (output is unaffected)."
                            )

                    retry_lower_duration = 0
                    retry_upper_duration = None
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

    def _drain_video_writer(self, frame_queue, writer_process, file_writer):
        """Flush the frame queue and wait for the encoder to finish.

        Split out so a profiler can time the serial video-encode tail (the
        block spent waiting on ffmpeg after the last frame is produced) as its
        own stage instead of leaving it in the profile's unaccounted bucket.
        """
        frame_queue.put(None)  # sentinel: end of stream
        writer_process.join()
        file_writer.close()

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

        self._drain_video_writer(frame_queue, writer_process, file_writer)

        if os.path.exists(file_path_out):
            os.remove(file_path_out)
        os.rename(file_path, file_path_out)
        if (not hasattr(self, 'has_any_active_actors')) or (not self.has_any_active_actors):
            warnings.warn(
                "You rendered an empty scene! Did you forget to spawn() your Mobs?",
                EmptySceneWarning,
            )
