"""The frame-batching render loop, split out of :mod:`algan.scene`.

:class:`RenderLoopMixin` is mixed into :class:`~algan.scene.Scene` and is not
useful standalone (``self`` is always the Scene). It owns everything between
"the timeline has recorded animations" and "frames are streamed to the video
writer": batch sizing by memory budget, timeline state materialization and
primitive batching
(:meth:`~algan.render_loop.RenderLoopMixin._get_batch_of_primitives`),
the prefetch pipeline (:meth:`~algan.render_loop.RenderLoopMixin.get_frames`),
per-batch rendering
(:meth:`~algan.render_loop.RenderLoopMixin._render_primitive_batch`), and video
file output (:meth:`~algan.render_loop.RenderLoopMixin._render_to_video`).
"""

from __future__ import annotations

import bisect
import collections
import contextlib
import logging
import math
import os
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from queue import Empty, Full, Queue
from types import SimpleNamespace

import torch
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import algan.rendering.raytracing.settings as rt_settings_module
from algan.animation_timeline.animation_contexts import Off
from algan.environment import env_flag, env_float
from algan.errors import (
    AlganConfigurationError,
    AlganWarning,
    UnsupportedFeatureWarning,
    _user_stacklevel,
)
from algan.logging.logger import PERF, get_logger, resolve_progress_style
from algan.rendering.memory_model import (
    AffineFrameCost,
    ChunkMemoryModel,
    PeakRatioModel,
    chunk_signature,
)
from algan.rendering.post_processing.bloom import bloom_filter
from algan.rendering.primitives.bezier_circuit_primitive import BezierCircuitPrimitive
from algan.rendering.primitives.primitive import OutOfRenderMemory
from algan.rendering.raytracing.truncation import (
    path_tracer_fallback_hint,
    reset_truncations,
)
from algan.rendering.taichi_runtime import (
    _sync_devices,
    ensure_taichi_for_render,
    render_job_holding_the_arch,
)
from algan.settings import SETTINGS
from algan.settings._startup import _ANIMATION_DEVICE, render_device
from algan.utils.color_space import srgb_to_linear
from algan.utils.memory_utils import (
    InsufficientMemoryException,
    ManualMemory,
    _coalesced_memory_reclamation,
    auto_record_enabled,
    begin_cuda_peak,
    end_cuda_peak,
    get_num_available_bytes,
    is_cuda_oom,
    note_nonarena_peak,
    release_torch_memory,
    scene_excluded_from_gc,
)

logger = get_logger("scene")

# Below this many frames, the "log" style reports no progress at all. A bar is
# free to be short-lived; ten log lines about a two-second render are not.
_PROGRESS_MIN_LOGGED_FRAMES = 60

#: Share of a budget that a batch's *actor set* must take -- the part no frame
#: count can shrink -- before the window stops being the right lever and the
#: loop retreats behind a spawn to carry fewer actors instead (see
#: RenderLoopMixin._batch_actor_share). At half the budget, halving the frames
#: can at best halve the other half, so a shorter window buys little; below it
#: the batch is frame-bound and the ordinary search is the cheaper answer.
_ACTOR_SHARE_RETREAT = 0.5

#: Share of the render device's free memory (outside the arena) that a batch's
#: preparation may fill with the frame windows that materialize there (see
#: RenderLoopMixin._render_device_prep_budget). The rest is left for the
#: merge's and projection's transient out-of-arena scratch, which the batch
#: preflight bounds against the same headroom, and for the prefetched
#: successor batch that prepares while this one renders.
_RENDER_PREP_FRACTION = 0.4

#: Share of the arena's remaining bytes above which a batch's *scene* is
#: considered to leave too little for frames whose cost has not been measured
#: yet (see RenderLoopMixin._prepared_batch_fits_render_arena). Below it, a
#: first batch keeps the full window the geometry allows.
_UNMEASURED_SCENE_SHARE = 0.5

#: Frames such a batch is held to until the chunk model has measured one. Long
#: enough to batch usefully, short enough that a window sized against a zero
#: frame cost cannot commit the render to geometry it will not be able to
#: trace over.
_UNMEASURED_PROBE_FRAMES = 8

#: Maximum multiplicative growth of the next fetched frame window after the
#: arena has accepted a batch. The fitted arena capacity can be optimistic
#: because it includes frame-independent scene bytes; an unbounded jump can
#: therefore rematerialize a much larger batch only to reject it immediately.
#: 1.6 preserved the accepted 7/11/12-frame partition in the UHD Metal
#: benchmark while avoiding its failed 23-frame speculative fetch.
_ARENA_FETCH_GROWTH_LIMIT = 1.6

#: Most stills one render batch of a still job may carry (see
#: RenderLoopMixin._still_group_ends).
_STILL_GROUP_MAX_FRAMES = 16


#: Clean batches in a row after which the arena preflight's learned safety
#: margin halves, and the ceiling that wait doubles up to after each failure
#: that follows a lowering (RenderLoopMixin._note_render_arena_success). One:
#: while the margin is up the batches are a few frames long, so a slower start
#: spends most of a scene recovering (HD scene 12: 36 batches of 3 frames).
_ARENA_MARGIN_DECAY_BATCHES = 1
_ARENA_MARGIN_DECAY_MAX_BATCHES = 64


def _other_device_bytes(device):
    """``(held by others, device total)`` in bytes, or None where unknown.

    "Others" is everything but this process's PyTorch allocator -- other
    processes, the display, driver and compiler runtimes: what
    ``mem_get_info`` says is in use, less what PyTorch has reserved (the render
    arena included).
    """
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    try:
        free, total = torch.cuda.mem_get_info(device)
        reserved = torch.cuda.memory_reserved(device)
    except Exception:  # noqa: BLE001 -- telemetry only; None means "cannot say"
        return None
    return max(0, int(total) - int(free) - int(reserved)), int(total)


#: How much more of the device other holders must have taken, since the render
#: began, for an under-modelled batch to be blamed on them rather than on the
#: render itself: the larger of 256 MB and 1/12 of the device.
_OTHER_HOLDER_MIN_BYTES = 256 << 20


def _next_arena_fetch_cap(duration, arena_frames):
    """Bound the next materialized window without shrinking what just fit."""
    return min(
        max(1, int(duration), int(arena_frames)),
        max(1, int(duration * _ARENA_FETCH_GROWTH_LIMIT)),
    )


def _one_frame_does_not_fit_message():
    return (
        "The prepared scene plus one rendered frame does not fit in the "
        "allocated render memory. Please lower the resolution, anti-alias "
        "level, or scene complexity." + path_tracer_fallback_hint()
    )


def _render_device_pool_bytes(device):
    """Bytes the render device's memory pool is taken to hold, in total.

    The one figure every out-of-arena budget is a share of:
    :meth:`RenderLoopMixin._gpu_merge_headroom_bytes` subtracts the arena from
    it, and :meth:`RenderLoopMixin._render_device_prep_budget` takes a fraction
    of what the arena leaves.

    It is deliberately the device's *total* rather than what is free at the
    moment of asking. These budgets size the next batch window, and a window is
    not a harmless performance choice: the frames that share a merge share the
    batch-wide chord-count and promotion decisions, so a window that moved with
    another tenant's momentary VRAM use moved pixels from one run to the next
    -- measured on the T4 box as a second window of 8 frames in one process and
    11 in another, with ~5% of the frame's pixels differing.
    ``available_memory_override`` wins on a measured device, exactly as it does
    in :func:`~algan.utils.memory_utils.get_num_available_bytes`, so a run that
    pins the pool pins the windows with it.

    **Never infinite, for any device.** Both budgets used to answer
    ``float("inf")`` for anything that was not CUDA or CPU, which meant MPS.
    That does not fail loudly; it silently makes every guard they feed
    unsatisfiable -- ``estimated_merge_peak > headroom`` cannot be true of
    infinity -- so a Metal render sized its windows against the arena alone
    while the merge and projection scratch, which live outside the arena, were
    bounded by nothing at all. That is what killed ``materials_and_lighting``
    two thirds of the way through on a 7 GB Mac (``DESIGN_mps_support.md``
    §1.4). A device this function does not know now gets the CPU's ceiling
    rather than a free pass.
    """
    device = torch.device(device)
    if device.type in ("cuda", "mps"):
        override = SETTINGS.computing.available_memory_override
        if override is not None:
            return int(override)
        if device.type == "cuda":
            _, total_bytes = torch.cuda.mem_get_info(device)
            return int(total_bytes)
        # Metal's ``recommendedMaxWorkingSetSize``: the driver's own answer for
        # how much of a unified-memory machine a process should hold. It is the
        # analogue of CUDA's total, and it is already what
        # ``get_num_available_bytes`` measures MPS against.
        return int(torch.mps.recommended_max_memory())
    # A CPU render has a renderer working-memory budget too --
    # ``cpu_render_memory_budget``, which is what the arena itself is a fraction
    # of -- so the merge has the same finite
    # headroom outside the arena that a card does. Leaving this infinite left
    # the CPU with no window-shrinking lever at all: every budget that bounds a
    # batch before it is built was CUDA-only, so CPU took the largest window
    # the geometry allowed and met the arena's limit mid-render instead.
    return int(SETTINGS.computing.cpu_render_memory_budget)


@contextlib.contextmanager
def _render_progress(total):
    """Report render progress on the frame the user is waiting for.

    Yields a callable to invoke once per frame written.

    Where it can be drawn, this is a tqdm bar, which buys the estimate the old
    in-place percentage line never gave: renders run for minutes, and "22.9%"
    does not tell you whether to wait. Where stderr is being captured into a
    stored log -- pytest, CI -- it degrades to at most ten log lines, because
    there a bar's carriage returns are kept rather than acted on and it becomes
    hundreds of lines. ``logging.logger.resolve_progress_style`` decides which;
    ``ALGAN_PROGRESS`` / ``set_progress_style`` override it.

    Whether to report at all is settled once, on entry, rather than per frame:
    a bar is an object with a lifetime, so a level or style change mid-render
    can no longer conjure or retire one.
    """
    if not logger.isEnabledFor(logging.INFO):
        yield lambda: None
        return

    style = resolve_progress_style()
    if style == "none":
        yield lambda: None
        return

    if style == "log":
        done = 0
        step = max(1, total // 10)

        def report_logged():
            nonlocal done
            done += 1
            if total >= _PROGRESS_MIN_LOGGED_FRAMES and done % step == 0:
                logger.info(
                    "Rendering %d/%d frames (%.0f%%)", done, total, 100 * done / total
                )

        yield report_logged
        return

    bar = tqdm(
        total=total,
        desc="Rendering",
        unit="frame",
        file=sys.stderr,
        dynamic_ncols=True,
        # Global average rather than tqdm's default EWMA (smoothing=0.3).
        # Frames do not arrive at a steady rate: the memory model sizes each
        # batch at runtime and grows it geometrically, so a batch's frames land
        # together and the next stalls while it preps, and an OOM retry
        # re-renders a shrunken window. An EWMA tracks those bursts and swings
        # the estimate hardest over the early batches, which is when people
        # actually read it.
        smoothing=0,
    )
    try:
        # Route console logging through tqdm.write for the bar's lifetime, so a
        # warning or a PERF batch-split mid-render prints above the bar instead
        # of smearing it. Only handlers writing to stdout/stderr are swapped, so
        # a user's file handler is left alone; a console handler that is not a
        # StreamHandler at all (Manim's rich one) is out of reach and still
        # collides. Algan's logger does not propagate, so it is redirected by
        # name rather than reached through the root.
        with logging_redirect_tqdm(loggers=[get_logger(), logging.root]):
            yield lambda: bar.update(1)
    finally:
        bar.close()


class EmptySceneWarning(Warning):
    pass


def write_frames_from_queue(queue, file_writer):
    while True:
        item = queue.get()
        if item is None:  # Sentinel value to signal the end
            break
        # ``(frame, copies)``: a held frame arrives once with its repeat count
        # (``_VideoWriter.put``); a bare frame is written once.
        frame, copies = item if isinstance(item, tuple) else (item, 1)
        array = frame.numpy()
        for _ in range(copies):
            _write_frame(file_writer, array)


def _write_frame(file_writer, array):
    """Pipe one frame to the encoder.

    moviepy's ``FFMPEG_VideoWriter.write_frame`` writes ``array.tobytes()``,
    a full copy of the frame per write; the pipe takes the array's own buffer
    just as well. Same bytes, same order: at 1080p that raised the encode's
    throughput from 140 to 178 frames per second on the dev box, and the
    encoder is what a hold-heavy scene waits on. Anything that is not
    moviepy's writer, a non-contiguous array, or a failed write goes through
    ``write_frame`` itself, which also produces moviepy's error report.
    """
    stdin = getattr(getattr(file_writer, "proc", None), "stdin", None)
    if (
        stdin is None
        or not getattr(array, "flags", None)
        or not array.flags["C_CONTIGUOUS"]
        or _opt_disabled_render_loop("writerview")
    ):
        file_writer.write_frame(array)
        return
    try:
        stdin.write(memoryview(array))
    except OSError:
        file_writer.write_frame(array)


def _opt_disabled_render_loop(name):
    from algan.animation_timeline.timeline import _opt_disabled

    return _opt_disabled(name)


class _VideoWriter:
    """A bounded encoder queue whose failures reach its sole producer."""

    def __init__(self, file_writer):
        self.file_writer = file_writer
        self.queue = Queue(maxsize=8)
        self._error = None
        self._thread = threading.Thread(
            target=self._run, name="algan-video-writer", daemon=True
        )

    def start(self):
        self._thread.start()

    def _run(self):
        try:
            write_frames_from_queue(self.queue, self.file_writer)
        except BaseException as exc:
            # Publish before exiting. A producer blocked by backpressure must
            # learn why the consumer stopped, including on the final frame.
            self._error = exc

    def _raise_if_failed(self):
        if self._error is not None:
            raise self._error

    def put(self, frame, copies=1):
        """Queue ``frame`` to be written ``copies`` times in a row.

        A held frame (static-frame reuse) is ONE queue entry however many
        times it repeats, so the bound limits distinct frames in flight --
        the memory it exists for -- and a long hold no longer blocks the
        render thread until the encoder has drained all but eight copies.
        Measured at HD on the backprop explainer, render-thread time outside
        every profiled stage was 0.25-0.4 s per second of video, over half
        of a hold-heavy scene's render.
        """
        item = frame if copies == 1 or frame is None else (frame, int(copies))
        while True:
            self._raise_if_failed()
            try:
                # Poll so a dead encoder cannot strand a full queue, and so
                # Ctrl-C remains observable on Windows while waiting for space.
                self.queue.put(item, timeout=0.1)
                break
            except Full:
                pass
        self._raise_if_failed()

    def _join(self):
        while self._thread.is_alive():
            self._thread.join(timeout=0.1)

    def _discard_frames(self):
        while True:
            try:
                self.queue.get_nowait()
            except Empty:
                return

    def abort(self):
        # Only the render thread produces frames. Once it enters cleanup we
        # can discard the backlog and insert a sentinel without blocking,
        # whether the consumer has failed, is writing, or is waiting on get().
        self._discard_frames()
        self.queue.put_nowait(None)
        self._join()
        self._discard_frames()

    def finish(self):
        try:
            self.put(None)
            self._join()
            self._raise_if_failed()
        except BaseException:
            self.abort()
            raise


def _max_duration_that_fits(requested_frames, fits):
    """Largest positive runtime for which the monotone ``fits`` predicate is
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


def _aux_source_registry(scene):
    """The armed object-id pass's ``{Mob.id: actor}`` registry, or ``None``.

    The pass is armed for one render by setting ``scene._aux_id_registry`` to a
    dict (and disarmed by deleting it); while armed, every primitive the batch
    takes from an actor is stamped with that actor's ``Mob.id``
    (:func:`_stamp_primitive_sources`), which the collections and the scene
    merge carry into ``tri_obj_source_ids`` / ``circuit_source_ids``. See
    :mod:`algan.rendering.pass_identity`. A camera-view pass is a copy of the
    Scene's ``__dict__``, so it shares the very same dict.
    """
    registry = getattr(scene, "_aux_id_registry", None)
    return registry if isinstance(registry, dict) else None


def _stamp_primitive_sources(primitives, actor, registry):
    """Record ``actor`` as the source of each of ``primitives``.

    ``_source_mob_id`` is a plain int (not a tensor, so nothing uploads or
    counts it; not ``_rt_``-prefixed, so ``slice_time_window`` keeps it). A
    primitive that already names a source -- an aggregate that stamped its
    parts more precisely (``Arrow3D``) -- keeps it. The actor itself goes into
    ``registry`` so the ids resolve back to Mobs after the render; a packed
    view shares its pack's id, which is why the actor (the pack, whose
    ``parents`` chain is intact) is what is recorded.
    """
    mob_id = int(actor.id)
    registry.setdefault(mob_id, actor)
    for primitive in primitives:
        if (
            getattr(primitive, "_source_mob_id", None) is None
            and getattr(primitive, "_circuit_source_ids", None) is None
        ):
            primitive._source_mob_id = mob_id


def _primitive_source_device(primitive, fallback=None):
    """Device holding a not-yet-projected primitive's source geometry."""
    for name in (
        "corners",
        "colors",
        "normals",
        "mob_center",
        "next_segment_inds",
    ):
        value = getattr(primitive, name, None)
        if torch.is_tensor(value):
            return value.device
    if fallback is not None:
        return torch.device(fallback)
    return _ANIMATION_DEVICE


def _projection_anti_alias_level(scene, primitives):
    """Choose conservative camera projection scale before scene merging.

    The exact analytic-raster route is known only after packed-scene metadata
    exists.  Projection happens earlier, so use output resolution whenever the
    primitive classes and live settings make that route possible.  If a later
    material/legacy check falls back, Bezier tessellation remains at least as
    fine as the AA=2 reference and its bounds are merely more conservative.
    """
    requested = max(1, int(scene.video_settings.supersampling))
    rt_settings = SETTINGS.raytracing
    from algan.rendering.raytracing.primitives import (
        RayTracedBezierCircuitPrimitive,
        RayTracedTrianglePrimitive,
    )

    if not primitives:
        return requested, False
    if int(rt_settings.samples_per_pixel) > 1:
        # The path tracer renders at output resolution (jittered sub-pixel
        # samples are its anti-aliasing), so projection tessellates for the
        # output pixel grid.
        return 1, False
    if (
        not rt_settings.hybrid_raster
        or not rt_settings.analytic_aa
        or float(getattr(scene.camera, "near", 0.0) or 0.0) > 0.0
    ):
        return requested, False
    ray_types = (RayTracedTrianglePrimitive, RayTracedBezierCircuitPrimitive)
    if not all(isinstance(primitive, ray_types) for primitive in primitives):
        return requested, False

    has_tri = any(isinstance(p, RayTracedTrianglePrimitive) for p in primitives)
    has_bez = any(isinstance(p, RayTracedBezierCircuitPrimitive) for p in primitives)
    possible = (
        (has_tri or has_bez)
        and (not has_tri or rt_settings.analytic_aa_tri_active())
        and (not has_bez or rt_settings.analytic_aa_bez_active())
    )
    return (1 if possible else requested), bool(possible)


def _slice_render_state(render_state, start, end, total_frames):
    """Return a frame-window view of an immutable render-state snapshot."""
    start = int(start)
    end = int(end)
    total_frames = int(total_frames)

    def sliced(value):
        if (
            torch.is_tensor(value)
            and value.ndim > 0
            and int(value.shape[0]) == total_frames
        ):
            return value[start:end]
        return value

    return {
        **(
            {"frame_indices": render_state["frame_indices"][start:end]}
            if "frame_indices" in render_state
            else {}
        ),
        "ray_origin": sliced(render_state["ray_origin"]),
        "screen_point": sliced(render_state["screen_point"]),
        "screen_basis": sliced(render_state["screen_basis"]),
        # Optional: snapshots of cameras without a lens (test doubles) omit it,
        # and a key missing here would silently drop the lens from every
        # prefix retry and overlapped-prep window.
        **(
            {"camera_lens": sliced(render_state["camera_lens"])}
            if "camera_lens" in render_state
            else {}
        ),
        "lights": [
            (sliced(origin), sliced(color), sliced(aux))
            for origin, color, aux in render_state["lights"]
        ],
        # The kept light objects stay aligned with ``lights``. A light that
        # overlaps the fetched window but not this prefix keeps its (all-zero)
        # rows -- zero rows are inert everywhere, so the prefix renders the
        # same as a fresh fetch of it would.
        "light_objects": render_state.get("light_objects"),
        "light_active": [
            sliced(active)
            for active in render_state.get(
                "light_active", [None] * len(render_state["lights"])
            )
        ],
    }


def _frame_window_bounds(start, end, frame_indices=None):
    """Real frame envelope for a dense window or a slice of selected frames."""
    if frame_indices is None:
        return start, end
    return frame_indices[start], frame_indices[end - 1] + 1


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
    frame_indices=None,
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
            frame_indices=frame_indices,
        )

    if torch.is_tensor(background):
        background = background.to(device)
    else:
        background = torch.as_tensor(background, device=device)

    if background.dim() > 1:
        if background.shape[0] == 1:
            from algan.animation_timeline.timeline import _opt_disabled

            if not _opt_disabled("staticbg"):
                from algan.rendering.raytracing.scene_builder import (
                    _StaticImageBackground,
                )

                # One frame's rows, standing for every frame of the chunk:
                # the copies below would be byte-identical frame to frame,
                # and at a super-sampled resolution the float temporaries
                # that built them were the largest allocation of the render.
                rows = background.reshape(-1, background.shape[-1])
                rows = torch.cat((rows[:1], rows))
                rows = torch.add(0.5, rows, alpha=255).clamp_(0, 255)
                return _StaticImageBackground(
                    rows.to(torch.uint8), new_ind - current_ind
                )
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


def _decode_light_rgb(rgb, frame_local):
    """sRGB-decode ``rgb``; a still batch's (``frame_local``) frame by frame.

    The decode's ``pow`` rounds one ulp apart in a vectorized loop and in its
    scalar tail, so decoded as one array a frame's light could differ from
    the same frame decoded alone (see ``scene_builder._decode_frames``).
    """
    if frame_local and rgb.dim() and rgb.shape[0] > 1:
        return torch.cat([srgb_to_linear(frame) for frame in rgb.split(1)])
    return srgb_to_linear(rgb)


def _check_post_processes(post_processes):
    """Reject a non-callable pass before the render rather than after it.

    Each pass is applied to finished frames, so a bad entry used to surface as
    ``TypeError: 'int' object is not callable`` once the whole render had
    already been paid for.
    """
    if post_processes is None:
        return
    if callable(post_processes):
        raise AlganConfigurationError(
            "post_processes takes a sequence of passes, not one pass. Wrap it: "
            "post_processes=(bloom_filter,)."
        )
    for index, process in enumerate(post_processes):
        if not callable(process):
            raise AlganConfigurationError(
                f"post_processes[{index}] is not callable: got "
                f"{type(process).__name__}. Each pass is a function applied to "
                f"finished frames, such as bloom_filter or "
                f"partial(bloom_filter, glow_spread=0.015). Pass () for no "
                f"post-processing."
            )


def _framewise_post_process(process, batch_native=True):
    """Adapt a still-image pass to preserve one callback per output frame.

    Sparse still rendering may put several requested timestamps in one render
    batch. ``Scene.save_frame`` historically rendered each still independently,
    so user passes observed a one-frame batch on every invocation. Keep that
    extension-point contract without giving up sparse batching of the expensive
    scene render. Built-in bloom remains batch-native unless ``batch_native`` is
    off, as it is for stills that share a render batch (see
    ``RenderLoopMixin.get_frames``).
    """
    base = process.func if isinstance(process, partial) else process
    if base is bloom_filter and batch_native:
        return process

    # Keywords post_process_frames passes the pass it wraps (bloom's
    # premultiplied_over; see ``_framewise_of`` below) reach every call.
    def framewise(frames, memory, **kwargs):
        if frames.shape[0] <= 1:
            return process(frames, memory=memory, **kwargs)

        output = None
        expected_shape = None
        expected_dtype = None
        stable_pointers = None
        for index in range(frames.shape[0]):
            try:
                produced = process(frames[index : index + 1], memory=memory, **kwargs)
                if produced.ndim == 0 or produced.shape[0] == 0:
                    raise RuntimeError("A still-frame post-process produced no frames")
                # Match the old one-still path: if a custom pass unexpectedly
                # returns several frames, the still writer keeps the last one.
                produced = produced[-1:]
                if output is None:
                    expected_shape = tuple(produced.shape[1:])
                    expected_dtype = produced.dtype
                    output = memory.get_tensor(
                        (frames.shape[0], *expected_shape), expected_dtype
                    )
                    stable_pointers = memory.get_pointers()
                elif (
                    tuple(produced.shape[1:]) != expected_shape
                    or produced.dtype != expected_dtype
                ):
                    raise AlganConfigurationError(
                        "A post-process used by save_frame(at=[...]) must return "
                        "the same per-frame shape and dtype for every still."
                    )
                output[index : index + 1].copy_(produced)
            finally:
                if stable_pointers is not None:
                    # Reclaim this callback's temporary arena allocations while
                    # retaining the aggregate output for the next pipeline pass.
                    memory.set_pointers(stable_pointers)
        return output

    # So post_process_frames still recognises the pass it adapts.
    framewise._framewise_of = process
    return framewise


def _framewise_post_processes(post_processes, batch_native=True):
    return tuple(
        _framewise_post_process(process, batch_native) for process in post_processes
    )


class RenderLoopMixin:
    """Frame batching, batch preparation, and the render/video-output loop
    (mixed into :class:`~algan.scene.Scene`).
    """

    #: Cached spawn/despawn bounds + interval index for the render's actor
    #: list. Instance attribute on first write; see :meth:`_actor_window_index`.
    _actor_window_cache = None

    #: Whether frames are exported for linear-light ``over`` compositing; see
    #: :meth:`~algan.scene.Scene.set_premultiplied_over`, which owns the
    #: setting. Declared here because the render loop reads it directly and a
    #: Scene need not have run ``__init__`` to be rendered from -- the memory
    #: preflight tests build one with ``Scene.__new__``. Off is the inert
    #: value, so the default is the pre-feature behaviour.
    premultiplied_over = False

    def _batch_prep_context(self):
        """The context a render puts around **all** of its batch preparation.

        Preparing a batch replays recorded animated functions, and replay calls
        a recorded function's *undecorated* body -- so the
        ``record_funcs=False`` wrap that ``animated_function`` normally applies
        is absent. A recorded function whose body calls another animated
        function (``Cylinder.set_start_point`` -> ``_move_between_points`` ->
        ``move_to``) therefore **records a new event every time it is
        replayed**. The render never sees this because its batch loop runs
        inside this context.

        Anything else that calls :meth:`_get_batch_of_primitives` -- every prep
        benchmark and probe in this repo -- must enter it too, or it silently
        grows the timeline on every call: measured at +6 events per call on a
        three-animation scene, and +31..99 per call on the reference scene.
        That corrupts the thing being measured (it re-resolves replay windows
        and invalidates the event-window caches every call, neither of which a
        render does) as well as the Scene.

        Exposed as a method rather than copied into each harness so there is
        one definition to keep in step with the render loop.
        """
        return Off(
            record_attr_modifications=False,
            record_funcs=False,
            priority_level=math.inf,
            animation_manager=self.animation_manager,
        )

    def _actor_window_index(self, actors):
        """Spawn/despawn bounds for ``actors``, indexed for window queries.

        Building this per batch is O(actors) with two ``TimelineEvent`` walks
        each, and there are O(scene) batches -- O(n^2) over a render. Timing is
        fixed for the whole render (which is what let the timestamps be read
        once per batch in the first place), so it can be read once per *render*
        instead, and sorted so a batch touches only the actors near its window.

        Cached against the actor list's identity and length: a render builds
        one list and reuses it for every batch, so a new render is a new list
        and rebuilds. Holding the list in the cache is what makes identity a
        safe key -- it cannot be freed and its id reused.

        Deliberately *not* keyed on the global timing revision, which looks
        like the safer choice and is a trap: it is bumped whenever a timespan
        is configured, including for the transient mobs a render itself
        creates, so it changes during a render and turned this cache into a
        per-batch rebuild -- measured at 257 ms a batch against the 96 ms
        unindexed scan it replaced. Timing is fixed for the runtime of a
        render, which is the same invariant that let the timestamps be read
        once per batch before.

        Actors that never spawned (``spawn < 0``) are dropped here rather than
        tested per batch; ``despawn < 0`` means "never despawns" and becomes
        ``+inf`` so the query is a plain interval overlap.
        """
        key = len(actors)
        cache = self._actor_window_cache
        if cache is not None and cache[0] is actors and cache[1] == key:
            return cache[2]

        rows = []
        for actor in actors:
            spawn = actor.lifespan.start()
            if spawn < 0:
                continue
            despawn = actor.lifespan.end()
            rows.append((actor, spawn, math.inf if despawn < 0 else despawn))
        spawns = torch.tensor([r[1] for r in rows], dtype=torch.float64)
        despawns = torch.tensor([r[2] for r in rows], dtype=torch.float64)
        order = torch.argsort(spawns, stable=True)
        index = (
            [r[0] for r in rows],
            spawns,
            despawns,
            order,
            spawns[order],
            torch.cummax(despawns[order], 0).values,
        )
        self._actor_window_cache = (actors, key, index)
        return index

    @staticmethod
    def _actors_in_window(index, start_time, cutoff):
        """Positions in ``index``'s actor list that overlap ``[start_time, cutoff]``.

        Ascending, because the caller's downstream order (anchor priority) is
        the authored actor order and must not change.

        The bounds mirror the predicate exactly, including its boundaries:
        ``spawn <= cutoff`` upward, and ``despawn >= start_time`` downward via
        the running maximum despawn (non-decreasing, so everything before the
        first position that reaches ``start_time`` has already despawned).
        """
        _, _, despawns, order, sorted_spawns, running_max_despawn = index
        hi = int(
            torch.searchsorted(
                sorted_spawns, torch.tensor(cutoff, dtype=torch.float64), right=True
            )
        )
        # right=False: the predicate keeps despawn == start_time.
        lo = int(
            torch.searchsorted(
                running_max_despawn,
                torch.tensor(start_time, dtype=torch.float64),
                right=False,
            )
        )
        if lo >= hi:
            return order[:0]
        candidates = order[lo:hi]
        return candidates[despawns[candidates] >= start_time].sort().values

    def _still_group_ends(self, actors, frame_indices, start, end):
        """Where a still job's render batches must end: ascending offsets.

        A batch carries every Mob alive anywhere in its window, so stills may
        share one only while that is exactly the set each of them selects on
        its own -- nothing spawns or despawns between the first and the last.
        Then the batch's per-frame cost is each still's own, and the memory
        budget sizes it as it would a video window. Read off the lifespan index
        alone, so nothing is materialized to decide it. A group never spans a
        change of the live set, and holds at most ``_STILL_GROUP_MAX_FRAMES``.
        """
        fps = self.frames_per_second
        index = self._actor_window_index(actors)

        def alive(first, last):
            return len(
                self._actors_in_window(
                    index, frame_indices[first] / fps, (frame_indices[last] + 1) / fps
                )
            )

        ends = []
        first = start
        while first < end:
            own = alive(first, first)
            last = first
            while (
                last + 1 < end
                and last + 1 - first < _STILL_GROUP_MAX_FRAMES
                and alive(last + 1, last + 1) == own
                and alive(first, last + 1) == own
            ):
                last += 1
            ends.append(last + 1)
            first = last + 1
        return ends

    @staticmethod
    def _stills_may_share_batches():
        """Whether this job's stills may share render batches at all.

        Not under the path tracer, whose light-sampling tables and adaptive
        sampling are built over a whole batch, nor under the wavefront memory
        trim, which compacts a batch by its union of visible primitives, nor
        with tonemapping inside the composite, where bloom hands a frame back
        as bytes or as floats depending on whether it glows: no per-frame
        check covers those, so each still keeps a batch of its own.
        """
        rt_settings = SETTINGS.raytracing
        return (
            _STILL_GROUP_MAX_FRAMES > 1
            and int(rt_settings.samples_per_pixel) <= 1
            and not rt_settings.wf_mem_trim
            and rt_settings.is_post_process_tonemap_enabled()
        )

    @staticmethod
    def _frames_deciding_alike(primitives, num_frames):
        """Leading frames of a prepared still batch that decide alike alone.

        The merge notes, per frame, the inputs of every choice it makes for
        the batch as a whole (``_merge_scene``'s ``frame_signature``); frames
        noting what the first one does would each make those choices alike
        alone, so a batch of them renders each exactly as it would alone.
        Anything unverifiable -- no merged scene, a primitive whose projection
        could not choose per frame -- keeps the first frame only.
        """
        first = primitives[0]
        prepared = getattr(first, "_rt_prepared_host_scene", None)
        merged = (
            prepared[0]
            if prepared is not None
            else getattr(first, "_rt_merged_scene", None)
        )
        alike = None if merged is None else merged.get("_frames_alike")
        return 1 if alike is None else min(int(alike), num_frames)

    def _prepare_merged_host_scene(
        self, primitive_batch, *, render_state=None, track_peak=None
    ):
        """Return the cached source-device scene used for upload/preflight."""
        first = primitive_batch[0]
        cached = getattr(first, "_rt_prepared_host_scene", None)
        if cached is not None:
            return cached

        from algan.rendering.raytracing.scene_builder import _merge_scene

        # Read the batch's immutable tensors, never the live light's animation
        # state: this method also runs on the prefetch worker.
        lights = []
        if render_state is not None:
            objects = render_state.get("light_objects")
            if objects is None:
                objects = getattr(self, "light_sources", ())
            active = render_state.get("light_active", [None] * len(objects))
            for light, (origin, color, aux), visible in zip(
                objects, render_state["lights"], active
            ):
                lights.append(
                    SimpleNamespace(
                        origin=origin,
                        light_color=color,
                        _render_aux=aux,
                        _render_active=visible,
                        light_type=getattr(light, "light_type", -1),
                        width=getattr(light, "width", None),
                        height=getattr(light, "height", None),
                    )
                )
        merged_host = _merge_scene(
            primitive_batch,
            light_sources=lights,
            track_peak=track_peak,
            # A still batch is checked frame by frame before it renders.
            frame_signature=bool(getattr(self, "_frame_local_batches", False)),
        )
        env_map = getattr(self, "environment_map", None)
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
        transient out-of-place build scratch (see settings.merge_on_gpu).

        The arena block was reserved from a fraction of the pool at
        ``get_frames`` start; what the device holds outside it is the headroom
        the merge draws from, with a margin left for Taichi's own allocation
        growth during the render that follows.

        The device's pool (:func:`_render_device_pool_bytes`, which is where
        the reasoning about *why* it is a total and not a live measurement
        lives) less the arena's actual size, with a margin. A device genuinely
        short of memory still lands on the merge's own out-of-memory retry,
        which is exact.
        """
        total_bytes = _render_device_pool_bytes(self.memory.data.device)
        return int(max(0, total_bytes - len(self.memory)) * 0.9)

    @staticmethod
    def _may_slice_across_spawns():
        """Whether a fetched batch may be sliced when a mob spawns inside it.

        The prefix then carries actors that have not spawned by the time it
        ends. Their geometry is inert: materialization zeroes a mob's opacity
        outside its lifespan, and ``_pack_frame_visibility`` gives a primitive
        empty per-frame bounds wherever its alpha is below ``min_alpha``, so it
        never enters the BVH on those frames -- nothing un-spawned is drawn.

        It is not byte-identical to re-fetching the prefix, though: carrying
        the extra primitives reorders the merged arrays and the STBVH, so
        shared-edge depth ties and interpolation boundaries land differently.
        The residual is edge-local and a couple of levels deep (see
        ``benchmarks/_prespawn_invisibility_check.py``); both renders are
        correct, and re-fetching costs a full rematerialization per batch.
        Set ``ALGAN_SLICE_ACROSS_SPAWNS=0`` to rematerialize instead.
        """
        return env_flag("ALGAN_SLICE_ACROSS_SPAWNS", True)

    def _fetched_window_has_stable_actor_set(self, actors, start_ind, end_ind):
        """Whether no renderable actor spawns inside a fetched frame window.

        Only consulted under ``ALGAN_SLICE_ACROSS_SPAWNS=0`` (see
        :meth:`_may_slice_across_spawns`), where prefix slicing is restricted
        to windows for which it is exactly equivalent to fetching the prefix.
        Actors that despawn inside the window are safe either way: their
        already-materialized opacity becomes zero.
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
        if not env_flag("ALGAN_REUSE_FETCHED_BATCH", True):
            return False
        if total_frames <= 1 or not primitive_batch:
            return False

        rt_settings = SETTINGS.raytracing

        if not rt_settings.project_on_gpu_active():
            return False
        target_device = render_device()
        for primitive in primitive_batch:
            if getattr(primitive, "_rt_projected", False):
                return False
            if not callable(getattr(primitive, "slice_time_window", None)):
                return False
            if not getattr(primitive, "frame_dependent_source_attrs", ()):
                # The base method is intentionally inert until a primitive
                # declares its time-bearing source tensors.
                return False
            if _primitive_source_device(primitive) == target_device:
                return False
        return True

    def _slice_fetched_batch(
        self, primitive_batch, render_state, duration, total_frames
    ):
        primitives = [
            primitive.slice_time_window(0, duration, total_frames)
            for primitive in primitive_batch
        ]
        return primitives, _slice_render_state(render_state, 0, duration, total_frames)

    def _release_preflight_candidate(self, primitive_batch):
        """Drop projected/merged state belonging to a rejected arena probe.

        Keyed off the ``_rt_`` prefix, which is the same contract
        :meth:`~algan.rendering.primitives.primitive.RenderPrimitive.slice_time_window`
        reads it as: a prefixed attribute is one a projection rebuilds.
        """
        for primitive in primitive_batch:
            for name in tuple(vars(primitive)):
                if name.startswith("_rt_"):
                    setattr(primitive, name, None)
        self.memory.reset()
        release_torch_memory(force_gc=False)

    def _select_largest_fitting_fetched_prefix(
        self,
        primitive_batch,
        render_state,
        total_frames,
        post_processes,
        transparent_background,
        *,
        known_failing=None,
    ):
        """Preflight slices of one fetched source batch and keep the largest.

        The old outer retry loop rematerialized the timeline and rebuilt every
        source primitive for each binary-search candidate.  Here the pristine
        animation-device batch is fetched once.  Each probe is a shallow
        frame-window view uploaded independently for projection, so rejected
        probes cannot mutate it.

        ``known_failing`` is a runtime already preflighted and rejected (the
        prefetch worker's full-window build): the first probe is skipped when
        it would repeat it, and the search continues exactly as if that probe
        had just failed.

        Returns ``(primitives, runtime, render_state)`` or ``None`` when this
        batch cannot safely use the reuse path.
        """
        total_frames = int(total_frames)
        if not self._can_slice_fetched_batch(primitive_batch, total_frames):
            return None

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
            from algan.rendering.raytracing import settings as rt_settings

            estimated_peak = int(
                rt_settings.project_gpu_peak_factor * gpu_project_input_bytes(candidate)
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
                # A single frame is the smallest thing that can be rendered, so
                # only the exact terms may reject it (see the preflight).
                require_estimates_fit=duration > 1,
                num_frames=duration,
            )
            if fits:
                return candidate, candidate_state
            self._release_preflight_candidate(candidate)
            return None

        # Test the largest runtime allowed by projection first.  It is often
        # already the final answer, turning the previous retry cascade into one
        # source fetch and one exact preflight.
        if known_failing is not None and upper >= known_failing:
            upper = int(known_failing)
        else:
            result = probe(upper)
            if result is not None:
                logger.debug(
                    "Arena planner selected %s/%s fetched frames on its first "
                    "exact preflight.",
                    upper,
                    total_frames,
                )
                return result[0], upper, result[1]

        if upper <= 1:
            raise OutOfRenderMemory(_one_frame_does_not_fit_message())

        low = 1
        high = upper - 1
        best = 0
        while low <= high:
            duration = self._next_probe_duration(low, high)
            result = probe(duration)
            if result is None:
                high = duration - 1
                continue

            best = duration
            if True:  # runtime == high:
                logger.debug(
                    "Arena planner selected %s/%s fetched frames without "
                    "rematerializing the batch.",
                    duration,
                    total_frames,
                )
                return result[0], duration, result[1]

            # A larger prefix may fit.  Release this prepared candidate while
            # retaining only its runtime; keeping two merged scenes resident
            # would invalidate the next headroom measurement.
            self._release_preflight_candidate(result[0])
            low = duration + 1

        if best <= 0:
            raise OutOfRenderMemory(_one_frame_does_not_fit_message())

        # The final binary-search step can be a failure immediately above the
        # best fitting runtime, so recreate that winning prefix once.
        result = probe(best)
        if result is None:
            result = probe(best)
            raise OutOfRenderMemory(
                "Render-arena fit was not monotone while selecting a batch."
            )
        logger.debug(
            "Arena planner selected %s/%s fetched frames without "
            "rematerializing the batch.",
            best,
            total_frames,
        )
        return result[0], best, result[1]

    def _begin_batch_cost_measurement(self):
        """Start a fresh cost fit for a newly fetched batch.

        Each term is affine in the frame count, with an intercept fixed by the
        batch's *actor set* -- so a new fetch, which selects a different set,
        starts a new line (see :class:`AffineFrameCost`).
        """
        self._batch_costs = collections.defaultdict(AffineFrameCost)

    def _note_batch_cost(self, term, num_frames, needed_bytes, usable_bytes):
        """Record what one preflight term cost this batch, and its budget."""
        if not num_frames or needed_bytes <= 0:
            return
        # A later reservation can exhaust an initially positive budget. Keep
        # that verdict: discarding it leaves the earlier optimistic capacity
        # live and repeats the same rejected preflight probes every batch.
        self._batch_costs[term].observe(num_frames, needed_bytes, max(0, usable_bytes))

    def _batch_frame_capacity(self):
        """Frames the tightest measured term leaves room for, or ``None``.

        Zero means no window is short enough: the actor set alone overruns some
        term's budget, and only a batch carrying fewer actors will fit.
        """
        capacities = [
            capacity
            for cost in self._batch_costs.values()
            if (capacity := cost.max_frames_for()) is not None
        ]
        return min(capacities) if capacities else None

    def _batch_actor_share(self):
        """Largest share of a term's cost this batch's actor set fixes.

        The part no frame count can shrink, as a fraction of what the term
        costs over the whole fetched window. Near 1.0 the window is the wrong
        lever entirely; well below it the batch is frame-bound and the ordinary
        frame search is the cheaper answer.
        """
        shares = [
            share
            for cost in self._batch_costs.values()
            if (share := cost.actor_share()) is not None
        ]
        return max(shares) if shares else None

    def _describe_batch_costs(self):
        return ", ".join(
            f"{term} {cost.describe()} of {(cost.budget or 0) / 1e6:.1f} MB"
            for term, cost in sorted(self._batch_costs.items())
        )

    def _previous_spawn_boundary(self, actors, start_ind, end_ind):
        """Largest window end that admits strictly fewer actors than ``end_ind``.

        Batch preparation selects the actors that have spawned by the window's
        end, so what decides how much *actor* geometry a batch carries is the
        window's reach over the spawn schedule -- not its length. Shortening a
        window inside a stretch with no spawn in it drops no actor at all,
        which is why a frame-count search cannot relieve an actor-bound batch.

        ``None`` when there is no spawn to retreat behind.
        """
        fps = self.frames_per_second
        start_time = start_ind / fps
        end_time = end_ind / fps
        latest = None
        for actor in actors:
            if not hasattr(actor, "get_render_primitives"):
                continue
            try:
                spawn_time = float(actor.lifespan.start())
            except (AttributeError, TypeError, ValueError):
                continue
            if start_time < spawn_time <= end_time and (
                latest is None or spawn_time > latest
            ):
                latest = spawn_time
        if latest is None:
            return None
        # Selection is ``spawn <= end / fps``, so the window has to end
        # strictly before that spawn for its actor to be left out.
        boundary = math.ceil(latest * fps) - 1
        while boundary > start_ind and boundary / fps >= latest:
            boundary -= 1
        return boundary if boundary > start_ind else None

    def _next_probe_duration(self, low, high, use_hint=True):
        """Next frame count to try within ``[low, high]``.

        The last preflight measured what its batch actually consumed, so it can
        say roughly how many frames there is room for. Aiming at that lands on
        the answer in one probe where halving takes several. It may only pull
        the target *below* the halving point, never above it, so the search
        keeps its guaranteed halving progress when the estimate is useless (on
        a batch that overflowed it overshoots, by the frame-independent part of
        whichever term rejected it).

        ``use_hint=False`` for callers whose next window will be materialized
        afresh after a *sliced* measurement: the slice carried the whole
        fetched window's actor set, so its estimate describes a batch the
        caller is no longer proposing to build, and following it would repeat
        the degenerate window it is retreating from.
        """
        midpoint = (low + high + 1) // 2
        hint = self._batch_frame_capacity() if use_hint else None
        if hint is None or hint >= midpoint:
            return midpoint
        return max(low, hint)

    def _prepared_batch_fits_render_arena(
        self,
        primitive_batch,
        render_state,
        post_processes,
        transparent_background,
        *,
        require_estimates_fit=True,
        num_frames=None,
    ):
        """Whether the prepared scene and at least one frame fit exactly.

        The scene upload grows the arena's reverse pointer; camera/lights,
        output, wavefront state and post-processing grow the forward pointer.
        Preflighting both sides lets the outer batching loop binary-search a
        maximum fitting prepared runtime without rendering speculative frames.

        Two of the terms are exact (the scene's arena bytes, and an actual
        out-of-memory raised by the projection or the merge) and the rest are
        estimates -- the modelled per-frame cost and the transient peaks of the
        out-of-arena builds. ``require_estimates_fit=False`` drops the estimated
        terms, leaving only the exact ones. Callers pass it for a single-frame
        batch: there is no smaller window to retreat to, so a rejection there
        aborts the render outright, and a *guess* must never be what does that.
        The render's own out-of-memory retry remains the backstop.

        ``num_frames`` is the batch's frame count. Given it, each term's cost
        is recorded against it (see ``_note_batch_cost``), which is what lets
        the caller separate what the frame count buys from what the batch's
        actor set costs regardless -- and so aim its next window, or decide
        that no window is short enough and fewer actors are needed.
        """
        self._last_arena_preflight = None
        if not getattr(self.memory, "managed", False):
            return True

        # A batch the prefetch worker prepared under prefetch-gpu-prep arrives
        # already projected and merged. Its builds ran beside a live render,
        # so: their transient peaks are unmeasurable (any reading includes the
        # render's allocations) and are deliberately not observed -- the
        # predictors keep what they learned from the un-overlapped batches --
        # and their proactive estimates are moot, since the builds already
        # ran (bounded there against derated headroom; see
        # _prepare_batch_on_worker). Everything below them -- the exact arena
        # bytes, the frame-cost model, the verdict -- is unchanged.
        overlapped = bool(
            getattr(primitive_batch[0], "_rt_prep_overlapped", False)
        ) and all(
            getattr(primitive, "_rt_projected", False) for primitive in primitive_batch
        )

        rt_settings = SETTINGS.raytracing
        from algan.rendering.raytracing.scene_builder import (
            get_merged_scene_arena_nbytes,
            gpu_merge_input_bytes,
            gpu_project_input_bytes,
        )
        from algan.rendering.raytracing.settings import (
            hdr_frame_dtype,
            is_post_process_tonemap_enabled,
        )
        from algan.rendering.raytracing.tracer import (
            effective_anti_alias_level,
        )

        # Prefetch defers projection to this render thread when it runs on the
        # device (project-on-gpu); otherwise it merely finishes any CPU
        # projection the worker didn't complete. Its transient device scratch
        # (source geometry + shading workspace + packed _rt_* output) lives in
        # the pool's non-arena headroom, so -- like the merge below -- estimate
        # its peak from the source-geometry bytes and shrink the window before
        # attempting it, with the OOM handler as the exact fallback.
        project_inputs = 0
        project_token = None
        overlap_inputs = (
            getattr(primitive_batch[0], "_rt_overlap_inputs", None)
            if overlapped
            else None
        )
        if overlap_inputs is not None:
            # The builds are done, so their estimates are moot -- but the
            # window-sizing cost terms are not. Noting them exactly as the
            # un-overlapped preflight does bounds the next window's capacity
            # by the same terms in both arms.
            headroom = self._gpu_merge_headroom_bytes()
            self._note_batch_cost(
                "projection",
                num_frames,
                overlap_inputs[0],
                self._project_peak_ratio.max_inputs_for(headroom),
            )
            self._note_batch_cost(
                "merge",
                num_frames,
                overlap_inputs[1],
                self._merge_peak_ratio.max_inputs_for(headroom),
            )
        # The worker may also have projected a batch without merging it (the
        # merge estimate exceeded its derated headroom, or the merge ran out of
        # memory). That batch arrives unstamped, so it is not ``overlapped``,
        # but only its unprojected remainder -- possibly nothing -- is left to
        # size and measure here.
        projection_pending = not all(
            getattr(primitive, "_rt_projected", False) for primitive in primitive_batch
        )
        if projection_pending and rt_settings.project_on_gpu_active():
            # Read now: projecting releases the source geometry this sums.
            project_inputs = gpu_project_input_bytes(primitive_batch)
            estimated_project_peak = self._project_peak_ratio.predict(project_inputs)
            headroom = self._gpu_merge_headroom_bytes()
            # Scale the window by the *inputs* the window controls, not by the
            # predicted peak: the build's fixed part does not shrink with the
            # frame count, so dividing it in would pin the window where it is.
            self._note_batch_cost(
                "projection",
                num_frames,
                project_inputs,
                self._project_peak_ratio.max_inputs_for(headroom),
            )
            if require_estimates_fit and estimated_project_peak > headroom:
                logger.debug(
                    "GPU projection peak estimate %.1f MB exceeds pool "
                    "headroom %.1f MB [%s]; shrinking frame window.",
                    estimated_project_peak / 1e6,
                    headroom / 1e6,
                    self._project_peak_ratio.describe(),
                )
                return False
            project_token = begin_cuda_peak(self.memory.data.device)
        if not overlapped:
            try:
                self._prewarm_render_batch(primitive_batch, render_state)
                if project_token is not None:
                    # Projection ran on this thread (project-on-gpu defers it
                    # here precisely so no concurrent render pollutes the
                    # counter), so the peak it just reached bounds the next
                    # batch's estimate.
                    self._project_peak_ratio.observe(
                        project_inputs, end_cuda_peak(project_token)
                    )
                    project_token = None
            except (InsufficientMemoryException, RuntimeError) as exc:
                # Device projection overran the pool headroom. Drop partial state
                # and report not-fitting so the caller shrinks the frame window.
                # (Also treat a Taichi-allocator OOM as such; re-raise real errors.)
                if not isinstance(exc, InsufficientMemoryException) and not is_cuda_oom(
                    exc
                ):
                    raise
                primitive_batch[0]._rt_merged_scene = None
                primitive_batch[0]._rt_prepared_host_scene = None
                release_torch_memory(force_gc=False)
                logger.debug("Arena preflight: projection ran out of memory (%r).", exc)
                return False
        if not all(
            getattr(primitive, "_rt_projected", False) for primitive in primitive_batch
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
        merge_inputs = 0
        if gpu_merge and not overlapped:
            # Read now: merging nulls the packed _rt_* arrays this sums.
            merge_inputs = gpu_merge_input_bytes(primitive_batch)
            estimated_merge_peak = self._merge_peak_ratio.predict(merge_inputs)
            headroom = self._gpu_merge_headroom_bytes()
            # See the projection term: scale by inputs, not by predicted peak.
            self._note_batch_cost(
                "merge",
                num_frames,
                merge_inputs,
                self._merge_peak_ratio.max_inputs_for(headroom),
            )
            if require_estimates_fit and estimated_merge_peak > headroom:
                logger.debug(
                    "GPU merge peak estimate %.1f MB exceeds pool headroom "
                    "%.1f MB [%s]; shrinking frame window.",
                    estimated_merge_peak / 1e6,
                    headroom / 1e6,
                    self._merge_peak_ratio.describe(),
                )
                return False
        try:
            merged_host, env_map = self._prepare_merged_host_scene(
                primitive_batch, render_state=render_state
            )
        except (InsufficientMemoryException, RuntimeError) as exc:
            # The device build overran the pool headroom. Drop any partial
            # merge state and report the batch as not fitting so the caller
            # shrinks the frame window and retries. (Also treat a Taichi-
            # allocator OOM as such; re-raise real errors.)
            if not isinstance(exc, InsufficientMemoryException) and not is_cuda_oom(
                exc
            ):
                raise
            primitive_batch[0]._rt_merged_scene = None
            primitive_batch[0]._rt_prepared_host_scene = None
            release_torch_memory(force_gc=False)
            logger.debug("Arena preflight: scene merge ran out of memory (%r).", exc)
            return False
        scene_bytes = get_merged_scene_arena_nbytes(
            merged_host, self.memory, persist=True
        )
        if gpu_merge and not overlapped:
            # The build just ran and reported its own peak, so the multiplier
            # that bounds the *next* one is measured rather than guessed.
            measured = int(merged_host.get("_gpu_merge_peak_bytes", -1))
            if measured >= 0:
                self._merge_peak_ratio.observe(merge_inputs, measured)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "GPU merge: est peak %.1f MB (measured %s MB) [%s], "
                    "headroom %.1f MB, arena scene %.1f MB.",
                    estimated_merge_peak / 1e6,
                    f"{measured / 1e6:.1f}" if measured >= 0 else "n/a",
                    self._merge_peak_ratio.describe(),
                    self._gpu_merge_headroom_bytes() / 1e6,
                    scene_bytes / 1e6,
                )
        elif gpu_merge and overlapped and logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Overlapped batch: merge prepared on the worker, arena scene "
                "%.1f MB, headroom %.1f MB (peak not observed).",
                scene_bytes / 1e6,
                self._gpu_merge_headroom_bytes() / 1e6,
            )
        bytes_remaining = self.memory.get_num_bytes_remaining()
        margin = int(getattr(self, "_arena_unmodeled_bytes", 0))
        self._note_batch_cost(
            "arena", num_frames, scene_bytes, bytes_remaining - margin
        )
        if scene_bytes > bytes_remaining:
            logger.debug(
                "Arena preflight: scene %.1f MB exceeds the %.1f MB remaining "
                "in the %.1f MB arena (tris=%s circuits=%s).",
                scene_bytes / 1e6,
                bytes_remaining / 1e6,
                len(self.memory) / 1e6,
                merged_host.get("num_triangles", 0),
                merged_host.get("num_circuits", 0),
            )
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

        aa = effective_anti_alias_level(
            merged_host,
            self.video_settings.supersampling,
            light_sources=lights,
            environment_map=env_map,
            near_clip=float(getattr(self.camera, "near", 0.0) or 0.0),
            far_clip=float(getattr(self.camera, "far", 0.0) or 0.0),
            transparent_background=transparent_background,
        )
        render_height = self.num_pixels_screen_height * aa
        render_width = self.num_pixels_screen_width * aa
        render_channels = 5 if transparent_background else 4
        # Linear-HDR buffer: the composite writes linear HDR here and post
        # tonemaps last, so bloom runs on unclamped HDR. dtype from
        # hdr_frame_dtype() -- float32 by default, opt-in float16 (RGBA16F,
        # half the memory) on GPUs with fast FP16.
        frame_dtype = (
            hdr_frame_dtype() if is_post_process_tonemap_enabled() else torch.uint8
        )
        samples = max(1, int(rt_settings.samples_per_pixel))
        # What one frame costs on top of the scene is *measured*, not modelled.
        # Until this job has rendered a chunk there is nothing to measure, so
        # the preflight arbitrates on the scene alone -- which is the exact
        # part, and the part that decides whether a batch is renderable at all.
        # An optimistic first batch is corrected by the render's own retry.
        signature = chunk_signature(
            width=render_width,
            height=render_height,
            channels=render_channels,
            dtype=frame_dtype,
            samples_per_pixel=samples,
            num_triangles=merged_host.get("num_triangles", 0),
            num_circuits=merged_host.get("num_circuits", 0),
        )
        modelled_forward = self._chunk_memory_model.predict_preflight(signature, 1)
        forward_bytes = modelled_forward or 0
        need_bytes = scene_bytes + forward_bytes
        self._last_arena_preflight = (need_bytes, bytes_remaining)
        # With no measurement yet, the frame term above is *zero*, so a first
        # batch is sized on its geometry alone. That is affordable while the
        # scene is a small part of the arena -- whatever the frame turns out to
        # cost, there is room for it -- and reckless once the scene is most of
        # it, because what remains has to cover every frame in the window and
        # the scene may densify later in that window. So while the frame cost
        # is unmeasured and the scene is over ``_UNMEASURED_SCENE_SHARE`` of
        # what is left, hold the window down to a probe length: those frames
        # measure the line, and every batch afterwards is sized against a
        # measurement rather than against a zero. Deliberately gated on the
        # share, so an arena with room to spare keeps its full first window.
        if (
            modelled_forward is None
            and require_estimates_fit
            and num_frames > _UNMEASURED_PROBE_FRAMES
            and scene_bytes > _UNMEASURED_SCENE_SHARE * max(1, bytes_remaining)
        ):
            logger.debug(
                "Arena preflight: %s-frame window with an unmeasured frame "
                "cost and a scene of %.1f MB in %.1f MB remaining; probing at "
                "%s frames first.",
                num_frames,
                scene_bytes / 1e6,
                bytes_remaining / 1e6,
                _UNMEASURED_PROBE_FRAMES,
            )
            return False
        # Refine the arena term now that the frame cost is known: it is the
        # scene, not the frame buffers, that the frame count scales.
        self._note_batch_cost(
            "arena", num_frames, scene_bytes, bytes_remaining - margin - forward_bytes
        )
        fits = need_bytes <= bytes_remaining - margin
        if not (fits or require_estimates_fit):
            # Single-frame batch: the modelled frame cost is the only thing
            # rejecting it, and there is no smaller window left to retreat to.
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Arena preflight: modelled frame cost %.1f MB does not fit "
                    "alongside the %.1f MB scene in %.1f MB [%s]; rendering the "
                    "frame anyway rather than failing on an estimate.",
                    forward_bytes / 1e6,
                    scene_bytes / 1e6,
                    bytes_remaining / 1e6,
                    self._chunk_memory_model.describe(signature),
                )
            return True
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Arena preflight %s: scene %.1f + frame %.1f = %.1f MB vs "
                "%.1f MB remaining - %.1f MB margin (aa=%s, %sx%s, tris=%s, "
                "circuits=%s).",
                "fits" if fits else "rejects",
                scene_bytes / 1e6,
                forward_bytes / 1e6,
                need_bytes / 1e6,
                bytes_remaining / 1e6,
                margin / 1e6,
                aa,
                render_width,
                render_height,
                merged_host.get("num_triangles", 0),
                merged_host.get("num_circuits", 0),
            )
        return fits

    def _observed_chunk_frames(self, duration):
        """Frame count to credit a rendered chunk's measured arena peak to.

        Normally the chunk that was planned. When the renderer had to
        sub-divide it -- an out-of-memory split, or the Monte Carlo path budget
        -- the high-water mark belongs to the largest sub-window it actually
        launched, and crediting it to the planned count reads the per-frame
        cost as roughly half of what it is. The model then plans the same
        over-large chunk again and splits again, for chunk after chunk, with
        nothing in the loop ever learning: the split is recovered inside the
        tracer, so the outer render-failure path never sees it either.
        """
        launched = getattr(self.memory, "last_launch_frames", None)
        if not launched or launched >= duration:
            return duration
        logger.debug(
            "chunk of %d frames was rendered in sub-windows of at most %d; "
            "attributing its arena peak to %d frames",
            duration,
            launched,
            launched,
        )
        return int(launched)

    def _note_render_arena_underestimate(self):
        """Grow the preflight safety margin after a batch that passed the
        arena preflight still failed to render: some allocation is not being
        modeled, so future preflights must leave real slack. The failed
        batch's own (need, remaining) pair makes the margin large enough to
        reject at least that exact configuration; repeated failures grow it
        geometrically.
        """
        prev = int(getattr(self, "_arena_unmodeled_bytes", 0))
        last = getattr(self, "_last_arena_preflight", None)
        observed = 0
        if last is not None:
            need, remaining = last
            observed = max(0, int(remaining) - int(need)) + 1
        self._arena_unmodeled_bytes = max(prev * 2, observed, 8 << 20)
        # A failure after the margin was last lowered says the lowering was
        # premature, so the next one waits twice as long: the margin settles
        # where it holds instead of re-failing on a fixed period.
        if getattr(self, "_arena_margin_decayed", False):
            self._arena_margin_decay_after = min(
                2
                * getattr(
                    self, "_arena_margin_decay_after", _ARENA_MARGIN_DECAY_BATCHES
                ),
                _ARENA_MARGIN_DECAY_MAX_BATCHES,
            )
        self._arena_margin_decayed = False
        self._arena_margin_clean_batches = 0
        # Only a failure another holder of the device can account for may be
        # decayed away (see _note_render_arena_success). One the render caused
        # itself -- an allocation outside the arena that grows with the window
        # -- keeps its margin: growing the windows back only reproduced it, and
        # on Windows the driver pages instead of failing, so nothing re-raises
        # the margin (measured 195 s against 211-255 s on HD scene 12 with the
        # per-frame background copies restored).
        now = _other_device_bytes(render_device())
        start = getattr(self, "_arena_other_bytes_at_start", None)
        external = False
        other_growth = 0
        if now is not None and start is not None:
            other_growth = now[0] - start[0]
            external = other_growth > max(_OTHER_HOLDER_MIN_BYTES, now[1] // 12)
        if not external:
            self._arena_margin_persistent = True
        if getattr(self, "_arena_margin_persistent", False):
            logger.warning(
                "Arena preflight under-modeled the render; raising its safety "
                "margin to %.1f MB for the rest of this job.",
                self._arena_unmodeled_bytes / 1e6,
            )
        else:
            logger.warning(
                "Arena preflight under-modeled the render while another holder "
                "took %.1f MB more of the device; raising its safety margin to "
                "%.1f MB until batches render cleanly again.",
                other_growth / 1e6,
                self._arena_unmodeled_bytes / 1e6,
            )

    def _note_render_arena_success(self):
        """Let the preflight safety margin shrink back after clean batches.

        The margin (:meth:`_note_render_arena_underestimate`) is learned from a
        single failure. When another process took device memory the render had
        counted on, that failure need not recur -- the other holder may let go
        -- and a margin that never shrinks turns it into a job of tiny batches.
        Such margins, and only those (a failure the render caused itself keeps
        its margin for the job), shrink back: after
        ``_arena_margin_decay_after`` batches in a row render without failing,
        the margin halves (and drops to zero below 8 MB). A failure after a
        decay doubles that wait (up to ``_ARENA_MARGIN_DECAY_MAX_BATCHES``), so
        a margin the scene really needs is re-learned at most a few times.
        ``ALGAN_OPT_DISABLE=margindecay`` keeps the old never-shrinking margin.
        """
        margin = int(getattr(self, "_arena_unmodeled_bytes", 0))
        if margin <= 0 or getattr(self, "_arena_margin_persistent", False):
            return
        from algan.animation_timeline.timeline import _opt_disabled

        if _opt_disabled("margindecay"):
            return
        clean = getattr(self, "_arena_margin_clean_batches", 0) + 1
        wait = getattr(self, "_arena_margin_decay_after", _ARENA_MARGIN_DECAY_BATCHES)
        if clean < wait:
            self._arena_margin_clean_batches = clean
            return
        self._arena_unmodeled_bytes = margin // 2 if margin // 2 >= (8 << 20) else 0
        self._arena_margin_clean_batches = 0
        self._arena_margin_decayed = True
        logger.log(
            PERF,
            "Arena preflight safety margin lowered to %.1f MB after %d clean batches.",
            self._arena_unmodeled_bytes / 1e6,
            wait,
        )

    def _reset_render_arena_after_failure(self):
        """Release every allocation owned by a failed render attempt.

        OOM exceptions retain their traceback until the ``except`` block has
        exited. Those frames can own arena views and ordinary CUDA tensors, so
        retry cleanup must run afterwards and must collect reference cycles.
        Resetting both arena pointers releases its forward and persistent
        allocations without reallocating the backing tensor.
        """
        self.memory.reset()
        release_torch_memory(force_gc=True)

    def _render_primitive_batch(
        self,
        primitive_batch,
        start_ind,
        end_ind,
        save_image=False,
        post_processes=(),
        transparent_background=False,
        background=None,
        render_state=None,
        aux_sink=None,
    ):
        """Render one prepared primitive batch for a frame interval.

        With ``aux_sink`` the renderer also traces the aux passes, and each
        chunk's passes go to ``aux_sink`` just before its frames are yielded.
        """
        with torch.no_grad():
            camera = self.camera
            if render_state is None:
                render_state = self._materialize_render_state(start_ind, end_ind)
            camera.ray_origin = render_state["ray_origin"]
            camera.screen_point = render_state["screen_point"]
            camera.screen_basis = render_state["screen_basis"]
            projection_aa, projection_analytic = _projection_anti_alias_level(
                self, primitive_batch
            )
            camera.screen_width = self.num_pixels_screen_width * projection_aa
            camera.screen_height = self.num_pixels_screen_height * projection_aa
            camera.output_screen_width = self.num_pixels_screen_width
            camera.output_screen_height = self.num_pixels_screen_height
            camera.analytic_raster = projection_analytic
            # The snapshot carries only lights whose lifespan intersects this
            # window (see _materialize_render_state); everything downstream of
            # here -- vertex shading, the packed light rows, the route
            # decision -- must consume the same filtered list, or the zipped
            # snapshot tensors would land on the wrong light objects.
            render_lights = render_state.get("light_objects")
            if render_lights is None:
                render_lights = self.light_sources
            light_active = render_state.get("light_active", [None] * len(render_lights))
            for light, (origin, light_color, aux), visible in zip(
                render_lights, render_state["lights"], light_active
            ):
                light.origin = origin
                light.light_color = light_color
                light._render_aux = aux
                # The per-frame lifespan mask travels with the other snapshot
                # tensors: build_area_light_quads reads it off whichever object
                # reaches it, and the tracer's own _merge_scene fallback is
                # handed these live lights rather than the prepared shims. A
                # missing attribute there would silently mean "always active"
                # and would render a despawned panel as opaque black geometry.
                light._render_active = visible

            self.memory.scene = self
            original_pointers = self.memory.get_pointers()
            # Projection builds out of place in pool headroom, so the arena
            # recorder cannot see it; measure it from torch's counters instead
            # so project_gpu_peak_factor can be a measurement rather than a
            # guess. The input size has to be read *now*: projecting releases
            # the source geometry it is computed from. Skipped entirely unless
            # a calibration run is recording.
            _measuring = auto_record_enabled()
            _project_token = None
            _project_inputs = 0
            if _measuring:
                try:
                    from algan.rendering.raytracing.scene_builder import (
                        gpu_project_input_bytes as _project_input_bytes,
                    )

                    # Only the primitives this loop will actually project: one
                    # already projected on the prefetch worker has had its
                    # source geometry released, and reading it back raises.
                    _pending = [
                        primitive
                        for primitive in primitive_batch
                        if not getattr(primitive, "_rt_projected", False)
                    ]
                    _project_inputs = _project_input_bytes(_pending) if _pending else 0
                    _project_token = begin_cuda_peak(self.memory.data.device)
                except Exception:  # noqa: BLE001
                    # Calibration instrumentation must never be able to break
                    # a render; a missed sample only costs corpus coverage.
                    _measuring = False
                    _project_token = None
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
                    primitive, fallback=camera.ray_origin.device
                )
                scratch = ManualMemory(0, device=source_device, managed=False)
                try:
                    primitive.memory = scratch
                    primitive.project_to_screen(camera, render_lights)
                finally:
                    primitive.memory = original_memory
            if _measuring:
                with contextlib.suppress(Exception):
                    note_nonarena_peak(
                        "project", _project_inputs, end_cuda_peak(_project_token)
                    )

            # Reclaim animation-phase residuals before render batching.
            release_torch_memory(force_gc=False)

            # Projection ran on the source (CPU) device; the merge + STBVH
            # build ran on the CPU or (default) the render device. Upload each
            # unique finished storage directly into the persistent end of
            # ManualMemory before sizing frame chunks, so the exact scene
            # footprint is subtracted from the arena automatically. On the CPU
            # merge no CUDA-side cat/BVH scratch ever sat beside the pool; on
            # the GPU merge that scratch has already been freed (its transient
            # peak was bounded against the pool headroom in the preflight).
            rt_settings = SETTINGS.raytracing
            from algan.rendering.raytracing.scene_builder import (
                copy_merged_scene_to_arena,
            )

            merged_host, env_map = self._prepare_merged_host_scene(
                primitive_batch, render_state=render_state
            )
            device_scene = copy_merged_scene_to_arena(
                merged_host, self.memory, persist=True
            )
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
                release_torch_memory(force_gc=False)

            render_pointers = self.memory.get_pointers()
            from algan.rendering.raytracing.settings import (
                hdr_frame_dtype,
                is_post_process_tonemap_enabled,
            )
            from algan.rendering.raytracing.tracer import (
                effective_anti_alias_level,
            )

            aa = effective_anti_alias_level(
                merged_host,
                self.video_settings.supersampling,
                light_sources=render_lights,
                environment_map=env_map,
                near_clip=float(getattr(camera, "near", 0.0) or 0.0),
                far_clip=float(getattr(camera, "far", 0.0) or 0.0),
                transparent_background=transparent_background,
            )
            render_height = self.num_pixels_screen_height * aa
            render_width = self.num_pixels_screen_width * aa
            render_channels = 5 if transparent_background else 4
            # Linear-HDR buffer (hdr_frame_dtype: f32 default, opt-in f16) --
            # see the matching note in the deterministic render path.
            frame_dtype = (
                hdr_frame_dtype() if is_post_process_tonemap_enabled() else torch.uint8
            )
            samples = max(1, int(rt_settings.samples_per_pixel))
            # Batches whose peak lies on the same line share a fit. Nothing
            # here describes *what* gets allocated -- only what would put a
            # batch on a different line.
            signature = chunk_signature(
                width=render_width,
                height=render_height,
                channels=render_channels,
                dtype=frame_dtype,
                samples_per_pixel=samples,
                num_triangles=merged_host.get("num_triangles", 0),
                num_circuits=merged_host.get("num_circuits", 0),
            )
            model = self._chunk_memory_model
            bytes_remaining = self.memory.get_num_bytes_remaining()

            current_ind = start_ind
            # How far this batch has actually handed frames to the caller.
            # A chunk that fails after earlier chunks were emitted leaves the
            # render resumable from here rather than having to be abandoned;
            # see the render-failure handler in ``get_frames``.
            self._last_emitted_time_ind = start_ind
            while True:
                if getattr(self.memory, "managed", False):
                    duration = model.plan(
                        signature, end_ind - current_ind, bytes_remaining
                    )
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            "chunk %d frames from model [%s], %.1f MB free",
                            duration,
                            model.describe(signature),
                            bytes_remaining / 1e6,
                        )
                else:
                    # Unmanaged mode deliberately uses PyTorch's ordinary
                    # allocator.  There is no finite arena to size against,
                    # and arbitrary custom post-process callables remain valid
                    # because their allocations do not need an arena planner.
                    duration = end_ind - current_ind
                new_ind = current_ind + duration

                logger.debug(f"rendering batch with runtime {duration}")

                background_source = (
                    self.background_frame if background is None else background
                )
                bgf = _prepare_background_for_chunk(
                    background_source,
                    screen_width=self.num_pixels_screen_width,
                    screen_height=self.num_pixels_screen_height,
                    anti_alias_level=aa,
                    current_ind=current_ind,
                    new_ind=new_ind,
                    frames_per_second=(
                        self.frames_per_second if callable(background_source) else 1
                    ),
                    device=render_device(),
                    frame_indices=(
                        render_state["frame_indices"][
                            current_ind - start_ind : new_ind - start_ind
                        ]
                        if "frame_indices" in render_state
                        else None
                    ),
                )
                # Pressure-gated gc (like every other steady-state call site):
                # a forced full collection here cost ~150 ms per frame window
                # and reference counting already frees the previous window's
                # buffers; release_torch_memory still collects cycles when the device
                # is genuinely near capacity.
                release_torch_memory(force_gc=False)
                # Baseline the high-water mark at this chunk's starting level,
                # so what it reaches afterwards is this chunk's own footprint
                # rather than the whole job's.
                chunk_base = render_pointers[0] + len(self.memory) - render_pointers[1]
                self.memory.max_pointer = chunk_base
                self.memory.last_launch_frames = None
                self.memory.last_chunk_capacity_limited = False
                if aux_sink is not None:
                    # Never hand on a stale batch's passes.
                    self.__dict__.pop("_pending_aux_passes", None)
                frames = primitive_batch[0].render(
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
                    anti_alias_level=self.video_settings.supersampling,
                    light_sources=render_lights,
                    memory=self.memory,
                    post_processes=post_processes,
                    # Per-frame thin lens, passed explicitly rather than hung on
                    # the live camera: only the path tracer reads it, and
                    # projection never needs it.
                    camera_lens=render_state.get("camera_lens"),
                    **({"aux_passes": True} if aux_sink is not None else {}),
                )
                if aux_sink is not None:
                    # The renderer's side channel (render_batch_raytraced);
                    # the frames themselves stay a plain tensor.
                    aux = self.__dict__.pop("_pending_aux_passes", None)
                    if aux is None:
                        raise RuntimeError(
                            "aux passes were requested but the render kernel "
                            "produced none"
                        )
                    if int(aux["depth"].shape[0]) != len(frames):
                        raise RuntimeError(
                            f"the render kernel produced aux passes for "
                            f"{int(aux['depth'].shape[0])} frames alongside "
                            f"{len(frames)} rendered frames"
                        )
                    aux_sink(aux)
                yield frames
                # Control is back, so the caller has taken every frame up to
                # ``new_ind``: those are final and a later chunk's failure
                # must not re-render them.
                self._last_emitted_time_ind = new_ind
                # The chunk rendered, so its peak is now known exactly. This is
                # the whole memory model: no allocation is described anywhere,
                # only what the arena actually reached.
                if getattr(self.memory, "managed", False):
                    model.observe(
                        signature,
                        self._observed_chunk_frames(duration),
                        max(0, self.memory.max_pointer - chunk_base),
                        capacity_limited=self.memory.last_chunk_capacity_limited,
                    )

                # Some batch-wide data is allocated at the arena's persistent
                # end from inside the first chunk and read by every later one:
                # the tracer's raster projection / bounds tables, and the trees
                # of a deferred STBVH build re-homed into the arena the merged
                # scene lives in. Hold the reverse pointer open exactly that
                # far -- those and nothing else -- so the per-chunk rewind below
                # stops reclaiming what the next chunk still points into.
                # Published by the tracer rather than read off the arena, so an
                # unrelated persistent allocation inside a chunk can never be
                # retained by accident.
                from algan.rendering.raytracing.tracer import (
                    ARENA_RETAINED_REVERSE_POINTER,
                )

                retained = device_scene.get(ARENA_RETAINED_REVERSE_POINTER)
                if retained is not None and retained < render_pointers[1]:
                    render_pointers = (render_pointers[0], retained)
                    bytes_remaining = retained - render_pointers[0]

                self.memory.set_pointers(render_pointers)
                current_ind = new_ind
                if current_ind >= end_ind:
                    break

            self.memory.set_pointers(original_pointers)
            self.memory.max_pointer = self.memory.current_pointer = (
                len(self.memory) - self.memory.current_reverse_pointer
            )
            # Camera/screen/light state is no longer reset here: batch prep
            # (_get_batch_of_primitives) resets and re-materializes it at the
            # start of each batch, and may already be running on a worker
            # thread for the next batch while this render executes.

    def _render_background_batch(
        self,
        start_ind,
        end_ind,
        post_processes=(),
        transparent_background=False,
        background=None,
        frame_indices=None,
        aux_sink=None,
    ):
        """Yield background-only frame batches for ``[start_ind, end_ind)``.

        Used for frame windows with no renderable primitives (an empty scene,
        or a stretch where nothing is spawned) so the output video still
        covers the window. Mirrors ``_render_primitive_batch``'s frame
        finalization -- background prefill into the render arena followed by
        the standard post-processing chain -- with the ray tracer itself
        skipped, so these frames match what the tracer produces when nothing
        is visible. With ``aux_sink``, each batch's aux passes (all misses:
        nothing is there to record) go to it just before the batch is yielded.
        """
        from algan.rendering.post_processing.post_process import (
            post_process_frames,
        )
        from algan.rendering.raytracing.scene_builder import (
            _downsample_background,
            _prefill_background,
        )
        from algan.rendering.raytracing.settings import (
            hdr_frame_dtype,
            is_post_process_tonemap_enabled,
        )

        aa = max(1, int(self.video_settings.supersampling))
        # Mirror the tracer's anti-aliasing strategy (render_batch_raytraced):
        # default super-sampled buffer averaged down in post-processing;
        # ALGAN_INPLACE_AA keeps the buffer at output resolution.
        inplace_aa = env_flag("ALGAN_INPLACE_AA", False)
        if inplace_aa:
            width = self.num_pixels_screen_width
            height = self.num_pixels_screen_height
            post_aa = 1
        else:
            width = self.num_pixels_screen_width * aa
            height = self.num_pixels_screen_height * aa
            post_aa = aa
        channels = 5 if transparent_background else 4
        # Linear-HDR buffer for background-only windows (hdr_frame_dtype);
        # matches the primitive path so the two never disagree.
        frame_dtype = (
            hdr_frame_dtype() if is_post_process_tonemap_enabled() else torch.uint8
        )
        device = self.memory.data.device
        background_source = self.background_frame if background is None else background

        original_pointers = self.memory.get_pointers()
        bytes_remaining = self.memory.get_num_bytes_remaining()
        # Background-only windows are sized by the same measured model as the
        # primitive path. Their own signature: no geometry, so their line is
        # much shallower and must not be mixed with a rendered batch's.
        signature = chunk_signature(
            width=width,
            height=height,
            channels=channels,
            dtype=frame_dtype,
            samples_per_pixel=1,
            num_triangles=0,
            num_circuits=0,
        )
        model = self._chunk_memory_model

        current_ind = start_ind
        while current_ind < end_ind:
            if getattr(self.memory, "managed", False):
                duration = model.plan(signature, end_ind - current_ind, bytes_remaining)
            else:
                # Unmanaged mode uses PyTorch's ordinary allocator; there is
                # no finite arena to size against (see _render_primitive_batch).
                duration = end_ind - current_ind
            new_ind = current_ind + duration

            bgf = _prepare_background_for_chunk(
                background_source,
                screen_width=self.num_pixels_screen_width,
                screen_height=self.num_pixels_screen_height,
                anti_alias_level=aa,
                current_ind=current_ind,
                new_ind=new_ind,
                frames_per_second=(
                    self.frames_per_second if callable(background_source) else 1
                ),
                device=render_device(),
                frame_indices=(
                    frame_indices[current_ind - start_ind : new_ind - start_ind]
                    if frame_indices is not None
                    else None
                ),
            )
            # In-place AA samples the background once per output pixel, so a
            # super-sampled image background must be averaged down first
            # (solid colors are resolution-free).
            if aa > 1 and inplace_aa:
                bgf = _downsample_background(
                    bgf,
                    aa,
                    duration,
                    self.num_pixels_screen_height,
                    self.num_pixels_screen_width,
                )
            # Pressure-gated gc: the background-only path allocates almost
            # nothing per window, so a forced full collection here was pure
            # fixed cost (see the _render_primitive_batch call site).
            release_torch_memory(force_gc=False)
            chunk_base = original_pointers[0] + len(self.memory) - original_pointers[1]
            self.memory.max_pointer = chunk_base
            out = self.memory.get_tensor(
                (duration, width * height, channels), frame_dtype
            )
            _prefill_background(out, bgf, 0, device, background_frames=duration)
            frames = post_process_frames(
                self.memory,
                out.view(duration, height, width, channels),
                anti_alias_level=post_aa,
                post_processes=list(post_processes),
                apply_fxaa=self.video_settings.fxaa,
                premultiplied_over=self.premultiplied_over,
                _linear_output=getattr(self, "_linear_output", False),
            )
            if getattr(self.memory, "managed", False):
                model.observe(
                    signature, duration, max(0, self.memory.max_pointer - chunk_base)
                )
            self.memory.set_pointers(original_pointers)
            if aux_sink is not None:
                from algan.rendering.raytracing.aux_passes import empty_aux_passes

                aux_sink(
                    empty_aux_passes(
                        len(frames),
                        self.num_pixels_screen_height,
                        self.num_pixels_screen_width,
                    )
                )
            yield frames
            current_ind = new_ind

    def _is_batchable_surface(self, actor):
        """True if this actor's geometry build can be stacked with same-shaped
        peers into one tensor pass (see surface.get_render_primitives_batched).
        Requires the stock Surface build (no subclass override), the plain
        vertex-color path, and computed normals. Set ALGAN_BATCH_SURFACE_PREP=0
        to disable batching (A/B against the per-surface path).
        """
        if not env_flag("ALGAN_BATCH_SURFACE_PREP", True):
            return False
        from algan.mobs.surfaces.surface import Surface

        if not isinstance(actor, Surface):
            return False
        if type(actor).get_render_primitives is not Surface.get_render_primitives:
            return False
        if actor._has_color_texture or actor.ignore_normals:
            return False
        if (
            getattr(actor, "material_texture", None) is not None
            or getattr(actor, "normal_texture", None) is not None
        ):
            return False
        return not (
            actor is self.camera
            or actor is self.camera.screen
            or actor in self.light_sources
        )

    def _is_batchable_bezier(self, actor):
        """True if this bezier circuit's primitive build can be merged with
        same-shaped peers into one vectorized pass (see
        bezier_circuit.build_render_primitives_batched). Requires the stock
        BezierCircuitCubic build methods, a non-empty circuit, un-batched
        control points, and singleton rows for the per-circuit attributes.
        Set ALGAN_BATCH_BEZIER_PREP=0 to disable (A/B against the per-actor
        path).
        """
        if not env_flag("ALGAN_BATCH_BEZIER_PREP", True):
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
        if actor._stroke_style_key() is not None:
            return False
        if getattr(actor, "_nonplanar_plan", None) is not None:
            # Renders as PN patches and/or per-run circuits, neither of which
            # the vectorized circuit pack knows how to build.
            return False
        if actor.control_points.parent_batch_sizes is not None:
            return False
        timeline = self.timeline_manager
        try:
            for attr in (
                "opacity",
                "basis",
                "glow",
                "stroke_width",
                "location",
            ):
                if (
                    timeline.attr_to_timeline[attr].mob_id_to_inds[actor.id].numel()
                    != 1
                ):
                    return False
            loc_inds = timeline.attr_to_timeline["location"].mob_id_to_inds
            if loc_inds[actor.control_points.id].numel() % 4 != 0:
                return False
            timeline.attr_to_timeline["color"].mob_id_to_inds[actor.grid.id]
            timeline.attr_to_timeline["color"].mob_id_to_inds[actor.border_grid.id]
        except (KeyError, AttributeError):
            return False
        return True

    def _bezier_block_key(self, actor):
        """``_bezier_group_key`` for an actor that may not be batchable.

        The draw-order walk keys every circuit by the block it will be merged
        into, including the packed and non-batchable ones that never reach
        ``_build_deferred_beziers``; those have no timeline rows for the key's
        texture-row counts, and are keyed by their batch identifier alone.
        """
        try:
            return self._bezier_group_key(actor)
        except (KeyError, AttributeError):
            from algan.rendering.primitives.bezier_circuit_primitive import (
                BezierCircuitPrimitive,
            )

            return BezierCircuitPrimitive.batch_identifier_for(
                actor.num_texture_points, actor.filled, actor._stroke_style_key()
            )

    def _authored_draw_order(self):
        """This Scene's actors in Manim's draw order, and what it costs in depth.

        Returns ``(rank, bias)``. ``rank`` orders *every* actor: each authored
        tree walked parent-first with the roots in creation order -- Manim's
        flattened family. The sort it replaces was by hierarchy depth
        descending, which keeps parents ahead of their children but interleaves
        unrelated trees: a depth-3 node of one tree landed ahead of a depth-2
        node of another, which is what split an arrow's shaft and tip around a
        grid they crossed.

        ``bias`` is that order expressed in coplanarity bins, for the circuits
        only. Coplanar circuits tie on distance and are resolved by their
        position in the merged arrays, and that position follows the draw order
        only *within one merge block* -- filled circuits, stroked ones and each
        distinct texture-grid shape are packed separately, and a block lands
        wherever its first member did. A bias is therefore needed exactly where
        the walk crosses a block boundary, and one bin there is enough, so the
        count is the number of alternations rather than the number of Mobs (23
        for the 117 circuits of the manim-compatibility scene). It is centred
        on zero so the scene straddles the plane it was authored on instead of
        drifting off it.

        Computed over the whole authored Scene rather than the frame window's
        actors, which is what makes it stable: were it derived from the live
        set, a despawn elsewhere would renumber the survivors and step their
        depths mid-render.
        """
        from algan.mobs.bezier_circuit import BezierCircuitCubic

        seen = set()
        order = []

        def visit(root):
            stack = [root]
            while stack:
                mob = stack.pop()
                if id(mob) in seen:
                    continue
                seen.add(id(mob))
                order.append(mob)
                children = getattr(mob, "children", None)
                if children:
                    stack.extend(reversed(list(children)))

        for actor in self.actors:
            if not getattr(actor, "parents", None):
                visit(actor)
        # An actor registered without ever being reachable from a root keeps
        # its registration position rather than being dropped from the order.
        for actor in self.actors:
            visit(actor)

        rank = {id(mob): i for i, mob in enumerate(order)}

        circuits = [
            mob
            for mob in order
            if isinstance(mob, BezierCircuitCubic)
            and not mob.empty
            and hasattr(mob, "get_render_primitives")
        ]
        # ``z_index`` is Manim's primary key over the flattened family, and
        # Python's sort is stable, so equal values keep the walk's order.
        circuits.sort(key=lambda mob: mob.z_index)

        bias = {}
        steps = 0
        previous = None
        for mob in circuits:
            # Either half of the key can put this circuit's merged position out
            # of draw order: a different block packs it elsewhere, and a
            # different z_index means the sort above moved it relative to the
            # collection order, which the walk alone still follows.
            key = (mob.z_index, self._bezier_block_key(mob))
            if previous is not None and key != previous:
                steps += 1
            previous = key
            bias[id(mob)] = steps
        centre = steps // 2
        return rank, {key: value - centre for key, value in bias.items()}

    def _bezier_group_key(self, actor):
        from algan.rendering.primitives.bezier_circuit_primitive import (
            BezierCircuitPrimitive,
        )

        timeline = self.timeline_manager
        tex_rows = (
            timeline.attr_to_timeline["color"].mob_id_to_inds[actor.grid.id].numel()
        )
        border_tex_rows = (
            timeline.attr_to_timeline["color"]
            .mob_id_to_inds[actor.border_grid.id]
            .numel()
        )
        return (
            BezierCircuitPrimitive.batch_identifier_for(
                actor.num_texture_points, actor.filled, actor._stroke_style_key()
            ),
            tex_rows,
            border_tex_rows,
            actor.render_primitive,
        )

    def _build_deferred_beziers(self, deferred):
        """Build one merged bezier primitive per group of deferred circuits
        in a single vectorized pass (see
        bezier_circuit.build_render_primitives_batched). The merged primitive
        is attached to the group's first entry (matching the position the
        group's collection had in the per-actor path); later entries stay
        empty.

        A group is a ``(batch group key, run index)`` pair: get_batch_of_
        primitives stamps ``entry["run"]`` when it splits an identifier that
        clashes with raw primitives into maximal batchable runs, and merging
        per run is what keeps each merged collection on the span its
        circuits' raw primitives would have occupied. Entries from the
        gated-off path carry no run index at all; ``get`` maps them onto run
        0, which collapses the key to the historical grouping exactly -- and
        no group key either, so it is computed here for them instead.
        """
        from algan.mobs.bezier_circuit import build_render_primitives_batched

        groups = {}
        for entry in deferred:
            key = entry.get("group_key")
            if key is None:
                key = self._bezier_group_key(entry["actor"])
            groups.setdefault((key, entry.get("run", 0)), []).append(entry)
        registry = _aux_source_registry(self)
        for entries in groups.values():
            mega = build_render_primitives_batched([e["actor"] for e in entries], self)
            if registry is not None:
                # One circuit per actor, in entry order (``_is_batchable_bezier``
                # guarantees the single row), so the lane is the actors' ids.
                for entry in entries:
                    registry.setdefault(int(entry["actor"].id), entry["actor"])
                mega._circuit_source_ids = [int(e["actor"].id) for e in entries]
            entries[0]["prebuilt"] = [mega]

    def _build_deferred_surfaces(self, deferred):
        """Build geometry for all deferred surfaces, one stacked tensor pass
        per (grid shape, materialized location shape) group (see
        surface.get_render_primitives_batched).
        """
        from algan.mobs.surfaces.surface import get_render_primitives_batched

        groups = collections.defaultdict(list)
        for entry in deferred:
            actor = entry["actor"]
            key = (
                actor.grid_width,
                actor.grid_height,
                # Shape only: read the location uncopied rather than clone
                # every grid once per surface per batch just for ``.shape``.
                tuple(actor.grid.get_animated_attribute("location", copy=False).shape),
            )
            groups[key].append(entry)

        registry = _aux_source_registry(self)
        for entries in groups.values():
            prims = get_render_primitives_batched([e["actor"] for e in entries])
            for entry, p in zip(entries, prims):
                if isinstance(p, list):
                    entry["prims"] = p
                else:
                    entry["prims"] = [p] if p is not None else []
                if registry is not None:
                    _stamp_primitive_sources(entry["prims"], entry["actor"], registry)

    def _scene_has_renderable_actors(self, start_time_ind, end_time_ind):
        """Whether any spawned renderable actor's lifespan intersects the
        frame window (mirroring ``_get_batch_of_primitives``' candidate
        filter). Cheap -- only lifespan timestamps are evaluated -- so the
        empty-scene warning can fire before rendering starts rather than
        after it finishes.
        """
        # Local: algan.rendering.lights reaches Mob, which cannot be imported
        # while this module is still initializing.
        from algan.rendering.lights import LIGHT_AREA_SAMPLE

        start_time = start_time_ind / self.frames_per_second
        end_time = end_time_ind / self.frames_per_second
        actors = list(self.actors)
        if (
            SETTINGS.raytracing.samples_per_pixel > 1
            and SETTINGS.raytracing.pt_area_light_quads
        ):
            actors.extend(
                light
                for light in self.light_sources
                if getattr(light, "light_type", -1) == LIGHT_AREA_SAMPLE
            )
        return any(
            (actor.lifespan.start() >= 0)
            and (actor.lifespan.start() <= end_time)
            and ((actor.lifespan.end() >= start_time) or actor.lifespan.end() < 0)
            and (
                hasattr(actor, "get_render_primitives")
                or getattr(actor, "light_type", -1) == LIGHT_AREA_SAMPLE
            )
            for actor in actors
        )

    def _warn_vertex_baked_lighting(self):
        """Warn about spawned Mobs whose shading can only be baked into vertex
        colors while this Scene's lighting rig asks for more than that bake
        delivers (extended lights, shadows, an environment map).

        ``set_material`` makes the same check against the lights that exist when
        it is called; this one is what catches the ordinary authoring order,
        where the material is chosen before the lights are spawned or before
        ``shadows`` is turned on. Cheap: the rig is resolved first, so the
        ordinary scene never walks its actors at all, and the walk itself reads
        one attribute before it evaluates a lifespan (which is not free -- see
        ``_get_batch_of_primitives``). No primitives are built.
        """
        from algan.rendering.shaders.materials import (
            _PER_FRAGMENT_ADVICE,
            _lighting_beyond_vertex_bake,
            _shades_per_fragment,
        )

        # A diagnostic must not be able to abort a render, and the render-loop
        # tests drive this mixin with stub Scenes that carry only the state the
        # loop itself reads -- so every attribute is optional here.
        features = _lighting_beyond_vertex_bake(
            getattr(self, "light_sources", None) or (),
            environment_map=getattr(self, "environment_map", None),
        )
        if not features:
            return
        baked = [
            actor
            for actor in getattr(self, "actors", None) or ()
            if hasattr(actor, "get_render_primitives")
            and not _shades_per_fragment(getattr(actor, "shader", None))
            and actor.is_spawned()
        ]
        if not baked:
            return
        # Name what the author chose -- the material -- falling back to the
        # shader for a Mob shaded through set_shader directly.
        kinds = sorted(
            {
                type(actor.material).__name__
                if getattr(actor, "material", None) is not None
                else getattr(actor.shader, "__name__", "a custom shader")
                for actor in baked
            }
        )
        warnings.warn(
            f"{len(baked)} spawned Mob(s) use shading Algan can only bake into "
            f"vertex colors ({', '.join(kinds)}), so {'; '.join(features)}. "
            f"{_PER_FRAGMENT_ADVICE}",
            UnsupportedFeatureWarning,
            stacklevel=_user_stacklevel(),
        )

    def _never_spawned_root_mobs(self):
        """Root actors whose whole subtree never spawned but which own geometry.

        These are authored Mobs that will silently not appear in the video --
        the commonest beginner mistake. Cheap: only the hierarchy and lifespan
        timestamps are consulted, no primitives are built.

        Reference geometry is excluded by construction rather than by a special
        case: a Mob built with ``add_to_scene=False`` never enters ``actors``,
        and that flag is exactly what marks a Mob as never intended to be shown.

        Each remaining clause is load-bearing. ``not actor.parents`` collapses a
        container and its children into one report (an unspawned ``Tex``
        registers half a dozen actors). ``is_spawned_in_subtree`` rather than
        ``is_spawned`` because containers are routinely left unspawned while
        their children spawn individually. And the renderable test runs over
        descendants because ``get_render_primitives`` lives on leaf classes --
        a ``Text`` does not have it, its character batch does.
        """
        return [
            actor
            for actor in self.actors
            if not actor.parents
            and not actor.is_spawned_in_subtree()
            and any(
                hasattr(d, "get_render_primitives") for d in actor.get_descendants()
            )
        ]

    def _render_device_prep_budget(self):
        """Render-device bytes a batch's preparation may hold at once.

        The animation-device budget is a setting (``cpu_render_memory_budget``) and
        so is this one, in effect: a fixed share of what the device's pool
        (:func:`_render_device_pool_bytes`) holds outside the arena's fraction.
        The merge's and projection's own out-of-arena scratch stays bounded
        separately by the batch preflight; a device genuinely short of memory
        falls back on the render's out-of-memory retry, as the animation
        budget does.
        """
        total_bytes = _render_device_pool_bytes(render_device())
        outside_arena = total_bytes * (
            1.0 - float(SETTINGS.computing.rendering_memory_fraction)
        )
        return int(outside_arena * _RENDER_PREP_FRACTION)

    def _get_batch_of_primitives(
        self,
        start_time_ind,
        max_end_time_ind,
        actors,
        max_mem_used,
        *,
        frame_indices=None,
    ):
        """Build the largest renderable primitive batch within the memory budget."""
        from algan.rendering.lights import LIGHT_AREA_SAMPLE

        real_start, real_end = _frame_window_bounds(
            start_time_ind, max_end_time_ind, frame_indices
        )
        max_end_time = real_end / self.frames_per_second
        start_time = real_start / self.frames_per_second
        # Spawn/despawn timestamps are read several times each below (twice per
        # actor in each of the two filters, and once per actor on every step of
        # the runtime search). Each read walks a TimelineEvent to its span and
        # recomputes the context rescaling, which on a scene with tens of
        # thousands of actors made this the single hottest function in batch
        # preparation. Timing is fixed for the whole render, so read each
        # actor's pair once -- once per *render*, indexed by time, so a batch
        # scans its own window rather than the whole scene (which was O(n^2)
        # across a render, and is what _actor_window_index exists for).
        index = self._actor_window_index(actors)
        indexed_actors, spawns = index[0], index[1]
        candidates = self._actors_in_window(index, start_time, max_end_time)
        primitive_actors = [
            (indexed_actors[i], float(spawns[i]))
            for i in candidates.tolist()
            if hasattr(indexed_actors[i], "get_render_primitives")
        ]

        # Precompute memory per timestep once to avoid redundant calls inside binary search loop
        actor_mem = [
            (
                spawn,
                actor._get_memory_used_per_timestep(),
                actor._get_render_device_memory_used_per_timestep(),
            )
            for actor, spawn in primitive_actors
        ]
        # Unknown callback dependencies materialize all texture rows, even
        # panels outside this actor window. The actor-only estimate misses
        # those entirely. Use the full requested interval conservatively;
        # preparation retries still cover callback-owned allocations.
        fallback_animation_mem, fallback_render_mem = (
            self.timeline_manager._unbounded_texture_memory_per_timestep(
                start_time, max_end_time
            )
        )
        # Two budgets: what a frame allocates on the animation device, and
        # what it allocates on the render device -- a wide attribute's window
        # (a texture) materializes there, outside the render arena, and a
        # batch that fits the first budget by a mile can still exhaust the
        # second. Read once per fetch, since it is a measurement.
        max_render_mem_used = (
            self._render_device_prep_budget()
            if fallback_render_mem or any(render_mem for _, _, render_mem in actor_mem)
            else float("inf")
        )

        # Binary search for the largest batch that fits the animation-device
        # budget.  The selected actor set grows monotonically with runtime, so
        # the memory predicate is monotone too.
        def get_duration():
            requested_duration = min(
                max_end_time_ind - start_time_ind,
                SETTINGS.computing.max_animation_batch_size,
            )

            def fits(duration):
                _, last = _frame_window_bounds(
                    start_time_ind, start_time_ind + duration, frame_indices
                )
                cutoff = last / self.frames_per_second
                mem_used = 0
                render_mem_used = 0
                for spawn, mem, render_mem in actor_mem:
                    if spawn <= cutoff:
                        mem_used += mem * duration
                        render_mem_used += render_mem * duration
                mem_used = max(mem_used, fallback_animation_mem * duration)
                render_mem_used = max(render_mem_used, fallback_render_mem * duration)
                return (
                    mem_used <= max_mem_used and render_mem_used <= max_render_mem_used
                )

            return _max_duration_that_fits(requested_duration, fits)

        duration = get_duration()
        _, last = _frame_window_bounds(
            start_time_ind, start_time_ind + duration, frame_indices
        )
        spawn_cutoff = last / self.frames_per_second
        actors = [
            indexed_actors[i]
            for i in self._actors_in_window(index, start_time, spawn_cutoff).tolist()
        ]
        selected_indices = (
            frame_indices[start_time_ind : start_time_ind + duration]
            if frame_indices is not None
            else None
        )
        time_inds = (
            torch.tensor(selected_indices, dtype=torch.int64)
            if selected_indices is not None
            else torch.arange(start_time_ind, start_time_ind + duration)
        )

        timeline = self.timeline_manager
        # Restrict base-state queries to actors that can contribute to this
        # frame window. Animation replay retains global row ids, and the
        # timeline conservatively falls back to all rows for user callbacks or
        # updaters whose dependencies cannot be discovered safely. Stills that
        # share a batch replay frame by frame, so each frame's state is exactly
        # its alone state (see TimelineManager._replay_frame_groups).
        timeline.replay_frame_by_frame = bool(
            getattr(self, "_frame_local_batches", False)
        )
        try:
            timeline.set_state_to_times(
                time_inds / self.frames_per_second, active_mobs=actors
            )
        finally:
            timeline.replay_frame_by_frame = False

        # Each bucket holds the batch identifier's primitives and merged
        # collections as an ordered list of ``(is_finished_collection,
        # primitive_class, primitive)`` items, in actor order. A bucket used
        # to be a ``[class marker, flat list]`` pair and could therefore hold
        # either raw primitives or finished collections, never both -- which
        # is what forced the all-or-nothing bezier group revert. With the
        # groups split into runs (above) a bucket legitimately mixes both:
        # a run's merged collection shares its identifier with the very raw
        # primitives that caused the clash. The walk below emits each
        # maximal run of raw primitives through the same per-class emission
        # as before, so what lands downstream is the same concatenation the
        # all-raw path produced.
        grouped_primitives = collections.defaultdict(list)
        # The object-id pass's source registry, ``None`` unless that pass is
        # armed for this render -- in which case every primitive taken from an
        # actor below (and in the deferred builders) is stamped with its Mob.
        aux_registry = _aux_source_registry(self)
        # Surfaces sharing a grid shape are not built one-by-one: their state
        # is materialized per-actor below (in anchor-priority order, exactly as
        # before), but the geometry build is deferred so all of them can run as
        # one stacked tensor pass (_build_deferred_surfaces). ordered_items
        # records primitives / deferred entries in actor order so the final
        # grouping (and thus the merged collection layout) is unchanged.
        ordered_items = []
        deferred_surfaces = []
        deferred_beziers = []
        if env_flag("ALGAN_COPLANAR_DRAW_ORDER", True):
            draw_rank, draw_bias = self._authored_draw_order()
            missing = len(draw_rank)
            collection_order = sorted(
                actors, key=lambda x: draw_rank.get(id(x), missing)
            )
        else:
            # The historical order: a global sort by hierarchy depth. Kept for
            # A/B against the draw-order rule, which moves coplanar 2-D output.
            draw_bias = {}
            collection_order = sorted(
                actors, key=lambda x: x.anchor_priority, reverse=True
            )
        for actor in collection_order:
            if not hasattr(actor, "get_render_primitives"):
                continue
            if id(actor) in draw_bias:
                # Consumed by the circuit primitive builders; ``None`` leaves a
                # Mob built outside a render loop on its authored z_index.
                actor._draw_bias = float(draw_bias[id(actor)])
            if self._is_batchable_surface(actor):
                # ``kind`` tells the deferred-bezier run walk below that this
                # entry's primitives are raw (already built by then) without
                # probing for whichever payload key happens to be filled.
                entry = {"actor": actor, "prims": None, "kind": "surface"}
                deferred_surfaces.append(entry)
                ordered_items.append(entry)
                continue
            if self._is_batchable_bezier(actor):
                entry = {
                    "actor": actor,
                    "prims": None,
                    "prebuilt": None,
                    "kind": "bezier",
                }
                deferred_beziers.append(entry)
                ordered_items.append(entry)
                continue
            capture = getattr(self, "_camera_view_captures", {}).get(id(actor))
            if capture is not None:
                from algan.rendering.camera_views import _view_primitive

                primitive = _view_primitive(actor, capture, time_inds)
            else:
                primitive = actor.get_render_primitives()
            if primitive is not None:
                if not isinstance(primitive, list):
                    primitive = [primitive]
                if aux_registry is not None:
                    _stamp_primitive_sources(primitive, actor, aux_registry)
                ordered_items.append(primitive)

        if deferred_surfaces:
            self._build_deferred_surfaces(deferred_surfaces)

        if deferred_beziers and env_flag("ALGAN_BEZIER_GROUP_RUNS", True):
            # A non-batchable primitive sharing a group's batch identifier
            # would be concatenated into the same collection, interleaved by
            # actor order -- which used to force the *whole group* back to the
            # per-actor build. The old comment called such groups rare; on the
            # reference scene they reverted 51.5% of all circuits, because a
            # single packed circuit (a Text's glyphs, say) poisons every
            # batchable peer sharing its identifier. Wholesale reversion is
            # not needed to keep the merged layout identical: within one
            # batch identifier, each deferred circuit sits after some number
            # of raw primitives of that identifier and before the rest, so
            # splitting the group into maximal runs of consecutive batchable
            # actors -- each run merged on its own -- puts every merged
            # collection on exactly the span its circuits' raw primitives
            # would have occupied, and the bucket's concatenation comes out in
            # the per-actor path's order. ``entry["run"]`` records that
            # position: the count of raw primitives sharing the entry's batch
            # identifier seen so far in this walk. "Raw" means anything that
            # will be registered into grouped_primitives outside a merged
            # collection -- both plain per-actor lists and a deferred
            # surface's already-built primitives. Surfaces cannot actually
            # collide with a circuit's identifier today, but counting them
            # costs one dict lookup apiece and makes the invariant true by
            # construction rather than by a class-name accident.
            # The key is stamped on the entry rather than recomputed in
            # _build_deferred_beziers: it costs two timeline row lookups per
            # circuit, and this walk already has to take it for every one of
            # them to find the identifier.
            raw_counts = collections.Counter()
            for item in ordered_items:
                if isinstance(item, dict):
                    if item["kind"] == "bezier":
                        key = self._bezier_group_key(item["actor"])
                        item["group_key"] = key
                        item["run"] = raw_counts[key[0]]
                        continue
                    prims = item["prims"]  # a deferred surface: built above
                else:
                    prims = item
                for p in prims:
                    raw_counts[p.get_batch_identifier()] += 1
            self._build_deferred_beziers(deferred_beziers)
        elif deferred_beziers:
            # A non-batchable primitive sharing a group's batch identifier
            # would have been concatenated into the same collection,
            # interleaved by actor order; fall back to the per-actor build
            # for such (rare) groups so the collection layout is unchanged.
            # ALGAN_BEZIER_GROUP_RUNS=0 keeps this all-or-nothing revert (A/B
            # against the run splitting above).
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
                        entry["prims"] = (
                            primitive if isinstance(primitive, list) else [primitive]
                        )
                        if aux_registry is not None:
                            _stamp_primitive_sources(
                                entry["prims"], entry["actor"], aux_registry
                            )
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
                    grouped_primitives[collection.get_batch_identifier()].append(
                        (True, None, collection)
                    )
                continue
            primitives = item["prims"] if isinstance(item, dict) else item
            if not primitives:
                continue
            for p in primitives:
                grouped_primitives[p.get_batch_identifier()].append(
                    (False, p.__class__, p)
                )

        primitive_collections = []
        for _, items in grouped_primitives.items():
            run = []
            run_class = None
            for is_collection, primitive_class, primitive in items:
                if is_collection:
                    if run:
                        self._emit_primitive_collections(
                            run_class, run, primitive_collections
                        )
                        run = []
                    primitive.memory = self.memory
                    primitive.scene = self
                    primitive_collections.append(primitive)
                    continue
                # Last raw item wins, exactly as the old single class marker
                # was overwritten by each registration; identifiers are
                # class-derived, so a run mixing classes cannot arise without
                # two classes sharing one identifier string.
                run_class = primitive_class
                run.append(primitive)
            if run:
                self._emit_primitive_collections(run_class, run, primitive_collections)
        render_state = self._materialize_render_state(
            start_time_ind,
            start_time_ind + duration,
            **(
                {"frame_indices": selected_indices}
                if selected_indices is not None
                else {}
            ),
        )
        if (
            not primitive_collections
            and SETTINGS.raytracing.samples_per_pixel > 1
            and SETTINGS.raytracing.pt_area_light_quads
            and any(
                getattr(light, "light_type", -1) == LIGHT_AREA_SAMPLE
                for light in render_state["light_objects"]
            )
        ):
            # The render loop dispatches empty batches straight to the
            # background writer. Supply a non-rendering collection so a
            # light-only scene reaches the ordinary merge, which adds the
            # emitter geometry before building its BVH. No actor is authored.
            from algan.rendering.raytracing.primitives import RayTracedTrianglePrimitive
            from algan.rendering.shaders.pbr_shaders import null_shader

            padding = RayTracedTrianglePrimitive(
                corners=torch.zeros((duration, 3, 3)),
                colors=torch.zeros((1, 3, 5)),
                shader=null_shader,
            )
            collection = RayTracedTrianglePrimitive(triangle_collection=[padding])
            collection.memory = self.memory
            collection.scene = self
            primitive_collections.append(collection)
        # The batch's primitives are built: the texture windows that fed them
        # (a whole image per frame, on the render device) have no reader left
        # and would otherwise sit beside the next batch's until this one has
        # rendered. See AnimationTimeline.release_wide_windows.
        timeline.release_wide_windows()
        return primitive_collections, start_time_ind + duration, render_state

    def _emit_primitive_collections(self, primitive_class, primitives, out):
        """Build the collections for one run of same-identifier raw
        primitives and append them to ``out``.

        This is the per-class emission the final grouping loop in
        _get_batch_of_primitives has always applied to a whole bucket,
        factored out unchanged so a bucket that mixes raw runs with finished
        collections (see the run splitting above) can emit its raw runs
        between the collections it walks past. Called once over a bucket
        with no finished collections in it, it must -- and does -- produce
        exactly the collections that bucket produced before the split, in
        the same order, with ``.memory`` / ``.scene`` set the same way.
        """
        if not primitives:
            return
        if primitive_class is BezierCircuitPrimitive:
            max_bezier_batch_size = 50000
            counts = torch.tensor([p.corners.shape[1] for p in primitives]).cumsum(0)
            num_sub_batches = (counts[-1] // max_bezier_batch_size) + 1
            current_ind = 0
            for _i in range(num_sub_batches):
                inds = (counts > max_bezier_batch_size).nonzero()
                if len(inds) == 0:
                    next_ind = len(primitives)
                else:
                    next_ind = max(inds[0], current_ind + 1)
                out.append(
                    primitive_class(
                        triangle_collection=primitives[current_ind:next_ind]
                    )
                )
                current_ind = next_ind
                out[-1].memory = self.memory
                out[-1].scene = self
                if current_ind >= len(primitives):
                    break
                counts -= counts[current_ind - 1]
        else:
            textured = []
            colored = []
            for p in primitives:
                if (
                    getattr(p, "uvs", None) is not None
                    or getattr(p, "texture_map", None) is not None
                ):
                    textured.append(p)
                else:
                    colored.append(p)
            if colored:
                out.append(primitive_class(triangle_collection=colored))
                out[-1].memory = self.memory
                out[-1].scene = self
            # Textured primitives are batched one per collection: a
            # collection carries a single texture map set (color/material/
            # normal), so merging two differently-textured primitives
            # would drop all but the first primitive's maps. Their
            # geometry is still merged into one kernel launch downstream
            # (see _merge_scene).
            for p in textured:
                out.append(primitive_class(triangle_collection=[p]))
                out[-1].memory = self.memory
                out[-1].scene = self

    @_coalesced_memory_reclamation()
    def _prewarm_render_batch(self, primitives, render_state):
        """Run a batch's ``project_to_screen`` (+ the CPU merge when merge and
        project both stay on the CPU) ahead of the render.

        Called on the prefetch worker (from ``fetch_batch``) when projection is
        on the CPU, so the work stays hidden behind the previous batch's
        render; when projection is on the render device (default; see
        settings.project_on_gpu) it is deferred to the render thread instead
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
        by ``_render_primitive_batch``; on any failure the un-projected remainder
        (and the merge) simply runs on the render thread as before.
        """
        try:
            from algan.rendering.raytracing.primitives import (
                RayTracedBezierCircuitPrimitive,
                RayTracedTrianglePrimitive,
            )
            from algan.rendering.raytracing.scene_builder import (
                upload_primitive_source,
            )
        except Exception:
            return
        if not primitives or not isinstance(
            primitives[0], (RayTracedTrianglePrimitive, RayTracedBezierCircuitPrimitive)
        ):
            return
        rt_settings = SETTINGS.raytracing

        aa, analytic_raster = _projection_anti_alias_level(self, primitives)
        # Projection runs on the render device by default (see
        # settings.project_on_gpu); the primitive source geometry and the
        # camera/light snapshot are moved there so the packed _rt_* outputs are
        # built on it (ready for the GPU merge, no upload). Off keeps
        # projection on the snapshot's source (CPU) device.
        gpu_project = rt_settings.project_on_gpu_active()
        project_device = render_device() if gpu_project else None

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
        camera.output_screen_width = self.num_pixels_screen_width
        camera.output_screen_height = self.num_pixels_screen_height
        camera.analytic_raster = analytic_raster

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
                        primitive, fallback=render_state["ray_origin"].device
                    )
                scratch = scratch_by_device.get(source_device)
                if scratch is None:
                    scratch = ManualMemory(0, device=source_device, managed=False)
                    scratch_by_device[source_device] = scratch
                primitive.memory = scratch
                # Marked until it completes: a projection that raises part-way
                # leaves its primitive half-transformed, which fetch_batch must
                # not hand on for re-projection (see prewarm_batch there).
                primitive._rt_projection_interrupted = True
                primitive.project_to_screen(camera, lights)
                primitive._rt_projection_interrupted = False
                primitive._rt_projected = True
            finally:
                primitive.memory = original_memory
        # The merge + STBVH build ride the prefetch worker only when they run
        # on the CPU. When they run on the render device (the default; see
        # settings.merge_on_gpu) they are deferred to the render thread so
        # their transient device peak is measured/bounded without a concurrent
        # render polluting the pool -- unless overlap is enabled and its own
        # conditions hold, in which case _prepare_batch_on_worker below has
        # already run them here.
        if not rt_settings.merge_on_gpu_active():
            self._prepare_merged_host_scene(primitives, render_state=render_state)

    def _overlap_headroom_fraction(self):
        """Share of the pool headroom an overlapped build must fit inside.

        The render thread's preflight bounds a projection or merge against the
        whole of ``_gpu_merge_headroom_bytes`` because nothing else is running.
        A build launched on the prefetch worker runs beside a live render that
        draws on the same headroom, so it may only claim this share of it; the
        rest belongs to the render. Setting, with an env override read live so
        an A/B script can retune between renders. An out-of-range override
        warns and falls back to the setting, like every env read: a mistyped
        knob must not silently drop the derate.
        """
        fraction = env_float(
            "ALGAN_OVERLAP_HEADROOM_FRACTION",
            SETTINGS.computing.overlap_pool_headroom_fraction,
        )
        if not 0.0 < fraction <= 1.0:
            warnings.warn(
                f"ALGAN_OVERLAP_HEADROOM_FRACTION={fraction!r} is outside "
                "(0, 1]; using the setting's "
                f"{SETTINGS.computing.overlap_pool_headroom_fraction!r} instead.",
                AlganWarning,
                stacklevel=_user_stacklevel(),
            )
            return SETTINGS.computing.overlap_pool_headroom_fraction
        return fraction

    def _overlap_gpu_prep_active(self):
        """Whether this fetch may prepare its batch on the prefetch worker.

        Requires the setting (default off; env override
        ``ALGAN_PREFETCH_GPU_PREP`` read live), both GPU builds active, and
        the calling thread to be the prefetch worker: a synchronous fetch has
        no render to hide behind, and letting it "overlap" would swap the
        render thread's full-headroom estimates for derated ones and so move
        window decisions between arms without any concurrency to show for it.
        """
        if not env_flag(
            "ALGAN_PREFETCH_GPU_PREP", SETTINGS.computing.prefetch_gpu_prep
        ):
            return False
        rt_settings = SETTINGS.raytracing
        if not (
            rt_settings.project_on_gpu_active() and rt_settings.merge_on_gpu_active()
        ):
            return False
        return threading.current_thread().name.startswith("algan-batch-prep")

    def _await_prefetched_batch(self, pending):
        """Block until the prefetch worker hands over the next batch.

        A method rather than an inline ``pending.result()`` so the profiler
        can time it: it is the render thread's idle time waiting on scene
        preparation, i.e. the part of prep that is on the critical path.
        Everything the worker did that this does not wait for was free.

        An ordinary method, not a ``staticmethod``: the profiler's hook
        replaces the class attribute with a plain function, which a
        staticmethod's bound call then hands ``self`` as well -- a profiled
        multi-batch render died on the first prefetched batch that way.
        """
        return pending.result()

    def _prepare_batch_on_worker(self, primitive_batch, render_state):
        """Run this batch's GPU projection + merge here on the prefetch worker,
        while the previous batch renders.

        This is the overlap half of what the arena preflight otherwise does on
        the render thread: the same builds, on the same device, against the
        same inputs, so the packed arrays are byte-identical to preparation on
        the render thread. What is deliberately *not* kept is the peak
        bookkeeping -- a build timed beside a live render reports a peak that
        includes the render's allocations, so the batch arrives stamped
        ``_rt_prep_overlapped`` and the preflight skips both the observations
        and the now-moot estimates (the predictors keep the estimates they
        learned from the un-overlapped batches).

        Each build is bounded proactively, as on the render thread, but
        against the headroom derated by :meth:`_overlap_headroom_fraction`.
        A declined estimate leaves the batch untouched for the render thread,
        which then runs today's path unchanged -- including its right to
        shrink the frame window before paying for a build. An out-of-memory
        from a build that was attempted clears its partial state and likewise
        defers. Either way the work is not wasted, only re-placed; the one
        genuinely speculative cost is that a batch prepared whole here can
        still fail its exact arena check on the render thread, which falls
        back to the ordinary refetch-at-a-shorter-window retry.

        Runs strictly after :meth:`_get_batch_of_primitives` on the same worker
        (so the timeline materialization it reads is complete) and strictly
        before the render thread's ``pending.result()`` handover, so none of
        the state written here is ever touched concurrently.
        """
        from algan.rendering.raytracing.primitives import (
            RayTracedBezierCircuitPrimitive,
            RayTracedTrianglePrimitive,
        )
        from algan.rendering.raytracing.scene_builder import (
            gpu_merge_input_bytes,
            gpu_project_input_bytes,
        )

        if not isinstance(
            primitive_batch[0],
            (RayTracedTrianglePrimitive, RayTracedBezierCircuitPrimitive),
        ):
            return
        # The first batch(es) of a job have no calibrated predictor to bound
        # the builds with and no render to hide behind anyway: leave them on
        # the render thread exactly as today.
        if not (
            self._project_peak_ratio.is_calibrated()
            and self._merge_peak_ratio.is_calibrated()
        ):
            return

        headroom = int(
            self._gpu_merge_headroom_bytes() * self._overlap_headroom_fraction()
        )
        project_inputs = gpu_project_input_bytes(primitive_batch)
        estimated_project_peak = self._project_peak_ratio.predict(project_inputs)
        if estimated_project_peak > headroom:
            logger.debug(
                "Overlapped projection estimate %.1f MB exceeds derated pool "
                "headroom %.1f MB [%s]; leaving the batch to the render "
                "thread.",
                estimated_project_peak / 1e6,
                headroom / 1e6,
                self._project_peak_ratio.describe(),
            )
            return
        self._prewarm_render_batch(primitive_batch, render_state)
        if not all(
            getattr(primitive, "_rt_projected", False) for primitive in primitive_batch
        ):
            return
        merge_inputs = gpu_merge_input_bytes(primitive_batch)
        estimated_merge_peak = self._merge_peak_ratio.predict(merge_inputs)
        if estimated_merge_peak > headroom:
            logger.debug(
                "Overlapped merge estimate %.1f MB exceeds derated pool "
                "headroom %.1f MB [%s]; leaving the merge to the render "
                "thread.",
                estimated_merge_peak / 1e6,
                headroom / 1e6,
                self._merge_peak_ratio.describe(),
            )
            return
        try:
            # track_peak=False: a peak measured beside the live render counts
            # the render's own allocations, and measuring it would reset the
            # process peak counter under that render. The value would be
            # discarded anyway (the preflight skips it for overlapped
            # batches), so the build simply does not measure.
            self._prepare_merged_host_scene(
                primitive_batch, render_state=render_state, track_peak=False
            )
        except (InsufficientMemoryException, RuntimeError) as exc:
            # The overlapped build overran even the derated headroom. Drop
            # any partial merge state; whatever projected cleanly stays
            # projected (the render thread skips it), and the merge itself
            # reruns there under the full-headroom estimates.
            if not isinstance(exc, InsufficientMemoryException) and not is_cuda_oom(
                exc
            ):
                raise
            primitive_batch[0]._rt_merged_scene = None
            primitive_batch[0]._rt_prepared_host_scene = None
            release_torch_memory(force_gc=False)
            logger.debug("Overlapped scene merge ran out of memory (%r).", exc)
            return
        primitive_batch[0]._rt_prep_overlapped = True
        # What the builds read, for the preflight's window-sizing cost terms:
        # it notes them as for any batch, though it observes no peaks.
        primitive_batch[0]._rt_overlap_inputs = (project_inputs, merge_inputs)
        logger.debug("Batch prepared on the prefetch worker (overlap).")

    def _materialize_render_state(self, start_ind, end_ind, *, frame_indices=None):
        """Materialize camera/screen/light state over ``[start_ind, end_ind)``
        and extract the plain tensors the renderer consumes (this used to be
        the first thing _render_primitive_batch did). Returning a snapshot
        instead of writing camera attributes means the render thread never
        reads animated state -- by the time a batch renders, prep for the
        *next* batch may be mutating that state on a worker thread.

        Lights whose lifespan never intersects the frame window are left out
        of the snapshot entirely (``light_objects`` carries the kept objects,
        aligned with ``lights``): an unspawned or already-despawned light
        contributes nothing, so packing it would only cost per-light kernel
        work in every batch after its despawn. A light that is live for part
        of the window is kept; its out-of-lifespan frames materialize
        zero-color rows, which every lighting path treats as inert (the
        shadow fans and the default shaders skip zero-color rows), so the
        output does not depend on where batch boundaries happen to fall
        relative to a light's spawn.
        """
        from algan.rendering.lights import LIGHT_AREA_SAMPLE

        camera = self.camera
        # Batch preparation is CPU/source-device work.  Keeping this snapshot
        # beside the materialized animation tensors prevents the prefetch worker
        # from allocating the next batch on the render device while the current
        # batch is still resident there.
        camera_location = camera.location
        device = camera_location.device
        fps = self.frames_per_second
        real_start, real_end = (
            (frame_indices[0], frame_indices[-1] + 1)
            if frame_indices is not None
            else (start_ind, end_ind)
        )
        window_start_time = real_start / fps
        window_end_time = real_end / fps
        lights = []
        light_objects = []
        light_active = []
        frame_times = None
        frame_local = bool(getattr(self, "_frame_local_batches", False))
        for light in self.light_sources:
            # Same lifespan-overlap test as the render loop's actor filter:
            # start < 0 means never spawned, end < 0 means never despawned.
            try:
                spawn_time = float(light.lifespan.start())
                despawn_time = float(light.lifespan.end())
            except (AttributeError, TypeError, ValueError):
                spawn_time = 0.0
                despawn_time = -1.0
            if spawn_time < 0 or spawn_time > window_end_time:
                continue
            if 0 <= despawn_time < window_start_time:
                continue
            light_objects.append(light)
            # Geometry has a lifespan independently of emission: an active
            # zero-intensity panel is opaque, a despawned panel is absent.
            active = None
            if (
                SETTINGS.raytracing.samples_per_pixel > 1
                and SETTINGS.raytracing.pt_area_light_quads
                and getattr(light, "light_type", -1) == LIGHT_AREA_SAMPLE
            ):
                if frame_times is None:
                    frame_times = (
                        torch.tensor(frame_indices, device=device)
                        if frame_indices is not None
                        else torch.arange(start_ind, end_ind, device=device)
                    ) / fps
                active = frame_times >= max(0.0, spawn_time)
                if despawn_time >= 0:
                    active = active & (frame_times <= despawn_time)
            light_active.append(active)
            loc = light.location
            # The one ingest point for light color, and the decode has to
            # happen here rather than at the pack: alpha and opacity below are
            # linear scalars and intensity below is a linear per-frame row, so
            # srgb_to_linear(c * k) is not srgb_to_linear(c) * k. Channel 3 is
            # glow, not color, so only 0:3 is decoded. This is what three.js
            # does too -- its Color is already linear by the time WebGLLights
            # multiplies in intensity.
            light_rgba = light.color
            if rt_settings_module.linear_color_space:
                light_rgba = torch.cat(
                    (
                        _decode_light_rgb(light_rgba[..., :3], frame_local),
                        light_rgba[..., 3:],
                    ),
                    -1,
                )
            col = light_rgba[..., :-1] * light_rgba[..., -1:] * light.opacity
            # Per-frame intensity row ([T, 1, 1] once materialized, so it
            # broadcasts directly). Kept LAST -- after alpha and opacity -- so
            # a constant intensity still computes ((rgb * glow) * opacity) * k
            # exactly as before. Lights without an intensity attribute (the
            # stub lights some render-loop tests drive this mixin with) skip it.
            intensity = getattr(light, "intensity", None)
            if intensity is not None:
                col = col * intensity
            is_ext = getattr(light, "_is_extended", None)
            if is_ext is not None and is_ext():
                # Extended light (see algan.rendering.lights): snapshot its
                # emitter sample positions and packed aux parameter columns.
                # Area lights expand into K samples, each carrying 1/K of the
                # light's power.
                loc_f = loc.reshape(loc.shape[0], -1)[:, :3]  # [T, 3]
                col_f = col.reshape(col.shape[0], -1)  # [T, C]
                pos_rows = light._get_sample_positions(loc_f)  # [T, K, 3]
                k = pos_rows.shape[-2]
                col_rows = (
                    (col_f / k if k > 1 else col_f).unsqueeze(-2).expand(-1, k, -1)
                )
                aux = light._build_aux(loc_f)  # [T, K, LIGHT_AUX_COLS]
                radiance_cols = getattr(light, "_AUX_RADIANCE_COLS", None)
                if radiance_cols is not None:
                    # Radiance-bearing aux columns (a hemisphere's ground
                    # color) scale with the light's per-frame opacity and
                    # intensity, like the RGB columns above -- so frames
                    # outside the light's lifespan pack a genuinely all-zero
                    # (inert) row rather than a row that keeps emitting from
                    # its aux columns. Two SEPARATE multiplies, intensity then
                    # opacity, because that is the order _build_aux used to bake
                    # in ((ground * intensity) * opacity); float multiplication
                    # is not associative, so folding the two scalars first
                    # could differ in the last bit and move a
                    # constant-intensity render.
                    a, b = radiance_cols
                    opacity = light.opacity
                    if intensity is not None:
                        aux[..., a:b] = aux[..., a:b] * intensity
                    aux[..., a:b] = aux[..., a:b] * opacity.reshape(
                        opacity.shape[0], 1, 1
                    )
                lights.append(
                    (pos_rows.to(device), col_rows.to(device), aux.to(device))
                )
            else:
                lights.append(
                    (
                        loc.unsqueeze(-2).to(device),
                        col.unsqueeze(-2).to(device),
                        None,
                    )
                )
        # ``[T, 1, 2]`` (aperture diameter, focus distance). getattr: the camera
        # doubles some tests drive this mixin with have no lens.
        lens_fn = getattr(camera, "_get_render_lens", None)
        return {
            **({"frame_indices": frame_indices} if frame_indices is not None else {}),
            "ray_origin": camera_location.unsqueeze(-2).to(device),
            "screen_point": camera.screen.location.unsqueeze(-2).to(device),
            "screen_basis": camera._get_render_screen_basis().to(device),
            **({"camera_lens": lens_fn().to(device)} if lens_fn is not None else {}),
            "lights": lights,
            "light_objects": light_objects,
            "light_active": light_active,
        }

    def get_frames(
        self,
        start_time_ind,
        end_time_ind,
        background=None,
        post_processes=(bloom_filter,),
        manual_memory=True,
        *,
        frame_indices=None,
        _post_process_per_frame=False,
        _independent_frames=False,
        aux_passes: bool = False,
        aux_sink=None,
    ):
        """Yield frames and always release per-render state on exit.

        With ``frame_indices``, start/end are offsets into that strictly
        increasing sequence of non-negative integer frame indices. Only those
        frames are materialized and rendered; gaps cost no frame storage.
        Without it, start/end retain their ordinary timeline-index meaning.

        ``_independent_frames`` (stills) renders every frame exactly as it would
        alone, from only its own Mobs. A batch carries every Mob alive anywhere
        in its window, so frames share one only while nothing spawns or
        despawns between them (:meth:`_still_group_ends`), at most
        ``_STILL_GROUP_MAX_FRAMES`` of them. Inside such a batch every choice
        that could move a frame's pixels is made per frame -- timeline replay,
        circuit chord counts and outline bounds, circuit geometry reuse, color
        decoding, post-processing --
        and a batch whose frames would decide anything else differently alone
        (route, opacity and material gates, triangle promotion; see
        ``_merge_scene``'s frame signature) is split before it renders
        (:meth:`_frames_deciding_alike`). The path tracer gives every still a
        batch of its own (:meth:`_stills_may_share_batches`).

        With ``aux_passes=True``, ``aux_sink`` (a callable, required then) is
        called exactly once per yielded batch, immediately before the batch is
        yielded, with that batch's compositing passes: a dict of CPU tensors
        (``"depth"``, ``"normal"``, ``"mob_id"``) whose leading dimension is
        the batch's frame count (``algan.rendering.raytracing.aux_passes``
        describes them). Background-only batches deliver passes that record
        nothing; a live camera view's capture passes compute none. Off (the
        default), nothing extra is allocated, traced or called.

        The wrapper is deliberately outside the implementation generator so
        its ``finally`` also runs for OOMs, worker failures, and callers that
        close the generator before consuming every frame.
        """
        if aux_passes and not callable(aux_sink):
            raise TypeError("aux_passes=True needs a callable aux_sink")
        # Forwarded only when requested, so the implementation generators are
        # called exactly as before when the passes (or independent frames)
        # are off.
        forwarded = {"aux_passes": True, "aux_sink": aux_sink} if aux_passes else {}
        if _independent_frames:
            forwarded["independent_frames"] = True
        if frame_indices is not None:
            import operator

            try:
                values = tuple(frame_indices)
                if any(isinstance(value, bool) for value in values):
                    raise ValueError
                frame_indices = tuple(operator.index(value) for value in values)
                if (
                    any(value < 0 for value in frame_indices)
                    or any(a >= b for a, b in zip(frame_indices, frame_indices[1:]))
                    or not 0 <= start_time_ind <= end_time_ind <= len(frame_indices)
                ):
                    raise ValueError
            except (TypeError, ValueError) as exc:
                raise AlganConfigurationError(
                    "frame_indices must be strictly increasing non-negative integers; "
                    "start/end must bound a slice of that sequence"
                ) from exc
            if start_time_ind == end_time_ind:
                return
        _check_post_processes(post_processes)
        if _independent_frames and post_processes:
            # Bloom included: it decides from a whole chunk whether any frame
            # glows, and a still's pixels must not depend on its batch-mates.
            post_processes = _framewise_post_processes(
                post_processes, batch_native=False
            )
        elif _post_process_per_frame and post_processes:
            post_processes = _framewise_post_processes(post_processes)
        # Every frame-producing path validates the arch before launching a
        # kernel: a render device changed since the last job needs a different
        # arch, or arguments would stage through the wrong device. Video jobs
        # also hold the arch through encoding and already made this selection;
        # this idempotent check covers direct frame-generator callers too.
        ensure_taichi_for_render()
        original_background = self.background_frame
        original_memory = self.memory
        # A render job is the scope of the truncation instrument: its counters
        # start at zero here and each ceiling gets one warning per job, so a
        # second save_video reports its own render rather than inheriting the
        # first one's totals and its already-spent warnings.
        reset_truncations()
        with torch.no_grad(), render_job_holding_the_arch():
            # Rendering is inference-only, but the scope is local to Algan so
            # importing the library does not alter PyTorch autograd globally.
            # ``no_grad`` rather than ``inference_mode``: a render that leaves
            # the Scene re-renderable (``save_video(reset=False)``) leaves
            # tensors behind that later authoring mutates in place, and an
            # inference tensor may not be mutated once the mode has exited.
            # The scene is excluded from garbage collection for the runtime:
            # the per-batch reclaim only ever needs to find the cycles this
            # render made, and walking the authored scene to find them cost
            # more than the reclaim saved (see scene_excluded_from_gc).
            try:
                from algan.rendering.camera_views import (
                    _live_views,
                    _render_with_camera_views,
                )

                views = _live_views(self)
                if views:
                    # Each pass owns the GC/arena scope. Nesting gc.freeze /
                    # unfreeze would unfreeze the outer scope prematurely.
                    yield from _render_with_camera_views(
                        self,
                        views,
                        start_time_ind,
                        end_time_ind,
                        background=background,
                        post_processes=post_processes,
                        manual_memory=manual_memory,
                        frame_indices=frame_indices,
                        **forwarded,
                    )
                    return
                with scene_excluded_from_gc():
                    yield from self._get_frames_impl(
                        start_time_ind,
                        end_time_ind,
                        background=background,
                        post_processes=post_processes,
                        manual_memory=manual_memory,
                        **(
                            {"frame_indices": frame_indices}
                            if frame_indices is not None
                            else {}
                        ),
                        **forwarded,
                    )
            finally:
                # _get_frames_impl has drained its prep worker before returning
                # here, including errors and abandoned generators. Release
                # render-local device geometry before memory-pressure cleanup.
                cache = self.__dict__.pop("_bezier_geometry_cache", None)
                if cache is not None:
                    cache.clear()
                self.__dict__.pop("_frame_local_batches", None)
                # Release the arena before the arch scope considers a deferred
                # compiler reset. WDDM charges GPU allocations against host
                # commit too; its freed storage must be reclaimable before the
                # runtime decides whether to throw away warm kernels.
                self.background_frame = original_background
                render_memory = self.memory
                if render_memory is not None and render_memory is not original_memory:
                    render_memory.data = None
                self.memory = original_memory

    def _get_frames_impl(
        self,
        start_time_ind,
        end_time_ind,
        background=None,
        post_processes=(bloom_filter,),
        manual_memory=True,
        *,
        frame_indices=None,
        aux_passes=False,
        aux_sink=None,
        independent_frames=False,
    ):
        # The aux passes' sink, or None when they are off (see get_frames).
        aux_sink = aux_sink if aux_passes else None
        if end_time_ind <= start_time_ind:
            if aux_sink is not None:
                from algan.rendering.raytracing.aux_passes import empty_aux_passes

                aux_sink(
                    empty_aux_passes(
                        0,
                        self.num_pixels_screen_height,
                        self.num_pixels_screen_width,
                    )
                )
            yield []
            return

        self.original_background_frame = self.background_frame
        if background is not None:
            self.background_frame = background

        transparent_background = self.background_is_transparent()
        if self.premultiplied_over and transparent_background:
            from algan.rendering.post_processing.post_process import (
                _validate_premultiplied_over,
            )

            _validate_premultiplied_over()
        self._warn_vertex_baked_lighting()

        for light in self.light_sources:
            light.is_primitive = True
        actors = [self.camera, self.camera.screen, *self.light_sources, *self.actors]
        save_image = False

        self.memory = ManualMemory(
            SETTINGS.computing.rendering_memory_fraction,
            managed=manual_memory,
        )
        # Safety margin learned from render failures this job: when a batch
        # that passed the arena preflight still fails to render, the preflight
        # under-estimated some allocation, so subsequent preflights must leave
        # at least this much slack (see _note_render_arena_underestimate).
        self._arena_unmodeled_bytes = 0
        # How that margin shrinks back (see _note_render_arena_success).
        self._arena_margin_clean_batches = 0
        self._arena_margin_decay_after = _ARENA_MARGIN_DECAY_BATCHES
        self._arena_margin_decayed = False
        self._arena_margin_persistent = False
        self._arena_other_bytes_at_start = _other_device_bytes(render_device())
        self._last_arena_preflight = None
        self._begin_batch_cost_measurement()
        # Frames the arena had room for in the last accepted batch, used to size
        # the next fetch (see fetch_end_for). None until a batch has been
        # measured: the first fetch of a job has nothing to go on.
        self._arena_fetch_frame_cap = None
        # Measured chunk-peak model, carried across every batch of this job so
        # only the first one pays to probe.
        self._chunk_memory_model = ChunkMemoryModel()
        # The merge and the projection build outside the arena, so the chunk
        # model cannot see them; their multipliers are measured from the builds
        # themselves, seeded by the previous guesses until one has run.
        self._merge_peak_ratio = PeakRatioModel(
            rt_settings_module.merge_gpu_peak_factor
        )
        self._project_peak_ratio = PeakRatioModel(
            rt_settings_module.project_gpu_peak_factor
        )

        # Adaptive gen-fused forecast (settings.wf_gen_fused == "auto") is fed
        # per-batch render timings below; a new job restarts its batch count.
        _rt_settings = SETTINGS.raytracing

        _rt_settings._begin_render_job()

        with self._batch_prep_context():
            current_time_ind = start_time_ind

            max_animate_mem = int(
                SETTINGS.computing.animation_memory_fraction
                * get_num_available_bytes(_ANIMATION_DEVICE)
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
            prefetch_enabled = env_flag("ALGAN_PREFETCH_BATCHES", True)
            # Grad mode is thread-local; mirror the caller's in the worker so
            # prep-created tensors carry the same autograd state the render
            # thread's do.
            grad_enabled = torch.is_grad_enabled()

            # Stills (see get_frames): batches end where each still's group
            # does, and preparation makes its choices per frame (read off the
            # Scene by the projection and the merge, worker included).
            self._frame_local_batches = bool(independent_frames)
            group_ends = (
                self._still_group_ends(
                    actors, frame_indices, start_time_ind, end_time_ind
                )
                if independent_frames
                and frame_indices is not None
                and self._stills_may_share_batches()
                else None
            )

            def group_end_for(time_ind):
                if group_ends is None:
                    return time_ind + 1
                return group_ends[bisect.bisect_right(group_ends, time_ind)]

            # Once stills turn out not to decide alike (see the check below),
            # fetch at most as many at a time as last agreed, doubling after
            # each run of agreeing batches -- a run that lengthens each time a
            # doubling fails. Stills that never agree (adaptive dicing under a
            # moving camera) then waste little on speculation, and stills that
            # agree again soon regain whole groups. None: whole groups.
            alike_hint = None
            alike_streak = 0
            alike_patience = 1

            def materialize_batch(time_ind, batch_end_ind):
                if independent_frames:
                    # Never past the end of time_ind's group (see get_frames):
                    # every fetch, retry and prefetch of this job comes
                    # through here.
                    batch_end_ind = min(batch_end_ind, group_end_for(time_ind))
                    if alike_hint is not None:
                        batch_end_ind = min(batch_end_ind, time_ind + alike_hint)
                while True:
                    try:
                        return self._get_batch_of_primitives(
                            time_ind,
                            batch_end_ind,
                            actors,
                            max_animate_mem,
                            **(
                                {"frame_indices": frame_indices}
                                if frame_indices is not None
                                else {}
                            ),
                        )
                    except (
                        InsufficientMemoryException,
                        OutOfRenderMemory,
                        MemoryError,
                        RuntimeError,
                    ) as prep_exc:
                        if not isinstance(
                            prep_exc,
                            (
                                InsufficientMemoryException,
                                OutOfRenderMemory,
                                MemoryError,
                            ),
                        ) and not is_cuda_oom(prep_exc):
                            raise
                        duration = batch_end_ind - time_ind
                        if duration <= 1:
                            raise
                        batch_end_ind = time_ind + max(1, duration // 2)
                    # Preparation allocates outside the arena: a dense
                    # texture query can OOM before the render retry ever
                    # sees a batch. Drop partial state and the exception's
                    # traceback before retrying, on the same prep worker.
                    self.timeline_manager.clear_buffers()
                    release_torch_memory(force_gc=True)
                    logger.log(
                        PERF,
                        "Batch preparation exceeded memory; retrying frames %s:%s.",
                        time_ind,
                        batch_end_ind,
                    )

            def prewarm_batch(batch, time_ind):
                """Run whatever of the batch's builds belong on this thread.

                Pre-run the ray tracer's vertex shade + packing
                (project_to_screen) and merged-scene / STBVH build here (all
                torch-only) so they ride the prefetch: batch b+1's prep runs on
                the worker while batch b renders, turning seconds of
                otherwise-serial render-thread CPU work into hidden time.
                ALGAN_PREFETCH_MERGE=0 falls back to projecting + merging on the
                render thread. When projection runs on the render device
                (settings.project_on_gpu) it is deferred to the render thread
                entirely -- GPU work on this worker would contend with the
                in-flight render and pollute the transient-peak stats -- so
                only the CPU-projection path prewarms here, unless
                prefetch-gpu-prep is on, in which case _prepare_batch_on_worker
                takes the GPU builds anyway (skipping their peak observations;
                see its docstring).

                The overlapped builds run on a full-window copy when the batch
                can be sliced, keeping the fetched batch pristine beside it: a
                window the arena then rejects is trimmed to a prefix on the
                render thread, exactly as un-overlapped batches are, instead of
                being refetched and rebuilt from scratch.

                A failed build defers to the render thread, which finishes
                whatever was left. Returns the batch to hand on, or ``None``
                when that is unsafe: a projection that raised part-way can leave
                its primitive half-transformed (vertex shading accumulates into
                ``colors`` in place; PN dicing and stroke expansion replace
                ``corners``), and projecting it again would apply those steps
                twice.
                """
                from algan.rendering.raytracing import settings as rt_settings

                source = None
                # The gated overlap is consulted first: it requires the GPU
                # builds itself (see _overlap_gpu_prep_active), so on a
                # CPU-projection render this falls through to the legacy
                # worker prewarm exactly as before.
                if self._overlap_gpu_prep_active():
                    build, what = self._prepare_batch_on_worker, "overlapped batch prep"
                    duration = batch[1] - time_ind
                    if self._can_slice_fetched_batch(batch[0], duration):
                        source = batch
                        built, built_state = self._slice_fetched_batch(
                            batch[0], batch[2], duration, duration
                        )
                        batch = (built, batch[1], built_state)
                elif not rt_settings.project_on_gpu_active():
                    build, what = self._prewarm_render_batch, "render-batch prewarm"
                else:
                    return batch
                try:
                    build(batch[0], batch[2])
                except Exception as e:
                    logger.warning(
                        f"{what} failed (deferring to the render thread): {e}"
                    )
                if source is not None:
                    # Anything short of a fully projected copy (declined,
                    # uncalibrated, interrupted) is discarded: the pristine
                    # batch then takes the un-overlapped path unchanged.
                    if not all(
                        getattr(primitive, "_rt_projected", False)
                        for primitive in batch[0]
                    ):
                        return source
                    batch[0][0]._rt_overlap_source = (source[0], source[2])
                    return batch
                if any(
                    getattr(primitive, "_rt_projection_interrupted", False)
                    for primitive in batch[0]
                ):
                    return None
                return batch

            def fetch_batch(time_ind, batch_end_ind=None):
                if batch_end_ind is None:
                    batch_end_ind = end_time_ind
                with torch.set_grad_enabled(grad_enabled):
                    batch = materialize_batch(time_ind, batch_end_ind)
                    if not batch[0] or not env_flag("ALGAN_PREFETCH_MERGE", True):
                        return batch
                    prepared = prewarm_batch(batch, time_ind)
                    if prepared is not None:
                        return prepared
                    # Rebuild the same window from the timeline rather than
                    # hand a half-projected batch on, and leave the fresh one
                    # wholly to the render thread.
                    batch_end_ind = batch[1]
                    del batch
                    self.timeline_manager.clear_buffers()
                    release_torch_memory(force_gc=True)
                    logger.log(
                        PERF,
                        "Batch projection was interrupted; rebuilding frames %s:%s.",
                        time_ind,
                        batch_end_ind,
                    )
                    return materialize_batch(time_ind, batch_end_ind)

            def fetch_end_for(time_ind):
                """End index to materialize up to, given what the arena took.

                The animation-device budget alone routinely asks for far more
                frames than the render arena can hold, and materializing them
                is the expensive half of a batch. Capping the request at the
                arena's own measured capacity (``_arena_fetch_frame_cap``,
                refreshed from every accepted batch) means the common case
                fetches once instead of fetching, being rejected, and
                rematerializing. The cap is a hint only: it grows straight back
                whenever the scene thins out, because the capacity it is
                refreshed from is measured against the whole arena rather than
                against the previous window.
                """
                cap = self._arena_fetch_frame_cap
                if not cap:
                    return end_time_ind
                return min(end_time_ind, time_ind + int(cap))

            executor = (
                ThreadPoolExecutor(max_workers=1, thread_name_prefix="algan-batch-prep")
                if prefetch_enabled
                else None
            )
            pending = None
            pending_end_ind = end_time_ind
            retry_end_ind = None
            retry_lower_duration = 0
            retry_upper_duration = None

            def drain_pending():
                """Finish and discard a prep the loop is about to invalidate.

                Batch preparation writes the shared timeline buffers, so only
                one may run at a time: anything that refetches on this thread
                waits here first.
                """
                nonlocal pending
                if pending is not None:
                    with contextlib.suppress(Exception):
                        pending.result()
                    pending = None

            try:
                while True:
                    _sync_devices()
                    s = time.time()
                    if retry_end_ind is not None:
                        drain_pending()
                        logger.debug(
                            f"Fetching batch {current_time_ind}:{retry_end_ind}."
                        )
                        primitives, new_time_ind, render_state = fetch_batch(
                            current_time_ind, retry_end_ind
                        )
                        retry_end_ind = None
                    elif pending is not None:
                        logger.debug(
                            f"Fetching batch {current_time_ind}:{pending_end_ind}."
                        )
                        primitives, new_time_ind, render_state = (
                            self._await_prefetched_batch(pending)
                        )
                        pending = None
                    else:
                        drain_pending()
                        fetch_end_ind = fetch_end_for(current_time_ind)
                        logger.debug(
                            f"Fetching batch {current_time_ind}:{fetch_end_ind}."
                        )
                        primitives, new_time_ind, render_state = fetch_batch(
                            current_time_ind, fetch_end_ind
                        )
                    _sync_devices()
                    e = time.time()
                    logger.debug("Batch fetch took %.2f seconds", e - s)

                    # NOTE: starting the successor's preparation here, before
                    # the arena preflight, was measured and is a LOSS (+15% on
                    # this project's reference scene). The preflight regularly
                    # shortens the window it was handed, and preparation writes
                    # the shared timeline buffers, so a speculative prep started
                    # at the wrong boundary cannot simply be abandoned -- the
                    # loop has to wait it out before it can prepare the right
                    # one. That serialized wasted prep costs far more than the
                    # preflight-length idle it was meant to fill.
                    # A fresh fetch selects a different actor set, and the
                    # actor set is what fixes the intercept of every cost the
                    # preflight weighs. Start its measurements over.
                    self._begin_batch_cost_measurement()
                    if new_time_ind <= current_time_ind:
                        raise OutOfRenderMemory(
                            "Insufficient memory to render this scene, "
                            "please reduce the number of Mobs used."
                            + path_tracer_fallback_hint()
                        )

                    duration = new_time_ind - current_time_ind
                    fetched_duration = duration
                    planned_prefix = None
                    # A batch the worker built under prefetch-gpu-prep is a
                    # full-window copy; the pristine fetched batch rides along
                    # so a window that does not fit can still be trimmed to a
                    # prefix here instead of being refetched.
                    overlap_source = (
                        getattr(primitives[0], "_rt_overlap_source", None)
                        if primitives
                        else None
                    )
                    if overlap_source is not None:
                        primitives[0]._rt_overlap_source = None
                    if (
                        retry_upper_duration is None
                        and primitives
                        and (
                            self._may_slice_across_spawns()
                            or self._fetched_window_has_stable_actor_set(
                                actors,
                                *_frame_window_bounds(
                                    current_time_ind, new_time_ind, frame_indices
                                ),
                            )
                        )
                    ):
                        known_failing = None
                        if overlap_source is not None:
                            # The worker's build is the same full-window probe
                            # the prefix search would start with.
                            if self._prepared_batch_fits_render_arena(
                                primitives,
                                render_state,
                                post_processes,
                                transparent_background,
                                require_estimates_fit=duration > 1,
                                num_frames=duration,
                            ):
                                planned_prefix = (primitives, duration, render_state)
                            else:
                                self._release_preflight_candidate(primitives)
                                primitives, render_state = overlap_source
                                known_failing = duration
                        if planned_prefix is None:
                            planned_prefix = (
                                self._select_largest_fitting_fetched_prefix(
                                    primitives,
                                    render_state,
                                    duration,
                                    post_processes,
                                    transparent_background,
                                    known_failing=known_failing,
                                )
                            )
                    # The pristine batch is needed only to choose the window;
                    # holding it through the render would keep its source
                    # geometry alive for nothing.
                    overlap_source = None

                    # Two budgets, one arena. What the frame count buys is
                    # bounded above by _batch_frame_capacity; what the batch's
                    # actor set costs regardless is _batch_actor_share, and no
                    # frame count touches it. When the actor set is what
                    # dominates, shortening the window relieves nothing --
                    # slicing keeps every actor the fetch selected, and even a
                    # rematerialized shorter window keeps them unless it also
                    # reaches back past a spawn. Retreat behind the last spawn
                    # instead: that is the only lever on the actor term.
                    actor_share = self._batch_actor_share()
                    spawn_boundary = None
                    if (
                        primitives
                        # Dense spawn-boundary indices cannot be used as
                        # offsets in a sparse selection. Its ordinary bounded
                        # prefix/OOM search still shrinks the selected window.
                        and frame_indices is None
                        and actor_share is not None
                        and actor_share >= _ACTOR_SHARE_RETREAT
                    ):
                        spawn_boundary = self._previous_spawn_boundary(
                            actors, current_time_ind, new_time_ind
                        )
                    if spawn_boundary is not None:
                        logger.debug(
                            "Batch is actor-bound (%.0f%% of a term's cost is "
                            "fixed by its actor set: %s); refetching %s:%s to "
                            "carry fewer actors.",
                            actor_share * 100,
                            self._describe_batch_costs(),
                            current_time_ind,
                            spawn_boundary,
                        )
                        if planned_prefix is not None:
                            self._release_preflight_candidate(planned_prefix[0])
                        retry_upper_duration = min(
                            duration,
                            retry_upper_duration
                            if retry_upper_duration is not None
                            else duration,
                        )
                        if primitives:
                            primitives[0]._rt_device_scene = None
                            primitives[0]._rt_prepared_host_scene = None
                            primitives[0]._rt_merged_scene = None
                        del primitives, planned_prefix
                        self.memory.reset()
                        release_torch_memory(force_gc=False)
                        retry_end_ind = spawn_boundary
                        continue

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
                                # Nothing smaller than one frame can be
                                # attempted, so estimates lose their vote here
                                # rather than aborting the render.
                                require_estimates_fit=duration > 1,
                                num_frames=duration,
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
                            raise OutOfRenderMemory(_one_frame_does_not_fit_message())
                        logger.warning(
                            "Prepared batch does not fit the render arena; "
                            "binary-searching the largest fitting runtime."
                        )
                        if primitives:
                            primitives[0]._rt_device_scene = None
                            primitives[0]._rt_prepared_host_scene = None
                            primitives[0]._rt_merged_scene = None
                        del primitives
                        self.memory.reset()
                        release_torch_memory(force_gc=False)
                        target_duration = self._next_probe_duration(
                            max(1, retry_lower_duration),
                            max(1, retry_upper_duration - 1),
                        )
                        retry_end_ind = current_time_ind + target_duration
                        continue

                    if independent_frames and duration > 1 and primitives:
                        # Stills share this batch only if each would make
                        # every batch-wide choice alike alone; otherwise
                        # refetch the leading frames that do (see get_frames).
                        agreeing = self._frames_deciding_alike(primitives, duration)
                        if agreeing < duration:
                            logger.debug(
                                "Stills %s:%s would render differently together; "
                                "keeping the first %s in this batch.",
                                current_time_ind,
                                new_time_ind,
                                agreeing,
                            )
                            if alike_hint is not None:
                                alike_patience *= 2
                            alike_hint = agreeing
                            alike_streak = 0
                            primitives[0]._rt_device_scene = None
                            primitives[0]._rt_prepared_host_scene = None
                            primitives[0]._rt_merged_scene = None
                            del primitives
                            self.memory.reset()
                            release_torch_memory(force_gc=False)
                            retry_end_ind = current_time_ind + agreeing
                            continue
                    if (
                        independent_frames
                        and alike_hint is not None
                        and duration >= alike_hint
                    ):
                        alike_streak += 1
                        if alike_streak >= alike_patience:
                            alike_streak = 0
                            alike_hint *= 2
                            if alike_hint >= _STILL_GROUP_MAX_FRAMES:
                                alike_hint = None

                    if retry_upper_duration is not None:
                        retry_lower_duration = max(retry_lower_duration, duration)
                        if False:  # retry_upper_duration - retry_lower_duration > 1:
                            # This candidate fits, but a failed upper bound
                            # leaves room to probe a larger prepared batch
                            # without emitting speculative frames.
                            if primitives:
                                primitives[0]._rt_device_scene = None
                                primitives[0]._rt_prepared_host_scene = None
                                primitives[0]._rt_merged_scene = None
                            del primitives
                            self.memory.reset()
                            release_torch_memory(force_gc=False)
                            target_duration = (
                                retry_lower_duration + retry_upper_duration
                            ) // 2
                            retry_end_ind = current_time_ind + target_duration
                            continue

                    # Carry the arena's verdict on this batch into the next
                    # fetch. Materializing a batch costs seconds, and without
                    # this every batch re-derives the same window from scratch:
                    # it fetches whatever the animation-device budget allows,
                    # the arena rejects it, and the timeline is rematerialized
                    # at half the size -- for every batch of the job, not just
                    # the first.
                    if primitives:
                        arena_frames = self._batch_frame_capacity() or 0
                        fitted = duration
                        if (
                            independent_frames
                            and retry_upper_duration is None
                            and duration == fetched_duration
                        ):
                            # A still batch ends where its group, or its frames
                            # deciding alike, do: not the arena's verdict, so
                            # it does not shrink the next fetch.
                            fitted = max(
                                duration,
                                self._arena_fetch_frame_cap or _STILL_GROUP_MAX_FRAMES,
                            )
                        # Never below what just fit: the estimate reads the
                        # scene's frame-independent bytes as if they scaled, so
                        # it under-shoots (harmlessly) on a batch that fit.
                        self._arena_fetch_frame_cap = _next_arena_fetch_cap(
                            fitted, arena_frames
                        )

                    # Only prefetch the successor once the current runtime is
                    # final. A speculative successor would start at the wrong
                    # boundary while the binary preflight search is active (and
                    # see the measurement note above the preflight).
                    if executor is not None and new_time_ind < end_time_ind:
                        pending_end_ind = fetch_end_for(new_time_ind)
                        pending = executor.submit(
                            fetch_batch, new_time_ind, pending_end_ind
                        )
                    if len(primitives) > 0:
                        s = time.time()
                        logger.debug(
                            "Rendering %.2f seconds of video.",
                            (new_time_ind - current_time_ind) / self.frames_per_second,
                        )
                        produced_output = False
                        retry_after_render_failure = False
                        try:
                            for frame_batch in self._render_primitive_batch(
                                primitives,
                                current_time_ind,
                                new_time_ind,
                                save_image,
                                post_processes,
                                transparent_background,
                                background,
                                render_state=render_state,
                                **(
                                    {"aux_sink": aux_sink}
                                    if aux_sink is not None
                                    else {}
                                ),
                            ):
                                produced_output = True
                                yield frame_batch
                        except (
                            InsufficientMemoryException,
                            OutOfRenderMemory,
                            RuntimeError,
                        ) as render_exc:
                            # A Taichi launch can OOM as a bare RuntimeError from
                            # its own allocator once the render chunk retry is
                            # exhausted; treat it as a render OOM here too (and
                            # re-raise any genuine, non-OOM RuntimeError).
                            if not isinstance(
                                render_exc,
                                (InsufficientMemoryException, OutOfRenderMemory),
                            ) and not is_cuda_oom(render_exc):
                                raise
                            # Frames already handed to the caller are final, so
                            # the retry resumes from the first one that is not
                            # rather than re-rendering the window from its
                            # start. Without this, a batch that emitted even
                            # one chunk could not be shrunk at all: the chunk
                            # split inside the tracer only shrinks the *trace*,
                            # while the batch's merged geometry -- the term
                            # that actually did not fit -- is fixed by the
                            # window. Every long batch densifying later in its
                            # window therefore reached "insufficient memory to
                            # ray trace a single frame" with the designed
                            # backstop standing down, which is exactly the case
                            # the backstop exists for. Measured on
                            # tests/full_renders' materials_and_lighting: a
                            # 123-frame window whose 625 MB scene left 234 MB
                            # of an 819 MB arena for the frame.
                            resume_ind = int(
                                getattr(
                                    self, "_last_emitted_time_ind", current_time_ind
                                )
                            )
                            resume_ind = max(current_time_ind, resume_ind)
                            if produced_output and resume_ind > current_time_ind:
                                remaining = new_time_ind - resume_ind
                                current_time_ind = resume_ind
                                duration = max(1, remaining)
                            elif produced_output:
                                # Nothing survived the batch, so there is no
                                # prefix to keep and no smaller window that
                                # this failure justifies.
                                raise
                            if duration <= 1:
                                raise
                            self._note_render_arena_underestimate()
                            # Not a failure: the model sizes a chunk from
                            # the batch's first frames and cannot see a
                            # scene that densifies later, so this retry is
                            # the designed backstop. Saying "failed" here,
                            # at WARNING, and repeating the exception's
                            # advice to lower the resolution made a healthy
                            # render look broken.
                            logger.log(
                                PERF,
                                "Frame batch did not fit; retrying "
                                f"{current_time_ind}:{new_time_ind} at half "
                                "runtime.",
                            )
                            # A prefetched successor starts at the old end and
                            # is invalid after this split. Drain and discard it
                            # before rematerializing the smaller current batch.
                            drain_pending()
                            if primitives:
                                primitives[0]._rt_device_scene = None
                                primitives[0]._rt_prepared_host_scene = None
                                primitives[0]._rt_merged_scene = None
                            del primitives
                            # The arena preflight approved this runtime and
                            # was wrong, so it cannot arbitrate durations just
                            # below it: binary-probing upward would converge to
                            # runtime-1, render-fail again, and repeat -- an
                            # O(frames) cascade of ~seconds-long refetches.
                            # Back off geometrically instead: cap the bounds at
                            # half so the halved candidate renders immediately,
                            # and let the raised preflight margin (see above)
                            # arbitrate everything afterwards.
                            retry_lower_duration = 0
                            retry_upper_duration = max(1, duration // 2)
                            retry_end_ind = current_time_ind + max(1, duration // 2)
                            retry_after_render_failure = True
                        if retry_after_render_failure:
                            # This deliberately runs after the exception handler:
                            # only then has Python released the exception state
                            # and traceback frames that may own CUDA tensors.
                            self._reset_render_arena_after_failure()
                            continue
                        self._note_render_arena_success()
                        # Drop this batch's arena-view caches before the arena
                        # is reset/reallocated: a rendered primitive's
                        # ``_rt_device_scene`` holds tensors carved from the
                        # render arena, and any surviving one keeps the whole
                        # arena buffer alive past teardown (``data = None`` only
                        # frees the buffer once no views reference it). The
                        # failure paths already clear these; do it on the
                        # success path too so the last batch never pins the
                        # arena into the next render/reset.
                        if primitives:
                            primitives[0]._rt_device_scene = None
                            primitives[0]._rt_prepared_host_scene = None
                            primitives[0]._rt_merged_scene = None
                        del primitives
                        # Free previous batch data before allocating next batch.
                        release_torch_memory(force_gc=False)
                        _sync_devices()
                        e = time.time()
                        logger.debug(
                            "Rendered frames %d:%d in %.2f seconds",
                            current_time_ind,
                            new_time_ind,
                            e - s,
                        )
                        if _rt_settings._note_batch_rendered(
                            new_time_ind - current_time_ind,
                            e - s,
                            end_time_ind - new_time_ind,
                        ):
                            logger.debug(
                                "Adaptive gen-fused: forecasted remaining "
                                "render time justifies compiling the fused "
                                "generation kernels; fusing from the next "
                                "batch (output is unaffected)."
                            )
                    else:
                        # No renderable primitives in this window (an empty
                        # scene, or a stretch where nothing is spawned): still
                        # emit background-only frames so the output video
                        # covers the window instead of silently dropping it.
                        logger.debug(
                            f"No active actors in {current_time_ind}:"
                            f"{new_time_ind}; rendering background only."
                        )
                        for frame_batch in self._render_background_batch(
                            current_time_ind,
                            new_time_ind,
                            post_processes=post_processes,
                            transparent_background=transparent_background,
                            background=background,
                            **(
                                {
                                    "frame_indices": frame_indices[
                                        current_time_ind:new_time_ind
                                    ]
                                }
                                if frame_indices is not None
                                else {}
                            ),
                            **({"aux_sink": aux_sink} if aux_sink is not None else {}),
                        ):
                            yield frame_batch

                    retry_lower_duration = 0
                    retry_upper_duration = None
                    current_time_ind = new_time_ind
                    if new_time_ind >= end_time_ind:
                        break
            finally:
                # Always drain the worker before leaving (normal completion,
                # error, or abandoned generator): a prep still running while
                # the caller resets or reuses the scene would race it.
                if pending is not None:
                    with contextlib.suppress(Exception):
                        pending.result()
                if executor is not None:
                    executor.shutdown(wait=True)
                # And always put the timeline back. Materializing points every
                # attribute's ``active_state`` at a buffer holding this window's
                # frames; leaving one behind makes the next read of any
                # attribute return a window of values where the caller expects
                # the authored one, which surfaces far from here as a shape
                # mismatch while constructing an unrelated Mob. This used to sit
                # above the ``finally``, so it ran only on normal completion --
                # and a caller that stops consuming part way through, which the
                # comment above already contemplates, left the scene
                # materialized. Clearing is idempotent.
                self.timeline_manager.clear_buffers()

    def _drain_video_writer(self, writer):
        """Flush the frame queue and wait for the encoder to finish.

        Split out so a profiler can time the serial video-encode tail (the
        block spent waiting on ffmpeg after the last frame is produced) as its
        own stage instead of leaving it in the profile's unaccounted bucket.
        """
        writer.finish()
        writer.file_writer.close()

    def _recorded_end_time_for_render(self):
        """End of everything recorded so far, in seconds.

        Ordinarily this is the root context's end, because a render happens
        with every block closed. Called from inside an unfinished block the
        active context is the innermost open one, whose window covers only its
        own block -- an enclosing :class:`~.Sync` can already hold animations
        running past it -- so the whole open chain is consulted. Every open
        context shares one timeframe (a ``runtime`` rescales its block
        retroactively, on exit), so their ends are directly comparable.
        """
        end = 0.0
        context = self.animation_manager.context
        while context is not None:
            end = max(end, context.timespan.original_end)
            context = context.prev_context
        return end

    def _render_to_video(
        self,
        file_writer,
        file_path,
        file_path_out,
        post_processes=(bloom_filter,),
        background=None,
        despawn_camera_and_lights=True,
        preserve_authoring_state=False,
        passes=None,
    ):
        """Stream rendered frame batches to the configured video writer.

        ``preserve_authoring_state`` is set for a render that leaves the Scene
        re-renderable (``save_video(reset=False)``): the frame window and the
        replay-window resolution the render derives are rolled back afterwards,
        so authoring can continue -- including inside a block that has not
        finished yet -- and render again. See
        :meth:`~algan.animation_timeline.timeline.AnimationTimeline.preserving_authoring_state`.

        ``passes`` is an :class:`algan._render_passes.VideoPassJob` when
        auxiliary passes were requested: its encoders run beside the video's
        and its sequences are published right after the video file is.
        """
        ensure_taichi_for_render()
        # The encoder still owns queued CPU frames after get_frames exits.
        # Keep deferred compiler reclamation outside the complete video job:
        # the implementation drains the writer, restores the timeline, and
        # releases its last frame locals before this outer arch scope exits.
        with render_job_holding_the_arch():
            return self._render_to_video_impl(
                file_writer,
                file_path,
                file_path_out,
                post_processes=post_processes,
                background=background,
                despawn_camera_and_lights=despawn_camera_and_lights,
                preserve_authoring_state=preserve_authoring_state,
                **({"passes": passes} if passes is not None else {}),
            )

    def _render_to_video_impl(
        self,
        file_writer,
        file_path,
        file_path_out,
        post_processes=(bloom_filter,),
        background=None,
        despawn_camera_and_lights=True,
        preserve_authoring_state=False,
        passes=None,
    ):
        previous_scene_times = (
            [list(pair) for pair in self.scene_times]
            if preserve_authoring_state
            else None
        )
        try:
            with torch.no_grad():
                self._stream_video_frames(
                    file_writer,
                    file_path,
                    post_processes,
                    background,
                    despawn_camera_and_lights,
                    preserve_authoring_state,
                    **({"passes": passes} if passes is not None else {}),
                )
        finally:
            if previous_scene_times is not None:
                self.scene_times[:] = previous_scene_times

        if os.path.exists(file_path_out):
            os.remove(file_path_out)
        os.rename(file_path, file_path_out)
        if passes is not None:
            # After the video, so the passes never describe a render whose
            # video failed to publish.
            passes.publish()

    def _static_frame_runs(self, start_ind, end_ind, background, post_processes):
        """Which of a video's frames to render, and how often to write each.

        A frame is a *repeat* when the recording proves its state equals the
        previous frame's, and is then written again instead of rendered again
        (``SETTINGS.raytracing.experimental.reuse_static_frames``). The test
        replicates, comparison for comparison, how materialization reads the
        recording at a frame's time (``time_ind / frames_per_second``, float32):

        * no function application or updater is active at either frame (the
          replay's own ``start <= t < end`` test, on the same float32 windows);
        * every attribute's base state has the same edit rank at both frames
          (``_query_row_states`` depends on time only through that rank);
        * every lifespan end-point mask is unchanged (``record_end_points``);
        * no actor spawns or despawns between them (the batch's actor set).

        Equal recorded state gives equal primitives and render state, and the
        deterministic renderer and per-frame post-processing then give equal
        pixels. Anything whose output can vary with the frame itself returns
        None (render every frame): the path tracer's per-frame sampling, a
        callable background, traced paths, live camera views, and
        post-processes from outside Algan.

        Returns
        -------
        tuple or None
            ``(frame_indices, repeats)``: the absolute frames to render, in
            order, and how many consecutive video frames each one fills.
        """
        total = int(end_ind) - int(start_ind)
        if not rt_settings_module.reuse_static_frames or total < 2:
            return None
        if int(rt_settings_module.samples_per_pixel) != 1:
            return None
        chosen_background = (
            background if background is not None else self.background_frame
        )
        if callable(chosen_background) and not torch.is_tensor(chosen_background):
            return None
        for process in post_processes or ():
            function = process.func if isinstance(process, partial) else process
            module = getattr(function, "__module__", "") or ""
            if not module.startswith("algan.rendering.post_processing"):
                return None
        from algan.rendering.camera_views import _live_views

        if _live_views(self):
            return None
        timeline = self.timeline_manager
        if len(timeline._traced_paths):
            return None
        timeline._resolve_replay_windows()

        # Exactly the render's frame times: _get_batch_of_primitives divides
        # the int64 frame indices by the frame rate.
        times = torch.arange(int(start_ind), int(end_ind)) / self.frames_per_second
        changed = torch.zeros(total, dtype=torch.bool)
        changed[0] = True

        def steps(per_frame):
            changed[1:] |= (per_frame[1:] != per_frame[:-1]).to(changed.device)

        def counts_at_or_before(values, at, right=True):
            dtype = torch.promote_types(values.dtype, at.dtype)
            ordered = values.reshape(-1).to(dtype).sort().values
            return torch.searchsorted(
                ordered, at.to(device=ordered.device, dtype=dtype), right=right
            ).cpu()

        # Animations and updaters: a frame at which one is active differs
        # from both of its neighbours. ``ends`` is floored at ``starts`` so an
        # inverted window counts as never active, as the replay test has it.
        functions = timeline.function_timeline
        windows = []
        if functions.function_applications:
            windows.append(functions._windows()[:2])
        if functions.updaters:
            windows.append(functions._updater_windows()[:2])
        for starts, ends in windows:
            if starts.numel() == 0:
                continue
            ends = torch.maximum(ends, starts)
            active = counts_at_or_before(starts, times) > counts_at_or_before(
                ends, times
            )
            changed |= active
            changed[1:] |= active[:-1]

        for attribute in timeline.attr_to_timeline.values():
            attribute.prepare_for_queries()
            if attribute._edits_sorted:
                index = attribute._prepared_queries(times)
                steps(
                    torch.searchsorted(
                        index.unique_timestamps,
                        times.to(index.unique_timestamps.device).contiguous(),
                        right=True,
                    ).cpu()
                )
            if attribute.record_end_points and attribute.pointer:
                attribute._refresh_end_points()
                bounds = attribute._end_points.reshape(-1, 2)
                # Rows whose ``start <= t`` / ``t < end`` flips between frames.
                steps(counts_at_or_before(bounds[:, 0], times))
                steps(counts_at_or_before(bounds[:, 1], times))

        # The batch's actor set, which _actors_in_window tests in float64.
        spawns, despawns = [], []
        for actor in [
            self.camera,
            self.camera.screen,
            *self.light_sources,
            *self.actors,
        ]:
            spawn = actor.lifespan.start()
            if spawn < 0:
                continue
            despawn = actor.lifespan.end()
            spawns.append(spawn)
            despawns.append(math.inf if despawn < 0 else despawn)
        if spawns:
            times64 = (
                torch.arange(int(start_ind), int(end_ind), dtype=torch.float64)
                / self.frames_per_second
            )
            steps(
                counts_at_or_before(torch.tensor(spawns, dtype=torch.float64), times64)
            )
            steps(
                counts_at_or_before(
                    torch.tensor(despawns, dtype=torch.float64), times64, right=False
                )
            )

        rendered = changed.nonzero().view(-1)
        if rendered.numel() == total:
            return None
        boundaries = torch.cat((rendered, torch.tensor([total])))
        repeats = (boundaries[1:] - boundaries[:-1]).tolist()
        frame_indices = tuple((rendered + int(start_ind)).tolist())
        return frame_indices, repeats

    def _stream_video_frames(
        self,
        file_writer,
        file_path,
        post_processes,
        background,
        despawn_camera_and_lights,
        preserve_authoring_state,
        passes=None,
    ):
        writer = None
        # The auxiliary-pass job (algan._render_passes.VideoPassJob), if any:
        # its per-pass encoders receive every frame the video does, repeats
        # included, from the batches get_frames delivers through its sink.
        pass_job = passes
        try:
            self.scene_times.append(
                [
                    self.scene_times[-1][0],
                    (
                        round(
                            self._recorded_end_time_for_render()
                            * self.frames_per_second
                        )
                    ),
                ]
            )
            self._initialize_frames()

            # Closing the camera/light lifespans is only meaningful when the scene
            # is being finalized; skipping it leaves the scene re-renderable after
            # a save_video(reset=False). Both lifespans stay open past the last
            # rendered frame index either way, so the output is unaffected.
            if despawn_camera_and_lights:
                self.camera.despawn(animate=False)
                for light in self.light_sources:
                    light.despawn(animate=False)

            if not self._scene_has_renderable_actors(*self.scene_times[-1]):
                warnings.warn(
                    "You are rendering an empty scene! Did you forget to spawn() your Mobs?",
                    EmptySceneWarning,
                    stacklevel=_user_stacklevel(),
                )

            self.file_path = file_path
            self.file_writer = file_writer

            writer = _VideoWriter(file_writer)
            self.frame_queue = writer.queue
            writer.start()
            if pass_job is not None:
                pass_job.open(self, _VideoWriter)
            # The snapshot is taken here rather than around the whole render call:
            # the fade-out and the zero-runtime guard record on the timeline
            # first, and edits made after a snapshot would fall outside it.
            preserve = (
                self.timeline_manager.preserving_authoring_state(
                    preserve_replay_resolution=(
                        self.animation_manager.context.prev_context is not None
                    )
                )
                if preserve_authoring_state
                else contextlib.nullcontext()
            )
            start_ind, end_ind = self.scene_times[-1]
            total_frames = max(1, end_ind - start_ind)
            with (
                preserve,
                _render_progress(total_frames) as report_frame,
                contextlib.ExitStack() as frame_cleanup,
            ):
                runs = self._static_frame_runs(
                    start_ind, end_ind, background, post_processes
                )
                repeats = None
                frame_window, sparse = tuple(self.scene_times[-1]), {}
                if runs is not None:
                    frame_indices, repeats = runs
                    frame_window = (0, len(frame_indices))
                    sparse = {"frame_indices": frame_indices}
                    logger.log(
                        PERF,
                        "Rendering %d of %d frames; the rest repeat the frame "
                        "before them.",
                        len(frame_indices),
                        end_ind - start_ind,
                    )
                # Explicitly close the generator on encoder failure too: its
                # finally releases materialization buffers and prep workers.
                aux = (
                    {}
                    if pass_job is None
                    else {"aux_passes": True, "aux_sink": pass_job.sink}
                )
                batches = self.get_frames(
                    *frame_window,
                    background=background,
                    post_processes=post_processes,
                    manual_memory=True,
                    **sparse,
                    **aux,
                )
                close_frames = getattr(batches, "close", None)
                if close_frames is not None:
                    frame_cleanup.callback(close_frames)
                from algan.animation_timeline.timeline import _opt_disabled

                rendered = 0
                for frame_batch in batches:
                    encoded = (
                        None if pass_job is None else pass_job.take(len(frame_batch))
                    )
                    for position, frame in enumerate(frame_batch):
                        copies = 1 if repeats is None else repeats[rendered]
                        rendered += 1
                        if _opt_disabled("writerrepeats"):
                            for _ in range(copies):
                                writer.put(frame)
                                if pass_job is not None:
                                    pass_job.put(encoded, position)
                                report_frame()
                            continue
                        writer.put(frame, copies)
                        if pass_job is not None:
                            pass_job.put(encoded, position, copies)
                        # After the put: the queue is bounded and feeds the
                        # encoder thread, so reporting first would run the
                        # progress ahead of the actual encode.
                        for _ in range(copies):
                            report_frame()
            self._drain_video_writer(writer)
            if pass_job is not None:
                pass_job.finish()
        except BaseException:
            if writer is not None:
                writer.abort()
            if pass_job is not None:
                pass_job.abort()
            try:
                file_writer.close()
            except Exception:
                # Cleanup must not replace an encoder/rendering error or a
                # cancellation. The worker is already joined before closing.
                logger.debug("Video writer cleanup failed", exc_info=True)
            raise
        finally:
            if pass_job is not None:
                pass_job.release(self)
