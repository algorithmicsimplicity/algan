"""The viewer's state: one Scene, one render worker, one lock.

Everything the browser asks for goes through a :class:`ViewerSession`. It owns
the Scene being inspected and serialises every touch of it, because two things
about Algan's renderer make concurrent access wrong rather than merely slow:

* A render binds Taichi's arch and allocates the render arena for its duration,
  so two renders at once fight over both.
* Reading an attribute "at time t" materializes the Scene's whole timeline at
  that time. It is global mutation, undone afterwards -- so a render sharing the
  Scene would read the wrong state.

So there is one worker thread that renders, one lock that every Scene access
takes, and HTTP handler threads that wait on results rather than computing them.

Frames are rendered lazily, a chunk at a time, and cached as encoded PNGs. A
seek does not cancel the chunk already running -- ``get_frames`` is a generator
and abandoning it mid-batch would waste the batch's materialization -- but it
does redirect what the worker renders next, so the wait is bounded by one chunk
rather than by the rest of the video.
"""

from __future__ import annotations

import io
import threading
import time

import torch

from algan.rendering import fragment_capture
from algan.settings.video_settings import PREVIEW
from algan.viewer import hierarchy
from algan.viewer.pixels import PixelRecord

#: How many frames one render call produces. A render call has real fixed cost
#: -- the arena, the batch cost models, the prefetch pool -- so asking for one
#: frame at a time pays it per frame; measured on a CPU container, eight frames
#: cost barely more than one.
CHUNK_FRAMES = 12

#: Stop prefetching once this many frames are cached, so a long video does not
#: fill memory with PNGs nobody scrolled to.
MAX_CACHED_FRAMES = 900


class ViewerSession:
    """A Scene, rendered on demand for a browser to page through."""

    def __init__(self, scene, video_settings=None):
        self.scene = scene
        self.video_settings = self._resolve_settings(scene, video_settings)
        self.fps = int(self.video_settings.frames_per_second)
        self.width, self.height = self.video_settings.resolution
        # Snapshotted, not followed: frames already rendered are cached, and a
        # Scene re-authored underneath a viewer would leave those frames showing
        # geometry that no longer exists. A second ``view()`` call is the way to
        # see later additions.
        self.duration = float(scene._recorded_end_time_for_render())
        # At least one frame: a scene authored but never advanced still has a
        # first frame to look at, and a zero-length scrubber is unusable.
        self.total_frames = max(1, round(self.duration * self.fps))

        self._lock = threading.RLock()
        self._cache: dict[int, bytes] = {}
        self._order: list[int] = []
        self._wanted = 0
        self._generation = 0
        self._error: str | None = None
        self._frame_ready = threading.Condition(self._lock)
        self._work = threading.Event()
        self._closed = False
        self._work.set()
        self._worker = threading.Thread(
            target=self._run, name="algan-viewer-render", daemon=True
        )
        self._worker.start()

    # -- settings ---------------------------------------------------------

    @staticmethod
    def _resolve_settings(scene, video_settings):
        """What to render at: small and quick, but on the Scene's own clock.

        The PREVIEW preset is 10 fps, and a viewer that renumbered a 30 fps
        scene's frames would report a frame index that does not exist in the
        video the script actually produces. So the resolution and supersampling
        come from PREVIEW and the frame rate stays the Scene's.
        """
        if video_settings is not None:
            return video_settings
        return PREVIEW.set(
            frames_per_second=int(scene.video_settings.frames_per_second)
        )

    # -- public surface ---------------------------------------------------

    def state(self):
        """Everything the page needs to lay itself out."""
        with self._lock:
            cached = sorted(self._cache)
            error = self._error
        return {
            "duration": self.duration,
            "fps": self.fps,
            "total_frames": self.total_frames,
            "width": self.width,
            "height": self.height,
            "resolution": list(self.video_settings.resolution),
            "supersampling": int(self.video_settings.supersampling),
            "cached": _ranges(cached),
            "cached_count": len(cached),
            "error": error,
        }

    def frame(self, index, timeout=120.0):
        """The PNG for a frame, rendering it first if it is not cached."""
        index = max(0, min(int(index), self.total_frames - 1))
        deadline = time.monotonic() + timeout
        with self._lock:
            png = self._cache.get(index)
            if png is not None:
                return png
            # Point the worker here ONCE. Re-aiming it on every wakeup would
            # bump the generation each time and the worker, seeing the target
            # move, would abandon its chunk before finishing a single frame.
            self._wanted = index
            self._generation += 1
        self._work.set()
        with self._lock:
            while True:
                png = self._cache.get(index)
                if png is not None:
                    return png
                if self._error:
                    raise RuntimeError(self._error)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"frame {index} was not rendered in time")
                self._frame_ready.wait(remaining)

    def prefetch(self, index):
        """Ask the worker to render around ``index`` without waiting for it."""
        with self._lock:
            self._wanted = max(0, min(int(index), self.total_frames - 1))
            self._generation += 1
        self._work.set()

    def time_of(self, index):
        """The timestamp a frame index lands on."""
        return float(index) / float(self.fps)

    def frame_of(self, seconds):
        """The frame index a timestamp lands on, clamped to the video."""
        return max(0, min(round(float(seconds) * self.fps), self.total_frames - 1))

    # -- hierarchy --------------------------------------------------------

    def roots(self):
        with self._lock:
            return hierarchy.roots(self.scene)

    def children(self, node, include_components=False):
        with self._lock:
            mob = hierarchy.index(self.scene).get(int(node))
            if mob is None:
                return None
            return hierarchy.children(mob, include_components)

    def attributes(self, node, frame=None):
        with self._lock:
            mob = hierarchy.index(self.scene).get(int(node))
            if mob is None:
                return None
            at = None if frame is None else self.time_of(int(frame))
            return {
                "node": int(node),
                "label": hierarchy.mob_label(mob),
                "type": type(mob).__name__,
                "at": at,
                "attributes": hierarchy.attributes_at(self.scene, mob, at),
            }

    # -- pixels -----------------------------------------------------------

    def pixel(self, frame, x, y):
        """The fragment list behind one pixel of one frame.

        Renders that frame again with the capture armed: the record only exists
        while the chunk that made it is in flight, so an inspection is a render
        rather than a lookup. Cheap in practice -- one frame, already warm.
        """
        frame = max(0, min(int(frame), self.total_frames - 1))
        x, y = int(x), int(y)
        if not (0 <= x < self.width and 0 <= y < self.height):
            return {"available": False, "reason": "outside the frame"}
        with self._lock:
            fragment_capture.arm()
            try:
                self._render_range(frame, frame + 1, store=False)
            finally:
                captures = fragment_capture.disarm()
            if not captures:
                return {
                    "available": False,
                    "frame": frame,
                    "x": x,
                    "y": y,
                    "reason": (
                        "This frame produced no per-pixel record. Nothing covers "
                        "it, or the render took a route that does not build one "
                        "(more than one sample per pixel, or a near clip plane)."
                    ),
                }
            by_id = hierarchy.mob_by_timeline_id(self.scene)
            fragments = []
            raw = 0
            for capture in captures:
                record = PixelRecord(capture, by_id)
                if record.width != self.width or record.height != self.height:
                    continue
                for frame_rel in _frame_rels(record):
                    found = record.fragments(x, y, frame_rel)
                    if found:
                        fragments.extend(found)
                        raw += record.raw_fragment_count(x, y, frame_rel)
            self._annotate(fragments, frame)
        return {
            "available": True,
            "frame": frame,
            "time": self.time_of(frame),
            "x": x,
            "y": y,
            "fragments": fragments,
            "raw_fragments": raw,
        }

    def _annotate(self, fragments, frame):
        """Fill in each fragment's owning Mob's authored colour at this frame.

        The colour fetch's own fifth lane is not the opacity a user authored, so
        the number the panel labels as opacity comes from the timeline instead.
        Read once for the whole list, since materializing is not cheap.
        """
        wanted = {f["mob_id"] for f in fragments if f.get("mob_id") is not None}
        if not wanted:
            return
        index = hierarchy.mob_by_timeline_id(self.scene)
        colours = {}
        # One materialization for every mob in the list, not one each: holding
        # the timeline at a time is the expensive half of the read.
        with hierarchy.materialized(self.scene, self.time_of(frame)):
            for mob_id in wanted:
                mob = index.get(mob_id)
                if mob is None:
                    continue
                for row in hierarchy.attributes_of(self.scene, mob):
                    if row["name"] == "color" and row["value"]:
                        colours[mob_id] = row["value"]
        for fragment in fragments:
            colour = colours.get(fragment.get("mob_id"))
            if colour and len(colour) >= 5:
                fragment["mob_color"] = colour[:3]
                fragment["opacity"] = colour[4]
            else:
                fragment["mob_color"] = None
                fragment["opacity"] = None

    # -- rendering --------------------------------------------------------

    def _run(self):
        while not self._closed:
            self._work.wait()
            self._work.clear()
            if self._closed:
                return
            try:
                # Keep going while there is anything left to render, so one
                # nudge from the page turns into a run of chunks ahead of it.
                while not self._closed and self._render_next():
                    pass
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    self._error = f"{type(exc).__name__}: {exc}"
                    self._frame_ready.notify_all()

    def _render_next(self):
        """Render one chunk from where the page is looking. False when idle."""
        with self._lock:
            start = self._next_gap()
            if start is None:
                return False
            # A frame someone is waiting for is rendered whatever the cache
            # holds; running ahead of them is what stops at the cap.
            if start != self._wanted and len(self._cache) >= MAX_CACHED_FRAMES:
                return False
            generation = self._generation
            end = min(start + CHUNK_FRAMES, self.total_frames)
            self._render_range(start, end, store=True, generation=generation)
        return True

    def _next_gap(self):
        """The first uncached frame at or after the playhead, else anywhere."""
        for index in range(self._wanted, self.total_frames):
            if index not in self._cache:
                return index
        for index in range(0, self._wanted):
            if index not in self._cache:
                return index
        return None

    def _render_range(self, start, end, *, store, generation=None):
        """Render ``[start, end)`` and, if asked, cache the PNGs.

        Runs with the lock held. ``preserving_authoring_state`` is what keeps
        the Scene re-renderable: a render resolves the timeline's replay windows
        into fixed timestamps, and leaving those behind makes every later render
        stop its animations early.
        """
        scene = self.scene
        previous = scene.video_settings
        explicit = getattr(scene, "_video_settings_explicit", False)
        index = start
        try:
            if self.video_settings is not previous:
                scene.set_video_settings(self.video_settings)
            preserve = scene.animation_manager.context.prev_context is not None
            with (
                torch.inference_mode(),
                scene.timeline_manager.preserving_authoring_state(
                    preserve_replay_resolution=preserve
                ),
            ):
                for batch in scene.get_frames(start, end):
                    for row in range(batch.shape[0]):
                        if store:
                            self._store(index, batch[row])
                        index += 1
                    if self._closed:
                        # Shutting down. Leave the generator here; closing it
                        # runs its own cleanup, and the enclosing context
                        # managers still restore the Scene on the way out.
                        break
                    if generation is not None and self._generation != generation:
                        # The page moved. Finish this batch -- it is already
                        # materialized -- but do not start the next one.
                        break
        finally:
            if scene.video_settings is not previous:
                scene.set_video_settings(previous, _explicit=explicit)
            scene._video_settings_explicit = explicit

    def _store(self, index, frame):
        from PIL import Image

        buffer = io.BytesIO()
        Image.fromarray(frame.contiguous().numpy()).save(buffer, format="PNG")
        self._cache[index] = buffer.getvalue()
        self._order.append(index)
        while len(self._order) > MAX_CACHED_FRAMES:
            self._cache.pop(self._order.pop(0), None)
        self._frame_ready.notify_all()

    def close(self, timeout=30.0):
        """Stop the worker and wait for it to actually leave the Scene alone.

        Waiting is the point. A render materializes the Scene's timeline at the
        frames it is drawing and restores it when it finishes; a worker still
        running after the viewer is gone would leave the Scene materialized
        under whoever authors next, which shows up as shape mismatches in code
        nowhere near here.
        """
        self._closed = True
        self._work.set()
        worker = self._worker
        if worker is not None and worker.is_alive():
            worker.join(timeout)
        self._worker = None


def _frame_rels(record):
    """The chunk-relative frame indices a capture holds."""
    return [f - record.time_start for f in record.frames] or [0]


def _ranges(indices):
    """Collapse sorted frame indices into ``[start, end]`` pairs for the page."""
    out = []
    for index in indices:
        if out and index == out[-1][1] + 1:
            out[-1][1] = index
        else:
            out.append([index, index])
    return out
