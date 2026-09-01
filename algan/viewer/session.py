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
What that lock must *not* do is stand between the page and answers it could have
had without touching the Scene at all, which is what the next paragraph is about.

There are two locks, not one, and the split is the difference between a viewer
that answers and one that does not. ``_scene_lock`` guards the Scene itself and
is held for as long as a render or a materialized read takes. ``_lock`` guards
only this object's own bookkeeping -- the frame cache, the playhead, the error --
and is never held across anything slow. Routes that need no Scene access
(``/api/state``, a cached ``/frame/N.png``) therefore answer immediately even
while a chunk is rendering.

The worker is also the *lowest-priority* user of the Scene. Python locks are not
fair: a worker that releases the Scene lock at the end of a chunk and re-takes it
at the top of the next one wins that race against a request that has been waiting
since before the chunk started, and can keep winning for the whole video. So a
request announces itself in ``_scene_demand`` before it queues, and the worker
stands aside while that count is non-zero and abandons the chunk it is in at the
next batch boundary. A request waits for the batch already in flight, not for the
rest of the video.

Frames are rendered lazily, a chunk at a time, and cached as encoded PNGs. A
seek does not cancel the chunk already running -- ``get_frames`` is a generator
and abandoning it mid-batch would waste the batch's materialization -- but it
does redirect what the worker renders next, so the wait is bounded by one chunk
rather than by the rest of the video.
"""

from __future__ import annotations

import contextlib
import io
import threading
import time

import torch

from algan.rendering import fragment_capture
from algan.settings.video_settings import (
    HD,
    LD,
    MD,
    PREVIEW,
    PRODUCTION,
    SMOKE_TEST,
    THUMBNAIL,
    UHD,
)
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

#: The built-in presets the resolution picker offers, in the order it shows
#: them: smallest first, so the cheap ones to render are the easy ones to reach.
BUILT_IN_PRESETS = (
    ("SMOKE_TEST", SMOKE_TEST),
    ("THUMBNAIL", THUMBNAIL),
    ("PREVIEW", PREVIEW),
    ("LD", LD),
    ("MD", MD),
    ("HD", HD),
    ("PRODUCTION", PRODUCTION),
    ("UHD", UHD),
)

#: How many finished pixel inspections to keep. Each is a small dict, and
#: keeping them is what makes the page's poll for a slow one free once it lands.
MAX_CACHED_PIXELS = 256


class ViewerSession:
    """A Scene, rendered on demand for a browser to page through."""

    def __init__(self, scene, video_settings=None):
        self.scene = scene
        self._options, self._current_option = self._build_options(scene, video_settings)
        self.video_settings = self._options[self._current_option][1]
        self.fps = int(self.video_settings.frames_per_second)
        self.width, self.height = self.video_settings.resolution
        # Snapshotted, not followed: frames already rendered are cached, and a
        # Scene re-authored underneath a viewer would leave those frames showing
        # geometry that no longer exists. A second ``Scene.view()`` call is the
        # way to see later additions.
        self.duration = float(scene._recorded_end_time_for_render())
        # At least one frame: a scene authored but never advanced still has a
        # first frame to look at, and a zero-length scrubber is unusable.
        self.total_frames = max(1, round(self.duration * self.fps))

        # Bookkeeping only, and never held across a render: see the module
        # docstring for why the two locks are not one.
        self._lock = threading.RLock()
        # The Scene: renders, materialized reads, fragment captures.
        self._scene_lock = threading.RLock()
        #: How many threads are queued for ``_scene_lock``. Guarded by ``_lock``.
        self._scene_demand = 0
        self._cache: dict[int, bytes] = {}
        self._order: list[int] = []
        self._wanted = 0
        self._generation = 0
        self._error: str | None = None
        self._frame_ready = threading.Condition(self._lock)
        self._scene_free = threading.Condition(self._lock)
        #: Finished inspections, keyed by ``(frame, x, y)``, and the keys of the
        #: ones still being computed. Both guarded by ``_lock``.
        self._pixels: dict[tuple[int, int, int], dict] = {}
        self._pixel_order: list[tuple[int, int, int]] = []
        self._pixel_jobs: set[tuple[int, int, int]] = set()
        self._pixel_ready = threading.Condition(self._lock)
        #: Bumped whenever the render resolution changes. Everything cached
        #: before a bump is of the wrong size, and the page puts it in the frame
        #: URL so the browser's own HTTP cache cannot serve a stale PNG either.
        self._epoch = 0
        self._work = threading.Event()
        self._closed = False
        self._work.set()
        self._worker = threading.Thread(
            target=self._run, name="algan-viewer-render", daemon=True
        )
        self._worker.start()

    # -- settings ---------------------------------------------------------

    @staticmethod
    def _build_options(scene, video_settings):
        """Every resolution the picker offers, and which one to start on.

        **Every option carries the same frame rate**, so the picker changes the
        size of a frame and never which frame an index names. The presets come
        with frame rates of their own and adopting one would renumber the whole
        video -- PREVIEW is 10 fps, so a 30 fps Scene viewed at PREVIEW's clock
        would report frame indices that do not exist in the video the script
        produces. A preset therefore contributes its *resolution* and
        supersampling only. The rate itself is the one the session opened on:
        whatever ``view()`` was given, or else the Scene's.

        The Scene's own settings and the ones ``view()`` was given are listed
        even when they duplicate a preset's size -- they are the two the user
        named, and being able to pick them by name is the point.

        Returns ``({name: (title, settings)}, starting name)``.
        """
        opened_on = (
            video_settings if video_settings is not None else scene.video_settings
        )
        fps = int(opened_on.frames_per_second)
        options = {
            name: (name, preset.set(frames_per_second=fps))
            for name, preset in BUILT_IN_PRESETS
        }
        options["SCENE"] = ("Scene", scene.video_settings.set(frames_per_second=fps))
        if video_settings is not None:
            options["VIEW"] = ("View", video_settings)
            return options, "VIEW"
        return options, "PREVIEW"

    def resolution_options(self):
        """The picker's rows: a name to post back, and a label to show.

        Labelled ``(height, width)``, which is the order asked for -- note it is
        the reverse of ``VideoSettings.resolution``, which is ``(width, height)``.
        """
        rows = []
        for name, (title, settings) in self._options.items():
            width, height = settings.resolution
            rows.append(
                {
                    "name": name,
                    "label": f"{title}: ({height}, {width})",
                    "width": width,
                    "height": height,
                }
            )
        return rows

    def set_resolution(self, name):
        """Re-render everything at another of the offered resolutions.

        Returns the new state, or ``None`` if there is no such option.

        Takes the Scene lock, so it waits out the batch in flight rather than
        swapping the size under a render that has already read it. Everything
        cached is then wrong by definition and goes: the frames, and the pixel
        inspections whose coordinates were in the old frame's grid.
        """
        key = str(name).upper()
        entry = self._options.get(key)
        if entry is None:
            return None
        settings = entry[1]
        with self._scene(), self._lock:
            if key != self._current_option:
                self._current_option = key
                self.video_settings = settings
                self.width, self.height = settings.resolution
                # Invariant, not an update: every option is built with the
                # session's own frame rate, so the playhead keeps its meaning
                # across a change of size. Recomputed anyway so that an option
                # that ever did carry another rate could not slip through.
                self.fps = int(settings.frames_per_second)
                self.total_frames = max(1, round(self.duration * self.fps))
                self._wanted = min(self._wanted, self.total_frames - 1)
                self._cache.clear()
                self._order.clear()
                self._pixels.clear()
                self._pixel_order.clear()
                self._epoch += 1
                # Stop the chunk in flight at its next batch: it is drawing the
                # old size into a cache that no longer wants it.
                self._generation += 1
                self._frame_ready.notify_all()
                self._pixel_ready.notify_all()
        self._work.set()
        return self.state()

    # -- scene access -----------------------------------------------------

    @contextlib.contextmanager
    def _scene(self):
        """Hold the Scene for the block, announcing the wait before queueing.

        ``_scene_demand`` counts *waiters*, not holders, so it drops back to
        zero the moment this thread is served. The render worker consults it
        before starting a chunk and between batches within one, which is what
        stops a request from losing the lock race indefinitely.
        """
        with self._lock:
            self._scene_demand += 1
        try:
            self._scene_lock.acquire()
        finally:
            with self._lock:
                self._scene_demand -= 1
                self._scene_free.notify_all()
        try:
            yield
        finally:
            self._scene_lock.release()

    def _scene_is_wanted(self):
        """Is anything queued for the Scene behind the holder of it?"""
        with self._lock:
            return self._scene_demand > 0

    def _stand_aside(self, timeout=30.0):
        """Wait for the queue to clear before taking the Scene speculatively.

        The timeout is a backstop, not a schedule: a page that somehow asked
        without pause would otherwise stop the worker rendering entirely, and
        a viewer that never renders another frame is worse than one that
        answers a request a batch late.
        """
        deadline = time.monotonic() + timeout
        with self._lock:
            while self._scene_demand and not self._closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return
                self._scene_free.wait(remaining)

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
            "epoch": self._epoch,
            "resolution_name": self._current_option,
            "resolution_options": self.resolution_options(),
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
        with self._scene():
            return hierarchy.roots(self.scene)

    def children(self, node, include_components=False):
        with self._scene():
            mob = hierarchy.index(self.scene).get(int(node))
            if mob is None:
                return None
            return hierarchy.children(mob, include_components)

    def attributes(self, node, frame=None):
        with self._scene():
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

    def pixel(self, frame, x, y, wait=3.0):
        """The fragment list behind one pixel of one frame.

        Computed on a thread of its own and reported back over however many
        requests it takes, rather than by holding one request open until it is
        done. That is not a style preference: the *first* inspection of a
        session compiles a Taichi kernel variant for the capture-armed render
        path, which was measured at **12 s with an idle worker and 67 s with one
        still rendering**, and a browser asked to wait that long for a single
        response gives up and reports it to the page as ``TypeError: Failed to
        fetch`` -- discarding an answer that was on its way.

        So this waits ``wait`` seconds, which is long enough that a warm
        inspection (~2 s) still answers in one round trip, and otherwise returns
        ``{"pending": True}`` for the page to poll on. Results are cached by
        ``(frame, x, y)``, so the poll that finally lands costs nothing and
        re-inspecting a pixel is free.
        """
        frame = max(0, min(int(frame), self.total_frames - 1))
        x, y = int(x), int(y)
        if not (0 <= x < self.width and 0 <= y < self.height):
            return {"available": False, "reason": "outside the frame"}
        key = (frame, x, y)
        deadline = time.monotonic() + float(wait)
        with self._lock:
            while True:
                done = self._pixels.get(key)
                if done is not None:
                    return done
                if key not in self._pixel_jobs:
                    self._pixel_jobs.add(key)
                    threading.Thread(
                        target=self._compute_pixel,
                        args=(key, self._epoch),
                        name=f"algan-viewer-pixel-{frame}",
                        daemon=True,
                    ).start()
                remaining = deadline - time.monotonic()
                if remaining <= 0 or self._closed:
                    return {"pending": True, "frame": frame, "x": x, "y": y}
                self._pixel_ready.wait(remaining)

    def _compute_pixel(self, key, epoch):
        """Run one inspection and publish it, whether it works or not.

        An exception here has to become a *result*, not a lost thread: the page
        is polling for this key and would otherwise poll until it gave up, with
        nothing to show for it.

        ``epoch`` is the resolution this was started under. If it has moved on,
        the answer describes a frame of a different size and is dropped -- the
        waiter then finds neither a result nor a job and starts a fresh one.
        """
        try:
            payload = self._inspect_pixel(*key)
        except Exception as exc:  # noqa: BLE001 -- reported to the page as-is
            payload = {
                "available": False,
                "frame": key[0],
                "x": key[1],
                "y": key[2],
                "reason": f"{type(exc).__name__}: {exc}",
            }
        with self._lock:
            if epoch == self._epoch:
                self._pixels[key] = payload
                self._pixel_order.append(key)
                while len(self._pixel_order) > MAX_CACHED_PIXELS:
                    self._pixels.pop(self._pixel_order.pop(0), None)
            self._pixel_jobs.discard(key)
            self._pixel_ready.notify_all()

    def _inspect_pixel(self, frame, x, y):
        """The inspection itself: one capture-armed render, read at one pixel."""
        with self._scene():
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
        # Outside the bookkeeping lock, and behind anything already queued for
        # the Scene: rendering ahead is speculative, and a request is not.
        self._stand_aside()
        if self._closed:
            return False
        with self._scene():
            self._render_range(
                start, end, store=True, generation=generation, yielding=True
            )
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

    def _render_range(self, start, end, *, store, generation=None, yielding=False):
        """Render ``[start, end)`` and, if asked, cache the PNGs.

        Runs with ``_scene_lock`` held. ``preserving_authoring_state`` is what
        keeps the Scene re-renderable: a render resolves the timeline's replay
        windows into fixed timestamps, and leaving those behind makes every
        later render stop its animations early.

        ``yielding`` is for the worker's speculative chunks: it gives up the
        rest of the chunk once something is queued for the Scene, so a request
        waits for the batch in flight rather than for the chunk. A caller that
        is itself serving a request (``pixel``) leaves it off -- it is the one
        being waited for, and abandoning its own render would answer nothing.
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
                    if yielding and self._scene_is_wanted():
                        # Somebody is queued behind this chunk. Frames dropped
                        # here are not lost: ``_next_gap`` finds them again on
                        # the next pass, and the wait to be served is now one
                        # batch rather than the rest of the video.
                        break
        finally:
            if scene.video_settings is not previous:
                scene.set_video_settings(previous, _explicit=explicit)
            scene._video_settings_explicit = explicit

    def _store(self, index, frame):
        from PIL import Image

        buffer = io.BytesIO()
        # Encoded outside the lock: PNG compression is the slow part, and
        # holding the bookkeeping lock through it would stall ``/api/state``
        # for exactly the reason the two locks exist.
        Image.fromarray(frame.contiguous().numpy()).save(buffer, format="PNG")
        png = buffer.getvalue()
        with self._lock:
            self._cache[index] = png
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
        with self._lock:
            # A worker parked in ``_stand_aside`` is not waiting on ``_work``,
            # and a request parked on an inspection needs to be let go too.
            self._scene_free.notify_all()
            self._frame_ready.notify_all()
            self._pixel_ready.notify_all()
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
