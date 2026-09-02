"""GPU memory arena and accounting for the render loop.

:class:`ManualMemory` is a bump-allocator arena for render-time tensors. Callers
snapshot the allocation pointer, allocate freely, and restore it to free
everything since the snapshot -- deterministic and far cheaper than relying on
the caching allocator across a frame batch.

The rest of the module is the accounting that keeps a render inside its budget:
available-byte queries, ``ensure_render_headroom``, CUDA peak-tracking scopes,
and :class:`InsufficientMemoryException` / ``is_cuda_oom`` for the retry path that
shrinks the frame window when a batch does not fit.

:class:`AllocationRecorder` and ``ManualMemory.scope()`` are **diagnostics
only** -- they attribute arena usage per stage when you are investigating, and do
not participate in batch sizing. That is
:mod:`algan.rendering.memory_model`'s job, which measures the arena's actual
high-water mark rather than modelling it.
"""

from __future__ import annotations

import gc
import sys
from contextlib import contextmanager

import torch

from algan.constants.math import GIGABYTES
from algan.environment import env_int
from algan.settings import SETTINGS
from algan.settings._startup import render_device


class InsufficientMemoryException(Exception):
    pass


def is_cuda_oom(exc):
    """True if ``exc`` is an out-of-memory failure from *either* GPU allocator.

    PyTorch raises :class:`torch.OutOfMemoryError`, but Taichi kernel launches
    allocate from their own CUDA pool (``cuMemAllocAsync``) and surface
    exhaustion as a plain :class:`RuntimeError` wrapping the driver string
    (``CUDA_ERROR_OUT_OF_MEMORY``). The render arena bump-allocator never hits
    the driver mid-render, so a batch that over-committed VRAM fails *inside a
    Taichi launch* (typically the post-process tonemap) rather than as a torch
    OOM -- and the retry loops, which only knew the torch type, let it escape.
    Matching the driver message lets the same ``release_torch_memory`` + window-split
    retry recover it (``torch.cuda.empty_cache`` hands torch's reserved-but-free
    blocks back to the driver, which is exactly the memory Taichi needs).
    """
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        return (
            "out of memory" in msg
            or "cuda_error_out_of_memory" in msg
            or "cudaerrormemoryallocation" in msg
        )
    return False


def get_num_available_bytes(device=torch.device("cuda")):
    device = torch.device(device)
    # A pinned figure stands in for the *measured* branches only. The CPU
    # branch already returns a setting, so it is deterministic as it is, and
    # routing it through this override too would silently retune animation
    # batch sizing along with the render arena.
    override = SETTINGS.computing.available_memory_override
    if override is not None and device.type in ("cuda", "mps"):
        return int(override)
    if device.type == "cuda":
        # ``release_torch_memory`` acts on PyTorch's current CUDA device.  The render
        # arena may target a different indexed device, so make that device
        # current while reclaiming its cached blocks before measuring it.
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
            free_bytes, _ = torch.cuda.mem_get_info(device)
        return free_bytes
    elif device.type == "mps":
        allocated_bytes = torch.mps.driver_allocated_memory()
        total_bytes = torch.mps.recommended_max_memory()
        free_bytes = total_bytes - allocated_bytes
        free_bytes = min(free_bytes, 1 * GIGABYTES)
        return free_bytes
    else:
        return SETTINGS.computing.max_cpu_memory_used


def _gpu_memory_pressure(threshold=0.8):
    """True when the CUDA device is using more than ``threshold`` of its memory
    (driver-level, so it accounts for Taichi + torch + everything).
    """
    if not torch.cuda.is_available():
        return True  # No CUDA telemetry; keep the original (always-gc) behavior.
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        return (total_bytes - free_bytes) > threshold * total_bytes
    except Exception:
        return True


#: Reclaimable torch cache below which a steady-state ``release_torch_memory`` call is
#: not worth its cost (see :func:`release_torch_memory`). One HD frame buffer's worth.
_MIN_RECLAIMABLE_BYTES = 128 << 20


def _reclaimable_cuda_bytes():
    """Bytes torch is holding cached and not currently using, or 0 off CUDA."""
    if not torch.cuda.is_available():
        return 0
    try:
        return int(torch.cuda.memory_reserved()) - int(torch.cuda.memory_allocated())
    except Exception:
        return 0


def release_torch_memory(force_gc=True):
    """Reclaim freed memory back to the allocators.

    ``gc.collect()`` walks the entire Python object graph and dominates this
    call (~0.2s each on a large scene; it was costing ~40% of a small render
    when called several times per frame batch). It is only needed to break
    *reference cycles* -- reference counting already frees the (explicitly
    nulled) geometry tensors immediately -- so it is skipped unless the GPU is
    actually under memory pressure (where reclaiming cyclic garbage matters for
    avoiding OOM) or ``force_gc`` is set. A render additionally freezes the
    authored scene out of collection entirely (:func:`scene_excluded_from_gc`),
    which is what makes the surviving collections cheap.

    ``torch.cuda.empty_cache()`` is *not* cheap: it drains the device and hands
    every cached block back to the driver, which on Windows/WDDM costs tens of
    milliseconds per call whether or not there is anything to hand back
    (measured at ~79 ms a call, 33 s of a four-minute render, once the gc above
    stopped dominating it). It is therefore gated on the same memory pressure
    as the collection, plus a worthwhile amount actually being reclaimable --
    or on the caller forcing it, which every failure/retry path does. This is
    self-regulating: a cache left unreclaimed shows up as *driver-level* used
    memory, so it raises the pressure that triggers the next reclaim.

    Sizing decisions are unaffected either way:
    :func:`get_num_available_bytes` reclaims unconditionally before it
    measures, so every batch and chunk still sees the same free-byte figure.
    """
    pressured = force_gc or _gpu_memory_pressure()
    if pressured:
        gc.collect()
    if (
        torch.cuda.is_available()
        and pressured
        and (force_gc or _reclaimable_cuda_bytes() >= _MIN_RECLAIMABLE_BYTES)
    ):
        torch.cuda.empty_cache()
    if torch.mps.is_available():
        torch.mps.empty_cache()


@contextmanager
def scene_excluded_from_gc():
    """Keep the authored scene out of every collection made inside the block.

    ``release_torch_memory`` runs ``gc.collect()`` several times per frame batch to break
    the reference cycles a batch leaves behind before the device runs out of
    memory, and a collection walks *every* tracked object in the process. An
    authored scene is millions of them -- one per Mob, per recorded edit, per
    retained tensor -- all live from the first frame to the last, so each
    collection re-walked the whole scene to find the handful of cycles the batch
    actually produced. Measured on this project's reference scene: 0.24 s per
    call, ~100 s of a four-minute render, every second of it holding the GIL
    against the batch-prep worker that is supposed to be running concurrently.

    ``gc.freeze()`` moves everything that already exists into a permanent
    generation collections skip, so the render's collections walk only what the
    render itself allocated. Nothing leaks: the frozen objects are the scene,
    which is live throughout the render, and ``gc.unfreeze()`` returns them to
    the ordinary generations on the way out (including on an error, so a failed
    render does not leave the process with collection disabled for its scene).
    The one collection before freezing keeps pre-existing garbage collectable.
    """
    gc.collect()
    gc.freeze()
    try:
        yield
    finally:
        gc.unfreeze()


def ensure_render_headroom(device, min_free_fraction=0.15):
    """Return torch's reserved-but-free CUDA blocks to the driver when free
    VRAM is low, so a following Taichi kernel launch has room.

    Taichi allocates from its own CUDA pool (``cuMemAllocAsync``), which cannot
    draw on torch's caching allocator. When both share a device and free memory
    runs low, a Taichi launch (typically the post-process tonemap) OOMs even
    though torch is holding plenty of reclaimable cached blocks -- see
    ``is_cuda_oom``. The retry loops recover from that, but only after gc +
    re-rendering the chunk; reclaiming *proactively* here avoids the round-trip.

    Gated on driver-level free memory so the common (plentiful) case pays only a
    cheap ``mem_get_info`` probe: ``torch.cuda.empty_cache()`` (~ms, and it
    forces the next batch to re-acquire blocks from the driver) runs only when
    free VRAM drops below ``min_free_fraction`` of the device total -- exactly
    the regime where a Taichi launch is at risk. ``no-op`` off CUDA. Returns
    True iff it actually reclaimed.
    """
    if device is None or not torch.cuda.is_available():
        return False
    device = torch.device(device)
    if device.type != "cuda":
        return False
    try:
        with torch.cuda.device(device):
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            if free_bytes < min_free_fraction * total_bytes:
                torch.cuda.empty_cache()
                return True
    except Exception:
        pass
    return False


_PEAK_FLOOR = {}

# Arm the recorder on every managed arena built from here on. Only the
# calibration driver sets this; a normal render leaves it False and pays a
# single boolean test per arena construction.
_AUTO_RECORD = False
_RECORDED_ARENAS = []


def set_auto_record(enabled):
    """Record every managed arena created from now on (calibration only).

    The render arena is built deep inside ``get_frames`` and dropped when the
    job ends, so a driver that wants a whole render's allocation stream cannot
    reach in and arm it. Arenas armed this way are retained in a registry that
    outlives the render; call :func:`clear_recorded_arenas` between runs.
    """
    global _AUTO_RECORD
    _AUTO_RECORD = bool(enabled)
    if not _AUTO_RECORD:
        _RECORDED_ARENAS.clear()


_NONARENA_PEAKS = []


def note_nonarena_peak(name, input_bytes, peak_bytes):
    """Record a transient peak taken *outside* the arena (calibration only).

    The GPU merge and the projection build out of place in pool headroom, so
    the arena recorder cannot see them and their size has to come from torch's
    allocator counters instead. Ignored unless :func:`set_auto_record` is on,
    so a normal render pays one boolean test.
    """
    if _AUTO_RECORD and int(input_bytes) > 0:
        _NONARENA_PEAKS.append((str(name), int(input_bytes), int(peak_bytes)))


def auto_record_enabled():
    """Whether calibration recording is armed (see :func:`set_auto_record`).

    Lets callers skip work that only a calibration run needs, rather than
    computing it and discarding it on every render.
    """
    return _AUTO_RECORD


def recorded_nonarena_peaks():
    """``(name, input_bytes, peak_bytes)`` samples, oldest first."""
    return list(_NONARENA_PEAKS)


def recorded_arenas():
    """Managed arenas armed by :func:`set_auto_record`, oldest first."""
    return list(_RECORDED_ARENAS)


def clear_recorded_arenas():
    _RECORDED_ARENAS.clear()
    _NONARENA_PEAKS.clear()


def begin_cuda_peak(device):
    """Start measuring a region's peak torch CUDA allocation.

    ``torch.cuda.reset_peak_memory_stats`` is process-global, so a component
    that measures its own peak destroys the number the profiler reports for the
    whole render -- which is why the GPU merge's peak tracking had to default
    off. Peaks are absolute high-water marks, so the displaced value is not
    lost, merely remembered: :func:`peak_allocated` returns the max of the live
    counter and everything these regions have reset away. Nesting is safe.

    Returns an opaque token for :func:`end_cuda_peak`, or ``None`` off CUDA.
    Only torch's allocator is visible -- Taichi's ``cuMemAllocAsync`` pool is
    not (see :func:`is_cuda_oom`) -- so callers still need slack.
    """
    if device is None or not torch.cuda.is_available():
        return None
    device = torch.device(device)
    if device.type != "cuda":
        return None
    saved = torch.cuda.max_memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    return (device, saved, torch.cuda.memory_allocated(device))


def end_cuda_peak(token):
    """Finish a :func:`begin_cuda_peak` region.

    Returns the bytes allocated above its entry point (0 off CUDA).
    """
    if token is None:
        return 0
    device, saved, base = token
    peak = torch.cuda.max_memory_allocated(device)
    key = (device.type, device.index)
    _PEAK_FLOOR[key] = max(_PEAK_FLOOR.get(key, 0), saved, peak)
    return max(0, peak - base)


@contextmanager
def cuda_peak_scope(device):
    """Block form of :func:`begin_cuda_peak`.

    Yields a callable returning the region's peak bytes above entry; it keeps
    reporting the final value after the block exits.
    """
    token = begin_cuda_peak(device)
    measured = []

    def peak():
        if measured:
            return measured[0]
        if token is None:
            return 0
        return max(0, torch.cuda.max_memory_allocated(token[0]) - token[2])

    try:
        yield peak
    finally:
        measured.append(end_cuda_peak(token))


def reset_peak_floor():
    """Forget displaced peaks, alongside a ``reset_peak_memory_stats`` call.

    Callers that reset the process counter to start a fresh measurement window
    (the profiler, between runs) must clear the floor too, or the previous
    window's peak leaks into this one's.
    """
    _PEAK_FLOOR.clear()


def peak_allocated(device=None):
    """Process-wide peak torch CUDA allocation, including peaks that an
    intervening :func:`begin_cuda_peak` region reset off the live counter.
    """
    if not torch.cuda.is_available():
        return 0
    device = torch.device(device) if device is not None else torch.device("cuda")
    if device.type != "cuda":
        return 0
    key = (device.type, device.index)
    return max(torch.cuda.max_memory_allocated(device), _PEAK_FLOOR.get(key, 0))


def _caller_qualname():
    """Qualified name of the first frame outside this module.

    Recorded beside each allocation so a stale-table diagnostic can name the
    function that introduced a new buffer. Deliberately *not* part of the
    calibration fingerprint -- moving an allocation between helpers must not
    invalidate a table whose byte model is unchanged.
    """
    frame = sys._getframe(1)
    while frame is not None:
        module = frame.f_globals.get("__name__", "")
        if module != __name__:
            code = frame.f_code
            return f"{module}.{getattr(code, 'co_qualname', code.co_name)}"
        frame = frame.f_back
    return "<unknown>"


class ScopeRecord:
    """One recorded :meth:`ManualMemory.scope` frame.

    ``events`` is the ordered allocation/temp-nesting stream that the memory
    calibration replays; ``peak_forward``/``peak_reverse`` are the scope's own
    high-water marks relative to its entry pointers.
    """

    __slots__ = (
        "name",
        "params",
        "entry_forward",
        "entry_reverse",
        "exit_forward",
        "exit_reverse",
        "events",
        "children",
        "alloc_count",
        "peak_forward",
        "peak_reverse",
    )

    def __init__(self, name, entry_forward, entry_reverse, params=None):
        self.name = name
        # Shape parameters the scope's size is driven by (frames, pool slots,
        # primitive counts, ...), supplied by the annotation site. Recording
        # them lets calibration fit coefficients straight from production
        # renders instead of re-deriving the drivers from call context.
        self.params = params or {}
        self.entry_forward = entry_forward
        self.entry_reverse = entry_reverse
        self.exit_forward = entry_forward
        self.exit_reverse = entry_reverse
        self.events = []
        self.children = []
        self.alloc_count = 0
        self.peak_forward = 0
        self.peak_reverse = 0

    def total_bytes(self):
        """Alignment-free sum of this scope's own allocations."""
        return sum(event[4] * event[5] for event in self.events if event[0] == "alloc")

    def __repr__(self):
        return (
            f"ScopeRecord({self.name!r}, params={self.params}, "
            f"allocs={self.alloc_count}, "
            f"peak_forward={self.peak_forward}, "
            f"peak_reverse={self.peak_reverse})"
        )


class AllocationRecorder:
    """Captures every :class:`ManualMemory` allocation into a scope tree.

    ``get_tensor`` is the arena's single allocation entry point, so recording
    there covers every current *and future* buffer with no per-site
    annotation. That is what lets the memory tables be regenerated by
    measurement rather than maintained by hand.
    """

    def __init__(self):
        self.root = ScopeRecord("<root>", 0, 0)
        self.abandoned = []
        self._stack = [self.root]

    @property
    def current(self):
        return self._stack[-1]

    def scopes(self, name):
        """Every recorded scope with ``name``, outermost first."""
        found = []

        def walk(record):
            if record.name == name:
                found.append(record)
            for child in record.children:
                walk(child)

        walk(self.root)
        return found

    def push_scope(self, name, forward, reverse, params=None):
        record = ScopeRecord(name, forward, reverse, params)
        self.current.children.append(record)
        self._stack.append(record)
        return record

    def pop_scope(self, record, forward, reverse):
        # ``ManualMemory.reset`` (the render-failure path) can clear the stack
        # while scopes are still open, so a pop may find its record already
        # gone. Unwinding to it when present and ignoring it otherwise keeps an
        # OOM retry from corrupting later recordings.
        if record not in self._stack:
            return
        while self._stack[-1] is not record:
            self._stack.pop()
        record.exit_forward = forward
        record.exit_reverse = reverse
        self._stack.pop()

    def note_alloc(self, dtype, persist, numel, itemsize, forward, reverse):
        self.current.events.append(
            (
                "alloc",
                _caller_qualname(),
                str(dtype),
                bool(persist),
                int(numel),
                int(itemsize),
            )
        )
        for record in self._stack:
            record.alloc_count += 1
            record.peak_forward = max(
                record.peak_forward, forward - record.entry_forward
            )
            record.peak_reverse = max(
                record.peak_reverse, record.entry_reverse - reverse
            )

    def note_temp(self, kind, clear_persist=False):
        self.current.events.append(
            ("temp_push", bool(clear_persist)) if kind == "push" else ("temp_pop",)
        )

    def clear(self):
        """Drop scopes left open by an aborted render.

        ``ManualMemory.reset`` runs while an OOM is still unwinding, so any
        open scope holds a truncated event stream that stops at the failing
        allocation. Detaching those keeps a partial measurement from being
        mistaken for a complete one -- which would bias a calibrated
        coefficient *downwards*, the one direction that causes OOMs. They stay
        on ``abandoned`` for diagnostics.
        """
        if len(self._stack) > 1:
            outermost = self._stack[1]
            parent = self._stack[0]
            if outermost in parent.children:
                parent.children.remove(outermost)
            self.abandoned.append(outermost)
        self._stack = [self.root]


class _NullScope:
    """Zero-cost stand-in for :meth:`ManualMemory.scope` when not recording."""

    __slots__ = ()

    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


_NULL_SCOPE = _NullScope()


class _RecordingScope:
    __slots__ = ("memory", "name", "params", "record")

    def __init__(self, memory, name, params):
        self.memory = memory
        self.name = name
        self.params = params
        self.record = None

    def __enter__(self):
        recorder = self.memory._recorder
        self.record = recorder.push_scope(
            self.name,
            self.memory.current_pointer,
            self.memory.current_reverse_pointer,
            self.params,
        )
        return self.record

    def __exit__(self, exc_type, exc_val, exc_tb):
        recorder = self.memory._recorder
        if recorder is not None and self.record is not None:
            recorder.pop_scope(
                self.record,
                self.memory.current_pointer,
                self.memory.current_reverse_pointer,
            )
        # Never suppress: an OOM here must still reach the window-shrink retry.
        return False


class TempMemoryContext:
    """Restore the arena pointers on exit.

    ``clear_persist`` additionally rewinds the persistent (reverse) pointer to
    its value at entry. ``persist_floor`` is a zero-argument callable returning
    a reverse pointer that rewind must not cross, or ``None`` -- for the caller
    that allocates something batch-lived at the persistent end from *inside*
    the scope and needs it readable after the scope closes. It is read on exit,
    not on entry, because that allocation happens during the block. Explicit
    rather than inferred from the arena, so an unrelated persistent allocation
    inside the scope is still reclaimed.
    """

    def __init__(self, memory, clear_persist, persist_floor=None):
        self.memory = memory
        self.clear_persist = clear_persist
        self.persist_floor = persist_floor

    def __enter__(self):
        self.initial_pointer = self.memory.current_pointer
        self.initial_reverse_pointer = self.memory.current_reverse_pointer
        recorder = self.memory._recorder
        if recorder is not None:
            recorder.note_temp("push", self.clear_persist)
        return self.memory

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Emitted before the pointers move so the recorded stream matches the
        # order a replay must reproduce, and on the exception path too (an OOM
        # unwind still closes the scope it opened).
        recorder = self.memory._recorder
        if recorder is not None:
            recorder.note_temp("pop")
        self.memory.current_pointer = self.initial_pointer
        if self.clear_persist:
            reverse = self.initial_reverse_pointer
            floor = self.persist_floor() if self.persist_floor is not None else None
            if floor is not None:
                reverse = min(reverse, floor)
            self.memory.current_reverse_pointer = reverse
        # Never suppress an exception.  Pointer restoration is especially
        # important on the error path: callers use a temp scope around an
        # arena-backed operation and then retry a smaller frame window.
        return False


class ManualMemory:
    def __init__(
        self,
        portion_of_available_memory_used,
        device=None,
        managed=True,
        *,
        num_bytes=None,
    ):
        if device is None:
            device = render_device()
        self.current_pointer = 0
        self.max_pointer = 0
        # Largest frame window a render kernel actually launched for the chunk
        # in progress. The tracer sets it when it has to sub-divide a chunk (an
        # out-of-memory split, or the Monte Carlo path budget); the batching
        # loop resets it before each chunk and reads it afterwards so the
        # measured peak is attributed to the window that produced it rather
        # than to the larger window that was planned. ``None`` means the chunk
        # rendered whole.
        self.last_launch_frames = None
        self.stack = []
        self.managed = managed
        # Diagnostic: ALGAN_ARENA_POISON=<byte 0..255> fills every fresh arena
        # allocation with that byte (0xFF reads as NaN in a float lane and -1 in
        # an int lane). The arena is a bump allocator, so an allocation that is
        # read before it is written sees whatever the previous chunk -- or the
        # previous *job*, or the allocator's earlier tenant -- left there; a
        # render whose pixels move under the poison has such a read. Costs a
        # fill per allocation, so it is off (-1) unless asked for.
        self._poison = env_int("ALGAN_ARENA_POISON", -1) if managed else -1
        # Off by default: production pays one ``is not None`` test per
        # allocation.  See ``recording()`` and ``set_auto_record()``.
        self._recorder = None
        if managed and _AUTO_RECORD:
            self._recorder = AllocationRecorder()
            _RECORDED_ARENAS.append(self)

        if num_bytes is None:
            num_bytes = (
                int(get_num_available_bytes(device) * portion_of_available_memory_used)
                if managed
                else 1
            )
        num_bytes = max(0, int(num_bytes))
        self.data = torch.empty((num_bytes,), device=device, dtype=torch.uint8)
        self.length = len(self.data)
        self.current_reverse_pointer = self.length

    def __len__(self):
        return self.length

    def get_pointers(self):
        return self.current_pointer, self.current_reverse_pointer

    def set_pointers(self, pointers):
        pointers = [*pointers]
        self.current_pointer = pointers[0]
        self.current_reverse_pointer = pointers[1]

    def get_percent_used(self):
        if not len(self):
            return 0.0
        return 1.0 - self.get_num_bytes_remaining() / len(self)

    def get_num_bytes_remaining(self):
        return self.current_reverse_pointer - self.current_pointer

    def clone(self, x, **kwargs):
        new_x = self.get_tensor(x.shape, x.dtype, **kwargs)
        new_x[:] = x
        return new_x

    def cast(self, x, dtype, **kwargs):
        new_x = self.get_tensor(x.shape, dtype=dtype, **kwargs)
        new_x[:] = x
        return new_x

    def get_tensor(self, shape, dtype=torch.float, persist=False):
        if not self.managed:
            return torch.empty(shape, dtype=dtype, device=self.data.device)
        reverse = persist

        def get_shape(shape):
            shape = [int(_.item()) if hasattr(_, "item") else int(_) for _ in shape]
            # Scalars have no last dimension to widen into bytes. Represent
            # them as one element; callers still receive a scalar view below.
            scalar = not shape
            if scalar:
                shape = [1]
            element_size = dtype.itemsize
            byte_shape = list(shape)
            byte_shape[-1] *= element_size
            return shape, byte_shape, element_size, scalar

        logical_shape, byte_shape, num_bytes, scalar = get_shape(shape)

        pointer = self.current_pointer if not reverse else self.current_reverse_pointer

        def get_bap():
            remainder = pointer % num_bytes
            if not reverse:
                byte_align_offset = (num_bytes - remainder) if (remainder > 0) else 0
            else:
                byte_align_offset = -remainder
            return byte_align_offset

        byte_align_offset = get_bap()

        def get_numel():
            # return np.prod(shape) +  byte_align_offset
            nu = byte_shape[0]
            for x in byte_shape[1:]:
                nu = nu * x
            if reverse:
                nu = nu * -1
            return nu

        numel = get_numel()
        pointer = pointer + byte_align_offset
        new_pointer = pointer + numel

        def error_check():
            if (
                (new_pointer < self.current_pointer)
                if reverse
                else (new_pointer > self.current_reverse_pointer)
            ):
                raise InsufficientMemoryException

        error_check()

        def get_x():
            if reverse:
                x = self.data[new_pointer:pointer]
            else:
                x = self.data[pointer:new_pointer]
            return x

        def get_data():
            x = get_x()
            if self._poison >= 0:
                x.fill_(self._poison)
            if reverse:
                self.current_reverse_pointer = new_pointer
            else:
                self.current_pointer = new_pointer
            # old_max = self.max_pointer
            self.max_pointer = max(
                self.max_pointer,
                self.current_pointer + (self.length - self.current_reverse_pointer),
            )
            # if self.max_pointer > old_max:
            #    LoggerManager.instance().log_message(f'Reached {self.max_pointer} bytes, {self.max_pointer / len(self)}%')
            recorder = self._recorder
            if recorder is not None:
                numel = 1
                for extent in logical_shape:
                    numel *= extent
                recorder.note_alloc(
                    dtype,
                    reverse,
                    numel,
                    num_bytes,
                    self.current_pointer,
                    self.current_reverse_pointer,
                )
            x = x.view(byte_shape).view(dtype).view(logical_shape)
            if scalar:
                x = x.view(())
            return x

        return get_data()

    def reset(self):
        self.current_pointer = 0
        self.current_reverse_pointer = self.length
        self.max_pointer = 0
        self.stack = []
        if self._recorder is not None:
            # Called on the render-failure path, potentially with scopes still
            # open on the unwinding stack; drop them so the next attempt starts
            # from a clean tree.
            self._recorder.clear()

    def save_pointer(self):
        self.stack.append(self.current_pointer)

    def reset_pointer(self):
        self.current_pointer = self.stack[-1]
        self.stack = self.stack[:-1]

    def temp(self, clear_persist=False, persist_floor=None):
        return TempMemoryContext(self, clear_persist, persist_floor)

    def scope(self, name, **params):
        """Label the enclosing region for memory calibration.

        A no-op unless :meth:`recording` is active, so scopes can be left in
        the production render path permanently. Scope names are the terms the
        batch-size model composes (output buffers, wavefront tile state,
        post-processing, ...), so they must stay in step with how
        ``chunk_memory_required`` adds them up.

        ``params`` are the shape quantities the scope's size is driven by --
        ``frames``, ``pool``, ``num_triangles`` and so on. They are recorded
        alongside the allocations so calibration can fit coefficients from an
        ordinary render. Keyword evaluation is not free, so annotation sites
        pass only what the model actually regresses on.
        """
        if self._recorder is None:
            return _NULL_SCOPE
        return _RecordingScope(self, name, params)

    def note_scope_params(self, **params):
        """Attach shape parameters to the innermost open scope.

        For quantities that are only known *inside* the region -- the fragment
        count the sparse-coverage COUNT kernel produces, for instance -- which
        therefore cannot be passed to :meth:`scope`. A no-op when not
        recording.
        """
        recorder = self._recorder
        if recorder is not None:
            recorder.current.params.update(params)

    @contextmanager
    def recording(self):
        """Record every allocation made inside the block.

        Yields the :class:`AllocationRecorder`. Nesting is not supported: the
        calibration driver and the one-frame probe are the only callers, and
        both own the arena for the runtime.
        """
        previous = self._recorder
        recorder = AllocationRecorder()
        self._recorder = recorder
        try:
            yield recorder
        finally:
            self._recorder = previous
