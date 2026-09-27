"""What a render says while it waits for kernels.

A cold render used to print one line -- "compiling the GPU kernels can take
several minutes" (on the CPU too) -- and then nothing for as long as that
took. This module replaces it with an account of the wait:

* a header the first time in a render that a kernel misses the source-keyed
  index and has to go through the frontend, naming the device it compiles
  for and pointing at ``algan warmup``;
* one line per kernel that was really compiled (not one the offline cache
  served), numbered, with its time and the time so far -- and, when a
  precompile pool is running (``kernel_precompile.py``), how far the
  background workers have got;
* one line per kernel the render waited for a background worker to finish;
* a heartbeat every :data:`HEARTBEAT_SECONDS` while a single kernel is still
  compiling, so a three-minute megakernel is not three silent minutes;
* a summary when the render ends, if it compiled anything, and one line each
  when a pool starts and finishes.

Everything goes through the ``algan`` logger at ``INFO``, like the rest of the
render's progress, so ``ALGAN_LOG_LEVEL=WARNING`` silences it. The state is
per render job: it resets when the outermost job starts, so a second render in
one process says nothing unless it compiles something new.
"""

from __future__ import annotations

import functools
import threading
import time

from algan.logging.logger import get_logger

#: How long one kernel may compile before the heartbeat says so, and then how
#: often it repeats.
HEARTBEAT_SECONDS = 20.0

_LOCK = threading.Lock()


def _never_raises(function):
    """Reporting must not be the reason a compile or a render fails."""

    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except Exception:  # noqa: BLE001
            return None

    return wrapped


class _State:
    __slots__ = (
        "rendering",
        "is_render",
        "started",
        "header_shown",
        "compiled",
        "compiled_seconds",
        "from_workers",
        "in_progress",
        "announced",
        "heartbeat",
        "last_beat",
    )

    def __init__(self):
        self.rendering = False
        #: Whether this account belongs to a render job, rather than to the
        #: time outside one (a kernel launched while a script authors).
        self.is_render = False
        self.started = time.perf_counter()
        self.header_shown = False
        self.compiled = 0
        self.compiled_seconds = 0.0
        self.from_workers = 0
        #: ``id(kernel) -> (name, started)`` for materializations under way.
        self.in_progress = {}
        #: Kernels whose backend compile was announced before it started.
        self.announced = set()
        self.heartbeat = None
        self.last_beat = {}


_STATE = _State()


def _format_seconds(seconds):
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, seconds = divmod(int(round(seconds)), 60)
    return f"{minutes} min {seconds:02d} s"


def _device_phrase():
    from algan.rendering.taichi_runtime import (
        taichi_arch_is_cpu,
        taichi_arch_is_cuda,
        taichi_arch_is_metal,
    )

    if taichi_arch_is_cuda():
        return "the GPU (CUDA)"
    if taichi_arch_is_metal():
        return "the GPU (Metal)"
    if taichi_arch_is_cpu():
        return "the CPU"
    return "the GPU"


def _pool_suffix():
    from algan.rendering.kernel_precompile import active_pool

    pool = active_pool()
    if pool is None:
        return ""
    finished, total = pool.counts()
    if not total or finished >= total and not pool.active:
        return ""
    return f"; background workers: {finished} of {total} done"


def _show_header_locked():
    """The first line of a slow render. Called with :data:`_LOCK` held."""
    if _STATE.header_shown:
        return None
    _STATE.header_shown = True
    return (
        f"Preparing render kernels for {_device_phrase()}. The first render on "
        "this machine, and the first after an update, compiles them, which can "
        "take several minutes; they are cached, so later renders start in "
        "seconds. `algan warmup` precompiles the common set in parallel."
    )


def _log(lines):
    logger = get_logger()
    for line in lines:
        if line:
            logger.info(line)


# ---------------------------------------------------------------------------
# The render's scope
# ---------------------------------------------------------------------------


@_never_raises
def render_started():
    """The outermost render job began: a fresh account."""
    global _STATE
    with _LOCK:
        previous = _STATE
        _STATE = _State()
        _STATE.rendering = True
        _STATE.is_render = True
        # A header already printed outside a render (a kernel launched while
        # authoring) is not repeated by the render that follows it; one
        # printed by an earlier render is, if this one compiles too.
        _STATE.header_shown = previous.header_shown and not previous.is_render


@_never_raises
def render_finished():
    """The outermost render job ended: summarize, if anything was compiled."""
    with _LOCK:
        state = _STATE
        state.rendering = False
        state.in_progress.clear()
        compiled, seconds, from_workers = (
            state.compiled,
            state.compiled_seconds,
            state.from_workers,
        )
        elapsed = time.perf_counter() - state.started
    if not compiled and not from_workers:
        return
    parts = []
    if compiled:
        parts.append(
            f"{compiled} compiled here ({_format_seconds(seconds)} of compiling)"
        )
    if from_workers:
        parts.append(f"{from_workers} from background workers")
    _log(
        [
            f"Render kernels ready: {', '.join(parts)}; render took "
            f"{_format_seconds(elapsed)}. They are cached for later renders."
        ]
    )


# ---------------------------------------------------------------------------
# Events from the compile boundary (`taichi_runtime`) and the index
# ---------------------------------------------------------------------------


@_never_raises
def materializing(kernel, name):
    """A new specialization is being materialized; it may or may not be slow."""
    with _LOCK:
        _STATE.in_progress[id(kernel)] = (name, time.perf_counter())
        if _STATE.heartbeat is None and _STATE.rendering:
            _STATE.heartbeat = threading.Thread(
                target=_heartbeat,
                args=(_STATE,),
                name="algan-compile-heartbeat",
                daemon=True,
            )
            _STATE.heartbeat.start()


@_never_raises
def materialized(kernel):
    """Done, by whatever route: nothing is under way for it any more."""
    with _LOCK:
        _STATE.in_progress.pop(id(kernel), None)


@_never_raises
def index_missed():
    """A specialization missed the source-keyed index: the frontend is about to run."""
    with _LOCK:
        header = _show_header_locked()
    _log([header])


#: A backend compile is announced before it starts when the frontend alone
#: took this long, or when a previous compile of the same spec took
#: :data:`_ANNOUNCE_EXPECTED` or more.
_ANNOUNCE_FRONTEND = 5.0
_ANNOUNCE_EXPECTED = 10.0


@_never_raises
def backend_starting(kernel, name, frontend_seconds, expected):
    """The backend compile of a possibly long kernel is about to start.

    The heartbeat cannot speak during it: ``Program.compile_kernel`` holds the
    GIL for its whole duration (measured: one 20 s gap in a thread ticking
    every 50 ms while the CPU backend compiled ``sheet_resolve_shade_arena``).
    So a kernel that is already slow, or was slow last time, says so up front.
    """
    if frontend_seconds < _ANNOUNCE_FRONTEND and (expected or 0.0) < _ANNOUNCE_EXPECTED:
        return
    with _LOCK:
        header = _show_header_locked()
        _STATE.announced.add(id(kernel))
        number = _STATE.compiled + _STATE.from_workers + 1
    hint = f"; about {_format_seconds(expected)} last time" if expected else ""
    _log(
        [
            header,
            f"  kernel {number}: compiling {_short(name)}{hint}{_pool_suffix()} ...",
        ]
    )


@_never_raises
def compiled(kernel, name, seconds, cold):
    """A specialization finished compiling in this process.

    ``cold`` when the backend really compiled it, rather than loading an
    artifact from the offline cache; only those get a line, unless the compile
    was announced (then the line says the cache served it after all).
    """
    suffix = _pool_suffix() if cold else ""
    with _LOCK:
        _STATE.in_progress.pop(id(kernel), None)
        announced = id(kernel) in _STATE.announced
        _STATE.announced.discard(id(kernel))
        if not cold:
            lines = (
                [
                    f"  {_short(name)}: loaded from the kernel cache "
                    f"({_format_seconds(seconds)})"
                ]
                if announced
                else []
            )
        else:
            header = _show_header_locked()
            _STATE.compiled += 1
            _STATE.compiled_seconds += seconds
            number = _STATE.compiled + _STATE.from_workers
            elapsed = time.perf_counter() - _STATE.started
            lines = [
                header,
                f"  kernel {number}: {_short(name)} compiled in "
                f"{_format_seconds(seconds)} ({_format_seconds(elapsed)} so far{suffix})",
            ]
    _log(lines)


@_never_raises
def waited_for_worker(job, waited, ready):
    """The render waited ``waited`` s for a background worker to finish ``job``."""
    if not ready:
        return
    with _LOCK:
        header = _show_header_locked()
        _STATE.from_workers += 1
        number = _STATE.compiled + _STATE.from_workers
    _log(
        [
            header,
            f"  kernel {number}: {job.name} compiled by a background worker "
            f"(waited {_format_seconds(waited)}{_pool_suffix()})",
        ]
    )


def _short(name):
    """``module.qualname[specialization=...]`` -> ``qualname``."""
    return name.split("[", 1)[0].rsplit(".", 1)[-1]


def _heartbeat(state):
    while True:
        time.sleep(HEARTBEAT_SECONDS / 4)
        with _LOCK:
            if state is not _STATE or not state.rendering:
                state.heartbeat = None
                return
            now = time.perf_counter()
            lines = []
            for key, (name, started) in list(state.in_progress.items()):
                running = now - started
                last = state.last_beat.get(key, started)
                if running >= HEARTBEAT_SECONDS and now - last >= HEARTBEAT_SECONDS:
                    state.last_beat[key] = now
                    header = _show_header_locked()
                    if header:
                        lines.append(header)
                    lines.append(
                        f"  still compiling {_short(name)} "
                        f"({_format_seconds(running)} so far{_pool_suffix()})"
                    )
        _log(lines)


# ---------------------------------------------------------------------------
# Events from the precompile pool
# ---------------------------------------------------------------------------


@_never_raises
def pool_started(pool, workers):
    jobs = len(pool.jobs)
    _log(
        [
            f"Compiling {jobs} render kernels for {_device_phrase()} in {workers} "
            f"background worker process{'es' if workers != 1 else ''} ({pool.reason}): "
            "the kernel cache does not have them for this version yet."
        ]
    )


@_never_raises
def pool_event(pool, kind, payload):
    if kind == "fatal":
        get_logger().warning(f"Background kernel compilation stopped: {payload}")
        return
    if kind != "finished":
        return
    jobs = pool.jobs
    compiled = sum(job.status == "compiled" for job in jobs)
    reused = sum(job.status in ("cached", "offline-hit") for job in jobs)
    taken = sum(job.state == "taken" for job in jobs)
    failed = [job for job in jobs if job.state == "failed"]
    elapsed = (pool.finished or time.perf_counter()) - pool.started
    parts = [f"{compiled} compiled"]
    if reused:
        parts.append(f"{reused} already cached")
    if taken:
        parts.append(f"{taken} left to the render")
    if failed:
        parts.append(f"{len(failed)} not compiled")
    _log(
        [
            f"Background kernel compilation finished in {_format_seconds(elapsed)}: "
            + ", ".join(parts)
            + "."
        ]
    )
