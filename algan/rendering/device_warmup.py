"""Load the render device's torch kernels while the script is still authoring.

The problem
-----------
PyTorch loads its CUDA kernels lazily (``CUDA_MODULE_LOADING=LAZY``, the
default): the first call of each op in a process loads the module that holds
it, 3-100 ms apiece. A render calls a few dozen distinct ops, so a fresh
process's first render paid ~0.7-0.8 s for loading alone on an RTX 5080
(``bmm`` 85-100 ms for cuBLAS, ``index_select`` 65 ms, the sorts, scans and
reductions 10-35 ms each; measured with every cache warm), on the critical
path, while the script before it spent one to two seconds authoring on the
CPU with the device idle. With a cold CUDA driver cache (``CUDA_CACHE_PATH``,
kept under ``ALGAN_CACHE_DIR``: a new machine, a driver update,
``algan.clear_cache()``) the driver JIT-compiles those kernels and the same
ops cost ~15 s -- the 14 s a cold render spent before its first Taichi
kernel. The driver serializes that JIT (four threads took as long as one), so
one thread is all this uses.

The mechanism
-------------
A daemon thread, started while ``import algan`` is still running in a
process that is plainly a scene script (and by the render daemon at launch),
calls each op the renderer uses once on tiny tensors of the dtypes it uses
them with. The loading happens inside the op, where torch's bindings have
released the GIL, so the thread costs the authoring thread a few
milliseconds of GIL in total; the rest runs on another core.

Why it cannot change a render
-----------------------------
* It only touches its own tensors, and only on CUDA: a CPU or MPS render
  device at import starts nothing (MPS compiles its kernels per shape, so
  there is nothing comparable to load ahead).
* It is over before the render begins: the first render job stops it and
  joins it (:func:`finish`), so nothing it allocates overlaps what the render
  measures -- the memory model's peaks, the arena preflight. What it leaves
  behind is what the render's own first calls would have created (the
  caching allocator's small segment, the cuBLAS handle and workspace, which
  return to torch's per-thread pools when the thread exits and are reused by
  the render thread).
* Every step is wrapped: one that fails is skipped. A script that never
  renders only makes the work wasted, and so does a render device changed
  after import -- the thread checks the device before each step and stops
  as soon as it is no longer the one it warms.
* At exit the thread is stopped and joined (bounded), so it is never inside a
  driver call while the process tears CUDA down, and never holds it open.

``ALGAN_STARTUP_WARMUP=0`` turns it off. It is read where the thread would
start: at ``import algan`` in a scene script (so it must be set before the
import), and in the render daemon at its launch and again at each run, so a
client's ``0`` stops a warm-up still running for that run and its ``1``
starts one the launch environment had turned off. It never changes a
picture, so the daemon does not refuse a client over it.
"""

from __future__ import annotations

import atexit
import contextlib
import threading
import time
import types

from algan.environment import env_flag

_THREAD = None
_STOP = threading.Event()
_LOCK = threading.Lock()
#: What the last warm-up did, for tests and for anyone inspecting a process:
#: ``{"ops": int, "failed": int, "seconds": float, "stopped": bool,
#: "device_moved": bool}``.
LAST_RUN = None


def enabled():
    """Whether the start-up warm-up may run in this process at all."""
    return env_flag("ALGAN_STARTUP_WARMUP", True) and not env_flag(
        "ALGAN_PRECOMPILE_WORKER", False
    )


def _ops(device):
    """The warm-up steps, in order: ``[(name, callable), ...]``.

    One entry per kernel family the renderer was measured paying a first-call
    load for (``explainer`` and ``graphics`` PREVIEW renders on CUDA,
    2026-10), with the dtypes it calls them with. The operand tensors are
    built by the first steps rather than up front, because on a cold driver
    cache building them alone is seconds of JIT and the thread must be
    stoppable between any two. Sizes matter where torch picks a different
    kernel for small inputs: sorts, scans and ``unique`` below 4096 elements
    take a small-sort path the render never uses, so those get 8192.
    """
    import torch

    from algan.rendering.mps_compat import accumulate_dtype

    # The renderer's wide accumulators, through the one module allowed to name
    # float64 (test_mps_friendly enforces it); float64 on CUDA, where this runs.
    wide = accumulate_dtype()
    big = 8192
    n = types.SimpleNamespace()

    def floats():
        n.f = torch.ones(8, device=device)
        n.f3 = torch.ones((4, 3), device=device)
        n.m = torch.ones((1, 2, 2), device=device)
        n.d = torch.ones(8, device=device, dtype=wide)

    def ints():
        n.i = torch.arange(8, device=device)
        n.keys = torch.arange(big, 0, -1, device=device)

    def derived():
        n.i32 = n.i.to(torch.int32)
        n.b = n.f > 0.5
        n.fkeys = n.keys.to(torch.float32)
        n.mask = (n.i % 2) == 0

    return [
        ("floats", floats),
        ("ints", ints),
        ("derived", derived),
        ("bmm", lambda: torch.bmm(n.m, n.m)),
        ("index_select", lambda: n.f.index_select(0, n.i)),
        ("index", lambda: n.f[n.i]),
        ("index_mask", lambda: n.i[n.mask]),
        ("index_put", lambda: n.f.clone().index_put_((n.i,), n.f)),
        ("gather", lambda: torch.gather(n.b, 0, n.i)),
        (
            "scatter_reduce",
            lambda: [
                n.i32.clone().scatter_reduce_(0, n.i, n.i32, reduce)
                for reduce in ("amax", "amin", "sum")
            ],
        ),
        ("pow", lambda: (n.f.pow(2), n.f.pow(0.5), n.f.pow(3.0))),
        ("vector_norm", lambda: torch.linalg.vector_norm(n.f3, dim=-1)),
        ("sort", lambda: torch.sort(n.keys, stable=True)),
        ("sort_float", lambda: torch.sort(n.fkeys, stable=True)),
        ("argsort", lambda: torch.argsort(n.keys)),
        (
            "unique",
            lambda: torch.unique(n.keys, return_inverse=True, return_counts=True),
        ),
        ("unique_dim", lambda: torch.unique(n.f3, dim=0, return_inverse=True)),
        (
            "unique_consecutive",
            lambda: torch.unique_consecutive(n.keys, return_counts=True),
        ),
        ("cumsum", lambda: (torch.cumsum(n.keys, 0), torch.cumsum(n.fkeys, 0))),
        ("repeat_interleave", lambda: torch.repeat_interleave(n.i % 3)),
        ("bincount", lambda: torch.bincount(n.i)),
        ("remainder", lambda: (n.i % 3, n.f % 2.0)),
        (
            "div_modes",
            lambda: (
                torch.div(n.i, 2, rounding_mode="floor"),
                torch.div(n.i, 2, rounding_mode="trunc"),
            ),
        ),
        ("div", lambda: n.f / n.f),
        ("mul", lambda: (n.f * n.f, n.i * n.i)),
        (
            "clamp",
            lambda: (n.f.clamp_min(0.1), n.f.clamp_max(0.9), n.f.clamp(0.1, 0.9)),
        ),
        ("sum", lambda: (n.f3.sum(dim=1), n.b.sum(), n.i.sum())),
        ("mean", lambda: n.f3.mean(dim=1)),
        (
            "amax_amin",
            lambda: (n.f3.amax(dim=1), n.f3.amin(dim=1), n.f.amax(), n.i.amax()),
        ),
        ("argmax_argmin", lambda: (n.f.argmax(), n.i.argmin(), n.f.argmin())),
        ("aminmax", lambda: torch.aminmax(n.f)),
        (
            "minimum_maximum",
            lambda: (
                torch.minimum(n.i, n.i),
                torch.maximum(n.f, n.f),
                torch.minimum(n.f, n.f),
            ),
        ),
        ("all_any", lambda: (n.b.all(dim=0), n.b.any())),
        (
            "compare",
            lambda: (
                n.f == n.f,
                n.f > 0.5,
                n.i == n.i,
                n.f < n.f,
                n.i >= 1,
                n.i != n.i,
            ),
        ),
        ("logic", lambda: (n.b & n.b, n.b | n.b, ~n.b, torch.logical_not(n.b))),
        ("shift", lambda: (n.i << n.i, n.i >> 1, n.i & n.i, n.i | n.i)),
        (
            "unary",
            lambda: (
                n.f.sqrt(),
                -n.f,
                n.f.reciprocal(),
                n.f.abs(),
                n.f.ceil(),
                n.f.floor(),
            ),
        ),
        ("log", lambda: (n.f.log2(), n.d.log2(), n.f.log(), n.f.exp())),
        ("trig", lambda: (n.f.sin(), n.f.cos(), torch.atan2(n.f, n.f))),
        ("where", lambda: torch.where(n.b, n.f, n.f)),
        (
            "cat",
            lambda: (
                torch.cat([n.f, n.f]),
                torch.cat([n.i, n.i]),
                torch.stack([n.f, n.f]),
            ),
        ),
        (
            "convert",
            lambda: (
                n.i.float(),
                n.f.long(),
                n.b.long(),
                n.i32.long(),
                n.f.to(wide),
                n.d.float(),
                n.f.half(),
            ),
        ),
        ("clone", lambda: (n.f.clone(), n.f3.t().contiguous())),
        ("nonzero", lambda: torch.nonzero(n.mask)),
        (
            "fill",
            lambda: (
                torch.zeros(4, device=device),
                torch.full((4,), 2.0, device=device),
            ),
        ),
    ]


def _resolved(device):
    """``device`` with an index: a bare ``cuda`` names the calling thread's current GPU."""
    import torch

    if device.type == "cuda" and device.index is None:
        with contextlib.suppress(Exception):
            return torch.device("cuda", torch.cuda.current_device())
    return device


def _device_moved(device):
    """Whether the render device is no longer the one this warm-up is for.

    The whole device, index included: a move from ``cuda:0`` to ``cuda:1``
    leaves nothing on the first GPU worth loading.
    """
    try:
        from algan.settings._startup import render_device

        return _resolved(render_device()) != device
    except Exception:  # noqa: BLE001
        return False


def _run(device):
    global LAST_RUN
    import torch

    started = time.perf_counter()
    done = failed = 0
    stopped = moved = False
    on_device = (
        torch.cuda.device(device) if device.type == "cuda" else contextlib.nullcontext()
    )
    try:
        with torch.inference_mode(), on_device:
            for _name, op in _ops(device):
                if _STOP.is_set():
                    stopped = True
                    break
                if _device_moved(device):
                    moved = True
                    break
                try:
                    op()
                    done += 1
                except Exception:  # noqa: BLE001 -- a warm-up op is never worth an error
                    failed += 1
    except Exception:  # noqa: BLE001
        failed += 1
    LAST_RUN = {
        "ops": done,
        "failed": failed,
        "seconds": time.perf_counter() - started,
        "stopped": stopped,
        "device_moved": moved,
    }


def _start_thread(device, reason):
    global _THREAD
    _STOP.clear()
    _THREAD = threading.Thread(
        target=_run,
        args=(device,),
        name=f"algan-device-warmup ({reason})",
        daemon=True,
    )
    _THREAD.start()


def start(reason="import"):
    """Start the warm-up thread for the current render device, if it is CUDA.

    Once per process, and silent: returns whether a thread was started.
    """
    if not enabled():
        return False
    with _LOCK:
        if _THREAD is not None:
            return False
        try:
            from algan.settings._startup import render_device

            device = render_device()
        except Exception:  # noqa: BLE001
            return False
        if device.type != "cuda":
            return False
        # Resolved here, on the thread that will render: a bare ``cuda`` means
        # this thread's current GPU, which the warm-up thread does not share.
        _start_thread(_resolved(device), reason)
        return True


def start_at_import():
    """From ``import algan``: start only in a process plainly running a scene script.

    The test the daemon handoff uses (``daemon_client.is_scene_script_run``),
    so not a test runner, a notebook, ``python -c``, ``-m`` or the ``algan``
    CLI, where a background thread issuing device work would be a surprise.
    """
    try:
        from algan.daemon_client import is_scene_script_run

        if not is_scene_script_run():
            return False
    except Exception:  # noqa: BLE001
        return False
    return start("import")


#: How long the first render job waits for the step in flight. Warm, a step is
#: milliseconds; on a cold driver cache the slowest JIT measured ~5 s, so this
#: is only reached by a driver that is not answering at all.
RENDER_JOIN_SECONDS = 30.0


def finish(timeout=RENDER_JOIN_SECONDS, *, log=False, starting="the render"):
    """Stop the warm-up and wait for it: called before the first render job.

    Steps not yet taken are skipped -- the render loads what it needs itself
    -- and the one in flight is waited for, so the render starts with no other
    thread issuing device work. Returns whether the thread is gone; with
    ``log``, a thread still running at the timeout is reported (at ``PERF``),
    naming what (``starting``) then goes ahead beside it.
    """
    thread = _THREAD
    if thread is None:
        return True
    if thread is threading.current_thread():
        return False
    _STOP.set()
    thread.join(timeout)
    gone = not thread.is_alive()
    if not gone and log:
        with contextlib.suppress(Exception):
            from algan.logging.logger import PERF, get_logger

            get_logger().log(
                PERF,
                "The start-up device warm-up was still inside one step after "
                "%.0f s; %s starts while it finishes that step.",
                timeout,
                starting,
            )
    return gone


def on_daemon_run():
    """A daemon run is starting: apply this run's ``ALGAN_STARTUP_WARMUP``.

    The daemon swaps each client's live variables in per run, so this is where
    a client's value can still matter: ``0`` stops a warm-up the launch
    started, ``1`` starts one the launch environment had turned off (once per
    process either way; the daemon is warm after its first render).
    """
    if enabled():
        return start("daemon run")
    finish(log=True, starting="this daemon run")
    return False


def active():
    """Whether the warm-up thread is running (for tests)."""
    thread = _THREAD
    return thread is not None and thread.is_alive()


def _reset_for_tests():
    """Forget a finished warm-up so a test can start another."""
    global _THREAD, LAST_RUN
    finish()
    with _LOCK:
        _THREAD = None
        LAST_RUN = None
        _STOP.clear()


# Bounded by one step: a cold driver JIT of the slowest is ~5 s.
atexit.register(lambda: finish(timeout=10.0))
