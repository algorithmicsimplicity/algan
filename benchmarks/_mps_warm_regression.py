"""Why is the SECOND render in a process slower than the first, on Metal only?

``profile_scene(runs=2)`` prints run 1 as "cold" and run 2 as "warm", and warm
should be the faster of the two -- the only thing it saves is the Taichi kernel
compile, but that is all it should need to save. On the CPU it behaves:
``nn_scene_PREVIEW`` measures 209.3 s then 95.6 s, 2.2x faster warm. On the Mac
runner the same shape of run went 595.7 s cold and then had to be killed 30
minutes into the warm pass.

``profile_scene`` prints its table only at the very end, so a run that is killed
reports nothing at all -- which is how three Mac jobs in a row cost an hour each
and produced no reading. This script prints a line **after every render**, so a
timeout still leaves the comparison behind, and it reports the three numbers
that would explain the effect:

* the arena the render was given (``rendering_memory_fraction`` of whatever
  ``get_num_available_bytes`` reported at the time);
* the MPS pool figures that feed it;
* how many primitive batches and render chunks the render then took, since a
  smaller arena buys narrower windows and every batch re-pays projection, the
  merge and the BVH build -- which on Metal are eager CPU torch.

It deliberately does NOT install the profiler: its hooks synchronize the device
around every kernel launch, and the question here is about wall time between
renders, not about attributing it.

Usage:
  uv run python benchmarks/_mps_warm_regression.py [runs] [quality]
"""

from __future__ import annotations

import multiprocessing
import os
import resource
import sys
import threading
import time
from pathlib import Path

#: The scene's texture lives beside the performance benchmarks, and the harness
#: runs its command from the repository root rather than from there. Resolved
#: against this file so the script works from any working directory.
WORLD_MAP = str(Path(__file__).resolve().parent / "performance" / "world_map.png")

os.environ.setdefault("ALGAN_USE_DAEMON", "0")
os.environ.setdefault("ALGAN_VIDEO_ENCODER", "software")

import psutil  # noqa: E402
import torch  # noqa: E402

from algan import *  # noqa: E402, F403
from algan.mobs.neural_nets.neural_net import NeuralNetMLPV3  # noqa: E402
from algan.utils import memory_utils as mu  # noqa: E402

RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 2
QUALITY = (sys.argv[2] if len(sys.argv) > 2 else "UHD").upper()
PRESET = {"UHD": UHD, "HD": HD, "MD": MD, "PREVIEW": PREVIEW}[QUALITY]
#: Stop before starting a render that would push the process past this many
#: seconds. A macOS runner is reclaimed before its timeout and a reclaimed job
#: publishes NOTHING (agent_guidance/gpu_harnesses.md), so a run that would not
#: finish is worth less than one that stops and reports.
BUDGET = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0

#: ``pin-arena`` as a fourth argument: never let a later render size itself
#: smaller than the first one did.
#:
#: Run 37 traced the warm penalty to launch count, not to reclaim. Its counters
#: are flat where the reclaim hypothesis needed them to move -- 0.1 to 1.9 s a
#: chunk inside ``release_torch_memory``, and cache clears rising 1 -> 7 a chunk
#: through the cold pass without moving its import count off 12388. What DOES
#: move is between the renders: 12388 imports a chunk cold against 48766 warm,
#: 3.94x, against a per-chunk cost of 31.2 s cold and 55.2 s warm. The 36378
#: extra imports divide into the 24.0 s gap at 0.66 ms each, which is this box's
#: measured dispatch cost. The arena is 1898 MB cold and 1226 MB warm, 0.65x,
#: because ``driver_allocated_memory`` reads as a high-water mark here (it
#: climbs monotonically 2.91 -> 4.49 G across the cold pass while live bytes sit
#: at 1.91 G, and the pressured drains do not bring it down).
#:
#: So: smaller arena -> narrower launches -> more of them. Pinning the arena
#: tests exactly that link. If warm returns to cold's per-chunk cost, the defect
#: is the sizing probe; if it stays slow, it is not.
PIN_ARENA = "pin-arena" in sys.argv[4:]

#: Filled by the hooks below, reset per render.
STATS = {"arenas": [], "batches": 0, "chunks": 0}
WALL_START = time.perf_counter()

#: When the render in progress started, so the per-chunk trace can say where
#: its time went rather than only how much there was of it.
RENDER_STARTED = [0.0]

#: When the chunk in flight began, for the heartbeat below.
CHUNK_STARTED = [0.0]

#: Highest driver_allocated seen since the last chunk line, and when the
#: heartbeat last spoke. The boundary readings understate the in-chunk peak by
#: ~1.1 G, which is what run 41's heartbeat found.
PEAK = [0]
#: Lowest host_free seen since the last chunk line. The unified-memory
#: hypothesis says THIS is the number that runs out, not the GPU one.
TROUGH = [1 << 62]
LAST_BEAT = [0.0]

#: Reclaim accounting, reset at every chunk boundary. The per-chunk trace from
#: run 36 put the warm cost INSIDE the chunks (warm 54.8 s a chunk against a
#: cold 34.8 s over the same 18 chunks and 12 batches), not in batch
#: preparation -- warm preparation is the faster of the two, 8.4 s against
#: 28.2 s. So the preflight search is not what costs, and the suspect is the
#: reclaim path: `_gpu_memory_pressure` judges from `driver_allocated_memory`,
#: which counts cached-but-free blocks, and after run 1 that figure sits at
#: 4.56 G of a 4.67 G recommended max while torch's LIVE bytes are 0.00 G. Over
#: the 0.8 threshold, every one of the nineteen `force_gc=False` reclaim sites
#: pays a full gc.collect() and drops the zero-copy import cache, which the
#: next launch of every kernel then re-imports. These counters say whether that
#: is what the extra ~20 s a chunk is.
RECLAIM = {"calls": 0, "pressured": 0, "seconds": 0.0}

_real_release = mu.release_torch_memory


def _release(force_gc=True):
    started = time.perf_counter()
    RECLAIM["calls"] += 1
    if force_gc or mu._gpu_memory_pressure():
        RECLAIM["pressured"] += 1
    try:
        return _real_release(force_gc=force_gc)
    finally:
        RECLAIM["seconds"] += time.perf_counter() - started


mu.release_torch_memory = _release

_real_available = mu.get_num_available_bytes
#: The first render's free-byte figure, which every later one is held to.
_FIRST_AVAILABLE = []


def _available(*args, **kwargs):
    got = _real_available(*args, **kwargs)
    if not _FIRST_AVAILABLE:
        _FIRST_AVAILABLE.append(got)
        return got
    return max(got, _FIRST_AVAILABLE[0]) if PIN_ARENA else got


mu.get_num_available_bytes = _available

_real_arena_init = mu.ManualMemory.__init__


def _arena_init(self, portion, *args, **kwargs):
    _real_arena_init(self, portion, *args, **kwargs)
    if getattr(self, "managed", False):
        STATS["arenas"].append(int(len(self)))


mu.ManualMemory.__init__ = _arena_init

from algan.render_loop import RenderLoopMixin  # noqa: E402

_real_batch = RenderLoopMixin._render_primitive_batch


def _batch(self, *args, **kwargs):
    STATS["batches"] += 1
    yield from _real_batch(self, *args, **kwargs)


RenderLoopMixin._render_primitive_batch = _batch


def _heartbeat():
    """Print every 30 s while a chunk is in flight.

    The chunk line is printed at chunk START, so a chunk that takes 12 minutes
    and a process that has died look identical from the log until the next line
    appears -- which has twice cost a diagnosis: run 38's cold pass had one
    455.6 s chunk among 25-42 s ones, and run 40 sat past 730 s on chunk 5 with
    the driver figure at 3.14 G of 4.67, nowhere near the ceiling that killed
    run 39. A live process says so here, with the pool figures at the moment of
    asking, so "slow" and "dead" stop looking the same. A daemon thread, so it
    never holds the interpreter open.

    Its first outing found what the chunk lines cannot see: DURING a chunk the
    driver figure reaches 4.85-4.87 G, while at chunk boundaries it reads
    3.70-4.07 G. The peak is ~1.1 G above every number this instrument had been
    reporting, and above the 4.67 G recommended max -- so the headroom reserve
    was calibrated against a figure that understates the peak by a quarter of
    the pool. Hence the watermark below: sample often, report the peak at the
    next chunk line, so the reserve can be set against the real high point
    rather than against the troughs between chunks.
    """
    while True:
        time.sleep(2)
        if torch.mps.is_available():
            PEAK[0] = max(PEAK[0], torch.mps.driver_allocated_memory())
        TROUGH[0] = min(TROUGH[0], psutil.virtual_memory().available)
        held = time.perf_counter() - CHUNK_STARTED[0]
        if not CHUNK_STARTED[0] or held < 45:
            continue
        if time.perf_counter() - LAST_BEAT[0] < 30:
            continue
        LAST_BEAT[0] = time.perf_counter()
        print(
            f"      ... chunk {STATS['chunks']} still running, {held:.0f} s in"
            f" | {_pool()}",
            flush=True,
        )

import algan.rendering.raytracing.tracer as rtr  # noqa: E402

_real_wavefront = rtr.raytrace_render_wavefront


def _wavefront(*args, **kwargs):
    STATS["chunks"] += 1
    # Printed BEFORE the chunk runs, so the gap between two lines is that
    # chunk's cost and a long gap before chunk 1 is the batch preparation --
    # which is what a preflight that binary-searches the window looks like.
    started = time.perf_counter() - RENDER_STARTED[0]
    print(
        f"    chunk {STATS['chunks']:>3} begins at +{started:7.1f} s | "
        f"reclaim {RECLAIM['calls']:>4} calls, {RECLAIM['pressured']:>4} pressured, "
        f"{RECLAIM['seconds']:6.1f} s | zc {ZC['clears']:>4} clears, "
        f"{ZC['imports']:>6} imports | peak {PEAK[0] / 2**30:.2f}G | {_pool()}",
        flush=True,
    )
    CHUNK_STARTED[0] = time.perf_counter()
    PEAK[0] = 0
    TROUGH[0] = 1 << 62
    RECLAIM["calls"] = RECLAIM["pressured"] = 0
    RECLAIM["seconds"] = 0.0
    ZC["clears"] = ZC["imports"] = 0
    return _real_wavefront(*args, **kwargs)


import algan.rendering.mps_zero_copy as zc  # noqa: E402

#: The reclaim hypothesis charges most of its cost NOT to the reclaim call but
#: to the launches after it: clearing the zero-copy import cache means the next
#: launch of every kernel re-imports every arena array it takes, and the widest
#: take twenty. So count the clears and the re-imports too; reclaim seconds
#: alone would under-report the mechanism by design.
ZC = {"clears": 0, "imports": 0}

_real_clear = zc.clear_import_cache
_real_import = zc.import_tensor


def _clear_cache():
    ZC["clears"] += 1
    return _real_clear()


def _import_tensor(tensor, element_shape=()):
    ZC["imports"] += 1
    return _real_import(tensor, element_shape)


zc.clear_import_cache = _clear_cache
zc.import_tensor = _import_tensor

rtr.raytrace_render_wavefront = _wavefront

# Every module that did `from ...memory_utils import release_torch_memory` holds
# its own binding, so rebinding the name on memory_utils alone intercepts
# nothing -- which is what a first local run showed: 0 calls against a render
# that makes several a chunk. Rebind wherever the original is already bound,
# after every algan module this script needs has been imported.
for _name, _module in list(sys.modules.items()):
    if not _name.startswith("algan"):
        continue
    if getattr(_module, "release_torch_memory", None) is _real_release:
        _module.release_torch_memory = _release
    if getattr(_module, "clear_import_cache", None) is _real_clear:
        _module.clear_import_cache = _clear_cache
    if getattr(_module, "import_tensor", None) is _real_import:
        _module.import_tensor = _import_tensor
    if getattr(_module, "get_num_available_bytes", None) is _real_available:
        _module.get_num_available_bytes = _available


def _pool():
    """The two MPS figures the free-bytes probe is built from, in GiB."""
    if not torch.mps.is_available():
        return "n/a"
    return (
        f"driver={torch.mps.driver_allocated_memory() / 2**30:.2f}G "
        f"current={torch.mps.current_allocated_memory() / 2**30:.2f}G "
        f"recommended={torch.mps.recommended_max_memory() / 2**30:.2f}G "
        f"| host_free={psutil.virtual_memory().available / 2**30:.2f}G "
        f"rss={psutil.Process().memory_info().rss / 2**30:.2f}G"
    )


def _host():
    """Host-side footprint, which the MPS pool is carved out of.

    The runner has 7 GB of unified memory and a 4.67 GB
    ``recommendedMaxWorkingSetSize``, so a live child process or a growing
    resident set competes with the render arena rather than sitting beside it.
    ``children`` is here because run 34 exited with a leaked-semaphore warning
    that run 35 did not: one semaphore explains no memory by itself, but an
    unreaped child that owns it would.
    """
    kids = multiprocessing.active_children()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is bytes on macOS, kilobytes on Linux.
    scale = 2**30 if sys.platform == "darwin" else 2**20
    return f"rss_peak={rss / scale:.2f}G children={len(kids)}{[c.name for c in kids] or ''}"


def scene():
    duration = 0.3
    SETTINGS.raytracing.set(shadows=False)
    with Off():
        nn = NeuralNetMLPV3([5, 5, 5, 5]).move(LEFT).spawn()
        x = ImageMob(WORLD_MAP).move_next_to(nn, LEFT).spawn()
        label = (
            Text("Neural Net MLP v3 processing an image of the globe")
            .move_next_to(nn, DOWN)
            .spawn()
        )
    with Sync(runtime=duration):
        nn.move(UP)
        x.color_texture = x.color_texture * 0.5
        label.move(RIGHT * 2)


def main():
    print(
        f"quality={QUALITY} runs={RUNS} budget={BUDGET or 'none'} "
        f"pin_arena={PIN_ARENA}",
        flush=True,
    )
    threading.Thread(target=_heartbeat, daemon=True).start()
    print(f"pool before any render: {_pool()} | {_host()}", flush=True)
    for i in range(1, RUNS + 1):
        STATS["arenas"].clear()
        STATS["batches"] = STATS["chunks"] = 0
        SceneManager.reset()
        Scene.set_video_settings(PRESET)
        scene()
        started = time.perf_counter()
        RENDER_STARTED[0] = started
        Scene.save_video(
            os.path.join("algan_outputs", f"warm_regression_run{i}.mp4"),
            video_settings=PRESET,
            reset=True,
            ffmpeg_params=["-crf", "17", "-preset", "ultrafast"],
        )
        elapsed = time.perf_counter() - started
        arenas = ", ".join(f"{b / 2**20:.0f}M" for b in STATS["arenas"]) or "none"
        print(
            f"RUN {i}: {elapsed:.1f} s | arenas [{arenas}] | "
            f"batches {STATS['batches']} | chunks {STATS['chunks']}",
            flush=True,
        )
        print(f"  pool after run {i}: {_pool()} | {_host()}", flush=True)
        if BUDGET and (time.perf_counter() - WALL_START) > BUDGET:
            print(
                f"stopping after run {i}: {time.perf_counter() - WALL_START:.0f} s "
                f"is past the {BUDGET:.0f} s budget, and a job that is reclaimed "
                "reports nothing at all",
                flush=True,
            )
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
