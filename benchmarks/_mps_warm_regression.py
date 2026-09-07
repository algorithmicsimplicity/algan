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

import os
import sys
import time
from pathlib import Path

#: The scene's texture lives beside the performance benchmarks, and the harness
#: runs its command from the repository root rather than from there. Resolved
#: against this file so the script works from any working directory.
WORLD_MAP = str(Path(__file__).resolve().parent / "performance" / "world_map.png")

os.environ.setdefault("ALGAN_USE_DAEMON", "0")
os.environ.setdefault("ALGAN_VIDEO_ENCODER", "software")

import torch  # noqa: E402

from algan import *  # noqa: E402, F403
from algan.mobs.neural_nets.neural_net import NeuralNetMLPV3  # noqa: E402
from algan.utils import memory_utils as mu  # noqa: E402

RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 2
QUALITY = (sys.argv[2] if len(sys.argv) > 2 else "UHD").upper()
PRESET = {"UHD": UHD, "HD": HD, "MD": MD, "PREVIEW": PREVIEW}[QUALITY]

#: Filled by the hooks below, reset per render.
STATS = {"arenas": [], "batches": 0, "chunks": 0}

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

import algan.rendering.raytracing.tracer as rtr  # noqa: E402

_real_wavefront = rtr.raytrace_render_wavefront


def _wavefront(*args, **kwargs):
    STATS["chunks"] += 1
    return _real_wavefront(*args, **kwargs)


rtr.raytrace_render_wavefront = _wavefront


def _pool():
    """The two MPS figures the free-bytes probe is built from, in GiB."""
    if not torch.mps.is_available():
        return "n/a"
    return (
        f"driver={torch.mps.driver_allocated_memory() / 2**30:.2f}G "
        f"current={torch.mps.current_allocated_memory() / 2**30:.2f}G "
        f"recommended={torch.mps.recommended_max_memory() / 2**30:.2f}G"
    )


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
    print(f"quality={QUALITY} runs={RUNS}", flush=True)
    print(f"pool before any render: {_pool()}", flush=True)
    for i in range(1, RUNS + 1):
        STATS["arenas"].clear()
        STATS["batches"] = STATS["chunks"] = 0
        SceneManager.reset()
        Scene.set_video_settings(PRESET)
        scene()
        started = time.perf_counter()
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
        print(f"  pool after run {i}: {_pool()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
