"""torch.profiler capture (CPU+CUDA) of a short UHD render of the nn scene.

Run from ``benchmarks/performance``.  usage: probe_torch_profiler_uhd.py [frames]

Answers one question: of the wall time the stage profiler charges to
``wavefront_loop``'s *exclusive* column (13.2 s of a 29.9 s UHD run on a T4),
how much is Taichi kernel time, how much is torch GPU work, and how much is
host-side Python?  The stage timers cannot tell those apart because they sync at
stage boundaries; a torch profile can.
"""

import os
import sys

os.environ["ALGAN_USE_DAEMON"] = "0"

import torch
from torch.profiler import ProfilerActivity, profile

from algan import *  # noqa: F403
from algan.mobs.neural_nets.neural_net import NeuralNetMLPV3
from algan.scene_manager import SceneManager

frames = int(sys.argv[1]) if len(sys.argv) > 1 else 6


def scene(duration):
    SETTINGS.raytracing.set(shadows=True)
    with Off():
        nn = NeuralNetMLPV3([5, 5, 5, 5]).move(LEFT).spawn()
        x = ImageMob("world_map.png").move_next_to(nn, LEFT).spawn()
        label = (
            Text("Neural Net MLP v3 processing an image of the globe")
            .move_next_to(nn, DOWN)
            .spawn()
        )
    with Sync(duration=duration):
        nn.move(UP)
        x.color_texture = x.color_texture * 0.5
        label.move(RIGHT * 2)


# warm run (compiles kernels / fills the memory model), then the profiled run
scene(frames / 60)
Scene.save_video(
    "tp_warm", UHD, overwrite=True, reset=True, ffmpeg_params=["-preset", "ultrafast"]
)
SceneManager.reset()
scene(frames / 60)
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False
) as prof:
    r = Scene.save_video(
        "tp_prof",
        UHD,
        overwrite=True,
        reset=True,
        ffmpeg_params=["-preset", "ultrafast"],
    )
print(f"rendered {r.duration_seconds:.1f}s for {frames} frames", flush=True)
ka = prof.key_averages()
print("================ BY CUDA TIME ================", flush=True)
print(ka.table(sort_by="cuda_time_total", row_limit=50, max_name_column_width=70))
print("================ BY CPU TIME =================", flush=True)
print(ka.table(sort_by="self_cpu_time_total", row_limit=40, max_name_column_width=70))
