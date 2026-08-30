from __future__ import annotations

import os

os.environ["ALGAN_PREFETCH_BATCHES"] = "0"

# Benchmarks must never be measured inside a warm daemon: it keeps adaptive
# renderer state (the memory model's batch-size fit) across runs, so one
# benchmark would be timed against whatever ran before it.
os.environ.setdefault("ALGAN_USE_DAEMON", "0")

from algan import *
from algan.mobs.neural_nets.neural_net import NeuralNetMLP
from algan.rendering.raytracing.primitives import (
    RayTracedTrianglePrimitive,
)
from algan.utils.profiling_utils import profile_scene

RENDERER_SETTINGS.triangle_primitive = RayTracedTrianglePrimitive


def neural_net():
    with Off():
        nn = (
            NeuralNetMLP([10, 10, 10])
            .set_material(MeshBasicMaterial(color=GREEN))
            .spawn()
        )
        Text("Neural Network").move_next_to(nn, UP).spawn()
    # nn.activate()
    with Seq(duration=1):
        nn.move(DOWN)


video_settings = HD  # .set(ssaa=2, fps=60)
profile_scene(neural_net, video_settings, "neural_net", runs=2, kernel_profiler=False)
# neural_net()
# render_to_file('nn_profile2', video_settings=video_settings)
# render_all_funcs(__name__, video_settings=video_settings)
