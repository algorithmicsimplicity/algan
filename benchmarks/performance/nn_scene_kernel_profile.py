"""The two nn benchmark scenes, re-run with Taichi's per-kernel GPU profiler on.

``nn_scene_PREVIEW.py`` / ``nn_scene_UHD.py`` are the wall-clock benchmarks and
deliberately keep ``kernel_profiler=False``: enabling it re-initializes the
Taichi runtime and adds per-launch instrumentation, which is exactly what you do
not want in the number you are trying to move.

This script renders the *same* scenes with it on, so a round of optimization can
also see where GPU time (as opposed to wall time) goes, and which kernels are
launch-bound rather than compute-bound. Read it alongside the wall-clock report,
never instead of it.

    python benchmarks/performance/nn_scene_kernel_profile.py PREVIEW
    python benchmarks/performance/nn_scene_kernel_profile.py UHD
"""

from __future__ import annotations

import os
import sys

os.environ["ALGAN_USE_DAEMON"] = "0"

from algan import *  # noqa: E402
from algan.mobs.neural_nets.neural_net import NeuralNetMLPV3  # noqa: E402
from algan.utils.profiling_utils import profile_scene  # noqa: E402

QUALITY = (sys.argv[1] if len(sys.argv) > 1 else "PREVIEW").upper()
# PREVIEW renders 5 s of animation, UHD 0.5 s: the two scripts differ only in
# that and in the preset, so the 4K arm stays a ~30 s job rather than a 5 min one.
DURATION = {"PREVIEW": 5, "UHD": 0.5}[QUALITY]
SETTINGS_PRESET = {"PREVIEW": PREVIEW, "UHD": UHD}[QUALITY]


def scene():
    SETTINGS.raytracing.set(shadows=True)

    with Off():
        nn = NeuralNetMLPV3([5, 5, 5, 5]).move(LEFT).spawn()
        x = ImageMob("world_map.png").move_next_to(nn, LEFT).spawn()
        label = (
            Text("Neural Net MLP v3 processing an image of the globe")
            .move_next_to(nn, DOWN)
            .spawn()
        )

    with Sync(runtime=DURATION):
        nn.move(UP)
        x.color_texture = x.color_texture * 0.5
        label.move(RIGHT * 2)


profile_scene(
    scene,
    SETTINGS_PRESET,
    f"nn_{QUALITY}_kp",
    runs=2,
    kernel_profiler=True,
    save_video_kwargs={"ffmpeg_params": ["-crf", "17", "-preset", "ultrafast"]},
)
