"""A/B driver: nn scene at PREVIEW with the weight-floor exit off/on.

One process per arm (the gate feeds a ti.template(), so each arm must compile
its own variant in its own process). Usage:

    uv run python scratch_perf/r3/ox/ab_weight_floor.py <off|on> <output.mp4>
"""

from __future__ import annotations

import os
import sys

os.environ["ALGAN_USE_DAEMON"] = "0"

ARM = sys.argv[1]
OUTPUT = sys.argv[2]
assert ARM in ("off", "on")

WORLD_MAP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "world_map.png")

# Pinned identically in both arms so the frame-window split (and therefore
# the merged array layout) is byte-reproducible across processes. 8 GiB.
MEMORY_OVERRIDE = 8 * 1024**3

t0 = __import__("time").time()

from algan import (  # noqa: E402
    PREVIEW,
    ImageMob,
    DOWN,
    LEFT,
    Off,
    RIGHT,
    Scene,
    SETTINGS,
    Sync,
    Text,
    UP,
)
from algan.mobs.neural_nets.neural_net import NeuralNetMLPV3  # noqa: E402

SETTINGS.computing.set(available_memory_override=MEMORY_OVERRIDE)
SETTINGS.raytracing.experimental.weight_floor_exit = ARM == "on"
print(f"[ab] weight_floor_exit = {SETTINGS.raytracing.experimental.weight_floor_exit}")
print(f"[ab] available_memory_override = {MEMORY_OVERRIDE}")


def scene():
    run_time = 5
    SETTINGS.raytracing.set(shadows=True)

    with Off():
        nn = NeuralNetMLPV3([5, 5, 5, 5]).move(LEFT).spawn()
        x = ImageMob(WORLD_MAP).move_next_to(nn, LEFT).spawn()
        label = (
            Text("Neural Net MLP v3 processing an image of the globe")
            .move_next_to(nn, DOWN)
            .spawn()
        )

    with Sync(run_time=run_time):
        nn.move(UP)
        x.color_texture = x.color_texture * 0.5
        label.move(RIGHT * 2)


with Scene() as s:
    scene()
    result = s.save_video(
        OUTPUT,
        video_settings=PREVIEW,
        overwrite=True,
        ffmpeg_params=["-c:v", "libx264rgb", "-qp", "0"],
    )

print(f"[ab] arm={ARM} status={result.status} output={result.output_path}")
print(f"[ab] wall {__import__('time').time() - t0:.1f}s")
