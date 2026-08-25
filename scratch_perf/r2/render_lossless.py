"""Render the nn benchmark scene once, losslessly.  usage: render_lossless.py <out.mp4> [QUALITY]

Lossless because an H.264 re-encode amplifies a single-channel difference into
thousands of differing pixels, which makes a byte-identity comparison useless
(see the round-1 notes).  The memory override is pinned so both arms of an A/B
choose the same batch windows -- a different window legitimately moves pixels,
because chord counts and dice levels are batch-wide maxima.
"""

import os
import sys
import time

os.environ["ALGAN_USE_DAEMON"] = "0"

from algan import *  # noqa: F403
from algan.mobs.neural_nets.neural_net import NeuralNetMLPV3

OUT = sys.argv[1] if len(sys.argv) > 1 else "out.mp4"
QUALITY = sys.argv[2].upper() if len(sys.argv) > 2 else "PREVIEW"
_PRESETS = {"UHD": UHD, "HD": HD, "PREVIEW": PREVIEW, "MD": MD, "LD": LD}

SETTINGS.computing.set(available_memory_override=3 * 1024**3)
SETTINGS.raytracing.set(shadows=True)

run_time = 5.0 if QUALITY == "PREVIEW" else 0.5
with Off():
    nn = NeuralNetMLPV3([5, 5, 5, 5]).move(LEFT).spawn()
    x = ImageMob("world_map.png").move_next_to(nn, LEFT).spawn()
    label = (
        Text("Neural Net MLP v3 processing an image of the globe")
        .move_next_to(nn, DOWN)
        .spawn()
    )
with Sync(run_time=run_time):
    nn.move(UP)
    x.color_texture = x.color_texture * 0.5
    label.move(RIGHT * 2)

t0 = time.perf_counter()
r = Scene.save_video(
    OUT,
    _PRESETS[QUALITY],
    overwrite=True,
    reset=True,
    ffmpeg_params=["-c:v", "libx264rgb", "-qp", "0"],
)
print(f"rendered {OUT} ({QUALITY}) in {time.perf_counter() - t0:.1f}s", flush=True)
