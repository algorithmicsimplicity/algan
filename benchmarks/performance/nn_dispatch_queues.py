"""Queue geometry of the Q1/Q2 primary shadow dispatch.  usage: <script> [QUALITY]

The A/B in ``nn_ablation.py`` answers whether ``ALGAN_DEVICE_DISPATCH`` is
faster.  This answers *what shape of work it is given*, which is the thing that
explains the answer, because the two arms do not launch the same grid:

* the reference arm compacts accepted events on the host and launches
  ``num_events * num_lights`` threads;
* the device-counted arm never learns the count on the host, so it launches
  ``capacity * num_lights`` -- and capacity is the number of *source sheets* in
  the slice, not the number of accepted events.

So the on-arm's trace grid is ``1 / acceptance_rate`` times the reference's.
The surplus threads read the device header, fail the ``idx < count`` guard and
return, but they are still launched.  A low acceptance rate is therefore the
mechanism to look for first when the toggle loses.

Renders the same scene as ``nn_ablation.py`` at the requested preset, for a
fraction of a second -- the queue shape is a resolution property, not a
frame-count one, so a couple of frames is the whole measurement.
"""

import os
import sys

os.environ["ALGAN_USE_DAEMON"] = "0"
os.environ["ALGAN_DEVICE_DISPATCH"] = "1"

from algan import *  # noqa: F403, E402
from algan.mobs.neural_nets.neural_net import NeuralNetMLPV3  # noqa: E402
from algan.rendering.raytracing import shadow_dispatch  # noqa: E402

QUALITY = (sys.argv[1].upper() if len(sys.argv) > 1 else "UHD").strip()
_PRESETS = {"UHD": UHD, "HD": HD, "PREVIEW": PREVIEW, "MD": MD, "LD": LD}

WINDOWS = []
_prepare = shadow_dispatch.prepare_primary_shadow_dispatch


def _record(memory, **kwargs):
    dispatch = _prepare(memory, **kwargs)
    n = kwargs["n"]
    accepted = int(kwargs["accepted"][:n].ne(0).sum().item())
    WINDOWS.append((n, accepted))
    print(
        f"window {len(WINDOWS) - 1}: sheets={n} accepted={accepted} "
        f"rate={accepted / max(1, n):.4f} overlaunch={n / max(1, accepted):.2f}x",
        flush=True,
    )
    return dispatch


# The pipeline imports the symbol directly, so patch it where it is called.
shadow_dispatch.prepare_primary_shadow_dispatch = _record
import algan.rendering.raytracing.raster_pipeline as raster_pipeline  # noqa: E402

raster_pipeline.prepare_primary_shadow_dispatch = _record


def scene():
    SETTINGS.raytracing.set(shadows=True)
    with Off():
        nn = NeuralNetMLPV3([5, 5, 5, 5]).move(LEFT).spawn()
        x = (
            ImageMob("benchmarks/performance/world_map.png")
            .move_next_to(nn, LEFT)
            .spawn()
        )
        label = (
            Text("Neural Net MLP v3 processing an image of the globe")
            .move_next_to(nn, DOWN)
            .spawn()
        )
    with Sync(runtime=0.1):
        nn.move(UP)
        x.color_texture = x.color_texture * 0.5
        label.move(RIGHT * 2)


scene()
Scene.save_video(f"nn_dispatch_queues_{QUALITY}", _PRESETS[QUALITY], overwrite=True)

sheets = sum(n for n, _ in WINDOWS)
accepted = sum(a for _, a in WINDOWS)
print(f"\nQUALITY={QUALITY}  coverage windows={len(WINDOWS)}")
print(f"source sheets={sheets}  accepted events={accepted}")
if sheets:
    rate = accepted / sheets
    print(f"acceptance rate={rate:.4f}  trace over-launch={1 / max(rate, 1e-9):.2f}x")
