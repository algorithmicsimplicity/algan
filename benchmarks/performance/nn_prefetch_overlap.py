"""Does overlapping the arena preflight pay?  usage: <script> base|ovl [QUALITY]

`SETTINGS.computing.prefetch_gpu_prep` lets the batch-prep worker run batch
b+1's render-device projection and scene merge *while batch b renders*, instead
of leaving both to the render thread's arena preflight, which otherwise runs
with nothing in flight.  It is off by default.  On a T4 the preflight measured
16.8% of a warm UHD render and ~30% of a PREVIEW one, so this is the largest
lever in the profile that needs no new code.

Same scene as ``nn_ablation.py base``, so the arms are comparable to it.
Two arms, one process each:

base   the default -- preparation on the render thread, between batches
ovl    ``prefetch_gpu_prep=True``

Both arms print an OVERLAP line reporting how many batches were actually
prepared on the worker.  That line is the point of this script rather than
``nn_ablation.py ovl``: the overlap is gated on the GPU projection *and* merge
both being active, it is skipped for the first batch of a render by design, and
a failure inside it is caught and downgraded to a render-thread preparation with
only a logged warning.  Every one of those is silent in a timing, and each turns
the arm into a null measurement that reads as "neutral".  ``base`` must report 0
and ``ovl`` must report a nonzero count, or the comparison means nothing.

Note the ceiling this implies: the first batch of a render never overlaps, so a
two-batch render can hide at most the *second* batch's preparation.  Renders
with more batches have more to gain.
"""

import os
import sys

os.environ["ALGAN_USE_DAEMON"] = "0"

from algan import *  # noqa: F403, E402
from algan.mobs.neural_nets.neural_net import NeuralNetMLPV3  # noqa: E402
from algan.render_loop import RenderLoopMixin  # noqa: E402
from algan.utils.profiling_utils import profile_scene  # noqa: E402

ARM = (sys.argv[1] if len(sys.argv) > 1 else "base").strip().lower()
QUALITY = (sys.argv[2].upper() if len(sys.argv) > 2 else "UHD").strip()
if ARM not in ("base", "ovl"):
    raise SystemExit(f"arm must be 'base' or 'ovl', got {ARM!r}")
_PRESETS = {"UHD": UHD, "HD": HD, "PREVIEW": PREVIEW, "MD": MD, "LD": LD}

OVERLAPPED = []
_prepare_on_worker = RenderLoopMixin._prepare_batch_on_worker


def _counting_prepare(self, primitive_batch, render_state):
    result = _prepare_on_worker(self, primitive_batch, render_state)
    OVERLAPPED.append(True)
    print(f"OVERLAP: batch prepared on the worker (#{len(OVERLAPPED)})", flush=True)
    return result


RenderLoopMixin._prepare_batch_on_worker = _counting_prepare


def scene():
    duration = 5.0 if QUALITY == "PREVIEW" else 0.5
    SETTINGS.raytracing.set(shadows=True)
    if ARM == "ovl":
        SETTINGS.computing.set(prefetch_gpu_prep=True)

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

    with Sync(runtime=duration):
        nn.move(UP)
        x.color_texture = x.color_texture * 0.5
        label.move(RIGHT * 2)


print(f"ARM={ARM}  QUALITY={QUALITY}", flush=True)
profile_scene(
    scene,
    _PRESETS[QUALITY],
    f"nn_pf_{ARM}_{QUALITY}",
    runs=2,
    kernel_profiler=False,
    save_video_kwargs={"ffmpeg_params": ["-crf", "17", "-preset", "ultrafast"]},
)

print(f"\nOVERLAP TOTAL: {len(OVERLAPPED)} batches prepared on the worker")
if ARM == "ovl" and not OVERLAPPED:
    print(
        "OVERLAP VERDICT: the ovl arm never overlapped a batch -- this timing is "
        "a null measurement, not a neutral result.",
        flush=True,
    )
elif ARM == "base" and OVERLAPPED:
    print("OVERLAP VERDICT: the base arm overlapped; the arms are not separated.")
else:
    print("OVERLAP VERDICT: arm behaved as intended.")
