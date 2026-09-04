"""Size the UHD render's cost levers by ablation.  usage: nn_ablation.py <arm>

Each arm is one process (settings behind ``ti.static`` gates are baked at kernel
compile time, so two arms in one process would silently share the first arm's
code).  Every arm renders the same scene as ``nn_scene_UHD.py`` and reports the
warm RUN 2 wall time, so the arms are directly comparable to the ``r2base``
baseline of 29.90 s.

Arms
----
base       the reference configuration
noshadow   ``shadows=False`` -- sizes the shadow half of ``wavefront_shade``
           plus the whole of ``raster_shadow_trace``
sec1       ``ALGAN_ANALYTIC_AA_SECONDARY=1`` (set in the environment, this
           script only records it) -- one continuation ray per reflective
           fragment instead of four
b1         ``max_bounces=1``
b0         ``max_bounces=0`` -- no continuations at all, so the sheet resolve
           alone; the floor this scene could reach
ovl        ``prefetch_gpu_prep=True`` -- run the batch's projection and GPU
           merge on the prefetch worker, beside the previous batch's render,
           instead of on the render thread between batches. Output-identical by
           construction (same builds, same inputs, same device), so this arm's
           video digest is the parity check as well as the timing.
"""

import os
import sys

os.environ["ALGAN_USE_DAEMON"] = "0"

from algan import *  # noqa: F403
from algan.mobs.neural_nets.neural_net import NeuralNetMLPV3
from algan.utils.profiling_utils import profile_scene

ARM = sys.argv[1] if len(sys.argv) > 1 else "base"
# Second argument picks the quality preset, so the same arms can be run at
# PREVIEW on a 4 GB card and at UHD on the T4.
QUALITY = sys.argv[2].upper() if len(sys.argv) > 2 else "UHD"


def scene():
    duration = 5.0 if QUALITY == "PREVIEW" else 0.5
    SETTINGS.raytracing.set(shadows=(ARM != "noshadow"))
    if ARM == "b1":
        SETTINGS.raytracing.set(max_bounces=1)
    elif ARM == "b0":
        SETTINGS.raytracing.set(max_bounces=0)
    elif ARM == "ovl":
        SETTINGS.computing.set(prefetch_gpu_prep=True)

    with Off():
        nn = NeuralNetMLPV3([5, 5, 5, 5]).move(LEFT).spawn()
        x = ImageMob("world_map.png").move_next_to(nn, LEFT).spawn()
        label = (
            Text("Neural Net MLP v3 processing an image of the globe")
            .move_next_to(nn, DOWN)
            .spawn()
        )

    with Sync(runtime=duration):
        nn.move(UP)
        x.color_texture = x.color_texture * 0.5
        label.move(RIGHT * 2)


# Codegen knobs ride in the output tag, not just in the log. Every step of a
# harness run writes into one `algan_outputs/`, so two arms that share a tag
# also share their video filenames -- and the digests the runner reports then
# all come from whichever step ran last, which silently destroys the parity
# half of an A/B. These four change only how the kernels are COMPILED, so the
# frames must come out identical, and that is exactly the claim the digest is
# there to check.
_CODEGEN_ENV = (
    ("ALGAN_GPU_MAX_REG", "reg"),
    ("ALGAN_OPT_LEVEL", "opt"),
    ("ALGAN_ADV_OPT", "adv"),
    ("ALGAN_ANALYTIC_AA_SECONDARY", "sec"),
)
_SUFFIX = "".join(
    f"_{short}{os.environ[name]}" for name, short in _CODEGEN_ENV if name in os.environ
)

print(
    f"ARM={ARM}  QUALITY={QUALITY}  "
    + "  ".join(
        f"{name}={os.environ.get(name, '(default)')}" for name, _ in _CODEGEN_ENV
    ),
    flush=True,
)
_PRESETS = {"UHD": UHD, "HD": HD, "PREVIEW": PREVIEW, "MD": MD, "LD": LD}
profile_scene(
    scene,
    _PRESETS[QUALITY],
    f"nn_abl_{ARM}_{QUALITY}{_SUFFIX}",
    runs=2,
    kernel_profiler=False,
    save_video_kwargs={"ffmpeg_params": ["-crf", "17", "-preset", "ultrafast"]},
)
