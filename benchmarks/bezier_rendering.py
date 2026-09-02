from __future__ import annotations

import os

os.environ["ALGAN_PREFETCH_BATCHES"] = "0"
os.environ["ALGAN_ADV_OPT"] = "0"

import torch

# Benchmarks must never be measured inside a warm daemon: it keeps adaptive
# renderer state (the memory model's batch-size fit) across runs, so one
# benchmark would be timed against whatever ran before it.
os.environ.setdefault("ALGAN_USE_DAEMON", "0")

from algan import *
from algan.mobs.neural_nets.neural_net import NeuralNetMLPV3
from algan.utils.profiling_utils import profile_scene


def Boxed(mob, color=BLUE, buffer=0.1, *args, **kwargs):
    return Group(
        mob,
        SurroundingRectangle(
            mob,
            *args,
            color=color.lerp(BLACK, 0.8).lerp(PURE_BLUE, 0.1).set_opacity(0.95),
            stroke_color=torch.lerp(color, BLACK, 0.2),
            buffer=buffer,
            stroke_width=1,
            **kwargs,
        ),
    )


def GlowTex(c, *args, **kwargs):
    m = (
        Tex(*args, **kwargs)
        .set(
            color=c + GLOW * 0.01,
            stroke_color=torch.lerp(c, WHITE, 0.9),
            stroke_width=0.8,
        )
        .scale(0.75)
    )
    return m


text_string = ("a" * 50 + "\n") * 50


def text_scene():
    with Off():
        nn = NeuralNetMLPV3([3, 3, 3]).spawn()
        mob = Boxed(GlowTex(GREEN, text_string)).spawn()
    with Sync(runtime=1):
        mob.move(LEFT)
        nn.move(LEFT)
    return


set_log_level("DEBUG")

profile_scene(text_scene, UHD.set_frames_per_second(60), runs=2, kernel_profiler=False)
