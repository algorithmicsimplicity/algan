import os
os.environ['ALGAN_PREFETCH_BATCHES'] = "0"

import torch
from algan import *
from algan.mobs.neural_nets.neural_net import NeuralNetMLP, NeuralNetMLPV3
from algan.utils.profiling_utils import profile_scene

Boxed = lambda mob, color=BLUE, buffer=0.1, *args, **kwargs: Group(mob,
                                                                          SurroundingRectangle(mob,
                                                                                               color=color.lerp(BLACK, 0.8).lerp(PURE_BLUE, 0.1).set_opacity(
                                                                                                   0.95),
                                                                                               border_color=torch.lerp(
                                                                                                   color, BLACK, 0.2),
                                                                                               buffer=buffer,
                                                                                               border_width=1, *args,
                                                                                               **kwargs))
def GlowTex(c, *args, **kwargs):
    m = Tex(*args, **kwargs).set(color=c + GLOW * 0.01,
                                border_color=torch.lerp(c, WHITE, 0.9),
                                            border_width=0.8).scale(0.75)
    return m
text_string = ('a' * 50 + '\n') * 50

def text_scene():
    with Off():
        nn = NeuralNetMLPV3([3, 3, 3]).spawn()
        mob = Boxed(GlowTex(GREEN, text_string)).spawn()
    with Sync(run_time=1):
        mob.move(LEFT)
        nn.move(LEFT)
    return

set_log_level('DEBUG')

profile_scene(text_scene, UHD.set_frames_per_second(60), runs=2, kernel_profiler=False)
