import os
os.environ["ALGAN_USE_DAEMON"] = "0"

from algan import *
from algan.utils.profiling_utils import profile_scene
from algan.mobs.neural_nets.neural_net import NeuralNetMLPV3

def scene():
    run_time=5
    SETTINGS.raytracing.set(shadows=True)
    
    with Off():
        nn = NeuralNetMLPV3([5, 5, 5, 5]).move(LEFT).spawn()
        x = ImageMob('world_map.png').move_next_to(nn, LEFT).spawn()
        label = Text('Neural Net MLP v3 processing an image of the globe').move_next_to(nn, DOWN).spawn()

    with Sync(run_time=run_time):
        nn.move(UP)
        x.color_texture = x.color_texture.view(x.texture_width, -1, 5) * 0.5
        label.move(RIGHT*2)

profile_scene(scene, PREVIEW, "nn_PREVIEW", runs=2, kernel_profiler=False)