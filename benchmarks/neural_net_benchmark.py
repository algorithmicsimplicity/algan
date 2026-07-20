import os
os.environ["ALGAN_PREFETCH_BATCHES"] = "0"
from algan import *
from algan.mobs.neural_nets.neural_net import NeuralNetMLP
from algan.utils.profiling_utils import profile_scene
from algan.rendering.raytracing.primitives import RayTracedPNTrianglePrimitive, RayTracedTrianglePrimitive
RENDERER_SETTINGS.triangle_primitive = RayTracedTrianglePrimitive

def neural_net():
    with Off():
        nn = NeuralNetMLP([10, 10, 10]).set_material(MeshBasicMaterial(color=GREEN)).spawn()
        label = Text('Neural Network').move_next_to(nn, UP).spawn()
    #nn.activate()
    with Seq(run_time=1):
        nn.move(DOWN)


render_settings = HD#.set_anti_alias_level(2).set_frames_per_second(60)
profile_scene(neural_net, render_settings, 'neural_net', runs=2, kernel_profiler=False)
#neural_net()
#render_to_file('nn_profile2', render_settings=render_settings)
#render_all_funcs(__name__, render_settings=render_settings)