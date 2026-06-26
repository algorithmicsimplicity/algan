from algan import *
from algan.mobs.neural_nets.neural_net import NeuralNetMLP
from algan.rendering.raytracing import enable_ray_tracing
from algan.rendering.raytracing.primitives import set_wavefront
from algan.utils.profiling_utils import profile_scene


def neural_net():
    with Off():
        nn = NeuralNetMLP([10, 10, 10]).set_material(MeshStandardMaterial(color=GREEN)).spawn()
        label = Text('Neural Network').move_next_to(nn, UP).spawn()
    nn.move(DOWN)


enable_ray_tracing(1, pn_triangles=True, fragment_shading=True, shadows=False)
render_settings = UHD
render_settings.anti_alias_level = 2
set_wavefront(True)
profile_scene(neural_net, render_settings, 'neural_net')