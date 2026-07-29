from algan import *
from algan.rendering.raytracing import enable_ray_tracing


def render_static_triangles():
    n = 100
    m = Sphere(grid_height=n, grid_width=n).scale(3).spawn()
    Scene.wait(5)


rs = HD
rs.fxaa = False
SETTINGS.computing.set(animation_memory_fraction=0.4, rendering_memory_fraction=0.4)
enable_ray_tracing()
render_all_funcs(__name__, rs)
