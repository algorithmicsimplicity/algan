from algan import *
import manim as mn


def render_static_triangles():
    mobs = Group([Sphere() for _ in range(9)]).arrange_in_grid().spawn()
    mobs.wait(100)


#COMPUTING_DEFAULTS.compiled = False
COMPUTING_DEFAULTS.max_animate_batch_size = 1000
#COMPUTING_DEFAULTS.render_device = torch.device('cpu')
render_all_funcs(__name__, PREVIEW)
