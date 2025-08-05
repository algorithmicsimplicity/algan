from algan import *
import manim as mn


def render_static_text():
    mobs = Group([ManimMob(mn.Text('a')) for _ in range(250)]).arrange_in_grid().scale(1/10).spawn()
    mobs.wait(2)


#COMPUTING_DEFAULTS.compiled = False
COMPUTING_DEFAULTS.max_animate_batch_size = 1000
#COMPUTING_DEFAULTS.render_device = torch.device('cpu')
render_all_funcs(__name__, HD)
