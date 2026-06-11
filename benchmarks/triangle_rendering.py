from algan import *
import manim as mn


def render_static_triangles():
    m1 = QuadTriangulated(torch.stack([UP, RIGHT, DOWN, LEFT]), color=RED).spawn()
    m2 = QuadTriangulated(torch.stack([UP, RIGHT, DOWN, LEFT]), color=GREEN).move(OUT*0.01).spawn()
    m1.move(RIGHT)
    m1.move(LEFT*2)
    #mobs = Group([Sphere() for _ in range(9)]).arrange_in_grid().spawn()
    #mobs.wait(1)


#COMPUTING_DEFAULTS.compiled = False
rs = PREVIEW
rs.fxaa = False
#COMPUTING_DEFAULTS.render_device = torch.device('cpu')
render_all_funcs(__name__, rs)
