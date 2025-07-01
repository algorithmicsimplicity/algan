from algan import *
import manim as mn

def render_static_triangles():
    mobs = Group([Sphere() for _ in range(9)]).arrange_in_grid().spawn()
    #mobs.wait(2)

render_all_funcs(__name__, PREVIEW)

