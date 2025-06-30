from algan import *
import manim as mn

def render_static_beziers():
    mobs = ManimMob(mn.Text('abcdefir\nsbmbbkl')).scale(2).spawn()
    mobs.wait(2)

render_all_funcs(__name__, HD)

