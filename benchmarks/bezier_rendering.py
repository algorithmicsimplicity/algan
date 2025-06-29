from algan import *
import manim as mn

def render_static_beziers():
    mobs = ManimMob(mn.MathTex('abcdefghijklmnopqrstuvwxyz')).spawn()
    mobs.wait(2)

render_all_funcs(__name__, PREVIEW)

