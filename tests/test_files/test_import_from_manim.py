import manim as mn

from algan.animation.animation_contexts import Sync, Off
from algan.constants.spatial import *#RIGHT, LEFT, IN, OUT, ORIGIN, UP
from algan.mobs.manim_mob import ManimMob
from algan.utils.algan_utils import render_all_funcs


def test_manim_text():
    with Off():
        x = ManimMob(mn.Text('abba')).spawn()#TriangleTriangulated(control_points, PolygonVertices, constants=GREEN)
        #x2 = get_mob()
        x.border_color = PURE_GREEN
        #x.portion_of_curve_drawn = 0
    with Sync(run_time=2):
        #x.location = RIGHT
        x.color = PURE_RED
        x.border_width = 10
        x.portion_of_curve_drawn = 1
    x.wait(0.5)
    x.despawn()
    #with Synchronized():
    #    x.rotate(-360, UP)

    #x2.rotate(-45, UP)
    #x.move(RIGHT)
    #x.location = x.location + RIGHT
    return


def test_manim_tex():
    def Vector(elements):
        with Off():
            return ManimMob(mn.Tex('2')).set(border_width=0, border_color=PURE_GREEN).move(RIGHT*1.3)
            return ManimMob(mn.DecimalMatrix([[_] for _ in elements])).set(border_width=0).move(RIGHT*1.3)
    x = Vector(elements=[0.2, 0.2, 0.2]).spawn()
    #with Sequenced(run_time=3):
    #    x.move(UP*0.9)
    x.wait()


def test_manim_sphere():
    x = ManimMob(mn.Sphere()).scale(0.2).spawn()
    x.rotate(180, UP)
    x.wait()


render_all_funcs(__name__, start_index=-2, max_rendered=1)
