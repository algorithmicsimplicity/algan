from algan.animation.animation_contexts import Sync, Off
from algan.constants.spatial import *#RIGHT, LEFT, IN, OUT, ORIGIN, UP
from algan.mobs.bezier_circuit import BezierCircuitCubic, BezierCurveCubic
from algan.utils.algan_utils import render_all_funcs

p = torch.stack((
    torch.stack((UP*0.5, UP*0.4+RIGHT*0.1, UP*0.2+RIGHT*0.5, RIGHT*0.1)),
    torch.stack((RIGHT*0.1, RIGHT* 0.4 + DOWN*0.3, RIGHT*0.3 + DOWN*0.4, DOWN*0.1)),
))

#p = p - p.mean((0,1))
#p = p * 4

get_mob = lambda r=0: BezierCurveCubic(p).spawn()


def test_bezier_curve():
    with Off():
        x1 = get_mob()
        x2 = get_mob()
        x2.color = PURE_RED
    with Sync():
        x1.rotate(30, UP)
        x2.rotate(-360, UP)


render_all_funcs(__name__, start_index=0, max_rendered=1)
