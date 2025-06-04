import torch.nn.functional as F

from algan.animation.animation_contexts import Sync, Off
from algan.constants.spatial import *#RIGHT, LEFT, IN, OUT, ORIGIN, UP
from algan.mobs.bezier_circuit import BezierCircuitCubic
from algan.mobs.shapes_2d import TriangleTriangulated
from algan.utils.algan_utils import render_all_funcs


p = torch.stack((
    torch.stack((UP, UP+RIGHT*0.1, UP+RIGHT*0.2, UP+RIGHT*-0.2+DOWN*0.5)),
    torch.stack((UP+RIGHT*-0.2+DOWN*0.5, UP+RIGHT*-0.3+DOWN*0.4, UP+RIGHT*-0.2+DOWN*0.1, UP)),
))

p = p - p.mean((0,1))
p = p * 4

get_mob = lambda r=0: BezierCircuitCubic(p).spawn()


def test_bezier_circuit_and_triangle():
    with Off():
        x1 = get_mob()
        x2 = TriangleTriangulated(torch.stack((UP * 0.5,
                                               F.normalize(RIGHT + DOWN, p=2, dim=-1) * 0.5,
                                               F.normalize(LEFT + DOWN, p=2, dim=-1) * 0.5)),
                                  color=torch.stack([PURE_RED, PURE_BLUE, PURE_GREEN])).spawn()
    with Sync():
        x1.rotate(30, UP)
        x2.rotate(-360, UP)
    #x2.rotate(-45, UP)
    #x.move(RIGHT)
    #x.location = x.location + RIGHT
    return


render_all_funcs(__name__, start_index=0, max_rendered=-1)
