from algan.animation.animation_contexts import Sync, Off
from algan.constants.spatial import *#RIGHT, LEFT, IN, OUT, ORIGIN, UP
from algan.mobs.bezier_circuit import BezierCircuitCubic
from algan.utils.algan_utils import render_all_funcs

p = torch.stack((
    torch.stack((UP, UP+RIGHT*0.1, UP+RIGHT*0.2, UP+RIGHT*-0.2+DOWN*0.5)),
    torch.stack((UP+RIGHT*-0.2+DOWN*0.5, UP+RIGHT*-0.3+DOWN*0.4, UP+RIGHT*-0.2+DOWN*0.1, UP)),
))

p = p - p.mean((0,1))
p = p * 4

get_mob = lambda r=0: BezierCircuitCubic(p).spawn()


def test_bezier_circuit_basic():
    with Off():
        x1 = get_mob()
        x2 = get_mob()
        x2.color = BLUE
    with Sync():
        x1.rotate(30, UP)
        x2.rotate(-360, UP)


def test_bezier_circuit_complex():
    with Off():
        c = torch.stack((LEFT+DOWN*0.5, RIGHT*0.5+UP*0.3, LEFT*0.4+UP*0.33, LEFT*0.7+DOWN*0.7))
        #c = torch.stack((c, c.flip(0)+DOWN*0.2))
        c = c.unsqueeze(0)
        x = BezierCircuitCubic(c).spawn()
        #x = get_bezier_points(p).spawn()
    #with Sequenced():
    #    x.location = x.location + torch.randn_like(x.location * 0.05)
    #    x.location = x.location + torch.randn_like(x.location * 0.05)
    x.wait(1)
    #x2.rotate(-45, UP)
    #x.move(RIGHT)
    #x.location = x.location + RIGHT
    return

render_all_funcs(__name__, start_index=0, max_rendered=1)
