import torch.nn.functional as F

from algan.animation.animation_contexts import Sync, Off
from algan.constants.spatial import *#RIGHT, LEFT, IN, OUT, ORIGIN, UP
from algan.mobs.group import Group
from algan.mobs.shapes_2d import TriangleTriangulated
from algan.utils.algan_utils import render_all_funcs


s = 0.1
get_mob = lambda r=0: TriangleTriangulated(torch.stack((UP * s,
                                                        F.normalize(RIGHT+DOWN,p=2,dim=-1) * s,
                                                        F.normalize(LEFT+DOWN,p=2,dim=-1) * s)), color=torch.stack([PURE_RED, PURE_BLUE, PURE_GREEN])).spawn()


def test_parents():
    with Off():
        xs = Group([get_mob() for _ in range(3)]).arrange_in_line(RIGHT)
    with Sync():
        #TODO add in support for synchronized animating of parent and children,
        # currently it doesn't work because render_all always forces children -> parent state materialization
        # (even if parent needs to be called first, as it is below, so what we need to do is stratafy
        # state materialization into levels, each level is the number of animations playing previously in the same time.
        # we already have code which yoinks out the i'th level in animatable modification history,
        #all we need to do is, in render, iterate through levels one by one until no actor returns a non-zero state.

        xs.rotate(90, OUT)
        [x.move(RIGHT * 0.5) for x in xs]
        #xs.rotate(90, OUT)
        #[x.rotate(-45, OUT) for x in xs]
    return


render_all_funcs(__name__, start_index=-1, max_rendered=-1)
