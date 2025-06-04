import torch.nn.functional as F

from algan.animation.animation_contexts import Sync
from algan.constants.spatial import *#RIGHT, LEFT, IN, OUT, ORIGIN, UP
from algan.mobs.shapes_2d import TriangleTriangulated
from algan.utils.algan_utils import render_all_funcs


s = 0.1
get_mob = lambda r=0: TriangleTriangulated(torch.stack((UP * s,
                                                        F.normalize(RIGHT+DOWN,p=2,dim=-1) * s,
                                                        F.normalize(LEFT+DOWN,p=2,dim=-1) * s)), color=torch.stack([PURE_RED, PURE_BLUE, PURE_GREEN])).spawn()


def test_synchronized():
    x = get_mob()
    with Sync():
        x.move(RIGHT)
        x.move(UP)
        x.rotate(360, UP+OUT+RIGHT)
        x.rotate(720, UP + LEFT)
        x.scale(3)
    return


render_all_funcs(__name__, start_index=-1, max_rendered=-1)
