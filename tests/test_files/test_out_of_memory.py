import torch.nn.functional as F

from algan.animation.animation_contexts import Sync
from algan.constants.spatial import *#RIGHT, LEFT, IN, OUT, ORIGIN, UP
from algan.utils.algan_utils import render_all_funcs
from algan.mobs.shapes_2d import TriangleTriangulated


get_mob = lambda a=0: TriangleTriangulated(torch.stack((UP * 10,
                                                        F.normalize(RIGHT+DOWN,p=2,dim=-1) * 10,
                                                        F.normalize(LEFT+DOWN,p=2,dim=-1) * 10)), color=torch.stack([PURE_RED, PURE_BLUE, PURE_GREEN])).spawn()


def test_OOM():
    with Sync():
        ts = [get_mob() for _ in range(5)]


render_all_funcs(__name__, start_index=0, max_rendered=1)

