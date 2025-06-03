import torch.nn.functional as F

from algan.animation.animation_contexts import Sync, Off, Seq
from algan.constants.spatial import *#RIGHT, LEFT, IN, OUT, ORIGIN, UP
from algan.mobs.shapes_2d import TriangleTriangulated
from algan.utils.algan_utils import render_all_funcs


get_mob = lambda r=0: TriangleTriangulated(torch.stack((UP * 0.5,
                                                        F.normalize(RIGHT+DOWN,p=2,dim=-1) * 0.5,
                                                        F.normalize(LEFT+DOWN,p=2,dim=-1) * 0.5)), color=torch.stack([PURE_RED, PURE_BLUE, PURE_GREEN])).spawn()


def test_context():
    with Off(spawn_at_end=True):
        x1 = get_mob().move(LEFT*0.5)
        x2 = get_mob().move(RIGHT*0.5)

    with Sync():
        with Seq(run_time=1):
            x1.move(UP*0.2)
            x1.move(RIGHT*0.2)
            #x1.data.history.attribute_modifications['location'][1][1]()
            #x1.data.history.attribute_modifications['location'][1][2]()
        x2.move(UP*0.4)
    return


render_all_funcs(__name__, start_index=0, max_rendered=-1)
