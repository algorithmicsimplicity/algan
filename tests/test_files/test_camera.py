import torch.nn.functional as F

from algan.animation.animation_contexts import Seq
from algan.constants.spatial import *#RIGHT, LEFT, IN, OUT, ORIGIN, UP
from algan.utils.algan_utils import render_all_funcs
from algan.mobs.shapes_2d import TriangleTriangulated


get_mob = lambda r=0: TriangleTriangulated(torch.stack((UP * 0.5,
                                                        F.normalize(RIGHT+DOWN,p=2,dim=-1) * 0.5,
                                                        F.normalize(LEFT+DOWN,p=2,dim=-1) * 0.5)), color=torch.stack([PURE_RED, PURE_BLUE, PURE_GREEN])).spawn()


def test_camera():
    x1 = get_mob()
    #x1.move(RIGHT*0.5)
    with Seq():
        for d in [LEFT, UP, RIGHT, DOWN]:
            x1.move(d)
            x1.scene.camera.move_to_make_mob_center_of_view(x1)
    return


render_all_funcs(__name__, start_index=0, max_rendered=-1)
