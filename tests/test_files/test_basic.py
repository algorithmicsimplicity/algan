import torch.nn.functional as F

from algan.animation.animation_contexts import Sync, AnimationContext
from algan.constants.spatial import *#RIGHT, LEFT, IN, OUT, ORIGIN, UP
from algan.mobs.shapes_2d import TriangleTriangulated
from algan.utils.algan_utils import render_all_funcs



get_mob = lambda r=0: TriangleTriangulated(torch.stack((UP * 0.5,
                                                        F.normalize(RIGHT+DOWN,p=2,dim=-1) * 0.5,
                                                        F.normalize(LEFT+DOWN,p=2,dim=-1) * 0.5)), color=torch.stack([PURE_RED, PURE_BLUE, PURE_GREEN])).spawn()


def test_move():
    x = get_mob()
    x.move(RIGHT)
    return


def test_rotate():
    x = get_mob()
    x.rotate(360, UP)
    x.rotate(-360, UP)


def test_sync():
    x = get_mob()
    x2 = get_mob()
    with Sync():
        x.move(LEFT*0.5)
        x2.move(RIGHT*0.5)
    with Sync():
        x.rotate(360, OUT)
        x2.rotate(-360, OUT)


def test_multi():
    x = get_mob()
    with Sync():
        x.move(RIGHT*0.5)
        x.move(UP*0.5)
        x.rotate(180, OUT)
        x.rotate(180, RIGHT)


def test_prespawn():
    with AnimationContext(prevent_spawn=True):
        x = get_mob()
    for _ in range(10):
        x.move(RIGHT/10)
    x.spawn()
    x.move(LEFT)

render_all_funcs(__name__, start_index=1, max_rendered=1)
