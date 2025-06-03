import torch.nn.functional as F

from algan.constants.spatial import *#RIGHT, LEFT, IN, OUT, ORIGIN, UP
from algan.mobs.shapes_2d import TriangleTriangulated
from algan.settings.render_settings import UHD
from algan.utils.algan_utils import render_all_funcs


get_mob = lambda r=0: TriangleTriangulated(torch.stack((UP * 0.5,
                                                        F.normalize(RIGHT+DOWN,p=2,dim=-1) * 0.5,
                                                        F.normalize(LEFT+DOWN,p=2,dim=-1) * 0.5)), color=torch.stack([PURE_RED for _ in range(3)])).set(glow=1).spawn()


def test_glow():
    x = get_mob()
    #x.location = x.location + RIGHT
    x.glow = 1#0.1
    x.glow = 0.2
    x.glow = 0.5
    x.glow = 1
    #x.move(RIGHT)
    return

render_all_funcs(__name__, start_index=0, max_rendered=-1)
