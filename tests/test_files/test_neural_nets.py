import torch.nn.functional as F

from algan.constants.spatial import *#RIGHT, LEFT, IN, OUT, ORIGIN, UP
from algan.mobs.neural_nets.neural_net import NeuralNetMLP
from algan.utils.algan_utils import render_all_funcs
from algan.mobs.shapes_2d import TriangleTriangulated


get_mob = lambda r=0: TriangleTriangulated(torch.stack((UP * 0.5,
                                                        F.normalize(RIGHT+DOWN,p=2,dim=-1) * 0.5,
                                                        F.normalize(LEFT+DOWN,p=2,dim=-1) * 0.5)), color=torch.stack([PURE_RED, PURE_BLUE, PURE_GREEN])).spawn()


def test_mlp():
    nn = NeuralNetMLP([3, 3, 3]).spawn()
    nn.activate()


render_all_funcs(__name__, start_index=0, max_rendered=1)
