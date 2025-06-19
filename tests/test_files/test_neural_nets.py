from algan.mobs.neural_nets.neural_net import NeuralNetMLP
from algan.utils.algan_utils import render_all_funcs


def test_mlp():
    nn = NeuralNetMLP([3, 3, 3]).spawn()
    nn.activate()


render_all_funcs(__name__, start_index=0, max_rendered=1)
