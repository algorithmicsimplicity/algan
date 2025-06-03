import copy

import torch

from algan.defaults.style_defaults import DEFAULT_SPRITE_COLOR


class TextureMap:
    def __init__(self, specification=None):
        if specification is None:
            specification = DEFAULT_SPRITE_COLOR
        self.specification = specification
        if isinstance(specification, torch.Tensor):
            self.get_texture = self.get_constant_color#lambda coords: specification.view(([1] * (coords.dim()-1)) + [-1]).expand(list(coords.shape[:-1]) + [-1])

    def get_constant_color(self, coords):
        return self.specification.view(([1] * (coords.dim()-1)) + [-1]).expand(list(coords.shape[:-1]) + [-1])

    def __add__(self, other):
        if isinstance(other, TextureMap):
            other = other.specification
        cop = copy.deepcopy(self)
        cop.specification = cop.specification + other
        return cop

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if isinstance(other, TextureMap):
            other = other.specification
        cop = copy.deepcopy(self)
        cop.specification = cop.specification * other
        return cop

    def __rmul__(self, other):
        return self.__mul__(other)

    def __call__(self, *args, **kwargs):
        return self.get_texture(*args, **kwargs)