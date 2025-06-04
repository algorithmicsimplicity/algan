import torch

from algan.animation.animation_contexts import Off
from algan.mobs.mob import Mob


class Renderable(Mob):
    """
    Base class for all objects that appear on screen.
    """

    def on_create(self):
        with Off():
            self.opacity = 0
        self.opacity = self.max_opacity

    def on_destroy(self):
        self.opacity = torch.tensor((0.0,)).view(1)
