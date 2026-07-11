import torch

from algan.animation.animation_contexts import Off, Seq
from algan.mobs.mob import Mob


class Renderable(Mob):
    """
    Base class for all objects that appear on screen.
    """

    def on_create(self):
        opacity = self.opacity
        with Seq():
            with Off():
                self.opacity = 0
            self.opacity = opacity

    # The fade above writes the whole subtree (recursive opacity set):
    # _create_recursive records it once per spawned subtree instead of once
    # per mob (see Animatable._create_recursive).
    on_create._is_stock_fade = True

    def on_destroy(self):
        self.opacity = torch.tensor((0.0,)).view(1)
