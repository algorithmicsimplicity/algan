import torch

from algan.animation.animation_contexts import Off, Seq
from algan.mobs.mob import Mob


class Renderable(Mob):
    """
    Base class for all objects that appear on screen.
    """

    def on_create(self):
        try:
            opacity = self.opacity
        except:
            print('debug')
            opacity = self.opacity
        with Seq():
            with Off():
                self.opacity = 0
                #self.glow = 0
            self.opacity = opacity

    def on_destroy(self):
        #with Sync():
        self.opacity = torch.tensor((0.0,)).view(1)
            #self.glow = torch.tensor((0.0,)).view(1)
