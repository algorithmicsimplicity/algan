import torch
import torch.nn.functional as F

import sys
sys.path.insert(-1, '/')
from algan.animation.animation_contexts import Seq  # , Runtime, Sequential
#from constants import BLUE
from algan.constants.spatial import RIGHT, UP
#from mobs.group import Group
from algan.mobs.text import Tex
from algan.utils.algan_utils import render_all_funcs
#from mobs.shapes_2d import Rectangle
from algan.utils.tensor_utils import unsquish, squish


def text():
    x = Tex("Hello", font_size=90)
    #p = x.character_mobs[0].location
    c1 = x.character_mobs[0]
    with Seq():
        c1.move(UP*0.1)
        c1.move(RIGHT* 0.1)
    return
    if True:
        k = 1
        x.character_mobs[0].tiles.location = x.character_mobs[0].tiles.location + squish((torch.randn_like(unsquish(x.character_mobs[0].tiles.location,0,k)[:,:1])*0.3).expand(-1,k,-1,-1))
        x.character_mobs[0].tiles.location = x.character_mobs[0].tiles.rotate(360, F.normalize(UP+RIGHT, p=2, dim=-1))
    x.move(UP)
    #with Runtime(1):
    #    x.rotate(180, OUT)
    return

render_all_funcs(__name__)
