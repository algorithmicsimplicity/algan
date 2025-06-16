from importlib.metadata import version

__version__ = '0.0.7'# version(__name__)

import os
import shutil
import torch

torch.set_grad_enabled(False)
c = torch.inference_mode()
c.__enter__()

try:
    @torch.compile
    def _dummy_func(x):
        return x + 1

    # Test the dummy function
    _dummy_func(torch.tensor(1.0))

    compiled = torch.compile
    print('using torch.compile')
except:
    compiled = lambda x: x

from algan.defaults.batch_defaults import *
from algan.defaults.device_defaults import *
from algan.defaults.style_defaults import *
from algan.defaults.render_defaults import *
from algan.defaults.directory_defaults import *

from algan.settings.render_settings import *

from algan.constants.spatial import *
from algan.constants.color import *
from algan.constants.math import *
from algan.rendering import camera

from algan.mobs.mob import *
from algan.mobs.manim_mob import *
from algan.mobs.group import *
from algan.mobs.text import *
from algan.mobs.image_mob import *
from algan.mobs.surfaces.surface import *
from algan.mobs.shapes_3d import *
from algan.mobs.shapes_2d import *
from algan.mobs.bezier_circuit import *

from algan.animation.animation_contexts import *

from algan.utils.algan_utils import *


def clear_cache():
    f = os.path.join(algan.defaults.directory_defaults.DEFAULT_DIR, 'algan_cache')
    if os.path.exists(f):
        shutil.rmtree(f)