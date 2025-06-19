from importlib.metadata import version

__version__ = version(__name__)

import os
import shutil
import torch

from algan.defaults.batch_defaults import *
from algan.defaults.device_defaults import *
from algan.defaults.style_defaults import *
from algan.defaults.render_defaults import *
from algan.defaults.directory_defaults import *

from algan.utils.memory_utils import ManualMemory

torch.set_grad_enabled(False)
c = torch.inference_mode()
c.__enter__()

try:
    @torch.compile
    def _dummy_func(x):
        return x + 1

    # Test the dummy function
    _dummy_func(torch.tensor(1.0))

    #compiled = torch.compile
    #print('using torch.compile')
    compiled = lambda x: x
except:
    compiled = lambda x: x


class SceneManager:
    _instance = None
    _memory = None
    _scene_class = None
    _scene_initializer = None

    def __init__(self):
        raise RuntimeError('Call SceneTracker.instance() instead of SceneTracker().')

    @classmethod
    def set_scene_class(cls, scene_class, scene_initializer):
        cls._scene_class = scene_class
        cls._scene_initializer = scene_initializer

    @classmethod
    def reset(cls):
        AnimationManager.reset()
        cls._instance = None
        return cls.instance()

    @classmethod
    def instance(cls):
        if cls._instance is None:
            if cls._memory is None:
                cls._memory = ManualMemory(((int((torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0))
                                                 *algan.defaults.batch_defaults.DEFAULT_PORTION_MEMORY_USED_FOR_RENDERING))))
            cls._instance = cls._scene_class(memory=cls._memory)
            cls._instance.scene_initializer = cls._scene_initializer
            cls._instance.reset_scene()
        return cls._instance


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
from algan.scene import Scene

from algan.animation.animation_contexts import *

from algan.utils.algan_utils import *

from algan.rendering.lights import *


def clear_cache():
    f = os.path.join(algan.defaults.directory_defaults.DEFAULT_DIR, 'algan_cache')
    if os.path.exists(f):
        shutil.rmtree(f)

def default_scene_initializer(scene):
    scene.camera = Camera(location=algan.constants.spatial.CAMERA_ORIGIN)
    scene.light_sources = [PointLight(location=scene.camera.location + UP * 1 + RIGHT * 5 + OUT * 1,
                                     color=WHITE).spawn(animate=False)]

SceneManager.set_scene_class(algan.scene.Scene, default_scene_initializer)
