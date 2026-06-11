from functools import wraps
from importlib.metadata import version

__version__ = version(__name__)

import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import shutil
import torch
torch.set_grad_enabled(False)
c = torch.inference_mode()
c.__enter__()
#import openvino.torch
#import torch_tensorrt


def exported(function=None, *, example_inputs=None, dynamic_shapes=None):
    def _decorate(func):
        class ModuleWrapper(torch.nn.Module):
            def __init__(self, func):
                super().__init__()
                self.forward = func

            #def forward(self, *args, **kwargs):
            #    return self.func(*args, **kwargs)

        mod = ModuleWrapper(func)
        #mod.forward = func
        ep = torch.export.export_for_inference(mod, example_inputs, dynamic_shapes=dynamic_shapes, strict=False,)
        return ep.module()
        @wraps(func)
        def wrapper_func(*args, **kwargs):
            return ep.module()(*args, **kwargs)
        return wrapper_func
    if function:
        return _decorate(function)
    return _decorate

'''default_compile_operation = lambda x: torch.compile(x, dynamic=False, fullgraph=False, backend="tensorrt",
                                                    options={"min_block_size": 1,
                                                             "use_python_runtime": True, }
                                                    )#"onnxrt")# mode="reduce-overhead")#backend='cudagraphs')'''
default_compile_operation = lambda x: torch.compile(x, dynamic=True, fullgraph=False)


class CudaStream():
    def __enter__(self):
        self.stream = None
        if False:#COMPUTING_DEFAULTS.compiled:
            self.stream = torch.cuda.Stream()
            self.stream.wait_stream(torch.cuda.default_stream(torch.device('cuda')))
            self.context = torch.cuda.stream(self.stream)
            self.context.__enter__()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type is not None:
            return False
        if self.stream is not None:
            return self.context.__exit__(exc_type, exc_value, exc_traceback)
        return True


def compile_wrapper(function):
    compiled_function = default_compile_operation(function)

    def _decorate(func, compiled_func):
        @wraps(func)
        def wrapper_func(*args, **kwargs):
            if COMPUTING_DEFAULTS.compiled:
                return compiled_func(*args, **kwargs)
            return func(*args, **kwargs)

        return wrapper_func

    return _decorate(function, compiled_function)


try:
    @default_compile_operation
    def _dummy_func(x):
        with torch.no_grad():
            return x + 1

    # Test the dummy function
    with torch.no_grad():
        _dummy_func(torch.tensor(1.0))

    # compiled = torch.compile
    # print('using torch.compile')
    compiled = lambda x: x
    cuda_compiled = lambda x: x#compile_wrapper
except Exception as e:
    #raise e
    #print('PyTorch Compilation is unavailable, most likely due to running on Windows OS.')
    compiled = lambda x: x
    cuda_compiled = lambda x: x
#not_compiled = torch.compiler.disable(recursive=True)
not_compiled = lambda x: x
def csync(f):
    def _sync(*args, **kwargs):
        torch.cuda.synchronize()
        return f(*args, **kwargs)

    return _sync

from algan.settings.defaults import *
from algan.settings.style_defaults import *
from algan.settings.logging_defaults import *

from algan.utils.memory_utils import ManualMemory


class SceneManager:
    _instance = None
    _memory = None
    _scene_class = None
    _scene_initializer = None

    def __init__(self):
        raise RuntimeError("Call SceneManager.instance() instead of SceneManager().")

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
                cls._memory = None  # ManualMemory(algan.defaults.batch_defaults.DEFAULT_PORTION_MEMORY_USED_FOR_RENDERING)
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

from algan.animation.manim_animations import *


def clear_cache():
    f = DIRECTORY_DEFAULTS.cache_directory
    if os.path.exists(f):
        shutil.rmtree(f)


def default_scene_initializer(scene):
    scene.camera = Camera(location=CAMERA_ORIGIN).spawn(animate=False)
    scene.light_sources = [
        PointLight(
            location=scene.camera.location + UP * 1 + RIGHT * 5 + OUT * 1, color=WHITE
        ).spawn(animate=False)
    ]


SceneManager.set_scene_class(Scene, default_scene_initializer)
SceneManager.instance()


def make_manim_dir():
    from manim import config

    for tex_dir in [config.get_dir("tex_dir"), config.get_dir("text_dir")]:
        if not tex_dir.exists():
            tex_dir.mkdir(parents=True)


make_manim_dir()
