from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("algan")
except PackageNotFoundError:
    # Source archives may be run directly without installed package metadata.
    __version__ = "0+unknown"

import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import shutil
import torch

# Algan never needs gradients: all animation math is pure tensor arithmetic.
# Inference mode is entered process-wide (and never exited) because it must
# cover every tensor the library ever creates, including module-level
# constants. NOTE for library consumers: this means importing algan disables
# autograd in the importing process; do not import algan into a process that
# also trains torch models.
torch.set_grad_enabled(False)
c = torch.inference_mode()
c.__enter__()

from algan.settings.defaults import *
from algan.settings.style_defaults import *
from algan.logging.logger import get_logger, set_log_level

from algan.utils.memory_utils import ManualMemory
from algan.scene_manager import SceneManager

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
from algan.mobs.three_d_models import ThreeDModelMob, TriangleMesh
from algan.mobs.numeric_display import NumericDisplay
from algan.scene import Scene

from algan.animation.animation_contexts import *
from algan.utils.algan_utils import *
from algan.rendering.lights import *

set_environment_map = Scene.set_environment_map
from algan.rendering.shaders.materials import *
from algan.rendering.shaders.material_shaders import (
    basic_material_shader,
    lambert_shader,
    phong_shader,
    standard_shader,
    physical_shader,
    toon_shader,
    normal_shader,
    matcap_shader,
    depth_shader,
)
from algan.rendering.shaders.pbr_shaders import (
    default_shader,
    basic_pbr_shader,
    null_shader,
)
from algan.rendering.shaders.fragment_shaders import (
    FragmentStage,
    cosine_color,
    STAGE_DEFAULT,
    STAGE_UNLIT,
    STAGE_LAMBERT,
    STAGE_PHONG,
    STAGE_STANDARD,
    STAGE_PHYSICAL,
)
from algan.rendering.raytracing.shading_taichi import (
    _ggx_distribution as ggx_distribution,
    _smith_geometry as smith_geometry,
    _shading_normal as shading_normal,
    _prep_normal as prep_normal,
    _light as fragment_light,
    _light_vis as fragment_light_vis,
)

from algan.rendering.raytracing.tracer import render_batch_raytraced

from algan.settings.kernel_settings import KERNEL_SETTINGS
KERNEL_SETTINGS.render_kernel = render_batch_raytraced

from algan.animation.manim_animations import *
from algan.animation.indication import *
from algan.animation.timeline import TimelineManager


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


# The scene itself is created lazily, on the first SceneManager.instance()
# call (e.g. the first Mob construction or render_to_file()).
SceneManager.set_scene_class(Scene, default_scene_initializer)

# Re-exported for backwards compatibility; it now runs lazily on first Tex use.
from algan.mobs.text import make_manim_dir
