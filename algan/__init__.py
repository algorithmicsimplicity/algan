from __future__ import annotations


def __getattr__(name):
    # PEP 562: resolve __version__ lazily -- the importlib.metadata lookup
    # costs ~0.1 s and almost no session reads it. (Underscored names are
    # excluded from `from algan import *`, so star-imports don't force it.)
    if name == "__version__":
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("algan")
        except PackageNotFoundError:
            # Source archives may be run without installed package metadata.
            return "0+unknown"
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


import os
import sys

from algan.environment import (
    warn_for_unknown_algan_environment_variables as _warn_for_unknown_algan_environment_variables,
)

_warn_for_unknown_algan_environment_variables()

# If a general daemon is running (`python -m algan.daemon`), hand this script
# to it and exit with its result -- the point of the daemon is that a warm
# process has already paid the ~7 s import and ~65 s of Taichi kernel
# preparation below. With no daemon this costs one `isfile` and returns, so
# scripts run exactly as before; see algan/daemon_client.py for the full set
# of conditions and fallbacks. It must stay *here*, above the torch and
# taichi imports, or a client would pay for the thing it is avoiding.
from algan.daemon_client import maybe_handoff as _maybe_handoff

_maybe_handoff()

# The project vendors the subset of Manim Community used for SVG/Tex and
# compatibility Mobs.  Expose it under Manim's normal top-level package name
# before importing any Algan mob modules; those modules intentionally use the
# public ``manim`` import path.
from algan.external_libraries import manim as _vendored_manim

sys.modules.setdefault("manim", _vendored_manim)

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

from algan.settings import *
from algan.settings._startup import _ANIMATION_DEVICE, _RENDER_DEVICE

if _ANIMATION_DEVICE.type != "cpu":
    torch.set_default_device(_ANIMATION_DEVICE)
torch.set_default_dtype(torch.float32)
from algan.errors import *
from algan.logging.logger import get_logger, set_log_level, set_progress_style

get_logger().info(f"Rendering device set to {_RENDER_DEVICE}")

from algan.constants.color import *
from algan.constants.math import *
from algan.constants.spatial import *
from algan.scene_manager import SceneManager
from algan.settings.video_settings import *
from algan.utils.memory_utils import ManualMemory
from algan.utils.taichi_fast_launch import apply as _apply_taichi_fast_launch

# Taichi is imported (via the rendering modules above) but no kernel has
# materialized yet -- install the warm-start memoization now so every kernel
# compiled in this process benefits (see utils/taichi_warmstart.py), plus
# the cached fast launcher that skips Taichi's per-launch Python argument
# re-validation on repeat launches (see utils/taichi_fast_launch.py).
from algan.utils.taichi_warmstart import apply as _apply_taichi_warmstart

_apply_taichi_warmstart()
_apply_taichi_fast_launch()

from algan.animatable_base.animatable import *
from algan.animatable_base.mob import *
from algan.mobs.bezier_circuit import *
from algan.mobs.group import *
from algan.mobs.image_compat import *
from algan.mobs.image_mob import *
from algan.mobs.manim_compat import *
from algan.mobs.manim_mob import *
from algan.mobs.manim_parity import *
from algan.mobs.numeric_display import NumericDisplay
from algan.mobs.opengl_compat import *
from algan.mobs.point_cloud import *
from algan.mobs.shapes_2d import *
from algan.mobs.shapes_3d import *
from algan.mobs.surfaces.surface import *
from algan.mobs.text import *
from algan.mobs.three_d_models import ThreeDModelMob, TriangleMesh
from algan.project import Project
from algan.rendering import camera

# Manim names its root class Mobject; Algan's native equivalent is Mob.  Its
# abstract graph and OpenGL renderer-specific bases likewise map to Algan's
# renderer-independent classes.
Mobject = Mob
GenericGraph = Graph
install_opengl_aliases(globals())
from algan.animation_timeline.animation_contexts import *
from algan.rendering.lights import *
from algan.scene import Scene
from algan.sound.audio_effect import AudioEffect, AudioManager
from algan.utils.algan_utils import *


def set_environment_map(*args, **kwargs):
    """Set the environment map on the current active scene."""
    return SceneManager.instance().current_scene.set_environment_map(*args, **kwargs)


from algan.constants.material_presets import *
from algan.rendering.shaders.material_shaders import (
    basic_material_shader,
    depth_shader,
    lambert_shader,
    matcap_shader,
    normal_shader,
    phong_shader,
    physical_shader,
    standard_shader,
    toon_shader,
)
from algan.rendering.shaders.materials import *
from algan.rendering.shaders.pbr_shaders import (
    basic_pbr_shader,
    default_shader,
    null_shader,
)

SETTINGS.style.set(default_shader=default_shader)
from algan.rendering.raytracing.shading_taichi import (
    _ggx_distribution as ggx_distribution,
)
from algan.rendering.raytracing.shading_taichi import (
    _light as fragment_light,
)
from algan.rendering.raytracing.shading_taichi import (
    _light_vis as fragment_light_vis,
)
from algan.rendering.raytracing.shading_taichi import (
    _prep_normal as prep_normal,
)
from algan.rendering.raytracing.shading_taichi import (
    _shading_normal as shading_normal,
)
from algan.rendering.raytracing.shading_taichi import (
    _smith_geometry as smith_geometry,
)
from algan.rendering.raytracing.tracer import RenderPlan, render_batch_raytraced
from algan.rendering.shaders.fragment_shaders import (
    STAGE_DEFAULT,
    STAGE_LAMBERT,
    STAGE_PHONG,
    STAGE_PHYSICAL,
    STAGE_STANDARD,
    STAGE_UNLIT,
    FragmentStage,
    cosine_color,
)
from algan.rendering.shaders.fragment_stage_library import fresnel_rim, glass_ball
from algan.settings.kernel_settings import KERNEL_REGISTRY

KERNEL_REGISTRY.render_kernel = render_batch_raytraced

from algan.animation_timeline.timeline import TimelineManager
from algan.animations.changing import *
from algan.animations.indication import *
from algan.animations.manim_animations import *
from algan.animations.movement import *


def clear_cache(taichi_kernels=False):
    """Delete Algan's content caches (tessellations, manim Tex/Text, audio).

    The Taichi offline kernel cache lives inside the cache directory too
    (the environment-selected Taichi cache directory) but is spared by default:
    it holds compiled kernels (minutes to rebuild), is version-keyed, and is
    never invalidated by scene-content changes. Pass
    ``taichi_kernels=True`` to wipe it as well (e.g. before
    A/B-benchmarking kernel edits -- the offline cache does not invalidate on
    ``@ti.func`` changes).
    """
    f = SETTINGS.paths.cache_directory
    if not os.path.exists(f):
        return
    if taichi_kernels:
        shutil.rmtree(f)
        return
    from algan.settings._startup import _TAICHI_CACHE_DIRECTORY

    keep = os.path.normcase(os.path.abspath(_TAICHI_CACHE_DIRECTORY))
    for entry in os.listdir(f):
        p = os.path.join(f, entry)
        if os.path.normcase(os.path.abspath(p)) == keep:
            continue
        if os.path.isdir(p):
            shutil.rmtree(p)
        else:
            os.remove(p)


def default_scene_initializer(scene):
    scene.camera = Camera(scene=scene, location=CAMERA_ORIGIN).spawn(animate=False)
    scene.light_sources = []
    PointLight(
        scene=scene,
        location=scene.camera.location + UP * 1 + RIGHT * 5 + OUT * 1,
        color=WHITE,
    ).spawn(animate=False)


# The SceneManager singleton is created lazily. Its default Scene is created
# only when current_scene is first requested (e.g. by Mob construction or a
# module-level render call).
SceneManager.set_scene_class(Scene, default_scene_initializer)

# Re-exported for backwards compatibility; it now runs lazily on first Tex use.
# Curate star imports. ``from algan import *`` is the documented entry point,
# so it is effectively the public API: it must expose mobs, animations,
# contexts, materials, settings and authoring constants, and nothing else.
# Internal helpers stay reachable at their real import path; they simply do not
# land in the user's namespace, where names like ``mean``, ``interpolate`` and
# ``offset`` would shadow whatever the user imported before Algan.
from types import ModuleType as _ModuleType

from algan.mobs.text import make_manim_dir

# Whole modules whose public names are implementation detail.
_INTERNAL_EXPORT_MODULES = (
    "algan.animatable_base.mob_hierarchy",
    "algan.animatable_base.mob_layout",
    "algan.animatable_base.mob_materials",
    "algan.animatable_base.mob_morph",
    "algan.animatable_base.mob_movement",
    "algan.animatable_base.mob_orientation",
    "algan.animation_timeline.timeline",
    "algan.mobs.nonplanar_circuit",
    "algan.rendering.logical_pn",
    "algan.utils.file_utils",
    "algan.utils.lazy_import",
    "algan.utils.python_utils",
    "algan.utils.tensor_utils",
)

# Individually internal names from modules that are otherwise public.
_INTERNAL_EXPORT_NAMES = frozenset(
    {
        # authoring/session plumbing
        "default_scene_initializer",
        "get_logger",
        "install_opengl_aliases",
        "active_scene_for_new_mob",
        "animation_manager_bound",
        "animation_manager_context",
        "animation_manager_for",
        "prepare_kwargs",
        # render-primitive construction
        "build_render_primitives_batched",
        "get_render_primitives_batched",
        "compute_grid_vertex_normals",
        "get_grid_to_triangle_indices",
        "grid_to_triangle_vertices",
        "effective_triangle_primitive",
        "render_batch_raytraced",
        "point_to_tensor2",
        "color_to_texture_map",
        "midpoint",
        "mid_point",
        # service registries: not user settings
        "KERNEL_REGISTRY",
        "RENDERER_REGISTRY",
        # internal rate-func/animation steps
        "draw_step",
        "undraw_step",
        "passing_flash_step",
        "wiggle_step",
        "wiggle",
        "there_and_back",
        # tooling and dev utilities
        "make_manim_dir",
        "missing_manim_mobjects",
        "validate_manim_mobject_parity",
        "combine_scenes",
        "concatenate_videos",
        "get_file_writer",
        "profile_func",
        "BatchedMobViewSequence",
        # internal constants
        "HANDLED_FUNCTIONS",
        "TIME_PARAMETER_NAME",
        "SPAWN_VERSION",
        "STRUCTURE_VERSION",
        "HIERARCHY_VERSION",
        "MANIM_COMMUNITY_VERSION",
        "MANIM_MOBJECT_NAMES",
        "MANIM_OPENGL_MOBJECT_NAMES",
        "MANIM_PRIVATE_MOBJECT_NAMES",
        "MANIM_EXTERNAL_TOOL_MOBJECT_NAMES",
        # Helpers that are genuinely useful when writing custom animations, but
        # too specialised to spend a name in every user's namespace. They stay
        # importable from the module that defines them -- see
        # docs/source/advanced_user_tutorials/extending_algan.rst.
        "project_onto_basis",  # algan.geometry.geometry
        "map_global_to_local_coords",  # algan.geometry.geometry
        "map_local_to_global_coords",  # algan.geometry.geometry
        "rotate_vector_around_axis",  # algan.geometry.geometry
        "get_rotation_around_axis",  # algan.geometry.geometry
        "get_rotation_between_bases",  # algan.geometry.geometry
        "get_orthonormal_vector",  # algan.geometry.geometry
        "batch_mobs",  # algan.utils.mob_utils
        "pack_animatable_rows",  # algan.utils.mob_utils
        "pack_member_rows",  # algan.utils.mob_utils
        "animate_lagged_by_location",  # algan.utils.animation_utils
        "render_all_funcs",  # algan.utils.algan_utils (or Scene.render_all_funcs)
        "null_shader",  # algan.rendering.shaders.pbr_shaders
        "DEFAULT_RATE_FUNC",  # algan.animation_timeline.animation_contexts
    }
)

# Public names that the rules above would otherwise miss.
# FragmentStage instances are neither callable nor upper-case, so the rules
# below do not pick them up.
_EXTRA_EXPORTS = ("cosine_color", "fresnel_rim", "glass_ball", "rate_funcs")


def _is_root_export(name, value):
    if name.startswith("_") or name in _INTERNAL_EXPORT_NAMES:
        return False
    if isinstance(value, _ModuleType):
        return False
    origin = getattr(value, "__module__", "") or ""
    if origin in _INTERNAL_EXPORT_MODULES:
        return False
    if callable(value):
        return origin == "algan" or origin.startswith("algan.")
    return name.isupper()


__all__ = (
    *sorted(name for name, value in globals().items() if _is_root_export(name, value)),
    *_EXTRA_EXPORTS,
)
