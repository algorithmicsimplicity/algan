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

# The project vendors the geometry subset of Manim Community that the
# compatibility layer converts (see algan/external_libraries/manim/VENDORING.md
# for what is in it and why it is not the manim distribution). Expose it under
# Manim's normal top-level package name before importing any Algan mob module;
# those modules intentionally use the public ``manim`` import path, and so do
# user scripts that build Mobjects to hand to ManimMob.
from algan.external_libraries import manim_alias as _manim_alias

_vendored_manim = _manim_alias.install()


# A warning filter and a pydub converter fix-up used to sit here: the `manim`
# distribution dragged pydub in, and pydub probes PATH for ffmpeg as it is
# imported, so every `import algan` warned "Couldn't find ffmpeg or avconv" on
# a machine carrying only the build imageio-ffmpeg bundles. Nothing in Algan's
# dependency set imports pydub any more, so both are gone.

# Taichi prints "[Taichi] version ..." to *stdout* the moment it is imported
# (taichi/_lib/utils.py), so a script whose stdout is data got the banner mixed
# into it and `algan --version` printed two versions. Its own opt-out is this
# variable, read at that import; ``setdefault`` so anyone who wants the banner
# can still ask for it. It is not an ALGAN_ variable, hence not declared in
# algan/environment.py -- it belongs to Taichi, like TI_OFFLINE_CACHE_FILE_PATH.
os.environ.setdefault("ENABLE_TAICHI_HEADER_PRINT", "False")
import shutil

import torch

# Algan never needs gradients: all animation math is pure tensor arithmetic.
# That is a property of Algan's own work, not of the process, so the grad mode
# is switched off around the render entry points (``RenderLoopMixin.get_frames``
# and friends) rather than here. A process-global ``set_grad_enabled(False)``
# plus a never-exited ``torch.inference_mode()`` used to be entered right here,
# which meant importing algan permanently disabled autograd for the importing
# process -- a notebook that imported it could never train afterwards.
from algan.settings import *
from algan.settings._startup import _ANIMATION_DEVICE, render_device

if _ANIMATION_DEVICE.type != "cpu":
    torch.set_default_device(_ANIMATION_DEVICE)
torch.set_default_dtype(torch.float32)
from algan.errors import *
from algan.logging.logger import get_logger, set_log_level, set_progress_style

# DEBUG, not INFO: every `import algan` printed this line to stderr, including
# a plain `algan --help`. It is a diagnostic, and `algan check` reports the same
# device on demand.
get_logger().debug(f"Rendering device set to {render_device()}")

from algan.constants.color import *
from algan.constants.math import *
from algan.constants.spatial import *
from algan.rendering.taichi_runtime import (
    install_render_arch_guard as _install_render_arch_guard,
)
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
# The MPS zero-copy conversion, which turns torch MPS tensors into ndarrays
# over their own MTLBuffer so Taichi binds them instead of copying them through
# the host (see rendering/mps_zero_copy.py). A no-op on every machine but a Mac
# running the patched Taichi build, and it goes on before the arch guard so the
# guard stays the outermost wrapper.
from algan.rendering.mps_zero_copy import (  # noqa: E402
    install_zero_copy_launch as _install_zero_copy_launch,
)

_install_zero_copy_launch()
# Last, so it is the outermost wrapper on Kernel.__call__: the fast launcher
# bypasses whatever it wrapped on a plan hit, and the arch guard must see those
# launches too. It is what brings Taichi up, since no kernel module does that
# at import any more -- the arch depends on SETTINGS.computing.render_device,
# which a script can still change at this point.
_install_render_arch_guard()

from algan.animatable_base.animatable import *
from algan.animatable_base.mob import *
from algan.animation_timeline.animation_contexts import *
from algan.mobs.bezier_circuit import *
from algan.mobs.group import *
from algan.mobs.image_mob import *
from algan.mobs.manim_adapters import *
from algan.mobs.manim_mob import *
from algan.mobs.numeric_display import DecimalNumber
from algan.mobs.shapes_2d import *
from algan.mobs.shapes_3d import *
from algan.mobs.surfaces.procedural_textures import *
from algan.mobs.surfaces.surface import *
from algan.mobs.text import *
from algan.mobs.three_d_models import Model3D, TriangleMesh
from algan.project import Project
from algan.rendering import camera
from algan.rendering.lights import *
from algan.scene import Scene
from algan.sound.audio_effect import AudioEffect, AudioManager
from algan.utils.algan_utils import *

# There is deliberately no module-level wrapper for a Scene method here.
# ``Scene.set_environment_map`` is the one spelling, and ``Scene.foo(...)``
# already resolves the active Scene when called on the class -- so the wrapper
# bought nothing and cost the namespace a second name for one thing. It used to
# sit between these two blocks, which is why they need the split marker to stay
# apart now that it is gone: the shader/material imports below must land after
# the Mob modules above, and sorting them into one block would hoist
# ``material_presets`` above them.
# isort: split
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
    null_shader,
)

SETTINGS.style.set(default_material=DiffuseMaterial())
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
from algan.rendering.raytracing.truncation import TruncationCounts
from algan.rendering.shaders.fragment_shaders import (
    STAGE_LAMBERT,
    STAGE_MANIM,
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

# The Manim compatibility layer is a separate surface, reached as
# ``algan.manim`` rather than star-imported here: every name in it means
# "Manim's version, by Manim's conventions", and mixing the two namespaces is
# what made ``Square`` (degrees) and ``Arc`` (radians) indistinguishable. The
# import is eager so ``import algan.manim as mn`` needs no second import, but
# nothing it defines reaches ``algan.__all__``.
#
# It must stay *below* the Mob imports and behind this assignment, which keeps
# isort from hoisting it into the block above: ``algan.manim`` imports ``Mob``,
# which imports ``algan.animated_function``, so pulling it up leaves this module
# half-initialised and the import fails.
_MANIM_NAMESPACE_ANCHOR = None
from algan import manim as _manim_namespace  # noqa: E402, F401


def clear_cache(include_kernels=False):
    """Delete Algan's content caches (tessellations, manim Tex/Text, audio).

    The Taichi offline kernel cache lives inside the cache directory too
    (the environment-selected Taichi cache directory) but is spared by default:
    it holds compiled kernels (minutes to rebuild), is version-keyed, and is
    never invalidated by scene-content changes.

    Parameters
    ----------
    include_kernels
        Whether to wipe the compiled Taichi kernels as well. Defaults to False.
        :func:`clear_cached_kernels` is the same thing said outright, and is
        what to reach for before A/B-benchmarking a kernel edit -- the offline
        cache does not invalidate on ``@ti.func`` changes.
    """
    f = SETTINGS.paths.cache_directory
    if not os.path.exists(f):
        return
    if include_kernels:
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


def clear_cached_kernels():
    """Delete the compiled Taichi kernels, and Algan's content caches with them.

    The offline kernel cache does not invalidate when an imported ``@ti.func``
    changes, so a kernel edit A/B-benchmarked against a warm cache measures the
    old kernel. Clearing costs a cold compile of minutes.

    This is :func:`clear_cache` with ``include_kernels=True``; the kernel cache
    lives inside the same directory, so there is no way to drop one without the
    other.
    """
    clear_cache(include_kernels=True)


from algan.scenes.default_scene import default_scene_initializer
"""def default_scene_initializer(scene):
    scene.camera = Camera(scene=scene, location=CAMERA_ORIGIN).spawn(animate=False)
    scene.light_sources = []
    PointLight(
        scene=scene,
        location=scene.camera.location + UP * 1 + RIGHT * 5 + OUTWARD * 1,
        color=WHITE,
    ).spawn(animate=False)"""


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
    "algan.rendering.mps_compat",
    "algan.rendering.mps_zero_copy",
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
        # Primitive builders. `api_settings.md`'s star-import rule keeps these
        # out of the namespace: they are what the shape classes are assembled
        # from (a bare triangle's vertex buffer, a triangulated quad), not
        # something an authoring script reaches for, and each carries Mob's
        # generic docstring rather than one of its own. Still importable from
        # `algan.mobs.shapes_2d` / `algan.mobs.triangulated_bezier_circuit`,
        # which is where the benchmarks and the renderer tests take them from.
        "TriangleVertices",
        "TriangleTriangulated",
        "QuadTriangulated",
        "TriangulatedBezierCircuit",
        # Degree/radian boundary factors. `DEGREES` and `RADIANS` are the two
        # multipliers a script writes; these two are library-internal, and four
        # names for two factors -- two of which read as synonyms and differ by
        # 57x -- is exactly the collision the curated namespace exists to stop.
        "DEGREES_TO_RADIANS",
        "RADIANS_TO_DEGREES",
        # The Manim-compatibility surface: Manim's field of view, Manim's
        # default 3-D shading and the material that carries it. They mean
        # "Manim's version of this", so they belong beside the rest of that
        # layer in `algan.manim` rather than in a namespace of Algan's own
        # names. `use_manim_defaults()` installs them for you.
        "manim_fov",
        "manim_shader",
        "ManimMaterial",
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
        # the render device's public face is SETTINGS.computing.render_device;
        # this accessor is how engine code reads it without binding it
        "render_device",
        # internal rate-func/animation steps
        "draw_step",
        "undraw_step",
        "passing_flash_step",
        "wiggle_step",
        "wiggle",
        "there_and_back",
        # Video encoding: codec probing, encoder selection and the moviepy
        # binary override. All internal to what save_video does; the user-facing
        # controls are its ``codec``/``ffmpeg_params`` arguments and
        # SETTINGS.paths.ffmpeg_binary.
        "check_codec_is_available",
        "resolve_encode_binary",
        "select_video_encoder",
        "override_moviepy_ffmpeg_binary",
        # Surface topology and tessellation plumbing.
        "wrap_pad_texture",
        "surface_closed_axes",
        "surface_weld_flags",
        "orient_faces_outward",
        # Timeline recording introspection, engine memory, version counters.
        "attr_ranges_for_mob",
        "release_torch_memory",
        "ANIMATABLE_PROPERTY_VERSION",
        # Colour coercion: deliberately passes tensors through untouched so a
        # per-row colour buffer is not collapsed to one colour, which makes it
        # the wrong shape for a user-facing constructor. Every public entry
        # point (Mob colour kwargs, materials) already calls it for you.
        "to_color",
        # Raw Taichi shading functions: microfacet BSDF maths taking kernel
        # memory arguments, unusable from an authoring script.
        "fragment_light",
        "fragment_light_vis",
        "prep_normal",
        "shading_normal",
        "smith_geometry",
        "ggx_distribution",
        # NOTE: the fragment-shader callables (``phong_shader``,
        # ``standard_shader``, ...) are deliberately NOT here. They look like
        # engine internals and one tutorial passage imports them by module
        # path, but the star import is their real contract: the executable
        # ``.. algan::`` examples in shaders_and_materials.rst open with
        # ``from algan import *`` and then pass ``standard_shader`` to
        # ``set_fragment_shader``, and two unit tests assert their presence in
        # ``__all__``. They are authoring vocabulary, not plumbing.
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
        "MANIM_PANGO_MOBJECT_NAMES",
        "MANIM_PRIVATE_MOBJECT_NAMES",
        "MANIM_EXTERNAL_TOOL_MOBJECT_NAMES",
        "MANIM_UNVENDORED_MOBJECT_NAMES",
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
        "DEFAULT_EASING",  # algan.animation_timeline.animation_contexts
    }
)

# Public names that the rules above would otherwise miss.
# FragmentStage instances are neither callable nor upper-case, so the rules
# below do not pick them up.
_EXTRA_EXPORTS = ("cosine_color", "fresnel_rim", "glass_ball", "easings")


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
