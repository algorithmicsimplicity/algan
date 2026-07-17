import os

import torch
import sys

from algan.logging.logger import get_logger
from algan.settings.render_settings import LD
from algan.rendering.shaders.pbr_shaders import default_shader
from algan.constants.math import GIGABYTES

__all__ = ["COMPUTING_DEFAULTS", "DIRECTORY_DEFAULTS", "RENDERING_DEFAULTS"]

cuda_available = False
if torch.cuda.is_available():
    try:
        # torch.cuda.is_available() can lie (driver/runtime mismatch), so
        # probe with a real allocation before committing to cuda.
        torch.zeros((1,)).cuda() + 1
        cuda_available = True
    except Exception:
        pass

accelerator = torch.device(
        "cuda"
        if cuda_available
        else ("mps" if torch.mps.is_available() else "cpu")
    )


# The settings singletons below are plain classes (not dataclasses): every
# field is a class attribute, mutated in place via the singleton instance
# (e.g. ``COMPUTING_DEFAULTS.render_on_cpu = True``).
class ComputingDefaults:
    compiled = False
    portion_of_memory_used_for_animating = 0.15
    portion_of_memory_used_for_rendering = 0.4
    max_animate_batch_size = 10000
    max_cpu_memory_used = 2 * GIGABYTES
    animation_device = torch.device("cpu")
    render_device = accelerator
    # Keep the legacy override in sync with the auto-detected render device.
    # Taichi must not be initialized with ti.gpu on CPU-only systems: that
    # makes Taichi probe CUDA and then Vulkan, whose failed initialization
    # can segfault on headless machines.
    render_on_cpu = accelerator.type == "cpu"
    use_torch_scatter = True
    allow_save_frame = True


COMPUTING_DEFAULTS = ComputingDefaults()
# torch.set_default_device installs a global TorchFunctionMode that intercepts
# every torch call in Python (millions of calls per render). The animation
# device is cpu -- torch's factory default -- so only pay that cost when a
# non-default device is actually requested.
if COMPUTING_DEFAULTS.animation_device.type != "cpu":
    torch.set_default_device(COMPUTING_DEFAULTS.animation_device)
torch.set_default_dtype(torch.float32)
get_logger().info(f"Rendering device set to {COMPUTING_DEFAULTS.render_device}")


class DirectoryDefaults:
    base_directory = sys.path[0]
    # Algan's per-user home directory: every persistent cache lives under
    # ``cache_directory`` (audio stamps, manim Tex/Text geometry + LaTeX
    # output, bezier tessellations, and the Taichi offline kernel cache),
    # keyed by content hashes so it is shared safely across projects.
    # Override with the ALGAN_HOME / ALGAN_CACHE_DIR env vars, or mutate the
    # attributes before first use. The paths are evaluated once, at import:
    # reassigning ``cache_directory`` at runtime does *not* move
    # ``taichi_cache_directory`` (reassign it too if you want it to follow).
    algan_directory = os.environ.get(
        "ALGAN_HOME", os.path.join(os.path.expanduser("~"), ".algan")
    )
    cache_directory = os.environ.get(
        "ALGAN_CACHE_DIR", os.path.join(algan_directory, "cache")
    )
    # Dedicated home for Taichi's offline kernel cache (compiled-kernel
    # artifacts). Kept as its own setting because clearing content caches
    # must not throw away compiled kernels (a ~minutes-long recompile).
    # NOTE: consumed by ``ti.init`` while ``import algan`` is still running,
    # so mutating it from user code is too late -- move it via the env vars
    # above (or TI_OFFLINE_CACHE_FILE_PATH, which takes precedence).
    taichi_cache_directory = os.path.join(cache_directory, "taichi")
    output_filename = "algan_render_output"
    output_directory = "algan_outputs"
    output_path = None


DIRECTORY_DEFAULTS = DirectoryDefaults()


class RenderingDefaults:
    settings = LD
    shader = None


RENDERING_DEFAULTS = RenderingDefaults()
RENDERING_DEFAULTS.shader = default_shader
