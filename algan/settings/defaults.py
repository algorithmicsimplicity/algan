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
        torch.zeros((2,)).cuda() + 1
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
    render_on_cpu = False
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
    cache_directory = os.path.join(base_directory, "algan_cache")
    output_filename = "algan_render_output"
    output_directory = "algan_outputs"
    output_path = None


DIRECTORY_DEFAULTS = DirectoryDefaults()


class RenderingDefaults:
    settings = LD
    shader = None


RENDERING_DEFAULTS = RenderingDefaults()
RENDERING_DEFAULTS.shader = default_shader
