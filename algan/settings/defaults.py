import os

import torch
import sys
from dataclasses import dataclass

from algan.settings.render_settings import LD
from algan.rendering.shaders.pbr_shaders import default_shader
from algan.constants.math import GIGABYTES

__all__ = ["COMPUTING_DEFAULTS", "DIRECTORY_DEFAULTS", "RENDERING_DEFAULTS"]

cuda_available = False
if torch.cuda.is_available():
    try:
        torch.zeros((2,)).cuda() + 1
        cuda_available = True
    except:
        pass

accelerator = torch.device(
        "cuda"
        if cuda_available
        else ("mps" if torch.mps.is_available() else "cpu")
    )

@dataclass
class ComputingDefaults:
    compiled: bool = False
    portion_of_memory_used_for_animating: float = 0.15
    portion_of_memory_used_for_rendering: float = 0.4
    max_animate_batch_size = 10000
    max_cpu_memory_used = 2 * GIGABYTES
    animation_device = torch.device("cpu")
    render_device = accelerator
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
print(f"Rendering device set to {COMPUTING_DEFAULTS.render_device}")


@dataclass
class DirectoryDefaults:
    base_directory = sys.path[0]
    cache_directory = os.path.join(base_directory, "algan_cache")
    output_filename = "algan_render_output"
    output_directory = "algan_outputs"
    output_path = None


DIRECTORY_DEFAULTS = DirectoryDefaults()


@dataclass
class RenderingDefaults:
    settings = LD
    shader = None


RENDERING_DEFAULTS = RenderingDefaults()
RENDERING_DEFAULTS.shader = default_shader
