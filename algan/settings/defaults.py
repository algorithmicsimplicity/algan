import torch
import sys
from dataclasses import dataclass

from algan.settings.render_settings import LD
from algan.rendering.shaders.pbr_shaders import default_shader
from algan.constants.math import MEGABYTES

__all__ = ['COMPUTING_DEFAULTS', 'DIRECTORY_DEFAULTS', 'RENDERING_DEFAULTS']

@dataclass
class ComputingDefaults:
    compiled: bool = False
    portion_of_memory_used_for_animating: float = 0.2
    portion_of_memory_used_for_rendering: float = 0.6
    max_animate_batch_size = 1000
    max_cpu_memory_used = 1024 * MEGABYTES
    animation_device = torch.device('cpu')
    render_device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.mps.is_available() else 'cpu'))

COMPUTING_DEFAULTS = ComputingDefaults()
torch.set_default_device(COMPUTING_DEFAULTS.animation_device)
torch.set_default_dtype(torch.float32)
print(f'Rendering device set to {COMPUTING_DEFAULTS.render_device}')

@dataclass
class DirectoryDefaults:
    base_directory = sys.path[0]
    output_filename = 'algan_render_output'
    output_directory = 'algan_outputs'
    output_path = None

DIRECTORY_DEFAULTS = DirectoryDefaults()

@dataclass
class RenderingDefaults:
    settings = LD
    shader = None

RENDERING_DEFAULTS = RenderingDefaults()
RENDERING_DEFAULTS.shader = default_shader
