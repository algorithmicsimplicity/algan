from dataclasses import dataclass
from algan.settings.abstract_settings import Settings


@dataclass
class KernelSettings(Settings):
    render_kernel = None

KERNEL_SETTINGS = KernelSettings()