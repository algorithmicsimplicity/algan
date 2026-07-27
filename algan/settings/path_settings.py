from dataclasses import dataclass
import os
import sys

from algan.settings.abstract_settings import Settings
from algan.settings._startup import _CACHE_DIRECTORY


@dataclass
class PathSettings(Settings):
    """Runtime-adjustable content-cache and output paths.

    ``ALGAN_HOME`` and the Taichi offline-cache path are initialization-only
    environment configuration. ``cache_directory`` remains public because
    Algan's content caches are consulted lazily and can safely move at runtime.
    """

    base_directory: str = sys.path[0]
    cache_directory: str = str(_CACHE_DIRECTORY)
    output_filename: str = "algan_render_output"
    output_directory: str = "algan_outputs"
    output_path: str | None = None

    def __post_init__(self):
        for name in (
            "base_directory",
            "cache_directory",
            "output_filename",
            "output_directory",
        ):
            object.__setattr__(self, name, os.fspath(getattr(self, name)))
        if self.output_path is not None:
            object.__setattr__(self, "output_path", os.fspath(self.output_path))
