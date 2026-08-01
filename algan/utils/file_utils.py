from __future__ import annotations

import os

from algan.constants.color import Color
from algan.settings._startup import _ANIMATION_DEVICE


def resolve_asset_path(file_path):
    """Return a usable path for an image the user referred to by name.

    A relative path is tried against the working directory first, then against
    the directory of the script being run. Without that second lookup,
    ``ImageMob('world_map.jpg')`` only finds an image sitting next to your
    script when you happen to launch Python from that same directory.

    An unresolvable path is returned unchanged so the loader still reports the
    name the user actually wrote.
    """
    path = os.fspath(file_path)
    if os.path.isabs(path) or os.path.exists(path):
        return path

    # Imported here to keep this module out of the settings import ordering.
    from algan.settings.path_settings import _main_script_path

    script = _main_script_path()
    if script is not None:
        candidate = os.path.join(os.path.dirname(script), path)
        if os.path.exists(candidate):
            return candidate
    return path


def get_image(file_path):
    if isinstance(file_path, str):
        import torchvision  # deferred: ~0.2 s of import algan

        file_path = (
            torchvision.io.read_image(resolve_asset_path(file_path))
            .to(_ANIMATION_DEVICE)
            .permute(1, 2, 0)
        )
        file_path = file_path.float() / 255
    return Color.add_defaults(file_path)
