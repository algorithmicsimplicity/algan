from __future__ import annotations

import torchvision

from algan.constants.color import Color
from algan.settings.defaults import COMPUTING_DEFAULTS


def get_image(file_path):
    if isinstance(file_path, str):
        file_path = (
            torchvision.io.read_image(file_path)
            .to(COMPUTING_DEFAULTS.animation_device)
            .permute(1, 2, 0)
        )
        file_path = file_path.float() / 255
    return Color.add_defaults(file_path)
