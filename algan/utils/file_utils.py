"""Locating and loading the image assets a scene refers to by name.

Every user-facing path argument -- :class:`~algan.mobs.image_mob.ImageMob`,
:meth:`~algan.scene.Scene.set_background`,
:meth:`~algan.scene.Scene.set_environment_map` and the 3-D model loaders --
routes through :func:`resolve_asset_path`, so a file sitting beside the running
script loads regardless of the working directory.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from algan.constants.color import Color
from algan.errors import AlganConfigurationError
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


# Pillow's unsigned 16-bit single-channel modes: what a 16-bit greyscale PNG or
# TIFF opens as.
_SIXTEEN_BIT_GREYSCALE_MODES = frozenset({"I;16", "I;16L", "I;16B", "I;16N"})


def _decode_pil_image(pil_image) -> np.ndarray:
    """Decode a Pillow image to float32 ``[H, W, 3]`` RGB or ``[H, W, 4]`` RGBA.

    Every loader that reads an image file goes through here, so a file's mode
    cannot change how its pixels are interpreted. 8-bit modes are scaled from
    0-255 to ``[0, 1]``: greyscale is replicated across the color channels, a
    palette is looked up, CMYK is converted, and an alpha band or a
    ``transparency`` entry (the ``tRNS`` of a palette, greyscale or RGB PNG)
    becomes the fourth channel.

    Unsigned 16-bit greyscale keeps its precision, scaled from 0-65535.
    Converting it through Pillow's 8-bit modes would clip every sample above
    255 to white. Pillow before 10.3 opened a 16-bit greyscale PNG as ``"I"``
    rather than ``"I;16"``; a PNG cannot hold any other 32-bit integer image, so
    that mode is read the same way when it came from a PNG. A 32-bit float image
    is taken as already normalized, as a float tensor is.
    """
    mode = pil_image.mode
    transparency = pil_image.info.get("transparency")
    if mode in _SIXTEEN_BIT_GREYSCALE_MODES or (
        mode == "I" and pil_image.format == "PNG"
    ):
        samples = np.asarray(pil_image)
        channels = [samples.astype(np.float32) / 65535] * 3
        if transparency is not None:
            # A greyscale tRNS names the one sample value that is transparent.
            channels.append((samples != transparency).astype(np.float32))
        return np.stack(channels, -1)
    if mode == "F":
        return np.repeat(np.asarray(pil_image, dtype=np.float32)[..., None], 3, -1)
    bands = pil_image.getbands()
    has_alpha = "A" in bands or "a" in bands or transparency is not None
    rgb_or_rgba = pil_image.convert("RGBA" if has_alpha else "RGB")
    array = np.asarray(rgb_or_rgba, dtype=np.float32)
    array /= 255
    return array


def get_image(file_path):
    if isinstance(file_path, (str, os.PathLike)):
        # Imported here, not at module scope, to keep PIL off algan's import-time
        # critical path.
        from PIL import Image

        with Image.open(resolve_asset_path(file_path)) as pil_image:
            array = _decode_pil_image(pil_image)
        file_path = torch.from_numpy(array).to(_ANIMATION_DEVICE)
    elif not torch.is_tensor(file_path):
        # A numpy array is the natural way to hand over pixels -- it is what
        # every imaging library returns and what Manim's ImageMobject takes --
        # and it used to reach ``torch.zeros_like`` as-is and fail there.
        try:
            file_path = torch.as_tensor(np.asarray(file_path))
        except (TypeError, ValueError) as not_an_array:
            raise AlganConfigurationError(
                f"Expected image pixels (a tensor or array of shape [H, W, C]) "
                f"or a path to an image file, got {type(file_path).__name__}."
            ) from not_an_array
        if file_path.dtype == torch.uint8:
            file_path = file_path.float() / 255
        else:
            file_path = file_path.float()
    return Color.add_defaults(file_path)
