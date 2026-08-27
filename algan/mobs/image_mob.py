"""Images as flat textured surfaces.

:class:`ImageMob` puts a picture on screen: a file path or an RGBA array becomes
a quad carrying the image as a texture map, sampled per fragment by the renderer,
so it stays sharp independently of the mesh's resolution.

It is a :class:`~algan.mobs.surfaces.surface.Surface`, so it moves, rotates and
deforms like any other 3-D Mob -- but it is deliberately unlit: a picture shows
its own colours rather than a lighting gradient, whatever the Scene's lights are
doing.

Paths resolve against the working directory and then against the directory
holding the main script, so an image sitting beside your ``.py`` file loads
wherever you launch Python from.

See :doc:`/advanced_user_tutorials/images_and_textures`.
"""

from __future__ import annotations

import torch
import torch.types

import algan.utils.file_utils as file_utils
from algan.constants.color import Color
from algan.errors import AlganConfigurationError
from algan.mobs.surfaces.surface import Surface
from algan.rendering.shaders.pbr_shaders import null_shader
from algan.utils.lazy_import import LazyModule, isinstance_if_loaded

# Deferred: an ImageMob is usually built from a file path / array; only the
# ManimMob conversion path hands it a manim ImageMobject, and in that case
# manim is already imported. The isinstance checks below therefore must not
# force the ~2 s manim import (isinstance_if_loaded is False for free while
# manim was never loaded).
_manim = LazyModule("manim", extras=("algan.utils.manim_svg_cache",))


class ImageMob(Surface):
    """A flat 2-D rectangular
    :class:`~algan.mobs.surfaces.surface.Surface` with color set according
    to a given image (or image file path).

    The picture is shown unlit -- its own colours, unaffected by the Scene's
    lights.

    Parameters
    ----------
    rgba_array_or_file_path
        An array of RGBA data, or a string containing the path to an image file from which
        RGBA data will be read, used to color the surface.
    textured
        If True, shade the four-vertex surface by sampling the image as a texture.
        If False, create one mesh vertex per image pixel and assign the pixel
        values directly as vertex colors.
    **kwargs
        Passed to :class:`~algan.mobs.surfaces.surface.Surface`.

    """

    _morph_family = "image"

    def __init__(
        self,
        rgba_array_or_file_path: torch.Tensor | str,
        textured: bool = True,
        **kwargs,
    ):
        submob = rgba_array_or_file_path
        if isinstance_if_loaded(rgba_array_or_file_path, _manim, "ImageMobject"):
            rgba_array = Color.add_defaults(
                torch.from_numpy(submob.pixel_array).float() / 255
            )
        else:
            rgba_array = file_utils.get_image(rgba_array_or_file_path)

        if rgba_array.dim() < 3:
            # Reported here, where the argument was written, rather than as
            # ``IndexError: tuple index out of range`` from the shape lookup
            # two lines down.
            raise AlganConfigurationError(
                f"An image needs a height, a width and colour channels: "
                f"expected an array of shape [H, W, C] (C of 3, 4 or 5) or a "
                f"path to an image file, got an array of shape "
                f"{tuple(rgba_array.shape)}."
            )
        h = rgba_array.shape[-3]
        w = rgba_array.shape[-2]
        aspect_ratio = w / h
        surface_colors = rgba_array.transpose(-3, -2).flip(-2).contiguous()

        super().__init__(
            coord_function=lambda uv: torch.cat(
                (
                    (uv[..., :1] - 0.5) * aspect_ratio,
                    (uv[..., 1:] - 0.5),
                    torch.zeros_like(uv[..., :1]),
                ),
                -1,
            ),
            grid_height=h if not textured else 2,
            grid_width=w if not textured else 2,
            color_texture=surface_colors if textured else None,
            **kwargs,
        )
        if not textured:
            self.grid._setattr_without_record("color", surface_colors.flatten(-3, -2))
        # A picture plane is unlit: null_shader returns the albedo unchanged,
        # so the image shows its own colours instead of a lighting gradient.
        # Called after super().__init__ so it reaches the grid that actually
        # renders (set_shader walks the descendants), and before spawn, as
        # set_shader requires.
        self.set_shader(null_shader)
        if isinstance_if_loaded(rgba_array_or_file_path, _manim, "ImageMobject"):
            self.scale(torch.tensor((submob.width / 2, submob.height / 2, 1)).float())
            self.move_to(submob.get_center())

    def setattr_absolute(self, attr_name, value):
        if attr_name == "color":
            self.glow = value[..., -2:-1]
            return self
        return super().setattr_absolute(attr_name, value)
