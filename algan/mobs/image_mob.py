import torch
import torch.types
from algan.mobs.surfaces.surface import Surface
from algan.constants.color import Color
import algan.utils.file_utils as file_utils
from algan.utils.lazy_import import LazyModule, isinstance_if_loaded

# Deferred: an ImageMob is usually built from a file path / array; only the
# ManimMob conversion path hands it a manim ImageMobject, and in that case
# manim is already imported. The isinstance checks below therefore must not
# force the ~2 s manim import (isinstance_if_loaded is False for free while
# manim was never loaded).
_manim = LazyModule("manim", extras=("algan.utils.manim_svg_cache",))


class ImageMob(Surface):
    """A flat 2-D rectangular :class:`~.Surface` with color set according
    to a given image (or image file path).

    Parameters
    ----------
    rgba_arra_or_file_path
        An array of RGBA data, or a string containing the path to an image file from which
        RGBA data will be read, used to color the surface.
    ignore_normals
        If True the surface will have no normals (i.e. will not interact with lighting).
    **kwargs
        Passed to :class:`~Surface` .

    """

    def __init__(self, rgba_array_or_file_path: torch.Tensor | str, **kwargs):
        submob = rgba_array_or_file_path
        if isinstance_if_loaded(rgba_array_or_file_path, _manim, "ImageMobject"):
            rgba_array = Color.add_defaults(torch.from_numpy(submob.pixel_array).float() / 255)
        else:
            rgba_array = file_utils.get_image(rgba_array_or_file_path)

        h = rgba_array.shape[-3]
        w = rgba_array.shape[-2]
        aspect_ratio = w / h

        super().__init__(
            coord_function=lambda uv: torch.cat(
                (
                    (uv[..., :1] - 0.5) * aspect_ratio,
                    (uv[..., 1:] - 0.5),
                    torch.zeros_like(uv[..., :1]),
                ),
                -1,
            ),
            grid_height=2,
            grid_width=2,
            color_texture=rgba_array.transpose(-3, -2).flip(-2),
            **kwargs,
        )
        if isinstance_if_loaded(rgba_array_or_file_path, _manim, "ImageMobject"):
            self.scale(torch.tensor((submob.width / 2, submob.height / 2, 1)).float())
            self.move_to(submob.get_center())

    def setattr_absolute(self, attr_name, value):
        if attr_name == "color":
            self.glow = value[..., -2:-1]
            return self
        return super().setattr_absolute(attr_name, value)
