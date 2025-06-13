import torch.types
from algan.mobs.surfaces.surface import Surface
import algan.utils.file_utils as file_utils


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

    def __init__(self, rgba_array_or_file_path:torch.Tensor|str, ignore_normals=True, **kwargs):
        rgba_array = file_utils.get_image(rgba_array_or_file_path)

        h = rgba_array.shape[-3]
        w = rgba_array.shape[-2]

        super().__init__(grid_height=h, grid_width=w, color_texture=rgba_array.transpose(-3,-2).flip(-2),
                         ignore_normals=ignore_normals, **kwargs)
