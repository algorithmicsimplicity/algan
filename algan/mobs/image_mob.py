from algan.mobs.surfaces.surface import Surface
import algan.utils.file_utils as file_utils


class ImageMob(Surface):
    def __init__(self, rgba_array_or_file_path, ignore_normals=True, **kwargs):
        rgba_array = file_utils.get_image(rgba_array_or_file_path)

        h = rgba_array.shape[-3]
        w = rgba_array.shape[-2]

        super().__init__(grid_height=h, grid_width=w, color_texture=rgba_array.transpose(-3,-2).flip(-2),
                         ignore_normals=ignore_normals, **kwargs)
