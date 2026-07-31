"""Manim-compatible raster image Mobs backed by :class:`~algan.mobs.image_mob.ImageMob`."""
from __future__ import annotations

import pathlib

import numpy as np
import torch
from PIL.Image import Resampling

from algan.animation_timeline.animation_contexts import Off
from algan.constants.color import Color
from algan.mobs.image_mob import ImageMob
from algan.mobs.surfaces.surface import Surface
from algan.utils.file_utils import get_image


def _load_rgba5(filename_or_array) -> torch.Tensor:
    value = str(filename_or_array) if isinstance(filename_or_array, pathlib.PurePath) else filename_or_array
    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value)
        if value.dtype == torch.uint8:
            value = value.float() / 255
    result = get_image(value).to(dtype=torch.get_default_dtype())
    return result.clamp(0, 1)


def _rgba5_to_uint8(value: torch.Tensor) -> np.ndarray:
    rgba = torch.cat((value[..., :3], value[..., -1:]), -1)
    return (rgba.detach().cpu().clamp(0, 1).numpy() * 255).round().astype(np.uint8)


def _color_to_rgb8(color) -> np.ndarray:
    if hasattr(color, "to_rgb"):
        rgb = color.to_rgb()
    elif isinstance(color, Color):
        rgb = color.rgb.detach().cpu().numpy()
    elif isinstance(color, str):
        rgb = Color(color).rgb.detach().cpu().numpy()
    else:
        rgb = np.asarray(color, dtype=float).reshape(-1)[:3]
    return np.clip(np.asarray(rgb) * 255, 0, 255).round().astype(np.uint8)


class AbstractImageMobject(ImageMob):
    """Renderer-independent equivalent of Manim's abstract image base."""

    def __init__(
        self,
        scale_to_resolution,
        pixel_array_dtype="uint8",
        resampling_algorithm=Resampling.BICUBIC,
        **kwargs,
    ):
        _rgba5 = kwargs.pop("_algan_rgba5", None)
        self.pixel_array_dtype = pixel_array_dtype
        self.scale_to_resolution = scale_to_resolution
        self.set_resampling_algorithm(int(resampling_algorithm))
        if _rgba5 is None:
            # The Manim base is intentionally abstract.  Give it valid flat
            # geometry without registering a dummy texture whose dimensions
            # could constrain later ImageMobjects on Algan's shared timeline.
            Surface.__init__(
                self,
                coord_function=lambda uv: torch.cat(
                    (uv - 0.5, torch.zeros_like(uv[..., :1])), dim=-1
                ),
                grid_height=2,
                grid_width=2,
                ignore_normals=True,
                **kwargs,
            )
        else:
            ImageMob.__init__(self, _rgba5, **kwargs)

    def set_color(self, color, alpha=None, family=True):
        raise NotImplementedError

    def set_resampling_algorithm(self, resampling_algorithm):
        if not isinstance(resampling_algorithm, int):
            raise ValueError(
                "resampling_algorithm has to be an int Pillow resampling constant"
            )
        self.resampling_algorithm = resampling_algorithm
        return self

    def reset_points(self):
        # ImageMob always owns the four sampled corners of its native surface.
        return self

    def get_pixel_array(self):
        raise NotImplementedError


class ImageMobject(AbstractImageMobject):
    def __init__(
        self,
        filename_or_array,
        scale_to_resolution=1080,
        invert=False,
        image_mode="RGBA",
        **kwargs,
    ):
        self.scale_to_resolution = scale_to_resolution
        self.image_mode = image_mode
        self.invert_image = invert
        self.pixel_array_dtype = kwargs.pop("pixel_array_dtype", "uint8")
        resampling = kwargs.pop("resampling_algorithm", Resampling.BICUBIC)
        self.path = (
            pathlib.Path(filename_or_array)
            if isinstance(filename_or_array, (str, pathlib.PurePath))
            else None
        )

        rgba5 = _load_rgba5(filename_or_array)
        if invert:
            rgba5 = rgba5.clone()
            rgba5[..., :3] = 1 - rgba5[..., :3]
        self.pixel_array = _rgba5_to_uint8(rgba5)
        self.orig_alpha_pixel_array = self.pixel_array[..., 3].copy()

        # Translate common Manim style names instead of forwarding unknown
        # raster-specific options to Algan's Surface/Mob constructor.
        initial_fill_opacity = kwargs.pop("fill_opacity", None)
        for key in ("fill_color", "stroke_color", "stroke_width", "stroke_opacity"):
            kwargs.pop(key, None)

        super().__init__(
            scale_to_resolution,
            pixel_array_dtype=self.pixel_array_dtype,
            resampling_algorithm=resampling,
            _algan_rgba5=rgba5,
            **kwargs,
        )
        # Animatable installs generic ``set_color``/``set_opacity`` methods on
        # the concrete class while registering Mob attributes. Restore the
        # image-aware variants after that registration step.
        type(self).set_color = type(self)._set_color_compat
        type(self).set_opacity = type(self)._set_opacity_compat
        self.set_color = self._set_color_compat
        self.set_opacity = self._set_opacity_compat

        if initial_fill_opacity is not None:
            self.set_opacity(initial_fill_opacity)

        # Match Manim's stable image sizing rule (Cairo's frame height is 8
        # scene units). ImageMob's unscaled surface has height 1.
        if scale_to_resolution:
            height = self.pixel_array.shape[0] / scale_to_resolution * 8.0
            with Off(animation_manager=self.animation_manager):
                self.scale(height)

    def _sync_texture(self):
        rgba = torch.from_numpy(self.pixel_array).to(
            device=torch.get_default_device(), dtype=torch.get_default_dtype()
        ) / 255
        rgba5 = torch.cat(
            (rgba[..., :3], torch.zeros_like(rgba[..., :1]), rgba[..., 3:4]), -1
        )
        texture = rgba5.transpose(-3, -2).flip(-2)
        self.color_texture = texture
        return self

    def get_pixel_array(self):
        return self.pixel_array

    def _set_color_compat(self, color, alpha=None, family=True):
        self.pixel_array[..., :3] = _color_to_rgb8(color)
        if alpha is not None:
            self.pixel_array[..., 3] = int(round(255 * alpha))
        self._sync_texture()
        if family:
            for child in self.get_non_component_children():
                if hasattr(child, "set_color"):
                    child.set_color(color)
        return self

    def _set_opacity_compat(self, alpha):
        self.pixel_array[..., 3] = np.rint(
            self.orig_alpha_pixel_array.astype(np.float32) * alpha
        ).astype(np.uint8)
        return self._sync_texture()

    def fade(self, darkness=0.5, family=True):
        return self.set_opacity(1 - darkness)

    def interpolate_color(self, mobject1, mobject2, alpha):
        if mobject1.pixel_array.shape != mobject2.pixel_array.shape:
            raise AssertionError("Mobject pixel array shapes incompatible for interpolation")
        values = (
            mobject1.pixel_array.astype(np.float32) * (1 - alpha)
            + mobject2.pixel_array.astype(np.float32) * alpha
        )
        self.pixel_array = values.round().astype(np.uint8)
        self._sync_texture()

    def get_style(self):
        alpha = float(self.pixel_array[..., 3].mean() / 255)
        return {"fill_opacity": alpha}


class ImageMobjectFromCamera(ImageMobject):
    def __init__(
        self,
        camera,
        default_display_frame_config=None,
        **kwargs,
    ):
        self.camera = camera
        if default_display_frame_config is None:
            default_display_frame_config = {
                "stroke_width": 3,
                "buff": 0,
            }
        self.default_display_frame_config = default_display_frame_config
        pixel_array = getattr(camera, "pixel_array", None)
        if pixel_array is None:
            pixel_array = np.zeros((1, 1, 4), dtype=np.uint8)
        super().__init__(pixel_array, scale_to_resolution=False, **kwargs)
        with Off(animation_manager=self.animation_manager):
            self.scale(3)

    def get_pixel_array(self):
        pixel_array = getattr(self.camera, "pixel_array", self.pixel_array)
        if isinstance(pixel_array, torch.Tensor):
            pixel_array = pixel_array.detach().cpu().numpy()
        self.pixel_array = np.asarray(pixel_array)
        return self.pixel_array

    def add_display_frame(self, **kwargs):
        from algan.mobs.shapes_2d import SurroundingRectangle

        config = dict(self.default_display_frame_config)
        config.update(kwargs)
        # Algan's native name is ``buffer``; accept Manim's ``buff``.
        if "buff" in config:
            config["buffer"] = config.pop("buff")
        if "stroke_width" in config:
            config["border_width"] = config.pop("stroke_width") / 2
        if "stroke_color" in config:
            config["border_color"] = config.pop("stroke_color")
        # The frame is added as a child but Algan renders registered actors
        # rather than walking the hierarchy, so it has to join the scene to be
        # drawn at all.
        self.display_frame = SurroundingRectangle(
            self, scene=self.scene, **config
        )
        self.add_children(self.display_frame)
        return self


OpenGLImageMobject = ImageMobject

__all__ = [
    "AbstractImageMobject",
    "ImageMobject",
    "ImageMobjectFromCamera",
    "OpenGLImageMobject",
]
