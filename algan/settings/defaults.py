"""Compatibility aliases for the pre-SETTINGS defaults API."""
from algan.errors import AlganConfigurationError
from algan.settings import SETTINGS


class _ComputingDefaultsProxy:
    _aliases = {
        "portion_of_memory_used_for_animating": "animation_memory_fraction",
        "portion_of_memory_used_for_rendering": "rendering_memory_fraction",
        "max_animate_batch_size": "max_animation_batch_size",
    }

    def __getattr__(self, name):
        if name in {"animation_device", "render_device", "render_on_cpu"}:
            env = "ALGAN_ANIMATION_DEVICE" if name == "animation_device" else "ALGAN_RENDER_DEVICE"
            raise AlganConfigurationError(
                f"{name} is initialization-only; set {env} before importing algan"
            )
        return getattr(SETTINGS.computing, self._aliases.get(name, name))

    def __setattr__(self, name, value):
        if name.startswith("_"):
            return object.__setattr__(self, name, value)
        if name in {"animation_device", "render_device", "render_on_cpu"}:
            env = "ALGAN_ANIMATION_DEVICE" if name == "animation_device" else "ALGAN_RENDER_DEVICE"
            raise AlganConfigurationError(
                f"{name} is initialization-only; set {env} before importing algan"
            )
        SETTINGS.computing.set(**{self._aliases.get(name, name): value})


class _RenderingDefaultsProxy:
    @property
    def settings(self):
        return SETTINGS.video

    @settings.setter
    def settings(self, value):
        SETTINGS.video.set(**value.to_dict())

    @property
    def shader(self):
        return SETTINGS.style.default_shader

    @shader.setter
    def shader(self, value):
        SETTINGS.style.set(default_shader=value)


COMPUTING_DEFAULTS = _ComputingDefaultsProxy()
DIRECTORY_DEFAULTS = SETTINGS.paths
STYLE_DEFAULTS = SETTINGS.style
RENDERING_DEFAULTS = _RenderingDefaultsProxy()

__all__ = [
    "COMPUTING_DEFAULTS",
    "DIRECTORY_DEFAULTS",
    "STYLE_DEFAULTS",
    "RENDERING_DEFAULTS",
]
