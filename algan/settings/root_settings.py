"""The settings root object and whole-tree snapshots.

:class:`AlganSettings` is the type of :data:`algan.SETTINGS`. It owns the section
instances, keeps their identity stable across mutation, and refuses assignment to
a section attribute -- replacing a section would strand every reference already
taken to it.

:class:`SettingsSnapshot` captures the whole tree at once.
``SETTINGS.snapshot()`` and ``SETTINGS.restore(...)`` are the save/restore pair
for longer-lived flows; ``SETTINGS.override(...)`` is the scoped form for a block
that spans several sections.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

from algan.errors import AlganConfigurationError
from algan.settings.computing_settings import ComputingSettings
from algan.settings.path_settings import PathSettings
from algan.settings.raytracing_settings import RayTracingPreset, RayTracingSettings
from algan.settings.style_settings import StyleSettings
from algan.settings.video_settings import LD, VideoSettings


@dataclass(frozen=True)
class SettingsSnapshot:
    """Immutable snapshot of all public runtime-adjustable settings."""

    computing: ComputingSettings
    paths: PathSettings
    style: StyleSettings
    video: VideoSettings
    raytracing: RayTracingPreset


class AlganSettings:
    """Stable process-global root for runtime-adjustable Algan settings.

    Section objects keep stable identities. Call ``SETTINGS.video.set(...)``
    rather than replacing ``SETTINGS.video`` so imports made by engine modules
    always observe the current values.
    """

    __slots__ = (
        "computing",
        "paths",
        "style",
        "video",
        "raytracing",
        "_skip_save_frame",
    )

    #: The five sections that are public configuration. ``__slots__`` also
    #: carries ``_skip_save_frame``, an engine flag rather than a setting: the
    #: docs build sets it so an example's ``save_frame`` call renders nothing,
    #: and it must not appear in ``dir(SETTINGS)``, ``repr`` or a snapshot
    #: beside ``video`` and ``raytracing`` as though a user should reach for it.
    _SECTIONS = ("computing", "paths", "style", "video", "raytracing")

    def __init__(self):
        object.__setattr__(self, "computing", ComputingSettings())
        object.__setattr__(self, "paths", PathSettings())
        object.__setattr__(self, "style", StyleSettings())
        object.__setattr__(self, "video", LD.as_mutable())
        object.__setattr__(self, "raytracing", RayTracingSettings())
        object.__setattr__(self, "_skip_save_frame", False)

    def __setattr__(self, name, value):
        if name == "_skip_save_frame":
            object.__setattr__(self, "_skip_save_frame", value)
            return
        if name in self._SECTIONS:
            raise AlganConfigurationError(
                f"SETTINGS.{name} has stable identity; call SETTINGS.{name}.set(...)"
            )
        raise AttributeError(name)

    def __dir__(self):
        return sorted(self._SECTIONS) + ["snapshot", "restore", "override"]

    def __repr__(self):
        sections = "\n".join(
            f"  {name}={getattr(self, name)!r}" for name in self._SECTIONS
        )
        return f"SETTINGS(\n{sections}\n)"

    def snapshot(self) -> SettingsSnapshot:
        return SettingsSnapshot(
            computing=self.computing.as_preset(),
            paths=self.paths.as_preset(),
            style=self.style.as_preset(),
            video=self.video.as_preset(),
            raytracing=self.raytracing.as_preset(),
        )

    def restore(self, snapshot: SettingsSnapshot):
        if not isinstance(snapshot, SettingsSnapshot):
            raise AlganConfigurationError(
                "SETTINGS.restore requires a SettingsSnapshot"
            )
        self.computing.set(**snapshot.computing.to_dict())
        self.paths.set(**snapshot.paths.to_dict())
        self.style.set(**snapshot.style.to_dict())
        self.video.set(**snapshot.video.to_dict())
        self.raytracing._restore(snapshot.raytracing.to_dict())
        return self

    @contextmanager
    def override(self, **sections):
        """Temporarily override one or more sections and restore on exit.

        Example: ``with SETTINGS.override(raytracing={"samples_per_pixel": 4}):``
        """
        unknown = [name for name in sections if name not in self._SECTIONS]
        if unknown:
            raise AlganConfigurationError(f"Unknown settings section '{unknown[0]}'")
        snapshot = self.snapshot()
        try:
            for name, values in sections.items():
                if not isinstance(values, dict):
                    raise AlganConfigurationError(
                        f"SETTINGS.override({name}=...) requires a dict of fields"
                    )
                getattr(self, name).set(**values)
            yield self
        finally:
            self.restore(snapshot)
