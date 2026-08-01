"""Shared base for Algan's process-global singleton services.

Only :class:`~algan.scene_manager.SceneManager` currently uses this base.
Timeline, animation, and audio managers are ordinary per-scene objects.
"""
from __future__ import annotations


class Singleton:
    """Lazily-created process-global singleton base."""

    _instance = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._instance = None

    def __init__(self):
        name = type(self).__name__
        raise RuntimeError(f"Call {name}.instance() instead of {name}().")

    @classmethod
    def _create(cls):
        raise NotImplementedError

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls._create()
        return cls._instance

    @classmethod
    def reset(cls):
        cls._instance = None
