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
        """Get the one instance of this class, creating it on first use.

        Returns
        -------
        Singleton
            The same object on every call until :meth:`reset`.
        """
        if cls._instance is None:
            cls._instance = cls._create()
        return cls._instance

    @classmethod
    def reset(cls):
        """Forget the instance, so the next :meth:`instance` call builds a new one.

        Anything still holding the old instance keeps it; this only changes
        what :meth:`instance` returns from now on.
        """
        cls._instance = None
