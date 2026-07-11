"""Shared base for algan's process-global singletons.

:class:`SceneManager <algan.scene_manager.SceneManager>`,
:class:`TimelineManager <algan.animation.timeline.TimelineManager>`,
:class:`AnimationManager <algan.animation.animation_contexts.AnimationManager>`
and :class:`AudioManager <algan.sound.audio_effect.AudioManager>` all follow
the same pattern: a lazily-created instance reached via ``cls.instance()``,
dropped by ``cls.reset()`` so the next access recreates it. Configuration that
must survive a reset (scene class, speech source, ...) lives in class
attributes on the subclass, not on the instance.
"""


class Singleton:
    """Lazily-created process-global singleton.

    Subclasses implement :meth:`_create` returning the instance. Direct
    construction raises; go through :meth:`instance`. Never cache the returned
    object across a :meth:`reset` — it is replaced wholesale.
    """

    _instance = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Each subclass gets its own slot; otherwise resetting one manager
        # would clobber the cached instance of another via the shared base.
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
