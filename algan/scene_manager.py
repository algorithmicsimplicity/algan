"""Owner of the process-global :class:`~algan.scene.Scene` singleton.

Importing ``algan`` configures this manager with the default scene class and
initializer (camera + point light); the scene itself is created lazily on the
first :meth:`SceneManager.instance` call. All code that needs "the current
scene" must go through :meth:`SceneManager.instance` rather than caching the
returned object, because :meth:`SceneManager.reset` replaces it wholesale
(along with the animation timeline).
"""

from algan.utils.singleton import Singleton


class SceneManager(Singleton):
    _memory = None
    _scene_class = None
    _scene_initializer = None

    @classmethod
    def set_scene_class(cls, scene_class, scene_initializer):
        cls._scene_class = scene_class
        cls._scene_initializer = scene_initializer

    @classmethod
    def reset(cls):
        # Imported here: animation_contexts and timeline both import this
        # module at import time, so importing them at module level would be
        # circular.
        from algan.animation_timeline.animation_contexts import AnimationManager
        from algan.animation_timeline.timeline import TimelineManager

        AnimationManager.reset()
        TimelineManager.reset()
        super().reset()
        return cls.instance()

    @classmethod
    def _create(cls):
        scene = cls._scene_class(memory=cls._memory)
        scene.scene_initializer = cls._scene_initializer
        # Publish the instance before running the initializer: the camera and
        # light mobs it constructs call SceneManager.instance() re-entrantly.
        cls._instance = scene
        scene.reset_scene()
        return scene
