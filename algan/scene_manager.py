"""Process-global owner of Algan's active-scene stack.

Only :class:`SceneManager` is a singleton.  Scenes themselves are ordinary,
self-contained objects: each owns its timeline, animation, and audio managers.
Creating a scene pushes it onto this manager's stack, making it the destination
for subsequently constructed mobs that were not given an explicit ``scene``.
"""

from __future__ import annotations

from contextlib import contextmanager

from algan.utils.singleton import Singleton


class SceneManager(Singleton):
    """Maintain the stack of active :class:`~algan.scene.Scene` objects.

    ``SceneManager.instance()`` returns the manager (not a Scene).  Access the
    topmost scene through :attr:`current_scene`.  A default scene is created
    lazily when the stack is empty, preserving Algan's traditional module-level
    authoring style while allowing nested, isolated scenes.
    """

    _memory = None
    _scene_class = None
    _scene_initializer = None

    @classmethod
    def _create(cls):
        manager = cls.__new__(cls)
        manager._scene_stack = []
        manager._creating_default_scene = False
        return manager

    @classmethod
    def set_scene_class(cls, scene_class, scene_initializer):
        cls._scene_class = scene_class
        cls._scene_initializer = scene_initializer

    @property
    def scene_stack(self):
        """An immutable snapshot of the active-scene stack."""
        return tuple(self._scene_stack)

    @property
    def current_scene(self):
        """Return the topmost active scene, creating the default lazily."""
        if not self._scene_stack:
            self._create_default_scene()
        return self._scene_stack[-1]

    def _create_default_scene(self):
        manager_type = type(self)
        scene_class = manager_type._scene_class
        scene_initializer = manager_type._scene_initializer
        if scene_class is None:
            raise RuntimeError(
                "SceneManager has not been configured with a Scene class"
            )
        if self._creating_default_scene:
            raise RuntimeError("Recursive default Scene creation")
        self._creating_default_scene = True
        try:
            # Scene.__init__ pushes itself before running its initializer, so
            # camera/light mobs constructed by the initializer resolve back to
            # this partially-created but already active scene.
            scene_class(
                memory=manager_type._memory,
                scene_initializer=scene_initializer,
            )
        finally:
            self._creating_default_scene = False

    def push(self, scene):
        """Push ``scene`` and make it active.

        A scene may occur at most once in the stack.  Re-entering an already
        active scene is therefore a no-op; reactivating an inactive scene moves
        it to the top.
        """
        if scene in self._scene_stack:
            if self._scene_stack[-1] is scene:
                return scene
            raise RuntimeError(
                "Cannot reactivate a Scene covered by another active Scene"
            )
        self._scene_stack.append(scene)
        scene._terminated = False
        return scene

    @contextmanager
    def activating(self, scene):
        """Temporarily make ``scene`` current while preserving the stack.

        This is used for scene-owned callbacks such as custom initializers when
        an explicit, currently inactive scene is reset or rendered.
        """
        if self._scene_stack and self._scene_stack[-1] is scene:
            yield scene
            return
        snapshot = list(self._scene_stack)
        self._scene_stack = [item for item in self._scene_stack if item is not scene]
        self._scene_stack.append(scene)
        try:
            yield scene
        finally:
            self._scene_stack = snapshot

    def terminate(self, scene):
        """Pop ``scene`` from the stack.

        Stack discipline is deliberate: terminating a covered scene would make
        containment ambiguous, so only the current topmost scene can be popped.
        Calling this for an already-terminated scene is harmless.
        """
        if scene not in self._scene_stack:
            scene._terminated = True
            return scene
        if self._scene_stack[-1] is not scene:
            raise RuntimeError("Only the current active Scene can be terminated")
        self._scene_stack.pop()
        scene._terminated = True
        return scene

    @classmethod
    def reset(cls):
        """Discard every scene and return a fresh default active scene.

        This retains the singleton manager object itself; only its scene stack
        is reset.  Existing scene references remain valid objects but are marked
        terminated and no longer receive implicitly-created mobs.
        """
        manager = cls.instance()
        for scene in manager._scene_stack:
            scene._terminated = True
        manager._scene_stack.clear()
        return manager.current_scene
