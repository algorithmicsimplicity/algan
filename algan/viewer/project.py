"""A lazy project catalogue and at most one live Scene viewer."""

from __future__ import annotations

import threading

from algan.viewer.session import ViewerSession


class ProjectViewerSession:
    """Author scenes on first selection; keep only the active worker/cache.

    Opening the catalogue or polling it never calls ``load_scene``. A selection
    drains the previous renderer and inspections before authoring another scene,
    since authoring and rendering both use the process-global scene context.
    Authored recordings are retained until close, so revisits do not re-author.
    Selection versions reject requests belonging to a discarded scene, including
    requests arriving during a slow authoring pass or after a failed selection.
    """

    def __init__(self, scenes, load_scene, video_settings=None):
        self._scenes = dict(scenes)
        if not self._scenes:
            raise ValueError("A project viewer needs at least one scene")
        self._load_scene = load_scene
        self._authored = {}
        self._video_settings = video_settings
        self._lock = threading.RLock()
        self._switch_lock = threading.Lock()
        self._closed = False
        self._version = 0
        self._scene_id = None
        self._session = None
        self._loading_scene_id = None
        self._error = None

    def _open(self, scene_id):
        if scene_id not in self._authored:
            # Cache only successful authoring. A failed scene can be retried by
            # clicking its tab, but polling must never retry user code.
            self._authored[scene_id] = self._load_scene(scene_id)
        scene, settings = self._authored[scene_id]
        return ViewerSession(scene, self._video_settings, raytracing=settings)

    def session_for_request(self, version=None):
        with self._lock:
            if self._closed:
                raise ValueError("The project viewer is closed")
            if version is not None and int(version) != self._version:
                raise ValueError("The selected scene changed; refresh the viewer")
            if self._session is None:
                raise ValueError("Select a scene tab and wait for it to finish loading")
            return self._session

    def state(self):
        with self._lock:
            return {
                **(self._session.state() if self._session is not None else {}),
                "scene_id": self._scene_id,
                "scene_version": self._version,
                "loading_scene_id": self._loading_scene_id,
                **({"error": self._error} if self._session is None else {}),
                "scenes": [
                    {"id": scene_id, "name": name}
                    for scene_id, name in self._scenes.items()
                ],
            }

    def select_scene(self, scene_id):
        """Load one stable ID; invalid IDs leave the current viewer alone."""
        with self._switch_lock:
            with self._lock:
                if self._closed:
                    raise ValueError("The project viewer is closed")
                if scene_id not in self._scenes:
                    return None
                if scene_id == self._scene_id:
                    return self.state()
                previous = self._session
                # Invalidate the old selection *before* closing/authoring. The
                # catalogue stays responsive, without exposing a closed worker.
                self._session = None
                self._scene_id = None
                self._version += 1
                self._loading_scene_id = scene_id
                self._error = None
            try:
                if previous is not None:
                    # Never time out and start authoring over a live renderer.
                    previous.close(timeout=None)
                session = self._open(scene_id)
            except Exception as exc:
                message = (
                    f"Could not load project scene {self._scenes[scene_id]}: "
                    f"{type(exc).__name__}: {exc}"
                )
                with self._lock:
                    self._error = message
                raise RuntimeError(message) from exc
            finally:
                with self._lock:
                    self._loading_scene_id = None
            with self._lock:
                self._session = session
                self._scene_id = scene_id
            return self.state()

    def close(self):
        with self._switch_lock:
            with self._lock:
                self._closed = True
                session = self._session
                self._session = None
                self._scene_id = None
            if session is not None:
                session.close(timeout=None)
            self._authored.clear()
            self._load_scene = None
