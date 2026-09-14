"""A project catalogue and one live Scene viewer, never concurrent renderers."""

from __future__ import annotations

import threading

from algan.viewer.session import ViewerSession


class ProjectViewerSession:
    """Retain authored scenes, but keep only the active scene's worker/cache.

    A selection waits for the old worker and all materialized reads to leave
    the scene before creating its replacement. HTTP requests resolve to an
    immutable session reference; the selection version rejects delayed requests
    rather than interpreting an old node/pixel/frame in a different scene.
    """

    def __init__(self, scenes, video_settings=None):
        self._scenes = {
            scene_id: (name, scene, settings)
            for scene_id, name, scene, settings in scenes
        }
        if not self._scenes:
            raise ValueError("A project viewer needs at least one scene")
        self._video_settings = video_settings
        self._lock = threading.RLock()
        self._switch_lock = threading.Lock()
        self._closed = False
        self._version = 0
        self._scene_id = next(iter(self._scenes))
        self._session = self._open(self._scene_id)

    def _open(self, scene_id):
        _, scene, settings = self._scenes[scene_id]
        return ViewerSession(scene, self._video_settings, raytracing=settings)

    def session_for_request(self, version=None):
        with self._lock:
            if self._closed:
                raise ValueError("The project viewer is closed")
            if version is not None and int(version) != self._version:
                raise ValueError("The selected scene changed; refresh the viewer")
            return self._session

    def state(self):
        with self._lock:
            return {
                **self._session.state(),
                "scene_id": self._scene_id,
                "scene_version": self._version,
                "scenes": [
                    {"id": scene_id, "name": entry[0]}
                    for scene_id, entry in self._scenes.items()
                ],
            }

    def select_scene(self, scene_id):
        """Switch to a stable project ID; unknown IDs leave the viewer alone."""
        if scene_id not in self._scenes:
            return None
        with self._switch_lock:
            with self._lock:
                if self._closed:
                    raise ValueError("The project viewer is closed")
                if scene_id == self._scene_id:
                    return self.state()
                previous = self._session
            # Never time out and launch a second renderer over the same arena.
            # close also drains inspections, and refuses queued stale requests.
            previous.close(timeout=None)
            session = self._open(scene_id)
            with self._lock:
                self._session = session
                self._scene_id = scene_id
                self._version += 1
            return self.state()

    def close(self):
        with self._switch_lock:
            with self._lock:
                self._closed = True
                session = self._session
            session.close(timeout=None)
