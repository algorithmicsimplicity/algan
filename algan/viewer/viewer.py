"""``Scene.view()``: open the Scene you have authored in an interactive viewer."""

from __future__ import annotations

import threading
import webbrowser

from algan.logging.logger import get_logger
from algan.viewer.server import ViewerServer
from algan.viewer.session import ViewerSession

logger = get_logger()


class ViewerHandle:
    """A running viewer, and the means to stop it.

    Returned by :meth:`~algan.scene.Scene.view`. Also a context manager, so a
    script can open a viewer for as long as it needs one and be sure the port is
    released.
    """

    def __init__(self, server, session):
        self._server = server
        self._session = session
        self._thread = None

    @property
    def url(self) -> str:
        """The address the viewer is served at, session token included."""
        return self._server.url

    def url_for(self, path: str) -> str:
        """One route's full URL, with this session's token attached.

        Every route but the page itself needs the token, and :attr:`url`
        already carries a query string, so a script or a test addressing the
        API asks for its URL here rather than concatenating onto that one.
        """
        return self._server.url_for(path)

    @property
    def session(self) -> ViewerSession:
        """The viewer's own handle on the Scene, for tests and scripting."""
        return self._session

    def serve_forever(self) -> None:
        """Serve on this thread until :meth:`stop` is called."""
        self._server.serve_forever()

    def start(self) -> ViewerHandle:
        """Serve on a background thread and return immediately."""
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._server.serve_forever,
                name="algan-viewer-http",
                daemon=True,
            )
            self._thread.start()
        return self

    def stop(self) -> None:
        """Stop serving and release the port."""
        self._server.shutdown()
        self._server.server_close()
        self._session.close()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.stop()
        return False


def _view(
    scene=None,
    video_settings=None,
    *,
    port: int = 0,
    open_browser: bool = True,
    block: bool = True,
) -> ViewerHandle:
    """Implementation of :meth:`algan.scene.Scene.view`.

    Private on purpose. The viewer is reached from the Scene and nowhere else --
    ``view`` is too general a name to export from a package whose ``__all__`` a
    user dumps into their own namespace with ``from algan import *``. The
    user-facing contract, including every parameter, is documented on
    :meth:`~algan.scene.Scene.view`; ``scene`` here is the Scene to look at, and
    defaults to the active one.
    """
    if scene is None:
        from algan.scene import Scene

        scene = Scene.current()
    session = ViewerSession(scene, video_settings)
    server = ViewerServer(session, port=port)
    handle = ViewerHandle(server, session)
    logger.info("Algan viewer serving at %s", handle.url)
    if open_browser:
        # A machine with no browser (a container, a remote shell) raises or
        # silently does nothing; either way the URL has already been printed.
        try:
            webbrowser.open(handle.url)
        except Exception:  # noqa: BLE001
            logger.info("Could not open a browser; visit %s", handle.url)
    if not block:
        return handle.start()
    print(f"Algan viewer: {handle.url}  (Ctrl-C to stop)")
    try:
        handle.serve_forever()
    except KeyboardInterrupt:
        # Say it before stopping, not after: ``stop`` waits for the render batch
        # in flight, which can take tens of seconds, and a Ctrl-C that prints
        # nothing until it is over reads as a viewer that ignored it.
        print("\nAlgan viewer stopping (waiting for the batch in flight)...")
    finally:
        handle.stop()
        print("Algan viewer stopped.")
    return handle
