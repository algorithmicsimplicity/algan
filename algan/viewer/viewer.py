"""``view()``: open the Scene you have authored in an interactive viewer."""

from __future__ import annotations

import threading
import webbrowser

from algan.logging.logger import get_logger
from algan.viewer.server import ViewerServer
from algan.viewer.session import ViewerSession

logger = get_logger()


class ViewerHandle:
    """A running viewer, and the means to stop it.

    Returned by :func:`view`. Also a context manager, so a script can open a
    viewer for as long as it needs one and be sure the port is released.
    """

    def __init__(self, server, session):
        self._server = server
        self._session = session
        self._thread = None

    @property
    def url(self) -> str:
        """The address the viewer is served at."""
        return self._server.url

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
    """Open the Scene in an interactive viewer.

    Starts a small web server on this machine and points a browser at it. The
    page plays the Scene as it stands, and lets you stop on a frame and ask what
    is in it: the Scene's mobs as a tree with their animatable attributes, any
    pixel's colour, and the list of surfaces behind that pixel, nearest first,
    each with its depth and the mob it came from.

    Frames are rendered as you reach them rather than up front, so the window
    opens immediately and seeking costs one chunk of frames. Nothing is written
    to disk, and the Scene is left exactly as authored -- you can keep adding to
    it, or call :meth:`~algan.scene.Scene.save_video`, afterwards.

    The video is the Scene as it stands when you call this. Frames already
    rendered are kept, so if you go on authoring the same Scene from a REPL
    (``block=False``), open a new viewer to see the additions rather than
    expecting this one to grow.

    Parameters
    ----------
    scene
        The Scene to look at. Defaults to ``None``, meaning the active Scene.
    video_settings
        Resolution and anti-aliasing to render at, normally a preset such as
        ``HD``. Defaults to ``None``, meaning the ``PREVIEW`` preset's
        resolution at the Scene's own frame rate -- so seeking stays quick while
        the frame numbers still match the video the Scene would produce.
    port
        Port to serve on. Defaults to 0, meaning any free port.
    open_browser
        Whether to open the page in your default browser. Defaults to True.
    block
        Whether to serve until interrupted. Defaults to True, which is what a
        script wants: the viewer stays up until you close it with Ctrl-C. False
        returns immediately with the viewer running in the background, which is
        what a REPL or a test wants. Note that a blocking viewer running on the
        warm render daemon occupies it until you stop it, since the daemon runs
        one script at a time.

    Returns
    -------
    ViewerHandle
        The running viewer. It carries the ``url`` being served and a ``stop()``
        that shuts it down, and works as a context manager.

    Animation
    ---------
    Records nothing and renders nothing until the page asks for a frame. The
    Scene's timeline, its mobs and its video settings are all left as they were.

    Examples
    --------
    .. code-block:: python

        square = Square().spawn()
        square.move(RIGHT)

        view()  # opens a browser, serves until Ctrl-C

        handle = view(block=False)  # keep scripting while it runs
        print(handle.url)
        handle.stop()
    """
    if scene is None:
        from algan.scene import Scene

        scene = Scene.instance()
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
        print("\nAlgan viewer stopped.")
    finally:
        handle.stop()
    return handle
