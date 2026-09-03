"""The viewer's HTTP surface: static page, frame images, and a small JSON API.

Deliberately the standard library and nothing else. Algan ships no GUI toolkit,
and a viewer is not a good reason to make every install carry one -- a browser is
already on the machine, and HTML gives the tree, the canvas and the editable
fields for free. It also means the whole thing is drivable from a test without a
display.

Routes:

``GET /``                           the page
``GET /static/<name>``              its script and stylesheet
``GET /api/state``                  runtime, frame rate, size, what is cached
``GET /frame/<n>.png``              one rendered frame, rendering it if need be
``GET /api/hierarchy``              the Scene's root nodes
``GET /api/children?node=``         one node's children
``GET /api/attrs?node=&frame=``     one node's animatable attributes
``GET /api/fragments?frame=&x=&y=`` the fragment list behind one pixel
``POST /api/resolution?name=``      re-render everything at another resolution
``POST /api/shutdown``              stop serving

Everything but the page itself and its own static files carries a per-session
token, minted when the server is built and handed to the browser in the URL
:meth:`ViewerServer.url` returns. Binding to ``127.0.0.1`` is not on its own
enough to keep other pages out: a page on any origin can POST to a localhost
URL without being allowed to read the answer, which is all it takes to hit
``/api/shutdown``, and a name that resolves to 127.0.0.1 (DNS rebinding) turns
the whole API into same-origin for that page. The token closes the first --
it is unguessable and not in a cookie, so a cross-site request cannot carry it
-- and the ``Host`` check closes the second, because a rebound page reaches
here under its own hostname rather than ``localhost``.
"""

from __future__ import annotations

import ipaddress
import json
import mimetypes
import os
import secrets
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

STATIC = Path(__file__).parent / "static"

#: Hostnames a request may name in its ``Host`` header. A browser reaching the
#: viewer by its address uses one of these; anything else is another name that
#: has been pointed at this machine.
_ALLOWED_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "[::1]"})


def _host_is_local(host_header: str | None) -> bool:
    """Whether ``Host`` names this machine rather than a rebound domain."""
    if not host_header:
        # HTTP/1.1 requires it; a client that omits it is not a browser.
        return False
    host = host_header.rsplit(":", 1)[0] if _has_port(host_header) else host_header
    if host in _ALLOWED_HOSTS:
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def _has_port(host_header: str) -> bool:
    """Whether ``Host`` carries a ``:port`` suffix, IPv6 brackets allowed."""
    if host_header.startswith("["):
        return host_header.rfind("]") < host_header.rfind(":")
    return host_header.count(":") == 1


class _Handler(BaseHTTPRequestHandler):
    server_version = "Algan"

    # The page polls; the default logger would print a line per poll.
    def log_message(self, format, *args):  # noqa: A002
        pass

    @property
    def session(self):
        return self.server.session

    def _refused(self, route, query) -> bool:
        """Whether this request was refused; the refusal is already sent.

        The page and its static files are answered to anyone who can reach the
        port: they carry no Scene data and do nothing. Everything else needs
        the session token.
        """
        if not _host_is_local(self.headers.get("Host")):
            self._error(
                HTTPStatus.FORBIDDEN,
                "the Algan viewer answers only to localhost; this request "
                f"named the host {self.headers.get('Host')!r}",
            )
            return True
        if route == "/" or route.startswith("/static/"):
            return False
        token = (query.get("t") or [""])[0]
        if not secrets.compare_digest(token, self.server.token):
            self._error(
                HTTPStatus.FORBIDDEN,
                "missing or wrong viewer session token; open the URL the "
                "viewer printed, which carries one",
            )
            return True
        return False

    def do_GET(self):  # noqa: N802
        url = urlparse(self.path)
        query = parse_qs(url.query)
        route = url.path
        if self._refused(route, query):
            return None
        try:
            if route == "/":
                return self._file(STATIC / "index.html")
            if route.startswith("/static/"):
                name = route[len("/static/") :]
                # Serve only what this package ships, by name, so a crafted
                # path cannot walk out of the static directory.
                target = (STATIC / name).resolve()
                if target.parent != STATIC.resolve() or not target.is_file():
                    return self._error(HTTPStatus.NOT_FOUND, "no such file")
                return self._file(target)
            if route.startswith("/frame/") and route.endswith(".png"):
                index = int(route[len("/frame/") : -len(".png")])
                return self._png(self.session.frame(index))
            if route == "/api/state":
                return self._json(self.session.state())
            if route == "/api/hierarchy":
                return self._json({"roots": self.session.roots()})
            if route == "/api/children":
                node = int(_one(query, "node"))
                components = _one(query, "components", "0") == "1"
                rows = self.session.children(node, components)
                if rows is None:
                    return self._error(HTTPStatus.NOT_FOUND, "no such node")
                return self._json({"node": node, "children": rows})
            if route == "/api/attrs":
                node = int(_one(query, "node"))
                frame = _one(query, "frame", None)
                rows = self.session.attributes(
                    node, None if frame in (None, "") else int(frame)
                )
                if rows is None:
                    return self._error(HTTPStatus.NOT_FOUND, "no such node")
                return self._json(rows)
            # Not ``/api/pixel``, however much it wants to be. "pixel" is the
            # canonical tracking-pixel keyword, and EasyPrivacy -- which uBlock
            # Origin, Adblock Plus and Ghostery all ship by default -- carries
            # generic ``/pixel?`` rules that match on the path alone, host
            # irrelevant, so a page served from 127.0.0.1 is filtered like any
            # other. The extension cancels the request before it is sent, the
            # page sees a bare ``TypeError``, and the reported symptom is that
            # inspection hangs for eight seconds (the client's retry ladder) and
            # then fails with a network error no server log can explain.
            if route == "/api/fragments":
                return self._json(
                    self.session.pixel(
                        int(_one(query, "frame", "0")),
                        int(_one(query, "x", "0")),
                        int(_one(query, "y", "0")),
                    )
                )
            if route == "/api/prefetch":
                self.session.prefetch(int(_one(query, "frame", "0")))
                return self._json({"ok": True})
        except (TypeError, ValueError) as exc:
            return self._error(HTTPStatus.BAD_REQUEST, str(exc))
        except TimeoutError as exc:
            # A frame that is still rendering is not an error the page should
            # give up on; it retries, and meanwhile shows that it is waiting.
            return self._error(HTTPStatus.SERVICE_UNAVAILABLE, str(exc))
        except Exception as exc:  # noqa: BLE001
            return self._error(
                HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(exc).__name__}: {exc}"
            )
        return self._error(HTTPStatus.NOT_FOUND, "no such route")

    def do_POST(self):  # noqa: N802
        url = urlparse(self.path)
        if self._refused(url.path, parse_qs(url.query)):
            return None
        if url.path == "/api/shutdown":
            self._json({"ok": True})
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return None
        if url.path == "/api/resolution":
            # POST, not GET: this one throws away every rendered frame and
            # starts the video again at another size.
            try:
                name = _one(parse_qs(url.query), "name")
                payload = self.session.set_resolution(name)
            except (TypeError, ValueError) as exc:
                return self._error(HTTPStatus.BAD_REQUEST, str(exc))
            except Exception as exc:  # noqa: BLE001
                return self._error(
                    HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(exc).__name__}: {exc}"
                )
            if payload is None:
                return self._error(HTTPStatus.NOT_FOUND, "no such resolution")
            return self._json(payload)
        return self._error(HTTPStatus.NOT_FOUND, "no such route")

    # -- replies ----------------------------------------------------------

    def _send(self, status, body, content_type, *, cache=False):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        # A frame's bytes never change once rendered, but the API's answers do.
        self.send_header(
            "Cache-Control", "public, max-age=86400" if cache else "no-store"
        )
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload, status=HTTPStatus.OK):
        self._send(status, json.dumps(payload).encode(), "application/json")

    def _png(self, data):
        self._send(HTTPStatus.OK, data, "image/png", cache=True)

    def _file(self, path):
        kind = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self._send(HTTPStatus.OK, path.read_bytes(), kind)

    def _error(self, status, message):
        self._json({"error": message}, status)


def _one(query, name, default=...):
    """One query parameter, or the default; missing without one is an error."""
    values = query.get(name)
    if not values:
        if default is ...:
            raise ValueError(f"missing query parameter {name!r}")
        return default
    return values[0]


class ViewerServer(ThreadingHTTPServer):
    """An HTTP server bound to one :class:`~algan.viewer.session.ViewerSession`."""

    daemon_threads = True

    # POSIX only, and deliberately so. There it means "do not make me wait out
    # TIME_WAIT", which is what you want when re-running a script on a fixed
    # ``port=``. On Windows the same flag means something else entirely: it lets
    # a second socket bind a port that is *already in use*, and connections then
    # go to whichever bound last. Two viewers on one port is not a hypothetical
    # -- it happened here, four deep, and the failure it produces is vicious:
    # requests are split between the servers, and killing one makes the port
    # start refusing connections instantly, which a page reports as
    # ``TypeError: Failed to fetch``. Better to fail loudly at bind time.
    allow_reuse_address = os.name != "nt"

    def __init__(self, session, host="127.0.0.1", port=0):
        super().__init__((host, port), _Handler)
        self.session = session
        #: This session's key to its own API. Minted per server, never
        #: persisted, and reachable only by whoever was given the URL.
        self.token = secrets.token_urlsafe(16)

    @property
    def url(self):
        """The address to open, token included -- what the viewer prints."""
        return self.url_for("/")

    @property
    def origin(self):
        """The scheme, host and port, with no path and no token."""
        host, port = self.server_address[:2]
        return f"http://{host}:{port}"

    def url_for(self, path):
        """One route's full URL, with this session's token attached.

        The token is what every route but the page needs, so building a
        request URL by concatenating onto :attr:`url` does not work -- that one
        already carries a query string. This is the way to address a route from
        outside the browser.
        """
        separator = "&" if "?" in path else "?"
        return f"{self.origin}{path}{separator}t={self.token}"
