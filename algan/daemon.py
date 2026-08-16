r"""Warm-process render daemon: re-run a scene script without paying startup.

Every fresh ``python scene.py`` pays ~8 s of library import plus ~10 s of
Taichi kernel preparation before the first pixel renders. This daemon pays
them once: it keeps the process (and Taichi's in-process kernel cache) alive
and re-executes the scene script on demand, so from the second render on the
only cost is the render itself.

Usage::

    .venv/Scripts/python.exe -m algan.daemon               # general daemon
    .venv/Scripts/python.exe -m algan.daemon scene.py [options] [-- script args]

**General mode (no SCRIPT) is the one to leave running.** The daemon publishes
a state file at ``$ALGAN_HOME/daemon.json`` and then serves whatever scripts
come to it: every subsequent ``python any_scene.py`` notices the state file
during ``import algan`` and hands itself over (see :mod:`algan.daemon_client`),
so scripts are launched exactly as they always were and simply start rendering
in ~1 s. Launch it once, forget about it. With no daemon running, or with
``ALGAN_USE_DAEMON=0``, scripts run in their own process exactly as before.

Concurrent scripts are **queued and run one at a time**, in arrival order --
which is also what this project needs on Windows, where two live render
processes fight over the output mp4. A waiting client is told its position.

Triggers (a render is never interrupted; triggers arriving mid-render
coalesce into at most one queued re-run):

* **Enter** in the daemon terminal re-renders; ``q`` quits. This is the
  primary workflow: edit in your editor, save, switch to the daemon, Enter.
* A **localhost TCP socket** (default port 46711; ``--port``, or env
  ``ALGAN_DAEMON_PORT``; ``--no-serve`` disables) accepts the line commands
  ``render`` / ``ping`` / ``quit``. Bind an editor key to the stdlib
  one-liner (deliberately not ``-m algan.daemon``, which would import the
  whole library just to poke the socket)::

      python -c "import socket;s=socket.create_connection(('127.0.0.1',46711),2);s.sendall(b'render\\n');print(s.recv(16).decode().strip())"

* ``--watch`` re-renders when the scene script or any of its sibling helper
  modules change on disk (polled; coalesced; never interrupts).

Between runs the daemon restores a clean slate:

* ``SceneManager.reset()`` -- fresh scene, camera, light and timeline.
* ``SETTINGS.snapshot()`` / ``SETTINGS.restore()`` resets every public
  runtime settings section to its import-time value, so one run cannot leak
  configuration into the next. Private adaptive renderer state is retained.
* User helper modules -- modules imported from the script's directory tree
  are evicted from ``sys.modules`` so the next run picks up their edits (the
  daemon prints what it evicted). Modules imported from elsewhere are NOT
  reloaded.

Limits: edits to algan itself require a daemon restart -- already-imported
modules stay stale, and editing ``*_taichi.py`` kernel sources under a live
Taichi JIT can compile mixed-version kernels (the daemon warns when it sees
algan sources change). Keep to one rendering process at a time on Windows.

Two more that are specific to serving other processes:

* **Startup-only settings cannot be adopted from a client.**
  ``ALGAN_RENDER_DEVICE`` and friends are read while Torch/Taichi initialise,
  i.e. when the *daemon* started. A script that sets one to a different value
  is refused with an explanation and runs cold in its own process, rather than
  being silently rendered on the wrong device.
* **Anything that can reach 127.0.0.1 can ask the daemon to execute a path.**
  Requests must carry the token from the state file, which lives in the user's
  home directory (mode 0600 where the platform honours it). Do not forward the
  port off-host.
"""

from __future__ import annotations

import _thread
import argparse
import contextlib
import json
import os
import queue
import runpy
import secrets
import socketserver
import struct
import sys
import threading
import time
import traceback

# This process *is* a daemon: it must never hand its own work to another one,
# and neither must the scripts it executes. Set before algan is imported,
# because the handoff hook fires during that import -- without this, launching
# a second daemon while one is live would make the second serve itself to the
# first.
os.environ["ALGAN_DAEMON_CHILD"] = "1"

import algan  # noqa: E402, F401  (the whole point: pay the import once, up front)
from algan import SceneManager
from algan import daemon_client as _dc
from algan.settings import SETTINGS

DEFAULT_PORT = int(os.environ.get("ALGAN_DAEMON_PORT", "46711"))
_ALGAN_DIR = os.path.dirname(os.path.abspath(algan.__file__))

# The daemon's own console, captured before any run can redirect sys.stdout to
# a client socket. Daemon chatter always lands here, never in a client's
# output stream.
_CONSOLE = sys.stdout


def _say(msg):
    print(f"[algan-daemon] {msg}", file=_CONSOLE, flush=True)


def _is_under(path, root):
    path = os.path.normcase(os.path.abspath(path))
    root = os.path.normcase(os.path.abspath(root))
    try:
        return os.path.commonpath([path, root]) == root
    except ValueError:  # different drives on Windows
        return False


class _SettingsSnapshot:
    """Import-time snapshot of the public settings and service registries."""

    def __init__(self):
        self._settings = SETTINGS.snapshot()
        from algan.settings.kernel_settings import KERNEL_REGISTRY
        from algan.settings.renderer_settings import RENDERER_REGISTRY

        self._renderer = dict(vars(RENDERER_REGISTRY))
        self._kernel = dict(vars(KERNEL_REGISTRY))

    def restore(self):
        SETTINGS.restore(self._settings)
        from algan.settings.kernel_settings import KERNEL_REGISTRY
        from algan.settings.renderer_settings import RENDERER_REGISTRY

        vars(RENDERER_REGISTRY).clear()
        vars(RENDERER_REGISTRY).update(self._renderer)
        vars(KERNEL_REGISTRY).clear()
        vars(KERNEL_REGISTRY).update(self._kernel)


class _AlganSourceGuard:
    """Warn when algan's own sources change under a live daemon."""

    def __init__(self):
        self._mtimes = {}
        for dirpath, dirnames, filenames in os.walk(_ALGAN_DIR):
            dirnames[:] = [
                d for d in dirnames if d not in ("external_libraries", "__pycache__")
            ]
            for fn in filenames:
                if fn.endswith(".py"):
                    path = os.path.join(dirpath, fn)
                    with contextlib.suppress(OSError):
                        self._mtimes[path] = os.stat(path).st_mtime_ns

    def warn_if_changed(self):
        changed = []
        for path, mtime in self._mtimes.items():
            try:
                if os.stat(path).st_mtime_ns != mtime:
                    changed.append(path)
            except OSError:
                changed.append(path)
        if not changed:
            return
        shown = ", ".join(os.path.relpath(p, _ALGAN_DIR) for p in changed[:5])
        more = f" (+{len(changed) - 5} more)" if len(changed) > 5 else ""
        _say(f"WARNING: algan sources changed since startup: {shown}{more}")
        _say(
            "WARNING: imported algan modules are stale -- restart the "
            "daemon to pick these up."
        )
        if any(p.endswith("_taichi.py") for p in changed):
            _say(
                "WARNING: *_taichi.py changed under a live JIT: newly "
                "compiled kernel variants would mix old and new source. "
                "Restart the daemon before rendering anything new."
            )


class _StateFile:
    """Publishes ``$ALGAN_HOME/daemon.json`` for the lifetime of the daemon.

    Its presence is how an ordinary ``python scene.py`` discovers that a
    daemon exists at all -- and its absence is what keeps the discovery check
    down to a single ``isfile`` for everyone who never launches one.
    """

    def __init__(self, port):
        self.path = _dc.state_path()
        self.token = secrets.token_hex(16)
        self._payload = {
            "protocol": _dc.PROTOCOL_VERSION,
            "port": port,
            "pid": os.getpid(),
            "token": self.token,
            "env": _dc.startup_env(),
        }

    def write(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        # Create with owner-only permissions *before* the token reaches the
        # disk; the file is a capability to execute arbitrary paths.
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        fd = os.open(self.path, flags, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(self._payload, fh)
        with contextlib.suppress(OSError):
            os.chmod(self.path, 0o600)

    def remove(self):
        with contextlib.suppress(OSError):
            # Only clear our own registration -- a daemon started later owns
            # the file now and must not be de-registered by our shutdown.
            with open(self.path, encoding="utf-8") as fh:
                if json.load(fh).get("pid") != os.getpid():
                    return
            os.remove(self.path)


class _RunJob:
    """One client's request to run a script, and the pipe back to that client.

    Frames are written from whichever thread is producing output (the main
    thread, during the run); the handler thread only waits on :attr:`done`.
    """

    def __init__(self, request, stream):
        self.request = request
        self.script = request["script"]
        self.argv = list(request.get("argv", ()))
        self.cwd = request.get("cwd") or os.getcwd()
        self.isatty_out = bool(request.get("isatty_out"))
        self.isatty_err = bool(request.get("isatty_err"))
        self.done = threading.Event()
        self._stream = stream
        self._lock = threading.Lock()
        self._closed = False

    def send(self, kind, payload=b""):
        with self._lock:
            if self._closed:
                return
            try:
                _dc.write_frame(self._stream, kind, payload)
                self._stream.flush()
            except OSError:
                # The client went away (Ctrl-C, killed terminal). Keep
                # rendering -- the run may still be producing a file someone
                # wants -- but stop trying to talk to a closed socket.
                self._closed = True

    def info(self, message):
        self.send(_dc.FRAME_INFO, message)

    def finish(self, code):
        self.send(_dc.FRAME_EXIT, struct.pack("!i", int(code)))
        with self._lock:
            self._closed = True
        self.done.set()


class _ClientStream:
    """``sys.stdout``/``sys.stderr`` stand-in: tees to console and client.

    ``fileno`` deliberately reports the daemon console's descriptor rather
    than raising: subprocesses (ffmpeg, in particular) inherit it, so their
    output lands in the daemon's terminal instead of crashing the run. That is
    a known asymmetry -- Python-level output reaches the client, C-level and
    subprocess output does not.
    """

    def __init__(self, console, job, kind, isatty):
        self._console = console
        self._job = job
        self._kind = kind
        self._isatty = isatty

    def write(self, text):
        if isinstance(text, bytes):
            text = text.decode("utf-8", "replace")
        self._console.write(text)
        self._job.send(self._kind, text)
        return len(text)

    def writelines(self, lines):
        for line in lines:
            self.write(line)

    def flush(self):
        with contextlib.suppress(Exception):
            self._console.flush()

    def isatty(self):
        return self._isatty

    def fileno(self):
        return self._console.fileno()

    @property
    def encoding(self):
        return getattr(self._console, "encoding", "utf-8")

    @property
    def errors(self):
        return getattr(self._console, "errors", "replace")

    closed = False


@contextlib.contextmanager
def _client_streams(job):
    """Redirect Python-level stdout/stderr to ``job`` for the duration."""
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = _ClientStream(old_out, job, _dc.FRAME_STDOUT, job.isatty_out)
    sys.stderr = _ClientStream(old_err, job, _dc.FRAME_STDERR, job.isatty_err)
    try:
        yield
    finally:
        sys.stdout, sys.stderr = old_out, old_err


def _refuse(stream, reason):
    with contextlib.suppress(OSError):
        _dc.write_frame(stream, _dc.FRAME_REFUSE, reason)
        stream.flush()


def _user_modules(script_dir):
    """Names of loaded modules whose source lives under the script's tree."""
    names = []
    for name, module in list(sys.modules.items()):
        file = getattr(module, "__file__", None)
        # Require a real absolute path: torch sets e.g.
        # torch.ops.__file__ = "torch.ops", which would otherwise resolve
        # relative to the cwd and match the script tree.
        if (
            isinstance(file, str)
            and os.path.isabs(file)
            and _is_under(file, script_dir)
            and not _is_under(file, _ALGAN_DIR)
        ):
            names.append(name)
    return names


class _TriggerHandler(socketserver.StreamRequestHandler):
    def handle(self):
        try:
            line = self.rfile.readline().decode(errors="replace").strip()
        except OSError:
            return
        command = line.lower()
        if command == "render":
            self.server.events.put(("render", "socket"))
            self.wfile.write(b"ok\n")
        elif command == "quit":
            self.server.events.put(("quit", "socket"))
            self.wfile.write(b"ok\n")
        elif command == "ping":
            self.wfile.write(b"pong\n")
        elif command == "run":
            self._handle_run()
        elif command.startswith("cancel "):
            self._handle_cancel(line.split(None, 1)[1].strip())
        else:
            self.wfile.write(b"err: expected run | cancel | render | ping | quit\n")

    def _handle_run(self):
        """Accept a client's script, queue it, and block until it has run."""
        try:
            header = self.rfile.read(4)
            if len(header) != 4:
                return
            (length,) = struct.unpack("!I", header)
            request = json.loads(self.rfile.read(length).decode("utf-8"))
        except (OSError, ValueError, struct.error):
            _refuse(self.wfile, "malformed run request")
            return

        if not secrets.compare_digest(
            str(request.get("token", "")), self.server.state.token
        ):
            _say("rejected a run request with a bad token")
            _refuse(self.wfile, "bad daemon token -- is the state file stale?")
            return
        if request.get("protocol") != _dc.PROTOCOL_VERSION:
            _refuse(
                self.wfile,
                f"daemon speaks protocol {_dc.PROTOCOL_VERSION}, client speaks "
                f"{request.get('protocol')!r} -- restart the daemon",
            )
            return
        mismatch = _dc.describe_env_mismatch(
            request.get("env") or {}, self.server.state._payload["env"]
        )
        if mismatch is not None:
            _say("refused a run: " + mismatch.splitlines()[0])
            _refuse(self.wfile, mismatch)
            return
        script = request.get("script") or ""
        if not os.path.isfile(script):
            _refuse(self.wfile, f"script not found on the daemon's host: {script}")
            return

        job = _RunJob(request, self.wfile)
        depth = self.server.events.qsize()
        self.server.events.put(("run", job))
        if depth or self.server.busy.is_set():
            job.info(f"queued behind {depth + 1} run(s) -- waiting")
        # ThreadingTCPServer gives this connection its own thread, so blocking
        # here holds the socket open for the run's output without stalling
        # anything else.
        job.done.wait()

    def _handle_cancel(self, token):
        if not secrets.compare_digest(str(token), self.server.state.token):
            self.wfile.write(b"err: bad token\n")
            return
        if self.server.busy.is_set():
            # do_run executes on the main thread, so this raises
            # KeyboardInterrupt inside the running script -- the same thing
            # Ctrl-C would have done had the script owned the terminal.
            _say("cancel requested by a client")
            _thread.interrupt_main()
            self.wfile.write(b"ok\n")
        else:
            self.wfile.write(b"idle\n")


def _start_socket(events, port, state, busy):
    try:
        server = socketserver.ThreadingTCPServer(("127.0.0.1", port), _TriggerHandler)
    except OSError as e:
        _say(
            f"trigger socket unavailable on 127.0.0.1:{port} ({e}); "
            "stdin trigger still works."
        )
        return None
    server.daemon_threads = True
    server.events = events
    server.state = state
    server.busy = busy
    threading.Thread(
        target=server.serve_forever, daemon=True, name="algan-daemon-socket"
    ).start()
    _say(f"trigger socket on 127.0.0.1:{port} -- poke with:")
    _say(
        '  python -c "import socket;s=socket.create_connection'
        f"(('127.0.0.1',{port}),2);s.sendall(b'render\\n');"
        'print(s.recv(16).decode().strip())"'
    )
    return server


def _start_stdin(events):
    def loop():
        interactive = False
        with contextlib.suppress(Exception):
            interactive = sys.stdin.isatty()
        got_line = False
        while True:
            try:
                line = input()
            except (EOFError, OSError):
                # Quit only for a deliberate close of a live interactive
                # session (Ctrl+Z/Ctrl+D after use). A detached, piped or
                # broken stdin running dry just disables this trigger --
                # the socket / --watch / Ctrl+C still control the daemon.
                if interactive and got_line:
                    events.put(("quit", "stdin closed"))
                else:
                    _say(
                        "stdin trigger inactive (no interactive terminal); "
                        "use the socket, --watch, or Ctrl+C."
                    )
                return
            got_line = True
            command = line.strip().lower()
            if command in ("q", "quit", "exit"):
                events.put(("quit", "stdin"))
                return
            if command in ("", "r", "render"):
                events.put(("render", "stdin"))
            else:
                _say(f"unknown command {command!r} (Enter = re-render, q = quit)")

    threading.Thread(target=loop, daemon=True, name="algan-daemon-stdin").start()


class _Watcher:
    """Poll the script + its helper modules; enqueue a render on change."""

    def __init__(self, events):
        self.events = events
        self._paths = set()
        self._mtimes = {}
        self._lock = threading.Lock()
        threading.Thread(
            target=self._loop, daemon=True, name="algan-daemon-watch"
        ).start()

    def set_paths(self, paths):
        with self._lock:
            self._paths = set(paths)
            self._mtimes = {p: self._stat(p) for p in self._paths}

    @staticmethod
    def _stat(path):
        try:
            s = os.stat(path)
            return (s.st_mtime_ns, s.st_size)
        except OSError:
            return None

    def _loop(self):
        while True:
            time.sleep(0.5)
            with self._lock:
                changed = []
                for path in self._paths:
                    now = self._stat(path)
                    if now != self._mtimes.get(path):
                        self._mtimes[path] = now
                        changed.append(path)
            if changed:
                self.events.put(
                    (
                        "render",
                        "changed: " + ", ".join(os.path.basename(p) for p in changed),
                    )
                )


def _drain(events, first):
    """Coalesce queued triggers: any quit wins, else one render.

    Returns ``(event, deferred)``. Client ``run`` requests are **never**
    coalesced -- each is a different script and every client is owed its own
    output and exit code -- so any encountered while draining is handed back
    for the caller to re-queue, preserving arrival order.
    """
    if first[0] != "render":
        return first, []
    deferred = []
    while True:
        try:
            nxt = events.get_nowait()
        except queue.Empty:
            return first, deferred
        if nxt[0] == "quit":
            return nxt, deferred
        if nxt[0] == "run":
            deferred.append(nxt)
        # extra renders coalesce into the one we already hold


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m algan.daemon",
        description="Warm-process algan render daemon: keeps the library "
        "and compiled kernels loaded, re-running SCRIPT on "
        "demand.",
        epilog="Triggers: Enter in this terminal; the localhost socket "
        "(see startup banner); --watch. Script args go after '--'.",
    )
    parser.add_argument(
        "script",
        nargs="?",
        help="scene script to (re-)execute. Omit it for a general daemon that "
        "serves whatever scripts are launched against it.",
    )
    parser.add_argument(
        "script_args", nargs="*", help="arguments passed through to the script"
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="also re-render when the script or its helper modules change on disk",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"trigger socket port (default {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--no-serve", action="store_true", help="do not open the trigger socket"
    )
    parser.add_argument(
        "--no-initial-render",
        action="store_true",
        help="wait for a trigger instead of rendering once at startup",
    )
    args = parser.parse_args(argv)

    script = os.path.abspath(args.script) if args.script else None
    if script is not None and not os.path.isfile(script):
        parser.error(f"script not found: {script}")
    if script is None and args.watch:
        parser.error("--watch needs a SCRIPT to watch")
    if script is None and args.no_serve:
        parser.error(
            "a general daemon with no SCRIPT and no socket has nothing to do; "
            "give it a SCRIPT or drop --no-serve"
        )
    if script is not None:
        _add_to_path(os.path.dirname(script))

    events = queue.Queue()
    busy = threading.Event()
    snapshot = _SettingsSnapshot()
    guard = _AlganSourceGuard()
    state = _StateFile(args.port)
    server = None if args.no_serve else _start_socket(events, args.port, state, busy)
    if server is not None:
        state.write()
        _say(f"serving any script launched on this machine (state: {state.path})")
        _say("scripts run normally -- `python scene.py` will find this daemon")
    watcher = _Watcher(events) if args.watch else None
    _start_stdin(events)

    run_count = 0
    # Every script tree the daemon has executed. User modules from all of them
    # are evicted before each run, so an edit is always picked up and one
    # script's helpers never satisfy another's import.
    script_dirs = set()
    last = {"script": script, "args": list(args.script_args), "cwd": os.getcwd()}

    def reset_state():
        evicted = sorted({n for d in script_dirs for n in _user_modules(d)})
        for name in evicted:
            sys.modules.pop(name, None)
        if evicted:
            _say("reloading: " + ", ".join(evicted))
        snapshot.restore()
        SceneManager.reset()

    def execute(path, script_args, cwd, reason):
        """Run one script to completion. Returns its exit code."""
        nonlocal run_count
        run_count += 1
        guard.warn_if_changed()
        if run_count > 1:
            reset_state()
        script_dirs.add(os.path.dirname(path))
        _add_to_path(os.path.dirname(path))
        _say(f"run #{run_count} ({reason}): {path}")
        old_argv, old_cwd = sys.argv, os.getcwd()
        started = time.perf_counter()
        code = 0
        try:
            sys.argv = [path] + list(script_args)
            with contextlib.suppress(OSError):
                os.chdir(cwd)
            runpy.run_path(path, run_name="__main__")
            _say(f"run #{run_count} finished in {time.perf_counter() - started:.1f} s")
        except SystemExit as e:
            code = 0 if e.code is None else (e.code if isinstance(e.code, int) else 1)
            _say(
                f"run #{run_count} exited (code {code}) after "
                f"{time.perf_counter() - started:.1f} s"
            )
        except KeyboardInterrupt:
            code = 130
            _say(f"run #{run_count} interrupted; state will be reset on the next run")
        except BaseException:  # noqa: BLE001 -- one script must not kill the daemon
            code = 1
            traceback.print_exc()
            _say(
                f"run #{run_count} FAILED after "
                f"{time.perf_counter() - started:.1f} s -- fix the script "
                "and re-trigger"
            )
        finally:
            sys.argv = old_argv
            with contextlib.suppress(OSError):
                os.chdir(old_cwd)
        if watcher is not None and path == script:
            watcher.set_paths(
                {path}
                | {
                    getattr(sys.modules[n], "__file__", None)
                    for n in _user_modules(os.path.dirname(path))
                }
                - {None}
            )
        return code

    def do_local(reason):
        target = last["script"]
        if target is None:
            _say("no script to re-run yet -- launch one and it will land here")
            return
        busy.set()
        try:
            execute(target, last["args"], last["cwd"], reason)
        finally:
            busy.clear()

    def do_job(job):
        busy.set()
        job.send(_dc.FRAME_START)
        last.update(script=job.script, args=job.argv, cwd=job.cwd)
        try:
            with _client_streams(job):
                code = execute(job.script, job.argv, job.cwd, "client")
        except BaseException:
            code = 1
            traceback.print_exc()
        finally:
            busy.clear()
            job.finish(code)

    if script is not None and not args.no_initial_render:
        do_local("startup")
    elif watcher is not None:
        watcher.set_paths({script})
    _say("ready -- Enter = re-run the last script, q = quit")

    try:
        while True:
            try:
                event = events.get(timeout=0.5)
            except queue.Empty:
                continue
            event, deferred = _drain(events, event)
            for item in deferred:
                events.put(item)
            kind, payload = event
            if kind == "quit":
                _say(f"quitting ({payload})")
                break
            if kind == "run":
                do_job(payload)
            else:
                do_local(payload)
            _say("ready -- Enter = re-run the last script, q = quit")
    except KeyboardInterrupt:
        _say("quitting (Ctrl+C)")
    finally:
        state.remove()
        if server is not None:
            server.shutdown()
        _drop_pending(events)
    return 0


def _add_to_path(directory):
    """Match ``python scene.py``: the script's directory is importable."""
    if directory and directory not in sys.path:
        sys.path.insert(0, directory)


def _drop_pending(events):
    """Release any queued clients so they fall back instead of hanging."""
    while True:
        try:
            kind, payload = events.get_nowait()
        except queue.Empty:
            return
        if kind == "run":
            payload.send(_dc.FRAME_REFUSE, "the daemon shut down before this run")
            payload.done.set()


if __name__ == "__main__":
    sys.exit(main())
