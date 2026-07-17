"""Warm-process render daemon: re-run a scene script without paying startup.

Every fresh ``python scene.py`` pays ~8 s of library import plus ~10 s of
Taichi kernel preparation before the first pixel renders. This daemon pays
them once: it keeps the process (and Taichi's in-process kernel cache) alive
and re-executes the scene script on demand, so from the second render on the
only cost is the render itself.

Usage::

    .venv/Scripts/python.exe -m algan.daemon scene.py [options] [-- script args]

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
* Settings snapshot/restore -- public scalar toggles of
  ``algan.rendering.raytracing.settings`` and the ``*_DEFAULTS`` singletons
  are reset to their import-time values, so one run's ``set_*`` calls cannot
  leak into the next. (Shallow: in-place mutation of a tensor held by a
  default would leak; reassignments -- the normal pattern -- are covered.
  Private state is deliberately kept, e.g. the adaptive gen-fused decision.)
* User helper modules -- modules imported from the script's directory tree
  are evicted from ``sys.modules`` so the next run picks up their edits (the
  daemon prints what it evicted). Modules imported from elsewhere are NOT
  reloaded.

Limits: edits to algan itself require a daemon restart -- already-imported
modules stay stale, and editing ``*_taichi.py`` kernel sources under a live
Taichi JIT can compile mixed-version kernels (the daemon warns when it sees
algan sources change). Keep to one rendering process at a time on Windows.
"""
import argparse
import os
import queue
import runpy
import socketserver
import sys
import threading
import time
import traceback

import algan  # noqa: F401  (the whole point: pay the import once, up front)
from algan import SceneManager

DEFAULT_PORT = int(os.environ.get("ALGAN_DAEMON_PORT", "46711"))
_ALGAN_DIR = os.path.dirname(os.path.abspath(algan.__file__))

_SIMPLE = (bool, int, float, str, tuple, type(None))


def _say(msg):
    print(f"[algan-daemon] {msg}", flush=True)


def _is_under(path, root):
    path = os.path.normcase(os.path.abspath(path))
    root = os.path.normcase(os.path.abspath(root))
    try:
        return os.path.commonpath([path, root]) == root
    except ValueError:  # different drives on Windows
        return False


class _SettingsSnapshot:
    """Import-time snapshot of the mutable settings surface."""

    def __init__(self):
        from algan.rendering.raytracing import settings as rt_settings

        self._module_values = [
            (rt_settings, {
                name: value for name, value in vars(rt_settings).items()
                if not name.startswith("_") and name.isupper()
                and isinstance(value, _SIMPLE)
            }),
        ]
        self._object_dicts = []
        for attr in ("COMPUTING_DEFAULTS", "RENDERING_DEFAULTS",
                     "STYLE_DEFAULTS", "DIRECTORY_DEFAULTS"):
            obj = getattr(algan, attr, None)
            if obj is not None:
                self._object_dicts.append((obj, dict(vars(obj))))
        try:
            from algan.settings.renderer_settings import RENDERER_SETTINGS
            self._object_dicts.append(
                (RENDERER_SETTINGS, dict(vars(RENDERER_SETTINGS))))
        except Exception:
            pass

    def restore(self):
        for module, values in self._module_values:
            for name, value in values.items():
                setattr(module, name, value)
        for obj, saved in self._object_dicts:
            live = vars(obj)
            for key in [k for k in live if k not in saved]:
                del live[key]
            live.update(saved)


class _AlganSourceGuard:
    """Warn when algan's own sources change under a live daemon."""

    def __init__(self):
        self._mtimes = {}
        for dirpath, dirnames, filenames in os.walk(_ALGAN_DIR):
            dirnames[:] = [d for d in dirnames
                           if d not in ("external_libraries", "__pycache__")]
            for fn in filenames:
                if fn.endswith(".py"):
                    path = os.path.join(dirpath, fn)
                    try:
                        self._mtimes[path] = os.stat(path).st_mtime_ns
                    except OSError:
                        pass

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
        shown = ", ".join(os.path.relpath(p, _ALGAN_DIR)
                          for p in changed[:5])
        more = f" (+{len(changed) - 5} more)" if len(changed) > 5 else ""
        _say(f"WARNING: algan sources changed since startup: {shown}{more}")
        _say("WARNING: imported algan modules are stale -- restart the "
             "daemon to pick these up.")
        if any(p.endswith("_taichi.py") for p in changed):
            _say("WARNING: *_taichi.py changed under a live JIT: newly "
                 "compiled kernel variants would mix old and new source. "
                 "Restart the daemon before rendering anything new.")


def _user_modules(script_dir):
    """Names of loaded modules whose source lives under the script's tree."""
    names = []
    for name, module in list(sys.modules.items()):
        file = getattr(module, "__file__", None)
        # Require a real absolute path: torch sets e.g.
        # torch.ops.__file__ = "torch.ops", which would otherwise resolve
        # relative to the cwd and match the script tree.
        if (isinstance(file, str) and os.path.isabs(file)
                and _is_under(file, script_dir)
                and not _is_under(file, _ALGAN_DIR)):
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
        else:
            self.wfile.write(b"err: expected render | ping | quit\n")


def _start_socket(events, port):
    try:
        server = socketserver.ThreadingTCPServer(
            ("127.0.0.1", port), _TriggerHandler)
    except OSError as e:
        _say(f"trigger socket unavailable on 127.0.0.1:{port} ({e}); "
             "stdin trigger still works.")
        return None
    server.daemon_threads = True
    server.events = events
    threading.Thread(target=server.serve_forever, daemon=True,
                     name="algan-daemon-socket").start()
    _say(f"trigger socket on 127.0.0.1:{port} -- poke with:")
    _say("  python -c \"import socket;s=socket.create_connection"
         f"(('127.0.0.1',{port}),2);s.sendall(b'render\\n');"
         "print(s.recv(16).decode().strip())\"")
    return server


def _start_stdin(events):
    def loop():
        interactive = False
        try:
            interactive = sys.stdin.isatty()
        except Exception:
            pass
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
                    _say("stdin trigger inactive (no interactive terminal); "
                         "use the socket, --watch, or Ctrl+C.")
                return
            got_line = True
            command = line.strip().lower()
            if command in ("q", "quit", "exit"):
                events.put(("quit", "stdin"))
                return
            if command in ("", "r", "render"):
                events.put(("render", "stdin"))
            else:
                _say(f"unknown command {command!r} "
                     "(Enter = re-render, q = quit)")

    threading.Thread(target=loop, daemon=True,
                     name="algan-daemon-stdin").start()


class _Watcher:
    """Poll the script + its helper modules; enqueue a render on change."""

    def __init__(self, events):
        self.events = events
        self._paths = set()
        self._mtimes = {}
        self._lock = threading.Lock()
        threading.Thread(target=self._loop, daemon=True,
                         name="algan-daemon-watch").start()

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
                    ("render",
                     "changed: " + ", ".join(os.path.basename(p)
                                             for p in changed)))


def _drain(events, first):
    """Coalesce queued triggers: any quit wins, else one render."""
    kind, reason = first
    while True:
        try:
            nxt = events.get_nowait()
        except queue.Empty:
            return kind, reason
        if nxt[0] == "quit":
            return nxt
        # extra renders coalesce into the one we already hold


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m algan.daemon",
        description="Warm-process algan render daemon: keeps the library "
                    "and compiled kernels loaded, re-running SCRIPT on "
                    "demand.",
        epilog="Triggers: Enter in this terminal; the localhost socket "
               "(see startup banner); --watch. Script args go after '--'.")
    parser.add_argument("script", help="scene script to (re-)execute")
    parser.add_argument("script_args", nargs="*",
                        help="arguments passed through to the script")
    parser.add_argument("--watch", action="store_true",
                        help="also re-render when the script or its helper "
                             "modules change on disk")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"trigger socket port (default {DEFAULT_PORT})")
    parser.add_argument("--no-serve", action="store_true",
                        help="do not open the trigger socket")
    parser.add_argument("--no-initial-render", action="store_true",
                        help="wait for a trigger instead of rendering once "
                             "at startup")
    args = parser.parse_args(argv)

    script = os.path.abspath(args.script)
    if not os.path.isfile(script):
        parser.error(f"script not found: {script}")
    script_dir = os.path.dirname(script)
    # Match `python scene.py`: the script's directory is importable.
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    events = queue.Queue()
    snapshot = _SettingsSnapshot()
    guard = _AlganSourceGuard()
    server = None if args.no_serve else _start_socket(events, args.port)
    watcher = _Watcher(events) if args.watch else None
    _start_stdin(events)

    run_count = 0

    def do_run(reason):
        nonlocal run_count
        run_count += 1
        guard.warn_if_changed()
        if run_count > 1:
            evicted = _user_modules(script_dir)
            for name in evicted:
                del sys.modules[name]
            if evicted:
                _say("reloading: " + ", ".join(sorted(evicted)))
            snapshot.restore()
            SceneManager.reset()
        _say(f"run #{run_count} ({reason})")
        old_argv = sys.argv
        started = time.perf_counter()
        try:
            sys.argv = [script] + list(args.script_args)
            runpy.run_path(script, run_name="__main__")
            _say(f"run #{run_count} finished in "
                 f"{time.perf_counter() - started:.1f} s")
        except SystemExit as e:
            _say(f"run #{run_count} exited (code {e.code}) after "
                 f"{time.perf_counter() - started:.1f} s")
        except KeyboardInterrupt:
            _say(f"run #{run_count} interrupted; state will be reset on "
                 "the next run")
        except Exception:
            traceback.print_exc()
            _say(f"run #{run_count} FAILED after "
                 f"{time.perf_counter() - started:.1f} s -- fix the script "
                 "and re-trigger")
        finally:
            sys.argv = old_argv
        if watcher is not None:
            watcher.set_paths(
                {script} | {getattr(sys.modules[n], "__file__", None)
                            for n in _user_modules(script_dir)} - {None})

    if not args.no_initial_render:
        do_run("startup")
    elif watcher is not None:
        watcher.set_paths({script})
    _say("ready -- Enter = re-render, q = quit")

    try:
        while True:
            try:
                event = events.get(timeout=0.5)
            except queue.Empty:
                continue
            kind, reason = _drain(events, event)
            if kind == "quit":
                _say(f"quitting ({reason})")
                break
            do_run(reason)
            _say("ready -- Enter = re-render, q = quit")
    except KeyboardInterrupt:
        _say("quitting (Ctrl+C)")
    finally:
        if server is not None:
            server.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
