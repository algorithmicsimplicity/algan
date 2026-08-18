r"""Thin client that hands a scene script to an algan daemon, starting one if
none is running.

A fresh ``python scene.py`` pays several seconds of library import plus ~20 s
of Taichi kernel preparation before the first pixel renders, even with a warm
offline cache. :mod:`algan.daemon` pays that once and keeps the process warm;
this module is the half that lets an *ordinary* script use it without being
launched any differently.

The flow, from the script's point of view:

1. ``import algan`` reaches :func:`maybe_handoff` before any heavy import.
2. If a daemon is running -- there is a state file at
   ``$ALGAN_HOME/daemon.json`` -- the client ships
   ``(cwd, script path, argv, environment)`` to it, streams the run's
   stdout/stderr back to its own, and exits with the daemon's exit code. The
   client never imports torch or taichi, so the whole round trip is Python
   startup plus the render itself.
3. If none is running, the client starts one in the background and waits for it
   to come up (:func:`_autostart`), then hands off as above. The first run
   therefore costs what it always did -- the daemon pays the same cold start
   the script would have -- and every later run starts warm. Disable with
   ``ALGAN_AUTO_DAEMON=0``.

**Any problem falls back to a normal in-process run** -- a refused handshake, a
daemon that is not listening, a spawn that fails or is slow to come up. The one
thing that is *not* recoverable is a daemon that dies after the script has
started executing: the script's side effects have already happened, so
re-running it locally could duplicate them. That case reports an error and
exits non-zero.

``ALGAN_USE_DAEMON=0`` disables the handoff even when a daemon is running, and
``ALGAN_AUTO_DAEMON=0`` disables only the starting of new ones.
``ALGAN_DAEMON_CHILD=1`` is set by the daemon around its own execution of a
script, and is what stops the handoff from recursing.

A run on the daemon is meant to be indistinguishable from a run in its own
process, since with auto-start even a first run lands there. What the daemon
reproduces: ``sys.argv``, the working directory, the full environment, stdout
and stderr including subprocess and C-level output, tty-ness of both streams,
and the exit code. What it deliberately does not: ``stdin`` (connected to
``os.devnull``; the daemon's own stdin is its re-render trigger) and ``atexit``
handlers (``runpy`` does not run them, and a warm process never shuts down).

Nothing heavier than the standard library and :mod:`algan.environment` (which
is itself stdlib-only, and already imported by the time this module loads) may
be imported here, and the module must stay that way: it runs before
``algan/__init__.py`` has imported torch, and its whole value is that a client
process never pays for what it does not use.
"""

from __future__ import annotations

import contextlib
import json
import os
import socket
import struct
import sys
import time

from algan.environment import (
    env_flag,
    env_float,
    env_int,
    env_str,
    startup_environment_variables,
)

#: Bumped when the wire format changes. A client and daemon that disagree
#: refuse each other rather than misparse, and the client falls back.
#:
#: 2 -- the run request carries the client's full environment (``env_full``),
#: which the daemon applies for the duration of the run.
PROTOCOL_VERSION = 2

#: Seconds to wait for the daemon's TCP accept. Short on purpose: a daemon
#: that cannot answer promptly is not worth blocking a cold run for.
CONNECT_TIMEOUT = env_float("ALGAN_DAEMON_TIMEOUT", 2.0)

# Frame kinds on the daemon -> client stream. Each frame is one kind byte, a
# 4-byte big-endian length, then that many payload bytes.
FRAME_STDOUT = b"O"
FRAME_STDERR = b"E"
FRAME_INFO = b"I"  # daemon chatter (queue position); utf-8, shown on stderr
FRAME_REFUSE = b"R"  # handshake rejected; utf-8 reason; client falls back
FRAME_START = b"S"  # the script is now executing -- fallback is no longer safe
FRAME_EXIT = b"X"  # payload is a 4-byte big-endian signed exit code

#: Environment variables consumed while Torch and Taichi initialise. The daemon
#: baked its values at launch and cannot change them, so a client whose values
#: differ is refused and runs cold rather than silently rendering on the wrong
#: device or against the wrong cache. Declared in :mod:`algan.environment`
#: alongside every other variable Algan honors; see also the
#: "Initialization-only settings" section of ``CLAUDE.md``.
STARTUP_ENV = startup_environment_variables()


class DaemonUnavailable(Exception):
    """The daemon cannot serve this run; the caller should run in-process.

    Raised only while it is still safe to fall back -- that is, before the
    daemon reports that the script has started executing.
    """


class DaemonUnreachable(DaemonUnavailable):
    """Nothing answered at the registered address.

    Distinct from a daemon that answered and *refused*: this means the state
    file names a process that is gone -- killed, crashed, or left behind by a
    reboot -- so the registration is stale and should be replaced rather than
    obeyed on every subsequent run.
    """


class DaemonRunFailed(Exception):
    """The daemon accepted the run and then died. Falling back is unsafe."""


def algan_home():
    """The ``$ALGAN_HOME`` directory (default ``~/.algan``), as a path string.

    Duplicated from :mod:`algan.settings._startup` rather than imported: that
    module imports torch, which is exactly what a client is avoiding.
    """
    home = env_str("ALGAN_HOME") or os.path.join(os.path.expanduser("~"), ".algan")
    return os.path.expanduser(home)


def state_path():
    """Path of the daemon's state file. Its absence means "no daemon"."""
    return os.path.join(algan_home(), "daemon.json")


def startup_env():
    """The subset of the environment that the daemon bakes in at launch."""
    return {name: env_str(name, "") for name in STARTUP_ENV}


def describe_env_mismatch(client_env, daemon_env):
    """Human-readable report of startup-env differences, or ``None`` if same."""
    diffs = [
        f"  {name}: this script wants {client_env.get(name, '') or '<unset>'!r}, "
        f"daemon was launched with {daemon_env.get(name, '') or '<unset>'!r}"
        for name in STARTUP_ENV
        if client_env.get(name, "") != daemon_env.get(name, "")
    ]
    if not diffs:
        return None
    return (
        "startup-only settings differ from the running daemon's:\n"
        + "\n".join(diffs)
        + "\nThese are read while Torch/Taichi initialise, so the daemon "
        "cannot adopt them. Restart the daemon with these values, or set "
        "ALGAN_USE_DAEMON=0 for this script."
    )


# --------------------------------------------------------------------------
# Framing
# --------------------------------------------------------------------------


def write_frame(stream, kind, payload=b""):
    """Write one ``kind``-tagged frame. ``payload`` may be bytes or str."""
    if isinstance(payload, str):
        payload = payload.encode("utf-8", "replace")
    stream.write(kind + struct.pack("!I", len(payload)) + payload)


def read_frame(stream):
    """Read one frame. Returns ``(kind, payload)``, or ``(None, b"")`` at EOF."""
    header = _read_exactly(stream, 5)
    if header is None:
        return None, b""
    kind = header[:1]
    (length,) = struct.unpack("!I", header[1:])
    payload = _read_exactly(stream, length) if length else b""
    if payload is None:
        return None, b""
    return kind, payload


def _read_exactly(stream, count):
    chunks = []
    remaining = count
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


# --------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------


def read_state():
    """Parse the daemon state file, or return ``None`` if there is no daemon."""
    try:
        with open(state_path(), encoding="utf-8") as fh:
            state = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(state, dict) or "port" not in state or "token" not in state:
        return None
    return state


def should_try(main_module=None):
    """Whether this process is an ordinary scene-script run worth handing off.

    Deliberately conservative: anything that is not plainly ``python foo.py``
    -- a REPL, ``python -c``, a notebook, a test runner, the daemon's own
    execution of a script -- runs in-process as before. A false positive would
    silently reroute an unrelated process into a shared daemon, which is much
    worse than a false negative costing one cold start.
    """
    if env_flag("ALGAN_DAEMON_CHILD", False):
        return False
    if not env_flag("ALGAN_USE_DAEMON", True):
        return False
    if sys.flags.interactive:
        return False
    # Test runners and notebooks import algan as a library, not as a script.
    if "pytest" in sys.modules or "PYTEST_CURRENT_TEST" in os.environ:
        return False
    if "ipykernel" in sys.modules:
        return False
    main_module = main_module or sys.modules.get("__main__")
    path = getattr(main_module, "__file__", None)
    if not isinstance(path, str) or not path.endswith(".py"):
        return False
    return os.path.isfile(path)


def script_of(main_module=None):
    """Absolute path of the running ``__main__`` script."""
    main_module = main_module or sys.modules.get("__main__")
    return os.path.abspath(main_module.__file__)


def run_remote(state, script, argv=None, cwd=None, out=None, err=None):
    """Execute ``script`` on the daemon described by ``state``; return its code.

    Raises :class:`DaemonUnavailable` while falling back is still safe, and
    :class:`DaemonRunFailed` once the script has begun executing.
    """
    out = out if out is not None else _binary(sys.stdout)
    err = err if err is not None else _binary(sys.stderr)
    request = {
        "protocol": PROTOCOL_VERSION,
        "token": state["token"],
        "cwd": cwd if cwd is not None else os.getcwd(),
        "script": script,
        "argv": list(argv if argv is not None else sys.argv[1:]),
        "env": startup_env(),
        # The whole environment, applied by the daemon for the duration of the
        # run. Without it a script reads the *daemon's* environment, so
        # ``MY_VAR=x python scene.py`` would silently see the daemon's value.
        # Localhost-only, token-authenticated, and same-user, which is the same
        # trust boundary the script path itself already crosses.
        "env_full": dict(os.environ),
        "isatty_out": _isatty(sys.stdout),
        "isatty_err": _isatty(sys.stderr),
    }
    try:
        sock = socket.create_connection(
            ("127.0.0.1", int(state["port"])), CONNECT_TIMEOUT
        )
    except OSError as exc:
        raise DaemonUnreachable(f"could not reach the daemon ({exc})") from exc

    started = False
    with sock:
        sock.settimeout(None)
        stream = sock.makefile("rwb")
        payload = json.dumps(request).encode("utf-8")
        # The request needs no kind byte -- only the daemon's replies are
        # multiplexed -- so it is a bare 4-byte length and that many bytes.
        stream.write(b"run\n" + struct.pack("!I", len(payload)) + payload)
        stream.flush()
        _install_cancel_handler(state)
        while True:
            kind, data = read_frame(stream)
            if kind is None:
                if started:
                    raise DaemonRunFailed(
                        "the algan daemon stopped responding mid-run. The "
                        "script may have partially completed; re-run it "
                        "deliberately rather than assuming it did nothing."
                    )
                raise DaemonUnreachable("the daemon closed the connection")
            if kind == FRAME_REFUSE:
                raise DaemonUnavailable(data.decode("utf-8", "replace"))
            if kind == FRAME_START:
                started = True
            elif kind == FRAME_INFO:
                err.write(b"[algan-daemon] " + data + b"\n")
                _flush(err)
            elif kind == FRAME_STDOUT:
                out.write(data)
                _flush(out)
            elif kind == FRAME_STDERR:
                err.write(data)
                _flush(err)
            elif kind == FRAME_EXIT:
                (code,) = struct.unpack("!i", data)
                return code


def log_path():
    """Where an auto-started daemon's console output is appended."""
    return os.path.join(algan_home(), "daemon.log")


def _open_log():
    """Open the daemon log for append, trimming it if it has grown large."""
    path = log_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cap = env_int("ALGAN_DAEMON_LOG_MAX_BYTES", 4 * 1024 * 1024)
    with contextlib.suppress(OSError):
        if cap > 0 and os.path.getsize(path) > cap:
            os.replace(path, path + ".old")
    return open(path, "ab")


def _spawn_daemon():
    """Start a detached general daemon. Returns the Popen, or None on failure.

    The child is fully detached: it must outlive this process, which is about
    to hand it a script and exit.
    """
    import subprocess

    idle = env_int("ALGAN_DAEMON_IDLE_TIMEOUT", 1800)
    # The daemon inherits this environment, so the startup-only variables it
    # bakes in match ours and the handoff that follows cannot mismatch.
    # ALGAN_DAEMON_CHILD must not carry over: it would tell the new daemon's
    # own ``import algan`` that it is somebody's child and must not serve.
    env = dict(os.environ)
    env.pop("ALGAN_DAEMON_CHILD", None)
    kwargs = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = 0x00000008 | 0x00000200  # DETACHED | NEW_GROUP
    else:
        kwargs["start_new_session"] = True
    try:
        log = _open_log()
    except OSError:
        return None
    try:
        with log:
            return subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "algan.daemon",
                    "--idle-timeout",
                    str(idle),
                ],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                # Never hold the project directory open: on Windows that blocks
                # deleting it. The daemon chdirs per run anyway.
                cwd=algan_home(),
                env=env,
                **kwargs,
            )
    except (OSError, ValueError):
        return None


def _autostart():
    """Start a daemon and wait for it to publish its state. May return None.

    Returns the new daemon's state dict, or ``None`` if auto-start is disabled,
    the spawn failed, or it did not come up inside the readiness timeout -- in
    every one of those cases the caller simply runs in-process, and a daemon
    that is merely slow will serve the *next* run.
    """
    if not env_flag("ALGAN_AUTO_DAEMON", True):
        return None
    os.makedirs(algan_home(), exist_ok=True)
    proc = _spawn_daemon()
    if proc is None:
        return None
    timeout = env_float("ALGAN_DAEMON_START_TIMEOUT", 60.0)
    _warn(
        "starting a background render daemon so later runs skip the startup "
        f"cost (log: {log_path()}). Disable with ALGAN_AUTO_DAEMON=0."
    )
    deadline = time.monotonic() + max(0.0, timeout)
    while time.monotonic() < deadline:
        state = read_state()
        if state is not None:
            return state
        if proc.poll() is not None:  # died before publishing anything
            _warn(f"the background daemon exited early; see {log_path()}")
            return None
        time.sleep(0.25)
    _warn(
        "the background daemon is still starting; running this script "
        "in-process (it will serve the next run)"
    )
    return None


def _clear_stale_state(state):
    """Delete a state file that still names the daemon we failed to reach.

    Guarded by the token so a daemon that registered in the meantime keeps its
    file. Without this, one hard-killed daemon (SIGKILL, a crash, a reboot)
    would leave a registration that every later run tries, fails, and falls
    back from -- and auto-start would never fire, because a state file exists.
    """
    with contextlib.suppress(OSError, ValueError):
        current = read_state()
        if current is not None and current.get("token") == state.get("token"):
            os.remove(state_path())


def _dispatch(script):
    """Run ``script`` on a daemon. Returns its exit code, or None to run here."""
    state = read_state()
    if state is not None:
        try:
            return run_remote(state, script)
        except DaemonUnreachable as exc:
            _clear_stale_state(state)
            _warn(f"removed a stale daemon registration ({exc})")
        except DaemonUnavailable as exc:
            _warn(f"not using the algan daemon: {exc}")
            return None
    state = _autostart()
    if state is None:
        return None
    try:
        return run_remote(state, script)
    except DaemonUnavailable as exc:
        _warn(f"not using the algan daemon: {exc}")
        return None


def maybe_handoff():
    """Run this process's script on a daemon. Does not return if it does.

    Called from ``algan/__init__.py`` before any heavy import. On success the
    process exits with the daemon's code; otherwise it returns and the import
    continues normally.
    """
    try:
        if not should_try():
            return
        script = script_of()
    except Exception:  # never let the fast path break an ordinary import
        return

    try:
        code = _dispatch(script)
    except DaemonRunFailed as exc:
        _warn(str(exc))
        _exit(1)
    except KeyboardInterrupt:
        _exit(130)
    except Exception as exc:  # noqa: BLE001 -- a broken client must not block work
        _warn(f"not using the algan daemon: {exc!r}")
        return
    if code is None:
        return
    _exit(code)


def _exit(code):
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            stream.flush()
    # _exit, not sys.exit: this process imported half of algan and has no
    # teardown worth running, and atexit hooks belong to the daemon's run.
    os._exit(code)


def _warn(message):
    try:
        sys.stderr.write(f"[algan] {message}\n")
        sys.stderr.flush()
    except Exception:
        pass


def _binary(stream):
    return getattr(stream, "buffer", None) or stream


def _flush(stream):
    with contextlib.suppress(Exception):
        stream.flush()


def _isatty(stream):
    try:
        return bool(stream.isatty())
    except Exception:
        return False


def _install_cancel_handler(state):
    """Forward Ctrl-C to the daemon instead of orphaning a live render.

    Killing the client alone would leave the daemon rendering, and on Windows
    an orphaned render holds its output mp4 locked. A second Ctrl-C gives up
    on the polite route and takes the client down.
    """
    import signal

    seen = []

    def handler(signum, frame):
        seen.append(1)
        if len(seen) > 1:
            raise KeyboardInterrupt
        try:
            with socket.create_connection(
                ("127.0.0.1", int(state["port"])), CONNECT_TIMEOUT
            ) as sock:
                sock.sendall(b"cancel " + state["token"].encode("ascii") + b"\n")
        except OSError:
            raise KeyboardInterrupt from None

    # ValueError/OSError: not the main thread, or no signal support here.
    with contextlib.suppress(ValueError, OSError):
        signal.signal(signal.SIGINT, handler)
