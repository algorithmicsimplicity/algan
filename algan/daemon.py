r"""Warm-process render daemon: re-run a scene script without paying startup.

Every fresh ``python scene.py`` pays several seconds of library import plus
~20 s of Taichi kernel preparation before the first pixel renders, even with a
warm offline cache. This daemon pays them once: it keeps the process (and
Taichi's in-process kernel cache) alive and re-executes the scene script on
demand, so from the second render on the only cost is the render itself.

Usage::

    .venv/Scripts/python.exe -m algan.daemon               # general daemon
    .venv/Scripts/python.exe -m algan.daemon scene.py [options] [-- script args]

**Launching one by hand is optional.** An ordinary ``python scene.py`` starts a
general daemon itself when none is running, runs on it, and leaves it warm for
the next script (see :mod:`algan.daemon_client`; ``ALGAN_AUTO_DAEMON=0``
disables it, ``ALGAN_USE_DAEMON=0`` disables the daemon entirely). Launch one
by hand when you want it in a terminal you can watch, or want Enter-to-re-render.

**General mode (no SCRIPT) is the one to leave running.** The daemon publishes
a state file at ``$ALGAN_HOME/daemon.json`` and then serves whatever scripts
come to it: every subsequent ``python any_scene.py`` notices the state file
during ``import algan`` and hands itself over, so scripts are launched exactly
as they always were and simply start rendering in ~1 s.

Concurrent scripts are **queued and run one at a time**, in arrival order --
which is also what this project needs on Windows, where two live render
processes fight over the output mp4. A waiting client is told its position.

Triggers (a render is never interrupted; triggers arriving mid-render
coalesce into at most one queued re-run):

* **Enter** in the daemon terminal re-renders; ``q`` quits. This is the
  primary workflow: edit in your editor, save, switch to the daemon, Enter.
* A **localhost TCP socket** (preferred port 46711; ``--port``, or env
  ``ALGAN_DAEMON_PORT``; ``--no-serve`` disables) accepts the line commands
  ``render`` / ``ping`` / ``quit``, each of which must carry the token from
  the state file. If the preferred port is taken the daemon binds an
  ephemeral one instead of exiting, and publishes it in the state file --
  which is where clients look anyway. Bind an editor key to::

      algan daemon render     # also: algan daemon ping, algan daemon quit

  Those subcommands read ``$ALGAN_HOME/daemon.json`` for the port and the
  token and send the line for you.

* ``--watch`` re-renders when the scene script or any of its sibling helper
  modules change on disk (polled; coalesced; never interrupts).

When a run ends the daemon restores a clean slate, before it goes idle rather
than at the start of the next run -- so what it holds while waiting is the warm
process and nothing else:

* ``SceneManager.reset()`` -- fresh scene, camera, light and timeline.
* ``SETTINGS.snapshot()`` / ``SETTINGS.restore()`` resets every public
  runtime settings section to its import-time value, so one run cannot leak
  configuration into the next. Private adaptive renderer state is retained.
* User helper modules -- modules imported from the script's directory tree
  are evicted from ``sys.modules`` so the next run picks up their edits (the
  daemon prints what it evicted). Modules imported from elsewhere are NOT
  reloaded.
* **The render's GPU memory goes back to the driver**: one ``gc.collect()``
  (the scene's object graph is cyclic, so refcounting alone frees almost none
  of it) and one ``torch.cuda.empty_cache()``. Measured on a 4 GB card, an
  idle daemon holding 1.6 GB after a 90-frame render now holds ~0.1 GB, at a
  cost of ~0.15 s and no measurable change to the next render.
  ``ALGAN_DAEMON_RELEASE_MEMORY=0`` keeps the memory cached instead.

**Edits to algan itself are handled, not merely warned about.** The daemon
fingerprints every algan source file at startup and re-checks at every run
launch; if anything changed it refuses the run and shuts down, so the script
executes in a fresh process that loads the edited code. A new daemon starts on
the next run. This costs a cold start -- that is what editing the library has
always cost -- but it can no longer render with stale modules or compile
mixed-version kernels from a half-edited ``*_taichi.py``. An edit that lands
*during* a run is not caught, exactly as it is not caught for a plain ``python
scene.py``. Keep to one rendering process at a time on Windows.

A run served here is meant to be indistinguishable from one in its own
process: ``sys.argv``, the working directory, the caller's full environment,
stdout/stderr at the descriptor level (so ffmpeg and other subprocesses reach
the caller) and their tty-ness are all reproduced. ``stdin`` is not -- it is
connected to ``os.devnull``, because the daemon's own stdin is its re-render
trigger -- and ``atexit`` handlers do not run, because ``runpy`` does not run
them and a warm process never shuts down.

Three more limits that are specific to serving other processes:

* **Startup-only settings cannot be adopted from a client**, with one
  exception. ``ALGAN_ANIMATION_DEVICE`` and friends are read while Torch/Taichi
  initialise, i.e. when the *daemon* started. A script that sets one to a
  different value is refused with an explanation and runs cold in its own
  process, rather than being silently rendered on the wrong device.
  ``ALGAN_RENDER_DEVICE`` **is** adopted (:func:`_adopt_render_device`): it
  only seeds ``SETTINGS.computing.render_device``, which owns the value from
  then on, and every render re-selects Taichi's arch from it. A script that
  wants the other device is served warm; if that crosses the CPU/GPU line its
  first render pays one kernel-preparation pass, which is still far less than
  the cold start refusing it used to cost.
* **Neither can settings read while algan is imported.** The renderer's
  toggles become module-level defaults during ``import algan``, which in a
  daemon happened at *its* launch -- so a script that sets one before its own
  ``import algan``, the way every A/B script in ``benchmarks/`` selects an
  arm, would otherwise be served by a process that never saw it and would
  render with the daemon's values instead. Those are refused too
  (:func:`algan.daemon_client.describe_import_env_mismatch`); the swapped-in
  environment covers every variable read live, so flipping one *during* a run
  works here exactly as it does cold. The corollary is that a daemon started
  by a script with non-default toggles serves only scripts that set the same
  ones: stop it if you want one baked with the defaults.
* **Anything that can reach 127.0.0.1 can ask the daemon to execute a path.**
  Every request -- ``run``, ``cancel``, ``render``, ``ping`` and ``quit``
  alike -- must carry the token from the state file, which lives in the user's
  home directory (mode 0600 where the platform honours it). Do not forward the
  port off-host.
* **The daemon must be the same Algan.** The state file records the
  interpreter, the prefix, the package directory and the version it was
  started with, and a client whose own differ is not served: it runs cold in
  its own process rather than executing against another virtualenv's
  site-packages (:func:`algan.daemon_client.describe_interpreter_mismatch`).
"""

from __future__ import annotations

import _thread
import argparse
import codecs
import contextlib
import hashlib
import io
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
# first. Written through os directly for the same reason: importing
# algan.environment to reach its accessors would import algan, which is the
# very import this has to precede.
os.environ["ALGAN_DAEMON_CHILD"] = "1"

import algan  # noqa: E402, F401  (the whole point: pay the import once, up front)
from algan import SceneManager
from algan import daemon_client as _dc
from algan.environment import env_flag, env_int, env_str
from algan.logging.logger import apply_environment_logging
from algan.settings import SETTINGS
from algan.settings.path_settings import output_filename_for, output_root_for
from algan.utils.memory_utils import release_torch_memory


def default_port():
    """The trigger socket's default port.

    Read when the command line is parsed rather than at import: it configures
    the client/daemon transport, not anything a script renders, so there is no
    reason a warm process could not honour a value set after its own import.
    """
    return env_int("ALGAN_DAEMON_PORT", 46711)


_ALGAN_DIR = os.path.dirname(os.path.abspath(algan.__file__))

# The environment algan was imported with, captured immediately after that
# import and never touched again. A run swaps ``os.environ`` for the client's
# (see :func:`_run_context`), so this is the only record of the values that
# were live at the moment the import-time settings read them -- which is what
# :func:`_dc.describe_import_env_mismatch` compares a client against.
_IMPORT_ENVIRON = dict(os.environ)


def _capture_console():
    """A stream on the daemon's real stdout, immune to per-run redirection.

    A run replaces file descriptor 1 with a pipe (see :func:`_run_context`), so
    a console captured as plain ``sys.stdout`` would start writing *into* that
    pipe -- daemon chatter would leak into the client's output, and the pump
    thread echoing to the console would feed itself. Duplicating the descriptor
    first gives a handle on the real terminal that no later ``dup2`` can move.
    """
    try:
        fd = os.dup(sys.stdout.fileno())
    except (OSError, ValueError, AttributeError):
        return sys.stdout  # not a real file (pytest capture, pythonw): as-is
    try:
        return open(
            fd,
            "w",
            buffering=1,
            encoding=getattr(sys.stdout, "encoding", None) or "utf-8",
            errors="replace",
        )
    except OSError:
        os.close(fd)
        return sys.stdout


# The daemon's own console, captured before any run can redirect stdout to a
# client socket. Daemon chatter always lands here, never in a client's stream.
_CONSOLE = _capture_console()


def _say(msg):
    print(f"[algan-daemon] {msg}", file=_CONSOLE, flush=True)


def _is_under(path, root):
    path = os.path.normcase(os.path.abspath(path))
    root = os.path.normcase(os.path.abspath(root))
    try:
        return os.path.commonpath([path, root]) == root
    except ValueError:  # different drives on Windows
        return False


def _torch_reserved_bytes():
    """VRAM torch holds from the driver, or ``None`` when there is no CUDA."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return int(torch.cuda.memory_reserved())
    except Exception:  # noqa: BLE001 -- a memory report must never fail a run
        return None


def _release_run_memory():
    """Hand a finished run's GPU memory back to the driver.

    A daemon exists to stay warm, not to stay *full*: between runs none of the
    render's memory is wanted, and on a small card an idle daemon holding it is
    what makes the next real render -- or anything else on the GPU -- run out.
    Measured on a 4 GB GTX 1050 after a 90-frame render: 1348 MiB still
    allocated and 1500 MiB reserved while idle, released in 0.12 s.

    Two steps, and both are needed. The scene's object graph is cyclic (mobs
    hold their children, the timeline holds every mob), so dropping the
    daemon's reference to it frees almost nothing by refcount alone --
    ``gc.collect()`` is what actually releases the tensors. They then sit in
    torch's caching allocator, visible to the next run but not to any other
    process, until ``torch.cuda.empty_cache()`` returns the blocks to the
    driver. :func:`algan.utils.memory_utils.release_torch_memory` does both.

    The cost is paid by the *next* run, which re-acquires its blocks from the
    driver instead of from the cache; that is a few milliseconds against a
    render. ``ALGAN_DAEMON_RELEASE_MEMORY=0`` keeps the memory instead.
    """
    if not env_flag("ALGAN_DAEMON_RELEASE_MEMORY", True):
        return
    before = _torch_reserved_bytes()
    started = time.perf_counter()
    release_torch_memory(force_gc=True)
    after = _torch_reserved_bytes()
    if before is None or after is None:
        return
    freed = before - after
    if freed >= (64 << 20):
        _say(
            f"released {freed / 2**20:.0f} MiB of GPU memory in "
            f"{time.perf_counter() - started:.2f} s"
        )


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


class _SourceDigest:
    """Content fingerprint of every algan source file.

    The daemon's loaded modules are frozen at import; if the files on disk no
    longer match them, a run served here would render with code that is not
    what a fresh interpreter would load. :meth:`changed_since` is what the
    daemon checks at every run launch to refuse exactly that (see
    ``DESIGN_daemon_lifecycle.md``).

    Content is hashed rather than stat'd, for two reasons. A gate must never
    *miss* an edit, and mtime can be preserved across one. And ``git
    checkout`` / ``stash`` / ``rebase`` rewrite mtimes wholesale without
    changing content, so an mtime gate would force a pointless cold restart on
    every branch switch. Hashing the whole tree costs ~10 ms for the ~300 files
    (5 MB) involved, once per run launch.

    ``external_libraries/`` is included: it is vendored and read-only by
    policy, but it is imported into this same interpreter, so an edit there is
    exactly as stale as any other. A file that cannot be read records the error
    in place of its hash, which compares unequal and so errs toward restarting.
    """

    def __init__(self, files):
        self.files = files

    @classmethod
    def capture(cls, root=None):
        root = _ALGAN_DIR if root is None else root
        files = {}
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d != "__pycache__"]
            for name in filenames:
                if not name.endswith(".py"):
                    continue
                path = os.path.join(dirpath, name)
                key = os.path.relpath(path, root).replace(os.sep, "/")
                try:
                    with open(path, "rb") as handle:
                        files[key] = hashlib.sha256(handle.read()).hexdigest()
                except OSError as exc:
                    files[key] = f"<unreadable:{exc.errno}>"
        return cls(files)

    def changed_since(self, baseline):
        """Sorted relative paths that differ from ``baseline``.

        Added and removed files are included. Empty means this daemon's
        sources are still the ones it imported.
        """
        names = set(self.files) | set(baseline.files)
        return sorted(n for n in names if self.files.get(n) != baseline.files.get(n))


def _stale_message(changed):
    """Explain a staleness refusal to whoever asked for the run."""
    shown = ", ".join(changed[:5])
    more = f" (+{len(changed) - 5} more)" if len(changed) > 5 else ""
    message = (
        f"algan sources changed since this daemon started ({shown}{more}); "
        "the daemon is shutting down and this run will execute in a fresh "
        "process."
    )
    if any(name.endswith("_taichi.py") for name in changed):
        message += (
            " A *_taichi.py changed, so the next process also pays a full "
            "kernel recompile."
        )
    return message


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
            # Which Algan this is, in the four terms a client can check
            # without importing anything heavy. ``$ALGAN_HOME`` defaults to
            # ``~/.algan`` for every project on the machine, so without these
            # a daemon started from one virtualenv would happily execute
            # another's script against its own site-packages -- and the
            # source digest, which hashes *this* tree, cannot see that.
            "python": sys.executable,
            "prefix": sys.prefix,
            "algan_path": _ALGAN_DIR,
            "algan_version": _dc.algan_version(),
        }

    def set_port(self, port):
        """Record the port actually bound (see :func:`_start_socket`)."""
        self._payload["port"] = port

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
        # The client's whole environment, applied for the runtime of the run
        # so the script reads the variables its caller set, not the daemon's.
        env = request.get("env_full")
        self.env = dict(env) if isinstance(env, dict) else None
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


class _RunStream(io.TextIOWrapper):
    """``sys.stdout``/``sys.stderr`` for one run: a stream on the real fd.

    Built directly on the descriptor rather than wrapping whatever
    ``sys.stdout`` happened to be, so Python-level writes provably reach the
    redirected descriptor -- and therefore the client -- even if something
    earlier in the process replaced the stream object with one that is not
    backed by a file descriptor at all.

    ``isatty`` reports the *client's* tty-ness rather than the pipe's. Code
    that adapts to a terminal -- tqdm's progress bars in ``render_loop``, most
    obviously -- would otherwise silently degrade to one line per update.
    """

    def __init__(self, fd, isatty):
        super().__init__(
            io.BufferedWriter(io.FileIO(fd, "w", closefd=False)),
            encoding="utf-8",
            errors="replace",
            line_buffering=True,
        )
        self._algan_isatty = bool(isatty)

    def isatty(self):
        return self._algan_isatty


class _Pump:
    """Forward everything written to one descriptor into client frames.

    The write end replaces file descriptor 1 or 2 for the runtime of a run,
    so Python-level writes, C-level writes and *subprocess* writes (ffmpeg's,
    via moviepy) all travel one ordered channel and all reach the client. The
    previous implementation replaced ``sys.stdout`` only, so anything that
    reached the descriptor directly was lost to the daemon's own terminal.
    """

    def __init__(self, job, kind, console):
        self._job = job
        self._kind = kind
        self._console = console
        self.read_fd, self.write_fd = os.pipe()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name=f"algan-daemon-pump-{kind.decode()}"
        )
        self._thread.start()

    def _loop(self):
        # Incremental, because a read can split a multi-byte character.
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        try:
            with os.fdopen(self.read_fd, "rb", buffering=0) as reader:
                while True:
                    chunk = reader.read(65536)
                    if not chunk:
                        break
                    self._emit(decoder.decode(chunk))
        except OSError:
            pass
        finally:
            with contextlib.suppress(Exception):
                self._emit(decoder.decode(b"", True))

    def _emit(self, text):
        if not text:
            return
        with contextlib.suppress(Exception):
            self._console.write(text)
            self._console.flush()
        self._job.send(self._kind, text)

    def close(self, timeout=5.0):
        """Close the write end and wait for the drain to finish."""
        with contextlib.suppress(OSError):
            os.close(self.write_fd)
        self._thread.join(timeout)


def _swap_std_handle(fd):
    """Point Windows' std handle at ``fd``; returns a restore token or None.

    ``os.dup2`` moves the C runtime's descriptor, which covers Python and the
    CRT -- but a child process launched by :mod:`subprocess` inherits from
    ``GetStdHandle``, not from the descriptor table. Without this, ffmpeg would
    keep writing to the daemon's own console on Windows even though every
    Python-level write was correctly redirected.
    """
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        import msvcrt

        which = -11 if fd == 1 else -12
        kernel32 = ctypes.windll.kernel32
        kernel32.GetStdHandle.restype = ctypes.c_void_p
        kernel32.GetStdHandle.argtypes = [ctypes.c_uint32]
        kernel32.SetStdHandle.argtypes = [ctypes.c_uint32, ctypes.c_void_p]
        previous = kernel32.GetStdHandle(which)
        kernel32.SetStdHandle(which, ctypes.c_void_p(msvcrt.get_osfhandle(fd)))
        return (which, previous)
    except Exception:  # noqa: BLE001 -- best effort; the run must still work
        return None


def _restore_std_handle(token):
    if token is None:
        return
    with contextlib.suppress(Exception):
        import ctypes

        which, previous = token
        ctypes.windll.kernel32.SetStdHandle(which, ctypes.c_void_p(previous))


def _adopt_render_device():
    """Adopt the client's ``ALGAN_RENDER_DEVICE`` for this run.

    The one startup variable a warm process *can* take from a client. It is
    read at import only to seed ``SETTINGS.computing.render_device``, which
    owns the value from then on, and Taichi re-selects its arch at the start of
    every render (``taichi_runtime.ensure_taichi_for_render``) -- so applying
    the client's value here makes the run render where a cold one would.

    Called from :func:`execute` after ``reset_state()`` has restored the
    settings snapshot, so the value never leaks into the next run. It runs
    before the script does, so the script can still change the device itself.

    A change across the CPU/GPU line costs the next render one
    kernel-preparation pass, since ``ti.init`` discards the compiled kernels of
    the old arch. That is still far less than the cold start the refusal used
    to force, and a daemon serving one device repeatedly pays it once.

    An unusable device raises here exactly as it would at import in a cold
    process; the run reports it and fails rather than rendering somewhere else.
    """
    from algan.settings._startup import coerce_device

    device = coerce_device(
        env_str("ALGAN_RENDER_DEVICE", "auto"), "ALGAN_RENDER_DEVICE"
    )
    if device != SETTINGS.computing.render_device:
        _say(f"adopting this script's render device: {device}")
        SETTINGS.computing.set(render_device=device)


@contextlib.contextmanager
def _run_context(job):
    """Make one run look as much like its own process as a warm one can.

    Reproduced: stdout and stderr at the descriptor level (so subprocess and
    C-level output reach the client), their tty-ness, and the caller's full
    environment. Not reproduced, deliberately: ``stdin``, which is connected to
    ``os.devnull`` because the daemon's own stdin is its re-render trigger.

    ``job`` is ``None`` for a locally triggered re-run, where there is no
    client to stream to and no environment to adopt; stdin is still isolated,
    so a script cannot eat the trigger.
    """
    saved_stdin = sys.stdin
    saved_environ = dict(os.environ)
    saved_out, saved_err = sys.stdout, sys.stderr
    pumps = []
    saved_fds = []
    handles = []
    run_streams = []
    try:
        with contextlib.suppress(OSError, ValueError):
            # Deliberately not a context manager: it has to outlive this
            # statement and is closed in the finally below. noqa: SIM115.
            sys.stdin = open(os.devnull)  # noqa: SIM115
        if job is not None:
            if job.env is not None:
                os.environ.clear()
                os.environ.update(job.env)
                # Must survive the swap: it is what stops a python subprocess
                # started by the script from handing *itself* to this daemon
                # and deadlocking behind the run that spawned it.
                os.environ["ALGAN_DAEMON_CHILD"] = "1"
            for stream, fd, kind, isatty in (
                (saved_out, 1, _dc.FRAME_STDOUT, job.isatty_out),
                (saved_err, 2, _dc.FRAME_STDERR, job.isatty_err),
            ):
                # Anything already buffered belongs to the daemon, not this run.
                with contextlib.suppress(Exception):
                    stream.flush()
                pump = _Pump(job, kind, _CONSOLE)
                pumps.append(pump)
                saved_fds.append((fd, os.dup(fd)))
                os.dup2(pump.write_fd, fd)
                handles.append(_swap_std_handle(fd))
                run_streams.append(_RunStream(fd, isatty))
            sys.stdout, sys.stderr = run_streams
        yield
    finally:
        sys.stdout, sys.stderr = saved_out, saved_err
        # Close the run's streams while their descriptors still point at the
        # pipes, or their tail would flush onto the daemon's own console.
        for stream in run_streams:
            with contextlib.suppress(Exception):
                stream.close()
        for token in handles:
            _restore_std_handle(token)
        # Restore the descriptor before closing our copy of the write end: the
        # pump only sees EOF once every duplicate of it is gone.
        for fd, saved in saved_fds:
            with contextlib.suppress(OSError):
                os.dup2(saved, fd)
                os.close(saved)
        for pump in pumps:
            pump.close()
        with contextlib.suppress(Exception):
            sys.stdin.close()
        sys.stdin = saved_stdin
        os.environ.clear()
        os.environ.update(saved_environ)


def _refuse(stream, reason):
    with contextlib.suppress(OSError):
        _dc.write_frame(stream, _dc.FRAME_REFUSE, reason)
        stream.flush()


def _is_plumbing_frame(filename):
    """Whether a traceback frame belongs to the machinery, not the script."""
    if filename in ("<frozen runpy>", "<string>"):
        return True
    base = os.path.basename(filename)
    return base == "runpy.py" or os.path.abspath(filename) == os.path.abspath(__file__)


def strip_plumbing_frames(tb):
    """Drop the daemon's and ``runpy``'s frames from the top of ``tb``.

    A script that raises under the daemon showed ``algan/daemon.py ... in
    execute`` and two ``runpy`` frames above its own first line -- frames a
    plain ``python scene.py`` would never print and that say nothing about the
    error. Only the *leading* run of them is dropped, so a script that itself
    calls ``runpy`` still shows that call.

    Returns ``tb`` unchanged if every frame is plumbing: an empty traceback
    would be worse than an honest one.
    """
    stripped = tb
    while stripped is not None and _is_plumbing_frame(
        stripped.tb_frame.f_code.co_filename
    ):
        stripped = stripped.tb_next
    return tb if stripped is None else stripped


def _print_script_traceback(exc):
    """Report a user script's exception as its own process would have."""
    traceback.print_exception(type(exc), exc, strip_plumbing_frames(exc.__traceback__))


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
        # ``verb [token]``. Every verb needs the token: this socket executes
        # arbitrary paths and stops the process, and a bare ``quit`` from any
        # local process used to be enough to stop somebody else's daemon.
        # ``run`` carries its token inside the JSON request that follows.
        verb, _, token = line.partition(" ")
        command = verb.lower()
        token = token.strip()
        if command == "run":
            self._handle_run()
            return
        if command == "cancel":
            self._handle_cancel(token)
            return
        if command not in ("render", "ping", "quit"):
            self.wfile.write(b"err: expected run | cancel | render | ping | quit\n")
            return
        if not self._token_ok(token):
            return
        if command == "render":
            self.server.events.put(("render", "socket"))
            self.wfile.write(b"ok\n")
        elif command == "quit":
            self.server.events.put(("quit", "socket"))
            self.wfile.write(b"ok\n")
        else:
            self.wfile.write(b"pong\n")

    def _token_ok(self, token):
        """Whether ``token`` is this daemon's. Answers the caller when not."""
        if secrets.compare_digest(str(token), self.server.state.token):
            return True
        _say("rejected a trigger with a bad token")
        self.wfile.write(
            b"err: bad token -- pass the one from the daemon state file "
            b"(algan daemon render | ping | quit do this for you)\n"
        )
        return False

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
        # Last of the handshake checks, and it must stay in the handshake: by
        # the time ``execute`` runs, ``do_job`` has already sent FRAME_START
        # and the client can no longer fall back safely. Refusing here means
        # the run happens in a fresh process that loads the edited source.
        changed = _SourceDigest.capture().changed_since(self.server.sources)
        if changed:
            _say(f"refusing a run: algan sources changed ({len(changed)} file(s))")
            _refuse(self.wfile, _stale_message(changed))
            self.server.events.put(("quit", "algan sources changed"))
            return
        # Settings this daemon baked in when it imported algan. Unlike a
        # startup-variable mismatch, one here is invisible in the output --
        # the script simply renders with the daemon's toggles instead of its
        # own -- which is why it is refused rather than warned about. It comes
        # after the staleness gate so that a stale daemon still shuts down for
        # the next client instead of merely refusing this one.
        import_mismatch = _dc.describe_import_env_mismatch(
            request.get("env_full") or {}, _IMPORT_ENVIRON
        )
        if import_mismatch is not None:
            _say("refused a run: " + import_mismatch.splitlines()[0])
            _refuse(self.wfile, import_mismatch)
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


class _TriggerServer(socketserver.ThreadingTCPServer):
    """The trigger socket, with one deviation from the default.

    ``allow_reuse_address`` (SO_REUSEADDR) is set on POSIX so a daemon can bind
    immediately after its predecessor exited. Without it the port sits in
    TIME_WAIT for a minute or so, and since a daemon now shuts down every time
    algan's sources change, the replacement started by the next run would fail
    to bind and exit -- leaving no daemon at all for the rest of that minute,
    exactly during a burst of library edits. The flag still does not permit two
    live listeners on one port, so a spawn race keeps its loser.

    Not set on Windows, where SO_REUSEADDR means something else: it lets two
    sockets bind the same address *simultaneously*, which would let two daemons
    each believe they were serving.
    """

    allow_reuse_address = sys.platform != "win32"


def _bind_trigger_socket(port, allow_fallback):
    """Bind the trigger socket, falling back to an ephemeral port.

    The preferred port (46711, or ``ALGAN_DAEMON_PORT``) is a convenience, not
    an address anyone needs: clients discover the daemon through the state
    file, which carries whatever port was actually bound. It can be held by
    another user, another ``ALGAN_HOME``, or a daemon of ours that has not let
    go of it yet -- and a daemon that exits over that leaves no state file, so
    every later run spawns another one that fails the same way, for ever.
    Binding port 0 instead costs nothing and ends that loop.

    An explicit ``--port`` is not a preference but an instruction, so it does
    not fall back: it fails loudly and the daemon runs without a socket.
    """
    try:
        return _TriggerServer(("127.0.0.1", port), _TriggerHandler)
    except OSError as first:
        if not allow_fallback:
            raise
        _say(f"port {port} is unavailable ({first}); binding an ephemeral port")
    return _TriggerServer(("127.0.0.1", 0), _TriggerHandler)


def _start_socket(events, port, state, busy, sources, allow_fallback=True):
    try:
        server = _bind_trigger_socket(port, allow_fallback)
    except OSError as e:
        _say(
            f"trigger socket unavailable on 127.0.0.1:{port} ({e}); "
            "stdin trigger still works."
        )
        return None
    port = server.server_address[1]
    state.set_port(port)
    server.daemon_threads = True
    server.events = events
    server.state = state
    server.busy = busy
    server.sources = sources
    threading.Thread(
        target=server.serve_forever, daemon=True, name="algan-daemon-socket"
    ).start()
    _say(f"trigger socket on 127.0.0.1:{port} -- poke with:")
    _say("  algan daemon render   (also: algan daemon ping | quit)")
    return server


def _start_stdin(events):
    def loop():
        # Bind the real stdin once, rather than calling input(), which
        # re-reads ``sys.stdin`` on every call: a run replaces that with
        # os.devnull (_run_context), and this loop would then read an instant
        # EOF and quit the daemon in the middle of serving somebody.
        stream = sys.stdin
        interactive = False
        with contextlib.suppress(Exception):
            interactive = stream.isatty()
        got_line = False
        while True:
            try:
                line = stream.readline()
                if not line:
                    raise EOFError
            except (EOFError, OSError, ValueError):
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
    port = default_port()
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help=f"trigger socket port. Without this the daemon prefers {port} and "
        "falls back to an ephemeral one when it is taken; given explicitly, "
        "the port is an instruction and binding it is allowed to fail",
    )
    parser.add_argument(
        "--no-serve", action="store_true", help="do not open the trigger socket"
    )
    parser.add_argument(
        "--no-initial-render",
        action="store_true",
        help="wait for a trigger instead of rendering once at startup",
    )
    parser.add_argument(
        "--idle-timeout",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="exit after this long with nothing to do (0 = never, the default "
        "for a hand-launched daemon; auto-started ones pass a real value)",
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
    sources = _SourceDigest.capture()
    wanted_port = port if args.port is None else args.port
    state = _StateFile(wanted_port)
    server = (
        None
        if args.no_serve
        else _start_socket(
            events,
            wanted_port,
            state,
            busy,
            sources,
            allow_fallback=args.port is None,
        )
    )
    if server is None and script is None:
        # Nothing can ever reach this process: no socket to serve clients, and
        # no script for the stdin trigger to re-run. Under auto-start that is
        # what a lost spawn race looks like, and an immortal idle process
        # holding a CUDA context is the last thing it should leave behind.
        _say("no trigger socket and no SCRIPT -- nothing to serve; exiting")
        return 1
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

    # Whether the daemon's state is fresh enough to run into. Set False for
    # the runtime of a run and True again by ``release_after_run``; a release
    # that failed leaves it False so the next run resets before it starts
    # rather than inheriting whatever the failure left behind.
    clean = {"state": True}

    def reset_state():
        evicted = sorted({n for d in script_dirs for n in _user_modules(d)})
        for name in evicted:
            sys.modules.pop(name, None)
        if evicted:
            _say("reloading: " + ", ".join(evicted))
        snapshot.restore()
        SceneManager.reset()

    def release_after_run():
        """Reset for the next run *now*, and give its memory back.

        The reset used to happen at the start of the following run, which left
        the finished run's scene -- and every tensor it holds -- resident for
        however long the daemon sat idle. Doing it on the way out costs the
        same and means an idle daemon holds nothing but the warm process it
        exists to be.
        """
        try:
            reset_state()
            _release_run_memory()
            clean["state"] = True
        except BaseException:  # noqa: BLE001 -- tidying must not kill the daemon
            traceback.print_exc()
            clean["state"] = False

    def execute(path, script_args, cwd, reason):
        """Run one script to completion. Returns its exit code."""
        nonlocal run_count
        run_count += 1
        if not clean["state"]:
            reset_state()
        clean["state"] = False
        # Point the output defaults at *this* script. They are resolved once,
        # when the settings are constructed -- which in a daemon is its own
        # startup, with no user script in sight -- so without this every
        # client's video would land in the daemon's directory under the name
        # "algan_render_output" instead of beside the script as its stem.
        # Applied after reset_state()'s restore, and before the script runs so
        # it can still override them itself.
        SETTINGS.paths.set(
            output_root=output_root_for(path),
            output_filename=output_filename_for(path),
        )
        _adopt_render_device()
        # Same argument as the render device: the client's ALGAN_LOG_LEVEL /
        # ALGAN_PROGRESS were read into a live logger at *this process's*
        # import, so re-read them here and the run reports at the verbosity the
        # script asked for instead of the daemon's. Neither is part of
        # SETTINGS.snapshot(), so reset_state() cannot restore them; that is
        # why the call resets an unset variable to its default rather than
        # leaving it, and why one client's DEBUG does not follow the next.
        apply_environment_logging()
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
        except BaseException as exc:  # noqa: BLE001 -- must not kill the daemon
            code = 1
            _print_script_traceback(exc)
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

    def stale_quit():
        """Shut down if algan's sources no longer match what we imported.

        The socket handshake makes this call for client runs; this is the
        locally triggered path (Enter, the socket's ``render``, ``--watch``),
        where there is no client to refuse and the daemon simply stands down.
        """
        changed = _SourceDigest.capture().changed_since(sources)
        if not changed:
            return False
        _say(_stale_message(changed))
        events.put(("quit", "algan sources changed"))
        return True

    def do_local(reason):
        target = last["script"]
        if target is None:
            _say("no script to re-run yet -- launch one and it will land here")
            return
        if stale_quit():
            return
        busy.set()
        try:
            with _run_context(None):
                execute(target, last["args"], last["cwd"], reason)
        finally:
            busy.clear()
            release_after_run()

    def do_job(job):
        busy.set()
        job.send(_dc.FRAME_START)
        last.update(script=job.script, args=job.argv, cwd=job.cwd)
        try:
            with _run_context(job):
                code = execute(job.script, job.argv, job.cwd, "client")
        except BaseException:
            code = 1
            traceback.print_exc()
        finally:
            busy.clear()
            # Release the client first: the tidy-up below is the daemon's own
            # housekeeping and the script has nothing left to wait for.
            job.finish(code)
            release_after_run()

    if script is not None and not args.no_initial_render:
        do_local("startup")
    elif watcher is not None:
        watcher.set_paths({script})
    _say("ready -- Enter = re-run the last script, q = quit")

    idle_timeout = max(0.0, args.idle_timeout)
    idle_since = time.monotonic()
    try:
        while True:
            try:
                event = events.get(timeout=0.5)
            except queue.Empty:
                # Only an auto-started daemon sets a timeout; one launched by
                # hand stays until it is told to go. ``busy`` cannot be set
                # here (runs occupy this thread), but check it anyway so the
                # rule survives anyone moving execution off it.
                if (
                    idle_timeout
                    and not busy.is_set()
                    and time.monotonic() - idle_since > idle_timeout
                ):
                    _say(f"exiting after {idle_timeout:.0f}s idle")
                    break
                continue
            idle_since = time.monotonic()
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
            idle_since = time.monotonic()
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
