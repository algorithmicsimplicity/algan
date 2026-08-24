=================
The Render Daemon
=================

When you first run an Algan program, there are some overhead costs associated
with starting up the renderer. Firstly, Algan's dependencies (mainly Torch and
Taichi) must be imported and initialized. Secondly, Taichi must compile all of
the rendering kernels. All up, this takes about twenty seconds, give or take.

To reduce the start-up time and make iterating on a scene more convenient, Algan
employs a **render daemon**: a copy of Algan kept alive in another process, with
its kernels already compiled. When you import Algan in a script, the client half
of the daemon looks for one that is already running and, if it finds one, hands
the script over to it. If none is found, Algan launches one in the background.
Either way, the start-up cost is paid once and every later run begins rendering
almost immediately.

Nothing is required of you to get this. A plain ``python scene.py`` uses it.

How the handoff works
=====================

``import algan`` reaches the client before any heavy import happens:

#. A running daemon publishes a state file at ``~/.algan/daemon.json`` (or
   ``$ALGAN_HOME/daemon.json``). Its absence means "no daemon".
#. If the file is there, the client sends the daemon the working directory, the
   script path, ``sys.argv`` and the environment, streams the run's stdout and
   stderr back to its own, and exits with the daemon's exit code. The client
   itself never imports Torch or Taichi, so the round trip costs Python start-up
   plus the render.
#. If no daemon is running, the client starts one in the background, waits for it
   to publish its state file, and then hands off as above. That first run costs
   what it always did; later ones start warm.

A run on the daemon is meant to be indistinguishable from a run in its own
process. ``sys.argv``, the working directory, the environment, stdout and stderr
(at the descriptor level, so ``ffmpeg`` and other subprocesses reach you) and the
tty-ness of both streams are all reproduced. Two things deliberately are not:
``stdin`` is connected to the null device, because the daemon's own stdin is its
re-render trigger, and ``atexit`` handlers do not run, because a warm process
never shuts down.

**Concurrent scripts are queued and run one at a time**, in arrival order. A
waiting client is told its position. On Windows this is what you want anyway: two
live render processes fight over the output file.

**Anything that goes wrong falls back to a normal in-process run** -- a refused
handshake, a daemon that is not listening, a spawn that fails or is slow to come
up. The one unrecoverable case is a daemon that dies *after* the script has
started executing: the script's side effects have already happened, so re-running
it locally could duplicate them. That reports an error and exits non-zero.

Launching one by hand
=====================

You do not have to, but a hand-launched daemon gives you a terminal you can watch
and an Enter-to-re-render loop:

.. code-block:: bash

    python -m algan.daemon                      # a general daemon
    python -m algan.daemon scene.py             # ... that also owns one script
    python -m algan.daemon scene.py --watch     # ... and re-renders on save

In the daemon's own terminal, **Enter** re-renders the last script and ``q``
quits. That is the primary hand-launched workflow: edit in your editor, save,
switch to the daemon, press Enter.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Effect
   * - ``SCRIPT [args...]``
     - The script to re-execute. Omit it for a general daemon that serves
       whatever scripts are launched against it. Script arguments go after
       ``--``.
   * - ``--watch``
     - Also re-render when the script or its sibling helper modules change on
       disk. Needs a ``SCRIPT``.
   * - ``--port PORT``
     - Trigger-socket port. Default 46711.
   * - ``--no-serve``
     - Do not open the trigger socket. Needs a ``SCRIPT``.
   * - ``--no-initial-render``
     - Wait for a trigger instead of rendering once at startup.
   * - ``--idle-timeout SECONDS``
     - Exit after this long with nothing to do. ``0`` (never) is the default for
       a hand-launched daemon; an auto-started one is given a real value.

Triggering a re-render
======================

A render is never interrupted, and triggers arriving mid-render coalesce into at
most one queued re-run. There are three ways in:

* **Enter** in the daemon's terminal.
* **The trigger socket** on ``127.0.0.1``, port 46711 by default. It accepts the
  line commands ``render``, ``ping`` and ``quit``. Bind an editor key to the
  standard-library one-liner -- deliberately not ``python -m algan.daemon``,
  which would import the whole library just to poke a socket:

  .. code-block:: bash

      python -c "import socket;s=socket.create_connection(('127.0.0.1',46711),2);s.sendall(b'render\n');print(s.recv(16).decode().strip())"

* **``--watch``**, which polls the script and its sibling modules for changes.

The socket also accepts ``cancel`` (with the state file's token), which raises
``KeyboardInterrupt`` inside the running script -- the same thing Ctrl-C would
have done had the script owned the terminal -- and ``run``, which is the handoff
protocol the client uses and not something to drive by hand.

Stopping it
===========

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Way
     - When to use it
   * - ``q`` then Enter, or Ctrl+C
     - A daemon you launched in a terminal.
   * - ``quit`` on the trigger socket
     - Anything, including a background daemon. Same one-liner as above with
       ``b'quit\n'`` in place of ``b'render\n'``.
   * - Kill the process
     - The ``pid`` is in ``~/.algan/daemon.json``.
   * - Wait
     - An auto-started daemon exits by itself after two hours idle
       (``ALGAN_DAEMON_IDLE_TIMEOUT``).
   * - Edit Algan's own source
     - The daemon shuts itself down (see below).

Stopping it is also how you clear baked-in configuration: a daemon launched by a
script that set a non-default renderer toggle serves only scripts that set the
same one, so stop it if you want one baked with the defaults.

Where its output goes
=====================

An **auto-started** daemon has no terminal, so its console output is appended to
``~/.algan/daemon.log``. That is the first place to look when a run behaves
oddly, or when the client warns that the background daemon exited early. The log
is trimmed to ``ALGAN_DAEMON_LOG_MAX_BYTES`` (4 MB by default), with the previous
contents kept alongside as ``daemon.log.old``.

A **hand-launched** daemon prints to its own terminal instead, and does not write
the log at all.

Both live under ``$ALGAN_HOME``, which defaults to ``~/.algan``:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - File
     - Contents
   * - ``daemon.json``
     - The state file: port, pid, an access token, and the startup environment
       the daemon baked in. Its absence means no daemon is running.
   * - ``daemon.log``
     - Console output of auto-started daemons.

What survives a run, and what does not
======================================

When a run ends the daemon restores a clean slate -- on the way out rather than at
the start of the next run, so what it holds while idle is the warm process and
nothing else:

* The Scene, camera, lights and timeline are reset.
* Every public :doc:`settings <settings>` section is restored to its import-time
  value, so one run cannot leak configuration into the next. Private adaptive
  renderer state is kept deliberately.
* Helper modules imported from the script's directory tree are evicted from
  ``sys.modules``, so the next run picks up their edits. Modules imported from
  anywhere else are **not** reloaded.
* The render's GPU memory goes back to the driver: one ``gc.collect()`` and one
  ``torch.cuda.empty_cache()``. On a 4 GB card an idle daemon that used to hold
  1.6 GB after a 90-frame render now holds about 0.1 GB, for ~0.15 s and no
  measurable change to the next render. ``ALGAN_DAEMON_RELEASE_MEMORY=0`` keeps
  the memory cached instead.

.. important::

    **Edits to Algan itself are handled, not merely warned about.** The daemon
    fingerprints every Algan source file at startup and re-checks it at every run
    launch; if anything changed it refuses the run and shuts down, so the script
    executes in a fresh process that loads the edited code, and a new daemon
    starts on the next run. This costs a cold start -- which is what editing the
    library has always cost -- but it can no longer render with stale modules or
    compile a mixed-version kernel out of a half-edited ``*_taichi.py``.

    An edit that lands *during* a run is not caught, exactly as it is not caught
    for a plain ``python scene.py``.

When the daemon refuses a run
=============================

Two classes of setting cannot be adopted from a client, so a script that wants
different values for them is refused and runs cold instead. That is deliberate:
being served would silently render the wrong thing.

* **Startup-only settings** -- ``ALGAN_RENDER_DEVICE``,
  ``ALGAN_ANIMATION_DEVICE`` and friends are read while Torch and Taichi
  initialize, which happened when the *daemon* started.
* **Import-time settings** -- most renderer toggles become module-level defaults
  during ``import algan``, which in a daemon happened at its launch. A script
  that sets one *before* its own ``import algan`` -- which is how every A/B
  script in ``benchmarks/`` selects an arm -- would otherwise be served by a
  process that never saw it.

Variables read *live* are unaffected: the client's environment is swapped in for
the run, so flipping one between two renders inside a script works warm exactly
as it does cold. See :doc:`settings` for the two categories.

Turning it off
==============

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Variable
     - Effect
   * - ``ALGAN_USE_DAEMON=0``
     - Disable the handoff entirely, even when a daemon is running.
   * - ``ALGAN_AUTO_DAEMON=0``
     - Keep using a daemon that is already running, but never start a new one.

Benchmarks in this repository set ``ALGAN_USE_DAEMON=0``, because a warm process
also carries the previous run's adaptive renderer state -- which is exactly what
you do not want when measuring.

Environment variables
=====================

.. list-table::
   :header-rows: 1
   :widths: 40 18 42

   * - Variable
     - Default
     - Meaning
   * - ``ALGAN_USE_DAEMON``
     - ``1``
     - Use a daemon at all.
   * - ``ALGAN_AUTO_DAEMON``
     - ``1``
     - Start one when none is running.
   * - ``ALGAN_DAEMON_PORT``
     - ``46711``
     - Trigger-socket port.
   * - ``ALGAN_DAEMON_TIMEOUT``
     - ``2.0``
     - Seconds the client waits to connect before falling back.
   * - ``ALGAN_DAEMON_START_TIMEOUT``
     - ``60.0``
     - Seconds to wait for an auto-started daemon to publish its state file. A
       daemon that is merely slow serves the *next* run.
   * - ``ALGAN_DAEMON_IDLE_TIMEOUT``
     - ``7200``
     - Seconds of idleness after which an auto-started daemon exits.
   * - ``ALGAN_DAEMON_LOG_MAX_BYTES``
     - ``4194304``
     - Size at which ``daemon.log`` is rotated to ``daemon.log.old``.
   * - ``ALGAN_DAEMON_RELEASE_MEMORY``
     - ``1``
     - Return the render's GPU memory to the driver when a run ends.
   * - ``ALGAN_HOME``
     - ``~/.algan``
     - Where ``daemon.json`` and ``daemon.log`` live.

``ALGAN_DAEMON_CHILD`` is set by the daemon around its own execution of a script,
and is what stops the handoff from recursing. Do not set it yourself.

.. warning::

    **Anything that can reach ``127.0.0.1`` can ask the daemon to execute a
    path.** Requests must carry the token from the state file, which lives in
    your home directory (mode 0600 where the platform honours it), but do not
    forward the port off-host.

See Also
========

* :mod:`algan.daemon` -- the daemon itself, and the reference for everything on
  this page.
* :doc:`performance_and_quality` -- the render costs the daemon does *not*
  remove.
* :doc:`settings` -- the startup-only and initialization-only settings a warm
  daemon cannot adopt.
