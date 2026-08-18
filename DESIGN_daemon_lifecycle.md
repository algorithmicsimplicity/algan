# Algan — Daemon Lifecycle: Design Document

Status: DESIGN ONLY. Nothing here is implemented.

Two changes to [`algan/daemon.py`](algan/daemon.py) and
[`algan/daemon_client.py`](algan/daemon_client.py):

1. **A staleness gate.** A daemon whose loaded algan sources no longer match
   what is on disk refuses the run and shuts down, instead of rendering with
   stale code. This is a **correctness** change and the reason the document
   exists.
2. **Auto-start.** When no daemon is running, an ordinary `python scene.py`
   leaves one warmed behind it, so the *next* run starts in ~1s instead of
   paying ~20s of Taichi kernel preparation. This is a **convenience** change
   for users who are not editing the library.

They compose: **auto-start + shutdown-on-stale = auto-restart**, with no
supervisor process, no `os.execv`, and no new lifecycle machinery. The daemon
dies; the next run brings up a fresh one that imports the edited source. Every
mechanism this needs — refusal, client fallback, queued-client release,
quit-after-current-run — already exists and is already tested.

**Explicit non-goal: making library development fast.** A restart is a cold
start by definition. Editing a `*_taichi.py` invalidates the `.tic` key (source
paths, line numbers and inlined `ti.func` bodies all participate in it — see
§2 of [`DESIGN_frontend_trace_cache.md`](DESIGN_frontend_trace_cache.md)), so a
kernel edit costs a real cold compile. Editing non-kernel algan Python costs
~20s of kernel prep plus imports. That is what those edits cost today without a
daemon, and this design does not improve it. It only guarantees you never
*silently* get the old code instead.

---

## 1. What is wrong today

`_AlganSourceGuard` ([`daemon.py:147`](algan/daemon.py)) walks `_ALGAN_DIR` at
startup and records `st_mtime_ns` for the 143 `.py` files outside
`external_libraries/` and `__pycache__/`. `warn_if_changed()` runs at the top of
every `execute()` ([`daemon.py:649`](algan/daemon.py)), names up to five changed
files, and adds a louder warning when any of them ends in `_taichi.py`.

So detection exists. What it does with the detection is print

> WARNING: imported algan modules are stale -- restart the daemon to pick these up.

and then **run the script anyway, against the stale modules**. `reset_state()`
([`daemon.py:636`](algan/daemon.py)) cannot help: `_user_modules()` explicitly
excludes anything under `_ALGAN_DIR` ([`daemon.py:347`](algan/daemon.py)), by
design. Correctness rests entirely on a human reading a warning and acting on
it, in a terminal they may not be looking at.

Three defects in the detector itself, which matter once it becomes a gate
rather than a hint:

* **New files are invisible.** The mtime map is built once at startup, so a
  module added afterwards is never checked.
* **The baseline is never updated**, so after one edit it re-warns identically
  forever, which trains the reader to ignore it.
* **mtime is the wrong signal.** `git checkout`, `git stash` and `git rebase`
  rewrite mtimes wholesale without changing content. An mtime-based gate would
  force a pointless ~20s restart on every branch switch — including switching
  away and back.

---

## 2. Staleness detection: hash the content

Replace the mtime map with a SHA-256 over the sorted `(relative path,
sha256(bytes))` list of **every** `.py` file under `_ALGAN_DIR`. One digest for
the whole tree; equality is the whole test.

**Cost.** 143 files / 3.3 MB excluding `external_libraries/`, plus 164 files /
2.0 MB if it is included. 307 files and 5.3 MB hashes in roughly 10 ms with a
warm page cache. This runs once per run-launch, not on a poll loop, so it is
free at the frequency that matters.

**Include `external_libraries/`.** The current guard skips it, presumably for
walk speed. It is vendored and read-only *by policy* (`CLAUDE.md`), but it is
imported into the same interpreter, so an edit there is exactly as stale as any
other. At 10 ms for the full tree there is no reason to trade correctness for
it. Keep excluding `__pycache__/`; `.pyc` files are derived, and the source is
what is being hashed.

**Why hashing beats mtime, restated as the design argument:** the gate must be
conservative in one direction only. Missing a real edit renders wrong output.
Reporting a change that did not happen costs a cold start. Content hashing is
exact in both directions, which means it never misses an edit *and* never fires
on a branch switch that restores identical content. mtime is wrong in both
directions at once.

**Hash all files, not just imported ones.** Scoping to `sys.modules` entries
under `_ALGAN_DIR` would be cheaper and more precise, but it cannot see a
*newly created* module that a future run would import. Over-restarting is safe;
under-restarting is the failure this document exists to prevent.

---

## 3. Where the check goes: the handshake, not `execute()`

This is the load-bearing detail. `do_job` sends `FRAME_START`
([`daemon.py:709`](algan/daemon.py)) **before** calling `execute`
([`daemon.py:713`](algan/daemon.py)), and `FRAME_START` is precisely the point
after which the client considers fallback unsafe
([`daemon_client.py:264`](algan/daemon_client.py), and the `DaemonRunFailed`
docstring). A check inside `execute` — where `warn_if_changed` lives now — is
too late to refuse cleanly.

So the staleness check moves into `_TriggerHandler._handle_run`, alongside the
token, protocol, env-mismatch and script-exists checks
([`daemon.py:387-410`](algan/daemon.py)). That block is exactly the right home:
it runs before the job is queued, and it already has the refusal path this
needs.

```
_handle_run
  token ok?  protocol ok?  startup-env matches?  script exists?
  └─ algan sources unchanged since launch?   ← new, last of the five
       no ──► _refuse("algan sources changed …")     [FRAME_REFUSE]
              events.put(("quit", "stale algan sources"))
              return                                  (job never queued)
```

What the client does with that refusal is already built and already tested:
`FRAME_REFUSE` → `DaemonUnavailable` ([`daemon_client.py:262`](algan/daemon_client.py))
→ `maybe_handoff` warns and returns ([`daemon_client.py:299`](algan/daemon_client.py))
→ the import proceeds and the script runs in-process, in a fresh interpreter
that loads the edited source. `test_refusal_before_start_is_recoverable`
(`tests/unit_tests/test_daemon_client.py:243`) already covers that path.

The refusal text should say what happened and what will happen, since it
surfaces to the user as `[algan] not using the algan daemon: <reason>`:

> algan sources changed since this daemon started (algan/rendering/…/raster_taichi.py
> and 2 others); the daemon is shutting down and this run will execute in a
> fresh process.

### 3.1 The local-trigger path

Enter on the daemon's stdin, `render` on the socket, and `--watch` all reach
`do_local` ([`daemon.py:696`](algan/daemon.py)), where there is no client to
refuse. There the daemon prints the same message and quits — it is the
developer's own terminal, and the next `python scene.py` starts a fresh one.

`warn_if_changed`'s current call site inside `execute` is then dead and should
be deleted rather than left as a second, weaker signal.

---

## 4. Shutdown ordering

`("quit", …)` goes on the existing event queue rather than exiting inline, and
everything else follows from machinery that is already there:

* **A render in progress is never interrupted.** `do_job`/`do_local` run on the
  main thread, which is not reading the queue during a run. The quit is
  processed when the current run returns.
* **Quit wins the coalescing.** `_drain` ([`daemon.py:557`](algan/daemon.py))
  already promotes a quit over queued renders.
* **Queued clients are released, not dropped.** `_drop_pending`
  ([`daemon.py:761`](algan/daemon.py)) sends each queued job a `FRAME_REFUSE`,
  so they fall back to cold in-process runs — which is the correct outcome,
  since they too would have run against stale code.
* **The state file is removed in the existing `finally`**
  ([`daemon.py:748`](algan/daemon.py)), and `_StateFile.remove` is pid-guarded
  ([`daemon.py:217`](algan/daemon.py)) so a daemon started in the meantime is
  not de-registered.

Deliberately *not* removing the state file early, at the moment staleness is
detected: a client arriving during the dying daemon's last render would then
find no daemon and auto-start a second one, putting two heavy warm-ups on the
GPU at once. Leaving the file in place means such a client connects, gets
refused for staleness, and runs cold — correct, and sequential.

### 4.1 The one case this does not cover

A source edit that lands *while a run is executing* is not caught: the run was
validated at handshake time and finishes on the code it started with. That is
the same semantics as editing a file during a plain `python scene.py`, and the
`*_taichi.py` JIT hazard (`CLAUDE.md`, `AGENTS_DETAILED.md:357`) is unchanged
for that window. Document it; do not try to fix it. Fixing it would mean
interrupting a render, which is worse than the problem.

---

## 5. Auto-start

`maybe_handoff` currently returns as soon as `read_state()` finds no state file
([`daemon_client.py:290`](algan/daemon_client.py)). The question is what to do
instead, and the answer is: **nothing, until the current run has finished.**

Three options were considered:

| | Run 1 | Contention | Run 1 output |
|---|---|---|---|
| A. Spawn at import, run in-process | cold | **two heavy warm-ups at once** | normal |
| B. Spawn at import, wait, hand off | cold | none | via the daemon protocol |
| C. Spawn after the run finishes | cold | none | normal |

**Take C.** A duplicates ~20-27s of Taichi and Torch initialization across two
processes simultaneously, which on a modest GPU risks an OOM during what is
supposed to be a transparent convenience. B avoids that and is tempting, but it
routes the very first run's output through `_ClientStream`, which carries a
documented asymmetry ([`daemon.py:269`](algan/daemon.py)): Python-level output
reaches the client, C-level and subprocess output — ffmpeg's, notably — does
not. Making a user's *first ever* run behave differently from plain `python
scene.py` is a bad trade for a few seconds.

C keeps run 1 byte-for-byte the run it is today, spawns the daemon afterwards,
and has it warm by the time a human has looked at their video and edited
something. Runs 2+ are warm.

### 5.1 Trigger

Spawn from an `atexit` hook, gated on all of:

* `should_try()` is true — the existing conservative check
  ([`daemon_client.py:184`](algan/daemon_client.py)) that this is a plain
  `python foo.py` and not a REPL, notebook, `python -c`, or the daemon's own
  child. Note it already returns False under pytest
  ([`daemon_client.py:200`](algan/daemon_client.py)), so **the test suite never
  auto-starts a daemon** — no new test isolation is required.
* `ALGAN_AUTO_DAEMON` is on (default true).
* `read_state()` still returns None — re-checked at exit, because another
  process may have started one during this run.
* **A render actually happened.** A module-level flag set by
  `save_video`/`save_frame`. A script that imports algan and renders nothing
  should not leave a GPU-resident process behind.

`atexit` does not run on `os._exit` or a hard crash, which is the right
behavior: a script that died is not evidence anyone wants a daemon.

### 5.2 Spawn mechanics

`subprocess.Popen([sys.executable, "-m", "algan.daemon", "--idle-timeout", N])`,
fully detached:

* stdin `DEVNULL`; stdout/stderr to `$ALGAN_HOME/daemon.log` (append, truncated
  when it exceeds a cap). The daemon's `_say` chatter has to go *somewhere*, and
  a detached process has no terminal.
* POSIX: `start_new_session=True`. Windows: `DETACHED_PROCESS |
  CREATE_NEW_PROCESS_GROUP`.
* `cwd=algan_home()`, so the daemon never holds a project directory open —
  on Windows that would block deleting it. The daemon `os.chdir`s per run
  anyway ([`daemon.py:663`](algan/daemon.py)).
* Environment inherited as-is, so the startup-env values the daemon bakes in
  match the process that spawned it. `ALGAN_DAEMON_CHILD` is explicitly cleared
  in the child env (belt and braces — `should_try()` already refuses to reach
  here from inside a daemon run).

### 5.3 Telling the user

One line to stderr, once, at spawn:

```
[algan] warmed a background render daemon for next time (log: ~/.algan/daemon.log,
        exits after 30 min idle). Disable with ALGAN_AUTO_DAEMON=0.
```

A background process holding a CUDA context that the user did not ask for and
cannot see is not acceptable; this line is part of the feature, not decoration.

---

## 6. Two daemon-side fixes auto-start requires

**A general daemon that cannot open its socket must exit.** `_start_socket`
returns None when the port is taken ([`daemon.py:440`](algan/daemon.py)) and
`main` carries on. With a SCRIPT that is reasonable — stdin still triggers
re-renders. With no SCRIPT it produces a process with no trigger, no state file
(it is only written `if server is not None`,
[`daemon.py:622`](algan/daemon.py)) and nothing to do, which idles forever
holding VRAM. Today that needs a hand-typed command to happen; under auto-start
a spawn race produces it. Fix: if `server is None and script is None`, log and
return.

**An idle timeout.** `--idle-timeout SECONDS` (0 = never, the default for a
hand-launched daemon). The main loop already polls with
`events.get(timeout=0.5)` ([`daemon.py:730`](algan/daemon.py)), so this is a
counter reset on every event and every run, checked in the `queue.Empty` branch,
and suppressed while `busy` is set. Only auto-started daemons pass the flag: a
daemon someone launched deliberately stays until told to quit.

---

## 7. Environment variables

Three new names in `_RUNTIME_VARIABLES` in
[`algan/environment.py`](algan/environment.py) (alphabetically, beside the
existing `ALGAN_USE_DAEMON`), each read through an accessor at the point of use
per `CLAUDE.md`:

| Name | Default | Meaning |
|---|---|---|
| `ALGAN_AUTO_DAEMON` | `1` | Leave a warm daemon behind after a successful render. |
| `ALGAN_DAEMON_IDLE_TIMEOUT` | `1800` | Seconds before an auto-started daemon exits. |
| `ALGAN_DAEMON_STALE_CHECK` | `1` | Kill switch for the staleness gate. |

`ALGAN_DAEMON_STALE_CHECK=0` is documented as **unsafe** — it re-enables exactly
the silent-stale-render behavior this design removes. It exists only so that a
misfiring hash check (a generated file appearing under `algan/`, a filesystem
that cannot be read reliably) can be worked around without reverting, and its
warning should say so.

These are runtime variables, not startup ones: they do not participate in
`STARTUP_ENV` and a client whose value differs from the daemon's is not
refused.

---

## 8. Tests

In `tests/unit_tests/test_daemon_client.py` (client half, stdlib-only, no GPU)
and a new `tests/unit_tests/test_daemon_staleness.py`. All unmarked — these are
feature tests for the daemon, and per `tests/README.md` the `fast` marker is for
tests that a change *elsewhere* is liable to break.

Staleness digest:
* unchanged tree → identical digest across two computations;
* edited file content → different digest;
* **touched but identical content → identical digest** (the git-checkout case,
  and the reason for hashing rather than stat'ing);
* new file added → different digest;
* file deleted → different digest;
* `__pycache__/` churn → identical digest.

Gate and shutdown:
* stale sources at handshake → `FRAME_REFUSE` arrives and `FRAME_START` never
  does (extend `_FakeDaemon`, `tests/unit_tests/test_daemon_client.py:172`);
* the client raises `DaemonUnavailable`, not `DaemonRunFailed`, so
  `maybe_handoff` falls back rather than exiting non-zero;
* a stale handshake while `busy` is set queues a quit and does not interrupt the
  in-flight run;
* queued jobs at shutdown each receive a refusal (`_drop_pending`).

Auto-start:
* no spawn when `ALGAN_AUTO_DAEMON=0`, when `should_try()` is false, when a
  state file already exists, or when no render occurred;
* spawn is detached and does not inherit `ALGAN_DAEMON_CHILD`;
* a general daemon whose port is taken exits instead of idling;
* the idle timeout fires and removes the state file.

---

## 9. Documentation to update

The current guidance tells people to do by hand what this makes automatic, so
it changes rather than being extended:

* [`algan/daemon.py`](algan/daemon.py) module docstring — the "Limits: edits to
  algan itself require a daemon restart" paragraph.
* [`algan/daemon_client.py`](algan/daemon_client.py) module docstring — the
  auto-start path. Also **fix its startup figure**: it claims "~65 s of Taichi
  kernel preparation" where `daemon.py:4` says "~10 s" and
  `DESIGN_frontend_trace_cache.md` measures ~20s. Three numbers, three places;
  measure once and make them agree.
* `CLAUDE.md`, Taichi gotchas — "Restart the daemon after changing any Algan
  source" becomes "the daemon detects this and restarts itself".
* `AGENTS_DETAILED.md:357-358`, same two lines.
* `docs/source/contributing/development.rst` if it mentions the daemon.

---

## 10. Phases

| Phase | Deliverable |
|---|---|
| 1 | Content-hash digest replacing `_AlganSourceGuard`'s mtime map, with its tests. No behavior change yet — still warn-only. |
| 2 | The handshake gate (§3) and shutdown ordering (§4). This is the correctness fix and is independently shippable; without §5 the daemon simply has to be relaunched by hand, as today. |
| 3 | The two daemon-side fixes in §6 (socket-failure exit, idle timeout). |
| 4 | Auto-start (§5), off by default behind `ALGAN_AUTO_DAEMON` until it has been exercised on Windows and Linux. |
| 5 | Default on; docs in §9 updated. |

Phase 2 is the one that matters and does not depend on any of the others. If
auto-start turns out to be more trouble than it is worth, phases 1-3 still
leave the daemon safe to use during library development, which it is not today.

---

## 11. Hazards

| Hazard | Mitigation |
|---|---|
| Stale render (the whole point) | Content-hash gate at the handshake, before `FRAME_START` (§3) |
| Edit lands mid-run | Not caught; documented (§4.1). Interrupting a render is worse |
| `*_taichi.py` edited under a live JIT | Unchanged for the mid-run window; the gate prevents *starting* a run after such an edit |
| Two daemons warming at once | State file kept until exit (§4); spawn deferred to `atexit` (§5) |
| Orphan GPU-resident process | Idle timeout (§6), one-line notice at spawn (§5.3), `ALGAN_AUTO_DAEMON=0` |
| Daemon with no trigger and no work | Exit when the socket fails and there is no SCRIPT (§6) |
| Branch switches forcing pointless restarts | Content hashing, not mtime (§2) |
| Spawn race between two finishing scripts | Loser fails to bind the port and now exits (§6); the state file is written only by the winner |
| Test suite spawning daemons | `should_try()` already returns False under pytest (§5.1) |
