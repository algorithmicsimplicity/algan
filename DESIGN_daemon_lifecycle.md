# Algan — Daemon Lifecycle: Design Document

Status: **IMPLEMENTED** (all phases). Validated on Linux/CPU; the Windows and
CUDA half is outstanding — see `HANDOFF_daemon_lifecycle.md` for what remains
and how to run it. §13 records what implementation found that this plan did
not anticipate.

Two changes to [`algan/daemon.py`](algan/daemon.py) and
[`algan/daemon_client.py`](algan/daemon_client.py):

1. **A staleness gate.** A daemon whose loaded algan sources no longer match
   what is on disk refuses the run and shuts down, instead of rendering with
   stale code. This is a **correctness** change and the reason the document
   exists.
2. **Auto-start.** When no daemon is running, an ordinary `python scene.py`
   starts one, runs itself on it, and leaves it warm, so every later run starts
   in ~1s instead of paying ~20s of Taichi kernel preparation. This is a
   **convenience** change for users who are not editing the library, and it
   requires first closing the gap between "run on the daemon" and "run in your
   own process" (§5).

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

### 4.1 Mid-run edits: an equivalence, not a gap

A source edit that lands *while a run is executing* is not caught, and that is
correct rather than a residual hazard. Because the gate runs at **every** run
launch — client handshake and the local Enter/socket paths alike — the
following holds at the instant any run starts:

> daemon-loaded sources == disk == what a fresh interpreter would load

From that instant the two cases are identical. Already-imported modules are
frozen either way; a module imported lazily after the edit picks up new source
either way; and the Taichi JIT reads `*_taichi.py` at first launch of a kernel
variant either way, which is the mixed-version hazard `CLAUDE.md` and
`AGENTS_DETAILED.md:357` already warn about for plain `python scene.py`.

So the claim to make in the docs is not "the daemon has one remaining stale
window" but **"the daemon is exactly as safe as no daemon"**. Do not try to
close it further: the only way to do so is to interrupt a render, which is
worse than the problem.

---

## 5. Run parity: closing the gap between daemon and in-process runs

Auto-start means a user's **first ever** run executes on the daemon. That is
only acceptable if running on the daemon is indistinguishable from running in
your own process. Today it is not, in four ways. Three are bugs worth fixing
regardless of auto-start; the fourth is documented and left alone.

### 5.1 Output forwarding (the one that matters)

`_ClientStream` ([`daemon.py:269`](algan/daemon.py)) replaces `sys.stdout` and
`sys.stderr` at the *Python* level. Its `fileno()` deliberately reports the
daemon console's descriptor, so anything writing to fd 1/2 directly — ffmpeg
via moviepy, C extensions, torch warnings raised from C++ — lands in the
daemon's terminal (a log file, once auto-started) and never reaches the user.
The docstring calls this out as a known asymmetry.

Fix it at the descriptor level. During a run:

1. `os.pipe()`, `os.dup2` the write end onto fds 1 and 2, and pump the read end
   into `FRAME_STDOUT`/`FRAME_STDERR` frames from a reader thread.
2. Point `sys.stdout`/`sys.stderr` at wrappers over those fds.

That gives **one ordered channel** carrying Python-level, C-level and
subprocess output together — which also fixes an ordering bug that exists
today, where the two classes of output travel by different paths and cannot be
interleaved correctly.

Two details that are easy to get wrong:

* **`isatty` must keep lying.** The current `_ClientStream.isatty()` reports the
  *client's* tty-ness, shipped in the request as `isatty_out`/`isatty_err`. That
  is what makes the tqdm progress bars in
  [`render_loop.py:148`](algan/render_loop.py) render as progress bars. A pipe
  is not a tty, so a naive redirect silently degrades them to one line per
  update. The wrappers must override `isatty()` with the client's value, exactly
  as today.
* **Windows needs `SetStdHandle` as well as `dup2`.** Child processes inherit
  their handles from `GetStdHandle(STD_OUTPUT_HANDLE)`, not from the CRT file
  descriptor, so `os.dup2` alone captures Python but not ffmpeg. This is the one
  genuinely platform-specific piece of the change.

The daemon's own `_say` chatter is unaffected: `_CONSOLE` is captured at import
([`daemon.py:110`](algan/daemon.py)) precisely so daemon output never enters a
client's stream. That existing decision is what makes this fix clean.

### 5.2 Environment variables

Only the startup subset travels with a request
([`daemon_client.py:231`](algan/daemon_client.py)), and it is compared, not
applied. Everything else — `MY_OUTPUT_DIR=/tmp python scene.py` — is silently
read from the **daemon's** environment instead of the caller's. That is a bug
today and a serious one under auto-start.

Ship the client's full `os.environ` in the request and swap it in for the
duration of the run, restoring afterwards. This is the same pattern `execute`
already applies to `sys.argv` and the working directory
([`daemon.py:659-666`](algan/daemon.py)), so it fits the existing shape. The
startup subset keeps its current behavior — refused on mismatch
([`daemon.py:400`](algan/daemon.py)), because those are baked into Torch and
Taichi at launch and cannot be adopted per run.

### 5.3 stdin

The daemon's stdin is its own Enter-to-re-render trigger
([`daemon.py:462`](algan/daemon.py)), so a script calling `input()` would
compete with the trigger thread for input. Connect the run's stdin to
`DEVNULL` explicitly and document it. Scene scripts do not read stdin, and
adding a client→daemon stdin channel to the protocol is not worth it for a case
nobody has.

### 5.4 `atexit`

`runpy.run_path` does not run the script's `atexit` handlers — they would queue
until the daemon itself exits. Document; do not emulate. A scene script whose
output depends on `atexit` is already relying on interpreter shutdown, which a
warm process legitimately does not perform.

---

## 6. Auto-start

With §5 done, the first run can simply use the daemon it starts. `maybe_handoff`
currently gives up as soon as `read_state()` returns None
([`daemon_client.py:290`](algan/daemon_client.py)); instead it spawns a daemon,
waits for it to publish its state file, and hands off.

The rejected alternative was to run the first script in-process and spawn the
daemon afterwards from an `atexit` hook. It keeps run 1 maximally normal, but it
performs the ~20-27s of Torch and Taichi initialization **twice** — once in the
script, once again in the background daemon — and the second one starts exactly
when the user is most likely to launch their next run. Spawn-and-hand-off does
the work once. Run 1's wall clock is the same either way, since the daemon does
the same cold start the script would have done.

### 6.1 Trigger and gating

Spawn when all of:

* `should_try()` is true — the existing conservative check
  ([`daemon_client.py:184`](algan/daemon_client.py)) that this is a plain
  `python foo.py` and not a REPL, notebook, `python -c`, or the daemon's own
  child. It already returns False under pytest
  ([`daemon_client.py:200`](algan/daemon_client.py)), so **the test suite never
  auto-starts a daemon** — no new test isolation is required.
* `ALGAN_AUTO_DAEMON` is on (default true).
* `read_state()` returns None.

### 6.2 Spawn mechanics

`subprocess.Popen([sys.executable, "-m", "algan.daemon", "--idle-timeout", N])`,
fully detached:

* stdin `DEVNULL`; stdout/stderr to `$ALGAN_HOME/daemon.log` (append, truncated
  past a size cap). A detached process has no terminal, and `_say` has to go
  somewhere.
* POSIX: `start_new_session=True`. Windows: `DETACHED_PROCESS |
  CREATE_NEW_PROCESS_GROUP`.
* `cwd=algan_home()`, so the daemon never holds a project directory open — on
  Windows that blocks deleting it. The daemon `os.chdir`s per run anyway.
* Environment inherited as-is, so the startup-env values it bakes in match the
  spawning process and the immediately-following handoff cannot mismatch.
  `ALGAN_DAEMON_CHILD` is explicitly cleared in the child env.

### 6.3 Waiting, and giving up

Poll for the state file, then hand off as usual. Two bounds, because a first run
must never hang on this:

* **Readiness timeout** (`ALGAN_DAEMON_START_TIMEOUT`, default 60s). The daemon
  publishes its state file only after Torch and Taichi are up, which is the
  whole ~20-27s. On timeout, stop waiting and run in-process; the daemon is
  still warming and will serve the *next* run.
* **Spawn failure** — `Popen` raises, or the process exits immediately — falls
  back to in-process silently, exactly as every other daemon failure does today.

The worst case is therefore "first run took its normal cold time, plus up to the
timeout, and then ran normally", never a hang.

### 6.4 Telling the user

One line to stderr at spawn:

```
[algan] starting a background render daemon so later runs skip the ~20s
        startup (log: ~/.algan/daemon.log, exits after 30 min idle).
        Disable with ALGAN_AUTO_DAEMON=0.
```

A background process holding a CUDA context that the user did not ask for and
cannot see is not acceptable; this line is part of the feature, not decoration.

---

## 7. Two daemon-side fixes auto-start requires

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

## 8. Environment variables

Three new names in `_RUNTIME_VARIABLES` in
[`algan/environment.py`](algan/environment.py) (alphabetically, beside the
existing `ALGAN_USE_DAEMON`), each read through an accessor at the point of use
per `CLAUDE.md`:

| Name | Default | Meaning |
|---|---|---|
| `ALGAN_AUTO_DAEMON` | `1` | Start a daemon when none is running. |
| `ALGAN_DAEMON_IDLE_TIMEOUT` | `1800` | Seconds before an auto-started daemon exits. |
| `ALGAN_DAEMON_START_TIMEOUT` | `60` | Seconds to wait for a spawned daemon before running in-process. |

These are runtime variables, not startup ones: they do not participate in
`STARTUP_ENV` and a client whose value differs from the daemon's is not refused.

**There is deliberately no kill switch for the staleness gate.** An earlier
draft had `ALGAN_DAEMON_STALE_CHECK=0` as an escape hatch for a "misfiring"
hash check. No such misfire could be constructed: nothing under `algan/`
generates or rewrites `.py` files, and every failure mode that does exist —
a file read while an editor is mid-save, a transient filesystem error — hashes
*differently* and therefore over-restarts, costing one cold start and
self-correcting on the next run. A switch whose only effect is to re-enable
silent stale renders is worse than no switch.

---

## 9. Tests

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

Run parity (§5):
* a subprocess writing to fd 1 during a run reaches the client, not just the
  daemon console (spawn `python -c "import os; os.write(1, b'x')"` inside a
  fake run and assert the bytes arrive as a `FRAME_STDOUT`);
* Python-level and fd-level writes arrive **in issue order**, which is the
  ordering bug the single-channel design fixes;
* `sys.stdout.isatty()` inside a run reports the *client's* value, not the
  pipe's, so tqdm still renders progress bars;
* a client env var reaches the script's `os.environ`, and the daemon's own
  environment is restored afterwards;
* a startup-env var still refuses the run rather than being applied.

Auto-start:
* no spawn when `ALGAN_AUTO_DAEMON=0`, when `should_try()` is false, or when a
  state file already exists;
* spawn is detached and does not inherit `ALGAN_DAEMON_CHILD`;
* readiness timeout expires → the run proceeds in-process rather than hanging;
* spawn failure (`Popen` raises) → in-process fallback, no traceback;
* a general daemon whose port is taken exits instead of idling;
* the idle timeout fires and removes the state file.

---

## 10. Documentation to update

The current guidance tells people to do by hand what this makes automatic, so
it changes rather than being extended:

* [`algan/daemon.py`](algan/daemon.py) module docstring — the "Limits: edits to
  algan itself require a daemon restart" paragraph.
* [`algan/daemon_client.py`](algan/daemon_client.py) module docstring — the
  auto-start path, and the `_ClientStream` asymmetry note in
  [`daemon.py:269`](algan/daemon.py), which §5.1 makes obsolete. Also **fix the
  startup figure**: the client docstring claims "~65 s of Taichi kernel
  preparation" where `daemon.py:4` says "~10 s" and
  `DESIGN_frontend_trace_cache.md` measures ~20s. Three numbers, three places;
  measure once and make them agree.
* `CLAUDE.md`, Taichi gotchas — "Restart the daemon after changing any Algan
  source" becomes "the daemon detects this and restarts itself".
* `AGENTS_DETAILED.md:357-358`, same two lines.
* `docs/source/contributing/development.rst` if it mentions the daemon.

---

## 11. Phases

| Phase | Deliverable |
|---|---|
| 1 | Content-hash digest replacing `_AlganSourceGuard`'s mtime map, with its tests. No behavior change yet — still warn-only. |
| 2 | The handshake gate (§3) and shutdown ordering (§4). **The correctness fix, and independently shippable** — without the rest, a stale daemon shuts down and has to be relaunched by hand, which is still strictly better than rendering stale. |
| 3 | Run parity (§5): fd-level output forwarding, per-run environment, stdin to `DEVNULL`. Each is a standalone bug fix and each improves the daemon for people already using it by hand. |
| 4 | The two daemon-side fixes in §7 (socket-failure exit, idle timeout). |
| 5 | Auto-start (§6), off by default behind `ALGAN_AUTO_DAEMON` until exercised on Windows and Linux. |
| 6 | Default on; docs in §10 updated. |

Phases 2 and 3 both stand alone and neither depends on auto-start. If
auto-start turns out to be more trouble than it is worth, phases 1-4 still
leave the daemon safe to use during library development — which it is not
today — and fix three real bugs for its existing users.

Phase 3 is a hard prerequisite for phase 5: auto-start is what makes a user's
*first* run land on the daemon, and shipping that before output and environment
behave identically would turn two latent bugs into everyone's first impression.

---

## 12. Hazards

| Hazard | Mitigation |
|---|---|
| Stale render (the whole point) | Content-hash gate at the handshake, before `FRAME_START` (§3) |
| Edit lands mid-run | Out of scope by equivalence, not by omission: the daemon matches plain `python scene.py` exactly (§4.1) |
| `*_taichi.py` edited under a live JIT | Unchanged for the mid-run window; the gate prevents *starting* a run after such an edit |
| ffmpeg / C-level output vanishing into the daemon log | fd-level redirect with a pump thread (§5.1) |
| tqdm progress bars degrading to one line per update | Wrappers override `isatty()` with the client's value (§5.1) |
| Subprocess output escaping on Windows | `SetStdHandle` alongside `dup2` (§5.1) |
| Script reading the daemon's env instead of the caller's | Full env shipped and swapped per run (§5.2) |
| First run hanging on a daemon that never starts | Readiness timeout, then in-process fallback (§6.3) |
| Orphan GPU-resident process | Idle timeout (§7), one-line notice at spawn (§6.4), `ALGAN_AUTO_DAEMON=0` |
| Daemon with no trigger and no work | Exit when the socket fails and there is no SCRIPT (§7) |
| Branch switches forcing pointless restarts | Content hashing, not mtime (§2) |
| Two daemons started at once | Loser fails to bind the port and now exits (§7); the state file is written only by the winner |
| Test suite spawning daemons | `should_try()` already returns False under pytest (§6.1) |

---

## 13. What implementation found that this plan did not

Three defects surfaced only once the pieces ran together. All three are
consequences of the daemon becoming something that starts and stops *routinely*
rather than being launched once by hand, which is the part of this design that
had no precedent to reason from.

**The port does not free instantly (`TIME_WAIT`).** `socketserver` does not set
`SO_REUSEADDR`, so for roughly a minute after a daemon exits its port cannot be
rebound. That was harmless when daemons were launched by hand and lived for
days; it is fatal now that every source edit shuts one down, because the
replacement started by the next run fails to bind and — thanks to §7's new exit
— correctly exits, leaving no daemon at all for the rest of that minute, right
in the middle of a burst of library edits. `_TriggerServer` sets the flag on
POSIX only: on Windows `SO_REUSEADDR` instead permits two simultaneous binds,
which would let two daemons both believe they were serving.

**A hard-killed daemon poisoned auto-start permanently.** `SIGKILL`, a crash or
a reboot leaves the state file behind, and the original design treated the
file's existence as proof of a daemon. Every later run would find it, fail to
connect, fall back cold, and never start a replacement — auto-start defeated
forever by one stale file. `run_remote` now distinguishes *unreachable* from
*refused* (`DaemonUnreachable`), and `_dispatch` clears a registration it could
not reach, token-guarded so a daemon that registered in the meantime survives.
A refusal is left alone: that daemon is alive and answering.

**Output landed in the daemon's directory.** `output_root` and
`output_filename` are resolved once, when the settings are constructed — in a
daemon that is its own startup, where there is no user script — so every
client's video went to the daemon's own directory named
`algan_render_output.mp4` instead of beside the script under its stem. It
predates this work (a hand-launched daemon has it too, masked by launching from
the project directory) but auto-start's `cwd=algan_home()` made it systematic,
and it is the most visible parity break there is: your video is not where you
left your script. `path_settings` now exposes `output_root_for(script)` /
`output_filename_for(script)`, which both the import-time defaults and the
daemon call, so the two cannot drift; `execute` applies them per run, after the
settings restore and before the script runs.

**`python -m` looked exactly like a scene script.** `should_try`
([`daemon_client.py:184`](algan/daemon_client.py)) documents itself as accepting
only a plain `python foo.py`, and names `-m` among the invocations that must
never reach a daemon — but nothing checked for it. Under `-m` the `__main__`
module is the *package's* own `__main__.py`, which ends in `.py` and exists on
disk, so both path tests passed. CI caught it the only way it could: the docs
build runs `python -m sphinx`, `docs/source/conf.py` imports algan, and the
documentation build was handed to a render daemon. `__main__.__spec__` is set
under `-m` and `None` for a script, which is the discriminator.

This one is worth dwelling on, because it is the shape of the whole risk in
§6. Handing off used to require someone to have launched a daemon deliberately,
which made a false positive rare and its blast radius small. Auto-start made
handoff the default, so every latent false positive in `should_try` became a
default behaviour change for a class of programs nobody had considered. The
same reasoning drove the benchmark opt-out: 19 scripts under `benchmarks/` are
plain `python foo.py` that import algan, and measuring inside a warm daemon
means measuring against the adaptive renderer state left by the previous run.

The lesson worth carrying: §5 asked "what does a run on the daemon fail to
reproduce?" and got three of four. The fourth was found by rendering a real
scene and looking for the file. Enumerating divergences from the code is
necessary and not sufficient — and the question §5 *should* also have asked is
"which programs will now be handed to a daemon that were not before?"
