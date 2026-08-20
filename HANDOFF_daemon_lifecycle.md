# Handoff — Daemon Lifecycle: validation on Windows + CUDA

**You are picking up finished, working code that has only been exercised on one
platform.** Every phase of `DESIGN_daemon_lifecycle.md` is implemented and green
on Linux/CPU. What is missing is the half that box cannot speak for: Windows,
and a real GPU. This document is self-contained — you should not need to read
the conversation that produced the branch.

Branch: `claude/taichi-kernel-startup-2jtmrp`. No PR has been opened.

---

## 1. What this changes, in one paragraph

Algan has a warm-process render daemon (`algan/daemon.py`) that keeps torch and
Taichi's compiled kernels loaded so a re-render costs ~1 s instead of ~20 s of
Taichi kernel preparation. Two things were wrong with it. It would happily serve
a run using **stale code** after you edited algan itself — it printed a warning
and rendered anyway. And it only existed if you remembered to launch it by hand,
so ordinary users never got the benefit. Now: the daemon fingerprints every
algan source file and refuses (and shuts down) the moment they change, so a
stale render is impossible; and an ordinary `python scene.py` starts a daemon
when none is running, runs on it, and leaves it warm. Those two compose into
self-restarting behaviour with no supervisor process: the daemon dies on an
edit, the next run starts a fresh one.

Because auto-start means a user's **first** run now lands on the daemon,
"running on the daemon" had to become indistinguishable from "running in your
own process". Three parity bugs were fixed to get there (output descriptors,
environment, output paths).

---

## 2. Why it is built this way (the two decisions worth not re-litigating)

**Why a content hash and not mtimes.** A gate must never miss an edit, and
mtime can be preserved across one. More practically: `git checkout` / `stash` /
`rebase` rewrite mtimes wholesale without changing a byte, so an mtime gate
would shut the daemon down and force a cold restart on every branch switch,
including switching away and back. Hashing all 307 `.py` files under `algan/`
(5.3 MB, including `external_libraries/`) costs ~10 ms and runs once per run
launch.

**Why the check lives in the socket handshake.** `do_job` sends `FRAME_START`
*before* calling `execute`, and `FRAME_START` is the point after which the
client can no longer safely fall back (re-running locally could duplicate side
effects). So the check sits in `_TriggerHandler._handle_run` alongside the
token/protocol/env checks, where refusing is still free. Do not move it into
`execute` — that reintroduces the bug.

**What is deliberately not covered.** An edit landing *while a run executes* is
not caught. Because the gate runs at every run launch, daemon-loaded sources
equal disk at the instant a run starts, so from there the daemon behaves exactly
as a fresh interpreter would — the exposure is identical to editing during a
plain `python scene.py`. Closing it would mean interrupting a render, which is
worse. There is also **no kill switch** for the gate, on purpose: its only
possible effect would be to re-enable silent stale renders.

---

## 3. Files changed

| File | What |
|---|---|
| `algan/daemon.py` | `_SourceDigest` (content fingerprint) replacing `_AlganSourceGuard`; `_stale_message`; handshake gate; `stale_quit()` for local triggers; `_RunStream`/`_Pump`/`_run_context` replacing `_ClientStream`/`_client_streams`; `_swap_std_handle` (Windows); `_capture_console`; `_TriggerServer` (SO_REUSEADDR); `--idle-timeout`; exit when socket fails with no SCRIPT; per-run output-path defaults; stdin trigger reads a bound stream instead of `input()` |
| `algan/daemon_client.py` | `PROTOCOL_VERSION` 1→2; ships `env_full`; `DaemonUnreachable`; `_clear_stale_state`; `_dispatch`; `_autostart`/`_spawn_daemon`/`_open_log`/`log_path` |
| `algan/settings/path_settings.py` | `output_root_for(script)` / `output_filename_for(script)` extracted so the daemon and the import-time defaults share one rule |
| `algan/environment.py` | registers `ALGAN_AUTO_DAEMON`, `ALGAN_DAEMON_IDLE_TIMEOUT`, `ALGAN_DAEMON_LOG_MAX_BYTES`, `ALGAN_DAEMON_START_TIMEOUT` |
| `tests/unit_tests/test_daemon_staleness.py` | new — 11 tests on the digest and refusal text |
| `tests/unit_tests/test_daemon_run_context.py` | new — 13 tests on run parity |
| `tests/unit_tests/test_daemon_client.py` | extended — auto-start, stale-registration recovery, unreachable-vs-refused |
| `CLAUDE.md`, `AGENTS_DETAILED.md` | the "restart the daemon by hand" guidance, which is now wrong |
| `DESIGN_daemon_lifecycle.md` | plan of record; §13 lists what implementation found |

New environment variables (all runtime, none startup-only):

| Name | Default | Meaning |
|---|---|---|
| `ALGAN_AUTO_DAEMON` | `1` | Start a daemon when none is running |
| `ALGAN_DAEMON_IDLE_TIMEOUT` | `7200` | Seconds before an auto-started daemon exits |
| `ALGAN_DAEMON_START_TIMEOUT` | `60` | Seconds to wait for a spawned daemon before running in-process |
| `ALGAN_DAEMON_LOG_MAX_BYTES` | `4194304` | Rotate `~/.algan/daemon.log` past this size |

---

## 4. What has already been verified (Linux, CPU, this branch)

Do not redo these; they are recorded so you know what is *not* in question.

* `pytest -q tests/unit_tests` — **1063 passed, 91 skipped**, 149 s.
* `pytest -q --fast` — see §8; run it yourself on your machine regardless.
* End-to-end against a real daemon, comparing daemon-served output to an
  `ALGAN_USE_DAEMON=0` control: Python-level stdout/stderr, direct `os.write`
  to fds 1 and 2, subprocess stdout/stderr, argv, cwd, exit code (7),
  environment variable propagation, stdin returning EOF, and `isatty()`
  reporting True through a pty.
* Staleness gate: editing `algan/constants/spatial.py` under a live daemon
  produced a refusal naming the file, an in-process fallback whose output
  matched the control exactly, and a clean daemon shutdown that removed the
  state file.
* Auto-start: cold first run spawned a daemon and ran on it; second run took
  **0.23 s** against **4.79 s** for the same script with `ALGAN_USE_DAEMON=0`
  (both without any kernel work — with a render the gap is far larger).
* A real render through the daemon: 26.4 s cold (kernel compile included),
  0.23 s warm, 4-frame 32×32 mp4, landing beside the script.
* Idle timeout fires and removes the state file; stale registration from a
  `SIGKILL`ed daemon is detected, removed, and replaced.

---

## 4b. Verified on Windows + CUDA (2026-08-20, GTX 1050, this checkout)

Run on this machine while working through §5/§6; what is still open is listed
at the end.

* **§5.1 `SetStdHandle`.** `tests\unit_tests\test_daemon_run_context.py` passes
  here, `test_subprocess_output_reaches_the_client` included — so the handle
  swap does take effect. A real render's ffmpeg output also reaches the client.
* **§5.2 Detached spawn.** Auto-start spawned a daemon that outlived its
  parent, showed no console window, and logged to `~/.algan/daemon.log`.
* **§5.3 Port rebinding.** Not a problem in practice: the staleness gate shut a
  daemon down and the very next run bound 46711 again, six times over, with no
  `SO_REUSEADDR` and no wait. `SO_EXCLUSIVEADDRUSE` is not needed.
* **§5.5 The gate fires on Windows.** Editing `algan\constants\spatial.py`
  under a warm daemon produced a refusal naming `constants/spatial.py`, an
  in-process run that read the *edited* value, and a daemon that removed its
  state file and exited. Reverting the edit did the same again (content hash,
  not mtime), and editing a `*_taichi.py` added the recompile warning.
* **§6.2 GPU residency.** An idle daemon held 1601 MiB of a 4096 MiB card after
  a 90-frame render. It now hands that back when the run ends (§14 of the
  design doc), so an idle daemon sits at ~125 MiB and the idle timeout is no
  longer the only thing standing between other GPU work and that VRAM.
* **§6.4 Renders are unchanged.** `pytest -q --fast` 213 passed (66 s of its
  75 s budget) and `pytest -q tests\unit_tests` 1228 passed / 90 skipped. No
  baseline was regenerated. `tests\full_renders` was **not** run.

Still open: §5.4 (project-directory deletion and mp4 locking), §6.1 (replacing
the "~20 s" docstring figures with a measured one), §6.3 (confirming exactly
one process allocates VRAM during a cold first run), §6.5 (tqdm on a real
Windows terminal).

---

## 5. What you need to verify — Windows

This is the bulk of the remaining work. The daemon is used on Windows in this
project (`CLAUDE.md` notes the one-render-at-a-time rule because orphaned
renders hold output mp4s locked), so these paths matter.

### 5.1 `SetStdHandle` — the one genuinely platform-specific piece

`_swap_std_handle` / `_restore_std_handle` in `algan/daemon.py`. On POSIX,
`os.dup2` is enough: a child launched by `subprocess` inherits descriptor 1. On
Windows a child inherits from `GetStdHandle(STD_OUTPUT_HANDLE)`, which `dup2`
does not move — so without `SetStdHandle` ffmpeg would keep writing to the
daemon's own log while every Python-level write was correctly redirected. **This
code has never executed.** It is wrapped in a bare `except Exception` returning
`None`, so a failure degrades silently to "subprocess output is lost" rather
than crashing — which means a passing render does not prove it works. Test it
explicitly:

```
.venv\Scripts\python.exe -m pytest -q tests\unit_tests\test_daemon_run_context.py
```

`test_subprocess_output_reaches_the_client` is the one that matters. If it
fails, the handle swap is not taking effect; check the `ctypes` argtypes
(`GetStdHandle.restype` must be `c_void_p`, or a 64-bit handle is truncated to
a 32-bit int and the restore corrupts the daemon's console).

### 5.2 Detached spawn

`_spawn_daemon` uses `creationflags = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP`
(`0x8 | 0x200`). Verify the spawned daemon survives its parent exiting, does not
pop up a console window, and writes to `~/.algan/daemon.log`.

### 5.3 `SO_REUSEADDR` is deliberately off on Windows

`_TriggerServer.allow_reuse_address = sys.platform != "win32"`, because on
Windows the flag permits two *simultaneous* binds — which would let two daemons
both think they were serving. Verify the consequence: after a daemon exits, can
the next run bind port 46711 promptly? If Windows turns out to hold the port
the way Linux's `TIME_WAIT` does, the spawn will fail and exit (correctly, but
leaving no daemon). If you see that, the fix is `SO_EXCLUSIVEADDRUSE` rather
than turning `allow_reuse_address` on. Test: start a daemon, `quit` it via the
socket, immediately run a script, and check whether a daemon comes back.

### 5.4 Windows file locking

The daemon holds no project directory open (`cwd=algan_home()`), which was
chosen for this reason. Confirm a project directory can be deleted while a
daemon is warm, and that the daemon does not lock the output mp4 between runs.

### 5.5 Path handling

`_SourceDigest` normalises separators (`replace(os.sep, "/")`) so digests are
comparable, and `_is_under` already handles cross-drive paths. Confirm the gate
fires on Windows by editing a file under `algan\` while a daemon is warm.

---

## 6. What you need to verify — CUDA

### 6.1 The startup saving is real

The headline number (~20 s of Taichi preparation per cold run) comes from
`DESIGN_frontend_trace_cache.md`, measured on CUDA. Reproduce it:

```
set ALGAN_LOG_TAICHI_COMPILES=1
.venv\Scripts\python.exe your_scene.py          # cold, no daemon
```

then the same scene twice more with a daemon warm, and record the three
numbers. **The docstrings in the tree disagree** — `daemon.py` says ~20 s,
`daemon_client.py` said ~65 s before this branch, `DESIGN_frontend_trace_cache.md`
measures 27.4 s total for a trivial `save_frame`. One measured number should
replace all of them; both module docstrings now say "~20 s" and should be
corrected to whatever you actually measure.

### 6.2 GPU residency and the idle timeout

An auto-started daemon holds a CUDA context. Confirm with `nvidia-smi` that it
appears when a daemon starts and is released when the idle timeout fires
(default 7200 s; test with `--idle-timeout 30` on a hand-launched daemon).
Judgement call worth making with real numbers in front of you: **is 2 hours
the right default** for a background process sitting on VRAM? Lower it if it
gets in the way of other GPU work — though see §4b: an idle daemon now
holds ~125 MiB rather than 1601 MiB, which is most of what made the number
urgent.

### 6.3 Two heavy processes must not overlap

Auto-start was chosen (over the alternative of running the first script
in-process and warming a daemon afterwards) specifically to avoid two
simultaneous torch+Taichi initialisations. Confirm on a real GPU that a cold
first run shows exactly one process allocating VRAM.

### 6.4 Renders are unchanged

Nothing here touches the renderer, so no baseline should move. Establish that:

```
.venv\Scripts\python.exe -m pytest -q --fast
.venv\Scripts\python.exe -m pytest -q tests\full_renders
```

`tests/full_renders` skips itself when `CI` is set; it needs
`ALGAN_RUN_FULL_RENDERS=1` on a machine whose `expected_outputs_cuda/`
baselines are its own. **If a baseline moves, something is wrong** — investigate
rather than re-baselining. The one plausible mechanism is the output-path change
in §3 (`path_settings.py`), which alters *where* files are written, not their
contents.

### 6.5 tqdm through the daemon on a real terminal

`_RunStream.isatty()` reports the client's tty-ness so progress bars survive.
Verified through a pty on Linux; confirm on a Windows terminal that a real
render's progress bar renders as a bar and not as one line per update.

---

## 7. Known gaps and judgement calls left open

* **`atexit` handlers do not run** for a daemon-served script — `runpy` does not
  run them and a warm process never shuts down. Documented in both module
  docstrings, not emulated. If a real scene script turns out to depend on this,
  reconsider.
* **stdin is `os.devnull`.** The daemon's own stdin is its Enter-to-re-render
  trigger, so a script cannot have it. A script calling `input()` gets `EOFError`
  — the same as `stdin=DEVNULL` in a subprocess. Adding a client→daemon stdin
  channel to the protocol was judged not worth it for a case scene scripts do
  not have.
* **`ALGAN_DAEMON_CHILD=1` is visible to the script**, forced into the swapped
  environment so a python subprocess started by the script cannot hand *itself*
  to the daemon and deadlock behind the run that spawned it. It is the one
  environment variable that differs from a plain run.
* **The daemon's console tee.** `_Pump` echoes run output to the daemon's own
  console as well as the client. On an auto-started daemon that console is
  `~/.algan/daemon.log`, so the log grows with every run's output; it rotates at
  `ALGAN_DAEMON_LOG_MAX_BYTES` (4 MB). Check the rotation actually keeps it
  bounded over a long session.
* **Protocol version is now 2.** A client and daemon that disagree refuse each
  other and the client falls back, so a half-updated checkout degrades safely
  rather than misparsing — but anyone with a daemon running from before this
  branch must restart it (the staleness gate will do that for them on the first
  run, since the sources changed).

---

## 8. Reproducing the Linux verification

The end-to-end probe used during development is worth re-running on Windows.
Create `probe.py`:

```python
import os, subprocess, sys
import algan  # noqa: F401  (triggers the handoff)

print("PY_STDOUT ok", flush=True)
print("PY_STDERR ok", file=sys.stderr, flush=True)
os.write(1, b"FD1_DIRECT ok\n")
os.write(2, b"FD2_DIRECT ok\n")
subprocess.run([sys.executable, "-c",
                "import sys; print('SUBPROC_STDOUT ok');"
                " print('SUBPROC_STDERR ok', file=sys.stderr)"], check=True)
print("ENV_PROBE =", os.environ.get("PROBE_VAR", "<missing>"))
print("CWD =", os.path.basename(os.getcwd()))
print("ARGV =", sys.argv[1:])
print("ISATTY_OUT =", sys.stdout.isatty())
try:
    input(); print("STDIN read something")
except EOFError:
    print("STDIN eof ok")
sys.exit(7)
```

Run it twice — once with `ALGAN_USE_DAEMON=0` for a control, once normally —
and diff. The only expected differences are the Taichi banner (the daemon
printed it to its own log at startup) and `ALGAN_DAEMON_CHILD`.

Then the staleness gate, which is the correctness claim:

```
# with a daemon warm
echo # touch >> algan\constants\spatial.py
.venv\Scripts\python.exe probe.py
```

Expect: a refusal on stderr naming `constants/spatial.py`, the script running
in-process (the Taichi banner reappears), and the daemon gone. Revert the edit.

---

## 9. Closing the project out

1. Work through §5 and §6, fixing what breaks.
2. Replace the "~20 s" figures in `algan/daemon.py` and
   `algan/daemon_client.py` module docstrings with your measured number (§6.1).
3. Revisit the `ALGAN_DAEMON_IDLE_TIMEOUT` default with real VRAM numbers (§6.2).
4. Open a PR. `.github/pull_request_template.md` is the layout — What and why /
   Rendered output / Verification / Docs. On **Rendered output**: state that no
   baselines were regenerated and name the suites that establish it, on which
   hardware. Say plainly that the Linux/CPU evidence in §4 came from a
   CPU-only cloud session and cannot speak for CUDA — name the machine you ran
   the rest on. Write the body yourself; do not paste a generated summary
   (`CLAUDE.md` is emphatic about this, and the auto-generated description has
   been wrong on every PR this repo has had).
5. Delete this file in the same PR, or keep it if §5/§6 turned up follow-up
   work worth tracking.
