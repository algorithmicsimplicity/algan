# Running a script on a GPU that is not this machine

This box has no GPU. Two harnesses exist so that measuring something on real
GPU hardware costs one launch rather than a new piece of infrastructure:

| Harness | Hardware | Entry point |
| --- | --- | --- |
| **Mac runner** | GitHub's Apple-silicon runner: virtualized M1, **real** Metal GPU, 3 CPUs, 7 GB | `.github/workflows/run_on_mac.yaml` |
| **Kaggle T4** | Kaggle notebook: Tesla T4 (Turing, 16 GB), 4 vCPUs, ~30 GB weekly quota | `scripts/kaggle/` + the Kaggle MCP |

Both run **any command in this repository** and hand back its output. Neither
is a test: nothing here guards a regression, so nothing here runs on the
ordinary push matrix.

**Pick by question, not by convenience.** The T4 answers "how fast, and how
much VRAM" for CUDA — it is the only box that runs the real render path at UHD.
The Mac answers "does this work at all on Metal, and how does MPS compare to
its own CPU". The Mac is a *virtualized* instance: its compute numbers are
sound, its **per-launch and per-copy numbers are not** (a synchronized dispatch
measured 432 µs there against 2.0 µs on its CPU — a virtualization tax on
submission that no physical Mac pays). Never rank a many-small-kernel stage
from Mac timings.

**Neither box baselines pixels.** `expected_outputs_cuda/` was baselined on the
user's Pascal card, so `tests/fast`'s pixel comparison fails on the T4 and on
the Mac *on master*. That failure is not your change. Compare arms against each
other, on one box, instead.

---

## Mac runner

### Launch

Two entry points, because a `workflow_dispatch` only exists once the workflow
file is on the **default branch** — a GitHub rule, not a choice.

**Dispatch** (preferred, leaves no commit):

```
mcp__github__actions_run_trigger  method=run_workflow
    owner=algorithmicsimplicity repo=algan
    workflow_id=run_on_mac.yaml
    ref=<your branch>          # the workflow file AND the code come from here
    inputs={
      "command": "uv run python benchmarks/_mps_vs_cpu_torch_speed.py",
      "arms": "mac-mps,mac-cpu",
      "env": "ALGAN_VIDEO_ENCODER=software",
      "latex": false,
      "timeout_minutes": "60"
    }
```

**Request file** (works from a branch whose `run_on_mac.yaml` has not reached
master yet): write the same keys to `.github/gpu-run/mac.json` and push. The
push itself triggers the run.

```json
{
  "command": "uv run python benchmarks/_foo.py --runs 3",
  "arms": ["mac-mps", "linux-cpu"],
  "env": {"ALGAN_VIDEO_ENCODER": "software"},
  "latex": false,
  "timeout_minutes": 60
}
```

Inputs and file are resolved by `.github/workflows/scripts/resolve_gpu_request.py`
(dispatch inputs win). Arms are `mac-mps`, `mac-cpu`, `linux-cpu`; each pins
`ALGAN_RENDER_DEVICE` and runs the command from the repository root. `linux-cpu`
is the **control** — without it, a Mac arm that reports nothing is ambiguous
between "Metal refused" and "the harness is broken".

Things worth setting deliberately:

* **`latex: true`** if the scene uses `Tex`/`MathTex`. Off by default because
  BasicTeX plus the packages is ~4 minutes and most measurement scripts never
  touch it. A scene that needs it and did not ask fails minutes later, inside
  the render.
* **`taichi_wheel_run_id`**. The MPS arm installs a patched Taichi wheel from a
  `taichi_build.yaml` run (default `33342025517`). **On stock Taichi the Apple
  GPU is refused and Algan renders on the CPU**, so an MPS arm without a wheel
  silently duplicates the CPU arm. `"none"` opts out on purpose — which is a
  real thing to measure, since it is what an unpatched Mac user gets.
* **`arms`**. Free minutes, but 5 concurrent macOS jobs across the whole
  account. Two mac arms is two slots.

### Wait

~10–14 min of setup before the command runs (brew, `uv sync`, and the wheel),
so poll at 5-minute intervals, not tighter. Nothing notifies you.

```
mcp__github__actions_list  method=list_workflow_runs resource_id=run_on_mac.yaml
    workflow_runs_filter={"branch": "<your branch>"}
mcp__github__actions_get   method=get_workflow_run resource_id=<run id>
```

`status: completed` plus a `conclusion`. `success` means the command exited 0 —
the `pipefail`/`tee` in the run step is what makes that true; without it a
pipeline reports only its last command and a crashed script goes green.

> ### Status is stale; the log is not
>
> **Both** of these APIs served `in_progress` for an hour after the run had
> finished, with per-step timestamps frozen mid-job. The first run of this
> harness looked like a 60-minute hang in the render; it had actually finished
> in 100 seconds. Kaggle does the same thing — `get_notebook_session_status`
> answered `RUNNING` long after its notebook had exited 0.
>
> So do not diagnose from a status field. **Try to read the output**: on
> GitHub, `get_job_logs` 404s while a job genuinely runs and returns the whole
> transcript the moment it does not; on Kaggle,
> `list_notebook_session_output` returns the log of a finished run. A
> `cancel_workflow_run` answering *"Cannot cancel a workflow run that is
> completed"* is the same tell.

### Read

```
mcp__github__actions_list  method=list_workflow_jobs resource_id=<run id>
mcp__github__get_job_logs  job_id=<job id> return_content=true tail_lines=300
```

Read the **"Report the environment Algan resolved"** step first: it prints
`ALGAN_DEVICE`. An `mps` arm that says `cpu` measured the CPU arm twice, and
that is invisible in a timing.

The job summary carries the last 400 kB of output; the `run-on-mac-<arm>`
artifact carries all of it plus anything under `algan_outputs/`.

---

## Kaggle T4

The MCP is `https://www.kaggle.com/mcp` (token auth). Code reaches the box by
**cloning this repository at a branch**, so:

> **Push the branch before launching.** There is no patch-payload path and there
> should not be — the repository is public and pushing is one command. A run
> against an unpushed branch fails at the fetch, having spent a session slot.

### Launch

```
uv run python scripts/kaggle/make_notebook.py \
    --tag t4-baseline \
    --branch claude/my-branch \
    --step "uhd:python benchmarks/performance/nn_scene_UHD.py" \
    --step "preview:python benchmarks/performance/nn_scene_PREVIEW.py" \
    --env ALGAN_VIDEO_ENCODER=software \
    --out /tmp/nb.py
```

Then `mcp__Kaggle__save_notebook` with `text` = that file's contents and the
config the generator printed to stderr:

```json
{"newTitle": "algan-t4-baseline", "enableGpu": true, "enableInternet": true,
 "machineShape": "NvidiaTeslaT4", "kernelType": "script", "language": "python",
 "kernelExecutionType": "SaveAndRunAll", "isPrivate": true,
 "sessionTimeoutSeconds": 3600}
```

> ### Re-launching the same tag needs the notebook's `id`
>
> A second `save_notebook` with the same `newTitle` and no `id` is refused —
> *"The requested title ... is already in use"* — and that is the common case,
> because resuming a tag is the whole point of the persistent working
> directory. Pass the `kernel_id` the first save returned (as `id`); the save
> then makes a new version of that notebook and runs it. Getting this wrong
> costs nothing but a round trip, unlike everything else on this page.

> ### `machineShape` must be exactly `"NvidiaTeslaT4"`
>
> An unrecognised value **fails silently and expensively**. Kaggle drops it,
> saves with the generic `machine_shape: "Gpu"`, and satisfies that with
> whatever it has — often a **Tesla P100**, whose compute capability 6.0 this
> torch build refuses. Algan then falls back to the CPU: every arm renders on
> two slow vCPUs, `nvidia-smi` still shows a GPU at the top of the log, and the
> notebook is still titled "t4". Fourteen arms of CPU numbers were collected
> that way before anyone noticed, and a UHD arm dying of `OutOfRenderMemory`
> read like a real regression.
>
> `torch.cuda.is_available()` is **True** on that P100, so the obvious probe
> sails through. `runner.py` asks Algan instead
> (`algan.settings._startup.render_device()`) and aborts the run — but the save
> has still cost a session slot, so get the string right.

The generated body only bootstraps (apt ≈ 25 s, clone, `pip install -e` ≈ 50 s)
and then hands over to `scripts/kaggle/runner.py` **in the clone**. Read that
file to know what a run does; do not put logic in the body.

The install carries the **`pango` extra** by default (`--extras`), which is what
the apt list's Pango/Cairo headers were always there for: without `manimpango`,
`algan.Text` falls back to LaTeX's text mode, and the image has no TeX — so any
scene with a `Text` in it (both `nn_scene_*` benchmarks) cannot run at all. It
builds from source (~1 min) and `site-packages` does not persist between
sessions, so it is paid per run. `--extras ''` opts out.

`/kaggle/working` persists between runs of the same notebook, which is what
carries the clone, the Taichi kernel cache (`ALGAN_CACHE_DIR`) and the results.
**Steps are resumable**: a step that already exited 0 under the same `--tag` is
skipped, so a session cut short by the timeout is continued by re-saving rather
than redone. `--force` re-runs everything.

### Wait

```
mcp__Kaggle__get_notebook_session_status  {"userName": "algorithmicsimp", "kernelSlug": "algan-<tag>"}
```

The Kaggle user is **`algorithmicsimp`** (not the GitHub name), and the slug is
`algan-<tag>` — that is what `--tag` is for.

Nothing notifies you — arm a background timer (`sleep 420` as a background Bash
task) and poll when it fires. And see **Status is stale; the log is not**
above: `RUNNING` here has been observed an hour after a run finished, so read
the output rather than believing the status.

> **The batch GPU session limit is 2, and a *queued* run holds one.**
> `get_notebook_session_status` returns `{}` for a queued run, which is
> indistinguishable from "no session", and `get_accelerator_quota` shows
> `time_used` flat and `time_reserved: 0s`. So a run that has not started looks
> exactly like a run that was never submitted. **Do not re-save to "retry"** —
> each save consumes the other slot. The only signal that distinguishes them is
> trying to save a *different* notebook, which answers
> `{"error": "Maximum batch GPU session count of 2 reached."}`. There is no MCP
> tool that lists or cancels sessions by id (`cancel_notebook_session` wants a
> `kernelSessionId` you have no way to obtain), so a wedged run can only be
> waited out — hence the modest `sessionTimeoutSeconds`.

Do not use `get_notebook_info` to check on a run: it echoes the entire notebook
source back. Use it once, deliberately, to read `machine_shape` back.

`COMPLETE` does not mean success, and `ERROR` does not mean the run said why.
Both are read the same way.

### Read

```
mcp__Kaggle__list_notebook_session_output  {"userName": "algorithmicsimp", "kernelSlug": "algan-<tag>", "pageSize": 1}
```

**Pass `pageSize: 1`.** It trims the `files` list to one entry and does not
touch `log`, which is the part you want; without it you pay for hundreds of
`algan_cache/**` URLs (Taichi `.tic` blobs, manim glyph SVGs).

The response blows the tool-result token cap and gets spilled to a file, and
its `log` key is **a JSON string**, not a string. So:

```
uv run python scripts/kaggle/read_output.py <spilled-tool-result.json> --out /tmp/run.log
```

That writes the transcript and prints a digest in reading order: **device
first**, then step boundaries, timings, verdicts, and the final `RESULTS` line.
Then grep `/tmp/run.log` for whatever the run was about.

### Reading the numbers

* **Verify the device before reading a single number.** The `machineShape` trap
  is silent and the log is the only place it shows.
* **Read RUN 2, never a step's `seconds`.** `results.json`'s per-step `seconds`
  is whole-script wall clock and includes the cold Taichi JIT, which differs
  between two arms of one A/B because each toggle value compiles its own kernel
  variant. On one round that made a UHD arm look **8.4% slower** with a feature
  on (126.09 s vs 116.29 s) when the warm numbers were 25.85 vs 25.69 — neutral.
  `profile_scene(runs=2)` prints `RUN 1 (cold)` and `RUN 2 (warm (steady
  state))`; only the second is a measurement.
* **Size a stage from the unsynced profile in RUN 2**, not from a
  sync-bracketed harness. A harness that fences each launch charges it the queue
  it drains: one such script reported ~12 s and ~16 s per resolve mode on a
  render whose `sheet_resolve_shade` totals 0.3 s in the profile, and that
  mistake ranked a whole optimization.
* **Cross-step video SHAs are a free parity check.** `runner.py` records a
  sha256 per output mp4/png in `results.json` and in the `RESULTS` line, so two
  arms of a byte-identical A/B should print the same digest.
* **Never take a determinism or pixel reading while another process is using
  the GPU.** Free VRAM at job start sets the arena size, which sets tile sizes
  and batch windows; a concurrent job changes the pixels. Check `nvidia-smi`
  (the harness prints it) before trusting a comparison.

**The T4 picks NVENC by itself.** The first run's encoder line read
`Encoding video with h264_nvenc`. That is the right default for throughput and
the wrong one for comparing against an x264-encoded baseline, so pin
`ALGAN_VIDEO_ENCODER=software` whenever the output bytes are the measurement.

### Both harnesses, verified

Each harness's first run rendered `scripts/gpu_smoke.py --runs 2` on the branch
that added it. Use these as the "the plumbing works" reference, not as
performance numbers — one moving square at PREVIEW is not a benchmark:

| Arm | device / Taichi arch | cold | warm | job |
| --- | --- | --- | --- | --- |
| Kaggle T4 | `cuda`, `arch=cuda`, torch 2.10.0+cu128, py3.12 | 26.9 s | 0.93 s | 133 s total |
| mac-mps | `mps`, `arch=metal`, torch 2.7.1, patched wheel | 32.3 s | 1.40 s | ~85 s |
| linux-cpu (control) | `cpu`, `arch=x64` | 17.2 s | 0.76 s | ~100 s |

Reference numbers, Kaggle T4, master @ `95271dac`, warm RUN 2:
`nn_scene_UHD.py` **29.90 s** (30 frames @ 3840×2160; cold 85.85 s),
`nn_scene_PREVIEW.py` **6.25 s** (50 frames; cold 32.72 s).

---

## Adding to either harness

The offline halves are tested (`tests/unit_tests/test_gpu_harness.py`): the
request resolver, the notebook generator, and the device guard. Both failure
modes they guard are silent and late — an empty matrix produces a green run
that measured nothing, and a malformed notebook body fails after apt, clone and
install have already been paid for. Keep them covered.
