---
name: kaggle-t4-measurement
description: "Running algan render benchmarks on a Kaggle T4 through the Kaggle MCP: the working harness, and every trap that cost time"
metadata:
  node_type: memory
  type: reference
  modified: 2026-08-26
---

The Kaggle MCP (`https://www.kaggle.com/mcp`, token auth) is the T4 measurement
box for algan perf work. `[[kaggle-mcp-remote-gpu]]` in the personal memory
covers first-time setup, the image contents and the accelerator string; this
file is the **operating manual** for a perf session, written after round 2
(2026-08-26).

## The harness

`scratch_perf/kaggle/make_notebook.py` generates the notebook body;
`scratch_perf/kaggle/make_chunks.py` splits an oversized payload. Both are in
the repo. Generate, then pass the body as `save_notebook`'s `text`:

```
uv run python scratch_perf/kaggle/make_notebook.py --tag <tag> --repo <worktree> \
    --snapshot <tagname> --arm "uhd:nn_scene_UHD.py" --arm "preview:nn_scene_PREVIEW.py"
```

Kernel config that works: `enableGpu: true`, `enableInternet: true`,
`machineShape: "NvidiaTeslaT4"`, `kernelExecutionType: "SaveAndRunAll"`,
`isPrivate: true`, **`sessionTimeoutSeconds: 3600`**. Arms are resumable — an
arm that already exited 0 in a previous session of the same tag is skipped, so a
cut-short run is continued by re-saving rather than redone.

**`machineShape` must be exactly `"NvidiaTeslaT4"`, and a wrong value fails
SILENTLY and expensively.** An unrecognised string (`"GpuT4x2"` was invented
and tried, 2026-08-27) is dropped: the notebook is saved with the generic
`machine_shape: "Gpu"`, which Kaggle satisfies with whatever it has — often a
**Tesla P100**. Then the whole thing goes quiet rather than failing:

* the P100 is cuda capability 6.0 and this torch build supports (7.0)-(12.0),
  so torch refuses `sm_60` and Algan's `_auto_render_device` falls back to CPU;
* every arm renders on two slow vCPUs, `nvidia-smi` at the top of the log still
  shows a GPU, and the notebook is still called "t4 perf";
* `nn_scene_UHD` then dies with `OutOfRenderMemory` (UHD on CPU), which reads
  like a real regression and is not.

Two runs (tags `memo1`, `memo2`) collected fourteen arms of CPU numbers this
way before anyone noticed. **The check that catches it is `Rendering device set
to <x>` in each arm's log, or the generator's guard** — and the guard has to ask
ALGAN, not torch: `torch.cuda.is_available()` returns **True** on that P100 (it
reports the device and only rejects the arch later), so the obvious probe
sails straight through. `make_notebook.py` now aborts the run unless
`algan.settings._startup._RENDER_DEVICE.type` is `cuda`.

Confirm the shape took by reading `machine_shape` back from
`get_notebook_info` — it echoes the whole source, so do it once, deliberately.

Per-run fixed cost: apt ~25 s + `pip install -e` ~50 s. `/kaggle/working`
persists between runs of the same notebook, which is what carries the git clone,
the Taichi kernel cache (`ALGAN_CACHE_DIR=/kaggle/working/algan_cache`) and the
payload store.

## Traps, in the order they will bite

**The batch GPU session limit is 2, and a *queued* run holds one.**
`get_notebook_session_status` returns `{}` for a queued run — indistinguishable
from "no session" — and `get_accelerator_quota` shows `time_used` flat and
`time_reserved: 0s`. So a run that has not started looks exactly like a run that
was never submitted. **Do not re-save to "retry"**: each save consumes the other
slot. The only signal that tells you what is happening is trying to save a
*different* notebook, which answers
`{"error": "Maximum batch GPU session count of 2 reached."}`. There is no MCP
tool that lists sessions or cancels one by id (`cancel_notebook_session` wants
an integer `kernelSessionId` you have no way to obtain), so a wedged run can
only be waited out — hence the modest `sessionTimeoutSeconds`. Once, both slots
were held for over an hour.

**Nothing notifies you when a run finishes.** Arm a background timer
(`sleep 420; echo ...` as a background Bash task) and poll
`get_notebook_session_status` when it fires.

**`get_notebook_info` echoes the entire notebook source back.** It is the
expensive way to learn a version number. Use `get_notebook_session_status`.

**`COMPLETE` does not mean success**, and `ERROR` gives you nothing directly:
`list_notebook_session_output` returns the *stderr stream* of a failed run,
which is where a traceback shows up. For a successful run, download files with
`download_notebook_output(filePath=...)`, which mints a signed
`kaggleusercontent.com/kf/<session id>/...` URL that plain `curl` can fetch.

**Getting code onto the box is the real problem.** If the branch can be pushed
to GitHub, use `--branch <name>` and the notebook just fetches it — everything
below is unnecessary. Without push:

* the notebook seeds from the public tip and applies a gzip+base64
  `git diff --binary --unified=1 <base>` carried **inline in the body**;
* **a payload past ~10 kB does not survive being emitted inline in one piece.**
  An 18 kB base64 blob arrived as 11 kB. So every payload carries
  `assert len(PATCH_B64) == N` and a sha256 assert *before the notebook touches
  anything* — that turned a silent wrong-code render into a one-minute failure
  with an exact diagnosis;
* `make_chunks.py` splits a payload into ~6 kB chunk notebooks, each writing one
  `part.NNN` into `/kaggle/working/payloads/<key>/` and verifying its own sha;
  the render notebook assembles `part.*`. Chunk notebooks need no GPU;
* `--snapshot NAME` makes the render notebook **commit and tag the overlaid
  tree inside the Kaggle clone**, and `--from-tag NAME` starts a later run from
  it. Upload the payload in full once, then every later run is a small delta
  that fits inline. Use this.
* `-U1` on the diff is worth ~11% when the change is mostly reindentation.
* `upload_dataset_file` really does return a working GCS resumable-upload URL,
  but there is no create-dataset call to turn the object into something a
  notebook can attach. Dead end; do not spend time on it.

**Restrict the patch paths.** `--paths algan --paths benchmarks/performance` —
the round's briefs, reports, logs and archived patches under `scratch_perf/`
must never ride in a payload.

## What the T4 is and is not good for

Good: absolute warm/cold seconds at UHD, ablations behind env vars, and any
question about a Turing-class GPU. **Not** for the pixel-compared render suites
— `expected_outputs_cuda/` was baselined on the user's Pascal card. See
`[[algan-measurement-traps]]`.

Measured baselines on the Kaggle T4 (master @ `95271dac`, warm RUN 2):
`nn_scene_UHD.py` **29.90 s** (30 frames @ 3840x2160, cold 85.85 s),
`nn_scene_PREVIEW.py` **6.25 s** (50 frames, cold 32.72 s). See
`[[t4-round2-findings]]` for what those seconds are made of.
