---
name: t4-round3-findings
description: "Round 3 of the T4 optimization work (2026-08-26): recovered round-2 patches verified and landed, the immortal-ray fix, and where the T4 numbers ended"
metadata:
  node_type: memory
  type: project
  modified: 2026-08-26
---

Round 3 ran from a Claude Code cloud session (CPU-only) with Kaggle T4
notebooks as the measurement box and Ox Alpha as the implementation/audit
agent (five invocations; reviews appended to
`scripts/ox_alpha/ox_alpha_opencode_agent.md`). Branch
`claude/algan-t4-optimization-b5a10b`. Supersedes nothing — reads on top of
`[[t4-round2-findings]]`; `DESIGN_T4_optimization.md` carries the details.

## Where the numbers ended (Kaggle T4, warm RUN 2, nn scene)

| tree | UHD | PREVIEW |
|---|---|---|
| master @ `95271dac` | 29.90 s | 6.25 s |
| + round 2 (prep-side) | 27.81 s | 5.86 s |
| + sheet-chain kernels | 26.79 s | 5.72 s |
| + weight-floor exit | **24.68 s** | **5.60 s** |

**-17.5% at UHD, -10.4% at PREVIEW vs master.** With
`ALGAN_PREFETCH_GPU_PREP=1` (default OFF) UHD adds -3.5% within-session;
PREVIEW is flat (see below).

## What landed

* **`--branch` in the Kaggle harness actually implemented** — the flag
  existed but the notebook body ignored it, and the designated branch had
  never really been pushed (the local `origin/` ref was harness-seeded).
  A T4 measurement is now: push, `make_notebook.py --tag X --branch Y --arm
  ...`, `save_notebook`, poll. No payload machinery.
* **The round-2 WIP patches were recovered by the user** (they had never
  been committed) and archived under `scratch_perf/r2/patches/`:
  sheet-chain continuation, preflight overlap, continuation-rays
  (+ `spawn_counts.py`, its missing host half — apply-time copy documented
  in the archive commit).
* **Sheet-chain kernels** (pair-expand / band-stats / shell-ceiling),
  default ON, bit-identical (harness + 9 unit tests + per-toggle lossless
  A/Bs with launch proof). Verification found four defects in the recovered
  patch — worst: a `mask.is_cuda` gate that made every CPU check of
  pair-expand vacuous. `_window_pairs` census 119 -> 73 ops/call.
* **The immortal rays were never immortal** — they ride to the bounce cap,
  not `max_iters` (round 2 misread the bounce table's un-fillable last
  row). Real defect: the weight-floor exit sits at the bottom of the
  hit-drain loop, every in-place bounce branch `break`s past it, and the
  post-loop exits excluded bounced rays. **The fix's UHD win (-8.1%) came
  from the mid-chain, not the 5-7 tail**: bounce 2 carried 2.57M rays of
  which 1.3M were sub-floor; the drain now ends at bounce 4. Worst channel
  diff exactly 1 (inside every suite's tol-2 gate; committed baselines
  stay valid). **Default ON by the owner's explicit decision** — 1-LSB
  variation accepted without visual inspection, same posture as the
  pn_criterion fast_math exception.
* **Preflight overlap** verified end-to-end: byte-identical worker builds
  on the T4 at two memory pins, forced-OOM retry working under overlap.
  UHD -3.5%; **PREVIEW flat, and the reason matters**: at 3 batches the
  worker is the critical path (prep + overlapped builds exceed what the
  render can hide), so the wait relocates. The r2 ratios were measured at
  25-150 batches and do not transfer to T4 batch counts. Default OFF.

## Traps this round paid for (do not re-pay)

* **A profiled render is a distinct execution context.** The hand-written
  profiler merge shim did not forward kwargs; `_merge_scene` grew
  `track_peak`, and every `profile_scene` run crashed while every
  unprofiled render — including the entire local verification program and
  the T4 acceptance driver — passed. When a change touches a function the
  profiler wraps by hand, run one profiled render as verification. The
  generic `wrap_function` forwards `*args/**kwargs`; `merge_wrapper` was
  the one exception.
* **Kaggle T4 wall clock varies ~0.7 s across sessions** (uhd_off measured
  24.68 and 25.40 on consecutive sessions of the same tree). Quote
  within-session A/B deltas; never compare arms across sessions.
* **`kaggleusercontent.com` is blocked from the cloud session's proxy** —
  `download_notebook_output` URLs are unusable there. The session stdout
  stream via `list_notebook_session_output` carries everything
  (`results.json` is printed, profile reports ride the log); parse the
  `data` fields.
* **Checkpoint-commit an agent's in-flight workspace** when a stop-hook or
  restart risk demands it — clearly labeled WIP, superseded by the landing
  commit. Ox tolerates mid-run commits of its own tree and reports them.
* Container restarts kill background `opencode run` processes but not
  their sessions: resume with `opencode run -s <session-id>` and a
  continuation message; both resumed runs picked up cleanly.
