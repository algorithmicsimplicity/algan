# T4 optimization round 2 — worklog

Round 1 (branch `perf/t4-nn-scene-throughput`, merged to master at `fc100cd8`)
took warm PREVIEW 36.5 s -> 7.7 s and UHD 50.0 s -> 30.9 s on a Colab T4.
This round runs on the user's desktop (Windows, GTX 1050) with **Kaggle T4
notebooks as the measurement box**.

## Measurement harness

`scratch_perf/kaggle/make_notebook.py` generates the Kaggle notebook body.
Because nothing is pushed to GitHub, the notebook seeds itself from the public
tip (`df07859d`, which is what a fresh `git clone` lands on) and applies a
**gzip+base64 `git diff --binary`** of the working tree inline; `--extra` tars
in untracked files. Kaggle's file persistence keeps `/kaggle/working`, so the
clone, the pip install target and the Taichi kernel cache
(`ALGAN_CACHE_DIR=/kaggle/working/algan_cache`) survive between runs.

```
uv run python scratch_perf/kaggle/make_notebook.py --tag <tag> --repo <worktree> \
    --arm "uhd:nn_scene_UHD.py" --arm "preview:nn_scene_PREVIEW.py"
```

Then `save_notebook` with `slug algorithmicsimp/algan-t4-perf`,
`machineShape "NvidiaTeslaT4"`, `enableGpu`, `enableInternet`,
`kernelExecutionType SaveAndRunAll`. Results land in
`/kaggle/working/out/<tag>/`. Per-run overhead: apt ~25 s + pip ~50 s.

**Do not call `get_notebook_info`** — it echoes the whole notebook source back.
`get_notebook_session_status` is the cheap poll.

## Baseline on the Kaggle T4 (tag `r2base`, master @ 95271dac)

| scene | warm (RUN 2) | cold (RUN 1) |
|---|---|---|
| `nn_scene_UHD.py` (30 frames @ 3840x2160) | **29.90 s** | 85.85 s |
| `nn_scene_PREVIEW.py` (50 frames @ PREVIEW) | **6.25 s** | 32.72 s |

Reports: `scratch_perf/r2/t4_r2base_UHD.txt`, `..._PREVIEW.txt`.

### UHD is render-bound, and three Taichi kernels are most of it

**Read the `excl` column with care.** The stage timers subtract nested *stages*
from a stage's exclusive time but did **not** subtract the Taichi **kernels** a
stage launched — the kernel hooks write straight into `TIMERS.times` without
opening a stage. So every `excl` in a committed report is inflated by the
kernels underneath it. `charge_kernel_to_parent` (this round) fixes it;
`tests/unit_tests/test_profiler_stage_attribution.py` pins the contract.

Corrected, the 26.0 s render of 30 frames at 3840x2160 is:

| item | s | share | kind |
|---|---|---|---|
| `wavefront_shade` | 8.79 | 29% | Taichi kernel (8 bounce iterations/frame) |
| host-torch sheet chain | ~5.2 | 17% | `compact_sheets` 2.3, `window pairs` 1.14, `lexsort` 0.91, `shade class` 0.52, `fragment sort` 0.41, `prim split` 0.36 |
| `raster_shadow_trace` | 4.47 | 15% | Taichi kernel, 1 launch/frame |
| `wavefront_traverse_events` | 3.74 | 13% | Taichi kernel |
| `raster_tri_count` + `raster_tri_write` | 1.36 | 5% | Taichi kernels |
| prep + arena preflight | ~1.6 | 5% | |
| post-process | 0.71 | 2% | |

The stage table's raw numbers that led there:
`wavefront_loop` incl 26.015 / excl 13.218, of which 8.788 + 3.740 + 0.205 are
its own kernels, leaving ~0.5 s of real host work; `raster: sparse resolve`
excl 4.861 is `raster_shadow_trace` 4.470 + `sheet_resolve_shade` 0.302 + ~0.09 s
of host.

An independent `torch.profiler` capture of 6 frames agrees: 5.278 s of CUDA
time, 63% of it in those three kernels, ~30% in torch ops (sorts 211 ms,
`index_select` 175 ms, `gather` 172 ms, `cat` 167 ms, `copy_` 213 ms over 8261
calls, `stack` 154 ms, `fill_` 116 ms). Self CPU over the same window is 5.868 s,
so the host is about as busy as the device — prep on the worker thread accounts
for most of that.

### PREVIEW is preparation-bound

```
ray traced render total            5    3.245 incl  (52.0%)
Scene.get_batch_of_primitives      3    2.322 incl  (37.2%)
arena preflight (batch)            3    1.519 incl  (24.3%)   <-- serial on the render thread by design
AttributeTimeline.get          22702    0.807      (12.9%)
surfaces: get_render_primitives_batched  6  0.776  (12.4%)
```

Prep and render are essentially serial (2.32 + 3.25 + preflight ≈ 6.25), so
prefetch is not hiding prep here.

A **local** cProfile of one 5-frame prep call
(`scratch_perf/r2/probe_prep_cprofile.py`, CPU-only so it transfers) says
`_update_neural_net_idle` is 0.241 s of 0.761 s — a Python loop over 15 neurons
and 80 synapses — and `Surface.get_render_primitives_batched` another 0.296 s.

## Work in flight

Three Ox Alpha agents, one per git worktree so they cannot collide:

| tree | branch | brief | target |
|---|---|---|---|
| `D:\algan` | master | `scratch_perf/r2/ox/brief_wavefront_host.md` | instrument + cut `wavefront_loop`'s 13.2 s excl |
| `D:\algan_wt_prep` | `perf/r2-prep` | `scratch_perf/r2/ox/brief_prep_timeline.md` | batch the idle updater; cheapen `AttributeTimeline.get` |
| `D:\algan_wt_sheet` | `perf/r2-sheet` | `scratch_perf/r2/ox/brief_sparse_resolve.md` | the 4.86 s host chain in `shade_sparse_raster_coverage` |

`D:\algan_wt_lab` (`perf/r2-lab`) is mine: probes and notebook generation, kept
quiescent so a patch can be cut from a known-clean tree.

## Kaggle trap: the batch GPU session limit is 2, and a queued run holds a slot

`get_notebook_session_status` returns `{}` for a **queued** run — indistinguishable
from "no session". Two saves in a row therefore look like nothing happening at
all, and `get_accelerator_quota` confirms it: `time_used` does not move and
`time_reserved` stays `0s` while both slots are held.

The signal that says what is really going on comes from trying to save a
*different* notebook:

```
{"error": "Maximum batch GPU session count of 2 reached."}
```

So: **do not re-save to "retry" a run that looks stuck** — each save consumes
another of the two slots, and there is no MCP tool that lists sessions or their
ids, so `cancel_notebook_session` (which wants an integer `kernelSessionId`)
cannot be reached from the API alone. The session id does appear in the output
download URL of a *completed* run (`kaggleusercontent.com/kf/<session id>/...`),
which is no help for a stuck one.

Wait for the slots to clear instead, and batch as much as possible into each
run.

## Ablation: what the UHD render is actually spending time on (T4, tag `r2abl`)

Five arms of `benchmarks/performance/nn_ablation.py`, one process each (a
`ti.static` gate is baked at compile time, so arms cannot share a process), warm
RUN 2, 30 frames at 3840x2160:

| arm | warm | vs base | what it removes |
|---|---|---|---|
| `base` | 27.92 s | — | |
| `noshadow` | **16.25 s** | **-42%** | every shadow ray: `raster_shadow_trace` *and* `wavefront_shade`'s inline block |
| `b0` (`max_bounces=0`) | 16.40 s | -41% | the whole continuation bounce loop |
| `b1` (`max_bounces=1`) | 22.02 s | -21% | bounces 2..8 |
| `sec1` (`ALGAN_ANALYTIC_AA_SECONDARY=1`) | 24.97 s | -11% | 3 of every 4 continuation rays per reflective fragment |

The same two arms on the GTX 1050 at PREVIEW: base 29.35 s, `noshadow` 20.59 s
(-30%).

**Shadows are the single biggest lever: 42% of the UHD render.** And the two big
arms interact in a way worth spelling out — `raster_shadow_trace` alone is
~4.5 s, so of the 11.67 s that `noshadow` saves, roughly **7 s is shadow rays
fired from inside the bounce loop**, i.e. about 60% of the bounce loop's own
11.5 s. Optimising shadow rays therefore pays twice.

Per-bounce instrumentation (the sheet route's loop, PREVIEW on the 1050) says
the tail is not where the money is: bounce 0 holds **52%** of all
`wavefront_shade` time, bounces 0-2 hold 93%, and bounces 3-7 together are ~7%.
So a throughput cutoff that kills the tail is worth a few percent, not tens.

**Always pass a modest `sessionTimeoutSeconds`** (3600 is plenty for these
arms). A queued or wedged run holds one of the two batch GPU slots for its whole
timeout, and at the 9-hour default that is a day of measurement box gone.
Nothing notifies you when a run ends, either — arm a timer and poll
`get_notebook_session_status`.

## Round 1 of agents: what shipped and what did not

Integrated at `9f3fdb90` on `perf/r2-lab`, verified byte-identical against HEAD
(nn PREVIEW, lossless `libx264rgb -qp 0`, 50/50 frames, worst channel diff 0)
and `--fast` 277 passed. Locally the same render went 105.4 s -> 83.7 s.

| agent | outcome |
|---|---|
| bounce-loop instrumentation | **shipped** — per-iteration table; verified my `charge_kernel_to_parent` fix; no optimization found worth its price |
| prep / idle updater | **shipped** — batch preparation -21.8%, timeline reads 2841 -> 1436 per batch |
| receiver-facing shadow cull | **not shipped** — see below |

### The shadow cull is a measured no-op — do not re-derive it

The premise (surfaces facing away from a light still pay a full shadow fan) is
false on this branch: commit `f142f72d` put a **per-sample horizon guard** into
both trace sites (`raster_taichi.py:3047`, `wavefront_kernels_taichi.py:2984`),
so those BVH marches are already skipped. Implementing the whole-fan cull on top
removed 9% of *entered fans* but **0% of marched rays** — one ray over a whole
video — and kernel self-time was flat. The patch and the full stage-by-stage
soundness audit are kept at `scratch_perf/r2/patches/` and
`scratch_perf/r2/ox/REPORT_shadow_facing_cull.md`; the audit itself is worth
keeping (it establishes which stages carry `max(N·L, 0)` on every
vis-multiplied term, and that `event_snrm` holds the already-oriented normal).

The corollary matters for planning: **shadows being 42% of the UHD render is not
waste.** Those rays are marched because they contribute. Cutting them needs a
different idea, not a better cull.

## Round 2 of agents (in flight)

| tree | branch | target | share |
|---|---|---|---|
| `algan_wt_prep` | `perf/r2-preflight` | overlap the arena preflight with the render | PREVIEW 24% |
| `algan_wt_sheet` | `perf/r2-sheetchain` | the sheet route's remaining host-torch passes | UHD ~17% |
| `algan_wt_shade` | `perf/r2-shade` | classify and cut the 1.6 M continuation rays/frame | UHD 41% is the bounce loop |

All three start from `9f3fdb90` so their diffs compose.

## Getting code to Kaggle: the notebook body is the only channel, and it has a size limit

This box cannot push to GitHub (no credential helper, `gh` absent, the SSH key
is not accepted), so the Kaggle notebook seeds from the public tip and the local
changes ride inline as a gzip+base64 `git diff`. That works — up to a point.

**An 18 kB base64 payload arrived as 11 kB.** The notebook's
`assert len(PATCH_B64) == ...` caught it immediately (cost: one minute of GPU),
which is why every payload carries a length *and* a sha256 assert before it
touches anything. Put those asserts in first, always.

Two mechanisms handle it:

* `scratch_perf/kaggle/make_chunks.py` splits a payload into ~6 kB chunk
  notebooks. Each writes one `part.NNN` into the persisted payload store and
  verifies its own sha; the render notebook assembles `part.*` when it finds no
  `overlay.patch`. Chunk notebooks need no GPU.
* `--snapshot NAME` makes the render notebook **commit and tag** the overlaid
  tree inside the Kaggle clone, and `--from-tag NAME` starts a later run from
  it. So a payload is uploaded in full once, and everything after it is a small
  delta against that snapshot — which fits inline in one piece.

## Review of the integrated diff

The tracer change is 758 diff lines, almost all of it reindentation. Checked by
splitting the diff into lines that appear on both sides (pure reindentation) and
lines that appear on only one: the only genuine additions are the
`with _stage(...)` wrappers, `_BOUNCE_STAGE_CAP`, the bounce label, a deferred
import of `stage`, comments, and one line-wrap of the `pix_accum` allocation.
No control flow moved.

The `copy=False` changes are the ones that could corrupt silently, so each was
checked against its getter: `mob.location` / `mob.basis` are literally
`get_animated_attribute(...)` with the copy default, and `location.setter`
already reads `copy=False` internally — so the new call sites read the same
values, just uncopied, and every one of them feeds out-of-place arithmetic
before any write. `get_upwards_direction()` is
`F.normalize(unsquish(basis, -1, 3)[..., 1, :], p=2, dim=-1)`, which is exactly
what replaced it.

**Caveat on the test evidence**: `tests/unit_tests` under four concurrent
GPU-using agents is not a clean signal. A run of the integrated tree showed one
failure in `test_glossy_prefilter.py`, whose suite includes
`test_prefiltered_route_is_deterministic_across_processes` — and free VRAM at
job start decides the arena size, hence tile sizes, which is a documented
cross-process mover unless `available_memory_override` is pinned. Re-run it on a
quiet box before treating it as a regression.

## Verification debt carried into the next session

The full `tests/unit_tests` suite has **not** been run to completion on either
the round-2 integration (`9f3fdb90`) or the sheet-chain branch. Two attempts were
stopped for memory pressure: the suite spawns two ~2 GB render subprocesses at
once (its cross-process determinism test runs both arms concurrently), which on
this 4 GB / shared box took free RAM under 1.5 GB while agents were also
rendering.

What *is* established: `--fast` (277 tests, including the pixel-compared render)
passes on the integration, and the one failure the full suite produced —
`test_glossy_prefilter.py` — was independently confirmed by the sheet-chain
agent as **pre-existing cross-process nondeterminism** ("identical single pixel
(296, 820), identical delta"), not a regression.

Run the full suite on a quiet box before merging anything from this round.

## Full unit suite: PASSES on the integration (2026-08-26)

`scratch_perf/r2/run_suite_chunked.sh` on `9f3fdb90`, box otherwise idle:
**2077 passed, 132 skipped, 0 failed**, 14/14 chunks exit 0, 45 min wall.
Same pass count master reports.

The earlier `test_glossy_prefilter.py` failure is now positively explained
rather than merely suspected: it is chunk 5's first file, it took 28 minutes on
its own (its cross-process determinism renders), and it **passes** on a quiet
box. Contention, not the integration.
