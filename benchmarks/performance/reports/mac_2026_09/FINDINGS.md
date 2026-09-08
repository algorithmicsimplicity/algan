# Metal render on the Mac GPU: everything this round established

> **2026-09-08 correction:** [REORDER_FIX.md](REORDER_FIX.md) identifies an MPS
> output-offset bug that corrupts the ray queue and arena. The repaired queue
> reduces traversal from 275 to 4–5 passes per chunk and completes repeated
> uncapped UHD renders. This historical report's performance and virtualization
> conclusions must be read with that correction.

One self-contained record of a session spent optimising `nn_scene_UHD.py` on
Apple silicon. It covers what was fixed, what was measured, the one substantial
performance finding, and an unresolved failure that the same work introduced.
It is written so that someone picking this up cold needs nothing else.

**Read §0 first.** It was written last and it changes how to read §4 and §5.
Then §8 if you are deciding what to ship, §7 for the open failure.

---

## 0. The headline, established last: this box's GPU loses to its own CPU

The same scene, the same machine, the same arena, the same instrument, one
variable — the render device:

| | CPU (`mac-cpu`) | Metal (job 52, warm) |
| --- | ---: | ---: |
| total | **140.0 s** | 1010.3 s |
| arena | 1147 MB | 1147 MB |
| batches / chunks | 2 / 18 | 2 / 18 |
| steady-state chunk | **4.2 s** | 55.4 s |
| zero-copy imports per chunk | 0 | 73,044 |

**The CPU is 7.2× faster overall and 13.1× faster per chunk**, on a box whose
GPU has a 3× compute advantage (~1.3 TFLOP/s against ~450 GFLOP/s).

The launch arithmetic accounts for it:

```
73,044 imports/chunk × 432 µs  (this box's dispatch) = 31.6 s of a 55.4 s chunk
73,044 imports/chunk ×   2 µs  (a physical Mac)      =  0.15 s
```

So **the wall times in §4 are substantially a measurement of this runner's
virtualisation tax, not of the renderer.** The caveat under §1 said as much from
the start; it was quoted and then not applied, and most of this round ranked
work by those wall times anyway. Three consequences:

1. **Do not optimise against Mac-runner wall time.** Re-measure on a physical
   Mac or the Kaggle T4 before trusting any GPU-versus-baseline number here.
2. **§5 survives, because it is a launch-count finding** — fewer launches,
   faster — and it is internally consistent. Its *magnitude* is a property of
   this box.
3. **The real target is the launch count itself.** ~73,000 dispatches a chunk is
   design-level pressure that costs on any device with a non-trivial submission
   cost; on this one it is most of the render. That should displace the arena
   work at the top of any ranked list.

---

## 1. The workload and the box

* **Scene**: `benchmarks/performance/nn_scene_UHD.py` — a `NeuralNetMLPV3`, an
  `ImageMob` of the globe, and a text label, animated for 0.3 s. 18 frames at
  3840×2160, `shadows=False`, `libx264 -preset ultrafast`.
* **Box**: GitHub's `macos-latest` Apple-silicon runner. A *virtualised* M1
  (`VirtualMac2,1`), 3 CPUs, **7 GB unified memory**,
  `recommendedMaxWorkingSetSize = 4.67 GB`. The GPU is real hardware:
  ~1.3 TFLOP/s f32 sustained against ~450 GFLOP/s on its own CPU.
* **Compiler**: patched Quadrants 1.3.1 (`quadrants_build.yaml` run
  `33850787142`), zero-copy Metal ndarrays live.
* **Harness**: `.github/workflows/run_on_mac.yaml`, driven by
  `.github/gpu-run/mac.json`. See `agent_guidance/gpu_harnesses.md`.

> **Caveat that constrains every conclusion here.** This box's *compute* numbers
> are sound; its **per-launch numbers are not**. One synchronized dispatch
> measures **432 µs** here against 2.0 µs on its own CPU, and host↔device copies
> run at 6.0/3.2 GB/s. That is a virtualisation tax no physical Mac pays. Where
> a finding below depends on launch cost, it is flagged.

---

## 2. Where it started: the workload did not run at all

The first attempt died before rendering a frame:

```
algan.rendering.primitives.primitive.OutOfRenderMemory:
Insufficient memory to ray trace a single frame.
```

on a machine with 4.67 GB free. Cause: `get_num_available_bytes` clamped its
**MPS** branch to `min(free, 1 GiB)`. The render arena is
`rendering_memory_fraction` (0.4) of that, so **every Metal render on every Mac
sized its arena at 410 MB** — a 128 GB Mac Studio and an 8 GB Air got the same
arena. The clamp had no recorded rationale; the CUDA branch beside it reports
real free bytes, and `_render_device_pool_bytes` was *already* sizing
out-of-arena budgets from `recommendedMaxWorkingSetSize` on the same device.

Removing it is what makes everything below possible, and it is the largest
Mac-side win in the round: the difference between *cannot render 4K* and
*renders 4K*.

It also exposed two latent defects, because a frame window wider than one frame
per render chunk had never occurred on Metal before (§3).

---

## 3. Landed fixes

### 3.1 Bit-identical optimisations (no output change)

| Change | Where |
| --- | --- |
| Narrowed sheet lexsort keys; folded three readbacks into one; dropped two sorts | `rendering/raytracing/sheets.py` |
| Blocked the shading-class table's frame axis so a wide chunk cannot blow the transient | `sheets.py` `_shade_class` |
| Blocked the prim band rule's slope table on the same axis | `sheets.py` `_prim_split_after` |
| Narrowed the compaction's two probe reductions to widths Metal accepts | `sheets.py` |
| Keyed a Taichi ndarray argument by its base class, so Metal hits the source-key cache index | `utils/taichi_source_key.py` |

### 3.2 Metal correctness fixes

| Defect | Fix |
| --- | --- |
| Unbounded `[frames, N, 9]` tables → 6.45 GB allocation | Blocked the frame axis (above), plus `_FRAME_TABLE_BUDGET` / `_check_frame_table` guards |
| `IndexError: gl_bounds[gl_frame + 1]` from corrupt fragment keys | Same root cause |
| Corrupt packed 64-bit keys under the raster write-pass pair compaction | `raster_write_compact_active()` returns `False` on MPS (`rendering/raytracing/settings.py`) |
| `NotImplementedError: aten::index_copy.out` on MPS | `index_copy_rows()` in `rendering/mps_compat.py` — advanced-index assignment under `mps_friendly()` |

**Unfinished thread:** `_check_write_compaction` (gated on
`ALGAN_RASTER_KEY_CHECK`) printed nothing, so the compaction's *selection* is
correct and the true root cause of the key corruption is **not** identified. It
is gated off on Metal, not fixed.

### 3.3 The CPU branch had the same defect as the Metal one

`max_cpu_memory_used` defaulted to a flat **2 GB**, and the arena is
`rendering_memory_fraction` (0.4) of it, so **every** CPU machine got a 0.75 GB
arena — too small for one 4K frame. A UHD CPU render therefore died with
"Insufficient memory to ray trace a single frame" on the 7 GB runner, and would
have on a 512 GB workstation identically. That is structurally the same defect
as the 1 GiB Metal clamp in §2, in the branch nobody had pointed a 4K render at,
and it was found the same way: by rendering the scene and reading the error.

It now defaults to a share of the machine's RAM (`_CPU_MEMORY_SHARE`, 0.4),
never below the old 2 GB floor, falling back to that constant where memory
cannot be read. It remains an explicit setting for anyone pinning it.

### 3.4 Sizing (the subject of §5–§7)

* MPS 1 GiB clamp removed; branch sizes from the device.
* `_gpu_memory_pressure` answers per device instead of returning `True` for
  everything non-CUDA (that made all nineteen `force_gc=False` reclaim sites pay
  a full `gc.collect()` on Metal).
* `release_torch_memory`'s MPS block gated on the same pressure as the CUDA one.
* `get_num_available_bytes` MPS branch: clear import cache → `empty_cache()` →
  measure **`current_allocated_memory`**, capped at `0.4 ×` total RAM.

### 3.5 Test status

* `pytest -q --fast`: **537 passed** on the merged tree (36 s on a second run;
  the first pays a cold compile for master's new kernel).
* Full suite before the merge: **3383 passed, 146 skipped, 0 failures**.
* **Caveat:** `tests/full_renders` skipped 6 of 7 scenes ("no cpu_eager
  full-render baselines are available"), so the dense PN-geometry pixel
  comparison never ran. A tessellation/projection regression would be invisible
  to everything that did run.

---

## 4. Measured performance

**`nn_scene_UHD.py`, 18 frames at 3840×2160, shadows off**, via
`benchmarks/_mps_warm_regression.py` (two renders in one process, no profiler
hooks — its synchronising hooks would distort the wall time being measured).

| job | arena (cold/warm) | cold | warm | outcome |
| --- | --- | ---: | ---: | --- |
| 34 | 1898 / 1212 MB | 747.0 s | 1522.8 s | completed; exit 1 at shutdown |
| 36 | 1898 / 1226 MB | 810.4 s | 967.9 s | completed |
| 37 | 1898 / 1226 MB | 816.3 s | 1026.4 s | completed |
| 38 (`pin-arena`) | 1898 / 1898 MB | 1105.3 s [^stall] | **568.5 s** | completed |
| 39 (live-bytes) | 1911 MB | **649.8 s** | — | **killed** at warm chunk 14 |
| 47 (0.4 × RAM cap) | 1147 MB | 1019.5 s | — | killed at a 30-min timeout; **unclassified**, see §7.0 |
| **52 (post-merge, cap, 55-min timeout)** | **1147 / 1147 MB** | **1146.8 s** | **1010.3 s** | **completed — warm 0.88× cold** |
| **55 `mac-cpu` (post-merge, matched arena)** | **1147 MB** | **140.0 s** | — | **completed — see §0** |
| 55 `mac-mps` (cap off) | ~1.9 GB | — | — | crashed: MPS SymInt, §7.5 |
| 56 (cap off, `torch.compile` off) | ~1.9 GB | — | — | **wedged** at chunk 3, §7.6 |

[^stall]: Job 38's cold pass contained one 455.6 s chunk against 25–42 s for
every other chunk. A stall, not a baseline — and see §7, since it may be the
same failure as the wedges in a form that recovered.

**The cap costs time, and that is what a 30-minute timeout could not fit.** At
1147 MB a cold pass is ~1020 s against 649.8 s at 1911 MB, so a two-render job
needs ~34 minutes. Jobs sized against the old 20-minute expectation were killed
mid-flight; see §7.0.

### 4.1 Job 52: the pair completing, which settles several things at once

The first two-render UHD job to finish since the cap landed, on master's
optimised batching, with a timeout long enough to fit it (36.6 min of work).

| | cold | warm |
| --- | ---: | ---: |
| wall | 1146.8 s | **1010.3 s** |
| arena | 1147 MB | 1147 MB |
| batches | 2 | 2 |
| chunks | 18 | 18 |
| mean chunk | — | 55.4 s |
| in-chunk peak | 4.93 G | **5.61 G** |

* **The warm regression is gone.** Warm is **0.88× cold** — faster, which is the
  direction it belongs, since all a warm render saves is the kernel compile.
  Both renders size the *same* 1147 MB arena; the collapse to ~1226 MB that
  drove §5 does not happen.
* **Master's batching cut 12 batches to 2.** That is `808de65`, not this branch.
* **The peak reached 5.61 G — nearly a gigabyte over the 4.67 G
  `recommendedMaxWorkingSetSize` — and nothing failed.** See §7.4; this is the
  observation that ends the over-commit theory.

**PREVIEW (704×396), four renders in one process** — the control:

| render | wall | arena |
| ---: | ---: | ---: |
| 1 | 81.2 s | 1898 MB |
| 2 | **12.9 s** | 1828 MB |
| 3 | 12.7 s | 1827 MB |
| 4 | 11.1 s | 1819 MB |

Warm is **6.3× faster** with a stable arena. There is no general "second render
is slow" defect on this backend, and no instability at this size.

---

## 5. The finding: `driver_allocated_memory` is a high-water mark

This is the one substantial performance result of the round.

**The symptom.** At UHD the second render in a process was 1.19×, 1.26× and
2.04× *slower* than the first across three jobs — while its arena collapsed to
1212/1226/1226 MB from 1898 MB, reproducibly. The wall-time ratio is noisy; the
arena collapse is not.

**The chain, measured.**

| | cold | warm | ratio |
| --- | ---: | ---: | ---: |
| arena | 1898 MB | 1226 MB | 0.65× |
| zero-copy imports per chunk | 12,388 | 48,766 | 3.94× |
| cost per chunk (2–10) | 31.2 s | 55.2 s | 1.77× |

36,378 extra imports against a 24.0 s per-chunk gap is **0.66 ms per import** —
this box's dispatch cost. The warm render does no extra work; it does the same
work in four times as many launches. *(Launch-bound, so the magnitude is a
property of this virtualised box; the mechanism is not.)*

**Why the arena shrank.** `driver_allocated_memory` does not come back down
after `empty_cache()` on Metal. Within one pass it climbs monotonically
**2.91 → 4.49 G while live bytes hold flat at 1.91 G**, through seven pressured
drains a chunk in the last third that never move it. Sizing from it charges each
render for the previous one's peak.

**The A/B that proves it.** `_mps_warm_regression.py … pin-arena` holds every
later render to the first's free-byte figure:

* warm wall **1522.8/967.9/1026.4 s → 568.5 s**
* warm imports **48,766 → 12,388 per chunk, exactly cold's number**
* warm becomes **1.9× faster than cold**, the direction a warm render belongs

`driver_allocated` reached **4.76 G** doing it — past
`recommendedMaxWorkingSetSize` — with no failure, so on unified memory that
ceiling is advisory and the blocks behind the driver figure are reusable.

**The fix**: measure `current_allocated_memory` after the drain. Known gap:
Taichi's allocations outside torch are not in that figure.

---

## 6. Hypotheses raised and refuted

Recorded because each was plausible, each was killed by a measurement rather
than an argument, and someone will propose them again.

| # | Hypothesis | Killed by |
| --- | --- | --- |
| 1 | The preflight binary search re-pays projection/merge/BVH per rejected probe | Warm batch preparation is **8.4 s against cold's 28.2 s** — 3.4× *faster*. The cost is inside the chunks. |
| 2 | A gc storm: `_gpu_memory_pressure` reads `driver_allocated`, pinned above the 0.8 threshold, so all nineteen `force_gc=False` sites collect | `release_torch_memory` accounts for **0.1–1.9 s a chunk** against chunks costing 25–79 s; cache clears rise 1 → 7 a chunk without moving the import count off 12,388. |
| 3 | A memory knee at the 0.8 pressure threshold (cold chunks 11–12 spike to 75.6/78.9 s exactly as `driver` crosses 3.74 G) | Chunks 13–18 fall back to 26–44 s while still 7/7 pressured at 4.0–4.5 G, and the same two chunks spike in a job with a different memory profile. It is a **batch boundary**. |
| 4 | The render blocks writing to the ffmpeg pipe (an orphaned `ffmpeg-macos-aa` survives every wedge) | The heartbeat is a **separate daemon thread**; it froze too. A Python-level pipe block cannot stop it. The orphan is the video writer idling for frames that stopped coming — a consequence, not a cause. |
| 5 | A C call holding the GIL | Unnecessary once system-wide memory pressure is on the table, which stalls every thread. Not independently evidenced. |
| 6 | Host exhaustion, with `available + driver_allocated` a fixed ~5.4 G budget | Across the full trace that sum ranges **3.87 to 6.19 G**. Three adjacent rows fit; the model was fitted to them and shipped. Sizing from it produced an 80 MB arena and `OutOfRenderMemory` twice (jobs 43, 45). |
| 7 | Arena size governs the in-chunk peak | Job 47: peak **4.90 G cold / 5.04 G warm on a 1147 MB arena** — the same peak a 1898 MB arena reached. Arena size does not move the peak. |

**Method note.** Hypotheses 1, 2, 3, 6 and 7 were each committed to the tree
before being tested. Three shipped as default behaviour and had to be backed
out. The instrumentation in §9 exists because of that, and the discipline that
was missing throughout is: *measure the quantity the hypothesis is about, before
changing behaviour that depends on it.*

---

## 7. The open problem: the wedge

### 7.0 A classification error to read first

**Jobs 50 and 51 were NOT wedges.** They were rendering and printing the whole
time, and were killed by a `timeout_minutes` I had lowered from 50 to 30 on the
reasoning that "a wedge is visible within a minute, so there is nothing to learn
from the remaining half hour". That reasoning is fine for a wedge and wrong for
a slow run: with the arena capped at 1147 MB a cold pass alone takes ~1020 s, so
a pair needs ~34 minutes and a 30-minute timeout cannot fit one. I then read the
kill as a wedge, twice.

**Job 47 is now uncertain** for the same reason and should be treated as
unclassified rather than as evidence.

What that leaves as genuinely wedged is jobs **40, 41, 42 and 44** — each with
minutes to tens of minutes of total silence observed live, job 41 with 38
minutes of it. Every one of those ran with a ~1.4-1.9 GB arena, *before* the
0.4 x total-RAM cap. **No confirmed wedge has yet occurred with the cap in
place**, which is a materially better position than the rest of this section
was written to describe — and the reason the next run is a long-timeout one.

### 7.1 What happens

The process stops making progress and never resumes. It is **alive** — the
runner terminates it as an orphan at the step timeout, alongside a live
`ffmpeg-macos-aa` child. No traceback, no exception, no output of any kind,
including from a daemon thread whose entire body is `sleep(2)` and a `print`.

Sometimes it is a kill instead (job 39: exit code 1, no traceback). Once it
recovered on its own after 455.6 s (job 38).

### 7.2 Every occurrence

| job | configuration | wedged at | `driver` | in-chunk peak | `host_free` |
| --- | --- | --- | ---: | ---: | ---: |
| 38 | pin-arena 1898 MB | cold chunk 11 (**recovered**, 455.6 s) | — | — | — |
| 39 | live bytes, 1911 MB | warm chunk 14 (**killed**) | 4.68 G | — | — |
| 40 | live bytes, 1911 MB | cold chunk 5 | 3.14 G | — | — |
| 41 | live bytes, 1911 MB | after cold chunk 15 | 4.07 G | **4.85–4.87 G** | — |
| 42 | headroom 0.25, ~1.4 GB | cold chunk 1 | 3.56 G | — | — |
| 44 | no cap, 1911 MB | after chunk 3 | 2.96 G | 4.05 G | 1.19 G |
| 47 | 0.4 × RAM, 1147 MB | *unclassified* — 30-min timeout, see §7.0 | 3.87 G | 4.90/5.04 G | 1.12 G |
| ~~50~~ | post-merge, 1147 MB | **not a wedge** — progressing when killed | 2.40 G | 4.40 G | 1.19 G |
| ~~51~~ | post-merge, `torch.compile` off, cap ON | **not a wedge** — progressing when killed | — | — | — |
| 56 | post-merge, `torch.compile` off, **cap OFF** | after chunk 3 (36 min silent) | 2.96 G | 4.04 G | 1.51 G |

### 7.3 What is established

* **It is not arena size.** It happens at 1147, ~1400, 1898 and 1911 MB.
* **It is not the GPU working set alone.** Jobs 40 and 42 wedged at 3.14 G and
  3.56 G, well under the 4.67 G recommendation.
* **The in-chunk peak systematically exceeds the recommendation** — 4.40 to
  5.04 G — *regardless of arena size*, and the chunk-boundary readings
  understate it by ~1.1 G. Anything calibrated against boundary readings is
  calibrated against a trough.
* **Memory pressure is real and observed.** In job 44 the process's own resident
  set was *evicted while it worked*: **1.29 → 0.64 → 0.36 → 0.30 → 0.22 G**.
  That is macOS paging it out, not the render freeing anything.
* **But the machine is not out of memory at the moment it wedges.** `host_free`
  sits at **1.1–1.2 G** in jobs 44, 47 and 50. So exhaustion is not a sufficient
  explanation.
* **Never reproduced on CPU, and never at PREVIEW** (four renders, no
  instability, pool peaked at 3.16 G).
* **It has not been confirmed since the cap landed.** Jobs 50 and 51 both ran to
  their timeout still progressing (§7.0), so the claim that it "survived master's
  batching optimisation" is withdrawn — that job never wedged.

### 7.4 Job 52: the over-commit theory is dead

Job 52 ran both renders to completion with the in-chunk peak at **5.53–5.61 G
against a 4.67 G recommendation** — the highest figure recorded this round,
about 0.7 G above the peak in jobs that wedged — and nothing went wrong.
`host_free` sat at 1.2–2.5 G throughout, the same band as the wedged jobs.

So exceeding `recommendedMaxWorkingSetSize` is **normal on this box, not a
failure mode**, and neither the peak nor the host-free figure separates a
healthy run from a wedged one. Every hypothesis in this round that rested on
over-commit — hypotheses 6 and 7 in §6, and the reasoning behind the reserve
and the cap — is refuted by this run.

What actually changed between the wedges and job 52 is two things at once:
master's batching (`808de65`, 12 batches → 2) and the arena cap. The cap is
**not** established as the fix; it is confounded. §7.6 says how to separate them.

### 7.5 An uncapped post-merge render crashes on a real MPS bug

The MPS arm of job 55 (`ALGAN_MPS_HOST_SHARE=0`, ~1.9 GB arena) did not wedge —
it **crashed**, in `raster_pipeline._pair_expand_rows`:

```
rows = torch.empty((total, 8), dtype=torch.int32, device=device)
RuntimeError: RegisterMPS_0.cpp:7210:
SymIntArrayRef expected to contain only concrete integers
```

`total` is already `int(counts.sum().item())`, so a plain Python int reaches
that line in eager mode. A **SymInt** reaching it means the region was being
traced with dynamic shapes — i.e. `torch.compile`, which was on for this job and
whose Metal backend warns it is a prototype in every run and *fails* on
`_triangle_projection_fused` in every run.

**Confirmed by job 56.** The same configuration with `ALGAN_TORCH_COMPILE=0`
produced **no SymInt crash at all** — it rendered past the point job 55 died at.
So the prototype Metal codegen path owns this crash, and the fix is to pin
`torch_compile` off on MPS (or to stop `_pair_expand_rows` being traced).

That is the one hypothesis this round proposed and then confirmed rather than
retracted. It is **not** the wedge, though: job 56 wedged anyway (§7.6).

### 7.6 Answered by job 56: the cap is load-bearing

Job 56 ran the pair post-merge with `ALGAN_MPS_HOST_SHARE=0` (~1.9 GB arena) and
`ALGAN_TORCH_COMPILE=0`. It **wedged**:

```
22:50:34  chunk 3 begins at + 171.6 s | ... peak 4.04G | driver=2.96G | host_free=1.51G
23:26:45  ##[error]The operation was canceled.
```

**36 minutes of total silence after chunk 3**, the heartbeat included — the
signature established in §7.1, and the first wedge observed live rather than
inferred from a timeout. Terminated as an orphan with an ffmpeg child alive.

**The attribution is clean**, because two jobs differ in the cap alone:

| job | `torch.compile` | cap | outcome |
| --- | --- | --- | --- |
| 51 | off | **on** | progressing when its short timeout killed it |
| 52 | on | **on** | **completed**, cold 1146.8 s / warm 1010.3 s |
| 56 | off | **off** | **wedged** at chunk 3 |

51 and 56 hold `torch.compile` off and differ only in the cap, so the cap is
what separates a healthy run from a wedged one. **Keep it.** Master's batching
alone does not prevent the wedge, and `torch.compile` neither causes nor
prevents it.

The cost is real and should be stated with the decision: ~1147 MB against
~1911 MB, and cold 1146.8 s against 649.8 s, on this 7 GB box. On a machine with
memory the cap never binds (§3.4), so it costs nothing there.

Still unexplained: *why* a larger arena wedges. Nothing in §7.3 separates the
runs — job 56 wedged at `driver` 2.96 G, `host_free` 1.51 G and a 4.04 G peak,
all mid-range. The cap is a mitigation, not a diagnosis.

### 7.7 Untested hypotheses, in the order worth trying

0. **Already tested, and negative.** Job 51 ran with
   `ALGAN_TORCH_COMPILE=0` and behaved like job 50 with it on: both progressed
   until their timeout. The prototype codegen path is therefore **not** what
   distinguishes a wedging run from a healthy one, and hypothesis 1 below is
   retired as the leading candidate. It is left on record because the failing
   compile is still worth removing on its own account.

1. **`torch.compile` on Metal.** Every single run logs
   `torch.compile for Metal is an early protoype` from
   `torch/_inductor/codegen/mps.py`, followed by
   `torch.compile failed for _triangle_projection_fused (NameError: name 'ps0'
   is not defined)`. A failing prototype codegen path runs in every wedged job.
   **Test:** `SETTINGS.computing.torch_compile=False`. Cheapest decisive
   experiment available and it has never been run.
2. **The video encoder.** ffmpeg is alive in every wedge and holds 24 MB frames.
   **Test:** render with frame output to disk instead of streaming, or
   `save_frame` in a loop, and see whether the wedge survives.
3. **A Metal command buffer that never completes.** Consistent with the total
   silence, and would not be visible to any Python-level instrument.
   **Test:** `sample`/`spindump` the wedged process on the runner — the harness
   can run it after the render, or from a second background process.
4. **The macOS compressor.** `host_free` around 1.1 G with an evicted RSS is
   consistent with heavy compression; a VM with little or no swap has nowhere to
   go. **Test:** `vm_stat` / `memory_pressure` sampled beside the existing
   counters.
5. **Resolution threshold.** Never seen at PREVIEW, always at UHD.
   **Test:** HD and MD, to find where it starts.

### 7.8 A loose end

A `resource_tracker: There appear to be 1 leaked semaphore objects` warning
appears at shutdown in exactly the jobs that **died** (34, 39) and in none that
completed. Algan itself touches `multiprocessing` only for `cpu_count()`, and a
CPU render leaves `active_children() == []` with the tracker never started. The
likeliest source is torch inductor's async-compile `ProcessPoolExecutor` — which
ties back to hypothesis 1. It cannot cause a slowdown that precedes it, but it
may be a marker of the hard kill.

---

## 8. Recommendation

**Ship:** §3.1, §3.2 and the removal of the 1 GiB clamp. These are what make 4K
render on Metal at all, and they are independent of everything unresolved.

**Ship with the cap:** the `current_allocated_memory` fix (§5) together with the
`0.4 × total RAM` cap. The fix is correct — sizing from a high-water mark
charges each render for the last one's peak — but on its own it hands a 7 GB box
a 1.9 GB arena, and that is where the instability concentrated.

**Update from job 52:** the warm regression the §5 fix targets is gone in
practice — warm now runs at 0.88× cold with both renders on an identical arena.
That is the outcome the fix was for, at a cold cost the cap imposes.

**Do not claim the 568.5 s warm figure.** It was measured with a pinned 1.9 GB
arena on a box that cannot hold one safely. What survives is the real defect and
its removal: both renders now size the same arena, so the collapse to 1226 MB
and its 4× launch count are gone. On a Mac with real memory the cap never binds
and the arena is genuinely larger.

**Cost of the cap on this box:** cold 1019.5 s at a 1147 MB arena against
649.8 s at 1911 MB — about 57%. That is the price of not wedging, on a 7 GB
machine. It should be re-measured on a real Mac, where it does not apply.

**Keep the arena cap.** §7.6 settles it: with the cap off the render wedges at
chunk 3, with it on the pair completes. Two jobs differ in the cap alone, so the
attribution is clean. It costs ~500 s a render on a 7 GB box and nothing at all
on a machine with memory.

**`torch_compile` is now pinned off on MPS** (landed). `'auto'` declines there
via `_auto_declines_this_device` in `utils/torch_compile.py`, while
`torch_compile=True` and `ALGAN_TORCH_COMPILE=1` still force it on, so the Metal
backend stays re-testable when PyTorch fixes it. §7.5 is confirmed: that backend
owns the `SymIntArrayRef` crash, and it buys nothing in exchange — it also fails
on `_triangle_projection_fused` in every Mac run, after which that function runs
eagerly regardless.

It corrects a claim in the setting's own docs, too: "the switch can never fail a
render" was true of a compile that *fails* (warn once, run eagerly) and false of
a backend that compiles and then generates wrong code, which is what happened.

**Do not tune the arena further** on the strength of over-commit: job 52 peaked
at 5.61 G against a 4.67 G recommendation and was fine (§7.4).

**Post-merge note.** Job 50, the first run on master's optimised batching, is
faster per chunk — chunk 2 at 51.8 s against 67.3 s for the same chunk in job 47
— and did **not** wedge; it was still progressing when its timeout killed it.

---

## 9. Instrumentation, and how to reproduce

`benchmarks/_mps_warm_regression.py [runs] [quality] [budget_s] [pin-arena]`

Prints a line **per render** and **per chunk**, so a killed or reclaimed job
still leaves its comparison behind. Per chunk it reports: elapsed offset,
`release_torch_memory` calls / how many were pressured / seconds inside them,
zero-copy import-cache clears and re-imports, the driver **peak** since the last
chunk (sampled every 2 s), and the pool plus `host_free`/`rss`. A heartbeat
thread prints every 30 s while a chunk is in flight, past a 45 s floor — so a
slow chunk keeps talking and a wedged process goes silent.

`pin-arena` holds later renders to the first's free-byte figure (the §5 A/B).

Hooks rebind the patched names in **every module that imported them** — patching
`memory_utils` alone intercepts nothing, which cost one job to discover.

### Harness traps worth knowing

* **A macOS job is reclaimed well before `timeout_minutes`** (72 min against
  120; 57 against 100) and **a reclaimed job publishes nothing** — the
  `if: always()` upload never runs. Size commands for ~40 minutes and make them
  print per unit of work. Four jobs cost an hour each learning this.
* **The push trigger only fires when `mac.json` actually changes.** Rewriting it
  with identical values changes no bytes and launches nothing, silently. The
  `_request` field exists to guarantee a diff; **verify the run exists by head
  SHA before reporting that it is running.** This cost two rounds.
* `get_job_logs` returns 404 for an in-progress job, so a running job's output
  can only be read from the web UI.

---

## 10. Also unexplained

The `max_surfaces_per_ray` truncation warning fires with wildly varying counts
across otherwise identical passes — **0, 2, 62, 192, 559 rays** — and appeared
first in warm passes only. Unrelated to the above as far as these jobs show, and
worth a look on its own: identical geometry should truncate identically.
