# T4 round, 2026-09: the baseline that could be taken

Two full profiles of `nn_scene_UHD.py` and `nn_scene_PREVIEW.py` on a Kaggle
Tesla T4, at `53fbf36` (`perf-base1.log`) and `1eea23a` (`perf-base2.log`).
`perf-base1` additionally carries a pass of each scene with Taichi's per-kernel
GPU profiler on, and `perf-base2` a run of `benchmarks/_sheet_compact_breakdown.py`
at the 4K frame's real shapes.

> **The `t4_baseline/` and `t4_after/` reports next door do not describe this
> renderer and their `excl` columns cannot be compared to anything produced
> today.** They predate `charge_kernel_to_parent`, so a stage's "exclusive"
> time still contained every kernel it launched — which is how `wavefront_loop`
> came to read 12 s of "unattributed host work" on a render whose entire host
> side is under a second. They also predate the inline bounce stages.

## The numbers

Warm RUN 2, `libx264 -preset ultrafast`, GPU otherwise idle:

| scene | run 1 | run 2 | cold (RUN 1) |
| --- | --- | --- | --- |
| `nn_scene_UHD.py` (30 frames @ 3840x2160) | 17.69 s | 18.22 s | 116.92 / 117.84 s |
| `nn_scene_PREVIEW.py` (50 frames) | 5.33 s | 5.13 s | 78.60 / 77.33 s |

So **run-to-run spread is 3-4%**: a single-run difference smaller than that is
not a reading. Both scenes render byte-identical output across all four runs
(`sha256 df9086d3b323831e` / `8c755b30590d262b`).

The reference numbers in `agent_guidance/gpu_harnesses.md` (UHD 29.90 s,
PREVIEW 6.25 s) are from before `Sync(duration=)` was renamed to `runtime=`;
neither scene had run since.

## Where the warm time goes

**Taichi kernel GPU time is a minority of both renders**: 8.5 s of 17.7 s at
UHD, 1.0 s of 5.6 s at PREVIEW. Neither benchmark is bound by the ray tracer.

UHD (17.69 s), the items over 2%:

| stage | s | % |
| --- | --- | --- |
| `raster_shadow_trace` (kernel) | 2.70 | 15.3 |
| `wavefront_shade` (kernel) | 2.43 | 13.7 |
| `wavefront_traverse_events` (kernel) | 2.03 | 11.5 |
| `compact_sheets`, own time | 1.60 | 9.0 |
| arena preflight (projection + merge + BVH, **unoverlapped**) | 1.44 | 8.2 |
| `raster_tri_count` + `raster_tri_write` (kernels) | 0.99 | 5.6 |
| sheets lexsort | 0.76 | 4.3 |
| post-process device→host copy | 0.64 | 3.6 |
| PN dice | 0.42 | 2.4 |
| sheets shade class | 0.41 | 2.3 |

PREVIEW (5.33 s) inverts it — the render is 2.25 s and the *preparation* is the
critical path:

| stage | s | % |
| --- | --- | --- |
| `Scene._get_batch_of_primitives` (3 calls, partly on the prefetch worker) | 2.28 | 42.8 |
| ray traced render total | 2.25 | 42.2 |
| **arena preflight** | 1.62 | 30.4 |
| `surfaces: get_render_primitives_batched` | 0.98 | 18.3 |
| merge + BVH builds | 0.82 | 15.4 |
| `compact_sheets` | 0.61 | 11.4 |

`arena preflight` is projection, the GPU merge, the STBVH/refit builds and the
arena upload, and it runs **on the render thread between batches with nothing
in flight**: `SETTINGS.computing.prefetch_gpu_prep` is off by default, so
`_prepare_batch_on_worker` never runs.

## The cold cost is the largest number here

A user renders once per process, and that costs 117 s at UHD against a 17.7 s
warm render. The RUN 1 table attributes it:

* `sheet_resolve_shade` — 45.3 s, of which **45.0 s is the launch call**, i.e.
  Taichi compiling that one kernel. At PREVIEW it is 39.8 s.
* `wavefront_shade` — 16.0 s, 13.5 s of it launch.
* `logical PN: _dice_logical_pn` — 17.2 s against 0.38 s warm, and
  `raster: precompute tri projection` 5.96 s against 0.018 s: `torch.compile`.
* At PREVIEW, `surfaces: get_render_primitives_batched` is 59.9 s of its own
  time on the first call against 0.83 s warm — the same story.

**Both caches are supposed to make this a once-per-machine cost and neither
visibly does here.** The `preview` step runs in a fresh process *after* `uhd`
in the same session, with `ALGAN_CACHE_DIR` on the persistent disk and the
inductor cache in the container's `/tmp`, and still pays 39.8 s to compile
`sheet_resolve_shade` and 59.9 s in the surface build. Whether that is a cache
miss (different kernel variants per preset) or the caches not being consulted
at all is not yet established, and it is worth more than everything else on
this page: it is 5-7x the warm render.

## The codegen knobs are a dead end (`perf-cg-codegen-sweep.log`)

All eight arms in one session, so directly comparable, at UHD:

| arm | cold | warm | vs base |
| --- | --- | --- | --- |
| base (ptxas chooses) | 108 s | **17.75 s** | — |
| `ALGAN_GPU_MAX_REG=96` | 81 s | 17.87 s | +0.7% |
| `=128` | 82 s | 18.02 s | +1.5% |
| `=168` | 82 s | 18.15 s | +2.3% |
| `=200` | 81 s | 18.02 s | +1.5% |
| `=255` | 81 s | 17.88 s | +0.7% |
| `ALGAN_ADV_OPT=1` | **2661 s** | 17.35 s | −2.3% |
| `ALGAN_OPT_LEVEL=2` | — | never ran | — |

**Every register cap is worse than letting ptxas choose, in both directions.**
Capping low (96) buys occupancy by spilling and loses; capping high (255) buys
fewer spills at lower occupancy and also loses. So these kernels already sit at
a local optimum on that tradeoff and are *not* occupancy-limited in any way
`-maxrregcount` reaches. The `gpu_max_reg` docstring's tuning was done on the
author's Pascal card; on Turing the answer is "leave it alone".

`ALGAN_ADV_OPT=1` is worth its own warning. It **compiled for 2661 seconds** —
Taichi's full IR pipeline (whole-kernel CSE, alias analysis) is superlinear in
statement count, and these megakernels inline an entire shading and traversal
call graph into one body. The −2.3% it returned is inside the run-to-run band
(UHD has spread 17.30–18.45 s across sessions). Not viable at any price, and
the arm after it never ran because the wedge ate the session — which is how
`--step-timeout` was found to be inert (fixed in `scripts/kaggle/runner.py`).

Read with the k-buffer result: transposing the traverse→shade hit buffers from
`[num_active, kbuf, ch]` to `[kbuf, ch, num_active]`, which turns 24 scattered
4-byte accesses per ray per direction into coalesced ones, is byte-identical
and **neutral at UHD**. So the hot kernels are neither register-bound nor
bandwidth-bound; what is left is dependent-load latency and divergence in BVH
traversal, which only an algorithmic change touches.

## The shadow gather march is neutral too (`perf-sh-shadow-gather.log`)

`ALGAN_SHADOW_ANYHIT=gather` replaces the ordered march's *k* full BVH
traversal restarts (one per peeled translucent surface) with
`ceil((k+1)/kbuf)`. The 20 glass shells made this look like the obvious
algorithmic lever on `raster_shadow_trace` (2.70 s of 17.7 s, with 390 ms tail
launches). Alternating arms, one session:

| | base | gather | |
| --- | --- | --- | --- |
| UHD | 18.16 / 18.76 | 18.40 / 18.78 | +0.7% |
| PREVIEW | 5.31 | 5.22 | −1.7% |

Both inside the noise band, and **the digests are identical to base** on both
scenes — so on this geometry the documented seam-merge corner never arises,
and `k` is evidently small: the shadow rays are dominated by the opaque ground
plane, one traversal each, not by deep peels through the shells.

## What this scene has, which bounds several other ideas

`default_scene_initializer` spawns exactly **one** `PointLight`
(`algan/scenes/default_scene.py`). So:

* a light-major thread mapping in `raster_shadow_trace` (its grid is
  `num_events * num_lights`, indexed event-major) is a **no-op** — with one
  light the mapping is already warp-coherent;
* `wavefront_shade`'s `vis = ti.Vector([1.0] * (3 * max_shadow_lights))` uses
  **3 of its 48 lanes**. It is dynamically indexed, so it is a 192-byte
  per-thread *local memory* array, re-initialised across all 48 lanes on every
  drained surface, to carry 12 bytes. This scene is its worst case, and it is
  the one structural candidate the measurements above do not rule out —
  `-maxrregcount` cannot reach local memory, which is why the codegen sweep
  says nothing about it.

## Cross-session numbers are not comparable

Identical code, `nn_ablation base PREVIEW`, measured **4.69, 4.70, 4.76, 5.20,
5.31 s** across sessions — a 13% spread. UHD is far steadier (17.63–17.75 over
four sessions) but still moves ~3% run to run *within* a session.

So: **put both arms of an A/B in one session**, and treat a single-run
difference under ~3% at UHD (or under ~10% at PREVIEW, across sessions) as no
reading at all. Several early conclusions in this round had to be withdrawn for
exactly this reason.

## Reproducing

```
uv run python scripts/kaggle/make_notebook.py --tag perf-base1 \
    --branch <branch> \
    --step "uhd:python benchmarks/performance/nn_scene_UHD.py" \
    --step "preview:python benchmarks/performance/nn_scene_PREVIEW.py" \
    --step "uhd_kp:python benchmarks/performance/nn_scene_kernel_profile.py UHD" \
    --step "preview_kp:python benchmarks/performance/nn_scene_kernel_profile.py PREVIEW" \
    --env ALGAN_VIDEO_ENCODER=software --out /tmp/nb.py
```

then `save_notebook` per `agent_guidance/gpu_harnesses.md`. The kernel-profiler
variants cost about 0.3% of warm wall time on this box (17.67 s against 17.69 s),
so the per-kernel GPU table is effectively free to collect alongside.
