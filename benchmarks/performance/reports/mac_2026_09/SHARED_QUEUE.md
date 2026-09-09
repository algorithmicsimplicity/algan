# Where a warm Metal render's time goes, and the two fences it no longer takes

Measured 2026-09-09 on the Mac harness (`agent_guidance/gpu_harnesses.md`),
GitHub's virtualized M1 with 3 CPUs and 7 GB, against the locked published
`algan-quadrants` 1.3.0 (no wheel override), torch 2.7.1, after PR 120's
import-lifetime fix. Workload: the `nn_scene_UHD.py` fixture the earlier
reports use -- 18 frames at 3840x2160, shadows off, one frame per chunk, a
fixed 1720 MiB arena, software x264, CPU animation, `torch.compile` off.

The question was why the Apple GPU only matched the box's own CPU. The
answer, from a whole-render profile, is that the Taichi kernels were a small
part of a warm render; most of it was torch's own MPS work and the two host
fences the zero-copy launch wrapper took around every kernel launch.

## 1. The profile

`benchmarks/_mac_postfix_profile.py --coarse` now prints its stage table
into the job log (the artifact host is unreachable from some boxes that read
these runs). Run
[34299911993](https://github.com/algorithmicsimplicity/algan/actions/runs/34299911993):
CPU before (2 renders), MPS coarse (3), MPS with the ATen dispatch trace (2),
one VM. Times are **self** seconds -- a row excludes the spans nested inside
it -- so kernel rows exclude the fences and the fences exclude nothing.

**Warm MPS render, 61.3 s, the 17 steady-state chunks:**

| what | self seconds | notes |
| --- | ---: | --- |
| `coverage.compact_sheets` | 11.7 | torch ops: sorts, uniques, gathers over ~2.9M fragments |
| `stage.wavefront: - sparse setup` | 8.0 | mostly the `searchsorted().tolist()` readback waiting out the compaction's queued GPU work |
| `coverage.prepare_sparse_raster_coverage` | 7.9 | torch ops |
| `wait.quadrants.sync` (1067 calls) | 5.0 | the post-launch fence: **all Taichi kernel GPU time**, 0.29 s a chunk |
| `wait.torch.mps.synchronize` (1067 calls) | 3.8 | the pre-launch fence |
| `coverage._sibling_weights` | 1.0 | |
| `kernel.*` (1063 launches) | 0.6 | Python launch overhead, 0.56 ms a launch |
| `imports.*` (11 962 calls) | 0.2 | the zero-copy import cache |

**The same render's first chunk and prelude** carried 5.8 s of `kernel.*`
self time over 62 launches -- 2.8 s in one launch of
`sheet_resolve_shade_arena` -- against 0.6 s for the other 1063 launches. The
CPU arm's first chunk carried 1.9 s. Section 3 says what it was.

**The ATen trace** (the same render under `TorchDispatchMode`, 59.3 s):
28 040 ATen calls in the steady state, 29.5 s of self time. The largest
single rows were a 30-int readback (`copy_ mps -> cpu (30,)`, 17 x 0.46 s:
`_read_tile_alloc`, waiting out the tile's queued work), `index.Tensor` with a
boolean mask over 2.5M int64 (17 x 0.15 s: `band_class_groups`'s
`bands[starts]`), five `unique_consecutive` a chunk at 0.09 s and stable sorts
at 0.11 s. Its cProfile (in the artifact) puts 37 s of the 59 s inside ATen
op calls, 5.5 s in `ti.sync`, 4.0 s in `_mps_deviceSynchronize` and 4.1 s in
Quadrants' C++ launch path (3.6 ms a launch on this VM).

**The CPU arm, warm, 86.6 s** on the same VM: `compact_sheets` 30.4 s,
`wavefront_traverse_events_arena` 13.0 s, `prepare_sparse_raster_coverage`
7.7 s. So the GPU runs the kernels ~40x faster than three cores and the
compaction's torch ops ~2.5x faster, and paid 9 s of fences the CPU never had.

## 2. Two fences become none: Quadrants on torch's command queue

The wrapper took `torch.mps.synchronize()` before and `ti.sync()` after every
converted launch because torch and the compiler dispatched on separate Metal
command queues and nothing else ordered them (`DESIGN_mps_zero_copy.md`
§3.3). Quadrants 1.3 can instead dispatch on a queue it did not create
(`external_metal_command_queue`) and, told that the queue is torch's
(`external_metal_command_queue_is_torch_queue`), skips its own interop
syncs; Metal executes the command buffers of one queue in commit order.
`quadrants.interop.get_mps_command_queue()` reads the queue off torch's
default MPS stream.

`taichi_init_kwargs` now passes both on the Metal arch. The two fences
become:

* **before a launch**: `torch.mps.Event().record()`. Torch batches encoded
  work in an open command buffer and commits it lazily; `record` reaches
  `MPSHooks::recordEvent(syncEvent=true)` -> `MPSStream::synchronize(COMMIT)`
  -> `commitAndContinue`, which commits that buffer to the queue **without a
  CPU wait** (torch 2.7.1 sources, `aten/src/ATen/mps/MPSHooks.mm`,
  `MPSEvent.mm`, `MPSStream.mm`);
* **after a launch**: nothing. Quadrants submits its command list at the end
  of every launch when the queue is external
  (`quadrants/runtime/gfx/runtime.cpp`, `submit_current_cmdlist_if_timeout`,
  present at v1.3.0), so torch's next command buffer is queued behind the
  kernel, and a torch readback -- which waits as it always did -- waits for
  the kernel by FIFO order.

Both cache keys already exclude the queue pointer (`taichi_source_key`
excludes it from the fingerprint; Quadrants #850 from the C++ key), so
sharing costs no recompiles. `ALGAN_MPS_SHARED_QUEUE=0` keeps separate queues
and the blocking fences -- the A/B arm and the escape hatch -- and
`mps_zero_copy.STATS["shared_queue_launches"]` says which regime a run took.

### Measured

Run [34300735289](https://github.com/algorithmicsimplicity/algan/actions/runs/34300735289),
one VM, fresh process per arm, `_mac_postfix_profile.py --coarse`: a CPU
bracket, then MPS with the fences (`ALGAN_MPS_SHARED_QUEUE=0`), MPS on the
shared queue, and the same two again, ABBA. Parity between the two MPS arms'
third renders by `compare_warm_videos.py`.

RESULTS_PLACEHOLDER

## 3. The runtime was being torn down between renders

Every warm MPS render in run 34299911993 began with
`[Quadrants] Starting on arch=metal`; the CPU arm's process printed its
`Starting on arch=arm64` once. That is the 5.8 s of first-chunk kernel self
time: a fresh program creating every kernel's Metal pipeline from the offline
cache.

The cause is `release_torch_memory`'s host-memory pressure reset
(`reset_quadrants_for_memory_pressure`). On the runner the MPS pool holds
~4.7 GB of the 7 GB, so `psutil`'s available figure sits at 0.8-1.2 GB,
under the 15% threshold, on every call; a reset is recorded, deferred to the
job's exit, and taken there. It frees none of what that figure measures --
the Metal backend has no LLVM JIT and its program is small -- and costs the
next render its pipelines. The reset is now declined on the Metal arch. It
was written for Linux cgroup hosts and still runs there.

## 4. What is left, and why the GPU does not run away from the CPU here

With the fences gone, a warm UHD chunk on this box is, in round numbers,
1.0-1.3 s of torch MPS work (the sheet compaction's sorts, uniques, scans and
gathers over ~3M fragments, plus their readbacks), 0.3 s of Taichi kernels,
and 0.4 s of Python. The compaction is the same torch code the CPU arm runs
on three cores in ~1.8 s a chunk; MPSGraph's sort of 2.9M int64 keys takes
~0.11 s where a native radix sort would take a few milliseconds, and each of
the five `unique_consecutive` a chunk is a readback. That is the structural
remainder: the raster/compaction stage is torch-op-bound, and MPS's
implementations of those ops are only ~2.5x faster than this box's CPU.

Cheaper items measured but not taken this round, in order of size:

* `_read_tile_alloc`'s 30-int readback, 0.46 s a chunk: mostly the queued
  work it waits out, so it moves with whatever precedes it rather than being
  a cost of its own;
* `band_class_groups`'s boolean gather (landed as a scatter, ~0.15 s a chunk);
* `prepare._prewarm_render_batch` on the prefetch worker: 13-19 s wall on the
  MPS arm against 2-8 s on the CPU arm for the same work, contended by the
  render thread's dispatch on three cores. Off the critical path for this
  scene (the second batch's prep finishes before it is needed) but not for a
  scene with more batches.
