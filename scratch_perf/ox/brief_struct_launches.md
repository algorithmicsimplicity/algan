# Read-only audit: launch/sync structure of the deterministic render thread

Read /home/user/algan/CLAUDE.md first. READ-ONLY: modify no file except writing
your report to /home/user/algan/scratch_perf/ox/REPORT_struct_launches.md. No
renders, no pytest. Cite file:line for every claim. CPU-only; no wall-clock
claims. Other read-only audit sessions share this tree — ignore them.

Context: ranking STRUCTURAL candidates in how the render thread orchestrates
work per chunk on the default sheet route (tracer.py render_batch_raytraced ->
raster_pipeline.py prepare/shade_sparse_raster_coverage ->
sheet_resolve_taichi.py + the wavefront bounce loop).

Claims (CONFIRMED/REFUTED + line numbers):
1. Per render chunk on the default path, the host synchronises with the device
   at these points: list every forcing op (.item(), .tolist(), .nonzero(),
   .cpu(), .any() on device tensors...) in execution order for (i) a
   shadow-free opaque batch and (ii) a shadowed reflective batch.
2. The wavefront bounce loop syncs once per bounce iteration (compaction count
   readback) and once per accepted tile (rs_alloc readback); nothing overlaps
   chunk N's host-side torch passes with chunk N-1's device kernels inside the
   render thread (the prep-thread batch prefetch is separate and out of scope).
3. A bezier-circuit candidate pixel evaluates `_bezier_point_metrics` against
   every segment of its circuit — check bezier_acceleration.py: what pruning
   exists (interval lists? bands?), and at which call sites it IS and is NOT
   applied (raster emission vs shadow occlusion vs wavefront continuation
   hits).
4. shade_sparse_raster_coverage launches the resolve kernel twice on shadowed
   batches (mode 1 events, mode 2 shade), and mode 2 re-fetches everything mode
   1 already fetched (RENDERER_WORK_QUEUE.md item 9); the memoization that item
   scopes (widening the event tables by ~15 floats/sheet) is NOT built.
5. The resolve (sheet_resolve_shade) and wavefront_shade are monoliths carrying
   every material pipeline via ti.template composition. From the source: what
   per-thread state does the resolve walk carry (registers/locals per pixel),
   and which parts are live only for shadowed/reflective/refractive sheets yet
   paid by every thread?

Questions (answer from source only):
A. Which host-side torch passes remain between kernel launches per chunk on the
   sheet route (with SHEET_MASK_KERNEL and SHEET_RANK_KERNEL on), in execution
   order, and which of them force a sync?
B. Which launches or host passes consume inputs that are provably unchanged
   across the chunks of one batch (per-batch tables re-derived per chunk)?

End the report with a "What I did not verify" section.
