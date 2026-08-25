# Read-only audit: render-arena allocations read before they are written

Read `/content/algan/CLAUDE.md` first. READ-ONLY: modify no tracked file. No renders
(the GPU is in use by another session's measurements). Throwaway scripts under
`scratch_perf/ox/` may run on the CPU (`ALGAN_RENDER_DEVICE=cpu`).

## Context
`ManualMemory.get_tensor` (`algan/utils/memory_utils.py`) is a bump allocator over
one big device tensor and returns UNINITIALISED views (like `torch.empty`); helpers
such as `_arena_tensor(memory, shape, dtype, fill=None)` in
`algan/rendering/raytracing/raster_pipeline.py` fill only when asked. The arena is
rewound after every render chunk and reused, so an allocation that is read before
every element of it has been written sees whatever the previous chunk (or the
allocator's previous tenant) left there.

Measured symptom this audit serves: the same scene rendered from bit-identical merged
inputs gives different pixels (~5% of a 3840x2160 frame, on edges) depending on what
else the process allocated on the GPU beforehand -- exactly what a read-before-write
of arena memory would do. Two renders in one process agree; two processes disagree.

## Task
Enumerate every arena allocation on the deterministic sparse route
(`analytic_raster_route_active` true, `SAMPLES_PER_PIXEL == 1`, shadows on, the
split-sum glossy mode 3 active, refraction/split pool active because of translucent
PBR shells) and, for each, say whether every element is provably written before any
read, host or kernel. Sites to cover (not exhaustive -- follow the code):
- `tracer.py`: `_alloc_wavefront_state` (rs_ro/rd/acc/sca/int + stubs), `rs_pix`,
  `pix_accum` (both halves under glossy), `rs_alloc`, `rs_vis`, the
  `_ArenaRayCompactor` buffers `a`/`b`/`count`, `gen_meta`, `layer_offsets_t`,
  `gl_main`/`gl_pyr` (`_gloss_clear`), `aa_accum`, the per-tile `hit_f`/`hit_i` event
  buffers in `_drain_sparse_secondary`, `out` (prefilled by `_prefill_background`),
  the raster tables from `_build_raster_tables`.
- `raster_pipeline.py`: everything allocated in `prepare_sparse_raster_coverage`
  (`counts`, `accepts`, `frag_*_u`, the persistent `frag_*`, `covered_idx`,
  `run_offsets`, `sheet_*`) and in `shade_sparse_raster_coverage` (`sheet_offsets`,
  `sheet_accept`, `event_*`, `sheet_event_id`, `shadow_vis`, `dummy_*`).
- The kernels that read them: for each buffer name the kernel(s) and the condition
  under which an element is read, and whether a write to that element is guaranteed
  first (e.g. `event_pos` is written only for ACCEPTED sheets in resolve mode 1 -- is
  it read only through `acc_idx`? `rs_int` columns 0/1/3/4 for pool slots the resolve
  never allocated -- does `compact_ray_slots` with `scan_pool=True` read them?
  `pix_accum` reflection rows for pixels without a glossy branch? `hit_i`/`hit_f`
  entries beyond a ray's `num_hits`?).
- Also note any TORCH tensor created with `torch.empty` on the render path that is
  read partially initialised (e.g. `_query_row_states`' `out`, `_sparsely_written_zeros`).

Report to `scratch_perf/ox/REPORT_arena_reads.md`: a table (buffer, allocated at,
filled?, written by, read by, verdict PROVEN-SAFE / SUSPECT with the exact read that
can precede a write), suspects first. Quote file:line for every claim. Do not fix.
