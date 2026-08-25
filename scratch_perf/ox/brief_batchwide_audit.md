# Read-only audit: which per-batch decisions make rendered pixels depend on the batch window?

Read `/content/algan/CLAUDE.md` first. This is a READ-ONLY audit: do not modify any
tracked file. You may write throwaway scripts under `scratch_perf/ox/` and run them
with `ALGAN_USE_DAEMON=0 uv run python ...`, but the GPU is in use by another
session's measurements: **do not render anything** (no `save_video`/`save_frame`);
host-side probes on CPU tensors only (`ALGAN_RENDER_DEVICE=cpu` if a script must
import algan).

## Context
Algan materializes a *batch* of frames at once (`Scene.get_batch_of_primitives`,
`algan/render_loop.py`), builds one merged scene + STBVH per batch
(`algan/rendering/raytracing/scene_builder.py`) and renders it in chunks. Measured
fact on this branch: the same scene rendered with 3-frame batches and with 19-frame
batches produces different pixels at 3840x2160 (edges of text glyphs, silhouettes of
diced spheres, sub-pixel speckle over a textured flat quad; ~5% of pixels, up to 80
channel values), while at 704x396 the two are byte-identical. Renders with identical
batch windows are byte-identical to each other whatever the chunk/tile split.
`CLAUDE.md` documents one such mechanism (window split -> merged array padding ->
STBVH order -> depth-tie landing). I want the complete list.

## Questions (answer each with CONFIRMED/REFUTED and file:line evidence)
1. Bezier chord subdivision: `RayTracedBezierCircuitPrimitive._compute_samples_per_segment`
   (`algan/rendering/raytracing/primitives.py`) -- is the sample count per segment chosen
   from the maximum chord error over ALL frames of the batch (one count for the batch)
   rather than per frame? If so, a longer batch can pick a finer count for every frame.
2. Logical PN dicing: `LogicalPNTrianglePrimitive._required_subdivision_levels`,
   `_required_edge_levels`, `_required_patch_levels` -- are edge/patch levels per frame, or
   does any reduction (max/any) run across the frame axis, or across patches sharing an
   edge in a way that couples frames? Include the `max_diced_triangles` budget and the
   `_triangle_counts(...).sum(1) > budget` tests: what axis is `sum(1)`?
3. The merge: `_dedup_time` / `MERGE_DEDUP_TIME`, `_split_promotable` (constant-property
   promotion "constant across their corners (and frames)"), and any `_unify_time`
   broadcasting -- do any of these change per-frame VALUES (not just layout) depending
   on which frames share a batch? Promotion decided over the batch's frames is the one
   to check: a property constant over 3 frames but not over 19 would be promoted in one
   window and not the other -- does the promoted path shade identically to the
   per-vertex path?
4. The STBVH / refit BVH build (`stbvh.py`, `refit_bvh.py`, `segment_primitives_in_time`):
   does the tree's structure depend on the number of frames, and where exactly would a
   different tree change a closest-hit or any-hit result (shared-edge ties, epsilon
   tests)? Name the kernel sites (`raytrace_kernels_taichi.py`, `wavefront_kernels_taichi.py`).
5. Surface vertex normals / welds (`algan/mobs/surfaces/surface.py`
   `get_render_primitives_batched`, `surface_weld_flags`, `compute_grid_vertex_normals`)
   -- any reduction across the frame axis?
6. Anything else you find that reduces across the frame dimension of a `[T, ...]` tensor
   on the way to the kernels. `grep` for `.amax(0)`, `.max(dim=0)`, `.any(0)`,
   `.all(0)`, `amax(dim=0)`, `.max(0)` etc. in `algan/rendering/` and `algan/mobs/` and
   classify each hit as (a) per-frame-preserving, (b) batch-wide decision affecting pixel
   values, (c) batch-wide but layout-only.

For every CONFIRMED (b) item state: the exact decision, which frames it couples, whether
a longer batch can only make the tessellation finer (monotone) or can also make it
coarser, and whether it can cause a visible pop at a batch boundary. Do not propose
fixes; map the territory. Write the report to `scratch_perf/ox/REPORT_batchwide_audit.md`
with a summary table first.
