# Read-only audit: time expansion and geometry duplication in the scene merge

Read /home/user/algan/CLAUDE.md first. READ-ONLY: modify no file except writing
your report to /home/user/algan/scratch_perf/ox/REPORT_struct_merge.md. No
renders, no pytest. Cite file:line for every claim. CPU-only; no wall-clock
claims. Other read-only audit sessions share this tree — ignore them.

Context: ranking STRUCTURAL candidates around the merged-scene representation.
DESIGN_optimization_targets.md T4 records: "`scene_builder`'s
`_cat_collections` runs `_unify_time` over the primitives it concatenates, so a
single moving flat mesh anywhere in the scene expands the static one straight
back out at merge time." MERGE_DEDUP_TIME (raytracing/settings.py:490) is
default on. Sources: algan/rendering/raytracing/{scene_builder.py,
primitives.py, raster_pipeline.py, tracer.py}, algan/rendering/logical_pn.py.

Claims (CONFIRMED/REFUTED + line numbers):
1. `_unify_time` expands a static collection's arrays to the batch's full frame
   count whenever any collection of the same geometry type is per-frame — so
   the per-frame array cost is set by the most dynamic collection. List which
   merged fields expand (corners, normals, colours, uvs, material params,
   ids...) and the per-(frame, triangle) byte cost.
2. MERGE_DEDUP_TIME collapses identical time rows — describe its actual
   mechanism, granularity and key, and state what it does NOT collapse (e.g. a
   static mesh merged beside a moving one).
3. The per-batch projection and screen-bounds tables are computed over the full
   expanded [T, N] arrays with no static shortcut — a static triangle under a
   static camera is still projected once per frame. State whether any code
   detects the static-camera-and-static-row case.
4. A PN Surface that does not move and keeps its levels is re-diced from
   scratch every BATCH (T4's frame collapse is within-batch only); nothing
   caches diced microtriangles across batches. Same question for bezier
   circuits (chord counts + geometry build) across batches.
5. Downstream consumers (raster kernels, refit BVH build, shadow trace) index
   geometry as dense [T, N]; per-collection frame strides (stride-0 static
   rows) would require changes at — enumerate the consumer sites that assume a
   dense time axis.

Questions (answer from source only):
A. Which merged arrays are the largest per-batch allocations for a mixed
   static+moving scene, and which of those are pure time-broadcasts of one row?
B. Does the pipeline already compute static-ness somewhere (distinct_frames,
   geometry_static, frame-valid masks) that the merge then throws away by
   expanding anyway? List each such signal and where it dies.

End the report with a "What I did not verify" section.
