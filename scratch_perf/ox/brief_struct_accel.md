# Read-only audit: acceleration structures on the default deterministic path

Read /home/user/algan/CLAUDE.md first. This is a READ-ONLY audit: modify no file
except writing your report to
/home/user/algan/scratch_perf/ox/REPORT_struct_accel.md. Do not run renders or
pytest. Cite file:line for every claim. This container is CPU-only; make no
wall-clock claims. Other read-only audit sessions share this tree — ignore them.

Context: I am ranking STRUCTURAL speedup candidates for the default renderer
(samples_per_pixel==1, sheet route + wavefront continuations). Sources:
algan/rendering/raytracing/{stbvh.py, refit_bvh.py, scene_builder.py, tracer.py,
raster_taichi.py, wavefront_kernels_taichi.py, raytrace_kernels_taichi.py,
settings.py}, DESIGN_hybrid_raster.md §9/§13.

Give a CONFIRMED/REFUTED verdict with line numbers for each numbered claim,
then answer the questions.

Claims:
1. With BVH_REFIT default on, every tree a default-path batch builds is a
   RefitBVH; the classic STBVH build runs on no default-path batch (name any
   exception that still builds classic).
2. A batch builds up to six trees (3 geometry types x full+opaque) and
   BVH_DEFER skips building ALL of them when no shadow ray, no
   reflection/refraction and no SPP>1 path will traverse — i.e. the common
   shadow-free non-reflective batch builds zero trees.
3. A shadow ray walks, serially and each to completion, one tree per geometry
   type present in the batch; there is no combined mixed-type tree. With
   SHADOW_ANYHIT default False, describe what the shadow trace actually does
   per (event, light): full transmittance march, nearest-hit, or first-hit
   early exit — and what changes under SHADOW_ANYHIT=True/"gather".
4. The BVH topology (binned SAH over ever-visible primitives) is rebuilt from
   scratch every frame batch, even when the actor set and motion are unchanged
   from the previous batch; nothing caches topology across batches.
5. The refit tree's per-(frame, child) link words make tree memory scale with
   T x blocks. Give the bytes per (frame, block), and state whether the shipped
   code actually collapses static geometry to T=1 ("static geometry dedupes to
   T = 1", DESIGN_hybrid_raster.md §9) — where, and under what condition it
   fails to (e.g. one moving mob in the batch).

Questions (answer from source only):
A. Structurally, what does one shadow-visibility evaluation cost: how many tree
   walks per (event, light, fan sample), and which loops multiply
   (lights x SOFT_SHADOW_SAMPLES x trees)? Point at the exact loop nests in
   raster_taichi.py's shadow trace.
B. §13 item 3 proposes a single mixed-type any-hit tree for shadow rays. From
   the current code: what would its leaves have to carry per geometry type, and
   which call sites would consume it?
C. Is any visibility or traversal result recomputed that could be shared —
   e.g. the same (event, light) pair re-walked for several sheets/samples, or
   the opaque-prepass trees overlapping the full trees' work?

End the report with a "What I did not verify" section.
