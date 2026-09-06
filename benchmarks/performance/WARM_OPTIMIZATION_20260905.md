# Warm UHD render optimization

Target: the exact six-frame `nn_scene_UHD.py` scene, on the GTX 1050.
The user's supplied warm profile is 25.42 s; cold compilation is excluded.

## Ranked candidates

1. Reduce shadow and secondary BVH traversal work. Dedicated shadows cost
   6.37 s, secondary shading (including inline shadows) 6.03 s, and secondary
   traversal 3.16 s. These are disjoint kernel totals, not inclusive stages.
   First experiment: fetch the selected refit link directly instead of loading
   and branching over every sibling. Further candidates include retaining
   pending child distances to avoid retesting sibling boxes and spatially
   grouping shadow rays.
2. Tune occupancy and read-only memory access using the installed Quadrants
   patches. Register limits and read-only ndarray loads currently default off;
   both can change performance without changing the rendering algorithm.
3. Reduce raster discovery and sheet compaction (5.16 s combined). Candidates
   include fusing small tensor operations and reducing repeated stable sorts.
4. Reduce preparation and post-processing overhead. These have less total
   headroom than the ray kernels on this short scene.
5. Lower precision or adapt secondary sampling only after image validation.
   BVH bounds already use conservatively rounded float16. Changing geometry
   storage or sample counts has a higher risk of visible edge errors.

## Measurement

Use `.venv/Scripts/python.exe benchmarks/performance/nn_warm_experiment.py`.
The runner executes the original benchmark file, overriding only profiling
repetitions, output tag and explicitly requested compiler settings. No scene
copy can silently drift in duration, quality or encoder settings.

Use at least four repetitions, discard repetition one, and bracket candidate
runs with controls to expose thermal drift. `--legacy-refit-links` restores
the original link decoder before kernel compilation; each arm uses a separate
process. `--kernel-profiler` supplies device timings separately from clean
wall-time measurements. `--max-reg` and `--readonly` configure the compiler;
the live config is printed after rendering to verify the requested arm.

Never run concurrent GPU jobs or edit kernel source during a render.

## Results

Pending measurement and output validation.
