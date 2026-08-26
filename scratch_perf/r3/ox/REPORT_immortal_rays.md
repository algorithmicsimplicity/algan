# The "immortal rays" of the sheet route's bounce loop — diagnosed

Brief: `scratch_perf/r3/ox/brief_immortal_rays.md`. READ-ONLY on production
code was honored; every probe lives in `scratch_perf/r3/probes/` (inventory in
§8). Line numbers are against today's working tree.

## 0. Verdict up front

**The rays are not immortal.** On this container (Taichi CPU backend, PREVIEW,
the nn scene) a tail cohort of **exactly 30 rays** enters bounces 5, 6 and 7
with 100% survival — matching the T4 PREVIEW number quoted mid-task exactly —
and then **all 30 retire at bounce 7**, deterministically. No drain loop ever
reaches `max_iters`: the profile contains zero `bounce 8+` stages and the
per-tile launch arithmetic closes exactly (§2). What is real is a **missing
cheap exit**: every one of the 30 crosses the renderer's own significance floor
(`MIN_WEIGHT = 1e-3`, `raytrace_kernels_taichi.py:132`) no later than its
first recorded tail iteration, yet keeps bouncing for several further
generations because **the weight-floor exit is unreachable for any ray whose
last processed hit took an in-place reflection branch** — all three such
branches end in a `break` that jumps past the check (`break` at
`wavefront_kernels_taichi.py:3290`, `:3428`, `:3586`; check at `:3591-3594`,
which sits inside the hit-drain loop after those branches). The rays die only
when `bounces_left` reaches 0: the renderer refuses the next reflection
(`R := 0`, `wavefront_kernels_taichi.py:3104-3107`), their opaque last hit
zeroes the pass-through weight, and the floor test finally fires.

So DESIGN_T4_optimization.md §5 item 5 ("may be a correctness bug rather than
a performance one") resolves to: **image correct, termination logic
inconsistent, small but pure waste**, plus a wrong claim in §2 ("21 rays never
terminate, riding to `max_iters`" — they ride to the *bounce cap*; §6). A
one-line fix is proposed in §7; not implemented, per the brief.

Every claim below carries a line number or a measured number.
Reasoned-but-unmeasured statements are labelled **[inference]**.

## 1. What was measured on what

| artifact | what it is |
|---|---|
| `scratch_perf/r3/cpu_profile_nn_PREVIEW.txt` (copy also at `probes/report_nn_PREVIEW_cpu.txt`) | Profile of the stock `benchmarks/performance/nn_scene_PREVIEW.py`, runs=2, written by the benchmark itself on this container. RUN 1 end-to-end 102.47 s (cold), RUN 2 56.62 s (warm) — report lines 10 and 341. |
| `probes/probe_survivors.py`, `probes/probe_survivors2.py` → `survivors_v2.pt` | Monkeypatch-only instrumentation (no production file touched) of `tracer._alloc_wavefront_state`, `_ArenaRayCompactor.select`, `wavefront_traverse_events` and `wavefront_shade`. Records every compaction's surviving count per tile, and for small iterations clones each active ray's status/bounces_left/processed/num_hits/weight/origin/direction before and after shade, plus the accumulator delta the iteration committed. Full PREVIEW render. |
| `probes/probe_mat.py`, `probes/probe_micro.py` | Single-frame reruns (`Scene.save_frame(at=12)`, ~15 s) that pair each shade launch with its traverse batch and read the last-hit primitive's packed material row from the shade call's own arguments. |

Argument positions used by the probes were taken from the call sites with AST
parsing, not by hand: both `wavefront_traverse_events` calls are 50 args with
`hit_f` at 41 / `hit_i` at 42 (tracer.py:2593 and :3225), and both
`wavefront_shade` calls are 67 args with `rs_*` at 55-59, hits at 60-61,
`rs_pix` 62, `pix_accum` 63 (tracer.py:2646 and :3277).

Environment notes: the profiler's kernel wrapper calls
`torch.cuda.synchronize()` unconditionally (`profiling_utils.py:381`), which
raises with no NVIDIA driver, so probe drivers shim it to a no-op before
importing algan; `world_map.png` is passed to `ImageMob` as an absolute path
because assets resolve against the main script's directory
(`file_utils.get_image` → `resolve_asset_path`).

## 2. Reproduction: the phenomenon is fully visible on CPU

Warm-run bounce table (`cpu_profile_nn_PREVIEW.txt`, RUN 2 section):

```
   bounce  calls     rays in  traverse s   shade s continuations
        0     18      758566       2.111     0.921        225662
        1     18      225662       0.690     0.430        155994
        2     18      155994       0.469     0.458         17695
        3     18       17695       0.121     0.113           437
        4     18         437       0.017     0.017            30
        5     14          30       0.008     0.012            30
        6     14          30       0.009     0.021            30
        7     14          30       0.008     0.008              -
```

Three independent facts pin the loop's true length:

1. **No iteration beyond bounce 7 exists anywhere in the profile.** Stage
   labels cap at `bounce 8+` (`tracer.py:2587`, cap constant `:490`); grep
   counts zero such rows. A ray riding to
   `max_iters = MAX_SURFACES_PER_RAY + max_bounces·2 + 4 = 256 + 16 + 4 = 276`
   (`tracer.py:2309`; `MAX_SURFACES_PER_RAY` `raytrace_kernels_taichi.py:134`;
   `MAX_BOUNCES = 8` `raytracing/settings.py:29`) would produce ~268 more
   labeled iterations per frame-part.
2. **Launch-pair arithmetic closes exactly.** "drain active count" launches =
   **132 = 18 frame-parts × bounces 0–4 + 14 × bounces 5–7** (90 + 42). Every
   drain loop ended by bounce 7.
3. **Every tile's final compaction returns 0.** Probe v2 recorded all 150
   compactions of the run; all 18 per-tile sequences end in a 0 — e.g.
   `(45368, 13589, 9482, 1064, 30, 2, 2, 2, 0)`. The loop exits because
   `active.numel() == 0` (`tracer.py:2575`), never via the iteration cap.

The `-` in bounce 7's continuations column does not mean "kept going unobserved";
it means "never observable" — the table fills row N's continuations from row
N+1's rays-in, so the last row always shows `-`
(`profiling_utils.py:1282-1286`; docstring `:1265-1267`). §6 argues that this
misreading is the origin of the immortal-rays claim.

Machine-independence follows because the cohort is closed and deterministic:
the same geometry produces the same chains on any backend. The T4 facts given
mid-task (PREVIEW: exactly 30 entering bounces 5–7 at 100% survival; UHD:
~2855 entering bounce 7 with ~90%/bounce attrition) are consistent with the
CPU evidence: bigger frames have more pixels whose deepest mirror chain lasts
to the cap, and at UHD some members leave early via pass-through exits, hence
<100% survival there **[inference from the mechanism, not measured here]**.

## 3. Every exit from `active`, enumerated

Host loop: `while active.numel() > 0 and it < max_iters` (`tracer.py:2575`,
inside `_drain_sparse_secondary` `:2552`). A ray leaves `active` iff its
status stops being `_ACTIVE (0)`: compaction keeps exactly the slots with
`rs_int[r, 2] == 0` (`compact_ray_slots`, predicate at
`wavefront_kernels_taichi.py:131`; codes `_ACTIVE/_DONE` at `:79-80`; select at
`tracer.py:2717-2722`, scanning the whole pool here because a reflective batch
has `pool_ratio > 1` via `_split_pool_ratio` `tracer.py:391-443`). Statuses
are written only inside `wavefront_shade` (this route: compact=1,
first_iter=0):

| # | exit | predicate | site (wavefront_kernels_taichi.py unless noted) |
|---|---|---|---|
| E1 | miss / escaped | `num_hits == 0` → commit accum + leftover weight, set `_DONE` | `3655-3685` (store `3685`) |
| E2 | far clip | `(far_clip > 0) and (base_dist + t_hit > far_clip)` → done | `2589-2594` |
| E3 | weight floor | after each processed hit, `max(weight_r,g,b) < MIN_WEIGHT (1e-3)` → done | `3591-3594` (**inside** the hit-drain loop) |
| E4 | peel complete | post-loop `(not done) and (not bounced) and (num_hits < KBUF)` → done; under `opaque_closest` any non-bouncer retires immediately | `3600-3601`; variant `3596-3598` |
| E5 | surface ceiling | `processed >= MAX_SURFACES_PER_RAY (256)` → done, counted as truncation | `3602-3611` |

Non-spawn paths (never created, so not exits): pool-split continuations gated
on `wt_max > MIN_WEIGHT` (`3231`, `3303`, `3344`, `3380`, `3535`) and pool-slot
denial (`_reserve_continuation_slot` `903-918`; overflow retried host-side at
`tracer.py:3016-3034` and `3082-3096`). Seam-skipped hits `continue`
(`2625-2628`) without retiring anything, but still increment `processed`
(`2606-2607`).

Ways to remain `_ACTIVE` across iterations (status store at `3633`):

- **S1 — bounced in place.** Each reflect-here branch multiplies the weight by
  the reflected energy, sets `bounced=True`, and `break`s out of the hit-drain
  loop: glass reflect `3278-3290` (break `3290`), plain reflector
  `3416-3428` (break `3428`), custom-scatter reflect `3576-3586` (break
  `3586`). Post-loop, E4 excludes them explicitly (`not bounced`), so a
  bouncing ray has NO weight-based exit at all until it stops bouncing.
- **S2 — depth peeling.** `num_hits == KBUF (=4)` surfaces still queued
  (`KBUF`, `raytrace_kernels_taichi.py:353`).

The resolve side (bounce −1) mirrors this: far clip `sheet_resolve_taichi.py:284`,
sub-`MIN_ALPHA` sheet skip `370`, weight floor inside the walk `1041-1043`,
ceiling `1051-1063`, bounced primary stays `_ACTIVE` `1078-1099`, retire
otherwise `1100-1121`.

## 4. Who the 30 survivors are

All numbers from probe v2 (full PREVIEW render) unless noted. The cohort that
enters bounces 5/6/7 sums to **exactly 30 rays over 14 tiles** — per-tile
plateau sizes of 1–5, e.g. `..., 1064, 30, 2, 2, 2, 0` — matching the table
row-for-row. All 30 deaths were captured:

- **Death:** all at bounce 7, status 0→1, entering with `bounces_left = 0`
  and exiting with weight **exactly 0.0**. Weight on entry to the final
  iteration: 2.2e-12 … 2.2e-7 (measured, all 30). Mechanism as in §0.
- **Behaviour while alive:** every member bounced in place at *every*
  recorded tail iteration: `bounces_left` decrements by exactly 1 per
  iteration (…→2→1→0) while `num_hits` stays 1–2. `num_hits < KBUF` excludes
  S2 peeling (a non-bouncer with nh<KBUF dies via E4 the same iteration), so
  S1 is the only way these rays were still alive — confirmed by the bl ticks.
- **Weight decay is real and steep:** measured per-bounce factors ×0.046–×0.105
  across the cohort. Example full trace (slot 45820, window-local pixel
  index 185618 = 263·704 + 466 → kernel px 466, py 263):

  ```
  b3  w 2.52e-6 → 2.35e-7  bl 4→3   (bounce)
  b4  w 2.35e-7 → 1.81e-8  bl 3→2   (bounce)
  b5  w 1.81e-8 → 1.25e-9  bl 2→1   (bounce)
  b6  w 1.25e-9 → 7.46e-11 bl 1→0   (bounce)
  b7  w 7.46e-11 → 0.0     bl 0→0   DEAD
  ```

- **First sub-floor observation:** bounce 3 for some members, 4 for the rest
  (none is above the floor at its first recorded tail iteration; per-ray
  records were analyzed where cohorts shrink below 512 rays, i.e. bounces ≥3
  in the big tiles). **[inference]** Back-extrapolating the
  measured ratios puts their spawn weights near ~5e-3–1e-2 (legitimately above
  the floor, so the resolve was right to spawn them) and the actual crossing
  of MIN_WEIGHT around bounces 1–2; i.e. roughly 6 of their 8 bounces were
  spent below the significance floor.
- **Spawn class:** 28 of 30 accumulate into their own pixel's row (`accum_row
  < gloss_base`): the pooled-reflection family spawned by the sheet resolve
  under gate `rwt_max > MIN_WEIGHT` (`sheet_resolve_taichi.py:800`, spawns at
  `:771/:832/:841`). **2 are glossy-prefilter descendants** (`accum_row ≥
  gloss_base`: rows created for the split-sum substitution ray, born at weight
  `one3 = 1`, `sheet_resolve_taichi.py:884-896`) whose later generations are
  ordinary pooled reflections spawned inside the drain loop. Zero refraction,
  zero pool-slot denials observed anywhere in the run.
- **What they hit:** last-hit primitives are triangles (`htype 1`) carrying
  built-in pipeline id 5 = `_MID_PHYSICAL` (`shading_taichi.py:131`), packed
  slots read from `tri_mat` (`slot map`, `settings/raytracing/settings.py:2839-2862`):
  `metalness=0.0`, `transmission=0.0`, with `ior=1.5, roughness=0.18` at the
  final hits and `ior=5, roughness=0.12` mid-chain. Those are exactly the
  NeuronV3 materials — cores `MeshPhysicalMaterial(roughness=0.18)`
  (`mobs/neural_nets/neural_net.py:1089-1094`) and shells
  `MeshPhysicalMaterial(roughness=0.12, ior=5)` (`:1116-1128`). A dielectric
  ior=5 has Fresnel f0 = ((1−5)/(1+5))² ≈ 0.44 — a strong reflector, which is
  why chains here survive many generations before decaying under 1e-3. The
  per-bounce factors above are `alpha·R·mirror_share(roughness)` evaluated at
  each hit's geometry (`wavefront_kernels_taichi.py:3211-3213`,
  `_material_reflectance :1080-1167`, `_mirror_share :949`).
- **Pixel coordinates** (kernel coords, width 704; an output PNG's row is
  `height−1−py`, per `raster_pipeline.py:63`): a sample of the cohort:
  (466,263), (353,263), (353,143), (468,253), (408,228), (294,933), (466,150),
  (468,301), (415,266), (408,951) — full list in
  `probes/plateau_stats.json`.

## 5. The two brief hypotheses, checked explicitly

**(a) Two reflective surfaces re-spawning each other with weight that never
decays — REFUTED by measurement.** Nothing respawns at constant weight. Every
observed continuation multiplies its parent's weight by a factor ≤ 1 (the
cohort's factors are 0.046–0.105 per generation), and the only full-weight
spawns in the route are the once-per-pixel glossy-row rays at bounce 0
(`one3`, `sheet_resolve_taichi.py:889`), none of which reach the tail. What IS
true is the second half of the hypothesis's question: a product of
reflectivities can stay *above* the spawn gates for several generations
(ior=5 shells ⇒ factor ~0.07 ⇒ ~3.5 generations above 1e-3 from a 1e-2
spawn), but it cannot stay above it *forever*. The weight floor itself is
`MIN_WEIGHT = 1e-3` applied to the max colour component
(`raytrace_kernels_taichi.py:132`, test at `wavefront_kernels_taichi.py:3591`).

**(b) A ray whose weight is zero staying active because the exit tests
something else — REFUTED as stated, but adjacent to the truth.** Zero-weight
rays do exit: the E3 comparison `< MIN_WEIGHT` fires at exactly 0.0, and the
run shows it firing (all 30 deaths, plus ordinary pass-through retirements
with `w_post == 0`). The survivors' weights were never zero — they were small
but positive — and the exit did not "test something else"; it was simply
**never reached**, because every bounce branch `break`s past line 3591 and the
post-loop tests deliberately exclude bounced rays (`not bounced`,
`:3600-3601`). The precise failure is control-flow reachability of E3 for
S1 rays, not the predicate E3 evaluates.

Note the renderer's own comment at `:3602-3608` shows awareness that bounced
rays arrive at the post-loop block ("a ray still active here ... either
bounced or had hits left to drain") — but only the *surface-count* ceiling was
added there, not the weight floor.

## 6. Verdict

**Legitimate work, correctly rendered, with a missing cheap exit — and the
"rides to `max_iters`" claim in DESIGN_T4 §2 is factually wrong.**

1. **No loop runs past bounce 7** on this scene (§2); nothing reaches
   `max_iters = 276`. The cohort terminates at the bounce cap by design. The
   likely origin of the wrong claim is reading the bounce table's last-row `-`
   as "still alive": the instrumentation can never print a value there
   (`profiling_utils.py:1282-1286`).
2. **The image is not wrong.** Each ray's contribution is computed and
   committed correctly; nothing renders black or loops forever. What is wrong
   is consistency: a ray whose throughput falls under 1e-3 is treated as spent
   if it *pass-throughs* (E3 retires it) but as worth 8 full bounces if it
   *reflects*. Same transport budget, opposite treatment, purely because of
   which branch it happened to take.
3. **Cost is small but real, and it is what the tail rows measure.** With the
   cohort culled when it crosses the floor, no tile's drain would run past
   bounce 4–5 (every member is sub-floor by its first recorded tail
   iteration). Bounces 5–7 exist solely for these rays: warm CPU run
   0.066 s of 5.413 s bounce-loop kernel time (~1.2%), cold 0.071 s of
   13.648 s; T4 HD tail ≈ 0.78 s including launches (DESIGN_T4 §2 table).
   Per frame-part that is the "~3 extra launch pairs" the design doc counts —
   real, but caused by cap-riding sub-floor rays, not immortal ones.

## 7. Proposed fix (predicate + site; NOT implemented)

In `wavefront_shade`'s post-loop block, immediately after the peel-complete
test and before the surface-ceiling test (i.e. between lines 3601 and 3602 of
`wavefront_kernels_taichi.py`), add the same floor unconditionally:

```python
if ti.max(weight[0], ti.max(weight[1], weight[2])) < MIN_WEIGHT:
    done = True   # completion, not truncation: do NOT touch ALLOC_TRUNC_SURFACES
```

Effect: a ray whose throughput crosses the floor retires the same iteration
even if its last act was a bounce; the existing commit block (`3638-3654`)
deposits its accumulated colour and leftover throughput exactly as for any
other completion (with an env map, the leftover samples the map, `:3644-3650`
— unchanged). On the nn scene the plateau disappears: measured crossing
points put every cohort member's retirement at bounce ≤ 5 instead of 7.

Optional symmetry, not required for termination: the sheet resolve skips its
own floor check after a bounce too (break at `sheet_resolve_taichi.py:796`
before the `:1041` test); a bounced primary hands to the drain loop, which
would catch it one iteration after the fix above.

Byte-safety, stated honestly (**[inference]** — no A/B was run, per the
read-only constraint): this is not provably byte-identical. It retires
transport the renderer currently traces, bounded by the same envelope the
existing floor already accepts: dropped contribution ≤ w × radiance with
w < 1e-3 at cull time and measured decay ≤ ×0.105/generation afterwards, so
≲1.1e-3 of full scale pre-tonemap — under half an LSB of the u8 output for
any scene radiance ≤ ~2, and far less for these rays (their recorded
accumulator deltas are ≤ ~1e-3 per channel while alive, and their final
commits are exactly 0). Verification protocol per repo convention: alternating
A/B parity render (`benchmarks/_*_check.py` style), the fast suite's one
pixel-compared render, then `tests/full_renders` on CUDA and CPU with
baselines regenerated only after eyeballing the diff.

## 8. Probe inventory and artifacts

```
scratch_perf/r3/probes/
  run_nn_preview_cpu.py      driver: shims torch.cuda.synchronize, runs the stock benchmark
  repro_run.log              failed stock attempt before the shim (torch.cuda.synchronize raise)
  nn_preview_cpu.log         completed stock benchmark log (report written)
  probe_survivors.py/.log    probe v1 (small iterations, dict snapshots)
  probe_survivors2.py        probe v2 (all iterations, tensor snapshots)
  survivors_v2.pt            v2 raw records (selects/shades/traverses clones)
  analyze_v2.py              per-ray trace reconstruction -> survivor_traces.json
  probe_micro.py             single-frame sync/content sanity of traverse buffers
  probe_mat.py               paired traverse+shade capture incl. packed material rows
  plateau_stats.json         the 30 cohort members: pixels, classes, crossings, death weights
  report_nn_PREVIEW_cpu.txt  copy of the CPU profile report
scratch_perf/r3/ox/REPORT_immortal_rays.md   this file
```

Traps honored: `ALGAN_USE_DAEMON=0` on every run; no full unit suite; no
chasing of `tests/full_renders` baseline failures. Production files are
untouched by this task: everything under `/scratch_*` is gitignored, and the
one tracked-file change in git status (`DESIGN_T4_optimization.md`) belongs to
the concurrent session, not to this diagnosis.
One concurrent-session hazard materialized: the CPU profile report initially
landed at `benchmarks/performance/algan_profile_report_nn_PREVIEW.txt` and was
moved aside mid-session by another task; copies are preserved under
`scratch_perf/r3/` as listed above.
