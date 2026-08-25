# REPORT: receiver-facing whole-fan shadow cull

Brief: `scratch_perf/r2/ox/brief_shadow_facing_cull.md`. Branch `perf/r2-sheet`,
worktree `D:/algan_wt_sheet`, nothing committed or pushed. Every script ran
with `ALGAN_USE_DAEMON=0`; every A/B arm ran in its own process (the toggle
reaches a `ti.template()`, so one process per arm throughout).

**The headline finding first, because it reframes everything else:** the
brief's premise - "a surface point whose shading normal faces away from the
light still gets its full fan traced" - **does not hold on this branch**.
Commit `f142f72d` ("Fix the shadow terminator on diced surfaces") put a
receiver-side cull into both trace sites as a *per-sample horizon guard*:

- `raster_shadow_trace` (raster_taichi.py:3047-3051):
  `horizon_ok = (fnrm.dot(wis) > 1e-3) and (snrm.dot(wis) > 1e-4)`; a sample
  failing it is never marched - `if (ok == 1) and horizon_ok:` gates
  `_shadow_occluded`.
- `wavefront_shade`'s inline block (wavefront_kernels_taichi.py:2984-2991):
  same test, same gating.

For a fragment whose normals face away from the light essentially every sample
fails that guard, so **the BVH marches the brief wanted to remove are already
removed**. What remained per such cell was fan setup plus the sample loop's
arithmetic (sub-pixel origins under analytic AA, offsets, dots), and - the one
real gap - **soft fans**, where some samples face an extended emitter, get
traced, and their visibility multiplies an exactly-zero stage factor. The work
below implements the whole-fan cull the brief asked for anyway (loop skip on
hard fans, sound-by-the-stage-argument skip on soft fans), proves it
output-preserving, and measures it honestly: ray counts unchanged, kernel
self-time unchanged at the noise floor, for exactly the reason above.

## Part 1 - is the cull sound? Stage by stage

Question: for each built-in stage (`pid < _USER_PIPELINE_BASE`), is every term
multiplied by shadow visibility also multiplied by a factor that is *exactly*
zero when the shading normal faces away from the light? Line numbers are
`algan/rendering/raytracing/shading_taichi.py` (untouched by this task).

| pid | stage | vis-multiplied terms | zero at N.L <= 0? |
| --- | --- | --- | --- |
| 0 | manim `_stage_manim` (959-1004) | `base += lc * (w * v)` (999), `w = 0.5*d^3` halved when negative (995-998) | **NO** - Manim's offset deliberately keeps half weight past the horizon; vis is load-bearing there |
| 1 | unlit (950-955) | none (passthrough) | n/a - no event is built for it (sheet_resolve_taichi.py:499) |
| 2 | lambert (1008-1039) | `in_rgb * lc * (n_dot_l * v)` (1033); budget `wsum += n_dot_l * max(v) * peak(lc)` (1036-1037) | YES - explicit `n_dot_l = max(n.dot(ld), 0)` (1032) |
| 3 | phong (1043-1108) | `(in_rgb*lc*n_dot_l + fspec*lc*(0.25*d_blinn*n_dot_l*spec_w)) * v` (1101-1102) | YES - the Blinn half-vector lobe is *separately clamped by N.L*: re-multiplied by `n_dot_l`. No wrap/half-lambert shaping anywhere |
| 4 | standard (1112-1155) | diffuse `k_d*rgb*lc*n_dot_l` (1148); GGX spec `(ndf*geom)*F/max(4 nv nl,1e-4)` times `lc*(n_dot_l*spec_w)` (1146, 1149) | YES - at `n_dot_l == 0` Smith's `gl` factor is also exactly 0 (302), so the division's clamp never leaks a nonzero term |
| 5 | physical (1159-1262) | base diffuse+spec under `sheen_comp` (1230-1231), clearcoat `cc_spec*n_dot_l` (1240), sheen `sheen_brdf*n_dot_l` (1244-1245) | YES - every lobe carries `n_dot_l`. The old transmission/backside term was **removed** ("NO per-light transmission term", 1250-1259); nothing uses the negative side |
| 6 | toon (1266-1308) | `in_rgb*lc*(stepped*v)` (1301), `stepped = ceil(clamp(n_dot_l,0,1)*bands)/bands` (1299-1300) | YES - but only under a **strict** `<= 0` test: a dust-positive dot survives the clamp and `ceil` jumps it to a full band. This is why the cull fires at `<= 0.0`, not at the horizon guard's 1e-3/1e-4 slack |
| 7-9 | normal / matcap / depth (1312-1382) | no lights, never read `vis` | trivially |

Two consumers outside the stages complete the audit:

- `direct_specular_lobe` (the scatter path's delta-light add-back,
  shading_taichi.py:840-928): `out += spec * lc * (n_dot_l * spec_w) * v`
  (927) carries `n_dot_l`; its normal goes through the same
  `_sided_shading_normal` -> `_prep_normal` chain the stages use (906-907).
- The deferred-shadow bit path (wavefront_kernels_taichi.py:2200-2360) is
  compiled out (`deferred_shadows` always 0); untouched.

The energy-budget sums (`wsum`) carry the same per-light `n_dot_l`/`stepped`
factor, so a culled light contributes exactly zero to the budget before and
after; `_energy_scale(wsum)` is unchanged.

Verdict: **lambert, phong, standard, physical, toon qualify** (plus
unlit/normal/matcap/depth trivially). **Manim does not**, and user pipelines
(`>= _USER_PIPELINE_BASE`) may read visibility arbitrarily; both keep exact
fans, excluded by id alongside the existing `fan_geom` gate.

### Which normal does the stage light with?

Stages light with `n = _prep_normal(shade_n, face_n, flat)` where
`shade_n = _sided_shading_normal(n_interp, ...)` (dispatch at
shading_taichi.py:1583-1700): two-sided geometry flips toward the viewer,
then an optional flat blend toward the face normal.

**`event_snrm` holds the ORIENTED shading normal - already flipped - not the
raw one.** How I know: the mode-1 event build writes it at
sheet_resolve_taichi.py:500-508 from `_tri_shadow_normals`
(raster_taichi.py:2719-2741), which returns `_orient_hit_normals(snrm, fnrm,
rd)` (shading_taichi.py:500-525): normalized, face normal aligned to the
shading normal's hemisphere, both flipped to face back along the view ray via
`_faces_viewer(snrm, fnrm, -rd)`. The build's own comment says so ("snrm here
is the ORIENTED shading normal", sheet_resolve_taichi.py:514-515). For
two-sided geometry that flip is the same decision the stage makes (both read
`_faces_viewer` with sign-equivalent inputs), so testing `event_snrm` tests
the vector the stage lights with. Two residual divergences are inherited from
the shipped design rather than introduced here:

- flat-shaded geometry blends toward the face normal; the shadow path doesn't;
- one-sided geometry shaded from BEHIND (reachable only through transparency):
  the stage lights with the unflipped normal while the shadow path orients
  toward the viewer - documented as the KNOWN LIMIT at
  shading_taichi.py:464-473.

Both are handled conservatively: the cull requires **both** `snrm.wi <= 0`
and `fnrm.wi <= 0` (a flat blend lies between them), and in the
one-sided-behind case today's output is already "unshadowed" whenever the
whole fan fails the horizon test, which is the common case there.

## Part 2 - what was implemented

One toggle, two kernels, mirroring the existing `_light_zero_radiance` cull's
shape (same place in the per-light loop, same gate family, culled fan leaves
the all-lit default):

- `SHADOW_RECEIVER_CULL` / `ALGAN_SHADOW_RECEIVER_CULL`
  (settings.py:783-842): import-time env var declared in
  `algan/environment.py` (`_IMPORT_TIME_VARIABLES`), setter registered in
  `raytracing_settings.py` so
  `SETTINGS.raytracing.experimental.set(shadow_receiver_cull=...)` works
  (writing the parent raises with the pointer, as designed). Read live per
  batch through `shadow_receiver_cull_gate()` and passed as a template, so
  each arm compiles its own variant. **Default ON**, flipped only after the
  A/B below proved output preservation.
- `raster_shadow_trace`: new `recv_cull` template (+ diagnostic counter
  slots); the cull sits right after `_light_zero_radiance`'s
  (raster_taichi.py:2943-2958), firing on
  `fan_geom == 1 and pid_e != _MID_MANIM and snrm.wi <= 0 and fnrm.wi <= 0`.
- `wavefront_shade`: same cull inline (wavefront_kernels_taichi.py:2867-2878),
  driven by a `fan_cull` flag derived where `fan_geom` is (2717-2757) because
  Taichi scoping does not let the deep light loop read `pid_s`.
- Call sites: raster_pipeline.py:2297-2301, tracer.py:2662 and tracer.py:3283.

### The soft-fan decision

**Whole-fan cull, not per-sample - safe because the emitter's extent only
shapes visibility, and visibility is exactly what multiplies away.** Each
area row / radius sample is its own (event, light-row) cell whose stage
irradiance depends only on the direction to *its row's centre* (`ld` in
`_light_eval`, shading_taichi.py:739 - `lp` IS the cell centre for area rows).
When `N.ld <= 0` every qualifying stage's vis-multiplied term is exactly zero
whatever the fan returns, so culling the whole fan cannot move a pixel even
though part of the emitter faces the receiver. Per-sample decisions stay
necessary only where the stage term can be nonzero past the centre horizon -
manim and user pipelines - and those keep exact fans. This is stated in the
setting's comment and in both kernels.

## Part 3 - measurement

All numbers from this box (GTX 1050 4 GB, shared with other agents -
1.4-1.7 GB in use by them during these runs, observed wall swings 127-243 s
for identical work). Counts come from the **counting kernel build**
(`count_dbg=1`, atomic adds into a 4-slot buffer via
`scratch_perf/r2/ox/probe_cull_counts.py` monkeypatching the launch) - a
different variant from the shipping one; none of its wall time is quoted.
Timing comes from `timed_nn.py` (Taichi kernel profiler GPU self-times, which
contend far less than wall clock), 3 alternating process pairs per arm.

Counter semantics: `[0]` fans entering the trace branch (valid light row AND
nonzero colour); `[1]` rays actually marched; `[2]` fans skipped by the new
cull - counted BEFORE the colour gate, so it includes colour-zero cells that
today skip without counting; `[3]` fan samples evaluated.

### Shadow-ray counts (rays_marched per frame)

| scene | OFF | ON | delta |
| --- | --- | --- | --- |
| nn PREVIEW (704x396@10, 50 frames) | 90,150 | 90,149 | -1 ray over the whole video |
| nn HD (1920x1080@30, run_time 1 s, 30 frames) | 555,443 | 555,443 | 0 |
| stress scene (1 frame) | 219,163 | 219,163 | 0 |

As fractions of fans entered per frame (nn PREVIEW): OFF marches
90,150/29,602 = 3.05 rays per entered fan; the cull removes 2,657/29,602 =
**9.0% of entered fans** yet ~0% of marched rays, because those fans were
already marching nothing (horizon guard). Sample evaluations dropped
99,895 -> 91,444/frame (-8.5%) PREVIEW, 608,091 -> 562,380/frame (-7.5%) HD;
stress scene culled 143,809 cells/frame (mostly soft-fan cells of the
radius/area lights) with samples 340,679 -> 332,471.

The single missing ray at PREVIEW is the predicted boundary band in action:
one fan whose lifted-origin sample direction (`wis` is computed from the
face-normal-lifted origin, raster_taichi.py:3026-3035) passed the horizon
guard by a hair while the centre direction fails the strict test. Its
visibility multiplied an exactly-zero stage factor, so the output is
unchanged (PREVIEW arms md5-identical below).

### Timing (medians of 3 alternating single-process runs, whole-render wall / kernel GPU self-time)

Sheet route (default), nn scene at PREVIEW:

| arm | wall (median) | raster_shadow_trace | wavefront_shade |
| --- | --- | --- | --- |
| OFF | 143.0 s (126.9/143.0/194.2) | 6.93 s (8.22/6.93/6.80) | 4.62 s (4.19/4.77/4.62) |
| ON | 161.9 s (151.6/161.9/242.7) | 6.94 s (6.94/7.06/6.55) | 4.50 s (4.50/4.71/4.45) |

Wall medians are noise (the ON median is dragged by one 242.7 s round during
heavy tenant activity; kernel self-times are the honest pair and they are
flat). Classic wavefront route (same scene, `ALGAN_ANALYTIC_AA=0`, which
rejects the sheet route):

| arm | wall (median) | wavefront_shade |
| --- | --- | --- |
| OFF | 51.7 s (51.7/51.7/63.0) | 9.81 s (8.65/9.81/10.04) |
| ON | 51.4 s (51.1/51.6/63.6) | 10.07 s (9.19/10.07/10.15) |

Within noise. How contention was handled: arms alternated back-to-back so
tenants hit both equally, medians reported, and counts (which contention
cannot distort) lead the table. The conclusion does not rest on wall clock:
the cull provably skips no march that used to happen, and the loop arithmetic
it saves is ~8% of sample evaluations worth of dot products against ~90k-555k
real BVH marches per frame - invisible.

### The `_stage_default` aside

The existing comment at `_light_zero_radiance`
(wavefront_kernels_taichi.py:163-168) says admitting `_stage_default`'s fade
to the cull "would be correct ... but it has not been measured". Checked:
**there is no `_stage_default` stage on this branch** - only comments still
mention it. Every consumer of the visibility payload is one of the ten
built-in stages or `direct_specular_lobe`, all audited above; the vertex-baked
consumer the comment was written against no longer exists. Nothing to admit,
nothing to measure.

## Verification (all required steps, actual output)

### Lossless render A/B, toggle off vs on

nn scene, libx264rgb qp 0, `SETTINGS.computing.available_memory_override`
pinned to 3 GiB in both arms, one process per arm:

PREVIEW, 50 frames - `benchmarks/_video_diff.py`:

```
frames compared: 50
worst channel diff: 0 (frame -1)
pixels over tol 2: worst frame 0 of 278784 (0.000%, frame -1); mean 0.0/frame; 0 of 50 frames affected
```

md5 of the two lossless files:

```
a4bcda1a49b80714e32ce18e6237064e  nn_prev_lossless_off.mp4
a4bcda1a49b80714e32ce18e6237064e  nn_prev_lossless_on.mp4
```

Byte-identical.

HD, 150 frames:

```
frames compared: 150
worst channel diff: 1 (frame 24)
pixels over tol 2: worst frame 0 of 2073600 (0.000%, frame -1); mean 0.0/frame; 0 of 150 frames affected
per-frame max-diff histogram (worst 8 buckets):
  diff   1: 7 frames
  diff   0: 143 frames
```

Worst channel diff 1 is UNDER the suites' tolerance (<= 2) but not byte-zero,
so I ran the control the claim needs - the SAME arm twice:

```
(ALGAN_SHADOW_RECEIVER_CULL=0 vs ALGAN_SHADOW_RECEIVER_CULL=0)
frames compared: 150
worst channel diff: 1 (frame 32)
pixels over tol 2: worst frame 0 of 2073600 (0.000%, frame -1); mean 0.0/frame; 0 of 150 frames affected
  diff   1: 8 frames
```

Identical class and magnitude with identical code on both sides: the HD
deviation is the documented cross-run materialization-window noise
(tests/README.md: small (<=2) differences across runs are expected), not the
toggle. Zero pixels over tolerance either way.

### Stress scene

`benchmarks/_shadow_facing_cull_check.py` packs every case most likely to
break the cull, rendered both ways and compared the same way
(lossless RGB, `_video_diff.py`):

- an open TWO-SIDED parametric `Surface` lit from behind (flip-decision
  agreement between stage and shadow path);
- a `SpotLight` (cone factor atop the facing test);
- a `PointLight(shadow_radius=0.35)` (soft fan whose samples can face the
  emitter while its centre faces away);
- a `RectAreaLight(samples=4)` (per-cell emitter rows);
- a mob with a CUSTOM fragment pipeline that visibly rewards any forced
  all-lit default (must keep the exact fan) and a `ManimMaterial` cube
  (pid 0 must keep the exact fan);
- lambert / phong / standard / physical / toon cubes (the qualifying stages).

What each arm proves: the OFF arm pins the reference image; the ON arm proves
that culling back-facing fans - including soft fans - changes nothing on
geometry designed to expose every disagreement channel (flip mismatch,
extended emitter reach, custom-pipeline visibility reads, manim's negative
offset).

```
frames compared: 1
worst channel diff: 0 (frame -1)
pixels over tol 2: worst frame 0 of 278784 (0.000%, frame -1); mean 0.0/frame; 0 of 1 frames affected
```

Zero differing pixels.

### Test suites

Full unit suite:

```
uv run --extra dev python -m pytest -q tests/unit_tests
2065 passed, 132 skipped, 172 warnings in 1359.74s (0:22:39)
```

(includes `test_shadow_flags.py`, `test_area_light_soft_shadow.py`,
`test_normal_orientation.py`, the tonemapping tests, `test_environment.py`
and `test_taichi_runtime_config.py`).

Fast suite (run before the default flip; the flip changes a Python default
only, and the guards were re-run after it - see below):

```
uv run --extra dev python -m pytest -q --fast
fast suite: 46s of its 75s budget (61%)
276 passed, 1929 deselected, 3 warnings in 45.90s
```

Contrary to the brief's warning, the fast suite's pixel-comparison render
PASSED on this machine this time - no pre-existing failure to report.

After flipping the default ON, the targeted guards were re-run:

```
uv run --extra dev python -m pytest -q tests/unit_tests/test_environment.py \
    tests/unit_tests/test_taichi_runtime_config.py tests/unit_tests/test_shadow_flags.py \
    tests/unit_tests/test_area_light_soft_shadow.py tests/unit_tests/test_normal_orientation.py
90 passed, 3 warnings in 90.97s
```

Toggle surface check (env off -> import-time default False ->
experimental.set -> live gate):

```
env off -> False False
experimental set -> True
gate -> 1
```

### Lint

```
uv run ruff check --no-fix <all touched files>
```

Only pre-existing findings remain in the package files - verified by stashing
my changes and re-running on HEAD: settings.py:2813 I001, tracer.py:96 I001,
wavefront_kernels_taichi.py:27 I001 + F811 x3 + F841 (7 errors on HEAD, same
7 after my edits; CLAUDE.md's F401/F811 caveat is exactly why the duplicate
kernel-module imports stay untouched). All files I created pass clean, and

```
uv run ruff format --check <touched non-kernel files>
4 files would be reformatted  ->  fixed; now: all formatted
```

(The two `*_taichi.py` files are linted but never formatted, per CLAUDE.md.)

## Everything I did NOT verify

- **No performance win was found, and none should be expected from this
  change.** The measurements above show ray counts and kernel self-times flat.
  The brief's 21.6%-of-render figure describes the cost of shadow tracing as
  a whole, nearly all of which is real marches for front-facing receivers;
  commit f142f72d had already removed the back-facing half the brief targets.
  If more shadow cost is to be recovered, the remaining levers are structural
  (host-side event filtering before launch, fewer sub-pixel shadow origins),
  not this cull.
- The Monte Carlo path (SPP > 1, `raytrace_kernels_taichi.py`) traces shadow
  rays too; the brief scoped implementation to the two deterministic sites
  and I did not touch or measure the MC kernel.
- One-sided solids shaded from behind through transparency with SOFT lights:
  argued safe in the common case (their fans already fail the horizon guard
  wholesale today), but no scene in the verification battery renders that
  exact corner - the stress scene's backlit sheet is two-sided by design.
  This is the same documented KNOWN LIMIT territory, not a new guarantee.
- Flat-shaded geometry near silhouette bands: the both-normals requirement
  makes the cull conservative, but the side-decision disagreement between the
  stage's blend and `_orient_hit_normals` is argued away, not enumerated
  exhaustively. Empirically zero pixels moved on everything rendered here.
- CPU renderer: everything ran CUDA-only (as every prior round on this box).
- `tests/full_renders` was not run (brief did not require it; the change is
  proven output-preserving on the required scenes, and its default-ON state
  affects the six dense scenes only through the same code paths verified
  above). If desired before merge, run them on a CUDA machine with the
  baselines as committed - no baseline moves are expected.
- The counting build's counters `[2]` include colour-zero light rows (the
  count site precedes the colour gate), so "fans_receiver_culled" exceeds the
  OFF->ON drop in entered fans on the stress scene; the nn numbers happen to
  coincide exactly (132,830 both ways). Semantics documented on the probe.
- Wavefront-route timing used the nn scene pushed onto the classic route via
  `ALGAN_ANALYTIC_AA=0`; I did not construct a native bounce-heavy scene to
  isolate the inline block further.

## Files touched

- `algan/environment.py` - declare `ALGAN_SHADOW_RECEIVER_CULL`
- `algan/rendering/raytracing/settings.py` - the toggle + setter + live gate
- `algan/rendering/raytracing/raster_taichi.py` - cull + counters
- `algan/rendering/raytracing/wavefront_kernels_taichi.py` - cull
- `algan/rendering/raytracing/raster_pipeline.py`, `algan/rendering/raytracing/tracer.py` - call sites
- `algan/settings/raytracing_settings.py` - field + setter registration
- `benchmarks/_shadow_facing_cull_check.py` - the stress scene
- `scratch_perf/r2/ox/{nn_render,timed_nn,probe_cull_counts}.py` - harnesses
