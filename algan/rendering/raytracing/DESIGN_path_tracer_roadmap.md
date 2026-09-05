# Path tracer roadmap: what is deliberately missing, and what landing it takes

This is the engineering companion to the user-facing list in
`docs/source/advanced_user_tutorials/renderer_limitations.rst`: that page says
*what* the renderer does not do; this one says *why not yet* and sketches what
each feature costs to land in this codebase specifically. It is the plan of
record for the path tracer's remaining scope — update it when one of these
lands, the way `DESIGN_optimization_targets.md` is updated for optimization
work.


## What the path tracer is for

**It is the fallback that always works.** Not a second way to render what the
deterministic renderer already renders, and not an attempt to reproduce its
look. It exists for the scenes the deterministic renderer *cannot* do:

* **Global illumination**, and anything else that needs real transport —
  colour bleed, physically correct soft shadows, rough reflections that see
  the scene rather than a prefiltered proxy.
* **Scenes that exhaust memory in the deterministic renderer.** Its
  reflection/refraction branches split into a shared pool, so a scene with
  enough interacting transparent and reflective surfaces OOMs a single frame.
  The path tracer's per-path state is a fixed size and paths never split, so
  its memory is `slots x _PT_BYTES_PER_SLOT` and nothing else — it renders
  what the other one cannot fit.
* **Scenes with too many lights.** The deterministic renderer's cost is
  linear in light count and its shadows are capped at
  `ALGAN_MAX_SHADOW_LIGHTS` (16, with each `RectAreaLight` emitter sample
  eating a slot). A hundred-light scene is not slow there, it is impossible.
  The path tracer samples lights instead of summing them, so its cost per
  vertex is independent of how many there are.

Three consequences, and they are the reason several sections below reach a
different conclusion than they first did:

1. **Parity with the deterministic renderer is not a goal.** Where the two
   disagree, the path tracer should be *right*, not matching. A user reaches
   for it because the other renderer could not do the job, so there is
   usually no side-by-side to preserve — and the two already differ visibly
   (no ambient fill, no glossy prefilter, jittered rather than analytic AA).
   Partial parity that no one can rely on is worth less than correctness.
   See §5.
2. **Cost must not be linear in light count, anywhere.** That is one of the
   three reasons the fallback exists, so any `for li in range(num_lights)`
   in the hot path is a defect against the renderer's purpose, not a missing
   optimization. See §6.
3. **Bounded memory is a feature, not an accident.** "Renders what the
   deterministic renderer OOMs on" is a promise that fixed per-path state
   makes and that splitting takes away. Every continuation-pool proposal in
   §8 spends exactly the property that makes this renderer the fallback, and
   must be judged on that basis.

What it does *not* have to be: deterministic (the byte-reproducibility
guarantee was withdrawn — see contract 1), or a match for the deterministic
renderer's brightness, shading rate or edge treatment.

What it *must* be: **complete**. A fallback that refuses a feature leaves the
user with nowhere to go, and a fallback that silently drops one is worse. §9
is the audit of where it is not yet complete.

Status: the redesign's staged landing table (stages 1–6) is complete on this
branch — the wavefront skeleton and deterministic 2-D compositing, BSDF
sampling (cosine diffuse, spherical-cap VNDF GGX with Turquin compensation,
nested-media refraction), the power-weighted NEE table over delta/area light
rows, emissive triangles and the environment map's 2-D luminance CDF, the
Sobol–Owen sampler, Russian roulette, firefly clamping, jittered-pixel AA,
the closed-shell opacity ring, the `tests/path_traced/` baseline suite and
parity benchmark, and the OIDN RT denoiser with in-kernel albedo/normal AOVs.
On top of that table, §6a/§6b have landed: the Conty-Kulla light tree is now
the selection structure for next-event estimation (`light_tree.py`,
`pt_light_tree`), so emitter choice weighs distance and orientation rather
than power alone. §6a-ter has landed with them: a `RectAreaLight` is two
emissive triangles here rather than `K` packed cell rows
(`area_light_quads.py`, `pt_area_light_quads`). §6a-bis closed the last loop
that was linear in the light count: an authored-appearance material's direct
lighting is now sampled like everything else's past the shadow cap
(`pt_authored_light_sampling`), so the "too many lights" case is uncapped for
every material rather than for most of them.

Power-heuristic MIS covers the strategies that genuinely overlap: emissive
triangles — a `RectAreaLight`'s own quad included, since §6a-ter — and the
environment map, each sampled both by next-event estimation and by BSDF
continuations that land on them. It does **not** cover the remaining packed
light rows, and cannot: a delta light (point, spot, directional) has zero area
and is unhittable by construction, so no BSDF ray can find it. That is sound —
there is nothing to double-count — and it no longer costs anything visible.
The two consequences this paragraph used to name are both gone: a light row's
highlight is the same `_pt_lit_f_pdf` a BSDF ray would evaluate (§5), and an
area light **does** cast a reflected image in a mirror, because it is
geometry (§6a-ter). It is still invisible to camera rays and still not an
occluder, which is what the deterministic renderer does with it too.

What follows is the throughput baseline and the cheap wins the fallback role
puts first (§0), then everything the original plan named beyond the staged
table (§§1–4), then the divergences from the SOTA survey the redesign was
specified against that were *decided* rather than merely deferred (§§5–8),
then the completeness audit the fallback role demands (§9).

**The order of work, as of 2026-09-04.** The sections are numbered by the
history of the document, not by priority. The priority, under the purpose
above and the constraint that a fallback the user is *told* to reach for must
be as fast as it can be, is:

1. §0 — a measured baseline, then the kernel and host wins it ranks.
   **LANDED** (the T4 baseline, the denoiser in half precision with the
   adaptive pass-through, the opaque-batch gates, the packed ambient rows,
   the sampler hoists).
2. §0.3 — the defaults and the switch: what "turn on path tracing" means.
   **LANDED.**
3. §9 — the fallback never refuses. **LANDED**: custom scatter as a delta
   lobe, the never-refuses test, and the failure messages that point at the
   switch.
4. §2 — adaptive sampling. **LANDED**, with the stochastic gate that keeps
   deterministic pixels byte-identical.
5. §6 — the light tree and the authored-appearance sampling fix. **LANDED
   in full**: §6a/§6b's tree, §6a-quater's per-frame build, §6a-ter's
   area-light quads and §6a-bis's sampled authored lighting, each with a
   T4 A/B under `benchmarks/performance/reports/t4_2026_09/`. §5's single
   BSDF landed before them with §7's stratified lobe select, §0.2's
   sampler hoists and the self-intersection offset as one re-baseline.
   What this item leaves open: the area-light quads' leaf-bit end state
   (§6a-ter, 8% of device time on an area-light scene) and the authored
   surface's double count of an area light through its continuation.
6. §3 tier 2 and §8's pools — only behind a profile. §7's blue noise is
   built and measured (+2..4%, inside the noise), and ships off; what would
   make it pay is a per-dimension tile, which that section scopes.
7. §1 and §4 are not fallback work at all (each section says why) and sit
   behind everything above.


## The contract every one of these must land under

These are not preferences; each is load-bearing and tested.

1. **Sampler purity — but *not* byte-reproducibility.** Every random decision
   is a pure function of path identity: `(pt_seed, frame, pixel, dimension
   pair, sample index)` for Sobol pairs, plus the peel step for the hash RNG.
   Anything stochastic a new feature adds must draw from new, documented
   dimension pairs, never from shared mutable state.

   Purity is kept for what it buys *sampling*: stratification (a pixel's
   sample set is a proper low-discrepancy prefix, which is what makes
   progressive waves and future adaptive sampling sound), and independence
   from how a render was split into tiles, waves, chunks and batch windows —
   which matters because those splits come from the memory budget and change
   under the OOM retry. It is not kept in order to make two runs match.

   **The renderer does not promise byte-identical frames** (decided
   2026-08-29; the user-facing guarantee was withdrawn from
   `performance_and_quality.rst` and `renderer_limitations.rst`, and the
   three render-level reproducibility tests were deleted). It promises
   convergence. Concretely: accumulation may use atomics, paths may split,
   and summation order is not fixed by contract. None of that is *used*
   today — no path splits, so `pt_reduce` and the AOV reduction stay
   atomic-free and order-fixed, and frames do in fact reproduce — which is
   why `tests/path_traced/` can still pixel-compare at a pinned memory
   budget. Treat that as a convenience the current layout happens to
   provide, not as an invariant to defend. §8 is what it bought.
2. **The wavefront shape.** One traverse kernel shared with the deterministic
   renderer; `pt_shade` drains `kbuf`-sized hit batches; per-path state lives
   in `rs_ro/rs_rd/rs_sca/rs_int/rs_pix` + `pt_thru/pt_acc/pt_aov`. One
   compiled kernel serves every scene: per-scene facts ride runtime words
   (`nee_meta`) and runtime-gated branches, not new `ti.template()`
   arguments — a template argument multiplies the cold compile, which on the
   fallback is paid by a user who is already having a bad day. The one
   acceptable kind is a *two-valued* gate the deterministic renderer already
   compiles both sides of (the shadow any-hit mode, the closest-hit traverse;
   §0.2), so no new kernel body exists that did not before.

   The argument-count pressure this contract used to cite is stale: since
   arena packing (`arena_args_taichi.py`) `pt_shade_arena` takes 40
   parameters, not 59 of 64. New inputs still prefer widening an existing
   tensor (`nee_meta` has spare words, `nee_ref` can carry more row kinds)
   because that is cheaper than a parameter, not because the ceiling is near.
3. **Bounded per-path state.** All per-path and per-scene state is accounted
   in `_PT_BYTES_PER_SLOT` / scoped allocations so the tile/wave split and the
   OOM chunk-halving retry keep working — and, more than bookkeeping, the
   *size* is fixed: paths do not split, so a scene cannot make one path cost
   more memory than another. That is what lets this renderer finish scenes
   the deterministic one OOMs on, which is one of the three reasons it
   exists. A feature that makes per-path memory data-dependent is spending
   the renderer's purpose and needs to say so out loud, with a hard cap.
4. **The 2-D contract.** Camera-segment transparency composites with zero
   variance (`benchmarks/_pt_parity_check.py` holds flat interiors to ≤ 1
   channel count against the deterministic route at any spp). Note this is
   *not* a parity obligation of the kind §5 argues against: it is the one
   place matching is worth having, because a user who fell back to the path
   tracer for a 3-D reason should not lose text and vector-graphics quality
   as collateral. A feature that would make unlit stacks stochastic is wrong
   by construction here.
5. **The sampler dimension table** in `path_tracer_taichi.py`'s module
   docstring is the registry of who consumes randomness. Pairs
   `2 + 6b + 4, 5` are already **reserved for volumes**.


## 0. Throughput first: the baseline, the cheap wins, and the switch

Every section after this one ranks work by *variance*: what makes a sample
worth more. None of them asks what a sample *costs*, and until 2026-09-04
nothing in the repository did either — there was no path-tracer benchmark,
no recorded timing, and the denoiser had never been timed. For a renderer a
user is told to fall back to, that is the wrong first question. This section
is the throughput half of the plan, and it comes first because the profile it
produces is what should order everything below.

### 0.1 The baseline

`benchmarks/performance/pt_baseline.py` is the harness: three scenes
(an all-opaque lit scene, the same solids under 64 lights, and a 2-D text and
transparency stack), a warm RUN 2 per the gpu_harnesses rules, device-side
kernel times split into `pt_generate`, `wavefront_traverse_events`,
`pt_shade`, `compact_ray_slots`, `pt_reduce`, `finalize_samples` and the
denoiser, launch counts, and a `--deterministic` arm so the fallback's cost
is quoted *relative to what the user was rendering with*. Arms are selected
by environment variable, one process each (a `ti.static` gate is resolved at
compile time; `agent_guidance/taichi.md`). The T4 is the box to run it on;
the Mac's per-launch numbers are not sound (`agent_guidance/gpu_harnesses.md`).

Three numbers the baseline must produce before any section below is
scheduled: the share of a lit frame spent in traversal versus shading versus
host round-trips; the share of a text frame spent re-peeling the camera
segment per sample (the §2 case); and the denoiser's fixed per-frame cost,
which is independent of spp and bounds how low spp can usefully go.

**Measured, Kaggle T4, 2026-09-04**
(`benchmarks/performance/reports/t4_2026_09/pt_baseline_1.md` has every
arm). Five frames, 16 spp, 4 bounces, denoiser on, warm RUN 2:

| lit scene, 1280x720 | s | % |
| --- | --- | --- |
| host: prep, merge, encode | 0.879 | 50 |
| denoiser (fp32 torch U-Net) | 0.383 | 22 |
| `pt_shade` (NEE + shadow rays) | 0.248 | 14 |
| `wavefront_traverse_events` | 0.152 | 9 |
| generate + reduce + compact | 0.051 | 3 |

* The path tracer's kernels are **0.09 s per 720p frame** (~16 M paths/s
  with NEE and shadows); the deterministic renderer's kernels on the same
  scene are 8 ms per frame. End to end the fallback is **1.7x** the
  deterministic renderer here, because host prep and the denoiser set the
  wall clock, not transport.
* **The denoiser was the largest device-side item**: 77 ms per 720p frame,
  fixed per frame, fp32, NCHW, per tile, through plain `F.conv2d`. It was
  34% of the 2-D text arm, where it has zero-variance pixels to filter.
  **LANDED**: `denoise_precision` defaults to half precision with
  channels-last activations on CUDA (`pt_denoise_1.md` in the same
  reports directory): 1.7x on the filter at 720p and 1080p, at most one
  8-bit count of difference on a handful of channel samples, 9-14% end to
  end. Batching the tiles into one forward pass and compiling the U-Net
  are the next two steps if it climbs back to the top of the profile.
  **Also LANDED, after §2**: the filter takes the adaptive sampler's
  stochastic mask, passes every exact pixel through untouched and skips
  tiles whose core holds no flagged pixel — so a 2-D text frame, on which
  the denoiser was 29-44% of the adaptive wall clock at 720p-1080p
  (`pt_adaptive_1.md`), costs nothing to denoise, and the filter no longer
  softens exact edges (contract 4). `pt_error_target = 0` restores the
  whole-frame filter.
* **Many lights is flat**: 64 point lights cost what 3 do
  (`pt_shade` 20.1 ms against 18.2 ms at 320x180).
* **The 2-D arm**: 1.4 iterations per wave, 0.30 s of kernel time per five
  720p frames spent re-peeling zero-variance content 16 times. That sizes
  §2 at roughly a 16x reduction of that 0.30 s.
* **The two opaque switches** (§0.2) took 11% off the path tracer's device
  time at 720p, byte-identical video on both arms.
* **The host sync per iteration does not pay to remove** — see §0.2-bis.

### 0.2 Cheap wins the survey found in the kernel

Each is a half-day to a day, each ships behind an experimental kill switch,
and each has a byte-identical acceptance test on the scene class it applies
to — which is what makes them cheap to land without a re-baseline:

* **Shadow rays always took the full ordered march.** `_pt_nee_visibility`
  hardcoded `anyhit = 1`. The deterministic renderer chooses mode 3 (opaque
  any-hit, the march compiled out) when the batch is provably all-opaque
  (`has_transmissive`, `tri_has_translucent`, `bez_has_translucent`,
  `has_uncertain_texture_alpha` all clear — `tracer.py`'s decision). Shadow
  rays are the majority ray type in this renderer (one per light sample per
  lit crossing, against one continuation per bounce), so the path tracer now
  takes the same decision (`pt_shadow_anyhit`). Mode 2, the mixed-batch
  pre-pass, measured as a loss on the deterministic renderer and is not
  used. Acceptance: byte-identical on an all-opaque batch, up to the two
  corner cases `_shadow_occluded`'s docstring documents.
* **Secondary rays gathered a four-deep k-buffer on opaque scenes.** The
  shared traverse kernel has an `opaque_closest` mode (one nearest hit via
  `_nearest_surface_g`) that `path_trace_render` passed 0 for as a
  "deterministic-only rollout". A path tracer wants exactly closest-hit for
  an opaque batch; `pt_opaque_closest` passes it under the same
  `all_visible_opaque` gate the deterministic renderer uses. Acceptance:
  byte-identical, since the nearest hit of an opaque batch is the first
  k-buffer entry.
  Both opaque gates are host decisions on the merged batch's flags, so a
  single translucent primitive takes them off for the whole batch — and
  `Cube` ships `fill_opacity=0.75` (Manim's default), so a scene with a
  default cube is a mixed batch. `benchmarks/_pt_shadow_anyhit_check.py`
  reports what the host handed the kernels for exactly this reason: a pass
  where both arms ran mode 1 proved nothing. The pure-code-motion item that
  first shipped alongside these (decoding `nee_meta` under the hit test)
  is kept, but *without* a switch: it is byte-identical by construction
  and a template gate for it would have doubled the variant count for
  nothing, which is contract 2's whole concern.
* **The ambient / hemisphere fill scanned every light row per lit crossing**
  (the first bullet of §6a-bis). The host now appends the direction-less
  rows after the `E` sampled entries of `nee_ref` with their own kind, and
  their count rides `nee_meta`; the kernel loops the count. Byte-identical.
* **`nee_meta` was decoded per path per launch before the hit test**, so a
  path with no hits paid eleven loads for nothing. Moved under the test
  (no switch, see above).
* **Sampler overhead per draw — LANDED** (with the §5 re-baseline).
  `pt_sample_2d` re-derived the `(seed_root, key)` half of its seed on every
  call although it is constant for the whole path. The two hashes are now
  nested the other way round — `_pt_path_seed(seed_root, key)` once per
  `pt_shade` thread and once per `pt_generate` slot, then
  `_pt_pair_seed(path_seed, pair)` per draw — which is what makes the
  per-path half loop-invariant at all; the old spelling
  `combine(seed_root, combine(key, pair))` put a `pair`-dependent term
  inside *both* hashes. Per-dimension independence is unchanged, because
  `_pt_hash_combine` avalanches its second argument. `_pt_rng` took the same
  treatment (`_pt_rng_seeded`) and its unhoisted spelling is gone, leaving
  only the test probe on `pt_sample_2d`'s original signature. `pair_nee0` is
  hoisted per path, alongside the width of one crossing's pair block.
  The roulette draw still computes a full 2-D pair and keeps one component,
  deliberately: it is one draw per bounce and the dimension table documents
  the unused `x`.
* **The authored-appearance shadow loop — LANDED (§6a-bis).** It was the
  remaining linear-in-lights term and it needed the interface decision that
  section describes, which is why it is recorded there and not here.

### 0.2-bis The host sync every iteration

`_ArenaRayCompactor.select` reads `count.item()` after every traverse and
shade pair — a device sync per iteration — and the peel loop and the bounce
loop are fused into one host iteration (`pt_shade` performs at most one
scatter per launch and drains at most `kbuf` crossings), so a wave costs on
the order of *bounces plus crossings-over-four* round trips, each a sync and
three launches. At small tiles — low resolution, or a memory budget that
forces many waves — this is where the wavefront shape's overhead lives, not
in the inline shadow walk §8 worries about.

The fix, when the baseline shows the sync matters: allocate the hit batch
once per wave at the pool size the budget already charges
(`_PT_BYTES_PER_SLOT` counts `kbuf` events at `na == pool`), launch each
iteration over the *previous* live count with the compaction kernel writing
a `-1` sentinel over the tail and both kernels skipping `r < 0`, and read the
count back only every few iterations to decide termination. The traverse
kernel is shared, so its guard must be output-neutral for the deterministic
renderer; it is a single compare. This is a "measure first" item: the T4
number decides it, the CPU number does not (the CPU has no launch latency
to speak of, so it cannot see the cost).

**Measured 2026-09-04: deferred.** At 720p the arena budget puts a whole
frame's pixels in one wave with 16 samples in flight, so a render is 5
iterations per wave and 25 syncs per five frames; `compact_ray_slots` is
1.3% of wall and its wall time equals its device time, so the round trips
are hidden behind the kernels. The rewrite would only show at a small
memory budget, where waves are many and short. Revisit if a profile under
`available_memory_mb` pressure says so; until then the denoiser (§0.1) is
the item.

### 0.3 The defaults, and what "turn on path tracing" means

A fallback is reached for at the last minute, by a user whose scene just
failed. What the switch does by default therefore matters more than any
sampling improvement:

* **Fixed seed across frames is the default** (`pt_animated_seed = False`).
  `_pt_key` used to hash the frame into every sample, so residual noise
  re-rolled per frame — shimmer, which the eye and the denoiser both punish,
  and which forces spp *up*. Cycles ships the fixed seed and makes the
  animated seed opt-in; so does this renderer now. Static regions get
  identical estimates frame to frame; moving regions re-randomise through
  the geometry itself. Correlated error reads as a fixed noise texture, which
  is the better artifact for video. This was §3's tier 1, promoted from a
  switch to the default.
* **"Turn on path tracing" has one spelling** — LANDED as
  `render_loop.PATH_TRACER_FALLBACK_SPELLING`:
  `SETTINGS.raytracing.set(samples_per_pixel=16, max_bounces=2)`. Before,
  it meant "pick a `samples_per_pixel`" with `max_bounces` left at 8 and
  roulette from bounce 3 — the settings of a GI render, paid by a
  many-lights scene that needed one bounce. A preset *object* was
  considered and rejected: `RayTracingPreset` captures every one of the
  section's ~106 fields, so a `PATH_TRACED` constant would overwrite the
  user's other settings on `set(source=...)`, and the settings system has
  no partial-preset concept to add one to. A spelling the docs and the
  failure messages all agree on is the same affordance without a new type.
* **The failures name the switch** — LANDED. The 16-light shadow
  truncation warning and every one-frame `OutOfRenderMemory` now end with
  that spelling (the OOM hint is dropped when the path tracer is already
  the renderer that failed). The message at the failure is the
  documentation the user actually reads.
* **The user docs described a quality upgrade "at dramatically higher
  cost", never a fallback** — FIXED. `renderer_limitations.rst` and
  `performance_and_quality.rst` told the user to reach for the path tracer
  "when you need full light transport"; neither said it is the answer to a
  many-light scene or a scene that exhausts memory, and neither stated
  that the deterministic renderer's cost is linear in light count. Both
  now lead with the three failure classes and the spelling above.

### 0.4 What this section deliberately leaves alone

The sampler's arithmetic cost (~150–200 integer ops per 2-D draw, register
resident) is real but is not the bottleneck while a traversal costs more
than a hundred draws; it is on the §5 re-baseline list, not here. Kernel
variant count is protected by contract 2 and matters for the same reason
the defaults do: the fallback's cold compile is paid at the worst moment.
And the megakernel-ness of `pt_shade` — NEE, the shadow walk, BSDF sampling,
the authored pipeline, the shell ring and the AOVs in one body — is accepted
until a profile says register pressure is what limits occupancy; splitting
it is §8's shadow-ray queue, and §8 says why that waits.


## 1. Caustics

**Not fallback work.** Neither renderer produces caustics, so their absence
never leaves a scene with no renderer (the §9 test). This section is the
plan for a *new* capability and sits behind everything in §0, §2, §5, §6
and §9. When it lands it lands behind a switch that defaults off: MNEE adds
an iteration loop with a visibility ray per step inside the NEE block, on
the renderer whose cost per vertex is the thing being defended.

**Why absent.** A caustic is an `L (S)+ D` connection: light reaching a
diffuse vertex *through* specular bending. Unidirectional NEE cannot make that
connection — the shadow ray is a straight line, and ours deliberately passes
through glass tinted-but-unbent (the transparent-shadow convention, documented
in the limitations page). BSDF sampling finds the path only by luck: with
probability exactly zero for delta lights (a zero-area emitter cannot be hit
by a sampled ray — that is *why* delta lights are NEE-only), and with tiny
probability for emissive triangles and env maps — which arrives as fireflies
that `pt_firefly_clamp` then suppresses, since it clamps exactly the
rare-but-bright indirect contributions. So today: no caustics from delta
lights at all, and biased-away caustics from area emitters.

**What it would take**, in ascending order of scope:

* **Manifold next-event estimation (MNEE)** — Hanika et al. 2015; what
  Blender ships as "shadow caustics". Per NEE connection whose straight path
  crosses tagged transmissive geometry, Newton-iterate the refraction point on
  the specular manifold until the bent chain connects vertex → glass → light.
  This is the minimal production-grade addition and the recommended target:
  it is *deterministic given the path* (no new randomness, so it costs the
  sampler nothing), it lands inside `pt_shade`'s existing NEE loop, and its
  contributions arrive as direct light — outside the firefly clamp, so they
  are not biased away. Needs: smooth normals + derivatives on the caster
  (`tri_norm` exists), a bounded iteration loop with a fail-closed fallback
  (contribute nothing, count it — the `truncation.py` pattern), one extra
  visibility ray per iteration, and a caster declaration (a mob-level flag in
  the `declare_shadow_flags` family; tag closed transmissive shells by
  default). Scope: shadow caustics through one refractive interface pair;
  reflective and multi-interface caustics stay out and the docs keep saying
  so.
* **Photon mapping / SPPM.** A light-tracing pass depositing photons in a
  deterministic spatial hash, gathered at diffuse vertices. Handles every
  caustic class, but it is a second transport pipeline: emitter sampling from
  the light side, photon storage in the arena, radius/bias control, and a new
  kernel family. The memory model and the batch-window structure both grow
  real complexity.
* **BDPT / VCM.** Correct and general, and the worst fit: path identity (the
  purity the sampler contract rests on) becomes genuinely hard to preserve
  across connection strategies, and the wavefront state layout would need a
  light-subpath mirror. Not proposed.

**Verification when it lands:** a point light over a glass sphere on a white
floor — brute-force reference by rendering the same scene with the light
replaced by a tiny bright emissive sphere at very high spp and no clamp;
MNEE at low spp must land on the reference's focused spot within tolerance.


## 2. Adaptive sampling

**Why absent.** The wave loop renders uniform spp: `for sample_base in
range(0, samples, wave_samples)` over every pixel of the tile, and
`finalize_samples` divides by one scalar count.

**What it would take.** The attractive property in *this* codebase: unlit 2-D
content is zero-variance by construction, so adaptive sampling would let text
and vector-graphics scenes converge at the floor sample count while only lit
3-D regions pay — likely the single biggest speed win available to the path
tracer on Algan's actual workload. That "likely" is what §0.1's text-scene
arm exists to settle: every sample of a text pixel re-peels the camera
segment deterministically (contract 4), so the win is the peel cost times
the samples above the floor, and the baseline measures both factors.

Two things to keep straight when it lands. With the denoiser on, the error
target should be modest — the denoiser is what absorbs the residual, so
driving pixels to convergence on their own is paying twice. And adaptive
sampling does nothing for the *lit* pixels' cost per sample; for the
many-lights scene that is §6's job, not this one's.

The longer-term alternative, if the baseline says the camera-segment peel
dominates even at the floor count, is hybrid primary visibility: resolve the
camera segment once through the sheet route (exact analytic coverage, zero
variance, no per-sample cost) and start paths at the first lit vertex. It
would also give better anti-aliasing than jitter. It is a large structural
change and it is not proposed here; it is recorded so that the option is
weighed against adaptive sampling with numbers rather than rediscovered.

Sketch, staying inside the contract:

* **Error estimate:** split `accum` into even/odd half-sums (one extra
  [F, W·H, 4] tensor; the layout stays atomic-free and fixed-order). After a
  floor pass of `pt_min_samples`, the per-pixel relative half-buffer
  difference is a standard relMSE proxy — computed from deterministic sums,
  so the *decision* is deterministic too.
* **Pixel compaction:** further waves run on an active-pixel index list
  instead of the whole tile. The machinery exists in miniature: `rs_pix`
  already maps slot → pixel, and `_ArenaRayCompactor` already compacts rays;
  a per-wave pixel list is the same pattern one level up. Slot layout stays
  `r = k * active_pixels + p`, so `pt_reduce` gains an indirection through
  the list, nothing else.
* **Accounting:** a per-pixel sample-count tensor (or reusing a spare `accum`
  column); `finalize_samples` divides per pixel instead of by the scalar.
  The AOV reduction divides by the same counts.
* **Sampler purity:** untouched — each pixel's sample indices remain
  contiguous `0..n_p`, and `n_p` is itself a deterministic function of the
  rendered data.
* **Settings:** `samples_per_pixel` becomes the ceiling; experimental
  `pt_min_samples` and `pt_error_target` gate the loop (target 0 = today's
  uniform behaviour, the byte-parity escape hatch every PT feature has
  shipped with).

**Verification:** an all-unlit 2-D scene terminating at the floor count
(assert via the plan's sample tally); equal-error-vs-equal-time versus
uniform sampling on the `lit_and_shadowed` suite scene; the 2-D composite
still exact (contract 4).

### 2.1 LANDED 2026-09-04

Shipped as sketched, with one addition the measurements forced and one
correction. The settings are `pt_min_samples` (`ALGAN_PT_MIN_SAMPLES`, 4) and
`pt_error_target` (`ALGAN_PT_ERROR_TARGET`, **0.02**); `samples_per_pixel` is
now a ceiling, `RenderPlan.path_samples_mean` reports what was actually
taken (0 = the path tracer did not run), and one PERF line per chunk names
the pixels at the floor, at the ceiling and the mean.

**Adaptive sampling stops only pixels that were deterministic by
construction. That is what makes it safe.** A pixel is eligible to stop
before the ceiling only if *none* of its samples took a random decision, and
only then does the error estimate get a vote. `pt_shade` sets a sticky flag
(`_PT_ACC_STOCH`, one new `pt_acc` column) the first time a path gambles: a
lit crossing (next-event estimation picks one emitter out of the table), an
authored crossing or a custom scatter, or a lobe pick where
`w_sum - w_pass > 1e-9` (more than the pass-through branch was available).
An unlit stack (pass-through at probability 1), an unlit opaque absorb, and
an escape to the background or the environment map set nothing. `pt_reduce`
sums the flagged samples per pixel into a fourth column of the half-sum
buffer, and the host requires that count to be zero.

The reason is a correctness defect, not an optimisation. A half-buffer
difference cannot see this estimator's one failure mode: **a pixel whose
first samples all return zero has two halves that agree exactly**, and no
choice of target or eps distinguishes "converged at zero" from "has not
found the light yet". It was not hypothetical -- on
`tests/path_traced/scenes/lit_and_shadowed.py`, whose next-event table is
dominated by an emissive slab most surface points cannot see, a purely
statistical rule left **249 lit pixels of 9216 stuck at pure black** (255
counts, mean frame difference 4.91). A one-pixel dilation of the unconverged
set cut that to 31 pixels but could not remove it: the failure is structural,
and on the renderer that exists precisely because it must not refuse or
corrupt a scene, "rarer" is not an answer. The kernel already knows which
paths gambled, so it says so. With the gate the same scene renders
**byte-identical** to its uniform arm at mean 10.30 samples of 48.

The dilation is kept, now for the reason it is actually good at: a 2-D edge
pixel is deterministic given its jitter, so four jittered samples that happen
to agree would otherwise freeze a coverage value its neighbours are visibly
still resolving. It is grown from the unconverged pixels each round, never
from the rescued ones, so the ring stays one pixel wide, and it never revives
a pixel that has already stopped -- which is what keeps every live pixel's
sample count equal.

**What it is, mechanically.** `pt_reduce` sums the odd sample indices' RGB
into a second `[F, W·H, 4]` buffer (columns 0-2; column 3 is the stochastic
count), so with `n` samples so far the two half means are
`E = (accum − odd)/(n/2)` and `O = odd/(n/2)`, and

    err = max_c |E − O| / (max_c (E + O) + 0.02)

Every wave then runs over an explicit pixel LIST rather than a contiguous
tile span (`pt_generate` gains `pix_list`, `rs_pix` stores the global cell
and `ray_offset` is 0 for traverse and shade), and `pt_reduce` writes
through the same list. Slot layout is unchanged — `r = k · active + p` —
so `s_index = sample_base + r // tile_pixels` still enumerates each pixel's
contiguous Sobol prefix, because every pixel alive in a wave has received
the same count. Before `finalize_samples` the per-pixel sums are rescaled by
`samples / n_p` (and so are the denoiser's AOV sums), so the caller's single
scalar division is untouched. At `pt_error_target = 0` none of it runs and
no half-sum buffer is allocated, so that arm's memory model, batching and
output are byte-identical (`tests/path_traced` is 4 passed under
`ALGAN_PT_ERROR_TARGET=0`).

**The correction: the floor cannot be one wave.** At any resolution where the
budget fits a whole frame in one tile, `_pt_tile_shape` puts every sample in
flight at once — so the first wave would finish the render before a pixel
could be retired. The wave size is capped at the floor first and then at
most DOUBLES per wave, which bounds a pixel's overshoot past its true
stopping point at 2x and costs about 2x the launches.

**Measured, this CPU box** (4 vCPU x64, Quadrants 1.3.0, `pt_baseline.py`,
320x180, five frames, 16 spp, 4 bounces, denoiser off, warm RUN 2, median of
three):

| scene | target | mean spp | warm wall | frame diff vs target 0 |
| --- | --- | --- | --- | --- |
| `text_2d` | 0 | 16.00 | 0.834 s | — |
| `text_2d` | 0.02 | 5.54 | 0.622 s (**1.34x**) | max 119, mean 0.311, 883 px > 8 |
| `lit` | 0 | 16.00 | 1.511 s | — |
| `lit` | 0.02 | 5.94 | 1.535 s (neutral) | **max 0 — byte-identical** |

and across targets (288000 pixels, five frames):

| scene | target | max | mean | px > 8 |
| --- | --- | --- | --- | --- |
| `lit` | 0.01 / 0.02 / 0.05 | 0 | 0.000 | 0 |
| `text_2d` | 0.01 / 0.02 / 0.05 | 119 | 0.311 / 0.311 / 0.318 | 883 / 883 / 926 |

Four readings to carry forward.

* **Nothing that gambled was stopped, so the lit scene is exact.** `lit`
  takes 2.7x fewer samples and renders byte-identically: every sample it
  saved was a background pixel. That is also why the wall clock is neutral
  there — the samples adaptive sampling can safely drop on a lit scene are
  the cheapest ones (one traversal and out), which is §2's own prediction
  ("adaptive sampling does nothing for the *lit* pixels' cost per sample")
  measured.
* **The remaining cost is 2-D anti-aliasing, and only at edges.** All 883
  `text_2d` pixels past 8 counts sit on a geometry edge; no interior pixel
  moves (contract 4 holds exactly). `pt_min_samples` buys them back: a floor
  of 8 leaves 200 such pixels (mean 0.198, max 99) at mean 9.15 spp instead
  of 5.54 — half the win for a quarter of the residual.
* **The target is not the throughput knob.** Mean spp is flat from 0.005 to
  0.05: a pixel is either exactly converged or not eligible at all. 0.02 is
  picked from the tolerance analysis in `raytracing/settings.py` — it accepts
  a half-buffer difference of about 1.3–2.2 counts of 255 across four decades
  of linear radiance, which is why the metric needs no perceptual transform
  (a *relative* metric is invariant under any power law, so sqrt or a PU
  curve would only rescale it).
* **`tests/path_traced` at the shipped default is 1 failed, 3 passed.**
  `lit_and_shadowed` and `environment_and_refraction` now match their
  baselines within tolerance; `translucency_and_order` differs by up to 28
  counts on 65 pixels of 46080 (mean 0.011), every one on a translucent
  square's edge, because that scene renders at 8 spp so its edges get the
  floor of 4 where the baseline got 8. The baselines were deliberately not
  regenerated; the arm that must stay green is `ALGAN_PT_ERROR_TARGET=0`.

**What the T4 will measure**, and what this box cannot: §0.1's 2-D arm spent
0.30 s of kernel time per five 720p frames re-peeling zero-variance content
16 times, and this box's 320x180 render is too small for per-launch cost to
be read (`agent_guidance/gpu_harnesses.md`). The two numbers to take there
are the 720p `text_2d` wall against the target-0 arm — where the peel is
large and the ~2x extra launches are amortised — and whether the doubling
schedule's extra waves cost anything on `lit` at 720p, where they were
neutral here.


## 3. Temporal stability

**Why absent.** Frames are sampled and denoised independently, and the
sampler decorrelates frames by design (`frame` is hashed into every pair), so
residual noise re-rolls per frame — shimmer on lit 3-D content at low spp.
(2-D content is zero-variance and already rock-stable.)

**What it would take**, two tiers:

* **Tier 1 — correlated seeding: LANDED, as the default.** The frame is
  dropped from the pair key unless `pt_animated_seed` is set, so every frame
  reuses one sample set: static regions become perfectly stable (identical
  estimates), moving regions re-randomize through the geometry itself. One
  line in `_pt_key`, no new buffers, sampler purity untouched (the key is
  now `(pt_seed, frame-or-0, pixel, pair, index)`). An earlier draft made
  this an opt-in switch on the grounds that correlated error is "a choice";
  §0.3 says why it is the default instead: it is the standard configuration
  for animation, it is what Cycles ships, and it is the cheapest quality-per
  -spp win the renderer has. `pt_animated_seed = True` restores per-frame
  decorrelation for the cases that want it (a still-frame Monte Carlo
  average across frames, or a user who prefers shimmer to texture).
* **Tier 2 — motion-vector temporal filtering (SVGF-family).** Needs a
  velocity AOV: at the same first-non-delta vertex the albedo/normal guides
  use, record `tri_obj` + barycentrics (the `pt_aov` row has width to grow),
  then on the host re-project that material point through frame `f−1`'s
  geometry — the merged scene already carries per-frame `tri_pos`, so the
  correspondence is a gather, not a search. With velocity plus the existing
  albedo/normal guides, an EMA history buffer with disocclusion rejection is
  the literature-standard filter (OIDN has no public temporal model, so this
  is hand-rolled). The structural cost is that temporal state crosses render
  *chunks*: the render loop would carry a per-pixel history tensor across
  batch windows, and the OOM chunk-retry must roll it back — precedent exists
  (`truncation.py`'s snapshot/restore does exactly this for its counters).

**Verification:** tier 1 — adjacent frames of a static scene agreeing in
static regions (that is the *point* of the mode, and is a claim about two
frames of one render, not about two runs); tier 2 — temporal variance of a
fixed
camera path measurably below the per-frame arm at equal spp, with no ghosting
on the `translucency_and_order` suite scene (2-D content must pass through
the temporal filter unchanged).


## 4. Volumes and subsurface scattering

**Not fallback work**, for the same reason as §1: the deterministic renderer
has no volumes either, so nothing here is a scene the fallback refuses. It
is the largest new capability on the list and sits behind everything the
fallback role needs. Kept because the scaffolding decisions below (the
reserved sampler pairs, the media stack) constrain the work that *is*
scheduled.

**Why absent.** Not started; largest scope. What *does* exist is the
scaffolding: the refraction path carries a nested-media stack in `rs_sca`
(entry/exit tracking per closed shell), Beer–Lambert absorption is already
applied over interior chords for transmissive solids on both view and shadow
rays, `closed_shell` declarations identify watertight interiors, and sampler
pairs `2 + 6b + 4, 5` are reserved for exactly this.

**What it would take**, in landing order:

* **Homogeneous scattering media (v1).** Per-material `sigma_s` + phase
  anisotropy `g` (Henyey–Greenstein) alongside the existing
  `attenuation_color/_distance` (which already define `sigma_a`). In
  `pt_shade`: after traverse returns the next surface hit at `t_hit`, sample
  a medium-event distance `t_med ~ Exp(sigma_t)` from a reserved pair; if
  `t_med < t_hit`, the crossing becomes a *medium vertex* — HG-sample a new
  direction, run the NEE block from the interior point with transmittance
  along the shadow ray (analytic for homogeneous media — no ratio tracking
  needed), and continue. The wavefront loop barely changes shape: a medium
  event is "a scatter that consumed no hit", and the current media stack
  says which medium the segment is inside. MIS bookkeeping: phase-function
  pdf slots into the existing `_SCA_PREV_PDF` convention unchanged.
* **Heterogeneous media** (density fields, ratio tracking / delta tracking)
  are explicitly v2: they need a field representation the scene format does
  not have, and null-collision loops whose iteration counts are
  data-dependent (reproducible, but a real occupancy cost).
* **Subsurface scattering.** Once homogeneous media exist, random-walk SSS is
  the same machinery scoped to one shell's interior: high `sigma_s`, walk
  until the path re-crosses the *same* `tri_obj` surface (the identity the
  closed-shell ring already reads). That is the physically-faithful version
  and the one to land first; a Burley normalized-diffusion profile (sample a
  disk, probe-ray back onto the surface) is a later optimization for
  thick-media cost, not a prerequisite.
* **Denoiser interplay:** none required — the OIDN RT weights handle
  volumetric noise; medium vertices should write scatter albedo and a zero
  normal into the existing AOV guides.

**Verification:** a homogeneous slab against the closed-form
single-scatter + attenuation solution (the codebase's torch-quadrature
reference-test pattern from Stage 3); a dense-medium cube converging to its
diffusion limit.


## 5. One material, two direct-lighting responses — LANDED

**LANDED 2026-09-05**, as the section argued and with one addition the
completeness rule (§9) forced. `_pt_direct_response` is deleted; every
emitter kind — packed light rows included — now evaluates the surface through
`_pt_lit_f_pdf`, and the NEE estimator for a row is `f_cos * radiance /
p_sel`. `_light_eval` is untouched: decay, range fade, spot cone, the
one-sided area cosine and the `1/K` power fraction are the EMITTER model, and
only the SURFACE response changed. The direction-less ambient / hemisphere
rows contribute `e_diff * L` (the diffuse lobe's energy times the row's
radiance, with the hemisphere row's sky/ground blend still done by
`_light_eval` against the shading normal) and no specular fill — indirect
transport is what replaces that.

What moved, concretely: a Lambert surface under a light row is `pi` times
dimmer than it was, since `_pt_direct_response` had no `1/pi`; a rough metal
changes by the difference between `_smith_geometry`'s `k = (r+1)^2/8` remap
and the exact Smith `G2`, which is small at high roughness and grows as
roughness falls. The deterministic renderer's stage formulas and every one of
its baselines are untouched, exactly as this section predicted.

**Phong is GGX now, and that is why it still has a highlight.** Deleting the
Blinn-Phong term without giving `MeshPhongMaterial` a lobe would have made it
render identically to `MeshLambertMaterial` under the path tracer — a
silently dropped feature, which §9 says is worse than a refused one. So
`_pt_lit_lobes` gained a phong branch: `F0` is the authored `specular`
colour, the exponent converts by the standard `alpha = sqrt(2/(s + 2))`, and
the resulting roughness replaces the crossing's own for that material's NEE
responses *and* its continuation (`_pt_lit_lobes` returns it). The highlight
therefore moves and softens rather than disappearing, which
`renderer_limitations.rst` now says out loud.

Verification, in `tests/unit_tests/test_path_tracer.py`:
`test_area_light_row_and_emissive_quad_agree` is the acceptance test — a
Lambert and a GGX floor under a `RectAreaLight` and under an emissive quad of
matched radiance, agreeing within noise at 96 spp (the bound it holds them
to is 6% of the brighter arm);
`test_nee_light_row_direct_lighting_is_the_physical_bsdf` and
`test_area_light_matches_the_reference_integral` pin the row estimator
against closed-form and quadrature references instead of against the
deterministic renderer (the parity assertions they replaced are precisely
what this section deleted); and the furnace check gained a light-row arm,
`test_lambert_furnace_is_lossless_under_a_light_row`.

The rest of this section is the reasoning, kept.

**What was inconsistent.** A lit vertex answered "how much light comes back
toward the camera" with two different functions, chosen by *what kind of
emitter is asking*:

* **Packed light rows** (point / spot / directional / area cells) go through
  `_pt_direct_response`, which is the matching `shading_taichi` stage, term
  for term: `_smith_geometry`'s direct-lighting `k = (r+1)^2/8` remap, no
  `1/pi` on the diffuse term, phong's Blinn-Phong highlight.
* **Emissive triangles and the environment map** go through `_pt_lit_f_pdf`,
  which is the physical BSDF the continuation actually samples: `albedo/pi`
  diffuse, exact Smith `Lambda` for `G2`, Fresnel and Turquin compensation on
  the same `alpha = roughness^2` the VNDF sampler uses.

So a `RectAreaLight` and an emissive quad of the same radiance, in the same
place, do not light the same surface identically — and the discrepancy is
largest exactly where it is most visible, on smooth metals, because the two
`G` terms diverge as roughness falls.

**Why it is this way.** Brightness parity with the deterministic renderer was
taken as a product requirement: `spp == 1` is the default, every example in
the docs renders through it, and a user raising `samples_per_pixel` should not
watch their lighting change key. Light rows are the only emitters the
deterministic renderer has, so they were the only ones with a parity
obligation; emissive triangles and env maps had no counterpart to match, which
is why they were free to use the physical BSDF — and they had to, because MIS
is only correct when both ends evaluate the same function.

**Why that reasoning does not survive the renderer's stated purpose.** The
path tracer is the fallback for scenes the deterministic renderer cannot
render (see the top of this document). In those scenes there is no `spp == 1`
render to change key *from* — the comparison the parity requirement protects
does not exist. And where a scene *can* be rendered both ways, the two already
differ visibly: no ambient fill, no glossy prefilter, jittered instead of
analytic AA, real GI instead of none. Parity was already partial, and partial
parity is worth less than being right.

So the trade inverts. What the split costs is concrete: an emissive quad and a
`RectAreaLight` of matched radiance light the same surface differently; the
furnace tests cover only the transport half, so a future edit to
`_pt_direct_response` can break reciprocity silently; and it is a live trap
for anything that makes light rows MIS-able (making area lights hittable
geometry — which is also how an area light would gain its mirror image — needs
both ends to agree or the weights stop summing to one). What it buys is a
brightness match that no longer has a use case.

**What resolving it now takes — much less than it used to.** The previous
draft of this section proposed fixing the *deterministic* renderer's stage
formulas first (exact Smith `G2`, `1/pi` on diffuse, authored intensities
rescaled to compensate) so both renderers could converge on one BSDF. That was
a re-baseline of every committed frame in the repo on both devices, and it was
only necessary because parity had to be preserved through the change. It does
not: **delete `_pt_direct_response` and call `_pt_lit_f_pdf` for light rows
too.** The deterministic renderer keeps its stage formulas and its baselines
untouched; only `tests/path_traced/` re-baselines.

Two things to get right in that change. The NEE estimator for a light row
becomes `f_cos * radiance / p_sel` in place of the stage response, so the
delta-light radiometry from `_light_eval` (decay, range fade, spot cone,
one-sided area cosine) stays exactly as it is — it is the *emitter* model, and
only the *surface* response changes. And phong loses its Blinn-Phong highlight
in favour of GGX, which is a visible change to `MeshPhongMaterial` under the
path tracer and should be called out in the limitations page rather than
discovered.

**Verification when it lands:** a Lambert and a GGX surface lit by a
`RectAreaLight` and by an emissive quad of matched radiance agree to within
noise; the furnace tests extended to the NEE path.

**Land it with the other re-baselines, as one — DONE for four of the five.**
This change, §7's stratified lobe select, the sampler hoists in §0.2 and the
self-intersection offset in the final section each move every sample and each
say "re-baselines `tests/path_traced/`". They landed together, so
`tests/path_traced/` takes ONE re-baseline for the four of them; every one of
its three scenes moves, and the frames were compared side by side against the
committed baselines before it was taken. §6a-ter's area-light change (a
`RectAreaLight` as two emissive triangles) landed after this batch and needed
**no** re-baseline of its own: none of the three suite scenes carries a
`RectAreaLight`, and a render without one takes not one of its branches.
Deleting `_pt_direct_response` shrinks the shade kernel, which is a small
throughput win in its own right — the "second `_light_eval` call per ambient
row" this paragraph used to promise had already gone with §0.2's packed
ambient rows, so there was nothing left to remove there.


## 6. Many-light and emissive-mesh sampling

**6a, 6a-bis, 6a-ter, 6a-quater and 6b are all LANDED** (see the sections
below for what shipped and what each measured). Nothing in `pt_shade` is
linear in the light count any more.

**What existed.** One flat power-weighted CDF over every sampled light row,
every emissive triangle and one environment entry, rebuilt per render call
(`_build_nee_tables`). Selection was global and purely power-proportional: no
spatial term, no orientation cone, no BSDF awareness. It survives as the
`pt_light_tree = False` arm, byte for byte.

**Why treeing the *lights* looked unnecessary — and why that was wrong.** The
redesign plan said a tree over a handful of light rows could not pay, and an
earlier draft of this section agreed, proposing instead that the kernel simply
**sum every light row deterministically** (exact, zero variance, cheaper than
any selection scheme at small `N`).

That proposal contradicts the renderer's purpose. "Too many lights" is one of
the three reasons the path tracer exists: the deterministic renderer's cost is
linear in light count and its shadows cap out at 16, so a hundred-light scene
is impossible there and the user is told to fall back to here. A fallback
whose direct lighting is *also* linear in light count does not solve that
problem — it reproduces it. Summing is the right answer only for the small-`N`
case that was never the reason anyone reached for this renderer.

It is also worth retiring the usual argument for small `N` on its own terms.
"Only four lights, so a tree cannot pay" is false: a spatially-aware sampler
beats power-proportional selection even at two lights, because
power-proportional ignores distance entirely and will pick a bright light
across the room as often as a dim one against the surface. The survey quotes
PBRT-v4 §12.6 measuring a **2.72x MSE improvement on a two-light scene** for
exactly this reason.

### 6a. The tree covers the whole table, not just emitters — LANDED

So the light tree is not an optimization for the emissive-mesh case with the
light rows left outside it. It is the selection structure for **every** entry
the NEE table holds — delta rows, area-light cells, emissive triangles — with
the environment entry alongside as it is now. That makes per-vertex direct
lighting `O(log E)` in the total emitter count, which is the property the
"too many lights" use case actually needs, and it is what promotes 6b from
"worth doing once emissive meshes are a supported look" to **required for a
stated primary use case**.

**What shipped.** `algan/rendering/raytracing/light_tree.py` builds it host
side and `_pt_lt_importance` / `_pt_lt_descend` / `_pt_lt_pmf` in
`path_tracer_taichi.py` use it; `pt_light_tree` (`ALGAN_PT_LIGHT_TREE`,
default on) is the switch and off restores the flat CDF byte for byte.
Point, spot and rect-area-cell rows and emissive triangles go in the tree.
**Directional rows and the environment entry stay out** as a small
power-weighted flat list, picked with a position-independent probability
`P_inf / (P_inf + P_tree)` — and because a member's share inside that list is
`power / P_inf`, an infinite entry's *effective* probability comes out at
`power / P_total`, exactly the flat CDF's number. That is why `_NM_ENV_SHARE`
and the escape MIS weight needed no change at all: only the split among the
finite entries moved.

**One thing the paper does not tell you, and it inverted the first
measurement.** The importance is `power * |cos theta'| / d^decay`, not
`/ d^2`: Algan's light rows default to `decay = 0` and genuinely do not fade
with distance. A hard-coded inverse square aims the sampler at the near
lights while every light contributes the same, and measured **1.34x worse**
MSE than the flat CDF on the 32-light ring. The exponent therefore rides the
node (`LT_DECAY`, the minimum over the subtree — conservative in the
direction that keeps every emitter reachable); an emissive triangle always
carries 2, because there it is the area-to-solid-angle Jacobian rather than
an authored choice. The consequence to state plainly: **on default
`decay = 0` lights the tree matches the flat CDF rather than beating it**
(measured 1.03x), because there is nothing about them for a spatial
structure to discriminate by. It wins where the emitter model actually
varies with position — physical falloff, spot cones, one-sided area cells and
emissive triangles.

**Measured** (CPU, 32 point lights on a ring with `decay = 2` over a Lambert
floor, `pt_light_samples = 1`, 4 spp against a 128-spp reference rendered
with the tree off): mean squared error **12.5 with the tree against 109.2
without — 8.7x**, which is what `test_light_tree_cuts_many_light_variance`
asserts (at a 3x threshold). PBRT-v4 quotes 2.72x on *two* lights; more
lights is more room, as expected.

**Cost.** The build is host-side numpy, about **0.12 ms per node** and
`2E - 1` nodes: 0.2 ms at 2 entries, 7.5 ms at 32, 60 ms at 256, 236 ms at
1024. It is per *frame* because `light_pos` and `tri_pos` are per-frame
tensors — but frames whose emitter geometry is byte-identical share a row, so
a static light rig under moving geometry collapses to one tree per render
chunk however long the chunk is, and above `PER_FRAME_BUILD_BUDGET`
(`distinct frames x entries`) the build falls back to a single tree over the
union of every frame's bounds and cones: looser, still unbiased, and bounded
at roughly a quarter second per chunk. In the kernel,
`benchmarks/performance/pt_baseline.py --scene many_lights --resolution
320x180` (CPU, warm run 2, 64 lights) puts `pt_shade` device time at
**492.6 ms flat against 529.6 ms with the tree, +7.5%** — one importance pair
per level of a ~6-level descent, against a 6-step binary search, against a
kernel that is also tracing shadow rays and shading solids. On a bare
32-light `decay = 2` ring at the same resolution, where next-event estimation
is nearly all of what `pt_shade` does, the same descent measured +34%; take
that as the ceiling and this as the shape of a real frame. Against an 8.7x
variance win either is a large net gain in equal-error time.

Half of that overhead was inverse trigonometry, and it is gone: the node
packs `cos theta_o`, `sin theta_o` and `cos theta_e` rather than the angles,
and the importance evaluates `cos(theta - theta_o - theta_u)` through the
angle-subtraction identities (PBRT-v4's `CosSubClamped`). Measured on the
32-light ring, the straightforward `acos`/`asin`/`cos` version of the same
formula cost 844 ms of `pt_shade` device time where this one costs 674 ms,
over a flat-CDF baseline of 504 ms.

Two deliberate approximations in the build, both conservative and both
invisible to correctness (any positive importance is unbiased so long as both
MIS ends read the same tree): a node's cone is bounded about the normalized
*mean* of its members' axes rather than by the paper's incremental pairwise
union, and the split search scores a prefix's cone from the same bound on
running sums. Both exist because they vectorize and the union does not — the
build is dominated by host-side call count, not by arithmetic.

A node whose two children both score zero at a shading point — possible,
because the parent's union box and union cone can face the point when
neither child does — **splits the draw evenly** rather than abandoning the
sample. That costs a shadow ray that returns almost nothing and buys the
property the MIS weights rest on: the descent is a genuine distribution over
the leaves with no mass lost part-way down, which
`test_light_tree_selection_probabilities_sum_to_one` pins.

Still **not** done, and deliberately, the **sum-everything fast
path** for tiny tables:

A **sum-everything fast path** for tiny tables (a threshold on entry count,
in the low tens) is exact and zero-variance, but it costs `E` shadow rays per
vertex where the tree costs one, and with a denoiser downstream one
well-chosen sample is likely the faster route to equal perceived quality.
Do not build it first; build it if the §0.1 baseline on a two-light scene
asks for it.

### 6a-ter. A `RectAreaLight` is one emissive quad, not `K` rows — LANDED

**What shipped.** `algan/rendering/raytracing/area_light_quads.py` builds two
emissive triangles per `RectAreaLight` — centre from the light's own packed
sample rows, axes from its facing normal, radiance `colour * intensity / area`
(the matching §5's acceptance test does by hand) — and
`tracer._attach_area_light_quads` appends them to a **private copy** of the
merged scene, rebuilding the triangle BVH over the widened primitive set and
re-homing the widened tables into the arena. Private because the merge is the
persistent device scene the deterministic renderer may render from next: it
never sees these triangles at all, which is also why the camera-invisibility
test is not a leaf bit. The geometry is per frame, indexed like `light_pos`, so
a light that moves takes its quad with it. The `K` cell rows stay in
`light_col` (the authored-appearance branch still lights from them) but stop
being selectable in `_build_nee_tables`, so nothing is counted twice.

`pt_area_light_quads` (`ALGAN_PT_AREA_LIGHT_QUADS`, default on) is the switch,
host-side with no kernel variant; off is the packed-rows arm, byte for byte.

**What it costs on the host.** The widened copy rebuilds the *whole*
triangle tree (the merge does not keep its build inputs, and the traversal
takes one triangle tree per batch), so it is built **once per batch** and
cached on the batch-lived merge under `_pt_quad_widened`, with its arena
range retained through the per-window rewind the way the raster tables are
— `render_batch_raytraced` runs once per render *window*, which at 720p and
16 spp is one frame, and the first version rebuilt per window: measured
120 ms per frame on 3,200 triangles on the CPU box, i.e. a fifth of that
frame's wall. Per batch it is one extra split-BVH build, on the order of
the merge's own, plus a second copy of the triangle tables in the arena's
persistent end that the memory model does not plan for (a batch that only
just fit will retry smaller). The right end state is the quads entering
the merge itself with a camera-invisible leaf bit the deterministic
traversal tests and never sets; that touches the deterministic kernels and
is not done here.

**The falloff multiplier.** A row's emitter model is `d^-decay` times a range
fade and `RectAreaLight` defaults to `decay = 0` — no falloff at all — while a
physical emissive quad has inverse square built into transport. The difference
is a per-emitter radiance multiplier `d^(2 - decay) * fade(d)^2` applied
**at the emitter**, so both MIS ends evaluate it from the same distance (the
next-event end knows `ldist`, the BSDF-hit end knows `t_hit`) and the
power-heuristic weights still sum to one. Its two numbers per quad ride
`pt_emit_falloff`, one new arena entry on `pt_shade` rather than a new kernel
argument, indexed by `prim - quad_base` with `quad_base` on `nee_meta`
(`_NM_QUAD_BASE`). An ordinary emissive triangle is `prim < quad_base` and
takes neither branch, so emissive meshes are bit-identical. The light tree's
importance exponent reads the NET falloff for a quad rather than the
triangle's usual 2, because on a `decay = 0` light a hard-coded inverse square
aims the sampler at the near emitter for nothing (§6a measured that at 1.34x
worse).

**Camera-invisible, and not an occluder.** The one compare `prim >= quad_base`
in `pt_shade`'s drain loop, gated on `bounces_left >= max_b` — the camera
segment, the same reading the closed-shell ring takes — passes a primary ray
straight through while a ray that has bounced sees the light, which is what
puts it in a mirror. The quads are packed **non-opaque** so the k-buffer's
prune-behind-an-opaque-hit and `pt_opaque_closest` cannot hide the geometry
behind one while it is being skipped (that batch turns `all_visible_opaque`
off), and **non-casting** in the rebuilt tree — the `casts_shadows` leaf bit —
so a shadow ray walks through, matching the deterministic renderer where an
area light is not an occluder.

**Tests**, all in `tests/unit_tests/test_path_tracer.py` and none marked
`fast`: `test_area_light_quad_and_row_arms_agree` (Lambert and GGX floors, the
two arms within 6% at 96 spp),
`test_area_light_quad_falloff_follows_the_row_model` (`decay` 0 / 1 / 2 and a
non-zero `distance`, each arm against the
other), `test_area_light_quad_is_invisible_to_the_camera` (the light's own
pixels are pure background, with an emissive mob of the same size as the
control that proves the framing), `test_area_light_quad_shows_up_in_a_mirror`
(a smooth metal sphere: 16 pixels over 100/255 with quads, 0 without),
`test_area_light_quad_occludes_nothing` (a point light through a zero-intensity
panel), `test_area_light_quad_is_mis_covered_by_both_strategies`
(`max_bounces = 0`, where next-event carries the whole emitter at weight 1,
against `max_bounces = 3`, where the two strategies split it),
`test_area_light_quad_follows_a_moving_light` and
`test_area_light_quad_collapses_the_next_event_table` (a `samples = 16` light:
16 selectable entries becomes 2). The two pre-existing area-light tests —
`test_area_light_matches_the_reference_integral`, which pins the `decay = 0`
radiometry against a torch quadrature of the continuous area integral, and
§5's `test_area_light_row_and_emissive_quad_agree` — now run **through** the
quad path and still pass, which is the strongest statement available that the
new estimator is the same estimator.

**Variance**, `benchmarks/_pt_area_light_quad_variance.py` (CPU, 64x64, one
`samples = 16` area light over a Lambert floor with a smooth metal sphere,
adaptive sampling off, MSE against a 1024-spp reference, 4 seeds per arm):
**320.3 for the rows arm against 153.2 for the quads — 2.09x better at equal
spp**. The two 1024-spp references differ by 0.807 counts of 255 mean
absolute, which is the bias bound: they are two estimators of one emitter,
they agree to well under a channel count, and the 2.09x is therefore variance.
Most of it is the metal sphere, where a BSDF ray finds the emitter and a
next-event sample aimed at a near-delta lobe almost never does — the strategy
the rows arm does not have.

`tests/path_traced/` did not move at all — none of its three scenes carries a
`RectAreaLight` — so 6a-ter cost no re-baseline, which is why it is not in
§5's batch after all.

**On the T4** (`benchmarks/performance/reports/t4_2026_09/pt_arealight_1.md`,
the `lit` solids under four 16-sample area lights, 64 rows against 8
triangles): the quads arm is **5% faster end to end at 720p and 2% at
1080p**, with device time up 8% — the traverse half of that is the batch
losing `pt_opaque_closest` and the any-hit shadow query because the quads
are packed non-opaque, the shade half is the two-strategy MIS at emitter
hits — and host time down 165–190 ms from the next-event setup over 8
entries instead of 64. Variance at equal spp is 1.83x lower on the T4
(2.09x on the CPU box). The traverse cost is the argument for the leaf-bit
end state above.

**Known, and deliberately left — and §6a-bis did not close it.** An
authored-appearance material (manim, toon, matcap, a custom fragment pipeline)
lights from the packed rows, because that is the model those materials have,
while the quad is additionally geometry its continuation can find: such a
surface sees an area light slightly twice. §6a-bis changed *which* rows that
first term sums over — it may now sample them rather than sum them all — and
that is the whole of the change: the direct term still comes from the rows and
the continuation is still an ordinary Lambert bounce that can hit the quad, so
the double count is identical in both of its arms — the same surface, the same
two contributions, only the first one estimated rather than summed. (It exists
only where the quads do: with `pt_area_light_quads` off there is no geometry
for a continuation to find.) Closing it needs the continuation to know it left
an authored surface — path state, not an estimator — or the quads to carry an
"invisible to a path that came off an authored crossing" rule, and neither is
built. Physically-integrated materials — everything that goes through the
next-event table — are unaffected: for them the rows are withdrawn and only
the quad remains.

What §6a-bis *did* have to get right here is the other direction: its estimator
draws from a table of the LIGHT ROWS, not from the next-event entries, so a
`RectAreaLight` whose cell rows this section withdrew still reaches an authored
surface. Drawing from the next-event entries would have made an authored floor
under an area light go black.
`test_authored_sampling_lights_an_area_light_the_same` is the guard.

The rest of this section is the original plan, kept.

`RectAreaLight` expands to `K = k*k` cell rows carrying `1/K` of the power
each. That packing exists for the deterministic renderer's shadow fans; the
path tracer inherited it, and it costs the path tracer three things: `K`
table entries per light (one 4x4 area light is already 16 entries, so
"tiny" is reached long before the light count suggests), a per-cell jitter
special case in `_pt_light_sample_point`, and the two gaps the top of this
document admits — no mirror image, and a highlight from the stage formula
rather than transport.

The fix is to give the path tracer its own view of the light: **two emissive
triangles**, flagged invisible to camera rays. They then ride the
emissive-triangle path that already exists end to end — area sampling from
the table, `_pt_lit_f_pdf` on both ends, power-heuristic MIS, and a BSDF
ray that can find them, which is what puts the light in a mirror. The
camera-invisible flag is the only new piece: a bit on the triangle, in the
family of the `casts_shadows` leaf bit, tested where a camera-segment ray
would accept the hit. The deterministic renderer keeps its rows untouched.
This is a §5 re-baseline item (it moves every sample that touched an area
light) and belongs in that batch.

### 6a-quater. Build the tree per frame — LANDED

`light_pos` and `light_col` are per-frame tensors, and `_build_nee_tables`
runs once per render *chunk*. A tree whose bounds are the union over a
chunk's frames is unbiased (the pdf is whatever both MIS ends agree it is)
but its importance heuristic degrades for anything that moves — and Algan is
an animation engine, so lights move. `E` is small and the chunk's frame count
is small, so build one tree per frame, indexed by `f` the way `light_pos`
already is, rather than one union tree. The per-frame emission power the
"frame-animated emitters" gap in the final section describes falls out of the
same change.

**What shipped.** `build_light_trees` builds one tree per distinct frame of
the chunk and `lt_frame[f - time_start]` picks the row, so a light that moves
is followed rather than bounded by a chunk-wide union. Frames whose emitter
geometry is byte-identical share a row (a static rig under moving geometry
collapses to one build per chunk), and a chunk whose distinct-frame count
times entry count exceeds `PER_FRAME_BUILD_BUDGET` falls back to exactly the
union tree this section argues against — looser, still unbiased, and the
thing that keeps a host-side build from costing seconds on a long chunk of
genuinely moving emitters. `test_light_tree_follows_a_light_that_moves_
between_frames` is the guard. The per-frame *power* is still frame-0's: an
entry's weight is the number the flat table gives it, so which emitters are
sampleable at all did not change, and the "frame-animated emitters" gap in
the final section stays open.

### 6a-bis. Two loops that were linear in light count — LANDED

Independent of the tree, `pt_shade` used to walk every light row twice, and
under the purpose stated at the top of this document these were defects rather
than inefficiencies:

* **The ambient / hemisphere fill — FIXED (§0.2).** It used to scan all
  `num_lights` rows at every lit crossing to find the zero-to-two
  direction-less rows (`for li in range(num_lights)`, testing `lt_row` per
  row): in a 200-light scene, a 200-iteration scan per crossing per bounce
  to find two entries. The host now appends those rows to `nee_ref` after
  the sampled entries and the kernel loops their count from `nee_meta`.
* **The authored-appearance branch — FIXED (this section).** It looped all
  lights and traced a shadow ray for each up to `max_shadow_lights`, filling
  the `vis` vector `_run_frag_pipeline` expects. That is the deterministic
  renderer's cost model *and* its 16-light cap, running inside the fallback:
  such a surface in a hundred-light scene was lit by all hundred but shadowed
  by the first 16, and paid 16 shadow rays per crossing. (Not "lit by 16" —
  `_run_frag_pipeline` was handed the full `num_lights` and `_light_vis`
  returns fully-lit for any row past the payload, so the surplus lights lost
  their SHADOW, not their light.) These materials are opt-in rather than
  Algan's default (a shader-less mob is unlit, and the physically-integrated
  materials go through the NEE table), so it was a hole rather than the common
  path — but a hole in exactly the use case the renderer is advertised for,
  and the feature matrix did not mention that the cap was lifted only for some
  materials.

  It was harder than the first because `_run_frag_pipeline`'s interface is a
  per-light visibility vector, so it was an interface decision and not just a
  loop rewrite.

**What shipped.** The branch now fills the direction-less rows as the lit
branch does and **draws `pt_light_samples` of the remaining rows**, scaling
each drawn row's radiance by `1 / (S * p)`. `pt_authored_light_sampling`
(`ALGAN_PT_AUTHORED_LIGHT_SAMPLING`) is the switch, host-side, with **three**
states: `"off"` is the summing arm byte for byte, `"auto"` (the default) sums
inside `max_shadow_lights` and samples past it, `"always"` samples at any light
count. Three rather than two because of the bias below.

**The interface decision, which is the substance of this section: the weight
rides the light's RADIANCE, not its visibility.** `_run_frag_pipeline` and the
16-argument stage signature are untouched, and so is `shading_taichi.py`.
`pt_shade` passes `light_pos` and `light_col` through `_SampledLightView`, a
read-only view in `ArenaView`'s idiom (a tuple subclass, so `ti.static` passes
it through and it binds to a name in kernel scope) that rewrites
`view[tl, slot, c]` into `inner[tl, rows[slot], c]`, multiplied by
`scale[slot]` for `c < 3`. `rows` and `scale` are per-thread `ti.Vector`
locals, indexed from Python scope through the compiler's own subscript builder
exactly as `ArenaView` indexes the arena. Every built-in light-dependent stage
carries `lc` linearly in *both* its reflection and its `wsum` energy budget, so
`sum over slots g(r(s)) w_s vis_s` is unbiased for `sum over rows g(i) vis_i`.

The weight could not ride `vis`: `_light_vis` is `ti.static(shadows != 0)`-gated
and compiles out entirely when shadows are off, so a weight parked there would
be dead-code-eliminated and every shadowless path-traced render would be
silently wrong. A slot→row map through the stage signature was the other
candidate and is a public API break (`FragmentStage`'s contract, with
`_stage_cosine_color` as the shipped example users copy) that would also
recompile every deterministic shade kernel.

**The one deviation from the plan of record: the mode is a `ti.template()`
argument (`auth_sampled`), not a `nee_meta` word.** It is forced, not chosen.
The summing arm hands `_run_frag_pipeline` a row ordinal that can run *past*
`vis_lights` — a 40-light rig at the 16-slot cap is exactly the case this
section is about — so that arm cannot go through a per-thread slot map at all,
and a runtime mode would make every scene carry the map (and a select per
channel read) whether or not it uses one. Taichi specialises on template
arguments, so both arms still compile and run in ONE process, which is what the
parity tests need; a `ti.static` gate read off a setting would not. It costs no
runtime argument slot and no arena entry, and the mode-0 variant is the kernel
this file compiled before. The two runtime words that remain (`S` and the
authored table's length) ride `nee_meta`, whose width went 18 → 20.

**Where the sampled rows come from, and why not the next-event table.** A
separate small power-weighted CDF over the light rows, appended after the
ambient tail of `nee_ref` with its own self-normalised span of `nee_cdf` — no
new arena entry, only two tables that got longer, and built at all only in the
sampled mode so an `"off"` render's bytes are unchanged. Selecting from the
next-event entries instead (the plan's first draft, with non-light-row entries
rejected at weight 0) fails on §6a-ter: a `RectAreaLight` is two emissive
triangles there and its `K` cell rows are withdrawn, so an authored floor under
an area light would have lost its only light. It also wastes every draw that
lands on an emissive mesh or the environment, which do not light an authored
surface at all — in a scene where those hold most of the power, nearly all of
them. The cost of keeping the two apart is that the authored branch does not
get the light tree's spatial awareness; on `decay = 0` rows, which is Algan's
default, §6a measured the tree at 1.03x the flat CDF, so there is little there
to lose. Two bases, not one: the authored rows follow the ambient tail in
`nee_ref` but only the sampled entries in `nee_cdf`, because the ambient rows
have no selection probability at all.

**The residual bias, and why the switch has a third state.** `_stage_manim`
encodes to sRGB, adds its offset, clamps to `[0, 1]` and decodes — always,
linear working space or not — and `E[clamp(x)] != clamp(E[x])`. At `S = 1`
over 40 equal lights a sampled row carries 40x a light's radiance and clips, so
manim under a large rig reads darker and noisier than the sum. `"auto"`
therefore keeps the exact sum wherever it is affordable. Under
`ALGAN_LINEAR_COLOR=0` the illumination-budget normalisation `_energy_scale`
becomes `1 / max(wsum, 1)` of a now-random `wsum`, which is biased for the same
reason; under the default linear space it is exactly 1.0 and there is no such
term. And a **user** stage that uses a light's direction without multiplying by
its colour sees an unweighted sum over the sampled rows — documented on
`FragmentStage` and in `renderer_limitations.rst`.

**Sampler dimensions and adaptive sampling.** The draws spend the crossing's
own next-event pairs (`pair_cross0 + 2s` and `+ 1`): a crossing is either lit
or authored, never both, so no new dimension pair was needed, and `S` is capped
at `pt_light_samples` for exactly that reason. This also retires the hash-RNG
draw the summing arm uses for its per-light soft-shadow jitter, whose salt
`processed * 64 + li` aliases above 64 lights. `stoch = 1` was already set
unconditionally for this branch and stays; the comment beside it now names the
light pick as the primary reason so a future narrowing cannot drop it.

**Measured** (this 4-vCPU CPU box, `pt_baseline --scene
many_lights_authored --resolution 320x180`, 64 lights, 16 spp, 4 bounces, one
process per arm, warm RUN 2 — the arm is the `many_lights` rig with every solid
in `MeshToonMaterial` / `ManimMaterial`):

| arm | wall | `pt_shade` device | spp actually taken |
| --- | --- | --- | --- |
| `off` (the summing arm) | 6.267 s | 3693.9 ms (59.5% of wall) | 5.94 of 16 |
| `always` | 3.308 s | 517.1 ms (16.6% of wall) | 5.94 of 16 |

**7.1x less `pt_shade` device time and 1.89x less wall**, on a scene where the
shade kernel was 60% of the frame and is now 17% — traversal (468 ms) is the
larger item in the sampled arm. Adaptive sampling took the same 5.94 samples per
pixel in both, which is the point: the win is entirely per-crossing cost, not
fewer samples. `off` is also the arm that *misses* the shadows of 48 of the 64
lights, so this is not an equal-quality comparison in the sampled arm's
disfavour — it is cheaper and more correct at once.

**On the T4** (`benchmarks/performance/reports/t4_2026_09/pt_authored_1.md`,
the same scene, five frames, 16 spp ceiling with adaptive sampling,
denoiser on): `pt_shade` **1619 → 235 ms at 720p (6.9x) and 3894 → 598 ms at
1080p (6.5x)**, end to end **3.15 → 1.61 s and 5.85 → 2.55 s**, at the same
5.7 mean samples per pixel. An authored material now shades in about what
the physically-integrated `many_lights` rig does (260–275 ms at 720p) rather
than six times as much.

**Slots, and local memory.** `shadow_vis_slots` is now asked for
`ambient + sampled` rather than for the light count, so the `vis` payload a
64-light authored scene carries drops from the 16-slot cap (192 B per thread)
to one or two slots, against the 8-16 B the two new vectors cost at that width.
Net reduction where the mode is on; unchanged where it is off, since that arm
compiles neither vector.

**Also fixed here, one line:** `tracer.py` fired the `shadow_lights`
truncation warning for path-traced renders too (gated only on `shadow_flag and
num_lights > max_shadow_lights`), and its message told the user to render with
the path tracer while they already were. It is now `and samples <= 1`.

**Tests**, all in `tests/unit_tests/test_path_tracer.py` and none marked
`fast`: `test_authored_sampling_lands_on_the_sum_it_replaces` (a toon floor
under 8 point lights plus the two direction-less rows, the summing arm exact,
the two within a count and a half),
`test_authored_sampling_lands_on_the_sum_for_manim_too` (the same with the
looser bound the clamp earns),
`test_authored_sampling_lights_an_area_light_the_same` (the §6a-ter
interaction), `test_authored_sampling_shadows_a_light_past_the_deterministic_cap`
(a blocker over 40 lights: the sampled arm is measurably darker because it
shadows the 24 the summing arm cannot),
`test_authored_sampling_auto_is_the_sum_on_a_small_rig` (`torch.equal`, both
arms in one process),
`test_authored_sampling_is_inert_for_the_deterministic_renderer` and
`test_authored_sampling_rejects_an_unknown_mode`. `tests/path_traced` and
`tests/fast` do not move at the default; `tests/path_traced` gained
`authored_under_many_lights`, which is the first scene in the suite that
exercises this branch at all.

### 6b. Building the tree — LANDED

Conty Estevez & Kulla 2018, as in PBRT-v4's `BVHLightSampler` and Cycles:
each node carries its subtree's emitted power, its bounds, and an
**orientation cone** (axis, normal spread `theta_o`, emission spread
`theta_e`). A delta row is a degenerate leaf — a point (or, for a directional
row, a direction) with no area and a full cone — which is why one tree can
hold rows and triangles together rather than needing a separate structure per
emitter kind; that unification is what production light trees are for, and
per 6a it is the point here rather than an extra. Sampling descends
stochastically — at each node, score the
children by an importance heuristic in the shading point `x`, normalise, pick
one with a *rescaled* random number, and multiply the probabilities down to
the leaf. The heuristic is roughly `power * |cos theta'| / d^2`, with the cone
bounding how much of the node can face `x` at all. Rescaling the single random
number rather than drawing a fresh one per level is what preserves
stratification, which matters here because the entry-select draw is a Sobol
pair (§7).

This directly attacks what makes the flat CDF bad: back-facing emitters get
near-zero probability instead of full power-proportional probability, and
distant ones are discounted by `1/d^2`.

**Why not reuse the scene BVH.** The instinct is right that a BVH already
exists and that a second one sounds redundant, but the STBVH is the wrong tree
in four specific ways, and they are worth spelling out because only the first
is obvious:

1. **Wrong payload.** Kernel-facing nodes are `blocks [first_leaf, 8, arity]`
   — six bounds lanes, a packed frame interval, one pad — chosen so a node
   visit is one aligned 128-byte (or 64-byte f16) fetch. There is no room for
   power and a cone. That part is easy to fix with a parallel array, so it is
   the least of the four.
2. **Wrong contents.** It holds *every* triangle, and holds no light rows at
   all — so it could never be the whole answer, only the emissive-triangle
   half of it. Emitters are a tiny subset of what it does hold:
   with per-node power sums a zero-power subtree is skipped in O(1), so it
   would function — but you descend `log(N_all)` levels to reach `log(E)`
   worth of decisions, and the interior nodes carry no useful discrimination.
3. **Wrong shape.** It is built to minimise ray-traversal cost (SAH). A light
   tree wants to minimise *sampling variance*, which Conty-Kulla do with an
   SAH variant that also penalises wide orientation cones — a node grouping
   emitters that face opposite directions is bad for sampling however tight
   its bounds are. Reusing the SAH tree gives a working sampler with a
   needlessly poor selection distribution.
4. **Wrong axis, specifically to Algan.** The STBVH is *spatio-temporal*: its
   leaves are primitive *instances* with frame intervals, and the 4D Morton
   order deliberately clusters the same primitive at adjacent frames. For
   light sampling that is precisely backwards — you want distinct emitters
   clustered spatially at one frame, not one emitter clustered across time.
   Interior power sums would also be sums over instances that may not exist at
   the frame being shaded, which does not bias the result (the pdf is whatever
   both MIS ends agree it is) but does degrade the distribution.

Against that, a purpose-built tree is cheap: it is over `E` emitter entries
rather than `N` triangles, it is built host-side in torch inside
`_build_nee_tables` — which already computes per-triangle power and area — and
it is rebuilt per render call, not per frame. Production renderers all build a
separate light tree for these reasons, and they unify analytic lights into it;
here, per 6a, the light rows do not need to be in a tree at all.

**The one substantive cost: the MIS pdf becomes a query, not a lookup.**
Today both ends of the emissive MIS pair read one constant,
`tri_emit_prob[prim]` — the NEE side to form `pdf_sa`, the BSDF-hit side to
form `pdf_ne`. A spatially-varying sampler makes the selection probability a
function of the shading point, so at a BSDF hit on an emitter the kernel must
recompute *the probability that next-event estimation would have chosen this
triangle from the previous vertex*. That means a PMF query, and the standard
implementation is an upward walk from the emitter's leaf to the root using
stored parent pointers (cheaper than a top-down re-descent) — so the tree
needs a parent array as well as children.

The good news, and the reason this is cheaper here than it first looks: the
query needs the previous vertex, and **the path state already carries it**.
`rs_ro` is not updated on a pass-through crossing, only on a scatter, so
during the peel loop `ro` *is* the previous scatter point — which is exactly
why the current code can write `pdf_ne` in terms of `t_hit`, the distance
measured from it. A position-only importance heuristic therefore needs no new
per-path state at all. Wanting the shading-normal term too would need three
more floats: `rs_sca` is width 12 with columns 0–5 used by the path tracer, 6
free, and 7–11 owned by the nested-IOR stack, so one free column exists and a
widening would cost `_PT_BYTES_PER_SLOT` +12 bytes and a slightly smaller
tile. Start position-only.

**What shipped: position-only at both ends, and the normal term dropped
outright.** The tempting half-measure is to use PBRT's receiver-cosine bound
at the next-event call site, where the shading normal *is* in registers, and
to leave it out of the upward walk where it is not. That is exactly the bug
this section warns about: the two ends would evaluate different selection
pdfs and the power-heuristic weights would stop summing to one — silently,
as a small brightness error on emissive meshes. The alternative was widening
the shared `rs_sca` (which is `SCA_WIDTH_NESTED`, so it sizes the
*deterministic* renderer's per-ray state too) by three columns for one free
one. So both ends call one `_pt_lt_importance`, with no normal term, and
`test_light_tree_descent_and_pmf_agree` compares the descent's returned
probability against the upward walk over 4096 random points and trees —
equal to 1e-5 relative, which is the float precision of the two orderings.
Making that identity exact needed one more care: the descent forms the right
child's probability as `i1 / total` and never as `1 - p0`, so both walks
evaluate the same expression.

`tri_emit_prob` kept its array and its predicate role (`> 0` still marks
"this triangle is in the table"), and a new `tri_emit_entry` column maps a
triangle to its tree-local entry index so the walk can start at
`lt_entry_leaf[frame, entry]`. The kernel's arena spec grew seven entries
(`tri_emit_entry`, the two node tensors, the entry→leaf map, the frame→tree
row map and the infinite list's two) rather than seven kernel arguments —
`pt_shade` was already near Taichi's ceiling.

**Verification, as shipped:** the MIS-identity probe above (the single test
that keeps MIS correct); the leaf probabilities summing to one; the build's
determinism; a two-frame render whose moving light produces two different
trees; and the equal-spp variance comparison in 6a. Every furnace,
reference-integral and RectAreaLight test in `test_path_tracer.py` still
passes unchanged — the tree is a sampler and must not bias.

Still open from this section: delta-light direct lighting invariant to
`pt_light_samples`, which needs the sum-everything path 6a leaves undone.

**Closed: the T4 host residual was a measurement artefact.** With the build
memoized, the tree arm of the 64-light scene at 720p still read 200–365 ms
more host time per five-frame render than the flat arm, and nothing the
harness measures accounted for it — not the build (cache hits), not the
next-event setup (7 ms per chunk more, logged by the PERF line), and not
kernel execution. Under cProfile (`pt-cprofile-1`) the two arms are equal
to 5 ms end to end and every host cost the tree adds is attributed and
small (43 ms of `_build_light_tree_tables` over five chunks); slower Python
overlapped the residual away, which a fixed host cost cannot do, so it was
concurrent activity on the box (the software encoder process on few vCPUs,
most likely) landing on one arm's timeline. The light tree's real host
cost is ~10 ms per chunk. Numbers in
`benchmarks/performance/reports/t4_2026_09/pt_lighttree_1.md`. The same
profile showed the one explicit `gc.collect()` in `scene_excluded_from_gc`
costing 220–260 ms of a 2.3 s render on both arms — a §0 host item worth
its own look.


## 7. Sampler quality: stratified lobe selection, blue noise

Two cheap sampling improvements the survey named. Both have LANDED; the
second ships off, because it was measured and the measurement did not carry
it.

**Lobe selection drew white noise — LANDED.** `u_lobe` came from `_pt_rng`,
not from a Sobol pair, so the diffuse/specular/transmission/pass-through split
at every crossing was unstratified — and that decision drives which lobe a
bounce explores, so it was not a minor dimension.

The original reason was sound: the choice happens per surface *crossing*, and
crossings per bounce are unbounded in a translucent stack, so it had no fixed
dimension index. But Stage 3 solved exactly that problem for next-event
estimation by indexing pairs on `processed`. The same trick applies here, and
it is what shipped: a crossing's block widened from `2L` pairs to `2L + 1`,
the next-event pairs keeping their places inside it
(`2 + 6B + (2L+1)c + 2s`) and the lobe select taking the last one
(`2 + 6B + (2L+1)c + 2L`, x component only). The custom-scatter branch pick,
which shared the same white-noise draw, moved with it. Cost is a pair-index
constant and one `pt_sample_2d_seeded` call; sampler purity is unaffected
(both draws are pure functions of the same key), and it moved every existing
sample, so it rides the section 5 batch's single re-baseline of
`tests/path_traced/`. `test_dimension_pairs_never_collide` checks that the widened
arithmetic still partitions the pair space, over the render shapes a scene
can take.

**Blue-noise error distribution in screen space — LANDED 2026-09-05, and it
ships OFF.** `_pt_key` hashed `(frame, pixel)`, so neighbouring pixels got
independent sequences and the per-pixel error was white noise across the
image. Heitz et al. (SIGGRAPH 2019) distribute the *same* per-pixel
Owen-scrambled Sobol error as blue noise in screen space, which at low spp is
markedly better perceptually for identical cost and identical convergence.
That should matter more here than for a film renderer: Algan's workload is
animation at modest spp, fed to a denoiser, and both the human eye and the
denoiser's convolutional prior prefer high-frequency error. It is built,
tested and measured; the measurement is why the default is off.

### What shipped

**The tile.** `scripts/generate_blue_noise_tile.py` writes
`algan/rendering/raytracing/data/blue_noise_tile_64.npy` — a `uint16` 64x64
**permutation** of `0..4095` (8 KB), each entry a per-pixel sampler key.
Simulated annealing on Heitz's energy,
`Σ_{p,q} exp(−|p−q|²/σ_i² − |s_p−s_q|²/(σ_s²·D))` over a 7x7 *toroidal*
neighbourhood, where `s(v)` is the sampler's own first two draws in pairs
`(0, 3, 54)` — sub-pixel jitter, bounce 0's BSDF direction, and the first
crossing's light point at the shipped `max_bounces`/`pt_light_samples`.
Parameters of record are in the script's docstring (σ_i = 2.1, σ_s = 0.35,
300 sweeps, seeded; two minutes on one core, reproducible byte for byte).
Two deviations from the paper, both deliberate and both explained there: the
sample-space distance is squared and normalised per component rather than
raised to `d/2` (that exponent is calibrated for a per-dimension tile and goes
degenerate at `D = 12`), and there is one layer rather than one per dimension,
because this sampler has one per-pixel key. **Permutation** is load-bearing:
over a tile period the key multiset is the whole key set, so the assignment
cannot bias the estimator — picking keys freely from a larger pool optimises
better and is wrong, since a fixed key is a quadrature rule and only the
randomness of the assignment makes it unbiased.

**How it enters the key.** `_pt_bn_path_seed`: `path_seed =
hash(_PT_BN_SALT, tile[(y + oy) mod 64, (x + ox) mod 64])`, replacing
`_pt_path_seed(seed_root, _pt_key(f·anim, pixel))` and nothing else.
`pt_sample_2d_seeded`'s Sobol/Owen internals are untouched, so a pixel still
walks one Owen-scrambled Sobol sequence — only *which* pixel walks *which*
sequence changed. `(ox, oy)` is hashed from `(pt_seed, frame-or-0)` and is a
**toroidal shift**, not a rehash: a shift is an isometry of the tile's own
torus, so it preserves the optimisation while still decorrelating seeds and
(under `pt_animated_seed`) frames. That spelling is forced — the tile is
optimised against the map from tile value to sample sequence, so nothing
per-render may enter that map, which is why `_PT_BN_SALT` is a fixed constant
and `pt_seed` moves the lookup instead. Its one cost: two `(seed, frame)`
pairs landing on the same shift render identically, one chance in 4096.

**Transport.** The tile rides `nee_meta`'s tail (`_NM_BN_BASE`, with
`_NM_BLUE_NOISE` the switch word) — no new arena entry on `pt_shade`, no new
`ti.template()` variant, and `pt_generate` (not arena-packed) takes
`nee_meta` as one ordinary ndarray argument so both ends of the sampler read
one table. Contract 2 holds: one compiled kernel, a runtime word.

**The measurement** (`benchmarks/_pt_blue_noise_check.py`, CPU, 96x96, a
miniature `lit_and_shadowed`, `pt_error_target = 0` so both arms take equal
spp, 24 render seeds per arm, scored against a 1024-spp reference of the off
arm):

| spp | raw MSE | denoised MSE | low-frequency MSE |
| --- | --- | --- | --- |
| 2 | −0.7% ± 1.1% | **+1.9% ± 3.6%** | +3.5% ± 2.9% |
| 4 | +1.5% ± 1.6% | **+2.4% ± 4.5%** | +3.8% ± 3.2% |
| 8 | +3.9% ± 2.1% | **+2.7% ± 2.4%** | +3.1% ± 4.3% |

(positive = the blue-noise arm is better; the error bar is the quadrature sum
of the two arms' standard errors.) Raw MSE is equal, which is the prediction —
blue noise does not change convergence. The other two are positive in every
single arm, which is not nothing, and inside one standard error, which is not
a win either. **The bar for defaulting on was >10% denoised at 4 spp; 2.4% is
not it, so `pt_blue_noise` ships `False` and no baseline moved.**

**Why it is small, which is the useful finding.** The tile does what it was
built to do *in isolation*: on the pairs it covers it takes 1.3–1.6x the
low-frequency energy out of the error field at 1–2 samples (the script's
`--verify` reports it against a random permutation of the same keys). It is
the sharing that dilutes it. Heitz et al.'s tiles are **per dimension** — a
scrambling and a ranking value per pixel per dimension — while this sampler
derives every pair from ONE per-pixel key hashed with the pair index, so one
tile must serve every dimension at once and each gets a fraction of the
optimisation. Measured directly: one pair alone in the energy scores 2.4x on
its own pair and nothing on the others; three pairs score ~1.4x each; six
score ~1.13x each. A real render then spends most of its variance in pairs the
tile does not cover at all (later crossings, bounces 1..7, Russian roulette),
so ~10% of its low-frequency error is in scope and 3% comes back.

**What a follow-up would need**, if this is ever worth revisiting: the tile
inside `_pt_pair_seed` rather than `_pt_path_seed`, one layer per dimension
pair, i.e. a tile lookup per *draw* instead of per *path*. That is a signature
change to `pt_sample_2d_seeded` and every one of its call sites, a global
load in the inner sampling loop, and 8x-64x the shipped data — a different
change from the one this section scoped, and it should be gated on a profile
of that load, not on this section's numbers.

It composes with §3's Tier-1 correlated seeding rather than competing with it
— one changes how error is distributed across space, the other across time —
and `pt_animated_seed` on keeps the spatial property per frame, since the
frame only shifts the lookup.

**What it is not.** An earlier draft offered "a hash of pixel coordinates
mixed into the sample index" as the cheap way to land it. That gives white
noise with a different correlation structure, not blue noise: `_pt_key`
already hashes the pixel, and re-hashing it cannot shape the error's
spectrum. This is the reason the shipped answer is a table to generate and
carry rather than a hash to write.

**Verification:** `tests/unit_tests/test_path_tracer.py`'s
`test_blue_noise_*` — the tile loads as a permutation and its constants agree
with the kernel's; the **off arm reproduces `pt_sampler_probe` exactly**, per
pixel, with no tolerance (and `tests/path_traced` renders md5-identical to
the pre-change tree with the switch off); a blue-noise pixel's prefix is still
stratified; the error field's low-frequency energy drops by >15% on the
sampler probe while its total power does not move; and one real render
exercises the kernel branch and agrees with the hashed-key arm on the mean.
`test_dimension_pairs_never_collide` is untouched — no dimension pair was
added or moved.


## 8. Splitting, continuation pools, and efficiency-aware RR

Byte-reproducibility used to be contract 1, and it made this section short:
splitting was ruled out architecturally, so EARS was ruled out with it. That
constraint is gone (see contract 1), so this is now an open engineering
question rather than a closed one. What follows is the ranking, because "add
a continuation pool" is three different changes with three different payoffs.

**What a pool is for.** Today one slot holds one path and every path owns an
exclusive accumulator row, so a path can only ever *continue* — never fork.
Three separate things want forking, and they are worth separating because the
cheapest is not the famous one:

1. **A shadow-ray queue.** NEE visibility currently runs *inline* inside
   `pt_shade`: `_pt_nee_visibility` calls the shared shadow walk in the middle
   of the shade kernel, once per NEE sample per crossing. That is the one
   place `pt_shade` still behaves like a megakernel, and it is exactly the
   pattern the wavefront literature exists to remove — the shade kernel
   carries the traversal's register pressure and its divergence on every
   thread, including the threads doing no shading at all. Deferring shadow
   rays into a queue traversed by their own kernel is the most
   architecturally-aligned change available, and it needs no accumulator
   change at all (a visibility result comes back to the path that asked for
   it). It is also the prerequisite for raising `pt_light_samples` or
   `pt_light_samples`-style allocation without the cost landing inside
   `pt_shade`.

   Caveat, and the reason this is "measure first" rather than "do this next":
   the deterministic renderer keeps its shadow walks inline too, and that was
   a considered choice there. The queue trades kernel simplicity for a
   round-trip through global memory per shadow ray, and at Algan's light
   counts the inline version may well win. Profile before building it.

   And note where the wavefront overhead actually is today: not in the
   inline walk but in the host sync every iteration (§0.2-bis). A queue
   adds launches and syncs to a loop whose problem is launches and syncs.
   The sync fix comes first, and only a profile taken after it can say
   whether the shade kernel's register pressure is the next limit.

2. **A dielectric split pool.** At a glass surface the path picks
   reflect-or-refract stochastically (`w_spec` vs `w_trans`); the
   deterministic renderer *splits* there instead — that is what
   `refraction_flag` and `refract_initial_pool_ratio` are for — so glass is
   noisier per sample here. The textbook fix is to follow both branches at
   the first dielectric interface, weighted by Fresnel share: the narrowest
   useful pool, splitting factor 2, at a known and rare vertex type, bounded
   by a small depth.

   Read the precedent carefully, though. That same deterministic split pool
   is *why* glass-heavy scenes OOM there, which is one of the reasons a user
   would be on the path tracer for such a scene at all. Copying it copies the
   failure mode into the renderer whose job is to not have it. See the
   demotion below.

3. **A general splitting pool, for EARS.** EARS (Rath et al., SIGGRAPH 2022)
   treats RR and splitting as one continuous factor `n` per vertex: `n < 1` is
   roulette, `n > 1` is splitting, and it chooses `n` to maximise *efficiency*
   — `1 / (variance x cost)` — rather than to bound variance. Classic
   throughput RR (what this renderer does: survive with probability
   `max(throughput)` past `pt_rr_start_bounce`, floored at 0.05) only ever
   reduces work and picks its probability from a heuristic that is not
   derived from anything optimal. EARS estimates the second moment and the
   cost of continuing from a lightweight spatial cache learned across
   iterations, then solves for `n` by a fixed-point iteration the paper proves
   converges to the efficiency-optimal factors. Its predecessor ADRRS (Vorba &
   Křivánek 2016) uses a precomputed adjoint and a hand-tuned weight window;
   its successor MARS (Rath et al., SIGGRAPH Asia 2024) extends the allocation
   to NEE, which is the principled answer to §6's "how many light samples"
   question.

**The cost every pool shares, and why it is bigger here than elsewhere.**
Contract 3 is not bookkeeping: "renders scenes the deterministic renderer
OOMs on" is one of the three reasons this renderer exists, and it is true
*because* per-path state is a fixed size that no scene can inflate. The
deterministic renderer OOMs precisely because its reflection and refraction
branches split into a shared pool whose occupancy is data-dependent. Adding a
pool here reintroduces that failure mode into the renderer whose job is to not
have it. That does not forbid pools, but it does mean every one of them is
spending the property that makes this the fallback, and must therefore come
with a hard cap, honest `_PT_BYTES_PER_SLOT` accounting, and a degradation
path that is *worse output*, never an OOM.

**Recommendation: do not start with the pool.** Adaptive sampling (§2) should
come first, and it is not a stepping stone to splitting — it is a substitute
for most of what splitting would buy here, at a fraction of the structural
cost, and it moves in the right direction on memory rather than the wrong one.
Both answer "spend effort where the error is"; adaptive sampling answers it
per *pixel*, needs no pool, no atomics and no accumulator change, and suits
Algan's variance distribution unusually well because a large fraction of a
typical frame is unlit 2-D content that is zero-variance by construction and
should terminate at the floor sample count. Splitting answers the same
question per *vertex*, which is finer than Algan's shallow transport usually
needs; it earns its keep in production renderers largely because their shading
is expensive and their paths are deep.

**The dielectric split is demoted.** An earlier draft ranked it first among
the pools, on the grounds that glass is noisier here than under the
deterministic renderer at comparable cost. That was a parity argument, and
parity is not a goal (see the top of this document and §5) — worse, it is a
parity argument for adding back the exact splitting behaviour that makes the
deterministic renderer OOM on glass-heavy scenes, which is one of the reasons
a user would be on the path tracer for that scene in the first place. The
honest framing is that stochastic reflect-or-refract is *noisier per sample*
and *bounded in memory*, and bounded memory is the feature. If glass noise
turns out to be the real complaint, the first answers are more samples,
adaptive sampling, and the denoiser — all of which keep the memory profile.

So the order is: **§2 adaptive sampling → measure → the shadow-ray queue if
the profile says the inline NEE walks hurt (it is the only one of the three
that does not touch per-path memory at all) → a capped dielectric split only
if measurement, not parity, asks for it → EARS last, and only behind a scene
where RR and splitting are demonstrably the bottleneck.** Building a general
pool before a profile points at one would mean importing the deterministic
renderer's overflow-and-retry machinery, and its OOM behaviour, on
speculation.

**When a pool does land**, the things to get right: accumulation becomes
atomic (fine now, but the AOV reduction and `pt_reduce` both assume exclusive
rows and must change together); `_PT_BYTES_PER_SLOT` grows by the pool ratio,
which shrinks the tile and must stay honest or the OOM retry mis-sizes; the
split factor needs a hard ceiling so no scene can drive it unboundedly; the
sampler needs a per-branch decorrelation term in the pair key so split
siblings do not reuse one sequence; and `tests/path_traced/` moves to the
statistical criterion in `agent_guidance/memory_perf.md` rather than exact
pixel comparison.

**Path guiding** (SD-tree, Müller et al. 2017) stays deferred, on the survey's
own advice: it pays off on indirect-dominated transport, and the survey says
outright that surface-dominated scenes with simple lighting should stop before
it. Algan's workload is that case. Revisit only if adaptive sampling lands and
indirect noise is still the limit.

**Also deliberately not pursued**, for the record: spectral rendering (the
survey calls it optional for RGB-sufficient work, and Algan has no
live-action plates to match); OpenPBR/MaterialX as the material model (Algan's
public surface is deliberately Three.js-shaped, per the API rules in
`CLAUDE.md` — adopting OpenPBR would be an API decision, not a renderer one);
and BDPT/VCM/MLT, which the survey advises against as a primary integrator and
which §1 already rules out for this architecture.


## 9. Completeness: the fallback must not refuse or silently drop anything

A fallback that rejects a feature leaves the user with **no renderer at all**
for that scene, and one that silently drops a feature is worse — the frame
comes out wrong with nothing pointing at why. Under the purpose stated at the
top of this document, each of these is a bug against the renderer's role
rather than a missing nicety. Audited at the current head — the custom-scatter
and clip-plane entries are kept after their fixes because they are the worked
examples of the failure mode this section exists to catch (one refused
outright, one silently inert):

* **Custom scatter overrides were hard-rejected — LANDED.**
  `_build_render_plan` used to put `"custom scatter overrides"` in
  `unsupported_features` when `samples_per_pixel > 1`, so a scene using
  `FragmentStage(..., scatter=...)` raised `UnsupportedFeatureError` rather
  than rendering. If that scene was also one the deterministic renderer
  cannot fit — the exact case the fallback exists for — the user had nowhere
  to go, and the error message told them to set `samples_per_pixel` back to
  1, which is the thing that did not work.

  The stated reason was that arbitrary user continuation carries no sampling
  density for stochastic transport to weight. True, but the renderer already
  has a category for exactly that: a **delta lobe**. Refraction and tinted
  panes both take a deterministic direction and continue with `prev_pdf = 0`,
  which suppresses the MIS weight and treats whatever the ray finds next as
  covered by no NEE strategy. A custom scatter fits that mould precisely —
  call the user's function for the direction, continue with weight 1 and
  `prev_pdf = 0`, and the estimator stays consistent. It is their code and
  their density; treating it as a delta continuation is honest and is the
  same contract the built-in delta lobes get. The one genuine limitation is
  that such surfaces cannot be MIS-covered, which is already the case for
  every other delta lobe.

  That is what shipped. `pt_shade` takes `frag_scatters` beside
  `frag_pipelines` — the same batch-narrowed tuple `wavefront_shade` takes,
  so `tracer.py` builds it once for either renderer — and at an authored
  crossing whose pid carries a scatter it calls the user's `@ti.func` with
  the pipeline's shaded colour, commits the `contrib` it returns in place of
  `alpha * local` (the scatter has already folded coverage and shading into
  it, exactly as the deterministic renderer's `weight * contrib` assumes),
  and continues along **one** of the three returned branches: `w_pass`,
  `w_refl`, `w_trans` are their max components, one is picked from the
  existing `u_lobe` draw and its throughput divided by its selection
  probability, the same importance weighting the built-in lobe pick uses.
  Paths still do not split (contract 3), a pass-through keeps peeling the
  batch and a reflect/transmit ends the camera segment and spends a bounce,
  and `refraction` is passed to the scatter as 1 because this renderer can
  carry a transmitted branch even though it never traces both. A scene with
  no custom scatter passes `()`, every read is behind `ti.static`, and the
  compiled kernel is unchanged — `tests/path_traced` stays green at its
  existing baselines, which is the byte-identity guard.

  Note the one thing this does **not** buy: a scatter surface is still
  outside next-event estimation (it runs no NEE block, as every authored
  crossing does not), so light reaches it only through the sampled
  continuation. That is the documented limitation on
  `renderer_limitations.rst`, not a defect to fix here.

* **`camera.near` and `camera.far` — FIXED.** Near clipping used to be inert
  under the path tracer while the feature matrix advertised it: it is applied
  in `wavefront_generate_rays`, which the path tracer does not call, and
  `pt_generate` built primaries through `_generate_ray` with no near-clip
  term. Far clipping was simply unimplemented and documented as such.

  Both now land where the deterministic renderer puts them. `pt_generate`
  advances the primary origin to the near plane along the camera forward axis
  and seeds `base_dist` with the skipped distance (so screen-space widths and
  the far plane stay camera-relative rather than origin-relative), and
  `pt_shade` retires a path when `base_dist + t_hit` passes the far plane, at
  the same site in the drain loop `wavefront_shade` uses. Because `base_dist`
  accumulates across scatters in both renderers, the far plane clips *path
  length from the camera* identically. `far_clip` rides `nee_meta` (a new
  `_NM_FAR_CLIP` word) rather than a new kernel argument, per contract 2.

  Measured before the fix, 48x48, one red square, `camera.near = 100`: the
  deterministic frame clipped to mean 0 and the path-traced one was unchanged
  at 35.62. After: both clip to 0 for a near plane in front of the geometry
  and for a far plane behind it, and planes generous enough to contain the
  scene leave the frame within one channel count of unclipped. Covered by
  `test_camera_clip_planes_apply_under_path_tracing`.

* **The 16-light shadow cap is only partly lifted, and the docs do not say
  so — CLOSED (§6a-bis).** Physically-integrated materials sample shadows
  through the NEE table and were always uncapped; authored-appearance
  materials looped lights and stopped at `max_shadow_lights`. They now sample
  their rows too (`pt_authored_light_sampling`, `"auto"`), the truncation
  warning no longer fires for a path-traced render, and both
  `renderer_limitations.rst` and `performance_and_quality.rst` say which
  materials get what.

* **The failures do not point at the fallback.** `record_truncation
  ("shadow_lights", ...)` warns that lights past the cap cast no shadow and
  stops there; `OutOfRenderMemory` at a one-frame window says the frame did
  not fit and stops there. Neither names `samples_per_pixel`, so the user
  who hits exactly the failure this renderer exists for is not told it
  exists. The fix is a sentence in each message (§0.3); the docs fix is
  the same sentence in `renderer_limitations.rst`'s hard-limits table and
  in `performance_and_quality.rst`'s "when to use" paragraph.

**The standing rule this section implies:** when the deterministic renderer
gains a feature, the question is not "does the path tracer match it" (§5 says
that is not a goal) but "**can the path tracer still render a scene that uses
it**". A feature that only one renderer supports is fine when it is the path
tracer's; it is a hole when it is the deterministic renderer's, because the
fallback direction only runs one way.

This section outranks §§1–8 under the renderer's purpose. A missing feature
in those sections makes a scene noisier or slower; a hole here leaves the
user with no renderer at all.

**Verification.** Two tests, both of which exist now:

* **The fallback never refuses** — LANDED, as
  `test_the_fallback_refuses_nothing` in
  `tests/unit_tests/test_path_tracer.py`: `_build_render_plan` must return an
  empty `unsupported_features` for `samples_per_pixel > 1`, over every
  feature it inspects together *and* one at a time (so a refusal conditioned
  on a combination fails it too). This is the machine-checkable form of the
  rule above and the thing that would have caught the custom-scatter hole; it
  is built by enumerating the features that function reads rather than by
  listing scenes, so a future rejection added to it fails the test the moment
  it is written. It failed before the custom-scatter work and passes after.
  The wiring — that a real authored pipeline registers as a scatter and
  reaches the kernel — is a render, in the same file
  (`test_custom_scatter_renders_as_a_delta_continuation`, a black 45-degree
  panel between two out-of-frame red emissive walls: red in its pixels can
  only have bounced) and in `test_ux_regressions.py`'s
  `test_lifted_path_tracer_features_render`.
* **Clip planes apply** —
  `test_camera_clip_planes_apply_under_path_tracing` (landed with the fix
  above). Note the shape of the assertion, because it generalises: it checks
  both that a clipping configuration *clips* and that a generous one leaves
  the frame alone. The first half alone would pass on a renderer that
  clipped everything; the second alone would pass on the inert
  implementation this replaced.


## Smaller known gaps, for completeness

Tracked here so they are one search away, in rough order of effort:

* **Isolated black pixels beside glass — FIXED.** In
  `environment_and_refraction` two pixels per frame (10 over the five
  frames) came out pure black inside the prism's silhouette against a
  bright sky, every one of their four neighbours saturated.

  **It was not Russian roulette and not the `min_weight` floor** — the
  hypothesis this bullet used to carry. Instrumenting every retirement path
  in `pt_shade` for those pixels put all 8 of each pixel's samples on the
  **`ok == 0` rejected-direction absorb**: each one refracted into the
  prism at bounce 0 and was then killed at bounce 1, on the prism's *exit*
  face, by the GGX branch's `n_dot_l2 <= 1e-5` test. Scene-wide, 774 of
  2715 specular picks (29%) retired that way. Roulette could not have been
  it: `pt_rr_start_bounce` is 3 and these paths died at bounce ordinal 1,
  and the roulette / `min_weight` / far-clip / truncation counters for those
  pixels all read zero.

  The cause is the shading normal's *side*. A one-sided solid keeps its
  outward normal on a hit from inside (`_sided_shading_normal`, deliberately
  — a backface there is genuinely its inside), so a path travelling inside
  the glass meets the exit face with `shade_n . -rd < 0`. Two things then
  went wrong at once. `_pt_lit_lobes` clamped that cosine with
  `ti.max(..., 1e-4)`, which read a head-on interior hit as a *grazing* one
  and handed the reflection lobe the grazing Fresnel limit — measured
  `w_spec = 1.0` against `w_trans = 0.96`, so half the samples chose
  specular. And the VNDF sample was then drawn about `shade_n`, which puts
  every direction it can produce below that normal's horizon, so the pick
  was always rejected and `absorbed` zeroed the throughput. Total internal
  reflection is the same hit with the transmission branch shut
  (`w_trans = 0`, measured on one of the two pixels): there the specular
  lobe is the only branch, so **100%** of those samples died. Half a
  pixel's samples dying is invisible against a blown-out sky; all of them
  dying is a black pixel, and at 8 spp with p ≈ 0.5 that is ~0.4% of the
  prism's pixels — the handful observed.

  **The fix** is one ray-facing normal, `spec_n` in `pt_shade`: `shade_n`
  flipped to face the incoming ray, used for the GGX lobe's ONB, its
  `n_dot_v`, its half-vector and its horizon test, and mirrored by
  `ti.abs()` on `_pt_lit_lobes`' `n_dot_v`. Mirror reflection is invariant
  to the normal's sign, so this is the same interface — only the cosines
  and the horizon test change, and a front-facing hit gets
  `spec_n == shade_n` and is bit-identical. Nothing is killed to make it
  work: `ok == 0` still absorbs, because a below-horizon microfacet
  reflection genuinely has BRDF 0 and absorbing it is the unbiased answer;
  what changed is that the lobe is no longer *offered* at a weight it
  cannot honour. The selection weights are importance-sampling
  probabilities divided back out by `p_sel`, so moving them moves variance,
  not the mean. Measured: `ok == 0` retirements 774 → **0** on the scene,
  black pixels 2/frame → **0**, and a lossless glass cube in a uniform
  environment now renders as a flat white furnace.

  Tested by `test_glass_against_a_bright_sky_leaves_no_black_pixels`
  (`tests/unit_tests/test_path_tracer.py`): a glass prism at 2 spp against a
  bright environment map, counting pure-black pixels whose four neighbours
  are saturated — 30 of 9216 before the fix, 0 after.

  It moves `tests/path_traced/environment_and_refraction` (435 of 46080
  pixels differ, max 255 counts) and, unexpectedly, `lit_and_shadowed` —
  but only 2 pixels of 46080, one of them by 4 counts (frame 3, brighter,
  not darker). That scene has no transmissive material, so the crossing
  must be one whose *shading* normal is fractionally backfacing while the
  face is not — a grazing indirect hit — where the lobe used to be rejected
  and now reflects; a recovered sample of 48 is the right size for the
  move. `translucency_and_order` is byte-identical. **Both moved baselines
  still need re-recording** (`ALGAN_UPDATE_PATH_TRACED_BASELINES=1`, then
  `scripts/package_baselines.py`).

  Residual, out of scope here and worth its own bullet if it ever shows:
  the reflected TIR branch's tint is Schlick evaluated on the *inside*
  angle rather than KHR's air-side angle (which `_material_reflectance`
  already implements for the transmission gate), so a reflection just past
  the critical angle keeps ~4% instead of 100%. That is an energy
  *understatement* on a path that used to be killed outright, so it is
  strictly an improvement; correcting it means giving the specular lobe the
  same side-aware reflectance the transmission lobe already gets.
* **Frame-animated emitters — now tested.** The NEE table samples frame-0
  emission power (dark-at-frame-0 emitters stay unbiased through the BSDF
  path, weight 1), and the MIS pdf evaluates per-frame area. Pinned by
  `test_a_frame_animated_emitter_lights_exactly_the_frames_it_is_on`
  (`tests/unit_tests/test_path_tracer.py`): an emissive quad beside a Lambert
  floor, stepped instantaneously between frames of ONE render job (both frames
  in one chunk, so one table built from frame 0), against a static control lit
  on every frame. Measured at 128 spp, 64x36: dark at frame 0 takes **0**
  emissive table entries and still lights frame 1 to 131.97 against the
  control's 132.64 (**−0.5%**, all of it through BSDF hits at weight 1), while
  its own frame 0 reads exactly 0.00; bright at frame 0 takes 12 entries,
  matches the control at frame 0 bit for bit, and reads exactly 0.00 at frame
  1 — no frame-0 power leaks through the table into a frame whose emitter is
  off.
* **A mirror's image of a translucent closed shell still doubles.** The
  opacity ring covers the camera segment only, deliberately matching the
  deterministic route's identical bounce-loop gap (see the comment block at
  `solid_shell_alpha` in `settings.py`). Closing both means carrying surface
  identity through arbitrary bounce trees.
* **CUDA baselines for `tests/path_traced/` — RECORDED.** On the Kaggle T4
  (`pt-cudabase-1`): all four scenes, byte-identical on a re-render in the
  same session and again in a second session. `environment_and_refraction`
  and `translucency_and_order` are byte-identical to the CPU set;
  `lit_and_shadowed` and `authored_under_many_lights` differ by a few
  counts. Procedure in `tests/README.md`.
* **Self-intersection offsetting was a fixed world-space epsilon —
  FIXED in the path tracer.** Every spawned ray left along the geometric
  normal by `10 * min_hit_distance` (1e-3 world units, five sites in
  `pt_shade`), which is scale-dependent in both directions: acne on a scene
  authored at large coordinates, light leaking through thin geometry on one
  authored at small. Those five sites now call `_pt_offset_ray_origin`, which
  is Wachter & Binder (Ray Tracing Gems 2019, ch. 6): offset in integer float
  space by 256 ULPs, scaled by the hit point's own magnitude, with a fixed
  `1/65536` step below `1/32` where a relative step would round to nothing.
  The matching pull-back at the light end (`20 * min_hit_distance` in
  `_pt_nee_visibility`) scales the same way, through `_pt_shadow_tmax`, which
  measures the pull-back as a difference of two nearby points so the 1e7
  sentinel a directional row or an environment sample carries stays exact.
  `test_offset_ray_origin_scales_with_the_hit_point` probes it at 1e-3, 1 and
  1e3.

  **The deterministic renderer's own offsets are NOT changed** and none of
  its baselines move: this landed inside the path tracer only, and the
  frames it does move are `tests/path_traced/`'s, in the section 5
  re-baseline batch. Porting it to `wavefront_kernels_taichi` /
  `sheet_resolve_taichi` would re-baseline every committed frame in the
  repository on both devices, which is a change to make on its own.
