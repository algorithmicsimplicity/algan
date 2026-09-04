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

Power-heuristic MIS covers the two strategies that genuinely overlap:
emissive triangles and the environment map, each sampled both by next-event
estimation and by BSDF continuations that land on them. It does **not** cover
packed light rows, and cannot as the renderer stands — a delta light has zero
area and is unhittable by construction, and a `RectAreaLight` is a light row
rather than geometry, so no BSDF ray can find it either. That is sound (there
is nothing to double-count) but it has two visible consequences worth stating
plainly: an area light casts no reflected image in a mirror, and its highlight
on a glossy surface comes from the analytic stage formula rather than from
transport. §5 covers why those two ends do not agree numerically.

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
2. §0.3 — the defaults and the switch: what "turn on path tracing" means.
3. §9 — the fallback never refuses (custom scatter as a delta lobe, the
   never-refuses test, the failure messages that point at the switch).
4. §2 — adaptive sampling, if the baseline confirms the camera-segment peel
   and the unlit pixels are where the time goes.
5. §6 + §5 — the light tree, the authored-appearance sampling fix and the
   single BSDF, landed together as one re-baseline (§5 says why together).
6. §3 tier 2, §7's real blue noise, §8's pools — only behind a profile.
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
* **Sampler overhead per draw.** `pt_sample_2d` re-derives the
  `(seed_root, key)` half of its seed on every call although it is constant
  for the whole path; the roulette draw computes a full 2-D pair and keeps
  one component; `pair_nee0` is recomputed per crossing. All hoistable, none
  byte-identical (they move every sample), so they wait for the §5
  re-baseline rather than earning their own.
* **The authored-appearance shadow loop** (the second bullet of §6a-bis)
  is the remaining linear-in-lights term and needs the interface decision
  that section describes. It is not cheap and is listed there, not here.

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


## 5. One material, two direct-lighting responses

**What is inconsistent.** A lit vertex answers "how much light comes back
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

**Land it with the other re-baselines, as one.** This change, §7's
stratified lobe select, the sampler hoists in §0.2, the self-intersection
offset in the final section and §6's area-light change each move every
sample and each say "re-baselines `tests/path_traced/`". Four separate
re-baselines is four rounds of looking at frames and four release-asset
uploads (`tests/README.md`, "Where the heavy baselines live"); one is one.
Batch them behind a single branch and re-baseline once, on both devices.
Deleting `_pt_direct_response` also removes the second `_light_eval` call
per ambient row and shrinks the shade kernel, which is a small throughput
win in its own right.


## 6. Many-light and emissive-mesh sampling

**What exists.** One flat power-weighted CDF over every sampled light row,
every emissive triangle and one environment entry, rebuilt per render call
(`_build_nee_tables`). Selection is global and purely power-proportional: no
spatial term, no orientation cone, no BSDF awareness.

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

### 6a. The tree covers the whole table, not just emitters

So the light tree is not an optimization for the emissive-mesh case with the
light rows left outside it. It is the selection structure for **every** entry
the NEE table holds — delta rows, area-light cells, emissive triangles — with
the environment entry alongside as it is now. That makes per-vertex direct
lighting `O(log E)` in the total emitter count, which is the property the
"too many lights" use case actually needs, and it is what promotes 6b from
"worth doing once emissive meshes are a supported look" to **required for a
stated primary use case**.

A **sum-everything fast path** for tiny tables (a threshold on entry count,
in the low tens) is exact and zero-variance, but it costs `E` shadow rays per
vertex where the tree costs one, and with a denoiser downstream one
well-chosen sample is likely the faster route to equal perceived quality.
Do not build it first; build it if the §0.1 baseline on a two-light scene
asks for it.

### 6a-ter. A `RectAreaLight` is one emissive quad, not `K` rows

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

### 6a-quater. Build the tree per frame

`light_pos` and `light_col` are per-frame tensors, and `_build_nee_tables`
runs once per render *chunk*. A tree whose bounds are the union over a
chunk's frames is unbiased (the pdf is whatever both MIS ends agree it is)
but its importance heuristic degrades for anything that moves — and Algan is
an animation engine, so lights move. `E` is small and the chunk's frame count
is small, so build one tree per frame, indexed by `f` the way `light_pos`
already is, rather than one union tree. The per-frame emission power the
"frame-animated emitters" gap in the final section describes falls out of the
same change.

### 6a-bis. Two loops that are linear in light count today

Independent of the tree, `pt_shade` still walks every light row twice, and
under the purpose stated at the top of this document these are defects rather
than inefficiencies:

* **The ambient / hemisphere fill — FIXED (§0.2).** It used to scan all
  `num_lights` rows at every lit crossing to find the zero-to-two
  direction-less rows (`for li in range(num_lights)`, testing `lt_row` per
  row): in a 200-light scene, a 200-iteration scan per crossing per bounce
  to find two entries. The host now appends those rows to `nee_ref` after
  the sampled entries and the kernel loops their count from `nee_meta`.
* **The authored-appearance branch** (manim, toon, normal, matcap, depth and
  every `set_fragment_shader` pipeline) loops all lights and traces a shadow
  ray for each up to `max_shadow_lights`, filling the `vis` vector
  `_run_frag_pipeline` expects. That is the deterministic renderer's cost
  model *and* its 16-light cap, running inside the fallback: such a surface in
  a hundred-light scene gets shadows from the first 16 lights and pays 16
  shadow rays per crossing. These materials are opt-in rather than Algan's
  default (a shader-less mob is unlit, and the physically-integrated
  materials go through the NEE table), so this is a hole rather than the
  common path — but it is a hole in exactly the use case the renderer is
  advertised for, and the feature matrix does not mention that the cap is
  lifted only for some materials.

  The fix is harder than the first because `_run_frag_pipeline`'s interface is
  a per-light visibility vector. The options are to sample `pt_light_samples`
  lights from the table and fill only those slots (changing what the vector
  means, so the pipeline contract needs restating), or to keep the vector for
  the sampled subset and document authored-appearance materials as sampling
  their lighting like everything else. Either way it is an interface decision,
  not just a loop rewrite, which is why it is called out separately here.

### 6b. Building the tree

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

Two smaller consequences to keep straight: `tri_emit_prob` is also used as a
*predicate* (`> 0` gates the MIS weight and marks "this triangle is in the
table"), so that flag survives as an array even once the probability becomes a
query; and the NEE and MIS ends must call the *same* PMF routine, as they call
the same `_pt_lit_f_pdf` today — that identity is what makes the weights sum
to one, and it is the thing to test directly rather than by eye.

**Verification:** an emissive-mesh scene at equal time against the flat CDF
(the `_pt_furnace_check` / reference-integral pattern from Stage 3 gives the
ground truth); a unit probe asserting the descent's returned probability
equals the upward PMF walk for the same (point, triangle) pair, over random
points — that single test is what keeps MIS correct; and delta-light direct
lighting invariant to `pt_light_samples` once 6a lands, since summing the rows
makes that term exact.


## 7. Sampler quality: stratified lobe selection, blue noise

Two cheap sampling improvements the survey names and the implementation does
not have.

**Lobe selection draws white noise.** `u_lobe` comes from `_pt_rng`, not from
a Sobol pair, so the diffuse/specular/transmission/pass-through split at every
crossing is unstratified — and that decision drives which lobe a bounce
explores, so it is not a minor dimension. The dimension table's `2 + 6b + 0`
entry reserved an `x` slot for it that the kernel never reads (the table now
says so).

The original reason was sound: the choice happens per surface *crossing*, and
crossings per bounce are unbounded in a translucent stack, so it had no fixed
dimension index. But Stage 3 solved exactly that problem for next-event
estimation by indexing pairs on `processed` (`2 + 6B + 2(cL + s)`). The same
trick applies here: give the lobe select its own crossing-indexed pair after
the NEE block. Cost is a pair-index constant and one `pt_sample_2d` call;
sampler purity is unaffected (both draws are pure functions of the same key),
though it does move every existing sample, so it re-baselines
`tests/path_traced/`.

**No blue-noise error distribution in screen space.** `_pt_key` hashes
`(frame, pixel)`, so neighbouring pixels get independent sequences — error is
white noise across the image. Heitz et al. (SIGGRAPH 2019) distribute the
*same* per-pixel Owen-scrambled Sobol error as blue noise in screen space,
which at low spp is markedly better perceptually for identical cost and
identical convergence. That matters more here than for a film renderer:
Algan's workload is animation at modest spp, fed to a denoiser, and both the
human eye and the denoiser's convolutional prior prefer high-frequency error.

It also composes with §3's Tier-1 correlated seeding rather than competing
with it — one changes how error is distributed across space, the other across
time, and the pair is the standard low-spp animation configuration.

**What it is not.** An earlier draft offered "a hash of pixel coordinates
mixed into the sample index" as the cheap way to land it. That gives white
noise with a different correlation structure, not blue noise: `_pt_key`
already hashes the pixel, and re-hashing it cannot shape the error's
spectrum. Blue-noise error distribution needs an *optimised* per-pixel
permutation — Heitz et al.'s scrambling and ranking tiles, or the
precomputed permutation of Ahmed & Wonka 2021 / Belcour & Heitz 2021 —
which is shipped data (a 128x128 tile per dimension pair is tens of
kilobytes), applied inside `_pt_key` with no new kernel arguments and no
change to the purity contract. Small, but it is a table to generate and
carry, not a hash to write.

**Verification:** stratification tests as today (unchanged — they probe one
pixel's sequence); a low-spp render's error spectrum measurably
high-frequency-weighted; equal-spp perceptual comparison on the
`lit_and_shadowed` suite scene.


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
rather than a missing nicety. Audited at the current head — the clip-plane
entry is kept after its fix because it is the worked example of the failure
mode this section exists to catch:

* **Custom scatter overrides are hard-rejected.** `_build_render_plan` puts
  `"custom scatter overrides"` in `unsupported_features` when
  `samples_per_pixel > 1`, so a scene using `FragmentStage(..., scatter=...)`
  raises `UnsupportedFeatureError` rather than rendering. If that scene is
  also one the deterministic renderer cannot fit — the exact case the
  fallback exists for — the user has nowhere to go, and the error message
  tells them to set `samples_per_pixel` back to 1, which is the thing that
  did not work.

  The stated reason is that arbitrary user continuation carries no sampling
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
  so.** Physically-integrated materials sample shadows through the NEE table
  and are uncapped; authored-appearance materials still loop lights and stop
  at `max_shadow_lights`. The limitations page presents the cap as a single
  renderer-wide limit. See §6a-bis for the fix and why it is an interface
  decision.

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

**Verification.** Two tests, one of which exists now:

* **The fallback never refuses** — assert `_build_render_plan` returns an
  empty `unsupported_features` for `samples_per_pixel > 1` across every
  feature combination the deterministic renderer accepts. This is the
  machine-checkable form of the rule above and the thing that would have
  caught the custom-scatter hole; it is the test to add alongside the custom
  scatter work, and it should fail today. Build it by enumerating the
  features `_build_render_plan` inspects rather than by listing scenes, so a
  future rejection added to that function fails the test the moment it is
  written.
* **Clip planes apply** —
  `test_camera_clip_planes_apply_under_path_tracing` (landed with the fix
  above). Note the shape of the assertion, because it generalises: it checks
  both that a clipping configuration *clips* and that a generous one leaves
  the frame alone. The first half alone would pass on a renderer that
  clipped everything; the second alone would pass on the inert
  implementation this replaced.


## Smaller known gaps, for completeness

Tracked here so they are one search away, in rough order of effort:

* **Frame-animated emitters are untested.** The NEE table samples frame-0
  emission power (dark-at-frame-0 emitters stay unbiased through the BSDF
  path, weight 1), and the MIS pdf evaluates per-frame area — implemented,
  never pinned by a test. A two-frame scene with an emitter that brightens at
  frame 1 is the missing test, not new engine code.
* **A mirror's image of a translucent closed shell still doubles.** The
  opacity ring covers the camera segment only, deliberately matching the
  deterministic route's identical bounce-loop gap (see the comment block at
  `solid_shell_alpha` in `settings.py`). Closing both means carrying surface
  identity through arbitrary bounce trees.
* **CUDA baselines for `tests/path_traced/` do not exist.** Procedure is in
  `tests/README.md`; needs a CUDA machine.
* **Self-intersection offsetting is a fixed world-space epsilon.** Every
  spawned ray leaves along the geometric normal by `10 * min_hit_distance`
  (1e-3 world units, five sites in `pt_shade`), which is scale-dependent in
  both directions: acne on a scene authored at large coordinates, light
  leaking through thin geometry on one authored at small. The survey names
  Wächter & Binder (Ray Tracing Gems 2019) — offset in integer float space,
  scaled by the hit point's own magnitude — as a day-one item. It is a
  drop-in `@ti.func` replacement at those five sites plus the deterministic
  renderer's own offsets, and it re-baselines rendered output, so it is a
  change to make on its own.
