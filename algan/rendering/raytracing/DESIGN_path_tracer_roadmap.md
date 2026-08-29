# Path tracer roadmap: what is deliberately missing, and what landing it takes

This is the engineering companion to the user-facing list in
`docs/source/advanced_user_tutorials/renderer_limitations.rst`: that page says
*what* the renderer does not do; this one says *why not yet* and sketches what
each feature costs to land in this codebase specifically. It is the plan of
record for the path tracer's remaining scope — update it when one of these
lands, the way `DESIGN_optimization_targets.md` is updated for optimization
work.

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

What follows is everything the original plan named beyond the staged table
(§§1–4), then the divergences from the SOTA survey the redesign was specified
against that were *decided* rather than merely deferred (§§5–8) — each is a
place where Algan knowingly departs from what a production path tracer would
do, and each is reversible if the trade stops paying.


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
   (`nee_meta`) and runtime-gated branches, never new `ti.template()`
   arguments (variant explosion) — and the kernel sits at 59 of Taichi's 64
   runtime-argument ceiling, so new inputs prefer widening an existing tensor
   over adding one.
3. **The arena.** All per-path and per-scene state is accounted in
   `_PT_BYTES_PER_SLOT` / scoped allocations so the tile/wave split and the
   OOM chunk-halving retry keep working.
4. **The deterministic 2-D contract.** Camera-segment transparency composites
   with zero variance (`benchmarks/_pt_parity_check.py` holds flat interiors
   to ≤ 1 channel count against the deterministic route at any spp). A feature
   that would make unlit stacks stochastic is wrong by construction here.
5. **The sampler dimension table** in `path_tracer_taichi.py`'s module
   docstring is the registry of who consumes randomness. Pairs
   `2 + 6b + 4, 5` are already **reserved for volumes**.


## 1. Caustics

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
tracer on Algan's actual workload.

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

* **Tier 1 — correlated seeding (cheap, do first).** An experimental
  `pt_temporal_seed` mode that drops `frame` from the pair key so every frame
  reuses one sample set: static regions become perfectly stable (identical
  estimates), moving regions re-randomize through the geometry itself. One
  line in `_pt_key`, no new buffers, sampler purity untouched. Trade-off to
  document: correlated error reads as a fixed noise "texture" rather than
  shimmer — usually the better artifact for animation, but it is a choice,
  hence a switch rather than a new default.
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

**Why it is this way, and why that is defensible.** Brightness parity with the
deterministic renderer is a product requirement, not an implementation detail:
`spp == 1` is the default, every example in the docs renders through it, and a
user raising `samples_per_pixel` to reduce noise must not watch their lighting
change key. Light rows are the only emitters the deterministic renderer has,
so they are the only ones with a parity obligation. Emissive triangles and env
maps have no deterministic counterpart to match, which is exactly why they
were free to use the physical BSDF — and they had to, because MIS is only
correct when both ends of the pair evaluate the same function.

**Why it should not stay this way forever.** Three reasons, in ascending
severity:

1. It is a silent trap for any future feature that makes light rows
   MIS-able — making area lights hittable geometry is the obvious one, and it
   is also how an area light would gain its mirror image. The moment a BSDF
   ray can find a light row, the two ends must agree or the weights stop
   summing to one, and the failure is a subtle brightness error rather than
   an exception.
2. The white-furnace tests (`test_lambert_furnace_is_lossless`,
   `test_ggx_furnace_keeps_energy_with_compensation`) only exercise the
   transport half. Nothing pins the stage half's energy, so a future edit to
   `_pt_direct_response` can break reciprocity with no test failing.
3. It costs the renderer the ability to state a single answer to "what BSDF
   is this?", which every other physically-based decision here rests on.

**What resolving it takes.** The honest fix is to make the *deterministic*
renderer's stage formulas energy-correct (drop the `k` remap for the exact
Smith `G2`, put the `1/pi` on diffuse and rescale authored light intensity to
compensate) so both renderers converge on one BSDF, then delete
`_pt_direct_response` in favour of `_pt_lit_f_pdf` everywhere. That is a
re-baseline of every committed frame in the repo, on both devices — a
deliberate, self-contained change, not something to slip in beside another
feature. Until then this section is the record of the divergence, and any new
lit-vertex code must be explicit about which of the two it is implementing.

**Verification when it lands:** a Lambert and a GGX surface lit by a
`RectAreaLight` and by an emissive quad of matched radiance agree to within
noise; the furnace tests extended to the NEE path.


## 6. Many-light and emissive-mesh sampling

**What exists.** One flat power-weighted CDF over every sampled light row,
every emissive triangle and one environment entry, rebuilt per render call
(`_build_nee_tables`). Selection is global and purely power-proportional: no
spatial term, no orientation cone, no BSDF awareness.

**Where that is right.** For lights it is the correct call and the redesign
plan said so. Algan scenes carry single-digit light counts; a light BVH over
four rows would cost more to build and traverse than it saves, and the survey
recommends one (Conty Estevez & Kulla 2018) on the strength of production
scenes with thousands.

**Where it is already wrong.** Stage 3 put *emissive triangles* in the same
table, and those are not single-digit. One emissive mesh is thousands of
entries, and a global power CDF will happily pick a triangle on the far side
of the scene, facing away, occluded — a sample whose contribution is zero
before the shadow ray is even traced. The cost is paid per NEE draw per
crossing per bounce. This is precisely the regime the light-BVH literature
exists for, and Algan is now in it whenever a scene has a glowing mesh rather
than a glowing quad.

**Second, smaller regression.** `pt_light_samples` defaults to 1, so direct
lighting from *delta* rows is now stochastic: one entry drawn from the CDF per
vertex, weighted by `1/p_sel`. Before Stage 3 the kernel summed every row, so
delta direct lighting carried *no* variance at all — it was an exact analytic
sum, and the only noise in a simple lit scene came from indirect transport.
Single-sample selection introduces variance where there was none, and it grows
with the light count (an `N`-light rig now estimates its direct term from one
randomly chosen light per vertex). That is a bad trade at Algan's scale: the
CDF exists to bound cost when the emitter count is large, and the light-row
count never is.

**What to do**, in order of value per unit of work:

* **Split the table by cardinality.** Sum all delta and area light rows
  deterministically (bounded by `num_lights`, and already what the ambient /
  hemisphere fill does two blocks above), and keep the stochastic CDF for
  emissive triangles plus the env entry. Direct lighting goes back to
  noise-free, the emissive path keeps its MIS, and `pt_light_samples` becomes
  what it should be — a control on emitter sampling, not on lights. Small,
  local to `pt_shade`'s NEE block and `_build_nee_tables`; no new buffers.
* **A light BVH over the emissive-triangle entries** (Conty Estevez & Kulla
  2018: bounding box plus orientation cone per node, stochastic traversal on
  one rescaled random number, so stratification survives). Build it host-side
  beside the CDF, from the same frame-0 powers; the kernel walks it in place
  of the binary search. The MIS pdf stays computable — the traversal's
  selection probability replaces `tri_emit_prob[prim]`, which is already the
  single value both MIS ends read, so the change is contained to how that
  number is produced. This is the survey's "high-value, moderate-effort win"
  and is worth doing once emissive meshes are a supported look rather than an
  incidental one.

**Verification:** an emissive-mesh scene at equal time against the flat CDF
(the `_pt_furnace_check` / reference-integral pattern from Stage 3 gives the
ground truth); delta-light direct lighting invariant to `pt_light_samples`
once the table is split, since summing the rows makes that term exact.


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
time, and the pair is the standard low-spp animation configuration. Landing
it is a permutation applied inside `_pt_key` (a small precomputed ranking
tile, or a hash of pixel coordinates mixed into the sample index), no new
kernel arguments and no change to the purity contract.

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

2. **A dielectric split pool.** At a glass surface the path picks
   reflect-or-refract stochastically (`w_spec` vs `w_trans`). The
   deterministic renderer *splits* there — that is what `refraction_flag` and
   `refract_initial_pool_ratio` are for. So glass is noisier under the path
   tracer than under the deterministic renderer at comparable cost, on a
   headline Algan feature (`reflections_and_glass.rst`), and the fix is the
   textbook one: at the first dielectric interface, follow both branches and
   weight each by its Fresnel share. This is the narrowest useful pool —
   splitting factor 2, at a known and rare vertex type, bounded by a small
   depth — and the deterministic renderer's pool plus overflow retry is
   directly reusable precedent.

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

**Recommendation: do not start with the pool.** Adaptive sampling (§2) should
come first, and it is not a stepping stone to splitting — it is a substitute
for most of what splitting would buy here, at a fraction of the structural
cost. Both answer "spend effort where the error is"; adaptive sampling answers
it per *pixel*, needs no pool, no atomics and no accumulator change, and suits
Algan's variance distribution unusually well because a large fraction of a
typical frame is unlit 2-D content that is zero-variance by construction and
should terminate at the floor sample count. Splitting answers the same
question per *vertex*, which is finer than Algan's shallow transport usually
needs; it earns its keep in production renderers largely because their shading
is expensive and their paths are deep.

So the order is: **§2 adaptive sampling → measure → then (2) the dielectric
split, which is a concrete quality gap against the deterministic renderer,
and (1) the shadow queue if the profile says the inline walks are hurting →
EARS only if a measured scene shows RR/splitting is the remaining bottleneck.**
Building a general pool before there is a profile pointing at one would be
adding the deterministic renderer's overflow-and-retry machinery on
speculation.

**When a pool does land**, the things to get right: accumulation becomes
atomic (fine now, but the AOV reduction and `pt_reduce` both assume exclusive
rows and must change together); `_PT_BYTES_PER_SLOT` grows by the pool ratio,
which shrinks the tile and must stay honest or the OOM retry mis-sizes; the
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
* **Far clipping** is not applied by the path tracer (documented in the
  feature matrix).
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
