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
nested-media refraction), the power-weighted NEE table with power-heuristic
MIS (delta/area light rows, emissive triangles, the environment map's 2-D
luminance CDF and escape fold), the Sobol–Owen sampler, Russian roulette,
firefly clamping, jittered-pixel AA, the closed-shell opacity ring, the
`tests/path_traced/` baseline suite and parity benchmark, and the OIDN RT
denoiser with in-kernel albedo/normal AOVs. What follows is everything the
original plan named beyond that table, plus the one structural absence
(caustics) that is not a stage but a consequence of the architecture.


## The contract every one of these must land under

These are not preferences; each is load-bearing and tested.

1. **Byte-reproducibility per machine + memory budget.** Every random decision
   is a pure function of path identity — `(pt_seed, frame, pixel, dimension
   pair, sample index)` for Sobol pairs, plus the peel step for the hash RNG.
   No atomics; accumulation in a fixed order (`pt_reduce`, and the host-side
   AOV reduction). Anything stochastic a new feature adds must draw from new,
   documented dimension pairs, never from shared mutable state.
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
  it is *deterministic given the path* (no new randomness → reproducibility
  free), it lands inside `pt_shade`'s existing NEE loop, and its
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
  kernel family. Reproducibility survives if photon count and hash order are
  fixed, but the memory model and the batch-window structure both grow real
  complexity.
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

**Verification:** byte-reproducibility at fixed settings; an all-unlit 2-D
scene terminating at the floor count (assert via the plan's sample tally);
equal-error-vs-equal-time versus uniform sampling on the `lit_and_shadowed`
suite scene.


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
  line in `_pt_key`, no new buffers, reproducibility untouched. Trade-off to
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

**Verification:** tier 1 — two renders of a static scene at adjacent frames
byte-identical in static regions; tier 2 — temporal variance of a fixed
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
diffusion limit; byte-reproducibility throughout.


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
