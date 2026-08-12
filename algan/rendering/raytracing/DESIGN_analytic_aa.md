# Algan — Analytic Anti-Aliasing: Design Document

Status (§19 is the current front line: analytic AA now beats supersampled
`anti_alias_level = 2` on 8 of 11 feature configs and falls 7-9% short on the
other three, all of them reflection/refraction/specular CONTENT).

**PHASES 1 AND 2 SHIPPED** under the opt-in master toggle
`ALGAN_ANALYTIC_AA` (still default OFF). Bezier circuits (§13) and flat
triangles (§14–§16) are both covered, `ALGAN_ANALYTIC_AA_TRI` is now default ON,
and analytic AA at `anti_alias_level = 1` beats the plain aliased render on every
config in the check script — the sphere silhouette, a translucent mesh, sub-pixel
rods, text, slanted quads. Phases 3–4 (residual aliasing, the eligibility scan
and the default flip) are still design only. Read §13 for circuits, §14–§16 for
triangles, in that order: §14 is what was first built, §15 the fixed-point
rasterization that replaced its epsilon arbitration, §16 the sample-less-triangle
policy that closed the last gap. The rest of the document is the original design,
unchanged except where those sections correct it. §17 adds supersampled
continuation rays, which take the reflected image off §7's list of what coverage
cannot antialias.

Goal: replace supersampling (`RenderSettings.anti_alias_level`, default 2 → 4x
the pixels) with per-fragment **analytic coverage** — the fraction of the pixel
square each primitive actually covers — so a render at `anti_alias_level = 1`
matches or beats the current AA=2 edge quality at roughly a quarter of the
primary-visibility cost.

This document maps the existing pipeline, picks the insertion points, and
enumerates the correctness traps. The single most important conclusion is in
§5: naive "coverage as alpha" produces **visible seams on every internal edge of
every triangle mesh**, and the plan must budget for the fix, not discover it in
review.


================================================================================
1. HOW ANTI-ALIASING WORKS TODAY
================================================================================

Supersample-and-box-filter, end to end:

  * `RenderSettings.anti_alias_level` (`settings/render_settings.py:38`,
    default 2; `HD`/`MD`/`PRODUCTION` all inherit it).
  * `render_loop.py:807-812` sets `camera.screen_width/height` to
    `num_pixels_* aa`. Everything downstream — primitive projection, bezier
    flattening, the raster front-end, the wavefront — runs at that inflated
    resolution.
  * `tracer.py:697-716`: the default strategy is `width = screen_width * aa`,
    `kernel_aa = 1`, `post_aa = aa`. (`ALGAN_INPLACE_AA=1` instead loops `aa^2`
    jittered sub-pixel rays per output pixel — same sample count, less memory,
    still `aa^2` work. `wf_composite_accum_aa` / `wf_finalize_aa` implement it.)
  * `post_process.py:247-266`: strided box-average of the `aa x aa` block back
    down to the output resolution.
  * Only *one* ray per (sub)pixel is cast: `_generate_ray(..., 0.5, 0.5, ...)`
    — the sample is the sub-pixel centre. So AA=2 is a regular 2x2 grid, no
    jitter, fully deterministic.

Cost: everything scales with `aa^2` — candidate pairs, intersection tests,
fragment records, the fragment sort, the resolve, shadow events, bounce rays,
the composite, and the post-process downsample. AA=2 is a 4x tax on the
primary pipeline; AA=4 (`THUMBNAIL`) is 16x.

Two post-filters exist but are not substitutes: `fxaa` (wired via
`RenderSettings.fxaa`) and `SMAA` (`post_processing/anti_aliasing/smaa.py`,
implemented but **not wired into `post_process_frames`**). Both are morphological
guesses from the final image; they cannot recover sub-pixel geometry (a 0.7-pixel
text stem, a hairline) and they shimmer temporally.


================================================================================
2. WHERE ANALYTIC COVERAGE CAN LIVE
================================================================================

2.1 Not in the wavefront ray tracer
-----------------------------------
A ray tracer only learns about a primitive when a ray *hits* it. A pixel whose
centre misses a triangle by 0.2 px produces no hit at all, so there is nothing
to compute coverage from — you can only erode the inside of an edge, never
antialias its outside. Making the wavefront coverage-aware would mean dilating
the BVH slab tests and the barycentric acceptance in `_collect_hits` by the
projected pixel footprint, which changes traversal, K-buffer occupancy and
ordering everywhere. Out of scope; see §8 for what this implies for
raster-ineligible batches.

2.2 In the hybrid raster front-end — yes
-----------------------------------------
`ALGAN_HYBRID_RASTER` is now **default ON** (`settings.py:404`; note
`DESIGN_hybrid_raster.md` still says "default OFF" — stale), as are
`RASTER_SPARSE_COVERAGE` (`settings.py:543`) and `BVH_REFIT`. So the default
deterministic primary path is already a rasterizer, and it already has
everything analytic AA needs:

  * It enumerates candidates from a screen-space bbox that is **already dilated
    by one pixel** (`raster_pipeline.py:153-156`), so partially-covered edge
    pixels are already in the candidate set — no binning change needed.
  * It computes screen-space **edge functions** per candidate pixel
    (`_ss_pixel`, `raster_taichi.py:184-225`) — unnormalized signed distances
    to the three triangle edges, exactly the quantity analytic coverage needs.
  * For bezier circuits it computes the **distance to the nearest outline
    segment** (`_bezier_point_metrics`, `raytrace_kernels_taichi.py:242`)
    plus an inside/outside crossing parity — i.e. a signed distance field, for
    free.
  * It composites through a single scalar `alpha` per fragment
    (`raster_taichi.py:1321-1366`), which is the natural place to fold coverage
    in.
  * The sparse path (`prepare_sparse_raster_coverage`,
    `raster_pipeline.py:543`) emits *all* hits as sorted fragment records and
    truncates each pixel's run at its first opaque hit — a clean, per-fragment
    data model with no z-buffer to make per-sample.

So: **analytic AA is a raster front-end feature.** That also makes it a strong
forcing function for finishing item 2 of `DESIGN_hybrid_raster.md` §13.


================================================================================
3. COVERAGE FOR FLAT TRIANGLES
================================================================================

`_ss_pixel` already computes, at the pixel centre `q = (px+0.5, py+0.5)`:

    e0 = (sx2-sx1)*(qy-sy1) - (sy2-sy1)*(qx-sx1)     (and cyclic e1, e2)

`e_i` is the 2D cross product of edge `i` with `(vertex_i -> q)`; its magnitude
is `|edge_i| * perpendicular distance`. Therefore the **signed distance in
pixels** from the pixel centre to edge `i` is

    d_i = o * e_i / |edge_i|,     o = sign(n0 + n1 + n2)   (winding/backface)

with `d_i >= 0` inside. `sx/sy` are already in pixel units
(`_project_points`, `raster_pipeline.py:181-195`).

Implementation: extend the per-batch projection table
`precompute_triangle_projection` (`raster_pipeline.py:39`) from 10 to 13
columns, adding `1/|edge_i|` for the three edges, computed once per (frame,
triangle) in torch. Then per pixel the coverage is three multiplies:

    cov = clamp(d0 + 0.5, 0, 1) * clamp(d1 + 0.5, 0, 1) * clamp(d2 + 0.5, 0, 1)

Properties worth knowing:

  * **Exact** whenever only one edge crosses the pixel — the overwhelmingly
    common case.
  * Across a shared edge between two coplanar triangles the two coverages are
    `clamp(d+0.5)` and `clamp(-d+0.5)`, which sum to **exactly 1**. That makes
    the seam-union rule of §5 exact for edges (not for shared *vertices*, where
    3+ triangles meet and the product form under-counts).
  * An exact variant (clip the unit pixel square against three half-planes,
    Sutherland–Hodgman, ≤7 vertices, ~40 flops) is available as an opt-in
    refinement if corners prove visible. Do not start there.

Acceptance test change: `_ss_pixel` currently accepts only
`b_i >= -BARYCENTRIC_EPSILON` (`raster_taichi.py:213-214`). It must instead
accept when `cov > 0`, i.e. `d_i > -0.5` for all `i` (plus the half-diagonal
slack for corners). Two consequences:

  * **Barycentrics become extrapolated** (some `b_i < 0`) for pixels whose
    centre is outside the triangle. Depth `t` extrapolates harmlessly, but
    colour/normal/UV sampling must not: clamp the barycentric triple to the
    simplex before `_tri_color_g` / `_tri_normal_g` / `_tri_uv`, or textures
    will sample out of range at every silhouette.
  * Fragment count grows by the silhouette perimeter. For a 200 px circle:
    ~31.4k full + ~1.3k partial = 32.7k fragments, versus 125.6k at AA=2 —
    a 3.8x reduction. For small text (a 3x20 px stem) it is only ~1.6x. State
    that honestly in benchmarks: **text-heavy scenes win least.**

Ray-cast fallback (camera-plane straddlers, `_raycast_pixel`): no screen-space
edges available. Return `cov = 1` there (no AA). Rare and bounded; §13 item 8
of the hybrid-raster doc would fix it properly.


================================================================================
4. COVERAGE FOR BEZIER CIRCUITS (text and 2D shapes)
================================================================================

This is the cheapest and highest-value half of the work, because
`_bezier_point_metrics` already returns `min_dist_sq` — the squared distance to
the nearest flattened outline segment — and `crossings` gives inside/outside.
Signed distance in the circuit's plane is `sd = ±sqrt(min_dist_sq)`, and
`pixel_size = pixel_world_scale[f] * t` (`raster_taichi.py:344`) converts plane
units to pixels. Coverage:

    cov_fill   = clamp(0.5 - sd/pixel_size, 0, 1)
    cov_border = box-filter of the |sd| - border_w/2 band, same construction

Required changes in `_bez_pixel_hit` (`raster_taichi.py:308-366`):

  * **Query radius must always include half a pixel**: today `query_radius` is
    `max(|border_w|, outline_w)` and `min_dist_sq` is left at `1e30` when the
    query is skipped (`raytrace_kernels_taichi.py:279-283`). Widen to
    `query_radius + 0.71 * pixel_size` so an outside pixel within half a pixel
    of the outline still gets a distance.
  * **Emit fragments for outside-but-near pixels**: the current test is
    `inside or is_border` (line 360); it becomes `cov > 0`.
  * **Retire the `outline_w = 0.6 * pixel_size` dilation** (line 346). That
    hack exists to keep sub-pixel strokes from vanishing, and it is *not*
    AA-invariant (`pixel_world_scale` uses `screen_height * aa`,
    `tracer.py:741`, so at AA=1 it dilates twice as much in output pixels as at
    AA=2). With coverage the correct behaviour is to fade a thin feature by its
    width rather than fatten it — but a zero-area fill has no interior, so
    hairlines still need a floor. Expect visual iteration here; `Text` and
    `Tex` rendering is the acceptance test, not a unit test.

Two accuracy caveats:

  * **Flattening resolution.** `num_pixels_per_sample = 0.5`
    (`primitives/bezier_circuit_primitive.py:51`) is the max curve-to-chord
    error, measured against `camera.screen_height`, which is the *supersampled*
    height. At AA=2 that is 0.25 output pixels; at AA=1 it becomes 0.5 output
    pixels, and a continuous coverage function will show the flattening
    facets that box-filtering currently hides. Tighten to ~0.25 output pixels
    when analytic AA is on. This raises edge counts and `_bezier_point_metrics`
    cost, partially offsetting the win on text-heavy scenes — measure it.
  * **Anisotropy.** `pixel_size` is isotropic in the circuit plane. A strongly
    tilted plane has an elliptical pixel footprint, so a rotated `Text` will be
    over- or under-blurred along one axis. The correct fix needs the SDF
    gradient direction (computable inside `_bezier_point_metrics` — it already
    forms the closest-point vector, it just discards it) combined with the
    projected plane bases. Ship isotropic first; camera-facing text, which is
    the common case, is exact.

Circuits have **no shared-edge problem** (a glyph or shape is one closed
circuit), so scalar coverage is correct for them. That is why this half can ship
independently and first.


================================================================================
5. THE SEAM PROBLEM — the central correctness risk
================================================================================

Multiplying alpha by coverage is "coverage as alpha", and it is wrong wherever
two primitives of the same surface share an edge inside a pixel.

Concretely, two opaque triangles of a sphere mesh sharing an edge that cuts a
pixel 40/60:

    weight = 1
    A: alpha_eff = 0.40 -> acc += 0.40*cA, weight = 0.60
    B: alpha_eff = 0.60 -> acc += 0.36*cB, weight = 0.24

24% of the background bleeds through a pixel that is 100% covered. Every
internal edge of every `Surface`, `Sphere`, `Cylinder` and imported mesh gets a
faint lattice. Supersampling has no such artifact. **This is not a corner case
in Algan** — flat triangles are the default primitive and every 3D shape is a
dense grid of them.

Note also that the existing seam de-duplication
(`raster_taichi.py:1240-1244`: drop an edge-flagged hit within
`DEPTH_TIE_EPSILON` of the previous edge hit) is the *opposite* fix for the
*same* phenomenon — it exists because both triangles report a centre hit exactly
on a shared edge. Under coverage, dropping the second fragment leaves a 40%
hole. **The de-dup rule and the coverage rule must be designed and changed
together.**

Resolution ladder, in the order I recommend attempting:

**(a) Same-object coverage union (recommended).**
Add a per-triangle **object id** to the merged arrays. The merge already
concatenates per-primitive blocks, so this is a `repeat_interleave` of block
indices — essentially free — and no such id exists today (`scene_builder.py`
merged keys are all per-triangle attribute arrays; `tri_col_row` is a
*colour-row* index that constant-property promotion collapses per mob, so it is
a tempting but unreliable proxy). Carry it per fragment. In the resolve, while
consecutive fragments share an object id **union** their coverage instead of
compositing:

    if obj == obj_prev:  eff = min(cov, 1 - cov_run);  cov_run += eff
    else:                eff = cov;                    cov_run = cov

Exact for shared edges (§3 shows the two coverages sum to 1), O(1) registers,
robust to depth slope — which a `DEPTH_TIE_EPSILON`-window heuristic is **not**:
`DEPTH_TIE_EPSILON = 1e-4` world units is far smaller than the depth change
across one pixel of a near-edge-on surface, so a depth-window union would
silently stop working exactly where meshes alias worst. This rule replaces the
`seam_t` de-dup.

**(b) Analytic sample masks (MSAA-style), if (a) proves insufficient.**
Each fragment carries an N-bit mask (N = 4/8/16) of which sub-sample positions
it covers, derived analytically: `d_i` is affine, so
`d_i(q + δ_s) = d_i(q) + n̂_i · δ_s`, i.e. 3 FMAs and a sign test per sample —
no extra intersection work, and still **one shading per fragment**, which is
the whole point versus supersampling. Union of adjacent masks is exact at
shared vertices too. Costs: 6 more projection-table columns (`n̂_i`), one i32
per fragment, and — for exactness through translucent stacks — per-sample
transmittance (N vec3s), which the resolve cannot afford; the affordable
compromise is a per-sample *opaque* coverage mask plus scalar transmittance for
translucent fragments.

**(c) Two-kernel resolve split (pairs with either).**
The host already compacts covered pixels into `covered_idx`. A segment-max of
`cov < 1` over each pixel's run partitions covered pixels into "all fragments
fully cover" (the overwhelming majority; resolve unchanged and byte-identical)
and "has a partial fragment" (edge pixels, ~perimeter). Launch the existing
`raster_first_shade` on the first slice and a coverage-aware variant on the
second, passing an extra ordinal-indirection array `ord_idx` so the compact CSR
indexing still works (`o = ord_idx[t]; start = run_offsets[o]; pixel =
covered_idx[o]`). This keeps register pressure off the hot path — the resolve
already inherits the megakernel's 21-25% occupancy ceiling
(`DESIGN_hybrid_raster.md` §13 item 4), so a fatter monolithic resolve would
tax every pixel to serve the silhouette.


================================================================================
6. PIPELINE CHANGES, STAGE BY STAGE
================================================================================

Sparse path (`RASTER_SPARSE_COVERAGE`, the default):

  1. `precompute_triangle_projection` — 10 → 13 columns (`1/|edge_i|`), or 19
     with sample-mask normals.
  2. `raster_tri_count` / `raster_tri_write`, `raster_bez_count` /
     `raster_bez_write` — compute `cov`, accept on `alpha * cov > MIN_ALPHA`,
     write a new `frag_cov` SoA array (f32 first; u8 quantization only if the
     ≤2-per-channel test tolerance permits, which is marginal). Both COUNT and
     WRITE recompute the intersection, so both pay the coverage math — it is a
     handful of FMAs, fine.
  3. `prepare_sparse_raster_coverage` (`raster_pipeline.py:709-741`) — the
     opaque truncation must only fire on a **fully covering** opaque fragment
     (`opaque_s & (cov >= 1 - eps)`); a partially covering opaque hit stays in
     the run as an ordinary alpha fragment. Also: `frag_cov` must be carried
     through the `index_select` reorder and the persistent copy, and
     `discovery_bytes` (`raster_pipeline.py:765`) must gain its 4 bytes/frag.
  4. `raster_first_shade` — `alpha *= eff_cov` right before the existing
     `alpha = clamp(alpha, 0, 1)` (`raster_taichi.py:1321`); everything
     downstream (the four-way share split, the glow lane, reflect/refract
     continuations, `weight`) then follows automatically. Replace the `seam_t`
     de-dup with the §5 union rule.
  5. `raster_shadow_event_build` — it **replays the resolve's exact walk**, so
     its acceptance decisions must be updated identically or shadow ids will
     desynchronize from fragments. This is the easiest place to introduce a
     silent bug.

Dense path (`raster_iteration_zero`, still used under an env map / non-
post-process tonemapping): additionally, `raster_tri_z` / `raster_bez_z`
(`raster_taichi.py:406-466`) must only write the `atomicMin` z key for
**fully covering** fragments — a partial opaque fragment must not occlude the
whole pixel — and partial opaque fragments must be routed to the transparent
fragment stream instead. `_terminal_z_hit` is then always a full-coverage hit,
so it needs no coverage of its own.

Untouched: the bounce loop, `wavefront_shade`, the BVHs, the composite
(coverage arrives folded into `pix_accum`'s colour and leftover weight, so
transparent-background alpha at `wavefront_kernels_taichi.py:1256-1260` comes
out right for free), and the Monte Carlo path tracer.


================================================================================
7. WHAT ANALYTIC AA DOES **NOT** FIX
================================================================================

Supersampling antialiases *everything*. Analytic coverage antialiases *primary
geometric silhouettes*. Dropping to AA=1 therefore regresses:

  * **Hard shadow boundaries.** A shadow ray per shading point is binary, so
    shadow edges become stair steps. This is the biggest regression risk and it
    hits exactly the scenes that look most "3D". Mitigations, cheapest first:
    prefer non-zero `shadow_radius`/`shadow_angle` (the existing golden-angle
    fan already softens); or supersample *visibility only* — 2x2 jittered
    shadow rays per event in `raster_shadow_trace`, which is 4x of a stage that
    is a fraction of total cost, not 4x of everything. Measure before choosing.
    **DONE in §17/§19** (the second option: the event carries a world-space pixel
    footprint and the trace moves the query point over the pixel). Worth knowing
    how it measured, though: it was NOT what the shadow configs' residual against
    aa=2 turned out to be — that was §19.1's un-antialiased ray-cast fallback.
  * **Specular highlights** on curved surfaces (`_shade_tri_hit`) will crawl.
    No cheap fix without roughness-aware normal-distribution filtering.
  * **Texture minification.** Sampling is bilinear with no mip chain
    (`_sample_tex_vec5`), and a ray tracer has no screen-space derivatives, so
    a minified texture aliases. Ray differentials or a per-triangle LOD
    estimate would be a separate project.
  * **Reflections and refractions** — secondary rays remain one per pixel, so
    the *image inside* a mirror aliases even though the mirror's own outline
    does not. **CLOSED in §17**: a reflective or refractive hit now spawns
    `ANALYTIC_AA_SECONDARY_SAMPLES` continuations from different sub-pixel
    positions.
  * **Interpenetrating geometry** — where two surfaces cross, neither has an
    edge there, so coverage sees nothing.

Practical consequence: keep `anti_alias_level` as a working knob (analytic AA
must compose with AA≥2, not replace the mechanism), and consider wiring the
already-implemented SMAA as a cheap residual filter.


================================================================================
8. ROADBLOCKS
================================================================================

**8.1 AA level is global; raster eligibility is per batch.**
`anti_alias_level` fixes `camera.screen_width/height` for the whole render
(`render_loop.py:807`), but `use_raster` is decided per merged batch
(`tracer.py:1411-1421`) and excludes PN patches, custom scatter, near clipping,
mem-trim, and the legacy textured/sorted orchestrators. With analytic AA at
AA=1, an ineligible batch renders **aliased**, and if eligibility varies across
a scene's batches the aliasing flickers on and off mid-video. That is worse than
uniformly aliased.
Recommended handling: a pre-render eligibility scan at `render_to_file` time
(walk the actor registry for PN-triangle surfaces and custom scatter/shaders,
check `camera.near`, check the relevant settings) and fall back to supersampling
for the *entire* render if anything disqualifies it. Imperfect — merge-time
facts can still surprise it — so also log loudly, and add an assertion that a
render which committed to analytic AA never routes a batch to classic primary
traversal.

**8.2 Nothing is byte-identical.** Every rendered pixel changes. That collides
with the repo's optimization standard (byte-identical A/B parity). Analytic AA
is a *feature*, not an optimization, so the gate must be a visual/statistical
quality comparison (analytic-AA-at-1 vs supersampled-at-2 and -at-4), plus a
byte-identical kill-switch proving the toggle-off path is untouched. Full
re-baseline of `tests/expected_outputs_cuda/` and the docs example videos when
the default flips.

**8.3 Taichi kernel-cache staleness.** Editing `@ti.func`s (`_ss_pixel`,
`_bez_pixel_hit`, `_bezier_point_metrics`) does **not** invalidate the offline
cache. Run `clear_cache(taichi_kernels=True)` before every A/B, and
never edit `*_taichi.py` while a render process is live.

**8.4 Register pressure.** The resolve is occupancy-bound. Coverage adds one
float per fragment plus the union accumulator (cheap); sample masks add more.
Hence the two-kernel split of §5(c) rather than one fatter resolve.

**8.5 Memory and batch-size accounting shift.** At AA=1 the pixel count drops
4x, so tiles, wavefront state (`_wavefront_state_bytes_per_primary`), the
sparse discovery footprint
(`note_sparse_discovery_footprint`) and the HDR frame buffer all shrink, and
frames-per-batch rises ~4x. Good for throughput, but batch sizing is a tuned
system — expect surprises, especially the fragmentation the HDR float buffer
already causes on tiny scenes.

**8.6 Ancillary `aa`-scaled quantities.** Audit each: bezier border widths
(`bezier_circuit.py:310`, `:654`, scaled by `resolution*aa` and cancelled by
`pixel_world_scale`'s `screen_height*aa` — AA-invariant, verify after the
change), the `0.6 * pixel_size` outline dilation (**not** invariant, §4),
background-image resampling (`scene.py:394`), and the flattening tolerance
(§4).

**8.7 The Monte Carlo path tracer (SPP > 1) is unaffected** — it antialiases by
jittered sampling and should keep supersampling semantics untouched.


================================================================================
9. SETTINGS / GATE
================================================================================

As built (this section was written before §13–§16; those are the authority):

  ALGAN_ANALYTIC_AA (default 0)      Master toggle. Off ⇒ every code path is
                                     byte-identical to today.
  ALGAN_ANALYTIC_AA_TRI (default 1)  Triangle coverage (needs the master).
  ALGAN_ANALYTIC_AA_BEZ (default 1)  Circuit coverage.
  ALGAN_ANALYTIC_AA_SEAM (default 1) Seam rule: sum one object's disjoint
                                     sub-areas instead of compositing them.
                                     Off only to measure the difference.
  ALGAN_ANALYTIC_AA_SLIVER (drop)    Sample-less-triangle policy: drop | exact |
                                     exact_occ | area (§16).
  ALGAN_ANALYTIC_AA_SECONDARY (4)    Sub-pixel samples for what coverage cannot
                                     do analytically: continuations per
                                     reflective/refractive hit AND shadow-ray
                                     query positions. 1 | 2 | 4 | 8 (§17, §19).
  ALGAN_ANALYTIC_AA_SECONDARY_MIN_ENERGY (0.12)
                                     Share of the pixel a reflected/refracted
                                     branch must carry to be worth N samples
                                     instead of one (§19.2).
  ALGAN_ANALYTIC_AA_BEZ_MIN_HALF_WIDTH (0.3), ALGAN_ANALYTIC_AA_CHORD_TOLERANCE
                                     (0.25) — circuit stroke floor and
                                     flattening tolerance (§13.2, §4).

The sample COUNT is not on this list on purpose: it is a compile-time constant
in `raster_taichi.py` that no template argument carries, so an env var would let
the offline kernel cache serve the wrong variant (§16.4).

Plus `set_analytic_aa(...)` setters beside the existing ones in
`raytracing/settings.py`, read live (`rt_settings.X` at call time — importing by
value freezes them before user code runs; that bug has shipped before). A
`RenderSettings` field is the eventual user-facing surface, once §8.1's
eligibility scan exists.


================================================================================
10. VALIDATION PLAN
================================================================================

  * `benchmarks/_aa_analytic_kill_switch_check.py` — toggle OFF must be
    byte-identical to HEAD on the standard config matrix (opaque/translucent
    tri, bez, text, shadows hard+soft, glass, splits), the same eight configs
    `_raster_empty_skip_parity.py` already uses.
  * `benchmarks/_aa_analytic_quality.py` — render each scene at
    {AA=1 aliased, AA=1 analytic, AA=2 supersampled, AA=4 supersampled} and
    report per-pixel L1/L∞ and an edge-band histogram against the AA=4
    reference. Analytic-at-1 must beat AA=2 in the edge band.
  * `benchmarks/_aa_seam_check.py` — a subdivided sphere and a subdivided plane
    at grazing incidence, opaque and translucent. Scan for the lattice: max
    deviation between an interior pixel's colour and its fully-covered
    neighbours. This is the test that fails if §5 is skipped.
  * `benchmarks/_aa_analytic_ab.py` — in-process alternating A/B of GPU render
    time (wall-clock is thermally noisy, ~2x cross-process). Report separately
    for a triangle-dense scene, a text-dense scene, and a shadowed scene.
  * Visual: `Text`/`Tex` at small sizes, a thin `Line`, a rotated `Text` plane
    (anisotropy), and a fade transition (all-translucent, every fragment in the
    sorted stream).
  * Then the repo pixel suite, one process at a time on Windows, and a
    re-baseline commit.


================================================================================
11. PHASING
================================================================================

  Phase 1 — Bezier circuits only. `frag_cov` plumbing + SDF coverage + widened
  query radius + emission on `cov > 0` + flattening tolerance. No seam work
  needed (§4). Biggest quality-per-line win for Algan's dominant content
  (text, 2D shapes) and it exercises the whole `frag_cov` data path on the
  easy geometry. Expect near-`aa^2` savings on circuit-only scenes minus the
  tighter flattening.

  Phase 2 — Flat triangles + the seam fix. Projection-table columns, `_ss_pixel`
  acceptance + barycentric clamping, object id at merge, union rule replacing
  `seam_t`, coverage-aware opaque truncation and z-prepass, and the two-kernel
  resolve split. This is the bulk of the work and all of the risk.

  Phase 3 — Residual aliasing: shadow-visibility supersampling, SMAA wiring,
  the anisotropic circuit footprint, exact pixel-square clipping — each
  measured independently and defaulted on its own evidence.

  Phase 4 — §8.1 eligibility scan, `RenderSettings.analytic_aa`, default flip,
  re-baseline, docs.


================================================================================
12. ANTI-GOALS
================================================================================

  * Analytic AA in the wavefront tracer (§2.1).
  * Coverage as a pure post-process (that is FXAA/SMAA; already available and
    strictly weaker).
  * Temporal AA — needs jitter plus motion vectors, and would break the
    deterministic, frame-independent output the test suite depends on.
  * Per-sample shading of any kind. The entire value of analytic coverage is
    one shade per fragment; anything that shades per sub-sample has reinvented
    supersampling.
  * Removing `anti_alias_level`. It must keep working, and compose with
    analytic AA, for the residual aliasing of §7.


================================================================================
13. PHASE 1 AS BUILT (circuits) — 2026-07-25
================================================================================

Toggle: `ALGAN_ANALYTIC_AA` (default 0) / `set_analytic_aa(...)`, with
`ALGAN_ANALYTIC_AA_BEZ` subordinate. Off ⇒ byte-identical (repo pixel suite,
26/26, against pre-change baselines; `_raster_empty_skip_parity.py` 8/8 max|d|=0;
`_raster_sparse_coverage_parity.py` pixel-exact).

13.1 Measured
-------------
`benchmarks/_analytic_aa_bez_check.py` — mean L1 against a supersampled aa=4
reference at 320x180, and distinct luminance levels in the edge band (a hard
staircase has ~2 per edge, a resolved one a continuum):

    config    L1 aliased -> analytic (aa=1)    edge levels (aa=4 ref)
    tri            byte-identical               phase 1 leaves triangles alone
    slant        0.445 -> 0.239  (-46%)         186 ->  648   (696)
    text         0.670 -> 0.226  (-66%)          23 ->  168   (108)
    mixed        0.604 -> 0.289  (-52%)         478 ->  760   (827)
    border       0.451 -> 0.188  (-58%)         309 ->  742   (700)

Identical numbers on the dense tile path (`--dense`), which is a different
pipeline for circuits (z-prepass + `partial_only` second pass) — the two agree.

`benchmarks/_analytic_aa_bez_ab.py` — analytic AA at aa=1 versus the aa=2
default, scored against the same aa=4 reference:

    scene   res         arm            wall    L1 vs aa=4
    text    2560x1440   aa2 supersampled 2.00s   0.100
                        aa1 analytic     1.63s   0.098      1.22x
                        aa1 no AA        1.60s   0.249      1.25x  <- ceiling
    shapes  2560x1440   aa2 supersampled 3.95s   0.066
                        aa1 analytic     2.58s   0.068      1.53x
                        aa1 no AA        2.48s   0.123      1.59x  <- ceiling

Read this carefully. **Quality: analytic AA at aa=1 matches aa=2 supersampling**
(0.098 vs 0.100; 0.068 vs 0.066) — the design goal. **Speed: the win is
whatever dropping aa 2→1 is worth on that scene, and analytic coverage costs
almost none of it** (1.53x of an available 1.59x). It is NOT the ~4x the pixel
count suggests, because these short scenes are not render-bound: CPU prep and
video encode are common to every arm. Scenes that ARE render-bound will see
proportionally more; quote per-scene numbers, never a headline multiple.

13.2 As-built deltas from the design above
------------------------------------------
  * §4's `outline_w` retirement was done by *reinterpretation*, not removal:
    the classic `0.6 * pixel_size` fill dilation becomes
    `ANALYTIC_AA_BEZ_MIN_HALF_WIDTH` (default 0.3) as the minimum half-width of
    the drawn region, which reproduces the AA=2 reference stroke weight (the
    classic constant is 0.6 of a *supersample* pixel, hence 0.3 output pixels at
    aa=2 and 0.6 at aa=1 — it was never AA-invariant).
  * Colour classification had to widen with the region: a pixel in the
    half-pixel band *outside* a bordered circuit is border-coloured. `|d| <
    border_w` alone would have handed it the fill colour. (Superseded
    2026-07-31, §13.4: a filled circuit's border no longer reaches outside the
    outline at all.)
  * Triangles were left entirely alone rather than plumbed at coverage 1.0 in
    the kernels: the host pre-fills the `frag_cov` lane with 1.0, so
    `raster_tri_write` needs no change and a triangle-only scene is provably
    byte-identical. Phase 2 will make it write its own.
  * The dense tile path needed a second `partial_only` count/write pass over
    the *opaque* circuit candidates (§6): their fully covering pixels claim the
    z-prepass, their silhouette pixels must still blend. The sparse path (the
    default) needs no such thing — it has no z-prepass, only a truncation, which
    now fires solely on a fully covering opaque fragment.
  * `raster_shadow_event_build` applies the identical alpha scaling. It replays
    the resolve's walk, and a circuit fragment's alpha feeds `weight`, which
    gates termination; diverging there desynchronizes every shadow id from its
    fragment.

13.3 Known gap (circuits) — FIXED 2026-07-26
--------------------------------------------
The band form of the coverage filter was unexercised because a `Line` rendered
nothing at all: a filled `Square` drew while a `Line` in the same scene yielded
an entirely empty frame. The cause predated analytic AA and was **geometric,
not a coverage bug** — the packed polyline samples `t = k/n` for `k < n` only
and takes each cubic's endpoint from the first vertex of the segment it
connects to. That holds only where the connection is continuous. A segment that
CLOSES AN OPEN SUBPATH links back to a start point somewhere else, so its
endpoint was nobody's vertex and its final chord was simply missing; the
invisible closure edge ran from `t = (n-1)/n` instead of from `t = 1`. A
straight `Line` flattens to a single chord (`n = 1`, the curve-to-chord error is
zero), so its whole outline collapsed to one point.

Confirmed by forcing the chord count: at `n` chords exactly `(n-1)/n` of a
straight `Line` drew (`n=2` half, `n=4` three quarters), and a multi-segment
open path (`Line(path_arc=...)`, `Arrow`) was short by its last chord.

Fix: `_build_circuit_geometry` gives a segment whose connection is
discontinuous an explicit `t = 1` vertex (`needs_endpoint`, from the
`_bezier_connection_visibility` mask it already computed for the border flag),
so `verts_per_segment = num_samples + needs_endpoint` drives the vertex packing.
Closed circuits — `Square`, `Circle`, every glyph contour — have continuous
connections and are geometrically untouched. `_analytic_aa_bez_check.py` gained
the `unfilled` config (a straight `Line`, an arced one, an `Arrow` and an
unfilled `Circle`) now that there is something to measure.

13.4 Known gap (circuits) — the border's INNER edge — FIXED 2026-07-31
----------------------------------------------------------------------
Coverage antialiased the drawn region's *outer* boundary and nothing else, so a
bordered circuit resolved one edge continuously and the other by a hard
per-pixel classification: `in_border` was a BIT. The failure is loud whenever
the border is the only visible thing — an outlined glyph over a transparent
fill has a smooth outer contour and a stair-stepped inner one. Supersampling
never showed it, because there the classification runs at `aa`x resolution and
the box filter down-samples it.

Two coupled changes:

  * **The border of a FILLED circuit now runs inward** (`_circuit_point_region`,
    shared by the raster and classic paths). It was a band `|d| < border_w/2`
    straddling the outline, so raising `border_width` dilated the shape:
    neighbouring glyphs fused and bordered text went pudgy. The drawn region is
    now the fill alone, and the border is the part of it with `d <= border_w`.
    `_M_BORDER_W` became the FULL stroke width (the host dropped its `/2`), so
    apparent stroke weight is unchanged for a given `border_width`. Unfilled
    circuits have no interior to eat into and keep the centred band.
  * **The inner boundary gets its own box filter.** The fill-only region's
    coverage `clamp((d - border_w)/px + 0.5, 0, 1)` is subtracted from the total,
    and the remainder over the total is the border's share of the covered area.
    `_sample_circuit_color_blend` composites the two regions by area-weighted
    alpha — the premultiplied average supersampling converges to.

The share rides in the low 8 bits of the packed fragment ref
(`_pack_bez_ref`/`_decode_bez_ref`, and `_exact_fragment_order` shifts by 8 to
recover the layer) rather than in a per-fragment lane: it is a blend weight for
an 8-bit framebuffer, and a lane would have cost an ndarray argument through
five kernels plus every memory estimator. `_terminal_z_hit` needs it too —
full OUTER coverage is exactly the condition for reaching the z-prepass and says
nothing about the inner edge, so an opaque glyph's stroke straddles it there.

Circuits with `border_width = 0` (plain `Text`/`Tex`, plain fills) are
byte-identical: `outer_w = max(0, outline_w)` was already `outline_w`, and the
border share is gated on a non-zero width.


================================================================================
14. PHASE 2 AS BUILT (flat triangles) — 2026-07-25
================================================================================

Toggle: `ALGAN_ANALYTIC_AA_TRI`, **default OFF**, subordinate to
`ALGAN_ANALYTIC_AA`. With it off, everything below compiles out and circuits
behave exactly as §13 measured; the repo pixel suite (26/26) and both raster
parity suites stay byte-identical, and phase 1's numbers are unchanged.

It is off because it is not finished. What follows is what was established, so
the next attempt starts from evidence rather than from the top.

14.1 What is built and works
----------------------------
  * Coverage from the screen-space edge functions, via three reciprocal edge
    lengths added to the projection table (columns 10:12, allocated only when
    the feature is on). Acceptance widens from "centre inside" to "covers any
    sub-pixel sample", and barycentrics are projected back onto the simplex
    before they index colours/normals/UVs.
  * `tri_obj`, a per-triangle source-primitive id built at merge in the same
    three-block order `_geom` produces (§5(a)).
  * The seam rule: consecutive same-object fragments accumulate absorption
    ADDITIVELY against the transmittance from before the group started, rather
    than compositing multiplicatively. **This works and is the headline
    result** — on a subdivided sphere it takes interior notches from 2678 to 9,
    and the lattice visible over the whole object disappears
    (`_analytic_aa_bez_check.py seam`, which A/Bs against
    `ALGAN_ANALYTIC_AA_SEAM=0`).
  * Per-sample occlusion: each fragment carries an 8-sample mask and claims
    only samples no nearer opaque fragment has taken. This is what stops a
    mesh's back faces being ADDED to its front faces at a silhouette.
  * Silhouette gradation is right: edge levels on an opaque sphere go 280 →
    431..734 depending on configuration, against 608 for the aa=4 reference.
  * The z-prepass, the sparse path's opaque truncation, and the dense path's
    second `partial_only` pass over opaque candidates all handle partial
    coverage; `raster_shadow_event_build` replays the identical walk.

14.2 What does not work, and why
--------------------------------
A sub-pixel sample lying exactly ON an edge shared by two triangles is
evaluated by each from the opposite traversal of that edge. In floating point
the two results are not exact negatives, so **rounding alone decides which
triangle owns the sample**, differently from pixel to pixel. Both readings are
wrong in opposite directions:

  * let both CLAIM and both OCCLUDE it → the nearer takes it from its
    neighbour → dark speckle along every shared edge (499 notches at 16
    samples, 4626 at 8);
  * let both claim but neither occlude → a back-facing sliver picks it up at a
    silhouette → the mesh dilates by about a pixel, a visible bright halo
    (whole-frame L1 against the aa=4 reference 0.742 vs 0.243 aliased).

An epsilon band around the edge was the attempt to arbitrate this. It cannot:
the edge functions are cross products of screen coordinates in the hundreds, so
their rounding is already ~1e-5 pixels, and a band wide enough to be meaningful
is also wide enough to leak the halo. Measured both ways (1e-5 and 1e-3): seam
notches 8-9 either way, whole-frame L1 0.742 either way.

Also measured and rejected along the way, so they are not re-tried:

  * **Continuous area coverage** (product of clamped edge distances). Exact for
    an isolated edge, but an area cannot say WHERE in the pixel a fragment lies
    and the seam rule has to add sub-areas — so silhouette slivers sum to a
    halo and vertex wedges fail to sum to the pixel.
  * **Scaling coverage by the fresh-sample ratio** — fixed the halo (0.499 →
    0.338), made the seam worse (58 notches).
  * **Bounding the group total by the sample count** — the reverse trade.
  * **A depth window** to separate front from back faces: no threshold in pixel
    footprints survives grazing incidence, where it splits genuine shared edges
    and puts the lattice back.
  * **Winding-sign facing alone**, without occlusion: leaves the halo. Facing
    is still needed *with* occlusion, because a translucent mesh occludes
    nothing and its two sheets otherwise merge into one group and render
    opaque (L1 1.535 → 0.386 with facing restored).

14.3 What it would take
-----------------------
Evaluate the edge functions in FIXED POINT. Snapping the projected screen
coordinates to a sub-pixel integer grid makes the cross products exact
integers, hence exactly antisymmetric between two triangles sharing an edge —
at which point the classic **top-left fill rule** assigns every boundary sample
to exactly one triangle with no epsilon anywhere. Claim and occlude sets then
coincide, the masks partition the pixel exactly, and both failure modes go at
once. This is standard rasterizer practice and the reason hardware does it.

Cost/risks: int64 accumulators (a 4096-wide screen at 1/256 sub-pixel needs
~2^40 for a cross product), a fixed-point conversion in
`precompute_triangle_projection`, and care that the snapped coordinates are
shared between adjacent triangles (they are — both read the same vertex rows).

Only after that is it worth revisiting the translucent case, where scalar
transmittance still treats a mesh's two sheets as independently overlapping
rather than as the same sub-area seen twice (`trans` config: 0.386 vs 0.108
aliased). That one needs per-sample transmittance, which §5(b) already flags as
too register-heavy for this resolve — so the honest end state for translucent
meshes may be to leave coverage off for them.

14.4 Measurements (320x180, mean L1 against a supersampled aa=4 reference)
--------------------------------------------------------------------------
    config    subject                     aliased -> analytic   edge levels (ref)
    seam      subdivided sphere interior   9 notches vs 2678     (see 14.1)
    tri       two opaque spheres           0.243 -> 0.742        280 -> 431 (608)
    trans     translucent sphere           0.108 -> 0.386        108 -> 373 (340)
    slant/text/mixed/border (circuits, unaffected by phase 2)    see §13.1

Reproduce with `benchmarks/_analytic_aa_bez_check.py` (it forces triangle
coverage on for the `seam`/`tri`/`trans` configs and reports them as
characterisation rather than gates; the circuit configs remain hard gates).


================================================================================
15. FIXED-POINT RASTERIZATION (the §14.3 plan, carried out) — 2026-07-25
================================================================================

§14.3 said the way out was exact edge functions plus a top-left fill rule. That
is now implemented. It removed the epsilon arbitration entirely and took the
seam from a visible lattice to essentially nothing, but it did NOT close the
whole gap, and the reason is now understood and specific.

15.1 What was built
-------------------
  * Projected vertices snap to a **1/4096-pixel integer lattice** and the edge
    functions are evaluated in **int64**. Two triangles sharing an edge traverse
    it in opposite directions, and in exact integers their edge functions are
    then EXACT negatives (E = D x (Q-V1); the reverse gives -D x (Q-V2), equal
    because D x D = 0). In float they were merely near-negatives.
  * The classic **top-left fill rule** on top of that exactness: a sample lying
    on an edge counts only for the triangle whose traversal runs "down", or
    "left" when horizontal. Applied as a +1 integer bias, so no epsilon.
  * **Orientation comes from the exact integer area**, never the float edge
    sum. The float sum is the same quantity built from three large cancelling
    products, and two neighbours disagreeing about their winding was measured
    to be the cause of EVERY double-claimed sample.
  * The claim set and the occlusion set are now the same set, since the
    partition is exact.

Verified independently of the renderer: a standalone harness rasterizes random
quads split into two triangles and checks the masks. On consistently-wound
neighbours -- 256,000 pixel tests -- **zero samples are claimed by both and
zero are lost**. The rule is correct.

15.2 Two bugs found on the way, both worth remembering
------------------------------------------------------
  * **A name collision silently made the whole thing float.** The bias
    variables were called `b0/b1/b2`, which are already the float barycentrics
    earlier in `_ss_pixel`. Taichi keeps a local's type from its first
    assignment, so the integers became f32 -- values near 2^42 in a type with
    24 bits of mantissa -- and would also have corrupted the barycentrics on
    the way out. It announced itself only as a `TaichiWarning: Assign may lose
    precision: i64 <- f32`, which is easy to scroll past. Renaming them fixed
    a third of the remaining notches on its own.
  * **Block-scoped locals.** Values first assigned inside an `if ti.static(...)`
    block are not visible after it; two variables had to be hoisted to the
    fragment scope. Fails loudly, but confusingly (`TaichiNameError` pointing
    at the USE site).

15.3 What remains, precisely
----------------------------
The exact test answers "does this triangle CONTAIN this sample". A triangle
narrower than the sample spacing contains none, and near a silhouette a mesh
turns edge-on and produces a rim of exactly those. They cannot be dropped -- the
surface would have holes -- so they fall back to claiming the sample they come
closest to, with the continuous area estimate as their coverage. That fallback
is the entire remaining error, and it is a genuine dilemma:

    sliver occludes its claimed sample     seam 965 notches, tri L1 0.316
    sliver does not occlude                seam   9 notches, tri L1 0.520

Occluding steals the sample from the same-facing neighbour that really contains
it (notches); not occluding lets the mesh's back faces through at the rim
(silhouette dilation). Deferring the occlusion to the end of the coverage group
was tried and is no better: front and back faces alternate at a rim, so the
group closes constantly and the theft returns.

**Shipped setting: slivers do not occlude.** An interior lattice over every
mesh is far more objectionable than a one-pixel silhouette halo.

The fix is to stop approximating the sliver's area. The continuous product form
is a reconstruction filter -- it deliberately spreads coverage half a pixel past
the geometry -- so summing it over a rim of tiling slivers over-counts. The
EXACT area of (triangle n pixel square), by clipping the unit square against the
three half-planes (Sutherland-Hodgman, <= 7 vertices), does not: it is zero
outside the pixel and sums exactly over a tiling. Applied on the sliver path
only -- which is rare -- it should close the silhouette gap without touching
anything already working. That is the next step, and the last one identified.

15.4 Measurements after fixed point (same harness as §14.4)
-----------------------------------------------------------
    config    subject                    before §15        after §15
    seam      sphere interior notches    9 (epsilon band)  9   (now exact)
    tri       two opaque spheres         0.742             0.520
    trans     translucent sphere         0.386             0.237
    edge levels, tri                     431 (ref 608)     465
    edge levels, trans                   373 (ref 340)     327

The seam number is unchanged but is now earned rather than arbitrated: it comes
from an exact partition instead of an epsilon band that happened to be tuned
right, and it no longer trades against the silhouette. `tri` and `trans` both
improved substantially, and the whole-frame gap on `tri` is now attributable to
one identified mechanism (§15.3) rather than to accumulated approximation.


================================================================================
16. SAMPLE-LESS TRIANGLES — the last gap, closed — 2026-07-25
================================================================================

§15.3 named the remaining error precisely: a triangle narrower than the sample
spacing contains no sample, so the exact test has nothing to say about it, and
the fallback that gave it an approximate area dilated every silhouette. It also
proposed the fix — replace the approximate area with the EXACT clipped one — and
asserted that such triangles "cannot be dropped, the surface would have holes".

The exact area was built. **The assertion was wrong, and dropping is better.**

16.1 What was built
-------------------
  * `_pixel_clip_area`: the exact area of (triangle ∩ pixel square), as the
    boundary integral ½∮(x dy − y dx) taken around the triangle's outline
    *projected onto the square by a componentwise clamp*. The clamp is the
    nearest-point map onto the square, so the outline inside maps to itself and
    everything outside collapses onto the border, enclosing exactly the
    intersection. Each edge needs its four border-line crossings sorted (a
    five-comparator network); out-of-range ones clamp to an endpoint and drop out
    with no branching. Equivalent to Sutherland–Hodgman plus a shoelace, but with
    no vertex list — hence no dynamically indexed local array, which Taichi
    handles poorly — only a running accumulator.
  * `ANALYTIC_AA_SLIVER`, four policies for a sample-less triangle: `drop`
    (contribute nothing), `exact` (claim the nearest sample, weighted by the
    exact area, do not occlude), `exact_occ` (as `exact`, but occlude), and
    `area` (the §15 product form, kept only to measure against).
  * The policy travels to the resolve as a per-fragment mask bit
    (`_AA_SLIVER_OCC_BIT`), not as a template value, so the resolve — the
    occupancy-bound kernel — compiles once and serves every policy. Only the
    three geometry kernels get a per-policy variant, via the `aa` template value
    they already take (`1 + mode`), which also keeps the offline cache from
    serving one policy's `_ss_pixel` to another.
  * `benchmarks/_aa_clip_area_check.py` verifies the area function independently
    of the renderer, on the three properties the product form lacks: agreement
    with brute-force sampling (420 triangles across five scales plus deliberate
    slivers, all within 3e-3), exactly zero for a triangle disjoint from the
    pixel, and piece areas that sum to the whole over a fan-triangulated convex
    polygon (within 1e-4, the reference's own quantization).

16.2 Measured — every policy, same harness (320x180, mean L1 vs an aa=4 reference)
---------------------------------------------------------------------------------
`benchmarks/_analytic_aa_bez_check.py --sweep`. Ink is mean frame luminance
relative to the reference: >1 is silhouette dilation, <1 geometry lost.

    config  policy      L1      interior L1  notches  edge levels  ink
    tri     aliased     0.243      1.055         5       280        1.005
            area        0.520      1.227        10       465        1.046
            exact       0.174      1.185        16       596        1.009
            DROP        0.124      1.045        17       588        0.999
            exact_occ   1.035     12.514      5640       749        0.890
    trans   aliased     0.108      0.866         2       108
            area        0.237      1.018         3       327
            exact       0.108      0.963         2       310
            DROP        0.065      0.866         7       350
    thin    aliased     0.339      1.694         0       155        0.889
            area        0.764      1.523         0       477        1.288
            exact       0.100      1.367         0       373        1.023
            DROP        0.076      1.350         0       288        0.996
    seam    aliased     0.207      0.630         0       187        1.002
            area        0.362      0.745         9       465        1.016
            exact       0.145      0.655         8       491        1.003
            DROP        0.113      0.562         8       463        1.000

(aa=4 reference edge levels: tri 608, trans 340, thin 253, seam 440.)

Three things to read off this table:

  * **The exact area works, and confirms the §15.3 diagnosis.** `tri` 0.520 →
    0.174 and `thin`'s ink 1.288 → 1.023: the dilation really was the product
    form's half-pixel spread being summed over a rim of tiling slivers.
  * **Dropping is better still**, on every config and every column that matters.
    A sliver that claims a sample it does not contain is still positional noise;
    the exact area gets its MAGNITUDE right but not its POSITION.
  * **`exact_occ` is catastrophic** (5640 notches) and settles §15.3's dilemma
    for good: a sliver must never occlude.

16.3 Why dropping cannot open the hole §15.3 feared
---------------------------------------------------
Because the fill rule PARTITIONS the samples. Every sub-pixel sample inside the
mesh is contained by exactly one triangle, and that triangle contains a sample,
so by definition it is not a sliver. A sample a narrow triangle misses is
therefore owned by whichever neighbour of the tiling does contain it — dropping
the narrow one removes nothing. The only samples that can go unowned belong to
triangles whose projected area rounds to zero on the 1/4096-pixel lattice, which
is an error bounded by a lattice unit, not a hole.

This is also exactly what supersampling does: a sub-pixel sample either lands on
the geometry or it does not. The `thin` config is the direct test — rods 0.9,
0.45 and 0.22 output pixels wide, most of whose triangles contain no sample —
and dropping tracks the reference's total ink to 0.4%, against the 29% excess
the area fallback produced. The rods do not vanish; they fade, intermittently,
in the same pixels the reference fades them.

16.4 Sample count: 8, measured against 16
-----------------------------------------
The sample set is a compile-time constant (`_AA_SAMPLES`, the standard D3D 8x
and 16x patterns) and deliberately NOT a setting: it is baked into every
rasterizing kernel but is part of no template argument, so the offline cache
would serve an 8-sample kernel to a 16-sample build. Changing it means editing
that line and clearing the cache. Both were measured that way:

    config   8 samples             16 samples
    tri      L1 0.124  levels 588  L1 0.114  levels 601   (ref 608)
    trans    L1 0.065  levels 350  L1 0.065  levels 353
    thin     L1 0.076  levels 288  L1 0.093  levels 346
    seam     8 notches, int 0.562  1 notch,  int 0.636

    device time, 1280x720 two-sphere scene, warm run (benchmarks/_aa_sample_count_cost.py)
    raster_tri_count    98 ms -> 156 ms
    raster_tri_write   118 ms -> 140 ms
    raster_first_shade  74 ms ->  82 ms

16 improves `tri` slightly and nearly clears the residual seam notches, but
regresses `thin` and the seam's interior L1, for ~30% more time in the coverage
kernels. That is a wash bought at a real price, so **8 ships**.

16.5 What it is worth — triangles, against supersampling
--------------------------------------------------------
`benchmarks/_analytic_aa_bez_ab.py meshes` (the `meshes` scene added for this:
two spheres and a torus, one of each moving, 1280x720, alternating arms in one
process). The comparison that matters is not toggle-on-vs-off at one resolution
but analytic-at-1 against the shipped aa=2 default, both scored on an aa=4
reference:

    arm                          wall    L1 vs aa=4
    aa=2 supersampled            1.46s     0.079
    aa=1 analytic                1.14s     0.076     1.27x
    aa=1 no AA                   0.95s     0.114     1.53x   <- ceiling

Same shape as the circuit result in §13.1, and the same reading. **Quality:
analytic AA at aa=1 matches aa=2 supersampling** (0.076 vs 0.079) — the design
goal, now for triangles. **Speed: the win is whatever dropping aa 2→1 is worth
on that scene, and coverage costs almost none of it** (1.27x of an available
1.53x). It is not the 4x the pixel count suggests, because CPU prep and video
encode are common to every arm; quote per-scene numbers, never a headline
multiple. At 2560x1440 the aa=4 reference arm does not fit in 4GB, so this is
measured at 720p, where the scene is less render-bound than a real one — the
figure understates the render-side win.

16.6 What is left
-----------------
  * **8–17 interior notches** on a dense sphere (against 0–5 for the aliased
    render). Not the old lattice — those were thousands and covered the object.
    The mechanism is depth ORDER, not coverage: a fragment's `t` is the plane
    intersection extrapolated to the pixel centre, and for a near-tangent
    triangle whose true extent excludes that centre the extrapolation is far
    enough off to sort it against the wrong sheet, closing and reopening the
    coverage group. Bounded and not visible; fixing it means a per-fragment
    depth taken inside the triangle rather than at the pixel centre.
  * **Translucent meshes** remain approximate in the way §5(b) predicts: scalar
    transmittance treats a mesh's two sheets as independently overlapping rather
    than as one sub-area seen twice. Coverage still beats the aliased render
    there (0.065 vs 0.108), so this is no longer a reason to withhold it.
  * **Phases 3 and 4 are unchanged**: the residual aliasing of §7 (shadow
    edges above all) is untouched by any of this, and the §8.1 eligibility scan
    is still what stands between here and flipping `anti_alias_level` to 1 by
    default. §17 takes the first item off that list — the reflected image.


================================================================================
17. SUPERSAMPLED CONTINUATION RAYS — the reflected image — 2026-07-25
================================================================================

§7 listed what coverage does not antialias, and put reflections and refractions
on it: coverage resolves a mirror's own OUTLINE exactly, but the image *inside*
it is sampled by the continuation ray, and one continuation per pixel aliases no
matter how good the primary coverage is. That item is now closed.

`ANALYTIC_AA_SECONDARY_SAMPLES = N` (default 4, live only while the master
toggle is on) makes a reflective or refractive hit spawn N continuations instead
of one. Each is the primary ray re-generated through a different sub-pixel
position and re-intersected with that hit's own plane
(`_jittered_surface_sample`), carrying 1/N of the throughput. N == 1 compiles to
exactly the old single-ray code.

17.1 Design points worth knowing
--------------------------------
  * **Regular grid, not random jitter.** At N=4 the positions are the 2x2 grid
    `anti_alias_level = 2` supersamples at, so the reflected image is sampled
    where the arm this replaces samples it. Random offsets would also break the
    deterministic, frame-independent output the test suite depends on (§12) and
    make a mirror hiss between frames.
  * **One ray-plane solve, no re-traversal.** A flat triangle and a bezier
    circuit are both planes, so where a sub-pixel ray would have met the same
    primitive is exact.
  * **The shading normal is re-interpolated per sub-sample** (barycentrics from
    the same cross products, projected onto the simplex). This is not a detail:
    on a curved mirror the reflected direction turns by twice the normal's change
    across the pixel and curvature amplifies it, so reflecting several
    sub-samples off one shared normal *blurs* the reflection instead of resolving
    it. Measured on the mirror ball, sharing the normal left interior L1 at 1.92
    against 1.70 for the aa=2 arm; re-interpolating is what makes the sub-samples
    independent.
  * **The split happens once, at the primary hit.** Deeper bounces continue as
    single rays, so the cost is N times the secondary traversal, not N^depth.
  * **Sub-sample 0 continues in the pixel's own ray slot** (it carries the
    accumulated colour); the other N-1 go to the shared pool. Every branch
    commits both its colour and its leftover background weight when it retires,
    so 1/N each leaves the pixel's totals exactly as the single ray had them.

17.2 A real bug this exposed: coverage was dropping rim reflections
------------------------------------------------------------------
The resolve sends a reflection into the pixel's own ray slot only when the
reflected energy DOMINATES the pass-through (`refl_max >= cover_pass`); otherwise
the reflection must SPLIT — pool slot for the reflection, pass-through continues
— and that split path is compiled in only for a "splitting" batch, which
previously meant refraction, a custom scatter, or a semi-transparent reflector.

Analytic coverage makes every reflector's silhouette pixel partially covering,
hence `alpha < 1`, hence `cover_pass > 0` and `refl_max < cover_pass`. On a
mirror-only scene the split path was not compiled, so at those pixels the
reflection was **dropped outright** — a dark rim around every mirror, and worse
the better the coverage. It shipped with §14/§15 and no test covered it.

The fix is to treat any reflective scene under analytic AA as a splitting batch
(`_secondary_split_needed` in `tracer.py`), which is what secondary supersampling
needed anyway. On the mirror-ball config that alone took whole-frame L1 from
**0.346 to 0.153** — larger than the supersampling gain that follows it.

Two things made this silent and are worth remembering. A non-splitting batch gets
`pool_ratio == 1`, i.e. no spare pool slots at all; and at ratio 1 the host
IGNORES the pool's overflow flag, so failed slot reservations lose light
transport without a word. Anything that starts spawning continuations must join
the split flag, not invent a second notion of splitting — the pool sizing, the
active-set compaction and the fused-generation gate all read it.

Consequences to accept: such a batch also loses `opaque_closest`,
`opaque_prepass` and `mem_trim`, and its tile carries `max(2, N)` pool slots per
primary, so it fits fewer frames per batch.

17.3 Measured (320x180, mean L1 against a supersampled aa=4 reference)
---------------------------------------------------------------------
`benchmarks/_aa_secondary_check.py`. Interior = the lit region eroded by three
pixels, so the reflector's own antialiased silhouette cannot carry the result.

    config  arm       L1     interior L1  interior edge levels (ref)
    flat    aliased   0.486     0.958        450   (696)
            sec=1     0.338     0.781        561
            sec=2     0.303     0.644        628
            sec=4     0.289     0.585        660
            sec=8     0.279     0.544        698
            aa2       0.271     0.521        652
    mirror  aliased   0.261     1.998        227   (277)
            sec=1     0.153     1.923        259
            sec=4     0.137     1.930        260
            sec=8     0.128     1.826        280
            aa2       0.145     1.697        283
    glass   aliased   0.235     1.338        285   (397)
            sec=1     0.155     1.233        373
            sec=4     0.132     1.308        379
            aa2       0.116     0.998        375

`flat` (a flat mirror reflecting a straight edge) is the clean signal and is
monotone in N on every column: the reflected staircase resolves, reaching the
reference's own gradation at N=8. `mirror` (a mirror ball) and `glass` (a lens)
add extreme MINIFICATION of the reflected scene on top, which no sample count
fixes — their interior columns are therefore non-monotone in N and are reported
rather than gated. Note `mirror` at N=4 beats the aa=2 arm on whole-frame L1
while `flat` and `glass` land slightly short: supersampling antialiases
everything, including the shading and specular that analytic AA deliberately
leaves alone.

17.4 What it costs
------------------
`benchmarks/_analytic_aa_bez_ab.py mirror` (mirror ball + satellites + text,
1280x720, alternating arms in one process):

    arm                          wall    L1 vs aa=4
    aa=2 supersampled            3.34s     0.055
    aa=1 analytic + sec=4        3.32s     0.054     1.01x
    aa=1 no AA                   2.23s     0.115     1.50x

**On reflective content the trade is quality parity at cost parity.** The 1.27x
that analytic AA wins on a plain mesh scene (§16.5) is exactly what the four
secondary rays spend: you are paying for four reflection samples either way. What
you gain over aa=2 is the analytic silhouette; what you avoid is the 4x on
everything else. `sec=2` is available for most of the quality at part of the cost.

One practical cost: each (analytic, N) combination is its own compiled kernel
variant and the cold compile grows with the unrolled loop — N=8 took ~100s
against ~40s for N=1. Time warm runs only.

17.5 What is still NOT supersampled
-----------------------------------
  * **Stacked partially overlapping translucent fragments.** This is on §7's list
    and is NOT a continuation-ray problem, so jittered continuations do nothing
    for it: the fragments are all in the primary hit list, and the error is that
    scalar transmittance cannot say that two partially covering sheets occupy the
    SAME sub-area rather than independent ones. The mechanism that would fix it
    is per-sample transmittance (§5(b) — too register-heavy for this resolve) or
    delegating just those pixels to N jittered PRIMARY rays through the wavefront,
    which the pool could already carry. The trigger is cheap to detect during the
    walk (a partially covering, non-opaque fragment whose sample mask overlaps an
    earlier one from a different coverage group), and the resolve commits only
    after the walk, so it could discard and delegate. Not built. Coverage already
    beats the aliased render on the translucent config (§16.2), so this is a
    refinement, not a hole.
  * **Shadow edges** — still one shadow ray per shading point (§7). Its
    visibility accumulates every blocker's opacity, but the silhouette itself
    remains a single spatial sample. The same idea applies (2x2 jittered rays
    per shadow event in `raster_shadow_trace`) and is the next-most-visible
    item.
  * **Specular crawl and texture minification** — unchanged, and unaddressable by
    ray count alone at reasonable cost.
  * **The classic wavefront primary path.** These jittered spawns live in
    `raster_first_shade`, so a batch that routes away from the raster front-end
    gets neither coverage nor this — the same §8.1 hole, unchanged.


================================================================================
18. PER-SAMPLE TRANSMITTANCE — replaces the coverage group — 2026-07-25
================================================================================

The seam rule of §5(a) — sum the coverage of consecutive fragments that share a
source object — has been replaced by per-sample transmittance. It was not a
refactor: the group rule had a failure mode that made a scene **worse than no AA
at all**, and the replacement is exact where the group rule was heuristic.

18.1 The failure
----------------
The group rule assumes an object's fragments arrive CONSECUTIVELY in the depth
walk. Two INTERPENETRATING translucent meshes break that: within a pixel their
sheets interleave, so every switch closed and reopened a group, and a sheet split
across k fragments transmitted `(1 - a/k)^k` instead of `1 - a`. Too much light
got through, k varied per pixel, and the result was a speckled, too-dark overlap.

    two translucent spheres, mean L1 vs an aa=4 reference
      aliased (no AA)                    0.187
      analytic, group rule               0.581     <- 3x worse than no AA
      analytic, group rule off           0.551     <- i.e. the rule had stopped
                                                      working entirely
      analytic, per-sample transmittance 0.112     <- and now beats aa=2 (0.117)

Diagnostic worth keeping: the same two spheres SEPARATED IN DEPTH (overlapping on
screen but not interpenetrating) were fine under the group rule — 0.099 against
0.158 aliased. "Overlapping" was never the trigger; interpenetrating was.

18.2 What replaced it
---------------------
One array, `svis[s]`: how much light still reaches each sub-pixel sample.

    eff = sum(svis[s] for s in mask) / N        # what this fragment contributes
    for s in mask: svis[s] *= pass_through      # what it leaves for the rest

It subsumes three mechanisms and is exact where each was not:

  * **Shared edges** — disjoint masks sum to the pixel, so no lattice, and no
    object id is needed at all (`tri_obj` is now unused by the resolve).
  * **Opaque occlusion** — an opaque fragment zeroes its samples; the separate
    `occ_msk` is gone.
  * **Interleaved / interpenetrating surfaces** — each sample composites in true
    depth order, so consecutiveness stops mattering.

`weight` keeps only what cannot be per-sample: the CHROMATIC ratio of a tinted
transmitter, applied to the whole pixel scaled by the fragment's coverage. Exact
for a fully covering pane, which tinted glass almost always is. The pixel's
throughput is `weight * mean(svis)`.

Register cost is roughly neutral: N floats in, and `grp_absorb` (3), `grp_cov`,
`grp_obj`, `occ_msk` and a per-fragment `tri_obj` load out.

Side effects: the sliver policies `exact` and `exact_occ` now coincide (there is
no separate occlusion set to opt into), and an areal fragment — a circuit's SDF
coverage or a sliver's clipped area, both being a fraction of the pixel with no
POSITION — attenuates every sample uniformly instead of a subset exactly, which
is what circuits already did.


================================================================================
19. MATCHING anti_alias_level=2 EVERYWHERE — 2026-07-25
================================================================================

§13–§18 established that analytic AA beats the ALIASED render. The bar for
turning it on is different and higher: it has to match the shipped 2x2
supersampled default on everything, because supersampling antialiases every
quantity at once while coverage antialiases geometry and then needs a targeted
mechanism per remaining quantity (§7).

`benchmarks/_aa_match_aa2.py` is that gate: eleven configs, each covering one
aliasing SOURCE, scored as `L1(analytic @ aa=1) <= L1(supersampled @ aa=2)`
against a supersampled aa=4 reference. It also reports an ALIASED arm, which is
what separates "not antialiased yet" from "actively broken" — and it found one of
each.

19.1 The two findings that mattered
-----------------------------------
**Interpenetrating translucent meshes were broken** — §18, 0.581 against 0.187
aliased. Fixed by per-sample transmittance.

**The ray-cast fallback had no coverage, and it is not a rare case.** A triangle
that straddles the camera plane cannot be projected, so `_raycast_pixel` handled
it and reported coverage 1 — §3 called that "rare and bounded". It is neither:
ANY ground plane large enough to reach the horizon straddles the camera plane, so
its edges were byte-identical to no AA at all while everything else in the frame
was antialiased. Measured on a floor-and-sphere scene, a single row across the
floor's left edge:

    x                    21    22    23    24
    analytic (before)     0    98    98    97      <- identical to aliased
    aa=2                 47    47    71    98
    aa=4 reference       26    47    72    95

The fix is to answer the same set-membership question directly: cast one ray per
sub-pixel sample and test each against the triangle. That is the DEFINITION of
the sample mask, not an approximation of it, and it needs no near-plane clipping.
It took the shadow configs from 1.21x/1.24x of aa=2's error to **0.94x/0.97x**.

19.2 Wrong turns, recorded so they are not repeated
---------------------------------------------------
Both of the following were built, measured, and removed or corrected:

  * **Supersampling the SHADING at grazing incidence.** The theory was that a
    fully covered pixel on a near-edge-on surface spans enough world distance
    for the light term to swing inside it, and that this was the floor-scene
    residual. Built (per-sub-sample position, normal and barycentrics, N shading
    evaluations, gated on `|dot(n, rd)|`), measured, and it earned NOTHING on any
    config while costing 4x the shading — and made `spec` slightly worse. The
    residual was the ray-cast fallback above. Removed.
  * **Ungated secondary supersampling.** Every reflective fragment spawned N
    continuations, including the ~4% Fresnel sheen every PBR dielectric has. A
    plain glossy sphere with no mirror in the scene was paying four extra traced
    rays per pixel for 4% of its colour: 1.89s against aa=2's 0.96s, and
    slightly WORSE quality. Gated on the branch's share of the pixel
    (`ANALYTIC_AA_SECONDARY_MIN_ENERGY`) → 1.17s, now faster than aa=2. The
    lesson generalizes: the value of analytic coverage is that the expensive
    fallbacks fire only on the pixels that need them, so every fallback needs a
    per-fragment gate, not a per-scene one.

  * **Gating the N continuations on the FRAGMENT's total coverage.** Meant to
    stop a silhouette's several partial fragments each spawning N rays, which
    over-samples the rim relative to aa=2. It keys on the wrong quantity: in a
    dense mesh nearly EVERY fragment partially covers its pixel, because each
    triangle owns only a few of the samples. A "full coverage only" test
    therefore switched secondary sampling off almost everywhere and cost the
    glass config its entire refracted-image quality (0.127 -> 0.154, which is
    exactly the one-continuation number). The right predicate is per POSITION:
    one ray per sub-pixel continuation position the fragment actually covers,
    derived from its sample mask (`_sec_positions`). That is precisely what
    supersampling does -- one secondary ray per sub-pixel the primitive covers --
    and it restored the quality (0.125, the best measured) while keeping the rim
    saving.

A hypothesis that also proved wrong: that a splitting batch's 4-slot ray pool is
a penalty relative to supersampling. It is not — aa=2 has 4x the primaries at
ratio 1, analytic has 1x at ratio 4, so the slot budget and the tile count match.
What it does cost is `opaque_closest`, `opaque_prepass` and `mem_trim`, which the
split flag disables.

19.2a THE POOL-SIZING BUG — why it looked inherently slower
----------------------------------------------------------
Worth its own subsection: it made every timing in this document misleading until
it was found, and it took three attempts. `_split_pool_ratio` used
`max(REFRACT_INITIAL_POOL_RATIO, N)` = 4 slots per pixel. But the base of two
exists because a pixel can hold several splitting LAYERS -- a glass sphere has
two, its front and back sheets -- so with N continuations each, the requirement is
N per layer, i.e. 8. The pool overflowed on every tile, and an overflow
**discards the finished tile and re-renders it with half the primaries**:

    glass, 640x360x8, warm         wall    pool retries   resolve launches
      aa=2                        19.2s          1                4
      analytic, max(2,N) pool     45.4s          3               12   <- ~3x over
      analytic, base*N pool       21.5s          2                7
      + per-position gating       20.3s          1                4

The fix is `ratio = base * N`. Note how the symptom presents: no error, no
warning, just a wall time that makes the whole approach look wrong. `tracer.py`
keeps a `_WAVEFRONT_POOL_RETRIES` counter -- read it before concluding ANYTHING
about the cost of a splitting path.

19.3 Where it stands (320x180, mean L1 against a supersampled aa=4 reference)
----------------------------------------------------------------------------
    config      analytic@1     aa2   aliased   ratio   verdict
    mesh          0.204       0.225   0.376     0.91   beats aa2
    text          0.224       0.242   0.684     0.93   beats aa2
    shapes        0.307       0.301   0.604     1.02   within tolerance
    thin          0.061       0.303   0.346     0.20   beats aa2 5x
    trans         0.112       0.117   0.187     0.96   beats aa2
    shadow        0.159       0.169   0.262     0.94   beats aa2
    softshadow    0.161       0.166   0.250     0.97   beats aa2
    mirror        0.079       0.088   0.159     0.90   beats aa2
    spec          0.074       0.069   0.089     1.07   SHORT
    flat          0.289       0.271   0.486     1.07   SHORT
    glass         0.125       0.116   0.235     1.08   SHORT

This table is only meaningful while every quantity in the scene has an apparent
size INDEPENDENT of `anti_alias_level`. The reference is rendered by this same
engine at aa=4, so anything sized in internal render pixels rather than output
pixels differs between the reference and the arms, and the L1 it produces is a
geometry mismatch wearing an antialiasing costume. It moves the reference and
all three arms AT ONCE -- including `aliased`, which runs with analytic AA off
and therefore looks immune -- so a whole config drifting in lockstep is the
signature to look for, and the first thing to diff is the reference image, not
the feature under test.

This is not hypothetical: 255e67a dropped the `anti_alias_level` factor from the
bezier border width, which is consumed as `circuit_meta[_M_BORDER_W] *
pixel_size` with `pixel_size` built from `camera.screen_height` -- the INTERNAL
height. Borders became 1/aa thin, the aa=4 reference drew them at a quarter
width, and the four configs holding a bordered circuit read `flat` 1.263,
`glass` 0.464, against 0.289 and 0.125 here. `mesh`/`spec`/`mirror` are
triangle-only and never moved; the split between affected and unaffected configs
was exactly "contains a bordered bezier circuit". Fixed by scaling the packed
width by `_rt_projection_aa` (`camera.screen_height / output_screen_height`) --
the AA actually in force, which is 1 on the analytic route whatever
`anti_alias_level` was requested, and which is why the requested level is the
wrong thing to read. The test suite cannot catch this class of bug: it renders
at PREVIEW, where aa=1 makes internal and output pixels the same thing.

Eight of eleven beat aa=2 outright; the three that fall short do so by 7-9% and
are all the same class -- the CONTENT of a reflection, a refraction or a specular
lobe, where four sub-samples of a strongly minified image are simply four
samples, exactly as aa=2's are. They are far better than aliased in every case
(spec 0.074 vs 0.089, flat 0.289 vs 0.486, glass 0.127 vs 0.235). Absolute
magnitudes are small: 0.07-0.29 out of 255.

19.4 Speed — and why there IS a floor
-------------------------------------
The structural point, which no amount of tuning removes: analytic coverage
replaces the 4x PRIMARY cost, and cannot replace the SECONDARY cost. Matching
supersampling's quality inside a reflection or refraction means casting the same
number of rays into it; aa=2 casts one per sub-pixel, so analytic must too. Per
output pixel on the glass config:

                      primary rays    refracted rays
    aa=2                    4              ~8
    analytic, N=4           1              ~8

That scene's cost IS the refracted rays, so the three-quarters saved on primaries
is a small share of a small share, and the best available outcome is parity. It
measures 1.03-1.08x. Where the PRIMARY dominates -- meshes, text, 2D shapes, which
is nearly all Algan content -- it wins outright.

    traversal device time, glass, 640x360
      analytic, 1 continuation per fragment     296 ms   <- 2.3x faster than aa2
      supersampled aa=2                        688 ms
      analytic, 4 continuations per fragment   1070 ms   <- buys the quality

    render-bound wall clock, 1280x720, alternating arms in one process
      scene     aa=2      analytic @ aa=1
      meshes    1.46s        1.14s    1.27x faster, equal quality
      mirror    3.34s        3.32s    1.01x,        equal quality

So: faster on ordinary content, parity on reflective and refractive content. The
knob is `ANALYTIC_AA_SECONDARY_SAMPLES` -- 1 makes a refractive scene 2.3x faster
than aa=2 with a visibly worse refracted image, 4 matches aa=2's sampling density.

Do not read wall-clock ratios off the 320x180 configs in §19.3: at that size this
renderer sits at a floor dominated by CPU prep and video encode (the tiny-scene
floor), and the arms differ by less than that floor.

19.5 What would close the last three
------------------------------------
Nothing cheap. Each remaining config needs more samples of a minified secondary
image, which is the one thing that costs exactly what supersampling costs. The
honest options are (a) `ANALYTIC_AA_SECONDARY_SAMPLES = 8`, which helped `flat`
(0.279) but not `glass`, (b) filtered/mip-mapped secondary lookups, a separate
project (§7), or (c) accept a 7-9% shortfall on reflective, refractive and
sharply specular content in exchange for beating aa=2 everywhere else and being
faster overall.


================================================================================
20. ROUGHNESS-DRIVEN GLOSSY REFLECTIONS — 2026-07-29 (DISABLED)
================================================================================

Roughness-driven gloss is currently disabled by setting _GLOSSY_MIN_ROUGHNESS = 100
because it caused extreme speckling across frames, in the future it may be revisited
and enabled once it can be made stable across time.

...........

§7's list of what coverage cannot antialias had "specular highlights will crawl"
and "the image inside a reflection" on it. §17 closed the second by sampling the
reflected image at N sub-pixel positions. This closes something adjacent and
older: the reflected image was sampled N ways but always in the SAME direction,
because `roughness` never reached the bounce. A
`MeshStandardMaterial(roughness=0.18, metalness=0.75)` sphere showed a razor-thin
mirror arc of the text above it — correctly positioned, correctly tinted (the
metal lobe's F0 is the albedo), and hard as glass — while its direct highlight
was already a broad GGX blob. One material, described two contradictory ways.

The fix reuses the rays §17 already spawns: the N continuations that vary in
sub-pixel POSITION now vary in LOBE DIRECTION too.

20.1 What was actually wrong
----------------------------
`roughness` was authored, stored, read and discarded. `raster_first_shade` and
`raster_shadow_event_build` both did `reflectivity, _rough = _tri_extra_g(...)`;
`_scatter_impl` did not take it as an argument at all. Only `shading_taichi`'s
direct lobe ever saw it. So a rough metal and a mirror produced BYTE-IDENTICAL
reflections — measured, on the repro sphere: the reflected arc at roughness 0.18
and at 0.60 differed by nothing but the direct shading underneath it.

20.2 Which lobe — and why not the Monte Carlo one
-------------------------------------------------
There were two candidates and they are not close to each other.

  * The Monte Carlo megakernel already jitters its bounce:
    `rd_new = normalize(mirror + roughness * random_unit)`. Cheap, and the
    obvious thing to copy.
  * GGX / Trowbridge-Reitz half-vector sampling at `alpha = roughness^2`, which
    is what `shading_taichi._ggx_distribution` evaluates for the DIRECT
    highlight.

The second, because the first disagrees with the direct highlight the reflection
sits beside. At roughness 0.18 GGX puts the median microfacet 1.86° off the
normal (reflected deflection ~3.7°); the normal-perturbation lobe puts it at
~10°, a factor of 2.8, and the ratio is not even constant in roughness. A
reflection blurred 3x wider than the highlight on the same surface is two
materials in one shader. Scored against the closed-form GGX CDF (§20.5, Part A),
the MC lobe lands at KS 0.31–0.86 — it is a different distribution, not an
approximation of this one.

Cost is not the reason either way: both are a handful of transcendentals per tap.

20.3 Where the lobe lives, and where it deliberately does not
-------------------------------------------------------------
**Only at the primary hit, in `raster_first_shade`** — the same place, and the
same rays, as §17's split. All three of its reflection spawns take it (the glass
branch's reflected share, the pane / `split_refl` slot, and the mirror bounce);
`_scatter_impl`'s deeper bounces stay specular-perfect. §17.1's reason carries
over unchanged: the split happens ONCE, so the cost is N x the secondary
traversal rather than N^depth.

**A single continuation is never perturbed.** A fragment that takes the one-ray
path — `sec_aa == 1`, energy below `ANALYTIC_AA_SECONDARY_MIN_ENERGY`, or only
one covered sub-pixel position — keeps the exact mirror direction. This is the
whole discipline of the feature: one deterministic sample of a lobe is not a
blur, it is a mirror pointing the wrong way, and it would look worse than the
sharp reflection it replaced. Blurring needs several rays, so it lives where
several already exist.

**Transmission is not blurred.** Frosted glass is a separate lobe on a separate
branch, and the refracted spawn has no in-slot ray to keep coherent. Not built.

20.4 Cost: zero extra rays, zero extra pool slots
--------------------------------------------------
The taps are the ones `_sec_positions` already hands the fragment, so the spawn
count, the `1/sec_n` weights, the split flag and the pool sizing are all
untouched — measured identical to the pre-glossy build (§20.6). This was the
binding constraint: an analytic-AA reflective pixel already spawns up to
`ANALYTIC_AA_SECONDARY_SAMPLES` continuations and a dense mesh measures 6.10 per
covered pixel against a budget of 5, so there was no room for a lobe with a
sample budget of its own. Raising `_split_pool_ratio` is the wrong lever anyway
(§19.2a: the tile is `budget / ratio`, so the ratio scales the tile COUNT).

Reusing the same taps also DECORRELATES the two error sources: each tap now
differs from its neighbours in sub-pixel origin and in lobe direction at once.
The pairing is fixed (sub-pixel position `s` always draws lobe stratum `jtap`,
its ordinal among the positions this fragment covers), which is the desirable
arrangement rather than an accident — it is a Latin-hypercube pairing of two 1-D
stratifications, and it guarantees the four sub-pixels never share a lobe
stratum. The positional term is ±0.25 px against an angular term that reaches
tens of pixels, so the correlation cannot bias the result either way.

20.5 The per-pixel rotation is not a cosmetic dither
-----------------------------------------------------
`GLOSSY_INTERLEAVE` rotates each pixel's fan by a 4x4 Bayer index: the radial
stratum gets a Cranley-Patterson offset `(b + 0.5)/16` and the azimuth a
golden-angle multiple of the same `b` (irrational, so the two dimensions do not
correlate). It reads as an ordered dither, and it was expected to be a
cosmetic trade — four ghost images against speckle.

It is not. Without it every pixel samples the SAME four radial strata, so the
whole image is a 4-point quadrature of the lobe: measured KS 0.125 against the
analytic CDF, which is exactly the 1/(2*4) floor of a four-step empirical CDF,
at every roughness and at K=8 as well (0.063 = 1/16). It also TRUNCATES the
lobe: its outermost stratum sits at the 87.5th percentile where the rotated fan
reaches the 99.2nd, and GGX's tail is heavy. (The plain fan does measure ~18%
narrower, but do not read that as the truncation: the K=8 arm measures narrower
by the same margin for the staircase reason in §20.6. The truncation is a
provable property of the sampler; the measured indictment is the banding.)
With the rotation the same four rays per
pixel become a 64-point quadrature over a 4x4 block: KS 0.008, which IS the
1/(2*64) = 0.0078 floor for 64 points, i.e. the taps are an optimally
stratified quadrature of the analytic lobe rather than merely a close one.
(K=8: 0.004 against a 1/256 = 0.0039 floor.)

That is not luck, and it is the crisp reason to keep the rotation. The tap's
radial coordinate is `u = (j + (b + 0.5)/16) / K` for stratum `j < K` and Bayer
index `b < 16`, which is `(16j + b + 0.5) / 16K`; since `16j + b` runs over
0..16K-1 exactly once, the block's taps ARE the 16K quantiles `(i + 0.5)/16K` of
the lobe — the optimal stratified set, reached with no extra rays. The Bayer
index is not decorating the fan, it is COMPLETING its stratification; dropping
it throws away 15 of every 16 strata.

Verified for K = 2, 3, 4 and 8 (exact to the last bit), which matters because K
is `sec_n`, the number of sub-pixel positions the FRAGMENT covers, not the
setting. A silhouette fragment owning two positions still gets a perfectly
stratified 32-point sample of its lobe rather than an arbitrary pair.

Fixed in SCREEN space, so it is a function of the pixel and nothing else: the
same frame renders identically every time, and the pattern is stationary across
an animation. It cannot twinkle.

What it costs is visible as a texture in the transition band. With K taps a pixel
resolves a step edge into K+1 levels, so at the default K=4 a glossy gradient
is a 5-level ordered dither. That is inherent to four samples, not to the
rotation; `ANALYTIC_AA_SECONDARY_SAMPLES = 8` buys 9 levels through the existing,
already-sized mechanism.

20.6 Measured
-------------
`benchmarks/_glossy_ggx_check.py` is the ground truth, in two parts, because the
§19 gate cannot be one: every reflective config in `_aa_match_aa2.py` authors
roughness 0.0–0.05, and its aa=4 reference is rendered by this same engine — a
SHARP-reflection reference. It would have scored a correct glossy lobe as a
regression and a missing one as perfect.

**Part A — the tap set against the closed-form GGX CDF, no renderer involved.**
Kolmogorov–Smirnov distance between the taps' half-angle distribution and
`F(theta) = tan^2 / (a^2 + tan^2)`, over a 4x4 pixel block:

    roughness   GGX taps   plain fan   MC normal-perturbation   median half-angle
      0.05        0.008      0.125            0.861                  0.14°
      0.18        0.008      0.125            0.533                  1.86°
      0.35        0.008      0.125            0.315                  6.98°
      0.60        0.008      0.125            0.538                 19.80°
      1.00        0.008      0.125            0.562                 45.00°
    (K=8: 0.004 / 0.063 in the first two columns)

0.008 IS the 1/(2*64) floor for a 64-point set, so the taps are an optimally
stratified quadrature of the analytic lobe, not merely a close one. The MC
megakernel's lobe scores 0.32–0.86 — it is a different distribution, which is
what makes this test discriminating rather than self-confirming.

**Part B — rendered blur width, end to end.** A frame-filling flat mirror
reflecting a straight bright half-plane placed behind the camera; the 10–90%
rise of the reflected edge, swept over roughness.

    arm               width exponent   px/rad   spread %   dither RMS   banded %
    K=4 interleave         2.03          209       3.6        22.20       21.2
    K=4 plain fan          2.03          175       1.2         2.18       74.2
    K=8 interleave         2.02          173       2.4        14.53        9.7
    glossy off              n/a           17      87.5          —           —

    10-90% rise, px:  roughness  0.00   0.10   0.15   0.20   0.28   0.35
                 K=4 interleave  2.40  10.70  26.37  43.80  84.75 138.91
                     glossy off  2.40   2.40   2.40   2.40   2.40   2.40

The exponent is the model discriminator: GGX at `alpha = roughness^2` predicts
2.00 and a normal-perturbation lobe 1.00. All three glossy arms land at 2.02–2.03.
`px/rad` is the rendered width over the ANALYTIC lobe's own 10–90% deflection —
pure camera geometry, so it must not vary with roughness; it varies by 1–4%
across a 12x range in alpha, against 87.5% for the arm that ignores roughness.
The ~20% offset between the K=4 and K=8 columns is a measurement artifact (the
10–90 crossings are read off a 16- vs 32-step staircase, and the coarser one
biases them outward); Part A shows the two sample the identical distribution.

Two absolute goodness-of-fit metrics were tried first and neither discriminated,
so neither is reported: a fitted pixels-per-radian scale has a degenerate escape
at scale → 0, where every predicted curve collapses to a step and the arm that
ignores roughness scores BEST; and a width-normalised ESF shape L1 ends up
dominated by the 4x4 rotation's residual block wobble, scoring all three arms
equal to three decimals. The lobe's shape is established by Part A, exactly.

**Pool** (`benchmarks/_glossy_pool_check.py`, the repro sphere at 864x486):
2.587 continuation slots per covered pixel, BYTE-IDENTICAL with the lobe on and
off at every roughness, overflow flag 0, `_WAVEFRONT_POOL_RETRIES` 0. The lobe
redirects existing rays and spawns none, so §19.2a's failure mode is
structurally out of reach.

**Determinism** (`benchmarks/_glossy_determinism.py`): the same frame rendered
twice in one process is byte-identical with the lobe on. On moving content the
per-frame change is measured as max/median of the mean |delta| — a twinkling
frame shows up as a spike out of line with its neighbours. A spinning glossy
sphere scores 1.15 with the lobe against 1.56 sharp; an animated roughness ramp
2.23 against 4.41. The lobe LOWERS temporal spikiness on both, because the
quantity it blurs is one the sharp arm was resolving abruptly.

**Byte-identity**: `roughness = 0` output is byte-identical to the pre-change
build, and so is the whole repro sweep with `ALGAN_GLOSSY_REFLECTION=0`.

20.7 What this does not do
--------------------------
  * **Refraction stays sharp.** Frosted glass is a separate lobe on the
    transmitted branch. Not built.
  * **Deeper bounces stay sharp** — see §20.3. A mirror reflecting a rough metal
    shows that metal's reflection unblurred.
  * **Four taps is four taps.** A glossy transition resolves into K+1 levels, so
    at the default K=4 it is a 5-level ordered dither (RMS 22/255 measured on a
    black-to-white reflected edge, the worst case there is). High-contrast
    minified content — bright text reflected in a rough sphere — therefore
    speckles. `ANALYTIC_AA_SECONDARY_SAMPLES = 8` halves it (14.5) through the
    existing already-sized mechanism, and is the honest knob; there is no way to
    make four samples of a wide lobe smooth, and this is the same wall §19.5
    describes for the content of any minified secondary image.
  * **The classic wavefront primary path** gets none of this, like §17 before
    it: the lobe lives in `raster_first_shade`.


================================================================================
21. EXACT COVERAGE: WHAT IT FIXES, AND THE WALL IT HITS — 2026-08-12
================================================================================

Two defects were on the table. Circuit coverage did not take the boundary's
ANGLE into account, and triangle coverage was a count of eight sample points
rather than an analytic formula. The first is now fixed. The second turns out
not to be a defect that can be fixed in isolation, and the measurement showing
why is the most useful thing in this section.

21.1 The decomposition
----------------------
A fragment's coverage answers two questions that had been sharing one number:

    WHERE in the pixel is it   -> a set question; only a set answers it
    HOW MUCH of the pixel      -> a measure question; a real number answers it

`_coverage_density` now splits them. A fragment is `(cmsk, dens)`: the samples
it attenuates, and the fraction of each that it covers.

    eff = dens * sum(svis[s] for s in cmsk) / N
    for s in cmsk: svis[s] *= 1 - alpha * dens

This UNIFIED the two fragment kinds that previously needed separate branches at
five sites each. A mask whose popcount is the coverage gives `dens == 1` (every
triangle today); an areal fragment -- a circuit's SDF coverage, a sliver's
clipped area -- gives `cmsk == all` and `dens == cov`. Both come out
bit-identical to the branches they replaced, because `cov` is a multiple of 1/N
in the first case and the mask is already full in the second. The `areal` flag
is gone from both resolve walks.

21.2 Circuits: exact, angle-aware coverage (SHIPPED, default ON)
----------------------------------------------------------------
`clamp(d + 0.5, 0, 1)` is the exact area of (half-plane n pixel) for an
AXIS-ALIGNED boundary and for no other orientation -- it is the `b == 0` case of
the general formula. At 45 degrees it reports full coverage at `d = 0.5` where
the truth is `1 - (0.7071 - d)^2 = 0.957`, peaking at 0.043 of coverage.

`_halfplane_clip_area(n, d)` is that general formula in closed form, and
`_bezier_point_metrics` already had what it needs: the closest-point vector,
formed to get `min_dist_sq` and thrown away (ss4 predicted this exact use). It
is the local gradient of the SDF, so it turns the distance into a boundary LINE.
Applied to the outer silhouette, the unfilled band's nearer wall, and the
border's inner edge.

Measured against an aa=4 reference, whole-frame L1:

    config    box      exact     delta
    slant    0.1188   0.1093    +8.0%     long straight 24-degree edge
    border   0.0870   0.0850    +2.3%     the border's inner edge
    text     0.1081   0.1155    -6.8%     glyph-scale detail

The pattern is one boundary per pixel versus two. Linearizing to the NEAREST
segment models a 1px glyph stem as a half-plane and over-covers it; the box
filter's distance-based rounding happens to be closer there. The fix is a wedge
(square n H1 n H2, the two nearest segments), not a return to the box filter.

On the ss19 gate the exact form regresses NOTHING and improves four configs:
shapes 0.342 -> 0.331 (1.06x -> 1.03x of aa2), flat 0.398 -> 0.393, glass
0.153 -> 0.152, text 0.224 -> 0.223, the other seven identical. 8/11 beat aa=2 in
both arms; the trio that falls short (shapes, flat, glass) is the same either
way, and differs from the trio ss19.3 recorded (spec, flat, glass) through drift
since that measurement, not through this change.

**A HARNESS BIAS THAT INVERTS THE RESULT, and the reason the first run said the
opposite.** The aa=4 reference dilates a filled circuit by `0.6 * pixel_size` at
its own supersampled pixel = 0.15 output px; the analytic arms dilate by
`ANALYTIC_AA_BEZ_MIN_HALF_WIDTH = 0.3` output px. Every arm is therefore scored
against a reference whose silhouette sits 0.15 px inside it -- and a SHARPER
filter amplifies a fixed positional error in proportion to its slope, so the
exact form scores worse purely for being more faithful. At the shipped 0.3,
`slant` reads -0.6%; at a matched 0.15 it reads +8.0%. Match the dilation before
reading any circuit-coverage measurement here. The same comparison says the 0.3
default costs about TWICE as much L1 as the choice of filter does (slant 0.26 ->
0.12 on matching it alone) -- worth revisiting, but it is an appearance decision
about stroke weight, not an L1 one.

21.3 Triangles: exact area does NOT work (MEASURED, default OFF)
-----------------------------------------------------------------
Implemented behind `ALGAN_ANALYTIC_AA_EXACT_TRI`, dispatching on how many edges
reach the pixel: none -> 1, one -> `_halfplane_clip_area`, two or more ->
`_pixel_clip_area`. It is catastrophic, and not because of a bug:

    config   notches box -> exact    ink          whole-frame L1
    tri          13 ->  5920     1.000 -> 0.891      -730%
    seam          5 ->  6996     1.000 -> 0.953      -655%
    trans         7 ->  6866     1.000 -> 0.776     -1418%
    thin          0 ->     0     0.855 -> 0.530      -157%

A fragment's CLAIM and its OCCLUSION must be the same quantity, or the pixel
stops summing to one. With eight point samples both are `|M|/N`: quantized to
eighths, but consistent, and a tiling sums to exactly 1. That consistency was
the sample count's real job -- it was never only an estimate standing in for an
analytic formula, which is the framing this work started from.

Take the magnitude from the exact area and the two differ by up to 1/N.
Reconciling them needs `dens = A*N/|M|`, which EXCEEDS 1 whenever `A*N > |M|` --
the normal state of a boundary pixel, not an edge case. Clamping it (a sample
cannot be covered twice) can only ever lose energy, so every boundary fragment
under-claims, a shared edge no longer sums to 1, and the background leaks
through as interior notches. On a dense mesh every pixel is a boundary pixel for
several triangles, so it compounds into the ink figures above.

21.4 What looked like it would work: cells, not points  [SUPERSEDED by ss21.8]
--------------------------------------------------------------------------------
The reasoning below is left as written because the implementation followed it,
but its central claim -- that cells hold all three properties at once -- is
WRONG, and ss21.8 records the measurement that shows why. A cell is not ATOMIC:
two triangles can split one, and fractional coverage cannot say that their parts
are disjoint. Read ss21.8 before acting on this section.

Stop sampling the pixel at POINTS. Take a small number of CELLS tiling it, each
carrying the exact clipped area of the fragment within that cell:

    c_s = area(fragment n cell_s) / area(cell_s)

All three properties hold at once, which no point-set representation can manage:
the claim sums to `A` exactly, no cell can exceed 1, and a tiling still
partitions. `_pixel_clip_area` provably sums over a tiling already (property 3
of `benchmarks/_aa_clip_area_check.py`), so this also RETIRES the top-left fill
rule of ss15 -- whose entire job is to fake, for binary point ownership, the
partition that exact areas have for free.

Four quadrant cells look right: `svis` halves from 8 floats to 4 in an
occupancy-bound resolve, and each cell area is the same `_halfplane_clip_area`
with a shifted, scaled `d`. It costs the centroid depth evaluation of ss15 and
`_sec_positions` some resolution, which have to be re-measured. This is a
redesign of the mask, not an increment on it.

21.5 Validation
---------------
Everything in ss21 is byte-identical to the pre-change renderer except the
circuit filter itself: with `ALGAN_ANALYTIC_AA_EXACT=0` the fast suite's render
hashes identically (SHA256) to its pre-change baseline, which covers the
`_coverage_density` rewrite, the new `_bezier_point_metrics` arity, the
`_sliver_mode` re-encoding and the triangle machinery at once.

`_halfplane_clip_area` itself is checked standalone by
`benchmarks/_aa_clip_area_check.py`: agreement with an f64 polygon clip to
7.3e-08 over 1917 (normal, offset) pairs, agreement with `_pixel_clip_area` to
3.6e-07 on single-edge triangles, bit-exact collapse to the box filter when
axis-aligned, and `A(n,d) + A(-n,-d) == 1` bit-exactly on all 3690 boundaries
tested -- the seam property in its smallest form.

NOTE, both gates were DEAD when this work started and were repaired as part of
it: `_analytic_aa_bez_check.py` and `_aa_match_aa2.py` both imported
`render_to_file`/`RenderSettings`, removed from the public API, so neither had
run since. They now use `Scene(video_settings=...)` / `save_video`, and carry
`--exact-ab` and `--box` respectively for the arms above.

21.6 The two-segment boundary model — BUILT, NOT CORRECT (default OFF)
-----------------------------------------------------------------------
ss21.2 diagnosed the glyph regression as the half-plane MODEL rather than the
formula, and named the fix: take the two nearest segments. That is implemented
behind `ALGAN_ANALYTIC_AA_BEZ_WEDGE` and does not work yet. Dilation matched,
against the single half-plane:

    config    1 segment   2 segments
    text        0.1155      0.1231
    slant       0.1093      0.2467      <- a plain square, all convex corners
    border      0.0850      0.1916

Three bugs were found and fixed on the way, all worth keeping:

  * `a1 + a2 - 1` holds only for ANTIPARALLEL half-planes, whose complements are
    disjoint. For CO-parallel ones the complements are nested and the answer is
    `min(a1, a2)`; conflating them is wrong by up to 0.42.
  * The wedge apex sits `(|d1|+|d2|)/|cross|` away, which for an unrelated
    segment elsewhere in the shape is tens of pixels. Truncating its rays at any
    fixed length then misses the pixel and reports zero where the answer is "the
    first half-plane, unchanged". Short-circuit on trivial containment FIRST;
    that also bounds the apex to ~7px, which keeps the f32 shoelace accurate.
  * The signed distance to the second line is MINUS the projection of the
    closest-point vector onto its normal (the closest point is reached by
    stepping BACK along the normal). Without the negation every second
    half-plane is flipped, which turns a convex corner inside out.

What remains unsolved is ORIENTATION, and it is structural rather than a bug. A
second segment has no inward side of its own: the crossing parity is a property
of the QUERY, so it orients only the NEAREST segment. Every other segment's
normal has to come from the contour's handedness, calibrated as
`sign(cross(dir1, n1))` -- which needs `n1` to be segment 1's true normal. At a
CORNER the closest point on segment 1 is its endpoint, so `n1` points at the
vertex instead, the calibration sign is arbitrary, and the second half-plane
flips. Corners are exactly the pixels the model exists to fix, so restricting it
to interior closest-points is circular.

A correct version needs a per-segment inward side that does not depend on the
query -- most likely carried on the edge record at flatten time, where the
contour's traversal is known, rather than recovered in the kernel. Note that
even-odd fill means a hole's contour may be wound either way, so "left of the
direction" is not sufficient on its own.

`_two_halfplane_area` itself is sound and validated standalone (max error 0.0115
against an f64 polygon clip, exact for the far-segment case), as is the
second-nearest tracking in `_bezier_point_metrics`. Only the orientation is
missing.

21.7 Cell primitives for triangle coverage — VALIDATED  [wired in ss21.8]
-----------------------------------------------------------------------
The ss21.4 replacement for point sampling. Two primitives exist and are checked
standalone; nothing in the pipeline consumes them yet.

These primitives are sound and stay in the tree; what ss21.8 shows is that the
REPRESENTATION built on them does not hold, not that the areas are wrong.

`_cell_clip_area(vx, vy, k)` needs no clipping code of its own: a cell is the
pixel square translated to its centre and scaled by two, so the query maps
straight back onto `_pixel_clip_area`, which is already exact, already
validated, and already known to sum over a tiling.

`_pixel_clip_centroid(vx, vy)` carries the two first moments through the same
clamped boundary integral. It replaces the centroid-of-owned-samples that the
depth evaluation of ss15 depends on. Note why an area-weighted average of CELL
CENTRES will not do: it can fall outside the triangle, which puts the silhouette
mis-sorting of ss15 straight back. The centroid of (triangle n pixel) cannot --
an intersection of convex sets is convex.

Measured (`scratchpad/probe_cells.py`, to be folded into
`benchmarks/_aa_clip_area_check.py`):

    cell area vs brute force        max 0.0008   (240 triangles)
    cells sum to the whole area     max 1.19e-07 <- claim == total, exactly
    per-cell partition over a tiling max 0.0002  (200 fans)
    centroid vs brute force         max 0.0007
    centroid outside the triangle   0 of 130

The first and third are the two properties ss21.3 proved a point set cannot hold
at the same time.

STILL TO DO, and it is the bulk of the change:
  * pack 4 cells x 8 bits into the existing `frag_msk` int32 lane (no new
    per-fragment storage; the flag bits at `_AA_FLAG_SHIFT` need rehoming, and
    `_AA_SLIVER_BIT` may simply disappear -- a sub-sample sliver is a
    point-sampling artefact, and a cell records its area like anything else)
  * `_ss_pixel` emits cell areas + the clipped centroid instead of the
    fixed-point sample mask; the top-left fill rule can then retire, since exact
    areas summing over a tiling is the property it exists to fake
  * `_coverage_density` and both resolve walks move from 8 samples to 4 cells;
    `svis` halves, which the occupancy-bound resolve should like
  * `_sec_positions` and the shadow mask already think in 4 positions
  * the host prefill (`AA_MASK_ALL`) becomes an all-cells-full packed value
  * circuits emit per-cell areas too, so the two geometry types interoperate

21.8 Cells, built and measured — and why ss21.4 was wrong
-----------------------------------------------------------
Implemented behind `ALGAN_ANALYTIC_AA_EXACT_TRI` (which now selects the cell
representation): four exact per-cell areas packed 8 bits each into the existing
`frag_msk` lane, the clipped centroid replacing the sample centroid for depth,
and the resolve reading one coverage value per slot. Each cell is written into
TWO of the eight resolve slots, which keeps every loop bound and the 1/N
normalisation untouched -- the point path came out bit-identical through the
whole refactor, verified by SHA256 on the fast render.

    config   notches: points / cells      ink: points / cells
    tri            13 /   6942            1.000 / 0.946
    seam            5 /   8308            1.000 / 0.972
    trans           7 /   6734            1.000 / 0.897
    thin            0 /      0            0.855 / 0.999

WHY ss21.4 WAS WRONG. It claimed cells give exact magnitude, capacity AND the
partition. The first two hold. The third does not, and the reason is that a cell
is not ATOMIC. Two triangles that split one cell own DISJOINT parts of it;
fractional per-cell coverage cannot express that, so the resolve composites them
as independent and gets 1 - a*b (0.75 for an even split) where the answer is 1.
That is precisely the coverage-as-alpha lattice of ss5, reproduced one level
down. A sparse mesh hides it -- two cubes measured 13 notches -- and a
subdivided one, with many partial fragments per cell, does not.

Front-to-back accumulation, `min(coverage, remaining)` instead of a product,
fixes the tiling case exactly (notches to 0 on the cube scene) and then
OVER-claims wherever fragments genuinely overlap rather than tile (ink 1.009).
Depth does not separate the two cases: each fragment's depth is measured at its
own clipped centroid, and for two triangles sharing an edge those sit on
opposite sides of it, so the distances do not tie. Subdividing into more cells
shrinks the error only by converging back towards atomic units -- that is, back
to sample points.

THE GENERAL STATEMENT, which now has two independent implementations behind it:
a fragment's CLAIM and its OCCLUSION must be the same quantity, and only ATOMIC
sub-pixel ownership guarantees that. Eight sample points under a top-left fill
rule are atomic, which is why the shipped representation is consistent. The
magnitude quantizing to eighths is the PRICE of that consistency, not an
oversight to be engineered away, and any replacement has to supply atomicity
first and exactness second.

One real gain is stranded here: cells fix sub-pixel erosion outright (`thin` ink
0.855 -> 0.999), because a sliver's area lands in a cell instead of being dropped
for containing no sample. It is not separable -- harvesting it alone hands a
sliver a claim it cannot occlude, which is the halo the sliver policies already
measured in ss16.2. A representation that is atomic AND finite-area (per-cell
ownership, one owner per cell chosen by a fill rule, with the owner carrying the
cell's exact area) is the shape of a fourth attempt; it would need the fill rule
kept, not retired as ss21.4 supposed.

21.9 Surface-grouped coverage — the third attempt (INCOMPLETE, default OFF)
-----------------------------------------------------------------------------
ss21.3 and ss21.8 both end at the same wall. This section starts from the
question that reframes it: analytic anti-aliasing is a solved problem in 2D, so
what is different here?

THE ANSWER IS THE UNIT OF WORK, NOT THE FORMULA. FreeType, Skia, stb_truetype,
font-rs and Pathfinder all compute exact coverage, and they all rasterize one
CLOSED PATH at a time -- usually by signed-area accumulation along a scanline.
There is no occlusion inside a path, so there is no partition problem to solve.
Triangulate that same glyph and rasterize the triangles separately and the
problem appears immediately; they simply never do that.

For 3D meshes nobody computes exact analytic coverage. What ships instead:

  * hardware MSAA -- 4/8/16 atomic sample points, a coverage bitmask, a fill
    rule. This is our design; our eight fixed-point samples ARE 8x MSAA in
    software.
  * Carpenter's A-buffer (1984), the origin of the whole family -- a 4x8
    subpixel BITMASK.
  * TAA -- one jittered sample per pixel per frame, accumulated over time.
  * FXAA / SMAA -- edge detection and blur.

Every one is atomic samples or temporal accumulation. The point representation
is not a compromise we failed to escape; it is the industry answer, and the
eighth quantization is its documented price.

What we do differently, and what is worth fixing, is that coverage is computed
PER TRIANGLE and a surface then has to be reassembled from the pieces. The fill
rule, the partition and the per-sample transmittance all exist to perform that
reassembly. Our bezier circuits never pay it -- they SDF a whole outline, the
2D-rasterizer situation -- which is why the circuit side has always been in
better shape than the triangle side.

THE DESIGN. Give each fragment its source primitive, and then:

    within one surface, exact clipped areas ADD      (an interior edge stops
                                                      existing; summing over a
                                                      tiling is exact, which is
                                                      property 3 of
                                                      _aa_clip_area_check)
    between surfaces, coverage COMPOSITES            (what a 2D renderer does
                                                      between shapes)

with one refinement: a closed mesh's two SHEETS must be kept apart. Front and
back of a sphere both cover a silhouette pixel and cover the SAME part of it, so
their areas must not add -- the object's coverage is its larger sheet. The
facing bit separates them exactly and for free, which is what ss15 built it for.

WHAT IS BUILT (all of it gated, default off):

  * `tri_obj`, a per-triangle source-primitive id, rebuilt at merge from the
    concatenation order of `_geom` and threaded to BOTH resolve kernels. ss18
    deleted the original; this is the piece that has to exist before any
    surface-grouped scheme can be attempted, and it exists again.
  * `_ss_pixel` in this mode emits the exact clipped area as the fragment's
    coverage and puts ONLY the facing bit in the mask lane.
  * The resolve replaces the whole `svis` array with four scalars -- `rem`
    (transmittance ahead of the current object), `sheet_cov`, `obj_cov` and
    `obj_absorb` -- plus the current object and facing. Cheaper than the array
    it replaces, not more expensive.
  * The depth walk flushes an object when the id changes, and starts a new sheet
    when the facing changes within one object. A fragment contributes
    `max(obj_cov, sheet_cov + area) - max(obj_cov, sheet_cov)`, which sums
    exactly within a sheet and yields zero for a hidden back sheet.

STATUS: THE ACCOUNTING IS WRONG AND THE FAULT IS NOT ISOLATED.

    mesh scene: L1 1.18 against 0.16 aliased, ink 0.784, 1531 interior notches

Six fix-and-measure cycles each corrected a real bug without moving the result
materially. The bugs are worth recording because three of them are traps rather
than typos:

  * `tri_mat_id` is NOT an object id. It identifies the material, so two objects
    sharing one material merge into a single "surface" and the max() rule throws
    one of them away. This was used first as a zero-plumbing proxy; it is not
    one. Use `tri_obj`.
  * An earlier revision packed four per-cell areas into the mask lane. Bit 16 of
    that packing is cell 2's low bit and is also `_AA_BACKFACE_BIT`, so the
    sheet grouping read a facing bit that was effectively random.
  * The z-winner's in-kernel default and the host prefill both mean "fully
    covered" and were set to all-ones. All-ones sets bit 16, so every fragment
    that does not write its own mask -- every circuit, every z-winner -- read as
    back-facing.

THE NEXT STEP IS INSTRUMENTATION, NOT ANOTHER GUESS. Dump `sid`, facing, `cov`,
`contrib`, `rem`, `obj_absorb` per fragment for one known pixel and compare
against a walk computed by hand. Six rounds of inference between renders found
three real bugs and did not find this one; the remaining candidates (run
boundaries, the z-winner's position in the walk, interaction with the reflection
and pool paths that consume `weight`) are all cheap to eliminate from a dump and
expensive to eliminate by guessing.


21.10 CURRENT STATE AND NEXT STEPS — 2026-08-12
-------------------------------------------------
WHAT SHIPS. Circuit coverage is the exact angle-aware half-plane area
(`ALGAN_ANALYTIC_AA_EXACT`, default ON). The resolve reads every fragment
through one `_coverage_slots` call, with the `areal` branch gone. Everything
else is unchanged, and the whole of ss21 is byte-identical to the pre-change
renderer with that one flag off -- verified by SHA256 on the fast render, which
covers the resolve rewrite, a `@ti.func` arity change, a template re-encoding
and the triangle machinery in one shot.

WHAT IS GATED OFF, and none of it should be turned on without reading its
section first:

    ALGAN_ANALYTIC_AA_BEZ_WEDGE=1   two-segment circuit boundary   ss21.6
    ALGAN_ANALYTIC_AA_EXACT_TRI=1   surface-grouped triangles      ss21.9

THE RULE THAT GOVERNS ALL OF IT, now with three implementations behind it: a
fragment's CLAIM and its OCCLUSION must be the same quantity, and only ATOMIC
sub-pixel ownership guarantees that. Exact area breaks it directly. Cells look
like they escape it and do not, because a cell is not atomic. Surface grouping
is the one formulation that can hold it -- interior edges stop existing rather
than being partitioned -- which is why it is the attempt left standing.

NEXT STEPS, in the order I would take them:

  1. FINISH ss21.9 with a per-fragment dump. The plumbing is done and the theory
     is sound; what is left is a bug. This is the highest-value item because it
     removes per-triangle quantization from every silhouette in every scene.
  2. THE CIRCUIT WEDGE (ss21.6) needs a per-segment inward side that does not
     depend on the query. Compute it at flatten time, where the contour's
     traversal is known, and carry it on the edge record -- `edges_2d` is
     already 5 wide with column 4 holding `border_visible`, so this needs a
     sixth column or a sign packed into an existing one. Note that even-odd fill
     means a hole's contour may be wound either way, so "left of the direction"
     is not sufficient on its own; the parity just off the edge midpoint is.
  3. RE-EXAMINE `ANALYTIC_AA_BEZ_MIN_HALF_WIDTH`. At the shipped 0.3 it dilates
     a filled circuit by twice what the aa=4 reference does, and that costs
     about TWICE the L1 that the choice of AA filter does (ss21.2). It is an
     appearance decision about stroke weight, not an L1 one, so it wants a human
     looking at rendered Tex rather than a number.
  4. `shapes_and_timeline` in tests/full_renders was ALREADY stale at HEAD when
     this work started -- its render differs from its baseline deterministically
     (max 170) on unmodified code. It was deliberately left failing rather than
     re-baselined, because doing so would bake in an unreviewed change that is
     not part of this work.

TOOLING ADDED, and the reason the last third of this work went faster than the
first two thirds:

  * `benchmarks/_aa_iter.py` -- ONE frame per arm via `Scene.save_frame`, scored
    against cached supersampled references. Five scenes, each isolating one
    thing the model must get right (slant, stem, corner, glyph, mesh), reporting
    L1, ink, interior notches and max deviation. About twelve seconds a render
    against fifteen minutes for the video gates. Build the references once with
    `--ref`; arms are selected by environment variable so switching between them
    never edits code or invalidates the kernel cache.
  * `benchmarks/_aa_cell_check.py`, `benchmarks/_aa_wedge_check.py` -- standalone
    validation of the area primitives against an f64 polygon clip, in the manner
    of `_aa_clip_area_check.py`. Both caught real bugs that a render-level A/B
    would only have shown as a worse number. VALIDATE A PRIMITIVE STANDALONE
    BEFORE WIRING IT: skipping that step for `_two_halfplane_area` cost two
    render cycles to rediscover what a fifteen-second harness said immediately.

A NOTE ON THE GATES, because it cost real time here: `_analytic_aa_bez_check.py`
and `_aa_match_aa2.py` were both DEAD when this work started, importing
`render_to_file` and `RenderSettings` which the public API no longer has, so
neither had run since that change. `_aa_clip_area_check.py` was separately
broken by ruff's `from __future__ import annotations`, which turns Taichi kernel
annotations into strings (`benchmarks/*` now carries an I002 ignore so that
cannot recur). All three are repaired. `benchmarks/_rt2_i64_atomic_smoke.py` has
the same future-import breakage and has not been touched.
