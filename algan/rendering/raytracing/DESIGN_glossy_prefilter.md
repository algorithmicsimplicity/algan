# Split-sum glossy reflection (prefiltered reflection buffer)

Status: implemented in `glossy_prefilter_taichi.py` and the deterministic
renderer's frame-finalization path. This document records the design implementing
`benchmarks/renderer_audit/REPORT.md` §4.5.1. Read §4.5 and §4.5.1 first — they
are the measurement this is answering, and they name the two halves:

* the **DFG term** makes the reflected *energy* analytic, replacing
  `_mirror_share`'s throttle heuristic;
* a **prefiltered radiance** makes the reflected *shape* right, built by a
  screen-space reflection filter rather than by more taps.

Both `glossy_reflection` and `glossy_prefilter` default to `True` in the current
settings source. Set these public controls with `SETTINGS.raytracing.set(...)`;
low-level tuning such as the maximum pyramid level is experimental.
The old opt-in rollout described below is historical: eligible deterministic
reflections now use this route by default. Turning the glossy route off avoids
its additional buffers and gated kernels. This is a screen-space reflection
pyramid, not texture mipmapping and not the path tracer's rough-glass model.

## 1. What was wrong, in one paragraph

A rough metal reflects 4.7% of what it should because a single mirror ray is
throttled to the GGX CDF mass inside a narrow cone (`_mirror_share`) and the
remainder falls back to a metal's own shading, which has no diffuse term — so it
falls back to the 0.01 ambient fill. Turning `glossy_reflection` on spends the
existing `ANALYTIC_AA_SECONDARY_SAMPLES` taps over the lobe, which fixes the
energy (0.555 against the path tracer's 0.523) and gets the shape wrong in a way
that moves under half a pixel of camera motion by 320× the control. More taps
cannot fix it: the artefact is minification aliasing, and `1/sqrt(k)` is the
wrong lever against it.

## 2. The two halves

### 2.1 DFG — the environment BRDF

```
∫ L(l) f(l,v) dl  ≈  [ ∫ L(l) D(l) dl ] · [ ∫ f(l,v) dl ]
                      prefiltered radiance   BRDF integral (DFG)
```

The second factor is the lobe's *directional albedo*: a 2-D function of
`(n·v, roughness)` that costs no rays. Algan uses Karis's analytic fit
(`EnvBRDFApprox`, Karis 2014) rather than a LUT, because a LUT would be another
texture to pack, upload and address inside the shade kernel for a two-instruction
polynomial:

```
r  = roughness * (-1, -0.0275, -0.572, 0.022) + (1, 0.0425, 1.04, -0.04)
a  = min(r.x * r.x, exp2(-9.28 * NoV)) * r.x + r.y
A  = -1.04 * a + r.z
B  =  1.04 * a + r.w
E  = F0 * A + B          # per channel in F0
```

`F0` is the material's normal-incidence reflectance — `mix(dielectric_f0,
albedo, metalness)`, the same blend `_material_reflectance` performs before
Schlick's tail. `E` is what a reflection branch carries, in place of
`R * _mirror_share(roughness)`.

Sanity checks the implementation must hold to (they are the unit test):

* at `roughness = 0`, `E ≈ F0` at normal incidence and `E ≈ 1` at grazing —
  i.e. it degenerates to Schlick, which is why the mirror path can keep using
  Schlick and the two agree across the roughness threshold;
* `E` is monotonically decreasing in roughness at fixed `NoV` for a metal
  (`F0 ≈ 1`), and never exceeds 1 nor drops below 0;
* it is a *directional albedo*, so `1 - E` is what the surface keeps and the
  local shading term must use `E`, not the Schlick `R`, or the pixel gains or
  loses energy.

### 2.2 Prefiltered radiance — the first-bounce reflection buffer

One ray per glossy pixel, in the dominant (mirror) direction, with **throughput
1**, accumulating into a *second* per-pixel buffer `B`. The energy `W = weight *
alpha * E` is factored out and kept per pixel. After the frame's rays drain, the
composite is

```
final = finalize( A + W · prefilter(B) )
```

`prefilter(B)` is a mip pyramid of `B` sampled trilinearly at the level whose
box width matches the lobe's screen footprint. Factoring `W` out of the blur is
the point: blurring `W·L` instead bleeds a reflector's energy across its own
silhouette onto the background.

Deterministic by construction — the ray direction is a smooth function of
position, so nothing crawls, and there is no dither pattern to reconstruct.

**Scope: one prefiltered glossy event per pixel.** The first sheet in the walk
that qualifies takes it; every later reflective sheet in the same pixel, and
every deeper bounce, keeps the existing `_mirror_share` throttle. This is what
the separation can represent (one `W`, one radius, one `B` per pixel), and it is
the event that matters.

**CORRECTED 2026-08-31: "later sheet" means a later SURFACE, and the claim is
shared by the §4.4 band that made it.** A band's siblings are one surface's
coverage of the pixel, subdivided so each shades with its own normal
(`DESIGN_sheet_resolve.md` §4.4/§10.5), and §4.4's contract is that the
subdivision changes nothing the band commits. As first written this rule broke
it, in the one place the throttle and the split-sum disagree most: at roughness
0.35 the throttle is ~3% of Schlick where `E` is the lobe's whole directional
albedo, and §2.1's last sanity check says the local term must use `E` or the
pixel gains energy. A crease pixel's far sibling therefore kept energy the
interior of the same facet spends on its reflection — and since `E` is
per-channel and a metal's F0 is its albedo, a red metal gained it in RED. On
`tests/fast`'s Icosahedron that was a brighter, redder line down every interior
edge: the crease row `(207, 106, 94)` between facets at `(181, 103, 91)` and
`(187, 106, 94)`, i.e. +21 in red against +1 in green, outside the interval a
blend of the two facets can reach.

The claim is now the band's: every sibling shades with `E`, and their energies
SUM into the one row they share — one ray, in the claiming sibling's mirror
direction, at the weight the unsplit band would have spawned. The direction and
radius stay the claiming sibling's; a lobe wide enough to be worth prefiltering
spans the crease either way. Pixels with no subdivided band are untouched
(measured: on the fast scene the change moves 9,122 pixel-frames, all inside
the Icosahedron's screen box, worst 51, and nothing else in the frame).
`test_a_creases_siblings_share_the_pixels_prefiltered_claim` pins it: interior
local maxima on a lit rough-metal Icosahedron, 72 before and 0 after.

**CORRECTED 2026-10-20: the event can be claimed in the drain.** As written,
only a PRIMARY hit could claim it, so a rough reflector that a pixel sees in a
mirror or through glass got the throttle. For a metal, the throttle does not
give a softer reflection; it removes the reflection. The withheld lobe falls to
local shading, and for a metal that is the 0.01 ambient (§1). A satin gold cone
(roughness 0.32) standing on a perfect mirror floor reflected at about 1/18 of
its own brightness in linear light, olive where it is gold.
`test_a_rough_metal_seen_in_a_mirror_reflects_as_itself` uses a uniform
environment and a white metal mirror, where the exact answer is 1. Its
reflection / panel ratio was 0.064 per channel.

Now a pool ray that is not part of a glossy reflection, meeting a rough opaque
reflector with bounces left, makes the resolve's substitution (`R = E`). It
adds `weight * alpha * E` to the pixel's `W`. If the resolve did not claim the
pixel, the first such ray claims it, as the resolve would have:

* it spawns the one glossy ray (mirror direction, throughput 1, glossy row);
* it records the radius and `d_p`, the camera path length to the hit, which
  through a planar mirror is exactly the unfolded distance §3 needs;
* the hit itself carries on as the pass-through.

Two kinds of ray join the claim instead of making their own: rays that arrive
after it, and drain hits on a pixel the resolve already claimed. Their energy
sums into `W`, and the one ray carries the radiance. This is the band rule
above, applied to the drain:

* A mirror pixel's secondary taps are one reflection, seen from sub-pixel
  offsets.
* At a silhouette, the primary can claim the event while the uncovered part
  of the pixel sees the same object in the floor. That is the same surface
  twice. Left throttled, that share drew a dark dotted line along the contact
  between an object and its reflection.

Rays inside a glossy reflection still keep the throttle, because their hits
are the radiance the event prefilters.

Measured:

* the test scene's ratio went from 0.064 to 1.000;
* the reported cone's reflection / object ratio went from 0.31 to 0.96 in
  linear light (path tracer, 16 spp: 0.93 / 0.87 / 0.76 per channel).

**CORRECTED 2026-10-20: the event goes to the sheet reflecting the most light,
not the first.** First-come made the claim snap. A faint reflective sheet in
front of a rough mirror (a title layer fading out over chrome, a translucent
label) took the event, and the mirror behind it fell back to the throttle's
sharp, partial reflection. The frame the faint sheet's weight dropped under
`min_weight`, or it despawned, the mirror's reflection blurred all at once.

The resolve walks front to back and cannot know in advance which sheet
reflects most, so the claim can now change hands:

* The first qualifying sheet claims, as before, and its band shares the claim
  (the 2026-08-31 rule).
* A later qualifying sheet outside that band takes the claim when its
  `weight * alpha * E` exceeds the band's summed `W` (largest channel; ties
  stay with the earlier sheet). It is weighed with `E` on both sides.
* The band that loses the claim becomes what it would have been had it never
  claimed: its reflection leaves as one sharp pool ray into the pixel's own
  row at `W * _mirror_share(roughness)`, and the rest of `W`, already taken
  out of its local term, goes back in, at the colors its sheets shaded with.
  `E` stands in for its Schlick term, which is the one difference left.
* The glossy ray is spawned once the walk ends, from registers the claim
  fills (origin, direction, `d_p`, bounces, roughness, `m`), so a takeover
  needs no pool slot back.

The handover now happens where the two reflect equally, so what changes hands
is the weaker of two equal shares. Sweeping a rough-metal veil's opacity over
a rough mirror (0, 0.01, 0.02, 0.05, 0.1, 0.3, 0.45, 0.5, 0.55) moves the
frame in proportion to the step: 2 sRGB levels from 0 to 0.01, at most 9 for
each 0.05 step, and no jump anywhere. §3's coverage-weighted distance is the
other half of that.

On `tests/fast` it moves 44 pixel-frames by more than 2, in three frames,
all on the Icosahedron's silhouette: mostly less red, by up to 39, with green
and blue within 7, and a few pixels up to 11 redder. That reads as the §2.2
seam signature going away (inferred from the colors, not from a dump): a
larger sheet of the solid there had kept its lobe as local red under the
first-come rule, and now takes the event. On the 28 changed pixels of the
left and right silhouettes, measured against a blend of the facet and the
background, the mean red error fell from 28 to 13 levels; 22 moved closer.
`test_a_faint_veil_does_not_change_the_mirror_behind_it` pins the
veil case.

## 3. Blur radius

The lobe's median microfacet half-angle is `atan(alpha)` with `alpha =
roughness²`; a reflection deflects by twice the normal's tilt, so the reflected
lobe's angular scale is

```
sigma_angle = 2 * alpha
```

Its screen footprint depends on how far the reflected content is. Unfolding the
reflection, the reflected surface sits `d_r` beyond a primary hit `d_p` from the
eye, so a patch of angular size `sigma_angle` at the reflected surface is seen
at total distance `d_p + d_r`:

```
k        = d_r / (d_p + d_r)            # 0 at contact, 1 for a far reflection
sigma_px = k * sigma_angle / theta_px
```

`theta_px` is a pixel's angular size, computed in-kernel from the ray-generation
basis (`|pixel_basis_x| / |screen_point_of_this_pixel - cam_origin|`; for an
orthographic camera there is no angular size and the world pixel width
`|pixel_basis_x|` is used directly against `d_r * sigma_angle`).

`k` is what gives **contact hardening**: a reflection touching its reflector is
sharp, a distant one is fully blurred.

`d_r` is not one distance when the reflection passes a translucent surface:
part of it ends there and the rest goes on. So `k` is the mean of the cone
factor over what the glossy ray meets, each surface weighted by the share of
the ray it covers (its opacity times what earlier surfaces let through), and a
miss counts as 1. An opaque first hit covers all of it, which is the first
hit's `k` exactly. Recording only the nearest hit, as the route first did, let
a faint or fully transparent layer pull the whole reflection into contact. A
2%-opaque veil 0.4 in front of a rough mirror turned its blurred reflection
nearly sharp until the veil was gone. §4.3 says how the drain records it.

**CORRECTED 2026-10-21: near and far content are blurred in separate layers.**
`k` is per pixel, so where it jumps the blur radius jumps with it. On a rough
floor beside a bright post, the pixels whose glossy ray met the post were
blurred by a few pixels and their neighbours, whose ray went past it to the sky,
by the whole lobe; the reflection ended in a hard edge exactly where the radius
changed. The path tracer draws that edge soft, because the lobe of a pixel
beside the reflection still sees part of the post, at the post's small blur, in
front of the sky at the large one. In the logo's demo the same step drew the
sphere's reflection in the satin cone as a sharp-edged tongue that changed shade
as the camera moved, and the cone's reflection in the lacquer cube as a darker
triangle.

The composite therefore prefilters the reflection in two layers. Each pixel
writes its radiance into the pyramid split between a NEAR and a FAR layer by its
own `G`, the radius as a fraction of the full lobe's (the curved cone factor of
§3.1): wholly near below `G = 0.15`, wholly far above `0.35`, linear between so a
reflection whose distance varies smoothly has no seam. A pixel then blurs each
layer at that layer's radius and composites them:

```
refl = a * near(sigma_near) + (1 - a) * far(sigma_far)
```

`a` is the near layer's share of the valid texels in the footprint at
`sigma_near`, which is how much of the lobe the near content covers. A pixel
knows its own layer's radius; for the other one it takes that layer's mean `G`
over the widest footprint a near reflection can have at its scale
(`0.35 * 1.8 * S`), times its own full-lobe radius `S`. A pixel that is near
uses the near layer's mean `G` over its own near footprint rather than its own:
on the outline of a near reflection the coverage-weighted cone factor mixes the
object's with the sky's, and blurring at that mix drew the outline as a bright
thread. The far layer excludes the near texels, as the near object occludes the
far content in the reflection, and normalising by the far layer's own weight
fills the hole behind it from around it.

The pyramid does not know where one surface ends, and the near layer made that
visible. A rough surface in front of a glossy one writes, along its silhouette,
what its lobe meets just behind it: the glossy surface, up close, so near. The
glossy surface's pixels beside the silhouette found that in their near
footprint and took a share of it, which drew a bright outline round the wave
on the logo's chrome A and round its reflection in the floor. A surface's own
near reflection comes in at its own roughness, so the near layer also carries
`log2 S` and its square, and a pixel reads the footprint as two populations,
one at its own `log2 S` and one `d` away: the other's share of the near
validity is `E[e]^2 / E[e^2]` and `d = E[e^2] / E[e]`, for `e` the offset from
the pixel's own. That share comes off `a` by how far `d` is past a ramp from
0.5 to 1.5 (roughness within a factor of 1.19 counts as the same surface,
beyond 1.68 as another). A mean scale alone would not do: half the pixel's own
near content and half another's averages to a scale neither wrote at, so a
test on the mean throws the pixel's own share away with the other's. The test
applies in full to a far pixel and scales away as the pixel is itself near,
since a near pixel's own content is in its footprint whatever surrounds it;
held to the test on the mean, the chrome A's pixels reflecting the underside
of the wave lost their share to the wave's own and came out up to 45 levels
brighter.
`test_a_glossy_surface_beside_a_rougher_one_keeps_its_own_reflection` builds
the case by hand: the glossy pixels beside the boundary reflected 54 to 97
where the answer is 50, and reflect 50.

A pixel with no near texel within reach takes the far layer alone at its own
radius, which is the single blur exactly, so a frame with no near reflection
renders as before. `test_a_near_objects_reflection_has_a_soft_edge` measures
the largest one-pixel change across the post's reflection as a fraction of its
contrast: 0.79 to 1.0 before, 0.10 to 0.12 after, and 0.07 to 0.20 in the path
tracer.

### 3.1 Curved reflectors

The footprint above is a flat mirror's: turning the view ray by `delta` turns the
reflected ray by `delta`. A convex mirror also tilts its normal under the moving
hit point, by `kappa * d_p * delta` at curvature `kappa`, and the reflection swings
twice that on top, so the reflected content moves `m = 1 + 2 * kappa * d_p` times
faster than the view and the lobe covers `m` times fewer pixels. Unfolding as
before, content `d_r` past the hit is seen through a view angle of
`sigma_angle * d_r / (d_p + m * d_r)`, which is `k / (1 + (m - 1) k)` in terms of
the flat cone factor: exactly `k` at `m = 1`, `sigma_angle / m` for the sky, and
sharp again for content nearer than the mirror's focal distance (`R / 2` for a
sphere of radius `R`, where `d_p / m` tends).

`m` is measured per hit, from the hit triangle's own vertex normals
(`_reflector_magnification`): along each edge `(n_j - n_i) . e / |e|^2` is the
normal curvature in that direction, and `m` comes from the least of the three.
The footprint is an ellipse narrowed most along the most curved direction; an
isotropic blur sized by the flattest direction never undercuts it, is exact on a
sphere, and leaves a surface flat one way (a cylinder, a cone along its
generators) at the flat footprint. It also keeps the blur continuous: the
estimate is constant per triangle, and the greatest curvature jumps between
neighbours wherever curvature varies (a cone's rings nearer the apex are more
curved), which with a geometric mean of the two drew sawtooth stripes and seams
across a satin cone.
Normals are turned to the viewer's side of the triangle first, so winding does
not matter. A concave direction counts as flat (a concave mirror's magnification
is unbounded near its focus, and the flat footprint is the widest blur the route
draws), and so does every Bezier circuit; the change only ever narrows a blur.

Without it, the logo's roughness-0.04 chrome sphere, 23 units from the camera
(`m` about 47), was blurred into a grey haze with coarse pyramid texels showing
along its rim, beside a sharp copy of itself in a mirror floor -- a deeper
bounce, which is never prefiltered (§5). `d_p` is known at the primary hit; `d_r`
is not known until the glossy ray hits something, so the ray records it — see
§4.3. A glossy ray that escapes to the background records nothing, which reads
as `k = 1`: the right answer for a reflection of the sky.

**CORRECTED 2026-10-21: the least curvature is the form's, not an edge's.** The
least of the three edges is the least curvature only when an edge lies along
the flattest direction. A triangle of a subdivided cone need not have an edge
along a generator; its least curved edge is then curved, and such triangles
came out magnified (1.44 on the demo's cone, 2.65 for the test's triangle)
against their neighbours' 1, drawing thin, sharper wedges across the satin cone
that moved with the camera. Normal
curvature is a quadratic form in the direction (Euler), and the three edges fix
its three coefficients, so `_least_curvature` solves for the form and takes its
smaller eigenvalue: zero on a cone whatever the triangle, `1/R` on a sphere as
before. It is never more than any edge's value, so the change only widens a
blur, and a sliver triangle, whose edges pin the form down poorly, keeps the
least edge.

`sigma_px` at roughness 0.35 and a PREVIEW frame is ~300 px — a rough metal
really does reflect an almost featureless average of its surroundings, which is
what the path tracer's "soft glow" reference is. That is why the prefilter is a
**mip pyramid** and not a separable blur: a 300-px separable blur is 600 taps
per pixel, and a strided approximation of it aliases.

## 4. Where it lands in the code

### 4.1 Settings (`raytracing/settings.py`)

* `GLOSSY_PREFILTER = env_flag("ALGAN_GLOSSY_PREFILTER", True)`. It only has an
  effect when `GLOSSY_REFLECTION` is on, which is still default **off**, so the
  default render is byte-identical. Turning glossy reflections on therefore
  gets the prefiltered route by default and the old tap fan only on request.
* `glossy_reflection_mode()` gains a value: `0` off, `1` fan, `2` fan +
  interleave (unchanged), **`3` prefiltered split-sum**. It reaches the resolve
  as a `ti.template()` exactly as before, so each mode keeps its own compiled
  variant.
* `set_glossy_reflection(enabled, *, interleave=None, prefilter=None)`.
* `GLOSSY_PREFILTER_MAX_LEVELS` (`ALGAN_GLOSSY_PREFILTER_LEVELS`, default 10)
  bounds the pyramid.

### 4.2 The resolve (`sheet_resolve_taichi.sheet_resolve_shade`)

Under `ti.static(glossy == 3)`, and for a sheet that is reflective, **not**
transmissive (no glass, no pane), has `roughness > _GLOSSY_MIN_ROUGHNESS`, has
bounces left, and either holds the pixel's claim or takes it (§2.2,
corrected 2026-10-20):

1. `R = E` (§2.1) instead of `R *= _mirror_share(rough)`, so the `share` term
   that follows keeps `1 - E` for local shading. Both shading modes (0 and 2)
   and the event-build mode (1) take this substitution, so the three walks stay
   identical.
2. A new branch, ahead of `elif is_pane or split_refl`, records **one** pool
   ray, which the walk spawns once it ends: mirror direction from the
   un-jittered surface point, `weight = (1,1,1)`, `bounces_left - 1`, and an
   accumulator row of `r + num_covered` — the glossy half of `pix_accum`
   (§4.3). No `sec_aa` fan: the taps are what the prefilter replaces.
3. It writes the per-pixel meta into **spare columns of `pix_accum`'s glossy
   row** rather than into a new ndarray. Both kernels involved sit at 72
   parameters, and Taichi's argument ceiling is why the environment map's
   placement already rides inside `layer_offsets`; a new ndarray argument is
   the one thing this change must not spend. The glossy row's layout is
   therefore

   | column | written by | meaning |
   | --- | --- | --- |
   | 0-3 | drain | reflected radiance + glow, premultiplied |
   | 4-6 | drain | leftover throughput (what the background shows through) |
   | 7 | drain | `sum(coverage / (base_dist + t_hit))` over the surfaces the glossy ray itself meets (§3) |
   | 8-10 | resolve, drain (§4.3) | `W = weight * alpha * E`, the factored-out energy |
   | 11 | resolve, drain (§4.3) | `sigma_angle / theta_px`, the full-lobe radius in pixels; `-1 / theta_px` where nothing has claimed the pixel |
   | 12 | resolve, drain (§4.3) | `d_p`, the camera path length to the claiming hit |
   | 13 | resolve, drain (§4.3) | `m`, the curved reflector's magnification (§3.1; 1 when flat) |

   so `pix_accum` is `(2 * primaries, 14)` on this route rather than
   `(primaries, 7)`. That is 4x a per-tile buffer of a few megabytes, and it
   costs no kernel argument at all.
4. It then performs the same occlusion write the non-glass reflective `else`
   branch does (`_run_svis_write` with `trans_share = 0`, `_run_redistribute`)
   and **continues the sheet walk** rather than bouncing the primary. The
   primary is now the pass-through, which is strictly better than today's
   `refl_max >= cover_pass` test: a silhouette pixel keeps both.

Everything under `mode == 1` (event build) still spawns nothing and writes no
ray state.

### 4.3 The drain (`wavefront_kernels_taichi.wavefront_shade`)

Glossy rays are ordinary pool rays. What marks one is its accumulator row:
`accum_pix >= gloss_base`, where `gloss_base` is the tile's primary count. That
tag rides for free — `rs_int[c, 4]` is copied to every continuation a ray
spawns, so a glossy ray's descendants accumulate into the glossy row too, which
is exactly the behaviour wanted (a reflection that bounces again is still part
of the same reflection).

`gloss_base` is passed in `layer_offsets[7]` rather than as a new kernel
argument — every extra ndarray argument costs a `set_arg` per launch and
per-use argument loads in the kernel, and Metal allows 31 buffers; the
"64-argument ceiling" is only a Python-side counter (`agent_guidance/taichi.md`).
The environment map placement rides there for the same reason. `0` means the feature is off. It is
paired with a `ti.template()` gate so that off compiles the whole thing out.

The one addition to the drain body: at each surface the glossy ray itself
meets, `ti.atomic_add(pix_accum[accum_pix, 7], cover / (base_dist + t_hit))`,
with `cover` the ray's remaining throughput (largest channel) times the
surface's opacity. The distance is the total path length from the camera, not
the segment. The composite reads `d_tot = 1 / column 7`, so
`1 - d_p / d_tot` is the coverage-weighted mean of each surface's cone factor,
a miss counting as 1 (§3). An opaque first hit gives exactly its own `d_tot`,
as the nearest-hit record this replaced did.

Only the glossy ray itself records, not its descendants. What a covering
surface reflects or refracts is what that surface shows, already counted at
its own distance; counting it again would pull the mean nearer. The ray
itself is the one still on the segment it was spawned on: its `base_dist` is
the claim's `d_p` (column 12) to the bit, because both are stored from one
value. Rows `[primaries:]` start at zero, so a glossy ray that never hits
anything reads as `k = 1`: a reflection of the sky, fully blurred.

The drain can also hold the pixel's event (§2.2, corrected 2026-10-20). The
resolve writes `-gl_px_per_rad` into column 11 of every pixel it shades. A
claim overwrites that value with a positive radius, so a scatter still reads
an unclaimed pixel as having no glossy branch. In the drain, a pool ray that
is not a glossy ray and meets a qualifying surface does three things:

1. It takes `R = E`.
2. It `atomic_add`s its `W` into columns 8-10.
3. It `atomic_max`es a tiny positive mark into column 11. The one ray that
   gets a negative value back is the claimant. It writes the radius from that
   value's magnitude, `d_p` and the hit's magnification `m` (§3.1, with the
   unfolded path length as `d_p`), and spawns the glossy ray into the pixel's
   glossy row.

A zero in column 11 means the resolve never ran on this route for the pixel,
so the drain falls back to the throttle. `gl_take` in
`wavefront_shade_arena` is the code.

Rows `[primaries:]` start zeroed with the rest of `pix_accum`.
`wf_composite_accum_sparse` is handed the first `primaries` rows as a view, so
it never sees the glossy half and is otherwise untouched.

### 4.4 The host (`tracer.py`, sparse route)

Two per-**frame** buffers, allocated once outside the tile loop:

| buffer | shape | contents |
| --- | --- | --- |
| `gl_main` | `(pixels_per_frame, 10)` f32 | `csum.rgba` (the pixel's linear pre-finalize color, background folded in), `W.rgb`, `sigma_px` (`< 0` = not a glossy pixel), `G` (the radius as a fraction of the full lobe's, before the clamp) and `S` (the full lobe's radius) |
| `gl_pyr` | `(~4/3 · pixels_per_frame, 14)` f32 | the mip pyramid of `B` in two layers, far then near (§3), each `(B.rgb, B.glow, v, v·G)` with `v` the layer's validity weight, the near layer followed by `(v·log2 S, v·log2² S)`, level 0 first. The glow lane is `out`'s column 3 (it is bloom coverage, not alpha) and it has to be prefiltered with the color or a blurred reflection would carry a sharp bloom mask. |

Per frame, not per batch: a batch is many frames and this would otherwise be the
dominant allocation. Covered ordinals are ordered by global pixel index, so a
frame's covered pixels are a contiguous ordinal range; when the prefilter is
active the tile loop **clamps each tile to the current frame's range** and
flushes at the boundary. The extra cost is at most one short tile per frame.

Per tile, after the drain and **before** `wf_composite_accum_sparse` (which
overwrites the raw prefilled background this reads):

* `gloss_scatter` — for each compact row with a glossy branch: write `csum`
  (the same `pix_accum·255 + leftover·bg` arithmetic the composite does), `W`,
  and `sigma_px = clamp(k · sigma_scale)` computed from the glossy row's `d_p`
  (column 12) and `d_tot` (one over column 7), with `G` and `S`, into
  `gl_main[pixel]`; and level 0 of `gl_pyr[pixel]`, `(w_f · B, w_f, w_f · G)`
  far and `(w_n · B, w_n, w_n · G, w_n · log2 S, w_n · log2² S)` near, with
  `B` the glossy row's radiance
  composited against the same background and `w_n = 1 - w_f` the pixel's near
  weight (§3).

Then `wf_composite_accum_sparse` runs unchanged and finalizes every covered
pixel including the glossy ones — its value for a glossy pixel is simply
overwritten below.

At the frame boundary:

* `gloss_pyramid_level` — one launch per level, a 2×2 weighted box reduction.
* `gloss_composite` — for each pixel with `sigma_px >= 0`, sample the pyramid
  trilinearly at `L = clamp(log2(sigma_px / 0.289), 0, levels - 1)` (a box of
  width `2^L` has standard deviation `2^L/sqrt(12) = 0.289 · 2^L`), divide each
  layer by its own sampled validity weight, composite the layers (§3), and
  write `out[f, p] = finalize(csum + W · prefiltered)`. A pixel with no near
  content within reach makes three trilinear fetches (the reach probe, core,
  halo); one beside near content makes up to six.
* reset both buffers for the next frame.

`gloss_finish` does all three in ONE launch: each pyramid level is its own
top-level loop (so its own offloaded task, run in order), then the composite's
loop, then the reset. Every kernel here computes a texel or a pixel through the
same `ti.func`, so the arithmetic and its order are the separate launches'. The
reset is cut to what the next frame reads: the composite puts each glossy
pixel's `sigma_px` back to -1 after reading it (the scatter writes every
`gl_main` column of a pixel it marks, and nothing reads an unmarked pixel past
`sigma_px`), and a last loop zeroes level 0 of `gl_pyr` (each higher level is
written whole by the next frame's reduction before anything reads it). The
reset before a chunk's first frame, `gloss_clear`, writes `gl_main` whole in
one coalesced pass and zeroes level 0 alone. `ALGAN_OPT_DISABLE=glossfuse`
restores the separate launches and `ALGAN_OPT_DISABLE=exactfills` the
strided clear; `tests/unit_tests/test_glossy_fused_finish.py` holds both to
them bit for bit, with everything the new reset skips poisoned.

`finalize` is `finalize_pixel_color` with the same tonemapping template value
and exposure the tile composite used, so a glossy pixel takes the identical
transfer path as its neighbours.

### 4.5 Memory

`gl_main` + `gl_pyr` is ~115 bytes per pixel of **one frame** (40 + 75; it was
53 before the two layers of §3) — 47 MB at PREVIEW, 238 MB at HD. Both come
from `ManualMemory`, so `rendering/memory_model.py`
measures them like everything else and the batch size adapts on its own. They
are allocated only when the mode is active.

Two things the *tile* sizer has to be told, which the batch model does not
cover — both were missing when this shipped, and both are why
`materials_and_lighting` and `solids_and_camera` failed with `OutOfRenderMemory`
at PREVIEW on a 600 MB arena:

* **Allocate the two frame buffers BEFORE `_auto_primary_per_tile` runs**, the
  way the classic route allocates `aa_accum` first. `WAVEFRONT_TILE_SAFETY` is
  1.0, so the tiler hands the tile *every* free arena byte; anything taken
  afterwards is taken out of the tile's own state. 16 MB at PREVIEW is nothing
  against the arena and everything against what is left of it by a batch's last
  chunk.
* **Charge the widened `pix_accum`.** `_WAVEFRONT_BYTES_PER_PRIMARY` is the
  measured 28 bytes of the plain `(primaries, 7)` row; this route allocates
  `(2 * primaries, 14)`, so the tile needs the extra 84 bytes per primary passed
  as `extra_bytes_per_primary` or it is fitted to a row it does not take.

Getting either wrong is not a graceful degradation. The tile overruns on its
first attempt and the halving retry cannot recover: a splitting batch holds the
continuation pool — the dominant allocation — fixed across retries, so only
`pix_accum` shrinks and the loop rides down to the one-covered-pixel diagnostic.
The retry now halves the pool alongside the primaries so a miss costs tiles
rather than the render, but that is a backstop, not the budget.

## 5. Deliberate limitations

* **Screen space.** A reflection of something off-screen or behind the reflector
  is the background, as it is today. The prefilter widens the lobe over what the
  screen has; on a reflector that is small on screen a wide lobe averages mostly
  its own pixels.
* **One glossy event per pixel** (§2.2). Among the primary's sheets it goes
  to the one reflecting the most light. A rough opaque reflector seen in a
  mirror or through glass claims it only when no primary sheet did. Bounces
  inside a glossy reflection keep the throttle.
* **A claim made in the drain is blurred at the primary ray's angular scale.**
  That is exact through a planar mirror, where path length unfolds. Through a
  curved mirror or a refraction it is an approximation.
* **An object and its mirror image share the pyramid.** Where their lobes
  overlap on screen, each one's prefilter averages the other's `B`. On the
  reported cone this moved the object itself by about -12 sRGB; its pixels
  had previously averaged only its own reflection buffer.
* **Curvature only on the pinhole.** An orthographic or blended projection keeps
  the flat footprint on a curved reflector (§3.1's `m` assumes the view ray's
  angle, not its offset, is what moves the hit point).
* **Transmissive materials are untouched.** Glass keeps Schlick and the existing
  refraction transport; the DFG substitution is for the opaque reflective branch,
  where the throttle was the visible error.
* **Several reflective sheets in one pixel** share one `B`, one `W` and one
  radius — the strongest sheet's. The others in the resolve's walk fall back
  to the throttle, and one that held the claim and lost it keeps a single
  sharp ray, not a `sec_aa` fan. A band challenging the claim is weighed one
  sheet at a time, so a band split across many siblings can lose to a smaller
  one-sheet band. Rough reflectors met in the drain join the event instead
  (§2.2, corrected 2026-10-20).
* **The contact distance is a mean.** A reflection that passes a
  half-transparent surface gets one radius between the surface's and the
  background's, not a sharp image of the surface over a blurred background.
* **Two distances, not many.** The near and far layers (§3) separate content
  by an order of magnitude in blur; two near objects at different distances
  share one near radius where their footprints overlap, the mean of theirs.
  The layers are screen-space like everything else here. Near content is kept
  to the surface that wrote it only by roughness (§3), so two touching
  surfaces of one roughness still share each other's near reflections within
  that small radius, and the far layer's wide blur mixes surfaces as it always
  has.
* **A box-mip prefilter is not a GGX prefilter.** It is the shape of the
  footprint, not of the lobe. That is the same approximation every real-time
  split-sum implementation makes for the radiance half.

## 6. Secondary tap count (`ANALYTIC_AA_SECONDARY_SAMPLES`)

Separate from the above, and a defect in its own right: the setting was snapped
to `8 / 4 / 2` because the sub-pixel positions were a hand-written table, so
`ALGAN_ANALYTIC_AA_SECONDARY=16` rendered exactly as 8, silently. Two things
capped it:

1. `_AA_SEC_JITTER` had entries for 1, 2, 4 and 8 only.
2. `_AA_SEC_OWNER` maps each of the 8 coverage samples to its *nearest* single
   position, so a fragment covering all 8 samples could never own more than 8
   positions — a 16-entry table alone would still have fired at most 8 rays.

Both are fixed:

* Positions for counts outside `{1, 2, 4, 8}` are generated at import from a
  Hammersley set. The four hand-written tables stay verbatim, because the
  render baselines were taken through them.
* **The ownership direction changes above eight**, and that asymmetry is
  deliberate rather than tidy. Up to eight taps the rule stays the forward one
  — each coverage sample owns its single nearest position — because the inverse
  rule does *not* reproduce it: the two disagree on 208 of the 256 possible
  coverage masks at `n = 8` alone, so swapping it wholesale would move renders
  at the default tap count. Above eight the direction flips to the inverse rule
  — each *position* assigned to its nearest coverage sample, which may own
  several — because forward ownership can never exceed one position per sample
  and there are only eight samples. That cap, not the table's length, is what
  rendered 16 and 32 as 8.

The ceiling is 32: the position mask is an `i32` bitfield, and the resolve
unrolls `ti.static(range(sec_aa))` at four call sites, so a high count is also
paid in kernel compile time. A value above it clamps and warns once per process
instead of silently snapping.

One wart the forward rule carries, found while pinning it and **deliberately
left**: at `n = 8`, position 1 is nobody's nearest, so a fully covered fragment
spawns seven rays where eight were asked for. (1, 2 and 4 partition their
positions completely; only 8 has the gap.) It is left because the fan is now
the legacy arm — `glossy_reflection` selects the prefilter — and changing what
an existing, measured configuration renders to tidy a path nothing recommends
is a bad trade against byte-identity. `tests/unit_tests/test_secondary_tap_
counts.py` asserts it rather than hiding it, so it cannot drift unnoticed and
whoever wants it fixed will find it stated.
