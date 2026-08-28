# Split-sum glossy reflection (prefiltered reflection buffer)

Status: design of record for the change implementing
`benchmarks/renderer_audit/REPORT.md` §4.5.1. Read §4.5 and §4.5.1 first — they
are the measurement this is answering, and they name the two halves:

* the **DFG term** makes the reflected *energy* analytic, replacing
  `_mirror_share`'s throttle heuristic;
* a **prefiltered radiance** makes the reflected *shape* right, built by a
  screen-space reflection filter rather than by more taps.

Nothing here changes the default render. The whole route is behind
`GLOSSY_REFLECTION`, which is off by default; with it off every kernel gate
below compiles out and no buffer is allocated.

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
sharp, a distant one is fully blurred. `d_p` is known at the primary hit; `d_r`
is not known until the glossy ray hits something, so the ray records it — see
§4.3. A glossy ray that escapes to the background never records one, and its
`d_r` stays at the initialised infinity, which is `k = 1`: the right answer for
a reflection of the sky.

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
bounces left, and is the first such sheet in this pixel's walk:

1. `R = E` (§2.1) instead of `R *= _mirror_share(rough)`, so the `share` term
   that follows keeps `1 - E` for local shading. Both shading modes (0 and 2)
   and the event-build mode (1) take this substitution, so the three walks stay
   identical.
2. A new branch, ahead of `elif is_pane or split_refl`, spawns **one** pool ray:
   mirror direction from the un-jittered surface point, `weight = (1,1,1)`,
   `bounces_left - 1`, and an accumulator row of `r + num_covered` — the
   glossy half of `pix_accum` (§4.3). No `sec_aa` fan: the taps are what the
   prefilter replaces.
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
   | 7 | drain | `min(base_dist + t_hit)` over the ray and its descendants |
   | 8-10 | resolve | `W = weight * alpha * E`, the factored-out energy |
   | 11 | resolve | `sigma_angle / theta_px`, the full-lobe radius in pixels |
   | 12 | resolve | `d_p`, the primary hit distance |

   so `pix_accum` is `(2 * primaries, 13)` on this route rather than
   `(primaries, 7)`. That is 3.7x a per-tile buffer of a few megabytes, and it
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
argument (the kernel is near Taichi's 64-argument ceiling — the same reason the
environment map placement rides there). `0` means the feature is off. It is
paired with a `ti.template()` gate so that off compiles the whole thing out.

The one addition to the drain body: at each hit of a glossy ray,
`ti.atomic_min(pix_accum[accum_pix, 7], base_dist + t_hit)` — the total path
length from the camera, not the segment. Two reasons it is that and not `t_hit`:
a descendant's `base_dist + t_hit` is always larger than its parent's, so the
minimum is exactly the *first* hit of the directly-spawned glossy ray (a
descendant's own short segment would win a `t_hit` minimum and report a
reflection as being in contact when it is not); and the composite already knows
`d_p`, so `d_r = d_tot - d_p` costs nothing there and saves the drain from
needing `d_p` at all. A glossy ray that never hits anything writes nothing, and
the host's initialised `+inf` is `k = 1`: a reflection of the sky, fully blurred.

Rows `[primaries:]` are initialised with column 7 at `+inf`.
`wf_composite_accum_sparse` is handed the first `primaries` rows as a view, so
it never sees the glossy half and is otherwise untouched.

### 4.4 The host (`tracer.py`, sparse route)

Two per-**frame** buffers, allocated once outside the tile loop:

| buffer | shape | contents |
| --- | --- | --- |
| `gl_main` | `(pixels_per_frame, 8)` f32 | `csum.rgba` (the pixel's linear pre-finalize color, background folded in), `W.rgb`, `sigma_px` (`< 0` = not a glossy pixel) |
| `gl_pyr` | `(~4/3 · pixels_per_frame, 5)` f32 | the mip pyramid of `B`: `(B.rgb, B.glow, v)` with `v` the validity weight, level 0 first. The glow lane is `out`'s column 3 (it is bloom coverage, not alpha) and it has to be prefiltered with the color or a blurred reflection would carry a sharp bloom mask. |

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
  (column 12) and `d_tot` (column 7), into `gl_main[pixel]`; and level 0 of
  `gl_pyr[pixel] = (B, 1)` with `B` the glossy row's radiance composited against
  the same background.

Then `wf_composite_accum_sparse` runs unchanged and finalizes every covered
pixel including the glossy ones — its value for a glossy pixel is simply
overwritten below.

At the frame boundary:

* `gloss_pyramid_level` — one launch per level, a 2×2 weighted box reduction.
* `gloss_composite` — for each pixel with `sigma_px >= 0`, sample the pyramid
  trilinearly at `L = clamp(log2(sigma_px / 0.289), 0, levels - 1)` (a box of
  width `2^L` has standard deviation `2^L/sqrt(12) = 0.289 · 2^L`), divide by
  the sampled validity weight, and write
  `out[f, p] = finalize(csum + W · prefiltered)`.
* zero both buffers for the next frame.

`finalize` is `finalize_pixel_color` with the same tonemapping template value
and exposure the tile composite used, so a glossy pixel takes the identical
transfer path as its neighbours.

### 4.5 Memory

`gl_main` + `gl_pyr` is ~53 bytes per pixel of **one frame** — 22 MB at PREVIEW,
111 MB at HD. Both come from `ManualMemory`, so `rendering/memory_model.py`
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
  `(2 * primaries, 13)`, so the tile needs the extra 76 bytes per primary passed
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
* **One glossy event per pixel** (§2.2). Deeper glossy bounces keep the throttle.
* **Transmissive materials are untouched.** Glass keeps Schlick and the existing
  refraction transport; the DFG substitution is for the opaque reflective branch,
  where the throttle was the visible error.
* **Several reflective sheets in one pixel** share one `B`, one `W` and one
  radius — the first qualifying sheet's. Later ones fall back to the throttle.
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
