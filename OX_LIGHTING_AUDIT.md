# Lighting → linear-value path: code audit

Read-only audit, 2026-08-22. No files modified; no renders or tests run.
Context: `TONEMAP_FINDINGS.md` §9 — the default tonemap is off, lit surfaces
that were calibrated against the curve now clip per-channel.

**Caveat on line numbers.** The tree moved *while this audit ran*: a hue-preserving
shade bound landed mid-audit as commit `5133478`, touching
`shading_taichi.py` (+20 lines at the tail of `_run_frag_pipeline`) and
`material_shaders.py` (`_recombine` rewritten), plus `test_tonemapping.py`.
All citations below are against the tree **including** that commit; where the
pre-commit behaviour matters (§4) it is called out explicitly.

---

## 1. `SETTINGS.raytracing.light_intensity` (`LIGHT_INTENSITY`, default π)

**Refined verdict: the brief's premise is half right, in an important way.**
`light_intensity` reaches only `path_trace_physical_stbvh`
(`algan/rendering/raytracing/raytrace_kernels_taichi.py:3292`, consumed at
`:3533` as `light_col[...] * light_intensity`, with `ambient` at `:3489-3493`)
— but that kernel is **never launched by any renderer**. The dispatch in
`tracer.py` launches exactly two backends:

* `samples_per_pixel == 1` (default): the deterministic wavefront,
  `raytrace_render_wavefront` (`tracer.py:1577`). Its light inputs are the
  packed arrays only (`tracer.py:1596-1598`); **no intensity scalar exists on
  this path**.
* `samples_per_pixel > 1`: `path_trace_scene_stbvh` (`tracer.py:1553`) — which
  receives **no lights at all**: `light_pos = light_col = None; num_lights = 0`
  (`tracer.py:1363-1365`). It gets `INDIRECT_BOUNCE_STRENGTH` instead
  (`tracer.py:1557`).

The only other reference to `path_trace_physical_stbvh` is
`tests/unit_tests/test_raytracing_unit.py:255`. So `LIGHT_INTENSITY` is read
only by a dead kernel. This is not a latent bug waiting to be found — it is
declared: `_INERT_FIELDS` in `algan/settings/raytracing_settings.py:149-162`
makes **writing** `light_intensity` raise `AlganConfigurationError` with the
message "'light_intensity' is not read by any renderer this build can launch".
Pinned by `tests/unit_tests/test_inert_settings.py`.

**What the default path uses instead:** the literal integers `1, 1`. The torch
vertex-shading call passes them positionally for
`(light_intensity, ambient_light_intensity)` at
`algan/rendering/raytracing/primitives.py:620-621`. That is what
`shading_taichi.py:46`'s docstring claim ("``light_intensity == ambient == 1``")
refers to — yes, literally hard-coded, at `primitives.py:620-621`. The Taichi
per-fragment stages go further: they have no intensity parameter in their
signature at all (`shading_taichi.py:554-557`, `:616-619`, `:645-648`,
`:679-682`, `:719-723`); they multiply the packed light colour only.

Note the vertex call site is itself conditional: `_shade_vertex_colors` skips
work when `_shaded_per_fragment()` is true (`primitives.py:586-589`), and
fragment shading is the default (`FRAGMENT_SHADING = True`,
`raytracing/settings.py:193`; a Mob's default shader maps to core id 0 via
`settings.py:2187` and `algan/__init__.py:153`). So in a default render the
vertex literals rarely execute; the fragment stages use packed colour only.

## 2. `AMBIENT_LIGHT` / `AMBIENT_STRENGTH`

* `AMBIENT_LIGHT = 0.0` (`raytracing/settings.py:96`): same story as §1 —
  consumed only by the dead physical kernel (`:3490-3492`), declared inert at
  `raytracing_settings.py:156-161`, write raises. Reaches **no live path**.
* `AMBIENT_STRENGTH = 0.1`: two independent hard-coded copies reach the live
  paths — `material_shaders.py:41` (torch shaders) and
  `shading_taichi.py:82` (kernel stages), used at `material_shaders.py:206,
  :237, :289, :383, :421` and `shading_taichi.py:633, :662, :698, :759`.
  Neither reads any setting; both are module constants. Because the renderer
  always passes `ambient_light_intensity == 1` (`primitives.py:621`; comment
  at `material_shaders.py:37-40`), the *effective* ambient coefficient is
  0.1 × env_map_intensity everywhere.

## 3. Multi-light combination in the deterministic path

The docstring's "each light applied in sequence with the running colour as
albedo" describes the *legacy vertex* loop:
`primitives.py:593-631` — `for light_source in light_sources:` overwrites
`self.colors[..., :d] = shaded` each iteration (`:627`). The kernel stages do
**not** do that any more; they are additive accumulations over a fixed albedo:

* `_stage_default` (`shading_taichi.py:572-611`): gathers
  `acc += lc * w` with `w = max(n·ld,0)^5 · 0.5 · vis` per light, plus
  `wsum += w·frac`; result `out = out·(1 − min(wsum,1)) + acc` (`:611`). The
  **base fade** is clamped (`min(wsum,1)`); the additive term `acc` is not.
* `_stage_lambert` (`:633-641`), `_stage_phong` (`:662-675`),
  `_stage_standard` (`:698-716`), `_stage_physical` (`:759-817`):
  `acc = ambient + emissive` once, then `acc += <direct term> · v` per light.

**It is a pure sum over lights; nothing normalises or clamps it inside the
stages.** Ceiling arithmetic for N white unit-intensity lights on a fully lit
white surface, committed behaviour:

* Lambert family: ambient `0.1·albedo` + `Σ albedo·lc·(n·l)` ≤ `0.1 + N`
  (specular lobes add more). Linear in N, unbounded.
* Default stage: `≤ min(1, 0.5N)·albedo + 0.5N·lc` → exactly **1.0 at N=1**
  (the "default rig lands exactly on 1.0" of the new comment), ~1.5–2.0 around
  N=2–3 depending on angles.

That is the mechanism behind the measured 2.397 peak: `tests/fast/scene.py:78-89`
rigs three lights (ambient 0.45 + directional 0.85 + point 0.6) onto materials
that include the unbounded Lambert/PBR stages.

## 4. Where the shaded colour is clamped

* **Torch vertex path (pre-`5133478`):** every `material_shaders.py` shader
  ended in `_recombine`, which clamped RGB to [0,1] per channel. Exceptions:
  `pbr_shaders.default_shader` returns its lerp **unclamped**
  (`pbr_shaders.py:207-212` — inherently bounded anyway, see below);
  `basic_pbr_shader` clamps at `pbr_shaders.py:157`; `null_shader`
  passthrough.
* **Taichi fragment path (pre-`5133478`): none.** The only clamp was
  commented out — `#out = ti.math.clamp(out, 0.0, 1.0)`. Downstream, the
  composite writes linear HDR straight through when post tonemap is on
  (`wf_composite`, `wavefront_kernels_taichi.py:1347-1351`;
  `finalize_pixel_color` t_val==3 arm, `raytrace_kernels_taichi.py:1887-1905`).

**So at HEAD the two paths disagreed** — torch truncated per channel at every
lit vertex, the kernel path never bounded — which is a real defect, independent
of everything else.

* **Working tree (in flight while auditing):** both sites now apply the *same*
  hue-preserving bound — floor at 0, divide all three channels by the peak
  when peak > 1, identity below 1, glow untouched:
  `shading_taichi.py:1098-1120` (end of `_run_frag_pipeline`) and
  `material_shaders.py:55-73` (`_recombine`; docstring cross-references the
  kernel site). Now committed as `5133478`, closing the disagreement for the
  built-in materials. The tail sits outside the stage dispatch, so it bounds
  every pipeline id — user pipelines included.
* **Final encode (both routes):** bloom has already run on the HDR buffer;
  the last step clamps — Taichi `tonemap_to_u8` method_id 0
  (`post_process.py:220`, kernel clamp in `tonemap_kernels_taichi.py`, the
  `method == 0` arm) or the torch fallback `clamp_(0.0, 1.0)` at
  `post_process.py:252-255` (exposure now applied first; §8 of
  TONEMAP_FINDINGS is fixed on this branch).

## 5. Glow flow

Glow is the 4th colour channel end to end, and **it bypasses every shading
bound**:

* Authored into `colors[..., -2]` (`primitives/triangle_primitive.py:159`,
  `bezier_circuit_primitive.py:186,205`); packed as corner channel 3.
* Torch shaders carry it as `glow_tail`, concatenated back **unbounded** by
  `_recombine` (`material_shaders.py:73`; explicit at `:61-62`).
* Kernel stages return `in_glow` untouched in slot 3 of their vec4
  (`shading_taichi.py:550, :612, :641, :675, :716, :817`); the new tail
  deliberately scales only `out`, returning `g` as-is (`:1116-1120`).
* Composite writes it to frame channel 3 (`wavefront_kernels_taichi.py:1347`);
  MC path treats glow as emission `albedo·glow` accumulating into the same
  lane (`raytrace_kernels_taichi.py:3482-3488`, averaged at `:1898`).
* Bloom turns it into above-1.0 RGB: `bloom_filter`
  (`post_processing/bloom.py`) early-outs when max glow ≤ 1e-5 (`:759-762`),
  reads channel 3 (`:788`), weights colour by `glow³·strength` (`:789-790`),
  multi-scale blurs, then **adds the halo back onto RGB channels 0-2**
  (`:845-850`). Bloom runs before the encode (`post_process_frames`,
  `post_process.py:276+`), on the float HDR buffer
  (`render_loop.py:1050-1052`).

So clamping/bounding shaded RGB at shade time would **not** destroy
glow-driven HDR: the halo is manufactured later, additively, from the glow
lane. The one interaction to keep: a shade-time bound must not clamp the glow
channel itself, or `glow³` loses its source.

## 6. Intensity semantics per light type

One uniform rule: `intensity` is a plain scalar multiplier on the light's
colour, applied once at snapshot time — `render_loop.py:2411-2414`
(`col = color.rgb·alpha·opacity`, then `col *= intensity`). It reaches every
consumer through the packed radiance columns (`scene_builder._pack_lights`,
docstring "intensity premultiplied" at `scene_builder.py:1860`). Per type,
all in `lights.py` unless noted:

* **PointLight** (`:211-268`): multiplier only; optional decay/range falloff
  applied in-kernel (`shading_taichi.py:488-499`).
* **DirectionalLight** (`:311-352`): multiplier only; no falloff; direction
  packed (`shading_taichi.py:459-461`).
* **AmbientLight** (`:355-363`): multiplier only. In-kernel it arrives along
  the normal with specular gated (`shading_taichi.py:462-464`), so the
  Lambert-family stages add a flat `albedo·lc` term **per ambient light** —
  yes, it compounds additively when several exist (no de-duplication, no
  maximum). In `_stage_default` each ambient light pulls the base colour
  toward `lc` with fixed weight 0.5 (`w = 1^5·0.5`).
* **HemisphereLight** (`:366-413`): sky scaled like any colour;
  ground colour premultiplied separately in `build_aux` (`:412`) and opacity-
  scaled at materialization (`render_loop.py:2429-2440`); blended by normal
  height in-kernel (`shading_taichi.py:465-474`). No double application.
* **SpotLight** (`:416-484`): point semantics + smoothstep cone
  (`shading_taichi.py:500-508`).
* **RectAreaLight** (`:487-606`): expanded to K emitter rows, each carrying
  colour/K (`render_loop.py:2425-2427`) and power fraction 1/K
  (`lights.py:177`); one-sided cosine emission per sample
  (`shading_taichi.py:509-513`). Sum over samples reconstructs the full
  intensity; `_stage_default` counts rows by fraction so K samples fade the
  base like one light (`:600-610`).
* (Env map ambient rides in as a synthetic ENV_SH row, `tracer.py:1347-1355`,
  evaluated `shading_taichi.py:475-487`.)

## 7. Candidate edit sites to enforce the invariant (map only)

1. The accumulation loops themselves — `shading_taichi.py:572-611`
   (default), `:633-641`, `:662-675`, `:698-716`, `:759-817`: normalise or
   energy-bound per-light terms.
2. The pipeline tail — `shading_taichi.py:1116-1119` (working tree): where
   the hue-preserving rescale now lives; any other policy would sit here.
   Covers every pipeline id, built-in and user.
3. The torch twin — `material_shaders.py:55-73` (`_recombine`), plus the
   unclamped `pbr_shaders.default_shader:207-212` and
   `basic_pbr_shader:157` if consistency wanted there too.
4. Upstream at the source — `render_loop.py:2411-2414`: cap effective
   radiance where intensity premultiplies; single choke point for both paths,
   both routes, and would also bound the MC-emission reading of colour.
5. Frame-buffer level — composite/finalise: `wavefront_kernels_taichi.py:1347-1351`,
   `raytrace_kernels_taichi.py:1887-1905`, the sheet-resolve composite
   (`sheet_resolve_taichi.py:869`). Bounds background+bloom input too, i.e.
   changes what bloom sees.
6. Encode level — `post_process.py:252-255` / `tonemap_kernels_taichi`
   method 0: display-referred only; cannot protect bloom's input (bloom runs
   earlier), so insufficient alone for anything but the final byte.
7. Ambient specifically — the per-AmbientLight compounding of §6 could be
   addressed at the packing (`scene_builder._pack_lights:1850`) or in the
   stages' treatment of `_LT_AMBIENT` rows (`shading_taichi.py:462-464`).

## 8. What would move

Pixel-compared suites (tolerance 2, per-device committed baselines):

* `tests/fast/test_fast_render.py` vs `tests/fast/expected_outputs_{cpu,cuda}/`
  — scene rigs three lights (`tests/fast/scene.py:78-89`) driving Lambert/PBR
  materials; measured peak 2.397 (§9 of TONEMAP_FINDINGS). Any change to lit
  values moves it; note `tests/fast/scene.py` carries no PN geometry, so
  tessellation-adjacent effects are invisible there.
* `tests/full_renders/test_full_renders.py` vs six scenes under
  `tests/full_renders/expected_outputs_{cpu,cuda}/` — **every scene has
  lights** (each has ≥ AmbientLight+DirectionalLight):
  `materials_and_lighting.py` (all six types + glow up to 2.5 + refraction),
  `solids_and_camera.py:39-46` (+ Hemisphere), `text_and_media.py:27-28`,
  `manim_compat_and_plots.py:31-32`, `complex_hierarchy_become.py:47-48`,
  `shapes_and_timeline.py:48-49`. Skips itself under `CI`.

Unit tests touching lighting values/behaviour (no pixel baselines):
`test_inert_settings.py` (pins the §1/§2 raises — any re-wiring of
`LIGHT_INTENSITY`/`AMBIENT_LIGHT` must update it),
`test_materials.py:454-508`, `test_deterministic_shadow_opacity.py:34`,
`test_render_truncations.py:407` (shadow-light ceiling),
`test_ux_regressions.py:608-627,1208-1209`, plus the locally-modified
`test_tonemapping.py`.

## Uncertainties

* The sheet-resolve composite is cited from its acc write
  (`sheet_resolve_taichi.py:869`) without a line-by-line clamp check; the
  wavefront composites I did check have none between shade and buffer.
* The new tail bounds each shading event; continuation rays re-shade per
  bounce, so a composited pixel sums several bounded events and can still
  exceed 1.0 before encode (bounded per event, not per pixel).
* §3's ceilings are the diffuse-dominated case; specular lobes raise them by
  a roughness-dependent amount.
