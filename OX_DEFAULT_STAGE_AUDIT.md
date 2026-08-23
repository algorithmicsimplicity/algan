# Default fragment stage "base fade" — code audit

Read-only audit, 2026-08-23. No files under `algan/` modified; no renders; no pytest.
Written against the working tree at HEAD `275e731`. All line numbers verified in the
tree unless explicitly labelled otherwise.

**Verdict up front:** your diagnosis is correct in mechanism and site. Two
refinements: (a) light *strength* has no effect on `wsum` at all — the fade weight
is purely geometric, so even two very dim lights fully annihilate the albedo;
(b) ambient-like rows are the guaranteed way to hit it (each contributes exactly
0.5), but any two near-head-on lights of any type do it too. The asymmetry
between `esum` (radiance-weighted) and `wsum` (geometric-only) is **not
documented intent**: the radiance weighting was introduced deliberately for the
energy budget in commit `1a4c9d2` with an explicit statement that geometric-only
counting is wrong, while `_stage_default`'s pre-existing base-fade sum was left
untouched.

---

## 1. Every implementation of "lerp the albedo toward the light colour"

There are exactly **two**, and they are the two you know:

**(a) `_stage_default` (Taichi)** — `algan/rendering/raytracing/shading_taichi.py:792-864`.
It *accumulates* a base fade across lights: per live row,
`w = d⁵ · 0.5 · v` (`:837`, `d = max(ld·n, 0)` from `:836`), then
`wsum += _vis_max_component(w) * frac` (`:856`) — **geometric-only**, no radiance
factor — with one blend at the end:
`out = out·(1 − min(wsum,1)) + acc·_energy_scale(esum)` (`:863`).
`acc += lc · w` (`:838`) and `esum += _vis_max_component(w) · max(lc)` (`:844-845`)
are radiance-carrying; only `wsum` is not.

**(b) `default_shader` (torch vertex)** — `algan/rendering/shaders/pbr_shaders.py:160-212`.
`diffuse_factor = relu(−(n·to_light))⁵ · 0.5` (`:203-206`) — same geometric-only
weight — then `torch.lerp(albedo_color, light_color, diffuse_factor)` (`:207-212`).
It does **not** accumulate: the caller loops lights and overwrites the running
colour each iteration (`primitives.py:660-698`), so N lights lerp sequentially,
the exact behaviour the kernel stage's docstring says it replaces
(`shading_taichi.py:796-803`).

**Places checked that do NOT implement it:**

- **Monte Carlo megakernel** (`raytrace_kernels_taichi.py`, the SPP>1 path):
  no material-id concept and no base fade. Ambient multiplies albedo
  (`:3639-3643`); next-event-estimation diffuse is
  `albedo · ((1−metallic)/π) · cosθᵢ · visibility · radiance` (`:3678-3687`);
  continuation throughput multiplies albedo (`:3707-3708`). Pure multiplication.
  (Grep for `lerp` / `** 5` in that file hits only Fresnel.)
- **Deterministic wavefront shade kernel** (`wavefront_kernels_taichi.py:2899-2906`)
  and the **sheet-resolve kernel** (`sheet_resolve_taichi.py:426-434`; imports
  `_shade_tri_hit` at `:69,:82`): both funnel through `_shade_tri_hit`
  (`raytrace_kernels_taichi.py:1688-1726`, `_run_frag_pipeline` call at `:1726`)
  → `_run_frag_pipeline` → pid 0 dispatches the **same** `_stage_default`
  (`shading_taichi.py:1325-1330` ungated chain, or solo path `:1300-1303`). One
  implementation, three launch sites — a fix there covers both routes.
- **Legacy sorted wavefront**: referenced by `shading_taichi.py:1131-1138` and
  imported lazily at `tracer.py:3676`, but the module
  `wavefront_sorted_kernels_taichi.py` **does not exist in the tree**, and its
  enabling setter raises "unsupported" (`settings.py:2310-2312`). Its injection
  helper `builtin_pipeline_fn` (`shading_taichi.py:1233-1244`) composes the same
  `_BUILTIN_STAGE_FNS`, so a revival would inherit the fixed stage.
- **`material_shaders.py` torch twins** (lambert/phong/standard/physical/toon/
  normal/matcap/depth): all multiply albedo — e.g. `lambert_shader`
  `ambient = rgb·kA; diffuse = rgb·radiance·n_dot_l` (`:286-287`). The only
  `torch.lerp` is on normals for flat shading (`:164`). No albedo→light lerp.
- **`basic_pbr_shader`** (`pbr_shaders.py:31-157`): additive
  `ambient + diffuse + specular`, albedo multiplied (`:101`, `:151`).
- **Shadow-fan culling sites** are not shading, but they corroborate the
  mechanism's lc-independence: both culling sites exclude `_MID_DEFAULT` from
  zero-radiance fan culling *because* its base fade accumulates a vis-weighted
  `w` even at `lc == 0` (`wavefront_kernels_taichi.py:161-165` docstring,
  `:2669-2677`; `raster_taichi.py:2857-2863`).
- **Vendored code** (`algan/external_libraries/`): manim's Cairo renderer is a
  different pipeline entirely; grep found no such rule. *(Verified by grep, not
  read line-by-line.)*

## 2. Which path runs for a mob with no material, under default settings

Chain, with defaults (SPP=1, deterministic backend):

1. A Mob built without `set_material` leaves `Mob.shader = None`
   (`animatable_base/mob.py:258`); when its triangles are built,
   `shader is None → SETTINGS.style.default_shader`
   (`rendering/primitives/triangle_primitive.py:187-189`), which is
   `pbr_shaders.default_shader` (`algan/__init__.py:153`).
2. `default_shader` **is core**: id 0 in the registry
   (`raytracing/settings.py:2551-2559`, entry `default_shader: 0`;
   `_shader_is_core` `:2647-2651`; `_shader_material_id` returns 0 for it,
   `:2637-2644`).
3. `_shaded_per_fragment()` (`raytracing/primitives.py:603-618`) is True iff
   `FRAGMENT_SHADING ∧ SAMPLES_PER_PIXEL ≤ 1 ∧ _shader_is_core(shader)`
   (`:614-618`). Defaults: `SAMPLES_PER_PIXEL = 1` (`settings.py:33`),
   `FRAGMENT_SHADING = True` (`settings.py:210`).
4. Therefore `_shade_vertex_colors` early-returns
   (`primitives.py:655-656`) — **the torch `default_shader` path is dead for
   material-less mobs under default settings** — and every hit is shaded
   per fragment in-kernel by `_stage_default`, on whichever deterministic route
   the batch takes (sheet resolve or classic wavefront; route decision
   `analytic_raster_route_active`, `tracer.py:542-635`; backend selection
   `tracer.py:966`).

The torch path is **NOT dead** under:

- **`SAMPLES_PER_PIXEL > 1`** (Monte Carlo backend, `tracer.py:966`):
  `_shaded_per_fragment()` is False, so `project_to_screen` →
  `_shade_vertex_colors` bakes lit colours per vertex with the sequential torch
  lerp (`render_loop.py:1261`, `:2354`; bake loop `primitives.py:660-698`),
  and the megakernel then applies its own ambient+NEE lighting on top
  (`raytrace_kernels_taichi.py:3632-3687`).
- **`set_fragment_shading(False)`** (`settings.py:2320-2331`): colours are
  Gouraud-baked the same way; the kernels skip the pipeline behind the
  compile-time gate (`sheet_resolve_taichi.py:401`,
  `wavefront_kernels_taichi.py:2619`).

So your fix as drafted changes the default-settings path only; the two settings
above keep rendering through the unfixed torch twin.

## 3. Is `wsum`'s omission of the radiance factor deliberate?

**No documented intent exists; the evidence says oversight.**

- The geometric-only form **predates visible history**: the repo's initial
  squashed commit `5df5b8a` ("Sheet resolve Phase 0...") already contains
  `wsum += w * frac` verbatim, with the docstring reasoning only about (i)
  additive-vs-sequential accumulation and (ii) the `frac` power-fraction
  weighting for area-light samples ("one area light displaces at most as much
  base colour as one point light would"). Nothing justifies excluding radiance
  from the count.
- Commit **`1a4c9d2`** ("Make lighting energy-conserving: normalise the
  illumination budget", 2026-08-22) introduced `esum` and the other stages'
  budgets. Its message states the opposite policy outright:

  > "Weighting the budget by the light's own colour is load-bearing, and the
  > first attempt got it wrong. Counting geometry alone charges a rig for how
  > many lights it has rather than how much light they emit, so tests/fast's
  > three dim lights (0.45/0.85/0.6) were billed as three full ones... Weighting
  > by peak(colour) makes three lights at 0.3 cost what one at 0.9 costs."

  The diff adds `esum` to `_stage_default` and radiance-weighted `wsum`s to the
  four PBR-family stages — but leaves `_stage_default`'s base-fade `wsum` line
  untouched (`git show 1a4c9d2 -- algan/rendering/raytracing/shading_taichi.py`
  shows `wsum += w * frac` as context, not a change). The same commit message
  even reports observing the fast scene's dim rig being mis-billed — by the
  budget, which it fixed — without noticing the base fade next to it bills the
  same way and was not fixed.
- Design docs: repo-wide grep of `*.md` for `wsum` / "base fade" finds only
  `OX_LIGHTING_AUDIT.md:83-84` (a description, not a rationale) and its §6 note
  "In `_stage_default` each ambient light pulls the base colour toward `lc`
  with fixed weight 0.5". No design doc claims the asymmetry.

Conclusion: inherited accident, contradicted by the project's own stated
policy eleven hours later.

## 4. Bit-identity for the default-lit scene

**The default rig** is one light: `default_scene_initializer`
(`algan/__init__.py:226-233`) spawns
`PointLight(location = camera + UP + 5·RIGHT + OUT, color=WHITE)` with default
`intensity=1.0`. Nothing else adds lights to an untouched Scene
(`scene.py:183`, `:435` start empty).

**What `lc` is for that row**, end to end:

- Ingest/decode once per frame: `col = srgb_to_linear(rgb) · alpha · opacity`,
  then `× intensity` only when intensity ≠ 1.0
  (`render_loop.py:2418-2430`). For this light: `WHITE = #FFFFFF`
  (`constants/color.py:274`) decodes to **exactly (1.0, 1.0, 1.0)** (verified by
  running `srgb_to_linear`: `tensor([1., 1., 1.])` — IEEE-exact since
  `(1.055/1.055)^2.4 = 1.0`), alpha = opacity = 1.0, and the intensity multiply
  is skipped entirely (`:2429-2430`).
- Packing: a plain `PointLight` (decay=distance=shadow_radius=0) is *not*
  extended (`lights.py:241-251`), so a scene holding only it keeps the compact
  C==3 packing (`scene_builder._pack_lights:1993-2013`), whose rows are raw RGB
  — docstring "RGB radiance (intensity premultiplied)" at `:1984`. `frac` reads
  1.0 in `_light_eval` (`shading_taichi.py:668`; the extended branch that could
  change it, `:673-675`, is compiled out for width-3 rows; even in a mixed pack,
  `_blank_aux` writes `aux[...,12] = 1/num_samples() = 1.0`, `lights.py:179`).
- Evaluation: decay=0, range=0 → no modifier touches `lc`
  (`shading_taichi.py:705-716` inert). So at the `wsum` line, `lc == (1,1,1)`
  exactly.

**Answer: yes — bit-identical, with the number max(lc) = 1.0 exactly.**
Your proposed term `_vis_max_component(w) · frac · ti.max(lc[0], max(lc[1],
lc[2]))` multiplies the old term by exactly 1.0, which is the IEEE identity;
`(v·1.0)·1.0 == v` bit for bit. The single default light's weight also peaks at
0.5 < 1, so `min(wsum,1)` never binds either (same argument as the existing
comment at `shading_taichi.py:857-862`).

Caveat: bit-identity holds only while that one light is the whole rig. Any
author-added light (or any non-white colour / non-1 intensity on the default
light) breaks exactness — which is the point of the change.

## 5. Entitlement check — do the other stages displace the albedo?

No. All five others either pass through or multiply:

| Stage | Lines | Albedo treatment |
| --- | --- | --- |
| `_stage_unlit` | `shading_taichi.py:783-788` | passthrough, unchanged |
| `_stage_lambert` | `:868-899` | `refl = in_rgb·(amb·env)` then `+= in_rgb·lc·(n·l)·v` — multiplied only; total scaled by `_energy_scale(wsum)` |
| `_stage_phong` | `:903-939` | same scheme + specular lobe (`specular·lc·…` carries no albedo, but nothing displaces `in_rgb`) |
| `_stage_standard` | `:943-986` | `f0`, `k_d` mix/scale `rgb`; every term is a product with `in_rgb` |
| `_stage_physical` | `:990-1093` | standard scheme + sheen compensation `sheen_comp ∈ [0,1]` multiplying the direct terms (`:1055-1062`), clearcoat, transmission |

Two near-misses, neither entitled to be called a base fade:

- Their energy budgets `wsum` ARE radiance-weighted
  (`n_dot_l · _vis_max_component(v) · max(lc)`, `:896-897`, `:936-937`,
  `:983-984`, `:1079-1080`) — i.e. the four younger stages already implement
  exactly the weighting you propose for `wsum`.
- Under the display-referred space (linear off), `_energy_scale`
  (`:140-174`, `scale = 1/max(weight,1)` at `:171-174`) dims the whole reflected
  total — albedo terms included — when total incident weight exceeds 1. It
  approaches but never reaches zero, and its weight counts radiance, so a dim
  rig is not penalised (that was `1a4c9d2`'s explicit goal). Under the default
  linear space it is exactly 1.0 (`:172`).

So `_stage_default` is the **only** stage that can zero the albedo, via
`min(wsum,1)` binding at `:863` whenever contributing weights sum ≥ 1.

## 6. Blast radius — test scenes that will move

Every scene below is default-shaded somewhere (Text/Tex/circuits carry no
material) AND lit by ≥ 2 lights with intensities ≠ 1. The ambient row alone
guarantees movement: today it contributes 0.5 to `wsum` regardless of its
intensity; after the fix it contributes 0.5·max(linear(WHITE))·intensity.
Default-stage pixels therefore keep measurably more albedo everywhere.

**`tests/fast/scene.py`** (compared pixel-wise vs `tests/fast/expected_outputs_{cpu,cuda}/`):
lights at `:78-89` — `AmbientLight(WHITE, 0.45)`,
`DirectionalLight(WHITE, 0.85, target ORIGIN)`, `PointLight(BLUE_A, 0.6)`.
Default-shaded mobs: title Text (`:91-93`), all four circuits Circle/Square/
RegularPolygon/Polygon (`:96-108`), Tex formula (`:129`), caption Text
(`:130-132`). (Cube/Icosahedron/Octahedron/faded Cube carry Lambert/Standard/
Basic materials, `:114-126` — those pixels stay put.) **Baseline moves.**

**`tests/full_renders/scenes/`** (six suites, each with committed CPU+CUDA baselines):

| Scene | Lights (with intensity) | Default-shaded mobs |
| --- | --- | --- |
| `materials_and_lighting.py` | Act 1 `:28-35`: Ambient(WHITE, 0.35) + Directional(WHITE, 1.0, shadow_angle 0.4). Act 3 `:172-198`: Point(YELLOW, 2.2, decay 1.0), Spot(BLUE_A, 5.0), RectArea(GREEN_A, 3.0, samples 4 → K=4), Hemisphere(MAROON_A/BLUE_E, 0.5). Act 4 `:226-232`: Ambient(WHITE, 0.45) + Directional(WHITE, 1.0) | title + all eight label Texts (`:37-43`, `:59-97`, `:161-166`, `:240-245`) |
| `solids_and_camera.py` | `:39-51`: Ambient(WHITE, 0.45) + Directional(WHITE, 0.8) + Hemisphere(BLUE_A/MAROON_E, 0.3) | title Text (`:53-58`) and any un-materialled circuits/labels |
| `text_and_media.py` | `:27-33`: Ambient(WHITE, 0.55) + Directional(WHITE, 1.0) | title, captions, glyph text (two ImageMobs excepted, `:61-66` Basic) |
| `manim_compat_and_plots.py` | `:31-37`: Ambient(WHITE, 0.6) + Directional(WHITE, 0.9) | everything (no `set_material` in file) |
| `complex_hierarchy_become.py` | `:47-53`: Ambient(WHITE, 0.55) + Directional(WHITE, 1.0) | everything (no `set_material` in file) |
| `shapes_and_timeline.py` | `:48-54`: Ambient(WHITE, 0.6) + Directional(WHITE, 0.9) | all circuits/shapes except one Basic quad (`:357`) |

**Bottom line: all seven pixel baselines move** (fast CPU+CUDA, full-render
CPU+CUDA). Movement concentrates on Text/glyph/circuit pixels; materialled
solids (Lambert/Standard/Physical/Basic ids ≠ 0) are unaffected because their
stages don't touch `wsum`'s base-fade role. Per CLAUDE.md, re-baselining needs
both device sets, and `ALGAN_UPDATE_FAST_BASELINE=1` writes only the CPU set.

## 7. Corrections and refinements to the diagnosis

Your mechanism, site, and arithmetic check out against the source:

- Ambient/hemisphere/env-SH rows get `ld = n`
  (`shading_taichi.py:679-681`, `:690`, `:703`); `_shading_normal` normalizes
  (`:280`), so `d = n·n = 1` and `w = 0.5·v` — **exactly 0.5 to `wsum`
  independent of intensity** (`:837`, `:856`). Verified.
- `_energy_scale` returns exactly 1.0 under the linear space (`:172`), so `acc`
  is unscaled. Verified.
- Your measured collapse is consistent with the code: with two ambient-like
  rows, output = `Σ lcᵢ · 0.5` — pure light colour — and the per-line
  differences in your numbers come from hemisphere rows blending sky/ground by
  the normal (`h = 0.5 + 0.5·n·up`, `:688-689`), so differently-oriented
  `Line3D` segments land on different greys. *(Reasoned from source; I did not
  re-measure.)*

Three refinements:

1. **Strength is irrelevant to the trigger.** `wsum`'s weight contains no
   `lc` — only geometry (`d⁵`), visibility, and `frac`. Two 0.01-intensity
   lights annihilate the albedo exactly as two 1.0 ones do; intensity only
   changes what little `acc` they leave behind. So "number **and strength** of
   lights" overstates it: the trigger is purely count × orientation. This makes
   the radiance-weighted fix strictly more necessary — it is what puts strength
   into the trigger.
2. **Ambient-like rows are sufficient but not necessary.** A head-on point or
   directional row also contributes its maximum `d⁵·0.5 = 0.5`; two near-head-on
   point lights reach `wsum ≈ 1` with no ambient anywhere. Area lights don't
   (their K samples carry `frac = 1/K`, `lights.py:175-179`, so they sum to ≤
   0.5 — the comment at `shading_taichi.py:846-855` documents exactly this).
3. **A second latent symptom your fix also cures:** because `row_live` gates on
   the *raw* row colour (`:824-831`) and deliberately ignores live modifiers
   (`:816-818`), a fragment outside a spot cone / beyond a range fade currently
   still fades its base by up to 0.5 while `acc` gains nothing — a darkening
   artifact. Radiance-weighting `wsum` by the *evaluated* `lc` zeroes that fade
   too. Note the shadow-fan culling comments lean on the fade's current
   lc-independence (`wavefront_kernels_taichi.py:161-165`,
   `raster_taichi.py:2859-2863`); after the fix those exclusions are merely
   conservative, not wrong, but worth a look.

One completeness gap: the proposed change fixes the kernel stage only. Under
`SAMPLES_PER_PIXEL > 1` or `set_fragment_shading(False)` the sequential torch
`default_shader` still lerps with the old geometric-only weight (see §2) —
probably acceptable (both are non-default paths), but the twin is where a
matching change would go if parity is wanted.
