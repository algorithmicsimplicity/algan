# OX_AREA_SHADOW_AUDIT — how an area light's shadow rays are built

Read-only audit. No file under `algan/`, `tests/` or `benchmarks/` was changed;
no render and no test suite was run. Every claim below was checked by reading
source; line numbers refer to the working tree as of this writing.

---

## Claim 1 — the packed row layout — CONFIRMED

The packed light row is 16 floats, and aux index `i` lands at packed column
`i + 3`.

- Layout, host side (`algan/rendering/raytracing/scene_builder.py:1984-1988`,
  docstring of `_pack_lights`):

  ```
  0:3  RGB radiance (intensity premultiplied)   9  cos outer (spot)
  3    light type id                            10 cos inner (spot)
  4    decay exponent                           11 shadow softness
  5    range (0 = infinite)                     12:15 ground RGB / SH
  6:9  direction                                15 power fraction (1/K)
  ```

- Same layout from the producer side
  (`algan/rendering/lights.py:185-198`, `Light.build_aux`): type 0, decay 1,
  distance 2, direction 3-5, cos outer 6, cos inner 7, shadow softness 8,
  ground/SH 9-11, power fraction 12 — all in **aux** indices.
- The `+3` offset is mechanical: rows are built as `torch.cat((c, a), -1)`
  with `c` exactly 3 wide (`scene_builder.py:2036`), so aux `i` → column
  `i + 3`. Spot checks: `PointLight.build_aux` writes
  `aux[..., 8] = self.shadow_radius` (`lights.py:269`) and both fans read the
  radius at `light_col[..., 11]` (`raster_taichi.py:2870`,
  `wavefront_kernels_taichi.py:2742`). `_blank_aux` writes the power fraction
  at aux 12 (`lights.py:179`) and `_light_eval` reads it at column 15
  (`shading_taichi.py:733`).

One refinement to the claim's wording: "cols 12-14 ground colour / SH" are
radiance-bearing for hemisphere/env-SH only, and `RectAreaLight` has
`_AUX_RADIANCE_COLS = None` (`lights.py:132`), so nothing scales an area row's
aux with opacity — consistent with the claim, just worth stating.

## Claim 2 — RectAreaLight never sets the shadow-radius column — CONFIRMED

`RectAreaLight.build_aux` (`lights.py:599-616`) is, in full:

```python
aux = self._blank_aux(location)
aux[..., 1] = self.decay
aux[..., 2] = self.distance
aux[..., 3:6] = self._directions(location).unsqueeze(-2)
return aux
```

It sets aux 1, 2 and 3:6 only. `_blank_aux` zero-fills
(`torch.zeros(...)` then sets aux 0 and aux 12, `lights.py:170-180`), so aux 8
→ packed column 11 stays **0.0 on every area emitter row**, at every frame.
Nothing else writes area rows: `_pack_lights` copies aux verbatim
(`scene_builder.py:2035`), and `_append_env_sh_light` (`tracer.py:933-958`)
only ever *appends* its own ENV_SH row and never mutates existing ones.

## Claim 3 — area rows emit one hard shadow ray each — CONFIRMED; the docstring claim is FALSE AS STATED

Both fans read the radius from column 11 and open a multi-sample fan only when
it is positive:

- Sheet-route fan, `raster_taichi.py:2865-2906`:
  ```python
  ltype = 0
  radius = 0.0
  if light_col.shape[2] > 3:
      ltype = ti.cast(light_col[tl, li, 3] + 0.5, ti.i32)
      if light_col.shape[2] > 11:
          radius = light_col[tl, li, 11]
  ...
  ns = 1
  b1 = ti.math.vec3(0.0, 0.0, 0.0)
  b2 = ti.math.vec3(0.0, 0.0, 0.0)
  if radius > 0.0:
      ns = SOFT_SHADOW_SAMPLES
      ...b1 = wi.cross(aref).normalized(); b2 = wi.cross(b1)
  ```
- Classic wavefront fan, `wavefront_kernels_taichi.py:2737-2791`: same shape —
  `radius = light_col[tl, li, 11]` (2742), `ns = 1` (2781),
  `if radius > 0.0: ns = SOFT_SHADOW_SAMPLES` (2784-2785).

For an area row `radius == 0.0`, so `ns` stays 1 and the loop body traces
exactly one ray aimed at the cell centre `lp` (`raster_taichi.py:2933-2941`;
`wavefront_kernels_taichi.py:2809-2812`). Confirmed in both fans.

**The docstring** (`raster_taichi.py:2760-2763`):

> "area lights are already expanded into packed sample rows and therefore
> naturally obtain soft visibility by averaging those rows in the material
> shader"

Answer, precisely: **the mechanism it names is real, the effect it promises is
not — false as stated.**

- True part: the material shader does average the rows. `_light_eval`
  (`shading_taichi.py:697-791`) evaluates each extended row independently and
  every built-in stage sums the lights; each area row carries 1/K of the power
  (column 15), so K rows reconstruct one whole light.
- False part: "soft visibility". Each row's own visibility is a single binary
  test at its cell centre (Claim 3), so the sum over rows is a sum of K step
  functions — piecewise constant with at most K+1 levels. That is the measured
  staircase: levels `[0.01, 0.25, 0.52, 0.74]`, flatness 0.87 vs the path
  tracer's monotone ramp at 0.49 (`benchmarks/renderer_audit/REPORT.md`
  §6.7). Averaging rows smooths the *irradiance* away from shadow edges
  slightly (each row shades with its own `wi`), but within the penumbra the
  result is quantized, not soft, at any finite `samples`. It approaches soft
  only as `samples → ∞`.

So "true-but-insufficient" would be generous: what the averaging cannot do is
produce the very thing the sentence claims it produces.

## Claim 4 — six free columns — REFUTED for column 11; CONFIRMED for 9, 10, 12, 13, 14

First half, confirmed: `RectAreaLight.build_aux` leaves aux 6, 7, 8, 9, 10, 11
untouched, so packed columns 9, 10, 11, 12, 13, 14 are 0.0 on every area row.

Second half ("read by nothing anywhere in the codebase") is **wrong about
column 11**. An exhaustive grep of every `light_col[..., n]` read in `algan/`
(and a check that no test/benchmark pokes these columns directly) gives:

| Column | Readers | Guard | Reachable with `ltype == 5`? |
| --- | --- | --- | --- |
| 9, 10 | `wavefront_kernels_taichi.py:183-184`; `shading_taichi.py:780-781` | `ltype == _LT_SPOT` | no |
| 9, 10, 11 | `shading_taichi.py:757-758` (env-SH `by`) | `ltype == _LT_ENV_SH` | no |
| **11** | `raster_taichi.py:2870`; `wavefront_kernels_taichi.py:2742` | **none — read for every extended row** | **yes** |
| 12, 13, 14 | `shading_taichi.py:745-747` (hemisphere ground); `shading_taichi.py:759-760` (env-SH `bz`) | `ltype == _LT_HEMISPHERE` / `_LT_ENV_SH` | no |
| 15 | `shading_taichi.py:733` (power fraction) | none (extended rows) | yes, but not claimed free |

Column 11 is read unconditionally in both shadow fans for any row whose width
exceeds 11 — that read is exactly the hook Design A plans to use, and it is
why claim 4 as written cannot stand. Its *value* on area rows (0.0) is of
course free to redefine; the column is not unread.

Genuinely free under `ltype == 5`: **columns 9, 10, 12, 13, 14 — five scalars.**
(A footnote asymmetry: the raster fan guards the col-11 read with
`shape[2] > 11`, the wavefront fan only with `shape[2] > 3`; harmless while
extended packing is always 16 wide.)

---

## Q1 — full call-site inventory for the fix

**Host-side packers / writers of light rows (all of them):**

1. `RenderLoopMixin._materialize_render_state` (`render_loop.py:2366-2474`)
   — the only `build_aux` caller (`:2444`); expands an area light into K
   emitter rows via `get_sample_positions` (`:2439`), divides colour by K
   (`:2441-2443`), applies `_AUX_RADIANCE_COLS` opacity scaling (`:2445-2456`).
   **This is where Design A/B would write new columns.**
2. `_pack_lights` (`scene_builder.py:1974-2039`) — concatenates RGB + aux into
   `[T, L, 16]`; also synthesizes the compact-packing whole-light fraction row
   for plain point lights sharing a pack with extended lights (`:2028-2033`).
   Called once, from `tracer.py:1462`.
3. `_append_env_sh_light` (`tracer.py:933-958`) — appends one ENV_SH row and
   pads compact packs to 16 columns. Writes cols 9:15 **on its own row only**.
4. Tensor-carrier shims (no packing, but they move `_render_aux` around and a
   fix that changed aux shapes would flow through them):
   `render_loop.py:1027-1033` (arena preflight snapshot), `:1200-1208`
   (batch snapshot re-attachment), `:2329-2334` (prefetch worker shim lights).

**Shadow fans / every reader that would change:**

1. `raster_shadow_trace` (`raster_taichi.py:2722-2985`) — the sheet route's
   sparse event fan; radius read at `:2870`, fan gate at `:2900`, sample loop
   `:2915-2977`. For Design B this is one edit site; for Design A none.
2. Inline fan in `wavefront_shade` (`wavefront_kernels_taichi.py:2653-2874`)
   — the classic wavefront route; radius read at `:2742`, fan gate `:2784`.
3. Deferred kernel `wavefront_shadow` (`wavefront_kernels_taichi.py:2150-2313`)
   — takes `light_pos` but **not `light_col`** (`:2176-2177`): it treats every
   row as a hard point light, no type, no radius. It is compiled out today —
   the tracer always passes `deferred_shadows == 0` (module comment
   `:204-206`; gates at `:2634`/`:2653`). Name it because reviving it would
   silently bypass any per-row visibility semantics.
4. **Monte Carlo megakernel** (`raytrace_kernels_taichi.py`): YES, it traces
   shadow rays from light rows — next-event estimation fires one
   `_transmittance` ray per row to the row position (`:3650-3692`) — but it
   reads **only columns 0-2** (`:3686-3688`). It consumes no type/radius/aux
   data, so neither design changes it and the MC path keeps hard per-row rays
   unless separately edited. Note also SPP > 1 rejects extended lights at
   preflight anyway (`tracer.py:966-978`, surfaced by
   `test_ux_regressions.py::test_monte_carlo_unsupported_features_fail_preflight`).
5. **`sheet_resolve_taichi.py` has no fan of its own — it delegates.** Mode 1
   builds candidate events (`sheet_resolve_taichi.py:139-148`);
   `raster_pipeline.py:2120` launches `raster_shadow_trace` on them; mode 2
   reads back the per-(event, light) `shadow_vis` triples
   (`raster_pipeline.py:2079-2087`). Its only light-row consumer is the shared
   shading stage library via `_light_eval` (`:121`, `:431`).
6. Shading readers (would NOT change, but consume rows):
   `_light_eval` (`shading_taichi.py:697-791`) and through it every built-in
   stage and custom fragment pipeline (`fragment_shaders.py:332`,
   `fragment_stage_library.py:55,103` pass `light_col` through as a template;
   neither indexes columns itself).

**Torch-side (non-Taichi) paths that consume light rows:** exactly one —
per-vertex shading, `RayTracedTrianglePrimitive._shade_vertex_colors`
(`primitives.py:656-705`), which feeds torch shaders
(`material_shaders.py`, `pbr_shaders.py`) per light. It **skips every extended
light** (`primitives.py:668-673`: `if getattr(light_source, "_render_aux",
None) is not None: continue`), so an area row can never reach a torch shader.
Its presence forces fragment shading instead (`tracer.py:593-604`,
`:1191-1196`). No torch path reads aux columns at all.

**Paths not named in the brief:** the two listed above — the deferred
`wavefront_shadow` kernel (never enabled, but a third fan-shaped site) and the
MC megakernel's NEE loop (does trace row shadow rays, RGB-only).

## Q2 — which columns are genuinely free

See the table under Claim 4. Summary: for `ltype == 5`, columns **9, 10, 12,
13, 14** are unreachable (spot-, hemisphere- and env-SH-guarded only);
column **11** is unconditionally read by both fans and is therefore not free
in the sense the claim uses, though its area-row value (0.0) is what defines
today's hard ray. Column 15 (power fraction) is read unconditionally for
extended rows and must keep carrying `1/K`. Checked across Taichi kernels,
`_light_eval`, the torch vertex path (skips extended rows entirely),
`tests/` and `benchmarks/` (no direct pokes of columns 8-14 anywhere outside
`algan/rendering/lights.py`).

## Q3 — is the cell centre entitled to speak for the cell?

No — not for visibility. The row position serves two roles and only one of
them survives scrutiny:

- As **emission origin**, the centre is a legitimate Monte Carlo point of a
  mean (that it should be an area-weighted integral instead is precisely the
  separate falloff defect, REPORT.md §6.7).
- As the **sole target of the visibility query**, it integrates the occlusion
  field over a set of measure zero of the cell. A blocker covering all of the
  cell except its exact centre-point casts full shadow; one covering everything
  but the centre casts none. Nothing averages the interior.

Precedents where one datum already stands for a region, and whether the
extent is recorded:

1. **Column 11 itself.** For point/spot lights it carries the emitting disk's
   world radius (`lights.py:226-228`), for directional lights
   `tan(half-angle)` (`lights.py:353`) — a region extent, explicitly recorded,
   and consumed by both fans to build the golden-angle disk/cone
   (`raster_taichi.py:2900-2932`). The machinery for "one datum = region" is
   already there; area rows simply leave it empty.
2. **The shadow event's pixel footprint.** Under `sec_aa`, the mode-1 build
   stores the sub-pixel world-space basis `event_dp` per event
   (`raster_taichi.py:2827-2831`) and the fan spreads its samples over it — a
   region datum whose extent is recorded and used, not thrown away.
3. **`pixel_world_scale` / `pixel_size_per_t`** rides into every
   `_shadow_occluded` call for epsilon scaling — again an extent that is kept.

What is discarded for area rows specifically: `RectAreaLight._rect_axes`,
`width`, `height` and `k` all exist host-side at expansion time
(`lights.py:586-597`, consumed at `render_loop.py:2439`), and the cell
half-extents follow from them — but `build_aux` records only decay, distance
and normal, so the region information dies at the packing boundary. The row,
as packed, cannot even recover its own cell size.

## Q4 — the two designs

### Design A — set aux 8 to an equivalent cell radius; no kernel change

What changes: one line in `RectAreaLight.build_aux` —
`aux[..., 8] = sqrt(width * height) / (k * sqrt(pi))` (area-matched disk:
cell area `w*h/k²` → `r = sqrt(area/pi)`). Everything downstream already
works: both fans read column 11 unconditionally, open a
`SOFT_SHADOW_SAMPLES`-ray golden-angle fan over a disk ⊥ `wi` centred at the
cell centre, and average occlusion over valid samples
(`raster_taichi.py:2900-2981`, `wavefront_kernels_taichi.py:2784-2874`).
Cost: area-light shadow rays go 1 → `SOFT_SHADOW_SAMPLES` (= 8 default,
`algan/settings/_startup.py:81`) per shaded fragment/event — ×8 on area rows
only; other lights untouched. Also needs the stale docstring at
`raster_taichi.py:2760-2763` rewritten.

What it gets wrong:

- **Solid-angle over-estimation at obliquity.** The true cell subtends
  `Ω ≈ A_cell·cos φ / d²`, where `φ` is the angle between `−wi` and the
  light's surface normal `n` (equivalently: between `wi` and the normal line,
  taken acute). The fan integrates a perpendicular disk of fixed solid angle
  `π r²/d² = A_cell/d²`. So the emitter's apparent size — and hence the
  penumbra width — is over-estimated by exactly **`1/cos φ`**: matched
  head-on, unbounded as `φ → 90°` (a surface lit edge-on gets blur from a
  disk that should have foreshortened to a sliver; note `_light_zero_radiance`
  culls backfaces only, so grazing-lit surfaces do reach the fan).
- **Shape/isotropy error.** The real footprint is a rectangle foreshortened
  along the plane of incidence; the disk blurs equally in every screen
  direction. Wrong penumbra *shape* for non-square cells at any angle.
- **Tiling error.** Disks cannot tile the k×k grid: neighbouring disks
  overlap slightly and corners are unsampled — mild cross-cell double counting
  in the union-of-blockers statistic. Second-order.
- Sample targets leave the light plane (`lp + off` with `off ⊥ wi`); harmless
  for a visibility query (what matters is angular spread from the shading
  point), but worth knowing when reasoning about horizon culls.

Is `calib_lights.json` grazing or head-on? **Head-on.** Light at
`(1.2, 1.8, 2.2)` aiming `(1.2, −0.9, 0)`, so
`n ≈ (0, −0.775, −0.632)`; probe_rect sphere r=0.6 at `(1.35, −0.9, 0)` sits
almost on the aim line. Computed: a top-of-probe shading point has
`φ ≈ 8°` (`1/cos φ ≈ 1.01`); wall points under the probe's silhouette
(front face z = −2.25) have `φ ≈ 20°` (`≈ 1.06`). So on THIS rig Design A's
obliquity error is ≤ ~6% in apparent size — effectively invisible. The error
would be gross only for rigs lit near the light plane's horizon.

### Design B — pack the rectangle basis into the free columns; kernel change

What changes: `build_aux` packs the cell frame — `right` unit axis (3 floats)
plus the two cell half-extents `w/(2k)`, `h/(2k)` — which fits the five free
columns **exactly** (e.g. right → 9, 10, 12; half-extents → 13, 14), leaving
col 11 at 0.0 or spending it as a marker. Both fans get a new branch keyed on
`ltype == _LT_AREA_SAMPLE` that builds sample offsets in the light plane
(`lp + right·u·hu + up·v·hv`) instead of the ⊥-wi disk; the up axis is
recoverable as `n × right` or packed by displacement of another column.
Cost: same ray count as Design A (`SOFT_SHADOW_SAMPLES` per row per shading
point), plus cold recompiles of both shade kernels and two hand-kept
fan implementations to keep in lockstep (the sheet resolve exists precisely
because such duplicated walks drifted once before).

What it gets wrong: essentially nothing geometrically — sampling actual
surface points foreshortens automatically, so penumbra width, shape and
aspect ratio are correct at every angle. Its costs are engineering risk
(two edit sites instead of zero; float accumulation contracts per fan must be
preserved) rather than image error. Like A, it does not touch the MC
megakernel, which stays hard-rayed.

Which is buildable: both; A is strictly smaller and lands entirely in
host/torch code with byte-level confidence, B is the geometrically honest one.
A's known error is quantified above; B's known cost is two kernels' worth of
care.

## Q5 — external invariants

**a. Convergence as `samples → ∞`.** Both designs have it, provided the
extent scales with the *cell*: cell size ∝ `1/k`, so Design A's equivalent
radius → 0 and Design B's half-extents → 0; each fan degenerates to today's
single centre ray, so the limit image equals today's-at-the-same-limit.
Hard-coded extents independent of k would break this.

**b. `samples = 1`.** k=1, the lone cell is the whole rectangle. Today: one
hard ray at the rectangle's centre — clearly wrong (a large softbox casts a
penumbra). Design A: one area-matched disk, `r = sqrt(w·h/π)`; right total
blur *amount* head-on, wrong shape (aspect ignored), over-blurred at
obliquity — approximate, but a genuine penumbra, i.e. closer to truth than
today. Design B: samples the actual rectangle in its plane — correct including
aspect ratio. So B is *right* at samples=1; A is approximately right head-on
and square, and degenerates badly for extreme aspects (a 1.8×0.1 rect becomes
an r≈0.24 disk).

**c. `width=1.8, height=0.1`.** Cells are `1.8/k × 0.1/k` — 18:1. Design A
samples a disk of equal *area*: blur leaks far past the sliver's short edges
and undershoots along its length; it no longer samples the emitting surface it
claims to, though total occlusion statistics stay roughly area-preserving.
Design B samples the exact elongated cells and still does.

**d. Byte-identical non-area lights.** **Design A guarantees it by
construction**: the only edits are inside `RectAreaLight.build_aux` (+docs);
other lights' rows are bit-identical, the fans' arithmetic for them is
untouched (their `radius` values and branches are pre-existing), and the
compact C==3 packing never sees column 11 at all. What could still break it:
(i) editing shared fan arithmetic (loop order, accumulation) while "just"
touching the area branch — the s-loop's summation order is part of the output
contract (`raster_taichi.py:2787-2794`); (ii) a `_taichi.py` source edit
forces recompile — fine after cache clear, but A/B benchmarking against stale
cache would mislead; (iii) indirect effects on chunk sizing (more shadow-ray
state per area row shifts arena peaks → different batch windows → the usual
≤2-channel torch rounding noise near boundaries; geometry unchanged keeps
this unlikely). Design B achieves the same guarantee only by discipline —
every new line gated on `ltype == 5` — since it edits the shared kernels.
Neither design touches `pn_criterion_kernel`, so the fast-math tessellation
caveat does not apply.

## Q6 — what pins today's behaviour

`SHADOWS` default is **False** (`settings.py:2343`). Occurrences found by
grepping `RectAreaLight` / `rect_area` / `LIGHT_AREA_SAMPLE` across `tests/`,
`benchmarks/`, `docs/`:

| Site | Renders? | Pixel-compared? | Shadows on? |
| --- | --- | --- | --- |
| `tests/full_renders/scenes/materials_and_lighting.py:186` (act 3; width 1.8, height 1.0, moves ±RIGHT·2.4 — the calib rig's twin) | yes | **yes** — `tests/full_renders` suite vs `expected_outputs_cpu/` + `expected_outputs_cuda/` (skips under CI unless `ALGAN_RUN_FULL_RENDERS=1`) | **yes** (`:22` `SETTINGS.raytracing.set(shadows=True)`; the scene docstring says it is the only scene that turns shadows on) |
| `tests/unit_tests/test_ux_regressions.py:1260` | renders until preflight raises `UnsupportedFeatureError` (SPP=4 + extended light) | no | n/a (rejected before render) |
| `tests/unit_tests/test_ux_regressions.py:678` | no (config validation only) | no | no |
| `tests/fast/` | scene has ambient/directional/point lights only, deliberately no shadows, no area light (`tests/fast/scene.py:26-27,44`) | yes, but pins nothing here — it is, however, the suite that certifies non-area bytes didn't move | no |
| `tests/unit_tests/test_raytracing_unit.py` (e.g. `:250`, `:713`) | kernel harnesses with compact 3-col rows | numeric asserts, no area rows | no |
| `benchmarks/_ext_lights_check.py:170` (`scene_area`, samples=16) | PNGs into `benchmarks/_tc_out/` | no — eyeball validation | yes (`set_ray_traced_shadows(True)`) |
| `benchmarks/renderer_audit/scenes/calib_lights.json` (`rect_area`), rendered by `algan_render.py` | yes | vs three.js captures (audit harness, not committed baselines) | **yes by default** (`algan_render.py:267,289`; `--no-shadows` flag) |
| `benchmarks/renderer_audit/TASK_area_light_shadow_banding.md`, `REPORT.md:73`, §6.7 (`:1269-1338`), `shadow_band_probe.py` | measurement texts/harness documenting the defect | scanline numbers quoted in REPORT.md | yes (probe rig) |
| `docs/source/advanced_user_tutorials/lighting_and_shadows.rst:212-229` (`.. algan:: LightingRectAreaLight`, samples=16) | during docs build only | no committed baseline | yes (`:221`) |
| `docs/.../performance_and_quality.rst:166`, `renderer_limitations.rst:303,513` | prose (MAX_SHADOW_LIGHTS slot accounting) | — | — |

Net: the only committed pixel baselines that will move are
`tests/full_renders`' act-3 frames (both device sets; CPU regenerable here,
CUDA needs a CUDA machine). Everything else is eyeball or audit harnesses.

## Q7 — independence from the falloff fix (REPORT.md §6.7)

§6.7 finds the same mean-vs-integral defect in the *falloff*: each sample
carries `1/K` of the power with no area element and (at `decay = 0`) no
distance term, so Algan floods the wall where three.js and the path tracer
pool under the rectangle (32× under the light, 145× at the edge; median 67×
against the path tracer); the specified correction — give each sample
`L · (A/K) · cosθₑ · cosθᵢ / d²` — "changes what `intensity` means for every
existing `RectAreaLight`", which is why it wants its own change and baselines.

Three sentences: the shadow fix and the falloff fix operate on disjoint
halves of the same rows — visibility per row lives in the two fans (and the
columns that drive them), while radiance/falloff per row lives in `_light_eval`
and the host packing — so a shadow-only fix is mechanically independent and
needs no redo when the falloff fix lands. They share exactly one dependency:
both key on the same k×k cell decomposition, so the falloff fix must keep the
same grid (as its specification does — `A/K` area elements on the same
samples), otherwise the shadow side's cell-extent encoding would need
repainting. Practically they remain two changes with two baselines, landing on
the same scene (`materials_and_lighting` act 3 / `cl_rect_area`) — REPORT.md
already schedules them that way ([4] is scoped to the shadow half alone).

---

## What I did not check

- **Everything here is source-reading; nothing was executed.** Per the brief I
  ran no render and no test, so no number below eye level (angles, factors,
  the 8°/20° incidence figures) was verified by running code — they are
  arithmetic on coordinates read out of `calib_lights.json`.
- `_shadow_occluded` / `_transmittance` internals were trusted from their call
  sites and comments (opacity accumulation, epsilon tiers); I did not read
  them line-by-line, so statements about what one occluded sample returns are
  second-hand.
- The MC megakernel's `light_intensity` scalar plumbing (where it comes from
  host-side) was not traced; I read only its NEE loop.
- `three_render.mjs` and `shadow_band_probe.py` were not opened; three.js-side
  claims are cited from `REPORT.md`/`TASK_area_light_shadow_banding.md`, not
  independently checked.
- The `sec_aa` interaction with a future soft area fan (fan spread over
  sub-pixel origins, `raster_taichi.py:2907-2923`) is reasoned from the code,
  not exercised; in particular whether Design A's disk should be rebuilt per
  sub-pixel origin (wi changes per origin) is noted as a design question, not
  settled.
- CUDA-specific behaviour of anything described (including whether the extra
  fan rays change occupancy-related truncation counters in practice) is
  uncheckable from this machine and was not attempted.
- Git history (when the `shape[2] > 11` guard appeared in the raster fan but
  not the wavefront one) was not investigated.
