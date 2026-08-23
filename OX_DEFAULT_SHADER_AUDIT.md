# OX_DEFAULT_SHADER_AUDIT — replacing Algan's default 3-D shader

Read-only audit of the plan in `ox_brief.md`. No file was edited, created or
deleted except this report; no test or render was run. Everything below is from
reading source; places where I reasoned rather than read are labelled
**[reasoning]**.

Citations are `file:line` as of this working tree.

---

## 0. Where the brief is wrong or self-contradictory (read this first)

1. **The brief states Manim's shading two different ways, and the safety of the
   id-0 shadow cull flips between them.** §3 says the faithful form is
   `rgb = rgb + light_term` — an *untinted scalar* add. That is what Manim
   actually does: `algan/external_libraries/manim/utils/color/core.py:1543-1547`
   (`to_sun = normalize(light_source - point)`, `light = 0.5 * dot(n, to_sun)**3`,
   negated halves, `shaded_rgb = rgb + light` — a scalar, no colour). Q4c instead
   posits a per-light contribution *tinted by `light_colour*`. These are not the
   same shader, and Q4c's verdict depends on which one is built (see §4c).
   A second, unavoidable divergence: Manim has exactly one hardcoded light; an
   Algan stage loops over the Scene's lights (`_stage_*` contract,
   `algan/rendering/raytracing/shading_taichi.py:741-747`), so "faithful" is only
   defined for a one-light rig.
2. **Q6 as posed cannot be satisfied literally.** An empty
   `get_shader_param_values()` means `flat_shading` never reaches slot 10,
   because the block is written by name out of parameters the mob registered via
   `set_shader` (details in §6). The fix is small and stated there.
3. **Q1's candidate list contains a trap:** `basic_pbr_shader` must NOT be used
   as the signature-length reference (it has 11 parameters, not 9). See §1.
4. **§2's mechanism claim is correct** (`TrianglePrimitive.__init__` reads
   `SETTINGS.style.default_shader` at
   `algan/rendering/primitives/triangle_primitive.py:187-189`; everything else
   follows from that single read), but note the fallback currently fires for
   `shader=None` only; `_shader_material_id(None)` maps to unlit id 1, never to
   id 0 (`algan/rendering/raytracing/settings.py:2606-2613`) — relevant to §4a.
5. **Q3's list is confirmed but incomplete in two ways**: `shapes_2d.py`'s site
   is also the route for every `Polyhedron` solid (Cube, Platonic solids,
   ConvexHull3D), and `ImageMob` reaches the `Surface` site. Details in §3.
6. Minor: §4's premise "id 0 is freed by (1)" holds only because the *settings*
   default is replaced in the same change; if `default_shader` were deleted
   without repointing `TrianglePrimitive`, the fallback would pass `None`, which
   packs as id 1 unlit (`settings.py:2611-2613`) — silently unlit mobs. The plan
   as written avoids this; do not split the two edits.

---

## Q1. Complete call-site inventory for deleting `pbr_shaders.default_shader`

Searched `algan/`, `tests/`, `benchmarks/`, `docs/` for `default_shader`
(import, name, docstring, `:func:` role). There is no top-level `examples/`
directory. Complete list:

| # | Site | What it does | Must become |
|---|------|--------------|-------------|
| 1 | `algan/rendering/shaders/pbr_shaders.py:160-212` | The definition (9 fixed params + body); module docstring names it at `:9` | Delete both; rewrite docstring line |
| 2 | `algan/__init__.py:149` | Imports it into the root namespace | Remove import |
| 3 | `algan/__init__.py:153` | `SETTINGS.style.set(default_shader=default_shader)` — the write that installs it as the default | Replace with `SETTINGS.style.set(default_material=DiffuseMaterial())` |
| 4 | `algan/__init__.py:363-366` (`__all__`) | Not listed literally; `_is_root_export` (`:350-360`) admits it because it is callable with `__module__` starting `"algan."`. Deleting #2 removes it from `__all__` automatically | Nothing (verify with `test_ux_regressions`) |
| 5 | `algan/animatable_base/mob_materials.py:14` | Import | Replace (see signature-reference row below) |
| 6 | `algan/animatable_base/mob_materials.py:96-98` | **Signature-length reference #1**: `num_shader_independent_params = len(inspect.signature(default_shader).parameters.keys())`, then slices the custom shader's extra params off that count (`:99-107`) | New reference (below) |
| 7 | `algan/rendering/raytracing/primitives.py:629-632` | **Signature-length reference #2**: `_ordered_shader_param_values` does `num_fixed = len(inspect.signature(default_shader).parameters)` then `extra_names = list(sig.keys())[num_fixed:]` | New reference (below) |
| 8 | `algan/rendering/shaders/fragment_shaders.py:119` | Imports it inside `_builtin_shader_to_stage` | Remove import |
| 9 | `algan/rendering/shaders/fragment_shaders.py:122` | Maps `default_shader → STAGE_DEFAULT` so `set_fragment_shader(default_shader)` resolves | Replace with `manim_shader → STAGE_MANIM` (or drop the entry; nothing else maps to id 0 after the change) |
| 10 | `algan/rendering/shaders/material_shaders.py:4` | Module docstring cites `default_shader` as the convention-defining signature | Repoint the citation (to `basic_material_shader` or a named constant) |
| 11 | `algan/rendering/raytracing/settings.py:2518,2521` | `_build_core_shader_ids` imports it and maps `default_shader: 0` | `manim_shader: 0` (import swap); update comment `:2508-2509` |
| 12 | `algan/rendering/raytracing/shading_taichi.py:18` | Module docstring calls it "the legacy diffuse" first stage | Rewrite |
| 13 | `algan/rendering/raytracing/shading_taichi.py:755` | `_stage_default`'s own docstring ("default_shader: diffuse lerp …") | Becomes `_stage_manim`'s docstring |
| 14 | `tests/unit_tests/test_materials.py:528-529` | Comment only ("default_shader does, so it's exercised by the render pipeline") — no code use | Update or delete comment |
| 15 | `docs/source/advanced_user_tutorials/settings.rst:181-184` | ``default_shader`` style-setting section | Rewritten per Q2 (listed here for completeness) |
| 16 | `docs/source/advanced_user_tutorials/shaders_and_materials.rst:201,208,212,227,228` | Five Sphinx `:func:` roles presenting it as the shipped diffuse vertex shader and the signature template | Repoint to `basic_material_shader` / new constant / `manim_shader` |
| 17 | `docs/source/advanced_user_tutorials/shaders_and_materials.rst:286` | Stage table: `STAGE_DEFAULT` "Resolved from ``default_shader``" | Row becomes `STAGE_MANIM` / manim wording |
| 18 | `OX_LIGHTING_AUDIT.md:107,207` | Prior audit prose citing `pbr_shaders.default_shader` behaviour (and `:81,181,193` cite `_stage_default`) | Historical document; annotate rather than rewrite |

Not call sites, checked to be sure: `benchmarks/` has zero hits;
`algan/manim_defaults.py` uses `basic_material_shader`, not `default_shader`
(`algan/manim_defaults.py:178,216`).

### The two signature-length references — what to replace them with

Both sites need "the number of shader-independent leading parameters", i.e. the
length of the canonical fixed prefix. Read from source:

- `default_shader` has exactly 9 parameters: `memory, vertex_location,
  vertex_normal, albedo_color, camera_location, light_origin, light_color,
  light_intensity, ambient_light_intensity` (`pbr_shaders.py:160-170`).
- `null_shader` has exactly the same 9, nothing more (`pbr_shaders.py:215-225`)
  — **valid** replacement.
- `basic_material_shader` has exactly the same 9, nothing more
  (`material_shaders.py:248-258`) — **valid** replacement.
- `basic_pbr_shader` has those 9 **plus** `smoothness`, `metallicness` =
  11 total (`pbr_shaders.py:31-43`) — **INVALID**. `len()` would return 11 and
  both sites would slice two real material parameters off every custom shader's
  extras (silently: `mob_materials.py:99-101` would stop registering
  `smoothness`/`metallicness` as animatable).

Every torch material shader in `material_shaders.py` repeats the same 9-name
prefix (the convention is documented at `material_shaders.py:3-10`), so any of
the two valid candidates gives identical counts today. **Recommendation**
[reasoning]: do not pick another function at all — introduce one named constant
next to the convention, e.g. `SHADER_FIXED_PARAM_COUNT = 9` (or better, a
`_FIXED_PARAM_NAMES` tuple) in `material_shaders.py`, and have both sites plus
the docstring cite that. It cannot be deleted out from under the API, it cannot
grow a parameter by accident without a test noticing, and it removes the only
remaining reason for `mob_materials.py` to import from `pbr_shaders`.
Note `tests/unit_tests/test_materials.py:47` already uses
`basic_material_shader` as its reference (`_NUM_BASE_PARAMS`), so deleting
`default_shader` leaves that test intact either way.

---

## Q2. Complete call-site inventory for renaming `SETTINGS.style.default_shader`

The field is declared on the dataclass at `algan/settings/style_settings.py:42`
(`default_shader: object | None = None`), described in the module docstring at
`:7`. Complete inventory:

**Writes**
- `algan/__init__.py:153` — `SETTINGS.style.set(default_shader=default_shader)`
  (installs the process default at import).
- `algan/manim_defaults.py:216` —
  `SETTINGS.style.set(default_shader=basic_material_shader)` inside
  `apply_manim_defaults` (plan item 5 changes this to `ManimMaterial()`).

**Reads**
- `algan/rendering/primitives/triangle_primitive.py:188` — the only engine read:
  `if shader is None: shader = SETTINGS.style.default_shader`.

**Tests**
- `tests/unit_tests/test_manim_defaults.py:25` — fixture saves
  `(style.default_shader, ...)`.
- `tests/unit_tests/test_manim_defaults.py:31-33` — restores it via
  `SETTINGS.style.set(default_shader=saved_style[0], ...)`.
- `tests/unit_tests/test_manim_defaults.py:103` — asserts
  `SETTINGS.style.default_shader is basic_material_shader` (this is the
  assertion the brief already knew about).

**Docs**
- `docs/source/advanced_user_tutorials/settings.rst:180-184` — the
  ``default_shader`` settings-table entry ("Defaults to ``None`` (Algan's
  built-in shading)").

**Round-trip / construction machinery (what breaks on a rename):**

- `StyleSettings` is a `@dataclass(Settings)` (`style_settings.py:35-36`). The
  base machinery is field-name-driven: `Settings._check_keys`
  (`algan/settings/abstract_settings.py:105-115`) raises
  `AlganConfigurationError` ("Unknown StyleSettings setting 'default_shader'.
  Did you mean 'default_material'?") for any stale keyword — so a missed call
  site fails loudly at `set()`, not silently. Good.
- `set()` goes through `dataclasses.replace` (`abstract_settings.py:121-134`),
  which is rename-safe once all callers are renamed.
- `__init_subclass__` auto-generates a per-field setter
  `set_default_shader` (`abstract_settings.py:63-80`); after the rename it will
  be `set_default_material`. No caller of `set_default_shader` exists anywhere
  (grepped; none found), so nothing to migrate.
- Snapshot/restore: `SETTINGS.snapshot()/restore()` capture whole section
  objects and replay them with `self.style.set(**snapshot.style.to_dict())`
  (`algan/settings/root_settings.py:82-101`; `to_dict` at
  `abstract_settings.py:179-183`). Keys are the dataclass field names, so the
  round trip is internally consistent across a rename within one process. There
  is no on-disk serialization of settings anywhere (snapshot objects are
  in-memory; `SettingsSnapshot`, `root_settings.py:27-35`).
- Construction: the only constructor call is `StyleSettings()` with no arguments
  (`root_settings.py:58`). **Nothing constructs it positionally**, so field
  order survives untouched; renaming does not move the field.
- No preset machinery involves `style` (presets exist for video/raytracing;
  `LD.as_mutable()` is applied to video, `root_settings.py:59`).

---

## Q3. Which Mobs actually consume the default

The single fallback point is `triangle_primitive.py:187-189` (confirmed). Every
route below passes `shader=self.shader`, and `Mob.__init__` sets
`self.shader = None` (`algan/animatable_base/mob.py:258`), so any Mob whose
author called neither `set_shader` nor `set_material` arrives as `None`.
`RayTracedTrianglePrimitive(TrianglePrimitive)` (`algan/rendering/raytracing/primitives.py:336`)
and `LogicalPNTrianglePrimitive(RayTracedTrianglePrimitive)` (`:1049`) inherit
the fallback through `*args/**kwargs` (`:1168-1175`).

Confirmed sites and the user-facing classes behind each:

1. **`algan/mobs/surfaces/surface.py:3068-3086`** — `LogicalPNTrianglePrimitive`,
   `shader=self.shader` at `:3073`. Classes: `Surface` itself (parametric), and
   everything built on it: `Sphere`, `Cylinder`, `Cone`, `Torus`, `Dot3D`
   (`shapes_3d.py:1093`), `Line3D` (`:1148`), the cylinder/cone parts of
   `Arrow3D` (`:938`), point-cloud dots (`point_cloud.py:167` delegates to child
   spheres), and **`ImageMob`** (`image_mob.py:35` — `class ImageMob(Surface)`).
   `Surface` sets no shader of its own (grepped `surface.py` for
   `set_shader/set_material`: none).
2. **`algan/mobs/pn_mesh.py:85-95`** — `LogicalPNTrianglePrimitive`,
   `shader=self.shader` at `:90`. Class: `PNMesh` only — deliberately internal,
   "the universal morph medium" (`pn_mesh.py:1,14-20`).
3. **`algan/mobs/three_d_models/mesh.py:337-374`** — `effective_triangle_primitive()`
   (= `RayTracedTrianglePrimitive`, `algan/settings/renderer_settings.py:17,26-28`),
   `shader=self.shader` at `:367`. Classes: `TriangleMesh` and `ThreeDModelMob`
   (.glb/.fbx imports).
4. **`algan/mobs/shapes_2d.py:508-534`** — `effective_triangle_primitive()`,
   `shader=self.shader` at `:532`. Classes: `TriangleTriangulated`,
   `TriangleVertices` (internal holder, `test_render_coverage_audit.py:57`),
   `QuadTriangulated` (two `TriangleTriangulated` children, `:556-571`), and —
   the addition to your list — **every `Polyhedron` solid**, because
   `Polyhedron` builds its faces from `TriangleTriangulated`
   (`algan/mobs/shapes_3d.py:1479`; classes: `Cube`, `Tetrahedron`,
   `Octahedron`, `Icosahedron`, `Dodecahedron`, `ConvexHull3D` at `:1982`).
   So the flat-mesh family reaches the default via shapes_2d, not via Surface.
   **One more route through this site (amendment):**
   `TriangulatedBezierCircuit` builds its tiles as `TriangleTriangulated`
   (`triangulated_bezier_circuit.py:1014`), which pulls in **plots.Arrow**
   (`plots.py:41`), `FunctionPlotMob`'s curve (`plots.py:301`), the
   **triangulated glyph text variants** (`text.py:309`; classes
   `TexTriangulated` / `TextTriangulated`, `text.py:862-877`) and the morph
   conversions (`morph_conversions.py:306`). Plain `Text`/`Tex` keep
   `triangulated = False` (`text.py:209`) and stay on bezier circuits.
5. **`algan/mobs/nonplanar_circuit.py:763-812`** (`build_patch_primitive`),
   `shader=circuit.shader` at `:792` into `LogicalPNTrianglePrimitive`. Reached
   from `build_render_primitives` for `plan.mode != "stroke"` (`:937-941`),
   which `BezierCircuitCubic.get_render_primitives` calls whenever
   `self._nonplanar_plan is not None` (`bezier_circuit.py:1042-1052`).

### (a) Does `BezierCircuitCubic` ever reach the triangle fallback?

**Yes, on exactly one route.** A circuit renders as triangles iff, at
construction, `classify_circuit(control_points, filled)` returns a non-None
plan (`bezier_circuit.py:435,532`) AND the circuit is filled — then each closed
sub-path becomes logical PN patches via `build_patch_primitive`
(`nonplanar_circuit.py:905-941`). Conditions, read from
`nonplanar_circuit.py:125-127,444-462`: `ALGAN_NONPLANAR_CIRCUITS` enabled
(default True), ≥4 control points in complete groups of 4, some sub-path's
covariance eigen-ratio above `PLANARITY_TOLERANCE` (i.e. genuinely
non-coplanar control points at construction time), `filled=True`. An unfilled
non-planar circuit takes `build_stroke_primitive` instead
(`nonplanar_circuit.py:919-935`), which builds a **circuit** primitive
(`circuit.render_primitive(...)`, `:858`) — no triangle fallback, and circuits
are never fragment-shaded anyway (the wavefront shades `htype == 1` only,
`wavefront_kernels_taichi.py:2846`).

**A texture grid is NOT a route.** The grid is colour samples laid over the
circuit's plane frame (`bezier_circuit.py:456-474`) consumed as
`texture_points` colours by the circuit primitive (`num_texture_points`,
`nonplanar_circuit.py:871-888`); it never produces triangles.

### (b) What `mobs/shapes_2d.py`'s primitive is for

`TriangleVertices.get_render_primitives` (`shapes_2d.py:508-553`) is the flat
triangles route: `TriangleTriangulated`, `TriangleVertices`,
`QuadTriangulated`, and all `Polyhedron` faces (via `shapes_3d.py:1479`). They
are authored as 2-D shapes but they are genuine 3-D triangle meshes (a `Mob`
with three corners; `QuadTriangulated` even parents two of them) — nothing
confines them to z=0; they simply default there.

### (c) Any Mob for which Lambert-lit would be clearly wrong?

**[reasoning]** Today's default is *already lit* (a diffuse lerp toward the
light colour, `pbr_shaders.py:198-212`), so nothing transitions unlit→lit;
everything transitions lerp-shaded→Lambert. Two candidates for "clearly wrong":

- The flat triangle mobs of (b) sitting in the authoring plane facing the
  camera: under Lambert + Algan's stock rig (one white PointLight offset up/
  right/out, `algan/__init__.py:226-233`) their brightness becomes
  orientation-dependent; rotate one to face away and it goes black. That is a
  look change users will see, arguably wrong for "2-D" content — but it is
  precisely the change being made, and `use_manim_defaults` scenes get
  `ManimMaterial` instead.
- **Why today's look is flat on this geometry — mechanism (amendment, read):**
  these mobs carry **all-zero vertex normals**. `TriangleVertices` stores
  `normals=None` unless passed (`shapes_2d.py:477-480`);
  `get_render_primitives` substitutes zeros and its re-normalization of zeros
  stays zero (`shapes_2d.py:519-530`). The kernel falls back to the geometric
  face normal for degenerate vertex normals
  (`raytrace_kernels_taichi.py:1633-1641`), so stages do receive *a* normal —
  but `_stage_default`'s weight is fifth-power (`d⁵·0.5`,
  `shading_taichi.py:795-796`), which keeps off-axis light nearly inert, while
  Lambert multiplies by `ambient + n·l` with `AMBIENT_STRENGTH_LINEAR = 0.01`
  (`shading_taichi.py:838-848`, `:107`). A camera-facing fill lit from well
  off its normal therefore drops to a small fraction (~20%) of its authored
  colour **[reasoning from the cited formulas — not a measured render]**. This
  lands hardest on the *decorative* members of this family — triangulated
  glyph fills, `plots.Arrow`, bare `TriangleTriangulated`,
  `QuadTriangulated` — rather than on solids, where the change is intended.
- `PNMesh` mid-morph: internal, not user-facing; its appearance while it is the
  morph carrier changes. Acceptable, worth a sentence in the PR.

Everything else in the list (solids, imported meshes, parametric surfaces) is
exactly the population a Lambert default is *for*.

One structural side effect worth stating: batching keys on shader identity,
`get_batch_identifier` returns `f"{self.__class__}_{id(self.shader)}"`
(`triangle_primitive.py:191-192`). After the change, unmaterialized mobs carry
the same function object as explicit `DiffuseMaterial` mobs and merge into one
collection where today they batch separately. **[reasoning]** This should be
pixel-neutral (mesh_keys/surface ids keep surfaces distinct,
`shapes_2d.py:545`), but it changes collection membership and is easy to
mistake for a regression in a diff video.

---

## Q4. Is in-kernel material id 0 safe to repurpose?

### (a) Zero-initialised / zero-padded material-id arrays

- **Per-primitive packing never writes zeros.** `_pack_material` builds the id
  row with `torch.full((1, N), _shader_material_id(shader))`
  (`raytracing/primitives.py:726-728`); `_pack_frag_pipeline` likewise
  `torch.full(..., pid)` (`:770`). The value always comes from the primitive's
  actual shader.
- **An unset shader never lands on 0.** `_shader_material_id(None)` returns 1,
  and unknown shaders return 1 (`settings.py:2606-2613`). Id 0 can only appear
  because a core shader mapped to 0 was genuinely present.
- **The empty-scene placeholder is zeros but is never read.**
  `scene_builder.py:1567` writes `scene["tri_mat_id"] = torch.zeros((1, 1),
  dtype=torch.int32)` when the batch has no triangles. Nothing dispatches on
  it: `_frag_pid_mask` returns `ALL_PIDS` unless the geometry type is active
  AND the merge-time id list exists (`tracer.py:3611-3623`), and kernels index
  `tri_mat_id` only for prims that exist.
- **The memory-trim banding treats 0 as lit, not unset.**
  `_build_mem_trim` sets `_UNLIT = 1` and computes
  `lit = (tri_mat_id != _UNLIT).any(0)` (`scene_builder.py:918-921`) — id 0
  gets a material block reserved (`needs_mat = lit`, `:935`). Consistent with
  repurposing 0 to another lit stage.
- **Custom-pipeline blocks are zero-padded, but can never be dispatched as 0.**
  `_pack_frag_pipeline` zero-fills the parameter block (`primitives.py:777`),
  and those primitives carry pid ≥ `_USER_PIPELINE_BASE = 6`
  (`shading_taichi.py:95`). The padding rule that governs slots
  (`shading_taichi.py:48-51,66-77`) is about the *parameter block*, not the id.
- **Parameter-block defaults** `_MAT_DEFAULTS` (`settings.py:2536-2572`) are
  slot defaults shared by every built-in id; nothing about them is id-0-specific.
- **STBVH build carries no material ids** (grep over `stbvh.py`: no hits).
- **Can such a slot reach `_run_frag_pipeline`?** The kernel reads
  `pid_arr[f % ..., prim]` (`shading_taichi.py:1260,1313`) for prims produced
  by traversal/emission, all of which were packed by `_pack_material`/scene
  merge from real values. I found **no path where a zero arrives as "unset"**
  — the only literal-zero producer is the empty-batch placeholder, which no
  prim index can reach. **[conclusion from the reads above]**

### (b) The `pids_present` bitmask

Read in full: `ALL_PIDS = -1` is the "assume everything" sentinel
(`shading_taichi.py:1132-1135`). `solo_pid` (`:1138-1171`): negative → -1
(runtime switch); a mask of **0** yields an empty `live` list → -1 (no special
meaning, just "not solo"); a mask whose only bit is bit 0 returns 0 and the
solo dispatch calls `_BUILTIN_STAGE_FNS[0]` unconditionally (`:1232-1246`) —
generic indexing, no special case. The gated loop compiles a branch per
reachable id with `mid != _MID_UNLIT and ((pids_present >> mid) & 1)`
(`:1322-1330`) — id 0 is branched like any other. The mask is built from
`torch.unique` over the very array the kernel indexes (`tracer.py:3594-3623`;
`scene_builder.py:1800-1803`), so bit 0 set ⇔ some primitive really carries id
0. **No special meaning attaches to mask 0 or to bit 0 anywhere.**

### (c) The zero-radiance shadow-cull exclusion — the highest-value question

The predicate, read from `_light_zero_radiance`
(`wavefront_kernels_taichi.py:147-197`): it returns 1 when light `li`'s
*evaluated radiance at this fragment* is exactly zero from geometry alone —
range fade (`:169-175`), spot-cone outer angle (`:176-190`), area-sample
backface (`:191-196`) — reproducing `_light_eval`'s factors bitwise. Its
docstring states the validity condition precisely (`:157-163`): skipping the
shadow fan (leaving `vis[li]` at its all-lit default) "cannot influence any
stage whose vis-multiplied terms all carry `lc` as a factor"; explicitly NOT
valid for `_stage_default`, "its base-colour fade accumulates a vis-weighted
`w` even at `lc == 0`". Both call sites gate `fan_geom = 1` on
`pid != _MID_DEFAULT` for built-in pids (`wavefront_kernels_taichi.py:2646-2658`;
`raster_taichi.py:2855-2863`) and skip the fan when it fires
(`wavefront_kernels_taichi.py:2750-2754`; `raster_taichi.py:2891-2894`);
zero-RAW-colour rows are skipped separately (`...2704-2709`; `raster_taichi.py:2895-2898`).

Why `_stage_default` needs the exclusion (read from `:750-817`): its output is
`out*(1 - min(wsum,1)) + acc`, where the fade weight `w = d⁵·0.5·v` (`:796`)
carries visibility but **not** `lc` — so at `lc == 0` the fan still moves the
base colour, and assuming "lit" would darken umbrae wrongly.

Now judge the proposed `_stage_manim` against the predicate, not the comment:

- **If tinted as Q4c states** (`offset += light_term * lc * v` positive;
  `offset += 0.5 * light_term * lc` negative): every vis-multiplied term
  carries `lc`, and the negative term never reads `vis` at all. Both branches
  vanish at `lc == 0`. Zero-RGB rows also contribute nothing, matching the
  fan-site skip. **The exclusion becomes newly UNNECESSARY**: `fan_geom = 1`
  is valid for the new id 0, and the `!= _MID_MANIM` special case can be
  deleted at both sites (keeping it is merely conservative). Not insufficient.
- **If faithful to Manim as §3 specifies** (untinted scalar `light_term`,
  no `lc` factor): the positive branch is a vis-weighted term *without* `lc` —
  exactly `_stage_default`'s disease. Worse, a despawned light's row would then
  need the same explicit `row_live` gate `_stage_default` has
  (`shading_taichi.py:782-791`), because the fan site skips zero-RGB rows for
  built-ins (`wavefront_kernels_taichi.py:2704-2709`) and an untinted stage
  would otherwise disagree with a batch boundary. **The exclusion remains
  REQUIRED**, and the stage must replicate the row-liveness gate.

So the answer flips on the design decision the brief left ambiguous (§0.1).
Whichever is chosen, the choice must be recorded in `_light_zero_radiance`'s
docstring (`:159-163`) and the two call-site comments
(`wavefront_kernels_taichi.py:2652-2656`; `raster_taichi.py:2859-2861`), which
currently explain the exclusion in terms of the old stage's base fade.

### (d) Anything else keyed off id 0

- **Monte Carlo megakernel**: routes through the same dispatcher —
  `_shade_tri_hit` → `_run_frag_pipeline`
  (`raytrace_kernels_taichi.py:1654-1696`, import at `:81`). No separate
  id-0 logic.
- **Sheet resolve**: gates event acceptance on `_MID_UNLIT`, not 0
  (`sheet_resolve_taichi.py:440-441`); packs the pid into `event_msk` as
  `pid_e << 8` (`:472`) and raster reads it back `>> 8`
  (`raster_taichi.py:2842`) — pid 0 ORs nothing into the low bits, harmless.
  Normal-computation skip is `_MID_UNLIT`-keyed too
  (`sheet_resolve_taichi.py:409-411`; `wavefront_kernels_taichi.py:2854-2857`).
- **Legacy sorted wavefront** (unsupported): buckets by pid and resolves
  id < 6 through `builtin_pipeline_fn` → `_BUILTIN_STAGE_FNS[pid]`
  (`tracer.py:3696-3717`; `shading_taichi.py:1174-1185`). Renaming element 0
  carries automatically; the bucket key `(1 << 8) | pid` is value-agnostic.
- **Tests**: `test_frag_pid_gate.py` asserts mask/solo mechanics using
  `_MID_STANDARD`/`_MID_UNLIT` symbols (`:54-56,60,75-92`) — id-value agnostic,
  unaffected. `test_fragment_shaders.py:138` uses a literal `0` in a fake
  `tri_material_ids` tuple as "some built-in id" — still true after the
  repurpose. `test_color_decode_boundary.py:149,169,188` uses id values 0 /
  `_USER_PIPELINE_BASE` purely as built-in-vs-custom markers for
  `_decode_merged_colors` — unaffected (0 stays built-in).
- **Serialization**: nothing serializes material ids. `SettingsSnapshot` holds
  settings sections only (`root_settings.py:27-35`); `algan/project.py` has no
  shader/mat-id references (grepped).
- **Profiling/debug**: `_FRAG_PID_LAST` (`tracer.py:95`) records masks, not id
  meanings; truncation counters are id-blind. The only "tooling" that knows id
  0 was the default is prose: `_light_zero_radiance`'s docstring and the two
  comments cited above, plus the pid table in the shading_taichi module
  docstring (`shading_taichi.py:25-32`) and `settings.py:2508-2509`.

**Verdict: repurposing id 0 is safe.** Every consumer treats ids as opaque
dispatch values; the two behavioural special cases are the fan-geom exclusions,
whose correctness under the new stage is decided by the tinted-vs-untinted
choice (§4c).

---

## Q5. Checklist for adding one new built-in stage (+ torch shader + Material)

Traced what `lambert_shader` / `STAGE_LAMBERT` / `DiffuseMaterial` touch.
Your expected list is confirmed and complete except for the two call-site
comments/exclusion sites from Q4c and the coverage-audit test. Full checklist:

1. **`algan/rendering/shaders/material_shaders.py`** — new `manim_shader`
   torch twin. Signature MUST open with the canonical nine params
   (convention documented at `:3-10`); declare `flat_shading: float = 0.0` as
   its one extra param (needed for Q6). Follow the `[.., 4]` albedo/glow
   channel layout (`:12-18`) and reuse `_split_albedo`, `_shading_normal`,
   `_recombine` (`:81-165`).
2. **`algan/rendering/raytracing/shading_taichi.py`**:
   - `_MID_DEFAULT = 0` → `_MID_MANIM = 0` (`:86`);
   - replace `_stage_default` (`:750-817`) with `_stage_manim`, same 14-arg
     stage contract (`:729-739`);
   - update `_BUILTIN_STAGE_FNS` (`:1128-1129`) — tuple position IS the pid;
   - ungated chain: swap the `_MID_DEFAULT` branch's callee (`:1266-1271`);
     gated loop and solo dispatch need NO edit (they iterate/index
     `_BUILTIN_STAGE_FNS`, `:1238-1246,1322-1330`);
   - prose: module docstring intro + pid table (`:16-32`),
     `MAX_SHADOW_LIGHTS` comment (`:189`), `_light_eval` docstring
     (`:626-629`).
3. **`algan/rendering/shaders/fragment_shaders.py`** — import swap (`:28-37`),
   `STAGE_DEFAULT = FragmentStage(_stage_default, _BUILTIN_MAT_SPECS)` →
   `STAGE_MANIM = FragmentStage(_stage_manim, _BUILTIN_MAT_SPECS)` (`:100`),
   map entry in `_builtin_shader_to_stage` (`:119,122`).
4. **`algan/rendering/raytracing/wavefront_kernels_taichi.py`** — import
   (`:66`), `_light_zero_radiance` docstring (`:159-163`), fan comment
   (`:2652-2656`), comparison (`:2657`).
5. **`algan/rendering/raytracing/raster_taichi.py`** — import (`:65`), comment
   (`:2860-2861`), comparison (`:2862`).
6. **`algan/rendering/raytracing/settings.py`** — `_build_core_shader_ids`:
   import + `manim_shader: 0` (`:2510-2528`); registry comment (`:2507-2509`).
   (`_MAT_DEFAULTS`/`_MAT_SLOTS` unchanged.)
7. **`algan/rendering/shaders/materials.py`** — `ManimMaterial(Material)`
   class with `shader = staticmethod(ms.manim_shader)` and
   `get_shader_param_values()` (see Q6); add to `__all__` (`:52-71`).
8. **`algan/__init__.py`** — remove `default_shader` import + install line
   (`:147-153`); import `manim_shader` alongside the other material shaders
   (`:135-145`); `STAGE_DEFAULT` → `STAGE_MANIM` in the fragment-shader import
   (`:174-183`); `SETTINGS.style.set(default_material=DiffuseMaterial())`.
   `__all__` is computed (`:350-366`): `manim_shader` lands via the callable
   rule, `STAGE_MANIM` via `name.isupper()`, `ManimMaterial` via
   `materials.__all__` through the star-import at `:146`. Nothing in
   `_INTERNAL_EXPORT_MODULES`/`_INTERNAL_EXPORT_NAMES` touches these names
   (`:253-342`; `null_shader` IS internal-listed at `:339` — irrelevant unless
   you pick it as the Q1 reference).
9. **`algan/settings/style_settings.py`** — field rename (`:42`) + docstring
   (`:7`).
10. **`algan/rendering/primitives/triangle_primitive.py`** — read
    `SETTINGS.style.default_material.shader` (`:187-189`).
11. **`algan/manim_defaults.py`** — `SETTINGS.style.set(default_material=
    ManimMaterial())` (`:216`) and adjust the surrounding comment (`:212-215`),
    which explains the *unlit* rationale that `basic_material_shader` served.
12. **Tests**:
    - `tests/unit_tests/test_manim_defaults.py` — fixture save/restore
      (`:24-34`) and the assertion (`:103`).
    - `tests/unit_tests/test_materials.py` — optionally add `ManimMaterial` to
      `ALL_MATERIALS` (`:49-59`); `test_param_contract` (`:208-217`) will then
      enforce that `get_shader_param_values()` keys equal the shader's extra
      params beyond `_NUM_BASE_PARAMS=9` (`:47`) — i.e. exactly
      `{"flat_shading"}`.
    - **`tests/unit_tests/test_render_coverage_audit.py` — this one bites.**
      It derives the required-covered set from `algan.__all__` subclasses of
      `Material` and fails on anything neither scene-covered nor in `EXEMPT`
      (`:26,50-75,86-105`). A public `ManimMaterial` must either appear in a
      full-render scene or be added to `EXEMPT` with a reason. Note `EXEMPT`
      currently lists `DiffuseMaterial` etc. as "legacy material API"
      (`:70-74`) — after this change `DiffuseMaterial` stops being legacy and
      becomes the default; the exemption reasons deserve updating.
    - `test_fragment_shaders.py` pattern (`:29-35`) if you want a resolution
      test for `manim_shader → STAGE_MANIM`.
13. **Docs** — `settings.rst:180-184`; `shaders_and_materials.rst` Vertex
    Shaders section (`:196-236`) and stage table row (`:283-286`); docstrings
    per `DOCSTRINGS.md`. Autosummary stubs under `docs/source/reference/` are
    **generated at build time** (the directory doesn't exist in-tree);
    `docs/source/reference_index/rendering.rst:11-14` lists modules, so new
    public names are picked up with no stub edits.
14. **Not needed** (verified): no seventh dispatch branch (id reused);
    `MAT_W`/`_BUILTIN_MAT_SPECS`/`_PHYSICAL_MAT_SPECS` untouched;
    `_MAT_SLOTS` untouched; STBVH untouched; `KERNEL_REGISTRY` untouched.

---

## Q6. Parameter block for a parameter-free stage

Mechanically, an empty `get_shader_param_values()` is coherent with all three
consumers:

- `set_material` registers the shader's extra params as animatable attrs via
  `set_shader(material.shader)` and then applies `get_shader_param_values()`
  over them (`mob_materials.py:277-293`); an empty dict is a no-op.
- `_pack_material` fills the block from `_MAT_DEFAULTS` and writes only
  named pairs present in `shader_param_names/values` (`primitives.py:730-756`);
  with no pairs it just writes `one_sided` (`:741-743`). A 12-slot spec costs
  nothing host-side — the block is always `MAT_W=30` wide
  (`shading_taichi.py:64`).
- Pipeline composition uses the FragmentStage's `param_specs` widths, not the
  material's dict (`fragment_shaders.py:167-206`), so `STAGE_MANIM` composes in
  custom pipelines with uniform offsets like every other built-in.

**But empty ≠ honours `flat_shading`.** Slot 10 reaches the kernel only if
(a) `flat_shading` is an animatable attr — i.e. a parameter of the *torch*
shader signature beyond the fixed nine (`mob_materials.py:95-113`) — and
(b) its value is written into the block — i.e.
`get_shader_param_values()` includes `"flat_shading"`, matched to
`_MAT_SLOTS["flat_shading"] == (10, 1)` (`settings.py:2583`) in
`_pack_material` (`primitives.py:734-736,751-755`). With a truly empty dict the
slot stays at the `_MAT_DEFAULTS` value 0.0 (`settings.py:2546-2547`) and
`ManimMaterial(flat_shading=True)` silently does nothing.

**What `ManimMaterial.get_shader_param_values()` should return:**

```python
{"flat_shading": self._flat()}
```

with `manim_shader` declaring `flat_shading: float = 0.0` as its sole extra
parameter. Then `manim_shader` is "no material parameters of its own" in every
meaningful sense (nothing colour/lobe-related), `test_param_contract` holds
(keys == extras == {"flat_shading"}), and the stage reads
`params[tm, prim, off + 10]` exactly like `_stage_lambert` does
(`shading_taichi.py:830-832`). `Material.__init__` already accepts and stores
`flat_shading` and `_flat()` converts it (`materials.py:268,279,298-299`), so
no constructor work is needed beyond inheriting.

---

## Q7. Colour space

### (a) Where the decode happens; `in_rgb` is linear

`scene_builder._decode_merged_colors` (`scene_builder.py:1882-1919`) is the
single crossing point, gated on `rt_settings.LINEAR_COLOR_SPACE` (`:1912-1913`):
it decodes channels `[..., :3]` of `tri_colors`, `circuit_colors`,
`circuit_border_colors` (`:1865-1869,1914-1918`), decodes colour textures as
they are appended (`:1262`; rationale `:1901-1906`), and finally the four
colour slots of the built-in material block — `_MAT_COLOR_SLOT_NAMES =
("emissive", "specular", "specular_color", "sheen_color")` (`:1879`),
applied per-primitive for ids `< _USER_PIPELINE_BASE` via
`_decode_material_block_colors` (`:1922-1944`). Deliberately not done in
`Color` so `mob.color` keeps reading display-referred and tweens stay
perceptual (`:1891-1896`). Therefore, with the default setting on, `in_rgb`
inside any fragment stage — including a new `_stage_manim` — is **linear
light**, and the faithful order is exactly the brief's
`linear → linear_to_srgb → add offset → clamp → srgb_to_linear`.
(Confirmed by the rendered round-trip tests,
`tests/unit_tests/test_color_decode_boundary.py:97-130`.)

### (b) Helpers, exact names/signatures

Taichi (`algan/rendering/raytracing/color_space_taichi.py`, a deliberate leaf
module):

```python
srgb_to_linear_f(c: ti.f32) -> ti.f32        # :20-31
linear_to_srgb_f(c: ti.f32) -> ti.f32        # :34-45
srgb_to_linear_v3(c) -> ti.math.vec3          # :48-53
linear_to_srgb_v3(c) -> ti.math.vec3          # :56-61
```

Torch (`algan/utils/color_space.py`):

```python
srgb_to_linear(c)    # tensor -> tensor, clamp_min(0) before pow :58-78
linear_to_srgb(c)    # encodes >1.0 rather than clamping       :81-103
```

### (c) Existing compile-time gate to reuse

Yes: `_linear_color_space()` (`shading_taichi.py:120-137`) — local import,
read live through the module object, and consumed under `ti.static` at
`:172` (`_energy_scale`) and `:1366` (`_run_frag_pipeline`'s peak bound). A new
stage should gate its round trip with `ti.static(bool(_linear_color_space()))`
so the display-referred arm compiles out. Heeding the CLAUDE.md warning: this
is resolved at kernel-compile time, so A/B-ing the two arms requires one
process per arm (and flipping the setting mid-process does nothing for kernels
already compiled).

### (d) Exposure / tonemapping interaction

The offset lands in the linear HDR buffer and is then treated like any other
radiance: exposure multiplies before the curve (`post_process.py:253-261`), a
tonemap curve maps it (`:245-251`, kernel twin
`tonemap_kernels_taichi.py:67-74`), and the sRGB OETF runs last
(`:263-276`). So:

- Under defaults (tonemapping off, exposure 1.0 — and
  `use_manim_defaults` forces tonemapping off, `manim_defaults.py:223`) the
  round trip is byte-faithful and nothing else is needed.
- If exposure ≠ 1 or a curve is on, the re-linearized offset is scaled/mapped
  exactly as Manim's byte value would NOT be — i.e. faithful-Manim look is
  conditional on those being neutral. There is no per-stage exposure hook and
  nowhere the offset *must* be pre-scaled; documenting the caveat in the stage
  docstring is the right treatment. **[reasoning from the cited pipeline
  order]**
- Note also the shared tail of `_run_frag_pipeline` (`:1341-1370`) applies to
  the stage's output: `max(out, 0)` always; the peak-rescale bound only when
  linear space is OFF. Manim's negative-half offset can push a dark albedo
  below range? No — the offset is non-negative (`0.5·(n·to_sun)³`, halved when
  negative, still ≥ 0), so the min-clamp is inert for it; but the *sum* over
  lights can exceed 1 and will ride whatever the tail/encoder do. **[reasoning]**

---

## Q8. What rendered output will move, and what pins it

### (a) Committed pixel baselines

Baseline directories: `tests/fast/expected_outputs_cpu/` +
`tests/fast/expected_outputs_cuda/` (one clip, `fast.mp4`), and
`tests/full_renders/expected_outputs_cpu/` +
`tests/full_renders/expected_outputs_cuda/` (six clips, one per scene).

- **`tests/fast/scene.py` — should NOT move.** Every triangle Mob in it carries
  an explicit material: `Cube`→MeshLambert, `Icosahedron`→MeshStandard,
  `Octahedron`→MeshBasic, faded `Cube`→MeshLambert (`scene.py:114-126`);
  everything else is bezier circuits and glyphs (`:91-108,129-132`), which
  never reach the triangle fallback (§3a). Lights are explicit (`:78-89`).
- **`complex_hierarchy_become.py` — MOVES.** Zero `set_material` calls in the
  file (grepped: none); its Spheres/Tetrahedrons/Cylinders
  (`:86,97,107,109,119,126,139,144`) all ride the current default lerp and
  would re-shade. Its framed `ImageMob` (`:81`) is default-shaded too
  (`ImageMob(Surface)`, `image_mob.py:35`).
- **`solids_and_camera.py` — MOVES.** Mostly materialized, but `Line3D`
  (`:194`) and `Dot3D` (`:195`) are bare (Line3D is Cylinder-based,
  Dot3D is Sphere-based) — both default-shaded today. `ConvexHull3D` right
  after them is materialized (`:205`).
- **`text_and_media.py` — MOVES, but via its bare `ImageMob`s, not the glTF
  (amendment; verified against the asset).** `ImageMob` subclasses `Surface`
  (`image_mob.py:35`) and both images in this scene are unmaterial'd
  (`:177-179`, `:185-188`) → Surface site, default shader today. The glTF
  does *not* ride the default: I parsed `assets/textured_icosphere.glb`'s JSON
  chunk — it declares exactly **1 mesh whose single primitive references
  material 0** — and with `pbr_materials=True` (the default,
  `model_mob.py:158-162,177,295-296`) the import applies a
  `MeshStandardMaterial` to it. A glb whose meshes lacked materials would fall
  back to the default (`model_mob.py:250-257`); this one does not.
  The `shader_sphere` beside it has an explicit fragment pipeline
  (`:195-199`) and is unaffected.
- **`materials_and_lighting.py` — unchanged.** Every Sphere/Prism in it sets a
  material (`:46-83,148-149,154,234-237`).
- **`shapes_and_timeline.py` — MOVES (amendment; corrects an earlier claim).**
  It contains one unmaterial'd triangle Mob:
  `gradient_triangle = TriangleTriangulated(...)` (`:315-321`, per-vertex
  colours only) reaches the fallback through `TriangleVertices`. The scene is
  lit (`AmbientLight` + `DirectionalLight`, `:48-49`), so the re-shade is
  visible. The neighbouring `QuadTriangulated` *is* covered
  (`MeshBasicMaterial`, `:357`), as is everything else in the file.
- **`manim_compat_and_plots.py` — unchanged.** `Axes`/`NumberPlane` (`:47,58`)
  are Manim VMobject content → circuits, not triangles.

Secondary, pixel-invisible but real: collection membership shifts because
unmaterialized mobs now share `lambert_shader`'s batch identity with explicit
DiffuseMaterial mobs (§3c, `triangle_primitive.py:191-192`). **[reasoning: no
direct pixel effect identified]**

Per CLAUDE.md, a CPU-side confirmation covers `tests/fast` and the portable
full-renders; regenerating `expected_outputs_cuda/` still needs a CUDA machine,
and tessellation-adjacent movement is invisible to `--fast` regardless.

### (b) Non-pixel tests asserting on shading / the current default

- `tests/unit_tests/test_manim_defaults.py:25,31-33` (fixture save/restore of
  `style.default_shader`) and `:103` (asserts `is basic_material_shader`) —
  must be updated to `default_material` / `ManimMaterial()`.
- `tests/unit_tests/test_frag_pid_gate.py` — asserts mask/solo *mechanics*
  with symbolic ids (`:49-68,74-98`); id-agnostic, survives unchanged.
- `tests/unit_tests/test_fragment_shaders.py:131-143` — uses literal `0` as a
  built-in pid in synthetic merged dicts (`:138`); still valid (0 remains a
  built-in). `:29-35` resolve test pattern worth extending to
  `manim_shader`.
- `tests/unit_tests/test_color_decode_boundary.py:142-194` — uses
  `tri_mat_id` 0 vs `_USER_PIPELINE_BASE` as built-in/custom markers for the
  decode; semantics unchanged by the repurpose.
- `tests/unit_tests/test_materials.py:47,208-217,524-537` — signature-count
  reference (`basic_material_shader`), param contract over `ALL_MATERIALS`,
  legacy-shader smoke test; only the comment at `:528-529` mentions
  `default_shader`.
- `tests/unit_tests/test_render_coverage_audit.py:50-105` — will fail until
  `ManimMaterial` is scene-covered or exempted (§5.12).
- No unit test asserts numeric pixel colours of an *unmaterialized* triangle
  Mob (checked the render-using unit tests: `test_deterministic_shadow_opacity.py`
  and friends all set materials or use circuits; `test_mesh_identity.py`'s bare
  `Cube`/`Icosahedron`/`Sphere` asserts are structural mesh-id assertions,
  `:152-200`).

### (c) Docs/design prose that would become false

- `docs/source/advanced_user_tutorials/shaders_and_materials.rst:196-236` —
  presents `default_shader` as the shipped diffuse vertex shader, the
  parameter-count convention, and the worked example; `:283-286` stage table.
- `docs/source/advanced_user_tutorials/settings.rst:180-184` — the
  `default_shader` settings entry.
- `OX_LIGHTING_AUDIT.md:81,107,181,193,207` — analyses of `_stage_default` /
  `default_shader` clamping and ambient behaviour (historical audit; annotate).
- `LINEAR_COLOR_WORK.md`, `TONEMAP_FINDINGS.md`, `AGENTS_DETAILED.md`,
  `DESIGN_optimization_targets.md` — grepped; **no prose describing the default
  shader's look**, so nothing there goes false. `AGENTS_DETAILED.md` mentions
  shader/material APIs generically only (`:164,196,347,491`).
- Docstrings that narrate the old default and should be touched in the same
  change: `material_shaders.py:3-10` (convention citation),
  `shading_taichi.py:16-32,626-629,755,189`,
  `wavefront_kernels_taichi.py:159,2652-2656`,
  `raster_taichi.py:2859-2861`, `manim_defaults.py:212-216`,
  `style_settings.py:7`, `pbr_shaders.py:9`.

---

## Method note

All claims above were verified by reading the cited files this session
(grep/read only; no execution). The few forward-looking statements — batch
membership neutrality in §3c/§8a, the exposure caveat in §7d, and the EXEMPT
decision in §5.12 — are labelled **[reasoning]**.

## Amendments (second-pass verification)

This report was re-verified against source in a second pass, and three items
were corrected or added; all corrections were made by re-reading the cited
files, and one asset was inspected directly:

1. **§8a `shapes_and_timeline.py` previously read "unchanged — no
   triangle-based Mobs". That was wrong.** The scene contains an unmaterial'd
   `TriangleTriangulated` (`gradient_triangle`, `:315-321`) and is lit
   (`:48-49`), so this baseline MOVES. Its `QuadTriangulated` is covered
   (`:357`). Net effect on the summary: **four** of the six full-render scenes
   move (`complex_hierarchy_become`, `solids_and_camera`,
   `shapes_and_timeline`, `text_and_media`); two do not.
2. **§8a `text_and_media.py`: the moving mechanism is the bare `ImageMob`s,
   not the glTF.** The glb's single mesh references a material and therefore
   receives `MeshStandardMaterial` at import (`pbr_materials=True` default);
   the two unmaterial'd `ImageMob`s (class `ImageMob(Surface)`) are what
   re-shade.
3. **§3 route list gained the `TriangulatedBezierCircuit → TriangleTriangulated`
   path** (plots.Arrow, FunctionPlotMob, TexTriangulated/TextTriangulated,
   morph conversions), and §3c gained the zero-vertex-normal mechanism with
   citations for why today's default renders these flat while Lambert will
   not.

A pre-amendment snapshot of this report is preserved outside the repository at
`/tmp/opencode/OX_DEFAULT_SHADER_AUDIT.pre_amendments.md`.
