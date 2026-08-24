# Adversarial review — per-mob shadow flags (`casts_shadows` / `receives_shadows`)

Reviewed: the uncommitted diff on `claude/per-mob-shadow-flags-uj5pim` as it stood at
09:20–09:33 today, plus the then-untracked `tests/unit_tests/test_shadow_flags.py` and
`benchmarks/_shadow_flags_check.py`. While this review was in progress the branch
author committed exactly that content as **`2708ac2` "Add per-mob casts_shadows /
receives_shadows flags"** (verified: the tracked-file hunks of my final snapshot are
byte-identical to `git diff 1e204d2 2708ac2`, and the two new files are in the commit).
All `file:line` references below are against `2708ac2` / the current tree.

Method: source reading only. Nothing was executed, staged or modified (the brief
forbade running pytest; a parallel render process was live throughout). Where a claim
below rests on reading rather than measurement it says so.

---

## Summary verdict

The mechanism is sound where it is wired, and the leaf-word/link-word surgery is
complete and safe — I found no missed decoder, no non-shadow ray that rejects a
non-caster, and no occlusion path that bypasses the bit. The real defects are at the
**edges of the wiring**, and two of them are reachable through the public API:

1. **(Highest severity) Mixed-flag diced collections partially suppress legitimate
   shadows** — `shadow_cast_flag`'s amax-over-frames reduction is only sound when one
   primitive carries one flag, and a diced collection of two mobs with different
   `casts_shadows` violates that (Q2).
2. **`ThreeDModelMob`/glTF silently ignores both flags** — its builder never declares
   them (Q1).
3. **A custom fragment pipeline ≥ 34 slots wide has its real slot-33 parameter read as
   `no_shadow_receive`** — the width guard prevents out-of-bounds reads, not semantic
   collision (Q5).

Details and the long tail under each question.

---

## Q1 — what did I fail to wire?

### Every construction site of the two primitives

There are exactly six places in `algan/` that construct a render primitive of these
classes (plus the generic collection wrappers in the batcher):

| # | Site | Class | Wired? |
|---|------|-------|--------|
| 1 | `algan/mobs/shapes_2d.py:522` (`TriangleVertices.get_render_primitives`) | `effective_triangle_primitive()` | ✅ declares at `shapes_2d.py:554` |
| 2 | `algan/mobs/surfaces/surface.py:3095` (`Surface._build_render_primitive`) | `LogicalPNTrianglePrimitive` | ✅ declares at `surface.py:3144`; reached from both `Surface.get_render_primitives` (`surface.py:2990`) **and** the batched fast path `get_render_primitives_batched` (`surface.py:721`) |
| 3 | `algan/mobs/nonplanar_circuit.py:787` (`build_patch_primitive`) | `LogicalPNTrianglePrimitive` | ✅ declares at `nonplanar_circuit.py:807` |
| 4 | `algan/mobs/three_d_models/mesh.py:363` (`TriangleMesh.get_render_primitives`) | `effective_triangle_primitive()` | ❌ **NOT wired** |
| 5 | `algan/mobs/pn_mesh.py:85` (`PNMesh.get_render_primitives`) | `LogicalPNTrianglePrimitive` | ❌ **NOT wired** |
| 6 | `algan/mobs/bezier_circuit.py:1181` (`BezierCircuitCubic._get_render_primitives`) | `RENDERER_REGISTRY.bezier_circuit_primitive` | ✅ declares at `bezier_circuit.py:1215` |
| 6b | `algan/mobs/nonplanar_circuit.py:859` (`build_stroke_primitive`) | same, via `circuit.render_primitive` | ✅ declares at `nonplanar_circuit.py:890` |

Collection wrappers: `render_loop.py:2222`, `:2243`, `:2253` build
`primitive_class(triangle_collection=...)`. These need no declaration of their own:
both `__init__`s gather every member of `_surface_params` per member, filling absent
members with `0.0` (= casts/receives, the old behaviour) — triangles at
`primitives.py:494-520`, circuits at `primitives.py:2707-2719`. A declared member's
value rides the merge; an undeclared member gets the inert fill. Correct.

### The specific mobs you asked about

* **`ImageMob`** — subclasses `Surface` (`image_mob.py:38`) and overrides no
  `get_render_primitives`, so it goes through site 2: **wired**. Note its
  `receives_shadows=False` is *inert anyway*, because it renders unlit and the recv
  gates short-circuit on `pid != _MID_UNLIT` (`sheet_resolve_taichi.py:465`,
  `wavefront_kernels_taichi.py:2701`). Its `casts_shadows=False` works (it is in the
  triangle tree). Consistent with the documented asymmetry; nothing misstated.
* **`ThreeDModelMob` / glTF import** — `ThreeDModelMob` holds `TriangleMesh` children
  (`model_mob.py:237`); `TriangleMesh.get_render_primitives` (`mesh.py:338-379`)
  constructs the primitive at site 4 and calls neither `declare_shadow_flags` nor even
  `declare_one_sided`/`declare_closed_shell`. The lone-primitive constructor's default
  `declare_shadow_flags(True, True)` (`primitives.py:551`) then fixes casting/receiving.
  **What a user gets: `three_d_model_mob.casts_shadows = False` (and
  `receives_shadows = False`) are silently ignored — the model keeps casting and
  receiving.** `resolved_shadow_flags()` cannot rescue this: nothing on the path ever
  reads it. This is the same mob class the CLAUDE.md entry for `two_sided` lists as
  "stays two-sided", i.e. a known user-facing surface, so I rank this a genuine gap,
  not an internal-only omission. (It is also the *only* wired-feature gap among the
  built-in mob families you listed.)
* **`point_cloud` (`PMobject`)** — builds `Dot3D` children or one `Dot3D.from_batches`
  pack (`point_cloud.py:135-153`); `Dot3D` is a `Sphere`, so primitives come from site
  2, and `resolved_shadow_flags()` walks up through the pack to the `PMobject`.
  **Wired**, including `pmobject.casts_shadows = False`.
* **`ManimMob`** — subclasses `BezierCircuitCubic` (`manim_mob.py:44`): wired via site
  6. Image submobjects become `ImageMob`s (wired). Batched children via `batch_mobs`
  produce another `BezierCircuitCubic`: wired. ✅
* **`Polyhedron`** — faces are `TriangleTriangulated` → `TriangleVertices` children
  (`shapes_3d.py:1660-1678`); primitives come from site 1, and
  `resolved_shadow_flags()` walks TriangleVertices → TriangleTriangulated → face
  Groups → `faces` Group → Polyhedron, so `poly.casts_shadows = False` lands.
  **Wired** — and covered by `test_flag_set_on_an_aggregate_reaches_the_geometry`
  (`test_shadow_flags.py:64`). Note the contrast with `two_sided`/`closed_shell`,
  which `Polyhedron.__init__` copies onto face mobs explicitly
  (`shapes_3d.py:1689-1691`): the flags rely solely on the ancestor walk. That works,
  but it is a second mechanism doing what the neighbours do with a first; fine, just
  worth knowing.
* **`from_batches`** (`Surface.from_batches` `surface.py:1179`,
  `BezierCircuitCubic.from_batches` `bezier_circuit.py:537`) — one packed Mob built
  from one representative constructor call, rows widened. The single packed mob's
  `_build_render_primitive` / `_get_render_primitives` runs the wired declaration once
  for the whole pack. **Wired**, with the inherent packing property that all members
  share one flag (documented elsewhere as "every member has the same … material").
* **`batch_mobs`** (`utils/mob_utils.py:188`) — packs N existing Mobs by deep-copying
  **mobs[0]** non-recursively (`mob_utils.py:228`) and re-batching rows. Plain
  attributes survive deepcopy, so *mobs[0]*'s instance flags become the whole pack's
  flags; **any other member's differing flag is silently dropped**. With all defaults
  (the overwhelmingly common case) nothing changes. See Q6 for the ranking.
* **`PNMesh`** — unwired (site 5), but deliberately internal ("This is deliberately
  internal", `pn_mesh.py:15`). It still reaches the screen: see Q6 item 4.

### Answering your four named declaration sites

All five declarations are present and correct as described (`surface.py:3144`,
`shapes_2d.py:554`, `bezier_circuit.py:1215`, `nonplanar_circuit.py:807` and `:890`),
each calling `self.resolved_shadow_flags()` / `circuit.resolved_shadow_flags()`. The
gap is not in those five — it is sites 4 and 5 never calling any of them.

---

## Q2 — does the flag survive the logical-PN dice?

**The carry itself: yes, verified.** `_dice_logical_pn` expands *every* member of
`_surface_params` — which since this diff includes `no_shadow_cast` /
`no_shadow_receive` (`primitives.py:420-428`) — to per-frame patch-corner arrays
(`primitives.py:2229-2234`), interpolates them onto the shared subdivision vertices
and scatters them to the diced microtriangle corners with everything else
(`primitives.py:2390-2395`), then `setattr`s the diced arrays back onto the primitive
(`primitives.py:2400-2401`) before `_pack_projected_flat_geometry` runs
(`project_to_screen`, `primitives.py:2411-2421`). So the frame-bounds packing at
`primitives.py:1032-1038` sees post-dice values. Interpolating a constant corner value
barycentrically reproduces the constant, and the consumer thresholds at `> 0.5`, so
float noise cannot flip it.

**Shape after dicing: correct.** Post-dice `no_shadow_cast` is `[T, M, 3, 1]` (M =
padded per-frame triangle width); `shadow_cast_flag` slices corner 0 (`dim >= 4`
branch), reduces amax over the trailing axis and over frames, and returns `[1, M]`
(`primitives.py:305-333`) — matching `lo.shape[1]` and concatenating cleanly in the
merge (`scene_builder.py:1432`). Padding tail rows read `0.0` → "casts", which is
harmless: they are alpha-zeroed (`primitives.py:2414-2420`) and bounded empty
(`_pack_frame_visibility`, `primitives.py:1039-1044`), so they emit no leaf that any
ray accepts.

**Meaning after dicing: wrong in one reachable case.** The reduction takes
`amax` over **frames** (`primitives.py:331-332`). For one mob that is exact — the flag
is fixed per mob, so every frame agrees. But a **collection** merges several mobs'
patches along the patch axis (`primitives.py:494-520`), and after dicing the *column*
a patch occupies moves from frame to frame, because each frame dices adaptively
(`counts`/`offsets`, `primitives.py:2159-2160`). Concretely:

> Sphere A (`casts_shadows=False`) + Sphere B (default) in one scene merge into ONE
> `LogicalPNTrianglePrimitive(triangle_collection=[pA, pB])` whenever they share a
> batch identifier — and they do: `get_batch_identifier`
> (`triangle_primitive.py:217-225`, extended at `primitives.py:1368-1373`) keys on
> class + shader + authored-params + tolerances, none of which is the shadow flag.
> In frame f0 column c hosts a patch of A (value 1.0); in frame f1, B having diced
> coarser, column c hosts a patch of B (value 0.0). The amax over frames makes
> `casts[c] = False` **for every frame**, so B's microtriangles sitting in column c in
> f1 do not cast a shadow that B should cast.

The error direction is one-way — a non-caster can never be promoted into casting —
but "my sphere's shadow is partly missing" is exactly the kind of silent wrongness
this feature must not have. It needs: ≥2 curved-surface mobs in one merge group with
*different* `casts_shadows`, and dice levels that shift column ownership between
frames (guaranteed under camera motion). Flat-triangle collections are unaffected
(their values are single-frame, so the amax over frames is trivially per-column
exact). Established from source; I did not render it (read-only brief) — a two-sphere
pixel probe against the `drop` oracle in `benchmarks/_shadow_flags_check.py` would
settle it in minutes.

Two smaller notes on `shadow_cast_flag` itself:

* The docstring says "taken amax over corners and frames", but the code slices corner
  0 *first* (`v[:, :, 0, :]`, `primitives.py:327-328`), so corners 1–2 are dropped
  before any reduction. Lossless today because every producer writes
  corner-uniform constants (`full_like`), but the comment overstates the guarantee a
  future producer gets.
* The `num_prims` argument is used only for the `None` fallback shape; fine.

---

## Q3 — is the leaf-word change complete and safe?

**Yes, with three stale copies outside production listed at the end.**

### Bit 15 was genuinely free

`leaf_tspan` is assembled as
`slot_t0.clamp(0, 2^15-1) | (slot_t1.clamp(0, 2^15-1) << 16)` (`stbvh.py:851-853`):
t0 lives in bits 0–14, bit 15 was never written, t1 in bits 16–30, bit 31 is the
interval-opaque flag applied afterwards (`stbvh.py:855-859`). So `LEAF_NOCAST_BIT`
(`stbvh.py:639`) collides with nothing, and narrowing t0 reads from `& 0xFFFF` to
`& 0x7FFF` is behaviour-neutral for unstamped trees (bit 15 clear ⇒ identical value).
The sign-based opaque tests (`tspan < 0`) don't see bit 15.

### Your load-bearing claim about `_build_blocks`: confirmed

`STBVH.__init__` builds `blocks` from the **node rows**, not from `leaf_tspan`:
`_build_blocks(nodes, first_leaf)` slices `nodes[1 : 1 + ARITY*first_leaf]` and packs
columns 6/7 (tmin/tmax floats, clamped to 15 bits) into lane 6 (`stbvh.py:135-169`,
called at `stbvh.py:222`). `_stamp_noncaster_bit` runs *after* construction
(`stbvh.py:892-899`) and touches only `bvh.leaf_tspan`. Therefore the three remaining
`& 0xFFFF` kernel reads — `_test_children` (`raytrace_kernels_taichi.py:711`) and
`_test_root` (`:895`, `:911`) — decode **node-derived block words**, which the stamp
never touches, and are correctly left at `0xFFFF`. (Their `>> 16 & 0x7FFF` t1 reads
were already narrow.) The refit tree's blocks likewise store link words bit-exactly —
raw int16 halves under `BLOCK_F16` (`refit_bvh.py:503-505`), bit-cast int32 otherwise
(`:512`) — so bit 29 survives packing; only lanes 0–5 get the directed-rounding
treatment.

### Every other decoder / whole-word consumer

I swept kernels, host torch code, tests and benchmarks:

* Production kernel decoders of the refit link word: six sites, all inside
  `_nearest_triangle_hit`, `_nearest_bezier_hit`, `_collect_hits` (×2) and
  `_anyhit_opaque_tri/bez` — every one updated to mask 29 bits *and* test
  `_REFIT_NOCAST_BIT` (`raytrace_kernels_taichi.py:1063-1067`, `1185-1189`,
  `2155-2159`, `2299-2303`, `2478-2480`, `2579-2581`). No remaining production
  decoder uses the old 30-bit mask.
* Host-side: nothing decodes either word. `scene_builder.py:404` uploads
  `leaf_tspan` by identity (byte-preserving); `_STBVH_TENSOR_FIELDS`
  (`scene_builder.py:48`) counts bytes. The refit builder writes the words
  (`refit_bvh.py:476-481`) and enforces the narrower index bound at
  `refit_bvh.py:304-307`. `nocast[safe_prim]` indexes original primitive ids, which
  stays aligned even for the opaque prepass BVH, because `_build_opaque_bvh` keeps the
  index space unchanged (empty bounds, `scene_builder.py:668-694`).
* Tests/benchmarks that decode leaf tspans on trees **they build themselves**
  (`tests/unit_tests/test_raytracing_unit.py:81-83, 130-136`, still `& 0xFFFF`;
  `test_scene_arena_upload.py:26` synthesizes `arange << 16`): unaffected today
  because their trees contain no non-casters, but they are latent stale decoders —
  if a future change feeds flagged trees into them they will misread t0. The new
  `test_shadow_flags.py` decoders use the correct masks (`:170-173`, `:212-217`,
  `:231-237`).
* **Stale production copies in a benchmark**: `benchmarks/_bvh_steps.py:184-185` and
  `:328-329` carry verbatim copies of the OLD leaf test (`(tspan & 0xFFFF) <= f`).
  Fed a tree containing a non-caster, these kernels would fold bit 15 into t0, fail
  the interval test, and silently skip the primitive — wrong traversal statistics in
  that benchmark only. They should be updated (or made to assert no non-casters).

Nothing compares or serializes either word whole anywhere else; no sort key, no hash.

---

## Q4 — shadow paths vs. the bit; and the converse

### Every way a primitive can occlude a light

1. **Sheet route (analytic-AA primary)** — mode 1 builds candidate events;
   `receives_shadows=False` hits build none (`sheet_resolve_taichi.py:460-465`), so
   mode 2 traces nothing for them and `lvis` stays lit. When events ARE built, the
   traced visibility comes from `_shadow_occluded` at `raster_taichi.py:2962` —
   shadow-ray context, flag carried (below). `shadow_vis` is written and consumed
   within the same resolve pass; there is no cached/reused visibility buffer that
   could bypass a later flag change.
2. **Classic wavefront inline shadows** — the light fan is gated on
   `(pid != _MID_UNLIT) && recv == 1` (`wavefront_kernels_taichi.py:2696-2701`);
   tracing goes through `_shadow_occluded` at `:2870`.
3. **Deferred shadow queue** — `_shadow_occluded` at `wavefront_kernels_taichi.py:2307`.
4. **The four arms inside `_shadow_occluded`** (`raytrace_kernels_taichi.py:2702-2797`):
   * march → `_nearest_surface_g(..., nocast=1)` (`:2891-2896`);
   * gather → `_collect_hits(..., nocast=1)` (`:3077-3082`);
   * any-hit modes 2/3 → `_shadow_anyhit_opaque` → `_anyhit_opaque_tri/bez`
     (`:2687-2697`). These two kernels test the bits **unconditionally** (no
     `ti.static(nocast)` gate, `:2478-2480`, `:2579-2581`) — but their ONLY callers
     are `_shadow_anyhit_opaque` and the mode-2 mid-march early-out, both shadow-ray
     contexts, so unconditional is correct here and costs the compile-time gate
     nothing that matters.

So: **no occlusion path bypasses the bit.** Any-hit early-out modes, the gather
march, the opaque prepass and the sheet route all either honour it or legitimately
shouldn't.

### The converse — a non-caster rejected on a non-shadow ray?

No. Every camera/reflection/refraction traversal passes `nocast=0`, compiling the
test out at compile time: `_nearest_surface` (`raytrace_kernels_taichi.py:2057`),
all six call sites in `wavefront_traverse` / `wavefront_traverse_events`
(`wavefront_kernels_taichi.py:1919-1921`, `1946-1948`, `1962-1964`, `2103-2105`,
`2134-2136`, `2150-2152`). The camera-side opaque prepass (nearest hit against the
opaque BVH) also runs `nocast=0`, so a non-casting solid still prunes gathering
behind it on camera rays — it stays visible and still occludes *visually*.
`_anyhit_opaque_*` reject non-casters unconditionally, but as established, nothing
but shadow queries reach them. The Monte Carlo megakernel needs nothing:
`det_shadows = bool(SHADOWS) and samples <= 1` (`tracer.py:1345`), confirmed.

---

## Q5 — entitlement: who reads slot 33?

**The width guard is correct against out-of-bounds and insufficient against semantic
collision — and the collision case exists.**

Mechanics, verified: built-in materials always pack a `MAT_W = 34` block
(`shading_taichi.py:78`; defaults row includes slot 33 = 0.0, `settings.py:2640-2648`,
asserted by `test_receive_slot_defaults_to_receiving`). Custom fragment pipelines pack
their OWN width `W = shader._frag_total_width` with slots numbered from **0**
contiguously across stages (`register_pipeline`, `fragment_shaders.py:177-207`;
packed at `primitives.py:865-902`). Mixed-width batches are right-zero-padded to the
widest W (`_cat_mat_blocks`, `utils.py:63-84`).

Consequences:

* Narrow custom pipeline (W ≤ 33): guard false → default receive. ✔ No OOB.
* Padded custom rows in a mixed batch (array widened to 34 by some built-in
  neighbour): slot 33 reads zero padding → default receive. ✔ Benign.
* **Custom pipeline with W ≥ 34**: guard TRUE, and slot 33 is a REAL parameter of
  that pipeline (slots start at 0, nothing reserves built-in ranges). If the user's
  slot-33 value exceeds 0.5, the mob **silently stops receiving shadows**; in
  `wavefront_shade` the whole visibility fan is additionally skipped for a hit whose
  pipeline may consume `vis` arbitrarily (`wavefront_kernels_taichi.py:2701-2704` —
  note `fan_exact`/`fan_geom` handling shows custom pids DO enter the fan when
  `recv_s == 1`), so the user pipeline also receives fabricated "everything lit"
  visibility bits. Reaching W ≥ 34 takes several stacked stages (~5 params each × 7),
  which is unusual but unvalidated — nothing caps or checks `_frag_total_width`.

The docstring-level claim repeated at both read sites — that the width test mirrors
"the same reasoning that keeps `_run_frag_pipeline`'s one_sided read on the built-in
branch" — is actually weaker than that reasoning: `one_sided` is only read under
`pid < _USER_PIPELINE_BASE` (`shading_taichi.py:1521-1526`), i.e. custom pipelines
are never asked the question at all. Slot 33 IS asked of custom pipelines whenever
the array happens to be wide enough. If you want the one_sided guarantee, gate the
two reads on `pid < _USER_PIPELINE_BASE` (built-ins only) instead of — or in addition
to — the width test; that closes the collision completely and matches the stated
reasoning. (`receives_shadows` on a custom-pipeline mob would then be inert like the
circuit case, which at least is honest.)

---

## Q6 — free rein, ranked

1. **Mixed-flag diced collections eat legitimate shadows** (Q2's finding; the most
   serious thing I found). Public API, silent, one-way-but-wrong. Fix direction if
   you want one: make the caster mask per-(frame, patch) survive to the leaf stamp
   (stamp per instance interval rather than reducing over frames), or split merge
   groups by resolved caster flag (cheap: one extra identifier component), or reduce
   with `amin` over frames per column *only within same-flag runs*. The middle option
   is one line in `get_batch_identifier` but changes batching.
2. **`ThreeDModelMob`/glTF ignores both flags** (Q1). One-line fix in
   `mesh.py:get_render_primitives` mirroring `surface.py:3144`; until then the
   docstring's promise ("Like `two_sided`, set it before the Mob is spawned") quietly
   doesn't hold for imported models.
3. **Custom-pipeline slot-33 collision** (Q5). Latent, low probability, high
   weirdness when hit. Gate on `pid < _USER_PIPELINE_BASE` to close.
4. **Cross-family morphs lose the flags mid-morph.** `become` between different
   morph families converts both ends to PN soups and renders THE SOUP during the
   transition (`mob_morph.py:1152-1226`, `_register_hierarchy_for_render(source_soup)`
   at `:1224`). `PNMesh.get_render_primitives` (`pn_mesh.py:80-95`) declares nothing,
   so a non-casting object **casts during its morph window** even though both
   endpoints honour the flag. (`_MORPH_ADOPTED_ATTRS` correctly carries the flags to
   the endpoint replacement — `mob.py:366-372` — so the seam is strictly interior to
   the morph.) Same-family morphs don't route through soups and are unaffected.
5. **`batch_mobs` collapses per-member flags to mobs[0]'s** (`mob_utils.py:228`).
   Inherent to packing (one packed mob = one flag), and invisible at defaults; worth a
   sentence in the docs rather than code.
6. **Benchmark-only stale leaf tests** — `benchmarks/_bvh_steps.py:184-185, 328-329`
   still decode `& 0xFFFF` on leaf words (Q3). Wrong numbers, wrong images never.
7. **Memory-model notes, both benign**: the +1 material-block slot and the new
   `[To, N]` bool masks are exactly the kind of allocation the measured memory model
   picks up automatically; nothing needed annotating. After a deferred BVH build,
   `build_deferred_bvhs` nulls `tri_frame_lo/hi` but retains `tri_frame_casts`
   (`scene_builder.py:836-907`) — consistent with how `tri_frame_opaque` already
   behaves, so at most one retained `[To, N]` bool per geometry type; harmless.
8. **Truncation counters unaffected**: rejecting leaves early only reduces work; the
   counters keep counting real events, zeros stay readings.
9. **Flag set after spawn**: the docstrings say set-before-spawn and the mechanism
   reads at primitive-build time, which happens per render job. Within one
   `save_video` the user can't intervene; across two `save_video(reset=False)` calls
   a changed flag takes effect on the second job. That is arguably finer than the
   docstring implies, not coarser — no action needed.

### Verified sound (so you don't re-litigate)

* Default-path inertness by construction: no declaration ⇒ `0.0` fills everywhere
  (merge fill `primitives.py:498-500`, `:2712-2716`), bit unset, slot at default;
  plus your measured byte-identity result on top.
* `ALGAN_PER_MOB_SHADOW_FLAGS=0` restores old behaviour host-side exactly, as
  claimed: `shadow_cast_flag` returns all-True (`primitives.py:323`), the material
  pair is skipped (`primitives.py:840`), so no word changes and no kernel variant
  moves; the runtime slot reads see constant 0.0. Statically verified.
* `resolved_shadow_flags()` (`mob.py:387-421`): DAG-safe (visited set), cycle-safe,
  early-exits when both flags settled; covers Cube/Polyhedron, Group, Text glyph
  packs (pack is a child of the Tex, `text.py:356`), PMobject, Arrow3D parts.
* Circuit asymmetry as documented: `declare_shadow_flags` accepts-and-ignores
  `receives` (`primitives.py:2681-2694`), pinned by
  `test_a_circuit_declares_casting_and_ignores_receiving`. I found the asymmetry
  stated correctly everywhere I looked, including the new CLAUDE.md paragraph.
* Refit `LINK_PRIM_MASK` narrowing: complete (Q3), guarded, and separable from its
  neighbours (`test_refit_link_word_keeps_prim_opacity_and_flag_separable`,
  `test_prim_index_range_is_enforced_at_the_narrower_bound`).

### Not established (couldn't be, from here)

* Whether the Q2 mixed-flag scenario actually moves pixels — source-established
  mechanism, unrendered (no pytest, parallel renders owned the machine).
* CUDA-specific behaviour of anything above; this checkout has no GPU.
* Whether any *user* (as opposed to built-in) pipeline in the wild is ≥ 34 slots —
  unknowable from the repo.
