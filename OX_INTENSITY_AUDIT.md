# Audit: making `Light.intensity` an animatable attribute

Read-only audit of the plan in the brief. No repo file was edited, created or
deleted; the single write is this report. Repo code was executed only through a
throwaway harness under `/tmp/opencode` with `ALGAN_USE_DAEMON=0`, which applies
the plan to `algan.rendering.lights.Light` **inside one process** (property
attach + constructor rework + `set_animated_attribute` override) and measures
the result. Everything below is labelled **[M]** (measured in that harness, on
this machine, CPU) or **[R]** (reasoning from source).

---

## A. The six numbered claims

### Claim 1 — register before `super().__init__()`, `_init_default_attr` after — **TRUE, with one mandatory refinement**

- The registration mechanism works pre-init: `register_attrs_as_animatable`
  (`algan/animatable_base/animatable.py:281-318`) creates `self.animatable_attrs`
  itself if absent (`:307-308`), and the property attach is skipped only when
  the class already has the name (`:341-342`); `Light` has no class-level
  `intensity` today (it is instance-assigned at `algan/rendering/lights.py:135`),
  so `hasattr(Light, "intensity")` is False and the property attaches. Because a
  property with a setter is a data descriptor, it also wins over any stale
  instance-dict entry left by today's line 135.
- **The refinement — measured, not theoretical.** Today's constructor line
  `self.intensity = _finite_number(...)` (`lights.py:135`) runs before
  `super().__init__()` and is reached again via kwarg forwarding from every
  subclass (`PointLight.__init__` → `super().__init__(intensity=intensity)`,
  `lights.py:239`). Once the property is attached *before* that line executes,
  the assignment routes into `set_animated_attribute`, which touches
  `self._prevent_recursive_sets` (`algan/animatable_base/mob.py:1080`) — an
  attribute created only later, in `Mob.__init__`/`Animatable.__init__`. **[M]**
  Every ordering that lets the property exist while `lights.py:135` still runs
  crashes with `AttributeError: '...object has no attribute
  '_prevent_recursive_sets'`. So "mirroring `Mob.__init__`" is not literally
  possible: Mob receives `location`/`color` as arguments (`mob.py:230-294`);
  Light manufactures its own. The validated value must be captured into a
  **local** (or read back from the local when seeding), i.e.
  `value = _finite_number(...); register; super().__init__();
  self._init_default_attr("intensity", cast_to_tensor(value))`. Keeping the raw
  assignment first and registering second also survives **[M]**, but leaves a
  stale float in `instance.__dict__` shadowed by the descriptor — harmless but
  confusing, and `__deepcopy__` copies it (`animatable.py:1261-1286`). With the
  local-variable form the whole flow works end-to-end **[M]**: construction,
  recorded animation, materialization `[T,1,1]`, clone, validation.

### Claim 2 — validation moves into a `set_animated_attribute` override; constructor keeps `_finite_number` — **TRUE, with one mandatory refinement**

- All three write routes reach the override **[M]**: `light.intensity = -1`
  (property fset → `set_animated_attribute`, `animatable.py:352-354`),
  `light.set_intensity(-1)` (generated instance closure → `__setattr__` → same
  setter, `animatable.py:696-713`), and `light.set(intensity=-1)`
  (`mob.py:1841-1845` iterates `self.__setattr__`). Each raised
  `AlganConfigurationError`.
- `_init_default_attr` does bypass the setter (writes rows directly via
  `tm.add_mob_attr`, `mob.py:346-360`), so the constructor's own
  `_finite_number` call must stay. Correct as stated.
- **The refinement — measured.** The override receives *tensors*, not just
  scalars: after any state materialization, attribute reads are `[T,1,1]` until
  `clear_buffers()` (**[M]**: shape `(30,1,1)`, `float()` raises `ValueError`;
  back to `(1,1,1)` after `clear_buffers()`). `Animatable.__deepcopy__`'s clone
  loop feeds exactly such a value through the setter
  (`setattr(clone, attr, getattr(self, attr))`, `animatable.py:1290-1291`) —
  **[M]** a naive `float(value)` override makes every `clone()` raise
  `ValueError` whenever the source was materialized earlier in the process.
  The override must validate tensor-tolerantly (e.g. cast then
  `torch.isfinite(v).all() and (v >= 0).all()`), which also keeps the finite
  check meaningful for animated values.

### Claim 3 — delete `Light.set_intensity`; the generated closure would shadow it — **TRUE**

- `generate_animatable_attr_set_get_methods` installs per-instance closures via
  `super().__setattr__(f"set_{attr}", ...)` (`animatable.py:713-716`), i.e.
  into the instance dict; a plain method is only a non-data descriptor, so the
  instance entry shadows it. Measured **[M]**: `"set_intensity" in
  light.__dict__` is True once registered pre-init (same mechanism as
  `"set_location"` for ordinary Mobs), while `"set_intensity" in
  vars(PointLight)` is False.
- `set_intensity` has **no callers anywhere** in `algan/`, `tests/`,
  `benchmarks/` or `docs/` — repo-wide grep finds only its definition
  (`lights.py:139-142`). Deleting it breaks nothing and removes dead code whose
  docstring ("Set the non-negative light intensity...") would otherwise lie
  about which code path runs.

### Claim 4 — renderer reads are exactly `_materialize_render_state` + `HemisphereLight.build_aux`; downstream eats the snapshot triple — **TRUE**

- Attribute reads/writes of a Light's `intensity` in all of `algan/`:
  `lights.py:135` (constructor write), `lights.py:141` (`set_intensity` write),
  `lights.py:422` (`build_aux` read), `render_loop.py:2428`
  (`float(getattr(light, "intensity", 1.0))` read). Nothing else.
- Downstream consumption is snapshot-only: `Scene._materialize_render_state`
  builds `(origin, color, aux)` triples (`render_loop.py:2457-2467`); the
  preflight wraps them in `_LightSnapshot` carrying exactly those three fields
  (`render_loop.py:1024-1033`); `_pack_lights` reads only
  `light.origin` / `light.light_color` / `light._render_aux`
  (`algan/rendering/raytracing/scene_builder.py:2000-2019`); `tracer.py` uses
  lights only for `_render_aux` presence tests and passes them to
  `_pack_lights` (`tracer.py:594, 975, 1193-1196, 1463`).
- The site at `render_loop.py:2428` genuinely breaks unconverted: `float()` on
  a multi-frame `[T,1,1]` tensor raises `ValueError` **[M]**.

### Claim 5 — materialized shapes `[T,1,W]`, broadcast works — **TRUE (measured)**

Re-measured in the harness after `timeline_manager.set_state_to_times(times)`:
`location [T,1,3]`, `color [T,1,5]`, `opacity [T,1,1]` — matching the brief's
numbers exactly — and the plan-having light's `intensity` came out `[T,1,1]`
float32, animating 0.85 → 3.0 across the window. `col =
rgba[...,:-1]*rgba[...,-1:]*opacity` is `[T,1,4]` and `col * intensity`
broadcasts to `[T,1,4]` cleanly **[M]**.

### Claim 6 — move HemisphereLight's intensity multiply out of `build_aux` into the `_AUX_RADIANCE_COLS` scaling — **TRUE, and REQUIRED, not optional**

- If `build_aux` keeps `aux[..., 9:12] = ground * self.intensity`
  (`lights.py:422`) while the loop also scales those columns by
  `opacity * intensity`, intensity is applied **twice** on hemisphere ground
  colour. Note the old line would even run without error against the tensor:
  `ground [3] * intensity [T,1,1] → [T,1,3]` broadcasts over K silently — so
  the double-apply would not announce itself.
- Ordering premise holds: the sRGB decode stays inside `build_aux`
  (`lights.py:420-421`) and the intensity multiply happens after it, in
  `_materialize_render_state` (`render_loop.py:2445-2456`). The comment's
  arithmetic claim is correct — measured `srgb_to_linear(c)*i !=
  srgb_to_linear(c*i)` (max abs diff 1.7e-01 for c≈(0.8,0.4,0.2), i=0.5)
  **[M]** — so decode-before-intensity must be preserved, and is.
- Out-of-lifespan inertness survives: the radiance scaling keeps opacity as a
  factor, and opacity is zeroed outside the lifespan during materialization
  (see C).

---

## B. Call-site inventory

**Attribute reads/writes of a Light's `intensity`** (complete; everything else
in the repo touching the token `intensity` is a different thing):

| Site | Kind | Breaks at `[T,1,1]`? |
| --- | --- | --- |
| `algan/rendering/lights.py:135` | constructor write | No — replaced/refactored under the plan (see A1) |
| `algan/rendering/lights.py:141-142` | write in `set_intensity` | No — method deleted (A3) |
| `algan/rendering/lights.py:422` | read in `HemisphereLight.build_aux` | Yes if kept: silently **double-applies** intensity once the loop scales too (A6). Removed by the plan |
| `algan/render_loop.py:2428-2430` | read + `float()` + conditional multiply | Yes: `float()` raises `ValueError` for T>1 **[M]**. This is the site the plan converts |
| `algan/render_loop.py:1024-1043` | snapshot triple (no `.intensity`) | No |
| `algan/rendering/raytracing/scene_builder.py:1984` (comment) | documents packed col 0:3 as "RGB radiance (intensity premultiplied)" | Still true under the plan |

Deliberately **different things**, found and excluded:
`SETTINGS.raytracing.light_intensity` (`algan/settings/raytracing_settings.py:134,159-169,205`;
setter `algan/rendering/raytracing/settings.py:2372-2375`; consumed only as a
kernel arg of the physical path tracer,
`raytrace_kernels_taichi.py:3477,3688`, exercised by
`tests/unit_tests/test_raytracing_unit.py:189,262`); `environment_intensity`
(`algan/scene.py:200,493,569,893`; `render_loop.py:475`; `tracer.py:1162,1471`);
material params `emissive_intensity` / `env_map_intensity` /
`specular_intensity` (`shading_taichi.py:42-44,911+`).

**Constructor-keyword usages (`intensity=...`)** — these keep working unchanged
because the signature and its validation stay. They are numerous and all
constant: `tests/fast/scene.py:78,83,88`; six scenes under
`tests/full_renders/scenes/` (notably `materials_and_lighting.py:28-33,175-197,226-231`);
~40 sites under `benchmarks/` (e.g. `_linear_color_check.py:255-266,371-372`,
`_cap_rim_probe.py:45-52`, `_tonemap_render_check.py:209`); docs examples
(`docs/source/advanced_user_tutorials/lighting_and_shadows.rst:90,117,140,...`).
None reads the attribute back expecting a float.

---

## C. Entitlement

**No new entitlement path opens, provided three conditions hold — two of which
the plan already satisfies, one of which is the validation itself.**

The mechanism that makes out-of-lifespan frames inert is *not* generic: only
the **opacity** timeline zeroes rows outside `[spawn, despawn)`
(`add_mob_attr` sets `record_end_points = attr == "opacity"`,
`algan/animation_timeline/timeline.py:2236-2240`; mask applied at
`timeline.py:1487-1503`). A new `intensity` timeline has
`record_end_points=False`, so its rows keep their values outside the lifespan.
That is harmless because intensity enters the render exclusively through
products in which opacity is also a factor:

- RGB path: `col = rgba[...,:-1] * rgba[...,-1:] * opacity`, then
  `* intensity` (`render_loop.py:2427-2430`) — opacity-zeroed frames stay zero.
- Aux path: `_AUX_RADIANCE_COLS` scaled by `opacity * intensity`
  (`render_loop.py:2445-2456` plus the claim-6 change) — same guarantee; the
  class comment's stated purpose ("a genuinely inert all-zero row",
  `lights.py:127-132`) is preserved verbatim by multiplying through the same
  opacity factor.
- Lights whose lifespan misses the window are filtered out entirely
  (`render_loop.py:2406-2409`); kept-but-partially-live lights ride the
  zero-row rule documented at `render_loop.py:2374-2383`.

The three conditions:

1. **Intensity must stay finite and ≥ 0.** A NaN/inf row would poison the
   products (`0 * inf = NaN`), resurrecting emission on out-of-lifespan frames
   *through* the opacity multiply. The `_finite_number(minimum=0.0)` validation
   (constructor + tensor-tolerant override, A2) is load-bearing for the
   entitlement invariant, not just user-friendliness. **[R]** grounded in the
   masked-fill semantics at `timeline.py:618-621`.
2. **The multiply chain must keep opacity as a factor** — it does, in both
   paths above.
3. **`build_aux` must stop multiplying by intensity** — required, see A6;
   otherwise hemisphere ground colour gets intensity twice (a live-light error,
   not a leak).

**RectAreaLight's `col_rows = col_f / k`** (`render_loop.py:2441-2443`): with
animated intensity folded into `col_f` upstream, dividing by k distributes per
frame exactly as today — today's order is likewise
`(rgb·glow·opacity)·intensity / k`. Each of the K emitter rows carries 1/K of
the (per-frame) power. No distortion, no leak. **[R]** over IEEE division
semantics; identical operation sequence to today.

Nothing was found where a legitimately live light's row is zeroed: the
intensity timeline is never endpoint-masked, and completed animations land on
their exact stored target values (measured: mid/end frames read exactly 3.0
**[M]**; edits store final values, replay holds final parameters —
`timeline.py:2776-2787`).

---

## D. Byte-identity for constant intensity

Invariant: **a scene whose lights never animate intensity renders byte-for-byte
identical to today.** Verdict: satisfied by the plan, on both arms.

- **intensity == 1.0:** today the multiply is skipped entirely
  (`if intensity != 1.0`, `render_loop.py:2429`); the plan multiplies by a
  `[T,1,1]` float32 tensor of 1.0 unconditionally. `x * 1.0f32` is bit-identical
  to `x` — measured over adversarial values including ±0.0 and inf **[M]**, and
  guaranteed by IEEE-754 (exact result, no rounding) **[R]**.
- **intensity == 0.85:** today `col * 0.85` (Python double wrapped-scalar);
  plan `col * tensor(float32(0.85))`. PyTorch converts a Python scalar to the
  tensor's dtype before an elementwise op, so both arms multiply by the same
  nearest-float32 value of the same double literal; `cast_to_tensor` uses
  `torch.get_default_dtype()`, which algan pins to float32
  (`algan/__init__.py:67`; `algan/utils/tensor_utils.py:70`). Measured
  bit-identical over 12k elements including extremes **[M]**. The association
  is unchanged — both arms compute `((rgb·glow)·opacity)·k`, appending the
  k-multiply last; regroupings were measured to change bits, but none occurs
  here **[M]**.
- **Where the constant comes from:** a never-animated attribute has no edit
  records; every frame reads the initial `current_state` value seeded by
  `_init_default_attr` — exactly the authored float, every frame **[R]** from
  `timeline.py:1141-1180` (sentinel final-state edit, timestamp inf), consistent
  with the measured t₀ read of 0.85 before the animation's start **[M]**.
- **Order of multiplies:** identical sequence, identical grouping (above). The
  only new arithmetic in a constant-intensity scene is the identity multiply —
  covered by the first bullet.
- **Not byte-identical, correctly:** mid-animation frames have no today-counterpart
  (that is the feature). The invariant is scoped to never-animating scenes and
  holds there.

Caveat for the record: this is measurement of the torch-CPU elementwise ops in
isolation plus source reasoning about the exact expression at
`render_loop.py:2427-2430`; I did not run full pixel-suite renders (read-only
audit; baselines are machine-specific anyway). The reasoning chain has no gap:
same inputs, same op, same dtype, same order.

---

## E. What else must move

Almost nothing; the generic machinery already treats a registered attribute
uniformly. Itemized:

- **`clone`** — nothing to change. `__deepcopy__` copies `animatable_attrs`
  (`animatable.py:1218`), regenerates accessors (`:1288`), and copies each
  attribute's rows through the property (`:1290-1291`). Measured working
  end-to-end **[M]**, including independent post-clone animation — contingent
  only on the tensor-tolerant validator (A2).
- **`become` / `_MORPH_ADOPTED_ATTRS`** — nothing. `_MORPH_ADOPTED_ATTRS` is for
  *plain* attrs (`("shader","two_sided","closed_shell")`, `mob.py:331`);
  intensity rides `animatable_attrs`, which become's value-copy loops handle
  (`mob_morph.py:984-990` same-kind; `:1258-1263` soup path). The batch-expansion
  and reorder loops skip width-1 attributes by shape guard
  (`mob_morph.py:603-610`, `691-696`); intensity rows are `[1,1,1]`, so they
  skip safely.
- **Packed-mob path** — nothing. Packing is driven by the same
  `animatable_attrs` bookkeeping; lights are never members of a pack in this
  repo, and the generic path needs no per-name knowledge.
- **`Scene.use_manim_defaults`** — nothing. It spawns a default-intensity
  PointLight and touches camera/material/background only
  (`algan/manim_defaults.py:205-230`).
- **Manim compat layer** — nothing. No light-intensity usage anywhere under
  `algan/mobs/manim_compat.py`, `opengl_compat.py`, `image_compat.py`,
  `point_cloud.py` (grep empty).
- **Docs autosummary stubs** — nothing checked in. `docs/source/reference/` is
  generated at build time (`docs/source/reference_index/rendering.rst:10`
  autosummary of `~rendering.lights`); keeping docstrings current (G) is what
  feeds it.
- **`algan/__init__.py` export curation** — nothing. No module-level name is
  added or removed; the Light classes remain exported as today.
- **Two implementation notes that are not file changes elsewhere but belong in
  the diff:** (i) the constructor-local refactor from A1; (ii) `Mob.set(intensity=...)`
  validity on non-lights is governed by the Scene-wide
  `attr_to_timeline` union (`mob.py:1744-1761`) — once any Light exists, a
  `Square.set(intensity=...)` passes the name check and writes an inert plain
  attribute. That behaviour is pre-existing for every per-class attribute (the
  code comments it), not introduced by this change; noting it so nobody mistakes
  it for a regression.

---

## F. Test coverage today, breakage, and placement

**Currently covering `Light.intensity` behaviour:**

- `tests/unit_tests/test_ux_regressions.py::test_light_parameters_are_validated_instead_of_silently_clamped`
  (defined at `:672-678`; **not** fast-marked — neighbouring markers are at
  `:634` and `:854`). Asserts `PointLight(intensity=-1)` raises
  `AlganConfigurationError`. **Does not break**: the constructor keeps
  `_finite_number` (claim 2). Not marked `fast`, so it is outside the dev loop
  but CI runs it.
- Pixel suites, which pin the *renderer-side* consumption of constant
  intensities (and, for the hemisphere, the exact `ground × intensity` path the
  plan moves):
  - `tests/fast/test_fast_render.py::test_the_fast_scene_renders_and_matches_its_baseline`
    (`:175`) — three lights at 0.45 / 0.85 / 0.6 (`tests/fast/scene.py:78-88`).
  - `tests/full_renders/test_full_render_scene.py::test_full_render_scene[materials_and_lighting]`
    — all six light types including `HemisphereLight(..., intensity=0.5)`
    with a ground colour (`scenes/materials_and_lighting.py:197-201`), i.e. the
    one scene that exercises the `build_aux` change. Skips itself under `CI`.
- Adjacent but distinct: `test_inert_settings.py` (settings-level
  `light_intensity` refusal, message text mentions `PointLight(intensity=2.0)`
  — unaffected), `test_tonemapping.py::...` `_lambert(light_intensity=...)`
  (kernel arg), `test_manim_shader_render.py:507` (intensity-1 rig — guarded by
  D's ×1.0 identity), `test_deterministic_shadow_opacity.py:48` (constructor
  kwarg).

None of these breaks under the plan *provided* the byte-identity argument in D
holds; if it does not, the pixel suites are what catches it.

**Where new tests belong.** There is no dedicated lights test file; the suite is
organised by subsystem (`tests/README.md:336-338`). Create
`tests/unit_tests/test_lights.py` for: the three write routes raising
`AlganConfigurationError` on negative/non-finite values (tensor-tolerant
validator included, since `clone()` exercises it — A2); assignment recording an
animation and materializing to `[T,1,1]` with the right per-frame values;
constant-intensity snapshot equivalence (the `col`/aux products equal today's
floats). The pixel side is already owned by the suites above; do not add a new
render fixture for it.

**Fast marker?** Default **no** (`tests/README.md:104-123`). The honest answer
to "which file must someone edit for this test to start failing" is
`algan/rendering/lights.py` — it is a feature test. The timeline machinery it
rides is already canaried wholesale by the marked timeline files, and the
render_loop-side consumption is watched by `tests/fast`'s own pixel comparison.
Marking one in would also require a row in `tests/README.md`'s table or
`test_fast_suite_curation.py` fails (`tests/README.md:143-146`). If the author
nonetheless wants one cheap canary — e.g. "an animated intensity changes the
packed light snapshot" — mark that single test and add the table row; do not
mark the module.

---

## G. Docstrings that become wrong or incomplete

Per `DOCSTRINGS.md`: lights are Tier 1 (§1), animatable attributes are API
surface documented in an `Attributes` section with units/ranges (§10), and
anything that records carries an `Animation` section saying recorded-vs-immediate,
default duration, propagation, and spawn-order constraints (§6). In
`algan/rendering/lights.py`:

1. **Module docstring, lines 8-10.** Says location/color/opacity are animatable
   and "(intensity, decay, cone angles, ...) are plain per-light constants."
   Becomes false. Move `intensity` into the animatable list; leave decay/angles
   as constants.
2. **`Light` class docstring, lines 116-123.** `intensity` documented only as
   "Scalar multiplier applied to the light's color." Needs: default (``1.0``),
   constraint (finite, ≥ 0 — and a `Raises` entry for
   `AlganConfigurationError`), an `Attributes` section entry (multiplier, unitless,
   range ≥ 0), and an `Animation` section stating: assignment /
   `set_animated_attribute` records an animation interpolating over the current
   context's duration (1 s default); propagates like any Mob attribute write;
   no spawn-order constraint (pre-spawn writes are instant setup, as with
   `location`).
3. **`Light.set_intensity`, lines 139-142.** Deleted with the method (claim 3);
   its promise ("Set the non-negative light intensity and return this light")
   must survive as the property/setter behaviour documented in (2).
4. **`PointLight` class docstring, lines 218-219** ("intensity — Color
   multiplier.") — same additions as (2), minus what it inherits; subclasses
   must not carry a stale "constant" implication either.
5. **`_AUX_RADIANCE_COLS` comment block, lines 127-131.** "those columns must
   scale with the light's per-frame opacity at materialization" → now
   `opacity * intensity`. Internal comment, but it is the contract the loop
   implements; leaving it stale invites the double-apply bug of A6 back.
6. **`HemisphereLight.build_aux`, lines 394-423.** The comment (lines 414-418)
   insists the decode happens "before the intensity multiply" *inside
   build_aux*. After the move the sentence's location claim is wrong even
   though its arithmetic point remains true; rewrite to say cols 9:12 carry the
   decoded ground colour, scaled by `opacity * intensity` downstream at
   materialization. The column table at lines 196-198 should note the scaling
   too.

Internal (not docstrings, but adjacent and will mislead): the ingest comment at
`algan/render_loop.py:2412-2417` ("alpha, opacity and intensity below are all
linear scalars").

**`.rst` files stating/implying constancy:**

- `docs/source/advanced_user_tutorials/lighting_and_shadows.rst:152-159` — the
  explicit note: intensity and shape parameters "are plain per-light constants,
  not animatable attributes ... To show two intensity settings, render two
  videos." Must be rewritten (keep the sentence about decay/distance/cone
  angles, which stays true).
- `docs/source/advanced_user_tutorials/lighting_and_shadows.rst:72-73` — "every
  light's ``location`` and ``color`` are animatable", implying by omission that
  intensity is not. Update the enumeration.
- Checked and clean: `renderer_limitations.rst:944-950` concerns
  `SETTINGS.raytracing.light_intensity` (different thing, still accurate);
  `reference_index/*.rst` contain no intensity claims; no other tutorial page
  states light intensity is fixed.

---

## Git status at end of run

```
$ git status --porcelain
?? OX_INTENSITY_AUDIT.md
```

The only entry is this report itself — the audit modified no tracked file and
created nothing else.
