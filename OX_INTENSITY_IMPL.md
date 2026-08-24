# Implementation: animatable `Light.intensity`

Companion to `OX_INTENSITY_AUDIT.md` (the plan) — this is the record of
implementing it. No commits were made; everything below is in the working
tree.

## What changed and why

### `algan/rendering/lights.py`

- **`Light.__init__`** registers `"intensity"` via
  `register_attrs_as_animatable(["intensity"], Light)` *before*
  `super().__init__()`, validates the constructor argument into a **local**
  with `_finite_number("intensity", intensity, minimum=0.0)`, and seeds the
  timeline row after the super chain with
  `self._init_default_attr("intensity", cast_to_tensor(intensity))`. The local
  (not a `self.intensity` assignment) is mandatory: once the property exists,
  that assignment would route into `set_animated_attribute`, which reads state
  `Animatable.__init__` has not built yet (audit §A1). Added the
  `cast_to_tensor` import, mirroring `mob.py`.
- **Validation moved to the one funnel**: deleted the dead `Light.set_intensity`
  method (no callers anywhere; the generated instance closure shadows it anyway)
  and overrode `set_animated_attribute` to run every write through
  `_validated_intensity`, which is tensor-tolerant (`torch.isfinite(...).all()`,
  `(value < 0).any()` on tensors; `_finite_number` for scalars). Tensor
  tolerance is required because materialized attribute values are `[T,1,1]` and
  `__deepcopy__` copies attributes through their setters.
- **Load-bearing comment** on `_validated_intensity`: a light outside its
  lifespan is made inert only by its *opacity* row being zeroed, and intensity
  reaches the render multiplied by that opacity — a NaN/inf intensity would turn
  `0 * inf` into NaN and resurrect emission on frames where the light does not
  exist. (The intensity timeline itself is not endpoint-masked;
  `record_end_points` is set only for `opacity`.)
- **`HemisphereLight.build_aux`** no longer multiplies the ground colour by
  intensity: `aux[..., 9:12] = ground`. Applying it there *and* in the
  radiance-column scaling at materialization would double-apply intensity, and
  the old line would have broadcast silently (`ground [3] * intensity [T,1,1]`)
  rather than erroring. The sRGB decode stays inside `build_aux`; the comment
  now says cols 9:12 carry the decoded ground colour and the opacity/intensity
  scaling happens downstream at materialization.
- **Comments/docstrings updated**: module docstring moves `intensity` into the
  animatable list (decay/distance/cone angles/emitter sizes stay constants);
  `_AUX_RADIANCE_COLS` class comment now says "opacity *and* intensity";
  `build_aux`'s column-table row for 9-11 notes the decode-here /
  scale-downstream split; `Light`'s class docstring gained the full parameter
  description, `Raises`, `Attributes` and `Animation` sections per
  `DOCSTRINGS.md`; `PointLight` points at the base class's entry instead of
  restating ("As on :class:`~.Light`: ...").

### `algan/render_loop.py` — `RenderLoopMixin._materialize_render_state`

- Replaced `float(getattr(light, "intensity", 1.0))` + conditional multiply with
  a per-frame read: `getattr(light, "intensity", None)`; when present,
  `col = col * intensity` where `intensity` is the materialized `[T,1,1]` row
  (shapes printed and verified, see below). The multiply stays **last**, after
  alpha and opacity, so constant-intensity scenes still compute
  `((rgb * glow) * opacity) * k` exactly as before; multiplying by an all-ones
  tensor is bit-exact under IEEE-754, so scenes at `intensity == 1.0` are
  unaffected too. Lights without any `intensity` attribute (the stub objects
  render-loop tests drive this mixin with) skip the multiply.
- In the `radiance_cols` block, the aux columns are scaled by intensity as a
  **separate multiply applied before** the opacity multiply — reproducing the
  association `build_aux` used to bake in, `(ground * intensity) * opacity`,
  because float multiplication is not associative and folding the two scalars
  first could differ in the last bit. A code comment says exactly this.
- Fixed the ingest comment above: alpha and opacity are linear scalars,
  intensity is now described as a linear per-frame row.

### Docs

- `docs/source/advanced_user_tutorials/lighting_and_shadows.rst`: the "Light
  Types" enumeration now includes ``intensity`` among the animatable
  attributes; the note claiming intensity is a plain per-light constant is
  rewritten (intensity animatable; shape parameters still constants) and now
  carries a new `.. algan:: LightingAnimatedIntensity` example (unique name,
  `from algan import *`, exactly one `Scene.save_video()`), house style.
- `docs/source/reference_index/rendering.rst`: added a "Runtime light
  attributes" section with a `.. py:attribute::`
  `algan.rendering.lights.Light.intensity` directive, copied in shape and
  wording from animation.rst's "Runtime Mob attributes" section (dynamically
  registered attributes never appear under `autoclass`).

### Tests — new `tests/unit_tests/test_lights.py`

Twelve tests, none marked `fast` (feature test for `lights.py`; marking one in
would also require a `tests/README.md` table row):

1. registration: `"intensity" in light.animatable_attrs`; the attribute reads
   back as a tensor, not a float;
2. recorded animation: value ramps 1 → 3 across a timed context, strictly
   between mid-animation, exact at both ends after materialization;
3. `Off()` lands instantly (value correct at t = 0);
4. validation on every write path: `PointLight(intensity=bad)`, assignment,
   `set_intensity`, `set(intensity=...)` each raise `AlganConfigurationError`
   matching "intensity", for `-1`, `-0.5`, NaN, `+inf`, `-inf`;
5. tensor-tolerant funnel: the materialized `[T,1,1]` passes
   `_validated_intensity` unchanged; non-finite tensors are rejected through
   the public write paths; a single-frame materialized tensor flows through the
   public setter onto another light;
6. renderer consumption: `_materialize_render_state(0, window)` colour rows
   track the ramp — last frame exactly 3x the first (rtol 1e-2), interior
   frames strictly between and monotonically increasing;
7. hemisphere ground colour (aux cols 9:12) scales **linearly** with animated
   intensity frame-by-frame against the materialized intensity rows — ratio
   `k`, not `k²`, so a double-apply fails this test;
8. byte-identity for a constant `intensity=0.85`: snapshot colour rows are
   `torch.equal` to today's arithmetic
   (`rgba[..., :-1] * rgba[..., -1:] * opacity * 0.85`, decode included)
   computed from the light's own materialized state.

## Audit premises that turned out wrong (or needed adjustment)

The audit's mechanics were all confirmed. Two things surfaced while writing
test 5:

1. **"clone() measured working end-to-end [M]" does not hold on this tree —
   pre-existing, and not about intensity.** On the clean tree at HEAD,
   `clone()` of *any* Mob whose timeline was materialized raises
   `RuntimeError` inside the generic basis-copy path
   (`squish` on an empty `(1, 0, 3, 3)` tensor), before the attribute copy ever
   reaches `intensity` (which sits last in `animatable_attrs`). Verified with a
   plain `Square` on stashed-clean HEAD: identical failure. Likewise, recording
   an edit on a Mob spawned *after* a materialization, then rematerializing,
   breaks generically (`IndexError` in the timeline's edit gather; reproduced
   with `Square.location` on clean HEAD). Consequences:
   - The brief's test 5 as specified ("clone(), assert no raise") cannot pass
     on this tree for reasons unrelated to this change. I reshaped it to pin
     what is actually ours — the validator accepts the `[T,1,1]` object the
     clone loop feeds it, non-finite tensors are rejected through the public
     paths — and documented the blocker in the test docstring.
   - The audit's core claim behind tensor tolerance stands independently: a
     scalar-only `float(value)` validator raises `ValueError` on exactly the
     `[T>1,1,1]` values the funnel must accept, and the validator is the one
     piece of this change that sees those values today.
2. **Bare `PointLight()` defaults to black, not white.** The default scene's
   light passes `color=WHITE` explicitly; a bare-constructed light's Mob
   default colour is black. My renderer-consumption test initially divided by
   an all-zero first frame (NaN ratios). Test fixed to pass `color=WHITE`;
   engine behaviour untouched.

## Verification output

All commands prefixed `ALGAN_USE_DAEMON=0`.

**1. New tests**

```
$ .venv/bin/python -m pytest -q tests/unit_tests/test_lights.py
12 passed, 3 warnings in 0.16s
```

**2. Neighbouring suites**

```
$ .venv/bin/python -m pytest -q tests/unit_tests/test_ux_regressions.py tests/unit_tests/test_materials.py \
    tests/unit_tests/test_doc_examples.py tests/unit_tests/test_lifecycle.py \
    tests/unit_tests/test_deterministic_shadow_opacity.py tests/unit_tests/test_fast_suite_curation.py \
    tests/unit_tests/test_manim_shader_render.py
565 passed, 126 skipped, 37 warnings in 89.16s (0:01:29)
```

**3. Fast suite**

```
$ .venv/bin/python -m pytest -q --fast
fast suite: 20s of its 75s budget (27%)
FAILED tests/fast/test_fast_render.py::test_the_fast_render...matches_its_baseline
1 failed, 274 passed, 1846 deselected, 3 warnings in 20.47s
```

The one failure is the **stated pre-existing CPU-baseline drift**, message
verbatim:

```
AssertionError: fast.mp4 differs from its baseline by up to 5 channel values
(worst at frame 27); see /home/user/algan/tests/fast/output_errors/fast.mp4
```

— exactly "5 channel values at frame 27" as declared in the brief for the
clean tree. Not chased, no baseline touched.

**4. Ruff**

```
$ .venv/bin/ruff check --no-fix algan tests      # exit 1: 24 findings
$ .venv/bin/ruff format --check algan tests      # exit 1 (before fixing my file)
```

Findings on my files (fixed during the work): `I001` un-sorted import block in
`tests/unit_tests/test_lights.py`, plus formatting — both repaired; my file now
passes both tools clean.

All remaining findings are **pre-existing**: I diffed the sorted finding list
on stashed-clean HEAD against my tree —

```
$ diff /tmp/opencode/ruff_clean.txt /tmp/opencode/ruff_mine.txt
IDENTICAL FINDING SETS (all pre-existing)
```

(24 entries: F401/SIM114 in `algan/constants/color.py`, D301 in
`algan/manim_defaults.py`, F401s in `algan/mobs/plots.py`, F841/I001/F811 in
the raytracing modules, I001 in `algan/scene.py`, F401 in
`algan/utils/audio_utils.py`, and I001/F401s in
`tests/unit_tests/test_rate_funcs_and_ux.py`.) Note `ruff check` is configured
with `fix = true`, so `--no-fix` was used throughout except deliberately on my
own new file.

**5. Byte-identity experiment (the point of the exercise)**

`tests/fast/scene.py` carries three constant non-unit intensities
(`AmbientLight(0.45)`, `PointLight 0.85 / 0.6`). After the change:

```
$ md5sum tests/fast/algan_outputs/fast.mp4
6e95d737ff48bf7d90ff86be76507a7a  tests/fast/algan_outputs/fast.mp4
```

Exactly the clean-tree value, stable across repeated renders on the changed
tree (rendered three times total: via `--fast`, via `tests/fast`, and again
via `tests/fast/test_fast_render.py`). Combined with the unchanged failure
message above, the change moves nothing for never-animating scenes.

**6. Hemisphere arm (not covered by `tests/fast`)**

Throwaway script `/tmp/opencode/hemi_ab.py` (not in the repo):
`SMOKE_TEST`-quality scene lit solely by
`HemisphereLight(color=WHITE, ground_color=(0.45, 0.3, 0.15), intensity=0.5)`
plus a sphere and floor, a few frames.

```
changed tree: b28725e90550467e779016f1afa2b64c  /tmp/opencode/hemi_changed.mp4
clean tree:   b28725e90550467e779016f1afa2b64c  /tmp/opencode/hemi_clean.mp4
```

Byte-identical (md5 equal), so moving the ground×intensity multiply out of
`build_aux` into the materialization loop changed nothing for a constant
intensity. `git status --porcelain` after `git stash pop`:

```
 M algan/render_loop.py
 M algan/rendering/lights.py
 M docs/source/advanced_user_tutorials/lighting_and_shadows.rst
 M docs/source/reference_index/rendering.rst
?? tests/unit_tests/test_lights.py
```

— nothing lost by the stash round-trip.

## Adversarial re-read of the diff

Two mandated questions:

**Is every name imported in the module that uses it?**
Yes. `cast_to_tensor` was newly imported into `lights.py` (the only new name
either engine module uses); `render_loop.py` needs nothing new (`getattr`,
existing locals). `ruff check --no-fix` reports zero F821/F401 in either file,
and the finding set matches clean HEAD exactly. The test module imports every
name it uses (ruff clean after fixes); the docs example resolves against the
live `algan` namespace (`test_doc_example_uses_public_api` passed).

**Does every tensor have the shape I think?** Printed, not assumed
(`/tmp/opencode/shape_probe.py`), over point + hemisphere + K=4 area lights +
a stub light without the attribute:

```
light.color          (45, 1, 5)      light.opacity   (45, 1, 1)
light.intensity      (45, 1, 1) float32
col (rgb*a*op)       (45, 1, 4)      col * intensity (45, 1, 4)
snapshot colour      (45, 1, 4) .. (45, 4, 4)        snapshot aux (45, K, 13)
aux[...,9:12]        (45, K, 3)      g * intensity   (45, K, 3)   # K in {1,4}
g*i*op               (45, K, 3)
stub light (no intensity attr): skip path OK, snapshot intact
```

So `[T,1,1]` broadcasts against both `col [T,1,4]` and `aux[...,9:12] [T,K,3]`
directly, as the code assumes; the stub-light path skips cleanly.

Other things checked while re-reading: the override's signature matches the
parent's (`attr, value, recursive=True`) and returns super's result; the
`Raises` role in the class docstring is well-formed RST
(`:class:`.AlganConfigurationError``); no remaining reader of `light.intensity`
exists anywhere in `algan/` besides the two sites this change converted
(grepped); `benchmarks/renderer_audit/three_render.mjs`'s `l.intensity` is a
JavaScript object built from its own JSON, not a Python Light; the
`SETTINGS.raytracing.light_intensity` kernel argument is a different thing,
untouched.

## What I did NOT verify

- **CUDA.** This container has no GPU. All verification is the CPU path; CUDA
  behaviour and CPU/CUDA divergence are uncheckable here. The committed
  `expected_outputs_cuda/` baselines were not touched and cannot be
  regenerated from here.
- **`tests/full_renders`.** Deliberately not run, per the brief — its
  baselines are per-machine and would say nothing here. (Note its
  `materials_and_lighting` scene is the dense pixel-level exercise of the
  hemisphere path; the throwaway A/B in step 6 stands in for it.)
- **A full Sphinx docs build.** The structural tiers ran (`test_doc_examples`
  passed, including the new directive's name/uniqueness/star-import/
  single-video rules and its authoring execution), but the rendered page and
  the new `.. py:attribute::` entry were not built/viewed with
  `docs/make_and_open_docs.py`.
- **The Monte Carlo path tracer arm** (SPP > 1): the fast scene renders on the
  deterministic route; the SPP>1 tracer consumes the same packed snapshots
  (audit §B/C reasoning), but no SPP>1 render was executed here.
- **Audio/Speech contexts and other exotic write routes** (updater-driven
  writes land through the same `set_animated_attribute` funnel and are covered
  by construction, but no dedicated test drives them).
- **The two pre-existing generic bugs documented above** (clone-after-
  materialization; edits recorded after a materialization by mobs spawned
  after it) were characterized and worked around in the test, not fixed — out
  of scope for this brief.
