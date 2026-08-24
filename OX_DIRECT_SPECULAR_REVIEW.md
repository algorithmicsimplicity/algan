# Review: the "missing factor of ~10" in the direct-specular add-back

**Verdict: nothing the scatter site passes `direct_specular_lobe` differs from
what the stage consumes — every argument agrees, verified against the packed
batch. The excess movement comes from the WEIGHT, not a lobe argument: under
the default glossy configuration (split-sum prefilter, mode 3) `R` is replaced
by `_material_env_brdf`'s directional albedo (~0.34 red) *before* the add-back
reads it, so the weight is ~25x the `R * _mirror_share(roughness)` ≈ 0.0135
the arithmetic assumed — and 0.336 × a sane 0.54 lobe reproduces the measured
205 → 230 exactly.**

The brief's premise ("something the scatter site passes differs") is refuted.
The unit-test identity transfers to the render; what fails is the prediction's
model of where `R`'s value comes from on this route.

## The invariant

For the add-back to move a pixel by only the predicted sliver, this must hold
at every hit it fires on:

> Every argument of `direct_specular_lobe` must carry the same value the
> fragment pipeline handed `_stage_standard` at that hit — same position, view
> direction, prepared normal, albedo, and transport scalars with
> `tri_extra(corner c) ≡ tri_mat(slot 8/9/12)`; `ior = 1.5` must make the
> lobe's F0 equal the stage's hard-coded 0.04; and `_energy_scale` must be
> exactly 1. Then lobe ≡ stage-specular (the unit test's identity), and
> Δpixel = w·α·(R + trans_share)·lobe is predictable from the stage's own term.

**No argument violates it.** What breaks the *prediction* is upstream of the
call: `(R + trans_share)` at the call site must be read after the same
throttling the prediction assumed. On the default route it is not.

## Lead-by-lead evidence

The batch was built host-side from a replica of `tests/fast/scene.py`'s rig
(ambient 0.45 white / directional 0.85 white / point 0.6 BLUE_A; `Icosahedron`
with `MeshStandardMaterial(RED, roughness=0.35, metalness=0.4)`), merged via
the real `_merge_scene`, inspected before any kernel compiles
(`/tmp/opencode/dump_batch.py`; values below are its output).

1. **Roughness — same number.** In this fully constant-promoted batch the
   scatter samplers read promoted triangles through their synthesized material
   map (`wavefront_kernels_taichi.py:312-343`, guard at `:329`; map synthesis
   in `scene_builder.py:594-617`, channels refl/rough/ior/transmission,
   bitmask `1|2|4|8` at `:617`). Dumped for the standard-material prims
   (`tri_mat_id == 4`, 20 tris): material-map texel `[0.4, 0.35, 1.5]`;
   stage side `tri_mat` slot 8 = 0.35, slot 9 = 0.4 (`shading_taichi.py:1108-
   1109`, always called with `off=0`: `:1581-1583`, `:1624-1626`,
   `:1690-1693`). Identical.
2. **IOR — 1.5, F0 = 0.04 both sides.** `MeshStandardMaterial = PBRMaterial`
   has no `ior` parameter (`materials.py:441-467`), so
   `_derive_material_surface_params` packs the fixed 1.5
   (`primitives.py:584-589`); the dumped texel carries m[2] = 1.5 and the map
   bitmask delivers it (`wavefront_kernels_taichi.py:370-376`). The lobe's
   `((1.5−1)/(1.5+1))² = 0.04` (`shading_taichi.py:898-899`) equals the
   stage's hard-coded `f0 = 0.04·(1−m)+rgb·m` (`:1117`) to the bit.
3. **prim/f — the same row.** Both consumers take `prim`/`f` from the same
   variables of the same sheet iteration (`sheet_resolve_taichi.py:400` set,
   `:434-442` shading vs `:602-607` add-back); the dump confirms prim 12's
   meta row points at its own group's maps.
4. **_energy_scale — confirmed exactly 1.0.** `LINEAR_COLOR_SPACE =
   env_flag("ALGAN_LINEAR_COLOR", True)` (`settings.py:78`) and the gate is
   compile-time (`shading_taichi.py:230-231`), so the scale compiles out.
   Also checked: normals are the same fetch on both paths (`sn` at
   `sheet_resolve_taichi.py:422-433` vs `normal` at `:518-521`, identical
   `_tri_normal_g(0, …)` calls), position/view/albedo are the same variables,
   lights and `lvis` are shared scope, and the monolith's copy
   (`wavefront_kernels_taichi.py:3087-3096`) reads the same inputs.

## What actually happens on tests/fast

`GLOSSY_REFLECTION` and `GLOSSY_PREFILTER` default **on**
(`settings.py:1734`, `:1765`), so `glossy_reflection_mode()` returns 3
(`settings.py:1790-1805`), passed as the resolve's template at
`raster_pipeline.py:2014`. For an opaque reflective sheet — the icosahedron:
reflectivity 0.4 ≥ 0, T = 0 ≤ 1e-4, roughness 0.35 > `_GLOSSY_MIN_ROUGHNESS`
= 1e-4 (`raster_taichi.py:2343`), bounces left > 0, `gl_taken` still False
(`sheet_resolve_taichi.py:236`) — the substitution branch runs and the
throttle branch does **not** (`sheet_resolve_taichi.py:538-564`):

```python
if ti.static(glossy == 3):
    if (not gl_taken) and ... and (T <= 1e-4) and (rough > _GLOSSY_MIN_ROUGHNESS) ...
        R = _material_env_brdf(...)      # :554 — R REPLACED wholesale
    else:
        R *= _mirror_share(rough)        # never reached for this material
```

By the time the share (`:582`) and the add-back (`:602`) read `R`, it holds
the lobe's directional albedo E(n·v), not the throttled Schlick value. The
substitution shipped earlier (`4ecc07c`, committed; only the +37-line
add-back block is new in this tree).

Numbers (red channel; replica script `/tmp/opencode/lobe_weight_check.py`):

| quantity | value |
| --- | --- |
| `_mirror_share(0.35)` | 0.0326 |
| Schlick R (face-on) × throttle — assumed weight | 0.413 × 0.0326 = **0.0135** |
| `_material_env_brdf` — actual weight | **0.336** (flat: 0.334–0.351 for n·v 0.5–0.95) |
| measured pixel | 205 → 230 ⇒ ΔL = decode diff = **0.181 linear** |
| implied lobe under assumed weight | 0.181 / 0.0135 = **13.4** ← the "30x paradox" |
| implied lobe under actual weight | 0.181 / 0.336 = **0.54** — an ordinary GGX specular for this rig |
| forward prediction at that lobe | assumed weight ⇒ +1.1 bytes ("predicts 2"); actual weight ⇒ +27 bytes vs **+25 measured** |

So the lobe never was 30x too big; the weight was ~25x bigger than assumed.
The stage carries the direct-light lobe locally at `share = 1 − R − T ≈ 0.66`
here, not ~0.987 — for the same reason.

## Consequence

The extended add-back's movement on `tests/fast` is arithmetically correct
behaviour, not an argument bug: under mode 3 the local share had already been
reduced by the full directional albedo E, so adding `E · lobe` restores unit
total weight, and delta lights remain disjoint from the traced ray's
environment return. The "factor of ten" was the prediction pricing the
add-back at a weight (`R·_mirror_share`) that no opaque reflective sheet ever
has on the default route. The scoping decision itself (gate at
`sheet_resolve_taichi.py:601`, `T > 1e-4`) can be revisited on that basis;
per the brief, nothing here was changed.

Scratch files: `/tmp/opencode/dump_batch.py` (batch dump),
`/tmp/opencode/lobe_weight_check.py` (numeric closure). No tracked file was
edited.

## Working-tree state

Uncommitted modifications to nine `algan/` files (the feature under review)
plus two benchmark docs were present throughout, except for a ~9-minute
window (09:36–09:46 UTC) when the concurrent session stashed `algan/`
(`git stash push -- algan/`) to run its suite and popped it back; all
citations above were re-checked against the restored tree. Line numbers refer
to that state.

## What I did not verify

* **No render was run.** The 205 → 230 endpoints are taken from
  `TASK_mirror_transmitted_lobe.md` §0 / `REPORT.md` §9.3.1, and the closure
  is arithmetic against them plus the pinned identity — not a fresh A/B
  render of the extended gate (that would have required editing tracked
  files). In particular the implied lobe 0.54 was not extracted per-pixel
  from a frame; it is inferred, and the conclusion is insensitive to it only
  because E is nearly flat across n·v for this material.
* **CPU only.** No CUDA machine was available; all route/template claims
  (mode 3 active, batch accepted by the sheet route, promoted maps as the
  scatter source) rest on defaults and source reading, consistent with but
  not re-measured by `tests/fast`.
* **GREEN/BLUE channels** were not measured; only the brief's red-channel
  figures were reproduced. (Its albedo makes E_g ≈ 0.06, so those channels
  should move far less.)
* **The wavefront-monolith site** (`wavefront_kernels_taichi.py:3087-3096`)
  was source-read, not executed: it applies the same weight off the same
  substituted `R`, but scenes routed through it (bounces, custom scatters)
  were not exercised.
* **Multi-sheet pixels** (the `gl_taken` latch across several qualifying
  sheets of one walk) were reasoned about, not observed; tests/fast's opaque
  solid shades one sheet per sample.
