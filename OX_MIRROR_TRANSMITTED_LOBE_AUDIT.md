# Audit: why Algan's mirror shows a transmitted, doubly-tinted lobe

Scope: read-only audit of `/home/user/algan` against the defect stated in
`benchmarks/renderer_audit/TASK_mirror_transmitted_lobe.md`. No render was run
(per the brief). The only execution was a throwaway numpy transcription of the
share arithmetic (`/tmp/opencode/q4_share_replica.py`, no taichi/algan import),
run for §Q4.

**Pinned to HEAD `1e204d2`; see "Working-tree state" below.** Every file:line
citation in this report refers to that commit.

Route context used throughout (all default settings): primary visibility
resolves through the **sheet route** (`HYBRID_RASTER`
`settings.py:870`, `ANALYTIC_AA` `settings.py:1261`, batch accepted at
`tracer.py:2389-2402`); its reflection/refraction continuations spawn into the
pool with `bounces_left - 1` (`sheet_resolve_taichi.py:649,662,699,762,771,…`)
and are drained by the **monolithic** `wavefront_shade`, whose built-in-material
arm is an inline copy of `_scatter_impl` (`wavefront_kernels_taichi.py:2908`
`if ti.static(len(frag_scatters) == 0):`). The legacy sorted pipeline that
calls `_scatter_impl` via `default_scatter` runs only when forced
(`tracer.py:2253-2262`); within the monolith, `_scatter_impl` itself is reached
only when the scene carries a custom fragment scatter (`:3353-3358`) or for
circuits (`:3364-3368`). This scene has built-in materials only, so the inline
copy is what executes — several findings below turn on exactly that.

The audited materials: `glass` = MeshPhysicalMaterial(metalness 0, roughness
0.05, transmission 1, ior 1.5, albedo linear (0.584, 0.761, 0.423));
`mirror` = MeshStandardMaterial(metalness 1, roughness 0, albedo white)
(`scenes/matlight_pbr_subset.json`; mapping in `benchmarks/renderer_audit/
algan_render.py:113-142`). Lights: ambient 0.35 + one directional; no
emissive geometry. Volumetric absorption is inert here:
`attenuation_distance` defaults to ∞ / colour to white ⇒ sigma packs as zeros
(`materials.py:495-496`, `materials.py:217-222`, `primitives.py:850-863`).

---

## Working-tree state (reported as the brief requires)

`git status --porcelain` was clean at session start and again immediately
before this report was written. At the final check, three kernel files carried
**uncommitted modifications this audit did not author**: file mtimes
07:57:33–07:57:51 fall inside the audit session, after the last clean check;
every command this audit ran is read-only (rg/sed/cat/git plus a python script
under /tmp), so the edits came from outside it — presumably concurrent work on
the defect itself. They were left untouched.

What they change, relative to the HEAD text cited below:

* `wavefront_kernels_taichi.py` (`_scatter_impl` ~:1283 and `wavefront_shade`
  ~:3049) and `sheet_resolve_taichi.py` (~:574): the locally-shaded share
  becomes `(one3 - R)` instead of `(one3 - R - trans_share)`. In-place, no
  line shifts. Under that arithmetic the four shares sum to
  `1 + trans_share`, not 1 — i.e. the Q4 identity no longer holds while the
  transmitted ray is still spawned with `trans_energy * tint` (`:3072`,
  `sheet:601`) — recorded here as an observation about a diff in progress,
  not analysed further.
* `shading_taichi.py` `_stage_physical` (~:1094): the ambient term's diffuse
  factor gains `(1.0 - transmission)`. Net +1 line, so shading_taichi
  citations at and beyond old line 1094 shift by +1 in the current tree
  (`:1101/:1102/:1109` → `:1102/:1103/:1110`; `:1145-1154`,
  `:1435`, `:1481/:1509/:1587` likewise).

All other citations (including every `wavefront_kernels_taichi.py` and
`sheet_resolve_taichi.py` line) are valid for both HEAD and the current tree.

---

## Q1 — Count the tints

**Verdict: DEAD END.** The through-path accumulates exactly two albedo factors
and the direct path accumulates the same two. The counts do not differ, so the
count is not the discriminator; per the brief's own rule, that is the answer
rather than a forced one.

Every multiplication by albedo/tint on the photon path
camera → mirror → glass entry → interior → glass exit → wherever:

| # | site | factor | lobe |
| --- | --- | --- | --- |
| 1 | Mirror Fresnel metal share: `f0_metal = clamp(albedo)`, `r_metal = f0_metal + (1-f0_metal)*tail` (`wavefront_kernels_taichi.py:1145-1146`), blended into `R` at `:1147`, called from `sheet_resolve_taichi.py:526-528` | mirror albedo (white ⇒ neutral) | reflected |
| 2 | Mirror locally-shaded share `share * color` where `color` is the stage output (albedo-multiplied lighting), `share = α(1-R-trans_share)` (`sheet_resolve_taichi.py:574-577`) | mirror albedo ×0 (share is 0: R≡1 for white metal) | local shading |
| 3 | Glass **entry**: transmitted ray weight `wt = weight * trans_energy * tint` (`sheet_resolve_taichi.py:601` for a primary; monolith `wavefront_kernels_taichi.py:3072` / `_scatter_impl:1318` for a bounce) | **glass albedo #1** | transmitted |
| 4 | Glass entry throttle-leak: `R *= _mirror_share(rough)` before `share = …(one3 - R - trans_share)` (`monolith:2976`, `sheet:550`, `glossy==3 else-branch`), leaving `share = r_diel·(1-_mirror_share)` ≈ 1.22 % of r_diel, multiplied by `color` (`monolith:3049-3053`) | **glass albedo #2** (small side-lobe) | local shading |
| 5 | Interior Beer-Lambert `weight *= exp(-sigma·seg)` (`monolith:3012-3017`) | none on this scene (sigma = 0, see preface) | (transmitted) |
| 6 | Glass **exit**: `wt = weight * trans_energy * tint` again (`monolith:3072`) | **glass albedo #3** — the dominant second factor | transmitted |
| 7 | Glass exit throttle-leak, same as #4 (`monolith:2976` + `:3049-3053`) | **glass albedo #4** (small side-lobe) | local shading |

Total on the dominant through-path (items 3 + 6): **two** glass-albedo factors.
Internal-reflection sub-paths accumulate none (the dielectric Fresnel lobe is
achromatic by construction, `:1147` with m=0).

The direct camera → glass path: item 3 at the entry (`sheet_resolve_taichi.py:601`)
and item 6 at the exit (`monolith:3072`) — **the same two**, plus the same two
small leaks. Counts equal.

So the double tint is real but symmetric across both paths: any photon that
crosses the solid is tinted once per interface crossing. What differs between
Algan and the reference is not how many tints a path accumulates but *which
lobe dominates the mirror disc* — the reference's untinted Fresnel lobe versus
Algan's doubly-tinted transmitted lobe. (Consistency check that survives Q1's
dead-end: the opaque control reads ONE albedo because with T=0 the only
albedo-carrying return is `contrib = α(1−R)·shaded` — matching the measured
g/r 1.28 vs predicted 1.30.)

## Q2 — The tint-on-spawn + inner-back-surface hypothesis

**Verdict: PARTIAL.** Tint-on-spawn confirmed — but it fires twice, at both
interfaces, and that alone accounts for albedo². The inner-surface-`shaded`
mechanism as stated is refuted as the carrier (its coefficient is zero at
T=1), though the machinery behind it is real and behaves as feared.

* **Spawn-time tinting — CONFIRMED, twice.** `trans_w = trans_energy * tint`
  at every transmissive split: `sheet_resolve_taichi.py:601`,
  `wavefront_kernels_taichi.py:3072`, `_scatter_impl:1318`. A solid crossing
  pays it at entry AND exit, so the through-ray carries `tint²` (g/r 1.698,
  cf. measured 1.77, single-tint 1.303).
* **`shaded` at the inner back surface — refuted as the carrier.** For T=1,
  m=0: `share = α(1 − R − trans_share)` with `trans_share = diel_pass·T =
  (1−r_diel)`, so `share = α(1 − r − (1−r)) ≡ 0` exactly (numeric table in Q4).
  `contrib = share * shaded` (`monolith:3050-3053`) is therefore identically
  zero at both glass interfaces regardless of what `shaded` contains. The one
  exception is the `_mirror_share` leak (item 4 above): scaling R before the
  subtraction leaves `share = r_diel·(1−0.98780) ≈ 1.22 %·r_diel` of genuinely
  albedo-tinted local shading on every glass hit — present, but far too small
  to carry the patch.
* **Is the inside shaded as inside or lit as front?** It is **lit as though it
  faced the light**, not ambient-only. The read of `_MAT_ONE_SIDED` happens in
  `_sided_shading_normal` (`shading_taichi.py:462-469`, slot 26 at `:92`),
  reached from `_run_frag_pipeline` (`:1481/:1509/:1587`), which the bounce
  loop reaches via `_shade_tri_hit` (`wavefront_kernels_taichi.py:2900`;
  view_dir = −rd at `raytrace_kernels_taichi.py:1732`). `Sphere` declares
  `two_sided = False` (`shapes_3d.py:454`) ⇒ one_sided ⇒ **no flip**
  (`shading_taichi.py:464-468`). At the exit hit the normal therefore stays
  outward while the viewer is inside: `n·v < 0`, clamped away at
  `shading_taichi.py:1102` (`n_dot_v ≥ 1e-4`), diffuse `k_d·rgb·lc·n_dot_l`
  with `n_dot_l = max(n·l, 0)` positive on the sun-facing hemisphere
  (`:1101,:1109`), ambient added unconditionally (`:1094`). The function's own
  KNOWN LIMIT note (`shading_taichi.py:451-460`) describes precisely this case,
  including that such a point skips its shadow test: the shadow fan orients
  normals back along the incoming ray (`_orient_hit_normals`,
  `wavefront_kernels_taichi.py:2697`; `shading_taichi.py:487-512`), so at an
  exit point the horizon cull rejects every sample and visibility stays
  all-lit (`:2864-2875` writes nothing when `n_valid == 0`).
* **Transmissive exemption from one-sidedness — NONE.** The transmission fold
  touches only the closed-shell flag: `closed_shell_ceiling_flag(closed_shell,
  transmission)` produces `_rt_tri_closed` (`primitives.py:1027-1038`), which
  is consumed host-side by the sheet-compaction coverage ceiling and "never
  reaches a kernel" (`primitives.py:539-543`; merged at
  `scene_builder.py:1389-1398`; read only at `sheets.py:909-917` under
  `SOLID_SHELL_ALPHA`). `one_sided` is declared separately
  (`primitives.py:512-527`, default False at `:509-510`), packed independently
  (`primitives.py:380-387`, `:762-781`), and nothing anywhere keys it off
  transmission. The claim in the brief checks out: `declare_closed_shell`'s
  transmit-fold affects the alpha ceiling only.

## Q3 — Who gets a ray, and where does lost reflection energy go?

**Verdict: CONFIRMED for the exhaustion redistribution (it is real but does
not fire on this path); drops elsewhere are silent removals, not
redistributions.**

Fate of each of the four shares in the default route (monolith inline arm;
`_scatter_impl` mirrors it):

* **shaded (`contrib`)** — always deposited (`monolith:3049-3053`). When
  `bounces_left <= 0`, R is zeroed *after* `_material_reflectance` and the
  throttle ran (`:2968 → :2976 → :2977-2980`) but *before* the share
  arithmetic (`:3049`): `share = α(one3 − 0 − trans_share)` absorbs exactly
  the would-be reflected energy. So yes — following `if bounces_left <= 0:
  R = 0` (`:1234-1238` in `_scatter_impl`, `:2977-2980` inline,
  `sheet_resolve_taichi.py:557-558`) through
  `share = alpha * (one3 - R - trans_share)`: **that is a redistribution from
  the achromatic specular lobe into the albedo-tinted locally-shaded lobe.**
  Same ordering in all three copies.
* **reflected** — gets a ray iff `refl_max > MIN_ALPHA` (1e-3,
  `raytrace_kernels_taichi.py:130`) and bounces remain: either the ray slot
  continues as the reflection (`monolith:3121-3133`, spends a bounce at
  `:3131`) or, when the pass-through outweighs it, a pool slot
  (`:3144-3177`). A pool overflow retries the tile host-side rather than
  losing transport (`kernel docstring :2429-2431`;
  `_reserve_continuation_slot:884-900`). If culled by `MIN_WEIGHT` (`:3146`)
  the energy is **removed silently** — `weight *= cover_pass` at `:3178`, no
  re-attribution to contrib. In the no-split-pool arm the lighter of
  reflection/pass-through is likewise dropped, not folded
  (`:3259-3278`; `_scatter_impl:1346-1353`).
* **transmitted** — pool slot gated `wt_max > MIN_WEIGHT` (`:3074`); failure
  drops it silently. If bounces are exhausted, `is_glass` is disabled
  (`:3032` requires `bounces_left > 0`) and the orphaned share continues
  **unbent** in the pass-through (`:3272-3278`,
  `weight *= cover3 + trans_energy * tint` at `:3276`): conserved, but no
  longer refracted.
* **missed (`cover_pass`)** — always keeps the primary walk (`:3178`, `:3217`,
  `:3254`, `:3431`).

Bounce budget: `MAX_BOUNCES = 8` (`settings.py:29`); sheet primaries start at
8 (`sheet_resolve_taichi.py:258`, value delivered via `layer_offsets[6]`,
`tracer.py:2477`); every spawned continuation carries parent−1
(`sheet_resolve_taichi.py:649/662/…`, `monolith:3104/:3167`; reflection-primary
decrements at `monolith:3131`, `sheet:711/:931`). The audited path spends at
most four events (mirror 1, entry 2, exit 3, plus at most one internal chain
link) — **it cannot run out**. Exhaustion needs ~8 interface events on one
ray (e.g. a long internal-TIR chain near the limb), after which the
redistribution above dumps accumulated interior energy into the tinted
`shaded` lobe. Real mechanism, wrong scene for it: it is not what paints the
measured patch.

## Q4 — Energy conservation of the four shares

**Verdict: CONFIRMED — the sum-to-1 claim holds exactly at the split, for this
material, at every cosine tested, on both sides. Where conservation actually
fails is downstream of the split (silent culls/drops, Q3), not in the share
arithmetic.**

Method: hand transcription of `_material_reflectance`
(`wavefront_kernels_taichi.py:1116-1149`) + the share block
(`:3039-3059` / `:1277-1291` / `sheet:569-582`), with and without the in-situ
`R *= _mirror_share(rough)` throttle (`:2976`, `sheet:550/556`) that
`_scatter_impl` does not apply. No taichi involved.

Sum check, α=1, glass material, worst channel deviation from 1 over cosines
0.05–1.00: **0.0e+00 unscaled and 0.0e+00 scaled** — necessarily, because
`share` is *defined* as `1 − R − trans_share`; the identity is true by
construction wherever R appears consistently (including the throttled variant
and the post-exhaustion R:=0 variant, which redistribute rather than violate).

Representative grazing cosine cos = 0.15 — the angles a mirror sees through at
the sphere's limb (green channel shown; dielectric R is achromatic):

| quantity | entering (cos_n = −0.15) | leaving (cos_n = +0.15) |
| --- | --- | --- |
| side test (`:1134`) | outside → Schlick at incident angle | `sin²θ_t = 2.199 > 1` ⇒ **TIR** (`:1136-1141`) |
| R (unscaled) | 0.4660 | 1.0000 |
| R (× _mirror_share(0.05) = 0.98780) | 0.4603 | 0.9878 |
| diel_pass | 0.5340 | 0.0000 |
| trans_share = diel_pass·T | 0.5340 | 0.0000 |
| shaded share (unscaled / scaled) | 0.0000 / 0.0057 | 0.0000 / 0.0122 |

Full table (entering side, unscaled R; leaving side below critical cosine is
TIR throughout):

| cos | entering R | entering trans_share | leaving R | leaving trans_share |
| --- | --- | --- | --- | --- |
| 0.05 | 0.7828 | 0.2172 | 1.0 (TIR) | 0.0 |
| 0.10 | 0.6069 | 0.3931 | 1.0 (TIR) | 0.0 |
| 0.15 | 0.4660 | 0.5340 | 1.0 (TIR) | 0.0 |
| 0.20 | 0.3546 | 0.6454 | 1.0 (TIR) | 0.0 |
| 0.30 | 0.2013 | 0.7987 | 1.0 (TIR) | 0.0 |
| 0.40 | 0.1147 | 0.8853 | 1.0 (TIR) | 0.0 |
| 0.50 | 0.0700 | 0.9300 | 1.0 (TIR) | 0.0 |
| 0.60 | 0.0498 | 0.9502 | 1.0 (TIR) | 0.0 |
| 0.80 | 0.0403 | 0.9597 | 0.0948 | 0.9052 |
| 1.00 | 0.0400 | 0.9600 | 0.0400 | 0.9600 |

Which lobe *should* dominate: at the entry interface alone, cos 0.15 is nearly
even (reflection 0.466 vs transmission 0.534) and transmission even wins
slightly there. But the leaving-side column is the decisive one: every internal
ray steeper than the 41.8° critical angle (cos ≤ 0.745) is total-internally-
reflected, so near the limb through-light is physically contained — whatever
returns to the mirror from limb-grazing geometry must be the untinted entry
Fresnel lobe (which reaches 0.78 at cos 0.05 and →1 as cos→0). Any light that
does cross twice exits through the face and pays two tints. That is the
reference's reading (g/r 0.95, concentrated); Algan's numbers put the
doubly-tinted through-path on top instead.

## Q5 — Roughness fade on the deep bounce

**Verdict: PARTIAL — the docstring's premise holds for the function but not
for the code that actually runs this scene; direct and mirrored paths are
faded equally here, and the real asymmetry is between code arms.**

All call sites of `_mirror_share`:

1. `wavefront_kernels_taichi.py:2976` — the monolith's inline scatter arm,
   applied unconditionally at **every drained hit**: fallback primaries and
   deep bounces alike.
2. `sheet_resolve_taichi.py:550` — glossy==3 sheets that do not qualify for
   the split-sum substitution (gate at `:543-548` requires `T <= 1e-4`, so the
   glass never takes the substitution; default mode is 3 via
   `glossy_reflection_mode()` `settings.py:1755-1767`, passed at
   `raster_pipeline.py:2014`).
3. `sheet_resolve_taichi.py:556` — glossy==0.

`_scatter_impl` takes neither roughness nor the fade — its docstring says so
(`:1204-1210`) and its body confirms (no call between `:1232` and `:1353`).

So, precisely:

* **Direct camera → glass (primary hit, sheet route): FADED** —
  `_mirror_share(0.05)` applied at `sheet_resolve_taichi.py:550` (default
  glossy==3, else-branch).
* **Mirror → glass (deep bounce, monolith inline arm): ALSO FADED** —
  `:2976`. The docstring's "this deep bounce takes neither roughness nor the
  fade" is accurate about `_scatter_impl` itself, i.e. about scenes with a
  custom fragment scatter (`:3353-3358`), circuits (`:3364-3368`) and the
  legacy sorted pipeline (`tracer.py:3698-3706`) — not about the default
  built-in-material path, which fades every hit.

There is therefore **no fade asymmetry between the direct and mirrored paths
on this scene** (both ×0.98780). Quantified: `_mirror_share(0.05) = 0.15⁴ /
(0.15⁴ + 0.05⁴) = 0.987805` — the throttle keeps 98.78 % of the Fresnel lobe
in the traced mirror ray and re-attributes the remaining **1.22 %** to the
locally shaded, albedo-tinted term (that re-attribution is what creates the
non-zero `share` rows in Q4's table). The asymmetry that does exist is between
arms: a scene carrying any user scatter renders bounced glass unfaded where
this scene fades it.

## Q6 — Is a datum being read by something with no right to it?

**Verdict: (a) PARTIAL — no datum is reused, but the exit interface is shaded
under the outward-facing convention, which is the substance of the concern;
(b) REFUTED, with one caveat worth keeping.**

**(a)** There is no cross-hit reuse: every hit shades fresh through
`_shade_tri_hit` (`wavefront_kernels_taichi.py:2900`) — the entry hit's
`color` is not carried anywhere. But the *convention* is shared: because the
sphere is one-sided, the exit-interface shading normal stays outward
(Q2), the lights are evaluated against it (`shading_taichi.py:1097-1098`), and
the result is numerically what an outward-facing lit surface point at that
position would get — full diffuse on the lit hemisphere plus unconditional
ambient, shadow-test skipped (Q2 for the mechanism, KNOWN LIMIT
`shading_taichi.py:451-460`). So the inward-facing exit interface is lit under
outward-facing assumptions. Its contribution is suppressed by `share = 0`
(T=1) except the 1.22 % throttle leak — small today, load-bearing if a future
change ever gives `share` weight at T≈1.

**(b)** `_material_reflectance`'s side test infers "leaving" from the sign of
`rd · normal` (`:1122`, gated by transmission at `:1134`). Its inputs are only
(rd, normal, material params) — nothing about a ray records how it was
spawned, so a mirror-reflected ray is classified exactly as a camera ray would
be at the same point. **No spawn-history sensitivity exists.** The caveat:
the normal in question is the INTERPOLATED shading normal
(`monolith:2947-2964` feeding `:2968`; `sheet:510-513` feeding `:526`), while
the geometric face normal decides `entering` for the IOR stack and the
refraction origin offset (`monolith:3085`, `:3090-3091`). Near the silhouette
the interpolated normal can flip relative to the geometric one, misclassifying
entering/leaving and applying air-side Schlick/TIR on the wrong side — equally
for camera- and mirror-spawned rays, and most probable exactly at the limb
where the mirror's highlight lives. That is grazing-angle fragility of the
inference, not a history sensitivity, and it is reported as such rather than
as the defect's cause.

---

## What I did not verify

Claims below rest on source-reading or static reasoning, not execution:

* **No render was run.** Every statement about which kernel processes which
  rays rests on reading routing conditions (`tracer.py:2253-2262`,
  `:2389-2402`; `raster_pipeline.py:2014`) and template gates — not on a trace
  or a frame. The TASK file's ruling that sheet-vs-wavefront agree within
  1.5/255 was taken as given, not re-measured.
* **Template values were not observed at runtime.** That this batch compiles
  `glossy==3`, `refraction==1`, empty `frag_scatters`, specific `sec_aa`,
  `ior_stack`, `mem_trim`, `opaque_closest` values follows from defaults and
  gate conditions cited above; a batch-dependent rejection could change them.
* **The numeric replica is a transcription, not the compiled kernel.** Taichi
  semantics (f32 rounding, `ti.pow`/`ti.exp` behaviour) were assumed faithful
  to Python floats; the sum-to-1 result is exact by construction anyway, so
  the conclusion does not hinge on float details, but the tabulated R values
  are replica outputs, not renderer outputs.
* **The measured pixel data behind §1** (g/r 1.77, brightest-40 means, etc.)
  were taken from the TASK file, not re-measured.
* **That the dominant mirror-disc content is through-transmitted light** is an
  inference from the measured g/r ≈ albedo² combined with the code paths
  above; no per-lobe decomposition was performed (it would need a render).
* **Pool-overflow retry and truncation counters** (`_reserve_continuation_slot`
  retry semantics, `MAX_SURFACES_PER_RAY` ceilings) were read, never exercised.
* **Bounce accounting** is static analysis of spawn/decrement sites
  (`bounces_left - 1` at the cited lines); no ray's actual counter trajectory
  was traced, so edge cases (e.g. a long internal-TIR chain reaching
  exhaustion) are argued, not observed.
* **`_energy_scale(wsum)`'s effect on the shaded term** (illumination-budget
  normalisation) was not unfolded; it scales `shaded` identically on both
  interfaces, so it does not change any verdict, but its exact value on this
  scene was not computed.
* **The uncommitted kernel edits described under "Working-tree state"** were
  observed in `git diff`, not executed or validated; the claim that they break
  the sum-to-1 identity is read off the diff's arithmetic, not measured.
