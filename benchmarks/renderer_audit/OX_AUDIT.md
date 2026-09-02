# OX_AUDIT — Light-transport correctness of Algan's renderer vs. the Three.js/glTF material model

**Date:** 2026-08-21 · **Repo:** `algan` @ `ecd6947` · **Method:** code reading + algebra only. **No render was run** (this audit is read-only); every quantity below is computed by hand from the cited expressions.

**References used:** KHR_materials_transmission / _volume / _clearcoat / _sheen specs (fetched from KhronosGroup/glTF main), Three.js `lights_physical_pars_fragment.glsl.js` / `lights_fragment_begin.glsl.js` / `lights_physical_fragment.glsl.js` (fetched from mrdoob/three.js dev, r18x-era), and three-gpu-pathtracer from prior knowledge (its source was **not** fetched — see "Not checked").

---

## 0. Verdicts on the ten claims

| # | Claim | Verdict |
|---|-------|---------|
| a | No colour management anywhere | **Confirmed.** No sRGB decode or encode exists anywhere in the package. |
| b | `_stage_physical` transmission term double-counts | **Confirmed**, with arithmetic below. |
| c | Fresnel wrong on the way out of a dense medium / at TIR | **Confirmed**, with one accidental special case that is right. |
| d | No volumetric absorption; per-interface tint | **Confirmed.** `attenuation_color`/`attenuation_distance` are accepted and silently dropped. |
| e | Roughness/reflection fade | **Confirmed**; it is a stand-in for the blur, and it is a *documented deliberate* trade-off (DESIGN_analytic_aa.md §20.7). |
| f | `AMBIENT_STRENGTH = 0.1` unconditional | **Confirmed.** |
| g | Light units / `light_intensity` default π | **Half right.** Falloff units match Three.js exactly; but `SETTINGS.raytracing.light_intensity` (π) is **dead code** — nothing reads it. |
| h | Diffuse missing 1/π | **Confirmed.** Exactly π in linear terms; ~1.73× in display terms at head-on mid-grey (the two errors partially cancel). |
| i | Clearcoat/sheen vs glTF | **Sheen: confirmed ad-hoc** (and worse than both references). **Clearcoat: Algan matches Three.js WebGL** and deviates only from the stricter glTF layering spec — *not clearly worse than the real-time reference*. |
| j | Anything else | Four extra findings (J1–J4 below), incl. glass casting fully opaque shadows and `SAMPLES_PER_PIXEL > 1` silently switching lighting models. |

---

## 1. Ranked findings (most worth-doing first)

Rank = physical severity × cheapness of fix.

---

### 1. Diffuse BRDF is missing 1/π — every lit surface is π× too bright in linear terms (claims g+h)

**What Algan does.** All four lit fragment stages accumulate diffuse as `k_d * albedo * lc * n_dot_l` with no reciprocal-π:

- `shading_taichi.py:600` (`_stage_lambert`): `acc += in_rgb * lc * (n_dot_l * v)`
- `shading_taichi.py:633` (`_stage_phong`): `acc += (in_rgb * lc * n_dot_l + ...) * v`
- `shading_taichi.py:674` (`_stage_standard`): `diffuse = k_d * rgb * lc * n_dot_l`
- `shading_taichi.py:735` (`_stage_physical`): `direct = k_d * rgb * lc * n_dot_l + ...`

and the PyTorch vertex twins do the same (`material_shaders.py:161,192,231,294,306`). The light row `lc` carries `color * opacity * intensity` premultiplied (`render_loop.py:2258-2261`) and `_pack_lights` adds nothing further (`scene_builder.py:1825-1890`); `_light_eval` applies `lc /= d^decay` for decayed points (`shading_taichi.py:450-453`) and no scale for directionals (`shading_taichi.py:419-421`).

**What the reference does.** Three.js: `reflectedLight.directDiffuse += irradiance * BRDF_Lambert(material.diffuseContribution) * (1.0 - F)` with `BRDF_Lambert(c) = RECIPROCAL_PI * c` and `irradiance = dotNL * directLight.color`. Directional irradiance = `color * intensity`; point = `color * intensity / d^decay` (candela). Algan's falloff arithmetic is otherwise a faithful copy — its range fade `clamp(1-(d/range)^4)^2` (`shading_taichi.py:454-459`) is literally Three.js's `pow2(saturate(1-pow4(d/cutoff)))`, and decay matches up to an epsilon-floor difference near d→0 (Three.js floors the denominator at 0.01; Algan floors d at 1e-4, so a decay-2 light can get ~10⁶× brighter at contact).

**Which is right / by how much.** The reference. In linear units Algan delivers exactly **π×** more diffuse response for the same intensity number. Display-space, the missing gamma (finding 9) partially masks it: a #808080 albedo, white `DirectionalLight(intensity=1)`, head-on → Algan pixel **128/255**, Three.js **≈74/255** (1.73×). The *shape* damage is the serious part — display values as n·l falls:

| n·l | Algan (/255) | Three.js (/255) |
|-----|------|------|
| 1.00 | 128 | 74 |
| 0.50 | 64 | 52 |
| 0.25 | 32 | 35 |
| 0.125 | 16 | 23 |

Algan's terminator falls ~2× steeper: spheres lit by directional light have a visibly harsher, more "banded" limb than the reference even where absolute levels coincide.

**Dead-setting alert (claim g).** `LIGHT_INTENSITY = π` (`settings.py:83`, "radiance scale of explicit point lights in physical mode") and `AMBIENT_LIGHT = 0.0` (`settings.py:85`) are **never read by any render path**. They are consumed only by `path_trace_physical_stbvh` (`raytrace_kernels_taichi.py:3121,3327-3332`), which does NEE with `diff = albedo*(1-metallic)/π` and `radiance * light_intensity` — but `tracer.py` launches only `path_trace_scene_stbvh` (`tracer.py:1436`) and the wavefront; the physical kernel is referenced solely by `tests/unit_tests/test_raytracing_unit.py:255`. The vertex path hardcodes `light_intensity=1, ambient_light_intensity=1` (`primitives.py:620-621`). So `SETTINGS.raytracing.set(light_intensity=...)` silently does nothing. Every cross-renderer comparison image differs for the boring reason this finding describes.

**Fix.** Multiply the four stage diffuse terms (and the vertex shaders) by `RECIPROCAL_PI`, or equivalently divide packed `lc` by π once in `_pack_lights` — one-line-per-site, self-contained. Output moves everywhere; baselines regenerate. **Bucket: `local`.**

---

### 2. Unconditional `AMBIENT_STRENGTH = 0.1` on every lit stage (claim f)

**What Algan does.** `AMBIENT_STRENGTH = 0.1` (`shading_taichi.py:82`, mirrored `material_shaders.py:39`), added unconditionally by `_stage_lambert` (`:593`), `_stage_phong` (`:622`), `_stage_standard` (`:658`), `_stage_physical` (`:719`) as `albedo * 0.1 * env_map_intensity`, plus the same in every vertex shader. No light is required.

**Reference.** Three.js adds nothing without an AmbientLight/HemisphereLight/environment IBL: `irradiance = getAmbientLightIrradiance(ambientLightColor)` starts at zero and `ambientLightColor` is uniform-derived.

**Verdict.** It is a fudge standing in for missing indirect light (INDIRECT_BOUNCE_STRENGTH defaults 0, `settings.py:79`; there is no GI on the deterministic path). Consequences: (i) a fully shadowed region can never go darker than 10% of albedo — a 0.5 grey floors at ~13/255, flattening shadow contrast; (ii) in a scene that *does* have honest indirect light (an environment map's SH row — `_LT_ENV_SH`, `shading_taichi.py:435-447` — or MC indirect bounces), the fudge **double-counts** ambient by a flat 0.1·albedo; (iii) black objects glow.

**Fix.** Gate it off by default (or fold into an actual AmbientLight the user must add). Trivial edit in five stage functions + shaders. **Bucket: `local`.**

---

### 3. Transmission is double-counted: a fake per-light glow on top of the traced refracted ray (claim b)

**What Algan does.** Two independent transports carry the same light:

1. `_stage_physical` adds, *inside the per-light loop* (`shading_taichi.py:750`):
   `direct += rgb * lc * (transmission * (1.0 - metalness) * 0.5)`
   (identical in the vertex shader, `material_shaders.py:332`). Note: **not multiplied by n_dot_l**, and it scales with the *number of lights*.
2. The scatter splits the hit four ways and traces the transmitted share as a real refracted ray tinted by albedo: `trans_share = diel_pass * T` (`wavefront_kernels_taichi.py:993`), `trans_w = trans_energy * tint` (`:1033`; duplicated `sheet_resolve_taichi.py:472,502` and `wavefront_kernels_taichi.py:2564,2593`), while the surface's own shaded colour enters weighted by `alpha * (1 - R - trans_share)` (`:998-1000`).

**Worked example** — `MeshPhysicalMaterial(transmission=0.5, ior=1.5, metalness=0, roughness=0)`, white albedo `a`, one white directional light of irradiance E, surface head-on to light and camera:

- Fresnel R = 0.04; `diel_pass = 0.96`; `trans_share = 0.48`.
- Stage output ≈ `k_d·a·E + fudge = 0.48·a·E + 0.25·a·E`.
- Local contribution = `(1 − 0.04 − 0.48) × stage = 0.48 × 0.73·a·E = 0.35·a·E`.
- Refracted ray additionally carries throughput `0.48·a` sampling whatever is behind.
- Reflected ray carries 0.04.

glTF reference (KHR_materials_transmission BTDF): `base = mix(diffuse_brdf(baseColor), specular_btdf * baseColor, transmission)` inside `fresnel_mix(ior=1.5, …)` — i.e. diffuse 0.48·(a/π)·E, transmitted **once** as 0.48·a·(scene behind), reflection 0.04. "Optical transparency does not require any changes whatsoever to the specular term."

So the fudge injects `0.48 × 0.25·a·E = 0.12·a·E` of energy per light that has no counterpart in the model — the transmitted light appears twice: once as the refracted image, once as a glow on the front face that ignores occluders behind the glass and ignores n·l (a light grazing from *behind* the surface still contributes, because the shadow fan leaves `vis[li]=1.0` when the light-facing cull skips, `wavefront_kernels_taichi.py:2365-2434`, and the term has no n·l factor). With N lights it is N× that.

**Verdict.** Reference is right; this is a visible error (glass looks self-luminous in the light's colour rather than showing the refracted scene). The `(1-T)` scaling of `k_d` (`shading_taichi.py:734`) *is* correct per the spec's `mix()` — that part Algan gets right.

**Fix.** Delete the term at `shading_taichi.py:750` and `material_shaders.py:332`. If some brightness is wanted for the no-refraction fallback (ior ≤ 1, pool absent), gate it to exactly those cases where no refracted ray is spawned. **Bucket: `local`.**

---

### 4. Fresnel on the way out of a dense medium: wrong-angle Schlick, and TIR carries the transmitted weight (claim c)

**What Algan does.**
- `_material_reflectance` (`wavefront_kernels_taichi.py:853`): `cosi = clamp(abs(rd.dot(n)))`, then Schlick with `r0 = ((1-ior)/(1+ior))²` — side-blind, evaluated on the *incident* angle whichever side the ray is on.
- `_refract_ray` (`:703-704`) detects `sin2_t > 1` and bends the ray into the mirror direction — but the caller still weights that ray with the *transmitted* share: `trans_w = trans_energy * tint` (`:1033`, `sheet_resolve_taichi.py:502`, `wavefront_kernels_taichi.py:2593`).

**Reference.** KHR_materials_volume, Fresnel section (normative): three cases — entering the denser medium: Schlick on incident angle; exiting with no TIR: Schlick evaluated at the **transmitted-side (low-IOR) angle**; exiting with TIR: **F = 1**.

**Numbers.** Glass (ior 1.5, θc = 41.8°), internal ray at 40°: exact Fresnel reflectance ≈ **0.245**; Schlick at the air-side angle ≈ 0.2456 (near-exact — this is why the spec picks it); Algan's inside-angle Schlick ≈ **0.041**. At 45° (TIR): true R = 1.0; Algan ≈ 0.042.

Consequences by case (white glass, albedo tint `tint`, transmission T):

- **T = 1, untinted:** at TIR both continuations travel the *same* mirror direction, so the combined weight `R + diel_pass·T·tint ≈ 0.042 + 0.958 = 1.0` — **accidentally correct** in total, at the cost of two redundant rays. Below the critical angle, though, the split between the reflected and refracted rays is wrong (0.041/0.959 instead of 0.245/0.755 at 40°), so polished-glass edges misplace Fresnel energy.
- **T < 1:** at TIR the mirror direction receives only `R + diel_pass·T·tint`; the remainder `1 − R − trans_share` is shaded *locally as if it were diffuse response to the lights*. At T = 0.5, half the energy that should reflect internally leaks into front-surface shading.
- **Tinted glass:** the TIR-reflected ray is multiplied by albedo (`tint`), but Fresnel reflection is achromatic — a (1, .5, .5)-tinted glass reflects TIR light as (1.0, 0.52, 0.52).

**Verdict.** Reference is right; visible in anything with thick transmissive solids (bright rims, glowing glass bottoms, wrong edge brightness). The claim as stated is confirmed, with the T=1-white special case noted.

**Fix.** One combined func computing side-aware eta, the KHR three-case Fresnel, TIR ⇒ R=1/trans_share=0, and the refracted direction; consume it at the three duplicated split sites (`_scatter_impl`, the inline block in `wavefront_shade`, `sheet_resolve_taichi`). The pieces (rd, outward geo normal, ior) are all already at each site. **Bucket: `moderate`** (the logic is small but the split arithmetic is triplicated and pinned by a parity harness; the side test needs the geometric normal, not the shading normal).

---

### 5. No volumetric absorption; transmitted light is tinted once per interface (claim d)

**What Algan does.** `AdvancedPBRMaterial` accepts `thickness`, `attenuation_color`, `attenuation_distance` (`materials.py:348-350`) with the comment "Stored for API parity; not used" (`:374-377`), and they are absent from `get_shader_param_values()` (`:381-399`) — they never reach the GPU. Transmission tinting is `trans_w = trans_energy * tint` with `tint = albedo` applied **at every interface crossing** (`wavefront_kernels_taichi.py:1033`, `sheet_resolve_taichi.py:502`).

**Reference.** KHR_materials_volume: `T(x) = attenuationColor^(x/attenuationDistance)` (Beer-Lambert along the actual interior path length; "Ray-tracers should ignore the thickness texture and use the actual, ray-traced distance"). BaseColor tints at the boundary; absorption happens over distance — "the overall color depends on the distance the light traveled."

**How wrong.** Two opposite errors: (i) entry+exit double-tints, so a thin coloured glass object gets `albedo²` (too dark — e.g. a (1,.5,.5) pane passes (1,.25,.25)); (ii) there is no distance term, so a metre-thick block tints identically to a thin shell (far too bright for deep volumes). Shape-dependent colour — the signature look of real glass — is unreachable.

**Fix.** Minimal honest version: plumb `attenuation_color`/`attenuation_distance` (+`thickness`) through `_derive_material_surface_params`/`tri_extra`/`circuit_meta` and apply `tint^(x/d)` using the segment length between entry and exit hits (the wavefront already tracks `base_dist`; a single-medium "inside" flag on the ray state suffices for non-nested scenes). True nested-dielectric tracking is a ray-payload change. **Bucket: `moderate`** for the thickness/single-medium version; `redesign` for full volume tracking.

---

### 6. Rough reflections: the lobe is faded, not integrated, and glossy blur is off by default (claim e)

**What Algan does.**
- `GLOSSY_REFLECTION = env_flag("ALGAN_GLOSSY_REFLECTION", False)` (`settings.py:1380`) — the deterministic GGX-lobe spread (`raster_taichi.py:2274-2320`, a genuine inverted-CDF GGX sampler) is **off by default**.
- With it off, primary continuations keep only `_mirror_share(roughness) = α₀²/(α₀² + r⁴)` with α₀ = 0.15² (`wavefront_kernels_taichi.py:776-817`), applied before the energy split (`sheet_resolve_taichi.py:452-457`, `wavefront_kernels_taichi.py:2534`). Roughness 0.1 → 0.83, 0.15 → 0.5, 0.2 → 0.24, 0.35 → 0.033.
- Deeper bounces are neither faded nor blurred (documented: DESIGN_analytic_aa.md §20.7); transmission is never blurred (documented: §20.3 "Transmission is not blurred … Not built").

**Reference.** A path tracer integrates the lobe: every sample draws a GGX perturbed direction with Fresnel/geometry weighting (three-gpu-pathtracer), so a rough metal shows a *blurred, full-energy* image of its surroundings. Three.js raster approximates it with the env-map BRDF (also full-energy).

**Is the fade physically motivated?** No — and the code says so itself (`_mirror_share` docstring: "This is the stand-in for a properly sampled glossy lobe"). It is energy-*conserving* (the faded share returns to local shading, which carries the direct-light GGX highlight), so it is not a leak; but the redirected energy becomes *diffuse-looking*, not blurred specular. Net effect: `MeshStandardMaterial(metalness=1, roughness=0.35)` in Algan renders as a weak (3%) sharp mirror over a lit albedo — no environment response beyond the flat 0.1 ambient or an explicitly supplied env map — where Three.js shows a bright blurry reflection dominating the metal's appearance. Metals suffer most because their entire look is specular. Also note the default `roughness=1.0` (`materials.py:299`) fades even dielectric Fresnel to ~0.05%, so default-parameter glass has essentially no reflection.

**Fix.** Turning `GLOSSY_REFLECTION` on by default is a one-line change (visual output moves; the DESIGN doc records why it was defaulted off — residual speckle). Real parity needs stochastic lobe sampling on continuation rays. **Bucket: `local`** for the default flip; **`redesign`** for true integration (stochastic sampling where the route is deterministic).

---

### 7. Sheen is an ad-hoc rim, not the Charlie BRDF (claim i, sheen half)

**What Algan does.** `shading_taichi.py:744-746`:
```python
sheen_term = sheen * pow(clamp(1.0 - n_dot_v), 1.0 + 8.0 * sheen_roughness)
direct += sheen_color * lc * sheen_term
```
No `n_dot_l`, no distribution normalization, no base-layer attenuation. Same in `material_shaders.py:321-324`.

**Reference.** glTF/KHR_materials_sheen and Three.js: `sheenSpecularDirect += irradiance * BRDF_Sheen(...)` where `irradiance = dotNL * lightColor` and `BRDF_Sheen = sheenColor * D_Charlie(sheenRoughness, n·h) * V_Neubelt(n·v, n·l)`; additionally the base is attenuated: `irradiance *= (1 - max3(sheenColor) * sheenAlbedo)`.

**Verdict.** Both references are right; Algan is worse than both. Because the term lacks n·l, a sheen surface lit from behind its horizon still gains rim energy proportional to `lc`; and it double-counts with the base lobe (no albedo scaling). Magnitude: small at default `sheen=0`; wrong wherever sheen is used.

**Fix.** Replace the expression with D_Charlie + V_Neubelt (both short, closed-form, in the fetched chunks) and multiply by `n_dot_l`; optionally add the albedo-scaling factor. Self-contained in the two shader files. **Bucket: `local`.**

---

### 8. Shadows: off by default, and blocked by coverage only — glass casts opaque shadows (claim j)

**What Algan does.**
- `SHADOWS = False` by default (`settings.py:1857`) — no shadows at all unless opted in.
- When on, the shadow march attenuates purely by coverage: `alpha = …; transmitted *= 1.0 - alpha` (`raytrace_kernels_taichi.py:2586-2596`; the MC `_transmittance` likewise, `:3080-3084`). `transmission`/`ior` are never consulted. A transmissive solid (alpha = 1) blocks the shadow ray completely.
- Compounding it, the opaque any-hit fast paths (`shadow_flag` 2/3, `tracer.py:1182-1191`) classify "translucent" as alpha < 1, so glass is opaque-flagged and early-outs would remain wrong even if the march were fixed.

**Reference.** Physics: a transparent interface passes roughly `(1-F)·T` of the light (caustics aside); KHR transmission exists precisely because this share is optically major. Three.js's rasterizer is equally crude here (shadow maps), so Algan matches naive WebGL — but diverges from the physically-based reading and from what its own transmission parameter promises.

**Verdict.** Visible and stark: a wine glass or ice cube throws a hard black shadow. Defensible as a rasterizer-parity choice, not as light transport.

**Fix.** In `_shadow_march_occluded`/`_shadow_gather_occluded`, multiply `transmitted` by `diel_pass·T` per hit (both quantities already packed in `tri_extra`/`circuit_meta`), and redefine the opaque leaf flag to account for transmission so the any-hit modes stay sound. **Bucket: `moderate`.**

---

### 9. No colour management anywhere (claim a) — biggest single error, most expensive fix

**What Algan does.** The entire pipeline runs on **display-referred values treated as arithmetic quantities**:
- Authoring: hex colours divided by 255 with no decode (`constants/color.py:93-98`); images `float()/255` with no decode (`utils/file_utils.py:53`).
- Lighting: all stages operate directly on those values (findings 1–7).
- Output: accumulators ×255 to uint8 (`wavefront_kernels_taichi.py:1158`, `sheet_resolve_taichi.py:901`); background prefill `torch.add(0.5, rows, alpha=255)` (`scene_builder.py:1980`).
- The only curves are tonemappers: Khronos PBR Neutral / AgX applied to `value/255` and rescaled (`raytrace_kernels_taichi.py:1752-1772`, `post_processing/post_process.py:230-246`). AgX's internal log encode is a tone curve, not an interchange transform; the `*_srgb` symbols in `agx_tonemap` (`:1717-1719`) are a primaries matrix, despite the name. The "linear HDR" buffer (`POST_PROCESS_TONEMAP`, `settings.py:66-74`) is linear *arithmetic* (unclamped float), not linear *colour*.

**Reference.** Three.js since r152: textures and colours decoded sRGB→linear, lit in linear, encoded on output (sRGB or via tonemapping+encode).

**Quantified.** Mid-grey 0.5 lit at half intensity head-on: Algan outputs 0.25 → **64/255**; a managed pipeline outputs `srgb_encode(0.214·0.5/π)` ≈ **52/255**. Head-on levels partly cancel against finding 1 (see the table there), but the *shape* cannot be fixed by scaling: Algan's display-space falloff is ~x¹·⁰ where the reference is ~x^0.42 — terminator gradients, highlight rolloff and saturated-colour response are all qualitatively different (channel-wise display-space scaling darkens saturated colours far more than linear scaling then encoding does: halving pure red gives display (0.5,0,0) in Algan vs (0.735,0,0) in a managed pipeline).

**Verdict.** The reference is unambiguously right; this is the largest single physical error in the list. But per the ground rules: a partial fix is not meaningfully available — decoding albedo without encoding output (or vice versa) makes things *worse*, and doing both touches every stage, both shader systems, the tonemappers, the bloom threshold, and every committed pixel baseline. **Bucket: `redesign`** — and it is genuinely the only correct fix. Cheapest credible stepping stone: a documented "authored colours are display-referred" statement plus the 1/π fix (finding 1), which together pin down most of the level error.

---

### 10. Bounce exhaustion and misc transport corners (claim j)

- **max_bounces (default 8, `settings.py:28`).** On the last permitted bounce the reflection share is zeroed *into local shading* (`R = 0` at `wavefront_kernels_taichi.py:2535-2538`, `sheet_resolve_taichi.py:458-459`, `_scatter_impl:949-953`) and transmission continues **unbent** (index-matched) as pass-through (`:2766-2772`). Rays then escape and pick up background×weight (`:2923-2939`). Verdict: a graceful, mostly-invisible degradation — better than the path-tracer convention of contributing black — but deep glass (entry+exit per crossing; internal Fresnel bounces) turns mirror-like at depth 8 and stops bending. Defensible; document it. Fix (if wanted): raise the floor for transmissive-only continuations (they are cheap) — `local`.
- **`SAMPLES_PER_PIXEL > 1` switches the lighting model, not just the sampler.** The MC kernel launched in that mode (`path_trace_scene_stbvh`, `tracer.py:1436`) has **no explicit lights and no shadows**: it emits the *vertex-shaded* colour (baked with the very stages above, `primitives.py:866` → `_shade_vertex_colors`) as emission and scatters stochastically (`raytrace_kernels_taichi.py:2811-2832`). The correct NEE kernel with `1/π` diffuse and `light_intensity` exists (`path_trace_physical_stbvh`) but is unreachable. So raising SPP changes brightness, shadows, and transmission behaviour at once. Verdict: surprising and undocumented as a *lighting* change; the fix is wiring the physical kernel in — `moderate`.
- **Multi-light vertex path drifts.** `_shade_vertex_colors` feeds each light the *previous light's output* as albedo (`primitives.py:593-631`), so ambient and albedo compound per light. Mostly moot (fragment shading is default for core materials, `settings.py:182`), but any non-core/custom Python shader with two point lights shades wrongly. `local`.
- **GGX geometry/normalization details.** Algan uses the UE4 separable Schlick-GGX with `k=(r+1)²/8` (`shading_taichi.py:112-118`) where Three.js uses height-correlated Smith `V_GGX_SmithCorrelated`; Schlick F90 is implicitly 1 whereas KHR_materials_specular scales F90 by `specular_intensity` at grazing. Cosmetic (<few % away from grazing). Not worth churn. —
- **Clearcoat (claim i, coat half): Algan is *not* worse than the real-time reference.** glTF's `fresnel_coat` layers as `mix(material, cc_brdf, cc·F_cc)` — base and emission scaled by `1 − cc·F_cc` (fetched spec, normative). Algan adds the coat lobe unattenuated (`shading_taichi.py:737-742`). But Three.js's own WebGL `RE_Direct_Physical` also simply adds `ccIrradiance * BRDF_GGX_Clearcoat(...)` with no base attenuation (verified in the fetched chunk); only the spec text and (to my recollection, unverified — see below) three-gpu-pathtracer implement the conserving mix. If Algan wants to claim glTF conformance, add the `(1 - cc·F_cc)` scale to `k_d`, the base spec, and emissive — `local`.

---

## 11. Documented deliberate limitations (different finding class)

These are stated in-repo as trade-offs, and this audit treats them as design decisions, not bugs: `_mirror_share`'s primary-only scope and the never-blurred transmission (DESIGN_analytic_aa.md §20.3/§20.7); the "crude transmission" comment at `shading_taichi.py:747-749`; the shadow-path normal-orientation KNOWN LIMIT in `_sided_shading_normal` (`shading_taichi.py:239-248`); the index-matched fallback when the refraction pool is absent (`_scatter_impl` docstring). Nothing in the DESIGN docs documents the missing colour transform, the missing 1/π, or the dead `light_intensity` setting — those are not claimed as choices anywhere I could find.

## 12. What I did not manage to check

- **No renders were executed.** All magnitudes are hand-computed from the cited expressions; I did not verify them against actual frames (the repo's own A/B tooling in `benchmarks/renderer_audit/` predates this audit and was left untouched).
- **three-gpu-pathtracer internals** were not fetched; statements about it (probabilistic coat layering, lobe sampling) rest on prior knowledge plus the glTF specs. Where that mattered (clearcoat), I leaned on the fetched Three.js chunk and said so.
- **Texture-path colour** beyond `get_image`: whether any loader elsewhere decodes (I found none), and how texture alpha interacts with the opaque leaf flags, is only partially traced.
- **The legacy sorted wavefront** (`wf_shade_event`) is marked unsupported and was not audited.
- **CUDA/CPU numeric divergence**, Taichi `fast_math` effects on the criterion kernels, and the Monte Carlo normal-perturbation lobe's true shape (taken from comments) were out of scope.
- **Env-map radiance units** (whether `_sample_env_map` values are commensurate with light units) were not pinned down.

---

## Appendix: exact expressions cited

```
shading_taichi.py:600   acc += in_rgb * lc * (n_dot_l * v)
shading_taichi.py:674   diffuse = k_d * rgb * lc * n_dot_l
shading_taichi.py:735   direct = k_d * rgb * lc * n_dot_l + spec * lc * (n_dot_l * spec_w)
shading_taichi.py:750   direct += rgb * lc * (transmission * (1.0 - metalness) * 0.5)
shading_taichi.py:744   sheen_term = sheen * pow(clamp(1.0 - n_dot_v), 1.0 + 8.0*sheen_roughness)
wavefront_kernels_taichi.py:853   cosi = ti.math.clamp(ti.abs(rd.dot(n)), 0.0, 1.0)
wavefront_kernels_taichi.py:863   diel_pass = (1.0 - m) * (1.0 - r_diel)
wavefront_kernels_taichi.py:993   trans_share = diel_pass * T
wavefront_kernels_taichi.py:998   share = alpha * (one3 - R - trans_share)
wavefront_kernels_taichi.py:1033  trans_w = trans_energy * tint
wavefront_kernels_taichi.py:817   return _MIRROR_SHARE_A2 / (_MIRROR_SHARE_A2 + a * a)
wavefront_kernels_taichi.py:703   if sin2_t > 1.0: out = rd - 2.0 * rd.dot(n) * n
raytrace_kernels_taichi.py:2594   transmitted *= 1.0 - alpha        # shadow march
raytrace_kernels_taichi.py:3328   diff = albedo * ((1.0 - metallic) / 3.14159…)   # dead kernel
primitives.py:620                 1,      # light_intensity, hardcoded
primitives.py:621                 1,      # ambient_light_intensity, hardcoded
```
