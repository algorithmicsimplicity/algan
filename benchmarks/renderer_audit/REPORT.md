# Algan vs Three.js — a rendering audit

Two renderers, one scene description, the same frame. Where they disagree, this
asks which one is closer to the physics, and fixes the cases where Algan is not
and the fix is contained.

Everything here was measured on this machine (a CPU-only cloud session,
Ubuntu 24.04, 4 vCPU, no GPU — so Algan ran its CPU path throughout). Nothing is
quoted from memory or from a design document without saying so.

---

## 1. How the comparison was set up

`SPEC.md` defines a small JSON scene format — camera, lights, spheres and boxes,
`MeshPhysicalMaterial` parameters — chosen so that both engines can express every
field *exactly*. Two back ends consume it:

| back end | what it is | file |
| --- | --- | --- |
| `algan` | Algan's own renderer via `Scene.save_frame` | `algan_render.py` |
| `three_raster` | `THREE.WebGLRenderer` r185, out of the box | `three_render.mjs` |
| `three_pathtrace` | `three-gpu-pathtracer` 0.0.24 on the same scene graph | `three_render.mjs` |

The path tracer matters: Three.js's rasterizer cannot reflect a scene without an
environment map and cannot refract at all, so on exactly the questions this audit
is about it is not a reference. `three-gpu-pathtracer` path-traces the *same*
`MeshPhysicalMaterial` objects, so it is the physically-based reading of the same
material description. Where the two Three.js modes disagree, the path tracer is
the reference and the rasterizer is reported as context.

Three.js ran with its defaults and no fudging toward Algan:
`ColorManagement.enabled = true`, `outputColorSpace = SRGBColorSpace`,
`toneMapping = NoToneMapping`, shadow maps on in the raster pass. Scene colours
are fed through `Color.setRGB(r, g, b, SRGBColorSpace)`, so an authored `0.8`
means the sRGB value `0.8` on both sides.

Algan ran with `SETTINGS.raytracing.set(shadows=True, tonemapping=False)`. Shadows
because Three.js has them on and Algan defaults them off; tonemapping off because
Algan defaults to Khronos PBR Neutral and Three.js defaults to none — with both
off, the only transfer function in play is the one the audit is measuring. (Algan's
tonemapper was checked against the Khronos curve and is a faithful implementation
of it: authored 0.2 → 41/255, which is exactly PBR Neutral's 0.04 black-point
offset. That is parity, not a discrepancy.)

### Two things had to be reconciled before any pixel comparison meant anything

**Algan's +z axis points into the screen.** `OUT`, the direction out of the screen
toward the viewer, is `(0, 0, -1)`, and a new Scene's camera sits at `z = -7`.
Three.js and glTF have the opposite sign. Porting a scene means negating every z
coordinate, every z direction, and every rotation about y. The docstring in
`algan/constants/spatial.py` said the opposite ("``OUT``/``IN`` along +z/-z") and
has been corrected as part of this work, along with the same claim in
`shapes_3d.py`.

**Algan's default Scene comes with a light.** `default_scene_initializer` spawns a
white `PointLight` beside the camera. Every scene here replaces the initializer
with one that spawns only the camera, so both engines see exactly the lights the
spec names.

With those settled, `scenes/calib_orient.json` — five unlit boxes, one per signed
axis — renders **identically** in the two engines: the same silhouettes to the
pixel, the same depth order, mean absolute difference 1.3/255 confined to
antialiased edges. Camera, field of view, projection and handedness agree exactly,
so everything that follows is shading and transport, not geometry.

![orientation calibration](out/calib_orient.compare.jpg)

---

## 2. The two global conventions that differ (and why only one is a defect)

`scenes/calib_diffuse.json` is one 0.8-grey roughness-1 metalness-0 sphere lit
head-on by a single white directional light of intensity 1, on black. The centre
pixel is a direct readout of each engine's diffuse convention:

| | centre pixel | as linear radiance |
| --- | --- | --- |
| Algan (tonemapping off) | **217** | 0.673 |
| Algan (default tonemapping) | 205 | 0.610 |
| Three.js raster | **122** | 0.195 |
| Three.js path tracer | 105 | 0.141 |

Three.js's number is exactly what theory says: `srgb_to_linear(0.8) / π = 0.192`,
encoded back to sRGB as 121. Algan's is `0.8 × (0.1 ambient + 0.96 k_d) = 0.85`,
written straight out as 217.

Two independent differences produce that gap.

### 2.1 The missing 1/π is a unit convention, not an error

Algan's diffuse term is `k_d · albedo · light_colour · n·l`, with no `1/π`.
Three.js uses `albedo/π`. In linear terms Algan is exactly π× brighter for the
same intensity number — a ratio of 3.17 measured against a predicted 3.14.

It is tempting to call that a bug, and Ox Alpha's independent audit
(`OX_AUDIT.md`, finding 1) ranks it first. **I disagree, and this is worth being
precise about.** A constant factor between two renderers is a choice of light
unit, not a physical claim: Algan's `DirectionalLight(intensity=1)` simply means
"irradiance π", i.e. "a white surface facing this light comes out white". Nothing
observable distinguishes that from Three.js's convention except the number the
user types. The renderer's own settings agree — `SETTINGS.raytracing.light_intensity`
defaults to π for exactly this reason.

Dividing the diffuse term by π *on its own* would make every existing Algan scene
3.14× darker and no more physically accurate. It is only worth doing as part of
2.2, which is a real defect, and where the two changes partly cancel.

### 2.2 Nothing in Algan is colour-managed — this one is a real defect

Measured directly (`transfer_probe.py` renders a flat slab at a series of
authored greys, with and without lights):

```
authored               0.05  0.10  0.20  0.30  0.40  0.50  0.60  0.70  0.80  0.90  1.00
as 8-bit                 13    26    51    76   102   128   153   178   204   230   255
unlit, no tonemap        13    26    51    77   102   128   153   179   204   230   255   <- identity
one directional light    14    28    55    82   109   136   163   190   217   244   255   <- x1.063
```

The unlit row is the identity map: an authored colour is written to the file
unchanged. Combined with the lit row, that pins the pipeline exactly — Algan
treats an authored sRGB value as a *linear reflectance*, does all its lighting
arithmetic on it, and writes the result out as if it were already sRGB-encoded.
There is no decode anywhere and no encode anywhere. (Ox Alpha traced the same
conclusion through the source independently and found the same: hex colours
divided by 255 with no decode, accumulators multiplied by 255 to uint8, and the
tonemappers as the only curves in the pipeline.)

The visible consequence is not brightness — that is 2.1 — it is **shape**. On
screen, a Lambertian falloff should be proportional to `n·l`. Algan's 8-bit
output is proportional to `n·l`, so the light it actually emits is proportional
to `n·l^2.2`: terminators fall away about twice as fast as they should, mid-shadow
detail is crushed, and saturated colours darken far more than they should when
dimmed (halving pure red gives a displayed (0.5, 0, 0) where a managed pipeline
gives (0.735, 0, 0)). Blending — antialiasing, transparency, reflections summing
with local shading — all happen in the wrong space too.

**Not fixed here, deliberately.** A partial fix is worse than none: decoding
albedo without encoding output, or vice versa, moves everything the wrong way.
Doing both is a contained change in principle — decode at pack time on the host,
encode once in `finalize_pixel_color`, and the tonemapper would then be operating
on linear values where it belongs — but it moves every pixel of every render,
invalidates both committed baseline sets, and needs the post-processing chain
(bloom threshold, FXAA, the HDR buffer) revisited with it. It is the right thing
to do and it deserves its own change, with its own baselines and its own
before/after. Worth noting that it composes cleanly with 2.1: with albedo decoded
and output encoded, dividing the diffuse term by π and multiplying light
intensities by π leaves the *authored* numbers meaning what they mean today.

---

## 3. Where Algan is already right

Worth stating plainly, because the rest of this document is a list of problems.

**Refraction is genuinely traced, and it is correct.** `scenes/calib_glass.json`
puts a clear ior-1.5 sphere in front of four coloured blocks. Algan produces the
inverted, magnified image a real glass ball produces, with the four-pointed dark
star where total internal reflection takes over at the rim — and it matches the
path tracer's structure closely enough to overlay.

![glass calibration](out/calib_glass.compare.jpg)

Three.js's **rasterizer cannot do this at all**: its `transmission` samples the
framebuffer behind the object, so the blocks appear the right way up, unrefracted.
On this axis Algan out-performs stock Three.js by a wide margin and matches the
path tracer.

**Mirror reflection is accurate.** On `scenes/calib_mirror.json`, the fraction of
the floor's brightness that survives reflection in a smooth metalness-1 sphere is
0.905 in Algan against 0.956 in the path tracer (the material's albedo is 0.95).
Three.js's rasterizer scores 0.0 — with no environment map a mirror has nothing
to reflect.

**Clearcoat matches the real-time reference.** Algan adds the coat lobe without
attenuating the base layer, which the glTF spec says to do — but Three.js's own
WebGL renderer does not attenuate it either. Algan is not worse than the reference
it is modelled on.

**Bounce exhaustion degrades gracefully.** At `max_bounces`, the reflection share
falls back into local shading rather than contributing black.

---

## 4. Discrepancies, ranked, with what was done about each

### 4.1 A transmissive solid casts a fully opaque shadow — FIXED

`scenes/calib_shadow.json` puts a clear glass sphere and an opaque sphere of the
same size side by side over a floor, lit straight down, so each shadow sits
directly beneath its sphere and the two are trivial to compare.

| | floor | glass sphere's shadow | opaque sphere's shadow |
| --- | --- | --- | --- |
| Algan (before) | 0.694 | 0.0070 — **1.0% of the floor** | 0.0070 — 1.0% |
| Three.js path tracer | 0.154 | 0.157 — **101.9% of the floor** | 0.007 — 4.6% |
| Three.js raster | 0.195 | 0.000 — 0% | 0.000 — 0% |

(Linear radiance; the glass patch exceeds 100% in the path tracer because the
sphere is a lens and concentrates light.)

Algan's glass blocked light **exactly as completely as an opaque sphere of the
same size**. The shadow march attenuated by coverage alone —
`transmitted *= 1.0 - alpha` — and never consulted `transmission` or `ior`, so a
window, a wine glass and a brick were the same object to a shadow ray. This was
the single largest structural error found.

![shadow calibration](out/calib_shadow.compare.jpg)

### 4.2 Fresnel was evaluated on the wrong side of a glass surface — FIXED

`_material_reflectance` applied Schlick's approximation to `abs(rd · n)`
regardless of which side of the interface the ray was on. Schlick is written for
a ray arriving from the thin side; a ray already inside glass reflects far more
than the same incident angle suggests, and past the critical angle it reflects
everything. Measured against the exact Fresnel equations at ior 1.5:

| angle inside the glass | before | after | exact |
| --- | --- | --- | --- |
| 20° | 0.040 | 0.040 | 0.042 |
| 35° | 0.040 | 0.067 | 0.086 |
| 40° | 0.041 | **0.246** | **0.245** |
| 41.8° (critical) | 0.041 | 0.908 | 0.891 |
| 45° | 0.042 | **1.000** | **1.000** |
| 80° | 0.410 | 1.000 | 1.000 |

So light leaving a solid was split six-to-one the wrong way just below the
critical angle, and beyond it, energy that should have been perfectly reflected
was passed through the surface instead. Worse, `_refract_ray` *did* detect total
internal reflection and bent the ray into the mirror direction — but the ray
carried the **transmitted** weight, which is tinted by the glass colour, so a
perfectly achromatic Fresnel reflection came out coloured.

The fix follows KHR_materials_volume's three normative cases: entering, Schlick
on the incident angle; leaving without TIR, Schlick on the air-side angle; leaving
past the critical angle, `F = 1` with the transmitted branch given zero weight.

**Scoped so nothing else moves**: the side test is gated on `transmission > 0`.
The renderer does not track which medium a ray is in, so "the far side of the
surface" is inferred from the normal — sound for a closed transmissive solid,
wrong for a back-facing hit on an ordinary opaque surface, which is not inside
anything. Verified: `calib_mirror.algan.png` is **byte-identical** across the
change (same md5), while `calib_glass` moves, and moves only in a ring at the
sphere's rim and along the arms of the TIR star — exactly where the exit-side
Fresnel acts.

`tests/unit_tests/test_dielectric_fresnel.py` pins all of this against the exact
Fresnel equations, including that an opaque material stays side-blind.

### 4.3 Transmission was counted twice — FIXED

`_stage_physical` added `rgb * lc * (transmission * (1 - metalness) * 0.5)` inside
its per-light loop — a glow proportional to transmission, with no `n·l`, scaling
with the number of lights — on top of the real refracted ray that the scatter
already traces. Glass lit from behind its own horizon still gained it.

### 4.4 Sheen was an ad-hoc rim, not a BRDF — FIXED

`sheen * (1 - n·v)^k` with no `n·l`, no distribution and no normalisation, so a
sheen surface gained rim energy from lights that do not reach it. Replaced with
the Charlie distribution and Neubelt visibility term that glTF and Three.js both
specify.

### 4.5 A rough metal reflects almost nothing — NOT FIXED (deliberate, documented)

Measured on `calib_mirror`, the fraction of the floor's brightness reflected by a
metalness-1 roughness-0.35 sphere:

| | reflection efficiency | reflected-image contrast |
| --- | --- | --- |
| Algan, default settings | **0.025** | 3.68 |
| Algan, `glossy_reflection` on | **0.620** | 1.04 |
| Three.js path tracer | **0.575** | 2.10 |
| Three.js raster (no env map) | 0.042 | 3.07 |

By default a rough metal in Algan reflects **2.5%** of what it should — a factor
of 23. With `SETTINGS.raytracing.experimental.set(glossy_reflection=True)` it
lands within 8% of the path tracer. This is the largest *visible* difference in
the showcase scene: Algan's gold sphere is a dark ball with a specular dot where
the path tracer's is a bright, blurred mirror of its surroundings.

**Not flipped, and the reason is good.** `settings.py` documents it: four
secondary taps cannot integrate a wide GGX lobe, and both ways of spending them
cost something visible in *motion* — the interleaved variant resolves into an
ordered dither that crawls as geometry moves, and the plain variant lands as four
discrete ghost copies. For an animation engine that is a worse artefact than a
dim reflection. The escape hatch is one setting and is right for stills.

What is worth changing is discoverability rather than the default: this is a
material-appearance control living among ~55 experimental performance switches.

### 4.6 No volumetric absorption — NOT FIXED

`attenuation_color`, `attenuation_distance` and `thickness` are accepted by
`MeshPhysicalMaterial`, documented, and then **silently dropped** — they never
reach the GPU. Instead the transmitted ray is tinted by the surface albedo at
*every* interface crossing, so a coloured pane is tinted twice (entry and exit)
and a metre-thick block is tinted exactly the same as a thin shell. Depth-dependent
colour — the most recognisable property of real coloured glass — is unreachable.

This is a genuine missing feature rather than a wrong formula, and it needs new
material slots plumbed through `scene_builder` into three kernel sites. The
tractable version is sketched in §6.

### 4.7 Two settings that silently do nothing — FIXED

`SETTINGS.raytracing.light_intensity` (default π) and
`SETTINGS.raytracing.ambient_light` are read only by `path_trace_physical_stbvh`,
a kernel `tracer.py` never launches — it is referenced only from a test. Setting
either had no effect on any render, silently.

### 4.8 Algan adds 10% ambient to every lit surface — NOT FIXED (documented choice)

Every lit stage adds `albedo × 0.1` with no light in the scene. Three.js adds
nothing without an ambient or environment light. So a fully shadowed region in
Algan can never go below 10% of albedo, and a black object glows slightly. It
stands in for the indirect light the deterministic path does not compute
(`indirect_bounce_strength` defaults to 0), and in a scene that *does* have honest
indirect light it double-counts.

This is an artistic default rather than a physics claim, and changing it would
darken every existing Algan scene. Left alone; recorded here so it is not
mistaken for a bug.

---

## 5. The main scene

`scenes/showcase.json` — mirror ball, clear glass sphere, tinted glass sphere,
rough gold sphere, clearcoat sphere, glass cube, sheen sphere, emitter, floor and
back wall, under a directional key and a blue point light.

![showcase](out/showcase.compare.jpg)

---

## 6. What a follow-up should do, in order

1. **Colour management** (§2.2), together with the `1/π` (§2.1). The largest
   remaining physical error, and the only one whose fix is a whole-engine change.
2. **Volumetric absorption** (§4.6). The tractable version does not need medium
   tracking: a transmitted ray is spawned *at* the entry surface, so when a hit is
   on the inside of a transmissive surface (`rd · n > 0`) that hit's own `t` is
   the interior path length. Apply `attenuation_color^(t/attenuation_distance)`
   there. Exact for a single convex solid, approximate for nested ones.
3. **Promote `glossy_reflection`** out of `experimental` (§4.5) so the trade-off
   is a choice users can find, and consider defaulting it on for `save_frame`,
   where the motion artefact that justifies the default cannot occur.
4. **Sheen albedo scaling and clearcoat base attenuation** (§4.4, §3) if glTF
   conformance rather than Three.js parity is the goal.

## 7. Files

| file | what it is |
| --- | --- |
| `SPEC.md` | the shared scene format |
| `scenes/*.json` | the scenes; `calib_*` isolate one question each |
| `algan_render.py` | Algan back end |
| `three_render.mjs` | Three.js back end, raster and path-traced |
| `compare.py` | side-by-side contact sheets, raw and exposure-matched |
| `metrics.py` | unit-free transport ratios (transmission and reflection efficiency) |
| `transfer_probe.py` | Algan's authored-colour → pixel transfer curve |
| `OX_AUDIT.md` | Ox Alpha's independent source-level audit |
| `../../OX_FIXES.md`, `../../OX_BEER.md` | Ox Alpha's implementation reports |

Reproduce:

```
<venv-python> benchmarks/renderer_audit/algan_render.py \
    benchmarks/renderer_audit/scenes/showcase.json --out out --no-tonemap
node benchmarks/renderer_audit/three_render.mjs \
    benchmarks/renderer_audit/scenes/showcase.json --out out --mode both --samples 64
<venv-python> benchmarks/renderer_audit/compare.py \
    out/showcase.algan.png out/showcase.three_pathtrace.png --out out
```

The Three.js back end needs `npm install three three-gpu-pathtracer playwright`
in a scratch directory; `three_render.mjs` documents how it finds it.
