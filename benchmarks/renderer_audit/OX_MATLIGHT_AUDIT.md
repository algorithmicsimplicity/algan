# Algan material shaders vs three.js r185 — source-level audit

Read-only audit. Every Algan claim cites `algan/...:line`; every three.js claim
cites the installed r185 source under
`.../scratchpad/three/node_modules/three/src/` (paths below abbreviated as
`three/src/...`). Nothing was rendered; nothing was modified.

**Why three.js is the spec here:** `materials.py:1-7` says Algan's material
classes "mirror the Three.js *mesh* materials -- the same material types,
property names and default settings", and `material_shaders.py:20-21` calls
the shaders "approximations of the Three.js GLSL materials". Divergence is
therefore measured against r185, not against taste.

## Where each side shades

| Algan site | Materials | Runs |
| --- | --- | --- |
| `raytracing/shading_taichi.py`: `_stage_unlit` :782, `_stage_default` :792, `_stage_lambert` :867, `_stage_phong` :902, `_stage_standard` :942, `_stage_physical` :989 | basic / lambert / phong / standard / physical | per fragment in the render kernel (`FRAGMENT_SHADING = True`, `raytracing/settings.py:210`) |
| `shaders/material_shaders.py`: `lambert_shader` :263, `phong_shader` :294, `standard_shader` :331, `physical_shader` :381, `toon_shader` :486, `normal_shader` :524, `matcap_shader` :547, `depth_shader` :572 | all eight | baked per vertex in torch before upload; only materials with no in-kernel port (toon/normal/matcap/depth + custom shaders) actually bake (`raytracing/primitives.py:649-698`, `materials.py:35-40`) |

The kernel stages and torch shaders are stated to be twins
(`shading_taichi.py:52-56`); where they differ from each other this is noted.

---

# Findings, ranked by visibility in a rendered frame

## F1. Every diffuse term is missing three's `1/pi` — lit surfaces render pi x hotter than the spec

**Algan.** Diffuse is `rgb * lightColor * n_dot_l`, unnormalized:
torch `material_shaders.py:287`; kernel `_stage_lambert`
`shading_taichi.py:892-893`. Same shape in phong (`material_shaders.py:321`,
`shading_taichi.py:932`), standard (`material_shaders.py:369`,
`shading_taichi.py:979`), physical (`shading_taichi.py:1061`) and toon
(`material_shaders.py:517`).

**three.js r185.** `RE_Direct_Lambert` is
`irradiance * BRDF_Lambert(diffuseColor)` with
`irradiance = dotNL * directLight.color`
(`ShaderChunk/lights_lambert_pars_fragment.glsl.js:13-16`) and
`BRDF_Lambert(c) = RECIPROCAL_PI * c` (`ShaderChunk/common.glsl.js:103-107`;
`RECIPROCAL_PI` at `common.glsl.js:5`). Identical structure in the phong and
toon direct terms (`lights_phong_pars_fragment.glsl.js:15-18`,
`lights_toon_pars_fragment.glsl.js:12-14`). (Q6 answered.)

**Is the factor the same pi everywhere?** Yes — its *absence* is uniform.
All five lit materials omit exactly one `RECIPROCAL_PI` on the diffuse lobe,
while specular lobes that carry their own pi internally (GGX distribution,
`shading_taichi.py:216-222` = torch `material_shaders.py:173-178`) keep it.
Net effect: the diffuse-to-specular balance differs from three.js by exactly
pi on every material.

**External invariant (energy must not exceed incident irradiance).**
three.js satisfies it by construction (`BRDF_Lambert` is normalized).
Algan's form integrates to `pi * albedo` over the hemisphere: any albedo above
`1/pi ~= 0.318` reflects more than it receives. In display-referred mode the
illumination budget (`_energy_scale`, `shading_taichi.py:140-174`, torch twin
`material_shaders.py:86-106`) partially masks this by renormalising over-lit
surfaces; under the default linear colour space it returns exactly 1.0 and
nothing masks it.

**What you see:** a white wall hit head-on by an intensity-1 white light goes
to pure white in Algan; in three.js the same surface renders at `1/pi ~= 0.318`
linear (~ sRGB byte 153, mid grey). Everything diffuse reads roughly one stop
hotter; metalness blends shift hue because diffuse outweighs specular by an
extra pi. *(Byte estimate is analytic through the sRGB OETF, not a rendered
pixel.)*

**Defect or convention?** Defect relative to the stated spec. The codebase
acknowledges the missing factor once — `raytracing/settings.py:109-111` sets
`LIGHT_INTENSITY = pi` "so a white light produces roughly albedo-level
Lambertian brightness" — but that setting is inert: it feeds only the never-
wired `path_trace_physical_stbvh` kernel, and `settings/raytracing_settings.py:149-164`
states no launchable renderer reads it. Live paths pass `light_intensity == 1`
(`primitives.py:687-688`); packed rows already carry intensity
(`render_loop.py:2427-2430`). Nothing cancels the pi anywhere reachable.

---

## F2. MeshDepthMaterial: linear Euclidean ramp vs hyperbolic NDC depth, driven by different near/far

**Algan** (`material_shaders.py:572-595`):

    d   = ||vertex_location - camera_location||_2     (Euclidean ray length)
    out = 1 - clamp((d - near)/(far - near), 0, 1)    replicated to RGB

with `near=0.1, far=100` authored **on the material** (`materials.py:585`),
unrelated to the Scene camera unless copied by hand.

**three.js r185.** `depthPacking` defaults to `BasicDepthPacking` (=3200;
`materials/MeshDepthMaterial.js:43`, `constants.js:1250`). The fragment shader
computes `fragCoordZ = 0.5 * vHighPrecisionZW[0] / vHighPrecisionZW[1] + 0.5`
(half NDC z plus one half) and writes
`gl_FragColor = vec4( vec3( 1.0 - fragCoordZ ), opacity )`
(`ShaderLib/depth.glsl.js:91,97`; docstring "White is nearest, black is
farthest" at `MeshDepthMaterial.js:6`).

Analytic forms (`n`=near, `f`=far, `d` = positive view-axis distance):

* three.js: `C(d) = n(f-d) / ((f-n)d)` — hyperbolic; `C(n)=1`, `C(f)=0`;
  near/far come from **whichever camera renders the pass** (its projection
  matrix produces `gl_Position.zw`).
* Algan: `C(d) = 1 - clamp((d-n)/(f-n), 0, 1)` — linear in Euclidean
  point-to-camera distance; near/far from **material defaults**.

three.js's RGBA packing variants (`depth.glsl.js:102-112`,
`packing.glsl.js:21-54`) have no Algan counterpart; Algan is grayscale only.

**What you see:** the ramps barely resemble each other away from the
endpoints. Example (n=0.1, f=100, object 10 units out): three gives
`0.1*90/(99.9*10) ~= 0.009` -> sRGB byte ~24 (near black); Algan gives
`(100-10)/99.9 ~= 0.90` -> byte ~243 (near white). Also three's value is
constant per axis-depth along a ray, while Algan's Euclidean distance grows
toward frame corners on planar walls, bending the ramp across the image.

**Defect or convention?** Declared approximation ("Approximates Three.js
depth packing with a simple linear luminance ramp",
`material_shaders.py:587-588`; listed in `materials.py:31-33`). Convention —
but visibly off-spec, and the near/far source mismatch is the surprising half:
`Camera` has real `near`/`far` clip fields (`rendering/camera.py:84-86`) that
the shader never consults.

---

## F3. MeshNormalMaterial packs world-space normals; three packs view-space

**Algan** (`material_shaders.py:524-544`): `out = n*0.5 + 0.5` with `n` the
world-space shading normal (`_shading_normal`, `material_shaders.py:142-165`).
The packing arithmetic matches `packNormalToRGB =
normalize(normal)*0.5+0.5` (`ShaderChunk/packing.glsl.js:2-4`) and
`gl_FragColor = vec4( normalize( normal ) * 0.5 + 0.5, ... )`
(`ShaderLib/meshnormal.glsl.js:77`) — but three's `normal` there is **view
space**: `vNormal = normalize(transformedNormal)`
(`ShaderChunk/normal_vertex.glsl.js:4`) where
`transformedNormal = normalMatrix * objectNormal`
(`ShaderChunk/defaultnormal_vertex.glsl.js:44`); FLAT_SHADED uses screen-space
derivatives of the view position (`normal_fragment_begin.glsl.js:4-8`).

**Concrete RGBs, before any transfer function:**

| Surface point | three.js (view space) | Algan (world space) |
| --- | --- | --- |
| sphere point facing the camera dead-on | `(0,0,+1)` -> **(0.5, 0.5, 1.0)** | `(0,0,-1)` -> **(0.5, 0.5, 0.0)** |
| normal pointing screen-right | view `+x` -> **(1.0, 0.5, 0.5)** | world `RIGHT=(1,0,0)` -> **(1.0, 0.5, 0.5)** |

Algan's dead-on value follows from the default camera at
`CAMERA_ORIGIN = ORIGIN + OUT*7 = (0,0,-7)` looking toward the origin
(`constants/spatial.py:28,33`): the camera-facing normal is world `-z`. The
screen-right case coincides only while the camera keeps identity orientation.

**External invariant ("a normal packing must map the camera-facing normal to
blue = 1").** three satisfies it always. Algan fails structurally: the blue
channel encodes world `z`, so toward-viewer normals read B=0. Orbit the camera
or rotate the mob and Algan's colours rotate with the world while three's stay
screen-fixed.

**Defect or convention?** Documented substitution — "Three.js uses *view-space*
normals; only the camera location (not its orientation) is available here"
(`material_shaders.py:537-539`). Convention, with a visible consequence: any
consumer expecting standard view-space encoding misreads every pixel whose B
channel it trusts.

---

## F4. MeshMatcapMaterial fallback: a different kind of function entirely

**Algan** (`material_shaders.py:547-569`):

    rim = clamp(1 - n_dot_v, 0, 1)^3
    out = rgb * (0.3 + 0.7 * n_dot_v) + rim * 0.4      (rim added untinted)

**three.js r185.** With no matcap texture assigned, the shader still builds a
matcap UV from the view-frame normal —
`uv = vec2(dot(x, normal), dot(y, normal)) * 0.495 + 0.5` with
`x = normalize(vec3(viewDir.z, 0, -viewDir.x))`, `y = cross(viewDir, x)`
(`ShaderLib/meshmatcap.glsl.js:87-90`) — and falls back to this built-in ramp,
quoted verbatim:

    vec4 matcapColor = vec4( vec3( mix( 0.2, 0.8, uv.y ) ), 1.0 ); // default if matcap is missing

(`meshmatcap.glsl.js:96-98`), then multiplies:
`outgoingLight = diffuseColor.rgb * matcapColor.rgb`
(`meshmatcap.glsl.js:102`). So the fallback is indexed by the **vertical
component of the view-frame normal** (`uv.y`), ramps grayscale 0.2 -> 0.8, and
is strictly multiplicative.

**Same kind of function?** No.

* three: scalar field over the full view-frame normal (azimuthal dependence
  through `uv.y`), multiplied into albedo; black albedo stays black; maximum
  output is `0.8 * albedo`.
* Algan: depends only on `n.v` (a radially symmetric cone, no azimuthal term)
  plus an **additive** rim. At the silhouette `out = 0.3*rgb + 0.4`: a pure-
  black mob gets a bright grey halo in Algan and stays black in three. Dead-on
  (`n.v = 1`) Algan returns the albedo at full strength; three tops out at
  `0.8 * albedo`.

**What you see:** Algan reads as a front-lit sphere with a white rim light;
three reads as a sphere lit from the top of frame, darker below, no rim. The
difference is stark on dark materials (halo vs none) and at silhouettes.

**Defect or convention?** Declared approximation (`materials.py:31-33`,
docstring `material_shaders.py:559-561`). Convention — but note the authored
`matcap` texture is never sampled either (warned: `materials.py:79-110`,
`emit_warnings` :302-309), so there is no path to spec behaviour even with art
supplied.

---

## F5. MeshToonMaterial: both band the diffuse irradiance scale — but the bands sit in different places

**Where the banding sits (Q1).** Both quantize a scalar function of `n.l`
that scales the diffuse term before ambient/emissive join:

* three: `irradiance = getGradientIrradiance(geometryNormal, light.direction)
  * directLight.color`, then
  `directDiffuse += irradiance * BRDF_Lambert(diffuseColor)`
  (`ShaderChunk/lights_toon_pars_fragment.glsl.js:10-16`);
  `getGradientIrradiance` samples `coord.x = dotNL * 0.5 + 0.5` against the
  gradient map (`ShaderChunk/gradientmap_pars_fragment.glsl.js:9-24`), wired
  in by `ShaderLib/meshtoon.glsl.js:69,74`.
* Algan: `diffuse = rgb * lc * intensity * stepped` with
  `stepped = ceil(clamp(n.l,0,1) * B) / B` (`material_shaders.py:509-517`);
  ambient and emissive unbanded (`material_shaders.py:515-520`).

So Algan bands a step function **of n.l**, not the final colour and not the raw
irradiance colour — structurally the same slot three bands.

**Thresholds.**

* three, N nearest-filtered texels: texel `i` owns
  `coord.x in [i/N, (i+1)/N)` => plateaus over
  `n.l in [-1 + 2i/N, -1 + 2(i+1)/N)`; **band edges at `n.l = -1 + 2i/N`,
  i = 1..N-1**; plateau values are the texel values (author-chosen; the red
  channel is replicated to RGB, `gradientmap_pars_fragment.glsl.js:15-17`).
  Half the domain lies below `n.l = 0`.
* Algan, B bands: value `k/B` on `n.l in ((k-1)/B, k/B]`, k = 1..B (plus 0
  only exactly at `n.l = 0`); **band edges at `n.l = k/B`, k = 1..B-1**;
  plateau values fixed at `k/B`.

**Do the edges coincide?** No, for two independent reasons:

1. *Domain mapping.* three spreads N steps uniformly over `[-1,1]` (spacing
   `2/N`); Algan spreads B uniformly over `[0,1]` (spacing `1/B`). At B=N=3:
   edges `{−1/3, +1/3}` vs `{+1/3, +2/3}`.
2. *Plateau values.* three's come from texture content (e.g. a hand-painted
   ramp); Algan's are a uniform staircase `k/B`. Even aligned edges would not
   align brightness.

**The no-gradientMap default is not banded at all.** Without
`USE_GRADIENTMAP`, three returns
`mix( vec3(0.7), vec3(1.0), smoothstep(0.7 - fw, 0.7 + fw, coord.x) )`
(`gradientmap_pars_fragment.glsl.js:19-22`) — one antialiased step from 0.7 to
1.0 centred at `coord.x = 0.7`, i.e. `n.l = 0.4`. Algan's default
(`bands=3.0`, `materials.py:541`) is three hard cel bands. A default Algan
toon mob shows crisp band boundaries where the three.js default shows a single
soft shadow line.

**External invariant (N bands => exactly N distinct levels).** Algan passes
for `n.l in (0,1]` (levels `{1/B ... 1}`); three passes iff its texel values
are distinct, which a genuine ramp satisfies. Both pass on count; they fail on
placement.

**Inputs consumed / entitlement.** `toon_shader` needs only light origin +
colour; fine. But as a baked material it silently sees plain point lights only
(the bake skips every extended light, `primitives.py:660-666`), while three's
toon responds to ambient/hemisphere (`RE_IndirectDiffuse_Toon`,
`lights_toon_pars_fragment.glsl.js:18-22`) and all direct types. Warned at
author time (`materials.py:151-180`), still dropped at render time. Multi-light
consequence: see F7.

**Defect or convention?** The `bands=` API is an Algan invention (documented,
`materials.py:527-530`). Edge/value placement is an undocumented divergence.
Overall convention; look for edge positions and hard-vs-soft default when
comparing frames.

---

## F6. Phong specular: no `(shininess*0.5 + 1)/pi` factor, no multiplicative n.l, no Fresnel

**three.js r185.** Specular is
`irradiance * BRDF_BlinnPhong(...) * specularStrength`, where
`irradiance = saturate(dotNL) * directLight.color`
(`lights_phong_pars_fragment.glsl.js:15-20`) and
(`ShaderChunk/bsdfs.glsl.js:3-31`, quoted):

    G_BlinnPhong_Implicit() = 0.25;                                   // bsdfs.glsl.js:6
    D_BlinnPhong(s, nh)     = RECIPROCAL_PI * (s*0.5 + 1.0) * pow(nh, s);  // bsdfs.glsl.js:12
    BRDF_BlinnPhong         = F_Schlick(specularColor, 1.0, dotVH) * G * D; // bsdfs.glsl.js:23-29

i.e. peak lobe weight `F * (s+2)/(8*pi)`, scaled by `dotNL` via the
irradiance, with Fresnel brightening at grazing incidence
(`F_Schlick(..., f90=1, ...)`, `common.glsl.js:109-120`).

**Algan.** Both paths:

* torch (`material_shaders.py:322-324`):
  `spec_term = clamp_min(n_dot_h,1e-4)**shininess`;
  `specular_out = specular * radiance * spec_term * (n_dot_l > 0)`.
* kernel (`shading_taichi.py:927-933`): same pow, gate
  `spec_w if n_dot_l > 0 else 0`.

So: **no `(s*0.5+1)/pi` normalization anywhere** (the audit's headline
question — answer: absent, in both sites); `n.l` appears only as a boolean
gate, not a multiplicative falloff; and the Fresnel factor `F_Schlick` is
replaced by the constant `specular` colour. Defaults otherwise match:
`specular=0x111111`, `shininess=30` on both sides
(`materials.py:379-380`; `materials/MeshPhongMaterial.js:63,71`).

**Consequences.**

* Peak height ratio Algan/three at the highlight centre is `~8*pi/(s+2)` per
  unit of `dotNL`: at s=30 that makes Algan's peak ~21% dimmer than three's —
  but broader and flatter, because three's normalized D concentrates energy as
  it narrows.
* Energy: three's form integrates to <= incident by construction. Algan's raw
  `pow(nh,s)` over-integrates for `s < ~23` (highlight can exceed what the
  normalized lobe allows; masked in practice by the small default
  `specular=0x111111 ~= 0.0667`, unmasked if a user raises it) and
  under-integrates for larger s.
* Grazing behaviour: three fades the highlight with `dotNL` and adds grazing
  Fresnel; Algan holds full height anywhere front-facing and never Fresnels.

**What you see:** highlights of the same nominal shininess land with different
peak brightness and footprint; cranking shininess sharpens but does not
brighten the spot in Algan, while in three sharper lobes grow toward
saturation; at grazing angles Algan's highlight stays where three's dies away.

**Inputs consumed / entitlement.** Reads specular/shininess slots (4..6, 7 of
the material block, `shading_taichi.py:33-46`), light rows, camera location
(for the half vector) — all legitimate. The boolean `n.l` gate does correctly
prevent back-facing specular. Nothing improper; the omissions above are the
issue.

**Defect or convention?** Defect relative to spec (undocumented divergence);
the module docstring claims Blinn-Phong parity (`material_shaders.py:311`,
`shading_taichi.py:907`).

---

## F7. Vertex bake is last-light-wins: two point lights on a baked material lose the first

Found during the entitlement pass. `_shade_vertex_colors` loops lights and
ASSIGNS each result into the same channels
(`raytracing/primitives.py:691-698`):

    for light_source in light_sources:
        ...
        self.colors[..., :d] = shaded      # overwrite per light

A mob shaded by any vertex-baked shader (toon/normal/matcap/depth/custom)
under two or more plain PointLights renders lit by **only the last light**
(each shader call re-adds ambient from scratch). three accumulates every light
into `ReflectedLight` (`lights_fragment_begin.glsl.js:66-80` loop).
`shading_taichi.py:52-56` documents the overwrite convention ("the renderer's
vertex path overwrites the colour per light"), and `_stage_default` fixes the
multi-light sum for the kernel path (`shading_taichi.py:797-803`) — so the
limitation is known upstream, but its visible effect on baked materials stands.

**What you see:** add a second PointLight aimed at a toon/matcap mob and
nothing changes except a swap to the new light's position/colour.

**Defect or convention?** Documented legacy convention for the bake path;
visibly divergent from three whenever a baked material meets >1 light.

---

# Light types: falloff, geometry, units

Packed-row layout reference: `scene_builder._pack_lights`
(`raytracing/scene_builder.py:1984-1988`) — RGB radiance 0:3 (intensity
premultiplied), type id 3, decay 4, range 5, direction 6:9, spot cos outer /
cos inner at packed columns 9/10 (= aux cols 6/7 of `Light._build_aux`,
`rendering/lights.py:185-198`), ground RGB / SH at 12:15, power fraction 15.
Intensity is multiplied in once, with no pi, at the ingest point
(`render_loop.py:2411-2430`); three likewise packs `color * intensity` per
type (`webgl/WebGLLights.js:281,313,362,375,409-410`) with no pi.

## Spot

* three: `getSpotAttenuation(coneCos, penumbraCos, angleCos) =
  smoothstep(coneCos, penumbraCos, angleCos)`
  (`ShaderChunk/lights_pars_begin.glsl.js:73-77`), applied in
  `getSpotLightInfo` (`lights_pars_begin.glsl.js:143-168`), with
  `coneCos = cos(angle)`, `penumbraCos = cos(angle * (1 - penumbra))`
  (`WebGLLights.js:316-317`).
* Algan (`shading_taichi.py:717-725`): aux col 6 packs `cos(outer angle)`;
  aux col 7 packs `cos(inner)` where `inner = outer*(1 - penumbra)`
  (`lights.py:486-492`). Kernel: `c = (-ld).dot(beam_axis)`,
  `t = clamp((c - cos_outer)/max(cos_inner - cos_outer, 1e-6), 0, 1)`, then
  `t*t*(3-2t)` — exactly GLSL smoothstep on the same two edges.

**Verdict:** the penumbra maps the same way (`angle*(1-penumbra)` on both
sides; identical smoothstep). Two deviations. (a) With `penumbra == 0` Algan
packs `cos_outer + 1e-4` to keep a hard edge well-defined (`lights.py:490-492`)
while three emits `smoothstep(e, e, x)`, which GLSL leaves undefined for
edge0 >= edge1 but which in practice renders as a hard cut — same image,
different route. *(The driver-behaviour claim is reasoned from the GLSL spec,
not verified on hardware.)* (b) Defaults differ: three's cone half-angle
default is 60 degrees (`lights/SpotLight.js:39`, radians API), Algan's is 30
degrees (`lights.py:449-455`, degrees API).

## Distance falloff (point/spot)

Identical window, different floors and defaults:

* three: `1/max(pow(d, decay), 0.01)` times `(saturate(1 - pow4(d/cutoff)))^2`
  when cutoff > 0 (`lights_pars_begin.glsl.js:56-71`);
  PointLight/SpotLight default `decay = 2` (`lights/PointLight.js:29`,
  `lights/SpotLight.js:39`).
* Algan: `lc /= max(d, 1e-4)^decay`; window `(1 - q^4)^2`, `q = d/range`
  (`shading_taichi.py:705-716`) — algebraically the same window as three;
  **default `decay = 0`**, i.e. no distance falloff (`lights.py:234`).

With matched `decay=2` the curves agree except near the origin: three caps the
boost at `1/0.01 = 100x`, Algan lets it run to `d^-2`. At engine defaults they
differ by `d^2`: Algan's point light illuminates equally at any distance.

## RectArea

* three: an analytic linearly-transformed-cosine integral of the whole
  rectangle — `LTC_Evaluate` builds the polygon's vector form factor on the
  sphere and horizon-clips it (`lights_physical_pars_fragment.glsl.js:252-314`;
  back-side bail at :260), feeding both specular and diffuse of
  **MeshPhysicalMaterial only** (`RE_Direct_RectArea_Physical` :465-523,
  `#define RE_Direct_RectArea` :646, loop guard
  `lights_fragment_begin.glsl.js:158-171`). Intensity is in nits (cd/m^2):
  `power = intensity * width * height * pi` (`lights/RectAreaLight.js:79-80`).
* Algan (`lights.py:497-616`): the rectangle expands into a k x k grid of
  cell-centred **point emitters**, each carrying `1/K` of the colour
  (`render_loop.py:2439-2443`) with one-sided cosine emission
  `lc * max((-ld).dot(rect_normal), 0)` (`shading_taichi.py:726-730`);
  per-sample decay/range apply as for point lights.

**What each converges to.** three converges to the rectangle's projected
solid-angle form factor (the cosine-lobe irradiance integral; saturates below
pi for a surrounding emitter). Algan's sum converges to the *mean over the
rectangle* of `cos_emitter * cos_surface / d^decay`, because each sample row
carries `1/K` of the power rather than an area element `A/K`: it is missing
the factor that would turn a mean into an integral. With the default
`decay = 0` it is not an approximation of any physical quantity at all —
independent of distance and of rectangle size (size only smooths sampling).

**Where the two must differ:**

* *Very close surface.* three's form factor stays bounded (<= ~pi); Algan with
  `decay=2` blows up like `1/d^2`, and with `decay=0` does not change at all
  as the surface approaches.
* *Grazing angle.* LTC handles horizon clipping analytically and exactly;
  Algan clamps per-sample cosines, which is qualitatively right but quantises
  the terminator into K bands at low sample counts.
* *The rectangle's own plane.* Behind the plane both give zero — LTC bails on
  orientation (:260), Algan's per-sample emission clamp goes negative-to-zero
  (`shading_taichi.py:728-730`). This one agrees.
* *Which materials it can light.* In three, rect-area lights illuminate only
  MeshPhysicalMaterial; in Algan every core-lit material receives the samples.

**Units:** not a single scalar ratio — nits times an analytic integral versus
a unitless multiplier times a mean-of-samples are different functions of
geometry, not scaled copies.

## Hemisphere

* three: `mix(groundColor, skyColor, 0.5 * dot(normal, direction) + 0.5)`
  (`lights_pars_begin.glsl.js:202-211`), where `direction` is derived from
  the light's own **position** transformed to view space
  (`WebGLLights.js:564-565`), then multiplied by `BRDF_Lambert`
  (a further `1/pi`) via `RE_IndirectDiffuse_*`.
* Algan (`shading_taichi.py:682-691`): `h = 0.5 + 0.5*n.dot(up)`;
  `lc = ground*(1-h) + sky*h`; applied as direct light with `ld = n`
  so the stage's `n.l` becomes exactly 1 — **no `1/pi`**. The axis is an
  explicit `up=` parameter defaulting to `UP` (`lights.py:387-408`),
  packed at aux cols 3:6; the ground colour is radiance-bearing aux 9:12,
  opacity-scaled (`lights.py:382-422`).

Blend shape identical; two divergences: the axis source (position vector vs
authored unit vector — same result for a light parked at `(0,1,0)`), and the
pi from F1 (Algan hemisphere light pi x hotter end to end).

## Units summary (exact scalar ratios, same authored Light color c and intensity i)

| Light | three r185 convention | Algan convention | End-to-end ratio (Algan / three) |
| --- | --- | --- | --- |
| Point | candela (`PointLight.js:25`), decay=2 => E = c*i/d^2 | unitless multiplier, decay=0 => E = c*i | engine defaults: `d^2/pi` (distance-dependent — **not one scalar**). Matched decay=2: **pi** |
| Spot | candela (`SpotLight.js:33`), decay=2, cone smoothstep | unitless, decay=0, identical cone math | as Point |
| Directional | irradiance multiplier, no falloff (`WebGLLights.js:281`) | same (`_light_eval` directional branch, `shading_taichi.py:676-678`) | **pi** |
| Ambient | summed into `ambientLightColor`, then `BRDF_Lambert` => c*i*albedo/pi (`WebGLLights.js:261-265`, `lights_fragment_begin.glsl.js:177`) | direct row with forced n.l=1 => c*i*albedo (`shading_taichi.py:679-681`) | **pi** |
| Hemisphere | mix(ground, sky, w)*albedo/pi | blend*albedo | **pi** |
| RectArea | nits x LTC solid-angle integral | unitless x mean of K point samples | **not a scalar** (different function) |

(The pi column is F1 restated per light type. Algan additionally has a
built-in ambient fill, `AMBIENT_STRENGTH` 0.1 display-referred / 0.01 linear,
`material_shaders.py:41-50`, `shading_taichi.py:98-107` — three has no
counterpart without an AmbientLight in the scene.)

---

# External invariants, checked per material

| Material | Invariant tested | three.js | Algan |
| --- | --- | --- | --- |
| MeshBasicMaterial / `_stage_unlit` | unlit must reproduce authored colour exactly | yes (`ShaderLib/meshbasic.glsl.js` outputs `diffuseColor`, no light terms) | yes — `material_shaders.py:259-260` returns albedo; `_stage_unlit` (`shading_taichi.py:782-788`) is a passthrough. Trivially satisfied on both sides |
| MeshLambertMaterial (and all diffuse) | reflected <= incident irradiance | yes (`BRDF_Lambert` normalized) | **no** — missing `1/pi` (F1); masked only partially by the display-referred energy budget |
| MeshPhongMaterial specular | lobe integrates to <= incident | yes (normalized D, implicit G) | **no in general** — unnormalized pow over-integrates for shininess <~ 23, under for larger s (F6) |
| MeshToonMaterial | N bands => N distinct levels | yes with distinct texels | yes for `n.l in (0,1]` (F5) |
| MeshNormalMaterial | camera-facing normal -> `(0.5, 0.5, 1)` | yes (view space) | **no** — world space + z mirror gives `(0.5, 0.5, 0)` (F3) |
| MeshMatcapMaterial | output never exceeds the sampled matcap * albedo | yes (multiplicative, max 0.8*albedo) | **no** — additive rim exceeds albedo at silhouettes (F4) |
| MeshDepthMaterial | monotonic near-white -> far-black over [near, far] | yes (hyperbolic) | yes (linear), but endpoints driven by different near/far than the render camera (F2) |

# Inputs consumed and entitlement

* `basic_material_shader` / `_stage_unlit`: consume nothing but albedo.
  Clean.
* All lit torch shaders receive `light_intensity` and
  `ambient_light_intensity` as literal `1`s from the bake loop
  (`primitives.py:687-688`) — both parameters are dead in practice;
  `AMBIENT_STRENGTH` is the real coefficient. A reader tuning the parameter
  would see nothing change.
* `env_map_intensity` is read by every core shader but scales only the flat
  ambient fill (e.g. `material_shaders.py:285`, slot 11 at
  `shading_taichi.py:878`): there is no environment map behind it. Slot
  overload, documented as an approximation (`standard_shader` docstring,
  `material_shaders.py:351-353`).
* `physical_shader` accepts `attenuation_sigma` and deliberately does not use
  it (absorption belongs to the wavefront bounce loop;
  `material_shaders.py:407-411`); `iridescence` accepted-unused
  (`shading_taichi.py:41`). Documented, clean.
* Baked materials (toon/normal/matcap/depth/custom) are fed plain point lights
  only — every extended light is skipped silently at bake time
  (`primitives.py:660-666`). Warned once at `set_material` time
  (`materials.py:151-180`), not at render time.
* The baked multi-light overwrite (F7) means "which lights exist" silently
  reduces to "the last one" for those materials.
* `depth_shader`'s near/far come from material defaults unrelated to the Scene
  camera (`materials.py:585`; `Camera.near/far` exist at
  `rendering/camera.py:84-86` and are never consulted).
* Kernel stages read the canonical block slots listed at
  `shading_taichi.py:33-46` plus the geometry-declared one-sided flag
  (slot 26) and Beer-Lambert sigma (slots 27-29, applied by the bounce loop,
  not the stages) — consistent with the packing rules.

# Reasoned but not verified by reading source

* GLSL nearest-filter texel-boundary arithmetic (F5 thresholds) follows from
  the OpenGL ES sampling definition plus
  `gradientmap_pars_fragment.glsl.js` — standard behaviour, not quoted from
  this three.js tree's docs.
* Driver handling of `smoothstep(e, e, x)` as a hard cut (Spot section).
* sRGB byte estimates in F1/F2 are analytic applications of the OETF, not
  rendered frames (this audit ran no renders, per its constraints).
* The claim that `Camera.get_forward_direction()` resolves to +z for the
  default scene rests on `CAMERA_ORIGIN = ORIGIN + OUT*7`
  (`constants/spatial.py:33`) plus the screen placement arithmetic in
  `rendering/camera.py:106`; the basis-row sign convention
  (`mob_orientation.py:230-240`, "local -z axis") was reconciled from those
  two rather than by executing a scene.

# Question-to-finding index

Q1 toon -> F5. Q2 normal -> F3. Q3 matcap -> F4. Q4 depth -> F2.
Q5 phong -> F6. Q6 lambert -> F1. Light types -> Spot/Distance/RectArea/
Hemisphere sections and units table.
