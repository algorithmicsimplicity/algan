# Renderer-audit scene spec

A single JSON document describing one still frame, rendered by two independent
back ends (`algan_render.py`, `three_render.mjs`) so their images can be
compared pixel-wise. The spec is deliberately small: only what both engines can
express *exactly*, so that a difference in the output is a difference in the
renderer and not in the scene.

Conventions shared by both back ends:

* Right-handed, **+Y up, +Z toward the viewer** — Algan's convention and
  Three.js's are already identical.
* Distances in world units, angles in **degrees**.
* Colours are **linear-ish RGB triples in [0, 1]** as authored. What each engine
  then does with them (gamma, tonemapping) is part of what the audit measures,
  so neither back end is allowed to "correct" for the other.
* `fov` is the **vertical** field of view in degrees (both engines agree).

```jsonc
{
  "name": "glass_and_metal",
  "render": {
    "width": 640, "height": 480,
    "background": [0.02, 0.025, 0.035],   // solid background colour
    "samples_per_pixel": 1                // three: pathtracer sample count
  },
  "camera": {
    "position": [0, 2.0, 12.0],
    "target":   [0, 0.2, 0],
    "up":       [0, 1, 0],
    "fov": 40, "near": 0.1, "far": 200
  },
  "lights": [
    // "directional": no falloff in either engine. `direction` points FROM the
    // light TOWARD the scene (Three.js's light.position is -direction).
    {"type": "directional", "direction": [-0.5, -0.8, -0.6],
     "color": [1, 0.97, 0.92], "intensity": 2.2},
    // "point": `decay` 0 = no falloff (both engines: attenuation == 1),
    // 2 = inverse-square. `distance` 0 = no range cutoff.
    {"type": "point", "position": [-5, 4, 4], "color": [0.55, 0.7, 1.0],
     "intensity": 1.0, "decay": 0, "distance": 0},
    {"type": "ambient", "color": [1, 1, 1], "intensity": 0.06}
  ],
  "objects": [
    {
      "name": "floor",
      "geometry": {"type": "box", "size": [40, 0.4, 40]},
      "position": [0, -1.2, 0],
      "rotation_y": 0,                     // degrees about +Y, applied about the
                                           // object's own centre
      "material": {
        "type": "physical",                // "physical" | "standard" | "basic" | "lambert"
                                           // | "phong" | "toon" | "normal" | "matcap" | "depth"
        "color": [0.5, 0.5, 0.54],
        "roughness": 0.85, "metalness": 0.0,
        "ior": 1.5, "transmission": 0.0,
        "clearcoat": 0.0, "clearcoat_roughness": 0.0,
        "sheen": 0.0, "sheen_roughness": 1.0, "sheen_color": [0, 0, 0],
        "emissive": [0, 0, 0], "emissive_intensity": 1.0,
        "specular_intensity": 1.0, "specular_color": [1, 1, 1],
        "opacity": 1.0,
        "attenuation_color": [1, 1, 1], "attenuation_distance": 0
      }
    }
  ]
}
```

Geometry types:

| `type`   | fields                                | meaning |
| -------- | ------------------------------------- | ------- |
| `sphere` | `radius`, `segments` (default 64)     | Three.js `SphereGeometry(radius, segments, segments/2)`; Algan `Sphere(radius=...)` |
| `box`    | `size` `[x, y, z]`                    | Three.js `BoxGeometry`; Algan `Prism(dimensions=...)` |

Every material field is optional; the defaults above are the defaults both back
ends apply. Fields both engines ignore are still allowed in the file so that one
spec can drive an engine that grows the feature later.

## Material types

Existing types: `basic`, `standard`, `physical`. Additional types:

| type | new fields | Algan class | three.js class |
| --- | --- | --- | --- |
| `lambert` | (`emissive`, `emissive_intensity`) | `MeshLambertMaterial` | `MeshLambertMaterial` |
| `phong` | `specular` (rgb, default [0.067,0.067,0.067] — three's own 0x111111), `shininess` (default 30) | `MeshPhongMaterial` | `MeshPhongMaterial` |
| `toon` | `bands` (default 3) — see note below | `MeshToonMaterial` | `MeshToonMaterial` |
| `normal` | — (ignores every field incl. colour) | `MeshNormalMaterial` | `MeshNormalMaterial` |
| `matcap` | — (no matcap image is sampled; see below) | `MeshMatcapMaterial` | `MeshMatcapMaterial` |
| `depth` | `near` (default 0.1), `far` (default 100) — Algan only, see below | `MeshDepthMaterial` | `MeshDepthMaterial` |

Notes the back ends must respect:

* **toon `bands`.** The two engines do not share a mechanism. three.js's default
  toon shading (no `bands`) is a 2-step smoothstep at `dotNL·0.5+0.5 = 0.7`
  mixing 0.7 → 1.0; its documented way to get N bands is a `gradientMap`, so
  when `bands` **is given** the three.js back end builds a
  `THREE.DataTexture` of N texels ramping 0..1 (texel *i* = round(i/(N−1)·255))
  with `NearestFilter` on both min and mag, `needsUpdate = true`, colorSpace
  left at its default — that translation is what the audit measures. Algan
  quantises directly: `ceil(dotNL·N)/N` with `dotNL` clamped to [0, 1]. Band
  edges therefore land in different places on each engine by construction.
* **depth `near`/`far`.** The engines define this material differently and the
  panel is **informational rather than a parity test**. Algan's
  `MeshDepthMaterial(near=, far=)` renders a *linear* ramp of camera distance
  over `[near, far]` (near = bright). three.js's plain `MeshDepthMaterial`
  takes no near/far fields: it uses the **camera's** projection and writes the
  non-linear `gl_FragCoord.z` depth (`vec3(1 - fragCoordZ)` under basic depth
  packing), so the spec's `near`/`far` are simply not passed to it. No custom
  shader hack is made to force agreement.
* **matcap.** Neither engine samples a matcap image here. Algan's
  `matcap_shader` approximates a default matcap with a view-facing diffuse term
  plus a rim highlight tinted by the base colour:
  `rgb·(0.3 + 0.7·dotNV) + rim³·0.4`. three.js's `meshmatcap_frag` without an
  assigned matcap texture substitutes its built-in default
  `vec4(vec3(mix(0.2, 0.8, uv.y)), 1.0)` (uv from the view-space normal) and
  multiplies the base colour into it. Informational.
* `basic`, `normal`, `matcap` and `depth` take no lighting fields (no
  `roughness`/`metalness`/`emissive`/...).
* **Algan caveat worth knowing when reading that panel:** `MeshToonMaterial`,
  `MeshNormalMaterial`, `MeshMatcapMaterial` and `MeshDepthMaterial` have no
  in-kernel shader port in Algan and are baked into vertex colours before the
  frame renders, so those four see only a plain point light's contribution,
  receive no shadows, and warn at `set_material` time when the rig asks for
  more (the warning is a finding, not noise). three.js shades all of these per
  fragment.

## Light types

Existing: `directional` (via `direction`), `point`, `ambient`. Additions:

* **`directional`, position+target form.** As an alternative to `direction`,
  a directional light accepts `"position": [x,y,z], "target": [x,y,z]`
  (light sits there, shines toward the target). Exactly one form must be
  given — both back ends reject a light carrying both or neither.
* **`spot`**: `position`, `target`, `color`, `intensity`, `angle` (half-angle in
  **degrees**, required — the engines' own defaults disagree), `penumbra`
  [0,1] (default 0), `decay` (default 0 = no falloff; overrides three's own
  constructor default of 2), `distance` (default 0 = unlimited). Both sides
  set `castShadow = true`.
* **`rect_area`**: `position`, `target`, `color`, `intensity`, `width`,
  `height` (required), plus optional `samples` (**Algan only**, default 4;
  number of deterministic emitter samples). The three.js side builds
  `RectAreaLight(color, intensity, width, height)`, positions it and
  `lookAt(target)`. It **requires `RectAreaLightUniformsLib.init()` once before
  the first render** (the back end imports it via the page import map as
  `three/addons/lights/RectAreaLightUniformsLib.js`; without it the light
  renders black), and it **cannot cast shadows in three.js** — unlike Algan's
  sampled-rect implementation. Recorded so the shadow difference is read as an
  engine limit, not a bug. One more three.js limit decides what a scene may
  put in front of one: `RE_Direct_RectArea` is defined only for
  `MeshPhysicalMaterial` (and so `MeshStandardMaterial`), so in three.js a
  rect-area light does not illuminate a `lambert`, `phong` or `toon` surface at
  all, while in Algan it illuminates every lit material. Give a rect-area
  light a `standard` receiver, or the panel measures that limit instead of the
  light.
* **`hemisphere`**: `color` (sky), `ground_color`, `intensity`. No shadows on
  either side.

Coordinates: every new light `position`, `target` and `direction` crosses the
frame boundary through each back end's flip helper (`_vec` in `algan_render.py`;
three.js needs none). A `target` is a point and flips too. A hemisphere light's
`up` stays `(0, 1, 0)` — Y is up in both frames; only Z negates.
