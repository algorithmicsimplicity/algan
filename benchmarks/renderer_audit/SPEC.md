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
        "type": "physical",                // "physical" | "standard" | "basic"
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
