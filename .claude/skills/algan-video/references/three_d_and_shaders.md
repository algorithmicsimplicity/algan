# 3D scenes, lighting, materials, textures and shaders

Only triangle geometry is lit: `Surface` and subclasses (`Sphere`, `Cylinder`,
`Cone`, `Torus`, `ImageMob`, `Dot3D`), `Cube`, `Prism`, the polyhedra,
`Model3D`, `TriangleMesh`, `TriangulatedBezierCircuit`. Bezier circuits
(`Square`, `Circle`, `Line`, `Text`, `Tex`, Manim vector Mobjects) are drawn in
their own colour, receive no light and no shadow, but do cast shadows and can be
made reflective or transmissive. For a lit flat shape use
`TriangulatedBezierCircuit`, `TextTriangulated` or `TexTriangulated`.

## Camera

```python
camera = Scene.get_camera()                      # at OUT * 7, looking at ORIGIN, FOV ~53 degrees
camera.move(OUT * 2)                             # dolly back
camera.rotate(90, UP, about=ORIGIN)              # turntable; stays aimed at the centre
camera.orbit(90, UP, about=ORIGIN)               # travels without turning
camera.look_at(point)
camera.center_on(mob)                            # reframe on a Mob
camera.set_fov(20)                               # telephoto; 90 = wide. Animatable
camera.add_updater(lambda cam, t: cam.look_at(ball.location))   # track a subject
with Off():
    camera.set_near_orthographic()               # parallel projection for technical diagrams
    camera.set_near(0.5); camera.set_far(50)     # clip planes; configuration, set once
```

Screen-relative placement (`move_to_screen_edge` etc.) reads the camera when
recorded. To keep a caption fixed through a camera move:
`Scene.get_camera().add_children([caption])` before positioning it.

## Lights

Every Scene starts with one white `PointLight` above and right of the camera.
Lights are Mobs: spawn them (inside `Off()`), animate `location`, `color`,
`intensity`, `orbit` them, despawn them. Spawning registers the light.

```python
with Off():
    Scene.clear_lights()
    SpotLight(location=UP * 6 + RIGHT * 4 + OUT * 4, target=ORIGIN, intensity=60,
              cone_angle=30, penumbra=0.5, decay=2, shadow_radius=0.3).spawn()   # key
    PointLight(location=LEFT * 6 + OUT * 2, intensity=4).spawn()                 # fill
    DirectionalLight(location=IN * 8 + UP * 4, target=ORIGIN,
                     color=Color((0.6, 0.7, 1.0))).spawn()                       # rim
    AmbientLight(intensity=0.25).spawn()
    HemisphereLight(color=BLUE, ground_color=(0.4, 0.3, 0.1), intensity=0.8).spawn()
    RectAreaLight(location=UP * 5, target=ORIGIN, width=4, height=4, samples=9,
                  intensity=30).spawn()                                          # softbox
```

- `PointLight`: omnidirectional. `decay=2` gives inverse-square falloff (raise
  `intensity` a lot), `distance` a range cutoff, `shadow_radius` a soft shadow.
- `DirectionalLight`: parallel rays from `location` toward `target`;
  `shadow_angle` (degrees) softens shadows.
- `AmbientLight`: flat fill; nearly every rig wants a little.
- `HemisphereLight`: sky colour from `color`, ground from `ground_color`.
- `SpotLight`: cone with `cone_angle` (half-angle) and `penumbra` 0..1.
- `RectAreaLight`: one-sided emitting rectangle; radiance is
  `color * intensity / area`; `samples` sets penumbra smoothness and shadow slot
  usage; keep `decay`/`distance` at their defaults.
- Shape parameters (`decay`, `distance`, cone angles, emitter sizes) are fixed
  per light, not animatable.

### Shadows

```python
SETTINGS.raytracing.set(shadows=True)            # off by default
```

Hard-edged unless the light has a size (`shadow_radius`, `shadow_angle`, or an
area light). Up to 16 shadow-casting light slots by default; a 3x3-sample area
light uses 9. `ALGAN_SOFT_SHADOW_SAMPLES` (default 8) and
`ALGAN_MAX_SHADOW_LIGHTS` are environment variables read before `import algan`.
Shadow rays respect opacity but travel straight through glass (no caustics).

### Environment maps

```python
Scene.set_environment_map("studio.png", intensity=1.0, ambient=True)   # equirectangular, sky at top
Scene.set_environment_map(None)                                          # remove
```

Acts as a skybox (visible in the background, reflections and refractions) and,
with `ambient=True`, lights the scene. Also accepts an `[H, W, 3]` array. This is
the single biggest improvement for metal and glass.

## Materials

Apply before `spawn()`. Numeric and colour properties become animatable
attributes in `snake_case` (`mob.roughness`, `mob.emissive_intensity`);
constructors accept Three.js `camelCase` (`emissiveIntensity=2`).

| Material | Lighting | Key properties (defaults) |
| --- | --- | --- |
| `MeshBasicMaterial` | Unlit | `color` |
| `MeshLambertMaterial` | Diffuse | `emissive` (black), `emissiveIntensity` (1) |
| `MeshPhongMaterial` | Blinn-Phong | `specular`, `shininess` (30), `emissive` |
| `MeshStandardMaterial` | PBR | `roughness` (1), `metalness` (0), `emissive`, `envMapIntensity` (1) |
| `MeshPhysicalMaterial` | PBR + extras | `clearcoat` (0), `clearcoatRoughness`, `ior` (1.5), `transmission` (0), `sheen`, `sheenColor`, `specularIntensity`, `iridescence` |
| `MeshToonMaterial` | Cel bands | `bands` (3) |
| `MeshNormalMaterial` | Normals as colour | `flatShading` |
| `MeshMatcapMaterial`, `MeshDepthMaterial` | Approximations | `color`; `near`, `far` |

A material's `color` defaults to `None` (keep the Mob's colour). Presets
(`WOOD`, `GLASS`, `PLASTIC`, `RUBBER`, `CERAMIC`, `STONE`, `MIRROR`,
`BRUSHED_METAL`, `CHROME`, `COPPER`) configure surface response and a flat base
colour only, no texture detail. Unconfigured 3D Mobs use
`SETTINGS.style.default_material` (`DiffuseMaterial`); replace it scene-wide
with `SETTINGS.style.set(default_material=MeshStandardMaterial(roughness=0.3))`.

Texture slots forwarded by `set_material`: `map`, `normal_map`, `roughness_map`
(green channel), `metalness_map` (blue channel), each a path or `[H, W, C]`
image, sampled per fragment on UV-bearing geometry (`Surface` family,
`TriangleMesh` with UVs). Property maps are static once spawned; `map` on a
`Surface` lands on the animatable `color_texture`. Other Three.js slots
(`envMap`, `aoMap`, `matcap`, ...) are accepted and dropped with a warning;
`wireframe`, `vertexColors` and non-default `side` are unsupported.

### Reflections and glass

- `metalness` 0 = dielectric, 1 = bare metal; `roughness` 0 = mirror, 1 =
  diffuse. Traced reflection fades out by about `roughness=0.35`;
  `SETTINGS.raytracing.set(glossy_reflection=True)` blurs a screen-space
  reflection instead.
- A fully metallic object in an empty scene renders black: add an environment
  map, surrounding objects, or back metalness off to 0.7..0.9. Flat or gently
  curved mirrors read better than a mirrored sphere.
- Glass: `MeshPhysicalMaterial(transmission=1.0, ior=1.5, roughness=0.0)` with
  `opacity=1.0`. `opacity` is coverage (fade), `transmission` is transparency.
  A tinted colour tints transmitted light. Put a patterned backdrop behind it.
  IOR: water 1.33, glass 1.5, sapphire 1.77, diamond 2.42.
- `SETTINGS.raytracing.set(max_bounces=8)` caps reflected/refracted depth. A
  solid glass sphere needs at least 4; dark patches inside glass mean too few.
  Lower it for drafts.

## Textures on surfaces

`Surface`-based Mobs take `[W, H, C]` tensors in `(u, v)` layout:
`color_texture` (5 channels), `roughness_texture`, `reflectivity_texture`,
`refractive_index_texture`, `glow_texture` (1 channel), `normal_texture`
(3, tangent space, `(0, 0, 1)` = unperturbed). Handing them a path raises;
use `surface.set_color_by_image(path)` or `get_image(path)`.

```python
globe = Sphere(radius=1.5, color_texture=get_checkerboard((RED, WHITE), resolution=8)).spawn()
globe.color_texture = get_stripes((BLUE, WHITE))              # animates, texel by texel
globe.color_texture = globe.color_texture.mult_opacity(0.5)   # arithmetic on the map
xyz = Sphere(radius=1.5).get_texture_locations((256, 256))    # world position per texel
world = ImageMob("map.png").scale(2).spawn()
world.set_shape_to(Sphere(radius=2, add_to_scene=False))      # reshape; texture follows
```

Closed surfaces wrap their maps seamlessly. Glow maps are baked per vertex
(raise `grid_width`/`grid_height` for detail). Normal maps only affect
per-fragment effects (reflection, refraction, shadows, fragment shading).
`Surface(..., render_tolerance_pixels=0.5)` controls tessellation; raise it for
close-up surfaces that are slow or out of memory.

2D circuits colour through a grid: `Square(grid_width=64, grid_height=64)` then
`set_color_by_function(lambda uv: rgb_or_rgba_or_5ch)` or
`set_color_by_image(path)`; `Line` gets a single `t` along its length. Both are
recorded as cross-fades.

## Imported models

```python
model = Model3D("dragon.glb", fit_to_size=2.0).spawn()   # recentred, bounding diagonal = 2 units
model.roughness = 0.1                                     # imported PBR materials are animatable
print(model.node_names); arm = model.get_part("LeftArm"); arm.rotate(45, OUT)
print(model.animation_names); model.play_animation("Walk", runtime=4, loop=2)   # rigid node animation only, no skinning
```

Symptoms: nothing visible means unit scale (use `fit_to_size`); black means a
metallic material with nothing to reflect; faceted means no authored normals;
slow means triangle count or normal maps.

## Custom shaders

### Fragment pipelines (preferred)

`mob.set_fragment_shader(stage_or_list)` before `spawn()`. A list runs left to
right; each stage receives the previous colour. Every stage parameter becomes an
animatable attribute (duplicate names across stages are suffixed). Lighting
stages: `STAGE_UNLIT`, `STAGE_LAMBERT`, `STAGE_PHONG`, `STAGE_STANDARD`,
`STAGE_PHYSICAL`, `STAGE_MANIM`; the vertex-shader functions `phong_shader`,
`standard_shader`, `lambert_shader`, `physical_shader`, `manim_shader`,
`basic_material_shader`/`null_shader` resolve to them. Library stages
(additive, layer over a lit base): `fresnel_rim` (`rim_color`, `rim_gain`,
`rim_power`), `glass_ball`. Example recolour stage: `cosine_color`
(`frequency`, `phase`).

```python
ball.set_fragment_shader([cosine_color, STAGE_STANDARD, fresnel_rim])
ball.rim_color = (0.4, 0.9, 1.0)     # width-3 tuple; TEAL_A[..., :3] to use a palette colour
```

Custom stage contract (copy `_stage_cosine_color` in
`algan/rendering/shaders/fragment_shaders.py`):

```python
from algan import *
from algan.taichi_compat import ti

@ti.func
def _stage_bands(pos, view_dir, n_interp, face_n, in_rgb, in_glow,
                 params: ti.template(), f, prim, off,
                 light_pos: ti.template(), light_col: ti.template(), num_lights,
                 shadows: ti.template(), vis, cam_pos):
    tm = f % params.shape[0]
    freq = params[tm, prim, off + 0]       # parameter slots follow the spec order; a width-3
    phase = params[tm, prim, off + 1]      # parameter occupies three consecutive slots
    k = 0.5 + 0.5 * ti.cos(pos[1] * freq + phase)
    return ti.math.vec4(in_rgb[0] * k, in_rgb[1] * k, in_rgb[2] * k, in_glow)

bands = FragmentStage(_stage_bands, [("frequency", 1, 6.0), ("phase", 1, 0.0)])
mob.set_fragment_shader([bands, STAGE_STANDARD]); mob.spawn()
mob.frequency = 12.0
```

Inputs: `pos` world position, `view_dir`, `n_interp` interpolated normal,
`face_n` face normal, `in_rgb`/`in_glow` colour from the previous stage,
`params[tm, prim, off + slot]` the animatable parameters, `light_pos`/`light_col`
packed light rows with `num_lights`, `shadows`/`vis` shadow visibility,
`cam_pos`. Return `ti.math.vec4(r, g, b, glow)`. A stage may also carry
`scatter=` (how the ray continues: pass-through, mirror, refract); the template
is `_scatter_forced_mirror` in the same file, and the contract is documented in
`algan/rendering/raytracing/shading_taichi.py`. A new pipeline costs one kernel
compile on its first render (cached afterwards). Both renderers run pipelines;
the path tracer picks one scatter branch at random per hit.

### Vertex shaders (PyTorch, per vertex)

`mob.set_shader(fn)` before `spawn()`. The function takes nine fixed parameters
then its own, which become animatable:

```python
def my_shader(memory, vertex_location, vertex_normal, albedo_color,
              camera_location, light_origin, light_color, light_intensity,
              ambient_light_intensity, banding=4.0):
    ...     # torch operations returning a colour per vertex
    return color
mob.set_shader(my_shader); mob.banding = 8.0
```

Working implementations to copy: `basic_material_shader` in
`algan/rendering/shaders/material_shaders.py` and `basic_pbr_shader`
(`smoothness`, `metallicness`) in `algan/rendering/shaders/pbr_shaders.py`. A
custom vertex shader is baked before the frame renders, sees one plain point
light and never receives shadows; Algan warns when such a Mob shares a scene
with other light types, an environment map or shadows. Reuse one function object
across Mobs: distinct shader objects batch separately.

## Renderer choice

`SETTINGS.raytracing.samples_per_pixel`: `1` (default) is the deterministic
hybrid renderer, noise-free, with exact analytic edges for 2D and text. `> 1`
is the Monte Carlo path tracer: global illumination, emissive surfaces as
lights, importance-sampled environment maps, physically blurred rough
reflections and refractions, every light shadowed, denoised by default
(`denoise=True`). Start with `samples_per_pixel=16, max_bounces=2` and raise
either when structure remains in the noise or indirect light is missing.
Supersampling does not apply to it. Neither renderer does caustics, ambient
occlusion, displacement, wireframe, or heterogeneous volumes.
