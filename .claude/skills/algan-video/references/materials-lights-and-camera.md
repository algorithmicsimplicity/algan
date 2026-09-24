# Materials, lights, cameras, and renderer selection

Implement the user's specified surface response and camera/light behavior. The
examples here name controls; they do not prescribe a lighting rig or visual style.

## Material installation and animation

Configure a material **before** spawn, then animate the properties registered on
the Mob:

```python
ball = Sphere()
ball.set_material(MeshStandardMaterial(roughness=0.6, metalness=0.0))
ball.spawn()
with Sync(runtime=2.0):
    ball.roughness = 0.2
    ball.metalness = 1.0
```

Changing the original Python material instance is not the same as animating the
Mob. Material constructors accept Three.js-style spellings where documented;
Mob attributes use snake case, for example `emissive_intensity` and
`clearcoat_roughness`. A material with `color=None` preserves the Mob's color.

| Material | Response controlled |
|---|---|
| `MeshBasicMaterial` | Unlit base color |
| `MeshLambertMaterial` | Diffuse lighting and emissive contribution |
| `MeshPhongMaterial` | Diffuse/specular lighting, including shininess |
| `MeshStandardMaterial` | PBR roughness and metalness |
| `MeshPhysicalMaterial` | Extended PBR response, including transmission, IOR, clearcoat, and sheen |
| `MeshToonMaterial` | Banded diffuse response |
| `MeshNormalMaterial`, `MeshDepthMaterial`, `MeshMatcapMaterial` | Diagnostic or approximate specialized appearances; verify their limitations |

Unconfigured 3D content uses the configured default material, currently
`DiffuseMaterial`. Flat 2D shapes and text are unlit; adding a light does not
make their ordinary flat fill behave like a lit 3D surface. Select the actual
geometry and material needed for the requested effect. For 3D geometry that
should show its own colors (glowing wires, textured display panels), pass
`unlit=True` to the constructor; it installs the same shading as
`MeshBasicMaterial` on the Mob and its parts.

Presets such as `GLASS`, `MIRROR`, `WOOD`, and `COPPER` configure material
properties and sometimes base color. They do not automatically add grain,
surface detail, or texture maps.

### Transparency is not glass transmission

`mob.opacity` controls coverage/transparency. Physical transmission and index of
refraction come from a transmissive material:

```python
ball = Sphere().set_material(
    MeshPhysicalMaterial(transmission=1.0, ior=1.5, roughness=0.1)
)
ball.spawn()
```

The values are mechanical examples. Use requested or explicitly established
values. Do not fake required refraction by lowering opacity without disclosure.
Reflection behavior comes from material controls such as roughness and metalness;
there are no separate general-purpose Mob reflectivity or refractive-index
setters. The scene must contain the required environment/objects for rays to
reflect or refract them.

Install `two_sided`, `casts_shadows`, and `receives_shadows` before spawning.
Do not attempt to animate a shader installation or these geometry declarations.
For an actual material-model switch, create an unspawned replacement and perform
a user-specified swap or transition. If replacing a Mob, account for its parent,
references, and updater dependencies; do not assume they automatically rebind.

`set_material`, `set_shader`, and `set_fragment_shader` install the shading
configuration. They are not three accumulating layers. To compose custom stages
with lighting, build one fragment pipeline; see [custom shaders](custom-shaders.md).

### Textures and supported slots

`map`, `normal_map`, `roughness_map`, and `metalness_map` forward to compatible
UV-bearing geometry. They accept paths or conventional image arrays. A Surface's
base-color texture is animatable, while material property maps are static.
Unsupported Three.js properties can be accepted with a warning and ignored;
constructor acceptance does not prove rendering support. Do not promise parity
with an entire Three.js material just because the class names match.

See [texture layouts](geometry-text-and-assets.md#images-and-texture-layouts)
for image versus UV ordering, channel conventions, and model asset handling.

## Lights

A Scene starts with a default PointLight. Inspect `Scene.get_light_sources()`
rather than assuming a blank rig. A newly spawned light registers with the Scene.
Only clear the existing rig when implementing a deliberate replacement:

```python
with Off():
    Scene.clear_lights()
    light = PointLight(location=UP * 3 + OUT * 2, intensity=2).spawn()
```

Configure setup lights in `Off()` so spawning them does not add unwanted opening
animation time. Lights are Mobs: location, color, and intensity are animatable.
`Scene.remove_light(light)` removes one; `Scene.clear_lights()` removes all.

Available light types include `PointLight`, `DirectionalLight`, `AmbientLight`,
`HemisphereLight`, `SpotLight`, and `RectAreaLight`. Read each signature for its
specific parameters. Directional and spot lights can point at a `target`;
spot `cone_angle` is in degrees. Hemisphere lights use a sky color and
`ground_color`. Light size, cone parameters, decay, and distance controls are
plain configuration rather than guaranteed timeline attributes. Set them before
rendering; do not write a sequence of assignments and expect interpolation.

Point lights default to no distance falloff. A `decay` setting changes attenuation
and the needed intensity. Use the requested physical or nonphysical convention
rather than silently choosing one. A finite distance cutoff is not identical
to an unbounded inverse-square light.

`SETTINGS.raytracing.set(shadows=True)` enables shadows. Verify whether each
object should cast and receive them. Built-in materials use in-kernel fragment
lighting and the scene's supported light rig. A custom plain PyTorch vertex
shader has a narrower path: plain PointLight lighting and no received shadows;
other lighting features can be warned about and omitted. A custom fragment
pipeline is the appropriate interface when the required effect needs in-kernel
lighting, not a way to make a vertex callback magically support every light.

An environment map is scene lighting/transport input:

```python
Scene.set_environment_map('assets/environment.hdr')
```

Use the supplied asset and verify that its format loads. A screen background
image is a different operation; it does not substitute for a 3D environment in
reflections. Do not add or choose an environment as unsolicited creative direction.

## Camera control

Get the scene's existing camera with `Scene.get_camera()`. It is a Mob, so common
movement and rotation methods apply. Configure its initial pose in `Off()` and
record subsequent moves in the appropriate timing contexts.

Read the default pose from the installed version (`camera.location`,
`camera.get_fov()`) instead of relying on a remembered default, and set the pose
explicitly for any composed shot. `fov` is the vertical field of view in degrees.
At distance `d` along the viewing direction the frame spans
`height = 2 * d * tan(fov / 2)` world units and `width = height * W / H` for the
output resolution `W x H`, which matters most for narrow portrait frames.
`camera.visible_size_at(point)` returns that `(width, height)` directly for the
plane through `point` facing the camera, using the current resolution (or a
`CameraView`'s capture resolution for its camera):

```python
width, height = Scene.get_camera().visible_size_at(ORIGIN).flatten().tolist()
```

| Operation | Effect |
|---|---|
| `camera.fly_to(p, look_at=t, via=w, look_at_via=u)` | Move position and aim together, horizon level, optionally curved |
| `camera.move(v)` | Translate without changing the pointing direction |
| `camera.look_at(point)` | Change aim without changing location |
| `camera.rotate(deg, axis, about=point)` | Rotate both camera location and orientation around the point |
| `camera.orbit(deg, axis, about=point)` | Move on the orbit without changing the pointing direction |
| `camera.center_on(mob)` | Reframe around the target |
| `camera.set_fov(degrees)` | Change the vertical field of view; supports animation |

`orbit` does **not** mean automatically look at the orbit center. A tracking
updater can make that relationship explicit.

`look_at` computes its rotation from the camera's position when it is recorded,
so `camera.move_to(p)` and `camera.look_at(t)` in one `Sync` do not keep the
target centred during the move. Use `fly_to` for a move whose position and aim
change together:

```python
camera = Scene.get_camera()
with Off():
    camera.fly_to(OUT * 12 + UP * 2, look_at=ORIGIN)       # establish the shot
with Seq(runtime=3):
    camera.fly_to(RIGHT * 4 + OUT * 7, look_at=UP * 0.5, via=RIGHT * 2 + OUT * 10)
```

It recomputes the aim from the interpolated position and target on every frame.
`via` bends the camera path through a waypoint and `look_at_via` bends the
target's path; each is reached halfway through the eased motion. Without
`look_at` the viewing direction is kept. World `UP` defines the horizon, and a
camera that starts tilted levels out over the move rather than jumping. Chain
shots by calling `fly_to` again; the next move starts from where the last ended.
The camera basis rows are `[right, up, forward]`, with `forward` the viewing
direction. Root rotation angles and FOV values are in degrees; trigonometric
functions used inside custom motion still take radians.

For ongoing targeting:

```python
camera = Scene.get_camera()
tracking = camera.add_updater(lambda mob, t: mob.look_at(subject.location))
# Author the desired motion and elapsed time here.
# camera.remove_updater(tracking) when tracking should stop.
```

`set_near_orthographic()` is an approximation achieved through the perspective
camera; do not call it mathematically exact orthographic projection.
`set_near(...)` and `set_far(...)` configure clip distances; they are not
animated timeline controls. Establish them during setup, not as purported
mid-shot keyframes. Do not guess depth-of-field or other camera parameter names;
inspect the installed Camera API when the requested shot needs those features.

Screen-relative placement resolves the camera when recorded. It does not pin a
caption throughout a later camera move. Use camera parenting or a replay-safe
updater as described in the geometry reference. Check projected text and object
bounds at several times along any camera move.

## Renderer mode and production checks

`samples_per_pixel=1` uses the deterministic hybrid renderer; a larger integer
selects the Monte Carlo path tracer. The path tracer uses scene illumination,
material transport, and stochastic sampling, so changing mode can change both
appearance and rendering cost. Set the mode explicitly when the requested
result depends on it.

```python
SETTINGS.raytracing.set(samples_per_pixel=1)
# Or use the established sample count for a path-traced delivery:
# SETTINGS.raytracing.set(samples_per_pixel=64)
```

Do not turn off shadows, reduce required bounces, replace transmission, change
material response, or alter final resolution merely to make the render finish
sooner without telling the user. Drafts can use separate lower-cost settings;
validate required effects in the intended final mode before a long export.
Reuse shared shader function objects and assets where possible, avoid unnecessary
geometry, and render individual Project scenes for iteration. These are authoring
practices, not instructions to patch the renderer or tune experimental kernels.

Source basis: [shading and camera sources](api-sources.md#materials-and-shaders).
