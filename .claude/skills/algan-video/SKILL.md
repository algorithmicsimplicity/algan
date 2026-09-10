---
name: algan-video
description: "Use Algan (a Python 2D/3D animation engine and Manim successor with a ray-traced renderer) to author and render a video from a user's brief. Use when asked to make an animation, explainer video, rendered clip, still frame, or 3D visualisation with Algan. Covers scripting, animating, updaters, text and LaTeX, 3D lighting and materials, custom shaders, audio, output and quality settings."
---

# Making videos with Algan

This skill is a tool manual. The user supplies the creative direction (what is
on screen, when, and why); you translate it into an Algan script and render it.
Do not invent content, pacing or styling the user did not ask for. When the
brief leaves a visual decision open, pick the plainest option that satisfies it
and say what you chose.

Algan is **lazy**: a script *records* animations on a Scene's timeline, and
`Scene.save_video()` renders that recording. Nothing appears unless it was
spawned, and nothing is computed until a render call.

The deeper catalogues live beside this file. Read the one your task touches:

| Task touches | Read |
| --- | --- |
| Mob classes and constructor args, every Mob method, contexts, easings, colours, `Scene` methods | `references/api_reference.md` |
| Cameras, lights, shadows, materials, reflections, glass, environment maps, textures, models, custom shaders | `references/three_d_and_shaders.md` |
| Text, LaTeX, numbers, images, SVG, Manim geometry, plots, audio, narration, multi-scene projects | `references/media_and_audio.md` |
| Output paths, quality presets, `SETTINGS`, transparency, performance, memory, troubleshooting | `references/output_and_performance.md` |

## 1. Running Algan

```bash
pip install algan          # once; see references/output_and_performance.md for extras
algan check                # prints versions, render device, LaTeX/ffmpeg status
python scene.py            # runs the script; writes algan_outputs/<name>.mp4 beside it
algan render scene.py -q HD     # same, forcing a quality preset the script does not set
```

Facts that matter when an agent runs the script rather than a person:

- **Every script must call** `Scene.save_video(...)` or `Scene.save_frame(...)`.
  Running a script that records animations and never renders produces nothing.
- **Do not call `Scene.view()`** in an unattended run. It starts a web viewer and
  blocks until Ctrl-C.
- **Keep all code below `from algan import *`.** A warm render daemon may re-run
  the script in another process; lines above the import execute twice.
- **The first render is slow** (tens of seconds to minutes) while kernels
  compile and cache. Later renders start warm. This is not a hang.
- `Text` needs a font backend and `Tex`/`MathTex` need a LaTeX install. On
  Linux install TeX Live first (`algan check` reports what was found).
- Set `ALGAN_USE_DAEMON=0` in the environment if the daemon misbehaves or you
  need a fully isolated run.

## 2. Recommended workflow

1. **Write `scene.py`** from the brief. Put scene setup in `with Off():`, then
   the animation beats, then one `Scene.save_video("name")`.
2. **Smoke-test cheaply**: `Scene.save_video("name", SMOKE_TEST)` (32x32, 2 fps)
   proves the script runs end to end in seconds.
3. **Check composition with stills**, which you can open with an image reader:
   ```python
   Scene.save_frame("check", at=[0.0, 2.5, 5.0])   # check_0.0.png, check_2.5.png, ...
   ```
   `at` is seconds on the timeline; a negative value counts back from the current
   time. Rendering never alters the Scene, so leave the `save_video` in place.
4. **Preview the motion** at the default preset (`LD`, 864x486 at 15 fps) or
   `PREVIEW`, and iterate.
5. **Final render** once, with a preset for that call only:
   `Scene.save_video("name", HD)`. `RenderResult.output_path` tells you where it
   went.

## 3. The core model

```python
from algan import *

with Off():                                  # instant setup, no animation recorded
    title = Text("Hello", font_size=72).move_to(UP * 2).spawn()
    box = Square(color=BLUE).spawn()

box.move(RIGHT * 2)                          # 1-second animation, then the next starts
box.color = YELLOW                           # assigning an animatable attribute animates it
with Sync(runtime=2):                        # everything inside plays together
    box.rotate(90, OUT)
    title.color = BLUE
Scene.wait(1)                                # hold
box.despawn()                                # fade out; optional

Scene.save_video("hello")                    # -> algan_outputs/hello.mp4
```

- **Mobs** are everything drawable: shapes, text, images, 3D solids, lights,
  the camera. A Mob must be `.spawn()`ed to appear; it fades in over 1 s unless
  spawned inside `Off()` or with `spawn(False)` (no fade, instant).
- **Animatable attributes**: `location`, `basis` (orientation and scale; change
  it through `rotate`/`scale`), `color`, `glow`, `opacity`, plus whatever a
  material or shader adds. Assignment records a 1-second interpolation. Reads
  return a **copy**, so `mob.location[0] = 1` changes nothing; write
  `mob.location = mob.location + RIGHT` or `mob.location += RIGHT`.
- **Colours** carry five channels: r, g, b, glow, opacity. `GREEN.set_glow(0.5)`,
  `RED.set_opacity(0.5)`. `BLUE * 0.5` also halves opacity, which silently makes a
  background transparent. Manim's palette (`RED`, `BLUE_E`, `TEAL_A`, ...),
  `WHITE`, `BLACK`, `TRANSPARENT` and `Color((r, g, b))` are available.
- **Common methods** (each is a 1 s animation unless a context says otherwise):
  `move(delta)`, `move_to(point, arc_angle=None)`, `rotate(deg, axis, about=None)`,
  `orbit(deg, axis, about=p)` (travel without turning), `scale(k)`,
  `look_at(point)`, `wait(s)`, `set(location=..., color=...)` for several
  attributes in one beat, and `become(other)`:
  ```python
  square = square.become(Circle(add_to_scene=False))   # reassign; target never drawn
  ```
- **Directions**: `RIGHT/LEFT` (x), `UP/DOWN` (y), `OUT/IN` (z, `OUT` is toward
  the viewer), `ORIGIN`. Angles are degrees. The default camera sits at
  `OUT * 7`, and the visible area at z=0 is about **12.4 wide by 7 tall**, so x
  spans roughly -6.2..6.2 and y roughly -3.5..3.5.

### Animation contexts

| Context | Behaviour |
| --- | --- |
| `Seq()` | One after another (the default outside any context). |
| `Sync()` | All start together. Same as `Lag(0)`. |
| `Lag(r)` | Next starts when the previous is fraction `r` through. |
| `Off()` | Instant, nothing recorded as motion. Use for setup. |
| `Audio(path)` / `Speech(text)` | Block runtime taken from a sound clip. See `references/media_and_audio.md`. |

All take `runtime=` (the whole block, animations rescaled to fit),
`runtime_per_part=` (each animation inside), and `easing=` (a rate function).
Contexts nest: a nested block counts as one animation for its parent and
inherits parameters it does not set. Wrap a routine in a function to reuse it.

```python
with Sync(runtime=3):
    with Seq():                     # circle's routine, three steps in 3 s total
        circle.move(LEFT * 3)
        circle.color = YELLOW
        circle.move(RIGHT * 3)
    square.rotate(360, OUT)         # plays alongside the whole routine
```

**Before a Mob is spawned its animations are always instant**, whatever the
surrounding context. `Scene.wait(s)` and `mob.wait(s)` insert a pause.

### Easing

Default is `easings.smooth` (ease in and out). Use `easings.identity` for
constant speed (turntables, orbits, clock hands), `easings.ease_out_quintic` for
arrivals, `easings.ease_in_expo`/`ease_out_expo` for sharp acceleration.
`easings.inversed(f)` reverses one. Any function mapping a tensor in [0, 1] to
[0, 1] works. Put `easing=` on the context that owns the motion, not an outer one
that also holds other animations.

## 4. Placing things

Prefer relative placement, then screen-relative, then absolute coordinates:

```python
label.move_next_to(box, DOWN, buffer=0.3)            # edge to edge
label.move_next_to(box, RIGHT, align_edge=DOWN)      # and share a bottom edge
caption.move_to_screen_edge(DOWN)                     # rest against a screen edge
logo.move_to_screen_corner((UP, LEFT))
hud.move_to_screen_position(0.9, 0.1)                 # (0,0) bottom-left, (1,1) top-right
diagram.fit_to_screen((0.0, 0.0), (0.5, 1.0))         # scale + move into the left half
diagram.fit_to_screen()                               # fill the frame
box.move_to(UP * 2 + LEFT * 3); box.y = 0             # absolute; x/y/z set one axis
box.scale_to_height(2.5); circle.scale_to_width(box.get_width())
```

Measurements (`get_width`, `get_height`, `get_center`, `get_bounding_box`) are
read at call time; use an updater if one Mob must keep tracking another.
`SETTINGS.style.buffer` (0.6) is the default gap for the layout methods. Screen
methods resolve the camera **once**, when recorded; to pin something during a
camera move, make it a child of the camera or use an updater.

### Groups and hierarchy

```python
row = Group([Square(color=BLUE).scale(0.3) for _ in range(6)]).spawn()
row.arrange_in_line(RIGHT, buffer=0.2)          # animated, like everything else
row.arrange_in_grid(2, row_buffer=0.5)
row.rotate(180, OUT); row.color = YELLOW        # propagates to every member
row[0].move(UP)                                 # members stay individually animatable
with Lag(0.3):
    for member in row:
        member.color = RED
parent.add_children([child_a, child_b])         # explicit parent/child link
```

A child moves, rotates, scales, recolours, spawns and despawns with its parent,
as if bolted to it; direct changes to the child ignore the parent. `Group` takes
a list or several Mobs, is indexable and iterable, and its centre is the
members' centre.

## 5. Text and mathematics

```python
title = Text("Euler's identity", font_size=64, weight="BOLD",
             color_map={"identity": YELLOW}).spawn()
formula = Tex(r"e^{i\pi}", "+ 1", "= 0", font_size=80).spawn()   # math mode, no $ needed
with Lag(0.5):
    for i in range(len(formula.tex_strings)):
        formula.get_segment(i).color = YELLOW    # per-segment: the pieces you passed
with Lag(0.2):
    for glyph in title.character_mobs:           # per-glyph (spaces excluded)
        glyph.color = BLUE
Text("Hand written", font_size=64).spawn(False).write(runtime=3)   # handwriting effect
counter = DecimalNumber(0.0, decimal_places=2).scale(2).spawn()
with Seq(runtime=3):
    counter.value = 100.0                        # counts up
```

Always use raw strings for LaTeX. `MathTex`, `Title`, `Paragraph`,
`BulletedList`, `MarkupText` also exist (Manim-compatible). Text and formulae are
vector outlines: they morph, scale and take gradients, but are drawn **unlit**.

## 6. Updaters (per-frame rules)

An updater runs every frame from the moment it is added until removed, on top of
whatever the timeline says. Use one for idle motion, following, or any rule whose
duration you do not know in advance.

```python
import torch
spin = triangle.add_updater(lambda mob, t: mob.rotate(t * 180, OUT))   # t = seconds since added
label.add_updater(lambda mob, t: mob.move_next_to(ball, DOWN))         # follow without inheriting orientation
ball.add_updater(lambda mob, t: mob.move(UP * 0.8 * torch.sin(t * 2 * PI)))
Scene.get_camera().add_updater(lambda cam, t: cam.look_at(ball.location))
Scene.wait(3)
triangle.remove_updater(spin)     # Mob keeps whatever state it reached
```

Rules that keep updaters correct:

- The signature is `(mob, t)`; declare `t` even when unused.
- **`t` is a torch tensor of shape `[frames, 1, 1]`**, not a float. Use
  `torch.sin`, `torch.exp`, ... never the `math` module (it fails at render time).
- Write state as a function of `t` from a fixed reference, never by accumulating
  increments; frames are evaluated in parallel batches.
- Updaters win over recorded animations of the same attribute on that frame.

## 7. Built-in and custom animations

Attention: `Indicate(mob)`, `Circumscribe(mob)`, `Flash(mob)`, `FocusOn(mob)`,
`Wiggle(mob)`, `Blink(mob, blinks=2)`, `ShowPassingFlash(outline, runtime=2)`,
`DrawBorderThenFill([a, b], runtime=2)` (spawn with `spawn(False)` first),
`AnimatedBoundary(mob).spawn()` (a Mob; `.stop()` freezes it).
Motion and deformation: `MoveAlongPath(dot, path_mob, runtime=3)`,
`ApplyMatrix(grid, torch.tensor([[1., .6], [0., 1.]]))`,
`ApplyPointwiseFunction`, `ApplyComplexFunction`, `Homotopy(mob, f(x, y, z, t))`,
`PhaseFlow(mob, vector_field, virtual_time=2.0)`, `ApplyWave(text)`. They take
`runtime=` and obey contexts. Their callbacks receive **batched torch tensors**.

A custom fixed-length animation describes one frame as a function of a swept
parameter:

```python
@animated_function(animated_args={"t": 0.0})     # start values of the swept args
def move_along(mob, t):
    mob.location = UP * np.sin(t) + RIGHT * (t - PI)   # one frame; assignments here are not separately animated

with Seq(runtime=3, easing=easings.identity):
    move_along(square, 2 * PI)                     # sweeps t from 0 to 2*pi
```

To place animations at times you compute yourself, name the context and move
its write pointer:

```python
with Seq() as ctx:
    start = ctx.current_time
    for i, mob in enumerate(mobs):
        ctx.current_time = start + delay_for[i]
        with Seq(runtime=1):
            mob.color = RED
    ctx.current_time = ctx.end_time        # resume sequential recording after the block
```

## 8. 3D: camera, lights, materials

3D solids (`Sphere`, `Cube`, `Cylinder`, `Cone`, `Torus`, `Prism`, Platonic
solids, `Surface(uv_function)`, `Model3D`) are lit and cast shadows; flat 2D
shapes and text are not lit.

```python
SETTINGS.raytracing.set(shadows=True)                # shadows are off by default (cost)
with Off():
    Scene.clear_lights()                             # drop the default point light
    DirectionalLight(location=UP * 8 + RIGHT * 4 + OUT * 4, target=ORIGIN,
                     intensity=3, shadow_angle=3).spawn()      # soft sun
    AmbientLight(intensity=0.3).spawn()              # keeps shadow sides from going black
    floor = Prism(width=9, height=0.2, depth=9, color=GREY).move(DOWN * 1.4)
    floor.set_material(MeshStandardMaterial(metalness=0.8, roughness=0.1)).spawn()
    ball = Sphere(radius=0.8, color=BLUE).set_material(
        MeshPhysicalMaterial(transmission=1.0, ior=1.5, roughness=0.0)).spawn()   # glass

camera = Scene.get_camera()
with Seq(runtime=4, easing=easings.identity):
    camera.rotate(360, UP, about=ORIGIN)             # turntable, stays aimed at the origin
camera.set_fov(30)                                   # animatable; dolly-zoom friendly
with Seq(runtime=3):
    ball.roughness = 0.6                             # material properties become animatable attrs
```

- `set_material`, `set_shader`, `set_fragment_shader`, `two_sided`,
  `casts_shadows`, `receives_shadows` must be set **before `spawn()`**.
- Materials are Three.js style: `MeshBasicMaterial` (unlit), `MeshLambertMaterial`,
  `MeshPhongMaterial`, `MeshStandardMaterial(metalness, roughness)`,
  `MeshPhysicalMaterial(+ transmission, ior, clearcoat, sheen)`,
  `MeshToonMaterial`, `MeshNormalMaterial`. Presets: `WOOD`, `GLASS`, `PLASTIC`,
  `RUBBER`, `CERAMIC`, `STONE`, `MIRROR`, `BRUSHED_METAL`, `CHROME`, `COPPER`.
- Lights: `PointLight`, `DirectionalLight`, `AmbientLight`, `HemisphereLight`,
  `SpotLight`, `RectAreaLight`. Spawn them inside `Off()` (each spawn is a 1 s fade
  otherwise). `location`, `color`, `intensity` animate; cone angles, decay and
  emitter sizes are fixed per light.
- **A mirror with nothing to reflect renders black.** Add
  `Scene.set_environment_map("panorama.png")` (equirectangular image) or
  surrounding geometry. **Glass with nothing behind it looks like nothing**: put a
  patterned backdrop behind it. `opacity` is coverage (fade in/out);
  `transmission` is glassiness.
- Camera config that is not animatable: `set_near`, `set_far`,
  `set_near_orthographic()` (parallel projection for diagrams). Set once, before
  spawning.
- Path tracing (global illumination, many lights, soft area shadows):
  `SETTINGS.raytracing.set(samples_per_pixel=16, max_bounces=2)`. Much slower;
  denoised by default. Keep `samples_per_pixel=1` (the deterministic renderer) for
  2D and text work.

Full light parameters, material tables, textures on surfaces and model import
are in `references/three_d_and_shaders.md`.

## 9. Custom shaders

Three levels, from simplest to most control. All are set before `spawn()`.

**Materials** (above) cover metal, plastic, glass, toon. Reach for a shader only
when the look is not a material.

**Fragment shader pipelines** run in the render kernel per pixel hit. Pass one
stage or a list applied left to right; each stage's parameters become animatable
attributes on the Mob:

```python
ball = Sphere(radius=1, color=BLUE_E)
ball.set_fragment_shader([standard_shader, fresnel_rim])   # lit PBR, then an additive rim light
ball.rim_color = (0.4, 0.9, 1.0)      # width-3 tuple; palette constants need [..., :3]
ball.rim_power = 3.0
ball.spawn()
ball.set_fragment_shader([cosine_color, STAGE_PHONG])       # recolour, then light (before spawn)
```

Lighting stages: `STAGE_UNLIT`, `STAGE_LAMBERT`, `STAGE_PHONG`, `STAGE_STANDARD`,
`STAGE_PHYSICAL`, `STAGE_MANIM` (or pass `phong_shader`, `standard_shader`, ...
which resolve to them). Shipped additive looks: `fresnel_rim`, `glass_ball`.
Example recolour stage: `cosine_color` (`frequency`, `phase`).

A **custom stage** is a Taichi `@ti.func` with the fixed stage signature plus a
list of `(name, width, default)` parameter specs:

```python
from algan import *
from algan.taichi_compat import ti

@ti.func
def _stage_bands(pos, view_dir, n_interp, face_n, in_rgb, in_glow,
                 params: ti.template(), f, prim, off,
                 light_pos: ti.template(), light_col: ti.template(), num_lights,
                 shadows: ti.template(), vis, cam_pos):
    tm = f % params.shape[0]
    freq = params[tm, prim, off + 0]          # slot 0 = first spec below
    k = 0.5 + 0.5 * ti.cos(pos[1] * freq)     # bands along world y
    return ti.math.vec4(in_rgb[0] * k, in_rgb[1] * k, in_rgb[2] * k, in_glow)

bands = FragmentStage(_stage_bands, [("frequency", 1, 6.0)])
mob.set_fragment_shader([bands, STAGE_STANDARD])
mob.frequency = 12.0                          # animatable
```

The stage receives the previous stage's colour in `in_rgb`/`in_glow` and returns
`vec4(r, g, b, glow)`. Read parameters from `params[tm, prim, off + slot]`. A
stage can also carry a `scatter=` function to change how rays bounce; copy
`forced_mirror_scatter` in `algan/rendering/shaders/fragment_shaders.py` as the
template. The first render with a new pipeline pays a kernel compile.

**Vertex shaders** (`set_shader`) are plain PyTorch functions evaluated per
vertex with nine fixed parameters followed by your own, which become animatable:

```python
def toon(memory, vertex_location, vertex_normal, albedo_color, camera_location,
         light_origin, light_color, light_intensity, ambient_light_intensity,
         bands=4.0):
    ...                                        # torch ops; return a colour per vertex
mob.set_shader(toon); mob.bands = 6.0
```

They see only a plain point light and never receive shadows, so prefer fragment
pipelines in lit scenes. Define one shader function and reuse it across Mobs;
different shader objects batch separately. Details and the shipped shader sources
to copy from are listed in `references/three_d_and_shaders.md`.

## 10. Images, textures, models, plots

```python
ImageMob("photo.png").scale(2).spawn()                       # flat textured plane; paths resolve beside the script
globe = Sphere(radius=1.5, color_texture=get_checkerboard((RED, WHITE))).spawn()
globe.color_texture = get_stripes((BLUE, WHITE))             # cross-fades texel by texel
Sphere().set_material(MeshStandardMaterial(map="earth.png", roughness_map="gloss.png"))
Model3D("robot.glb", fit_to_size=2.0).spawn()                # glTF/GLB/OBJ/PLY/STL; .fbx needs assimp
SVGMob("logo.svg").scale(2).spawn()
axes = Axes(x_range=(-3, 3, 1), y_range=(-1.5, 1.5, 0.5), x_length=9, y_length=4.5)
graph = axes.plot(lambda x: np.sin(x), color=YELLOW)          # returns a Mob; spawn both
square = Square(grid_width=64, grid_height=64, stroke_width=0)
square.set_color_by_function(lambda uv: torch.cat((uv[..., :1], 1 - uv[..., :1], uv[..., 1:]), -1))
```

Texture maps sample only on `Surface`-based Mobs (`Sphere`, `Cylinder`, `Torus`,
`ImageMob`, ...), not on `Cube` or polyhedra. 2D shapes take a colour grid
(`grid_width`/`grid_height`) for gradients and image fills. Manim geometry
imports via `ManimMob(mn.Mobject)` (import Manim as `import manim as mn`, never
star-import both); `Axes`, `NumberPlane`, `BarChart`, `Table`, `Brace`, `Arrow`,
`Arc`, `Star` and more are available natively with Manim's arguments.

## 11. Backgrounds, glow, post-processing, transparency

```python
Scene.set_background(Color([0.05, 0.05, 0.15]))              # whole Scene; not animatable
Scene.save_video("v", background="backdrop.png")             # this render only
Scene.save_video("v", background=lambda x, y, t: ...)        # procedural, per pixel per frame (torch tensors)
dot.glow = 0.4                                               # bloom makes glow visible; 0.3-0.5 is plenty
Scene.save_video("v", post_processes=())                     # disable bloom
Scene.save_video("v", post_processes=(partial(bloom_filter, strength=8), desaturate))
Scene.save_video("overlay.mov", background=TRANSPARENT)      # alpha output needs .mov (or .webm with codec)
```

A custom pass is `def f(frames, memory=None): ...` returning the frames (the
`memory` keyword is required). Import `bloom_filter` from
`algan.rendering.post_processing.bloom`.

## 12. Audio and narration

```python
with Audio("music.wav"):            # block runtime = clip length
    circle.rotate(360, OUT)
with Speech("Gradient descent follows the slope downhill."):   # synthesised; runtime = spoken length
    title.move(UP)
```

`Speech` needs a system TTS engine (built in on macOS/Windows; `espeak-ng` on
Linux). For recorded narration, align a transcript with
`get_speech_generator_from_file` and install it with
`Scene.current().audio_manager.set_speech_source(...)`. See
`references/media_and_audio.md`.

## 13. Output, quality and settings

```python
Scene.save_video("clip")                     # algan_outputs/clip.mp4, LD (864x486, 15 fps)
Scene.save_video("clip", HD)                 # this render only: 1920x1080, 30 fps
Scene.save_video("renders/final.mp4")        # a path with a directory is used as given
Scene.save_frame("still.png", at=1.5)        # one still
SETTINGS.video.set(HD, fps=60)               # default for every render; set at the top of the script
SETTINGS.paths.set(output_directory="renders")
SETTINGS.computing.set(render_device="cpu")  # or "cuda", "mps", "auto"; top of script
```

Presets: `SMOKE_TEST`, `PREVIEW`, `LD`, `MD` (720p30), `HD` (1080p30),
`PRODUCTION` (1440p60), `UHD` (2160p60), `THUMBNAIL`. Presets are immutable;
`HD.set(fps=24)` returns a copy. Mutate sections with `.set(...)`, never
`SETTINGS.video = HD`. `SETTINGS.video` is read when the Scene is created (the
first Mob creates it), so global changes go at the top of the script; per-render
arguments have no ordering constraint.

Other `save_video` keywords: `reset=True` (tear the Scene down afterwards),
`animate_fade_out=True`, `overwrite`, `codec`, `audio_codec`, `ffmpeg_params`.

Cost scales with pixels x frames x supersampling. Cheapest speedups while
drafting: a smaller preset, `SETTINGS.video.set(ssaa=1, fxaa=True)`, shadows off,
lower `max_bounces` in glassy scenes, `samples_per_pixel=1`. Out-of-memory advice
and the full settings map are in `references/output_and_performance.md`.

## 14. Multi-scene projects

For a longer video, give `Project` zero-argument scene functions; it renders any
subset with stable names and concatenates them. Scene functions do **not** call
`save_video`.

```python
def intro(): Text("Title", font_size=90).spawn(); Scene.wait(2)
def body():  Sphere(color=BLUE).spawn().rotate(360, UP)
project = Project([intro, body], file_path="talk.mp4", video_settings=PREVIEW)
project.render_video(); project.concatenate_videos()       # or project.render_video("body")
if __name__ == "__main__": project.run_cli()                # python talk.py --render-video 1 --video-settings HD
```

## 15. Pitfalls checklist

- Empty video: a Mob was never spawned (`NeverSpawnedMobWarning`), or the
  `spawn()` happened after the animations you expected.
- Everything happens at once or takes no time: you are inside `Off()`, or the Mob
  was not yet spawned when animated.
- Three seconds of darkness at the start: lights spawned outside `Off()`.
- Wrong colours after `become`/morph targets appear on screen: build targets
  with `add_to_scene=False`.
- `ValueError: only one element tensors...` at render time: `math` used on a
  tensor inside an updater, homotopy or surface function. Use `torch`.
- LaTeX fails: missing TeX install, non-raw string, or `$...$` added inside `Tex`.
- Material or shader has no effect: it was set after `spawn()`.
- Metal is black: nothing to reflect. Glass is invisible: nothing behind it.
- Shadows missing: `SETTINGS.raytracing.set(shadows=True)` not set.
- `.mp4` refused: the background is transparent; use `.mov`.
- Unexpectedly transparent output: a colour was scaled (`BLUE * 0.2`), scaling its
  alpha. Use `Color([...])` or `.set_opacity(1.0)`.
- `SETTINGS.video.set(...)` ignored: it ran after the first Mob created the Scene.
- `Scene.view()` hangs the run: it is interactive; use `save_frame` instead.
