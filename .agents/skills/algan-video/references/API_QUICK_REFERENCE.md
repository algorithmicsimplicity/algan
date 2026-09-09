# Algan API quick reference

Use this as a compact authoring reference. The current checkout remains authoritative.

## Scene lifecycle and output

```python
from algan import *

Circle().spawn()
Scene.wait(1)
Scene.save_video("example")
```

A Mob must be spawned to appear. `spawn()` normally fades in; `despawn()` normally fades out.

Primary output methods:

```python
Scene.save_video("name")
Scene.save_video("name", HD)
Scene.save_frame("shot")
Scene.save_frame("shot.png", at=1.5)
Scene.save_frame("shot.png", at=[0.5, 1.0, 1.5])
```

A bare output name goes under Algan's output directory. A path containing a directory is used as given.

Built-in video presets include:

- `SMOKE_TEST`: minimal pipeline check
- `PREVIEW`: fast composition/iteration
- `LD`: low quality/default
- `MD`: 720p-class
- `HD`: 1920x1080 at 30 fps
- `PRODUCTION`: 2560x1440 at 60 fps
- `UHD`: 3840x2160 at 60 fps
- `THUMBNAIL`: still-oriented preset

Set the default without replacing the settings object:

```python
SETTINGS.video.set(HD)
```

Build a modified preset immutably:

```python
HD_60 = HD.set(frames_per_second=60)
```

## Timeline contexts

Outside an explicit context, animations are sequential and typically one second each.

```python
with Off():
    # instantaneous setup
    ...

with Sync(runtime=2):
    # all children start together
    ...

with Seq(runtime=3):
    # children run one after another and the block is rescaled to 3 seconds
    ...

with Lag(0.3):
    # each child starts 30% of the way through the previous child
    ...
```

Useful timing/easing options:

```python
with Seq(runtime_per_part=0.5):
    ...

with Sync(runtime=4, easing=easings.identity):
    ...
```

`easings.smooth` is the normal default. `easings.identity` is useful for constant-speed camera or object orbits.

## Core Mob operations

```python
mob.move(RIGHT * 2)
mob.move_to(ORIGIN)
mob.rotate(90, OUT)
mob.rotate(180, UP, about=ORIGIN)
mob.scale(1.5)
mob.color = YELLOW
mob.opacity = 0.5
mob.glow = 0.3
mob.wait(0.5)
mob.despawn()
```

Common spatial constants include `ORIGIN`, `UP`, `DOWN`, `LEFT`, `RIGHT`, `OUT`, and `IN`.

Animatable attributes record assignments. Do not mutate a returned attribute tensor in place and expect it to be recorded.

Morphing:

```python
mob = mob.become(Triangle(add_to_scene=False))
```

Always assign the return value back.

## Groups and layout

```python
items = Group([
    Text("one"),
    Text("two"),
    Text("three"),
])
items.arrange_in_line(DOWN, buffer=0.35).move_to(ORIGIN).spawn()
```

Use Groups for layout and shared transforms rather than manually maintaining many unrelated coordinates.

## Text and formulas

```python
title = Text("Gradient descent", font_size=72).spawn()
formula = Tex(r"\nabla_\theta L(\theta)", font_size=64).spawn()
```

For independently animated formula parts:

```python
formula = Tex(r"L(", r"\theta", r")", font_size=72).spawn()
formula.get_segment(1).color = YELLOW
```

Visible text glyphs are available as `character_mobs` for staggered per-glyph animation.

## Camera

```python
camera = Scene.get_camera()

with Sync(runtime=3, easing=easings.identity):
    camera.rotate(180, UP, about=ORIGIN)
```

Useful camera methods include `look_at(...)` and `center_on(...)`.

## 3D materials

Set materials before spawning:

```python
sphere = Sphere(radius=1, color=BLUE).set_material(
    MeshStandardMaterial(
        metalness=0.1,
        roughness=0.3,
    )
)
sphere.spawn()
```

Physical effects use Three.js-style material classes such as `MeshStandardMaterial` and `MeshPhysicalMaterial`.

Material properties become animatable after a material is attached, for example:

```python
sphere.roughness = 0.8
```

Do not use obsolete ad-hoc reflectivity/refractive-index setters.

## Lighting and shadows

A Scene starts with a default white point light.

```python
light = Scene.get_light_sources()[0]
light.orbit(360, OUT, about=ORIGIN)
```

For explicit lighting:

```python
with Off():
    Scene.clear_lights()
    DirectionalLight(
        location=UP * 8 + RIGHT * 4 + OUT * 4,
        target=ORIGIN,
        color=WHITE,
        intensity=3,
    ).spawn()
    AmbientLight(color=WHITE, intensity=0.3).spawn()
```

Ray-traced shadows are disabled by default:

```python
SETTINGS.raytracing.set(shadows=True)
```

## Audio and narration

Tie animation duration to an audio clip:

```python
with Audio("music.wav"):
    mob.rotate(360, OUT)
```

Use narration timing:

```python
with Speech("Now we compare the two cases."):
    left.color = BLUE
    right.color = YELLOW
```

Recorded/custom speech sources are Scene-local:

```python
Scene.current().audio_manager.set_speech_source(generator)
```

## Multi-scene projects

Use one zero-argument function per scene:

```python
from algan import *

def intro():
    Text("Title", font_size=90).spawn()
    Scene.wait(2)

def body():
    Circle(color=BLUE).spawn()
    Scene.wait(1)

project = Project(
    [intro, body],
    file_path="video.mp4",
)

if __name__ == "__main__":
    project.run_cli()
```

Do not call `Scene.save_video()` inside functions managed by `Project`.

Useful project operations:

```python
project.render_video()
project.render_video(1)
project.render_video("body")
project.concatenate_videos()
```

## Viewer

For interactive local inspection:

```python
Scene.view()
```

The viewer is Scene-owned; there is no module-level `view`.

## Manim compatibility

Algan has a Manim compatibility layer under:

```python
import algan.manim as mn
```

Use it when porting or when a specific Manim-compatible geometry is needed. Prefer Algan's own animation, Scene, settings, materials, and rendering APIs for new work.

## Current documentation to consult

Inside an Algan checkout, useful authoring references include:

- `docs/source/new_user_tutorials/getting_started.rst`
- `docs/source/new_user_tutorials/basic_animations.rst`
- `docs/source/new_user_tutorials/combining_animations.rst`
- `docs/source/new_user_tutorials/three_d_basics.rst`
- `docs/source/advanced_user_tutorials/text_and_math.rst`
- `docs/source/advanced_user_tutorials/positioning_and_layout.rst`
- `docs/source/advanced_user_tutorials/cameras.rst`
- `docs/source/advanced_user_tutorials/lighting_and_shadows.rst`
- `docs/source/advanced_user_tutorials/shaders_and_materials.rst`
- `docs/source/advanced_user_tutorials/reflections_and_glass.rst`
- `docs/source/advanced_user_tutorials/audio_and_speech.rst`
- `docs/source/advanced_user_tutorials/saving_videos_and_images.rst`
- `docs/source/advanced_user_tutorials/multi_scene_projects.rst`
- `docs/source/advanced_user_tutorials/performance_and_quality.rst`
