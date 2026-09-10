# Text, images, external assets, audio and multi-scene projects

## Text

```python
title = Text("Euler's identity", font_size=64).move(UP * 1.5).spawn()
Text("bold", font_size=44, weight="BOLD"); Text("italic", slant="ITALIC")
Text("colored words", color_map={"colored": YELLOW})        # also font_map, slant_map, weight_map
Text("two\nlines", line_spacing=1.2, font="Times New Roman", gradient=(BLUE, GREEN))
```

`font_size` 48 by default (100 fills most of the frame); colour defaults to
`SETTINGS.style.text_color` (white). Pick either `font_size` or `scale` for a
label and animate with `scale`. Glyph outlines are Bezier circuits: they morph,
scale without pixelation and take colour grids, but are never lit.

Per glyph: `text.character_mobs` (visible glyphs only, no spaces). Handwriting:
`Text(...).spawn(False).write(runtime=3, lag_ratio=...)`; `spawn(False)` avoids
a fade-in before the writing. `write` does not change spawned state.

Linux needs a text backend: LaTeX (TeX Live) covers `Text` through LaTeX text
mode (ignoring font/weight/slant), and `pip install "algan[pango]"` restores
system fonts. macOS and Windows work out of the box.

## LaTeX

```python
formula = Tex(r"\frac{d}{dx}\left(x^2\right) = 2x", font_size=60).spawn()   # math mode
parts = Tex("e^{i\\pi}", "+ 1", "= 0", font_size=90).spawn()               # one segment per string
with Lag(0.5):
    for i in range(len(parts.tex_strings)):
        parts.get_segment(i).color = YELLOW
parts[3].color = RED                                    # one glyph
```

Raw strings always. `Tex` is already in math mode; never add `$`. Prose inside
a formula goes in `\text{...}`. `MathTex` is the Manim-compatible spelling.
Requires TeX Live, MiKTeX or MacTeX on `PATH` (needs `standalone`, `babel`,
`amsmath`, `amssymb`, `latex`, `dvisvgm`). Compiled glyphs are cached. Segments
are not `children` (a multi-part `Tex` has one child), so loop over
`get_segment`, not `children`.

## Numbers

```python
counter = DecimalNumber(0.0, decimal_places=2, integer_places=3).scale(2).spawn()
with Seq(runtime=3):
    counter.value = 100.0
```

`Integer`, `Variable`, `Matrix`, `Table` and friends follow Manim's arguments.

## Images

- `ImageMob("photo.png")` or an `[H, W, 4]`/`[H, W, 5]` tensor: a flat textured
  `Surface`. Paths resolve against the working directory, then the script's
  directory (same for `set_background`, `set_environment_map`, `Model3D`).
- `world.set_shape_to(Sphere(radius=2, add_to_scene=False))` wraps the image
  onto a globe; the texture follows the UVs.
- Image on a 2D shape: `Circle(grid_width=128, grid_height=128, stroke_width=0).set_color_by_image("photo.png")`.
- Gradient on a 2D shape: `set_color_by_function(lambda uv: ...)` with `uv[..., 2]`
  in [0, 1] (`u` left to right, `v` top to bottom over the circumscribing
  square). Per glyph on text by looping `character_mobs`. On a `Line` the
  function receives `t` from start to end.
- Background image: `Scene.save_video(background="backdrop.png")`.
- Procedural surface textures: `get_checkerboard`, `get_stripes`,
  `get_grid_lines`, `get_polka_dots`, `get_bricks`, `get_gradient`,
  `get_radial_gradient`, `get_noise`.

## SVG

`SVGMob("logo.svg").scale(2).spawn()`: path geometry only (convert text to
outlines first; raster images, filters and gradients are dropped). Paths become
children reachable through `.children`; the Mob is not indexable. The path is
resolved relative to the working directory, so prefer an absolute path.

## Manim geometry

```python
import manim as mn                                  # never star-import both libraries
plane = ManimMob(mn.ComplexPlane().add_coordinates()).spawn()
diagram = mn.VGroup(); diagram.add(mn.Axes(...)); plot = ManimMob(diagram).spawn()
ball = ManimMob(mn.Sphere(resolution=(16, 8)))       # Manim 3D arrives as real geometry
```

Manim animations (`Create`, `Transform`, `FadeIn`) do not exist here; animate
the imported Mob with Algan's methods. Use Manim colours inside Manim code and
Algan colours after wrapping. Manim's frame is 8 units tall against Algan's
~7, so imported diagrams usually want `scale` or `fit_to_screen`, or call
`Scene.use_manim_defaults()` first to match Manim's framing exactly.

Native Manim-argument classes need no wrapping: `Axes`, `ThreeDAxes`,
`NumberPlane`, `ComplexPlane`, `PolarPlane`, `NumberLine`, `BarChart`, `Table`,
`Graph`, `Brace`, `Arc`, `Annulus`, `Ellipse`, `Star`, `Arrow`, `Vector`,
`MathTex`, `Title`, `BulletedList`, `Code`, ... Their helper methods
(`axes.plot`, `axes.get_axis_labels`, `axes.c2p`) return Mobs to spawn.
`SETTINGS.style.set(shape_style_profile="manim")` makes built-in shapes adopt
Manim's constructor defaults (unfilled white outlines).

## Audio

```python
with Audio("music.wav"):                 # block runtime = clip length; wait_at_end=0
    circle.rotate(360, OUT)
    circle.scale(2)

from moviepy import AudioFileClip
with Audio(AudioFileClip("music.mp3").subclipped(10, 20)):
    mob.move(RIGHT)
```

Audio contexts nest with the others; a sound effect inside a narration segment:

```python
with Speech("The object now transforms into a triangle."):
    with Sync():
        with Audio("whoosh.wav"):
            pass
        mob = mob.become(Triangle(add_to_scene=False))
```

When a video has audio, `save_video` also writes `<stem>_script.txt` with the
transcript. `save_video(audio_codec=..., ffmpeg_params=[...])` overrides encoding.

## Narration

`Speech("text")` synthesises the line and sizes the block to it
(`wait_at_end=1` second by default). The default generator uses `pyttsx3`, which
drives the system engine: built in on macOS and Windows, `espeak-ng` on Linux
(`sudo apt install espeak-ng`); without one, `Speech` raises at synthesis.

Recorded narration: align a transcript to the recording and install the
generator on the Scene, then use `Speech` with each segment's exact text:

```python
from algan.utils.audio_utils import get_speech_generator_from_file   # needs pip install "algan[audio]"
generator = get_speech_generator_from_file(audio_file="narration.wav", transcript_file="narration.txt")
Scene.current().audio_manager.set_speech_source(generator)
with Speech("First we draw a circle."):
    diagram.scale(1.5)
```

Any callable `script -> moviepy audio clip` works as a generator (for a cloud
TTS, or pre-cut clips). The generator is per Scene. `Project(speech_source=...)`
installs one for every scene.

## Multi-scene projects

```python
from algan import *

def intro():
    Text("Gradient Descent", font_size=90).spawn(); Scene.wait(2)

def the_loss_surface():
    Sphere(color=BLUE).spawn().rotate(360, UP)

project = Project([intro, the_loss_surface], file_path="gradient_descent.mp4",
                  video_settings=PREVIEW, video_directory="clips",
                  screenshot_directory="stills", speech_source=None)
project.render_video()                       # all; or 1, "the_loss_surface", [0, "intro"]
project.render_video(video_settings=HD)      # final pass
project.render_screenshots("the_loss_surface", frames="perturbed", stop_early=True)
project.concatenate_videos()                 # stitch into file_path

if __name__ == "__main__":
    project.run_cli()
```

```bash
python video.py --render-video                    # everything
python video.py --render-video 1 outro            # by id or name
python video.py --render-screenshots --frames "s01_*" --stop-early
python video.py --concatenate-videos
python video.py --render-video --video-settings HD
```

Scene functions take no arguments and must not call `save_video`. Output stems
are `<index>_<name>` so a subset render keeps stable names. `render_video`
skips `save_frame` calls; `render_screenshots` runs only them.
`@algan_scene` marks a function as a scene entry point for
`render_all_funcs`.
