# FAQ: General Usage

## Why is nothing in my video?

Mobs have to be spawned. Constructing a `Square()` records nothing on its own —
`Square().spawn()` is what puts it on screen and makes it animatable. Algan warns
about this (`NeverSpawnedMobWarning`) when a Mob is never spawned.

## Why did my first render take minutes?

Almost none of that is your animation. Every fresh `python my_scene.py` re-imports
the library and compiles Taichi kernels before it can draw a pixel. Compiled kernels
are cached, so later runs are much faster, and the render daemon keeps a warm process
alive so you only pay it once:

```bash
uv run python -m algan.daemon my_scene.py --watch
```

See {doc}`../new_user_tutorials/getting_started`.

## Do I need a GPU?

No. Algan picks CUDA, then MPS, then CPU. CPU renders work and are just slower. Set
`ALGAN_RENDER_DEVICE` before `import algan` to override the choice.

## Can I render again after calling `save_video()`?

Yes. `save_video()` leaves the Scene exactly as you authored it: Mobs stay spawned,
the timeline keeps its recording, and you can render again — including a preview from
inside a `with` block that has not finished yet. Pass `reset=True` if you want the old
destructive behaviour.

## My change to `SETTINGS.video` had no effect

`SETTINGS.video` is read when a Scene is *constructed*, and Algan creates its default
Scene as soon as you build your first Mob. Set it at the top of your script, before any
Mob exists, or pass the settings to the render call instead:

```python
Scene.save_video("out", HD)
```

## Why does raising `samples_per_pixel` make my scene raise an error?

Because it changes renderer. Above 1 you get the Monte Carlo path tracer, which does
not implement environment maps, refractive materials, custom fragment-shader pipelines
or extended lights. Algan refuses rather than silently dropping them. See
{ref}`renderer-capabilities`.

## LaTeX fails to compile

Algan shells out to a real TeX installation — TeX Live, MiKTeX or MacTeX — so one has
to be on your `PATH`. Use raw strings (`r"..."`) so Python does not eat the backslashes,
and note that {class}`~algan.mobs.text.Tex` compiles in math mode, so `$` is never
needed. See {doc}`../advanced_user_tutorials/text_and_math`.

## My transparent video will not play

Use a `.mov` path. Algan's default codec for transparent output is `png`, which cannot
go in an MP4 (Algan rejects that outright) or a WebM (which needs its codec stated
explicitly). See {doc}`../advanced_user_tutorials/transparent_backgrounds`.

## Where did my output go?

`algan_outputs/` next to your script, unless you said otherwise. A bare filename goes to
the output directory; anything with a directory in it is used as written. `save_video()`
returns a result object whose `output_path` tells you exactly where the file landed.
