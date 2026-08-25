# FAQ: General Usage

## Why is my video empty?

Mobs must be spawned before they show up on screen. Constructing a `Square()` defines the object, but `Square().spawn()` is what actually puts it on screen and makes it animatable on the timeline. If you create a Mob without spawning it, Algan will warn you with a `NeverSpawnedMobWarning`.

## Why did my first render take a couple of minutes?

Almost none of that time is spent rendering your actual animation. On the very first run, Algan compiles its GPU raytracing kernels in Taichi. Those kernels are cached to disk, so every subsequent render starts right away.

To avoid Python startup overhead during development, use the render daemon with `--watch`:

```bash
uv run python -m algan.daemon my_scene.py --watch
```

See {doc}`../advanced_user_tutorials/the_render_daemon`.

## Do I need a dedicated GPU?

No. Algan will automatically detect and use CUDA (NVIDIA) or MPS (Apple Silicon). If no compatible GPU is found, it falls back to your CPU. CPU renders produce identical visual output, just at a slower rendering speed. You can override device selection with the `ALGAN_RENDER_DEVICE` environment variable.

## Can I keep animating after calling `save_video()`?

Yes! Calling `Scene.save_video()` does not destroy your scene. Mobs stay spawned and the timeline keeps its history, so you can continue adding animations or re-render at a different quality preset. If you want the old behavior that resets everything after rendering, pass `reset=True`.

## Why didn't my change to `SETTINGS.video` do anything?

`SETTINGS.video` is read when a `Scene` is first created, and Algan creates its default Scene as soon as your first Mob is instantiated. Make sure you set your settings at the very top of your script before creating any Mobs, or pass the preset directly into your render call:

```python
Scene.save_video("my_video", HD)
```

## Why did setting `samples_per_pixel > 1` raise an error?

Setting `samples_per_pixel` higher than 1 switches from the deterministic wavefront raytracer to the stochastic Monte Carlo path tracer. The path tracer doesn't support refractive materials, environment maps, or custom fragment pipelines. Instead of silently dropping those features, Algan raises an error to alert you. See {ref}`renderer-capabilities`.

## Why is my LaTeX not compiling?

Algan calls out to a local LaTeX installation (TeX Live, MiKTeX, or MacTeX) on your system `PATH`. Make sure to use raw strings (`r"..."`) in Python so backslashes aren't escaped, and remember that {class}`~algan.mobs.text.Tex` already runs in math mode (so you don't need `$...$`). See {doc}`../advanced_user_tutorials/text_and_math`.

## Why won't my transparent video play?

Make sure you render with a `.mov` container extension. Algan's default transparent video codec uses PNG frames, which is supported in QuickTime `.mov` containers but not in standard `.mp4` files. See {doc}`../advanced_user_tutorials/transparent_backgrounds`.

## Where are my rendered files saved?

By default, Algan creates an `algan_outputs/` folder in the same directory as your Python script. If you provide a bare filename like `"my_scene"`, it saves there. If you provide a path with folders (like `"renders/test.mp4"`), Algan respects that path directly. `Scene.save_video()` returns a `RenderResult` object whose `output_path` property tells you the exact file location.
