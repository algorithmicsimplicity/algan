# FAQ: General Usage

## Why is my video empty?

Mobs must be spawned before they appear. Constructing `Square()` defines the
object; `Square().spawn()` puts it on the Scene's timeline. Algan warns about
an unspawned Mob with `NeverSpawnedMobWarning`.

## Why is my first render slow?

The first render can include library startup and compilation of renderer kernel
variants. Algan uses Quadrants by default, with kernels written in the Taichi
language. Disk caches and the warm render daemon reduce repeat work, but a new
variant, compiler configuration or source edit may still require compilation.
The render itself also takes time; not every slow first run is a cache issue.

For a watched development script, use the installed environment's interpreter:

```bash
python -m algan.daemon my_scene.py --watch
```

See {doc}`../advanced_user_tutorials/the_render_daemon`.

## Do I need a dedicated GPU?

No. Automatic selection uses a supported CUDA GPU or Apple Silicon MPS device
when available, otherwise CPU. The renderers target the same visual result,
but floating-point differences mean device outputs need not be byte-identical.
Speed depends on the scene, backend and hardware. Set
`SETTINGS.computing.set(render_device="cpu")` before rendering, or use the
`ALGAN_RENDER_DEVICE` environment variable to seed that setting.

## Can I keep animating after calling `save_video()`?

Yes. By default, `Scene.save_video()` preserves spawned Mobs and timeline
history, so you can append animations or render again. Pass `reset=True` to
reset the Scene after rendering. An explicitly requested fade-out can append
animation, and a zero-duration video may need a one-frame timeline guard;
rendering is not a promise that no recording operation can occur.

## Why didn't my change to `SETTINGS.video` do anything?

`SETTINGS.video` seeds each Scene when it is created. A Mob can trigger creation
of the default Scene, but an earlier explicit Scene access can do so too. Set
defaults before creating the Scene, use `Scene.set_video_settings(...)` to
change the active Scene, or pass a one-off preset to a render call:

```python
Scene.save_video("my_video", HD)
```

## What changes when `samples_per_pixel` is greater than 1?

A value of 1 selects the deterministic hybrid raster/ray tracer. A value greater
than 1 selects the Monte Carlo path tracer; it is an explicit choice, not an
automatic fallback after a memory failure. The path tracer supports refractive
materials, environment maps, authored/custom fragment pipelines and homogeneous
scattering inside supported closed solids. Custom appearance stages and physical
BSDF materials do not necessarily have the same lighting semantics.

Homogeneous scattering and random-walk subsurface scattering require the path
tracer. Unsupported combinations are reported according to the configured
unsupported-feature policy. Increasing the sample count does not eliminate
ordinary geometry, configuration or memory errors. See {ref}`renderer-capabilities`.

## Why is my LaTeX not compiling?

Algan uses a local LaTeX installation (such as TeX Live, MiKTeX or MacTeX) on
`PATH`. Use raw strings (`r"..."`) so Python does not interpret backslashes.
{class}`~algan.mobs.text.Tex` runs in math mode; do not add a second pair of
`$...$` delimiters. See {doc}`../advanced_user_tutorials/text_and_math`.

## Why won't my transparent video play?

The container, codec and player must all support alpha. A `.mov` export is a
useful starting point: ordinary transparent output uses PNG frames by default,
while premultiplied-over output has its own encoder requirements. A standard
MP4/H.264 export is not an alpha-preserving substitute. See
{doc}`../advanced_user_tutorials/transparent_backgrounds`.

## Where are my rendered files saved?

By default, output goes under `algan_outputs/` beside the scene script. A bare
filename such as `"my_scene"` uses the configured output directory; a path with
a directory component, such as `"renders/test.mp4"`, is used as given.
`Scene.save_video()` returns a `RenderResult`; its `output_path` gives the
resolved location.
