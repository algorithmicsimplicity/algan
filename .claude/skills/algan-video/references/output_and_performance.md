# Output, quality, settings, performance and troubleshooting

## Installation and environment checks

```bash
pip install algan                         # Python 3.10-3.13; Linux x86-64, macOS Apple Silicon, Windows
pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install algan   # smaller CPU-only Linux install
algan check                               # versions, render device, LaTeX, ffmpeg, cache and output paths
algan new my_scene.py                     # scaffold a script
algan render scene.py [-q HD] [-o dir_or_file] [--no-daemon]
algan preview scene.py                    # PREVIEW preset
python -m algan.daemon scene.py --watch   # warm process that re-renders on save (interactive use)
```

Extras: `algan[pango]` (system fonts on Linux, compiles from source),
`algan[audio]` (transcript alignment), `algan[fbx]` (+ native assimp),
`algan[taichi]`. LaTeX: TeX Live (`texlive-latex-base texlive-latex-extra
texlive-fonts-recommended latexmk`), MiKTeX or MacTeX. Speech on Linux:
`espeak-ng`. A CUDA GPU (or Apple MPS) is used automatically; on Windows
install the CUDA PyTorch build from pytorch.org first.

The render daemon: `python scene.py` hands the script to a warm background
process if one exists (state file `~/.algan/daemon.json`, log
`~/.algan/daemon.log`). Consequences: code above `import algan` runs twice,
`stdin` is empty, `atexit` handlers do not run, and concurrent scripts queue.
`ALGAN_USE_DAEMON=0` disables the handoff; `ALGAN_AUTO_DAEMON=0` stops
auto-starting one; `algan daemon quit` stops a running one. A daemon refuses
scripts that set initialization-only variables to different values and runs
them cold instead.

## Where output goes

```python
Scene.save_video("my_video")             # algan_outputs/my_video.mp4 next to the script
Scene.save_video("my_video.mov")         # algan_outputs/my_video.mov
Scene.save_video("renders/final.mp4")    # renders/final.mp4, as written
Scene.save_video("/abs/path.mp4")        # exactly there
Scene.save_video()                       # SETTINGS.paths.output_filename (defaults derive from the script)
Scene.save_frame("shot")                 # shot.png at the current time
Scene.save_frame("shot.jpg", at=-0.5)    # half a second before the current time
Scene.save_frame("shot.png", at=[0.5, 1, 1.5])   # shot_0.5.png, shot_1.png, shot_1.5.png; returns a list
SETTINGS.paths.set(output_directory="renders")   # bare names land here; output_root is the base
```

`RenderResult`: `status` (`"rendered"`/`"skipped"`), `output_path`,
`walltime_seconds`, `render_plan` (which renderer ran, features requested,
`path_samples_mean`). `overwrite=False` skips an existing file.

Rendering never changes the Scene by default: render mid-script, keep
animating, render again. `save_video(reset=True)` tears the Scene down
afterwards (old Mobs become unusable). `animate_fade_out=True` records a
fade-out of everything at the end (also `SETTINGS.style.fade_out_on_scene_end`).

## Quality presets

| Preset | Resolution | fps | Use |
| --- | --- | --- | --- |
| `SMOKE_TEST` | 32x32 | 2 | Does the script run at all. |
| `PREVIEW` | 704x396 | 10 | Fastest usable look. |
| `LD` | 864x486 | 15 | Default; timing and composition. |
| `MD` | 1280x720 | 30 | Detail check; publishable. |
| `HD` | 1920x1080 | 30 | Normal final. |
| `PRODUCTION` | 2560x1440 | 60 | High-motion final. |
| `UHD` | 3840x2160 | 60 | 4K. |
| `THUMBNAIL` | 1280x720 | 1 | One still. |

Pass a preset to `save_video`/`save_frame` for that render only, or
`SETTINGS.video.set(HD)` at the top of the script for every render. Presets are
immutable; `HD.set(fps=24, ssaa=1)` returns a copy; `VideoSettings(resolution=(1080, 1920), frames_per_second=30)` builds one (portrait here). `SETTINGS.video`
is read when the Scene is created, which the first Mob does, so global changes
must precede any Mob (or use `Scene.set_video_settings(...)`).

## Settings

Sections `SETTINGS.video`, `.style`, `.paths`, `.computing`, `.raytracing`
(`.raytracing.experimental` for kernel switches). Mutate with `.set(...)` or
field assignment; never assign a section. Temporary:

```python
with SETTINGS.video.override(fps=12): Scene.save_video("draft")
with SETTINGS.override(video={"resolution": (640, 360)}, raytracing={"samples_per_pixel": 1}): ...
snap = SETTINGS.snapshot(); ...; SETTINGS.restore(snap)
```

Style defaults (set once at the top): `background` (BLACK), `frame` (letterbox
colour), `text_color` (WHITE), `buffer` (0.6), `fade_out_on_scene_end`,
`default_material`, `shape_style_profile` ("algan"/"manim"),
`border_placement` ("inward"/"centered").

Device: `SETTINGS.computing.set(render_device="cpu" | "cuda" | "cuda:1" | "mps" | "auto")`
at the top of the script (seeded by `ALGAN_RENDER_DEVICE`). Changing it across
the CPU/GPU line recompiles kernels; it is refused mid-render or once a textured
Mob exists. `ALGAN_ANIMATION_DEVICE` (default `cpu`) must be set in the
environment before `import algan`. `SETTINGS.computing.torch_compile` ("auto")
fuses PyTorch arithmetic; the first render pays for it.

Renderer: `SETTINGS.raytracing.set(samples_per_pixel=1, max_bounces=8, shadows=False, denoise=True, analytic_aa=True, texture_antialiasing=True, glossy_reflection=False, tonemapping=False, tonemap_method="agx", tonemap_exposure=1.0, unsupported_feature_policy="error")`
are the defaults. Turn `tonemapping` on for HDR lighting (bright highlights,
strong glow, environment maps).

Anti-aliasing: `SETTINGS.video.supersampling` (`ssaa`, default 2, costs its
square), analytic edge coverage (on), `SETTINGS.video.fxaa` (off; cheap
post-pass for drafts at `ssaa=1`).

## Backgrounds and post-processing

```python
Scene.set_background(Color([0.05, 0.05, 0.15]))          # solid; (BLUE * 0.15).set_opacity(1.0) to keep alpha at 1
Scene.save_video("v", background="backdrop.png")         # image scaled to the frame
def sunset(x, y, t):                                     # procedural: x, y in [0,1], t seconds, broadcastable tensors
    base = torch.zeros_like(t + y + x)
    return torch.cat([base + 0.35 * y, base + 0.1 * y, base + 0.3 * (1 - y), base, base + 1.0], -1)
Scene.save_video("v", background=sunset)                 # treated as opaque
```

Post-processing defaults to bloom (what makes `glow` visible). Customise with
`functools.partial(bloom_filter, strength=8, glow_spread=0.015, tail_weight=0.15, kernel_size=..., scale_factor=...)`
from `algan.rendering.post_processing.bloom`; `post_processes=()` disables all.
A custom pass is `def f(frames, memory=None): return frames` (torch tensor on the
render device, `[..., 5]` channels; the `memory` keyword is mandatory).

## Transparent output

```python
scene.save_video("overlay.mov", background=TRANSPARENT)                 # PNG frames in .mov
scene.save_video("overlay", background=RED.set_opacity(0.5))            # extension chosen: .mov
scene.save_video("overlay.webm", background=TRANSPARENT, codec="libvpx-vp9", ffmpeg_params=["-pix_fmt", "yuva420p"])
```

`.mp4` is refused for transparent output. A bare `.webm` without the codec
writes an unplayable file. Procedural backgrounds are always opaque. Transparency
is decided by the background colour's alpha, so scaled colours (`BLUE * 0.5`)
turn output transparent by accident. `Scene.set_premultiplied_over()` exports
coverage and additive glow in one linear premultiplied clip (ProRes 4444 for
`.mov`) for compositing tools; ordinary players show it wrongly.

Encoder: `libx264 -crf 17 -preset slower` by default, `h264_nvenc` when an
NVIDIA driver exposes it; pin with `ALGAN_VIDEO_ENCODER=software|nvenc|auto`.
`codec=`, `audio_codec=`, `ffmpeg_params=` pass through to FFmpeg.

## Performance

Cost, roughly in order: how much of the frame curved surfaces fill
(`render_tolerance_pixels`), resolution x supersampling x frame count,
refraction (splits rays), shadows x lights, triangle count, glow/bloom,
distinct shaders. Drafting checklist:

1. Work at `LD`/`PREVIEW`; render `HD` once at the end.
2. `SETTINGS.video.set(ssaa=1, fxaa=True)` for drafts.
3. Shadows off until the shot is blocked; lower `max_bounces` in glassy scenes.
4. Keep `samples_per_pixel=1` unless global illumination, dozens of lights, or
   memory-exhausting reflective geometry force the path tracer; then
   `samples_per_pixel=16, max_bounces=2`.
5. Hundreds of identical shapes: `Sphere.from_batches(centers, ...)` rather than
   hundreds of Mobs.
6. Reuse one shader function across Mobs.

Out of memory (`OutOfRenderMemory`): raise `render_tolerance_pixels` on close-up
surfaces, lower `ssaa` to 1, drop to a smaller preset, reduce geometry or
texture resolution, or raise `SETTINGS.computing.set(rendering_memory_fraction=0.6)`
if the device has room. Moving the camera further away costs nothing extra.

Caches live under `~/.algan/cache`: compiled kernels (minutes cold, instant
after), LaTeX and font glyphs, tessellations, audio. `clear_cache()` keeps the
kernels; `clear_cached_kernels()` drops everything.

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| Video is empty / `NeverSpawnedMobWarning` | Mob never `spawn()`ed. |
| Script finishes with no output | No `save_video`/`save_frame` call. |
| First run takes minutes | Kernel compilation; cached afterwards. |
| Animations happen instantly | Inside `Off()`, or the Mob was not yet spawned. |
| Video opens dark for seconds | Lights spawned outside `Off()`. |
| `ValueError: only one element tensors can be converted to Python scalars` | `math` on a tensor in an updater/surface/homotopy; use `torch`. |
| `TypeError` mid-render from a post-process | Pass lacks the `memory=None` keyword. |
| `Tex` fails | No LaTeX on `PATH`, non-raw string, or `$` added. |
| `Text` draws nothing on Linux | Install TeX Live or `algan[pango]`. |
| `Speech` raises | No system TTS engine (`espeak-ng` on Linux). |
| Material/shader ignored | Set after `spawn()`. |
| Metal object black | Nothing to reflect: environment map or neighbours. |
| Glass invisible | Nothing behind it. Dark patches inside: raise `max_bounces`. |
| No shadows | `SETTINGS.raytracing.set(shadows=True)`. Lights beyond 16 slots are lit but unshadowed. |
| `.mp4` refused | Transparent background; use `.mov`. |
| Output unexpectedly transparent | Scaled colour halved alpha; `.set_opacity(1.0)`. |
| `SETTINGS.video.set` had no effect | Ran after the first Mob created the Scene. |
| `SETTINGS.video = HD` raises | Use `SETTINGS.video.set(HD)`. |
| Setting an experimental switch on `SETTINGS.raytracing` raises | It lives on `SETTINGS.raytracing.experimental`. |
| Texture map ignored with a warning | Geometry has no UVs (`Cube`, polyhedra); use a `Surface`-based Mob. |
| Screen-pinned label drifts during a camera move | Screen methods resolve once; parent it to the camera or use an updater. |
| `become` target shows up / warns | Build it with `add_to_scene=False`. |
| Model invisible / black / faceted | `fit_to_size=2`; add an environment map; file has no normals. |
| Daemon serves stale or wrong config | `algan daemon quit`, or `ALGAN_USE_DAEMON=0`. |
