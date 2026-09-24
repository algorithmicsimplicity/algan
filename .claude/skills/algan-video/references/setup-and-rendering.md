# Setup and rendering

Use this reference to turn scene code into files. It is not an Algan development
or compiler-build guide. The API baseline and source links are in
[API sources](api-sources.md).

## Reuse a working environment

Use the interpreter associated with the user's project. For an existing virtual
environment, invoke `.venv/bin/python` on Unix or `.venv/Scripts/python.exe` on
Windows directly. Do not recreate or upgrade a working environment merely to
match this skill's source snapshot. Do not install another copy of Algan into a
different interpreter and then run the original one.

Run the bundled `scripts/check_environment.py` with that interpreter. It checks
package metadata, discoverability, and external executables without importing
Algan. Then run that environment's `algan check` for Algan's own diagnostics.
The bundled check does not initialize a GPU, prove a compiler backend works, or
replace an actual small render.

For a new environment, install the library through that environment's Python:

```bash
python -m pip install algan
algan check
```

The source-checked installation guide supports Python 3.10–3.13. Verify wheel
availability before selecting a newer interpreter. Algan's package dependencies
supply its matching compiler; do not independently install an arbitrary Taichi
or Quadrants version. Custom shader code should import the compiler facade with
`from algan.taichi_compat import ti`.

Optional dependencies are task-specific:

| Requested capability | Check |
|---|---|
| System-font `Text` | Pango/ManimPango support and the exact font family |
| `Tex` equations or the Linux text fallback | A working TeX installation and SVG conversion tools such as `dvisvgm` |
| Recorded narration alignment | The `algan[audio]` extra and the recorded audio/transcript |
| Default synthetic speech | The operating system's speech engine; Linux commonly needs eSpeak NG |
| FBX import | `algan[fbx]` and the native Assimp library |
| Video encoding | A usable FFmpeg binary; `algan check` reports its status |

On Linux, Pango-backed text is an extra because installing the Python binding
can require system development packages. A Debian/Ubuntu installation, when
system-package installation is authorized, is:

```bash
sudo apt-get install -y build-essential python3-dev libpango1.0-dev pkg-config
python -m pip install 'algan[pango]'
```

Without it, native `Text` can fall back to LaTeX text mode. This is not a guarantee
that the requested font or layout will be preserved. Do not silently replace
fonts, convert equations to plain text, or omit audio to get a successful run.

Record the Python executable, Algan distribution version, and imported source
location when diagnosing mismatched APIs. For a source installation, record the
checkout revision as well. Installation success is not rendering success.

## Script and CLI execution

Run a scene script with Python, or use Algan's CLI:

```bash
python scene.py
algan render scene.py -q preview
algan render scene.py -q hd -o renders/
algan render scene.py --no-daemon -- --seed 7
```

The CLI quality choices are `preview`, `ld`, `md`, `hd`, `production`, and `uhd`.
Arguments following `--`, and unrecognized script arguments, are forwarded to
the script. Do not invent a `--seed` option in a script that does not implement
one; the last command demonstrates forwarding only.

CLI defaults do not override explicit script choices. For example,
`Scene.save_video('clip', PREVIEW)` still requests PREVIEW when launched with
`-q hd`. The bundled single-scene examples explicitly request PREVIEW, so edit
that argument for a final delivery. A script's path containing a directory also
wins over a CLI output directory. Inspect the returned path rather than guessing.

Algan may hand a script to a warm render daemon during `import algan`. Keep
side effects such as file writes below the Algan import: code above it may run in
both the launching process and the daemon. Between runs the daemon evicts every
user module the script imported, wherever it lives (the script's folder, a
parent folder, another project), together with its bytecode and traceback
caches, so edits are picked up and tracebacks show current lines. Algan,
installed packages and a user package that contains a compiled extension module
stay loaded; the daemon says so for the latter, and it needs a daemon restart to
pick up edits. When Algan's own source changes, the daemon shuts down and the
run executes in a fresh process. Do not depend on interactive `input()`
or `atexit` cleanup in daemon-served scripts. `--no-daemon` bypasses it; for a
Python invocation, set `ALGAN_USE_DAEMON=0` before starting the process when a
fresh-process run is required. `ALGAN_AUTO_DAEMON=0` only prevents starting new
daemons; it is not the same as bypassing an existing one.

The interactive commands include `algan daemon ping`, `algan daemon render`,
`algan daemon cancel`, and `algan daemon quit`. Cancellation interrupts the
active script and retains queued scripts; quitting shuts the daemon down.
Do not start an indefinite interactive service as part
of an unattended export. First-time shader/kernel compilation can be substantial;
retain caches rather than clearing them on every scene edit.

Run `algan check` through the chosen environment to verify actual write access
to content, text, speech and kernel caches, daemon home and the working-directory
output location. Recent versions create a temporary probe file and remove it;
older versions only report paths. A successful import does not establish write
access. On Windows, run with the same user and sandbox as the actual render.

For a long source-project task, validation plus reviewed samples is the default
delivery; a full production export needs to be part of the requested scope.
When an export is requested, use the existing `profile_scene` report or Project's
`--profile` / `--estimate-render-time` on selected short representative scenes
if a cost estimate is useful. Sampling itself renders complete reference scenes.
Keep first-pass startup and compilation costs separate from warm throughput.

## Settings and the actual requested output

The settings sections have stable identity. Mutate them, do not replace them:

```python
SETTINGS.video.set(HD)
SETTINGS.video.set(frames_per_second=60)
custom_video = HD.set(resolution=(1080, 1920), frames_per_second=30)
```

A preset's `.set(...)` returns a new preset; it does not mutate `HD`. Resolution
is `(width, height)`. Video settings include `frames_per_second`,
`supersampling`, `fxaa`, and `audio_sample_rate`. `fps` and `ssaa` are supported
short forms, but full names are clearer in a reproducible delivery.

| Preset | Width × height | Frames/second |
|---|---:|---:|
| SMOKE_TEST | 32 × 32 | 2 |
| PREVIEW | 704 × 396 | 10 |
| LD | 864 × 486 | 15 |
| MD | 1280 × 720 | 30 |
| HD | 1920 × 1080 | 30 |
| PRODUCTION | 2560 × 1440 | 60 |
| UHD | 3840 × 2160 | 60 |
| THUMBNAIL | 1280 × 720 | 1 |

These are API facts, not recommendations for the user's video. Use the requested
resolution and frame rate. Establish the intended aspect ratio before recording
screen-relative layout; changing it only at export can invalidate placement.
A lower-resolution draft should retain that aspect ratio.

`SETTINGS.raytracing.set(samples_per_pixel=1)` selects the deterministic hybrid
renderer. A value above one selects the path tracer, whose lighting and sampling
can change the image. This is not interchangeable with video supersampling.
Never promise that a low-sample deterministic draft is a visual match for a
path-traced final. See [materials and cameras](materials-lights-and-camera.md).

## Scene ownership

`Scene.current()` gets the active Scene; `Scene.new()` creates another. The
class-style public calls such as `Scene.save_video(...)` operate on the active
Scene; instance calls operate on that instance. A Scene owns its camera, actors,
timeline, and audio. Do not move Mob instances between independent scenes as a
substitute for recreating them in their owning scene.

For a single script, the automatically available active Scene is sufficient.
For a collection of shots, use the [Project API](audio-and-projects.md) instead
of manually managing internal timeline or scene-manager objects. For repeated
execution in a notebook or another persistent authoring process, start a fresh
scene deliberately rather than appending the same animation repeatedly.

## Export calls

```python
video = Scene.save_video('renders/clip.mp4', video_settings=HD)
frame = Scene.save_frame('renders/check.png', video_settings=HD, at=0.75)
frames = Scene.save_frame('renders/check.png', at=[0.25, 0.75, 1.25])
print(video.status, video.output_path, video.walltime_seconds)
```

`save_frame` times are in seconds, not frame indices. Positive values select
authored timeline times; a negative value is relative to the current authored
time. Omitting `at` selects that current time. A sequence produces multiple files
with time suffixes and a list of results. Prefer inspecting after timing contexts
have closed, when block durations have been resolved.

The public parameter is `video_settings`, not `render_settings`; there is no
module-level `render_to_file` or substitute `RenderSettings` API. `save_video`
also accepts `background`, `post_processes`, `codec`, `audio_codec`,
`ffmpeg_params`, `overwrite`, `reset`, and `animate_fade_out`. Read the installed
signature before using policy-style options such as overwrite behavior.

A bare name goes under the configured output directory, normally
`algan_outputs`. A path with a directory component is used as supplied. Use a
unique or explicitly authorized output path and report `RenderResult.output_path`.
A result may describe a skipped existing file: do not call that a newly rendered
video. Rendering normally preserves the recording for another render;
`reset=True` explicitly resets it. A requested fade-out or the zero-duration
video guard can still add timeline work. Author the desired duration explicitly.

Opaque video commonly uses MP4. Use a compatible alpha-bearing format for a
transparent output; a `.mp4` filename does not preserve transparency. See
[compositing](compositing-and-postprocessing.md) for ordinary alpha versus the
special additive-glow export. Encoder hardware and rendering hardware are
separate; do not infer GPU rendering from the presence of an NVENC encoder.

`Scene.view()` is an interactive viewer, not an export method. It can block until
stopped. Use it only in an interactive workflow where that is useful and expected.

## Verification and reproducibility

Inspect representative rendered frames, including just before lifespan endpoints,
not only one attractive still. Check the first visible frame, required simultaneous
changes, updater dependencies, text glyphs, texture axes, transparency edges, and
final hold. Verify an audio-bearing clip by listening when possible; matching
stream durations does not prove synchronization.

Run `scripts/verify_video.py` on the returned file. It can assert dimensions,
frame rate, duration, and audio presence and optionally decode the entire clip.
It cannot verify visual intent, spoken words, usable transparency, or shader
correctness. A technical export report should say which of those were actually
checked and include the source, dependency/version information, asset locations,
and command needed to reproduce the output.

Source basis: [installation, settings, and export sources](api-sources.md#setup-and-export).
