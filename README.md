# Algan

<p align="center">
  <strong>Full-featured 2D/3D programmatic animation engine for explanatory mathematics and technical videos.</strong>
</p>

<p align="center">
  <a href="https://algorithmicsimplicity.github.io/algan"><img src="https://img.shields.io/badge/docs-algorithmicsimplicity.github.io%2Falgan-blue.svg" alt="Documentation" /></a>
  <a href="https://pypi.org/project/algan/"><img src="https://img.shields.io/pypi/v/algan.svg" alt="PyPI version" /></a>
  <a href="https://pypi.org/project/algan/"><img src="https://img.shields.io/pypi/pyversions/algan.svg" alt="Python versions" /></a>
  <a href="https://discord.gg/NvarFmvXKm"><img src="https://img.shields.io/discord/1122334455?color=7289da&label=Discord" alt="Discord Community" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT" /></a>
</p>

---

Algan is designed to be a successor to Manim with full-featured 3-D raytracing capabilities.

As seen on [AlgorithmicSimplicity](https://www.youtube.com/@algorithmicsimplicity).

---

## Key Features

- **Manim Feature Parity**: Everything you know and love from Manim.
- **GPU Ray Tracing**: High-fidelity optical effects including depth of field, area lights, glossy reflections, refractive glass, and soft shadows.
- **Declarative Timeline Contexts**: Intuitive animation staging with `Seq()`, `Sync()`, `Lag()`, `Off()`, and `Speech()` blocks makes animation code modular and re-usable.
- **Unified 2D/3D Geometry**: Seamless morphing and interpolation between 2D Bézier circuits and 3D meshes with `become()`.
- **Audio & Speech Alignment**: Automatic word-level forced alignment to synchronize on-screen animations with narration.

---

## Installation

Install via [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv add algan
```

Or via `pip`:

```bash
pip install algan
```

Every dependency ships wheels, so there is nothing to build and no system
package to install first. The installed footprint is large (~5 GB, mostly the
CUDA build of `torch` and its NVIDIA dependencies).

### Optional: Pango text

`Text` typesets through LaTeX's text mode by default. For Pango instead — which
gives you system fonts, per-span font/weight/slant styling, and Manim's
`MarkupText` and `Paragraph` — add the `pango` extra:

```bash
pip install "algan[pango]"
```

It is an extra rather than a dependency because `manimpango` publishes no Linux
wheel; on Linux it builds from source and wants Pango's headers first:

```bash
sudo apt-get install -y libpango1.0-dev pkg-config
```

For detailed platform-specific prerequisites (FFmpeg, GPU acceleration, optional LaTeX for formulas), see the [Installation Guide](https://algorithmicsimplicity.github.io/algan/installation/uv.html).

---

## Quickstart

Save this script as `scene.py`:

```python
from algan import *

# 1. Make 3-D objects with physical materials
sphere = Sphere(color=BLUE, radius=1.2)
sphere.set_material(
    MeshPhysicalMaterial(
        roughness=0.15,
        metalness=0.1,
        clearcoat=1.0,
        clearcoat_roughness=0.08,
    )
)

# 2. Define animation timeline with contexts
sphere.spawn()
with Sync(runtime=2):
    sphere.move(RIGHT * 2)
    sphere.rotate(180, OUT, about=ORIGIN)
    sphere.color = RED

# 3. Render video
Scene.save_video("quickstart.mp4")
```

Run with Python or the Algan CLI:

```bash
# Using python
python scene.py

# Using the algan CLI
algan render scene.py
```

The output video will be written to `algan_outputs/quickstart.mp4`.

---

## Command Line Interface (CLI)

Algan includes a first-class CLI:

```bash
algan check                 # Verify PyTorch, GPU acceleration, Taichi, FFmpeg, LaTeX & paths
algan new my_scene.py       # Scaffold a new scene script
algan render my_scene.py    # Render scene to video
```

`render` takes `-q {preview,ld,md,hd,production,uhd}` for the video preset and
`-o` for the directory or file to write to. Both fill in what the script leaves
open: a `Scene.save_video("intro")` still decides the name, and a path with a
directory in it still decides everything.

```bash
algan render my_scene.py -q hd -o renders/          # renders/intro.mp4, at HD
algan render my_scene.py --no-daemon -- --seed 7    # fresh process, args forwarded
```

A scene script may have a command line of its own (such as a
[`Project`](https://algorithmicsimplicity.github.io/algan/reference/algan.project.Project.html)
calling `run_cli()`) so any argument this CLI does not recognise is
forwarded to it, as is everything after `--`:

```bash
algan render project.py -q hd --render-video intro   # -q ours, --render-video the project's
algan render project.py -- --help                    # the project's help, not ours
```

Where both name the same thing, the script wins: `-q` sets the default preset,
and a `Project`'s own `--video-settings` (or its `video_settings=` argument)
overrides it.

### The warm render daemon

The first render of a session pays several seconds of library import plus
Taichi kernel preparation. Algan pays that once: the first `python scene.py`
starts a background daemon, and every later run hands its script to that warm
process and starts rendering in about a second. Nothing is launched
differently — it happens inside `import algan`.

```bash
algan daemon                # run one in this terminal (Enter re-renders, q quits)
algan daemon ping           # is one running?
algan daemon render         # re-render the last script (bind an editor key to it)
algan daemon quit           # stop it (algan daemon --stop is the same)
```

Those verbs each carry the token from the daemon's state file
(`$ALGAN_HOME/daemon.json`, default `~/.algan`), which is also where the port
lives — the daemon prefers 46711 and falls back to an ephemeral port when it is
taken.

**A script served by the daemon runs in another process**, and three things
follow from that:

- everything above `import algan` runs **twice** — once in your process, once
  in the daemon — so keep side effects below the import;
- `atexit` handlers do not run, because the warm process never shuts down;
- `stdin` is `/dev/null`, since the daemon's own stdin is its re-render trigger.

Everything else is reproduced: `sys.argv`, the working directory, the full
environment, stdout/stderr (including from ffmpeg and other subprocesses),
their tty-ness, and the exit code. `ALGAN_USE_DAEMON=0` runs in-process,
`ALGAN_AUTO_DAEMON=0` only stops new ones being started, and a script being
debugged is never handed off.

---

## Documentation

- **Documentation**: [https://algorithmicsimplicity.github.io/algan](https://algorithmicsimplicity.github.io/algan)
- **Tutorials**: [New User Tutorials](https://algorithmicsimplicity.github.io/algan/new_user_tutorials/index.html)
- **Discord Community**: [Join our Discord](https://discord.gg/NvarFmvXKm)
- **Issue Tracker**: [GitHub Issues](https://github.com/algorithmicsimplicity/algan/issues)

---

## License

Algan is licensed under the MIT License (see [LICENSE](LICENSE)). Copyright &copy; Algorithmic Simplicity.
