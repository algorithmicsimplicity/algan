# Algan

<p align="center">
  <strong>High-performance 2D/3D programmatic animation engine for explanatory mathematics and technical videos.</strong>
</p>

<p align="center">
  <a href="https://algorithmicsimplicity.github.io/algan"><img src="https://img.shields.io/badge/docs-algorithmicsimplicity.github.io%2Falgan-blue.svg" alt="Documentation" /></a>
  <a href="https://pypi.org/project/algan/"><img src="https://img.shields.io/pypi/v/algan.svg" alt="PyPI version" /></a>
  <a href="https://pypi.org/project/algan/"><img src="https://img.shields.io/pypi/pyversions/algan.svg" alt="Python versions" /></a>
  <a href="https://discord.gg/NvarFmvXKm"><img src="https://img.shields.io/discord/1122334455?color=7289da&label=Discord" alt="Discord Community" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT" /></a>
</p>

---

Algan is an animation engine designed to supersede traditional mathematical animation tools. It combines a declarative lazy recording timeline with GPU-accelerated ray tracing and Monte Carlo path tracing (powered by [Taichi](https://www.taichi-lang.org/)), Three.js-style physically based materials (PBR), and first-class audio synchronization.

As seen on [AlgorithmicSimplicity](https://www.youtube.com/@algorithmicsimplicity).

---

## Key Features

- **⚡ GPU Ray Tracing & Path Tracing**: High-fidelity optical effects including depth of field, area lights, glossy reflections, refractive glass, and soft shadows via Taichi GPU megakernels.
- **🎨 Three.js PBR Materials**: Familiar `MeshStandardMaterial` and `MeshPhysicalMaterial` with clearcoat, sheen, IOR, roughness, metalness, and transmission.
- **⏱️ Declarative Timeline Contexts**: Intuitive animation staging with `Seq()`, `Sync()`, `Lag()`, `Off()`, `Audio()`, and `Speech()` blocks.
- **🔄 Unified 2D/3D Geometry**: Seamless morphing and interpolation between 2D Bézier circuits and 3D meshes with `become()`.
- **🎙️ Audio & Speech Alignment**: Automatic word-level forced alignment to synchronize on-screen animations with narration.
- **🚀 Warm Process Daemon**: Fast feedback loops with automatic source change detection and zero-overhead scene re-runs.

---

## Comparison

| Feature | Algan | Manim | 3D DCC (Blender) |
| :--- | :--- | :--- | :--- |
| **Animation Model** | Lazy declarative recording | Procedural imperative updates | Keyframe graph editors |
| **Rendering Engine** | Hybrid Raster + Ray / Path Tracer (GPU) | Cairo / OpenGL raster | Cycles / EEVEE |
| **Materials** | Three.js PBR (`MeshStandardMaterial`, `MeshPhysicalMaterial`) | Flat vertex colors / simple shaders | Full node-based BSDF shaders |
| **Code-Driven** | 100% Python | 100% Python | Python API / GUI |
| **Audio Sync** | Built-in forced alignment (`Speech()`, `Audio()`) | Manual timestamp offsets | Manual timeline markers |
| **Angles** | Degrees ($0^\circ - 360^\circ$) | Radians ($0 - 2\pi$) | Degrees |

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

For detailed platform-specific prerequisites (FFmpeg, GPU acceleration, optional LaTeX for formulas), see the [Installation Guide](https://algorithmicsimplicity.github.io/algan/installation/uv.html).

---

## Quickstart

Save this script as `scene.py`:

```python
from algan import (
    BLUE,
    RED,
    WHITE,
    MeshPhysicalMaterial,
    Scene,
    Seq,
    Sphere,
    Sync,
)

# 1. Author 3D objects with physical materials
sphere = Sphere(color=BLUE, radius=1.2)
sphere.set_material(
    MeshPhysicalMaterial(
        roughness=0.15,
        metalness=0.1,
        clearcoat=1.0,
        clearcoat_roughness=0.08,
    )
)

# 2. Stage animations inside timeline contexts
with Seq():
    sphere.spawn()
    with Sync(run_time=2.0):
        sphere.rotate(180)
        sphere.move([2, 0, 0])
        sphere.color = RED

# 3. Render video
Scene.save_video("quickstart")
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
algan check                 # Verify PyTorch, GPU acceleration, Taichi, FFmpeg & LaTeX
algan new my_scene.py       # Scaffold a new scene script
algan render my_scene.py    # Render scene to video
algan preview my_scene.py   # Render it at the low-resolution preview preset
algan daemon                # Launch background warm render daemon
algan daemon --stop         # Stop the running daemon
```

`render` takes `-q {preview,ld,md,hd,production,uhd}` for the video preset and
`-o` for the directory or file to write to. Both fill in what the script leaves
open: a `Scene.save_video("intro")` still decides the name, and a path with a
directory in it still decides everything.

```bash
algan render my_scene.py -q hd -o renders/          # renders/intro.mp4, at HD
algan render my_scene.py --no-daemon -- --seed 7    # fresh process, args forwarded
```

A scene script may have a command line of its own — a
[`Project`](https://algorithmicsimplicity.github.io/algan/reference/algan.project.Project.html)
calling `run_cli()` does — so any argument this CLI does not recognise is
forwarded to it, as is everything after `--`:

```bash
algan render project.py -q hd --render-video intro   # -q ours, --render-video the project's
algan render project.py -- --help                    # the project's help, not ours
```

Where both name the same thing, the script wins: `-q` sets the default preset,
and a `Project`'s own `--video-settings` (or its `video_settings=` argument)
overrides it.

Because those two are settings, and a warm process shared with other runs cannot
be handed one run's settings, `-q` and `-o` run the script in the CLI's own
process and skip the [render daemon](https://algorithmicsimplicity.github.io/algan/advanced_user_tutorials/the_render_daemon.html).
A plain `algan render my_scene.py` is launched as its own process, exactly as
`python my_scene.py` is, and the daemon serves it warm.

---

## Documentation

- **Documentation**: [https://algorithmicsimplicity.github.io/algan](https://algorithmicsimplicity.github.io/algan)
- **Tutorials**: [New User Tutorials](https://algorithmicsimplicity.github.io/algan/new_user_tutorials/index.html)
- **Discord Community**: [Join our Discord](https://discord.gg/NvarFmvXKm)
- **Issue Tracker**: [GitHub Issues](https://github.com/algorithmicsimplicity/algan/issues)

---

## License

Algan is licensed under the MIT License (see [LICENSE](LICENSE)). Copyright &copy; Algorithmic Simplicity.
