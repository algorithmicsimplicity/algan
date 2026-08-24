"""Render one moment of a ``tests/full_renders`` scene at any quality.

The suite renders those scenes at ``PREVIEW`` (704x396), where a shape is often
sixty pixels across and an artifact sits right at the edge of legibility. This
replays the same recording and saves a single frame at whatever quality you
name, which is how you tell a real geometry fault from a low-resolution one.

    <venv-python> benchmarks/_full_render_frame_probe.py <scene> <seconds> [outname] [quality]

``scene`` is a file stem from ``tests/full_renders/scenes/`` (e.g.
``solids_and_camera``), ``seconds`` a time on the scene's own timeline -- the
suite's videos are 10 fps, so frame N of a baseline is N/10 seconds here.
``quality`` is any preset name exported by ``algan`` (default ``HD``).
"""

import importlib.util
import sys
from pathlib import Path

import algan
from algan import HD, Scene

SCENES = Path(__file__).resolve().parents[1] / "tests" / "full_renders" / "scenes"

if len(sys.argv) < 3:
    sys.exit(__doc__)

scene_name = sys.argv[1]
at = float(sys.argv[2])
name = sys.argv[3] if len(sys.argv) > 3 else f"{scene_name}_at_{at}"
quality = getattr(algan, sys.argv[4]) if len(sys.argv) > 4 else HD

path = SCENES / f"{scene_name}.py"
if not path.exists():
    sys.exit(
        f"no such scene: {path}\navailable: "
        + ", ".join(sorted(p.stem for p in SCENES.glob("*.py")))
    )

spec = importlib.util.spec_from_file_location("_full_render_probe_scene", path)
module = importlib.util.module_from_spec(spec)

with Scene() as scene:
    # The scene file records its animation on the active Scene; it never
    # renders, so save_frame is what materializes the moment asked for.
    spec.loader.exec_module(module)
    scene.save_frame(name, quality, at=at)
