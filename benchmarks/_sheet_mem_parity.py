"""Byte-identity A/B for the sheet-compaction memory reductions.

Renders a scene that exercises every path the edited code has a branch for --
PN surfaces (smooth shading class), flat polyhedra (the crease split), bezier
circuits and text (negative refs / the areal sheet), transparency (the opaque
prefix truncation), and a one-mesh closed solid (the coverage ceiling) -- and
prints a hash per frame. Run it with the old and the new
``sheets.py`` / ``raster_pipeline.py`` and diff the two outputs: the change is
a pure release-earlier / narrower-dtype rewrite, so every hash must match.

    <venv-python> benchmarks/_sheet_mem_parity.py > before.txt   # old files
    <venv-python> benchmarks/_sheet_mem_parity.py > after.txt    # new files
    diff before.txt after.txt
"""

import hashlib
import os

os.environ["ALGAN_USE_DAEMON"] = "0"

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from algan import *  # noqa: E402,F403
from algan.constants.math import GIGABYTES  # noqa: E402

# Free VRAM drifts between runs and the arena is sized from it, which
# re-windows the render; pin it so the two arms make identical batches.
SETTINGS.computing.set(available_memory_override=2 * GIGABYTES)

STEM = "_sheet_mem_parity"


def build():
    sphere = Sphere().scale(1.4).move(LEFT * 3).set_color(GREEN).spawn()
    cube = Cube().scale(1.1).move(RIGHT * 3).set_color(BLUE).spawn()
    glass = Sphere().scale(0.9).move(UP * 1.2).spawn()
    glass.opacity = 0.45
    circle = Circle().scale(0.8).move(DOWN * 1.8 + LEFT * 1.5).set_color(RED).spawn()
    label = Text("sheets").scale(0.6).move(DOWN * 2.4 + RIGHT * 1.5).spawn()
    with Sync():
        sphere.rotate(70, UP)
        cube.rotate(55, OUT + RIGHT)
        glass.move(RIGHT * 1.3)
        circle.rotate(40, OUT)
        label.move(UP * 0.3)


build()
for i, at in enumerate((0.0, 0.35, 0.7, 1.0)):
    path = Scene.save_frame(f"{STEM}_{i}.png", MD, at=at).output_path
    arr = np.asarray(Image.open(path).convert("RGB"))
    print(f"frame {i} t={at:4.2f} {hashlib.sha256(arr.tobytes()).hexdigest()}")
