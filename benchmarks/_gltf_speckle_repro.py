"""Isolate the glTF + PBR + normal-map sphere from tests/full_renders/text_and_media.

Renders the model alone with the same lights at a few of the rotation angles the
full scene passes through, at a resolution high enough to see individual pixels
on the lower-left limb.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT / "tests" / "full_renders")

from algan import *  # noqa: E402

OUT_DIR = ROOT / "benchmarks" / "_gltf_speckle_repro_out"
OUT_DIR.mkdir(exist_ok=True)

angle = float(sys.argv[1]) if len(sys.argv) > 1 else 200.0

with Off():
    AmbientLight(color=WHITE, intensity=0.55).spawn(animate=False)
    DirectionalLight(
        location=RIGHT * 4 + UP * 5 + OUT * 4,
        target=ORIGIN,
        color=WHITE,
        intensity=1.0,
    ).spawn(animate=False)
    model = ThreeDModelMob(
        "assets/textured_icosphere.glb",
        normalize=True,
        normalize_size=2.6,
    ).move(UP * 0.2)
    model.spawn(animate=False)
    model.rotate(angle, UP)

Scene.set_background(DARKER_GRAY)
Scene.save_frame(str(OUT_DIR / ("angle_%g" % angle)), MD, overwrite=True)
print("wrote", OUT_DIR / ("angle_%g.png" % angle))
