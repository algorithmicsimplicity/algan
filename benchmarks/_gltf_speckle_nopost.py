"""Same sphere, rendered through ``save_video`` so post-processing can be
switched off: ``NOPOST=1`` renders with ``post_processes=()``.

Writes ``benchmarks/_gltf_nopost_out/<name>.mp4`` and dumps frame 0 as a PNG,
then prints the left-rim profile row by row.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.chdir(REPO / "tests" / "full_renders")

from algan import *  # noqa: E402
from algan import HD, SETTINGS, Scene  # noqa: E402

OUTDIR = REPO / "benchmarks" / "_gltf_nopost_out"
OUTDIR.mkdir(parents=True, exist_ok=True)
SETTINGS.paths.set(output_root=str(OUTDIR), output_directory=".")

NOPOST = os.environ.get("NOPOST", "1") == "1"
Scene.set_background(DARKER_GRAY)

with Off():
    model = Model3D("assets/textured_icosphere.glb", fit_to_size=7.0).move(UP * 0.2)
    stack = [model]
    while stack:
        mob = stack.pop()
        stack.extend(getattr(mob, "children", ()) or ())
        tmap = getattr(mob, "texture_map", None)
        if tmap is None:
            continue
        new = tmap.clone()
        new[..., 0] = 0.85
        new[..., 1] = 0.55
        new[..., 2] = 0.25
        mob.texture_map = new

model.spawn(animate=False)
Scene.wait(0.3)

name = "nopost" if NOPOST else "withpost"
kwargs = {"post_processes": []} if NOPOST else {}
Scene.save_video(OUTDIR / f"{name}.mp4", HD, overwrite=True, **kwargs)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

cap = cv2.VideoCapture(str(OUTDIR / f"{name}.mp4"))
ok, frame = cap.read()
assert ok
cv2.imwrite(str(OUTDIR / f"{name}.png"), frame)
g = frame[..., 2].astype(int)
bg = int(np.median(g[:50, :50]))
print(f"{name}: background {bg}")
for y in range(190, 900, 30):
    xs = np.where(g[y] > bg + 8)[0]
    if len(xs) < 5:
        continue
    lo, hi = xs[0], xs[-1]
    print(y, "L", lo, g[y, lo - 1 : lo + 4], "| R", hi, g[y, hi - 3 : hi + 2])
