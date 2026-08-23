"""Film-strip a chained cross-family morph so its middle can be looked at.

`Cylinder -> Sphere -> Arrow`. The end frames are checked by
`_become_endstate_check.py`; this exists for the frames in between, which no
assertion covers and which is where the cross-family route looks worst:

* a solid becoming a solid tears into visibly separated strips while the
  independent PN triangles travel to their new counterparts,
* a solid becoming a STROKE-ONLY shape goes fully blank for roughly a third of
  the morph, because `_bezier_to_pn_soup` zeroes an unfilled circuit's opacity
  (there is no fill to convert) and the source therefore tweens its own opacity
  to zero before the real target spawns at the end.

Usage:  <venv-python> benchmarks/_become_chain_filmstrip.py
"""

from pathlib import Path

import cv2
import numpy as np

from algan import *

OUT = Path("/tmp/chain")
OUT.mkdir(exist_ok=True)

STEPS = [
    ("Cylinder", lambda: Cylinder(radius=0.5, height=1.4, color=BLUE)),
    ("Sphere", lambda: Sphere(radius=0.8, color=YELLOW)),
    ("Arrow", lambda: Arrow(LEFT * 1.4, RIGHT * 1.4, color=RED)),
]
SAMPLES = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]

frames = []
for step in range(len(STEPS) - 1):
    for sample in SAMPLES:
        path = OUT / f"s{step}_{sample:.2f}.png"
        with Scene() as scene:
            scene.use_manim_defaults()
            with Off():
                current = STEPS[0][1]().spawn()
            # replay the chain up to `step`, then sample inside step -> step+1
            for index in range(step):
                with Off():
                    nxt = STEPS[index + 1][1]()
                with Sync(run_time=1.0):
                    current = current.become(nxt)
            start = float(scene.animation_manager.context.timespan.current_time)
            with Off():
                nxt = STEPS[step + 1][1]()
            with Sync(run_time=1.0):
                current = current.become(nxt)
            end = float(scene.animation_manager.context.timespan.current_time)
            at = min(start + (end - start) * sample, end - 1e-4)
            scene.save_frame(str(path), LD, at=at, overwrite=True)
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        frames.append(image[..., :3] if image.shape[2] == 4 else image)
    print(f"rendered step {STEPS[step][0]} -> {STEPS[step + 1][0]}", flush=True)

rows = [
    np.concatenate(frames[i * len(SAMPLES) : (i + 1) * len(SAMPLES)], axis=1)
    for i in range(len(STEPS) - 1)
]
grid = np.concatenate(rows, axis=0)
scale = 1900 / grid.shape[1]
grid = cv2.resize(grid, (int(grid.shape[1] * scale), int(grid.shape[0] * scale)))
cv2.imwrite("/tmp/chain/chain_strip.png", grid)
print("wrote /tmp/chain/chain_strip.png")
print("row 1: Cylinder -> Sphere ; row 2: Sphere -> Arrow ; cols:", SAMPLES)
