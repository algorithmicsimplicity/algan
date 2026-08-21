"""Is the rim overshoot the reflection continuation?

Renders the glTF sphere with a flat base colour (so any colour that is not the
flat colour came from somewhere else) at ``max_bounces`` 1 and 0, and prints the
scanline profile across the left and right rims.

With ``max_bounces=0`` ``_scatter_impl`` zeroes ``R`` and no reflection ray is
traced, so if the rim overshoot is the self-reflection fired into the sphere by
the flipped shading normal, it must vanish.
"""

import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.chdir(REPO / "tests" / "full_renders")

from algan import *  # noqa: E402
from algan import HD, SETTINGS, Scene  # noqa: E402

BOUNCES = int(os.environ.get("BOUNCES", "1"))
OUTDIR = REPO / "benchmarks" / f"_gltf_bounce{BOUNCES}"
OUTDIR.mkdir(parents=True, exist_ok=True)
SETTINGS.paths.set(output_root=str(OUTDIR), output_directory=".")
SETTINGS.raytracing.set(max_bounces=BOUNCES)

Scene.set_background_color(DARKER_GRAY)

with Off():
    AmbientLight(color=WHITE, intensity=0.55).spawn(animate=False)
    DirectionalLight(
        location=RIGHT * 4 + UP * 5 + OUT * 4,
        target=ORIGIN,
        color=WHITE,
        intensity=1.0,
    ).spawn(animate=False)
    model = ThreeDModelMob(
        "assets/textured_icosphere.glb", normalize=True, normalize_size=2.6
    ).move(UP * 0.2)
    stack = [model]
    while stack:
        mob = stack.pop()
        stack.extend(getattr(mob, "children", ()) or ())
        tmap = getattr(mob, "texture_map", None)
        if tmap is not None:
            new = tmap.clone()
            new[..., 0], new[..., 1], new[..., 2] = 0.85, 0.55, 0.25
            mob.texture_map = new

model.spawn(animate=False)
Scene.save_frame("bounce", HD)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

np.set_printoptions(linewidth=250)
img = cv2.imread(str(OUTDIR / "bounce.png")).astype(int)
R = img[..., 2]
bg = int(np.median(R[0:50, 0:50]))
print(f"max_bounces={BOUNCES}  bg={bg}")
worst = 0
for y in range(400, 620, 2):
    row = R[y]
    xs = np.where(row > bg + 6)[0]
    if len(xs) < 20:
        continue
    l, r = xs.min(), xs.max()
    # overshoot = outermost covered pixel above the pixel just inside it
    worst = max(worst, row[l] - row[l + 1], row[r] - row[r - 1])
for y in [470, 510, 550, 570, 600]:
    row = R[y]
    xs = np.where(row > bg + 6)[0]
    if len(xs) < 20:
        continue
    l, r = xs.min(), xs.max()
    print(f" y={y} LEFT", row[l - 2 : l + 9], " RIGHT", row[r - 8 : r + 3])
print(f"worst rim overshoot (outer minus next-inner): {worst}")
