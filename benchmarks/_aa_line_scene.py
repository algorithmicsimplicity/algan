"""The visual half of the anti-aliasing sanity check.

Renders one frame containing both kinds of line -- :class:`Line` (a cubic
bezier circuit) and a thin :class:`Cylinder` (triangles) -- fanned out over a
range of slopes and interleaved so that each bezier line sits directly beside
the triangle line of the same slope. Also writes a 6x magnified crop of the
fan's centre, since anti-aliasing at 1:1 is exactly the thing the eye cannot
adjudicate.

Everything is at default quality settings; the only deliberate choice is a
black background and white lines, so nothing but coverage varies.

Run:  <venv-python> benchmarks/_aa_line_scene.py [--res md|hd|ld]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#: Slopes to fan the lines over, in degrees. Deliberately not a set of round
#: numbers: the interesting failures are at shallow slopes, where a line
#: advances a fraction of a pixel per column.
FAN_ANGLES = (2.0, 6.0, 14.0, 26.565, 45.0, 63.435, 76.0, 84.0, 88.0)

#: Matches the default ``Line`` stroke, measured at 0.098 world units wide.
CYLINDER_RADIUS = 0.049
LINE_LENGTH = 1.5
#: Perpendicular offset of each member of a pair from its cell centre.
PAIR_GAP = 0.32
#: Cell pitch in world units. The default camera frame is 12.44 x 7.0 units,
#: so a 3 x 3 grid at this pitch keeps every cell's content clear of its
#: neighbours' at every slope.
CELL_X, CELL_Y = 4.0, 2.2


def _cell_centre(index):
    col, row = index % 3, index // 3
    return ((col - 1) * CELL_X, (1 - row) * CELL_Y)


def build(scene):
    """Spawn the interleaved fan of bezier and triangle lines."""
    import torch

    from algan.constants.color import WHITE
    from algan.mobs.shapes_2d import Line
    from algan.mobs.shapes_3d import Cylinder
    from algan.mobs.text import Text
    from algan.rendering.shaders.materials import MeshBasicMaterial

    for i, angle in enumerate(FAN_ANGLES):
        phi = math.radians(angle)
        direction = torch.tensor([math.cos(phi), math.sin(phi), 0.0])
        cx, cy = _cell_centre(i)
        centre = torch.tensor([cx, cy, 0.0], dtype=torch.float32)
        # The pair is offset perpendicular to the line, so the two sit beside
        # each other at the same slope and never touch.
        perp = torch.tensor([-math.sin(phi), math.cos(phi), 0.0])
        gap = perp * PAIR_GAP

        half = direction * LINE_LENGTH / 2
        Line(
            centre + gap - half,
            centre + gap + half,
            color=WHITE,
            scene=scene,
        ).spawn()

        cylinder = Cylinder(
            radius=CYLINDER_RADIUS,
            height=LINE_LENGTH,
            direction=direction,
            color=WHITE,
            scene=scene,
        )
        cylinder.set_material(MeshBasicMaterial(color=WHITE))
        cylinder.move_to(centre - gap)
        cylinder.spawn()

        label = Text(f"{angle:g}", scene=scene, font_size=15, color=WHITE)
        label.move_to(centre + torch.tensor([-1.7, 0.0, 0.0]))
        label.spawn()

    caption = Text(
        "each pair: bezier Line (upper) and Cylinder of triangles (lower)",
        scene=scene,
        font_size=15,
        color=WHITE,
    )
    caption.move_to(torch.tensor([0.0, -3.2, 0.0]))
    caption.spawn()


def magnify(src, dst, box, factor=6):
    """Write a nearest-neighbour magnification of one crop of the frame."""
    import cv2

    image = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    x, y, w, h = box
    crop = image[y : y + h, x : x + w]
    big = cv2.resize(crop, (w * factor, h * factor), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(dst), big)
    return big.shape


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--res", default="hd", choices=("ld", "md", "hd"))
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    from algan.constants.color import BLACK
    from algan.scene import Scene
    from algan.settings.video_settings import HD, LD, MD

    video_settings = {"ld": LD, "md": MD, "hd": HD}[args.res]
    out_dir = Path(args.out) if args.out else REPO_ROOT / "algan_outputs" / "aa_check"
    out_dir.mkdir(parents=True, exist_ok=True)

    scene = Scene(video_settings=video_settings)
    build(scene)
    path = out_dir / "aa_fan.png"
    scene.save_frame(str(path), video_settings, background_color=BLACK)
    print(f"wrote {path}")

    width, height = video_settings.resolution
    # The 2-degree pair: the shallowest slope in the fan, where a line advances
    # a thirtieth of a pixel per column and aliasing is most visible.
    scale = height / 7.0  # the default camera frame is 7 world units tall
    cx, cy = _cell_centre(0)
    box = (
        int(width / 2 + cx * scale - 110),
        int(height / 2 - cy * scale - 70),
        220,
        140,
    )
    zoom = out_dir / "aa_fan_zoom.png"
    magnify(path, zoom, box)
    print(f"wrote {zoom}  (6x nearest-neighbour crop of {box})")
    return path


if __name__ == "__main__":
    main()
