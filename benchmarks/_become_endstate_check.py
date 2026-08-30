"""Does a finished `become` render as the target would?

The external invariant, checked the way this repo checks everything else --
against pixels.  For each pair we render two single frames at the same time and
settings:

  A: `source.become(target)`, sampled at the instant the morph ends.
  B: `target` alone, spawned, sampled at the same instant.

If `become` finished, A and B are the same picture.  Any channel difference is
a property the morph did not carry across.  The tolerance is the repo's own
(> 2 channel values fails), plus a count of how many pixels move at all.

Usage:  <venv-python> benchmarks/_become_endstate_check.py [--pairs a:b,c:d]
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

from algan import (
    BLUE,
    GREEN,
    LD,
    LEFT,
    RIGHT,
    UP,
    YELLOW,
    Arrow,
    Circle,
    Cube,
    Cylinder,
    Group,
    Line,
    Off,
    Polyhedron,
    Scene,
    Sphere,
    Square,
    Star,
    Sync,
    Tetrahedron,
    Text,
    Torus,
    Triangle,
)


def _polyhedron():
    vertices = torch.tensor(
        [
            [0.0, 0.6, 0.0],
            [-0.5, -0.3, 0.4],
            [0.5, -0.3, 0.4],
            [0.0, -0.3, -0.6],
        ]
    )
    return Polyhedron(vertices, [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]])


BUILDERS = {
    "Square": lambda: Square(color=BLUE),
    "SquareUnfilled": lambda: Square(color=BLUE, filled=False, stroke_width=0.05),
    "Circle": lambda: Circle(radius=0.7, color=YELLOW),
    "CircleBorder": lambda: Circle(radius=0.7, color=YELLOW, stroke_width=0.08),
    "Star": lambda: Star(color=GREEN),
    "Triangle": lambda: Triangle(color=GREEN),
    "Line": lambda: Line(LEFT, RIGHT),
    "Arrow": lambda: Arrow(LEFT, RIGHT),
    "Text": lambda: Text("hi"),
    "Sphere": lambda: Sphere(radius=0.8),
    "SphereGlow": lambda: Sphere(radius=0.8, glow=0.6),
    "Cylinder": lambda: Cylinder(radius=0.5, height=1.2),
    "Torus": lambda: Torus(),
    "Cube": lambda: Cube(size=1.0),
    "Tetrahedron": lambda: Tetrahedron(),
    "Polyhedron": _polyhedron,
    "Group2": lambda: Group(
        Square(color=BLUE).move(LEFT), Circle(color=YELLOW).move(RIGHT)
    ),
    "Group3": lambda: Group(
        Square(color=BLUE).move(LEFT * 1.5),
        Sphere(radius=0.4),
        Triangle(color=GREEN).move(RIGHT * 1.5),
    ),
    "SphereTranslucent": lambda: Sphere(radius=0.8, opacity=0.55),
    "SurfacePlane": lambda: __import__("algan").Surface(
        lambda u, v: torch.stack((u - 0.5, v - 0.5, torch.zeros_like(u)), -1),
        grid_width=6,
        grid_height=6,
    ),
    "SurfaceWaveCoarse": lambda: __import__("algan").Surface(
        lambda u, v: torch.stack(
            (u - 0.5, v - 0.5, 0.25 * torch.sin(6 * u) * torch.cos(6 * v)), -1
        ),
        grid_width=4,
        grid_height=9,
    ),
    "CrossedLines": lambda: __import__("algan").VGroup(
        Line(LEFT, RIGHT), Line(UP, UP * -1)
    ),
    "CrossedLinesGroup": lambda: Group(Line(LEFT, RIGHT), Line(UP, UP * -1)),
    "Arrow3D": lambda: __import__("algan").Arrow3D(),
    "DotCloud": lambda: __import__("algan").DotCloud(
        points=torch.tensor([[-0.6, 0.0, 0.0], [0.0, 0.5, 0.0], [0.6, -0.3, 0.0]])
    ),
    "Cross": lambda: __import__("algan").Cross(),
    "VGroupTwoSquares": lambda: __import__("algan").VGroup(
        Square(color=BLUE).move(LEFT), Square(color=BLUE).move(RIGHT)
    ),
}

DEFAULT_PAIRS = [
    ("Square", "Circle"),
    ("Square", "SquareUnfilled"),
    ("SquareUnfilled", "Square"),
    ("Circle", "CircleBorder"),
    ("CircleBorder", "Circle"),
    ("Square", "Star"),
    ("Text", "Square"),
    ("Square", "Text"),
    ("Square", "Sphere"),
    ("Sphere", "Square"),
    ("Sphere", "Cylinder"),
    ("Cylinder", "Sphere"),
    ("Sphere", "Torus"),
    ("Cube", "Sphere"),
    ("Sphere", "Cube"),
    ("Cube", "Tetrahedron"),
    ("Polyhedron", "Cube"),
    ("Sphere", "SphereGlow"),
    ("SphereGlow", "Sphere"),
    ("Sphere", "SphereTranslucent"),
    ("Group2", "Group3"),
    ("Group3", "Group2"),
    ("Square", "Group2"),
    ("Group2", "Square"),
    ("Line", "Arrow"),
    ("Arrow", "Line"),
]

OUTPUT = Path("algan_outputs/_become_endstate")


def _frame(path):
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"could not read {path}")
    return image.astype(np.int32)


def render_morph_end(source_name, target_name, path):
    with Scene() as scene:
        scene.use_manim_defaults()
        with Off():
            source = BUILDERS[source_name]().spawn()
            target = BUILDERS[target_name]()
        with Sync(duration=1.0):
            source.become(target)
        end = float(scene.animation_manager.context.timespan.current_time)
        scene.save_frame(str(path), LD, at=end - 1e-4, overwrite=True)


def render_target_only(target_name, path):
    with Scene() as scene:
        scene.use_manim_defaults()
        with Off():
            BUILDERS[target_name]().spawn()
        with Sync(duration=1.0):
            pass
        end = float(scene.animation_manager.context.timespan.current_time)
        scene.save_frame(str(path), LD, at=end - 1e-4, overwrite=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default=None)
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()

    if args.pairs:
        pairs = [tuple(item.split(":")) for item in args.pairs.split(",")]
    else:
        pairs = DEFAULT_PAIRS

    OUTPUT.mkdir(parents=True, exist_ok=True)
    failures = 0
    for source_name, target_name in pairs:
        tag = f"{source_name}__{target_name}"
        morph_path = OUTPUT / f"{tag}_morph.png"
        target_path = OUTPUT / f"{tag}_target.png"
        try:
            render_morph_end(source_name, target_name, morph_path)
            render_target_only(target_name, target_path)
            a = _frame(morph_path)
            b = _frame(target_path)
            if a.shape != b.shape:
                print(f"SHAPE   {tag}: {a.shape} vs {b.shape}")
                failures += 1
                continue
            difference = np.abs(a - b)
            peak = int(difference.max())
            moved = int((difference.max(axis=2) > 2).sum())
            total = a.shape[0] * a.shape[1]
            verdict = "ok     " if peak <= 2 else "DIFFERS"
            if peak > 2:
                failures += 1
            print(
                f"{verdict} {tag:<34} peak={peak:>4}  "
                f"pixels>2: {moved:>6} / {total} ({100 * moved / total:.2f}%)"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR   {tag}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            failures += 1
        sys.stdout.flush()
    print(f"\n{len(pairs) - failures} of {len(pairs)} pairs land on the target")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
