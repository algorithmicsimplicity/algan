"""Visual check: border geometry across filled/unfilled circuits and text.

Renders one frame with
  * a filled square with a visible fill AND a contrasting wide border
    (exercises the border/fill blend, not just border-over-nothing),
  * the same square with border_width 0 (must be unchanged by this work),
  * an unfilled circle and a Line (centred strokes, unchanged),
  * dense text (must not fuse as border_width rises).

    .venv/Scripts/python.exe benchmarks/_bez_border_shapes_check.py [name]
"""

from __future__ import annotations

import os
import sys

NAME = sys.argv[1] if len(sys.argv) > 1 else "shapes"
if NAME.endswith("_super"):
    os.environ["ALGAN_ANALYTIC_AA"] = "0"

from algan import *  # noqa: E402,F403


def main():
    with Off():
        for i, bw in enumerate((0, 4, 12)):
            (
                Square(color=BLUE, border_color=RED, border_width=bw)
                .spawn()
                .scale(0.8)
                .move(LEFT * 5 + UP * 2.5 + RIGHT * 3.2 * i)
            )
        Circle(
            color=GREEN, border_color=YELLOW, border_width=8, filled=False
        ).spawn().scale(0.8).move(RIGHT * 5 + UP * 2.5)
        Line(LEFT * 2, RIGHT * 2, color=WHITE, border_width=8).spawn().move(
            LEFT * 3 + DOWN * 1
        )
        Text("minimum", border_color=RED, border_width=6).spawn().scale(2).move(
            DOWN * 2.5
        )
    Scene.wait()
    out = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "_bez_border_aa_out", NAME
    )
    Scene.save_frame(out, PREVIEW)
    print("wrote", out + ".png")


main()
