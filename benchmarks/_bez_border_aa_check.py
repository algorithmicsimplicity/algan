"""Probe: anti-aliasing of a bezier circuit's OUTER vs INNER border edge.

Renders the 'e' glyph from ``debug/debug.py`` (fill invisible, opaque border)
under both AA routes and prints the horizontal luminance profile across one
scanline, so the outer (glyph outline) and inner (border -> fill) transitions
can be compared directly.

    .venv/Scripts/python.exe benchmarks/_bez_border_aa_check.py analytic
    .venv/Scripts/python.exe benchmarks/_bez_border_aa_check.py super
"""

from __future__ import annotations

import os
import sys

MODE = sys.argv[1] if len(sys.argv) > 1 else "analytic"
if MODE == "super":
    os.environ["ALGAN_ANALYTIC_AA"] = "0"

import numpy as np  # noqa: E402

from algan import *  # noqa: E402,F403
from algan.animations.manim_animations import _with_opacity  # noqa: E402


def main():
    with Off():
        text = Text("e").spawn().scale(20)
        for mob in text.character_mobs:
            mob.set_opacity_via_color(0)
            mob.border_width = 5
            mob.border_color = _with_opacity(mob.border_color, 1)
    Scene.wait()
    out = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "_bez_border_aa_out", f"{MODE}"
    )
    Scene.save_frame(out, PREVIEW.set(super_sampling_anti_aliasing=4))

    from PIL import Image

    im = np.array(Image.open(out + ".png").convert("L")).astype(float)
    print(f"[{MODE}] image {im.shape}")
    for row in (200, 250, 300):
        vals = im[row]
        runs = [(i, int(v)) for i, v in enumerate(vals) if v > 2]
        print(f"  row {row}: " + " ".join(f"{i}:{v}" for i, v in runs))


main()
