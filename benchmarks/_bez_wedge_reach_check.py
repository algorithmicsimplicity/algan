"""Regression check for the oriented wedge's REACH (DESIGN_analytic_aa_v2.md ss5.6).

The wedge models a circuit's local boundary with the two nearest flattened
walls. Those walls are only a boundary if they are close enough for the
coverage filter to read; two half-planes carry no notion of where their
segments end, so a pixel lying between the EXTENDED lines of a glyph's two long
diagonals -- the empty notch between the arms of an ``A`` -- reads as deeply
inside both and is painted at full coverage. That is a lone opaque speck in the
middle of nothing, and it shipped in all six full-render scenes until
``_bezier_point_metrics`` started applying its own ``query_radius`` to the
candidates the spatial grid hands back.

What this asserts, on a rendered frame of glyphs known to have interior
notches: no DETACHED ink. Every lit pixel must be reachable from a pixel the
glyph actually covers, so a speck standing alone in the background fails
whatever produced it. That is a property of the picture rather than of the
kernel's internals, so it keeps holding across coverage changes that legitimately
move edge pixels.

Run: .venv/Scripts/python.exe benchmarks/_bez_wedge_reach_check.py
"""

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# A warm daemon carries the previous run's renderer state and, worse here, its
# import-time settings; a check must render on its own terms.
os.environ.setdefault("ALGAN_USE_DAEMON", "0")

import cv2  # noqa: E402
import manimpango  # noqa: E402
import numpy as np  # noqa: E402

for _face in sorted((REPO / "tests" / "assets" / "fonts").glob("*.ttf")):
    manimpango.register_font(str(_face))

from algan import (  # noqa: E402
    DARKER_GRAY,
    DOWN,
    PREVIEW,
    PURPLE_A,
    SETTINGS,
    WHITE,
    AmbientLight,
    Group,
    Off,
    Scene,
    Text,
)

FONT = "Algan Test Sans"
# The letters whose outer contour is two long strokes with an empty region
# between them: exactly the configuration that puts a pixel inside both walls'
# extended lines while parity says outside.
WORDS = ("IMAGE", "AVATAR", "WXYZ", "MAKA")
OUT_DIR = REPO / "benchmarks" / "_bez_wedge_reach_out"


def render() -> np.ndarray:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SETTINGS.paths.set(output_root=str(OUT_DIR), output_directory=".")
    SETTINGS.computing.set(available_memory_override=1536 * 1024 * 1024)
    path = OUT_DIR / "reach.mp4"
    with Scene() as scene:
        Scene.set_background_color(DARKER_GRAY)
        with Off():
            AmbientLight(color=WHITE, intensity=0.55).spawn(animate=False)
            rows = Group(
                *(
                    Text(w, font_size=19, color=PURPLE_A, font=FONT).move(
                        DOWN * (i * 0.7 - 1.0)
                    )
                    for i, w in enumerate(WORDS)
                )
            )
            rows.spawn()
        Scene.wait(0.2)
        scene.save_video(
            str(path),
            video_settings=PREVIEW,
            overwrite=True,
            codec="libx264rgb",
            ffmpeg_params=["-crf", "0", "-preset", "fast"],
        )
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read a frame back from {path}")
    return frame


def detached_ink(frame: np.ndarray) -> list[tuple[int, int, int]]:
    """Lit connected components that no fully-covered pixel belongs to.

    A glyph is one solid run of ink with anti-aliased skirts, so its component
    contains pixels at (or near) full coverage. A wedge speck is a component of
    one or two pixels that is full-coverage AND touches nothing -- and an
    isolated skirt pixel from a legitimately faint stroke is not, which is why
    the test is "component with no faint member", not "small component".
    """
    bg = frame[0, 0].astype(int)
    delta = np.abs(frame.astype(int) - bg).max(2)
    lit = (delta > 8).astype(np.uint8)
    n, labels = cv2.connectedComponents(lit, connectivity=8)
    solid = delta >= int(delta.max() * 0.9)
    bad = []
    for c in range(1, n):
        mask = labels == c
        size = int(mask.sum())
        # A component made ENTIRELY of near-full-coverage pixels and small
        # enough to be no glyph part is detached ink.
        if size <= 4 and bool(solid[mask].all()):
            ys, xs = np.where(mask)
            bad.append((int(xs[0]), int(ys[0]), size))
    return bad


def main() -> int:
    frame = render()
    bad = detached_ink(frame)
    if bad:
        print(f"FAIL: {len(bad)} detached full-coverage component(s):")
        for x, y, size in bad:
            print(f"  at ({x},{y}), {size} px")
        print(
            "  A speck standing alone in the background is the wedge reading two "
            "far walls as a local boundary -- see DESIGN_analytic_aa_v2.md ss5.6."
        )
        return 1
    print(f"PASS: no detached ink over {len(WORDS)} words of glyphs with notches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
