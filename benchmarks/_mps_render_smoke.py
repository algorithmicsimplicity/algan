"""One frame through the whole renderer, on whatever device is configured.

The smallest thing that answers "does Algan render here at all". It exists for
the Apple GPU (``DESIGN_mps_support.md``), where the interesting failures are
not assertion failures: Metal answers an over-wide kernel with ``computeFunction
must not be nil`` and an int64 atomic with ``bind_pipeline`` -- both ``SIGABRT``
inside Taichi rather than exceptions Python can catch -- so a run that dies
without a traceback is itself the result. Running this before the test suite in
``.github/workflows/mps_probe.yaml`` costs a minute and separates "the renderer
aborts on this backend" from "one test disagrees about a pixel", which the
suite's output otherwise blends together.

It renders through ``save_frame``, so it needs no encoder and no LaTeX; the
geometry is chosen to reach the parts MPS-friendly mode touches -- overlapping
opaque solids (the one-mesh coverage ceiling and its facing-split sums), a
bezier circuit (the raster path), and a shadow (the sheet compaction).

    uv run python benchmarks/_mps_render_smoke.py

Prints what it resolved, writes ``mps_render_smoke/frame.png``, and exits
non-zero if the frame is empty, uniform, or not finite.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("ALGAN_USE_DAEMON", "0")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from algan import LD, OUTWARD, RIGHT, UP, Off, Scene, Sphere, Square  # noqa: E402
from algan.rendering.mps_compat import (  # noqa: E402
    accumulate_dtype,
    mps_friendly,
    reduction_index_dtype,
)
from algan.settings import SETTINGS  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "mps_render_smoke"


def main() -> int:
    print(f"render device    : {SETTINGS.computing.render_device}")
    print(f"mps_friendly     : {SETTINGS.computing.mps_friendly} -> {mps_friendly()}")
    print(f"accumulate dtype : {accumulate_dtype()}")
    print(f"reduction dtype  : {reduction_index_dtype()}")
    print(f"torch            : {torch.__version__}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame_path = OUTPUT_DIR / "frame.png"
    frame_path.unlink(missing_ok=True)

    with Scene() as scene:
        with Off():
            Square(size=2.4).spawn()
            Sphere(radius=0.65).move(RIGHT * 1.1 + UP * 0.35).spawn()
            Sphere(radius=0.5).move(OUTWARD * 1.2 + RIGHT * 0.3).spawn()
        scene.save_frame(str(frame_path), video_settings=LD)

    if not frame_path.exists() or frame_path.stat().st_size == 0:
        print("FAIL: no frame was written")
        return 1

    frame = np.asarray(Image.open(frame_path).convert("RGB")).astype(np.float64)
    print(f"frame            : {frame.shape} min {frame.min()} max {frame.max()}")
    if not np.isfinite(frame).all():
        print("FAIL: the frame is not finite")
        return 1
    if frame.max() == frame.min():
        print("FAIL: the frame is a single flat colour -- nothing was drawn")
        return 1
    # A frame of pure background would pass the test above on the vignette
    # alone, so ask for real ink: distinct values over a real share of it.
    if len(np.unique(frame.astype(np.uint8))) < 16:
        print("FAIL: the frame holds too few distinct values to be a render")
        return 1

    print(f"OK: rendered {frame_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
