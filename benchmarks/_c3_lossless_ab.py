"""The C.3 A/B under a lossless codec: how many pixels REALLY move?

Every diff so far -- and every pixel comparison the render suites make -- reads
back a lossy H.264 yuv420p stream, whose motion-compensated inter frames, 16x16
DCT blocks and 2x2 chroma subsampling all SPREAD a real change into pixels the
renderer never touched. libx264rgb at crf 0 is bit-exact RGB, so this diff is
the renderer's own output.
"""

import importlib.util
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from algan import PREVIEW, SETTINGS, Scene  # noqa: E402
from algan.rendering.raytracing import settings as rt_settings  # noqa: E402
from algan.scene_manager import SceneManager  # noqa: E402

FULL = REPO / "tests" / "full_renders"
OUT = FULL / "algan_outputs" / "_c3_ab"


def render(run_exact, out_name):
    rt_settings.set_analytic_aa(True, run_exact=run_exact)
    SceneManager.reset()
    with Scene() as scene:
        sp = importlib.util.spec_from_file_location(
            "_sc", FULL / "scenes" / "shapes_and_timeline.py"
        )
        mod = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(mod)
        scene.save_video(
            str(OUT / out_name),
            video_settings=PREVIEW,
            overwrite=True,
            animate_fade_out=True,
            codec="libx264rgb",
            ffmpeg_params=["-crf", "0", "-preset", "fast"],
        )


def main():
    conftest = FULL.parent / "conftest.py"
    spec = importlib.util.spec_from_file_location("_cft", conftest)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    os.chdir(FULL)
    SETTINGS.paths.set(
        output_root=str(FULL),
        output_directory="algan_outputs/_c3_ab",
        cache_directory=str(FULL / "algan_cache"),
    )
    SETTINGS.computing.set(available_memory_override=1536 * 1024 * 1024)
    render(False, "shapes_off_ll.mp4")
    render(True, "shapes_on_ll.mp4")

    import cv2
    import numpy as np

    a = cv2.VideoCapture(str(OUT / "shapes_off_ll.mp4"))
    b = cv2.VideoCapture(str(OUT / "shapes_on_ll.mp4"))
    i = 0
    nz = []
    while True:
        oa, fa = a.read()
        ob, fb = b.read()
        if not oa or not ob:
            break
        d = np.abs(fa.astype(np.int16) - fb.astype(np.int16))
        m = int(d.max())
        if m > 0:
            moved = np.nonzero(d.max(axis=2) > 0)
            nz.append(
                (
                    i,
                    m,
                    len(moved[0]),
                    list(zip(moved[1].tolist(), moved[0].tolist()))[:12],
                )
            )
        i += 1
    print(f"LOSSLESS ON vs OFF: {len(nz)} nonzero frames of {i}")
    for f, m, c, px in nz:
        print(f"  frame {f:3d} max {m:2d} px {c:4d}  at {px}")


if __name__ == "__main__":
    main()
