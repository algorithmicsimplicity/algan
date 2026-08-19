"""fc0f93f (UNCONDITIONAL lane reads): materials_and_lighting OFF vs ON,
rendered losslessly. The record says this configuration moved the scene by 42
channel values over 28,854 pixels (16% of a frame) -- a decoded-H.264 number.
This measures what the renderer itself changed."""

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
OUT = FULL / "algan_outputs" / "_c3_uncond"


def render(run_exact, out_name):
    rt_settings.set_analytic_aa(True, run_exact=run_exact)
    SceneManager.reset()
    with Scene() as scene:
        sp = importlib.util.spec_from_file_location(
            "_sc", FULL / "scenes" / "materials_and_lighting.py"
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
    OUT.mkdir(parents=True, exist_ok=True)
    os.chdir(FULL)
    SETTINGS.paths.set(
        output_root=str(FULL),
        output_directory="algan_outputs/_c3_uncond",
        cache_directory=str(FULL / "algan_cache"),
    )
    SETTINGS.computing.set(available_memory_override=1536 * 1024 * 1024)
    render(False, "materials_off_ll.mp4")
    render(True, "materials_on_ll.mp4")

    import cv2
    import numpy as np

    a = cv2.VideoCapture(str(OUT / "materials_off_ll.mp4"))
    b = cv2.VideoCapture(str(OUT / "materials_on_ll.mp4"))
    i = 0
    nz = []
    total = 0
    worst = 0
    while True:
        oa, fa = a.read()
        ob, fb = b.read()
        if not oa or not ob:
            break
        d = np.abs(fa.astype(np.int16) - fb.astype(np.int16))
        m = int(d.max())
        c = int((d.max(axis=2) > 0).sum())
        if m > 0:
            nz.append((i, m, c))
        total += c
        worst = max(worst, m)
        i += 1
    print(f"LOSSLESS unconditional ON vs OFF (materials): "
          f"{len(nz)} nonzero frames of {i}, {total} moved pixel-frames, "
          f"worst |d| {worst}")
    for f, m, c in nz[:20]:
        print(f"  frame {f:3d} max {m:2d} px {c}")


if __name__ == "__main__":
    main()
