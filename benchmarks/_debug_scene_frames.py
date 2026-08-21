"""Render reference PNG frames of the debug scene at fixed timestamps.

Usage:
    .venv/Scripts/python.exe benchmarks/_debug_scene_frames.py <tag>

Writes algan_outputs/profiling/frame_<tag>_t{...}.png for a fixed set of
timestamps covering all four acts, for before/after visual comparison.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _debug_scene_profile import build_scene  # noqa: E402

from algan import *  # noqa: F403
from algan.scene_manager import SceneManager

OUT_DIR = os.path.join("algan_outputs", "profiling")
TIMES = [2.0, 5.5, 10.5, 15.5]


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "run"
    os.makedirs(OUT_DIR, exist_ok=True)
    scene = SceneManager.reset()
    scene.set_video_settings(PREVIEW)
    build_scene()
    for t in TIMES:
        name = os.path.join(OUT_DIR, f"frame_{tag}_t{t:g}.png")
        Scene.save_frame(name, PREVIEW, at=t, overwrite=True)
        print(f"wrote {name}", flush=True)


if __name__ == "__main__":
    main()
