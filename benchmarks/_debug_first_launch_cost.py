"""Measure the per-process first-launch (module load/JIT) overhead.

Renders the same frame of the debug scene twice in one fresh process.
Frame 1 pays every kernel variant's first-launch cost (offline-cache load,
driver JIT, allocator warm-up); frame 2 pays none of it. The difference is
the fixed per-process overhead a plain `python debug/debug.py` run carries.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _debug_scene_profile import build_scene  # noqa: E402

from algan import *  # noqa: F403
from algan.scene_manager import SceneManager

OUT_DIR = os.path.join("algan_outputs", "profiling")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t_import = time.perf_counter()
    scene = SceneManager.reset()
    scene.set_video_settings(PREVIEW)
    build_scene()
    t_author = time.perf_counter()
    print(f"authoring: {t_author - t_import:6.2f}s", flush=True)
    for i in range(3):
        t0 = time.perf_counter()
        Scene.save_frame(
            os.path.join(OUT_DIR, f"first_launch_{i}.png"),
            PREVIEW,
            at=5.0,
            overwrite=True,
        )
        print(f"save_frame #{i}: {time.perf_counter() - t0:6.2f}s", flush=True)


if __name__ == "__main__":
    main()
