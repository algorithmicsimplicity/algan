"""Byte-identity parity check for the full debug/debug.py scene.

Renders the complete 174-frame scene at PREVIEW with the free-VRAM query
pinned (deterministic batch splits) and prints the output mp4's SHA256 plus
per-run wall time. Run once on the baseline to record the reference hash, and
once after a change; equal hashes = byte-identical output video.

Usage:
    .venv/Scripts/python.exe benchmarks/_debug_scene_parity.py [tag]
"""

import hashlib
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _debug_scene_profile import build_scene  # noqa: E402

from algan import *  # noqa: F403
from algan.scene_manager import SceneManager

OUT_DIR = os.path.join("algan_outputs", "profiling")

# Pin the measured free-VRAM figure so the arena (and therefore every batch
# split) is identical run to run; render output is not split-invariant.
PINNED_BYTES = 2_400_000_000


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "run"
    pinned = int(sys.argv[2]) if len(sys.argv) > 2 else PINNED_BYTES
    os.makedirs(OUT_DIR, exist_ok=True)
    SETTINGS.computing.set(available_memory_override=pinned)
    scene = SceneManager.reset()
    scene.set_video_settings(PREVIEW)
    build_scene()
    path = os.path.join(OUT_DIR, f"parity_{tag}.mp4")
    t0 = time.perf_counter()
    Scene.save_video(path, PREVIEW, overwrite=True)
    dt = time.perf_counter() - t0
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    print(f"PARITY {tag}: {dt:8.2f}s  sha256={digest}", flush=True)


if __name__ == "__main__":
    main()
