"""Debug-scene A/B for the opaque any-hit shadow early-out.

Renders the full 174-frame debug scene (materials zoo + lights + text;
contains transmissive glass, so batches select deferred mode 2) alternating
ALGAN_SHADOW_ANYHIT off/on within each round. Reports per-arm times (round
1 discarded -- kernel variants compile there), the sha256 of each arm's
last mp4, and the speedup.

The ON arm's mode is selectable: ``1`` = any-hit walks (deferred mode 2 on
this scene), ``gather`` (default) = kbuf gather-march (mode 4).

Usage:
    .venv/Scripts/python.exe benchmarks/_shadow_anyhit_debug_scene.py [reps] [1|gather]
"""

import hashlib
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _debug_scene_profile import build_scene  # noqa: E402

from algan import *  # noqa: F403,E402
from algan.rendering.raytracing import settings as rt_settings  # noqa: E402
from algan.scene_manager import SceneManager  # noqa: E402

OUT_DIR = os.path.join("algan_outputs", "profiling")
PINNED_BYTES = 2_400_000_000


def run_arm(tag, anyhit):
    rt_settings.set_shadow_anyhit(anyhit)
    SceneManager.reset()
    SETTINGS.computing.set(available_memory_override=PINNED_BYTES)
    scene = SceneManager.instance().current_scene
    scene.set_video_settings(PREVIEW)
    build_scene()
    path = os.path.join(OUT_DIR, f"anyhit_debug_{tag}.mp4")
    t0 = time.perf_counter()
    Scene.save_video(path, PREVIEW, overwrite=True)
    dt = time.perf_counter() - t0
    return path, dt


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    on_mode = sys.argv[2] if len(sys.argv) > 2 else "gather"
    on_value = "gather" if on_mode == "gather" else True
    print(f"ON arm mode: {on_value!r}", flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    t_off, t_on = [], []
    for r in range(reps):
        p_off, dt = run_arm("off", False)
        t_off.append(dt)
        print(f"round {r}: off {dt:7.2f}s", flush=True)
        p_on, dt = run_arm("on", on_value)
        t_on.append(dt)
        print(f"round {r}: on  {dt:7.2f}s", flush=True)
    sha_off = hashlib.sha256(open(p_off, "rb").read()).hexdigest()
    sha_on = hashlib.sha256(open(p_on, "rb").read()).hexdigest()
    keep_off = t_off[1:] if len(t_off) > 1 else t_off
    keep_on = t_on[1:] if len(t_on) > 1 else t_on
    print(
        f"DEBUG-SCENE anyhit A/B: sha_equal={sha_off == sha_on} "
        f"off={min(keep_off):7.2f}s on={min(keep_on):7.2f}s "
        f"speedup={min(keep_off) / min(keep_on):5.2f}x",
        flush=True,
    )
    print(f"  sha_off={sha_off}")
    print(f"  sha_on ={sha_on}")


if __name__ == "__main__":
    main()
