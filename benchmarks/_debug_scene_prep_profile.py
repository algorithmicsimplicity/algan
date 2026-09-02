"""cProfile the batch-prep (CPU) side of the debug scene's heavy window.

Runs entirely on the CPU (ALGAN_RENDER_DEVICE=cpu is set before import, so no
VRAM is touched and this can run beside a GPU render). Authors the full debug
scene, then profiles `_get_batch_of_primitives` over the expensive frame
window 15..71 -- exactly the work the prefetch worker does per batch.
"""

import cProfile
import io
import os
import pstats
import sys
import time

os.environ.setdefault("ALGAN_RENDER_DEVICE", "cpu")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _debug_scene_profile import build_scene  # noqa: E402

from algan import *  # noqa: F403
from algan.scene_manager import SceneManager


def main():
    scene = SceneManager.reset()
    scene.set_video_settings(PREVIEW)
    build_scene()
    scene._initialize_frames()
    for light in scene.light_sources:
        light.is_primitive = True
    actors = [scene.camera, scene.camera.screen, *scene.light_sources, *scene.actors]

    # _batch_prep_context is what the render loop puts around its batch loop.
    # Without it this profiles a different code path from the one it is meant
    # to measure: prep outside that context records new events on every replay
    # (see Scene._batch_prep_context), which re-resolves replay windows and
    # invalidates the event-window caches every call. A render does neither.
    with scene._batch_prep_context():
        # Warm one small window first (caches, JIT-ish paths), then profile the
        # heavy window like the render loop's worker would prepare it.
        scene._get_batch_of_primitives(0, 15, actors, 10**12)

        t0 = time.perf_counter()
        profiler = cProfile.Profile()
        profiler.enable()
        scene._get_batch_of_primitives(15, 71, actors, 10**12)
        profiler.disable()
    dt = time.perf_counter() - t0
    print(f"_get_batch_of_primitives(15, 71): {dt:.2f}s")

    out = io.StringIO()
    stats = pstats.Stats(profiler, stream=out)
    stats.sort_stats("cumulative").print_stats(45)
    print(out.getvalue())
    out2 = io.StringIO()
    stats2 = pstats.Stats(profiler, stream=out2)
    stats2.sort_stats("tottime").print_stats(35)
    print(out2.getvalue())


if __name__ == "__main__":
    main()
