"""A/B for the contiguous replay time-selector (``ALGAN_OPT_DISABLE=timeslice``).

``AnimationTimeline.set_state_to_times`` replays every recorded animated
function for the window, and while a function is replaying, the frame axis of
every attribute read and write goes through ``active_time_inds``.
``AttributeTimeline.get`` / ``modify`` already branch on it -- a ``slice`` reads
a view and writes a slice-assign, an index tensor pays an advanced-index gather
and a scatter -- and outside replay it is ``slice(None)``. The replay window is
an interval over ascending frame times, so the ``.nonzero()`` it used to pass is
contiguous and can be a slice.

This times ``get_batch_of_primitives`` -- the batch-prep worker's whole job --
with the selector on and off, alternating arms in one process because
wall-clock across processes on this machine swings ~2x with thermal state.

**It reports ~1.00x, and that is not the change's value.** Kept as the in-repo
smoke check (it proves both arms still render the same scene without needing the
external video project), but the debug scene is materials/lighting heavy and
barely replays any recorded functions, so there is almost nothing here for the
selector to act on. The measurement that can see it is
``videos/rl2/animations/_prep_timeslice_ab_s05.py`` on the reference scene,
which times the affected methods rather than the whole pass -- whole-prep wall
time on this machine swings 5.9-10.0 s between rounds *on either arm*, several
times the size of the effect. Do not read a ratio out of that column.

Runs on the CPU (``ALGAN_RENDER_DEVICE=cpu``), so it needs no VRAM and can run
beside a render. Not memory-capped: the sizes come from a real authored scene,
not from parameters.

    .venv/Scripts/python.exe benchmarks/_prep_timeslice_ab.py
"""

from __future__ import annotations

import os
import statistics
import sys
import time

os.environ.setdefault("ALGAN_RENDER_DEVICE", "cpu")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from _debug_scene_profile import build_scene  # noqa: E402

import algan.animation_timeline.timeline as tl  # noqa: E402
from algan import PREVIEW  # noqa: E402
from algan.scene_manager import SceneManager  # noqa: E402

WINDOWS = [(0, 40), (40, 90), (90, 140)]
ROUNDS = 3


def _prep(scene, actors, lo, hi):
    # batch_prep_context is what a render puts around its batch loop. Without
    # it a direct call records new events on every replay, which re-resolves
    # replay windows and invalidates the event-window caches every call --
    # none of which a render does, so the measurement would be of the harness.
    with scene.batch_prep_context():
        scene.get_batch_of_primitives(lo, hi, actors, 10**12)


def _arm(disabled):
    tl._OPT_DISABLED = frozenset({"timeslice"} if disabled else set())


def main():
    scene = SceneManager.reset()
    scene.set_video_settings(PREVIEW)
    build_scene()
    scene.initialize_frames()
    for light in scene.light_sources:
        light.is_primitive = True
    actors = [scene.camera, scene.camera.screen, *scene.light_sources, *scene.actors]

    # Warm every window on both arms first: the first pass over a window fills
    # per-mob descendant/range caches that neither arm should be charged for.
    for disabled in (True, False):
        _arm(disabled)
        for lo, hi in WINDOWS:
            _prep(scene, actors, lo, hi)

    samples = {True: [], False: []}
    for _ in range(ROUNDS):
        # Alternate, so a thermal drift over the run lands on both arms.
        for disabled in (True, False):
            _arm(disabled)
            t0 = time.perf_counter()
            for lo, hi in WINDOWS:
                _prep(scene, actors, lo, hi)
            samples[disabled].append(time.perf_counter() - t0)

    off = statistics.median(samples[True])
    on = statistics.median(samples[False])
    print(f"torch {torch.__version__}  windows={WINDOWS}  rounds={ROUNDS}")
    print(
        f"  tensor selector (timeslice OFF): {off * 1e3:8.1f} ms  {[f'{s * 1e3:.0f}' for s in samples[True]]}"
    )
    print(
        f"  slice  selector (timeslice ON ): {on * 1e3:8.1f} ms  {[f'{s * 1e3:.0f}' for s in samples[False]]}"
    )
    print(f"  speedup: {off / max(on, 1e-9):.3f}x")


if __name__ == "__main__":
    main()
