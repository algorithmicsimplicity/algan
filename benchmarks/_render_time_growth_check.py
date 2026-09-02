"""Does a REAL multi-batch render grow the timeline / re-resolve every batch?

A probe that calls ``_get_batch_of_primitives`` directly -- which is how every
prep-side measurement in this repo is taken -- runs *outside* the context
management a render sets up, and records new function applications on every
call. That is a harness artifact. This checks the same quantities inside a
genuine ``save_video`` split across several batches:

  * function applications recorded during the render;
  * ``_resolve_replay_windows`` invocations (it re-runs only when something was
    recorded, so this is the observable consequence);
  * ``FunctionTimeline._windows`` full rebuilds vs incremental extends.

Those three drive P4's cache design and P5's in-place checkpoint growth, so it
matters whether they are provoked by real renders or only by the harness.

    .venv/Scripts/python.exe benchmarks/_render_time_growth_check.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import algan.animation_timeline.timeline as tl  # noqa: E402
from algan import (  # noqa: E402
    LEFT,
    RIGHT,
    SETTINGS,
    SMOKE_TEST,
    UP,
    Cylinder,
    Scene,
    Square,
    Text,
)
from algan.scene_manager import SceneManager  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_replay_rec_out")

STATS = {"resolve": 0, "rebuild": 0, "extend": 0, "hit": 0, "batches": 0}


def instrument(ft):
    resolve = tl.AnimationTimeline._resolve_replay_windows
    windows = tl.FunctionTimeline._windows
    sst = tl.AnimationTimeline.set_state_to_times

    def probed_resolve(self):
        if not self._replay_windows_resolved:
            STATS["resolve"] += 1
        return resolve(self)

    def probed_windows(self):
        cache = self._window_cache
        n = len(self.function_applications)
        if cache is None:
            STATS["rebuild"] += 1
        elif cache[0] == n and cache[3] is not None:
            STATS["hit"] += 1
        else:
            STATS["extend"] += 1
        return windows(self)

    def probed_sst(self, times, active_mobs=None):
        STATS["batches"] += 1
        STATS.setdefault("counts", []).append(len(ft.function_applications))
        return sst(self, times, active_mobs=active_mobs)

    tl.AnimationTimeline._resolve_replay_windows = probed_resolve
    tl.FunctionTimeline._windows = probed_windows
    tl.AnimationTimeline.set_state_to_times = probed_sst
    return resolve, windows, sst


def main():
    os.makedirs(OUT, exist_ok=True)
    # Force several batches out of a short scene.
    SETTINGS.computing.set(max_animation_batch_size=8)
    scene = SceneManager.reset()
    scene.set_video_settings(SMOKE_TEST)

    bar = Cylinder(radius=0.15).spawn()
    label = Text("growth").spawn()
    box = Square().spawn()
    for _ in range(3):
        bar.set_start_point(LEFT * 2)
        bar.set_end_point(RIGHT * 2)
        bar.set_start_point(LEFT + UP)
        box.move(RIGHT)
        label.move(UP * 0.2)

    ft = scene.timeline_manager.function_timeline
    before = len(ft.function_applications)
    originals = instrument(ft)
    try:
        Scene.save_video(os.path.join(OUT, "growth.mp4"))
    finally:
        (
            tl.AnimationTimeline._resolve_replay_windows,
            tl.FunctionTimeline._windows,
            tl.AnimationTimeline.set_state_to_times,
        ) = originals
    after = len(ft.function_applications)

    print(f"batches prepared:            {STATS['batches']}")
    print(f"function applications:       {before} before -> {after} after")
    print(f"  per-batch counts:          {STATS.get('counts')}")
    print(f"_resolve_replay_windows runs:{STATS['resolve']:>4}")
    print(
        f"_windows cache:              {STATS['rebuild']} full rebuilds, "
        f"{STATS['extend']} extends, {STATS['hit']} hits"
    )
    assert STATS["batches"] > 1, "need a multi-batch render for this to mean anything"
    if after == before and STATS["resolve"] <= 1:
        print(
            "\nA real render records nothing and resolves once. The per-batch\n"
            "growth seen by direct _get_batch_of_primitives probes is a HARNESS\n"
            "ARTIFACT -- those calls run outside the render's context management."
        )
    else:
        print(
            f"\nA real render DOES record ({after - before} events) and re-resolves\n"
            f"{STATS['resolve']} times, so the per-batch cost is real."
        )


if __name__ == "__main__":
    main()
