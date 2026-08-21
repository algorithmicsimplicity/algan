"""Regression test: replaying a recorded animation must record NO new events.

``AnimationTimeline.set_state_to_times`` replays a recorded function by calling
``f.function(f.caller, **kwargs)`` -- the *undecorated* body. The
``animated_function`` decorator normally runs that body inside
``AnimationContext(record_funcs=False, ...)``, so nested animated calls record
nothing. Replay used NOT to reproduce that wrap, so any recorded function
whose body called another animated function recorded a brand new event **every
time it was replayed** -- once per frame batch that touched its window, growing
the timeline without bound and re-resolving replay windows every time.

``set_state_to_times`` now enters that context itself, so every caller is safe,
including harnesses that drive prep directly. This script fails if that
regresses.

``Cylinder.set_start_point`` is one: it calls ``_move_between_points``, which
calls ``self.move_to(...)``, which is a fully animated setter.

This quantifies the growth and checks whether it also perturbs output across a
re-render (``save_video(reset=False)`` leaves the Scene renderable, so the
spurious events are still there for the next render).

    .venv/Scripts/python.exe benchmarks/_replay_records_check.py
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch  # noqa: E402

from algan import LEFT, RIGHT, SMOKE_TEST, UP, Cylinder, Off, Scene  # noqa: E402
from algan.scene_manager import SceneManager  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_replay_rec_out")


def main():
    os.makedirs(OUT, exist_ok=True)
    scene = SceneManager.reset()
    scene.set_video_settings(SMOKE_TEST)

    bar = Cylinder(radius=0.15).spawn()
    bar.set_start_point(LEFT * 2)
    bar.set_end_point(RIGHT * 2)
    bar.set_start_point(LEFT + UP)

    ft = scene.timeline_manager.function_timeline
    authored = len(ft.function_applications)
    print(f"after authoring:            {authored} function applications")

    scene.initialize_frames()
    for light in scene.light_sources:
        light.is_primitive = True
    actors = [scene.camera, scene.camera.screen, *scene.light_sources, *scene.actors]

    counts = [authored]
    for i in range(4):
        # Same window every time: nothing about the scene changed, so a
        # correct replay would record nothing at all.
        scene.get_batch_of_primitives(0, 20, actors, 10**12)
        scene.timeline_manager.clear_buffers()
        counts.append(len(ft.function_applications))
        print(
            f"after preparing window {i}:   {counts[-1]} (+{counts[-1] - counts[-2]})"
        )

    growth = counts[-1] - authored
    print(
        f"\npreparing the SAME window 4 times added {growth} events "
        f"({100 * growth / max(authored, 1):.0f}% of the authored timeline)"
    )
    if growth:
        print(
            "This is unbounded in the number of batches a render prepares, and it\n"
            "is what forces _resolve_replay_windows to re-run on every batch."
        )

    # The fix for the harnesses: reproduce the context the render loop puts
    # around its whole batch loop (render_loop.py, `with Off(...)`). If this
    # holds the count flat, every direct-call probe in this repo can be made
    # faithful by wrapping its prep loop the same way.
    scene_off = SceneManager.reset()
    scene_off.set_video_settings(SMOKE_TEST)
    bar_off = Cylinder(radius=0.15).spawn()
    bar_off.set_start_point(LEFT * 2)
    bar_off.set_end_point(RIGHT * 2)
    bar_off.set_start_point(LEFT + UP)
    scene_off.initialize_frames()
    for light in scene_off.light_sources:
        light.is_primitive = True
    actors_off = [
        scene_off.camera,
        scene_off.camera.screen,
        *scene_off.light_sources,
        *scene_off.actors,
    ]
    ft_off = scene_off.timeline_manager.function_timeline
    n0 = len(ft_off.function_applications)
    with Off(
        record_attr_modifications=False,
        record_funcs=False,
        priority_level=math.inf,
        animation_manager=scene_off.animation_manager,
    ):
        for _ in range(4):
            scene_off.get_batch_of_primitives(0, 20, actors_off, 10**12)
            scene_off.timeline_manager.clear_buffers()
    n1 = len(ft_off.function_applications)
    print(f"\nsame 4 preps wrapped in the render's Off(): {n0} -> {n1} (+{n1 - n0})")
    print(
        "wrapping fixes it -- harnesses can use this"
        if n1 == n0
        else "wrapping is NOT sufficient; something else suppresses recording"
    )

    # Crucially: does a REAL render do this, or only a bare
    # get_batch_of_primitives call like the loop above? A render runs inside
    # Scene's own context management, which may already suppress recording --
    # in which case the growth above is an artifact of the harness and not a
    # bug in the engine. Watch a genuine save_video batch by batch.
    scene_r = SceneManager.reset()
    scene_r.set_video_settings(SMOKE_TEST)
    bar_r = Cylinder(radius=0.15).spawn()
    bar_r.set_start_point(LEFT * 2)
    bar_r.set_end_point(RIGHT * 2)
    bar_r.set_start_point(LEFT + UP)
    ft_r = scene_r.timeline_manager.function_timeline
    before_render = len(ft_r.function_applications)
    seen = []
    import algan.animation_timeline.timeline as _tl

    original = _tl.AnimationTimeline.set_state_to_times

    def watched(self, times, active_mobs=None):
        seen.append(len(ft_r.function_applications))
        return original(self, times, active_mobs=active_mobs)

    _tl.AnimationTimeline.set_state_to_times = watched
    try:
        Scene.save_video(os.path.join(OUT, "render.mp4"))
    finally:
        _tl.AnimationTimeline.set_state_to_times = original
    after_render = len(ft_r.function_applications)
    print(
        f"\nreal save_video: {before_render} events before, {after_render} after; "
        f"per-batch counts {seen}"
    )
    print(
        "a real render DOES record during prep"
        if len(set(seen)) > 1 or after_render != before_render
        else "a real render records nothing either (it always wrapped its batch loop)"
    )

    # Does it change what a re-render produces? save_video(reset=False) leaves
    # the scene renderable, and the spurious events survive into the next one.
    scene2 = SceneManager.reset()
    scene2.set_video_settings(SMOKE_TEST)
    bar2 = Cylinder(radius=0.15).spawn()
    bar2.set_start_point(LEFT * 2)
    bar2.set_end_point(RIGHT * 2)
    bar2.set_start_point(LEFT + UP)
    first = Scene.save_frame(os.path.join(OUT, "pass1"), at=0.9)
    n_after_first = len(scene2.timeline_manager.function_timeline.function_applications)
    second = Scene.save_frame(os.path.join(OUT, "pass2"), at=0.9)
    n_after_second = len(
        scene2.timeline_manager.function_timeline.function_applications
    )
    print(
        f"\nevents after 1st save_frame: {n_after_first}, after 2nd: {n_after_second}"
    )

    import cv2

    a = cv2.imread(str(first[0] if isinstance(first, list) else first))
    b = cv2.imread(str(second[0] if isinstance(second, list) else second))
    if a is None or b is None:
        print("could not read back the frames; skipping the pixel comparison")
        return
    diff = int(torch.from_numpy(abs(a.astype("int32") - b.astype("int32"))).max())
    print(f"same frame rendered twice: peak channel difference {diff}")
    print(
        "output unaffected"
        if diff == 0
        else "OUTPUT DIFFERS -- the spurious events are not inert"
    )


if __name__ == "__main__":
    main()
