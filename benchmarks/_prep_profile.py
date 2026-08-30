"""Profile ONLY the batch-prep path (set_state_to_times + get_render_primitives
+ collection build) of the bezier_rendering benchmark scene, without rendering.

Usage: .venv/Scripts/python.exe benchmarks/_prep_profile.py [num_batches]
"""

from __future__ import annotations

import io

import manim as mn

from algan import *
from algan.animation_timeline.animation_contexts import AnimationManager
from algan.animation_timeline.timeline import TimelineManager
from algan.mobs.neural_nets.neural_net import NeuralNetMLP
from algan.scene_manager import SceneManager
from algan.settings.defaults import COMPUTING_DEFAULTS
from algan.utils.memory_utils import get_num_available_bytes


def Boxed(mob, color=BLUE, buffer=0.1, *args, **kwargs):
    return Group(
        mob,
        SurroundingRectangle(
            mob,
            *args,
            color=color.lerp(BLACK, 0.8).lerp(PURE_BLUE, 0.1).set_opacity(0.95),
            stroke_color=torch.lerp(color, BLACK, 0.2),
            buffer=buffer,
            stroke_width=1,
            **kwargs,
        ),
    )


def GlowTex(c, *args, **kwargs):
    m = (
        ManimMob(mn.MathTex(*args, **kwargs))
        .set(
            color=c + GLOW * 0.01,
            stroke_color=torch.lerp(c, WHITE, 0.9),
            stroke_width=0.8,
        )
        .scale(0.75)
    )
    return m


text_string = ("a" * 50 + "\n") * 50


def text_scene():
    with Sync(duration=0.25):
        nn = NeuralNetMLP([3, 3, 3]).spawn()
        mob = Boxed(GlowTex(GREEN, text_string)).spawn()
    with Sync(duration=0.25):
        mob.move(LEFT)
        nn.move(LEFT)


def main():
    num_batches = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    scene = SceneManager.reset()
    scene.set_render_settings(HD)
    build_prof = cProfile.Profile()
    t0 = time.perf_counter()
    build_prof.enable()
    text_scene()
    build_prof.disable()
    t_build = time.perf_counter() - t0
    print(f"scene build: {t_build:.2f}s")
    if "--build-profile" in sys.argv:
        s = io.StringIO()
        pstats.Stats(build_prof, stream=s).sort_stats(
            pstats.SortKey.CUMULATIVE
        ).print_stats(40)
        print(s.getvalue())
        s = io.StringIO()
        pstats.Stats(build_prof, stream=s).sort_stats(pstats.SortKey.TIME).print_stats(
            30
        )
        print(s.getvalue())

    # Mimic render_to_video's prelude.
    scene.scene_times.append(
        [
            scene.scene_times[-1][0],
            round(
                AnimationManager.instance().context.timespan.original_end
                * scene.frames_per_second
            ),
        ]
    )
    scene.initialize_frames()
    scene.camera.despawn(animate=False)
    for light in scene.light_sources:
        light.despawn(animate=False)
    start_ind, end_ind = scene.scene_times[-1]
    print(f"frames: {start_ind}..{end_ind}")

    tm = TimelineManager.instance()
    n_events = len(tm.function_timeline.function_applications)
    n_edits = {a: len(t.edits) for a, t in tm.attr_to_timeline.items()}
    n_rows = {a: t.pointer for a, t in tm.attr_to_timeline.items()}
    print(f"function events: {n_events}")
    print(f"edits per attr: {n_edits}")
    print(f"rows per attr: {n_rows}")
    print(f"actors: {len(scene.actors[-1])}")

    for light in scene.light_sources:
        light.is_primitive = True
    actors = [
        scene.camera,
        scene.camera.screen,
        *scene.light_sources,
        *scene.actors[-1],
    ]
    max_animate_mem = int(
        COMPUTING_DEFAULTS.portion_of_memory_used_for_animating
        * get_num_available_bytes(COMPUTING_DEFAULTS.render_device)
    )
    print(f"max_animate_mem: {max_animate_mem / 2**20:.0f} MB")

    # This harness had the right idea before there was a name for it; it now
    # shares the render loop's own definition so the two cannot drift.
    with scene.batch_prep_context():
        prof = cProfile.Profile()
        cur = start_ind
        for _b in range(num_batches):
            if cur >= end_ind:
                break
            t0 = time.perf_counter()
            prof.enable()
            prims, new_ind, rs = scene.get_batch_of_primitives(
                cur, end_ind, actors, max_animate_mem
            )
            prof.disable()
            dt = time.perf_counter() - t0
            print(f"batch {cur}:{new_ind}  {dt:.3f}s  ({len(prims)} collections)")
            cur = new_ind

    s = io.StringIO()
    pstats.Stats(prof, stream=s).sort_stats(pstats.SortKey.CUMULATIVE).print_stats(45)
    print(s.getvalue())
    s = io.StringIO()
    pstats.Stats(prof, stream=s).sort_stats(pstats.SortKey.TIME).print_stats(35)
    print(s.getvalue())


if __name__ == "__main__":
    main()
