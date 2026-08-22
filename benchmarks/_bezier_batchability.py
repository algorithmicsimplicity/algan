"""Probe: how far the batched bezier build reaches, and what the group clash
costs -- the P9 measurement (DESIGN_optimization_targets.md, "P9 -- the
batched bezier build reaches 18.4% of the circuits"), rebuilt against the
bezier_rendering benchmark scene.

For several frame windows spread across the scene it classifies every circuit
actor as batched (merged by build_render_primitives_batched), reverted by the
group clash (batchable, but built per-actor because a raw primitive shared its
batch identifier), or rejected by _is_batchable_bezier -- splitting that last
one into `empty` (returns None immediately, ~free) and other reasons. The
outcomes are *measured*, not simulated: the probe wraps
build_render_primitives_batched and BezierCircuitCubic.get_render_primitives
and watches which actors each arm actually sends where, so running it with
ALGAN_BEZIER_GROUP_RUNS=0 (the old all-or-nothing revert) and =1 (run
splitting) shows exactly what the split moved.

It also times get_batch_of_primitives per window under both arms, alternating
the arms per round and taking medians -- un-alternated wall-clock A/Bs on this
class of machine produce noise that reads as a result.

    ALGAN_BEZIER_GROUP_RUNS=0 .venv/bin/python benchmarks/_bezier_batchability.py
    .venv/bin/python benchmarks/_bezier_batchability.py

Prep only, no render. batch_prep_context is mandatory: calling
get_batch_of_primitives outside it grows the timeline on every call and
corrupts the measurement.
"""

from __future__ import annotations

import os
import sys
import time

os.environ["ALGAN_PREFETCH_BATCHES"] = "0"
os.environ["ALGAN_ADV_OPT"] = "0"
# A warm daemon keeps adaptive renderer state across runs; this probe must
# measure its own scene, not whatever ran before it.
os.environ.setdefault("ALGAN_USE_DAEMON", "0")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import manim as mn  # noqa: E402
import torch  # noqa: E402

from algan import *  # noqa: E402
from algan.mobs.bezier_circuit import BezierCircuitCubic  # noqa: E402
from algan.mobs.neural_nets.neural_net import NeuralNetMLPV3  # noqa: E402
from algan.scene_manager import SceneManager  # noqa: E402
from algan.settings import SETTINGS  # noqa: E402
from algan.settings._startup import _ANIMATION_DEVICE  # noqa: E402
from algan.utils.memory_utils import get_num_available_bytes  # noqa: E402

WINDOW_WIDTH_FRAMES = 8
NUM_WINDOWS = 6
TIMING_ROUNDS = 6


def Boxed(mob, color=BLUE, buffer=0.1, *args, **kwargs):
    return Group(
        mob,
        SurroundingRectangle(
            mob,
            *args,
            color=color.lerp(BLACK, 0.8).lerp(PURE_BLUE, 0.1).set_opacity(0.95),
            border_color=torch.lerp(color, BLACK, 0.2),
            buffer=buffer,
            border_width=1,
            **kwargs,
        ),
    )


def GlowTex(c, *args, **kwargs):
    # ManimMob rather than algan's Tex: bezier_rendering.py's .set(border_
    # color=...) predates the animatable-property check and no longer runs
    # on a plain Tex; this is the same construction _bez_batch_parity.py uses.
    m = (
        ManimMob(mn.MathTex(*args, **kwargs))
        .set(
            color=c + GLOW * 0.01,
            border_color=torch.lerp(c, WHITE, 0.9),
            border_width=0.8,
        )
        .scale(0.75)
    )
    return m


text_string = ("a" * 50 + "\n") * 50


def text_scene():
    with Off():
        nn = NeuralNetMLPV3([3, 3, 3]).spawn()
        mob = Boxed(GlowTex(GREEN, text_string)).spawn()
    with Sync(run_time=1):
        mob.move(LEFT)
        nn.move(LEFT)


def clash_scene():
    """A scene built to exercise the group clash, which the benchmark scene
    turns out not to have at all.

    algan's ``Text`` packs its glyph circuits -- each is rejected by
    _is_batchable_bezier on the batched-control-points clause and its
    primitive goes into the merged arrays raw. Ordinary circuits spawned in
    the same frame batch share the glyphs' batch identifier (same texture-
    point count, same filled-ness), so under the old all-or-nothing rule
    every one of them was reverted to the per-actor build; with run
    splitting they batch into the runs between the glyphs.
    """
    with Off():
        txt = Text("Hello World").spawn()
        others = []
        for i in range(20):
            others.append(Circle(color=BLUE).move(RIGHT * (i - 10)).spawn())
            others.append(Square(color=RED).move(RIGHT * (i - 10) + UP * 2).spawn())
    with Sync(run_time=1):
        txt.move(LEFT)
        for m in others:
            m.move(UP)


class BuildWatcher:
    """Records which circuits each arm actually builds where.

    _build_deferred_beziers imports build_render_primitives_batched from its
    module at call time, so patching the module attribute is enough; the
    per-actor path goes through the class method.
    """

    def __init__(self):
        self.batched_ids = set()
        self.per_actor_ids = set()
        self._orig_brb = None
        self._orig_grp = None

    def attach(self):
        import algan.mobs.bezier_circuit as bez_mod

        watcher = self

        def counting_brb(actors, scene):
            watcher.batched_ids.update(id(a) for a in actors)
            return watcher._orig_brb(actors, scene)

        def counting_grp(self_mob):
            watcher.per_actor_ids.add(id(self_mob))
            return watcher._orig_grp(self_mob)

        self._orig_brb = bez_mod.build_render_primitives_batched
        self._orig_grp = BezierCircuitCubic.get_render_primitives
        bez_mod.build_render_primitives_batched = counting_brb
        BezierCircuitCubic.get_render_primitives = counting_grp

    def reset(self):
        self.batched_ids.clear()
        self.per_actor_ids.clear()


def classify(scene, watcher, circuits):
    rows = {
        "batched": [],
        "reverted": [],
        "rejected_empty": [],
        "rejected_bcp": [],
        "rejected_other": [],
        "unaccounted": [],
    }
    for a in circuits:
        if not scene._is_batchable_bezier(a):
            if a.empty:
                rows["rejected_empty"].append(a)
            elif getattr(a.control_points, "parent_batch_sizes", None) is not None:
                rows["rejected_bcp"].append(a)
            else:
                rows["rejected_other"].append(a)
        elif id(a) in watcher.batched_ids:
            rows["batched"].append(a)
        elif id(a) in watcher.per_actor_ids:
            rows["reverted"].append(a)
        else:
            rows["unaccounted"].append(a)
    return rows


def print_table(label, total_rows):
    n_total = sum(len(v) for v in total_rows.values())
    print(f"\n== {label}: {n_total} circuit appearances over {NUM_WINDOWS} windows ==")
    order = [
        ("batched", "batched"),
        ("reverted", "reverted by the group clash"),
        ("rejected", "rejected by _is_batchable_bezier"),
        ("rejected_empty", "  of which empty (~free)"),
        ("rejected_bcp", "  of which batched-control-points"),
        ("rejected_other", "  of which other reasons"),
        ("unaccounted", "unaccounted (probe bug if nonzero)"),
    ]
    rejected = (
        len(total_rows["rejected_empty"])
        + len(total_rows["rejected_bcp"])
        + len(total_rows["rejected_other"])
    )
    merged = dict(total_rows)
    merged["rejected"] = [None] * rejected
    for key, name in order:
        count = len(merged[key])
        share = f"{count / n_total:6.1%}" if n_total else "   n/a"
        print(f"| {name:<42} | {count:>5} | {share} |")


def probe_scene(label, build_scene_fn):
    scene = SceneManager.reset()
    # The bezier_rendering benchmark's settings (UHD at 60 fps).
    scene.set_video_settings(UHD.set_frames_per_second(60))
    build_scene_fn()
    # Mimic render_to_video's prelude (benchmarks/_prep_profile.py).
    scene.scene_times.append(
        [
            scene.scene_times[-1][0],
            round(
                scene.animation_manager.context.timespan.original_end
                * scene.frames_per_second
            ),
        ]
    )
    scene.initialize_frames()
    scene.camera.despawn(animate=False)
    for light in scene.light_sources:
        light.despawn(animate=False)
    start_ind, end_ind = scene.scene_times[-1]
    fps = scene.frames_per_second
    print(f"\n########## {label}: frames {start_ind}..{end_ind} ##########")

    for light in scene.light_sources:
        light.is_primitive = True
    # scene.actors is a flat list (it used to be per-run lists, which is what
    # _prep_profile.py's [-1] indexed into).
    actors = [
        scene.camera,
        scene.camera.screen,
        *scene.light_sources,
        *scene.actors,
    ]
    # The render loop's own budget (render_loop.py, fetch_batch's caller):
    # _prep_profile.py's COMPUTING_DEFAULTS import predates a settings
    # refactor and no longer resolves.
    max_animate_mem = int(
        SETTINGS.computing.animation_memory_fraction
        * get_num_available_bytes(_ANIMATION_DEVICE)
    )

    # Windows spread across the scene, each capped to WINDOW_WIDTH_FRAMES so
    # there is more than one of them (get_batch_of_primitives would otherwise
    # take the whole remaining scene in one batch).
    span = end_ind - start_ind
    step = max(1, span // NUM_WINDOWS)
    windows = [
        (min(s, end_ind - 1), min(s + WINDOW_WIDTH_FRAMES, end_ind))
        for s in range(start_ind, end_ind, step)
    ][:NUM_WINDOWS]

    watcher = BuildWatcher()
    watcher.attach()

    index = scene._actor_window_index(actors)

    def window_circuits(t0, t1):
        inds = scene._actors_in_window(index, t0 / fps, t1 / fps).tolist()
        return [
            index[0][i] for i in inds if isinstance(index[0][i], BezierCircuitCubic)
        ]

    with scene.batch_prep_context():
        # Warm-up pass (first-call lazy init must not land in any timing).
        for w0, w1 in windows:
            scene.get_batch_of_primitives(w0, w1, actors, max_animate_mem)

        # --- classification tables, one per arm ---
        arm = os.environ.get("ALGAN_BEZIER_GROUP_RUNS", "1")
        for flag_value, label in (
            ("0", "BEFORE (ALGAN_BEZIER_GROUP_RUNS=0, all-or-nothing revert)"),
            ("1", "AFTER  (ALGAN_BEZIER_GROUP_RUNS=1, run splitting)"),
        ):
            os.environ["ALGAN_BEZIER_GROUP_RUNS"] = flag_value
            totals = {
                k: []
                for k in (
                    "batched",
                    "reverted",
                    "rejected_empty",
                    "rejected_bcp",
                    "rejected_other",
                    "unaccounted",
                )
            }
            per_window = []
            for w0, w1 in windows:
                watcher.reset()
                scene.get_batch_of_primitives(w0, w1, actors, max_animate_mem)
                rows = classify(scene, watcher, window_circuits(w0, w1))
                for key, value in rows.items():
                    totals[key].extend(value)
                per_window.append((w0, len(rows["batched"]), len(rows["reverted"])))
            print_table(label, totals)
            print("  per window (start_ind, batched, reverted):", per_window)

        # --- alternating-arm wall clock ---
        os.environ["ALGAN_BEZIER_GROUP_RUNS"] = arm
        times = {("0", i): [] for i in range(len(windows))}
        times.update({("1", i): [] for i in range(len(windows))})
        for round_i in range(TIMING_ROUNDS):
            # Alternate arms per round; within a round walk every window.
            flag = str(round_i % 2)
            os.environ["ALGAN_BEZIER_GROUP_RUNS"] = flag
            for i, (w0, w1) in enumerate(windows):
                t_start = time.perf_counter()
                scene.get_batch_of_primitives(w0, w1, actors, max_animate_mem)
                times[(flag, i)].append(time.perf_counter() - t_start)

        print(
            "\n== get_batch_of_primitives wall clock (median of "
            f"{TIMING_ROUNDS // 2} rounds per arm, arms alternated) =="
        )
        for i, (w0, _) in enumerate(windows):
            old = sorted(times[("0", i)])[len(times[("0", i)]) // 2]
            new = sorted(times[("1", i)])[len(times[("1", i)]) // 2]
            ratio = new / old if old else float("nan")
            print(
                f"  window {w0:>4}: runs=0 {old * 1e3:8.1f} ms | "
                f"runs=1 {new * 1e3:8.1f} ms | new/old {ratio:.3f}x"
            )


def main():
    # The benchmark scene first, as specified; it measures 0 group clashes,
    # which is exactly why the clashing scene is probed alongside it.
    probe_scene("bezier_rendering benchmark scene", text_scene)
    probe_scene(
        "constructed clash scene (Text glyphs + ordinary circuits)", clash_scene
    )


if __name__ == "__main__":
    main()
