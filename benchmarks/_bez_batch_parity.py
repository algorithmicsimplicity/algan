"""Tensor-level parity check for the vectorized bezier primitive build.

Builds the bezier_rendering benchmark scene, materializes one batch of frame
times, then compares every attribute of the merged collection produced by
bezier_circuit.build_render_primitives_batched against the per-actor
get_render_primitives + BezierCircuitPrimitive(triangle_collection=...) path,
bitwise.

    .venv/Scripts/python.exe benchmarks/_bez_batch_parity.py
"""

from __future__ import annotations

import math  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import manim as mn  # noqa: E402
import torch  # noqa: E402

from algan import *  # noqa: E402
from algan.animation_timeline.animation_contexts import Off  # noqa: E402
from algan.mobs.bezier_circuit import build_render_primitives_batched  # noqa: E402
from algan.mobs.neural_nets.neural_net import NeuralNetMLP  # noqa: E402
from algan.scene_manager import SceneManager  # noqa: E402


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
    with Sync(run_time=0.25):
        nn = NeuralNetMLP([3, 3, 3]).spawn()
        mob = Boxed(GlowTex(GREEN, text_string)).spawn()
        # A few ordinary circuits so the groups have real size: the parity
        # check is only as good as the group it runs on.
        circles = [Circle(color=BLUE) for _ in range(8)]
        squares = [Square(color=RED) for _ in range(4)]
        for i, c in enumerate(circles):
            c.move(RIGHT * (i - 3)).spawn()
        for i, s in enumerate(squares):
            s.move(LEFT * (i + 1)).spawn()
    with Sync(run_time=0.25):
        mob.move(LEFT)
        nn.move(LEFT)


ATTRS = [
    "corners",
    "colors",
    "next_segment_inds",
    "normals",
    "border_width",
    "border_color",
    "mob_center",
    "grid_width",
    "grid_height",
    "basis1",
    "basis2",
    "z_index",
    "num_segments_per_object",
]


def main():
    scene = SceneManager.reset()
    scene.set_video_settings(PREVIEW)
    text_scene()
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

    with Off(
        record_attr_modifications=False, record_funcs=False, priority_level=math.inf
    ):
        # Materialize a mid-animation window (covers fade + move).
        times = torch.arange(start_ind, end_ind)
        scene.timeline_manager.set_state_to_times(times / scene.frames_per_second)

        actors = sorted(
            [a for a in scene.actors if hasattr(a, "get_render_primitives")],
            key=lambda x: x.anchor_priority,
            reverse=True,
        )
        bez = [a for a in actors if scene._is_batchable_bezier(a)]
        print(f"batchable bezier actors: {len(bez)}")
        if not bez:
            print("NOTHING TO CHECK")
            sys.exit(1)

        groups = {}
        for a in bez:
            groups.setdefault(scene._bezier_group_key(a), []).append(a)
        print(f"groups: {[(k[0][-20:], k[1], len(v)) for k, v in groups.items()]}")

        n_fail = 0
        for _key, group in groups.items():
            prims = [a.get_render_primitives() for a in group]
            old = group[0].render_primitive(triangle_collection=prims)
            new = build_render_primitives_batched(group, scene)
            for attr in ATTRS:
                a_old = getattr(old, attr)
                a_new = getattr(new, attr)
                if not torch.is_tensor(a_old):
                    ok = a_old == a_new
                else:
                    ok = (
                        a_old.shape == a_new.shape
                        and a_old.dtype == a_new.dtype
                        and torch.equal(a_old, a_new)
                    )
                if not ok:
                    n_fail += 1
                    if (
                        torch.is_tensor(a_old)
                        and a_old.shape == a_new.shape
                        and a_old.dtype == a_new.dtype
                    ):
                        d = (a_old.float() - a_new.float()).abs()
                        print(
                            f"  MISMATCH {attr}: shape {tuple(a_old.shape)} "
                            f"max diff {d.max().item()} "
                            f"n diff {(d > 0).sum().item()}"
                        )
                    else:
                        so = tuple(a_old.shape) if torch.is_tensor(a_old) else a_old
                        sn = tuple(a_new.shape) if torch.is_tensor(a_new) else a_new
                        do = a_old.dtype if torch.is_tensor(a_old) else type(a_old)
                        dn = a_new.dtype if torch.is_tensor(a_new) else type(a_new)
                        print(f"  MISMATCH {attr}: {so}/{do} vs {sn}/{dn}")
            # ``num_pixels_per_sample`` is in this list on purpose, and it is
            # what this harness caught when it was repaired: the batched
            # builder had drifted to 1 against the per-actor path's 0.5 -- a
            # curve flattened to twice the chord error on any batch the
            # analytic-AA route rejects. Keep it a plain equality check.
            for attr in ("num_texture_points", "filled", "num_pixels_per_sample"):
                if getattr(old, attr) != getattr(new, attr):
                    n_fail += 1
                    print(
                        f"  MISMATCH {attr}: {getattr(old, attr)} vs {getattr(new, attr)}"
                    )
            if type(old) is not type(new):
                n_fail += 1
                print(f"  MISMATCH class: {type(old)} vs {type(new)}")
        if n_fail:
            print(f"FAIL: {n_fail} mismatching attributes")
            sys.exit(1)
        print("ALL COLLECTION TENSORS BITWISE IDENTICAL")


if __name__ == "__main__":
    main()
