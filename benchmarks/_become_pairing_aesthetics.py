"""Render morph filmstrips so the pairing rule can be judged by eye.

`become`'s assignment is the thing that decides whether a morph reads as
"these parts moved there" or as "everything teleported".  It is invisible from
outside, so this renders the same morph under each available rule and lays the
frames out as a filmstrip: one row per rule, one column per sampled time.

Rules:
  blend      -- the shipped default: order, position and size, all normalized
                and weighted (the `_PAIR_*_WEIGHT` constants on MobMorphMixin)
  distance   -- the shipped opt-in, `minimize_movement=True`: pure proximity
  order      -- what the default USED to be, restored by monkeypatch: the order
                gap plus a distance capped at 1e-3, so geometry could only
                break exact ties. Kept as the A/B the change was made against.

Usage:  <venv-python> benchmarks/_become_pairing_aesthetics.py [--scene NAME]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from algan import (
    BLUE,
    GREEN,
    LD,
    LEFT,
    ORANGE,
    PURPLE,
    RED,
    RIGHT,
    UP,
    YELLOW,
    Circle,
    Group,
    Off,
    Rectangle,
    Scene,
    Square,
    Sync,
    Triangle,
)
from algan.animatable_base.mob_morph import MobMorphMixin

OUT = Path("/tmp/pairing_aesthetics")


# --------------------------------------------------------------------------
# The rule the default replaced, for A/B
# --------------------------------------------------------------------------


def install_legacy_order_cost():
    """Restore the pre-change rule: order, with distance as a 1e-6 tiebreak.

    This is what shipped before the assignment was rebalanced, kept so the two
    can be put side by side. Its distance term is capped at 1e-3 against an
    order gap spanning [0, 1], which is why it could only break exact ties.
    """
    original = MobMorphMixin._primitive_pair_cost

    def legacy_cost(
        self,
        source,
        target,
        *,
        source_index,
        target_index,
        source_count,
        target_count,
        minimize_movement,
        scene_span=None,
    ):
        compatibility = self._primitive_compatibility_rank(source, target)
        distance = float(
            (self._morph_center(source) - self._morph_center(target)).norm()
        )
        if minimize_movement:
            secondary = distance
        else:
            source_position = source_index / max(source_count - 1, 1)
            target_position = target_index / max(target_count - 1, 1)
            secondary = (
                abs(source_position - target_position) + min(distance, 1e3) * 1e-6
            )
        return compatibility * 1e6 + secondary

    MobMorphMixin._primitive_pair_cost = legacy_cost
    return original


def restore_cost(original):
    MobMorphMixin._primitive_pair_cost = original


# --------------------------------------------------------------------------
# Scenes whose "right answer" a viewer can name
# --------------------------------------------------------------------------


def bar_reorder():
    """Bars built in data order; the target is the same bars sorted by height.

    A viewer reads this as "the bars sorted themselves": each bar should slide
    to its new slot keeping its height.  Pairing by traversal order instead
    makes every bar stay put and change height.
    """
    heights_before = [0.6, 2.0, 1.1, 2.6, 1.6]
    heights_after = sorted(heights_before)
    colors = [BLUE, RED, GREEN, YELLOW, PURPLE]
    order = sorted(range(5), key=lambda index: heights_before[index])

    def build(heights, color_order):
        return Group(
            *[
                Rectangle(width=0.5, height=height, color=color).move(
                    RIGHT * (slot - 2) * 0.9 + UP * height / 2
                )
                for slot, (height, color) in enumerate(zip(heights, color_order))
            ]
        )

    return (
        build(heights_before, colors),
        build(heights_after, [colors[index] for index in order]),
    )


def scrambled_children():
    """The same four squares in both scenes; only the child ORDER differs.

    Nothing needs to move.  Any pairing that follows traversal order sends every
    square to a different corner.
    """
    places = [LEFT * 2 + UP, RIGHT * 2 + UP, RIGHT * 2 + UP * -1, LEFT * 2 + UP * -1]

    def build(order):
        return Group(*[Square(size=0.7).move(places[index]) for index in order])

    return build([0, 1, 2, 3]), build([2, 0, 3, 1])


def size_swap():
    """A big shape and a small one exchange places.

    Reads best as the big one travelling and staying big.  Pairing that ignores
    size lets the big one shrink in place while the small one grows in place.
    """
    return (
        Group(
            Circle(radius=0.9, color=ORANGE).move(LEFT * 2.5),
            Circle(radius=0.25, color=BLUE).move(RIGHT * 2.5),
        ),
        Group(
            Circle(radius=0.25, color=BLUE).move(LEFT * 2.5),
            Circle(radius=0.9, color=ORANGE).move(RIGHT * 2.5),
        ),
    )


def cluster_regroup():
    """Six dots in a row regroup into two triangles."""
    row = Group(
        *[
            Circle(radius=0.18, color=BLUE).move(RIGHT * (index - 2.5) * 0.8)
            for index in range(6)
        ]
    )
    clusters = []
    for side in (-1, 1):
        base = RIGHT * side * 2.0
        clusters.extend(
            [
                Circle(radius=0.18, color=BLUE).move(base + UP * 0.5),
                Circle(radius=0.18, color=BLUE).move(base + LEFT * 0.45 + UP * -0.35),
                Circle(radius=0.18, color=BLUE).move(base + RIGHT * 0.45 + UP * -0.35),
            ]
        )
    return row, Group(*clusters)


def shape_row():
    """Mixed shapes whose target arrangement reverses them."""

    def build(reverse):
        shapes = [
            Square(size=0.7, color=BLUE),
            Circle(radius=0.4, color=RED),
            Triangle(color=GREEN),
        ]
        if reverse:
            shapes = shapes[::-1]
        return Group(
            *[
                shape.move(RIGHT * (index - 1) * 1.6)
                for index, shape in enumerate(shapes)
            ]
        )

    return build(False), build(True)


SCENES = {
    "bar_reorder": bar_reorder,
    "scrambled_children": scrambled_children,
    "size_swap": size_swap,
    "cluster_regroup": cluster_regroup,
    "shape_row": shape_row,
}

RULES = ["blend", "distance", "order"]
SAMPLES = [0.0, 0.25, 0.5, 0.75, 1.0]


def render_strip(name, rule, tag):
    build = SCENES[name]
    paths = []
    for sample in SAMPLES:
        path = OUT / f"{tag}_{sample:.2f}.png"
        with Scene() as scene:
            scene.use_manim_defaults()
            with Off():
                source, target = build()
                source.spawn()
            start = float(scene.animation_manager.context.timespan.current_time)
            with Sync(duration=1.0):
                source.become(target, minimize_movement=(rule == "distance"))
            end = float(scene.animation_manager.context.timespan.current_time)
            at = start + (end - start) * sample
            scene.save_frame(str(path), LD, at=min(at, end - 1e-4), overwrite=True)
        paths.append(path)
    return paths


def main():
    import cv2

    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default=None)
    parser.add_argument(
        "--rules", default=",".join(RULES), help="comma separated subset of rules"
    )
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    names = [args.scene] if args.scene else list(SCENES)
    rules = args.rules.split(",")

    for name in names:
        rows = []
        for rule in rules:
            original = install_legacy_order_cost() if rule == "order" else None
            try:
                paths = render_strip(name, rule, f"{name}_{rule}")
            finally:
                if original is not None:
                    restore_cost(original)
            images = [cv2.imread(str(path), cv2.IMREAD_UNCHANGED) for path in paths]
            images = [
                image[..., :3] if image.shape[2] == 4 else image for image in images
            ]
            row = np.concatenate(images, axis=1)
            rows.append(row)
            print(f"  rendered {name} / {rule}", flush=True)
        grid = np.concatenate(rows, axis=0)
        scale = 1600 / grid.shape[1]
        grid = cv2.resize(
            grid, (int(grid.shape[1] * scale), int(grid.shape[0] * scale))
        )
        out_path = OUT / f"{name}_strip.png"
        cv2.imwrite(str(out_path), grid)
        print(f"wrote {out_path}  rows={rules}  cols={SAMPLES}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
