"""Report which source primitive `become` pairs with which target primitive.

Aesthetic quality of a morph is decided almost entirely by this assignment, and
it is invisible from the outside: the only way to see it is to intercept the
pairing.  This script monkeypatches `_pair_primitive_indices` to log its cost
matrix and its chosen assignment, then runs a set of scenes whose "obviously
right" answer a viewer could name by looking at them.

Usage:  <venv-python> benchmarks/_become_pairing_probe.py
"""

from __future__ import annotations

import torch

from algan import (
    BLUE,
    DOWN,
    GREEN,
    LEFT,
    RED,
    RIGHT,
    UP,
    YELLOW,
    Circle,
    Group,
    Off,
    Scene,
    Sphere,
    Square,
    Sync,
    Text,
    Triangle,
)
from algan.animatable_base.mob_morph import MobMorphMixin

_ORIGINAL = MobMorphMixin._pair_primitive_indices
_LOG = []


def _label(mob):
    center = MobMorphMixin._morph_center(mob)
    return f"{type(mob).__name__}@({center[0]:+.2f},{center[1]:+.2f})"


def _patched(self, sources, targets, minimize_movement):
    pairs, unmatched_sources, unmatched_targets = _ORIGINAL(
        self, sources, targets, minimize_movement
    )
    costs = torch.empty((len(sources), len(targets)), dtype=torch.float64)
    for i, source in enumerate(sources):
        for j, target in enumerate(targets):
            costs[i, j] = self._primitive_pair_cost(
                source,
                target,
                source_index=i,
                target_index=j,
                source_count=len(sources),
                target_count=len(targets),
                minimize_movement=minimize_movement,
            )
    _LOG.append(
        {
            "sources": [_label(mob) for mob in sources],
            "targets": [_label(mob) for mob in targets],
            "pairs": pairs,
            "unmatched_sources": unmatched_sources,
            "unmatched_targets": unmatched_targets,
            "costs": costs,
            "minimize_movement": minimize_movement,
        }
    )
    return pairs, unmatched_sources, unmatched_targets


MobMorphMixin._pair_primitive_indices = _patched


def report(title, build_source, build_target, minimize_movement=False):
    _LOG.clear()
    with Scene():
        with Off():
            source = build_source().spawn()
            target = build_target()
        with Sync(run_time=1.0):
            source.become(target, minimize_movement=minimize_movement)
    print(f"\n=== {title}  (minimize_movement={minimize_movement}) ===")
    if not _LOG:
        print("  (no hierarchy pairing ran -- single-primitive route)")
        return
    for entry in _LOG:
        print(f"  sources: {entry['sources']}")
        print(f"  targets: {entry['targets']}")
        for source_index, target_index in entry["pairs"]:
            distance = float(entry["costs"][source_index, target_index] % 1e6)
            print(
                f"    {entry['sources'][source_index]:>24s}"
                f"  ->  {entry['targets'][target_index]:<24s}"
                f"  secondary={distance:.4f}"
            )
        if entry["unmatched_sources"]:
            print(
                "    surplus sources: "
                + ", ".join(entry["sources"][i] for i in entry["unmatched_sources"])
            )
        if entry["unmatched_targets"]:
            print(
                "    surplus targets: "
                + ", ".join(entry["targets"][i] for i in entry["unmatched_targets"])
            )


def main():
    # 1. Same shapes, swapped places. Pairing by position is free and obviously
    #    right; pairing by traversal order makes both parts cross the screen.
    report(
        "two identical squares swap places",
        lambda: Group(
            Square(color=BLUE).move(LEFT * 2), Square(color=RED).move(RIGHT * 2)
        ),
        lambda: Group(
            Square(color=BLUE).move(RIGHT * 2), Square(color=RED).move(LEFT * 2)
        ),
    )
    report(
        "two identical squares swap places",
        lambda: Group(
            Square(color=BLUE).move(LEFT * 2), Square(color=RED).move(RIGHT * 2)
        ),
        lambda: Group(
            Square(color=BLUE).move(RIGHT * 2), Square(color=RED).move(LEFT * 2)
        ),
        minimize_movement=True,
    )

    # 2. Type match versus proximity: the near counterpart is a different shape.
    report(
        "square+circle -> circle-near, square-far",
        lambda: Group(
            Square(color=BLUE).move(LEFT * 3), Circle(color=RED).move(RIGHT * 3)
        ),
        lambda: Group(
            Circle(color=RED).move(LEFT * 3), Square(color=BLUE).move(RIGHT * 3)
        ),
    )

    # 3. Size: a big and a small target, sources of matching sizes but reversed
    #    traversal order.
    report(
        "big+small -> small+big (size should decide)",
        lambda: Group(
            Square(side_length=2.0).move(LEFT * 3),
            Square(side_length=0.4).move(RIGHT * 3),
        ),
        lambda: Group(
            Square(side_length=0.4).move(LEFT * 3),
            Square(side_length=2.0).move(RIGHT * 3),
        ),
    )

    # 4. Grid of four: rotate the arrangement by one position.
    def grid(colors):
        offsets = [LEFT + UP, RIGHT + UP, RIGHT + DOWN, LEFT + DOWN]
        return Group(
            *[
                Square(color=color).move(offset * 2)
                for color, offset in zip(colors, offsets)
            ]
        )

    report(
        "four squares rotate one position",
        lambda: grid([BLUE, RED, GREEN, YELLOW]),
        lambda: grid([YELLOW, BLUE, RED, GREEN]),
    )

    # 5. Unequal counts: three sources, five targets.
    report(
        "3 squares -> 5 squares",
        lambda: Group(
            *[Square().move(RIGHT * (index - 1) * 1.5) for index in range(3)]
        ),
        lambda: Group(
            *[Square().move(RIGHT * (index - 2) * 1.2) for index in range(5)]
        ),
    )

    # 6. Mixed families in a hierarchy.
    report(
        "square+sphere -> sphere+square",
        lambda: Group(
            Square(color=BLUE).move(LEFT * 2), Sphere(radius=0.5).move(RIGHT * 2)
        ),
        lambda: Group(
            Sphere(radius=0.5).move(LEFT * 2), Square(color=BLUE).move(RIGHT * 2)
        ),
    )

    # 7. Text: is a whole Text one primitive, or one per glyph?
    report("text 'ab' -> text 'ba'", lambda: Text("ab"), lambda: Text("ba"))
    report(
        "group of two texts",
        lambda: Group(Text("a").move(LEFT), Text("b").move(RIGHT)),
        lambda: Group(Text("b").move(LEFT), Text("a").move(RIGHT)),
    )

    # 8. Nested containers: does nesting depth affect traversal order pairing?
    report(
        "nested groups, mirrored",
        lambda: Group(
            Group(Square(color=BLUE).move(LEFT * 2), Triangle(color=RED).move(LEFT)),
            Group(
                Circle(color=GREEN).move(RIGHT), Square(color=YELLOW).move(RIGHT * 2)
            ),
        ),
        lambda: Group(
            Group(Square(color=YELLOW).move(LEFT * 2), Circle(color=GREEN).move(LEFT)),
            Group(Triangle(color=RED).move(RIGHT), Square(color=BLUE).move(RIGHT * 2)),
        ),
    )


if __name__ == "__main__":
    main()
