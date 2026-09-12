"""A math-explainer workload: what a Manim-style educational video renders.

The shape of the workload, not its content, is what is being measured, so it
carries the mix an explainer actually has:

* **glyph circuits** -- a title, a caption, two formulas and a row of numeric
  labels, ~180 glyphs of Pango ``Text`` (unicode maths rather than ``Tex`` so
  the scene needs no LaTeX toolchain on the GPU boxes; a glyph is the same
  cubic-bezier circuit either way);
* **plots** -- a Manim-compatibility ``Axes`` with two delegated plots, a
  ``NumberPlane``, a tangent ``Arrow`` and a ``Dot`` that tracks the curve
  through an updater;
* **diagrams** -- a 9-node graph of circles and lines spawned with ``Lag``,
  and a 6x6 heat-map of squares whose colours animate;
* **a little 3-D** -- one ``Sphere`` (PN dice) and one ``Cube`` turning in a
  corner, as explainers do for a "here is the surface" beat;
* **the animations** -- staggered spawns, synced moves, colour tweens, an
  ``Indicate``, a ``Circumscribe``, a glyph-morph ``become`` between the two
  formulas, and despawns. The camera never moves.

Everything is 2-D and unlit apart from the two solids, so there are no
shadows, no reflections and no refraction; shading cost is negligible and
the render is coverage, compaction and compositing -- plus the scene
preparation every batch pays for a few hundred small mobs.

The storyboard is written as fractions of the clip length so the same scene
measures at any duration::

    python benchmarks/performance/explainer_scene.py --quality PREVIEW --seconds 6
    python benchmarks/performance/explainer_scene.py --quality UHD --seconds 0.5

Two profiled runs; read RUN 2 (`agent_guidance/gpu_harnesses.md`).
"""

from __future__ import annotations

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _profile_cli  # noqa: E402

from algan import *  # noqa: E402

FORMULA_A = "xₙ₊₁ = xₙ − η ∇f(xₙ)"
FORMULA_B = "f(x) = ½ xᵀAx − bᵀx + c"

# The axes' placement, repeated here so the walker's updater can map graph
# coordinates to world space with tensor arithmetic. Asking the Axes itself
# (``c2p``) round-trips through the backing Manim object on every frame.
AXES_CENTER = LEFT * 3.4 + UP * 0.5
AXES_X_UNIT = 5.0 / 6.0  # x_length / x span
AXES_Y_UNIT = 3.2 / 4.0  # y_length / y span
AXES_Y_MID = 0.5  # midpoint of y_range


def mix(a, b, t):
    """Linear colour blend; ``Color`` is a tensor, so this stays a Color."""
    return a * (1.0 - t) + b * t


def scene(seconds: float):
    """Record the explainer; ``seconds`` is the whole clip's authored length."""
    SETTINGS.raytracing.set(shadows=False)
    Scene.set_background(DARKER_GRAY)

    def part(fraction):
        return seconds * fraction

    with Off():
        title = Text(
            "Gradient descent on a quadratic bowl",
            font_size=40,
            weight="BOLD",
            color=WHITE,
        ).move(UP * 3.1)
        caption = Text(
            "step size η trades speed against stability",
            font_size=22,
            color=GRAY_A,
        ).move(DOWN * 3.25)

        # Left: axes with the bowl, a tangent, and a dot walking the curve.
        axes = Axes(
            x_range=(-3, 3, 1),
            y_range=(-1.5, 2.5, 1),
            x_length=5.0,
            y_length=3.2,
        ).move(LEFT * 3.4 + UP * 0.5)
        bowl = axes.plot(lambda x: 0.35 * x * x - 1.0, color=YELLOW)
        slope = axes.plot(lambda x: 0.6 * x - 0.4, color=BLUE_B)
        walker = Dot(radius=0.11, color=RED).move(LEFT * 5.4 + UP * 1.6)
        tangent = Arrow(
            start=LEFT * 0.7, end=RIGHT * 0.7, color=ORANGE, stroke_width=4
        ).move(LEFT * 3.4 + UP * 1.9)

        # Middle: the formula, which later becomes the second formula.
        formula = Text(FORMULA_A, font_size=34, color=TEAL_A).move(UP * 1.9)

        # Right: a number plane with a vector, and a heat map.
        plane = NumberPlane(
            x_range=(-2, 2, 1), y_range=(-2, 2, 1), x_length=2.8, y_length=2.8
        ).move(RIGHT * 3.9 + UP * 0.9)
        vector = Vector(RIGHT * 0.9 + UP * 0.7, color=ORANGE).move(
            RIGHT * 3.9 + UP * 0.9
        )
        heat = Group(
            [Square(size=0.22, color=BLUE_E).scale(0.9) for _ in range(36)]
        ).arrange_in_grid(6, row_buffer=0.05, column_buffer=0.05)
        heat.move(RIGHT * 3.9 + DOWN * 1.7 - heat.get_center())

        # Bottom middle: a graph diagram of nine nodes and their edges.
        nodes = Group(
            [
                Circle(radius=0.16, color=PURPLE_A, stroke_color=WHITE, stroke_width=2)
                for _ in range(9)
            ]
        )
        for k, node in enumerate(nodes):
            angle = 2 * math.pi * k / 9
            node.move(RIGHT * 1.15 * math.cos(angle) + UP * 1.15 * math.sin(angle))
        nodes.move(DOWN * 1.55)
        edges = Group(
            [
                Line(
                    start=nodes[i].get_center(),
                    end=nodes[j].get_center(),
                    color=GRAY_B,
                    stroke_width=2,
                )
                for i, j in (
                    (0, 3),
                    (3, 6),
                    (6, 1),
                    (1, 4),
                    (4, 7),
                    (7, 2),
                    (2, 5),
                    (5, 8),
                    (8, 0),
                )
            ]
        )
        labels = Group([Text(str(k), font_size=18, color=WHITE) for k in range(9)])
        for node, label in zip(nodes, labels):
            label.move_to(node.get_center())
        # Numeric labels as plain glyph runs (``DecimalNumber`` typesets through
        # LaTeX, which the GPU boxes do not have).
        readouts = Group(
            [
                Text(f"{0.25 * (k + 1):.2f}", font_size=24, color=GRAY_A)
                for k in range(4)
            ]
        ).arrange_in_line(RIGHT, buffer=0.55)
        readouts.move(LEFT * 3.4 + DOWN * 2.1 - readouts.get_center())

        # A corner of 3-D: the surface the contours are slices of.
        bowl3d = Sphere(radius=0.42).set_material(
            MeshStandardMaterial(color=BLUE, roughness=0.5)
        )
        bowl3d.move(LEFT * 5.9 + DOWN * 2.6)
        cube = Cube(size=0.55).set_material(MeshLambertMaterial(color=MAROON_B))
        cube.move(RIGHT * 6.0 + DOWN * 2.6)

    # Everything in the walker's updater is host-side math on the timeline.
    def follow_curve(mob, t):
        # ``t`` is the elapsed time since the updater was added, as a
        # ``[frames, 1, 1]`` tensor: one location per materialized frame.
        frac = torch.clamp(t / max(seconds * 0.56, 1e-6), 0.0, 1.0)
        x = -2.6 + 5.2 * frac
        y = 0.35 * x * x - 1.0
        mob.move_to(
            AXES_CENTER
            + RIGHT * (x * AXES_X_UNIT)
            + UP * ((y - AXES_Y_MID) * AXES_Y_UNIT)
        )

    with Seq():
        # Act 1 -- the stage appears.
        with Sync(runtime=part(0.10)):
            title.spawn()
            axes.spawn()
            plane.spawn()
        with Lag(0.15, runtime=part(0.12)):
            bowl.spawn()
            slope.spawn()
            vector.spawn()
            formula.spawn()
            walker.spawn()
            tangent.spawn()
        with Sync(runtime=part(0.08)):
            heat.spawn()
            caption.spawn()
            bowl3d.spawn()
            cube.spawn()
            readouts.spawn()
        # Act 2 -- the diagram builds, node by node, while the walker descends.
        walk = walker.add_updater(follow_curve)
        with Lag(0.08, runtime=part(0.18)):
            for node, label in zip(nodes, labels):
                node.spawn()
                label.spawn()
        with Lag(0.06, runtime=part(0.10)):
            for edge in edges:
                edge.spawn()
        # Act 3 -- attention, colour and the formula morph.
        with Sync(runtime=part(0.16)):
            Indicate(nodes[4], runtime=part(0.16))
            Circumscribe(heat, runtime=part(0.16))
            for k, cell in enumerate(heat):
                cell.color = mix(BLUE_E, YELLOW, (k % 6) / 5.0)
            vector.rotate(50)
            tangent.rotate(-25)
            bowl3d.rotate(120, UP)
            cube.rotate(90, RIGHT)
            for k, readout in enumerate(readouts):
                readout.color = mix(GRAY_A, YELLOW, k / 3.0)
        with Sync(runtime=part(0.12)):
            formula.become(
                Text(FORMULA_B, font_size=34, color=TEAL_A, add_to_scene=False).move(
                    UP * 1.9
                )
            )
            slope.color = MAROON_A
            for k, cell in enumerate(heat):
                cell.color = mix(YELLOW, RED, (k // 6) / 5.0)
        walker.remove_updater(walk)
        # Act 4 -- part of the stage leaves.
        with Sync(runtime=part(0.14)):
            for edge in edges:
                edge.despawn()
            for node, label in zip(nodes, labels):
                node.despawn()
                label.despawn()
            plane.despawn()
            vector.despawn()


if __name__ == "__main__":
    _profile_cli.run(scene, "explainer", default_seconds=6.0)
