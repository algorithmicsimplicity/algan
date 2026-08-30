"""Flat 2-D geometry (bezier circuits) driven through the whole timeline API.

Everything on screen is a cubic-bezier circuit, so this scene is the pixel
reference for the circuit rasteriser: filled interiors, non-convex
triangulation, inward borders, and the analytic anti-aliasing on both the
silhouette and the border/fill seam.

On top of that geometry it exercises the recording side of the engine end to
end -- the four animation contexts and their nesting, ``duration`` rescaling,
rate functions, every indication animation, ``become`` morphing, updaters,
``wave_color``, ``DrawBorderThenFill`` and the spawn/despawn lifecycle.

Only native :mod:`algan.mobs.shapes_2d` geometry appears here; the
Manim-compatibility shapes live in ``manim_compat_and_plots`` so a regression in
one family cannot be mistaken for the other.  The shapes are laid out in
labelled, non-overlapping rows so a regression reads as a diff in one column.
"""

import torch

# The point-cloud family is Manim-compat, so Phase 1 of the API overhaul moved
# it out of the root namespace and behind ``algan.manim``.
import algan.manim as mn
from algan import *

# Pinned so the render does not depend on the host's fonts;
# tests/conftest.py registers the vendored faces.
FONT = "Algan Test Sans"

Scene.set_background(DARKER_GRAY)

# A non-convex circuit: five outer points and five inner points, so the fill
# rule and the triangulation of a re-entrant outline are both under test.
STAR_POINTS = (
    UP * 0.70,
    LEFT * 0.176 + UP * 0.243,
    LEFT * 0.666 + UP * 0.216,
    LEFT * 0.285 + DOWN * 0.093,
    LEFT * 0.411 + DOWN * 0.566,
    DOWN * 0.30,
    RIGHT * 0.411 + DOWN * 0.566,
    RIGHT * 0.285 + DOWN * 0.093,
    RIGHT * 0.666 + UP * 0.216,
    RIGHT * 0.176 + UP * 0.243,
)

# --------------------------------------------------------------------------
# Act 1 -- filled circuits spawn under Lag, outlined circuits under Sync.
# --------------------------------------------------------------------------
with Off():
    AmbientLight(color=WHITE, intensity=0.6).spawn(animate=False)
    DirectionalLight(
        location=RIGHT * 4 + UP * 5 + OUT * 4,
        target=ORIGIN,
        color=WHITE,
        intensity=0.9,
    ).spawn(animate=False)

    title = Text(
        "SHAPES AND TIMELINE",
        font_size=44,
        weight="BOLD",
        color=WHITE,
        font=FONT,
    ).move(UP * 3.05)

    filled = Group(
        Circle(radius=0.62, color=BLUE),
        Square(size=1.2, color=GREEN),
        RegularPolygon(5, radius=0.68, color=ORANGE),
        Polygon(*STAR_POINTS, color=YELLOW),
        Triangle(color=TEAL).scale(0.72),
        Rectangle(width=1.45, height=0.95, color=PURPLE),
    ).arrange_in_line(RIGHT, buffer=0.66)
    filled.move(UP * 1.45 - filled.get_center())

    filled_labels = Group(
        Text("circle", font_size=22, color=GRAY_A, font=FONT),
        Text("square", font_size=22, color=GRAY_A, font=FONT),
        Text("pentagon", font_size=22, color=GRAY_A, font=FONT),
        Text("non-convex", font_size=22, color=GRAY_A, font=FONT),
        Text("triangle", font_size=22, color=GRAY_A, font=FONT),
        Text("rectangle", font_size=22, color=GRAY_A, font=FONT),
    )
    for shape, label in zip(filled, filled_labels):
        label.move_to(shape.get_center() + DOWN * 1.15)

    # Borders run inward from the silhouette, so a border regression changes
    # this row without touching the filled row above it.
    outlined = Group(
        Circle(radius=0.6, color=TRANSPARENT, stroke_color=BLUE_A, stroke_width=6),
        Square(
            size=1.15,
            color=TRANSPARENT,
            stroke_color=GREEN_A,
            stroke_width=14,
        ),
        RegularPolygon(
            6,
            radius=0.66,
            color=MAROON_A,
            stroke_color=WHITE,
            stroke_width=4,
        ),
        Polygon(*STAR_POINTS, color=TRANSPARENT, stroke_color=YELLOW, stroke_width=4),
        Line(start=LEFT * 0.62, end=RIGHT * 0.62, color=ORANGE, stroke_width=4),
        Dot(radius=0.24, color=RED),
    ).arrange_in_line(RIGHT, buffer=0.74)
    outlined.move(DOWN * 0.65 - outlined.get_center())

    outlined_labels = Group(
        Text("thin border", font_size=20, color=GRAY_A, font=FONT),
        Text("thick border", font_size=20, color=GRAY_A, font=FONT),
        Text("border + fill", font_size=20, color=GRAY_A, font=FONT),
        Text("outlined star", font_size=20, color=GRAY_A, font=FONT),
        Text("line", font_size=20, color=GRAY_A, font=FONT),
        Text("dot", font_size=20, color=GRAY_A, font=FONT),
    )
    for shape, label in zip(outlined, outlined_labels):
        label.move_to(shape.get_center() + DOWN * 0.86)

with Seq():
    title.spawn()
    with Lag(0.18, duration=1.6):
        for shape in filled:
            shape.spawn()
    with Sync(duration=0.7):
        filled_labels.spawn()
    with Sync(duration=0.9):
        outlined.spawn()
        outlined_labels.spawn()

# --------------------------------------------------------------------------
# Act 2 -- nested contexts, retroactive duration rescaling and rate functions.
# --------------------------------------------------------------------------
with Off():
    rate_label = Text(
        "Seq / Sync / Lag / Off  +  rate functions",
        font_size=24,
        color=TEAL_A,
        font=FONT,
    ).move(DOWN * 2.5)

with Seq():
    rate_label.spawn()
    # The outer duration rescales every child retroactively: the Seq below runs
    # three edits back to back inside the same 2 seconds the Sync spends on one.
    with Sync(duration=2.0):
        with Seq():
            filled[0].move(UP * 0.5)
            filled[0].color = RED
            filled[0].move(DOWN * 0.5)
        with Sync():
            filled[1].rotate(45, OUT)
            filled[1].scale(1.15)
        with Lag(0.3):
            filled[2].rotate(72, OUT)
            filled[3].rotate(180, OUT)
            filled[4].rotate(120, OUT)
        # Two rate functions over the same span read as two different arrival
        # times for the same displacement.
        with Sync(rate_func=rate_funcs.linear):
            outlined[4].move(UP * 0.32)
        with Sync(rate_func=rate_funcs.ease_out_expo):
            outlined[5].move(UP * 0.32)
    with Sync(duration=1.2):
        filled[5].wave_color(GREEN, direction=RIGHT)
        outlined[2].stroke_color = YELLOW
    Scene.wait(0.2)

# --------------------------------------------------------------------------
# Act 3 -- every indication animation, one per column, staggered by Lag.
# --------------------------------------------------------------------------
with Off():
    rate_label.color = GRAY_B
    indication_label = Text(
        "indication animations",
        font_size=24,
        color=TEAL_A,
        font=FONT,
    ).move(DOWN * 3.0)

with Seq():
    indication_label.spawn()
    with Lag(0.55, duration=3.4):
        Indicate(filled[0], color=WHITE, duration=0.8)
        Wiggle(filled[1], scale_value=1.2, n_wiggles=4, duration=0.8)
        Circumscribe(filled[2], color=TEAL_A, buff=0.15, duration=0.9)
        Flash(filled[3], color=ORANGE, num_lines=10, flash_radius=0.75, duration=0.9)
        FocusOn(filled[4], duration=0.9)
        Blink(filled[5], time_on=0.2, time_off=0.15, blinks=2)
    with Sync(duration=1.5):
        ShowPassingFlash(outlined[0], time_width=0.3, duration=1.5)
        ShowPassingFlashWithThinningStrokeWidth(
            outlined[1], n_segments=6, time_width=0.35, duration=1.5
        )
        ApplyWave(filled_labels, direction=UP, amplitude=0.18, duration=1.5)
    Scene.wait(0.2)

# --------------------------------------------------------------------------
# Act 4 -- become morphing, updaters, DecimalNumber and hand-drawing.
# --------------------------------------------------------------------------
with Sync(duration=0.8):
    filled_labels.despawn()
    outlined_labels.despawn()
    outlined.despawn()
    indication_label.despawn()
    rate_label.despawn()

with Off():
    morph = Square(
        size=1.3,
        color=BLUE,
        stroke_color=WHITE,
        stroke_width=5,
    ).move(LEFT * 4.2 + DOWN * 1.4)
    hub = RegularPolygon(3, radius=0.55, color=MAROON).move(LEFT * 1.4 + DOWN * 1.4)
    satellite = Dot(radius=0.17, color=YELLOW).move(LEFT * 1.4 + DOWN * 1.4)
    counter = (
        DecimalNumber(
            0.0,
            decimal_places=1,
            integer_places=2,
            color=WHITE,
        )
        .scale(2.2)
        .move(RIGHT * 1.6 + DOWN * 1.4)
    )
    counter_frame = SurroundingRectangle(
        counter,
        color=TEAL_A,
        stroke_width=3,
        filled=False,
        buffer=0.22,
    )
    drawn = (
        RegularPolygon(
            3,
            radius=0.8,
            color=GREEN,
            stroke_color=WHITE,
            stroke_width=4,
        )
        .move(RIGHT * 4.5 + DOWN * 1.4)
        .spawn()
    )
    act_label = Text(
        "become            updater            counter            hand-drawn",
        font_size=20,
        color=GRAY_A,
        font=FONT,
    ).move(DOWN * 3.0)

with Seq():
    with Sync(duration=0.6):
        morph.spawn()
        hub.spawn()
        satellite.spawn()
        counter.spawn()
        counter_frame.spawn()
        act_label.spawn()
    # Two updaters: one drives the hub, the other tracks the frame it defines.
    spin_id = hub.add_updater(lambda mob, time: mob.rotate(time * 200.0, OUT))
    orbit_id = satellite.add_updater(
        lambda mob, time: mob.move_to(
            hub.get_center() + hub.get_right_direction() * 1.0
        )
    )
    # ``become`` morphs position as well as shape, so the targets are built
    # where the morphing Mob already is.
    with Sync(duration=1.2):
        morph.become(
            Circle(radius=0.7, color=TEAL, add_to_scene=False).move(
                LEFT * 4.2 + DOWN * 1.4
            )
        )
        counter.set_value(37.5)
    with Sync(duration=1.2):
        morph.become(
            Polygon(*STAR_POINTS, color=ORANGE, add_to_scene=False).move(
                LEFT * 4.2 + DOWN * 1.4
            )
        )
        counter.set_value(-4.0)
    DrawBorderThenFill([drawn], duration=1.4)
    satellite.remove_updater(orbit_id)
    hub.remove_updater(spin_id)
    Scene.wait(0.3)

# --------------------------------------------------------------------------
# Act 5 -- lifecycle: part of the scene leaves the frame, the rest despawns.
# --------------------------------------------------------------------------
with Sync(duration=1.4):
    filled[0].move_off_screen(LEFT, despawn=False)
    filled[5].move_off_screen(RIGHT, despawn=False)
    filled[2].move_off_screen(UP, despawn=False)
    morph.rotate(180, OUT)
    hub.scale(1.4)
    counter.set_value(0.0)

with Sync(duration=0.8):
    hub.despawn()
    satellite.despawn()
    drawn.despawn()
    act_label.despawn()
    counter_frame.despawn()
    morph.despawn()
    counter.despawn()
    filled.despawn()

# --------------------------------------------------------------------------
# Act 6 -- the raw primitives the higher-level shapes are assembled from.
# A per-vertex-coloured triangle, a hand-written cubic circuit, the quad
# families and the point-cloud family each reach the renderer by a different
# route, so they are worth a frame of their own.
# --------------------------------------------------------------------------
with Off():
    gradient_triangle = TriangleTriangulated(
        torch.stack((UP * 0.75, RIGHT * 0.7 + DOWN * 0.6, LEFT * 0.7 + DOWN * 0.6)),
        color=torch.stack([PURE_RED, PURE_GREEN, PURE_BLUE]),
    ).move(LEFT * 4.6 + UP * 0.5)

    # Two cubic segments closing into a leaf shape.
    leaf_controls = torch.stack(
        (
            torch.stack(
                (LEFT * 0.7, LEFT * 0.3 + UP * 0.9, RIGHT * 0.3 + UP * 0.9, RIGHT * 0.7)
            ),
            torch.stack(
                (
                    RIGHT * 0.7,
                    RIGHT * 0.3 + DOWN * 0.9,
                    LEFT * 0.3 + DOWN * 0.9,
                    LEFT * 0.7,
                )
            ),
        )
    )
    raw_circuit = BezierCircuitCubic(
        leaf_controls, color=TEAL, stroke_color=WHITE, stroke_width=4
    ).move(LEFT * 2.3 + UP * 0.5)

    quad = Quad(
        LEFT * 0.7 + DOWN * 0.55,
        RIGHT * 0.7 + DOWN * 0.55,
        RIGHT * 0.55 + UP * 0.6,
        LEFT * 0.8 + UP * 0.6,
        color=ORANGE,
    ).move(UP * 0.5)
    quad_triangulated = QuadTriangulated(
        torch.stack(
            (
                LEFT * 0.7 + DOWN * 0.55,
                RIGHT * 0.7 + DOWN * 0.55,
                RIGHT * 0.7 + UP * 0.6,
                LEFT * 0.7 + UP * 0.6,
            )
        ),
    )
    quad_triangulated.set_material(MeshBasicMaterial(color=PURPLE))
    quad_triangulated.move(RIGHT * 2.6 + UP * 0.5)

    surrounding = SurroundingRectangle(
        quad_triangulated,
        color=TEAL_A,
        stroke_width=3,
        filled=False,
        buffer=0.18,
    )

    primitive_labels = Group(
        Text("per-vertex colour", font_size=19, color=GRAY_A, font=FONT).move(
            LEFT * 4.6 + DOWN * 0.85
        ),
        Text("raw circuit", font_size=19, color=GRAY_A, font=FONT).move(
            LEFT * 2.3 + DOWN * 0.85
        ),
        Text("Quad", font_size=19, color=GRAY_A, font=FONT).move(DOWN * 0.85),
        Text("QuadTriangulated", font_size=19, color=GRAY_A, font=FONT).move(
            RIGHT * 2.6 + DOWN * 0.85
        ),
    )

with Seq():
    with Lag(0.15, duration=1.4):
        gradient_triangle.spawn()
        raw_circuit.spawn()
        quad.spawn()
        quad_triangulated.spawn()
    with Sync(duration=0.5):
        primitive_labels.spawn()
        surrounding.spawn()
    with Sync(duration=1.2):
        gradient_triangle.rotate(180, UP)
        raw_circuit.rotate(45, OUT)
        quad.rotate(-30, OUT)
    Scene.wait(0.3)

# --------------------------------------------------------------------------
# Act 7 -- point-cloud APIs, rendered through packed native sphere geometry.
# PGroup's members are intentionally not separate Scene actors: this exercises
# the composite's own primitive delegation as well as the three leaf classes.
# --------------------------------------------------------------------------
with Sync(duration=0.6):
    gradient_triangle.despawn()
    raw_circuit.despawn()
    quad.despawn()
    quad_triangulated.despawn()
    surrounding.despawn()
    primitive_labels.despawn()

with Off():
    dot_cloud = mn.DotCloud(
        points=torch.stack(
            (
                LEFT * 0.55 + DOWN * 0.45,
                RIGHT * 0.55 + DOWN * 0.45,
                UP * 0.55,
                ORIGIN,
            )
        ),
        stroke_width=10,
        color=YELLOW,
    ).move(LEFT * 4.2 + UP * 0.55)
    point_cloud_dot = mn.PointCloudDot(
        radius=0.62,
        density=5,
        stroke_width=7,
        color=BLUE_A,
    ).move(LEFT * 1.4 + UP * 0.55)
    true_dot = mn.TrueDot(stroke_width=16, color=GREEN_A).move(RIGHT * 1.4 + UP * 0.55)
    point_group = mn.PGroup(
        mn.DotCloud(
            points=torch.stack((LEFT * 0.42, RIGHT * 0.42, UP * 0.5)),
            stroke_width=9,
            color=ORANGE,
            add_to_scene=False,
        ),
        mn.TrueDot(
            center=DOWN * 0.48,
            stroke_width=14,
            color=PURPLE,
            add_to_scene=False,
        ),
    ).move(RIGHT * 4.2 + UP * 0.55)
    point_labels = Group(
        Text("DotCloud", font_size=20, color=GRAY_A, font=FONT).move(
            LEFT * 4.2 + DOWN * 0.75
        ),
        Text("PointCloudDot", font_size=20, color=GRAY_A, font=FONT).move(
            LEFT * 1.4 + DOWN * 0.75
        ),
        Text("TrueDot", font_size=20, color=GRAY_A, font=FONT).move(
            RIGHT * 1.4 + DOWN * 0.75
        ),
        Text("PGroup", font_size=20, color=GRAY_A, font=FONT).move(
            RIGHT * 4.2 + DOWN * 0.75
        ),
    )

with Seq():
    with Lag(0.15, duration=1.2):
        dot_cloud.spawn()
        point_cloud_dot.spawn()
        true_dot.spawn()
        point_group.spawn()
    with Sync(duration=0.5):
        point_labels.spawn()
    with Sync(duration=1.2):
        dot_cloud.rotate(90, OUT)
        point_cloud_dot.scale(1.25)
        true_dot.move(UP * 0.35)
        point_group.rotate(-90, OUT)
    Scene.wait(0.3)
