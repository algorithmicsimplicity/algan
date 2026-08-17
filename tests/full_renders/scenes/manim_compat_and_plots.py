"""The Manim-compatibility surface: plots, tables, braces and Manim animations.

Every Mob here is backed by a real Manim ``VMobject`` that Algan converts to its
own bezier geometry.  Two things have historically broken silently on this path
and both are visible in these frames:

* geometry handed back by a *delegated* Manim method (``axes.plot(...)``,
  ``brace.get_text(...)``) has to be registered as an actor on the owning
  Scene, otherwise it renders as nothing at all;
* the Manim-flavoured animations (``ApplyMatrix``, ``ApplyComplexFunction``,
  ``MoveAlongPath``, ``Homotopy``) have to materialise their target geometry
  once per frame rather than once per batch.

Compatibility Mobs are positioned and transformed *directly* rather than
through a parent ``Group``: a parent-driven transform leaves their backing
Manim object behind, which the accompanying unit test pins as a known defect.
"""

from algan import *

# Pinned so the render does not depend on the host's fonts;
# tests/conftest.py registers the vendored faces.
FONT = "Algan Test Sans"

Scene.set_background_color(DARKER_GRAY)

# --------------------------------------------------------------------------
# Act 1 -- axes with delegated plots, a brace, and a number plane.
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
        "MANIM COMPATIBILITY",
        font_size=42,
        weight="BOLD",
        color=WHITE,
        font=FONT,
    ).move(UP * 3.05)

    axes = Axes(
        x_range=(-3, 3, 1),
        y_range=(-2, 2, 1),
        x_length=5.2,
        y_length=3.2,
    ).move(LEFT * 3.3 + UP * 0.4)
    parabola = axes.plot(lambda x: 0.32 * x * x - 1.4, color=YELLOW)
    line_plot = axes.plot(lambda x: 0.5 * x, color=BLUE)
    brace = Brace(parabola, direction=DOWN)
    brace_text = brace.get_text("plot()")

    plane = NumberPlane(
        x_range=(-2, 2, 1),
        y_range=(-2, 2, 1),
        x_length=3.4,
        y_length=3.4,
    ).move(RIGHT * 3.6 + UP * 0.4)
    vector = Vector(RIGHT * 1.2 + UP * 1.0, color=ORANGE).move(RIGHT * 3.6 + UP * 0.4)
    dashed = DashedLine(
        start=LEFT * 1.5, end=RIGHT * 1.5, color=GREEN_A, stroke_width=5
    ).move(RIGHT * 3.6 + DOWN * 1.5)

with Seq():
    title.spawn()
    with Sync(run_time=0.8):
        axes.spawn()
        plane.spawn()
    with Lag(0.25, run_time=1.5):
        parabola.spawn()
        line_plot.spawn()
        vector.spawn()
        dashed.spawn()
    with Sync(run_time=0.7):
        brace.spawn()
        brace_text.spawn()
    with Sync(run_time=1.2):
        parabola.color = TEAL_A
        line_plot.color = MAROON_A
        vector.rotate(40)
    Scene.wait(0.2)

# --------------------------------------------------------------------------
# Act 2 -- tables, matrices and a graph.
# --------------------------------------------------------------------------
with Sync(run_time=0.8):
    for mob in (axes, parabola, line_plot, brace, brace_text, plane, vector, dashed):
        mob.despawn()

with Off():
    chart = BarChart(
        values=[2.0, 4.0, 1.5, 3.0],
        y_range=[0, 5, 1],
        x_length=4.0,
        y_length=2.8,
        bar_colors=["#4c9eff", "#5cd65c", "#ffb14c", "#ff6f6f"],
    ).move(LEFT * 3.6 + UP * 0.3)
    matrix = IntegerMatrix([[1, 2], [3, 4]]).scale(0.85).move(RIGHT * 0.4 + UP * 0.5)
    table = IntegerTable([[1, 2], [3, 4]]).scale(0.6).move(RIGHT * 3.9 + UP * 0.4)
    graph = Graph(
        [1, 2, 3, 4],
        [(1, 2), (2, 3), (3, 4), (1, 4)],
        layout="circular",
        layout_scale=0.85,
    ).move(DOWN * 1.95)
    data_label = Text(
        "BarChart      IntegerMatrix      IntegerTable      Graph",
        font_size=22,
        color=TEAL_A,
        font=FONT,
    ).move(DOWN * 3.15)

with Seq():
    with Lag(0.2, run_time=1.6):
        chart.spawn()
        matrix.spawn()
        table.spawn()
        graph.spawn()
    data_label.spawn()
    with Sync(run_time=1.2):
        matrix.rotate(20)
        graph.scale(1.25)
    Scene.wait(0.2)

# --------------------------------------------------------------------------
# Act 3 -- Manim-flavoured shapes and the Manim animation set.
# --------------------------------------------------------------------------
with Sync(run_time=0.8):
    for mob in (chart, matrix, table, graph, data_label):
        mob.despawn()

with Off():
    star = Star(outer_radius=0.75, inner_radius=0.34, color=YELLOW).move(
        LEFT * 4.4 + UP * 1.3
    )
    annulus = Annulus(inner_radius=0.3, outer_radius=0.7, color=PURPLE).move(
        LEFT * 2.2 + UP * 1.3
    )
    arc = ArcBetweenPoints(
        start=LEFT * 0.7, end=RIGHT * 0.7, angle=1.6, color=GREEN_A, stroke_width=6
    ).move(UP * 1.3)
    compat_arrow = Arrow(
        start=LEFT * 0.8, end=RIGHT * 0.8, color=ORANGE, stroke_width=8
    ).move(RIGHT * 2.2 + UP * 1.3)
    right_angle = RightAngle(
        Line(start=ORIGIN, end=RIGHT, add_to_scene=False),
        Line(start=ORIGIN, end=UP, add_to_scene=False),
        color=BLUE_A,
    ).move(RIGHT * 4.3 + UP * 1.3)

    grid = Square(side_length=1.3, color=BLUE).move(LEFT * 4.2 + DOWN * 1.4)
    traveller = Dot(radius=0.16, color=YELLOW).move(LEFT * 1.4 + DOWN * 1.4)
    path = Circle(radius=0.75, color=TRANSPARENT, border_color=GRAY_B, border_width=2)
    path.move(LEFT * 1.4 + DOWN * 1.4)
    waved = Square(side_length=1.2, color=GREEN).move(RIGHT * 1.4 + DOWN * 1.4)
    waved_y = float(waved.get_center().flatten()[1])
    bounded = Square(side_length=1.2, color=TRANSPARENT, border_width=0).move(
        RIGHT * 4.2 + DOWN * 1.4
    )
    boundary = AnimatedBoundary(bounded, max_stroke_width=10, cycle_rate=1.0)
    anim_label = Text(
        "ApplyMatrix    MoveAlongPath    Homotopy    AnimatedBoundary",
        font_size=21,
        color=TEAL_A,
        font=FONT,
    ).move(DOWN * 3.15)

with Seq():
    with Sync(run_time=0.9):
        star.spawn()
        annulus.spawn()
        arc.spawn()
        compat_arrow.spawn()
        right_angle.spawn()
    with Sync(run_time=0.9):
        grid.spawn()
        path.spawn()
        traveller.spawn()
        waved.spawn()
        bounded.spawn()
        boundary.spawn()
        anim_label.spawn()
    with Sync(run_time=1.8):
        # Directly transforming a compatibility Mob keeps its backing Manim
        # object in step, which is the supported way to move one.
        star.rotate(50)
        annulus.scale(1.25)
        compat_arrow.rotate(70)
        ApplyMatrix(grid, [[1.0, 0.6], [0.0, 1.0]], run_time=1.8)
        MoveAlongPath(traveller, path, run_time=1.8)
        # A genuine deformation, not a translation: the shear grows with y.
        Homotopy(
            waved,
            lambda x, y, z, t: (x + 0.55 * t * (y - waved_y), y, z),
            run_time=1.8,
        )
    with Sync(run_time=1.6):
        ApplyComplexFunction(grid, lambda z: z * (0.8 + 0.4j), run_time=1.6)
        MoveAlongPath(traveller, path, run_time=1.6)
        star.rotate(-50)
    Scene.wait(0.4)
