"""The one scene the fast suite renders: every geometry family in one frame.

This scene exists to be *cheap*, which makes its shape unusual — read this
before adding to it.

Taichi specialises the render kernel on which geometry families and features a
batch actually contains, and the first render of a given variant in a process
costs tens of seconds (measured: ~17 s for circuits alone, ~42 s for meshes,
~13 s for text) against ~0.5–1.5 s once that variant is warm.  The cost is
therefore paid **per kernel variant, not per scene**: five small scenes with
different geometry mixes cost five warm-ups, while one scene holding all of
them costs one.  That is the whole reason the fast suite is a single dense
scene, and the reason to extend this file rather than add a second one.

Held to the same constraint, the scene stays short (about three seconds) and
puts everything on screen at once in labelled, non-overlapping columns, so a
regression reads as a diff in one column:

* **cubic bezier circuits** — filled, border-only (borders run inward) and
  non-convex, so the fill rule, the triangulation of a re-entrant outline and
  the analytic AA on both the silhouette and the border/fill seam are covered;
* **glyph circuits** — ``Text`` and ``Tex``, which reach the rasteriser by the
  same route but are built through the Manim geometry cache;
* **flat triangle meshes** — ``Cube``, ``Icosahedron``, ``Octahedron``,
  straight into the flat-triangle BVH;
* **materials and lights** — the unlit, Lambert and PBR shading paths under an
  ambient, a directional and a point light;
* **the timeline** — all four animation contexts, a rate function, an updater
  and the spawn/despawn lifecycle, so a replay regression shows up as motion
  arriving at the wrong frame.

**No PN surfaces.** ``Sphere``, ``Cylinder``, ``Cone``, ``Torus`` and
``Surface`` tessellate to logical PN triangles, and adding any one of them to
this scene costs about 20 seconds of extra kernel specialisation — measured:
24.6 s for this scene, 44.8 s with a single ``Sphere`` in it.  That is a sixth
of the fast suite's whole budget for one geometry family, and the budget exists
to keep the loop under a minute.  The PN family is covered behaviourally by
``test_logical_pn_tessellation.py`` and ``test_surface_autotune.py``, and in
pixels by ``tests/full_renders/solids_and_camera`` — both outside the fast
suite, so a tessellation change is one of the few things this loop cannot see
(``CLAUDE.md`` says so under performance discipline).  Use a ``Polyhedron``
subclass when adding a solid here, never a ``Surface`` one.

Deliberately *not* here for the same reason: shadows, refraction, glow, Monte
Carlo sampling, glTF import and the camera moves.  Each pulls in another kernel
variant or another tracer path.  ``tests/full_renders/`` covers them all.

Like the full-render scenes, this file only *records* an animation; the harness
owns the Scene, the settings and the comparison.
"""

from algan import *

# Pinned so the render does not depend on the host's fonts;
# tests/conftest.py registers the vendored faces.
FONT = "Algan Test Sans"

Scene.set_background(DARKER_GRAY)

# Five outer and five inner points: a re-entrant outline, so the fill rule and
# the triangulation of a non-convex circuit are both under test.
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

with Off():
    # Three light types, so the unlit, Lambert and PBR paths are all driven by
    # something other than a constant.
    AmbientLight(color=WHITE, intensity=0.45).spawn(animate=False)
    DirectionalLight(
        location=RIGHT * 5 + UP * 5 + OUT * 4,
        target=ORIGIN,
        color=WHITE,
        intensity=0.85,
    ).spawn(animate=False)
    PointLight(
        location=LEFT * 3 + UP * 2 + OUT * 3,
        color=BLUE_A,
        intensity=0.6,
    ).spawn(animate=False)

    title = Text(
        "FAST SUITE", font_size=34, weight="BOLD", color=WHITE, font=FONT
    ).move(UP * 2.9)

    # Row 1 -- bezier circuits: filled, border-only, and non-convex.
    circuits = Group(
        Circle(radius=0.5, color=BLUE),
        Square(
            size=0.95,
            color=TRANSPARENT,
            stroke_color=GREEN_A,
            stroke_width=10,
        ),
        RegularPolygon(
            5, radius=0.55, color=MAROON_A, stroke_color=WHITE, stroke_width=4
        ),
        Polygon(*STAR_POINTS, color=YELLOW),
    ).arrange_in_line(RIGHT, buffer=0.7)
    circuits.move(UP * 1.35 - circuits.get_center())

    # Row 2 -- flat triangle meshes, one per shading path. Every solid here is
    # a Polyhedron, never a Surface: see the module docstring on why the PN
    # family is deliberately absent.
    cube = Cube(size=0.95).set_material(MeshLambertMaterial(color=ORANGE))
    cube.move(LEFT * 2.2 + DOWN * 0.5)
    metal = Icosahedron(edge_length=0.85).set_material(
        MeshStandardMaterial(color=RED, roughness=0.35, metalness=0.4)
    )
    metal.move(LEFT * 0.2 + DOWN * 0.5)
    unlit = Octahedron(edge_length=0.9).set_material(MeshBasicMaterial(color=TEAL))
    unlit.move(RIGHT * 1.8 + DOWN * 0.5)
    # Opacity is its own transport channel, not a shading term.
    faded = Cube(size=0.85, opacity=0.45).set_material(
        MeshLambertMaterial(color=PURPLE)
    )
    faded.move(RIGHT * 3.6 + DOWN * 0.5)

    # Row 3 -- glyph circuits, built through the Manim geometry cache.
    formula = Tex(r"e^{i\pi}+1=0", color=YELLOW).scale(0.8).move(DOWN * 2.1)
    caption = Text(
        "circuits / meshes / glyphs", font_size=20, color=GRAY_A, font=FONT
    ).move(DOWN * 2.9)

with Seq():
    title.spawn()
    # Lag staggers the spawns, so a replay regression moves geometry between
    # frames rather than changing a single still.
    with Lag(0.25, duration=1.0):
        for shape in circuits:
            shape.spawn()
    with Sync(duration=0.6):
        cube.spawn()
        metal.spawn()
        unlit.spawn()
        faded.spawn()
        formula.spawn()
        caption.spawn()

    # An updater writes every frame; a rate function makes two equal
    # displacements arrive at different times.
    spin = cube.add_updater(lambda mob, time: mob.rotate(time * 120.0, UP))
    with Sync(duration=1.0):
        metal.rotate(60, RIGHT)
        circuits[3].rotate(180, OUT)
        with Sync(rate_func=rate_funcs.linear):
            circuits[0].move(UP * 0.3)
        with Sync(rate_func=rate_funcs.ease_out_expo):
            circuits[1].move(UP * 0.3)
    cube.remove_updater(spin)

    # Part of the scene leaves, so the despawn half of the lifecycle is drawn.
    with Sync(duration=0.4):
        unlit.despawn()
        circuits[2].despawn()
