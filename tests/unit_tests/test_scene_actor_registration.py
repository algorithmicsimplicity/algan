"""Composite Mobs must register the geometry they are built out of.

Algan builds render primitives from the Scene's actor list -- ``Mob`` and
``Group`` define no ``get_render_primitives`` at all, so a composite renders
*only* through its registered leaf geometry.  A part built with
``add_to_scene=False`` and then attached with ``add_children`` is therefore
invisible: spawned, styled, carrying primitives, and never drawn.

Each test here pins a composite that used to be built exactly that way.
"""
from __future__ import annotations

import pytest

import algan


@pytest.fixture(autouse=True)
def fresh_scene_stack():
    algan.SceneManager.reset()
    yield
    algan.SceneManager.reset()


def invisible_geometry(mob):
    """Spawned geometry of ``mob`` that no actor will ever draw.

    Two legitimate ways exist for a descendant to render without being an actor
    itself, and both are excluded here:

    * an ancestor actor aggregates its descendants' primitives in its own
      ``get_render_primitives`` (:class:`~.Polyhedron` does this for its faces);
    * it is an index view (``mob[0]``), which shares its source's timeline rows
      and id, so the registered source draws the same geometry.
    """
    actors = list(mob.scene.actors)
    actor_ids = {id(actor) for actor in actors}
    actor_timeline_ids = {actor.id for actor in actors}

    def has_aggregating_ancestor(node):
        seen, stack = set(), list(node.parents)
        while stack:
            parent = stack.pop()
            if id(parent) in seen:
                continue
            seen.add(id(parent))
            if hasattr(parent, "get_render_primitives") and id(parent) in actor_ids:
                return True
            stack.extend(parent.parents)
        return False

    return [
        descendant
        for descendant in mob.get_descendants()
        if hasattr(descendant, "get_render_primitives")
        and descendant.is_spawned()
        and id(descendant) not in actor_ids
        and descendant.id not in actor_timeline_ids
        and not has_aggregating_ancestor(descendant)
    ]


def _spawned(mob):
    with algan.Off():
        mob.spawn()
    return mob


def test_animated_boundary_layers_are_registered():
    source = _spawned(algan.Square(color=algan.TRANSPARENT, border_width=0))
    boundary = _spawned(
        algan.AnimatedBoundary(source, max_stroke_width=14, cycle_rate=1.0)
    )

    actors = {id(actor) for actor in source.scene.actors}
    for layer in boundary.boundary_copies:
        assert all(
            id(descendant) in actors for descendant in layer.get_descendants()
        )
    assert invisible_geometry(boundary) == []


def test_animated_boundary_half_width_conversion():
    """Algan stores half-widths where Manim's public API takes full strokes."""
    source = _spawned(algan.Square(color=algan.TRANSPARENT, border_width=0))
    boundary = algan.AnimatedBoundary(source, max_stroke_width=14)

    assert boundary.max_stroke_width == 14
    assert boundary.max_border_width == 7


def test_paragraph_lines_are_registered():
    paragraph = _spawned(algan.Paragraph("hello", "world"))

    assert invisible_geometry(paragraph) == []


def test_paragraph_built_detached_registers_nothing():
    """``add_to_scene`` has to reach the lines, or morph targets leak actors.

    ``Paragraph.set_all_lines_alignments`` builds a detached Paragraph purely as
    a ``become`` target, so ``add_to_scene=False`` must keep its lines out of the
    scene too.
    """
    square = algan.Square()
    before = len(square.scene.actors)

    algan.Paragraph("hello", "world", scene=square.scene, add_to_scene=False)

    assert len(square.scene.actors) == before


@pytest.mark.parametrize("background", ["rectangle", "window", None])
def test_code_renders_all_of_its_parts(background):
    code = _spawned(
        algan.Code(code_string="x = 1\ny = x + 2", background=background)
    )

    assert invisible_geometry(code) == []


def test_code_built_detached_registers_nothing():
    square = algan.Square()
    before = len(square.scene.actors)

    algan.Code(
        code_string="x = 1",
        background="window",
        scene=square.scene,
        add_to_scene=False,
    )

    assert len(square.scene.actors) == before


def test_image_mobject_display_frame_is_registered():
    mob = algan.ImageMobjectFromCamera(algan.Scene.get_camera())
    mob.add_display_frame()
    _spawned(mob)

    actors = {id(actor) for actor in mob.scene.actors}
    assert id(mob.display_frame) in actors
    assert invisible_geometry(mob) == []


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda: algan.Text("ab"), id="Text"),
        pytest.param(lambda: algan.Tex("x^2"), id="Tex"),
        pytest.param(lambda: algan.NumericDisplay(1.5), id="NumericDisplay"),
        pytest.param(lambda: algan.Cube(), id="Cube"),
        pytest.param(lambda: algan.Dodecahedron(), id="Dodecahedron"),
        pytest.param(lambda: algan.Sphere(resolution=(4, 3)), id="Sphere"),
        pytest.param(
            lambda: algan.SurroundingRectangle(algan.Square()),
            id="SurroundingRectangle",
        ),
        pytest.param(
            lambda: algan.Group([algan.Square() for _ in range(3)]), id="Group"
        ),
        pytest.param(
            lambda: algan.Axes(x_range=(-2, 2, 1), y_range=(-2, 2, 1)), id="Axes"
        ),
        pytest.param(lambda: algan.Brace(algan.Square()), id="Brace"),
    ],
)
def test_composites_leave_no_invisible_geometry(build):
    assert invisible_geometry(_spawned(build())) == []
