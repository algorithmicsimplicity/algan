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
import algan.manim as mn


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
    source = _spawned(algan.Square(color=algan.TRANSPARENT, stroke_width=0))
    boundary = _spawned(
        algan.AnimatedBoundary(source, max_stroke_width=7, cycle_rate=1.0)
    )

    actors = {id(actor) for actor in source.scene.actors}
    for layer in boundary.boundary_copies:
        assert all(id(descendant) in actors for descendant in layer.get_descendants())
    assert invisible_geometry(boundary) == []


def test_animated_boundary_width_is_in_algan_units():
    """``max_stroke_width`` is Algan's unit, and reaches the layers verbatim.

    It used to be Manim's and be halved on the way in, which made
    ``AnimatedBoundary`` the one place in the root namespace where a stroke
    width meant something different from every other. Manim means twice this
    number by the same argument; ``algan.manim`` is where that lives.
    """
    source = _spawned(algan.Square(color=algan.TRANSPARENT, stroke_width=0))
    boundary = algan.AnimatedBoundary(source, max_stroke_width=7)

    assert boundary.max_stroke_width == 7
    widest = max(
        float(layer.stroke_width.reshape(-1).max())
        for layer in boundary.boundary_copies
    )
    assert widest == 7


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
    code = _spawned(algan.Code(code_string="x = 1\ny = x + 2", background=background))

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
    mob = mn.ImageMobjectFromCamera(algan.Scene.get_camera())
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
        pytest.param(lambda: algan.DecimalNumber(1.5), id="DecimalNumber"),
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


def unbuilt_geometry(mob):
    """Spawned geometry of ``mob`` that the actor walk never asks to build.

    ``invisible_geometry`` above excuses a descendant whose ancestor is an
    actor carrying ``get_render_primitives``, which assumes that builder
    reaches down to it.  The two stock builders do not: ``Surface`` and
    ``BezierCircuitCubic`` emit their own geometry and nothing else, so a part
    attached to a Cylinder is not drawn just because the Cylinder is -- which
    is how the cylinder and cone caps stayed unregistered and undetected.

    Spying on the builders answers the question exactly instead of assuming:
    run the collection the render loop runs (``get_render_primitives`` on each
    Scene actor, never a walk of the hierarchy) and see who gets asked.
    """
    geometry = [
        descendant
        for descendant in mob.get_descendants()
        if hasattr(descendant, "get_render_primitives") and descendant.is_spawned()
    ]
    asked = set()
    for part in geometry:
        original = part.get_render_primitives

        def spy(part=part, original=original):
            asked.add(id(part))
            return original()

        part.get_render_primitives = spy
    try:
        for actor in list(mob.scene.actors):
            builder = getattr(actor, "get_render_primitives", None)
            if builder is not None:
                builder()
    finally:
        for part in geometry:
            del part.get_render_primitives
    return [part for part in geometry if id(part) not in asked]


# ``show_ends``/``show_base`` promise a closed solid, and an open one is not
# merely a shading difference: a ray enters through the missing cap and hits
# whatever is inside.  That is what put a white speck of the axis Line3D on the
# red Arrow3D of ``tests/full_renders/solids_and_camera`` at the seam between
# its shaft and its head.
_CAPPED = {
    "cylinder": lambda **kw: algan.Cylinder(
        radius=0.4, height=1.0, show_ends=True, **kw
    ),
    "cone": lambda **kw: algan.Cone(base_radius=0.5, height=1.0, show_base=True, **kw),
    # Line3D is a capped Cylinder, and Arrow3D is one plus a capped Cone --
    # whose caps hang off parts that are not actors themselves.
    "line3d": lambda **kw: algan.Line3D(
        start=algan.LEFT, end=algan.RIGHT, radius=0.08, **kw
    ),
    "arrow3d": lambda **kw: algan.Arrow3D(
        start=algan.ORIGIN, end=algan.RIGHT * 1.1, shaft_radius=0.05, **kw
    ),
}


@pytest.mark.parametrize("name", sorted(_CAPPED))
def test_capped_solids_draw_their_caps(name):
    mob = _spawned(_CAPPED[name]())

    assert invisible_geometry(mob) == []
    assert unbuilt_geometry(mob) == []


def test_add_bases_after_spawn_draws_the_caps():
    """``Cylinder`` documents ``add_bases()`` as a post-construction capping.

    Spawning is recursive from the parent, so a cap attached after the tube has
    spawned never gets a spawn of its own -- and ``_actor_window_index`` drops a
    never-spawned actor, which is the open tube again. ``invisible_geometry``
    cannot see that on its own: it only considers spawned descendants.
    """
    cylinder = _spawned(algan.Cylinder(radius=0.4, height=1.0))

    cylinder.add_bases()

    assert cylinder.bottom_cap.is_spawned()
    assert cylinder.top_cap.is_spawned()
    assert invisible_geometry(cylinder) == []
    assert unbuilt_geometry(cylinder) == []


def test_add_bases_twice_keeps_one_pair_of_caps():
    """A second call re-aims the caps rather than stacking a second pair.

    The replaced pair would stay attached and registered behind the new one,
    drawing twice at the same depth.
    """
    cylinder = _spawned(algan.Cylinder(radius=0.4, height=1.0, show_ends=True))
    caps = (cylinder.bottom_cap, cylinder.top_cap)
    before = len(cylinder.scene.actors)

    cylinder.add_bases()

    assert (cylinder.bottom_cap, cylinder.top_cap) == caps
    assert len(cylinder.scene.actors) == before
    assert sum(1 for child in cylinder.children if child in caps) == 2


def test_uncapped_cone_registers_no_base():
    """``base_circle`` is built for every cone; only a capped one draws it."""
    cone = _spawned(algan.Cone(base_radius=0.5, height=1.0))

    assert cone.base_circle is not None
    assert not any(child is cone.base_circle for child in cone.children)
    assert id(cone.base_circle) not in {id(actor) for actor in cone.scene.actors}


@pytest.mark.parametrize("name", sorted(_CAPPED))
def test_detached_capped_solids_register_nothing(name):
    """A cap must not register itself when its solid was built detached.

    ``add_to_scene=False`` marks a Mob nobody intends to show -- a morph target,
    a measurement -- and a cap that registered anyway would draw a disc with no
    tube behind it.

    Counted over geometry rather than over every actor: ``Arrow3D`` also
    registers two ``opacity=0`` marker Mobs for ``get_start``/``get_end``, which
    predate this and draw nothing.
    """
    square = algan.Square()

    def renderable_actors():
        return [
            actor
            for actor in square.scene.actors
            if hasattr(actor, "get_render_primitives")
        ]

    before = len(renderable_actors())

    _CAPPED[name](scene=square.scene, add_to_scene=False)

    assert len(renderable_actors()) == before
