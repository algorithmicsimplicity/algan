"""Scene ownership of Mobs produced by the Manim compatibility layer.

Algan collects render primitives from the Scene's actor list rather than by
walking the mob hierarchy, so a Mob that was never registered with its Scene
cannot render no matter how it is spawned or styled.  These tests pin down which
conversions in ``algan.mobs.manim_compat`` register and which deliberately do
not.
"""
from __future__ import annotations

import manim as mn
import pytest

import algan
from algan.mobs.manim_mob import ManimMob


@pytest.fixture(autouse=True)
def fresh_scene_stack():
    algan.SceneManager.reset()
    yield
    algan.SceneManager.reset()


def _actor_ids(scene):
    return {id(actor) for actor in scene.actors}


def _renderable_actors(scene):
    """The actors that ``get_batch_of_primitives`` would consider."""
    return [
        actor
        for actor in scene.actors
        if actor.lifespan.start() >= 0 and hasattr(actor, "get_render_primitives")
    ]


def test_axes_plot_result_is_registered_with_the_owning_scene():
    axes = algan.Axes(x_range=(-3, 3, 1), y_range=(-1.5, 1.5, 0.5))
    graph = axes.plot(lambda x: x * x)

    assert graph.scene is axes.scene
    actors = _actor_ids(axes.scene)
    assert id(graph) in actors
    # Every renderable part of the conversion has to be an actor in its own
    # right, not just the root the caller was handed.
    assert all(id(descendant) in actors for descendant in graph.get_descendants())


def test_registered_plot_reaches_the_renderer_once_spawned():
    axes = algan.Axes(x_range=(-3, 3, 1), y_range=(-1.5, 1.5, 0.5))
    graph = axes.plot(lambda x: x * x)
    with algan.Off():
        graph.spawn()

    renderable = _renderable_actors(axes.scene)
    assert any(actor is graph for actor in renderable)
    assert any(
        actor.get_render_primitives() is not None
        for actor in renderable
        if actor is graph
    )


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(
            lambda axes: axes.plot_parametric_curve(
                lambda t: (t, t * t, 0.0), t_range=(-1, 1)
            ),
            id="plot_parametric_curve",
        ),
        pytest.param(
            lambda axes: axes.get_axis_labels("x", "y"), id="get_axis_labels"
        ),
        pytest.param(
            lambda axes: axes.get_graph_label(axes.plot(lambda x: x), "f"),
            id="get_graph_label",
        ),
    ],
)
def test_every_delegated_builder_registers_what_it_returns(build):
    axes = algan.Axes(x_range=(-3, 3, 1), y_range=(-1.5, 1.5, 0.5))
    result = build(axes)

    assert isinstance(result, algan.Mob)
    assert id(result) in _actor_ids(axes.scene)


def test_brace_get_text_is_registered():
    square = algan.Square()
    brace = algan.Brace(square)
    text = brace.get_text("width")

    assert id(text) in _actor_ids(square.scene)


def test_delegated_accessor_for_own_geometry_does_not_duplicate_actors():
    """``get_x_axis`` hands back geometry this Mob already draws.

    Registering a second conversion of it would put coincident duplicate
    geometry in the scene, so only newly built Mobjects are registered.
    """
    axes = algan.Axes(x_range=(-3, 3, 1), y_range=(-1.5, 1.5, 0.5))
    before = len(axes.scene.actors)

    axes.get_x_axis()

    assert len(axes.scene.actors) == before


def test_reading_a_manim_attribute_has_no_effect_on_the_scene():
    """Attribute access is a query: repeated reads must not accumulate actors."""
    axes = algan.Axes(x_range=(-3, 3, 1), y_range=(-1.5, 1.5, 0.5))
    before = len(axes.scene.actors)

    axes.x_axis
    axes.x_axis

    assert len(axes.scene.actors) == before


def test_copy_is_renderable_like_clone():
    square = algan.Square()
    circle = algan.Arc(add_to_scene=False)
    copied = circle.copy()

    assert id(copied) in _actor_ids(square.scene)


def test_manim_mob_add_to_scene_false_registers_nothing_at_all():
    """``add_to_scene`` governs the whole converted subtree, not just its root.

    One Manim Mobject becomes a whole Algan sub-hierarchy, and each renderable
    part is a separate actor, so a conversion asked not to join the scene must
    keep its children out too -- otherwise intermediate conversions (morph
    targets) pile up dead actors.
    """
    square = algan.Square()
    scene = square.scene
    before = len(scene.actors)

    mob = ManimMob(
        mn.VGroup(mn.Circle(), mn.Square()), scene=scene, add_to_scene=False
    )

    assert len(scene.actors) == before
    actors = _actor_ids(scene)
    assert id(mob) not in actors
    assert all(id(descendant) not in actors for descendant in mob.get_descendants())


def test_manim_mob_batching_registers_only_the_batched_mob():
    square = algan.Square()
    scene = square.scene

    mob = ManimMob(mn.VGroup(mn.Circle(), mn.Square()), scene=scene, batch=True)

    actors = _actor_ids(scene)
    assert id(mob) in actors
    assert all(id(child) not in actors for child in mob.submobjects)


def test_transforming_a_compat_mob_does_not_leak_morph_targets():
    """``move_to``/``scale``/``rotate`` morph through a throwaway conversion.

    Nothing of that target survives the ``become``, so none of it may be
    registered -- each transform used to leak one actor per Manim submobject.
    """
    axes = algan.Axes(x_range=(-3, 3, 1), y_range=(-1.5, 1.5, 0.5))
    with algan.Off():
        axes.spawn()
    before = len(axes.scene.actors)

    with algan.Off():
        axes.scale(1.5)
        axes.rotate(0.25)

    assert len(axes.scene.actors) == before


def test_structure_changing_delegated_method_keeps_its_additions_renderable():
    """``add_coordinates`` grafts new submobjects in via ``sync_from_manim``.

    Those grafted children are part of this Mob's hierarchy and must render, so
    unlike a morph target they are registered.
    """
    axes = algan.Axes(x_range=(-3, 3, 1), y_range=(-1.5, 1.5, 0.5))
    axes.add_coordinates()
    with algan.Off():
        axes.spawn()

    actors = _actor_ids(axes.scene)
    grafted = axes.get_non_component_children()
    assert grafted
    renderable = [
        descendant
        for child in grafted
        for descendant in child.get_descendants()
        if id(descendant) in actors
        and descendant.is_spawned()
        and hasattr(descendant, "get_render_primitives")
    ]
    assert renderable
