"""Contracts of the indication animations.

These are the most-used "look here" helpers, and they share two failure modes
that produce no error at all:

* the helper geometry several of them build (``Flash``'s spokes, ``FocusOn``'s
  spotlight, ``Circumscribe``'s frame) has to be registered as an actor on the
  owning Scene -- the render loop iterates ``scene.actors`` rather than walking
  the hierarchy, so unregistered geometry is simply invisible;
* they each take an explicit ``runtime`` that must win over the enclosing
  context, and must leave the Mob in the state they found it (that is what
  makes them safe to drop into an existing animation).
"""

from __future__ import annotations

import pytest
import torch

from algan import (
    ApplyWave,
    Blink,
    Circle,
    Circumscribe,
    Flash,
    FocusOn,
    Group,
    Indicate,
    Line,
    Off,
    Scene,
    Seq,
    ShowPassingFlash,
    ShowPassingFlashWithThinningStrokeWidth,
    Square,
    Wiggle,
)
from algan.constants.color import BLUE, PURE_GREEN


def _elapsed(scene):
    return float(scene.animation_manager.context.timespan.current_time)


def _state(mob, scene, times):
    scene.timeline_manager.set_state_to_times(torch.tensor(times))
    return (
        mob.location.clone(),
        mob.basis.clone(),
        mob.color.clone(),
        mob.scale_coefficient.clone(),
    )


@pytest.fixture
def scene():
    with Scene() as active:
        yield active


# --------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "animation",
    [
        lambda mob: Indicate(mob, duration=0.75),
        lambda mob: Wiggle(mob, n_wiggles=2, duration=0.75),
        lambda mob: Circumscribe(mob, duration=0.75),
        lambda mob: Flash(mob, num_lines=4, duration=0.75),
        lambda mob: FocusOn(mob, duration=0.75),
        lambda mob: ShowPassingFlash(mob, duration=0.75),
        lambda mob: ShowPassingFlashWithThinningStrokeWidth(
            mob, n_segments=3, duration=0.75
        ),
        lambda mob: ApplyWave(mob, duration=0.75),
    ],
    ids=[
        "Indicate",
        "Wiggle",
        "Circumscribe",
        "Flash",
        "FocusOn",
        "ShowPassingFlash",
        "ShowPassingFlashWithThinningStrokeWidth",
        "ApplyWave",
    ],
)
def test_indication_duration_wins_over_the_enclosing_context(scene, animation):
    with Off():
        square = Square(stroke_width=4).spawn()
    start = _elapsed(scene)
    with Seq(duration=None):
        animation(square)
    assert _elapsed(scene) - start == pytest.approx(0.75, abs=1e-6)


def test_blink_duration_comes_from_its_on_and_off_times(scene):
    with Off():
        square = Square().spawn()
    start = _elapsed(scene)
    Blink(square, time_on=0.2, time_off=0.1, blinks=2)
    # Two on/off cycles, plus a final "on" so the Mob is left visible.
    assert _elapsed(scene) - start == pytest.approx(0.2 * 3 + 0.1 * 2, abs=1e-6)


# --------------------------------------------------------------------------
# Helper geometry has to reach the renderer
# --------------------------------------------------------------------------
def _actors_of_type(scene, kind):
    return [actor for actor in scene.actors if type(actor) is kind]


def test_flash_registers_every_spoke_it_draws(scene):
    with Off():
        square = Square().spawn()
    before = len(_actors_of_type(scene, Line))
    Flash(square, num_lines=7, duration=0.5)
    assert len(_actors_of_type(scene, Line)) - before == 7


def test_focus_on_registers_its_spotlight_and_leaves_it_despawned(scene):
    with Off():
        square = Square().spawn()
    spotlight = FocusOn(square, duration=0.5)
    assert spotlight in scene.actors
    assert spotlight.is_despawned()


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"fade_in": True},
        {"fade_out": True},
        {"fade_in": True, "fade_out": True},
    ],
    ids=["passing-flash", "fade-in", "fade-out", "fade-both"],
)
def test_circumscribe_registers_its_frame_for_every_fade_combination(scene, kwargs):
    with Off():
        square = Square().spawn()
    before = len(scene.actors)
    Circumscribe(square, duration=0.5, **kwargs)
    assert len(scene.actors) > before, "the frame never became renderable"


def test_circumscribe_rejects_a_shape_it_cannot_build(scene):
    with Off():
        square = Square().spawn()
    with pytest.raises(ValueError, match="Rectangle or Circle"):
        Circumscribe(square, shape=Group, duration=0.5)


def test_flash_accepts_a_bare_point_as_well_as_a_mob(scene):
    with Off():
        Square().spawn()
    before = len(_actors_of_type(scene, Line))
    Flash(torch.tensor([0.5, 0.5, 0.0]), num_lines=5, duration=0.5)
    assert len(_actors_of_type(scene, Line)) - before == 5


def test_show_passing_flash_uses_stroke_clones_and_restores_source(scene):
    with Off():
        circle = Circle(color=BLUE, stroke_width=6).spawn()
    start = _elapsed(scene)
    original_points = circle.control_points.location.clone()
    original_opacity = circle.opacity.clone()
    actors_before = set(scene.actors)

    ShowPassingFlash(circle, time_width=0.2, duration=0.5)
    ShowPassingFlash(circle, time_width=0.2, duration=0.5)
    end = _elapsed(scene)

    flashes = [
        actor
        for actor in scene.actors
        if actor not in actors_before and type(actor) is Circle
    ]
    assert len(flashes) == 2
    assert all(not flash.filled for flash in flashes)
    assert all(flash.is_despawned() for flash in flashes)
    for flash in flashes:
        assert torch.allclose(flash.stroke_color, circle.color)

    assert circle.is_spawned()
    assert not circle.is_despawned()
    assert torch.allclose(circle.control_points.location, original_points)
    assert torch.allclose(circle.opacity, original_opacity)

    scene.timeline_manager.set_state_to_times(
        torch.tensor([start + 0.25, start + 0.75, end])
    )
    assert torch.allclose(
        circle.control_points.location,
        original_points.expand(3, -1, -1),
    )
    assert torch.allclose(circle.opacity[:2], torch.zeros_like(circle.opacity[:2]))
    assert torch.allclose(circle.opacity[2:], original_opacity)
    assert torch.allclose(
        flashes[0].control_points.location[0],
        flashes[1].control_points.location[1],
        atol=1e-6,
    )


# --------------------------------------------------------------------------
# The Mob is left as it was found
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "animation",
    [
        lambda mob: Indicate(mob, duration=0.5),
        lambda mob: Wiggle(mob, n_wiggles=2, duration=0.5),
        lambda mob: ApplyWave(mob, duration=0.5),
    ],
    ids=["Indicate", "Wiggle", "ApplyWave"],
)
def test_indication_restores_the_mob_it_decorated(scene, animation):
    with Off():
        square = Square(color=BLUE).spawn()
    start = _elapsed(scene)
    animation(square)
    end = _elapsed(scene)

    before = _state(square, scene, [start])
    after = _state(square, scene, [end])
    for original, restored in zip(before, after):
        assert torch.allclose(original, restored, atol=1e-4)


def test_indicate_flashes_the_colour_in_the_middle(scene):
    """A restored end state is only meaningful if something happened first."""
    with Off():
        square = Square(color=BLUE).spawn()
    start = _elapsed(scene)
    Indicate(square, scale_factor=1.5, duration=1.0)

    scene.timeline_manager.set_state_to_times(torch.tensor([start, start + 0.5]))
    assert not torch.allclose(square.color[0], square.color[1], atol=1e-3)


def test_indicate_grows_the_mob_in_the_middle(scene):
    with Off():
        square = Square(color=BLUE).spawn()
    start = _elapsed(scene)
    Indicate(square, scale_factor=1.5, duration=1.0)

    scene.timeline_manager.set_state_to_times(torch.tensor([start, start + 0.5]))
    assert float(square.basis[1].abs().max()) > float(square.basis[0].abs().max())


def test_indicate_scales_a_composite_around_its_anchor(scene):
    with Off():
        group = Group(
            Square().move(torch.tensor([-1.0, 0.0, 0.0])),
            Square().move(torch.tensor([1.0, 0.0, 0.0])),
        ).spawn()
    start = _elapsed(scene)
    Indicate(group, scale_factor=1.5, duration=1.0)

    scene.timeline_manager.set_state_to_times(torch.tensor([start, start + 0.5]))
    left = group[0].location[:, 0, 0]
    right = group[1].location[:, 0, 0]
    assert right[1] - left[1] == pytest.approx(
        1.5 * float(right[0] - left[0]), abs=1e-4
    )


def _alpha(mob):
    return float(mob.color[..., -1].max())


def test_blink_leaves_the_mob_visible_unless_asked_to_hide_it(scene):
    with Off():
        visible = Square().spawn()
        hidden = Square().move(torch.tensor([3.0, 0.0, 0.0])).spawn()
    with Seq():
        Blink(visible, time_on=0.1, time_off=0.1, blinks=1)
    end = _elapsed(scene)
    with Seq():
        Blink(hidden, time_on=0.1, time_off=0.1, blinks=1, hide_at_end=True)

    # Blink fades through the colour's alpha channel, not through ``opacity``,
    # so that parts carrying their own colours keep them.
    scene.timeline_manager.set_state_to_times(torch.tensor([end - 1e-3]))
    assert _alpha(visible) > 0.5
    scene.timeline_manager.set_state_to_times(torch.tensor([_elapsed(scene) - 1e-3]))
    assert _alpha(hidden) == pytest.approx(0.0, abs=1e-4)


# --------------------------------------------------------------------------
# Composites
# --------------------------------------------------------------------------
def test_indication_animations_accept_a_group(scene):
    with Off():
        group = Group(
            Square(color=BLUE),
            Circle(color=PURE_GREEN),
        ).spawn()
    start = _elapsed(scene)
    Indicate(group, duration=0.5)
    ApplyWave(group, duration=0.5)
    assert _elapsed(scene) - start == pytest.approx(1.0, abs=1e-6)
