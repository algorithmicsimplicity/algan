"""Contracts of ``Mob.become``.

``become`` is the only public operation that changes a Mob's *structure* while
it is on screen: it re-batches the Mob onto fresh timeline rows, pads whichever
side has fewer parts, and then animates every attribute across.  All three of
those steps fail quietly rather than loudly -- a mispaired part flies across the
screen, a dropped target leaves geometry behind, and a morph recorded against
the wrong rows corrupts an unrelated Mob.
"""

from __future__ import annotations

import pytest
import torch

from algan import (
    BLUE,
    LEFT,
    RIGHT,
    YELLOW,
    Circle,
    Group,
    Off,
    RegularPolygon,
    Scene,
    Square,
    Sync,
    Text,
)


@pytest.fixture
def scene():
    with Scene() as active:
        yield active


def _at(scene, mob, times):
    scene.timeline_manager.set_state_to_times(torch.tensor(times))
    return mob


def _geometry(mob):
    """Every point the Mob and its descendants own, at the current state."""
    points = [mob.location.reshape(-1, 3)]
    for child in mob.get_descendants():
        points.append(child.location.reshape(-1, 3))
    return torch.cat(points, 0)


def test_become_hands_back_a_spawned_registered_mob_to_keep_animating(scene):
    with Off():
        square = Square(color=BLUE).spawn()
    with Sync(run_time=1.0):
        morphed = square.become(Circle(radius=0.6, color=YELLOW))

    # ``detach_history=True`` may re-batch onto fresh rows and hand back a
    # different object, so the caller must use the *returned* Mob afterwards.
    # Whatever comes back has to be on screen and reachable by the renderer.
    assert morphed is not None
    assert morphed in scene.actors
    assert morphed.is_spawned()


def test_become_reaches_the_targets_appearance_and_travels_to_get_there(scene):
    with Off():
        square = Square(color=BLUE).spawn()
    start = float(scene.animation_manager.context.timespan.current_time)
    with Sync(run_time=1.0):
        morphed = square.become(Circle(radius=0.6, color=YELLOW))
    end = float(scene.animation_manager.context.timespan.current_time)

    _at(scene, morphed, [start, (start + end) / 2, end])
    colors = morphed.color.reshape(3, -1, 5)
    # Ends at the target colour, and is genuinely in between halfway through.
    assert torch.allclose(colors[2, :, :3], colors[2, :1, :3].expand_as(colors[2, :, :3]))
    assert not torch.allclose(colors[0], colors[2], atol=1e-3)
    assert not torch.allclose(colors[1], colors[0], atol=1e-3)
    assert not torch.allclose(colors[1], colors[2], atol=1e-3)


def test_become_morphs_position_as_well_as_shape(scene):
    """Documented Transform semantics.

    This is why scenes build their morph targets where the Mob already is.
    """
    with Off():
        square = Square().move(LEFT * 2).spawn()
    start = float(scene.animation_manager.context.timespan.current_time)
    with Sync(run_time=1.0):
        morphed = square.become(Circle(radius=0.6).move(RIGHT * 2))
    end = float(scene.animation_manager.context.timespan.current_time)

    _at(scene, morphed, [start, end])
    centers = _center_per_time(morphed, 2)
    assert centers[0][0] < -1.0
    assert centers[1][0] > 1.0


def _center_per_time(mob, count):
    points = [
        torch.cat(
            [mob.location[index].reshape(-1, 3)]
            + [child.location[index].reshape(-1, 3) for child in mob.get_descendants()],
            0,
        )
        for index in range(count)
    ]
    return [(p.amin(0) + p.amax(0)) / 2 for p in points]


@pytest.mark.parametrize("minimize_movement", [False, True])
def test_become_pads_whichever_side_has_fewer_parts(scene, minimize_movement):
    """A three-glyph word has to morph into a five-glyph one."""
    with Off():
        short = Text("ab", font_size=40).spawn()
        target = Text("abcde", font_size=40)
        # Measured before the timeline is materialized: authoring and
        # materialized state cannot be interleaved.
        target_width = float(target.get_width().reshape(-1)[0])
    with Sync(run_time=1.0):
        morphed = short.become(target, minimize_movement=minimize_movement)
    end = float(scene.animation_manager.context.timespan.current_time)

    _at(scene, morphed, [end])
    points = _geometry(morphed)
    width = float(points[:, 0].amax() - points[:, 0].amin())
    assert width == pytest.approx(target_width, rel=0.15)


def test_minimize_movement_keeps_parts_closer_to_where_they_started(scene):
    """The whole point of the flag: pairing by proximity, not by index."""

    def total_travel(minimize):
        with Scene() as isolated:
            with Off():
                source = Group(
                    *[Square(side_length=0.4).move(RIGHT * x) for x in (-2, 0, 2)]
                ).spawn()
            start = float(isolated.animation_manager.context.timespan.current_time)
            with Sync(run_time=1.0):
                # Same three squares, listed in reverse order.
                morphed = source.become(
                    Group(
                        *[Square(side_length=0.4).move(RIGHT * x) for x in (2, 0, -2)]
                    ),
                    minimize_movement=minimize,
                )
            end = float(isolated.animation_manager.context.timespan.current_time)
            isolated.timeline_manager.set_state_to_times(torch.tensor([start, end]))
            first = torch.cat(
                [c.location[0].reshape(-1, 3) for c in morphed.get_descendants()], 0
            )
            last = torch.cat(
                [c.location[1].reshape(-1, 3) for c in morphed.get_descendants()], 0
            )
            return float((last - first).norm(dim=-1).sum())

    assert total_travel(minimize=True) <= total_travel(minimize=False)


def test_become_rejects_a_target_of_a_different_primitive_type(scene):
    from algan import Sphere

    with Off():
        square = Square().spawn()
    with pytest.raises(NotImplementedError):
        square.become(Sphere(radius=0.5))


def test_become_does_not_spawn_or_mutate_the_target(scene):
    with Off():
        square = Square().spawn()
        target = Circle(radius=0.6)
    target_center_before = target.get_center().clone()

    with Sync(run_time=1.0):
        square.become(target)

    assert not target.is_spawned()
    assert torch.allclose(target.get_center(), target_center_before, atol=1e-5)


def test_chained_becomes_keep_animating(scene):
    """Each morph must hand back a Mob the next one can morph again."""
    with Off():
        mob = Square(color=BLUE).spawn()
    start = float(scene.animation_manager.context.timespan.current_time)
    with Sync(run_time=1.0):
        mob = mob.become(Circle(radius=0.6))
    with Sync(run_time=1.0):
        mob = mob.become(RegularPolygon(6, radius=0.6, color=YELLOW))
    end = float(scene.animation_manager.context.timespan.current_time)

    assert end - start == pytest.approx(2.0, abs=1e-6)
    _at(scene, mob, [start, (start + end) / 2, end])
    assert mob.color.shape[0] == 3
