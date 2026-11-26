"""Growing a Group one ``add`` at a time.

Render-free. ``group.add`` in a loop is the natural way to build a scatter plot
or a Group filled inside a loop, and it used to re-read every member's location
and re-check every member for cycles on every add: 2,000 adds took minutes,
against well under a second for one ``Group(mobs)``. These pin that a Group
grown by adds is the Group built at once, and that an add costs about the same
however many members there already are.
"""

import pytest
import torch

from algan import (
    DOWN,
    IN,
    LEFT,
    OUT,
    RIGHT,
    UP,
    Circle,
    Group,
    Mob,
    Off,
    Scene,
    SceneManager,
    Square,
)
from algan.animatable_base.mob_hierarchy import MobHierarchyMixin
from algan.animation_timeline.timeline import AttributeTimeline


@pytest.fixture(autouse=True)
def fresh_scene_stack():
    SceneManager.reset()
    yield
    SceneManager.reset()


def _members():
    """Mobs of several kinds, spread out: a Circle carries helper grids that
    boundary queries skip, and a nested Group re-centers its pivot when read.
    """
    mobs = []
    for i in range(9):
        offset = RIGHT * (i % 3) * 1.3 + UP * (i // 3) * 0.7 + OUT * (i % 2)
        mobs.append((Square(size=0.5) if i % 2 else Circle(radius=0.3)).spawn())
        with Off():
            mobs[-1].move_to(offset)
    inner = Group(Square(size=0.2).spawn(), Circle(radius=0.2).spawn().move(LEFT * 3))
    mobs.append(inner)
    return mobs


def _assert_same_group(grown, built):
    assert grown.children == built.children
    torch.testing.assert_close(grown.location, built.location, rtol=0, atol=1e-6)
    torch.testing.assert_close(
        grown.get_center(), built.get_center(), rtol=0, atol=1e-6
    )
    for direction in (LEFT, RIGHT, UP, DOWN, IN, OUT):
        torch.testing.assert_close(
            grown.get_boundary_point(direction),
            built.get_boundary_point(direction),
            rtol=0,
            atol=1e-6,
        )


def test_a_group_grown_by_adds_is_the_group_built_at_once():
    with Scene():
        mobs = _members()
        grown = Group().spawn()
        for mob in mobs:
            grown.add(mob)
            # The pivot after every add is the middle of what is in it so far.
            torch.testing.assert_close(
                grown.location,
                Group._midpoint_for(grown.children),
                rtol=0,
                atol=1e-6,
            )
        _assert_same_group(grown, Group(mobs))


def test_adds_still_measure_members_that_moved_between_them():
    # Something written between two adds -- a member moved, an animation, a
    # new Mob built -- has to be read, not taken from what the last add saw.
    with Scene():
        mobs = _members()
        grown = Group().spawn()
        for i, mob in enumerate(mobs):
            if i == 4:
                with Off():
                    mobs[0].move(LEFT * 4 + DOWN)
            if i == 6:
                mobs[1].move(UP * 3)
            if i == 7:
                mobs[2].exclude_from_boundary = True
            grown.add(mob)
            torch.testing.assert_close(
                grown.location,
                Group._midpoint_for(grown.children),
                rtol=0,
                atol=1e-6,
            )
        grown.remove_child(mobs[4])
        grown.add(Square(size=0.1).spawn().move(RIGHT * 6))
        with Off():
            grown.add(mobs[4])
        torch.testing.assert_close(
            grown.location, Group._midpoint_for(grown.children), rtol=0, atol=1e-6
        )


def test_re_adding_and_cycles_are_still_refused_or_ignored():
    from algan.errors import HierarchyError

    with Scene():
        a, b, c = Mob().spawn(), Mob().spawn(), Mob().spawn()
        g = Group(a)
        g.add(a, b)
        g.add(b)
        assert g.children == [a, b]
        outer = Group(g)
        with pytest.raises(HierarchyError, match="cycle"):
            g.add(outer)
        with pytest.raises(HierarchyError, match="more than once"):
            g.add(c, c)
        assert g.children == [a, b]


def _work_per_add(n, monkeypatch):
    """Timeline reads, descendant walks and cycle checks spent on ``n`` adds."""
    counts = {"reads": 0, "walks": 0, "cycle checks": 0}

    def counting(name, function):
        def counted(*args, **kwargs):
            counts[name] += 1
            return function(*args, **kwargs)

        return counted

    with Scene():
        with Off():
            mobs = [Mob(location=RIGHT * (i % 17) + UP * (i // 17)) for i in range(n)]
        for mob in mobs:
            mob.spawn()
        group = Group().spawn()
        group.add(Mob().spawn())
        with monkeypatch.context() as patch:
            patch.setattr(
                AttributeTimeline, "get", counting("reads", AttributeTimeline.get)
            )
            patch.setattr(
                MobHierarchyMixin,
                "get_descendants",
                counting("walks", MobHierarchyMixin.get_descendants),
            )
            patch.setattr(
                MobHierarchyMixin,
                "_contains_in_hierarchy",
                staticmethod(
                    counting("cycle checks", MobHierarchyMixin._contains_in_hierarchy)
                ),
            )
            for mob in mobs:
                group.add(mob)
        torch.testing.assert_close(
            group.location, Group._midpoint_for(group.children), rtol=0, atol=1e-6
        )
    return {name: count / n for name, count in counts.items()}


def test_an_add_costs_the_same_however_large_the_group(monkeypatch):
    # Counted, not timed, so a loaded machine cannot fail it. Each add used to
    # read every member's location, rebuild every member's descendant list and
    # re-check every member for a cycle, so the work per add grew with the
    # Group (about 4x from 40 to 160 members); now it does not.
    small = _work_per_add(40, monkeypatch)
    large = _work_per_add(160, monkeypatch)
    for name in small:
        assert large[name] <= 1.5 * small[name] + 2, (name, small, large)
