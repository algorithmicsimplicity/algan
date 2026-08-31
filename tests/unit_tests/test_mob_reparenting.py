"""The hierarchy is read when an animation is *recorded*, not when it plays.

Algan stores no local transforms: ``location`` and ``basis`` are world-space
rows in a shared buffer, and a parent transform is a delta written into its
descendants' rows at record time.  ``_apply_change`` resolves those rows through
``get_descendants()`` and ``modify_attribute_and_record`` stores the resolved
set on the event, so re-parenting between two recorded animations leaves the
first one alone and only redirects the second.

Everything here pins that contract from the outside, because nothing else does:
the suite covered cycle rejection and static hierarchies, and every one of these
would have passed just as happily against a build that resolved the hierarchy at
playback time and silently rewrote already-recorded frames.
"""

from __future__ import annotations

import math
import warnings

import pytest
import torch

from algan import Group, Mob, Off, Scene, SceneManager
from algan.animation_timeline.animation_contexts import Seq, Sync
from algan.errors import HierarchyChangedDuringUpdaterWarning, HierarchyError

# In the fast suite: these are pure timeline/Mob tests with no rendering, and
# what they pin is the record-time row set -- which any change to hierarchy
# bookkeeping, descendant caching or edit replay can move without touching this
# file.
pytestmark = pytest.mark.fast

RIGHT = torch.tensor([1.0, 0.0, 0.0])
UP = torch.tensor([0.0, 1.0, 0.0])
OUTWARD = torch.tensor([0.0, 0.0, 1.0])


def _empty_scene(scene):
    scene.camera = None
    scene.light_sources = []


@pytest.fixture(autouse=True)
def fresh_scene_stack():
    SceneManager.reset()
    yield
    SceneManager.reset()


@pytest.fixture
def scene():
    scene = Scene(scene_initializer=_empty_scene)
    yield scene
    scene.terminate()


def _mob(location, name="_"):
    return Mob(location=location, name=name, add_to_scene=False).spawn(animate=False)


def _locations_at(scene, times, mob):
    """``mob``'s location at each of ``times``, shape ``(len(times), 3)``."""
    scene.timeline_manager.set_state_to_times(torch.tensor([float(t) for t in times]))
    return mob.location[:, 0].clone()


def _hierarchy_warnings(caught):
    return [
        w
        for w in caught
        if issubclass(w.category, HierarchyChangedDuringUpdaterWarning)
    ]


def _assert_at(scene, times, mob, expected):
    torch.testing.assert_close(
        _locations_at(scene, times, mob),
        torch.tensor(expected),
        atol=2e-5,
        rtol=2e-5,
    )


# ---------------------------------------------------------------------------
# The headline case: move parent A, re-parent to B, move B.
# ---------------------------------------------------------------------------


def test_reparenting_redirects_later_animations_and_rewrites_no_earlier_one(scene):
    parent_a, parent_b, child = _mob([0, 0, 0]), _mob([0, 0, 5]), _mob([0, 2, 0])

    parent_a.add_children(child)
    parent_a.move(RIGHT * 3)  # t 0 -> 1, recorded while the child is under A

    parent_a.remove_child(child)
    parent_b.add_children(child)
    parent_b.move(UP * 4)  # t 1 -> 2, recorded while the child is under B

    parent_a.move(RIGHT * 10)  # t 2 -> 3, the child is gone from A

    times = [0.5, 1.0, 1.5, 2.0, 3.0]
    _assert_at(
        scene,
        times,
        child,
        [
            [1.5, 2.0, 0.0],  # mid-way through A's move: still carried by it
            [3.0, 2.0, 0.0],
            [3.0, 4.0, 0.0],  # mid-way through B's move
            [3.0, 6.0, 0.0],
            [3.0, 6.0, 0.0],  # A moves on alone
        ],
    )
    _assert_at(
        scene,
        times,
        parent_a,
        [
            [1.5, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [13.0, 0.0, 0.0],
        ],
    )


def test_reparenting_does_not_move_the_child(scene):
    """No local-to-parent transform exists, so the child keeps its world
    position and acquires none of the new parent's frame -- its ``z`` stays put
    even though the new parent sits five units away along that axis.
    """
    parent_a, parent_b, child = _mob([0, 0, 0]), _mob([0, 0, 5]), _mob([0, 2, 0])
    parent_a.add_children(child)

    parent_a.remove_child(child)
    parent_b.add_children(child)

    _assert_at(scene, [0.0], child, [[0.0, 2.0, 0.0]])


def test_a_detached_child_stops_following_its_former_parent(scene):
    parent, child = _mob([0, 0, 0]), _mob([0, 2, 0])
    parent.add_children(child)
    parent.remove_child(child)
    parent.move(RIGHT * 3)

    _assert_at(scene, [1.0], child, [[0.0, 2.0, 0.0]])
    _assert_at(scene, [1.0], parent, [[3.0, 0.0, 0.0]])


def test_rotation_pivots_about_whichever_parent_is_current(scene):
    parent_a, parent_b, child = _mob([0, 0, 0]), _mob([4, 0, 0]), _mob([0, 2, 0])

    parent_a.add_children(child)
    parent_a.rotate(90, OUTWARD)  # child swings about the origin

    parent_a.remove_child(child)
    parent_b.add_children(child)
    parent_b.rotate(90, OUTWARD)  # and now about (4, 0, 0)

    # A positive angle about OUTWARD turns counter-clockwise on screen, so
    # (0, 2) about (0, 0) lands on (-2, 0); (-2, 0) is then 6 to the left of
    # (4, 0), and a quarter turn about that puts it 6 below, at (4, -6).
    _assert_at(scene, [1.0, 2.0], child, [[-2.0, 0.0, 0.0], [4.0, -6.0, 0.0]])


def test_re_parenting_inside_one_sync_applies_both_parents(scene):
    """``Sync`` plays everything in it at once, and the hierarchy edit between
    the two moves takes effect immediately at record time -- so the child is
    genuinely under both animations over the same second.
    """
    parent_a, parent_b, child = _mob([0, 0, 0]), _mob([0, 0, 5]), _mob([0, 2, 0])
    parent_a.add_children(child)

    with Sync():
        parent_a.move(RIGHT * 3)
        parent_a.remove_child(child)
        parent_b.add_children(child)
        parent_b.move(UP * 4)

    _assert_at(scene, [0.5, 1.0], child, [[1.5, 4.0, 0.0], [3.0, 6.0, 0.0]])


# ---------------------------------------------------------------------------
# Several parents at once.
# ---------------------------------------------------------------------------


def test_a_mob_under_two_parents_accumulates_both_deltas(scene):
    """The hierarchy is a graph, not a tree.  This is what lets two overlapping
    Groups each arrange the same member.
    """
    first, second, shared = _mob([0, 0, 0]), _mob([0, 0, 0]), _mob([0, 2, 0])
    group_one = Group(first, shared)
    group_two = Group(shared, second)
    assert len(shared.parents) == 2

    with Off():
        group_one.move(RIGHT * 3)
        group_two.move(UP * 5)

    _assert_at(scene, [0.0], shared, [[3.0, 7.0, 0.0]])
    _assert_at(scene, [0.0], first, [[3.0, 0.0, 0.0]])
    _assert_at(scene, [0.0], second, [[0.0, 5.0, 0.0]])


def test_detaching_one_parent_leaves_the_other_driving(scene):
    parent_a, parent_b, child = _mob([0, 0, 0]), _mob([0, 0, 0]), _mob([0, 2, 0])
    child.add_parent(parent_a)
    child.add_parent(parent_b)

    child.remove_parent(parent_a)
    assert [id(p) for p in child.parents] == [id(parent_b)]

    with Off():
        parent_a.move(RIGHT * 3)
        parent_b.move(UP * 5)

    _assert_at(scene, [0.0], child, [[0.0, 7.0, 0.0]])


# ---------------------------------------------------------------------------
# Both halves of a link, from either side.
# ---------------------------------------------------------------------------


def test_add_parent_and_add_children_build_the_same_link(scene):
    parent, from_below, from_above = _mob([0, 0, 0]), _mob([0, 2, 0]), _mob([0, 3, 0])

    from_below.add_parent(parent)
    parent.add_children(from_above)

    assert [id(c) for c in parent.children] == [id(from_below), id(from_above)]
    assert [id(p) for p in from_below.parents] == [id(parent)]
    assert [id(p) for p in from_above.parents] == [id(parent)]

    with Off():
        parent.move(RIGHT * 3)
    _assert_at(scene, [0.0], from_below, [[3.0, 2.0, 0.0]])
    _assert_at(scene, [0.0], from_above, [[3.0, 3.0, 0.0]])


def test_add_parent_is_idempotent(scene):
    parent, child = _mob([0, 0, 0]), _mob([0, 2, 0])
    child.add_parent(parent)
    child.add_parent(parent)

    assert len(parent.children) == 1
    assert len(child.parents) == 1

    with Off():
        parent.move(RIGHT * 3)
    _assert_at(scene, [0.0], child, [[3.0, 2.0, 0.0]])


@pytest.mark.parametrize("detach", ["remove_parent", "remove_child"])
def test_detaching_from_either_side_drops_both_halves(scene, detach):
    parent, child = _mob([0, 0, 0]), _mob([0, 2, 0])
    child.add_parent(parent)

    if detach == "remove_parent":
        child.remove_parent(parent)
    else:
        parent.remove_child(child)

    assert parent.children == []
    assert child.parents == []


def test_remove_parent_ignores_a_mob_that_is_not_a_parent(scene):
    parent, child, stranger = _mob([0, 0, 0]), _mob([0, 2, 0]), _mob([9, 0, 0])
    child.add_parent(parent)

    assert child.remove_parent(stranger) is child
    assert [id(p) for p in child.parents] == [id(parent)]

    with pytest.raises(ValueError, match="not a child"):
        parent.remove_child(stranger)


def test_add_parent_rejects_a_non_mob(scene):
    """``add_children`` says what a bad child is; this side used to append
    anything at all, and now says what a bad parent is.
    """
    with pytest.raises(TypeError, match="must be an Animatable"):
        _mob([0, 0, 0]).add_parent("not a mob")


def test_add_parent_rejects_a_mob_from_its_own_subtree(scene):
    """The cycle walk goes down ``children`` now that both halves are linked,
    which is the same graph ``get_descendants`` traverses.
    """
    root, middle, leaf = _mob([0, 0, 0]), _mob([0, 1, 0]), _mob([0, 2, 0])
    middle.add_parent(root)
    leaf.add_parent(middle)

    with pytest.raises(HierarchyError, match="its own parent"):
        root.add_parent(root)
    with pytest.raises(HierarchyError, match="create a cycle"):
        root.add_parent(middle)
    with pytest.raises(HierarchyError, match="create a cycle"):
        root.add_parent(leaf)

    # A second root above the same chain is not a cycle.
    assert root.add_parent(_mob([0, 0, 9])) is root


# ---------------------------------------------------------------------------
# Lifespans and layout follow the live graph, unlike recorded animations.
# ---------------------------------------------------------------------------


def test_despawn_recursion_follows_the_new_parent(scene):
    parent_a, parent_b, child = _mob([0, 0, 0]), _mob([0, 0, 5]), _mob([0, 2, 0])
    parent_a.add_children(child)
    parent_a.remove_child(child)
    parent_b.add_children(child)

    parent_a.despawn(animate=False)
    assert not child.is_despawned()

    parent_b.despawn(animate=False)
    assert child.is_despawned()


def test_bounding_box_queries_follow_the_live_graph(scene):
    parent, child = _mob([0, 0, 0]), _mob([0, 3, 0])

    def top():
        return parent.get_boundary_point(UP).reshape(-1)[1].item()

    assert top() == pytest.approx(0.0)
    parent.add_children(child)
    assert top() == pytest.approx(3.0)
    parent.remove_child(child)
    assert top() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Group slices are non-owning views, but they still have to invalidate caches.
# ---------------------------------------------------------------------------


def test_a_group_slice_add_reaches_transforms(scene):
    """``get_descendants`` caches against the hierarchy version, so a slice that
    mutated ``children`` without bumping it went on serving the descendant set
    it had before the member arrived: ``len(view)`` said three, the move
    touched two.
    """
    first, second, third = _mob([0, 0, 0]), _mob([1, 0, 0]), _mob([2, 0, 0])
    group = Group(first, second, third)
    view = group[0:2]

    def descendant_ids():
        return [id(m) for m in view.get_descendants(include_self=False)]

    assert descendant_ids() == [id(first), id(second)]

    view.add(third)

    assert len(view) == 3
    assert descendant_ids() == [id(first), id(second), id(third)]
    with Off():
        view.move(RIGHT * 10)
    _assert_at(scene, [0.0], third, [[12.0, 0.0, 0.0]])


def test_a_group_slice_re_add_is_a_no_op(scene):
    """As documented, and as the owning path has always behaved -- the view used
    to raise ``HierarchyError: A child cannot occur more than once`` instead.
    """
    first, second = _mob([0, 0, 0]), _mob([1, 0, 0])
    view = Group(first, second)[0:2]

    assert view.add(first) is view
    assert len(view) == 2


def test_a_group_slice_takes_no_parent_link(scene):
    first, second = _mob([0, 0, 0]), _mob([1, 0, 0])
    group = Group(first, second)
    view = group[0:2]

    assert [id(p) for p in first.parents] == [id(group)]
    view.add(second)
    assert [id(p) for p in second.parents] == [id(group)]


# ---------------------------------------------------------------------------
# Updaters, unlike recorded animations, resolve against the live hierarchy.
# ---------------------------------------------------------------------------


def test_an_updater_resolves_its_subtree_at_materialization_not_record_time(scene):
    """Documented, not endorsed.

    The recorded-function replay path hands back the row set the event stored
    (``TimelineManager.replay_inds``); the updater path never sets
    ``_active_replay_event``, so every write inside an updater re-resolves its
    rows from the hierarchy as it stands when the frames are materialized.  A
    hierarchy edit made while an updater is live therefore reaches backwards
    over frames the updater already covered: here the child is detached at
    t = 1, and comes out unmoved at t = 0.5 as well.

    That is what the warning asserted below is for; this test is the proof the
    warning is not crying wolf.
    """
    parent_a, parent_b, child = _mob([0, 0, 0]), _mob([0, 0, 5]), _mob([0, 2, 0])
    parent_a.add_children(child)

    with Seq():
        updater_id = parent_a.add_updater(lambda mob, t: mob.move_to(RIGHT * t))
        Scene.wait(1)
        with pytest.warns(HierarchyChangedDuringUpdaterWarning):
            parent_a.remove_child(child)
        parent_b.add_children(child)
        Scene.wait(1)
        parent_a.remove_updater(updater_id)

    with Off(
        record_attr_modifications=False, record_funcs=False, priority_level=math.inf
    ):
        moved = _locations_at(scene, [0.5, 1.5], parent_a)
        stayed = _locations_at(scene, [0.5, 1.5], child)

    torch.testing.assert_close(
        moved, torch.tensor([[0.5, 0.0, 0.0], [1.5, 0.0, 0.0]]), atol=2e-5, rtol=2e-5
    )
    # Both frames, including the one before the detach.
    torch.testing.assert_close(
        stayed, torch.tensor([[0.0, 2.0, 0.0], [0.0, 2.0, 0.0]]), atol=2e-5, rtol=2e-5
    )


def test_the_warning_names_the_updater_the_mob_and_the_line(scene):
    parent, child = _mob([0, 0, 0]), _mob([0, 2, 0], name="dial")
    parent.add_children(child)

    def spin(mob, t):
        mob.move_to(RIGHT * t)

    with Seq():
        parent.add_updater(spin)
        Scene.wait(1)
        with pytest.warns(HierarchyChangedDuringUpdaterWarning) as caught:
            parent.remove_child(child)

    message = str(caught[0].message)
    assert "spin" in message  # which updater
    assert "test_mob_reparenting.py" in message  # and where it was added
    # The warning points at the author's own line, not at algan internals.
    assert caught[0].filename.endswith("test_mob_reparenting.py")


def test_the_warning_stays_quiet_once_the_updater_is_removed(scene):
    parent, child = _mob([0, 0, 0]), _mob([0, 2, 0])
    parent.add_children(child)

    with Seq(), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        updater_id = parent.add_updater(lambda mob, t: mob.move_to(RIGHT * t))
        Scene.wait(1)
        parent.remove_updater(updater_id)
        parent.remove_child(child)

    assert _hierarchy_warnings(caught) == []


def test_the_warning_stays_quiet_for_a_subtree_the_updater_never_addresses(scene):
    """The dependency set is not the test -- ``record_updater`` seeds it with
    the caller's whole subtree, so using it here would fire on any composite
    Mob.  What matters is whether the updater ever asked for *descendants*.
    """
    driven, other, child = _mob([0, 0, 0]), _mob([9, 0, 0]), _mob([0, 2, 0])

    with Seq(), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        driven.add_updater(lambda mob, t: mob.move_to(RIGHT * t))
        Scene.wait(1)
        other.add_children(child)

    assert _hierarchy_warnings(caught) == []


def test_a_non_recursive_updater_does_not_warn(scene):
    """``set_non_recursive`` writes one Mob's own rows, so its row set does not
    move when that Mob gains a child -- and a warning here would be a lie.
    """
    parent, child = _mob([0, 0, 0]), _mob([0, 2, 0])

    with Seq(), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        parent.add_updater(lambda mob, t: mob.set_non_recursive(location=RIGHT * t))
        Scene.wait(1)
        parent.add_children(child)

    assert _hierarchy_warnings(caught) == []


def test_the_warning_is_said_once_per_updater_and_parent(scene):
    """A loop that re-parents every iteration should not print a hundred
    identical paragraphs.
    """
    parent = _mob([0, 0, 0])
    children = [_mob([0, i + 1, 0]) for i in range(4)]

    with Seq(), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        parent.add_updater(lambda mob, t: mob.move_to(RIGHT * t))
        Scene.wait(1)
        for child in children:
            parent.add_children(child)
            parent.remove_child(child)

    assert len(_hierarchy_warnings(caught)) == 1


def test_a_mob_built_inside_an_updater_does_not_warn(scene):
    """An updater that constructs Mobs edits the hierarchy on every frame by
    construction; warning about its own edits would make the diagnostic
    useless for exactly the scenes that use it.
    """
    parent = _mob([0, 0, 0])

    def build(mob, t):
        mob.add_children(Mob(location=[0, 1, 0], add_to_scene=False))

    with Seq(), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        parent.add_updater(build)
        Scene.wait(1)

    assert _hierarchy_warnings(caught) == []
