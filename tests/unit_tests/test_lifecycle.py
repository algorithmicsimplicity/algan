import warnings

import pytest
import torch

from algan import RIGHT, Circle, Group, Lag, Off, Scene, Seq, Square, Sync
from algan.errors import NeverVisibleMobWarning

# In the fast suite: the spawn/despawn lifespan decides whether a Mob exists at
# a given frame at all, and containers inherit it from their children.
pytestmark = pytest.mark.fast


def test_unspawned_group_despawn_preserves_spawned_child_history():
    with Scene() as scene:
        child = Square()
        group = Group(child)

        with Seq():
            child.spawn(animate=False)
            child.move(RIGHT)
            group.despawn()

        scene.timeline_manager.set_state_to_times(torch.tensor([0.5, 1.5, 2.0]))

        assert torch.allclose(child.opacity[0], torch.ones_like(child.opacity[0]))
        assert 0 < float(child.opacity[1].mean()) < 1
        assert torch.count_nonzero(child.opacity[2]) == 0
        assert not group.is_spawned()
        assert not group.is_despawned()


def test_unspawned_group_with_spawned_children_animates():
    """A container is on screen through its children, so it animates.

    ``for mob in group: mob.spawn()`` leaves the group itself unspawned.
    Gating animation on the container's own spawn state applied its edits
    instantly *and* recorded nothing on the timeline, so the animation also
    contributed no time and the rendered video ended before it.
    """
    with Scene() as scene:
        child = Square()
        group = Group(child)

        with Seq():
            child.spawn(animate=False)
            start = scene.animation_manager.context.timespan.current_time
            with Sync(runtime=1.0):
                group.move(RIGHT)

        end = scene.animation_manager.context.timespan.original_end
        assert end == pytest.approx(start + 1.0)
        assert not group.is_spawned()

        scene.timeline_manager.set_state_to_times(
            torch.tensor([start, start + 0.5, end])
        )
        x = [float(child.location[i][..., 0].mean()) for i in range(3)]
        assert x[0] == pytest.approx(0.0, abs=1e-4)
        assert 0 < x[1] < 1
        assert x[2] == pytest.approx(1.0, abs=1e-3)


def test_fully_unspawned_mob_edits_stay_instant():
    with Scene() as scene:
        square = Square()

        with Seq(), Sync(runtime=1.0):
            square.move(RIGHT)

        # Nothing is on screen, so the edit is applied instantly and takes up
        # no time in the video.
        assert scene.animation_manager.context.timespan.original_end == 0
        assert float(square.location[..., 0].mean()) == pytest.approx(1.0)


def _spawn_move_despawn(mob):
    mob.spawn()
    mob.move(RIGHT)
    mob.despawn()


def _in_one_sync(scene):
    bead = Circle()
    with Sync(runtime=1.0):
        _spawn_move_despawn(bead)
    return bead


def _despawned_without_animation(scene):
    bead = Circle()
    with Sync():
        bead.spawn()
        bead.despawn(animate=False)
    return bead


def _in_a_rescaled_block(scene):
    bead = Circle()
    with Seq(runtime=3):
        scene.wait(1)
        with Sync():
            with Lag(0.5):
                bead.spawn()
            bead.despawn()
    return bead


def _container_of_separate_spawns(scene):
    beads = [Circle(), Square()]
    with Sync():
        for bead in beads:
            bead.spawn()
        Group(beads).despawn()
    return beads


def _seq_inside_the_sync(scene):
    bead = Circle()
    with Sync(runtime=1.0), Seq():
        _spawn_move_despawn(bead)


def _sequential(scene):
    _spawn_move_despawn(Circle())


def _instant_spawn(scene):
    bead = Circle()
    with Sync():
        bead.spawn(animate=False)
        bead.despawn()


def _spawned_in_off(scene):
    bead = Circle()
    with Sync():
        with Off():
            bead.spawn()
        bead.despawn()


def _all_in_off(scene):
    with Off():
        bead = Circle().spawn()
        bead.despawn()


def _despawned_in_a_later_block(scene):
    bead = Circle()
    with Sync():
        bead.spawn()
    with Sync():
        bead.despawn()


def _overlapping_lag(scene):
    with Lag(0.5):
        _spawn_move_despawn(Circle())


def _spawned_again_before_despawn(scene):
    bead = Circle().spawn()
    with Sync():
        bead.spawn()  # already spawned: nothing new to show
        bead.despawn()


def _scene_fade_out(scene):
    # What save_video(animate_fade_out=True) records at the end.
    with Sync():
        Circle().spawn()
        Square().spawn()
    scene.despawn_mobs(retain_history=True, runtime=0.5)


@pytest.mark.parametrize(
    ("author", "expected"),
    [
        (
            _in_one_sync,
            "Circle will never be visible: despawn() starts the exit at t=0.00s",
        ),
        (_despawned_without_animation, "Circle will never be visible"),
        (
            _in_a_rescaled_block,
            "Circle will never be visible: despawn() starts the exit at t=1.50s",
        ),
        (
            _container_of_separate_spawns,
            "2 Mobs (Circle, Square) will never be visible",
        ),
    ],
)
def test_despawn_with_its_animated_spawn_warns_once(author, expected):
    """A Mob despawned where its fade-in starts is never drawn: say so.

    The check reads lifespans only after the enclosing contexts have exited
    and rescaled them, and must point at the ``despawn()`` line.
    """
    with Scene() as scene, pytest.warns(NeverVisibleMobWarning) as caught:
        author(scene)
    warned = [w for w in caught if issubclass(w.category, NeverVisibleMobWarning)]
    assert len(warned) == 1
    assert str(warned[0].message).startswith(expected)
    assert "with Seq():" in str(warned[0].message)
    assert warned[0].filename == __file__
    with open(__file__, encoding="utf-8") as source:
        assert "despawn(" in source.read().splitlines()[warned[0].lineno - 1]


@pytest.mark.parametrize(
    "author",
    [
        _seq_inside_the_sync,
        _sequential,
        _instant_spawn,
        _spawned_in_off,
        _all_in_off,
        _despawned_in_a_later_block,
        _overlapping_lag,
        _spawned_again_before_despawn,
        _scene_fade_out,
    ],
)
def test_despawn_after_the_mob_was_shown_does_not_warn(author):
    with Scene() as scene, warnings.catch_warnings():
        warnings.simplefilter("error", NeverVisibleMobWarning)
        author(scene)


def test_never_visible_warning_is_authoring_time_only():
    with Scene() as scene:
        with pytest.warns(NeverVisibleMobWarning):
            bead = _in_one_sync(scene)
        with warnings.catch_warnings():
            warnings.simplefilter("error", NeverVisibleMobWarning)
            scene.timeline_manager.set_state_to_times(torch.tensor([0.25, 0.5]))
            assert torch.count_nonzero(bead.opacity > 1e-4) == 0
            scene.timeline_manager.clear_buffers()
