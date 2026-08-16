import pytest
import torch

from algan import RIGHT, Group, Scene, Seq, Square, Sync

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
            with Sync(run_time=1.0):
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

        with Seq(), Sync(run_time=1.0):
            square.move(RIGHT)

        # Nothing is on screen, so the edit is applied instantly and takes up
        # no time in the video.
        assert scene.animation_manager.context.timespan.original_end == 0
        assert float(square.location[..., 0].mean()) == pytest.approx(1.0)
