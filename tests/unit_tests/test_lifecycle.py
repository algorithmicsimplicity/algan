import torch

from algan import Group, RIGHT, Scene, Seq, Square


def test_unspawned_group_despawn_preserves_spawned_child_history():
    with Scene() as scene:
        child = Square()
        group = Group(child)

        with Seq():
            child.spawn(animate=False)
            child.move(RIGHT)
            group.despawn()

        scene.timeline_manager.set_state_to_times(
            torch.tensor([0.5, 1.5, 2.0])
        )

        assert torch.allclose(child.opacity[0], torch.ones_like(child.opacity[0]))
        assert 0 < float(child.opacity[1].mean()) < 1
        assert torch.count_nonzero(child.opacity[2]) == 0
        assert not group.is_spawned()
        assert not group.is_despawned()
