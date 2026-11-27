"""What the timeline does when authoring or replay goes wrong.

- An updater that raises while a batch is replayed left the timeline
  half-materialized: the batch's times and buffers stayed in place, and later
  authoring recorded against them. The error also did not say which updater
  raised, and ``if t > 1:`` on a batched ``t`` gave PyTorch's bare "Boolean
  value of Tensor ... is ambiguous".
- A Mob spawned after a ``rewind()`` past time zero got a negative start time,
  which is the "never spawned" sentinel, so it silently never appeared.
"""

import pytest
import torch

from algan import RIGHT, UP, Mob, Scene, Seq
from algan.errors import AlganConfigurationError
from algan.scene_manager import SceneManager

F32 = torch.float32


def _failing_batch_scene():
    scene = SceneManager.reset()
    m = Mob().spawn(animate=False)

    def upd(mob, t):
        if t.numel() > 1:
            raise ValueError("boom from user updater")
        mob.set(location=RIGHT * t)

    m.add_updater(upd)
    Scene.wait(2)
    return scene, m, upd


def test_failed_replay_leaves_authoring_state():
    scene, m, _ = _failing_batch_scene()
    timeline = scene.timeline_manager
    with pytest.raises(ValueError, match="boom"):
        timeline.set_state_to_times(torch.tensor([0.25, 0.75], dtype=F32))
    assert timeline._materialization_times is None
    assert timeline._materialized_mob_ids is None
    assert list(m.location.shape) == [1, 1, 3]
    m.move(RIGHT)
    location = timeline.attr_to_timeline["location"]
    rows = location.mob_id_to_inds[m.id]
    assert float(location.current_state[0, rows].reshape(-1)[0]) == pytest.approx(1.0)


def test_replay_error_names_the_updater():
    scene, _, upd = _failing_batch_scene()
    with pytest.raises(ValueError) as info:
        scene.timeline_manager.set_state_to_times(torch.tensor([0.25, 0.75], dtype=F32))
    text = "\n".join(getattr(info.value, "__notes__", [])) + str(info.value)
    assert "the updater on Mob" in text
    assert "upd" in text


def test_replay_error_hints_at_batched_time():
    scene = SceneManager.reset()
    m = Mob().spawn(animate=False)

    def upd(mob, t):
        if t > 1:
            mob.set(location=UP)
        else:
            mob.set(location=RIGHT * t)

    m.add_updater(upd)
    Scene.wait(2)
    with pytest.raises(RuntimeError) as info:
        scene.timeline_manager.set_state_to_times(torch.tensor([0.5, 1.5], dtype=F32))
    text = "\n".join(getattr(info.value, "__notes__", [])) + str(info.value)
    assert "torch.where" in text
    assert "one time per frame" in text


def test_spawn_at_negative_time_raises():
    SceneManager.reset()
    a = Mob().spawn(animate=False)
    b = Mob()
    with Seq() as context:
        a.move(RIGHT)
        context.rewind(3)
        with pytest.raises(AlganConfigurationError, match="before time zero"):
            b.spawn()
