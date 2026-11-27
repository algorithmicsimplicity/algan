"""An updater that writes opacity does not keep a despawned Mob on screen.

Materialization zeroes the rows of a Mob outside its lifespan, but updaters
run after that, so one writing ``opacity`` put a despawned Mob back at full
opacity for the rest of the video. The lifespan mask is applied again once the
updaters have run; inside the lifespan the updater's value still wins.
"""

import pytest
import torch

from algan import Mob, Scene, Seq
from algan.constants import easings
from algan.scene_manager import SceneManager

F32 = torch.float32


def _query(scene, mob, attr, times):
    timeline = scene.timeline_manager
    out = []
    with timeline.preserving_authoring_state():
        for t in times:
            timeline.set_state_to_times(torch.tensor([t], dtype=F32))
            try:
                out.append(float(getattr(mob, attr).reshape(-1)[0]))
            finally:
                timeline.clear_buffers()
    return out


def _opacity_scene(with_updater):
    scene = SceneManager.reset()
    m = Mob().spawn(animate=False)
    if with_updater:
        m.add_updater(
            lambda mob, t: mob.set(opacity=torch.ones_like(torch.as_tensor(t)))
        )
    Scene.wait(1)
    with Seq(easing=easings.identity):
        m.despawn()
    Scene.wait(1)
    return scene, m


@pytest.mark.parametrize("with_updater", [True, False])
def test_opacity_updater_does_not_survive_despawn(with_updater):
    scene, m = _opacity_scene(with_updater)
    got = _query(scene, m, "opacity", [0.5, 2.5])
    assert got == pytest.approx([1.0, 0.0], abs=1e-4)


def test_opacity_updater_may_override_fade_while_alive():
    scene, m = _opacity_scene(True)
    assert _query(scene, m, "opacity", [1.5])[0] == pytest.approx(1.0, abs=1e-4)


def test_a_batch_of_frames_masks_the_despawned_ones_too():
    scene, m = _opacity_scene(True)
    timeline = scene.timeline_manager
    with timeline.preserving_authoring_state():
        timeline.set_state_to_times(torch.tensor([0.5, 1.5, 2.5], dtype=F32))
        try:
            got = m.opacity.reshape(3, -1)[:, 0].tolist()
        finally:
            timeline.clear_buffers()
    assert got == pytest.approx([1.0, 1.0, 0.0], abs=1e-4)
