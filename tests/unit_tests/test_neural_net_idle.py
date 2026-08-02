import torch

from algan.animation_timeline.animation_contexts import Off
from algan.mobs.neural_nets.neural_net import NeuralNetMLPV3
from algan.scene_manager import SceneManager


def test_activated_idle_synapse_follows_its_moving_neurons():
    scene = SceneManager.reset()
    network = NeuralNetMLPV3([2, 2, 2]).spawn()
    source = network.layers[0][0]
    target = network.layers[1][0]
    synapse = target.synapses[0]

    network.activate()
    times = torch.tensor([1.25, 2.25, 3.25, 4.25])
    timeline = scene.timeline_manager
    with Off(record_attr_modifications=False, record_funcs=False):
        timeline.set_state_to_times(times)

    updater = timeline.function_timeline.updaters[network.idle_updater_id]
    historical_synapses = updater._history_clones[id(synapse)][1]
    incarnations = [synapse, *historical_synapses]
    visible = torch.stack(
        [candidate.grid.opacity.mean(-2)[..., 0] for candidate in incarnations]
    )
    locations = torch.stack(
        [
            candidate.location.reshape(len(times), -1, 3)[:, 0]
            for candidate in incarnations
        ]
    )
    shown = locations.gather(
        0, visible.argmax(0).view(1, -1, 1).expand(1, -1, 3)
    )[0]
    expected = (
        source.location.reshape(len(times), -1, 3)[:, 0]
        + target.location.reshape(len(times), -1, 3)[:, 0]
    ) * 0.5

    assert torch.equal(visible.sum(0), torch.ones_like(times))
    assert torch.equal(shown, expected)
    assert (shown[1:] - shown[:-1]).norm(dim=-1).min() > 1e-4
    timeline.clear_buffers()
    SceneManager.reset()
