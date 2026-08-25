import os

import pytest
import torch

from algan.animation_timeline.animation_contexts import Off
from algan.constants.spatial import LEFT, UP
from algan.environment import env_flag
from algan.geometry.geometry import map_global_to_local_coords
from algan.mobs.neural_nets.neural_net import (
    _IDLE_PARALLEL_RADIUS_FRACTION,
    NeuralNetMLPV3,
)
from algan.scene_manager import SceneManager


def test_idle_waypoints_are_squished_along_network_direction():
    SceneManager.reset()
    direction = torch.tensor([1.0, 2.0, -3.0])
    direction = direction / direction.norm()
    network = NeuralNetMLPV3([2, 3, 2], direction=direction)

    local_direction = map_global_to_local_coords(
        network.location, network.basis, network.location + direction
    )
    local_direction = local_direction / local_direction.norm()
    normalized_waypoints = network._idle_waypoints / network._idle_walk_radii.view(
        -1, 1, 1
    )
    parallel = (normalized_waypoints * local_direction).sum(dim=-1, keepdim=True)
    perpendicular = normalized_waypoints - parallel * local_direction

    assert parallel.abs().max() <= _IDLE_PARALLEL_RADIUS_FRACTION + 1e-6
    assert perpendicular.norm(dim=-1).max() <= 1 + 1e-6
    assert perpendicular.norm(dim=-1).amax() > parallel.abs().amax()
    SceneManager.reset()


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
    shown = locations.gather(0, visible.argmax(0).view(1, -1, 1).expand(1, -1, 3))[0]
    expected = (
        source.location.reshape(len(times), -1, 3)[:, 0]
        + target.location.reshape(len(times), -1, 3)[:, 0]
    ) * 0.5

    assert torch.equal(visible.sum(0), torch.ones_like(times))
    assert torch.equal(shown, expected)
    assert (shown[1:] - shown[:-1]).norm(dim=-1).min() > 1e-4
    timeline.clear_buffers()
    SceneManager.reset()


def _materialize_idle_buffers(dims, batched):
    """Build a net and materialize a window, returning every attr buffer."""
    previous = os.environ.get("ALGAN_BATCHED_IDLE_UPDATER")
    os.environ["ALGAN_BATCHED_IDLE_UPDATER"] = "1" if batched else "0"
    try:
        scene = SceneManager.reset()
        network = NeuralNetMLPV3(dims).move(LEFT).spawn()
        with Off():
            network.move(UP * 0.5)
        times = torch.arange(6) / scene.frames_per_second
        timeline = scene.timeline_manager
        with Off(record_attr_modifications=False, record_funcs=False):
            timeline.set_state_to_times(times, active_mobs=[network])
        buffers = {
            attr: tl.active_state.detach().clone()
            for attr, tl in timeline.attr_to_timeline.items()
        }
        directions = {}
        stack = [(network, "")]
        while stack:
            mob, path = stack.pop()
            d = getattr(mob, "direction", None)
            if torch.is_tensor(d):
                directions[f"{path}{type(mob).__name__}#{mob.id}"] = d.detach().clone()
            for i, child in enumerate(getattr(mob, "children", ())):
                stack.append((child, f"{path}{i}/"))
        timeline.clear_buffers()
        return buffers, directions
    finally:
        SceneManager.reset()
        if previous is None:
            os.environ.pop("ALGAN_BATCHED_IDLE_UPDATER", None)
        else:
            os.environ["ALGAN_BATCHED_IDLE_UPDATER"] = previous


@pytest.mark.fast
def test_batched_idle_updater_writes_what_the_loops_write():
    """The batched idle path must be bit-identical to the per-mob loops.

    Both arms materialize the same window on freshly built (deterministically
    seeded) nets and every attribute buffer -- plus the non-timeline
    ``direction`` attributes the updater assigns -- must compare equal under
    ``torch.equal``. This is the guard for ALGAN_BATCHED_IDLE_UPDATER.
    """
    assert env_flag("ALGAN_BATCHED_IDLE_UPDATER", True), (
        "the batched idle updater is expected to default on"
    )
    sequential, sequential_dirs = _materialize_idle_buffers([2, 3, 2], False)
    try:
        batched, batched_dirs = _materialize_idle_buffers([2, 3, 2], True)
    finally:
        pass
    assert set(sequential) == set(batched)
    for attr, value in sequential.items():
        assert torch.equal(value, batched[attr]), f"attribute {attr} diverged"
    assert set(sequential_dirs) == set(batched_dirs)
    for name, value in sequential_dirs.items():
        assert torch.equal(value, batched_dirs[name]), f"direction {name} diverged"
