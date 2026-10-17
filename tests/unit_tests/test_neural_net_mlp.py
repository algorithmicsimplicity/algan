"""The MLP neural-net mob: how it is wired, and the animations it plays.

``NeuralNetMLP`` lays a fully connected network out as neurons (a core inside
a shell) joined by thin cylindrical synapses. ``activate`` pulses it layer by
layer, ``forward`` feeds an input Mob in and brings an output Mob out,
``backward`` carries the error back from a label, ``train_step`` chains the
two, ``increment_weight`` / ``unincrement_weight`` single out one weight, and
the module function ``zap`` fires a pulse along temporary wires from one Mob
to another.

Everything is read off recorded or materialized state; nothing renders. The
nets have their idle drift switched off: it is ``test_neural_net_idle.py``'s
subject, and with it off every endpoint and colour the animations write can be
compared exactly.

A feature test: it breaks when this module is worked on, not when anything
else is, so it stays out of the fast suite (``tests/README.md``).
"""

import contextlib
import shutil

import pytest
import torch
from torch.testing import assert_close

from algan.animation_timeline.animation_contexts import Off
from algan.constants.color import BLUE, GREEN, PURE_BLUE, PURE_RED, RED
from algan.constants.spatial import LEFT, RIGHT, UP
from algan.mobs.neural_nets.neural_net import (
    NeuralNetMLP,
    NeuralNetMLPV2,
    NeuralNetMLPV3,
    Synapse,
    zap,
)
from algan.mobs.shapes_2d import Square
from algan.rendering.shaders.materials import MeshStandardMaterial
from algan.scene import Scene
from algan.scene_manager import SceneManager

pytestmark = pytest.mark.usefixtures("fresh_scene")

#: Float32 world positions a few units from the origin.
ATOL = 1e-5


def _still_net(dims, net_cls=NeuralNetMLP, **kwargs):
    """A spawned net with its idle drift switched off."""
    net = net_cls(list(dims), **kwargs).spawn(animate=False)
    net.remove_updater(net.idle_updater_id)
    return net


def _cursor(scene):
    """Where the next top-level animation of ``scene`` starts, in seconds."""
    return float(scene.animation_manager.context.timespan.current_time)


def _point(mob, frame=0):
    """A one-row Mob's location, at ``frame`` once materialized."""
    return mob.location.reshape(-1, 3)[frame]


def _endpoints(synapse, frame=0):
    """A synapse's (start, end), at ``frame`` once materialized.

    Its basis row 1 is the whole start-to-end vector, about its midpoint
    (``Cylinder._move_between_points``).
    """
    middle = synapse.location.reshape(-1, 3)[frame]
    axis = synapse.basis.reshape(-1, 9)[frame, 3:6]
    return middle - axis / 2, middle + axis / 2


def _points(mob, count, direction=UP):
    """``count`` points spread over ``mob`` along ``direction``, as ``(3,)`` tensors."""
    return [p.reshape(3) for p in mob.sample_points_in_direction(direction, count)]


def _input_points(net, mob):
    """Where ``forward`` takes hold of an input Mob: one point per first-layer neuron."""
    direction = -(net.get_forward_direction() + net.get_up_direction())
    return _points(mob, len(net.layers[0]), direction)


def _part_colours(net):
    """The colour rows of every part a colour wave pulses: the synapse, core
    and shell grids.
    """
    return {id(part): part.color.clone() for part in net._wave_pulsed_parts()}


def _assert_colours_unchanged(net, colours):
    for part in net._wave_pulsed_parts():
        assert_close(part.color, colours[id(part)], rtol=0, atol=1e-6)


@contextlib.contextmanager
def _materialized(scene, times):
    """Materialize ``scene`` at ``times``: attributes gain a leading time axis."""
    timeline = scene.timeline_manager
    with Off(
        record_attr_modifications=False,
        record_funcs=False,
        animation_manager=scene.animation_manager,
    ):
        timeline.set_state_to_times(torch.as_tensor(times, dtype=torch.float32))
    try:
        yield
    finally:
        timeline.clear_buffers()


def _incarnations(mob):
    """``mob`` and the clones ``detach_history`` handed its earlier rows to.

    ``wave_color`` refines a synapse's grid for the length of its wave, which
    splits the synapse's history: what is on screen before and during the wave
    belongs to clones.
    """
    splits = mob.scene.timeline_manager._updater_history_clones.get(mob.id, ())
    return [mob, *(clone for _, clone in splits)]


def _shown(mob, time):
    """The incarnation of ``mob`` whose lifespan holds ``time``."""
    for candidate in _incarnations(mob):
        start, end = candidate.lifespan.start(), candidate.lifespan.end()
        # -1 is "never": not spawned yet, or never despawned.
        if 0 <= start <= time and (end < 0 or time < end):
            return candidate
    return None


def _glow(part, times):
    """The mean glow of whichever incarnation of ``part`` is shown, per time."""
    values = []
    for frame, time in enumerate(times.tolist()):
        shown = _shown(part, time)
        values.append(0.0 if shown is None else float(shown.color[frame, :, 3].mean()))
    return torch.tensor(values)


def _onset(part, times, threshold=0.05):
    """When ``part``'s pulse starts: the first time its mean glow passes
    ``threshold``.
    """
    lit = (_glow(part, times) > threshold).nonzero()
    assert len(lit), "the part never pulsed"
    return float(times[lit[0, 0]])


def _peak_rgb(part, times, window=(0.0, float("inf"))):
    """``part``'s mean colour where its glow peaks within ``window`` seconds."""
    glow = _glow(part, times)
    inside = (times >= window[0]) & (times < window[1])
    frame = int(torch.where(inside, glow, torch.full_like(glow, -1)).argmax())
    return _shown(part, float(times[frame])).color[frame, :, :3].mean(0)


def _same_wire(a, b):
    return all(torch.allclose(p, q, atol=ATOL) for p, q in zip(a, b))


def _new_wires(scene, actors_before, between=None):
    """Every incarnation of the wires ``zap`` put up since ``actors_before``.

    ``zap`` makes plain ``Synapse``s, and a wire's history clones are actors
    too. ``between`` keeps the wires running between those (start, end) pairs.
    Read before materializing: the geometry compared is the authored one.
    """
    return [
        actor
        for actor in scene.actors
        if id(actor) not in actors_before
        and type(actor) is Synapse
        and (between is None or any(_same_wire(_endpoints(actor), w) for w in between))
    ]


def _assert_wires(wires, expected):
    """``wires`` run between exactly the (start, end) pairs ``expected``."""
    distinct = []
    for wire in wires:
        ends = _endpoints(wire)
        if not any(_same_wire(ends, other) for other in distinct):
            distinct.append(ends)
    assert len(distinct) == len(expected)
    for wire in expected:
        assert any(_same_wire(wire, other) for other in distinct), (
            f"no wire runs from {wire[0].tolist()} to {wire[1].tolist()}"
        )


def _visibility(wires):
    """How visible ``wires`` are at each materialized time.

    Materialization zeroes each incarnation's opacity outside its own lifespan,
    so the most visible of them is the one on screen.
    """
    return torch.stack(
        [(w.grid.color[..., 4] * w.grid.opacity[..., 0]).amax(-1) for w in wires]
    ).amax(0)


@pytest.mark.parametrize(
    "net_cls", [NeuralNetMLP, NeuralNetMLPV2, NeuralNetMLPV3], ids=["v1", "v2", "v3"]
)
def test_each_neuron_is_wired_to_every_neuron_of_the_layer_before(net_cls):
    """A neuron has one synapse from each neuron of the previous layer, in that
    layer's order, ending on it. A first-layer neuron instead has one input
    synapse reaching back half a layer spacing, which is where ``forward``
    takes hold of it. The synapses are the neuron's children, so they move
    with it.
    """
    spacing = 1.5
    net = net_cls([2, 3, 2], layer_spacing=spacing)

    for neuron in net.layers[0]:
        (synapse,) = neuron.synapses
        start, end = _endpoints(synapse)
        assert_close(end, _point(neuron), rtol=0, atol=ATOL)
        assert_close(start, _point(neuron) + LEFT * spacing / 2, rtol=0, atol=ATOL)

    for previous, layer in zip(net.layers, net.layers[1:]):
        for neuron in layer:
            assert len(neuron.synapses) == len(previous)
            for source, synapse in zip(previous, neuron.synapses):
                start, end = _endpoints(synapse)
                assert_close(start, _point(source), rtol=0, atol=ATOL)
                assert_close(end, _point(neuron), rtol=0, atol=ATOL)

    for layer in net.layers:
        for neuron in layer:
            for synapse in neuron.synapses:
                assert any(child is synapse for child in neuron.children)


def test_v2_net_builds_its_parts_from_standard_materials():
    """REGRESSION. ``NeuronV2`` and ``SynapseV2`` passed Three.js's camelCase
    ``envMapIntensity``, which the material constructors reject, so
    ``NeuralNetMLPV2`` -- documented as a drop-in replacement -- could not be
    built at all.

    Its docstring's look, read back off the parts: a glossy self-lit core in a
    translucent halo shell, joined by filament synapses, all shaded by
    ``MeshStandardMaterial`` with their own environment fill.
    """
    net = NeuralNetMLPV2([1, 2])
    neuron = net.layers[1][0]
    core, shell, synapse = neuron.core, neuron.shell, neuron.synapses[0]

    for part, fill in ((core, 4.0), (shell, 2.5), (synapse, 4.5)):
        assert part.shader is MeshStandardMaterial.shader
        assert float(part.env_map_intensity.reshape(-1)[0]) == pytest.approx(fill)
        assert float(part.metalness.reshape(-1)[0]) == 0
    assert float(core.roughness.reshape(-1)[0]) < float(shell.roughness.reshape(-1)[0])
    assert float(core.color.glow.reshape(-1)[0]) > 0
    assert float(core.color.opacity.reshape(-1)[0]) == 1
    assert float(shell.color.opacity.reshape(-1)[0]) < 1


def test_synapse_colours_are_jittered_the_same_way_on_every_build():
    """Each synapse takes its neuron's colour a quarter of the way towards a
    random one, for variety. The draws come from the module's own generator,
    reseeded per net: a net comes out the same however many nets or other
    random draws came before it, and building one takes nothing from torch's
    global generator.
    """

    def synapse_colours(net):
        return torch.stack(
            [
                synapse.color.reshape(-1, 5)[0]
                for layer in net.layers
                for neuron in layer
                for synapse in neuron.synapses
            ]
        )

    first = synapse_colours(NeuralNetMLP([2, 2]))
    NeuralNetMLP([3, 1])
    torch.rand(100)
    state = torch.get_rng_state()
    second = synapse_colours(NeuralNetMLP([2, 2]))

    assert torch.equal(torch.get_rng_state(), state)
    assert torch.equal(first, second)
    assert len({tuple(colour.tolist()) for colour in first}) == len(first)
    assert (first[:, :3] - GREEN[:3]).abs().max() <= 0.25 + 1e-6
    assert_close(first[:, 3:], GREEN[3:].expand(len(first), 2))


def test_activation_lights_the_net_layer_by_layer_and_settles_back(fresh_scene):
    """``activate`` pulses every neuron, a layer at a time from the input
    side: a neuron's incoming synapses light before it does, the neurons of a
    layer light together, the pulse is the colour asked for, and every part
    settles back to its own colour afterwards.
    """
    net = _still_net([1, 2, 1])
    resting = _part_colours(net)

    net.activate(color=PURE_BLUE)

    times = torch.arange(0.0, _cursor(fresh_scene) + 0.25, 0.025)
    with _materialized(fresh_scene, times):
        starts = []
        for layer in net.layers:
            onsets = [_onset(neuron.core.grid, times) for neuron in layer]
            assert len(set(onsets)) == 1, "a layer did not light together"
            starts.append(onsets[0])
            for neuron in layer:
                synapses = [_onset(s.grid, times) for s in neuron.synapses]
                assert max(synapses) < onsets[0], "a neuron lit before its synapses"
                peak = _peak_rgb(neuron.core.grid, times)
                assert_close(peak, PURE_BLUE[:3], rtol=0, atol=0.05)
        hidden = net.layers[1]
        assert_close(
            _glow(hidden[0].core.grid, times), _glow(hidden[1].core.grid, times)
        )
    assert starts == sorted(set(starts)), "the layers did not light in turn"
    _assert_colours_unchanged(net, resting)


def test_reverse_activation_runs_from_the_output_layer_and_nudges_the_weights(
    fresh_scene,
):
    """``activate(reverse=True)`` -- the backward pass -- runs the other way,
    from the output layer back, each neuron lighting before its incoming
    synapses. The neurons settle back to their own colour; the synapses, the
    weights, settle on theirs nudged up to a third of the way towards another,
    which is how the pass shows them being updated.
    """
    net = _still_net([1, 2, 1])
    resting = _part_colours(net)

    net.activate(reverse=True)

    times = torch.arange(0.0, _cursor(fresh_scene) + 0.25, 0.025)
    with _materialized(fresh_scene, times):
        starts = []
        for layer in net.layers:
            onsets = [_onset(neuron.core.grid, times) for neuron in layer]
            assert len(set(onsets)) == 1, "a layer did not light together"
            starts.append(onsets[0])
            for neuron in layer:
                synapses = [_onset(s.grid, times) for s in neuron.synapses]
                assert onsets[0] < min(synapses), "a synapse lit before its neuron"
    assert starts == sorted(set(starts), reverse=True), "not output layer first"

    for layer in net.layers:
        for neuron in layer:
            for part in (neuron.core.grid, neuron.shell.grid):
                assert_close(part.color, resting[id(part)], rtol=0, atol=1e-6)
            for synapse in neuron.synapses:
                before, after = resting[id(synapse.grid)], synapse.grid.color
                change = float((after[..., :3] - before[..., :3]).abs().amax())
                assert 1e-4 < change <= 1 / 3 + 1e-6
                assert_close(after[..., 3:], before[..., 3:], rtol=0, atol=1e-6)


@pytest.mark.parametrize("reverse", [False, True], ids=["forward", "reverse"])
def test_a_synapse_pulse_travels_the_way_the_signal_does(fresh_scene, reverse):
    """Along a synapse the pulse is a wave rather than a flash: in the forward
    pass it sets off from the source neuron's end and arrives at the
    target's, and the backward pass sends it the other way.
    """
    net = _still_net([1, 1])
    synapse = net.layers[1][0].synapses[0]
    source, target = _endpoints(synapse)

    net.activate(reverse=reverse)

    times = torch.arange(0.0, _cursor(fresh_scene), 0.025)
    with _materialized(fresh_scene, times):
        peak = float(times[int(_glow(synapse.grid, times).argmax())])
        wave = _shown(synapse.grid, peak)
        frames = [
            i for i, t in enumerate(times.tolist()) if _shown(synapse.grid, t) is wave
        ]
        span = target - source
        along = ((wave.location[frames[0]] - source) @ span) / span.dot(span)
        lit = wave.color[frames, :, 3] > 0.05
        assert lit.any(0).all(), "part of the synapse never lit"
        onset = times[frames][lit.float().argmax(0)]
    near_source = float(onset[along < 0.25].mean())
    near_target = float(onset[along > 0.75].mean())
    if reverse:
        assert near_target < near_source
    else:
        assert near_source < near_target


@pytest.mark.parametrize("reset", [True, False], ids=["reset", "no_reset"])
def test_forward_feeds_the_input_in_and_brings_the_output_out(fresh_scene, reset):
    """``forward`` takes the first layer's input synapses from where they wait
    to points spread over the input Mob -- neuron by neuron in order, so no two
    wires cross -- pulses the net, and brings the generator's Mob out against
    the middle output neuron, revealed in its own colours and then nudged
    clear along the net's direction. With ``reset`` (the default) the input
    synapses go back to waiting at the end; without it they stay on the
    input. The whole pass takes ``runtime`` seconds.
    """
    net = _still_net([2, 2, 3])
    source = Square(size=0.5).move(LEFT * 2).spawn(animate=False)
    waiting = [_endpoints(neuron.synapses[0]) for neuron in net.layers[0]]
    points = _input_points(net, source)
    # The lower neuron takes the lower point: the wires run side by side.
    assert points[0][1] < points[1][1]
    assert _point(net.layers[0][0])[1] < _point(net.layers[0][1])[1]
    authored = [
        p.color.clone() for p in Square(size=0.3, color=RED)._wave_pulsed_parts()
    ]

    output = net.forward(
        source, output_generator=lambda: Square(size=0.3, color=RED), reset=reset
    )

    assert isinstance(output, Square)
    assert _cursor(fresh_scene) == pytest.approx(3.0)
    appears = output.lifespan.start()
    assert 0 < appears < 3
    with _materialized(fresh_scene, torch.tensor([1.5, appears])):
        for neuron, point in zip(net.layers[0], points):
            start, end = _endpoints(_shown(neuron.synapses[0], 1.5), 0)
            assert_close(start, point, rtol=0, atol=ATOL)
            assert_close(end, _point(neuron, 0), rtol=0, atol=ATOL)
        for part in output._wave_pulsed_parts():
            assert part.color[1, :, 4].abs().max() == 0, "the output showed early"

    for part, colour in zip(output._wave_pulsed_parts(), authored, strict=True):
        # The reveal refines a coarsely sampled part to show its wave; a
        # uniformly coloured one comes back in its colour on every new row.
        assert_close(part.color, colour.expand_as(part.color), rtol=0, atol=1e-6)
    forward = net.get_forward_direction().reshape(3)
    middle = net.layers[-1][1]
    gap = output.get_boundary_point(-forward) - middle.get_boundary_point(forward)
    assert float(gap.reshape(3).dot(forward)) == pytest.approx(0.25, abs=1e-4)
    distances = [float((_point(output) - _point(n)).norm()) for n in net.layers[-1]]
    assert min(range(3), key=distances.__getitem__) == 1

    for neuron, point, (rest_start, rest_end) in zip(net.layers[0], points, waiting):
        start, end = _endpoints(neuron.synapses[0])
        assert_close(start, rest_start if reset else point, rtol=0, atol=ATOL)
        assert_close(end, rest_end, rtol=0, atol=ATOL)


def test_forward_without_an_output_generator_feeds_the_input_and_returns_none(
    fresh_scene,
):
    """REGRESSION. ``activate`` returns None when there is no output
    generator, and ``forward`` went on to move that None, so
    ``forward(inputs)`` -- with its own default ``output_generator=None`` --
    raised AttributeError.

    Without a generator the pass is the same, less the output: the input is
    taken hold of, the net pulses, and the input synapses go back to waiting,
    in ``runtime`` seconds.
    """
    net = _still_net([2, 1])
    source = Square(size=0.5).move(LEFT * 2).spawn(animate=False)
    waiting = [_endpoints(neuron.synapses[0]) for neuron in net.layers[0]]
    points = _input_points(net, source)

    assert net.forward(source, runtime=2) is None

    assert _cursor(fresh_scene) == pytest.approx(2.0)
    with _materialized(fresh_scene, torch.tensor([1.0])):
        for neuron, point in zip(net.layers[0], points):
            start, _ = _endpoints(_shown(neuron.synapses[0], 1.0), 0)
            assert_close(start, point, rtol=0, atol=ATOL)
    for neuron, (rest_start, rest_end) in zip(net.layers[0], waiting):
        start, end = _endpoints(neuron.synapses[0])
        assert_close(start, rest_start, rtol=0, atol=ATOL)
        assert_close(end, rest_end, rtol=0, atol=ATOL)


def test_reset_input_synapses_puts_the_inputs_back_before_the_first_layer(
    fresh_scene,
):
    """``reset_input_synapses`` sends each first-layer input synapse back to
    waiting -- from half a layer spacing before its neuron, to the neuron --
    in one second, wherever it had been pulled to.
    """
    net = _still_net([2, 1])
    waiting = [_endpoints(neuron.synapses[0]) for neuron in net.layers[0]]
    pulled_to = LEFT * 3 + UP * 2
    for neuron in net.layers[0]:
        neuron.synapses[0].set_start_point(pulled_to)
    pulled = _cursor(fresh_scene)

    net.reset_input_synapses()

    assert _cursor(fresh_scene) - pulled == pytest.approx(1.0)
    with _materialized(fresh_scene, torch.tensor([pulled, pulled + 1.0])):
        for neuron, (rest_start, rest_end) in zip(net.layers[0], waiting):
            synapse = neuron.synapses[0]
            assert_close(_endpoints(synapse, 0)[0], pulled_to, rtol=0, atol=ATOL)
            start, end = _endpoints(synapse, 1)
            assert_close(start, rest_start, rtol=0, atol=ATOL)
            assert_close(end, rest_end, rtol=0, atol=ATOL)


def test_backward_carries_the_error_from_the_label_through_the_output_into_the_net(
    fresh_scene,
):
    """Given an ``output`` and its ``label``, ``backward`` zaps the label's
    error into the output, then from the output into the output layer, along
    wires gone again by the end; the reverse pass, in ``color``, lights the
    output layer once the error has reached it. All in ``runtime`` seconds.
    """
    # A V3 net's synapses are SynapseV3s, so zap's wires -- plain Synapses --
    # and their history clones are told apart from the net's own by type.
    net = _still_net([1, 2, 1], NeuralNetMLPV3)
    output_neuron = net.layers[-1][0]
    output = Square(size=0.3, color=RED).move_to(_point(output_neuron) + RIGHT * 0.5)
    output.spawn(animate=False)
    label = Square(size=0.3, color=BLUE).move(RIGHT * 3.5).spawn(animate=False)
    into_output = list(zip(_points(label, 3), _points(output, 3)))
    into_net = list(zip(_points(output, 3), _points(output_neuron.shell, 3)))
    actors = {id(actor) for actor in fresh_scene.actors}
    start = _cursor(fresh_scene)

    assert net.backward(output, label, runtime=2) is net

    end = _cursor(fresh_scene)
    assert end - start == pytest.approx(2.0)
    _assert_wires(_new_wires(fresh_scene, actors), into_output + into_net)
    first = _new_wires(fresh_scene, actors, into_output)
    second = _new_wires(fresh_scene, actors, into_net)
    times = torch.arange(start, end + 0.25, 0.02)
    with _materialized(fresh_scene, times):
        first, second = _visibility(first), _visibility(second)
        reached = float(times[(second > 0.05).nonzero()[0, 0]])
        assert float(times[(first > 0.05).nonzero()[0, 0]]) < reached
        assert reached < _onset(output_neuron.core.grid, times, threshold=0.2)
        peak = _peak_rgb(output_neuron.core.grid, times)
        assert_close(peak, PURE_BLUE[:3], rtol=0, atol=0.05)
    assert first[times >= end].max() == 0, "a wire outlived the pass"
    assert second[times >= end].max() == 0, "a wire outlived the pass"


def test_train_step_runs_forward_in_one_colour_then_backward_in_another(fresh_scene):
    """``train_step`` is a forward pass that keeps the input attached, a
    backward pass from the label, and the output's exit: ``runtime`` seconds
    for each pass, pulsing in ``forward_color`` and then ``backward_color``,
    and a second for the output to go. It leaves the input synapses back
    waiting and the output gone.
    """
    net = _still_net([1, 2, 1], NeuralNetMLPV3)
    source = Square(size=0.5).move(LEFT * 2).spawn(animate=False)
    label = Square(size=0.3, color=BLUE).move(RIGHT * 3.5).spawn(animate=False)
    waiting = [_endpoints(neuron.synapses[0]) for neuron in net.layers[0]]
    actors = {id(actor) for actor in fresh_scene.actors}

    # The defaults swapped, so each colour can only have come from its argument.
    result = net.train_step(
        source,
        lambda: Square(size=0.3, color=RED),
        label,
        runtime=2,
        forward_color=PURE_BLUE,
        backward_color=PURE_RED,
    )

    assert result is net
    assert _cursor(fresh_scene) == pytest.approx(5.0)
    (output,) = [
        actor
        for actor in fresh_scene.actors
        if id(actor) not in actors and type(actor) is Square
    ]
    assert output.lifespan.end() == pytest.approx(5.0)
    core = net.layers[1][0].core.grid
    times = torch.arange(0.0, 5.0, 0.025)
    with _materialized(fresh_scene, times):
        forward_peak = _peak_rgb(core, times, (0, 2))
        backward_peak = _peak_rgb(core, times, (2, 4))
    assert_close(forward_peak, PURE_BLUE[:3], rtol=0, atol=0.05)
    assert_close(backward_peak, PURE_RED[:3], rtol=0, atol=0.05)
    for neuron, (rest_start, rest_end) in zip(net.layers[0], waiting):
        start, end = _endpoints(neuron.synapses[0])
        assert_close(start, rest_start, rtol=0, atol=ATOL)
        assert_close(end, rest_end, rtol=0, atol=ATOL)


@pytest.mark.skipif(
    not shutil.which("latex") or not shutil.which("dvisvgm"),
    reason="the increment label is Tex, which needs latex and dvisvgm",
)
def test_increment_weight_singles_out_one_weight_and_unincrement_undoes_it(
    fresh_scene,
):
    """``increment_weight`` picks the first synapse into the second layer,
    brightens it, and then writes ``w := w + 0.001`` beside it, below and to
    the left. ``unincrement_weight`` takes the label away and gives the weight
    back exactly the colour it had. No other part is touched.
    """
    net = _still_net([1, 2, 1])
    weight = net.layers[1][0].synapses[0]
    resting = _part_colours(net)

    assert net.increment_weight() is weight

    before, brightened = resting[id(weight.grid)], weight.grid.color
    assert_close(brightened[..., 3], before[..., 3] + 0.2)
    assert_close(brightened[..., [0, 1, 2, 4]], before[..., [0, 1, 2, 4]])
    label = net.increment_label
    assert label.lifespan.start() == pytest.approx(1.0), "not after the brightening"
    offset = _point(label) - _point(weight)
    assert float(offset[0]) < 0, "the label is not to the left"
    assert float(offset[1]) < 0, "the label is not below"

    net.unincrement_weight()

    assert label.lifespan.end() > label.lifespan.start()
    _assert_colours_unchanged(net, resting)


def test_zap_fires_a_pulse_along_new_wires_from_one_mob_to_the_other(fresh_scene):
    """``zap`` puts up wires between matching points of two Mobs -- three,
    spread along ``direction`` (``UP``) -- and sends a pulse of ``color`` along
    them from the first Mob to the second, in one second. The wires show only
    while the pulse is on them, and are gone afterwards.
    """
    first = Square(size=1).move(LEFT * 2).spawn(animate=False)
    second = Square(size=1).move(RIGHT * 2).spawn(animate=False)
    actors = {id(actor) for actor in fresh_scene.actors}

    zap(first, second, color=PURE_RED)

    assert _cursor(fresh_scene) == pytest.approx(1.0)
    wires = _new_wires(fresh_scene, actors)
    _assert_wires(wires, list(zip(_points(first, 3), _points(second, 3))))
    times = torch.arange(0.0, 1.25, 0.05)
    with _materialized(fresh_scene, times):
        shown = _visibility(wires)
        alpha = torch.cat(
            [w.grid.color[..., 4] * w.grid.opacity[..., 0] for w in wires], -1
        )
        x = torch.cat([w.grid.location[..., 0] for w in wires], -1)
        rgb = torch.cat([w.grid.color[..., :3] for w in wires], -2)
    lit = alpha.sum(-1) > 1
    fronts = ((alpha * x).sum(-1) / alpha.sum(-1).clamp_min(1e-6))[lit]

    assert shown[0] == 0, "the wires showed before the pulse"
    assert shown[times >= 1.0].max() == 0, "the wires outlived the zap"
    assert shown.max() > 0.5
    assert fronts.tolist() == sorted(fronts.tolist())
    assert fronts[-1] - fronts[0] > 1
    assert_close(
        rgb[alpha > 0.9],
        PURE_RED[:3].expand(int((alpha > 0.9).sum()), 3),
        rtol=0,
        atol=0.1,
    )


@pytest.mark.parametrize("num_points", [1, 5])
def test_zap_puts_up_as_many_wires_as_asked(fresh_scene, num_points):
    """REGRESSION. ``zap`` sampled the default three points of each Mob
    whatever ``num_points`` asked for: more than three raised IndexError, and
    fewer took the topmost instead of spreading over the Mobs.

    It puts up ``num_points`` wires, between points spread evenly along
    ``direction``: the first Mob's i-th to the second's i-th.
    """
    first = Square(size=1).move(LEFT * 2).spawn(animate=False)
    second = Square(size=2).move(RIGHT * 2 + UP).spawn(animate=False)
    actors = {id(actor) for actor in fresh_scene.actors}

    zap(first, second, num_points=num_points)

    expected = list(zip(_points(first, num_points), _points(second, num_points)))
    _assert_wires(_new_wires(fresh_scene, actors), expected)


def test_a_net_in_a_covered_scene_keeps_its_timing():
    """REGRESSION. ``backward`` opened its outermost context on the active
    Scene's animation manager rather than the net's, so for a net whose Scene
    was covered by another, its ``runtime`` was ignored: ``backward(runtime=3)``
    ran 4.49 seconds, and ``train_step``, which runs it, 9.49 rather than 7.

    A net records on its own Scene's timeline, and records the same there
    whichever Scene is active.
    """

    def record(covered):
        SceneManager.reset()
        home = Scene()
        net = _still_net([1, 1], scene=home)
        source = Square(size=0.5, scene=home).move(LEFT * 2).spawn(animate=False)
        label = Square(size=0.3, scene=home).move(RIGHT * 3).spawn(animate=False)
        active = Scene() if covered else home

        def output():
            return Square(size=0.3, color=RED, scene=home)

        durations = []
        for animate in (
            net.activate,
            lambda: net.forward(source, output),
            net.backward,
            lambda: net.train_step(source, output, label),
        ):
            start = _cursor(home)
            animate()
            durations.append(_cursor(home) - start)
        if covered:
            assert _cursor(active) == 0, "the active Scene recorded the net's animation"
        times = torch.arange(0.0, _cursor(home), 0.1)
        with _materialized(home, times):
            glow = net.layers[1][0].core.grid.color[..., 3].clone()
        return durations, glow

    durations, glow = record(covered=True)
    assert durations[2] == pytest.approx(3.0)
    assert durations[3] == pytest.approx(7.0)
    alone_durations, alone_glow = record(covered=False)
    assert durations == pytest.approx(alone_durations)
    assert_close(glow, alone_glow)
