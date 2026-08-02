import pytest
import torch

from algan.animation_timeline.animation_contexts import Off
from algan.constants.color import GRAY_E, PURE_RED
from algan.mobs.neural_nets.neural_net import NeuralNetMLP
from algan.mobs.shapes_2d import Square
from algan.scene_manager import SceneManager


@pytest.fixture(autouse=True)
def fresh_scene():
    SceneManager.reset()
    yield
    SceneManager.reset()


def test_activation_materializes_output_without_ramping_scalar_opacity():
    scene = SceneManager.instance().current_scene
    network = NeuralNetMLP([1, 1]).spawn(animate=False)
    output = network.activate(lambda: Square(color=GRAY_E), color=PURE_RED)

    # Record-time state is the resolved output: visible, non-glowing, and back
    # at the color authored by the output generator.
    torch.testing.assert_close(
        output.texture_points.color,
        GRAY_E.view(1, 1, 5).expand_as(output.texture_points.color),
    )
    for part in output._wave_pulsed_parts():
        torch.testing.assert_close(part.opacity, torch.ones_like(part.opacity))

    # wave_color temporarily refines the square's one-sample fill grid. Inspect
    # that visible incarnation exactly as the renderer materializes it.
    refined = max(
        (actor for actor in scene.actors if isinstance(actor, Square)),
        key=lambda actor: actor.num_texture_points,
    )
    assert refined is output
    assert refined.num_texture_points > 1
    assert refined.lifespan.end() < 0
    start = refined.lifespan.start()
    times = torch.linspace(start + 1e-4, start + 3 - 1e-4, 241)
    with Off(record_attr_modifications=False, record_funcs=False):
        scene.timeline_manager.set_state_to_times(times)

    scalar_opacity = refined.opacity.reshape(len(times), -1)
    torch.testing.assert_close(scalar_opacity, torch.ones_like(scalar_opacity))

    colors = refined.texture_points.color.reshape(len(times), -1, 5)
    peak_glow, peak_times = colors[..., 3].max(0)
    assert peak_glow.min() > 0.98
    peak_alpha = colors[..., 4].gather(0, peak_times.unsqueeze(0))[0]
    assert peak_alpha.min() > 0.98
    peak_rgb = colors[..., :3].gather(
        0, peak_times.view(1, -1, 1).expand(1, -1, 3)
    )[0]
    expected_peak_rgb = PURE_RED.rgb
    torch.testing.assert_close(
        peak_rgb,
        expected_peak_rgb.view(1, 3).expand_as(peak_rgb),
        atol=2e-2,
        rtol=0,
    )
    assert colors[-1, ..., 3].max() < 1e-3
    assert colors[-1, ..., 4].min() > 0.999
    scene.timeline_manager.clear_buffers()
