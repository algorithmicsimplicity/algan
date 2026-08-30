import pytest
import torch

from algan import (
    WHITE,
    Color,
    HemisphereLight,
    Off,
    PointLight,
    Scene,
    Seq,
    Sync,
)
from algan.errors import AlganConfigurationError
from algan.rendering.raytracing import settings as rt_settings
from algan.utils.color_space import srgb_to_linear

# Not marked fast: this is a feature test for algan/rendering/lights.py. The
# timeline machinery it rides is canaried wholesale by the marked timeline
# files, and render-loop consumption is watched by tests/fast's pixel
# comparison.


@pytest.fixture(autouse=True)
def fresh_scene_stack():
    from algan.scene_manager import SceneManager

    SceneManager.reset()
    yield
    SceneManager.reset()


def _snapshot_index_for(scene_state, light):
    """Which entry of a render snapshot belongs to ``light``."""
    return next(
        i
        for i, candidate in enumerate(scene_state["light_objects"])
        if candidate is light
    )


def test_intensity_is_registered_as_animatable():
    with Scene():
        light = PointLight()

        assert "intensity" in light.animatable_attrs
        # A timeline-backed row, not the plain float it used to be -- which is
        # what lets it carry a different value per frame.
        value = light.intensity
        assert isinstance(value, torch.Tensor)
        assert float(value.reshape(-1)[0]) == pytest.approx(1.0)


def test_intensity_assignment_records_an_animation():
    with Scene() as scene:
        light = PointLight()
        with Seq():
            light.spawn(animate=False)
            start = scene.animation_manager.context.timespan.current_time
            with Sync(duration=2.0):
                light.intensity = 3
        end = start + 2.0

        scene.timeline_manager.set_state_to_times(
            torch.tensor([start, start + 1.0, end, end + 1.0])
        )
        values = light.intensity.reshape(-1)

        assert values.shape == (4,)
        assert float(values[0]) == pytest.approx(1.0, abs=1e-4)
        assert 1.0 < float(values[1]) < 3.0
        assert float(values[2]) == pytest.approx(3.0, abs=1e-3)
        assert float(values[3]) == pytest.approx(3.0, abs=1e-3)


def test_intensity_write_in_off_context_is_instant():
    with Scene() as scene:
        light = PointLight()
        with Seq():
            light.spawn(animate=False)
        with Off():
            light.intensity = 3

        # No ramp: the value has already landed at the very first frame.
        scene.timeline_manager.set_state_to_times(torch.tensor([0.0]))
        assert float(light.intensity.reshape(-1)[0]) == pytest.approx(3.0)


@pytest.mark.parametrize("bad", [-1, -0.5, float("nan"), float("inf"), float("-inf")])
def test_intensity_validation_on_every_write_path(bad):
    with Scene():
        with pytest.raises(AlganConfigurationError, match="intensity"):
            PointLight(intensity=bad)

        light = PointLight()
        with pytest.raises(AlganConfigurationError, match="intensity"):
            light.intensity = bad
        with pytest.raises(AlganConfigurationError, match="intensity"):
            light.set_intensity(bad)
        with pytest.raises(AlganConfigurationError, match="intensity"):
            light.set(intensity=bad)


def test_intensity_validator_tolerates_materialized_rows():
    """Regression guard for the tensor-tolerant validator.

    ``Animatable.__deepcopy__`` copies every animatable attribute through its
    setter (``setattr(clone, attr, getattr(self, attr))``), so after any state
    materialization the value arriving at the validation funnel is a
    ``[T, 1, 1]`` tensor rather than a scalar. A scalar-only ``float(value)``
    validator raises ``ValueError`` on exactly that input.

    Note that landing such a write -- or cloning a Mob at all after
    materialization -- currently fails on generic timeline machinery for plain
    Mobs and ``location`` identically (pre-existing, before intensity is ever
    reached), so this pins only the funnel: tensors pass, non-finite tensors
    are rejected.
    """
    from algan.rendering.lights import _validated_intensity

    with Scene() as scene:
        light = PointLight()
        light.spawn(animate=False)
        light.intensity = 2

        scene.timeline_manager.set_state_to_times(torch.tensor([0.5, 1.5]))
        materialized = light.intensity
        assert tuple(materialized.shape) == (2, 1, 1)

        # The multi-frame row passes validation unchanged...
        assert torch.equal(_validated_intensity(materialized), materialized)
        # ...while a non-finite one is rejected through the public funnel.
        with pytest.raises(AlganConfigurationError, match="intensity"):
            light.intensity = torch.tensor([float("nan")])
        with pytest.raises(AlganConfigurationError, match="intensity"):
            light.set_intensity(torch.tensor([float("inf")]))

        # A single-frame materialized value flows through the public setter
        # onto another light without a scalar-only validator choking on it.
        recipient = PointLight(color=WHITE)
        recipient.spawn(animate=False)
        scene.timeline_manager.set_state_to_times(torch.tensor([1.5]))
        recipient.set_intensity(light.intensity)


def test_animated_intensity_reaches_the_renderer_per_frame():
    with Scene() as scene:
        light = PointLight(color=WHITE, location=[0, 4, 3])
        with Seq():
            light.spawn(animate=False)
            start = scene.animation_manager.context.timespan.current_time
            with Sync(duration=2.0):
                light.intensity = 3

        fps = scene.frames_per_second
        window = int(fps * 4)
        scene.timeline_manager.set_state_to_times(
            torch.arange(window, dtype=torch.float32) / fps
        )
        state = scene._materialize_render_state(0, window)

        rows = state["lights"][_snapshot_index_for(state, light)][1][..., :3].reshape(
            window, -1
        )
        first = rows[0]
        ratio = rows[-1] / first
        # The last frame sits past the animation's end, where the recorded
        # value holds exactly: three times the opening frame.
        assert torch.allclose(ratio, torch.full_like(ratio, 3.0), rtol=1e-2)

        quarter = round((start + 0.5) * fps)
        three_quarters = round((start + 1.5) * fps)
        g_quarter = rows[quarter] / first
        g_three_quarters = rows[three_quarters] / first
        assert bool((g_quarter > 1.0).all().item())
        assert bool((g_quarter < 3.0).all().item())
        assert bool((g_three_quarters > g_quarter).all().item())


def test_hemisphere_ground_colour_tracks_animated_intensity_exactly_once():
    with Scene() as scene:
        light = HemisphereLight(
            color=Color((0.2, 0.4, 1.0)),
            ground_color=(0.8, 0.5, 0.25),
        )
        with Seq():
            light.spawn(animate=False)
            with Sync(duration=2.0):
                light.intensity = 4

        fps = scene.frames_per_second
        window = int(fps * 4)
        scene.timeline_manager.set_state_to_times(
            torch.arange(window, dtype=torch.float32) / fps
        )
        state = scene._materialize_render_state(0, window)

        aux = state["lights"][_snapshot_index_for(state, light)][2]
        ground = aux[..., 9:12].reshape(window, -1)
        intensities = light.intensity.reshape(window, -1)
        first_frame = ground[0]

        # Linear in intensity, frame by frame -- a double-applied intensity
        # would square these ratios instead.
        expected = first_frame * intensities
        assert torch.allclose(ground, expected, rtol=1e-3)
        assert torch.allclose(ground[-1], 4.0 * first_frame, rtol=1e-3)


def test_constant_intensity_snapshot_matches_legacy_arithmetic():
    with Scene() as scene:
        light = PointLight(color=Color((1.0, 0.9, 0.8)), intensity=0.85)
        light.spawn(animate=False)

        n_frames = 4
        scene.timeline_manager.set_state_to_times(
            torch.arange(n_frames, dtype=torch.float32) / scene.frames_per_second
        )
        state = scene._materialize_render_state(0, n_frames)

        rgba = light.color
        if rt_settings.linear_color_space:
            rgba = torch.cat(
                (srgb_to_linear(rgba[..., :3]), rgba[..., 3:]),
                -1,
            )
        legacy = rgba[..., :-1] * rgba[..., -1:] * light.opacity * 0.85
        got = state["lights"][_snapshot_index_for(state, light)][1].reshape(
            legacy.shape
        )

        assert torch.equal(got, legacy)
