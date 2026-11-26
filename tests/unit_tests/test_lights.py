import math
import warnings

import pytest
import torch

from algan import (
    UP,
    WHITE,
    Color,
    DirectionalLight,
    HemisphereLight,
    Off,
    PointLight,
    RectAreaLight,
    Scene,
    Seq,
    SpotLight,
    Sync,
)
from algan.animatable_base.mob import Mob
from algan.errors import AlganConfigurationError, AlganWarning
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
        assert float(value.reshape(-1)[0]) == pytest.approx(math.pi)


def test_point_light_defaults_to_the_intensity_that_renders_white():
    # Irradiance units (see light-units): a white matte surface returns 1/pi
    # of a light, so PI is the bare PointLight that lights it white -- the
    # default scene light's intensity too.
    with Scene():
        assert float(PointLight().intensity.reshape(-1)[0]) == pytest.approx(math.pi)
        assert float(HemisphereLight().intensity.reshape(-1)[0]) == pytest.approx(1.0)


def test_intensity_assignment_records_an_animation():
    with Scene() as scene:
        light = PointLight(intensity=1.0)
        with Seq():
            light.spawn(animate=False)
            start = scene.animation_manager.context.timespan.current_time
            with Sync(runtime=2.0):
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
        light = PointLight(color=WHITE, location=[0, 4, 3], intensity=1.0)
        with Seq():
            light.spawn(animate=False)
            start = scene.animation_manager.context.timespan.current_time
            with Sync(runtime=2.0):
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
            with Sync(runtime=2.0):
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


@pytest.mark.parametrize("hex_color", [0x8B4513, "#8B4513"])
def test_hemisphere_ground_color_accepts_hex_like_every_material_color(hex_color):
    # three.js writes ``new HemisphereLight(0x87ceeb, 0x8b4513)``. Unparsed, the
    # int packed as a one-element radiance of 9127187 on every channel (about
    # 4.5e16 once decoded), and the string raised a bare TypeError from inside
    # the render loop.
    with Scene():
        from_hex = HemisphereLight(ground_color=hex_color)
        from_color = HemisphereLight(ground_color=Color("#8B4513"))
        location = torch.zeros(2, 3)
        got = from_hex._build_aux(location)[..., 9:12]
        expected = from_color._build_aux(location)[..., 9:12]

    assert torch.equal(got, expected)
    assert float(got.max()) < 1.0


@pytest.mark.parametrize(
    "bad",
    ["not a color", (0.5, 0.5), (0.5, float("nan"), 0.5), (0.5, -0.1, 0.5), True],
)
def test_hemisphere_ground_color_is_validated_at_construction(bad):
    with Scene(), pytest.raises(AlganConfigurationError, match="ground_color"):
        HemisphereLight(ground_color=bad)


def test_a_light_on_its_own_target_says_it_emits_nothing():
    # Both default to ORIGIN, so DirectionalLight() packs a zero direction and
    # lights nothing; it used to do so in silence.
    with Scene():
        light = DirectionalLight(color=WHITE)
        location = torch.zeros(3, 3)
        with pytest.warns(AlganWarning, match="DirectionalLight emits no light"):
            aux = light._build_aux(location)
        # The packing itself is unchanged: the warning is the whole fix.
        assert torch.equal(aux[..., 3:6], torch.zeros(3, 1, 3))
        # Once per light, not once per batch.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            light._build_aux(location)


@pytest.mark.parametrize("cls", [SpotLight, RectAreaLight])
def test_a_light_aimed_at_a_mob_where_it_stands_says_so(cls):
    with Scene():
        target = Mob(location=UP)
        light = cls(location=UP, target=target)
        with pytest.warns(AlganWarning, match="same point"):
            light._build_aux(torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))


def test_a_light_off_its_target_does_not_warn():
    with Scene():
        lights = [
            DirectionalLight(location=UP),
            SpotLight(location=UP),
            RectAreaLight(location=UP),
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            for light in lights:
                light._build_aux(torch.tensor([[0.0, 1.0, 0.0]] * 2))
