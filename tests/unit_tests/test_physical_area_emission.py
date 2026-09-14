"""The area-light authoring API cannot reintroduce nonphysical emission."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from algan.errors import AlganConfigurationError
from algan.rendering.lights import (
    LIGHT_AUX_COLS,
    PointLight,
    RectAreaLight,
    SpotLight,
)
from algan.rendering.raytracing.area_light_quads import _quad_geometry
from algan.scene_manager import SceneManager


@pytest.fixture(autouse=True)
def reset_scene():
    SceneManager.reset()
    yield
    SceneManager.reset()


@pytest.mark.parametrize("samples", [1, 4, 9])
def test_area_rows_and_panel_have_the_same_physical_normalization(samples):
    light = RectAreaLight(width=2, height=3, samples=samples)
    location = torch.tensor([[0.0, 0.0, 4.0]])
    aux = light._build_aux(location)
    assert bool((aux[..., 1] == 2.0).all())
    assert bool((aux[..., 2] == 0.0).all())
    origin = light._get_sample_positions(location)
    count = origin.shape[1]
    strength = torch.tensor([12.0, 6.0, 3.0])
    snapshot = SimpleNamespace(
        origin=origin,
        _render_aux=aux,
        light_color=(strength / count).reshape(1, 1, 3).expand(1, count, 3),
        width=2.0,
        height=3.0,
    )
    _pos, _normal, radiance = _quad_geometry(snapshot, 1, torch.device("cpu"))
    assert torch.allclose(radiance[0], strength / 6.0)
    # The same snapshot at another location has the same emitted radiance.
    snapshot.origin = origin + torch.tensor([0.0, 0.0, 40.0])
    _pos, _normal, farther = _quad_geometry(snapshot, 1, torch.device("cpu"))
    assert torch.equal(radiance, farther)


@pytest.mark.parametrize("samples", [1, 4, 16])
@pytest.mark.parametrize("frames", [1, 3])
def test_area_quad_geometry_uses_the_complete_aux_layout(samples, frames):
    """Shadow-budget columns must not change geometry, frame axes or radiance."""
    light = RectAreaLight(width=2, height=3, samples=samples)
    location = torch.tensor(
        [[0.0, 0.0, 4.0], [1.0, 0.5, 5.0], [-0.5, 1.0, 6.0]]
    )[:frames]
    aux = light._build_aux(location)
    count = light._num_samples()
    assert aux.shape == (frames, count, LIGHT_AUX_COLS)
    # The extended layout appends primary/bounce fan sizes after the original
    # 13 columns. Geometry must still read the normal from columns 3:6.
    aux[..., 13] = 7.0
    aux[..., 14] = 3.0
    scale = torch.arange(1, frames + 1, dtype=torch.float32).unsqueeze(-1)
    strength = torch.tensor([12.0, 6.0, 3.0]) * scale
    snapshot = SimpleNamespace(
        origin=light._get_sample_positions(location),
        _render_aux=aux,
        light_color=(strength / count).unsqueeze(1).expand(-1, count, -1),
        width=light.width,
        height=light.height,
    )

    pos, normal, radiance = _quad_geometry(snapshot, frames, torch.device("cpu"))

    assert pos.shape == (frames, 2, 9)
    assert normal.shape == radiance.shape == (frames, 3)
    right, up = light._rect_axes(location)
    a = location - right * light.width / 2 - up * light.height / 2
    b = location + right * light.width / 2 - up * light.height / 2
    c = location + right * light.width / 2 + up * light.height / 2
    d = location - right * light.width / 2 + up * light.height / 2
    expected_pos = torch.stack((torch.cat((a, b, d), -1), torch.cat((b, c, d), -1)), 1)
    torch.testing.assert_close(pos, expected_pos)
    torch.testing.assert_close(normal, light._directions(location))
    torch.testing.assert_close(radiance, strength / (light.width * light.height))


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("decay", 0),
        ("decay", 1),
        ("decay", 3),
        ("decay", float("nan")),
        ("distance", 1),
        ("distance", -1),
        ("distance", float("inf")),
    ],
)
def test_nonphysical_area_controls_fail_on_construction_and_later_writes(name, value):
    with pytest.raises(AlganConfigurationError, match=name):
        RectAreaLight(**{name: value})
    light = RectAreaLight()
    for write in (
        lambda: setattr(light, name, value),
        lambda: light.set(**{name: value}),
        lambda: light.set_non_recursive(**{name: value}),
    ):
        with pytest.raises(AlganConfigurationError, match=name):
            write()
    assert light.decay == 2.0
    assert light.distance == 0.0


def test_canonical_legacy_keywords_and_cloning_remain_supported():
    light = RectAreaLight(decay=2, distance=0)
    light.set(decay=2, distance=0)
    clone = light.clone()
    assert clone.decay == 2.0
    assert clone.distance == 0.0


def test_point_and_spot_artistic_falloff_is_not_changed():
    for cls in (PointLight, SpotLight):
        light = cls(decay=1, distance=5)
        assert light.decay == 1.0
        assert light.distance == 5.0
