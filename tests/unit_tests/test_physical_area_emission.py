"""The area-light authoring API cannot reintroduce nonphysical emission."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from algan.errors import AlganConfigurationError
from algan.rendering.lights import PointLight, RectAreaLight, SpotLight
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
