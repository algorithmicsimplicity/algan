"""``RayTracingPreset``: a frozen copy of the ray tracer's configuration.

``SETTINGS.raytracing.as_preset()`` captures one, ``SETTINGS.snapshot()`` holds
one, and ``preset.set(...)`` derives another, which ``SETTINGS.raytracing.set
(preset)`` later applies. A preset is made in one place and applied in another,
so what it accepts matters as much as what the live settings accept.

A feature test, so it sits outside the fast suite: ``test_settings_api.py``
holds the contracts every subsystem leans on; this file only breaks when the
preset type itself changes.
"""

from __future__ import annotations

import pytest

from algan import SETTINGS, AlganConfigurationError
from algan.settings import RayTracingPreset
from algan.settings.raytracing_settings import _IMPORT_FROZEN_FIELDS


def _preset() -> RayTracingPreset:
    return SETTINGS.raytracing.as_preset()


def test_a_preset_captures_the_live_values_and_stays_put():
    SETTINGS.raytracing.set(samples_per_pixel=3)
    preset = _preset()
    SETTINGS.raytracing.set(samples_per_pixel=5)

    assert preset.is_preset
    assert not SETTINGS.raytracing.is_preset
    assert preset.samples_per_pixel == 3
    assert preset.to_dict() == {
        **SETTINGS.raytracing.to_dict(),
        "samples_per_pixel": 3,
    }


def test_set_derives_a_copy_and_leaves_the_original_alone():
    preset = _preset()
    before = preset.samples_per_pixel

    derived = preset.set(samples_per_pixel=before + 7, shadows=False)

    assert derived is not preset
    assert derived.samples_per_pixel == before + 7
    assert derived.shadows is False
    assert preset.samples_per_pixel == before
    assert preset.set_samples_per_pixel(9).samples_per_pixel == 9
    assert preset.samples_per_pixel == before


def test_a_preset_cannot_be_assigned_to():
    preset = _preset()
    with pytest.raises(AlganConfigurationError, match="immutable"):
        preset.samples_per_pixel = 4


def test_reading_an_unknown_field_is_an_attribute_error():
    with pytest.raises(AttributeError):
        _preset().samples_per_pixl  # noqa: B018


def test_a_misspelled_field_is_refused_with_a_suggestion():
    with pytest.raises(
        AlganConfigurationError, match="Did you mean 'samples_per_pixel'"
    ):
        _preset().set(samples_per_pixl=4)


def test_set_merges_a_source_and_lets_keywords_win():
    preset = _preset()
    source = preset.set(samples_per_pixel=6, max_bounces=2)

    merged = preset.set(source, max_bounces=4)
    assert merged.samples_per_pixel == 6
    assert merged.max_bounces == 4

    SETTINGS.raytracing.set(samples_per_pixel=8)
    assert preset.set(SETTINGS.raytracing).samples_per_pixel == 8


def test_a_source_must_be_a_raytracing_configuration():
    with pytest.raises(AlganConfigurationError, match="got dict"):
        _preset().set({"samples_per_pixel": 4})


def test_to_dict_and_as_preset_hand_out_independent_copies():
    preset = _preset()
    values = preset.to_dict()
    values["samples_per_pixel"] = -100

    assert preset.samples_per_pixel != -100
    copy = preset.as_preset()
    assert copy is not preset
    assert copy.to_dict() == preset.to_dict()


def test_applying_a_preset_writes_its_values_to_the_live_settings():
    preset = _preset().set(samples_per_pixel=7, shadows=False)
    SETTINGS.raytracing.set(preset)

    assert SETTINGS.raytracing.samples_per_pixel == 7
    assert SETTINGS.raytracing.shadows is False


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("samples_per_pixel", -1, "at least 1"),
        ("samples_per_pixel", 2.5, "must be an integer"),
        ("samples_per_pixel", "lots", "must be an integer"),
        ("shadows", "yes", "True or False"),
    ],
)
def test_set_checks_a_value_where_it_is_written(field, value, message):
    """REGRESSION. ``preset.set`` checked only the field's name, so
    ``samples_per_pixel=-1`` made a preset that failed later, wherever it
    happened to be applied, while ``SETTINGS.raytracing.set`` -- and every
    video preset's ``set`` -- refuses the same value on the line that wrote it.
    """
    with pytest.raises(AlganConfigurationError, match=message):
        _preset().set(**{field: value})


def test_a_field_fixed_at_import_cannot_be_changed_through_a_preset():
    """REGRESSION. ``SETTINGS.raytracing.set(bvh_arity=8)`` is refused, since
    the kernels' layout is derived from it at import; but a preset accepted
    the write, and applying that preset -- which restores every field it
    carries -- put the new value in the renderer's storage, leaving host and
    compiled kernels disagreeing.
    """
    field = "bvh_arity"
    assert field in _IMPORT_FROZEN_FIELDS
    live = getattr(SETTINGS.raytracing, field)

    with pytest.raises(AlganConfigurationError, match="fixed when algan is imported"):
        _preset().set(**{field: live * 2})
    assert getattr(SETTINGS.raytracing, field) == live


def test_a_snapshot_round_trips_through_a_derived_preset():
    """Validating at ``set`` must not refuse what a snapshot already holds:
    deriving from a capture re-applies every field, frozen ones included.
    """
    preset = _preset()
    SETTINGS.raytracing.set(max_bounces=preset.max_bounces + 1)
    SETTINGS.raytracing.set(preset.set(preset))

    assert SETTINGS.raytracing.to_dict() == preset.to_dict()


def test_the_experimental_view_lists_and_scopes_only_its_switches():
    experimental = SETTINGS.raytracing.experimental
    switches = experimental.to_dict()

    assert switches
    assert set(switches) == set(dir(experimental))
    assert set(switches).isdisjoint(SETTINGS.raytracing.public_field_names())
    assert repr(experimental).startswith("RayTracingSettings.experimental(")
    assert all(name in repr(experimental) for name in switches)

    field = "area_light_soft_shadows"
    before = getattr(experimental, field)
    with experimental.override(**{field: not before}):
        assert getattr(SETTINGS.raytracing, field) is (not before)
    assert getattr(SETTINGS.raytracing, field) is before
