"""``SETTINGS.raytracing`` refuses the two settings no renderer reads.

``light_intensity`` and ``ambient_light`` are consumed only by
``raytrace_kernels_taichi.path_trace_physical_stbvh``, which ``tracer`` never
launches -- the deterministic path uses the wavefront tracer and
``samples_per_pixel > 1`` uses ``path_trace_scene_stbvh``. Setting either used
to be accepted and then do nothing at all, silently, which is the failure mode
worth a test: the guard has to keep *reads* working (engine code binds the
settings object and reads fields off it on the hot path) and has to keep a
snapshot restore round-tripping, while refusing a deliberate write.

If a future change wires the physical kernel in, delete the guard and this
file together -- the settings become real again.
"""

from __future__ import annotations

import math

import pytest

from algan import SETTINGS
from algan.errors import AlganConfigurationError

INERT = ("light_intensity", "ambient_light")


@pytest.fixture(autouse=True)
def _restore_settings():
    snapshot = SETTINGS.snapshot()
    yield
    SETTINGS.restore(snapshot)


@pytest.mark.parametrize("field", INERT)
def test_reading_an_inert_setting_still_works(field):
    value = getattr(SETTINGS.raytracing, field)
    assert isinstance(value, float)


def test_the_documented_defaults_are_unchanged():
    assert SETTINGS.raytracing.light_intensity == pytest.approx(math.pi)
    assert SETTINGS.raytracing.ambient_light == 0.0


@pytest.mark.parametrize("field", INERT)
def test_every_write_path_refuses(field):
    rt = SETTINGS.raytracing
    for write in (
        lambda: rt.set(**{field: 0.5}),
        lambda: setattr(rt, field, 0.5),
        lambda: getattr(rt, f"set_{field}")(0.5),
        lambda: rt.experimental.set(**{field: 0.5}),
    ):
        with pytest.raises(AlganConfigurationError) as excinfo:
            write()
        # The message has to point somewhere useful, not just say no.
        assert field in str(excinfo.value)
        assert "intensity=" in str(excinfo.value) or "AmbientLight" in str(
            excinfo.value
        )


def test_a_refused_write_leaves_the_value_alone():
    before = SETTINGS.raytracing.light_intensity
    with pytest.raises(AlganConfigurationError):
        SETTINGS.raytracing.set(light_intensity=99.0)
    assert SETTINGS.raytracing.light_intensity == before


def test_restoring_a_captured_configuration_still_round_trips():
    """A snapshot carries every field, inert ones included. Restoring one is
    not a request to tune anything, so it must not hit the guard.
    """
    preset = SETTINGS.raytracing.as_preset()
    SETTINGS.raytracing.set(shadows=not SETTINGS.raytracing.shadows)
    SETTINGS.raytracing.set(preset)
    assert SETTINGS.raytracing.light_intensity == pytest.approx(math.pi)


def test_an_ordinary_setting_is_unaffected():
    SETTINGS.raytracing.set(max_bounces=5)
    assert SETTINGS.raytracing.max_bounces == 5
