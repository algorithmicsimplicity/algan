"""Contracts of the one process-global ``SETTINGS`` object.

Engine modules read settings live off the section objects, so the section
identities have to stay stable and every write has to be validated at the point
of the write.  A silently-accepted typo here does not raise -- it renders the
wrong thing, which is exactly the failure mode these tests exist to prevent.
"""

from __future__ import annotations

import pytest
import torch

from algan import (
    HD,
    PREVIEW,
    SETTINGS,
    AlganConfigurationError,
    RayTracingSettings,
    VideoSettings,
)


@pytest.fixture(autouse=True)
def restore_settings():
    snapshot = SETTINGS.snapshot()
    yield
    SETTINGS.restore(snapshot)


# --------------------------------------------------------------------------
# Section identity
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "section", ["computing", "paths", "style", "video", "raytracing"]
)
def test_sections_have_stable_identity_and_reject_replacement(section):
    original = getattr(SETTINGS, section)
    with pytest.raises(AlganConfigurationError, match="stable identity"):
        setattr(SETTINGS, section, original)
    assert getattr(SETTINGS, section) is original


def test_unknown_root_attribute_is_an_attribute_error():
    with pytest.raises(AttributeError):
        SETTINGS.rendering  # noqa: B018


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def test_a_mistyped_setting_name_is_rejected_rather_than_stored():
    with pytest.raises(AlganConfigurationError):
        SETTINGS.video.set(frames_per_secondd=30)
    with pytest.raises(AlganConfigurationError):
        SETTINGS.raytracing.set(max_bounce=3)


def test_out_of_range_values_are_rejected():
    with pytest.raises(AlganConfigurationError):
        SETTINGS.video.set(frames_per_second=0)
    with pytest.raises(AlganConfigurationError):
        SETTINGS.computing.set(animation_memory_fraction=0.0)
    with pytest.raises(AlganConfigurationError):
        SETTINGS.style.set(buffer=-1)


def test_available_memory_override_replaces_the_measured_device_figure():
    """The knob that makes a render's frame-window split reproducible.

    Free device memory moves with allocator warmth, and the render loop sizes
    its frame windows from it, so an unpinned measurement makes the same scene
    split -- and therefore render -- differently from run to run.
    """
    from algan.utils.memory_utils import get_num_available_bytes

    assert SETTINGS.computing.available_memory_override is None
    with SETTINGS.computing.override(available_memory_override=1234567):
        assert get_num_available_bytes(torch.device("cuda")) == 1234567
        assert get_num_available_bytes(torch.device("mps")) == 1234567
        # The CPU branch already returns a setting, so it stays put.
        assert (
            get_num_available_bytes(torch.device("cpu"))
            == SETTINGS.computing.max_cpu_memory_used
        )
    assert SETTINGS.computing.available_memory_override is None


def test_available_memory_override_rejects_values_that_cannot_size_an_arena():
    for value in (0, -1, 2.5, True):
        with pytest.raises(AlganConfigurationError, match="available_memory_override"):
            SETTINGS.computing.set(available_memory_override=value)


def test_device_selection_answers_with_the_environment_variable_to_set():
    for name, variable in (
        ("render_device", "ALGAN_RENDER_DEVICE"),
        ("animation_device", "ALGAN_ANIMATION_DEVICE"),
    ):
        with pytest.raises(AlganConfigurationError, match=variable):
            SETTINGS.computing.set(**{name: "cpu"})


# --------------------------------------------------------------------------
# Presets
# --------------------------------------------------------------------------
def test_presets_are_immutable_and_set_returns_a_copy():
    assert HD.is_preset
    derived = HD.set(frames_per_second=60)
    assert derived is not HD
    assert derived.frames_per_second == 60
    assert HD.frames_per_second == 30
    with (
        pytest.raises(AlganConfigurationError, match="preset"),
        HD.override(frames_per_second=60),
    ):
        pass


def test_applying_a_preset_copies_its_values_into_the_live_section():
    SETTINGS.video.set(PREVIEW)
    assert SETTINGS.video.resolution == PREVIEW.resolution
    assert SETTINGS.video.frames_per_second == PREVIEW.frames_per_second
    # The live section is a copy, not the preset itself.
    assert SETTINGS.video is not PREVIEW
    assert not SETTINGS.video.is_preset


def test_set_rejects_a_settings_object_of_the_wrong_section():
    with pytest.raises(AlganConfigurationError, match="expected another"):
        SETTINGS.video.set(RayTracingSettings())
    with pytest.raises(AlganConfigurationError, match="expected"):
        SETTINGS.raytracing.set(VideoSettings((16, 16), 2))


# --------------------------------------------------------------------------
# Experimental switches
# --------------------------------------------------------------------------
def test_experimental_switches_are_not_settable_on_the_parent_section():
    with pytest.raises(AlganConfigurationError) as excinfo:
        SETTINGS.raytracing.set(hybrid_raster=True)
    message = str(excinfo.value)
    assert "experimental" in message
    assert "SETTINGS.raytracing.experimental.set" in message


def test_experimental_switches_round_trip_through_the_experimental_view():
    before = SETTINGS.raytracing.experimental.bvh_refit
    SETTINGS.raytracing.experimental.set(bvh_refit=not before)
    assert SETTINGS.raytracing.experimental.bvh_refit is (not before)
    SETTINGS.raytracing.experimental.set(bvh_refit=before)
    assert SETTINGS.raytracing.experimental.bvh_refit is before


def test_the_public_and_experimental_namespaces_do_not_overlap():
    public = {name for name in dir(SETTINGS.raytracing) if not name.startswith("_")}
    experimental = set(dir(SETTINGS.raytracing.experimental))
    assert experimental, "the experimental namespace should not be empty"
    assert public.isdisjoint(experimental)
    # ``shadows`` and ``samples_per_pixel`` describe what the renderer produces
    # and must stay on the public section.
    assert {"shadows", "samples_per_pixel", "max_bounces"} <= public


def test_a_write_reaches_the_module_globals_the_engine_reads_live():
    from algan.rendering.raytracing import settings as rt_settings

    before = rt_settings.MAX_BOUNCES
    SETTINGS.raytracing.set(max_bounces=before + 1)
    assert before + 1 == rt_settings.MAX_BOUNCES


# --------------------------------------------------------------------------
# Snapshot / restore / override
# --------------------------------------------------------------------------
def test_snapshot_and_restore_round_trip_every_section():
    snapshot = SETTINGS.snapshot()
    SETTINGS.video.set(frames_per_second=7)
    SETTINGS.raytracing.set(max_bounces=2, shadows=True)
    SETTINGS.style.set(buffer=1.75)
    SETTINGS.computing.set(max_animation_batch_size=123)

    SETTINGS.restore(snapshot)

    assert SETTINGS.video.frames_per_second == snapshot.video.frames_per_second
    assert SETTINGS.raytracing.max_bounces == snapshot.raytracing.max_bounces
    assert SETTINGS.raytracing.shadows == snapshot.raytracing.shadows
    assert SETTINGS.style.buffer == snapshot.style.buffer
    assert (
        SETTINGS.computing.max_animation_batch_size
        == snapshot.computing.max_animation_batch_size
    )


def test_restore_keeps_section_identity_so_live_readers_still_see_updates():
    sections = {
        name: getattr(SETTINGS, name)
        for name in ("computing", "paths", "style", "video", "raytracing")
    }
    SETTINGS.restore(SETTINGS.snapshot())
    for name, section in sections.items():
        assert getattr(SETTINGS, name) is section


def _raise_inside_override(target):
    with SETTINGS.video.override(frames_per_second=target):
        assert SETTINGS.video.frames_per_second == target
        raise RuntimeError("boom")


def test_override_restores_even_when_the_body_raises():
    before = SETTINGS.video.frames_per_second
    with pytest.raises(RuntimeError):
        _raise_inside_override(before + 5)
    assert SETTINGS.video.frames_per_second == before
