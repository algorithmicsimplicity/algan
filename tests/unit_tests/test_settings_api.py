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
    ComputingSettings,
    RayTracingSettings,
    VideoSettings,
)
from algan.settings._startup import render_device

# In the fast suite: engine modules read ``SETTINGS`` live off section objects
# they captured at import, so a change to how a section is written or restored
# reaches every subsystem at once.
pytestmark = pytest.mark.fast


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


@pytest.mark.parametrize(
    ("section", "field", "bad"),
    [
        ("video", "frames_per_second", 0),
        ("style", "buffer", -1),
        ("computing", "max_animation_batch_size", -5),
        ("paths", "output_directory", 123),
    ],
)
def test_assigning_a_field_validates_exactly_as_set_does(section, field, bad):
    """The short spelling must not be the one that skips the checks.

    ``SETTINGS.video.frames_per_second = 0`` used to store the zero and fail
    much later, somewhere inside a render, while
    ``SETTINGS.video.set(frames_per_second=0)`` refused it on the spot.

    Every dataclass section is listed, deliberately: the routing stands down
    until each declared field is present in ``__dict__``, so a section that
    stopped storing its fields there would quietly revert to unvalidated
    assignment. That is a silent regression, and this is what catches it.
    """
    settings_section = getattr(SETTINGS, section)
    before = getattr(settings_section, field)

    with pytest.raises(AlganConfigurationError):
        setattr(settings_section, field, bad)

    assert getattr(settings_section, field) == before, (
        "a refused assignment must leave the field exactly as it was"
    )


def test_assigning_a_field_normalizes_it_the_way_set_does():
    """``__post_init__`` coercions have to run on assignment too."""
    SETTINGS.style.buffer = 1
    assert isinstance(SETTINGS.style.buffer, float)
    assert SETTINGS.style.buffer == 1.0


def test_a_write_leaves_the_fields_it_did_not_change_alone():
    """Identity, not just equality: callers hold these objects.

    ``set`` used to write a deepcopy of *every* field, so changing ``buffer``
    swapped ``background_color`` for an equal stranger and anything holding the
    old Color stopped tracking the setting.
    """
    background = SETTINGS.style.background_color
    text = SETTINGS.style.text_color

    SETTINGS.style.set(buffer=0.7)
    assert SETTINGS.style.background_color is background
    assert SETTINGS.style.text_color is text

    SETTINGS.style.buffer = 0.8
    assert SETTINGS.style.background_color is background
    assert SETTINGS.style.text_color is text


def test_a_changed_field_is_still_applied_and_still_copied():
    """The narrowing must not skip a real change, or alias the caller's object.

    Copying is what stops a caller mutating their own value afterwards and
    silently reconfiguring the renderer.
    """
    original = SETTINGS.style.background_color
    replacement = original.clone()

    SETTINGS.style.set(background_color=replacement)

    assert SETTINGS.style.background_color is not original, "the change was skipped"
    assert SETTINGS.style.background_color is not replacement, "aliased the argument"
    assert bool((SETTINGS.style.background_color == replacement).all())


def test_a_section_can_still_be_constructed_directly():
    """Construction assigns fields one at a time, before ``set`` could work.

    The routing has to stand down until every declared field exists, or
    building a section from scratch would try to replace a half-built object.
    """
    section = ComputingSettings(max_animation_batch_size=42)
    assert section.max_animation_batch_size == 42
    # And it is a real, independent section, not a view on the live one.
    assert section is not SETTINGS.computing
    assert SETTINGS.computing.max_animation_batch_size != 42

    section.max_animation_batch_size = 43
    assert section.max_animation_batch_size == 43
    with pytest.raises(AlganConfigurationError):
        ComputingSettings(max_animation_batch_size=0)


def test_available_memory_override_rejects_values_that_cannot_size_an_arena():
    for value in (0, -1, 2.5, True):
        with pytest.raises(AlganConfigurationError, match="available_memory_override"):
            SETTINGS.computing.set(available_memory_override=value)


def test_the_animation_device_answers_with_the_environment_variable_to_set():
    """It is still initialization-only, and says so instead of "unknown".

    Every Mob's authoring state is allocated on it from the first ``Square()``
    onward, so there is no moment at which changing it would mean anything.
    """
    with pytest.raises(AlganConfigurationError, match="ALGAN_ANIMATION_DEVICE"):
        SETTINGS.computing.set(animation_device="cpu")


def test_render_on_cpu_points_at_the_field_that_replaced_it():
    with pytest.raises(AlganConfigurationError, match="render_device"):
        SETTINGS.computing.set(render_on_cpu=True)


def test_the_render_device_is_settable_and_normalizes_to_a_torch_device():
    original = SETTINGS.computing.render_device
    try:
        SETTINGS.computing.set(render_device="cpu")
        assert SETTINGS.computing.render_device == torch.device("cpu")
        assert isinstance(SETTINGS.computing.render_device, torch.device)
        # Direct assignment goes through the same validation as ``set``, which
        # it does not for any other field: an unvalidated device renders on the
        # wrong hardware rather than merely holding a silly number.
        SETTINGS.computing.render_device = torch.device("cpu")
        assert SETTINGS.computing.render_device == torch.device("cpu")
        # ``render_device`` is what the engine reads, never a bound copy.
        assert render_device() == SETTINGS.computing.render_device
    finally:
        SETTINGS.computing.set(render_device=original)


def test_the_render_device_rejects_what_it_cannot_render_on():
    with pytest.raises(AlganConfigurationError, match="render_device"):
        SETTINGS.computing.set(render_device="not-a-device")
    if not torch.cuda.is_available():
        with pytest.raises(AlganConfigurationError, match="CUDA"):
            SETTINGS.computing.set(render_device="cuda")


def test_the_render_device_cannot_change_mid_render():
    """The one path that could corrupt rather than merely be slow.

    Batch prep launches kernels from a worker thread, so a change while a job is
    running could have that thread re-initialize Taichi -- dropping every
    compiled kernel -- while the render thread is inside one.
    """
    from algan.rendering.taichi_runtime import render_job_holding_the_arch

    original = SETTINGS.computing.render_device
    other = "cpu" if original.type != "cpu" else "meta"
    try:
        with (
            render_job_holding_the_arch(),
            pytest.raises(AlganConfigurationError, match="render is in progress"),
        ):
            SETTINGS.computing.set(render_device=other)
        # And the counter unwinds, so the next render can still change it.
        SETTINGS.computing.set(render_device=original)
    finally:
        SETTINGS.computing.set(render_device=original)


def test_a_wide_attribute_freezes_the_render_device():
    """A texture's frame window is placed when the Mob is created.

    Nothing downstream re-asks, so the device must not move under it. The pin
    only arms when the wide attribute actually lands on the render device --
    on a CPU render device it does not, and there is nothing to protect.
    """
    from algan.animation_timeline.timeline import (
        WIDE_ATTR_MIN_CHANNELS,
        AttributeTimeline,
        clear_wide_attribute_device_pin,
        wide_attribute_device_pin,
    )

    original = SETTINGS.computing.render_device
    try:
        clear_wide_attribute_device_pin()
        AttributeTimeline(WIDE_ATTR_MIN_CHANNELS, buffer_size=2)
        pin = wide_attribute_device_pin()
        if pin is None:
            # CPU render device: no wide attribute is placed on it, so the
            # device stays free to change.
            SETTINGS.computing.set(render_device="cpu")
            return
        with pytest.raises(AlganConfigurationError, match="wide attribute"):
            SETTINGS.computing.set(render_device="cpu")
    finally:
        clear_wide_attribute_device_pin()
        SETTINGS.computing.set(render_device=original)


def test_resetting_the_scenes_releases_the_wide_attribute_pin():
    """The pin dies with the timelines holding it, which is what reset kills."""
    from algan.animation_timeline.timeline import (
        WIDE_ATTR_MIN_CHANNELS,
        AttributeTimeline,
        clear_wide_attribute_device_pin,
        wide_attribute_device_pin,
    )
    from algan.scene_manager import SceneManager

    try:
        clear_wide_attribute_device_pin()
        AttributeTimeline(WIDE_ATTR_MIN_CHANNELS, buffer_size=2)
        SceneManager.reset()
        assert wide_attribute_device_pin() is None
    finally:
        clear_wide_attribute_device_pin()


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
