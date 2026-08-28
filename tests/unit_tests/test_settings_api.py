"""Contracts of the one process-global ``SETTINGS`` object.

Engine modules read settings live off the section objects, so the section
identities have to stay stable and every write has to be validated at the point
of the write.  A silently-accepted typo here does not raise -- it renders the
wrong thing, which is exactly the failure mode these tests exist to prevent.
"""

from __future__ import annotations

import os
import subprocess
import sys

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


# --------------------------------------------------------------------------
# Ray-tracing values. Every field here writes through to a module-level global
# the kernels read, and the ones without a setter used to be written raw -- so
# a wrong type or a negative count arrived as a Taichi failure or a wrong image
# with nothing naming the setting behind it.
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("max_bounces", -1),  # a count cannot be negative
        ("max_bounces", "x"),  # nor a string
        ("max_bounces", True),  # nor a bool, which is an int subclass
        ("samples_per_pixel", 0),  # silently clamped to 1 before
        ("shadows", 1),  # a flag is True or False, not 1
        ("tonemap_exposure", "bright"),
        ("tonemap_method", "sepia"),  # its setter's own rejection
    ],
)
def test_a_bad_raytracing_value_is_refused_and_changes_nothing(field, bad):
    before = getattr(SETTINGS.raytracing, field)
    with pytest.raises(AlganConfigurationError) as excinfo:
        SETTINGS.raytracing.set(**{field: bad})
    assert field in str(excinfo.value), "the message must name the setting"
    assert getattr(SETTINGS.raytracing, field) == before


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("wavefront_tile_rays", 0),  # rays per tile; 0 is degenerate
        ("merge_gpu_peak_factor", 0),  # a multiplier the memory model scales by
        ("analytic_aa_chord_tolerance", 0),  # 0 asks for infinite subdivision
        ("shadow_eps_relative", float("nan")),  # NaN reaches a kernel silently
        ("analytic_aa_sliver", "nonsense"),  # an enumerated mode
        ("bvh_refit", "yes"),  # a flag, written raw before
    ],
)
def test_a_bad_experimental_value_is_refused_too(field, bad):
    before = getattr(SETTINGS.raytracing, field)
    with pytest.raises(AlganConfigurationError):
        SETTINGS.raytracing.experimental.set(**{field: bad})
    assert getattr(SETTINGS.raytracing, field) == before


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("max_bounces", 0, 0),  # documented: 0 means no bounces
        ("max_bounces", 4, 4),
        ("shadows", True, True),
        ("tonemap_exposure", 2, 2.0),  # an int for a float field normalizes
        ("tonemap_method", "agx", "agx"),
        ("indirect_bounce_strength", 0, 0.0),
    ],
)
def test_legitimate_raytracing_values_still_go_through(field, value, expected):
    SETTINGS.raytracing.set(**{field: value})
    stored = getattr(SETTINGS.raytracing, field)
    assert stored == expected
    assert type(stored) is type(expected)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("wf_gen_fused", True),  # bool forces the mode
        ("wf_gen_fused", "auto"),  # str selects the adaptive one
        ("shadow_terminator", "relax"),  # bool default, third state is a str
    ],
)
def test_a_mode_switch_may_be_spelled_as_a_bool_or_a_string(field, value):
    """The three fields the derived type check has to stand aside for.

    Their type is inferred from the value they ship with, which for these is
    only one of the two spellings their setter accepts.
    """
    SETTINGS.raytracing.experimental.set(**{field: value})


def test_an_unsupported_feature_is_not_flattened_into_a_configuration_error():
    """``UnsupportedFeatureError`` is a distinct type callers catch.

    It is also a subclass of ``AlganConfigurationError``, so the conversion of
    a setter's bare ``ValueError`` has to let Algan's own errors past first.
    """
    from algan.errors import UnsupportedFeatureError

    with pytest.raises(UnsupportedFeatureError):
        SETTINGS.raytracing.experimental.set(wf_textured=True)


def test_the_very_first_write_in_a_process_is_validated_too():
    """The checks must not depend on something having read a setting first.

    The accepted types are derived from the shipped defaults, and that table is
    populated as a side effect of resolving the legacy module. An earlier draft
    validated before resolving it, so the table was empty on the first write in
    a process and every value went through unchecked -- invisible to any test
    that read a field first, which is most of them. Run in a subprocess because
    by the time this module is imported the table is long since populated.
    """
    probe = """
import algan.settings.raytracing_settings as rs
from algan import SETTINGS
from algan.errors import AlganConfigurationError

assert not rs._DEFAULT_TYPES, "something resolved the module before the write"
try:
    SETTINGS.raytracing.set(max_bounces="x")
except AlganConfigurationError:
    print("VALIDATED")
"""
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        env=dict(os.environ, ALGAN_USE_DAEMON="0"),
    )
    assert result.returncode == 0, result.stderr
    assert "VALIDATED" in result.stdout, result.stdout


def test_a_captured_raytracing_configuration_still_round_trips():
    """Validation must not reject a snapshot of the live values.

    ``set(source=...)`` replays all 106 fields, experimental and inert
    included, so a rule that is too strict for a value the renderer itself
    produced would break ``SETTINGS.restore``.
    """
    preset = SETTINGS.raytracing.as_preset()
    SETTINGS.raytracing.set(max_bounces=3)
    SETTINGS.raytracing.set(preset)
    assert SETTINGS.raytracing.max_bounces == preset.to_dict()["max_bounces"]


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

    before = rt_settings.max_bounces
    SETTINGS.raytracing.set(max_bounces=before + 1)
    assert before + 1 == rt_settings.max_bounces


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


# --------------------------------------------------------------------------
# Coverage of the storage module
# --------------------------------------------------------------------------


def test_every_renderer_switch_is_reachable_through_SETTINGS():
    """No renderer configuration without a way to configure it.

    ``algan/rendering/raytracing/settings.py`` stores the renderer's
    configuration: module-level values with environment-variable defaults that
    engine code reads live. ``SETTINGS.raytracing`` (and ``.experimental``) is
    the only public way to write one.

    This used to be a real gap. The two layers spelled each field twice --
    ``hybrid_raster`` on the section, ``hybrid_raster`` in the module -- and a
    hand-maintained 119-row table joined them, so a switch whose row nobody
    added had a global, a setter, and no way to set it. Nine had accumulated.
    There is one spelling now and the field set is derived from the module, so
    the gap cannot reopen by omission; what this pins is that the derivation
    still sees every declaration, which a non-scalar default or a shadowing
    helper would break.
    """
    import ast
    import inspect

    from algan.rendering.raytracing import settings as storage

    declared = set()
    for node in ast.parse(inspect.getsource(storage)).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            name = target.id if isinstance(target, ast.Name) else None
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
        else:
            continue
        if name and not name.startswith("_") and name == name.lower():
            declared.add(name)

    unreachable = sorted(declared - set(SETTINGS.raytracing.field_names()))
    assert not unreachable, (
        "declared in the storage module but not reachable through SETTINGS -- "
        "either the default is not a scalar, or a helper of the same name "
        "shadows it:\n  " + "\n  ".join(unreachable)
    )


def test_no_renderer_switch_is_shadowed_by_a_helper_of_the_same_name():
    """A field and a function cannot share a name, and Python will not say so.

    With one spelling for each setting, a ``def`` later in the storage module
    than its field simply takes the name over: the field stops existing, drops
    out of ``SETTINGS`` silently, and any accessor that read it becomes a
    ``return`` of itself. Three did that the moment the two spellings merged.
    Nothing about it raises, so it needs asserting.
    """
    from algan.rendering.raytracing import settings as storage
    from algan.settings.raytracing_settings import _shadowed_fields

    shadowed = _shadowed_fields(storage)
    assert not shadowed, (
        "these renderer settings are declared as fields but the name is bound "
        "to a function by the time the module finishes -- rename the function "
        "(the field owns the name):\n  " + "\n  ".join(shadowed)
    )
