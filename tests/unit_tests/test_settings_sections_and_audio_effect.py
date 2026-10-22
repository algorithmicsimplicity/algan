"""Fast tests for the settings root, style section, renderer registry and AudioEffect."""

from __future__ import annotations

import math

import pytest
import torch

from algan.errors import AlganConfigurationError
from algan.settings.renderer_settings import (
    RENDERER_REGISTRY,
    RENDERER_SETTINGS,
    RendererRegistry,
    effective_triangle_primitive,
)
from algan.settings.root_settings import AlganSettings, SettingsSnapshot
from algan.settings.style_settings import (
    BORDER_PLACEMENTS,
    SHAPE_STYLE_PROFILES,
    StyleSettings,
)
from algan.sound.audio_effect import AudioEffect, AudioManager


def test_root_sections_have_stable_identity():
    settings = AlganSettings()
    with pytest.raises(AlganConfigurationError):
        settings.style = StyleSettings()


def test_root_rejects_unknown_attribute():
    settings = AlganSettings()
    with pytest.raises(AttributeError):
        settings.nonexistent = 1


def test_root_skip_save_frame_is_assignable_and_hidden():
    settings = AlganSettings()
    settings._skip_save_frame = True
    assert settings._skip_save_frame is True
    assert "_skip_save_frame" not in dir(settings)
    assert "_skip_save_frame" not in repr(settings)


def test_root_dir_lists_sections_and_methods():
    names = dir(AlganSettings())
    for expected in ("computing", "paths", "style", "video", "raytracing"):
        assert expected in names
    assert "override" in names


def test_snapshot_is_detached_from_later_changes():
    settings = AlganSettings()
    snap = settings.snapshot()
    assert isinstance(snap, SettingsSnapshot)
    settings.style.set(buffer=1.5)
    assert snap.style.buffer == pytest.approx(0.6)


def test_restore_puts_values_back_in_place():
    settings = AlganSettings()
    style = settings.style
    snap = settings.snapshot()
    settings.style.set(buffer=2.0)
    result = settings.restore(snap)
    assert result is settings
    assert settings.style is style
    assert style.buffer == pytest.approx(0.6)


def test_restore_rejects_non_snapshot():
    settings = AlganSettings()
    with pytest.raises(AlganConfigurationError):
        settings.restore({"style": {}})


def test_override_applies_and_restores():
    settings = AlganSettings()
    with settings.override(style={"buffer": 3.0}) as yielded:
        assert yielded is settings
        assert settings.style.buffer == pytest.approx(3.0)
    assert settings.style.buffer == pytest.approx(0.6)


def test_override_restores_after_exception():
    settings = AlganSettings()
    with pytest.raises(RuntimeError):
        _raise_inside_override(settings)
    assert settings.style.buffer == pytest.approx(0.6)


def _raise_inside_override(settings):
    with settings.override(style={"buffer": 4.0}):
        raise RuntimeError("boom")


def test_override_unknown_section():
    settings = AlganSettings()
    with pytest.raises(AlganConfigurationError, match="Unknown settings section"):
        _enter(settings.override(bogus={}))


def test_override_non_dict_section_value_restores():
    settings = AlganSettings()
    with pytest.raises(AlganConfigurationError, match="requires a dict"):
        _enter(settings.override(style={"buffer": 5.0}, video=3))
    assert settings.style.buffer == pytest.approx(0.6)


def _enter(cm):
    with cm:
        pass


def test_global_settings_override_is_restored():
    from algan.settings import SETTINGS

    before = SETTINGS.style.buffer
    try:
        with SETTINGS.override(style={"buffer": before + 1.0}):
            assert SETTINGS.style.buffer == pytest.approx(before + 1.0)
    finally:
        SETTINGS.style.set(buffer=before)
    assert SETTINGS.style.buffer == before


def test_style_defaults():
    style = StyleSettings()
    assert style.buffer == pytest.approx(0.6)
    assert style.fade_out_on_scene_end is False
    assert style.default_material is None
    assert style.shape_style_profile == "algan"
    assert style.border_placement == "inward"
    assert style.manim_stroke_width_ratio == pytest.approx(2.0)


def test_style_default_colors_are_independent():
    a = StyleSettings()
    b = StyleSettings()
    assert a.background is not b.background


def test_style_buffer_coerced_to_float():
    style = StyleSettings(buffer=1)
    assert isinstance(style.buffer, float)


@pytest.mark.parametrize("bad", [-0.1, math.inf, math.nan, "abc", None])
def test_style_rejects_bad_buffer(bad):
    with pytest.raises(AlganConfigurationError):
        StyleSettings(buffer=bad)


@pytest.mark.parametrize("bad", [0, -1.0, math.inf, math.nan, "x"])
def test_style_rejects_bad_stroke_ratio(bad):
    with pytest.raises(AlganConfigurationError):
        StyleSettings(manim_stroke_width_ratio=bad)


def test_style_rejects_non_bool_fade():
    with pytest.raises(AlganConfigurationError):
        StyleSettings(fade_out_on_scene_end=1)


def test_style_rejects_bad_border_placement():
    with pytest.raises(AlganConfigurationError, match="border_placement"):
        StyleSettings(border_placement="outward")


def test_style_rejects_bad_profile():
    with pytest.raises(AlganConfigurationError, match="shape_style_profile"):
        StyleSettings(shape_style_profile="blender")


def test_style_rejects_material_without_shader():
    with pytest.raises(AlganConfigurationError, match="default_material"):
        StyleSettings(default_material=object())


def test_style_accepts_duck_typed_material():
    class FakeMaterial:
        shader = None

    material = FakeMaterial()
    assert StyleSettings(default_material=material).default_material is material


@pytest.mark.parametrize("placement", BORDER_PLACEMENTS)
def test_style_accepts_each_border_placement(placement):
    assert StyleSettings(border_placement=placement).border_placement == placement


def test_style_profiles_constant():
    assert SHAPE_STYLE_PROFILES == ("algan", "manim")


def test_style_set_validates():
    style = StyleSettings()
    with pytest.raises(AlganConfigurationError):
        style.set(buffer=-1)
    assert style.buffer == pytest.approx(0.6)


def test_renderer_registry_alias_and_effective_primitive():
    assert RENDERER_SETTINGS is RENDERER_REGISTRY
    assert effective_triangle_primitive() is RENDERER_REGISTRY.triangle_primitive


def test_fresh_renderer_registry_matches_defaults():
    registry = RendererRegistry()
    assert registry.triangle_primitive is RENDERER_REGISTRY.triangle_primitive
    assert (
        registry.bezier_circuit_primitive is RENDERER_REGISTRY.bezier_circuit_primitive
    )


def test_audio_effect_stores_clip_and_start_func():
    clip = torch.zeros(2, 8)

    def start():
        return 1.5

    effect = AudioEffect(clip, start)
    assert effect.audio_clip is clip
    assert effect.start_time_func() == pytest.approx(1.5)


def test_audio_manager_script_and_chaining():
    manager = AudioManager()
    assert manager.append_script("  hello ") is manager
    assert manager.video_transcript == "hello\n\n"


def test_audio_manager_custom_speech_source():
    tone = torch.linspace(0, 1, 16)
    manager = AudioManager().set_speech_source(lambda script: (script, tone))
    script, audio = manager.get_speech("hi")
    assert script == "hi"
    assert torch.equal(audio, tone)
