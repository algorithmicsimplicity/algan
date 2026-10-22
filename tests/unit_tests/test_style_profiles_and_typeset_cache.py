"""Fast pure-Python tests for the Manim shape-style profile and typeset cache helpers."""

from __future__ import annotations

import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from algan.settings import SETTINGS
from algan.settings import shape_style_profiles as ssp
from algan.utils import typeset_cache as tc

pytestmark = pytest.mark.fast


class _FakeColor:
    def __init__(self, rgba):
        self._rgba = rgba

    def to_rgba(self):
        return self._rgba


class _FakeShape:
    def __init__(self, *args):
        self.args = args
        self.fill_color = _FakeColor((1.0, 0.0, 0.0, 1.0))
        self.fill_opacity = 0.5
        self.stroke_color = _FakeColor((0.0, 1.0, 0.0, 1.0))
        self.stroke_opacity = 1.0
        self.stroke_width = 4


class _UnfilledShape(_FakeShape):
    def __init__(self, *args):
        super().__init__(*args)
        self.fill_opacity = 0.0


class _BrokenShape:
    def __init__(self, *args):
        raise RuntimeError("boom")


def _fake_manim(**classes):
    return SimpleNamespace(
        LEFT="L", RIGHT="R", UP="U", __name__="fake_manim", **classes
    )


@pytest.fixture
def fresh_styles(monkeypatch):
    monkeypatch.setattr(ssp, "_SHAPE_STYLES", None)
    return monkeypatch


# --- shape_style_profiles ---------------------------------------------------


def test_mapping_tables_are_consistent():
    assert set(ssp._MANIM_SHAPE_CONSTRUCTION_ARGS) <= set(
        ssp._MANIM_SHAPE_STYLE_CLASSES
    )


def test_read_opacity_takes_max_of_tensor_like():
    assert ssp._read_opacity([0.2, 0.7, 0.1]) == pytest.approx(0.7)


def test_read_opacity_scalar():
    assert ssp._read_opacity(0.25) == pytest.approx(0.25)


def test_resolve_missing_class_returns_none(fresh_styles):
    fresh_styles.setattr(ssp, "_manim", _fake_manim())
    assert ssp._resolve_shape_style("Square", "Square") is None


def test_resolve_constructor_failure_returns_none(fresh_styles):
    fresh_styles.setattr(ssp, "_manim", _fake_manim(Square=_BrokenShape))
    assert ssp._resolve_shape_style("Square", "Square") is None


def test_resolve_snapshot_values(fresh_styles):
    fresh_styles.setattr(ssp, "_manim", _fake_manim(Square=_FakeShape))
    style = ssp._resolve_shape_style("Square", "Square")
    assert style is not None
    assert style["filled"] is True
    assert style["fill_opacity"] == pytest.approx(0.5)
    assert style["stroke_width_manim"] == pytest.approx(4.0)
    assert style["checker_color"] is None


def test_resolve_unfilled_shape(fresh_styles):
    fresh_styles.setattr(ssp, "_manim", _fake_manim(Circle=_UnfilledShape))
    style = ssp._resolve_shape_style("Circle", "Circle")
    assert style["filled"] is False


def test_resolve_passes_construction_args(fresh_styles):
    seen = []

    class Recording(_FakeShape):
        def __init__(self, *args):
            super().__init__(*args)
            seen.append(args)

    fresh_styles.setattr(ssp, "_manim", _fake_manim(Polygon=Recording))
    ssp._resolve_shape_style("Polygon", "Polygon")
    assert seen == [("L", "R", "U")]


def test_ensure_caches_once(fresh_styles):
    calls = []

    def fake_resolve(name, manim_name):
        calls.append(name)
        return None

    fresh_styles.setattr(ssp, "_manim", _fake_manim())
    fresh_styles.setattr(ssp, "_resolve_shape_style", fake_resolve)
    first = ssp._ensure_manim_shape_styles()
    second = ssp._ensure_manim_shape_styles()
    assert first is second
    assert len(calls) == len(ssp._MANIM_SHAPE_STYLE_CLASSES)


def test_style_for_off_profile_returns_none(fresh_styles):
    fresh_styles.setattr(SETTINGS.style, "shape_style_profile", "algan")
    assert ssp._manim_shape_style_for(type("Square", (), {})) is None


def test_style_for_converts_stroke_width_without_mutating(fresh_styles):
    snapshot = {"stroke_width_manim": 4.0, "filled": False}
    fresh_styles.setattr(ssp, "_SHAPE_STYLES", {"Square": snapshot})
    fresh_styles.setattr(SETTINGS.style, "shape_style_profile", "manim")
    fresh_styles.setattr(SETTINGS.style, "manim_stroke_width_ratio", 2.0)
    style = ssp._manim_shape_style_for(type("Square", (), {}))
    assert style["stroke_width"] == pytest.approx(2.0)
    assert "stroke_width" not in snapshot


def test_style_for_unmapped_class_returns_none(fresh_styles):
    fresh_styles.setattr(ssp, "_SHAPE_STYLES", {"Square": None})
    fresh_styles.setattr(SETTINGS.style, "shape_style_profile", "manim")
    assert ssp._manim_shape_style_for(type("Square", (), {})) is None
    assert ssp._manim_shape_style_for(type("Blob", (), {})) is None


# --- typeset_cache ----------------------------------------------------------


def test_build_directory_is_sibling(tmp_path):
    tex_dir = tmp_path / "Tex"
    assert tc.build_directory(tex_dir) == tmp_path / "Tex-build"


def test_write_atomically_writes_and_leaves_no_partials(tmp_path):
    target = tmp_path / "a.svg"
    assert tc.write_atomically(target, b"data") == target
    assert target.read_bytes() == b"data"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.svg"]


def test_building_cleans_up_on_error(tmp_path):
    target = tmp_path / "a.svg"

    def build():
        with tc.building(target) as partial:
            partial.write_bytes(b"x")
            raise RuntimeError("fail")

    with pytest.raises(RuntimeError):
        build()
    assert list(tmp_path.iterdir()) == []


def test_building_yields_absolute_unique_paths(tmp_path):
    target = tmp_path / "a.svg"
    with tc.building(target) as first, tc.building(target) as second:
        first.write_bytes(b"1")
        second.write_bytes(b"1")
        assert first.is_absolute()
        assert first != second
    assert target.read_bytes() == b"1"


def test_publish_keeps_existing_destination_on_oserror(tmp_path, monkeypatch):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.write_text("new")
    dst.write_text("old")

    def refuse(a, b):
        raise PermissionError("held open")

    monkeypatch.setattr(tc.os, "replace", refuse)
    tc._publish(src, dst)
    assert dst.read_text() == "old"


def test_publish_raises_when_destination_missing(tmp_path, monkeypatch):
    def refuse(a, b):
        raise PermissionError("nope")

    monkeypatch.setattr(tc.os, "replace", refuse)
    with pytest.raises(PermissionError):
        tc._publish(tmp_path / "src", tmp_path / "dst")


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("x^2", False),
        ("$x$", True),
        (r"\$5", False),
        (r"\text{a $x$ b}", False),
        (r"\text {a $x$}", False),
        (r"\textbf{a} $x$", True),
        (r"\mbox{$y$} + z", False),
        (r"\frac{$a$}{b}", True),
        (r"\text{a {b} $c$}", False),
        (r"\text{a} {b} $c$", True),
    ],
)
def test_has_math_mode_dollar(expression, expected):
    assert tc._has_math_mode_dollar(expression) is expected


@pytest.mark.parametrize(
    ("environment", "has_hint"),
    [
        (None, False),
        ("align*", True),
        (r"\begin{align*}", True),
        ("equation", True),
        ("center", False),
        ("alignedat", False),
    ],
)
def test_math_mode_dollar_hint(environment, has_hint):
    hint = tc._math_mode_dollar_hint("$x$", environment)
    assert (hint is not None) is has_hint


def test_math_mode_dollar_hint_none_without_dollar():
    assert tc._math_mode_dollar_hint("x^2", "align*") is None


def test_sweep_removes_only_stale_entries_once(tmp_path, monkeypatch):
    monkeypatch.setattr(tc, "_swept_build_dirs", set())
    builds = tmp_path / "Tex-build"
    builds.mkdir()
    old_dir = builds / "old"
    old_dir.mkdir()
    old_file = builds / "old.log"
    old_file.write_text("x")
    fresh = builds / "fresh"
    fresh.mkdir()
    stale = time.time() - tc._STALE_BUILD_SECONDS - 60
    os.utime(old_dir, (stale, stale))
    os.utime(old_file, (stale, stale))
    tc._sweep_stale_builds(builds)
    assert sorted(p.name for p in builds.iterdir()) == ["fresh"]
    os.utime(fresh, (stale, stale))
    tc._sweep_stale_builds(builds)
    assert fresh.exists()


def test_sweep_missing_directory_is_silent(tmp_path, monkeypatch):
    monkeypatch.setattr(tc, "_swept_build_dirs", set())
    tc._sweep_stale_builds(Path(tmp_path / "missing"))


def test_log_excerpt_without_log(tmp_path):
    note = tc._log_excerpt_or_note(tmp_path / "f.log", tmp_path)
    assert "f.tex" in note


def test_log_excerpt_moves_log_and_reports_no_error(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    builds = tmp_path / "builds"
    builds.mkdir()
    log = work / "f.log"
    log.write_text("nothing wrong here\n")
    note = tc._log_excerpt_or_note(log, builds)
    assert "names no error" in note
    assert (builds / "f.log").exists()


def test_to_algan_color_folds_opacity():
    color = ssp._to_algan_color(_FakeColor((0.2, 0.4, 0.6, 0.5)), 0.5)
    other = ssp._to_algan_color(_FakeColor((0.2, 0.4, 0.6, 0.25)))
    assert repr(color) == repr(other)


def test_resolve_reads_checkerboard_and_none_stroke_width(fresh_styles):
    class Checkered(_FakeShape):
        def __init__(self, *args):
            super().__init__(*args)
            self.stroke_width = None
            self.checkerboard_colors = [
                _FakeColor((1.0, 1.0, 1.0, 1.0)),
                _FakeColor((0.0, 0.0, 1.0, 1.0)),
            ]

    fresh_styles.setattr(ssp, "_manim", _fake_manim(Sphere=Checkered))
    style = ssp._resolve_shape_style("Sphere", "Sphere")
    assert style["stroke_width_manim"] == 0.0
    assert style["checker_color"] is not None
