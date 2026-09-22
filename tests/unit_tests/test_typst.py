"""Typst compilation, cached SVG metadata, and live Algan selections."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import algan
import algan.manim as mn
from algan import BLUE, RIGHT, UP, MathTypst, Scene, Seq, Typst, easings
from algan.utils import manim_svg_cache


@pytest.fixture
def scene():
    with Scene() as active:
        yield active


@pytest.fixture
def compiler(monkeypatch, tmp_path):
    typst = pytest.importorskip("typst")
    monkeypatch.setattr(algan.SETTINGS.paths, "cache_directory", tmp_path)
    manim_svg_cache._MEM_CACHE.clear()
    return typst


@pytest.mark.parametrize("factory", [Typst, MathTypst, mn.Typst, mn.MathTypst])
def test_optional_backend_has_actionable_error(factory, monkeypatch, tmp_path, scene):
    monkeypatch.setitem(sys.modules, "typst", None)
    monkeypatch.setattr(algan.SETTINGS.paths, "cache_directory", tmp_path)
    with pytest.raises(ImportError, match=r"algan\[typst\]"):
        factory("x")
    assert list(tmp_path.iterdir()) == []


def test_exports():
    assert {"Typst", "MathTypst"} <= set(algan.__all__)
    assert issubclass(MathTypst, Typst)
    assert mn.MANIM_UNVENDORED_MOBJECT_NAMES == ()


@pytest.mark.parametrize("factory", [MathTypst, mn.MathTypst])
@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.fast
def test_selection_animates_original_geometry_and_replays(
    compiler, scene, factory, batch
):
    equation = factory("{{ x : x }} + {{ y }}", batch=batch).spawn(animate=False)
    actors = tuple(scene.actors)
    parents = [tuple(part.parents) for part in equation._typst_parts]
    x, y = equation.select("x")[0], equation.select(0)[0]
    original_x = x.control_points.location.clone()
    original_y = y.control_points.location.clone()
    with Seq(runtime=2, easing=easings.linear):
        equation.select("x").move(UP)
    equation.select(0).color = BLUE
    assert tuple(scene.actors) == actors
    assert [tuple(part.parents) for part in equation._typst_parts] == parents
    assert torch.allclose(x.control_points.location, original_x + UP)
    assert torch.equal(y.control_points.location, original_y)
    scene.timeline_manager.set_state_to_times(torch.tensor([0.0, 1.0, 2.0]))
    try:
        expected = original_x + torch.tensor([0.0, 0.5, 1.0]).reshape(-1, 1, 1) * UP
        assert torch.allclose(x.control_points.location, expected, atol=1e-5)
        assert torch.allclose(y.control_points.location, original_y.expand(3, -1, -1))
    finally:
        scene.timeline_manager.clear_buffers()


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.fast
def test_cloned_selections_have_independent_rows(compiler, scene, batch):
    equation = MathTypst("{{ a }} + {{ b : b }}", batch=batch, track_baselines=True)
    # Warm the packed view cache before cloning; cached views must be rebound.
    b = equation.select("b")[0]
    original = b.control_points.location.clone()
    clone = equation.clone(add_to_scene=False, spawn=False)
    clone.select("b").move(UP)
    assert torch.equal(b.control_points.location, original)
    assert torch.allclose(clone.select("b")[0].control_points.location, original + UP)
    for old, new in zip(
        equation.get_baseline_frame(b), clone.get_baseline_frame(clone.select("b")[0])
    ):
        assert torch.allclose(new, old + UP.reshape(3), atol=1e-5)


def test_markup_and_repeated_math_labels(compiler, scene):
    text = Typst("A #box[red] <word> and #box[blue] <word>.")
    assert len(text.select("word")) == 7
    assert len({id(part) for part in text.select("word")}) == 7
    equation = MathTypst("{{ x : term }} + {{ x : term }} = {{ 2 }} {{ x }}")
    assert len(equation.select("term")) == 2
    assert len(equation.select(0)) == len(equation.select(1)) == 1
    assert len(equation.select("_grp-1")) == 1
    with pytest.raises(KeyError, match="Available labels"):
        equation.select("absent")
    with pytest.raises(IndexError):
        equation.select(-1)
    with pytest.raises(IndexError):
        equation.select(2)
    with pytest.raises(TypeError):
        equation.select(0.5)


def test_size_color_preamble_and_svg_strokes(compiler, scene):
    small = MathTypst("frac(a, b)", font_size=24)
    big = MathTypst("frac(a, b)", font_size=48)
    assert torch.allclose(
        big.get_bounding_box(), 2 * small.get_bounding_box(), atol=1e-5
    )
    bars_small = [p for p in small._typst_parts if not p.filled]
    bars_big = [p for p in big._typst_parts if not p.filled]
    assert bars_small
    assert len(bars_small) == len(bars_big)
    for a, b in zip(bars_small, bars_big):
        assert bool((a.stroke_width > 0).all())
        assert torch.allclose(b.stroke_width, a.stroke_width * 2)
    exact = Typst("A", height=2)
    bounds = exact.get_bounding_box()[..., 1]
    assert float(bounds.amax() - bounds.amin()) == pytest.approx(2)
    colors = Typst('#box[A] <a> #box[#text(fill: rgb("#ff0000"))[B]] <b>', color=BLUE)
    assert torch.allclose(colors.select("a")[0].color[..., :3], BLUE.rgb)
    assert torch.allclose(
        colors.select("b")[0].color[..., :3], torch.tensor([1.0, 0.0, 0.0])
    )
    custom = Typst("#message", typst_preamble='#let message = "Hello"')
    assert len(custom._typst_parts) == 5


@pytest.mark.parametrize("font_size", [0, -1, float("nan"), float("inf")])
def test_invalid_font_size_is_rejected_before_compilation(font_size, scene):
    with pytest.raises(ValueError, match="font_size"):
        Typst("x", font_size=font_size)


@pytest.mark.parametrize("batch", [False, True])
def test_baselines_follow_parent_and_selection_transforms(compiler, scene, batch):
    equation = MathTypst("{{ x }} + y", track_baselines=True, batch=batch)
    path = equation.select(0)[0]
    original = torch.stack(equation.get_baseline_frame(path))
    equation.move(RIGHT)
    equation.select(0).move(UP)
    assert torch.allclose(
        torch.stack(equation.get_baseline_frame(path)), original + RIGHT + UP, atol=1e-5
    )
    assert len(equation.baseline_frames) == 3
    assert MathTypst("x").baseline_frames == []
    with pytest.raises(ValueError, match="track_baselines"):
        equation.get_baseline_frame(MathTypst("{{ x }}").select(0)[0])


def test_svg_memory_and_disk_cache_preserve_labels_baselines_and_strokes(
    compiler, monkeypatch
):
    import manim

    first = manim.MathTypst("{{ x : numerator }} / y", track_baselines=True)
    generate = manim.MathTypst.generate_mobject

    def no_parse(self):
        pytest.fail("A warm Typst SVG should be rebuilt from its cached recipe")

    monkeypatch.setattr(manim.MathTypst, "generate_mobject", no_parse)
    for clear in (False, True):
        if clear:
            manim_svg_cache._MEM_CACHE.clear()
        cached = manim.MathTypst("{{ x : numerator }} / y", track_baselines=True)
        assert len(cached.select("numerator")) == 1
        assert len(cached.baseline_frames) == len(first.baseline_frames) > 0
        for actual, expected in zip(cached.submobjects, first.submobjects):
            np.testing.assert_allclose(actual.points, expected.points)
            np.testing.assert_allclose(actual.stroke_width, expected.stroke_width)
        np.testing.assert_allclose(cached.baseline_frames, first.baseline_frames)
    monkeypatch.setattr(manim.MathTypst, "generate_mobject", generate)
    untracked = manim.MathTypst("{{ x : numerator }} / y", track_baselines=False)
    assert untracked.baseline_frames == []
    assert manim_svg_cache._stable_key(untracked) != manim_svg_cache._stable_key(first)


def test_compilation_cache_tracks_source_fonts_version_and_runtime_directory(
    compiler, monkeypatch, tmp_path
):
    from manim.utils import typst_file_writing as writing

    calls = []
    compile_real = compiler.compile

    def counted(*args, **kwargs):
        calls.append((args, kwargs))
        return compile_real(*args, **kwargs)

    monkeypatch.setattr(compiler, "compile", counted)
    first = writing.typst_to_svg_file("Hello")
    assert writing.typst_to_svg_file("Hello") == first
    assert len(calls) == 1
    assert first.is_relative_to(tmp_path / "manim" / "Typst")
    assert first.with_suffix(".typ").is_file()
    font_dir = tmp_path / "fonts"
    font_dir.mkdir()
    fonts = writing.typst_to_svg_file("Hello", font_paths=[font_dir])
    assert fonts != first
    assert calls[-1][1]["font_paths"] == [str(font_dir.resolve())]
    monkeypatch.setattr(writing, "version", lambda _: "future-version")
    assert writing.typst_to_svg_file("Hello") != first
    monkeypatch.setattr(algan.SETTINGS.paths, "cache_directory", tmp_path / "elsewhere")
    assert writing.typst_to_svg_file("Hello").is_relative_to(tmp_path / "elsewhere")
    assert len(calls) == 4


def test_compiler_error_leaves_no_svg_cache_entry(compiler, scene):
    with pytest.raises(compiler.TypstError):
        MathTypst("frac(")
    assert not list(Path(algan.SETTINGS.paths.cache_directory).rglob("*.svg"))


def test_native_stroke_convention_matches_compatibility(compiler, scene):
    native = Typst("X", stroke_width=2)
    compatible = mn.Typst(
        "X", stroke_width=2 * algan.SETTINGS.style.manim_stroke_width_ratio
    )
    assert torch.equal(
        native._typst_parts[0].stroke_width, compatible._typst_parts[0].stroke_width
    )


def test_morph_result_keeps_target_selection(compiler, scene):
    source = MathTypst("{{ x }}").spawn(animate=False)
    target = MathTypst("{{ y : answer }}")
    result = source.become(target)
    assert isinstance(result, MathTypst)
    points = result.select("answer")[0].control_points.location.clone()
    result.select("answer").move(UP)
    assert torch.allclose(
        result.select("answer")[0].control_points.location, points + UP
    )


@pytest.mark.parametrize("factory", [MathTypst, mn.MathTypst])
@pytest.mark.parametrize("batch", [False, True])
def test_selection_after_parent_transforms(compiler, scene, factory, batch):
    equation = factory("{{ x }} + y", batch=batch)
    before = equation.select(0)[0].control_points.location.clone()
    equation.move(RIGHT)
    equation.scale(2)
    expected = (before + RIGHT - equation.get_center()) * 2 + equation.get_center()
    selected = equation.select(0)[0]
    assert torch.allclose(selected.control_points.location, expected, atol=1e-5)
    live_paths = [
        p
        for p in equation.get_descendants()
        if isinstance(p, algan.ManimMob) and not p.empty
    ]
    assert any(p.id == selected.id for p in live_paths)


def test_shape_matching_preserves_typst_selection(compiler, scene):
    equation = MathTypst("{{ x }} + {{ y }}").spawn(animate=False)
    result = algan.TransformMatchingShapes(equation, MathTypst("{{ y }} + {{ x }}"))
    before = result.select(1)[0].control_points.location.clone()
    result.select(1).move(UP)
    assert torch.allclose(result.select(1)[0].control_points.location, before + UP)


@pytest.mark.parametrize("batch", [False, True])
def test_compatibility_delegated_edits_rebind_selections(compiler, scene, batch):
    equation = mn.MathTypst("{{ x }} + y", batch=batch, track_baselines=True).spawn(
        False
    )
    before = equation.select(0)[0].control_points.location.clone()
    equation.shift(RIGHT)
    selected = equation.select(0)[0]
    assert torch.allclose(selected.control_points.location, before + RIGHT, atol=1e-5)
    assert any(p is selected for p in equation.get_descendants())
    assert selected in scene.actors
    equation.select(0).move(UP)
    assert torch.allclose(
        selected.control_points.location, before + RIGHT + UP, atol=1e-5
    )
    assert len(equation.baseline_frames) == 3


@pytest.mark.parametrize("factory", [Typst, MathTypst])
def test_empty_content_is_valid(compiler, scene, factory):
    empty = factory("")
    assert len(empty._typst_parts) == 0
    assert empty.baseline_frames == []
