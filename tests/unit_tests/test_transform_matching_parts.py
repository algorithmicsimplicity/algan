"""Identity matching must survive packed glyphs, replay, and later animation."""

from __future__ import annotations

import pytest
import torch

import algan
import algan.manim as mn
from algan import (
    BLUE,
    GREEN,
    RED,
    RIGHT,
    UP,
    BezierCircuitCubic,
    Circle,
    Group,
    MathTex,
    Off,
    Scene,
    Seq,
    Square,
    Sync,
    Tex,
    Text,
    TransformMatchingShapes,
    TransformMatchingTex,
    easings,
)
from algan.animations.transform_matching_parts import (
    _buckets,
    _match_buckets,
    _shape_key,
    _shape_parts,
    _tex_parts,
)
from algan.errors import AlganConfigurationError


@pytest.fixture
def scene():
    with Scene() as active:
        yield active


def _center(mob):
    points = mob.control_points.location.reshape(-1, 3)
    return (points.amin(0) + points.amax(0)) / 2


def _paths(scene, times):
    """Read the geometry the renderer sees, including every packed member."""
    scene.timeline_manager.set_state_to_times(torch.tensor(times, dtype=torch.float32))
    frames = [[] for _ in times]
    for actor in scene.actors:
        if not isinstance(actor, BezierCircuitCubic) or actor.empty:
            continue
        cp = actor.control_points
        sizes = cp.parent_batch_sizes
        sizes = [cp.location.shape[-2]] if sizes is None else sizes.tolist()
        for index, frame in enumerate(frames):
            for row, points in enumerate(cp.location[index].split(sizes)):
                alpha = actor.opacity[index, row, 0] * actor.color[index, row, -1]
                if float(alpha) > 1e-4:
                    center = (points.amin(0) + points.amax(0)) / 2
                    frame.append(
                        (
                            actor.color[index, row, :3].clone(),
                            center.clone(),
                            float(alpha),
                        )
                    )
    scene.timeline_manager.clear_buffers()
    return frames


def _colored(frame, color):
    return [
        (center, alpha)
        for rgb, center, alpha in frame
        if torch.allclose(rgb, color[:3], atol=1e-4)
    ]


def test_public_matching_helpers_are_exported():
    assert "TransformMatchingTex" in algan.__all__
    assert "TransformMatchingShapes" in algan.__all__
    assert "_shape_key" not in algan.__all__


@pytest.mark.fast
def test_clone_of_packed_view_owns_local_rows(scene):
    """Cloning a late member must not reuse offsets into its owner's rows."""
    with Off():
        square = Square()
        path = square.control_points.location.reshape(-1, 4, 3)
        batch = BezierCircuitCubic.from_batches([path, path + RIGHT, path + RIGHT * 2])
        selected = batch[2]
        clone = selected.clone(add_to_scene=False, spawn=False)
        original = selected.control_points.location.clone()
        assert clone.data_sub_inds is None
        assert clone.control_points.data_sub_inds is None
        assert torch.equal(clone.control_points.location, original)
        clone.move(UP)
        assert torch.equal(selected.control_points.location, original)
        assert torch.allclose(clone.control_points.location, original + UP)


@pytest.mark.parametrize("factory", [Tex, MathTex, mn.MathTex, mn.Tex])
def test_tex_keys_follow_typeset_segments(factory, scene):
    equation = factory("x", "+", "y")
    assert list(_buckets(_tex_parts(equation))) == ["x", "+", "y"]
    whole = factory("x+y")
    assert list(_buckets(_tex_parts(whole))) == ["x+y"]


def test_double_braces_and_nested_equations(scene):
    equation = Tex(r"{{x}} + {{y}}")
    assert list(_buckets(_tex_parts(equation))) == ["x", " + ", "y"]
    group = Group(equation, Group(Tex("z")))
    assert list(_buckets(_tex_parts(group))) == ["x", " + ", "y", "z"]


def test_batched_manim_tex_reads_live_rows(scene):
    equation = mn.MathTex("x", "+", "y", batch=True)
    before = [part.get_center().clone() for _, part in _tex_parts(equation)]
    with Off():
        equation.move(UP * 2)
    after = [part.get_center() for _, part in _tex_parts(equation)]
    for start, end in zip(before, after):
        assert torch.allclose(end, start + UP * 2, atol=1e-5)


def test_explicit_mapping_overrides_identity_and_combines_repetitions():
    pairs, outgoing, incoming = _match_buckets(
        {"x": [1, 2], "y": [3], "a": [4]},
        {"x": [5], "y": [6], "a": [7, 8]},
        {"x": "y", "a": "y"},
    )
    assert pairs == [([1, 2, 4], [6])]
    assert outgoing == [3]
    assert incoming == [5, 7, 8]


def test_shape_identity_ignores_translation_uniform_scale_and_color(scene):
    with Off():
        square = Square(color=RED)
        shifted = Square(color=BLUE).scale(2.7).move(RIGHT * 3 + UP)
        rotated = Square().rotate(30)
        stretched = Square().scale(torch.tensor([2.0, 1.0, 1.0]))
    assert _shape_key(square) == _shape_key(shifted)
    assert _shape_key(square) != _shape_key(rotated)
    assert _shape_key(square) != _shape_key(stretched)
    assert _shape_key(square) != _shape_key(Circle())


def test_shape_parts_split_native_glyphs_and_duplicate_groups_once(scene):
    word = Text("STOP")
    assert len(_shape_parts(word)) == 4
    assert len(_shape_parts(Group(word, Group(word)))) == 4
    assert set(_buckets(_shape_parts(word))) == set(
        _buckets(_shape_parts(Text("POST")))
    )


@pytest.mark.parametrize("factory", [Tex, MathTex, mn.MathTex])
def test_rearrangement_keeps_term_identity_during_replay(factory, scene):
    with Off():
        source = factory("x", "+", "y").spawn(False)
        target = factory("y", "+", "x").move(UP)
        for key, part in _tex_parts(source) + _tex_parts(target):
            part.color = {"x": RED, "y": GREEN, "+": BLUE}[key]
    start = source.get_center().clone()
    original_target = target.get_center().clone()
    scene.wait(1)
    with Sync(runtime=2, easing=easings.identity):
        result = TransformMatchingTex(source, target)
    assert result is not target
    assert result.is_spawned()
    assert source.is_despawned()
    assert not target.is_spawned()
    assert torch.allclose(target.get_center(), original_target)
    assert result.__class__ is target.__class__

    frames = _paths(scene, [0.5, 1, 2, 3])
    for color in (RED, GREEN, BLUE):
        samples = [_colored(frame, color) for frame in frames]
        assert all(len(sample) == 1 for sample in samples)
        before, first, middle, last = [sample[0][0] for sample in samples]
        assert torch.allclose(first, before, atol=1e-4)
        assert torch.allclose(middle, (before + last) / 2, atol=1e-4)
    assert _colored(frames[0], RED)[0][0][0] < _colored(frames[-1], RED)[0][0][0]
    assert _colored(frames[0], GREEN)[0][0][0] > _colored(frames[-1], GREEN)[0][0][0]
    # Backward/sparse materialization gives the same visible geometry.
    replay = _paths(scene, [3, 0.5, 2])
    for actual, expected in zip(replay, [frames[3], frames[0], frames[2]]):
        for color in (RED, GREEN, BLUE):
            assert torch.allclose(
                _colored(actual, color)[0][0], _colored(expected, color)[0][0]
            )


@pytest.mark.parametrize(
    "policy", [{}, {"transform_mismatches": True}, {"fade_transform_mismatches": True}]
)
def test_mismatches_finish_as_exact_target_with_no_temporary_geometry(policy, scene):
    with Off():
        source = Tex("x", "+", "x", color=RED).spawn(False)
        target = Tex("x", "+", "z", "+", "w", color=BLUE).move(UP)
    scene.wait(1)
    result = TransformMatchingTex(source, target, runtime=1, **policy)
    assert len(result) == len(target)
    assert result._matching_tex_keys == target._matching_tex_keys
    assert torch.equal(
        result._character_batch.control_points.location,
        target._character_batch.control_points.location,
    )
    frames = _paths(scene, [0.5, 1.5, 2])
    assert len(frames[0]) == 3
    assert len(frames[-1]) == 5
    assert len(_colored(frames[-1], BLUE)) == 5


def test_default_unmatched_parts_fade_in_place(scene):
    with Off():
        source = Square(color=RED).move(RIGHT * -2).spawn(False)
        target = Circle(color=BLUE).move(RIGHT * 2)
    TransformMatchingShapes(source, target, runtime=2)
    frame = _paths(scene, [1])[0]
    assert _colored(frame, RED)[0][1] == pytest.approx(0.5, abs=2e-6)
    assert _colored(frame, BLUE)[0][1] == pytest.approx(0.5, abs=2e-6)
    assert _colored(frame, RED)[0][0][0] == pytest.approx(-2)
    assert _colored(frame, BLUE)[0][0][0] == pytest.approx(2)


def test_shape_mapping_can_override_a_natural_match(scene):
    with Off():
        square = Square(color=RED).move(RIGHT * -2)
        circle = Circle(color=BLUE).move(RIGHT * 2)
        source = Group(square, circle).spawn(False)
        target_square = Square(color=RED).move(RIGHT * -2)
        target_circle = Circle(color=BLUE).move(RIGHT * 2)
        target = Group(target_square, target_circle)
    result = TransformMatchingShapes(
        source, target, key_map={square: target_circle, circle: target_square}
    )
    assert result.is_spawned()
    frame = _paths(scene, [0.5])[0]
    assert all(abs(float(center[0])) < 1e-4 for _, center, _ in frame)


def test_shape_anagram_moves_the_original_letter_to_its_new_position(scene):
    with Off():
        source = Text("STOP").spawn(False)
        target = Text("POST").move(UP)
        source.character_mobs[0].color = RED  # S moves from first to third.
        target.character_mobs[2].color = RED
    TransformMatchingShapes(source, target, runtime=2)
    frames = _paths(scene, [0, 1, 2])
    assert all(len(frame) == 4 for frame in frames)
    start, middle, end = [_colored(frame, RED)[0][0] for frame in frames]
    assert start[0] < end[0]
    assert torch.allclose(middle, (start + end) / 2, atol=1e-4)


def test_empty_shapes_and_one_sided_fades_preserve_target_opacity(scene):
    source = Group().spawn(False)
    with Off():
        target = Square(color=BLUE, opacity=0.4)
    result = TransformMatchingShapes(source, target)
    frames = _paths(scene, [0, 0.5, 1])
    assert frames[0] == []
    assert _colored(frames[1], BLUE)[0][1] == pytest.approx(0.2, abs=2e-6)
    assert _colored(frames[2], BLUE)[0][1] == pytest.approx(0.4, abs=2e-6)
    result = TransformMatchingShapes(result, Group())
    frames = _paths(scene, [1.5, 2])
    assert _colored(frames[0], BLUE)[0][1] == pytest.approx(0.2, abs=2e-6)
    assert frames[-1] == []


def test_horizontal_and_degenerate_shapes_have_finite_keys(scene):
    from algan import Line

    with Off():
        line = Line(-RIGHT, RIGHT)
        translated = Line(-RIGHT * 3, RIGHT * 3).move(UP)
        point = Square().scale(0)
    assert _shape_key(line) == _shape_key(translated)
    assert _shape_key(point) == _shape_key(point)


def test_matching_uses_the_source_scene_when_another_scene_is_active(scene):
    source = Tex("x").spawn(False)
    target = Tex("x")
    with Scene() as other:
        result = TransformMatchingTex(source, target)
        assert float(other.animation_manager.context.timespan.current_time) == 0
    assert result.scene is scene
    assert float(scene.animation_manager.context.timespan.current_time) == 1


def test_returned_tex_can_be_indexed_transformed_again_and_follow_parent(scene):
    with Off():
        source = Tex("x", "+", "y").spawn(False)
        parent = Group(source)
    first = TransformMatchingTex(source, Tex("y", "+", "x"), runtime=1)
    assert parent.children == [first]
    second = TransformMatchingTex(
        first, Tex("z", "+", "y"), key_map={"x": "z"}, runtime=1
    )
    assert parent.children == [second]
    with Off():
        second.get_segment(0).color = RED
    before = second.get_center().clone()
    parent.move(UP)
    assert torch.allclose(second.get_center(), before + UP)
    assert float(
        scene.animation_manager.context.timespan.current_time
    ) == pytest.approx(3)
    frames = _paths(scene, [0.5, 1.5, 2.5, 3])
    assert all(len(frame) == 3 for frame in frames)
    assert len(_colored(frames[-1], RED)) == 1


@pytest.mark.parametrize(
    ("source_text", "target_text"), [("", "x"), ("x", ""), ("", "")]
)
def test_empty_equations_keep_timing_and_endpoint(source_text, target_text, scene):
    source = Tex(source_text).spawn(False)
    with Seq(runtime=2):
        result = TransformMatchingTex(source, Tex(target_text))
    assert float(
        scene.animation_manager.context.timespan.current_time
    ) == pytest.approx(2)
    assert len(result) == bool(target_text)


def test_off_makes_matching_immediate(scene):
    with Off():
        source = Tex("x").spawn(False)
        result = TransformMatchingTex(source, Tex("x"), runtime=3)
    assert result.is_spawned()
    assert float(scene.animation_manager.context.timespan.current_time) == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"transform_mismatches": True, "fade_transform_mismatches": True},
        {"key_map": {"absent": "y"}},
        {"runtime": -1},
    ],
)
def test_invalid_options_fail_before_source_lifecycle_changes(kwargs, scene):
    source = Tex("x").spawn(False)
    with pytest.raises((AlganConfigurationError, ValueError)):
        TransformMatchingTex(source, Tex("y"), **kwargs)
    assert source.is_spawned()
    assert not source.is_despawned()
    assert float(scene.animation_manager.context.timespan.current_time) == 0


def test_validation_of_scene_spawn_and_text_kind(scene):
    source = Tex("x")
    target = Tex("y")
    with pytest.raises(AlganConfigurationError, match="Spawn"):
        TransformMatchingTex(source, target)
    source.spawn(False)
    with Scene():
        foreign = Tex("x")
    with pytest.raises(AlganConfigurationError, match="same Scene"):
        TransformMatchingTex(source, foreign)
    with pytest.raises(TypeError, match="LaTeX"):
        TransformMatchingTex(Text("x").spawn(False), target)
    with pytest.raises(AlganConfigurationError, match="whole source"):
        TransformMatchingShapes(source.character_mobs[0], Square())
