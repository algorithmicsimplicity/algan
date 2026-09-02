import copy

import manim as mn
import torch

from algan.animation_timeline.animation_contexts import Off
from algan.constants.color import BLUE, GREEN, RED, WHITE, YELLOW
from algan.mobs.bezier_circuit import BezierCircuitCubic
from algan.mobs.text import Tex, Text, TexTriangulated, TextTriangulated
from algan.mobs.triangulated_bezier_circuit import TriangulatedBezierCircuit
from algan.scene_manager import SceneManager
from algan.utils.mob_utils import BatchedMobViewSequence, batch_mobs
from algan.utils.tensor_utils import unsquish


def _square(x, repeats=1):
    corners = torch.tensor(
        [[x, 0.0, 0.0], [x + 1.0, 0.0, 0.0], [x + 1.0, 1.0, 0.0], [x, 1.0, 0.0]]
    )
    segments = []
    for _ in range(repeats):
        for start, end in zip(corners, corners.roll(-1, 0)):
            segments.append(
                torch.stack(
                    [start, start * (2 / 3) + end / 3, start / 3 + end * (2 / 3), end]
                )
            )
    return torch.stack(segments)


def test_direct_bezier_batch_matches_object_batch():
    SceneManager.reset()
    paths = [_square(0.0), _square(2.0)]
    color = torch.tensor([0.2, 0.4, 0.6, 1.0, 0.0])
    with Off(record_funcs=False, record_attr_modifications=False):
        individual = [
            BezierCircuitCubic(path, color=color, stroke_width=2, add_to_scene=False)
            for path in paths
        ]
        expected = batch_mobs(individual, add_to_scene=False)
        actual = BezierCircuitCubic.from_batches(
            paths,
            color=color,
            stroke_width=2,
            add_to_scene=False,
        )

    for attr in (
        "location",
        "basis",
        "color",
        "opacity",
        "stroke_width",
        "stroke_color",
    ):
        assert torch.equal(getattr(actual, attr), getattr(expected, attr))
    assert torch.equal(actual.control_points.location, expected.control_points.location)
    assert torch.equal(
        actual.border_grid.location,
        expected.border_grid.location,
    )
    assert torch.equal(
        actual.border_grid.color,
        expected.border_grid.color,
    )
    assert torch.equal(
        actual.control_points.parent_batch_sizes,
        expected.control_points.parent_batch_sizes,
    )

    actual_primitive = actual.get_render_primitives()
    expected_primitive = expected.get_render_primitives()
    for attr in (
        "corners",
        "colors",
        "next_segment_inds",
        "normals",
        "stroke_width",
        "stroke_color",
        "mob_center",
        "basis1",
        "basis2",
        "num_segments_per_circuit",
    ):
        assert torch.equal(
            getattr(actual_primitive, attr), getattr(expected_primitive, attr)
        ), attr


def test_border_texture_grid_is_independent_from_fill_texture_grid():
    SceneManager.reset()
    mob = BezierCircuitCubic(
        _square(0.0),
        color=WHITE,
        stroke_color=YELLOW,
        stroke_width=8,
        grid_width=2,
        add_to_scene=False,
    )
    fill_colors = torch.stack((RED, RED, BLUE, BLUE)).unsqueeze(0)
    border_colors = torch.stack((GREEN, BLUE, GREEN, BLUE)).unsqueeze(0)

    mob.grid.color = fill_colors
    mob.border_grid.color = border_colors
    primitive = mob.get_render_primitives()

    assert torch.allclose(primitive.colors, fill_colors.unsqueeze(-3))
    assert torch.allclose(primitive.stroke_color, border_colors.unsqueeze(-3))
    assert torch.allclose(mob.stroke_color, border_colors)

    # The circuit's ordinary color remains the fill API and must not overwrite
    # the independently-authored border child.
    mob.color = YELLOW
    assert torch.allclose(mob.border_grid.color, border_colors)

    # The compatibility-facing stroke_color property now aliases the border
    # texture child, so a uniform write still works as it did before.
    mob.stroke_color = RED
    assert torch.allclose(mob.grid.color, YELLOW.expand(1, 4, 5))
    assert torch.allclose(mob.border_grid.color, RED.expand(1, 4, 5))


def test_batched_views_are_lazy_cached_and_isolated():
    SceneManager.reset()
    packed = BezierCircuitCubic.from_batches(
        [_square(0.0), _square(2.0)], add_to_scene=False
    )
    views = BatchedMobViewSequence(packed, 2)

    assert views._views == {}
    first = views[0]
    assert first is views[0]
    assert list(views._views) == [0]
    assert first.location.shape[-2] == 1
    assert first.control_points.location.shape[-2] == 16
    assert first.border_grid.location.shape[-2] == 1

    second = views[1]
    second_before = second.location.clone()
    first.move([1.0, 0.0, 0.0])
    assert torch.equal(second.location, second_before)
    assert views[:] == [first, second]


def test_deepcopy_of_batched_views_discards_shared_view_cache():
    SceneManager.reset()
    packed = BezierCircuitCubic.from_batches(
        [_square(0.0), _square(2.0)], add_to_scene=False
    )
    views = BatchedMobViewSequence(packed, 2)
    second = views[1]

    cloned_views = copy.deepcopy(views)

    assert cloned_views.mob is not packed
    assert cloned_views._views == {}
    assert torch.equal(cloned_views[1].location, second.location)


def test_batched_view_wave_expands_shared_attribute_for_full_owner():
    SceneManager.reset()
    packed = BezierCircuitCubic.from_batches(
        [_square(0.0), _square(2.0)], add_to_scene=False
    )
    views = BatchedMobViewSequence(packed, 2)

    views[0].wave_color(WHITE)

    color_timeline = packed.scene.timeline_manager.attr_to_timeline["color"]
    control_point_rows = color_timeline.mob_id_to_inds[packed.control_points.id]
    assert control_point_rows.numel() == packed.control_points.location.shape[-2]

    # The second view uses nonzero control-point data_sub_inds.  It must reuse
    # the full shared allocation rather than shrinking it to its local size.
    views[1].wave_color(WHITE)


def test_direct_triangulated_text_batch_matches_object_batch():
    SceneManager.reset()
    manim_text = mn.Text("Hi")
    paths = [
        unsquish(torch.from_numpy(char.points).float().flip(-2), -2, 4).transpose(
            -3, -2
        )
        for char in manim_text.submobjects
    ]
    with Off(record_funcs=False, record_attr_modifications=False):
        individual = [
            TriangulatedBezierCircuit(
                path,
                invert=False,
                hash_keys=None,
                reverse_points=False,
                use_cache=False,
                add_to_scene=False,
            )
            for path in paths
        ]
        expected = batch_mobs(individual, add_to_scene=False)
        actual = TriangulatedBezierCircuit(
            paths,
            invert=False,
            hash_keys=None,
            reverse_points=False,
            use_cache=False,
            add_to_scene=False,
        )

    assert len(actual.get_descendants()) == len(expected.get_descendants())
    for mob_index, (actual_mob, expected_mob) in enumerate(
        zip(actual.get_descendants(), expected.get_descendants())
    ):
        for attr in ("location", "basis", "color", "opacity"):
            actual_value = getattr(actual_mob, attr)
            expected_value = getattr(expected_mob, attr)
            assert torch.equal(actual_value, expected_value), (
                mob_index,
                attr,
                (actual_value - expected_value).abs().max(),
            )
        if actual_mob.parent_batch_sizes is None:
            assert expected_mob.parent_batch_sizes is None
        elif mob_index < 2:
            assert torch.equal(
                actual_mob.parent_batch_sizes, expected_mob.parent_batch_sizes
            )
        else:
            assert (
                int(actual_mob.parent_batch_sizes.sum())
                == (actual_mob.location.shape[-2])
            )

    views = BatchedMobViewSequence(actual, 2)
    second_locations = [mob.location.clone() for mob in views[1].get_descendants()]
    views[0].scale(2)
    assert all(
        torch.equal(mob.location, before)
        for mob, before in zip(views[1].get_descendants(), second_locations)
    )


def test_text_geometry_variants_are_public_and_distinct():
    assert Text.triangulated is False
    assert Tex.triangulated is False
    assert TextTriangulated.triangulated is True
    assert TexTriangulated.triangulated is True
