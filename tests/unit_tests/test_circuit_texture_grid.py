"""The texture grid a bezier circuit is coloured over.

Covers the grid's shape (rectangular, with degenerate axes collapsing to one
row), the ``(u, v)`` domain laid across it, the ``set_color_by_*`` methods that
fill it in, and ``Line``'s guarantee that its first basis row points at its
start.
"""

from __future__ import annotations

import pytest
import torch

from algan.animation_timeline.animation_contexts import Off
from algan.constants.color import BLUE, RED
from algan.constants.spatial import DOWN, LEFT, RIGHT, UP
from algan.mobs.bezier_circuit import BezierCircuitCubic
from algan.mobs.shapes_2d import Circle, Line, Point, Square
from algan.mobs.text import Text
from algan.scene_manager import SceneManager


def _rgb(uv):
    """A colour whose channels read straight back as ``(u, v, 0)``."""
    return torch.cat((uv[..., :1], uv[..., 1:], torch.zeros_like(uv[..., :1])), -1)


def test_texture_grid_axes_are_independent():
    SceneManager.reset()
    square = Square(texture_grid_width=4, texture_grid_height=2, add_to_scene=False)
    assert (square.grid_width, square.grid_height) == (4, 2)
    assert square.num_texture_points == 8

    # Texels are laid out with the first basis (u) axis outermost, so the packed
    # rows reshape into [width, height] -- the same layout Surface uses.
    points = square.texture_points.location.reshape(4, 2, 3)
    first = square.basis[..., :3].reshape(3)
    second = square.basis[..., 3:6].reshape(3)
    center = square.location.reshape(3)
    for i, a1 in enumerate(torch.linspace(-1, 1, 4)):
        for j, a2 in enumerate(torch.linspace(-1, 1, 2)):
            expected = center + a1 * first * (1 + 1e-5) + a2 * second * (1 + 1e-5)
            torch.testing.assert_close(points[i, j], expected, atol=1e-5, rtol=0)


def test_texture_grid_height_mirrors_width_by_default():
    SceneManager.reset()
    square = Square(texture_grid_width=5, add_to_scene=False)
    assert (square.grid_width, square.grid_height) == (5, 5)


def test_degenerate_second_axis_defaults_to_one_row():
    SceneManager.reset()
    # A straight line's control points are collinear: its second basis row is
    # synthesized perpendicular to the path and every point of the shape maps to
    # the same v, so sampling it more than once buys nothing.
    line = Line(LEFT, RIGHT, texture_grid_width=8, add_to_scene=False)
    assert (line.grid_width, line.grid_height) == (8, 1)
    assert line.num_texture_points == 8

    # Explicitly asking for rows still gets them.
    thick = Line(
        LEFT, RIGHT, texture_grid_width=8, texture_grid_height=3, add_to_scene=False
    )
    assert (thick.grid_width, thick.grid_height) == (8, 3)

    # An arc is not collinear, so it keeps the square default.
    arc = Line(LEFT, RIGHT, path_arc=1.0, texture_grid_width=4, add_to_scene=False)
    assert (arc.grid_width, arc.grid_height) == (4, 4)

    # A Point has no extent at all.
    assert Point(texture_grid_width=4, add_to_scene=False).grid_height == 1


def test_default_grid_is_a_single_texel():
    SceneManager.reset()
    square = Square(add_to_scene=False)
    assert (square.grid_width, square.grid_height) == (1, 1)
    assert square.num_texture_points == 1


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (LEFT, RIGHT),
        (RIGHT, LEFT),
        (LEFT + DOWN, RIGHT + UP),
        # Endpoints that are equidistant from the centre in exact arithmetic and
        # an ulp apart in practice: the case an argmax tie-break resolves either
        # way from one machine to the next.
        (torch.tensor([[0.1, 0.0, 0.0]]), torch.tensor([[0.30000001, 0.0, 0.0]])),
    ],
)
def test_line_first_basis_points_from_center_toward_start(start, end):
    SceneManager.reset()
    line = Line(start, end, add_to_scene=False)
    first = line.basis[..., :3].reshape(3)
    toward_start = line.get_start().reshape(3) - line.location.reshape(3)
    torch.testing.assert_close(
        torch.nn.functional.normalize(first, dim=-1),
        torch.nn.functional.normalize(toward_start, dim=-1),
        atol=1e-5,
        rtol=0,
    )


def test_base_grid_spans_zero_to_one_and_centers_degenerate_axes():
    SceneManager.reset()
    grid = Square(
        texture_grid_width=3, texture_grid_height=2, add_to_scene=False
    ).get_base_grid()
    assert grid.shape == (3, 2, 2)
    torch.testing.assert_close(
        grid[..., 0], torch.tensor([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
    )
    torch.testing.assert_close(
        grid[..., 1], torch.tensor([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    )

    # A single-sample axis carries one colour for the whole span, so it is
    # evaluated at the middle of it rather than at one end.
    line_grid = Line(
        LEFT, RIGHT, texture_grid_width=2, add_to_scene=False
    ).get_base_grid()
    assert line_grid.shape == (2, 1, 2)
    torch.testing.assert_close(line_grid[..., 1], torch.full((2, 1), 0.5))


def test_set_color_by_function_writes_one_color_per_texel():
    SceneManager.reset()
    square = Square(texture_grid_width=3, texture_grid_height=2, add_to_scene=False)
    with Off(animation_manager=square.animation_manager):
        square.set_color_by_function(_rgb)

    colors = square.texture_points.color.reshape(3, 2, 5)
    grid = square.get_base_grid()
    torch.testing.assert_close(colors[..., 0], grid[..., 0])
    torch.testing.assert_close(colors[..., 1], grid[..., 1])
    # Missing channels take their defaults: no glow, fully opaque.
    torch.testing.assert_close(colors[..., 3], torch.zeros(3, 2))
    torch.testing.assert_close(colors[..., 4], torch.ones(3, 2))


def test_set_color_by_function_returns_self_and_accepts_rgba():
    SceneManager.reset()
    circle = Circle(texture_grid_width=2, add_to_scene=False)
    with Off(animation_manager=circle.animation_manager):
        result = circle.set_color_by_function(
            lambda uv: torch.cat((uv, uv[..., :1], uv[..., :1] * 0.5), -1)
        )
    assert result is circle
    torch.testing.assert_close(
        circle.texture_points.color[..., 4].reshape(-1),
        circle.get_base_grid()[..., 0].reshape(-1) * 0.5,
    )


def test_filled_circuit_colors_its_fill_and_unfilled_one_its_stroke():
    SceneManager.reset()
    filled = Square(texture_grid_width=2, border_color=RED, add_to_scene=False)
    border_before = filled.border_texture_points.color.clone()
    with Off(animation_manager=filled.animation_manager):
        filled.set_color_by_function(_rgb)
    assert not torch.equal(filled.texture_points.color, border_before)
    assert torch.equal(filled.border_texture_points.color, border_before)

    unfilled = Square(texture_grid_width=2, filled=False, add_to_scene=False)
    with Off(animation_manager=unfilled.animation_manager):
        unfilled.set_color_by_function(_rgb)
    # An unfilled circuit has no interior to show the colours in, so the stroke
    # takes them too.
    assert torch.equal(
        unfilled.border_texture_points.color, unfilled.texture_points.color
    )


def test_line_set_color_by_function_runs_from_start_to_end():
    SceneManager.reset()
    line = Line(LEFT * 2, RIGHT * 2, texture_grid_width=5, add_to_scene=False)
    with Off(animation_manager=line.animation_manager):
        line.set_color_by_function(
            lambda t: torch.cat((t, torch.zeros_like(t), 1 - t), -1)
        )

    colors = line.texture_points.color.reshape(5, 5)
    positions = line.texture_points.location.reshape(5, 3)
    start = line.get_start().reshape(3)
    end = line.get_end().reshape(3)
    # Redness has to grow with distance from the start, however the texels
    # happen to be ordered in the buffer.
    along = ((positions - start) * (end - start)).sum(-1) / (end - start).square().sum()
    torch.testing.assert_close(colors[:, 0], along, atol=1e-5, rtol=0)
    torch.testing.assert_close(colors[:, 2], 1 - along, atol=1e-5, rtol=0)


def test_set_color_by_image_lands_the_images_top_left_at_the_origin_of_uv():
    SceneManager.reset()
    # Quadrants, with rows running down the picture.
    image = torch.zeros(8, 8, 4)
    image[..., 3] = 1.0
    image[:4, :4, 0] = 1.0  # top-left red
    image[:4, 4:, 1] = 1.0  # top-right green
    image[4:, :4, 2] = 1.0  # bottom-left blue

    square = Square(texture_grid_width=8, texture_grid_height=8, add_to_scene=False)
    with Off(animation_manager=square.animation_manager):
        square.set_color_by_image(image)

    # [u, v]: u across, v down, so the image's top left lands at (0, 0).
    colors = square.texture_points.color.reshape(8, 8, 5)
    for (u, v), channel in (((0, 0), 0), ((-1, 0), 1), ((0, -1), 2)):
        brightest = int(colors[u, v, :3].argmax())
        assert brightest == channel
        assert colors[u, v, channel] > 0.9


def test_set_color_by_image_resamples_to_the_grid():
    SceneManager.reset()
    image = torch.rand(16, 12, 4)
    image[..., 3] = 1.0
    square = Square(texture_grid_width=5, texture_grid_height=3, add_to_scene=False)
    with Off(animation_manager=square.animation_manager):
        square.set_color_by_image(image)
    assert square.texture_points.color.shape[-2] == 15


def test_a_flat_circuit_says_how_to_get_a_grid():
    SceneManager.reset()
    square = Square(add_to_scene=False)
    for call in (
        lambda: square.set_color_by_function(_rgb),
        lambda: square.set_color_by_image(torch.rand(4, 4, 4)),
    ):
        with pytest.raises(ValueError, match="texture_grid_width"):
            call()


def test_a_function_returning_the_wrong_count_is_rejected():
    SceneManager.reset()
    square = Square(texture_grid_width=3, add_to_scene=False)
    with pytest.raises(ValueError, match="one color per texel"):
        square.set_color_by_function(lambda uv: torch.zeros(4, 3))


def test_multi_circuit_mobs_repeat_the_pattern_per_circuit():
    SceneManager.reset()
    text = Text("ab", texture_grid_width=2, texture_grid_height=2)
    circuit = next(
        mob
        for mob in text.get_descendants()
        if isinstance(mob, BezierCircuitCubic) and mob.num_texture_points > 1
    )
    objects = circuit.location.shape[-2]
    assert objects > 1
    with Off(animation_manager=circuit.animation_manager):
        circuit.set_color_by_function(_rgb)

    colors = circuit.texture_points.color.reshape(objects, 4, 5)
    for index in range(1, objects):
        assert torch.equal(colors[index], colors[0])


def test_a_single_glyph_view_colors_only_its_own_texels():
    SceneManager.reset()
    text = Text("ab", texture_grid_width=2, texture_grid_height=2)
    first, second = text.character_mobs[0], text.character_mobs[1]
    untouched = second.texture_points.color.clone()
    with Off(animation_manager=first.animation_manager):
        first.set_color_by_function(_rgb)

    torch.testing.assert_close(
        first.texture_points.color.reshape(2, 2, 5)[..., :2],
        first.get_base_grid(),
    )
    assert torch.equal(second.texture_points.color, untouched)


def test_direct_construction_matches_the_shape_helpers():
    SceneManager.reset()
    corners = torch.tensor(
        [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]]
    )
    segments = torch.stack(
        [
            torch.stack([a, a * (2 / 3) + b / 3, a / 3 + b * (2 / 3), b])
            for a, b in zip(corners, corners.roll(-1, 0))
        ]
    )
    circuit = BezierCircuitCubic(
        segments,
        color=BLUE,
        texture_grid_width=4,
        texture_grid_height=2,
        add_to_scene=False,
    )
    assert (circuit.grid_width, circuit.grid_height) == (4, 2)
    with Off(animation_manager=circuit.animation_manager):
        circuit.set_color_by_function(_rgb)
    assert circuit.texture_points.color.shape[-2] == 8
