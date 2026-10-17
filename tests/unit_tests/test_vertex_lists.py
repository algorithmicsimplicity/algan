"""A list of vectors is a list of points.

``cast_to_tensor`` says it combines a list of tensors into one tensor, and it
handled a list of numbers, but a list of the vector constants --
``[ORIGIN, RIGHT, UP]`` -- raised ``only one element tensors can be converted
to Python scalars``. So ``Polygon(ORIGIN, RIGHT, UP)`` worked and
``Polygon([ORIGIN, RIGHT, UP])`` did not, and an empty list of vertices
answered ``stack expects a non-empty TensorList``.
"""

import numpy as np
import pytest
import torch

from algan import ORIGIN, RIGHT, UP, AlganConfigurationError, Polygon, Scene
from algan.utils.tensor_utils import cast_to_tensor


@pytest.mark.parametrize(
    "points",
    [
        [ORIGIN, RIGHT, UP],
        (ORIGIN, RIGHT, UP),
        [np.zeros(3), np.array([1.0, 0, 0]), np.array([0, 1.0, 0])],
        [ORIGIN, [1, 0, 0], UP],
    ],
    ids=["list of tensors", "tuple of tensors", "list of arrays", "mixed"],
)
def test_a_list_of_vectors_casts_like_a_list_of_numbers(points):
    expected = cast_to_tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    torch.testing.assert_close(cast_to_tensor(points), expected)


def test_a_polygon_takes_its_vertices_as_one_list():
    with Scene():
        listed = Polygon([ORIGIN, RIGHT, UP], add_to_scene=False)
        spread = Polygon(ORIGIN, RIGHT, UP, add_to_scene=False)
        torch.testing.assert_close(listed.control_points, spread.control_points)


def test_an_empty_vertex_list_says_the_polygon_needs_vertices():
    with Scene(), pytest.raises(AlganConfigurationError, match="needs its vertices"):
        Polygon([])
