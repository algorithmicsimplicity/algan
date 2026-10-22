"""Degenerate inputs to Mob construction and transforms.

A degenerate argument should either work or be rejected with a clear
``AlganConfigurationError`` -- never collapse a Mob silently or surface as a
low-level ``TypeError`` from deep inside torch.
"""

from __future__ import annotations

import pytest
import torch

from algan import ORIGIN, OUT, RIGHT, Group, RegularPolygon, Scene, Square
from algan.errors import AlganConfigurationError


@pytest.fixture
def scene():
    with Scene() as active:
        yield active


@pytest.mark.parametrize("axis", [ORIGIN, (0, 0, 0)])
def test_rotate_about_zero_axis_is_rejected(scene, axis):
    square = Square()
    before = square.get_boundary_points_recursive().clone()
    with pytest.raises(AlganConfigurationError, match="axis"):
        square.rotate(90, axis)
    # The rejected call left the square intact rather than collapsed.
    assert torch.allclose(square.get_boundary_points_recursive(), before)


def test_orbit_about_zero_axis_is_rejected(scene):
    with pytest.raises(AlganConfigurationError, match="axis"):
        Square().orbit(90, ORIGIN, about=RIGHT)


def test_rotate_with_unnormalized_axis_still_works(scene):
    square = Square().rotate(90, OUT * 5)
    points = square.get_boundary_points_recursive()
    assert torch.isfinite(points).all()
    assert float(points.abs().max()) > 0.1


def test_empty_group_rotate_still_works(scene):
    Group().rotate(90, OUT)


@pytest.mark.parametrize("n", [3.5, "5", None])
def test_regular_polygon_rejects_non_integer_sides(scene, n):
    with pytest.raises(AlganConfigurationError, match="whole number"):
        RegularPolygon(n=n)


def test_regular_polygon_accepts_integral_float(scene):
    assert RegularPolygon(n=5.0).n == 5
