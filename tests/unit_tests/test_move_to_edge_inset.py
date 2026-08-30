"""``move_to_screen_edge`` insets the boundary *inside* the border, from anywhere.

The sign is the whole point of this file. ``move_to_screen_edge`` used to take its
inset direction from ``normalize(boundary - border)`` -- read off where the Mob
happens to be rather than off the edge that was asked for. The border is cast
from the boundary along the edge, so that difference is antiparallel to the edge
only while the Mob is inside the frame. Two cases fall out of it:

* a Mob already past the edge came to rest ``buffer`` *outside* the border,
  still off-screen, rather than being brought in;
* a boundary resting exactly *on* the border made the difference zero, and
  ``F.normalize`` amplified float32 noise into a whole unit vector, so rounding
  picked the direction. A Manim-compatibility ``Title`` lands exactly there,
  because Manim's ``to_edge`` insets by 0.5 from a frame 8 units tall and
  Algan's top border is at 3.5.

Neither was caught, because the existing coverage in
``test_manim_compat_movement.py`` measures the gap with ``.norm()``. An
unsigned gap cannot tell 0.6 inside from 0.6 outside. Everything here is signed.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from algan import DOWN, LEFT, RIGHT, SETTINGS, UP, SceneManager, Square

EDGES = {"up": UP, "down": DOWN, "left": LEFT, "right": RIGHT}


@pytest.fixture(autouse=True)
def reset_scene():
    SceneManager.reset()
    yield
    SceneManager.reset()


def _buffer() -> float:
    return float(SETTINGS.style.buffer)


def _signed_inset(mob, edge) -> float:
    """How far inside the border the Mob's boundary sits, along ``edge``.

    Positive is inside the frame, negative is past the border. The border is
    cast from the boundary along ``edge``, so the displacement between them is
    always parallel to ``edge``; its component along ``edge`` carries the sign
    that ``.norm()`` throws away.
    """
    edge = F.normalize(edge.to(torch.get_default_dtype()), p=2, dim=-1)
    boundary = mob.get_boundary_point(edge)
    border = mob.scene.camera.project_point_onto_screen_border(boundary, edge)
    return float(((border - boundary) * edge).sum(-1).reshape(-1)[0])


@pytest.mark.parametrize("edge_name", sorted(EDGES))
def test_move_to_edge_insets_a_mob_that_starts_inside(edge_name):
    edge = EDGES[edge_name]
    mob = Square().move_to_screen_edge(edge)

    assert _signed_inset(mob, edge) == pytest.approx(_buffer(), abs=2e-5)


@pytest.mark.parametrize("edge_name", sorted(EDGES))
def test_move_to_edge_brings_back_a_mob_that_starts_outside(edge_name):
    # Well past the border to begin with. This used to come to rest ``buffer``
    # *outside* the border -- still off-screen -- because the inset direction
    # flipped with the Mob's position.
    edge = EDGES[edge_name]
    mob = Square().move(edge * 8).move_to_screen_edge(edge)

    assert _signed_inset(mob, edge) == pytest.approx(_buffer(), abs=2e-5)


@pytest.mark.parametrize("edge_name", sorted(EDGES))
def test_move_to_edge_insets_a_boundary_resting_exactly_on_the_border(edge_name):
    # Put the boundary on the border, which is the degenerate case: the old
    # direction term was a zero vector normalized into noise.
    edge = EDGES[edge_name]
    mob = Square()
    boundary = mob.get_boundary_point(edge)
    border = mob.scene.camera.project_point_onto_screen_border(boundary, edge)
    mob.move(border - boundary)
    assert _signed_inset(mob, edge) == pytest.approx(0.0, abs=1e-4)

    mob.move_to_screen_edge(edge)

    assert _signed_inset(mob, edge) == pytest.approx(_buffer(), abs=2e-5)


@pytest.mark.parametrize("edge_name", sorted(EDGES))
def test_move_to_edge_is_idempotent(edge_name):
    edge = EDGES[edge_name]
    mob = Square().move_to_screen_edge(edge)
    once = mob.get_center().reshape(-1, 3)[0].clone()

    mob.move_to_screen_edge(edge)

    torch.testing.assert_close(
        mob.get_center().reshape(-1, 3)[0], once, atol=2e-5, rtol=0
    )


def test_move_to_screen_corner_insets_both_edges_from_outside():
    mob = Square().move(UP * 8 + RIGHT * 12).move_to_screen_corner((UP, RIGHT))

    assert _signed_inset(mob, UP) == pytest.approx(_buffer(), abs=2e-5)
    assert _signed_inset(mob, RIGHT) == pytest.approx(_buffer(), abs=2e-5)


@pytest.mark.parametrize("edge_name", sorted(EDGES))
def test_move_to_edge_leaves_the_other_axes_alone(edge_name):
    edge = EDGES[edge_name]
    mob = Square().move(UP * 0.4 + RIGHT * 0.3)
    start = mob.get_center().reshape(-1, 3)[0].clone()

    mob.move_to_screen_edge(edge)

    off_axis = edge.abs().reshape(-1) < 0.5
    torch.testing.assert_close(
        mob.get_center().reshape(-1, 3)[0][off_axis], start[off_axis], atol=2e-5, rtol=0
    )
