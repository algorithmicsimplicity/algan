"""A circuit's frame: batched exactly, facing OUTWARD, and immune to rounding.

``_circuit_frames`` frames many circuits at once (every glyph of a ``Text``),
and a pack must land on exactly the frames its members get alone. Which face a
circuit presents is the plane's own -- OUTWARD, else +y, else +x -- not a
by-product of which control points tie, so noise can never turn one over.
"""

from __future__ import annotations

import pytest
import torch

import algan.mobs.bezier_circuit as bezier_circuit
from algan import (
    Circle,
    Rectangle,
    RegularPolygon,
    SceneManager,
    Square,
    Star,
    Triangle,
)
from algan.mobs.text import mn


@pytest.fixture(autouse=True)
def fresh_scene():
    SceneManager.reset()
    yield
    SceneManager.reset()


def _segment(a, b):
    return torch.stack([a * (1 - t) + b * t for t in torch.linspace(0, 1, 4)])


def _polygon(corners):
    return torch.cat([_segment(a, b) for a, b in zip(corners, corners.roll(-1, 0))])


def _varied_circuits():
    generator = torch.Generator().manual_seed(7)
    circuits = []
    for i in range(24):
        n = int(torch.randint(3, 9, (1,), generator=generator))
        angles = torch.sort(torch.rand(n, generator=generator) * 6.283).values
        radii = 0.5 + torch.rand(n, generator=generator)
        corners = torch.stack(
            (radii * angles.cos(), radii * angles.sin(), torch.zeros(n)), -1
        )
        circuit = _polygon(corners)
        if i % 4 == 1:  # with a hole
            circuit = torch.cat((circuit, _polygon(corners.flip(0) * 0.3)))
        if i % 4 == 2:  # open
            circuit = circuit[:-4]
        if i % 5 == 3:  # tilted out of the screen plane
            rotation = torch.linalg.qr(torch.randn(3, 3, generator=generator))[0]
            circuit = circuit @ rotation
        circuits.append(circuit + torch.randn(3, generator=generator))
    circuits += [
        _segment(torch.tensor([0.0, 0, 0]), torch.tensor([2.0, 1, 0])),
        _segment(torch.tensor([0.0, 0, 0]), torch.tensor([0.0, 0, 3])),
        torch.zeros(4, 3),
    ]
    return circuits


def _glyphs(tex):
    typeset = mn.MathTex(tex, font_size=48)
    return [
        torch.from_numpy(glyph.points).float()
        for group in typeset.submobjects
        for glyph in group.submobjects
    ]


def test_a_circuit_gets_the_same_frame_alone_as_in_a_batch():
    circuits = _varied_circuits() + _glyphs(r"\sum_i x_i^2=\frac{a}{b}")
    locations, bases, synthesized = bezier_circuit._circuit_frames(circuits)
    assert not torch.isnan(bases).any()
    for index, circuit in enumerate(circuits):
        location, basis, alone = bezier_circuit._circuit_frames([circuit])
        assert torch.equal(location[0], locations[index])
        assert torch.equal(basis[0], bases[index])
        assert bool(alone[0]) == bool(synthesized[index])


def test_glyphs_and_shapes_face_outward_whatever_their_winding():
    outlines = _glyphs(r"w_{\rm new}=w_{\rm old}-\eta\,\text{abc 0123}")
    for mob in (
        Square(),
        Circle(),
        Triangle(),
        Rectangle(width=4, height=1),
        RegularPolygon(n=6),
        Star(),
    ):
        outline = mob.control_points.location.reshape(-1, 3).clone()
        outlines += [outline, outline.flip(0)]
    _, bases, _ = bezier_circuit._circuit_frames(outlines)
    normals = bases.reshape(-1, 3, 3)[:, 2]
    assert torch.allclose(normals, torch.tensor([0.0, 0.0, 1.0]).expand_as(normals))
    # Row 1 is +y for every one of them: an upright shape's own up is UP.
    assert (bases.reshape(-1, 3, 3)[:, 1, 1] > 0).all()


def test_rounding_cannot_turn_a_frame_round():
    generator = torch.Generator().manual_seed(3)
    outlines = _glyphs(r"\text{Hello World}\ 0123456789")
    for mob in (Square(), Circle(), Star()):
        outlines.append(mob.control_points.location.reshape(-1, 3).clone())
    _, reference, _ = bezier_circuit._circuit_frames(outlines)
    for shift in ([3.1, -2.7, 0.0], [1e-3, 0.0, 0.0], [0.0, 0.0, 0.0]):
        moved = [
            outline
            + torch.tensor(shift)
            + 1e-7 * torch.randn(outline.shape, generator=generator)
            for outline in outlines
        ]
        _, bases, _ = bezier_circuit._circuit_frames(moved)
        rows = bases.reshape(-1, 3, 3)
        before = reference.reshape(-1, 3, 3)
        # Every row points the same way: no axis flipped or swapped.
        assert ((rows * before).sum(-1) / (before * before).sum(-1) > 0.999).all()


@pytest.mark.parametrize(
    ("plane", "facing"),
    [
        ((0, 2), [0.0, 1.0, 0.0]),  # x-z plane: edge-on to z, faces +y
        ((1, 2), [1.0, 0.0, 0.0]),  # y-z plane: edge-on to z and y, faces +x
        ((0, 1), [0.0, 0.0, 1.0]),
    ],
)
def test_an_edge_on_plane_faces_up_then_right(plane, facing):
    corners2d = torch.tensor([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    for corners2d_wound in (corners2d, corners2d.flip(0)):
        corners = torch.zeros(4, 3)
        corners[:, plane[0]] = corners2d_wound[:, 0]
        corners[:, plane[1]] = corners2d_wound[:, 1]
        _, basis, _ = bezier_circuit._circuit_frames([_polygon(corners)])
        assert torch.allclose(basis[0, 6:], torch.tensor(facing))


def test_a_line_keeps_its_start_on_row_zero():
    start, end = torch.tensor([2.0, 1.0, 0.0]), torch.tensor([-1.0, 3.0, 0.0])
    location, basis, synthesized = bezier_circuit._circuit_frames(
        [_segment(start, end)]
    )
    assert bool(synthesized[0])
    assert torch.allclose(location[0] + basis[0, :3], start, atol=1e-6)
    assert torch.allclose(basis[0, 6:], torch.tensor([0.0, 0.0, 1.0]))
