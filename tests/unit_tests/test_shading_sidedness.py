"""Which side a hit is lit from, from the mob's declaration to the kernel's slot.

A mob whose geometry has an outside declares it
(:attr:`~algan.animatable_base.mob.Mob.two_sided` False), the primitive carries
that per triangle (``declare_one_sided``), and it arrives in the kernel as slot
``_MAT_ONE_SIDED`` of the per-primitive material block, where
``_run_frag_pipeline`` reads it once per hit. What it buys: a back-facing hit on
a solid is shaded as that solid's inside instead of being lit as though it faced
the camera -- the defect that made a half-transparent solid's far shell a second
lit front shell.

These are tensor assertions on the primitive build, no render and no Taichi, so
they are cheap; they are feature tests of that plumbing rather than of anything
the timeline or the Scene can break, so they stay out of the fast suite. The
rendered consequence is covered by ``tests/fast`` and ``tests/full_renders``.
"""

from __future__ import annotations

import pytest
import torch

from algan import (
    IN,
    LEFT,
    RIGHT,
    UP,
    Cube,
    Cylinder,
    Line3D,
    Octahedron,
    Off,
    Polyhedron,
    Scene,
    Sphere,
    Square,
    Surface,
    Text,
    Torus,
)
from algan.rendering.raytracing.primitives import RayTracedTrianglePrimitive
from algan.rendering.raytracing.settings import _MAT_DEFAULTS
from algan.rendering.raytracing.shading_taichi import _MAT_ONE_SIDED, MAT_W


def _primitives(mob):
    build = getattr(mob, "get_render_primitives", None)
    got = build() if build is not None else None
    if got is None:
        # A container (Text is a Group of glyph mobs) builds nothing itself.
        return [p for child in mob.children for p in _primitives(child)]
    return got if isinstance(got, list) else [got]


def _one_sided(mob):
    """The per-triangle one_sided flags of every triangle primitive of ``mob``."""
    values = []
    for primitive in _primitives(mob):
        flag = getattr(primitive, "one_sided", None)
        if flag is not None:
            values.extend(flag.reshape(-1).tolist())
    return values


_SOLIDS = {
    "sphere": lambda: Sphere(radius=0.5),
    "cylinder": lambda: Cylinder(radius=0.4, height=1.0),
    "line3d": lambda: Line3D(start=LEFT, end=RIGHT, thickness=0.1),
    "torus": lambda: Torus(major_radius=0.6, minor_radius=0.2),
    "cube": lambda: Cube(side_length=0.8),
    "octahedron": lambda: Octahedron(edge_length=0.8),
}

_SHEETS = {
    "surface": lambda: Surface(grid_width=4, grid_height=4),
    "square": lambda: Square(side_length=1.0),
    "text": lambda: Text("hi", font_size=20),
}


@pytest.mark.parametrize("name", sorted(_SOLIDS))
def test_a_solid_declares_an_outside(name):
    with Scene(), Off():
        mob = _SOLIDS[name]()
        mob.spawn(animate=False)
        assert mob.two_sided is False
        flags = _one_sided(mob)
        assert flags, f"{name} built no triangle primitive to carry the flag"
        assert all(v == 1.0 for v in flags), (
            f"{name} must reach the renderer as one-sided, or its far shell "
            "is lit as a second front shell"
        )


@pytest.mark.parametrize("name", sorted(_SHEETS))
def test_geometry_with_no_outside_stays_two_sided(name):
    with Scene(), Off():
        mob = _SHEETS[name]()
        mob.spawn(animate=False)
        assert mob.two_sided is True
        # Circuits (Square, Text) carry no flag at all: they are shaded
        # through a different primitive and were never one-sided.
        assert all(v == 0.0 for v in _one_sided(mob))


def test_an_open_polyhedron_cannot_declare_an_outside():
    """``Polyhedron`` takes arbitrary user geometry. A single triangle is not a
    closed shell, so "outward" has no answer and it must stay two-sided.
    """
    with Scene(), Off():
        mob = Polyhedron([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]])
        mob.spawn(animate=False)
        assert mob.two_sided is True
        assert all(v == 0.0 for v in _one_sided(mob))


def test_an_instance_can_ask_for_two_sided_lighting_back():
    """The escape hatch for an open built-in you want lit inside."""
    with Scene(), Off():
        mob = Cylinder(radius=0.4, height=1.0)
        mob.two_sided = True
        mob.spawn(animate=False)
        assert all(v == 0.0 for v in _one_sided(mob))


def test_a_merged_collection_keeps_each_mobs_declaration():
    """The batcher merges every same-shader mob into one primitive, so the flag
    has to survive per triangle -- a solid and a sheet in one batch must not
    take each other's side.
    """
    with Scene(), Off():
        solid = Octahedron(edge_length=0.8)
        sheet = Surface(grid_width=3, grid_height=3)
        for mob in (solid, sheet):
            mob.spawn(animate=False)
        members = _primitives(solid) + _primitives(sheet)
        merged = RayTracedTrianglePrimitive(triangle_collection=members)

        per_triangle = merged.one_sided.reshape(merged.one_sided.shape[1], -1)
        solid_tris = sum(int(p.corners.shape[1]) // 3 for p in _primitives(solid))
        assert bool((per_triangle[:solid_tris] == 1.0).all())
        assert bool((per_triangle[solid_tris:] == 0.0).all())


def test_the_declaration_lands_in_the_material_block_slot():
    """What the kernel actually reads: slot ``_MAT_ONE_SIDED`` of the block
    ``_run_frag_pipeline`` already has in hand at a hit.
    """
    with Scene(), Off():
        solid = Octahedron(edge_length=0.8)
        sheet = Surface(grid_width=3, grid_height=3)
        for mob in (solid, sheet):
            mob.spawn(animate=False)
        merged = RayTracedTrianglePrimitive(
            triangle_collection=_primitives(solid) + _primitives(sheet)
        )
        _mat_id, mat = merged._pack_material()

        assert mat.shape[-1] == MAT_W
        solid_tris = sum(int(p.corners.shape[1]) // 3 for p in _primitives(solid))
        slot = mat[..., _MAT_ONE_SIDED]
        assert bool((slot[:, :solid_tris] == 1.0).all())
        assert bool((slot[:, solid_tris:] == 0.0).all())


def test_the_blocks_default_is_the_historical_behaviour():
    """Zero is two-sided, and zero is what both the defaults row and the
    zero-padding of a narrower custom block supply -- so anything that does not
    declare keeps the lighting it had.

    The padding rule here has outgrown "one_sided is the LAST slot": slots may
    be appended after it, each safe for exactly one reason -- its own 0.0 also
    means the behaviour that existed before (slots 27..29 attenuation_sigma:
    0.0 is no absorption, what every material did before they existed), or
    nothing outside the pipeline id that owns it can reach it (slots 30..32,
    toon's num_bands and depth's near/far, whose non-zero defaults only ever
    land in a built-in block).
    """
    from algan.rendering.raytracing.shading_taichi import (
        _MAT_ATTENUATION_SIGMA,
    )

    assert len(_MAT_DEFAULTS) == MAT_W
    assert _MAT_DEFAULTS[_MAT_ONE_SIDED] == 0.0
    assert all(
        v == 0.0 for v in _MAT_DEFAULTS[_MAT_ONE_SIDED : _MAT_ATTENUATION_SIGMA + 3]
    )


def test_moving_a_solid_keeps_its_declaration():
    """The flag is rebuilt with the primitive every batch, so it must survive
    the transforms that rebuild the geometry.
    """
    with Scene(), Off():
        mob = Cylinder(radius=0.3, height=1.0)
        mob.spawn(animate=False)
        mob.move_between_points(LEFT + IN, RIGHT + UP)
        assert all(v == 1.0 for v in _one_sided(mob))


def test_every_triangle_primitive_carries_a_flag():
    """A primitive that never declared anything must still pack a value: the
    collection merge reads the attribute off every member.
    """
    with Scene(), Off():
        mob = Surface(grid_width=3, grid_height=3)
        mob.spawn(animate=False)
        for primitive in _primitives(mob):
            assert isinstance(getattr(primitive, "one_sided", None), torch.Tensor)
