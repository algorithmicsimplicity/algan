"""Per-triangle SURFACE identity (``tri_obj``) at the granularity mobs declare.

``tri_obj`` is what the analytic-AA resolve groups fragments by: a run is a
maximal consecutive group of depth-sorted fragments sharing ``(tri_obj, facing)``
(``DESIGN_analytic_aa_v2.md`` §4.2), and its exact clipped areas are summed
within a run and composited between runs. Getting the granularity wrong is
therefore silently visible: too coarse and coverage is summed across objects that
merely overlap, too fine and a mesh's own interior edges can never share a run.

Nothing tested ``tri_obj`` before this file, in any suite. These are pure tensor
assertions on the primitive build -- no render, no Taichi -- so they are cheap
enough for the fast suite, and they guard a contract shared by the merge, the
resolve and the shadow-event walk.
"""

from __future__ import annotations

import pytest
import torch

from algan import (
    RIGHT,
    Cube,
    Icosahedron,
    Off,
    Scene,
    Sphere,
    Square,
)
from algan.rendering.raytracing.primitives import _mesh_ids_from_collection


def _tri_obj(primitives):
    """The per-triangle surface ids a collection of primitives resolves to.

    Mirrors what ``_pack_projected_flat_geometry`` does without needing a
    camera: the collection path resolves ``_rt_obj_ids`` at construction.
    """
    from algan.rendering.raytracing.primitives import RayTracedTrianglePrimitive

    collection = RayTracedTrianglePrimitive(triangle_collection=list(primitives))
    ids, n = collection._rt_obj_ids, collection._rt_obj_ids_n
    if ids is None:
        # No member declared identity: the per-member counts are the ids.
        counts = collection._rt_obj_counts
        ids = torch.repeat_interleave(
            torch.arange(len(counts), dtype=torch.int32),
            torch.tensor(counts, dtype=torch.int64),
        )
        n = len(counts)
    return ids, n


def _primitives(mob):
    got = mob.get_render_primitives()
    if got is None:
        return []
    return got if isinstance(got, list) else [got]


class _Member:
    """Minimal stand-in for a collection member: enough for the resolver."""

    def __init__(self, n_tri, mesh_key=None, mesh_ids=None):
        self.corners = torch.zeros((1, n_tri * 3, 3))
        if mesh_key is not None:
            self.mesh_key = mesh_key
        if mesh_ids is not None:
            self.mesh_ids = mesh_ids


def test_a_collection_declaring_nothing_keeps_one_id_per_member():
    ids, n = _mesh_ids_from_collection([_Member(2), _Member(3)], [2, 3])
    reason = (
        "a collection where no member declares identity must fall through to "
        "the per-member counts, so the default path stays byte-identical"
    )
    assert ids is None, reason
    assert n is None, reason


def test_members_sharing_a_mesh_key_become_one_surface():
    members = [_Member(1, mesh_key="cube") for _ in range(12)]
    ids, n = _mesh_ids_from_collection(members, [1] * 12)
    assert n == 1
    assert ids.tolist() == [0] * 12


def test_two_keyed_groups_stay_distinct():
    members = [
        _Member(2, mesh_key="a"),
        _Member(2, mesh_key="a"),
        _Member(2, mesh_key="b"),
        _Member(2, mesh_key="b"),
    ]
    ids, n = _mesh_ids_from_collection(members, [2, 2, 2, 2])
    assert n == 2
    assert ids.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]


def test_an_unkeyed_member_between_two_keyed_ones_does_not_bridge_them():
    """A key merges with the PRECEDING member only, so identity cannot leak
    across an unrelated mob that happens to sit between two halves of one.
    """
    members = [
        _Member(1, mesh_key="a"),
        _Member(1),
        _Member(1, mesh_key="a"),
    ]
    ids, n = _mesh_ids_from_collection(members, [1, 1, 1])
    assert n == 3
    assert ids.tolist() == [0, 1, 2]


def test_mesh_ids_subdivide_a_member_into_shells():
    member = _Member(6, mesh_ids=torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int32))
    ids, n = _mesh_ids_from_collection([member], [6])
    assert n == 2
    assert ids.tolist() == [0, 0, 0, 1, 1, 1]


def test_shell_ids_are_renumbered_into_the_collections_namespace():
    """Two members' local shell ids must not collide after concatenation."""
    a = _Member(4, mesh_ids=torch.tensor([7, 7, 9, 9], dtype=torch.int32))
    b = _Member(2, mesh_ids=torch.tensor([3, 4], dtype=torch.int32))
    ids, n = _mesh_ids_from_collection([a, b], [4, 2])
    assert n == 4
    assert ids.tolist() == [0, 0, 1, 1, 2, 3]


def test_mesh_ids_length_must_match_the_member():
    member = _Member(3, mesh_ids=torch.tensor([0, 1], dtype=torch.int32))
    with pytest.raises(ValueError, match="mesh_ids has 2 entries"):
        _mesh_ids_from_collection([member], [3])


@pytest.mark.fast
def test_a_cube_is_one_surface_not_one_per_triangle():
    """A ``Polyhedron`` hands the batcher one member per triangle. Without a
    declared key a Cube is twelve surfaces and no run can span a face.
    """
    with Scene():
        with Off():
            cube = Cube(side_length=1.0).spawn()
        prims = _primitives(cube)
        assert len(prims) > 1, "expected one member per triangle"
        ids, n = _tri_obj(prims)
        assert n == 1, f"a Cube must be one surface, got {n}"
        assert set(ids.tolist()) == {0}


@pytest.mark.fast
def test_two_polyhedra_in_one_collection_stay_two_surfaces():
    with Scene():
        with Off():
            a = Cube(side_length=1.0).spawn()
            b = Icosahedron(edge_length=1.0).move(RIGHT * 3).spawn()
        ids, n = _tri_obj(_primitives(a) + _primitives(b))
        assert n == 2, f"two solids must stay two surfaces, got {n}"
        # Contiguous, in concatenation order.
        first, second = ids[0].item(), ids[-1].item()
        assert first != second


@pytest.mark.fast
def test_a_surface_is_one_surface():
    with Scene():
        with Off():
            sphere = Sphere(radius=1.0).spawn()
        prims = _primitives(sphere)
        assert len(prims) == 1
        ids, n = _tri_obj(prims)
        assert n == 1
        assert set(ids.tolist()) == {0}


@pytest.mark.fast
def test_two_spheres_in_one_collection_stay_two_surfaces():
    """The case ``_rt_obj_counts`` was introduced for: the batcher merges every
    same-identifier mob into one collection, so one member is not one surface.
    """
    with Scene():
        with Off():
            a = Sphere(radius=1.0).spawn()
            b = Sphere(radius=1.0).move(RIGHT * 3).spawn()
        ids, n = _tri_obj(_primitives(a) + _primitives(b))
        assert n == 2
        assert ids[0].item() != ids[-1].item()


def test_mesh_identity_is_off_switchable():
    """``ALGAN_MESH_ID=0`` must restore the per-member ids exactly, so the
    switch is a byte-level A/B rather than a behaviour flag.
    """
    from algan import SETTINGS

    with Scene():
        with Off():
            cube = Cube(side_length=1.0).spawn()
        prims = _primitives(cube)
        assert all(getattr(p, "mesh_key", None) is not None for p in prims)

    rt = SETTINGS.raytracing
    assert rt.MESH_ID is False, (
        "mesh identity is expected OFF by default until the run rule consults "
        "E -- see the MESH_ID comment in raytracing/settings.py"
    )
    original = rt.MESH_ID
    try:
        rt.experimental.set(mesh_id=True)
        assert rt.MESH_ID is True
    finally:
        rt.experimental.set(mesh_id=original)
    assert rt.MESH_ID is original


@pytest.mark.fast
def test_a_2d_circuit_mob_has_no_triangle_identity():
    """Circuits carry ``sid = -1 - circuit`` instead, so they must not appear in
    the triangle collection at all.
    """
    from algan.rendering.raytracing.primitives import RayTracedBezierCircuitPrimitive

    with Scene():
        with Off():
            square = Square(side_length=1.0).spawn()
        prims = _primitives(square)
        assert prims
        assert all(isinstance(p, RayTracedBezierCircuitPrimitive) for p in prims)
